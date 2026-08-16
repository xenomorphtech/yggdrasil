//! The JIT publisher and runtime helpers.
//!
//! Publisher: takes `ygg-jit`'s per-function code + relocations, lays them out
//! in physically contiguous frames, patches helper (Abs8) and sibling (PC-rel)
//! relocations through the HHDM alias, and maps the result RX in the code
//! zone. x86_64 needs no icache flush.
//!
//! Helpers: `extern "C"` functions generated code calls for everything
//! effectful. On error they exit the *process* directly (same observable
//! behavior as an interpreter trap), so generated code never handles failure.

use alloc::sync::Arc;
use alloc::vec::Vec;

use ygg_jit::{CompiledFn, HELPER_COUNT, Helper, RelocKind, RelocTarget};
use ygg_term::{HeapFull, Term};

use crate::atoms;
use crate::modload::{self, LoadedModule};
use crate::proc;

pub struct JitModule {
    pub fn_addrs: Vec<u64>,
    #[allow(dead_code)] // keeps the code pages alive conceptually; never freed yet
    span: (u64, usize),
}

/// Cranelift needs far more stack than a 64 KiB process stack: compilation
/// runs on a dedicated 1 MiB stack via a raw stack switch. Single-threaded by
/// construction (no safepoints inside, so no preemption mid-compile), but the
/// lock keeps it honest.
const COMPILE_STACK_SIZE: usize = 1024 * 1024;

#[repr(align(16))]
struct CompileStack([u8; COMPILE_STACK_SIZE]);

static mut COMPILE_STACK: CompileStack = CompileStack([0; COMPILE_STACK_SIZE]);
static COMPILE_LOCK: spin::Mutex<()> = spin::Mutex::new(());

#[unsafe(naked)]
unsafe extern "C" fn on_stack(arg: u64, f: extern "C" fn(u64) -> u64, new_rsp: u64) -> u64 {
    core::arch::naked_asm!(
        "mov rax, rsp",
        "mov rsp, rdx",
        "push rax",
        "call rsi",
        "pop rsp",
        "ret",
    )
}

struct CompileReq<'a> {
    module: &'a ygg_bytecode::Module,
    atom_map: &'a [u32],
    out: Option<Result<alloc::vec::Vec<CompiledFn>, ygg_jit::JitError>>,
}

extern "C" fn do_compile(p: u64) -> u64 {
    let req = unsafe { &mut *(p as *mut CompileReq) };
    req.out = Some(ygg_jit::compile_module(req.module, req.atom_map));
    0
}

/// Compile and publish a verified module. Returns None (and logs) on any
/// failure — the caller falls back to the interpreter.
pub fn compile_and_publish(module: &ygg_bytecode::Module, atom_map: &[u32]) -> Option<JitModule> {
    let _guard = COMPILE_LOCK.lock();
    let mut req = CompileReq {
        module,
        atom_map,
        out: None,
    };
    let top = unsafe {
        (&raw const COMPILE_STACK)
            .cast::<u8>()
            .add(COMPILE_STACK_SIZE)
    } as u64;
    unsafe { on_stack(&raw mut req as u64, do_compile, top) };
    let fns = match req.out.expect("compile never ran") {
        Ok(f) => f,
        Err(e) => {
            log::warn!("jit: compile failed ({e:?}), falling back to interpreter");
            return None;
        }
    };
    Some(publish(&fns))
}

fn publish(fns: &[CompiledFn]) -> JitModule {
    // Layout: 16-byte aligned functions in one span.
    let mut offsets = Vec::with_capacity(fns.len());
    let mut total = 0usize;
    for f in fns {
        total = (total + 15) & !15;
        offsets.push(total);
        total += f.code.len();
    }
    let pages = total.div_ceil(4096).max(1);
    let span = crate::mm::alloc_contig(pages, 1).expect("no frames for jit code");
    let base_write = crate::mm::phys_to_virt(span); // RW alias (HHDM)
    let base_exec = crate::vmm::map_code(span, pages as u64); // RX mapping

    unsafe {
        base_write.write_bytes(0xCC, pages * 4096); // int3 padding
        for (f, off) in fns.iter().zip(&offsets) {
            core::ptr::copy_nonoverlapping(f.code.as_ptr(), base_write.add(*off), f.code.len());
        }
    }

    let helper_addrs = helper_table();
    let fn_addrs: Vec<u64> = offsets.iter().map(|o| base_exec + *o as u64).collect();
    for (f, off) in fns.iter().zip(&offsets) {
        for r in &f.relocs {
            let target = match r.target {
                RelocTarget::Helper(h) => helper_addrs[h as usize],
                RelocTarget::Function(i) => fn_addrs[i as usize],
            };
            let write_at = unsafe { base_write.add(off + r.offset as usize) };
            let exec_at = base_exec + *off as u64 + r.offset as u64;
            match r.kind {
                RelocKind::Abs8 => unsafe {
                    write_at
                        .cast::<u64>()
                        .write_unaligned((target as i64 + r.addend) as u64);
                },
                RelocKind::PcRel4 => unsafe {
                    let rel = target as i64 + r.addend - exec_at as i64;
                    write_at.cast::<i32>().write_unaligned(rel as i32);
                },
            }
        }
    }

    log::info!(
        "jit: published {} fns, {} bytes at {:#x}",
        fns.len(),
        total,
        base_exec
    );
    JitModule {
        fn_addrs,
        span: (span, pages),
    }
}

fn helper_table() -> [u64; HELPER_COUNT] {
    let mut t = [0u64; HELPER_COUNT];
    t[Helper::SelfPid as usize] = rt_self as usize as u64;
    t[Helper::Send as usize] = rt_send as usize as u64;
    t[Helper::Recv as usize] = rt_recv as usize as u64;
    t[Helper::Spawn as usize] = rt_spawn as usize as u64;
    t[Helper::Safepoint as usize] = rt_safepoint as usize as u64;
    t[Helper::Print as usize] = rt_print as usize as u64;
    t[Helper::Eq as usize] = rt_eq as usize as u64;
    t[Helper::MakeTuple as usize] = rt_make_tuple as usize as u64;
    t[Helper::GetElem as usize] = rt_get_elem as usize as u64;
    t[Helper::Cons as usize] = rt_cons as usize as u64;
    t[Helper::Head as usize] = rt_head as usize as u64;
    t[Helper::Tail as usize] = rt_tail as usize as u64;
    t[Helper::PortOpen as usize] = rt_port_open as usize as u64;
    t[Helper::PortSubmit as usize] = rt_port_submit as usize as u64;
    t[Helper::CallExt as usize] = rt_call_ext as usize as u64;
    t[Helper::ExitAtom as usize] = rt_exit_atom as usize as u64;
    t[Helper::TrapBadarg as usize] = rt_trap_badarg as usize as u64;
    t[Helper::BinConst as usize] = rt_bin_const as usize as u64;
    t[Helper::BinFromList as usize] = rt_bin_from_list as usize as u64;
    t[Helper::BinToList as usize] = rt_bin_to_list as usize as u64;
    t[Helper::BinSize as usize] = rt_bin_size as usize as u64;
    t[Helper::BufToBin as usize] = rt_buf_to_bin as usize as u64;
    t[Helper::BinToBuf as usize] = rt_bin_to_buf as usize as u64;
    t[Helper::MapNew as usize] = rt_map_new as usize as u64;
    t[Helper::MapGet as usize] = rt_map_get as usize as u64;
    t[Helper::MapPut as usize] = rt_map_put as usize as u64;
    t[Helper::IsBinary as usize] = rt_is_binary as usize as u64;
    t[Helper::BinCat as usize] = rt_bin_cat as usize as u64;
    t[Helper::ListCat as usize] = rt_list_cat as usize as u64;
    t[Helper::BinPart as usize] = rt_bin_part as usize as u64;
    t[Helper::TailCallExt as usize] = rt_tail_call_ext as usize as u64;
    t[Helper::TailCallLocal as usize] = rt_tail_call_local as usize as u64;
    t[Helper::PortSubmit2 as usize] = rt_port_submit2 as usize as u64;
    t[Helper::BufWrite as usize] = rt_buf_write as usize as u64;
    t[Helper::SleepMs as usize] = rt_sleep_ms as usize as u64;
    t[Helper::BufNew as usize] = rt_buf_new as usize as u64;
    t[Helper::BufRead as usize] = rt_buf_read as usize as u64;
    t
}

// ---- runtime helpers ----

fn badarg() -> ! {
    proc::exit("badarg")
}

fn quota() -> ! {
    proc::exit("heap quota exceeded")
}

extern "C" fn rt_self() -> u64 {
    Term::pid(proc::current()).0
}

extern "C" fn rt_send(to: u64, msg: u64) -> u64 {
    match Term(to).as_pid() {
        Some(p) => {
            // Dead target is a no-op, BEAM-style.
            let _ = proc::send(p, Term(msg));
            0
        }
        None => badarg(),
    }
}

extern "C" fn rt_recv() -> u64 {
    proc::recv().0
}

extern "C" fn rt_spawn(fn_idx: u64, arg: u64) -> u64 {
    let arg = Term(arg);
    if arg.is_boxed() {
        badarg();
    }
    let Some(m) = modload::current_process_module() else {
        badarg()
    };
    if m.module.functions.get(fn_idx as usize).is_none() {
        badarg();
    }
    Term::pid(modload::spawn_fn(m, fn_idx as u32, arg)).0
}

extern "C" fn rt_safepoint() {
    proc::safepoint();
}

extern "C" fn rt_print(t: u64) {
    modload::print_term(Term(t));
}

extern "C" fn rt_eq(a: u64, b: u64) -> u64 {
    unsafe { ygg_term::eq(Term(a), Term(b)) as u64 }
}

extern "C" fn rt_make_tuple(ptr: *const Term, n: u64) -> u64 {
    let elems = unsafe { core::slice::from_raw_parts(ptr, n as usize) };
    proc::alloc_retry(|h| h.tuple(elems)).0
}

extern "C" fn rt_get_elem(t: u64, idx: u64) -> u64 {
    let t = Term(t);
    unsafe {
        if t.is_boxed() && t.kind() == ygg_term::Kind::Tuple && (idx as usize) < t.tuple_arity() {
            t.tuple_elem(idx as usize).0
        } else {
            badarg()
        }
    }
}

extern "C" fn rt_cons(h: u64, t: u64) -> u64 {
    proc::alloc_retry(|heap| heap.cons(Term(h), Term(t))).0
}

extern "C" fn rt_head(t: u64) -> u64 {
    let t = Term(t);
    unsafe {
        if t.is_boxed() && t.kind() == ygg_term::Kind::Cons {
            t.head().0
        } else {
            badarg()
        }
    }
}

extern "C" fn rt_tail(t: u64) -> u64 {
    let t = Term(t);
    unsafe {
        if t.is_boxed() && t.kind() == ygg_term::Kind::Cons {
            t.tail().0
        } else {
            badarg()
        }
    }
}

extern "C" fn rt_port_open(kind: u64) -> u64 {
    match crate::ports::open(kind as u8) {
        Some(t) => t.0,
        None => badarg(),
    }
}

extern "C" fn rt_port_submit(port: u64, op: u64, arg0: u64, tag: u64) -> u64 {
    let (Some(id), Some(a0), Some(tg)) = (
        Term(port).as_port(),
        Term(arg0).as_int(),
        Term(tag).as_int(),
    ) else {
        badarg()
    };
    let sqe = ygg_rings::Sqe {
        op: op as u32,
        tag: tg,
        arg0: a0 as u64,
        arg1: 0,
    };
    if crate::ports::submit(id, sqe) {
        0
    } else {
        badarg()
    }
}

extern "C" fn rt_port_submit2(port: u64, op: u64, arg0: u64, arg1: u64, tag: u64) -> u64 {
    let (Some(id), Some(o), Some(a0), Some(a1), Some(tg)) = (
        Term(port).as_port(),
        Term(op).as_int(),
        Term(arg0).as_int(),
        Term(arg1).as_int(),
        Term(tag).as_int(),
    ) else {
        badarg()
    };
    let sqe = ygg_rings::Sqe { op: o as u32, tag: tg, arg0: a0 as u64, arg1: a1 as u64 };
    if crate::ports::submit(id, sqe) {
        0
    } else {
        badarg()
    }
}

extern "C" fn rt_buf_write(buf: u64, off: u64, bin: u64) -> u64 {
    let (Some(id), Some(off)) = (Term(buf).as_int(), Term(off).as_int()) else {
        badarg()
    };
    let bin = Term(bin);
    let bytes = unsafe {
        if !bin.is_boxed() || bin.kind() != ygg_term::Kind::Binary || off < 0 {
            badarg()
        }
        bin.bin_bytes()
    };
    if crate::ports::buf_write(id as u64, off as usize, bytes) {
        Term::int(0).0
    } else {
        badarg()
    }
}

extern "C" fn rt_buf_new(size: u64) -> u64 {
    let Some(size) = Term(size).as_int() else { badarg() };
    if size < 0 || size > 64 << 20 {
        badarg()
    }
    Term::int(crate::ports::buf_create(alloc::vec![0u8; size as usize]) as i64).0
}

extern "C" fn rt_buf_read(buf: u64, off: u64, len: u64) -> u64 {
    let (Some(id), Some(off), Some(len)) =
        (Term(buf).as_int(), Term(off).as_int(), Term(len).as_int())
    else {
        badarg()
    };
    if off < 0 || len < 0 {
        badarg()
    }
    let Some(data) = crate::ports::buf_read(id as u64, off as usize, len as usize) else {
        badarg()
    };
    proc::alloc_retry(|h| h.binary(&data)).0
}

extern "C" fn rt_sleep_ms(ms: u64) {
    let Some(ms) = Term(ms).as_int() else { badarg() };
    if ms < 0 {
        badarg()
    }
    let _ = proc::recv_where(|_| false, Some(ms as u64));
}

extern "C" fn rt_call_ext(matom: u64, fatom: u64, ptr: *const Term, n: u64) -> u64 {
    let (Some(ma), Some(fa)) = (Term(matom).as_atom(), Term(fatom).as_atom()) else {
        badarg()
    };
    let args = unsafe { core::slice::from_raw_parts(ptr, n as usize) };
    let caller: Option<Arc<LoadedModule>> = modload::current_process_module();
    match modload::call_ext_dynamic(ma, fa, args) {
        Ok(t) => {
            if let Some(c) = caller {
                modload::note_running_pub(proc::current(), &c.name, c.version);
            }
            t.0
        }
        Err(trap) => modload::exit_for_trap(trap),
    }
}

extern "C" fn rt_exit_atom(a: u64) -> u64 {
    match Term(a).as_atom() {
        Some(idx) => proc::exit(atoms::name(idx)),
        None => badarg(),
    }
}

extern "C" fn rt_trap_badarg() -> u64 {
    badarg()
}

/// `ptr` points into the running module's bytecode (kept alive by the same
/// Arc that owns this generated code).
extern "C" fn rt_bin_const(ptr: *const u8, len: u64) -> u64 {
    let bytes = unsafe { core::slice::from_raw_parts(ptr, len as usize) };
    proc::alloc_retry(|h| h.binary(bytes)).0
}

extern "C" fn rt_bin_from_list(list: u64) -> u64 {
    let mut cur = Term(list);
    let mut bytes: alloc::vec::Vec<u8> = alloc::vec::Vec::new();
    unsafe {
        while cur.is_boxed() && cur.kind() == ygg_term::Kind::Cons {
            let Some(v) = cur.head().as_int() else { badarg() };
            if !(0..=255).contains(&v) {
                badarg();
            }
            bytes.push(v as u8);
            cur = cur.tail();
        }
    }
    if !cur.is_nil() {
        badarg();
    }
    proc::alloc_retry(|h| h.binary(&bytes)).0
}

extern "C" fn rt_bin_to_list(bin: u64) -> u64 {
    let t = Term(bin);
    let elems: alloc::vec::Vec<Term> = unsafe {
        if !t.is_boxed() || t.kind() != ygg_term::Kind::Binary {
            badarg();
        }
        t.bin_bytes().iter().map(|&b| Term::int(b as i64)).collect()
    };
    proc::alloc_retry(|h| h.list(&elems)).0
}

extern "C" fn rt_bin_size(bin: u64) -> u64 {
    let t = Term(bin);
    unsafe {
        if !t.is_boxed() || t.kind() != ygg_term::Kind::Binary {
            badarg();
        }
        Term::int(t.bin_bytes().len() as i64).0
    }
}

extern "C" fn rt_buf_to_bin(id: u64) -> u64 {
    let Some(id) = Term(id).as_int() else { badarg() };
    let Some(data) = crate::ports::buf_take(id as u64) else { badarg() };
    proc::alloc_retry(|h| h.binary(&data)).0
}

extern "C" fn rt_bin_to_buf(bin: u64) -> u64 {
    let t = Term(bin);
    let data = unsafe {
        if !t.is_boxed() || t.kind() != ygg_term::Kind::Binary {
            badarg();
        }
        t.bin_bytes().to_vec()
    };
    Term::int(crate::ports::buf_create(data) as i64).0
}

fn map_retry(f: impl Fn(&mut ygg_term::Heap) -> Result<Term, ygg_term::MapError>) -> u64 {
    loop {
        match proc::with_heap(&f) {
            Ok(t) => return t.0,
            Err(ygg_term::MapError::NonImmediateKey) => badarg(),
            Err(ygg_term::MapError::Heap(_)) => {
                if !proc::grow_current_heap() {
                    proc::exit("heap quota exceeded");
                }
            }
        }
    }
}

extern "C" fn rt_map_new(ptr: *const Term, n_pairs: u64) -> u64 {
    let flat = unsafe { core::slice::from_raw_parts(ptr, 2 * n_pairs as usize) };
    map_retry(|h| {
        let mut pairs: alloc::vec::Vec<(Term, Term)> =
            flat.chunks_exact(2).map(|c| (c[0], c[1])).collect();
        h.map_from_pairs(&mut pairs)
    })
}

extern "C" fn rt_map_get(map: u64, key: u64) -> u64 {
    let m = Term(map);
    unsafe {
        if !m.is_boxed() || m.kind() != ygg_term::Kind::Map {
            badarg();
        }
        match m.map_get(Term(key)) {
            Some(v) => v.0,
            None => badarg(),
        }
    }
}

extern "C" fn rt_map_put(map: u64, key: u64, val: u64) -> u64 {
    let m = Term(map);
    unsafe {
        if !m.is_boxed() || m.kind() != ygg_term::Kind::Map {
            badarg();
        }
        map_retry(|h| unsafe { h.map_put(m, Term(key), Term(val)) })
    }
}

extern "C" fn rt_is_binary(t: u64) -> u64 {
    let t = Term(t);
    unsafe { (t.is_boxed() && t.kind() == ygg_term::Kind::Binary) as u64 }
}

extern "C" fn rt_bin_cat(a: u64, b: u64) -> u64 {
    let (a, b) = (Term(a), Term(b));
    let joined: alloc::vec::Vec<u8> = unsafe {
        if !a.is_boxed()
            || a.kind() != ygg_term::Kind::Binary
            || !b.is_boxed()
            || b.kind() != ygg_term::Kind::Binary
        {
            badarg();
        }
        let mut v = a.bin_bytes().to_vec();
        v.extend_from_slice(b.bin_bytes());
        v
    };
    proc::alloc_retry(|h| h.binary(&joined)).0
}

extern "C" fn rt_list_cat(a: u64, b: u64) -> u64 {
    let mut elems: alloc::vec::Vec<Term> = alloc::vec::Vec::new();
    let mut cur = Term(a);
    unsafe {
        while cur.is_boxed() && cur.kind() == ygg_term::Kind::Cons {
            elems.push(cur.head());
            cur = cur.tail();
        }
    }
    if !cur.is_nil() {
        badarg();
    }
    proc::alloc_retry(|h| {
        let mut out = Term(b);
        for e in elems.iter().rev() {
            out = h.cons(*e, out)?;
        }
        Ok::<Term, ygg_term::HeapFull>(out)
    })
    .0
}

extern "C" fn rt_bin_part(bin: u64, off: u64, len: u64) -> u64 {
    let b = Term(bin);
    let (Some(off), Some(len)) = (Term(off).as_int(), Term(len).as_int()) else {
        badarg()
    };
    let part: alloc::vec::Vec<u8> = unsafe {
        if !b.is_boxed() || b.kind() != ygg_term::Kind::Binary || off < 0 || len < 0 {
            badarg();
        }
        let bytes = b.bin_bytes();
        let (off, len) = (off as usize, len as usize);
        if off + len > bytes.len() {
            badarg();
        }
        bytes[off..off + len].to_vec()
    };
    proc::alloc_retry(|h| h.binary(&part)).0
}

extern "C" fn rt_tail_call_ext(matom: u64, fatom: u64, ptr: *const Term, n: u64) {
    let (Some(ma), Some(fa)) = (Term(matom).as_atom(), Term(fatom).as_atom()) else {
        badarg()
    };
    let args = unsafe { core::slice::from_raw_parts(ptr, n as usize) };
    proc::set_tail_target(ma, fa, args.to_vec());
}

extern "C" fn rt_tail_call_local(fn_idx: u64, ptr: *const Term, n: u64) {
    let args = unsafe { core::slice::from_raw_parts(ptr, n as usize) };
    proc::set_tail_target_local(fn_idx as u32, args.to_vec());
}
