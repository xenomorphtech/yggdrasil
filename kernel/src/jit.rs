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
pub fn compile_and_publish(
    module: &ygg_bytecode::Module,
    atom_map: &[u32],
) -> Option<JitModule> {
    let _guard = COMPILE_LOCK.lock();
    let mut req = CompileReq { module, atom_map, out: None };
    let top = unsafe { (&raw const COMPILE_STACK).cast::<u8>().add(COMPILE_STACK_SIZE) } as u64;
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
    JitModule { fn_addrs, span: (span, pages) }
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
    let Some(m) = modload::current_process_module() else { badarg() };
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
    match proc::with_heap(|h| h.tuple(elems)) {
        Ok(t) => t.0,
        Err(HeapFull) => quota(),
    }
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
    match proc::with_heap(|heap| heap.cons(Term(h), Term(t))) {
        Ok(c) => c.0,
        Err(HeapFull) => quota(),
    }
}

extern "C" fn rt_head(t: u64) -> u64 {
    let t = Term(t);
    unsafe {
        if t.is_boxed() && t.kind() == ygg_term::Kind::Cons { t.head().0 } else { badarg() }
    }
}

extern "C" fn rt_tail(t: u64) -> u64 {
    let t = Term(t);
    unsafe {
        if t.is_boxed() && t.kind() == ygg_term::Kind::Cons { t.tail().0 } else { badarg() }
    }
}

extern "C" fn rt_port_open(kind: u64) -> u64 {
    match crate::ports::open(kind as u8) {
        Some(t) => t.0,
        None => badarg(),
    }
}

extern "C" fn rt_port_submit(port: u64, op: u64, arg0: u64, tag: u64) -> u64 {
    let (Some(id), Some(a0), Some(tg)) =
        (Term(port).as_port(), Term(arg0).as_int(), Term(tag).as_int())
    else {
        badarg()
    };
    let sqe = ygg_rings::Sqe { op: op as u32, tag: tg, arg0: a0 as u64, arg1: 0 };
    if crate::ports::submit(id, sqe) { 0 } else { badarg() }
}

extern "C" fn rt_call_ext(matom: u64, fatom: u64, ptr: *const Term, n: u64) -> u64 {
    let (Some(ma), Some(fa)) = (Term(matom).as_atom(), Term(fatom).as_atom()) else { badarg() };
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
