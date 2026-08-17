//! ygg-run: execute a luxpack (or single .yggm) on the host under the
//! Yggdrasil interpreter — the development-loop twin of running it on the OS.
//!
//! Usage: ygg-run <pack.luxpack> [entry_module] [fn_name]
//! Traps print the innermost module:function for fast diagnosis.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use anyhow::{Context, Result, bail};
use ygg_bytecode::Module;
use ygg_interp::{SystemApi, Trap};
use ygg_term::{Heap, Term};

fn host_ticks_ms() -> i64 {
    use std::sync::OnceLock;
    use std::time::Instant;
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_millis() as i64
}

#[derive(Default)]
struct Atoms {
    names: Vec<String>,
    index: HashMap<String, u32>,
}

impl Atoms {
    fn intern(&mut self, s: &str) -> u32 {
        if let Some(&i) = self.index.get(s) {
            return i;
        }
        let i = self.names.len() as u32;
        self.names.push(s.to_string());
        self.index.insert(s.to_string(), i);
        i
    }
    fn name(&self, i: u32) -> &str {
        self.names.get(i as usize).map(String::as_str).unwrap_or("?")
    }
}

struct Loaded {
    module: Module,
    atom_map: Vec<u32>,
}

/// A stashed tail-call target (mirrors kernel `proc::TailTarget`).
enum HostTail {
    Ext(u32, u32, Vec<Term>),
    Local(u32, Vec<Term>),
}

struct World {
    atoms: Atoms,
    modules: HashMap<String, Rc<Loaded>>,
    heap: Heap,
    #[allow(dead_code)]
    heap_buf: Vec<u64>,
    buffers: HashMap<i64, Vec<u8>>,
    next_buf: i64,
    call_depth: usize,
    tail_target: Option<HostTail>,
}

struct HostApi {
    world: Rc<RefCell<World>>,
    module: Rc<Loaded>,
}

impl SystemApi for HostApi {
    fn heap(&mut self) -> &mut Heap {
        // Single shared heap for the whole host run; lifetime is the process.
        unsafe { &mut *(&mut self.world.borrow_mut().heap as *mut Heap) }
    }
    fn self_pid(&self) -> u64 {
        1
    }
    fn send(&mut self, _to: Term, _msg: Term) -> Result<(), Trap> {
        eprintln!("ygg-run: send() ignored (no processes on host)");
        Ok(())
    }
    fn recv(&mut self) -> Term {
        panic!("ygg-run: recv() unsupported on host");
    }
    fn spawn(&mut self, _fn_idx: u32, _arg: Term) -> Result<u64, Trap> {
        panic!("ygg-run: spawn() unsupported on host");
    }
    fn safepoint(&mut self) {}
    fn atom_global(&mut self, local: u32) -> u32 {
        self.module.atom_map.get(local as usize).copied().unwrap_or(0)
    }
    fn print(&mut self, t: Term) {
        let mut s = String::new();
        let world = self.world.borrow();
        let _ = unsafe { ygg_term::fmt_term(t, &mut s, &|a| leak(world.atoms.name(a))) };
        println!("[bc] {s}");
    }
    fn port_open(&mut self, _kind: u8) -> Result<Term, Trap> {
        Err(Trap::Badarg)
    }
    fn port_submit(&mut self, _p: Term, _o: u8, _a: Term, _t: Term) -> Result<(), Trap> {
        Err(Trap::Badarg)
    }
    fn tail_call(&mut self, module_atom: u32, fname_atom: u32, args: &[Term]) {
        self.world.borrow_mut().tail_target =
            Some(HostTail::Ext(module_atom, fname_atom, args.to_vec()));
    }
    fn tail_call_local(&mut self, fn_idx: u32, args: &[Term]) {
        self.world.borrow_mut().tail_target = Some(HostTail::Local(fn_idx, args.to_vec()));
    }
    fn sleep_ms(&mut self, ms: u64) {
        std::thread::sleep(std::time::Duration::from_millis(ms));
    }
    fn ticks(&self) -> Term {
        Term::int(host_ticks_ms())
    }
    fn call_ext(&mut self, module_atom: u32, fname_atom: u32, args: &[Term]) -> Result<Term, Trap> {
        let (mname, fname, target, depth) = {
            let mut w = self.world.borrow_mut();
            w.call_depth += 1;
            let mname = w.atoms.name(module_atom).to_string();
            let fname = w.atoms.name(fname_atom).to_string();
            let target = w.modules.get(&mname).cloned();
            (mname, fname, target, w.call_depth)
        };
        let Some(target) = target else {
            eprintln!("TRAP: unknown module {mname} (depth {depth})");
            return Err(Trap::Badarg);
        };
        let Some(fn_idx) = target.module.function_named(&fname) else {
            eprintln!("TRAP: no function {mname}:{fname}");
            return Err(Trap::Badarg);
        };
        let r = trampoline(&self.world, target, fn_idx, args.to_vec());
        self.world.borrow_mut().call_depth -= 1;
        if let Err(t) = &r {
            eprintln!("TRAP {t:?} in {mname}:{fname}/{} (depth {depth})", args.len());
        }
        r
    }
    fn buf_to_bin(&mut self, id: i64) -> Result<Term, Trap> {
        let data =
            self.world.borrow_mut().buffers.remove(&id).ok_or(Trap::Badarg)?;
        self.heap().binary(&data).map_err(|_| Trap::HeapFull)
    }
    fn buf_new(&mut self, size: Term) -> Result<Term, Trap> {
        let size = size.as_int().ok_or(Trap::Badarg)?;
        if size < 0 || size > 64 << 20 {
            return Err(Trap::Badarg);
        }
        let mut w = self.world.borrow_mut();
        let id = w.next_buf;
        w.next_buf += 1;
        w.buffers.insert(id, vec![0u8; size as usize]);
        Ok(Term::int(id))
    }
    fn buf_read(&mut self, buf: Term, off: Term, len: Term) -> Result<Term, Trap> {
        let id = buf.as_int().ok_or(Trap::Badarg)?;
        let off = off.as_int().ok_or(Trap::Badarg)? as usize;
        let len = len.as_int().ok_or(Trap::Badarg)? as usize;
        let data = {
            let w = self.world.borrow();
            let data = w.buffers.get(&id).ok_or(Trap::Badarg)?;
            let end = off.checked_add(len).ok_or(Trap::Badarg)?;
            if end > data.len() {
                return Err(Trap::Badarg);
            }
            data[off..end].to_vec()
        };
        self.heap().binary(&data).map_err(|_| Trap::HeapFull)
    }
    fn buf_write(&mut self, buf: Term, off: Term, bin: Term) -> Result<Term, Trap> {
        let id = buf.as_int().ok_or(Trap::Badarg)?;
        let off = off.as_int().ok_or(Trap::Badarg)? as usize;
        let bytes = unsafe {
            if !bin.is_boxed() || bin.kind() != ygg_term::Kind::Binary {
                return Err(Trap::Badarg);
            }
            bin.bin_bytes().to_vec()
        };
        let mut w = self.world.borrow_mut();
        let data = w.buffers.get_mut(&id).ok_or(Trap::Badarg)?;
        let end = off.checked_add(bytes.len()).ok_or(Trap::Badarg)?;
        if end > data.len() {
            return Err(Trap::Badarg);
        }
        data[off..end].copy_from_slice(&bytes);
        Ok(Term::int(0))
    }
    fn bin_to_buf(&mut self, bin: Term) -> Result<Term, Trap> {
        let bytes = unsafe {
            if !bin.is_boxed() || bin.kind() != ygg_term::Kind::Binary {
                return Err(Trap::Badarg);
            }
            bin.bin_bytes().to_vec()
        };
        let mut w = self.world.borrow_mut();
        let id = w.next_buf;
        w.next_buf += 1;
        w.buffers.insert(id, bytes);
        Ok(Term::int(id))
    }
}

fn leak(s: &str) -> &'static str {
    Box::leak(s.to_string().into_boxed_str())
}

/// Run with TAIL_CALL_EXT trampolining (constant Rust stack per tail chain).
fn trampoline(
    world: &Rc<RefCell<World>>,
    mut target: Rc<Loaded>,
    mut fn_idx: usize,
    mut args: Vec<Term>,
) -> Result<Term, Trap> {
    loop {
        let mut api = HostApi { world: world.clone(), module: target.clone() };
        let r = ygg_interp::run_function(&target.module, fn_idx, &args, &mut api)?;
        if r != ygg_interp::TAIL_SENTINEL {
            return Ok(r);
        }
        let stashed = world.borrow_mut().tail_target.take().ok_or(Trap::BadCode)?;
        match stashed {
            HostTail::Ext(ma, fa, targs) => {
                let (mname, fname) = {
                    let w = world.borrow();
                    (w.atoms.name(ma).to_string(), w.atoms.name(fa).to_string())
                };
                let next = world.borrow().modules.get(&mname).cloned().ok_or(Trap::Badarg)?;
                let idx = next.module.function_named(&fname).ok_or(Trap::Badarg)?;
                target = next;
                fn_idx = idx;
                args = targs;
            }
            HostTail::Local(idx, targs) => {
                fn_idx = idx as usize;
                args = targs;
            }
        }
    }
}

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().collect();
    let use_jit = if let Some(i) = args.iter().position(|a| a == "--jit") {
        args.remove(i);
        true
    } else {
        false
    };
    let path = args.get(1).context("usage: ygg-run [--jit] <pack.luxpack> [entry] [fn]")?;
    let bytes = std::fs::read(path)?;

    let int_args: Vec<Term> =
        args.iter().skip(4).filter_map(|a| a.parse::<i64>().ok().map(Term::int)).collect();

    if use_jit {
        return run_jit(&bytes, args.get(2).cloned(), args.get(3).cloned(), int_args);
    }

    let mut heap_buf = vec![0u64; 32 * 1024 * 1024]; // 256 MiB host heap
    let heap = unsafe { Heap::new(heap_buf.as_mut_ptr().cast(), heap_buf.len() * 8) };
    let world = Rc::new(RefCell::new(World {
        atoms: Atoms::default(),
        modules: HashMap::new(),
        heap,
        heap_buf,
        buffers: HashMap::new(),
        next_buf: 1,
        call_depth: 0,
        tail_target: None,
    }));

    // Parse luxpack.
    let mut at = 0usize;
    let take = |at: &mut usize, n: usize| -> Result<&[u8]> {
        let s = bytes.get(*at..*at + n).context("truncated pack")?;
        *at += n;
        Ok(s)
    };
    let u32at = |at: &mut usize| -> Result<usize> {
        Ok(u32::from_le_bytes(take(at, 4)?.try_into()?) as usize)
    };
    if take(&mut at, 7)? != b"LUXPK1\n" {
        bail!("bad magic");
    }
    let elen = u32at(&mut at)?;
    let entry = String::from_utf8(take(&mut at, elen)?.to_vec())?;
    let count = u32at(&mut at)?;
    for _ in 0..count {
        let nlen = u32at(&mut at)?;
        let name = String::from_utf8(take(&mut at, nlen)?.to_vec())?;
        let dlen = u32at(&mut at)?;
        let data = take(&mut at, dlen)?;
        let module = Module::decode(data).map_err(|e| anyhow::anyhow!("{name}: {e:?}"))?;
        ygg_bytecode::verify::verify(&module).map_err(|e| anyhow::anyhow!("{name}: {e:?}"))?;
        let mut w = world.borrow_mut();
        let atom_map = module.atoms.iter().map(|a| w.atoms.intern(a)).collect();
        w.modules.insert(name.clone(), Rc::new(Loaded { module, atom_map }));
    }

    let entry = args.get(2).cloned().unwrap_or(entry);
    let fname = args.get(3).cloned().unwrap_or_else(|| "apply".to_string());
    let target =
        world.borrow().modules.get(&entry).cloned().context("entry module missing")?;
    let fn_idx = target
        .module
        .function_named(&fname)
        .with_context(|| format!("no {fname} in entry"))?;
    match trampoline(&world, target.clone(), fn_idx, int_args) {
        Ok(t) => {
            let mut s = String::new();
            let w = world.borrow();
            let _ = unsafe { ygg_term::fmt_term(t, &mut s, &|a| leak(w.atoms.name(a))) };
            println!("result: {s}");
            if s != "true" {
                std::process::exit(2);
            }
            Ok(())
        }
        Err(t) => bail!("entry trapped: {t:?}"),
    }
}

fn run_jit(
    bytes: &[u8],
    entry_override: Option<String>,
    fname: Option<String>,
    int_args: Vec<Term>,
) -> Result<()> {
    let mut at = 0usize;
    let take = |at: &mut usize, n: usize| -> Result<&[u8]> {
        let s = bytes.get(*at..*at + n).context("truncated pack")?;
        *at += n;
        Ok(s)
    };
    let u32at = |at: &mut usize| -> Result<usize> {
        Ok(u32::from_le_bytes(take(at, 4)?.try_into()?) as usize)
    };
    if take(&mut at, 7)? != b"LUXPK1\n" {
        bail!("bad magic");
    }
    let elen = u32at(&mut at)?;
    let entry = String::from_utf8(take(&mut at, elen)?.to_vec())?;
    let count = u32at(&mut at)?;

    let mut heap_buf = vec![0u64; 32 * 1024 * 1024];
    let heap = unsafe { Heap::new(heap_buf.as_mut_ptr().cast(), heap_buf.len() * 8) };
    let mut world = Box::new(jit_host::JWorld {
        atoms: Atoms::default(),
        modules: HashMap::new(),
        heap,
        heap_buf,
        stack: Vec::new(),
        tail_target: None,
    });
    for _ in 0..count {
        let nlen = u32at(&mut at)?;
        let name = String::from_utf8(take(&mut at, nlen)?.to_vec())?;
        let dlen = u32at(&mut at)?;
        let data = take(&mut at, dlen)?;
        let module = Module::decode(data).map_err(|e| anyhow::anyhow!("{name}: {e:?}"))?;
        ygg_bytecode::verify::verify(&module).map_err(|e| anyhow::anyhow!("{name}: {e:?}"))?;
        let atom_map: Vec<u32> = module.atoms.iter().map(|a| world.atoms.intern(a)).collect();
        let fns = ygg_jit::compile_module(&module, &atom_map)
            .map_err(|e| anyhow::anyhow!("{name}: jit {e:?}"))?;
        let fn_addrs = jit_host::load_exec(&fns);
        world.modules.insert(name, jit_host::JMod { module, atom_map, fn_addrs });
    }
    let entry = entry_override.unwrap_or(entry);
    let fname = fname.unwrap_or_else(|| "apply".to_string());
    let result = jit_host::run(world, &entry, &fname, int_args);
    // NOTE: the world was leaked into the static; fine for a CLI run.
    println!("jit result raw: {:#x}", result.0);
    match jit_host::atom_index("true") {
        Some(t) if result == Term::atom(t) => {
            println!("result: true");
            Ok(())
        }
        _ => {
            println!("result != true");
            std::process::exit(2);
        }
    }
}

// ---- host JIT mode: compile every module with ygg-jit, mmap, patch, run ----
// Reproduces the on-OS tier-1 path in userspace for differential debugging.

mod jit_host {
    use super::{Atoms, leak};
    use std::collections::HashMap;
    use ygg_bytecode::Module;
    use ygg_jit::{CompiledFn, HELPER_COUNT, Helper, RelocKind, RelocTarget};
    use ygg_term::{Heap, Term};

    pub struct JMod {
        pub module: Module,
        pub atom_map: Vec<u32>,
        pub fn_addrs: Vec<u64>,
    }

    pub struct JWorld {
        pub atoms: Atoms,
        pub modules: HashMap<String, JMod>,
        pub heap: Heap,
        pub heap_buf: Vec<u64>,
        pub stack: Vec<String>,
        pub tail_target: Option<super::HostTail>,
    }

    static mut W: *mut JWorld = std::ptr::null_mut();

    fn w() -> &'static mut JWorld {
        unsafe { &mut *W }
    }

    /// Look up a global atom in the (leaked) world after `run`.
    pub fn atom_index(name: &str) -> Option<u32> {
        w().atoms.index.get(name).copied()
    }

    fn die(msg: &str) -> ! {
        eprintln!("JIT TRAP: {msg}");
        eprintln!("call stack (innermost last):");
        for f in &w().stack {
            eprintln!("  {f}");
        }
        std::process::exit(3);
    }

    pub fn load_exec(fns: &[CompiledFn]) -> Vec<u64> {
        let mut offsets = Vec::new();
        let mut total = 0usize;
        for f in fns {
            total = (total + 15) & !15;
            offsets.push(total);
            total += f.code.len();
        }
        let size = total.max(1).next_multiple_of(4096);
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        } as *mut u8;
        assert!(!base.is_null() && base as isize != -1);
        let fn_addrs: Vec<u64> = offsets.iter().map(|o| base as u64 + *o as u64).collect();
        let helpers = helper_table();
        unsafe {
            for (f, off) in fns.iter().zip(&offsets) {
                std::ptr::copy_nonoverlapping(f.code.as_ptr(), base.add(*off), f.code.len());
            }
            for (f, off) in fns.iter().zip(&offsets) {
                for r in &f.relocs {
                    let target = match r.target {
                        RelocTarget::Helper(h) => helpers[h as usize],
                        RelocTarget::Function(i) => fn_addrs[i as usize],
                        // Host mode never passes a resolver, so no bound sites.
                        RelocTarget::Address(a) => a,
                    };
                    let at = base.add(off + r.offset as usize);
                    match r.kind {
                        RelocKind::Abs8 => {
                            at.cast::<u64>().write_unaligned((target as i64 + r.addend) as u64)
                        }
                        RelocKind::PcRel4 => {
                            let rel = target as i64 + r.addend - at as i64;
                            at.cast::<i32>().write_unaligned(rel as i32);
                        }
                    }
                }
            }
        }
        fn_addrs
    }

    pub fn invoke(addr: u64, args: &[Term]) -> Term {
        unsafe {
            let r = match args {
                [] => core::mem::transmute::<u64, extern "C" fn() -> u64>(addr)(),
                [a0] => core::mem::transmute::<u64, extern "C" fn(u64) -> u64>(addr)(a0.0),
                [a0, a1] => core::mem::transmute::<u64, extern "C" fn(u64, u64) -> u64>(addr)(a0.0, a1.0),
                [a0, a1, a2] => core::mem::transmute::<u64, extern "C" fn(u64, u64, u64) -> u64>(addr)(a0.0, a1.0, a2.0),
                [a0, a1, a2, a3] => core::mem::transmute::<u64, extern "C" fn(u64, u64, u64, u64) -> u64>(addr)(a0.0, a1.0, a2.0, a3.0),
                [a0, a1, a2, a3, a4] => core::mem::transmute::<u64, extern "C" fn(u64, u64, u64, u64, u64) -> u64>(addr)(a0.0, a1.0, a2.0, a3.0, a4.0),
                [a0, a1, a2, a3, a4, a5] => core::mem::transmute::<u64, extern "C" fn(u64, u64, u64, u64, u64, u64) -> u64>(addr)(a0.0, a1.0, a2.0, a3.0, a4.0, a5.0),
                [a0, a1, a2, a3, a4, a5, a6] => core::mem::transmute::<u64, extern "C" fn(u64, u64, u64, u64, u64, u64, u64) -> u64>(addr)(a0.0, a1.0, a2.0, a3.0, a4.0, a5.0, a6.0),
                [a0, a1, a2, a3, a4, a5, a6, a7] => core::mem::transmute::<u64, extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64) -> u64>(addr)(a0.0, a1.0, a2.0, a3.0, a4.0, a5.0, a6.0, a7.0),
                [a0, a1, a2, a3, a4, a5, a6, a7, a8] => core::mem::transmute::<u64, extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64) -> u64>(addr)(a0.0, a1.0, a2.0, a3.0, a4.0, a5.0, a6.0, a7.0, a8.0),
                [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9] => core::mem::transmute::<u64, extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) -> u64>(addr)(a0.0, a1.0, a2.0, a3.0, a4.0, a5.0, a6.0, a7.0, a8.0, a9.0),
                [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10] => core::mem::transmute::<u64, extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) -> u64>(addr)(a0.0, a1.0, a2.0, a3.0, a4.0, a5.0, a6.0, a7.0, a8.0, a9.0, a10.0),
                [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11] => core::mem::transmute::<u64, extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) -> u64>(addr)(a0.0, a1.0, a2.0, a3.0, a4.0, a5.0, a6.0, a7.0, a8.0, a9.0, a10.0, a11.0),
                [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12] => core::mem::transmute::<u64, extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) -> u64>(addr)(a0.0, a1.0, a2.0, a3.0, a4.0, a5.0, a6.0, a7.0, a8.0, a9.0, a10.0, a11.0, a12.0),
                [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13] => core::mem::transmute::<u64, extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) -> u64>(addr)(a0.0, a1.0, a2.0, a3.0, a4.0, a5.0, a6.0, a7.0, a8.0, a9.0, a10.0, a11.0, a12.0, a13.0),
                [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14] => core::mem::transmute::<u64, extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) -> u64>(addr)(a0.0, a1.0, a2.0, a3.0, a4.0, a5.0, a6.0, a7.0, a8.0, a9.0, a10.0, a11.0, a12.0, a13.0, a14.0),
                [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15] => core::mem::transmute::<u64, extern "C" fn(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) -> u64>(addr)(a0.0, a1.0, a2.0, a3.0, a4.0, a5.0, a6.0, a7.0, a8.0, a9.0, a10.0, a11.0, a12.0, a13.0, a14.0, a15.0),
                _ => die("arity > 16"),
            };
            Term(r)
        }
    }


    const SENTINEL: u64 = 7;

    pub fn run(world: Box<JWorld>, entry: &str, fname: &str, args: Vec<Term>) -> Term {
        unsafe { W = Box::into_raw(world) };
        let m = w().modules.get(entry).unwrap_or_else(|| die("entry module missing"));
        let fn_idx = m.module.function_named(fname).unwrap_or_else(|| die("no entry fn"));
        w().stack.push(format!("{entry}:{fname}"));
        trampoline(m, fn_idx, args)
    }

    fn trampoline(mut m: &'static JMod, mut fn_idx: usize, mut args: Vec<Term>) -> Term {
        loop {
            let r = invoke(m.fn_addrs[fn_idx], &args);
            if r.0 != SENTINEL {
                return r;
            }
            match w().tail_target.take().unwrap_or_else(|| die("sentinel without stash")) {
                super::HostTail::Ext(ma, fa, targs) => {
                    let mname = w().atoms.name(ma).to_string();
                    let fname = w().atoms.name(fa).to_string();
                    let next =
                        w().modules.get(&mname).unwrap_or_else(|| die("tail: unknown module"));
                    let idx =
                        next.module.function_named(&fname).unwrap_or_else(|| die("tail: no fn"));
                    m = next;
                    fn_idx = idx;
                    args = targs;
                }
                super::HostTail::Local(idx, targs) => {
                    fn_idx = idx as usize;
                    args = targs;
                }
            }
        }
    }

    fn heap() -> &'static mut Heap {
        &mut w().heap
    }

    extern "C" fn h_self() -> u64 {
        Term::pid(1).0
    }
    extern "C" fn h_send(_to: u64, _m: u64) -> u64 {
        0
    }
    extern "C" fn h_recv() -> u64 {
        die("recv on host")
    }
    extern "C" fn h_spawn(_f: u64, _a: u64) -> u64 {
        die("spawn on host")
    }
    extern "C" fn h_safepoint() {}
    extern "C" fn h_print(t: u64) {
        let mut s = String::new();
        let _ = unsafe { ygg_term::fmt_term(Term(t), &mut s, &|a| leak(w().atoms.name(a))) };
        println!("[bc] {s}");
    }
    extern "C" fn h_eq(a: u64, b: u64) -> u64 {
        unsafe { ygg_term::eq(Term(a), Term(b)) as u64 }
    }
    extern "C" fn h_make_tuple(ptr: *const Term, n: u64) -> u64 {
        let elems = unsafe { std::slice::from_raw_parts(ptr, n as usize) };
        heap().tuple(elems).unwrap_or_else(|_| die("heap full")).0
    }
    extern "C" fn h_get_elem(t: u64, idx: u64) -> u64 {
        let t = Term(t);
        unsafe {
            if t.is_boxed() && t.kind() == ygg_term::Kind::Tuple && (idx as usize) < t.tuple_arity()
            {
                t.tuple_elem(idx as usize).0
            } else {
                die("get_elem badarg")
            }
        }
    }
    extern "C" fn h_cons(h: u64, t: u64) -> u64 {
        heap().cons(Term(h), Term(t)).unwrap_or_else(|_| die("heap full")).0
    }
    extern "C" fn h_head(t: u64) -> u64 {
        let t = Term(t);
        unsafe {
            if t.is_boxed() && t.kind() == ygg_term::Kind::Cons {
                t.head().0
            } else {
                die("head badarg")
            }
        }
    }
    extern "C" fn h_tail(t: u64) -> u64 {
        let t = Term(t);
        unsafe {
            if t.is_boxed() && t.kind() == ygg_term::Kind::Cons {
                t.tail().0
            } else {
                die("tail badarg")
            }
        }
    }
    extern "C" fn h_port_open(_k: u64) -> u64 {
        die("port_open on host")
    }
    extern "C" fn h_port_submit(_p: u64, _o: u64, _a: u64, _t: u64) -> u64 {
        die("port_submit on host")
    }
    extern "C" fn h_port_submit2(_p: u64, _o: u64, _a0: u64, _a1: u64, _t: u64) -> u64 {
        die("port_submit2 on host")
    }
    extern "C" fn h_buf_write(_b: u64, _o: u64, _s: u64) -> u64 {
        die("buf_write on host")
    }
    extern "C" fn h_buf_new(_s: u64) -> u64 {
        die("buf_new on host")
    }
    extern "C" fn h_buf_read(_b: u64, _o: u64, _l: u64) -> u64 {
        die("buf_read on host")
    }
    extern "C" fn h_sleep_ms(ms: u64) {
        if let Some(ms) = Term(ms).as_int() {
            std::thread::sleep(std::time::Duration::from_millis(ms.max(0) as u64));
        }
    }
    extern "C" fn h_ticks() -> u64 {
        Term::int(super::host_ticks_ms()).0
    }
    extern "C" fn h_resume_tail(_token: u64) -> u64 {
        // The host never passes a resolver, so no direct-bound sites exist.
        die("resume_tail on host")
    }
    extern "C" fn h_bin_at(bin: u64, idx: u64) -> u64 {
        let b = Term(bin);
        let Some(idx) = Term(idx).as_int() else { die("bin_at: bad index") };
        unsafe {
            if !b.is_boxed() || b.kind() != ygg_term::Kind::Binary || idx < 0 {
                die("bin_at badarg");
            }
            let bytes = b.bin_bytes();
            if idx as usize >= bytes.len() {
                die("bin_at out of range");
            }
            Term::int(bytes[idx as usize] as i64).0
        }
    }
    extern "C" fn h_call_ext(ma: u64, fa: u64, ptr: *const Term, n: u64) -> u64 {
        let args = unsafe { std::slice::from_raw_parts(ptr, n as usize) }.to_vec();
        let mname = w().atoms.name(Term(ma).as_atom().unwrap_or(u32::MAX)).to_string();
        let fname = w().atoms.name(Term(fa).as_atom().unwrap_or(u32::MAX)).to_string();
        let Some(m) = w().modules.get(&mname) else {
            die(&format!("unknown module {mname}"))
        };
        let Some(fn_idx) = m.module.function_named(&fname) else {
            die(&format!("no fn {mname}:{fname}"))
        };
        if m.module.functions[fn_idx].arity as usize != args.len() {
            die(&format!("arity mismatch {mname}:{fname}"));
        }
        w().stack.push(format!("{mname}:{fname}/{}", args.len()));
        let r = trampoline(m, fn_idx, args);
        w().stack.pop();
        r.0
    }
    extern "C" fn h_exit_atom(a: u64) -> u64 {
        let name = Term(a).as_atom().map(|i| w().atoms.name(i).to_string());
        die(&format!("exit_atom({name:?})"))
    }
    extern "C" fn h_trap_badarg() -> u64 {
        die("badarg (inline tag check)")
    }
    extern "C" fn h_tail_call_ext(ma: u64, fa: u64, ptr: *const Term, n: u64) {
        let args = unsafe { std::slice::from_raw_parts(ptr, n as usize) }.to_vec();
        let (Some(ma), Some(fa)) = (Term(ma).as_atom(), Term(fa).as_atom()) else {
            die("tail_call_ext: bad atoms")
        };
        w().tail_target = Some(super::HostTail::Ext(ma, fa, args));
    }
    extern "C" fn h_tail_call_local(fn_idx: u64, ptr: *const Term, n: u64) {
        let args = unsafe { std::slice::from_raw_parts(ptr, n as usize) }.to_vec();
        w().tail_target = Some(super::HostTail::Local(fn_idx as u32, args));
    }
    extern "C" fn h_bin_const(ptr: *const u8, len: u64) -> u64 {
        let bytes = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
        heap().binary(bytes).unwrap_or_else(|_| die("heap full")).0
    }
    extern "C" fn h_bin_from_list(list: u64) -> u64 {
        let mut cur = Term(list);
        let mut bytes = Vec::new();
        unsafe {
            while cur.is_boxed() && cur.kind() == ygg_term::Kind::Cons {
                let Some(v) = cur.head().as_int() else { die("bin_from_list: non-int") };
                if !(0..=255).contains(&v) {
                    die("bin_from_list: out of range");
                }
                bytes.push(v as u8);
                cur = cur.tail();
            }
        }
        if !cur.is_nil() {
            die("bin_from_list: improper list");
        }
        heap().binary(&bytes).unwrap_or_else(|_| die("heap full")).0
    }
    extern "C" fn h_bin_to_list(bin: u64) -> u64 {
        let t = Term(bin);
        let elems: Vec<Term> = unsafe {
            if !t.is_boxed() || t.kind() != ygg_term::Kind::Binary {
                die("bin_to_list badarg");
            }
            t.bin_bytes().iter().map(|&b| Term::int(b as i64)).collect()
        };
        heap().list(&elems).unwrap_or_else(|_| die("heap full")).0
    }
    extern "C" fn h_bin_size(bin: u64) -> u64 {
        let t = Term(bin);
        unsafe {
            if !t.is_boxed() || t.kind() != ygg_term::Kind::Binary {
                die("bin_size badarg");
            }
            Term::int(t.bin_bytes().len() as i64).0
        }
    }
    extern "C" fn h_buf_to_bin(_id: u64) -> u64 {
        die("buf_to_bin on host")
    }
    extern "C" fn h_bin_to_buf(_b: u64) -> u64 {
        die("bin_to_buf on host")
    }
    extern "C" fn h_map_new(ptr: *const Term, n_pairs: u64) -> u64 {
        let flat = unsafe { std::slice::from_raw_parts(ptr, 2 * n_pairs as usize) };
        let mut pairs: Vec<(Term, Term)> = flat.chunks_exact(2).map(|c| (c[0], c[1])).collect();
        heap().map_from_pairs(&mut pairs).unwrap_or_else(|_| die("map_new failed")).0
    }
    extern "C" fn h_map_get(map: u64, key: u64) -> u64 {
        let m = Term(map);
        unsafe {
            if !m.is_boxed() || m.kind() != ygg_term::Kind::Map {
                die("map_get: not a map");
            }
            match m.map_get(Term(key)) {
                Some(v) => v.0,
                None => {
                    let k = Term(key)
                        .as_atom()
                        .map(|i| w().atoms.name(i).to_string());
                    die(&format!("map_get: missing key {k:?}"))
                }
            }
        }
    }
    extern "C" fn h_map_put(map: u64, key: u64, val: u64) -> u64 {
        let m = Term(map);
        unsafe {
            if !m.is_boxed() || m.kind() != ygg_term::Kind::Map {
                die("map_put: not a map");
            }
            heap().map_put(m, Term(key), Term(val)).unwrap_or_else(|_| die("map_put failed")).0
        }
    }
    extern "C" fn h_is_binary(t: u64) -> u64 {
        let t = Term(t);
        unsafe { (t.is_boxed() && t.kind() == ygg_term::Kind::Binary) as u64 }
    }
    extern "C" fn h_bin_cat(a: u64, b: u64) -> u64 {
        let (a, b) = (Term(a), Term(b));
        let joined: Vec<u8> = unsafe {
            if !a.is_boxed()
                || a.kind() != ygg_term::Kind::Binary
                || !b.is_boxed()
                || b.kind() != ygg_term::Kind::Binary
            {
                die("bin_cat badarg");
            }
            let mut v = a.bin_bytes().to_vec();
            v.extend_from_slice(b.bin_bytes());
            v
        };
        heap().binary(&joined).unwrap_or_else(|_| die("heap full")).0
    }
    extern "C" fn h_list_cat(a: u64, b: u64) -> u64 {
        let mut elems = Vec::new();
        let mut cur = Term(a);
        unsafe {
            while cur.is_boxed() && cur.kind() == ygg_term::Kind::Cons {
                elems.push(cur.head());
                cur = cur.tail();
            }
        }
        if !cur.is_nil() {
            die("list_cat: improper list");
        }
        let mut out = Term(b);
        for e in elems.into_iter().rev() {
            out = heap().cons(e, out).unwrap_or_else(|_| die("heap full"));
        }
        out.0
    }
    extern "C" fn h_bin_part(bin: u64, off: u64, len: u64) -> u64 {
        let b = Term(bin);
        let (Some(off), Some(len)) = (Term(off).as_int(), Term(len).as_int()) else {
            die("bin_part: non-int")
        };
        let part: Vec<u8> = unsafe {
            if !b.is_boxed() || b.kind() != ygg_term::Kind::Binary || off < 0 || len < 0 {
                die("bin_part badarg");
            }
            let bytes = b.bin_bytes();
            let (off, len) = (off as usize, len as usize);
            if off + len > bytes.len() {
                die("bin_part out of range");
            }
            bytes[off..off + len].to_vec()
        };
        heap().binary(&part).unwrap_or_else(|_| die("heap full")).0
    }

    fn helper_table() -> [u64; HELPER_COUNT] {
        let mut t = [0u64; HELPER_COUNT];
        t[Helper::SelfPid as usize] = h_self as usize as u64;
        t[Helper::Send as usize] = h_send as usize as u64;
        t[Helper::Recv as usize] = h_recv as usize as u64;
        t[Helper::Spawn as usize] = h_spawn as usize as u64;
        t[Helper::Safepoint as usize] = h_safepoint as usize as u64;
        t[Helper::Print as usize] = h_print as usize as u64;
        t[Helper::Eq as usize] = h_eq as usize as u64;
        t[Helper::MakeTuple as usize] = h_make_tuple as usize as u64;
        t[Helper::GetElem as usize] = h_get_elem as usize as u64;
        t[Helper::Cons as usize] = h_cons as usize as u64;
        t[Helper::Head as usize] = h_head as usize as u64;
        t[Helper::Tail as usize] = h_tail as usize as u64;
        t[Helper::PortOpen as usize] = h_port_open as usize as u64;
        t[Helper::PortSubmit as usize] = h_port_submit as usize as u64;
        t[Helper::CallExt as usize] = h_call_ext as usize as u64;
        t[Helper::ExitAtom as usize] = h_exit_atom as usize as u64;
        t[Helper::TrapBadarg as usize] = h_trap_badarg as usize as u64;
        t[Helper::BinConst as usize] = h_bin_const as usize as u64;
        t[Helper::BinFromList as usize] = h_bin_from_list as usize as u64;
        t[Helper::BinToList as usize] = h_bin_to_list as usize as u64;
        t[Helper::BinSize as usize] = h_bin_size as usize as u64;
        t[Helper::BufToBin as usize] = h_buf_to_bin as usize as u64;
        t[Helper::BinToBuf as usize] = h_bin_to_buf as usize as u64;
        t[Helper::MapNew as usize] = h_map_new as usize as u64;
        t[Helper::MapGet as usize] = h_map_get as usize as u64;
        t[Helper::MapPut as usize] = h_map_put as usize as u64;
        t[Helper::IsBinary as usize] = h_is_binary as usize as u64;
        t[Helper::BinCat as usize] = h_bin_cat as usize as u64;
        t[Helper::ListCat as usize] = h_list_cat as usize as u64;
        t[Helper::BinPart as usize] = h_bin_part as usize as u64;
        t[Helper::TailCallExt as usize] = h_tail_call_ext as usize as u64;
        t[Helper::TailCallLocal as usize] = h_tail_call_local as usize as u64;
        t[Helper::PortSubmit2 as usize] = h_port_submit2 as usize as u64;
        t[Helper::BufWrite as usize] = h_buf_write as usize as u64;
        t[Helper::SleepMs as usize] = h_sleep_ms as usize as u64;
        t[Helper::BufNew as usize] = h_buf_new as usize as u64;
        t[Helper::BufRead as usize] = h_buf_read as usize as u64;
        t[Helper::Ticks as usize] = h_ticks as usize as u64;
        t[Helper::ResumeTail as usize] = h_resume_tail as usize as u64;
        t[Helper::BinAt as usize] = h_bin_at as usize as u64;
        t
    }
}
