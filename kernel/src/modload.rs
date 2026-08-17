//! Module loading and bytecode process spawning.
//!
//! Modules arrive as `.yggm` blobs (limine boot modules now; storage/network
//! ports later), are decoded, and their local atoms are interned globally.
//! The verifier slots in here at M6; the two-version hot-loading table at M7.

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

use spin::Mutex;
use ygg_bytecode::{DecodeError, Module};
use ygg_interp::{SystemApi, Trap};
use ygg_term::{Heap, Term};

use crate::{atoms, boot, proc};

pub struct LoadedModule {
    pub name: String,
    /// Monotonic per-name version; bumped on each (re)load.
    pub version: u64,
    pub module: Module,
    /// Module-local atom index -> global atom index.
    pub atom_map: Vec<u32>,
    /// Tier-1 native code (None -> interpreter tier-0).
    pub jit: Option<crate::jit::JitModule>,
    /// Content-addressed and immortal: the name names the bytes, the slot is
    /// never replaced or purged, and external calls to it may be bound to its
    /// published code address at the callers' compile time.
    pub sealed: bool,
    /// Per function: can it return the tail sentinel to a native caller?
    /// Callers direct-binding to a function whose bit is false skip the
    /// sentinel check at the call site.
    pub tailers: Vec<bool>,
}

/// BEAM's two-version model: `current` serves new external calls; `old` keeps
/// in-flight processes alive until they migrate (or get purged).
struct Slot {
    current: Arc<LoadedModule>,
    old: Option<Arc<LoadedModule>>,
}

static MODULES: Mutex<alloc::collections::BTreeMap<String, Slot>> =
    Mutex::new(alloc::collections::BTreeMap::new());
/// Which (module, version) each live bytecode process is currently executing.
static RUNNING: Mutex<alloc::collections::BTreeMap<proc::Pid, (String, u64)>> =
    Mutex::new(alloc::collections::BTreeMap::new());

#[derive(Debug)]
pub enum LoadError {
    Decode(DecodeError),
    Verify(ygg_bytecode::verify::VerifyError),
}

pub fn load(name: &str, bytes: &[u8]) -> Result<Arc<LoadedModule>, LoadError> {
    load_full(name, bytes, true, false)
}

pub fn load_with_engine(
    name: &str,
    bytes: &[u8],
    use_jit: bool,
) -> Result<Arc<LoadedModule>, LoadError> {
    load_full(name, bytes, use_jit, false)
}

/// Load a content-addressed (immutable) module. Idempotent: a name that is
/// already sealed returns the existing instance untouched.
pub fn load_sealed(
    name: &str,
    bytes: &[u8],
    use_jit: bool,
) -> Result<Arc<LoadedModule>, LoadError> {
    if let Some(slot) = MODULES.lock().get(name)
        && slot.current.sealed
    {
        return Ok(slot.current.clone());
    }
    load_full(name, bytes, use_jit, true)
}

fn load_full(
    name: &str,
    bytes: &[u8],
    use_jit: bool,
    sealed: bool,
) -> Result<Arc<LoadedModule>, LoadError> {
    let module = Module::decode(bytes).map_err(LoadError::Decode)?;
    // The verifier is the isolation boundary: nothing unverified ever runs.
    ygg_bytecode::verify::verify(&module).map_err(LoadError::Verify)?;
    let atom_map: Vec<u32> = module.atoms.iter().map(|a| atoms::intern(a)).collect();
    let tailers = ygg_bytecode::sentinel_returners(&module);
    let jit = if use_jit {
        crate::jit::compile_and_publish(&module, &atom_map, &resolve_ext_direct)
    } else {
        None
    };

    let mut mods = MODULES.lock();
    let version = mods.get(name).map_or(1, |s| s.current.version + 1);
    let loaded = Arc::new(LoadedModule {
        name: String::from(name),
        version,
        module,
        atom_map,
        jit,
        sealed,
        tailers,
    });
    match mods.get_mut(name) {
        // Sealed names are immutable and direct-bound callers hold the
        // published code address: never swap the instance out.
        Some(slot) if slot.current.sealed => {
            return Ok(slot.current.clone());
        }
        Some(slot) => {
            slot.old = Some(core::mem::replace(&mut slot.current, loaded.clone()));
        }
        None => {
            mods.insert(
                String::from(name),
                Slot {
                    current: loaded.clone(),
                    old: None,
                },
            );
        }
    }
    log::info!(
        "modload: {} v{} ({} fns, {} atoms, {} bytes)",
        name,
        version,
        loaded.module.functions.len(),
        loaded.module.atoms.len(),
        bytes.len()
    );
    Ok(loaded)
}

pub fn current(name: &str) -> Option<Arc<LoadedModule>> {
    MODULES.lock().get(name).map(|s| s.current.clone())
}

/// Direct-bind hits/misses across all compiles (misses = sites left dynamic).
static EXT_BIND_HITS: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
static EXT_BIND_MISSES: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

pub fn ext_bind_stats() -> (u64, u64) {
    use core::sync::atomic::Ordering;
    (
        EXT_BIND_HITS.load(Ordering::Relaxed),
        EXT_BIND_MISSES.load(Ordering::Relaxed),
    )
}

/// Compile-time resolver for `CALL_EXT` sites: a callee that is sealed,
/// JIT'd, and of direct-call arity gets its published entry address baked
/// into the caller. The token is the callee's stable `LoadedModule` address
/// (immortal for sealed modules) for `ResumeTail` dispatch.
fn resolve_ext_direct(module_atom: u32, fname_atom: u32) -> Option<ygg_jit::ExtTarget> {
    use core::sync::atomic::Ordering;
    let resolved = (|| {
        let mods = MODULES.lock();
        let slot = mods.get(atoms::name(module_atom))?;
        let m = &slot.current;
        if !m.sealed {
            return None;
        }
        let jit = m.jit.as_ref()?;
        let idx = m.module.function_named(atoms::name(fname_atom))?;
        let f = &m.module.functions[idx];
        if f.arity > 16 {
            return None;
        }
        Some(ygg_jit::ExtTarget {
            addr: jit.fn_addrs[idx],
            token: Arc::as_ptr(m) as u64,
            arity: f.arity,
            may_tail: m.tailers.get(idx).copied().unwrap_or(true),
        })
    })();
    match &resolved {
        Some(_) => EXT_BIND_HITS.fetch_add(1, Ordering::Relaxed),
        None => EXT_BIND_MISSES.fetch_add(1, Ordering::Relaxed),
    };
    resolved
}

fn note_running(pid: proc::Pid, name: &str, version: u64) {
    RUNNING.lock().insert(pid, (String::from(name), version));
}

pub fn note_running_pub(pid: proc::Pid, name: &str, version: u64) {
    note_running(pid, name, version);
}

/// The module the current process is executing (by the RUNNING registry).
pub fn current_process_module() -> Option<Arc<LoadedModule>> {
    let (name, version) = RUNNING.lock().get(&proc::current()).cloned()?;
    let mods = MODULES.lock();
    let slot = mods.get(&name)?;
    if slot.current.version == version {
        Some(slot.current.clone())
    } else if slot.old.as_ref().is_some_and(|o| o.version == version) {
        slot.old.clone()
    } else {
        Some(slot.current.clone())
    }
}

/// Run a function of a loaded module, trampolining tail-call hops so
/// tail-recursive bytecode runs in constant native stack. Never compacts:
/// safe under nested `CALL_EXT` (outer frames hold live terms) and under
/// native callers that keep heap terms across the call.
pub fn invoke(m: &Arc<LoadedModule>, fn_idx: usize, args: &[Term]) -> Result<Term, Trap> {
    invoke_inner(m, fn_idx, args, false)
}

/// Like `invoke`, but each tail hop is a GC point: the stashed args are
/// treated as the process's *entire* live set and the heap is compacted.
/// Contract: the caller must be the outermost engine entry for this process
/// and must hold no process-heap terms across the call (spawn entries, or
/// drivers whose roots are all immediates).
pub fn invoke_gc(m: &Arc<LoadedModule>, fn_idx: usize, args: &[Term]) -> Result<Term, Trap> {
    invoke_inner(m, fn_idx, args, true)
}

fn invoke_inner(
    m: &Arc<LoadedModule>,
    fn_idx: usize,
    args: &[Term],
    gc: bool,
) -> Result<Term, Trap> {
    let mut module = m.clone();
    let mut fn_idx = fn_idx;
    let mut args: alloc::vec::Vec<Term> = args.to_vec();
    loop {
        let r = invoke_once(&module, fn_idx, &args)?;
        if r != ygg_interp::TAIL_SENTINEL {
            debug_assert!(r.0 & 7 != 7, "reserved word escaped as a term");
            return Ok(r);
        }
        next_tail_target(&mut module, &mut fn_idx, &mut args)?;
        if gc {
            proc::maybe_compact(&mut args);
        }
    }
}

/// Consume the stashed tail target into (module, fn_idx, args).
fn next_tail_target(
    module: &mut Arc<LoadedModule>,
    fn_idx: &mut usize,
    args: &mut alloc::vec::Vec<Term>,
) -> Result<(), Trap> {
    match proc::take_tail_target().ok_or(Trap::BadCode)? {
        proc::TailTarget::Ext(ma, fa, targs) => {
            let mname = atoms::name(ma);
            let target = current(mname).ok_or(Trap::Badarg)?;
            let fname = atoms::name(fa);
            let idx = target.module.function_named(fname).ok_or(Trap::Badarg)?;
            if target.module.functions[idx].arity as usize != targs.len() {
                return Err(Trap::Badarg);
            }
            note_running(proc::current(), &target.name, target.version);
            *args = targs;
            *module = target;
            *fn_idx = idx;
        }
        proc::TailTarget::Local(idx, targs) => {
            // Same module instance; the verifier already bounds-checked
            // the index and matched the arity.
            *args = targs;
            *fn_idx = idx as usize;
        }
    }
    Ok(())
}

/// Run the tail chain a direct-called sealed function left stashed, starting
/// in that callee's module (identified by its stable `LoadedModule` address).
/// Nested context: never compacts — the direct caller's frames hold live
/// terms, exactly like the dynamic `call_ext` path.
///
/// Safety: `token` must be a pointer previously produced by
/// `resolve_ext_direct` for a sealed module; sealed instances are immortal.
pub unsafe fn resume_tail_from(token: u64) -> Result<Term, Trap> {
    let ptr = token as *const LoadedModule;
    let mut module = unsafe {
        Arc::increment_strong_count(ptr);
        Arc::from_raw(ptr)
    };
    let mut fn_idx = 0usize;
    let mut args: alloc::vec::Vec<Term> = alloc::vec::Vec::new();
    loop {
        next_tail_target(&mut module, &mut fn_idx, &mut args)?;
        let r = invoke_once(&module, fn_idx, &args)?;
        if r != ygg_interp::TAIL_SENTINEL {
            debug_assert!(r.0 & 7 != 7, "reserved word escaped as a term");
            return Ok(r);
        }
    }
}

fn invoke_once(m: &Arc<LoadedModule>, fn_idx: usize, args: &[Term]) -> Result<Term, Trap> {
    if let Some(jit) = &m.jit {
        // Native fan-out for common arities; large-arity calls fall back to
        // the interpreter (same semantics, no JIT ABI for arity > 16).
        if args.len() <= 16 {
            let addr = jit.fn_addrs[fn_idx];
            let r = unsafe {
                match args {
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
                    _ => unreachable!(),
                }
            };
            return Ok(Term(r));
        }
    }
    let mut api = KernelApi { module: m.clone() };
    ygg_interp::run_function(&m.module, fn_idx, args, &mut api)
}

/// Resolve and run `module:fname(args)` against the *current* module table
/// (shared by the interpreter's and the JIT's call_ext paths). The callee's
/// version is noted in RUNNING; the caller restores its own afterwards.
pub fn call_ext_dynamic(module_atom: u32, fname_atom: u32, args: &[Term]) -> Result<Term, Trap> {
    let mname = atoms::name(module_atom);
    let target = current(mname).ok_or(Trap::Badarg)?;
    let fname = atoms::name(fname_atom);
    let fn_idx = target.module.function_named(fname).ok_or(Trap::Badarg)?;
    if target.module.functions[fn_idx].arity as usize != args.len() {
        return Err(Trap::Badarg);
    }
    note_running(proc::current(), &target.name, target.version);
    invoke(&target, fn_idx, args)
}

/// Terminate the current process for a trap (uniform across engines).
pub fn exit_for_trap(trap: Trap) -> ! {
    match trap {
        Trap::Exit(a) => proc::exit(atoms::name(a)),
        Trap::Badarg => proc::exit("badarg"),
        Trap::BadCode => proc::exit("badcode"),
        Trap::HeapFull => proc::exit("heap quota exceeded"),
    }
}

pub fn print_term(t: Term) {
    let mut s = alloc::string::String::new();
    let _ = unsafe { ygg_term::fmt_term(t, &mut s, &atoms::name) };
    crate::println!("[bc] {s}");
}

/// Kill every process still executing an out-of-date version of `name` and
/// drop the old version. Returns how many processes were killed.
pub fn purge(name: &str) -> usize {
    let current_version = match MODULES.lock().get_mut(name) {
        Some(slot) => {
            slot.old = None;
            slot.current.version
        }
        None => return 0,
    };
    let holdouts: Vec<proc::Pid> = {
        let mut running = RUNNING.lock();
        running.retain(|pid, _| proc::is_alive(*pid));
        running
            .iter()
            .filter(|(_, (m, v))| m == name && *v < current_version)
            .map(|(pid, _)| *pid)
            .collect()
    };
    for pid in &holdouts {
        proc::kill(*pid, "purged (old module version)");
    }
    holdouts.len()
}

/// Find a limine boot module by trailing path component.
pub fn boot_module_bytes(filename: &str) -> Option<&'static [u8]> {
    let resp = boot::MODULES.response()?;
    resp.modules()
        .iter()
        .find(|f| f.path().ends_with(filename))
        .map(|f| {
            // The file data lives in bootloader-reclaimable memory, which we
            // never reclaim (yet), so 'static is honest.
            unsafe { core::mem::transmute::<&[u8], &'static [u8]>(f.data()) }
        })
}

/// Spawn a process running `fname/1` of `module` with an *immediate* argument.
pub fn spawn(module: Arc<LoadedModule>, fname: &str, arg: Term) -> Option<proc::Pid> {
    assert!(!arg.is_boxed(), "spawn args must be immediates for now");
    let fn_idx = module.module.function_named(fname)? as u32;
    Some(spawn_fn(module, fn_idx, arg))
}

pub fn spawn_fn(module: Arc<LoadedModule>, fn_idx: u32, arg: Term) -> proc::Pid {
    let (name, version) = (module.name.clone(), module.version);
    let info = alloc::boxed::Box::new(SpawnInfo {
        module,
        fn_idx,
        arg,
    });
    let pid = proc::spawn(bytecode_entry, alloc::boxed::Box::into_raw(info) as u64);
    // Register the version at spawn time — a purge may race the first run.
    note_running(pid, &name, version);
    pid
}

struct SpawnInfo {
    module: Arc<LoadedModule>,
    fn_idx: u32,
    arg: Term,
}

extern "C" fn bytecode_entry(raw: u64) {
    let info = unsafe { alloc::boxed::Box::from_raw(raw as *mut SpawnInfo) };
    let f = &info.module.module.functions[info.fn_idx as usize];
    let args: &[Term] = if f.arity == 0 { &[] } else { &[info.arg] };
    // Spawn args are immediates (verifier-enforced) and this frame holds no
    // heap terms across the run: tail hops may compact.
    if let Err(trap) = invoke_gc(&info.module, info.fn_idx as usize, args) {
        exit_for_trap(trap);
    }
}

/// `SystemApi` over the real kernel.
struct KernelApi {
    module: Arc<LoadedModule>,
}

impl SystemApi for KernelApi {
    fn heap(&mut self) -> &mut Heap {
        // Single core, cooperative switching: while this process runs, nothing
        // else touches its heap (senders only run when scheduled, copy-on-send
        // happens under the table lock while we're switched out). SMP (M8)
        // moves this into the per-CPU current-process context.
        unsafe { &mut *proc::current_heap_ptr() }
    }
    fn self_pid(&self) -> u64 {
        proc::current()
    }
    fn send(&mut self, to: Term, msg: Term) -> Result<(), Trap> {
        let pid = to.as_pid().ok_or(Trap::Badarg)?;
        // Sending to a dead pid is a no-op, BEAM-style.
        let _ = proc::send(pid, msg);
        Ok(())
    }
    fn recv(&mut self) -> Term {
        proc::recv()
    }
    fn spawn(&mut self, fn_idx: u32, arg: Term) -> Result<u64, Trap> {
        if self.module.module.functions.get(fn_idx as usize).is_none() {
            return Err(Trap::BadCode);
        }
        Ok(spawn_fn(self.module.clone(), fn_idx, arg))
    }
    fn safepoint(&mut self) {
        proc::safepoint();
    }
    fn atom_global(&mut self, local: u32) -> u32 {
        self.module
            .atom_map
            .get(local as usize)
            .copied()
            .unwrap_or_else(|| atoms::intern("?"))
    }
    fn print(&mut self, t: Term) {
        print_term(t);
    }
    fn port_open(&mut self, kind: u8) -> Result<Term, Trap> {
        crate::ports::open(kind).ok_or(Trap::Badarg)
    }
    fn port_submit(&mut self, port: Term, op: u8, arg0: Term, tag: Term) -> Result<(), Trap> {
        let id = port.as_port().ok_or(Trap::Badarg)?;
        let arg0 = arg0.as_int().ok_or(Trap::Badarg)?;
        let tag = tag.as_int().ok_or(Trap::Badarg)?;
        let sqe = ygg_rings::Sqe {
            op: op as u32,
            tag,
            arg0: arg0 as u64,
            arg1: 0,
        };
        if crate::ports::submit(id, sqe) {
            Ok(())
        } else {
            Err(Trap::Badarg)
        }
    }

    fn port_submit2(
        &mut self,
        port: Term,
        op: Term,
        arg0: Term,
        arg1: Term,
        tag: Term,
    ) -> Result<(), Trap> {
        let id = port.as_port().ok_or(Trap::Badarg)?;
        let sqe = ygg_rings::Sqe {
            op: op.as_int().ok_or(Trap::Badarg)? as u32,
            tag: tag.as_int().ok_or(Trap::Badarg)?,
            arg0: arg0.as_int().ok_or(Trap::Badarg)? as u64,
            arg1: arg1.as_int().ok_or(Trap::Badarg)? as u64,
        };
        if crate::ports::submit(id, sqe) { Ok(()) } else { Err(Trap::Badarg) }
    }

    fn buf_write(&mut self, buf: Term, off: Term, bin: Term) -> Result<Term, Trap> {
        let id = buf.as_int().ok_or(Trap::Badarg)?;
        let off = off.as_int().ok_or(Trap::Badarg)?;
        let bytes = unsafe {
            if !bin.is_boxed() || bin.kind() != ygg_term::Kind::Binary || off < 0 {
                return Err(Trap::Badarg);
            }
            bin.bin_bytes()
        };
        if crate::ports::buf_write(id as u64, off as usize, bytes) {
            Ok(Term::int(0))
        } else {
            Err(Trap::Badarg)
        }
    }

    fn sleep_ms(&mut self, ms: u64) {
        // A never-matching receive: parks on the timer wheel, consumes no
        // mailbox message.
        let _ = proc::recv_where(|_| false, Some(ms));
    }

    fn ticks(&self) -> Term {
        Term::int(crate::irq::ticks() as i64)
    }

    fn buf_new(&mut self, size: Term) -> Result<Term, Trap> {
        let size = size.as_int().ok_or(Trap::Badarg)?;
        // Cap so bytecode can't exhaust the kernel heap (64 MiB covers any
        // realistic framebuffer or DMA backing).
        if size < 0 || size > 64 << 20 {
            return Err(Trap::Badarg);
        }
        Ok(Term::int(crate::ports::buf_create(alloc::vec![0u8; size as usize]) as i64))
    }

    fn buf_read(&mut self, buf: Term, off: Term, len: Term) -> Result<Term, Trap> {
        let id = buf.as_int().ok_or(Trap::Badarg)?;
        let off = off.as_int().ok_or(Trap::Badarg)?;
        let len = len.as_int().ok_or(Trap::Badarg)?;
        if off < 0 || len < 0 {
            return Err(Trap::Badarg);
        }
        let data =
            crate::ports::buf_read(id as u64, off as usize, len as usize).ok_or(Trap::Badarg)?;
        Ok(proc::alloc_retry(|h| h.binary(&data)))
    }

    fn buf_to_bin(&mut self, id: i64) -> Result<Term, Trap> {
        let data = crate::ports::buf_take(id as u64).ok_or(Trap::Badarg)?;
        Ok(proc::alloc_retry(|h| h.binary(&data)))
    }

    fn bin_to_buf(&mut self, bin: Term) -> Result<Term, Trap> {
        let data = unsafe {
            if !bin.is_boxed() || bin.kind() != ygg_term::Kind::Binary {
                return Err(Trap::Badarg);
            }
            bin.bin_bytes().to_vec()
        };
        Ok(Term::int(crate::ports::buf_create(data) as i64))
    }

    fn heap_grow(&mut self) -> bool {
        proc::grow_current_heap()
    }

    fn tail_call(&mut self, module_atom: u32, fname_atom: u32, args: &[Term]) {
        proc::set_tail_target(module_atom, fname_atom, args.to_vec());
    }

    fn tail_call_local(&mut self, fn_idx: u32, args: &[Term]) {
        proc::set_tail_target_local(fn_idx, args.to_vec());
    }

    /// The hot-loading migration point: resolve module:fname in the *current*
    /// module table and run it (in that module's context).
    fn call_ext(&mut self, module_atom: u32, fname_atom: u32, args: &[Term]) -> Result<Term, Trap> {
        let r = call_ext_dynamic(module_atom, fname_atom, args);
        // Back in the caller's (this Api's) module version.
        note_running(proc::current(), &self.module.name, self.module.version);
        r
    }
}
