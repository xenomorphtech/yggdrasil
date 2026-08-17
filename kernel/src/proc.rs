//! Processes: green threads with their own native stacks and isolated term
//! heaps, cooperatively switched at safepoints only.
//!
//! BEAM semantics, modernized where it counts:
//! - isolated per-process heaps; messages are deep-copied into the receiver
//! - links propagate exit signals (transitively, hop by hop at reap time)
//! - monitors deliver `{'DOWN', Ref, Pid, Reason}` messages
//! - selective receive: `recv_where` scans the mailbox with a predicate
//! - `receive ... after`: timeouts via the timer wheel, checked by the scheduler
//! - preemption: the timer interrupt sets `PREEMPT`; code yields at safepoints
//!
//! Heaps are fixed-size bump arenas for now — exhaustion is a quota breach and
//! kills the process (never the kernel). Semispace GC is a later milestone.

use alloc::boxed::Box;
use alloc::collections::{BTreeMap, VecDeque};
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

use spin::Mutex;
use ygg_term::{Heap, HeapFull, Term, copy_term};

use crate::atoms;

pub type Pid = u64;

/// Default process heap: 64 frames = 256 KiB (also the quota, until GC).
const DEFAULT_HEAP_PAGES: usize = 64;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum State {
    Runnable,
    Running,
    Waiting,
    Dead,
}

/// Per-process heap: a chain of physically contiguous spans, bump-allocated
/// via `cur` (always the newest span). Growth is non-moving — terms in old
/// spans stay valid, which is what keeps JIT-held pointers sound — and is
/// capped by `max_pages` (the quota). Compaction happens at trampoline safe
/// points (see `modload`), where the live set is exactly known.
pub struct ProcHeap {
    pub cur: Heap,
    spans: Vec<(u64, usize)>, // (phys, pages)
    max_pages: usize,
    total_pages: usize,
}

impl ProcHeap {
    fn new(initial_pages: usize, max_pages: usize) -> ProcHeap {
        let pages = initial_pages.min(max_pages).max(1);
        let phys = crate::mm::alloc_contig(pages, 1).expect("no frames for process heap");
        let cur = unsafe { Heap::new(crate::mm::phys_to_virt(phys), pages * 4096) };
        ProcHeap {
            cur,
            spans: alloc::vec![(phys, pages)],
            max_pages,
            total_pages: pages,
        }
    }

    /// Add a span (doubling, clamped to the quota). False at the cap.
    pub fn grow(&mut self) -> bool {
        if self.total_pages >= self.max_pages {
            return false;
        }
        let next = self.total_pages.min(self.max_pages - self.total_pages);
        let phys = match crate::mm::alloc_contig(next, 1) {
            Some(p) => p,
            None => return false,
        };
        self.spans.push((phys, next));
        self.total_pages += next;
        // Terms in the old span remain valid; only the bump target moves.
        self.cur = unsafe { Heap::new(crate::mm::phys_to_virt(phys), next * 4096) };
        true
    }

    /// Replace all spans with a single fresh one (compaction target).
    /// Returns the old spans for the caller to free *after* evacuation.
    pub fn begin_compact(&mut self, live_words: usize) -> Option<Vec<(u64, usize)>> {
        let pages = ((live_words * 8).div_ceil(4096)).clamp(4, self.max_pages);
        let phys = crate::mm::alloc_contig(pages, 1)?;
        let old = core::mem::replace(&mut self.spans, alloc::vec![(phys, pages)]);
        self.total_pages = pages;
        self.cur = unsafe { Heap::new(crate::mm::phys_to_virt(phys), pages * 4096) };
        Some(old)
    }

    pub fn total_pages(&self) -> usize {
        self.total_pages
    }
    pub fn span_count(&self) -> usize {
        self.spans.len()
    }
    fn take_spans(&mut self) -> Vec<(u64, usize)> {
        core::mem::take(&mut self.spans)
    }
}

pub fn free_spans(spans: Vec<(u64, usize)>) {
    for (phys, pages) in spans {
        crate::mm::free_frames(phys, pages);
    }
}

/// A message in flight: deep-copied out of the sender into its own kernel-heap
/// backing (BEAM's heap fragments). The receiver copies it into its heap at
/// receive time — so no CPU ever writes another CPU's process heap.
pub struct Fragment {
    /// Stable backing store (the Vec buffer address never changes after build).
    #[allow(dead_code)]
    backing: alloc::vec::Vec<u64>,
    root: Term,
}

unsafe impl Send for Fragment {}

impl Fragment {
    /// Deep-copy `term` (rooted in any live heap) into a fresh fragment.
    fn copy_from(term: Term) -> Fragment {
        let words = unsafe { ygg_term::term_size_words(term) }.max(1);
        let mut backing = alloc::vec![0u64; words];
        let mut heap = unsafe { Heap::new(backing.as_mut_ptr().cast(), words * 8) };
        let root = unsafe { copy_term(term, &mut heap) }.expect("fragment sized exactly");
        Fragment { backing, root }
    }

    /// Build a term directly in a fragment via `build` (retrying with a larger
    /// backing if the builder outgrows it).
    fn build(build: impl Fn(&mut Heap) -> Result<Term, HeapFull>) -> Fragment {
        let mut words = 32usize;
        loop {
            let mut backing = alloc::vec![0u64; words];
            let mut heap = unsafe { Heap::new(backing.as_mut_ptr().cast(), words * 8) };
            match build(&mut heap) {
                Ok(root) => return Fragment { backing, root },
                Err(HeapFull) => words *= 2,
            }
        }
    }
}

pub struct Process {
    pub state: State,
    /// Saved rsp while switched out.
    rsp: u64,
    stack_slot: u64,
    heap: ProcHeap,
    mailbox: VecDeque<Fragment>,
    /// Killed while Running on a core: honored at its next safepoint.
    kill_pending: bool,
    /// Bidirectional links (exit-signal propagation).
    links: Vec<Pid>,
    /// (ref, watcher) pairs watching *this* process.
    monitors: Vec<(u64, Pid)>,
    exit_reason: &'static str,
}

type Table = BTreeMap<Pid, Box<Process>>;

static TABLE: Mutex<Table> = Mutex::new(BTreeMap::new());
/// deadline tick -> pids to wake.
static WHEEL: Mutex<BTreeMap<u64, Vec<Pid>>> = Mutex::new(BTreeMap::new());
static NEXT_PID: AtomicU64 = AtomicU64::new(1);
static NEXT_REF: AtomicU64 = AtomicU64::new(1);

mod ctx {
    /// Switch stacks: save callee-saved regs + rsp to `*save`, resume `new_rsp`.
    /// Only ever called at safepoints, so caller-saved/FPU state needs no save.
    #[unsafe(naked)]
    pub unsafe extern "C" fn switch(save: *mut u64, new_rsp: u64) {
        core::arch::naked_asm!(
            "push rbp",
            "push rbx",
            "push r12",
            "push r13",
            "push r14",
            "push r15",
            "mov [rdi], rsp",
            "mov rsp, rsi",
            "pop r15",
            "pop r14",
            "pop r13",
            "pop r12",
            "pop rbx",
            "pop rbp",
            "ret",
        )
    }

    /// First frame of every process. The spawn stack seeds rbx = entry fn,
    /// r12 = arg; falls into exit when the body returns.
    #[unsafe(naked)]
    pub extern "C" fn trampoline() {
        core::arch::naked_asm!(
            "mov rdi, r12",
            "call rbx",
            "call {exit}",
            exit = sym super::exit_current,
        )
    }
}

pub fn current() -> Pid {
    crate::percpu::cpu().current.load(Ordering::Relaxed)
}

/// Queue a process as runnable on this core; kick one idle core so it can
/// steal without waiting for its next timer tick.
fn enqueue(pid: Pid) {
    let me = crate::percpu::cpu();
    me.runq.lock().push_back(pid);
    for c in crate::percpu::all() {
        if c.id != me.id && c.idle.load(Ordering::Relaxed) {
            let lapic = c.lapic_id.load(Ordering::Acquire);
            if lapic != u32::MAX {
                crate::irq::ipi(lapic, crate::irq::WAKE_VECTOR);
            }
            break;
        }
    }
}

pub fn is_alive(pid: Pid) -> bool {
    TABLE
        .lock()
        .get(&pid)
        .is_some_and(|p| p.state != State::Dead)
}

pub fn spawn(entry: extern "C" fn(u64), arg: u64) -> Pid {
    spawn_inner(entry, arg, None, DEFAULT_HEAP_PAGES)
}

/// Spawn with a custom heap size (pages). Heavy term churn (e.g. the network
/// stack) needs more than the default quota until per-process GC lands.
pub fn spawn_with_heap(entry: extern "C" fn(u64), arg: u64, heap_pages: usize) -> Pid {
    spawn_inner(entry, arg, None, heap_pages)
}

/// Spawn and atomically link to the current process.
pub fn spawn_link(entry: extern "C" fn(u64), arg: u64) -> Pid {
    spawn_inner(entry, arg, Some(current()), DEFAULT_HEAP_PAGES)
}

/// Spawn and atomically monitor: the monitor is registered before the child
/// can run (on any core), so the DOWN reason is never `noproc`.
pub fn spawn_monitor(entry: extern "C" fn(u64), arg: u64) -> (Pid, u64) {
    let r = NEXT_REF.fetch_add(1, Ordering::Relaxed);
    MONITOR_AT_SPAWN.lock().replace((r, current()));
    let pid = spawn_inner(entry, arg, None, DEFAULT_HEAP_PAGES);
    (pid, r)
}

/// Plumbing for `spawn_monitor` (set under no lock ordering constraints,
/// consumed inside `spawn_inner`'s table insertion).
static MONITOR_AT_SPAWN: Mutex<Option<(u64, Pid)>> = Mutex::new(None);

fn spawn_inner(
    entry: extern "C" fn(u64),
    arg: u64,
    link_to: Option<Pid>,
    heap_pages: usize,
) -> Pid {
    let (slot, top) = crate::vmm::map_stack();
    // Seed the stack so ctx::switch pops into the trampoline.
    // Layout from final rsp: [r15][r14][r13][r12=arg][rbx=entry][rbp][ret=trampoline]
    let rsp = top - 7 * 8;
    unsafe {
        let p = rsp as *mut u64;
        p.add(0).write(0);
        p.add(1).write(0);
        p.add(2).write(0);
        p.add(3).write(arg);
        p.add(4).write(entry as usize as u64);
        p.add(5).write(0);
        p.add(6).write(ctx::trampoline as usize as u64);
    }

    let heap = ProcHeap::new(16, heap_pages);

    let pid = NEXT_PID.fetch_add(1, Ordering::Relaxed);
    let mut links = Vec::new();
    let monitors: Vec<(u64, Pid)> = MONITOR_AT_SPAWN.lock().take().into_iter().collect();
    {
        let mut t = TABLE.lock();
        if let Some(parent) = link_to {
            if let Some(pp) = t.get_mut(&parent) {
                pp.links.push(pid);
                links.push(parent);
            }
        }
        t.insert(
            pid,
            Box::new(Process {
                state: State::Runnable,
                rsp,
                stack_slot: slot,
                heap,
                mailbox: VecDeque::new(),
                kill_pending: false,
                links,
                monitors,
                exit_reason: "normal",
            }),
        );
    }
    enqueue(pid);
    pid
}

/// Watch `target`; its death delivers `{'DOWN', Ref, Pid, Reason}` to us.
pub fn monitor(target: Pid) -> u64 {
    let me = current();
    let r = NEXT_REF.fetch_add(1, Ordering::Relaxed);
    let mut t = TABLE.lock();
    match t.get_mut(&target) {
        Some(p) if p.state != State::Dead => p.monitors.push((r, me)),
        _ => deliver_down(&mut t, me, r, target, "noproc"),
    }
    r
}

/// Run `f` against the current process's heap (for building terms to send).
pub fn with_heap<R>(f: impl FnOnce(&mut Heap) -> R) -> R {
    // Lock-free: the percpu current-heap pointer is set by this core's
    // scheduler at switch-in and only the owning (running) process touches
    // the heap — the same contract the interpreter tier already relies on
    // via `current_heap_ptr`. Every allocation helper lands here, so this
    // must not take the global TABLE lock.
    f(unsafe { &mut *current_heap_ptr() })
}

/// A stashed tail-call target awaiting the engine trampoline.
pub enum TailTarget {
    /// Cross-module: resolve `module:fname` in the *current* module table
    /// (this hop is the hot-load migration point).
    Ext(u32, u32, Vec<Term>),
    /// Local: function index in the module instance already running (BEAM
    /// local-call semantics — stays in the same version).
    Local(u32, Vec<Term>),
}

// The stash lives in percpu state, not the process table: it is written as
// an engine function's final act before returning the tail sentinel and read
// back immediately by the calling trampoline, with no safepoint or blocking
// point in between — so it can never survive across a context switch. This
// keeps the hottest path in the system (every tail-recursive iteration) off
// the global TABLE lock.
pub fn set_tail_target(module_atom: u32, fname_atom: u32, args: Vec<Term>) {
    *crate::percpu::cpu().tail_target.lock() = Some(TailTarget::Ext(module_atom, fname_atom, args));
}

pub fn set_tail_target_local(fn_idx: u32, args: Vec<Term>) {
    *crate::percpu::cpu().tail_target.lock() = Some(TailTarget::Local(fn_idx, args));
}

pub fn take_tail_target() -> Option<TailTarget> {
    crate::percpu::cpu().tail_target.lock().take()
}

/// Trampoline-point GC: the process has no live frames; `roots` is its entire
/// live term set. Compact into a fresh span sized to the live estimate and
/// free everything else. No-op unless there's actual pressure.
pub fn maybe_compact(roots: &mut [Term]) {
    let pid = current();
    let old_spans = {
        let mut t = TABLE.lock();
        let p = t.get_mut(&pid).expect("no current process");
        let pressured = p.heap.span_count() > 1
            || p.heap.cur.used_bytes() * 2 > p.heap.cur.capacity_bytes();
        if !pressured {
            return;
        }
        // term_size_words over-counts shared structure — a safe upper bound.
        let live: usize =
            roots.iter().map(|r| unsafe { ygg_term::term_size_words(*r) }).sum::<usize>()
                + roots.len()
                + 32;
        let Some(old) = p.heap.begin_compact(live) else {
            return; // allocation pressure: skip this cycle
        };
        unsafe { ygg_term::evacuate(roots, &mut p.heap.cur) }
            .expect("compaction target sized from live estimate");
        old
    };
    free_spans(old_spans);
}

/// Grow the current process's heap by one span. False at the quota cap.
pub fn grow_current_heap() -> bool {
    let pid = current();
    let mut t = TABLE.lock();
    t.get_mut(&pid).expect("no current process").heap.grow()
}

/// Allocate with grow-on-full retry; dies of quota breach only at the cap.
pub fn alloc_retry<R>(f: impl Fn(&mut Heap) -> Result<R, HeapFull>) -> R {
    loop {
        match with_heap(&f) {
            Ok(r) => return r,
            Err(HeapFull) => {
                if !grow_current_heap() {
                    exit("heap quota exceeded");
                }
            }
        }
    }
}

/// Raw pointer to the current process's heap, for the execution engines.
///
/// Lock-free: set by this core's scheduler at switch-in, cleared at
/// switch-out. Only the owning (running) process ever writes through it.
pub fn current_heap_ptr() -> *mut Heap {
    let p = crate::percpu::cpu().current_heap.load(Ordering::Relaxed);
    debug_assert!(p != 0, "no current process heap");
    p as *mut Heap
}

/// Build a term, growing the heap as needed; dies only at the quota cap.
pub fn build(f: impl Fn(&mut Heap) -> Result<Term, HeapFull>) -> Term {
    alloc_retry(f)
}

/// Switch from the current process back to this core's scheduler.
fn switch_to_scheduler(save: *mut u64) {
    let rsp = crate::percpu::cpu().sched_rsp.load(Ordering::Relaxed);
    unsafe { ctx::switch(save, rsp) }
}

pub fn yield_now() {
    let pid = current();
    debug_assert!(pid != 0, "yield outside a process");
    let rsp_ptr = {
        let mut t = TABLE.lock();
        let p = t.get_mut(&pid).unwrap();
        &raw mut p.rsp
    };
    // State stays Running until our scheduler has saved this context; it
    // requeues us afterwards (another core must never resume a stale rsp).
    crate::percpu::cpu()
        .post_switch
        .store(pid << 2 | 1, Ordering::Relaxed);
    switch_to_scheduler(rsp_ptr);
}

/// Interpreter back-edge / native-body poll point.
pub fn safepoint() {
    if crate::percpu::cpu().preempt.swap(false, Ordering::Relaxed) && current() != 0 {
        yield_now();
    }
}

pub fn exit(reason: &'static str) -> ! {
    let pid = current();
    let rsp_ptr = {
        let mut t = TABLE.lock();
        let p = t.get_mut(&pid).unwrap();
        p.state = State::Dead;
        p.exit_reason = reason;
        &raw mut p.rsp
    };
    switch_to_scheduler(rsp_ptr);
    unreachable!("dead process rescheduled")
}

pub extern "C" fn exit_current() -> ! {
    exit("normal")
}

/// Kill the *current* process from an exception handler (e.g. guard-page hit).
/// Abandons the interrupted context entirely.
pub fn fatal_current(reason: &'static str) -> ! {
    let pid = current();
    {
        let mut t = TABLE.lock();
        let p = t.get_mut(&pid).unwrap();
        p.state = State::Dead;
        p.exit_reason = reason;
    }
    static SCRATCH: AtomicU64 = AtomicU64::new(0);
    switch_to_scheduler(SCRATCH.as_ptr());
    unreachable!("dead process rescheduled")
}

/// Kill another process; reaped (and its exit signals delivered) when the
/// scheduler next sees it.
pub fn kill(pid: Pid, reason: &'static str) {
    let mut t = TABLE.lock();
    mark_dead(&mut t, pid, reason);
}

fn mark_dead(t: &mut Table, pid: Pid, reason: &'static str) {
    if let Some(p) = t.get_mut(&pid) {
        match p.state {
            State::Dead => {}
            // Running (possibly on another core): never reap under its feet.
            // Flag it; the process turns Dead at its next safepoint/receive
            // and is reaped by its own core's scheduler.
            State::Running => {
                if !p.kill_pending {
                    p.kill_pending = true;
                    p.exit_reason = reason;
                }
            }
            State::Runnable | State::Waiting => {
                p.state = State::Dead;
                p.exit_reason = reason;
                // Reaped when popped from a run queue.
                enqueue(pid);
            }
        }
    }
}

/// Copy `msg` (rooted in the sender's heap) into a fragment and enqueue it.
/// Returns false if the target is gone. The receiver copies it into its own
/// heap at receive time (and dies of quota breach there if it can't).
pub fn send(to: Pid, msg: Term) -> bool {
    // Copy before taking the table lock: only the sender's heap is read.
    let frag = Fragment::copy_from(msg);
    let mut t = TABLE.lock();
    deliver_fragment(&mut t, to, frag)
}

fn deliver_fragment(t: &mut Table, to: Pid, frag: Fragment) -> bool {
    let Some(p) = t.get_mut(&to) else {
        return false;
    };
    if p.state == State::Dead {
        return false;
    }
    p.mailbox.push_back(frag);
    if p.state == State::Waiting {
        p.state = State::Runnable;
        enqueue(to);
    }
    true
}

/// Build a message in a fragment and enqueue it (kernel-side senders: DOWN
/// messages, port completions). Returns false if the target is gone.
pub fn send_built(to: Pid, build: impl Fn(&mut Heap) -> Result<Term, HeapFull>) -> bool {
    let frag = Fragment::build(build);
    let mut t = TABLE.lock();
    deliver_fragment(&mut t, to, frag)
}

/// Selective receive: return the first mailbox message satisfying `pred`,
/// blocking until one arrives. `timeout_ms: Some(n)` gives up after ~n ms.
pub fn recv_where(pred: impl Fn(Term) -> bool, timeout_ms: Option<u64>) -> Option<Term> {
    let pid = current();
    let deadline = timeout_ms.map(|ms| crate::irq::ticks() + ms);
    loop {
        let rsp_ptr = {
            let mut t = TABLE.lock();
            let p = t.get_mut(&pid).unwrap();
            if p.kill_pending {
                p.state = State::Dead;
            }
            if p.state == State::Dead {
                // Killed since our last safepoint: don't take a message, just
                // hand control back so the scheduler reaps us.
                let ptr = &raw mut p.rsp;
                drop(t);
                switch_to_scheduler(ptr);
                unreachable!("dead process rescheduled");
            }
            // Copy each candidate into our heap speculatively; roll the bump
            // pointer back on predicate miss (nothing else allocates between).
            let mut quota_death = false;
            let mut matched: Option<Term> = None;
            for i in 0..p.mailbox.len() {
                let watermark = p.heap.cur.used_bytes();
                let copied = match unsafe { copy_term(p.mailbox[i].root, &mut p.heap.cur) } {
                    Ok(c) => c,
                    Err(HeapFull) => {
                        // A fresh span gives the copy contiguous room; retry
                        // next loop pass. At the cap, die of quota breach.
                        if p.heap.grow() {
                            continue;
                        }
                        quota_death = true;
                        break;
                    }
                };
                if pred(copied) {
                    p.mailbox.remove(i);
                    matched = Some(copied);
                    break;
                }
                p.heap.cur.truncate_to(watermark);
            }
            if quota_death {
                drop(t);
                exit("heap quota exceeded");
            }
            if matched.is_some() {
                return matched;
            }
            if let Some(d) = deadline
                && crate::irq::ticks() >= d
            {
                return None;
            }
            &raw mut p.rsp
        };
        if let Some(d) = deadline {
            WHEEL.lock().entry(d).or_default().push(pid);
        }
        // State stays Running until our scheduler saved the context; it then
        // transitions us to Waiting (or straight back to Runnable if a
        // message raced in) — no lost wakeups, no stale-context steals.
        crate::percpu::cpu()
            .post_switch
            .store(pid << 2 | 2, Ordering::Relaxed);
        switch_to_scheduler(rsp_ptr);
    }
}

pub fn recv() -> Term {
    recv_where(|_| true, None).unwrap()
}

pub fn recv_timeout(ms: u64) -> Option<Term> {
    recv_where(|_| true, Some(ms))
}

/// Wake sleepers whose deadline has passed.
fn check_wheel() {
    let now = crate::irq::ticks();
    let due: Vec<Pid> = {
        let mut w = WHEEL.lock();
        let mut due = Vec::new();
        while let Some((&d, _)) = w.first_key_value() {
            if d > now {
                break;
            }
            due.extend(w.pop_first().unwrap().1);
        }
        due
    };
    if !due.is_empty() {
        let mut t = TABLE.lock();
        for pid in due {
            match t.get_mut(&pid) {
                Some(p) if p.state == State::Waiting => {
                    p.state = State::Runnable;
                    enqueue(pid);
                }
                // Caught mid block-transition on another core: try next tick.
                Some(p) if p.state == State::Running => {
                    WHEEL.lock().entry(now + 1).or_default().push(pid);
                }
                _ => {}
            }
        }
    }
}

/// The scheduler loop for this core. Never returns.
pub fn run() -> ! {
    let me = crate::percpu::cpu();
    loop {
        // Contexts don't carry rflags: after e.g. a guard-page kill we resume
        // here with IF=0. The scheduler always runs with interrupts on.
        x86_64::instructions::interrupts::enable();
        me.phase.store(1, Ordering::Relaxed);
        // The timer wheel is time-driven and TICKS is BSP-owned.
        if me.id == 0 {
            check_wheel();
        }
        me.phase.store(2, Ordering::Relaxed);
        crate::ports::pump();
        me.phase.store(3, Ordering::Relaxed);
        // NOTE: pop and steal must be separate statements — holding our own
        // runq guard while locking another core's runq deadlocks the moment
        // two cores steal from each other.
        let mut next = me.runq.lock().pop_front();
        if next.is_none() {
            // Steal from the back of another core's queue.
            next = crate::percpu::all()
                .iter()
                .filter(|c| c.id != me.id)
                .find_map(|c| c.runq.lock().pop_back());
        }
        let Some(pid) = next else {
            // Nothing runnable anywhere: sleep until an interrupt (timer or
            // wake IPI) changes that.
            me.idle.store(true, Ordering::Relaxed);
            x86_64::instructions::interrupts::enable_and_hlt();
            me.idle.store(false, Ordering::Relaxed);
            continue;
        };
        me.phase.store(4, Ordering::Relaxed);
        let rsp = {
            let mut t = TABLE.lock();
            match t.get_mut(&pid) {
                Some(p) if p.state == State::Runnable => {
                    p.state = State::Running;
                    p.rsp
                }
                Some(p) if p.state == State::Dead => {
                    let fin = reap(&mut t, pid);
                    drop(t);
                    finalize(fin);
                    continue;
                }
                // Stale queue entry (waiting/duplicate/gone): skip.
                _ => continue,
            }
        };
        let heap_ptr = {
            let mut t = TABLE.lock();
            &raw mut t.get_mut(&pid).unwrap().heap.cur as u64
        };
        me.phase.store(5, Ordering::Relaxed);
        me.current.store(pid, Ordering::Relaxed);
        me.current_heap.store(heap_ptr, Ordering::Relaxed);
        me.switches.fetch_add(1, Ordering::Relaxed);
        unsafe { ctx::switch(me.sched_rsp.as_ptr(), rsp) };
        // The context we resumed from may have had IF=0 (e.g. the process was
        // killed inside the page-fault handler). Everything below may spin on
        // cross-core protocols (TLB shootdown) that need our interrupts on.
        x86_64::instructions::interrupts::enable();
        me.current.store(0, Ordering::Relaxed);
        me.current_heap.store(0, Ordering::Relaxed);

        me.phase.store(6, Ordering::Relaxed);
        // The outgoing context is fully saved now: perform its deferred
        // transition, then reap if it ended up Dead.
        let action = me.post_switch.swap(0, Ordering::Relaxed);
        let mut t = TABLE.lock();
        match (action & 3, action >> 2) {
            (1, apid) => {
                debug_assert_eq!(apid, pid);
                if let Some(p) = t.get_mut(&apid) {
                    if p.kill_pending {
                        p.state = State::Dead;
                    }
                    if p.state == State::Running {
                        p.state = State::Runnable;
                        enqueue(apid);
                    }
                }
            }
            (2, apid) => {
                debug_assert_eq!(apid, pid);
                if let Some(p) = t.get_mut(&apid) {
                    if p.kill_pending {
                        p.state = State::Dead;
                    }
                    if p.state == State::Running {
                        if p.mailbox.is_empty() {
                            p.state = State::Waiting;
                        } else {
                            // A message raced in while we were switching.
                            p.state = State::Runnable;
                            enqueue(apid);
                        }
                    }
                }
            }
            _ => {}
        }
        if t.get(&pid).is_some_and(|p| p.state == State::Dead) {
            me.phase.store(7, Ordering::Relaxed);
            let fin = reap(&mut t, pid);
            drop(t);
            finalize(fin);
        }
    }
}

/// Resources released outside the TABLE lock (see `reap`).
struct Finalize {
    pid: Pid,
    stack_slot: u64,
    heap_spans: Vec<(u64, usize)>,
}

/// Free a dead process's resources. MUST run with the TABLE lock dropped:
/// port teardown takes PORTS (pump orders PORTS -> TABLE) and stack unmap
/// broadcasts a TLB shootdown that must not stall other cores against TABLE.
fn finalize(fin: Finalize) {
    crate::vmm::unmap_stack(fin.stack_slot);
    free_spans(fin.heap_spans);
    crate::ports::close_owned_by(fin.pid);
}

/// Free a dead process and deliver its exit signals: DOWN messages to
/// monitors, death to links. Link propagation is hop-by-hop: linked processes
/// are marked dead here and their own reap continues the cascade.
/// Remove a dead process and deliver its exit signals (monitors, links) under
/// the TABLE lock. Resource teardown is returned for `finalize` — running it
/// here would invert the PORTS->TABLE lock order pump relies on.
fn reap(t: &mut Table, pid: Pid) -> Finalize {
    let p = t.remove(&pid).unwrap();
    log::info!("[proc] pid {} exited: {}", pid, p.exit_reason);

    for (r, watcher) in &p.monitors {
        deliver_down(t, *watcher, *r, pid, p.exit_reason);
    }
    if p.exit_reason != "normal" {
        for linked in &p.links {
            mark_dead(t, *linked, p.exit_reason);
        }
    }

    let mut p = p;
    Finalize {
        pid,
        stack_slot: p.stack_slot,
        heap_spans: p.heap.take_spans(),
    }
}

fn deliver_down(t: &mut Table, watcher: Pid, r: u64, dead: Pid, reason: &'static str) {
    let down = atoms::intern("DOWN");
    let reason = atoms::intern(reason);
    let frag = Fragment::build(|h| {
        h.tuple(&[
            Term::atom(down),
            Term::reference(r),
            Term::pid(dead),
            Term::atom(reason),
        ])
    });
    deliver_fragment(t, watcher, frag);
}
