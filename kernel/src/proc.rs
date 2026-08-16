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
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use spin::Mutex;
use ygg_term::{Heap, HeapFull, Term, copy_term};

use crate::atoms;

pub type Pid = u64;

/// Default process heap: 64 frames = 256 KiB (also the quota, until GC).
const HEAP_PAGES: usize = 64;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum State {
    Runnable,
    Running,
    Waiting,
    Dead,
}

pub struct Process {
    pub state: State,
    /// Saved rsp while switched out.
    rsp: u64,
    stack_slot: u64,
    heap: Heap,
    heap_span: u64,
    mailbox: VecDeque<Term>,
    /// Bidirectional links (exit-signal propagation).
    links: Vec<Pid>,
    /// (ref, watcher) pairs watching *this* process.
    monitors: Vec<(u64, Pid)>,
    exit_reason: &'static str,
}

type Table = BTreeMap<Pid, Box<Process>>;

static TABLE: Mutex<Table> = Mutex::new(BTreeMap::new());
static RUNQ: Mutex<VecDeque<Pid>> = Mutex::new(VecDeque::new());
/// deadline tick -> pids to wake.
static WHEEL: Mutex<BTreeMap<u64, Vec<Pid>>> = Mutex::new(BTreeMap::new());
static NEXT_PID: AtomicU64 = AtomicU64::new(1);
static NEXT_REF: AtomicU64 = AtomicU64::new(1);
static CURRENT: AtomicU64 = AtomicU64::new(0);
/// Scheduler context, saved when switching into a process.
static SCHED_RSP: AtomicU64 = AtomicU64::new(0);
/// Set by the timer interrupt; consumed by `safepoint`.
pub static PREEMPT: AtomicBool = AtomicBool::new(false);

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
    CURRENT.load(Ordering::Relaxed)
}

pub fn is_alive(pid: Pid) -> bool {
    TABLE.lock().get(&pid).is_some_and(|p| p.state != State::Dead)
}

pub fn spawn(entry: extern "C" fn(u64), arg: u64) -> Pid {
    spawn_inner(entry, arg, None)
}

/// Spawn and atomically link to the current process.
pub fn spawn_link(entry: extern "C" fn(u64), arg: u64) -> Pid {
    spawn_inner(entry, arg, Some(current()))
}

fn spawn_inner(entry: extern "C" fn(u64), arg: u64, link_to: Option<Pid>) -> Pid {
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

    let span = crate::mm::alloc_contig(HEAP_PAGES, 1).expect("no frames for process heap");
    let heap = unsafe { Heap::new(crate::mm::phys_to_virt(span), HEAP_PAGES * 4096) };

    let pid = NEXT_PID.fetch_add(1, Ordering::Relaxed);
    let mut links = Vec::new();
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
                heap_span: span,
                mailbox: VecDeque::new(),
                links,
                monitors: Vec::new(),
                exit_reason: "normal",
            }),
        );
    }
    RUNQ.lock().push_back(pid);
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
    let pid = current();
    let mut t = TABLE.lock();
    f(&mut t.get_mut(&pid).expect("no current process").heap)
}

/// Raw pointer to the current process's heap, for the execution engine.
///
/// Sound on single-core cooperative scheduling: only the running process (or
/// copy-on-send under the table lock while it's switched *out*) touches it.
pub fn current_heap_ptr() -> *mut Heap {
    let pid = current();
    let mut t = TABLE.lock();
    &raw mut t.get_mut(&pid).expect("no current process").heap
}

/// Build a term or die of quota breach.
pub fn build(f: impl FnOnce(&mut Heap) -> Result<Term, HeapFull>) -> Term {
    match with_heap(f) {
        Ok(t) => t,
        Err(HeapFull) => exit("heap quota exceeded"),
    }
}

/// Switch from the current process back to the scheduler.
fn switch_to_scheduler(save: *mut u64) {
    unsafe { ctx::switch(save, SCHED_RSP.load(Ordering::Relaxed)) }
}

pub fn yield_now() {
    let pid = current();
    debug_assert!(pid != 0, "yield outside a process");
    let (rsp_ptr, requeue) = {
        let mut t = TABLE.lock();
        let p = t.get_mut(&pid).unwrap();
        // Someone may have killed us since our last safepoint: stay Dead and
        // let the scheduler reap us (mark_dead already queued the pid).
        if p.state != State::Dead {
            p.state = State::Runnable;
        }
        (&raw mut p.rsp, p.state == State::Runnable)
    };
    if requeue {
        RUNQ.lock().push_back(pid);
    }
    switch_to_scheduler(rsp_ptr);
}

/// Interpreter back-edge / native-body poll point.
pub fn safepoint() {
    if PREEMPT.swap(false, Ordering::Relaxed) && current() != 0 {
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
        if p.state != State::Dead {
            // Marking the *current* process works too: it keeps running until
            // its next safepoint/receive, which refuses to resurrect it.
            p.state = State::Dead;
            p.exit_reason = reason;
            // Dead processes are reaped when popped from the run queue.
            RUNQ.lock().push_back(pid);
        }
    }
}

/// Copy `msg` (rooted in the sender's heap) into `to`'s heap and enqueue it.
/// Returns false if the target is gone. A receiver whose heap can't hold the
/// message dies of quota breach (no GC yet to save it).
pub fn send(to: Pid, msg: Term) -> bool {
    let mut t = TABLE.lock();
    send_locked(&mut t, to, msg)
}

fn send_locked(t: &mut Table, to: Pid, msg: Term) -> bool {
    let Some(p) = t.get_mut(&to) else { return false };
    if p.state == State::Dead {
        return false;
    }
    match unsafe { copy_term(msg, &mut p.heap) } {
        Ok(copied) => {
            p.mailbox.push_back(copied);
            if p.state == State::Waiting {
                p.state = State::Runnable;
                RUNQ.lock().push_back(to);
            }
            true
        }
        Err(HeapFull) => {
            mark_dead(t, to, "heap quota exceeded (mailbox overflow)");
            false
        }
    }
}

/// Build a message directly in `to`'s heap and enqueue it (kernel-side
/// senders: DOWN messages, port completions). Returns false if the target is
/// gone; a target whose heap is full dies of quota breach.
pub fn send_built(to: Pid, build: impl FnOnce(&mut Heap) -> Result<Term, HeapFull>) -> bool {
    let mut t = TABLE.lock();
    let Some(p) = t.get_mut(&to) else { return false };
    if p.state == State::Dead {
        return false;
    }
    match build(&mut p.heap) {
        Ok(m) => {
            p.mailbox.push_back(m);
            if p.state == State::Waiting {
                p.state = State::Runnable;
                RUNQ.lock().push_back(to);
            }
            true
        }
        Err(HeapFull) => {
            mark_dead(&mut t, to, "heap quota exceeded (mailbox overflow)");
            false
        }
    }
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
            if p.state == State::Dead {
                // Killed since our last safepoint: don't take a message, just
                // hand control back so the scheduler reaps us.
                let ptr = &raw mut p.rsp;
                drop(t);
                switch_to_scheduler(ptr);
                unreachable!("dead process rescheduled");
            }
            if let Some(i) = p.mailbox.iter().position(|&m| pred(m)) {
                return p.mailbox.remove(i);
            }
            if let Some(d) = deadline
                && crate::irq::ticks() >= d
            {
                return None;
            }
            p.state = State::Waiting;
            &raw mut p.rsp
        };
        if let Some(d) = deadline {
            WHEEL.lock().entry(d).or_default().push(pid);
        }
        // A send() (or wheel wake) between unlock and switch re-queues us —
        // single core, no lost wakeup.
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
            if let Some(p) = t.get_mut(&pid) {
                if p.state == State::Waiting {
                    p.state = State::Runnable;
                    RUNQ.lock().push_back(pid);
                }
            }
        }
    }
}

/// The scheduler loop. Runs on the boot stack; never returns.
pub fn run() -> ! {
    loop {
        check_wheel();
        crate::ports::pump();
        let Some(pid) = RUNQ.lock().pop_front() else {
            // Nothing runnable: wait for an interrupt to change that.
            x86_64::instructions::interrupts::enable_and_hlt();
            continue;
        };
        let rsp = {
            let mut t = TABLE.lock();
            match t.get_mut(&pid) {
                Some(p) if p.state == State::Runnable => {
                    p.state = State::Running;
                    p.rsp
                }
                Some(p) if p.state == State::Dead => {
                    reap(&mut t, pid);
                    continue;
                }
                // Stale queue entry (waiting/duplicate/gone): skip.
                _ => continue,
            }
        };
        CURRENT.store(pid, Ordering::Relaxed);
        unsafe { ctx::switch(SCHED_RSP.as_ptr(), rsp) };
        CURRENT.store(0, Ordering::Relaxed);

        let mut t = TABLE.lock();
        if t.get(&pid).is_some_and(|p| p.state == State::Dead) {
            reap(&mut t, pid);
        }
    }
}

/// Free a dead process and deliver its exit signals: DOWN messages to
/// monitors, death to links. Link propagation is hop-by-hop: linked processes
/// are marked dead here and their own reap continues the cascade.
fn reap(t: &mut Table, pid: Pid) {
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

    crate::vmm::unmap_stack(p.stack_slot);
    crate::mm::free_frames(p.heap_span, HEAP_PAGES);
    crate::ports::close_owned_by(pid);
}

fn deliver_down(t: &mut Table, watcher: Pid, r: u64, dead: Pid, reason: &'static str) {
    let down = atoms::intern("DOWN");
    let reason = atoms::intern(reason);
    let Some(w) = t.get_mut(&watcher) else { return };
    if w.state == State::Dead {
        return;
    }
    let msg = w.heap.tuple(&[
        Term::atom(down),
        Term::reference(r),
        Term::pid(dead),
        Term::atom(reason),
    ]);
    match msg {
        Ok(m) => {
            w.mailbox.push_back(m);
            if w.state == State::Waiting {
                w.state = State::Runnable;
                RUNQ.lock().push_back(watcher);
            }
        }
        Err(HeapFull) => mark_dead(t, watcher, "heap quota exceeded (mailbox overflow)"),
    }
}
