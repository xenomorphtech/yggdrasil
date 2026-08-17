//! Per-CPU state, reached through `gs:[0]`.
//!
//! Each CPU's `Cpu` struct starts with a pointer to itself; `IA32_GS_BASE`
//! points at the struct, so `cpu()` is a single `mov rax, gs:[0]`. Everything
//! that used to be a scheduler-global (current pid, preempt flag, scheduler
//! context, run queue) lives here so the same code runs unchanged on every
//! core.

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use spin::{Mutex, Once};

use crate::proc::Pid;

#[repr(C)]
pub struct Cpu {
    /// Must stay the first field: `cpu()` reads it via gs:[0].
    self_ptr: *const Cpu,
    pub id: u32,
    pub lapic_id: AtomicU32,
    /// Scheduler context, saved when switching into a process.
    pub sched_rsp: AtomicU64,
    /// Pid currently running on this core (0 = scheduler).
    pub current: AtomicU64,
    /// Set by this core's timer interrupt; consumed at safepoints.
    pub preempt: AtomicBool,
    /// `*mut Heap` of the running process — the execution engines' lock-free
    /// heap path. Written only by this core's scheduler at switch-in/out.
    pub current_heap: AtomicU64,
    pub runq: Mutex<VecDeque<Pid>>,
    /// Processes switched in on this core (work-stealing test statistic).
    pub switches: AtomicU64,
    /// Parked in hlt with an empty queue (wake-IPI hint).
    pub idle: AtomicBool,
    /// Deferred state transition for the process that just switched out:
    /// `pid << 2 | kind` (0 = none, 1 = requeue, 2 = block). Written by the
    /// outgoing process, consumed by this core's scheduler *after* the
    /// context is fully saved — this ordering is what makes it safe for
    /// another core to steal the process.
    pub post_switch: AtomicU64,
    /// Debug: which scheduler-loop step this core last entered.
    pub phase: AtomicU64,
    /// The running process's pending tail-call target. Core-local by
    /// construction: it is stashed as an engine function's final act before
    /// returning the tail sentinel and consumed immediately by the trampoline
    /// that made the call — no safepoint or blocking point lies between, so
    /// the window can never span a context switch. The Mutex is uncontended
    /// (same-core access only) and exists for interior mutability.
    pub tail_target: Mutex<Option<crate::proc::TailTarget>>,
}

unsafe impl Sync for Cpu {}
unsafe impl Send for Cpu {}

static CPUS: Once<Vec<&'static Cpu>> = Once::new();

/// Build the per-CPU table. Call once, before `gdt::init` on the BSP.
/// (The kernel heap is a static talc arena, alive from the first instruction.)
pub fn init(count: usize, bsp_lapic_id: u32) {
    CPUS.call_once(|| {
        (0..count)
            .map(|i| {
                let cpu = Box::leak(Box::new(Cpu {
                    self_ptr: core::ptr::null(),
                    id: i as u32,
                    lapic_id: AtomicU32::new(if i == 0 { bsp_lapic_id } else { u32::MAX }),
                    sched_rsp: AtomicU64::new(0),
                    current: AtomicU64::new(0),
                    preempt: AtomicBool::new(false),
                    current_heap: AtomicU64::new(0),
                    runq: Mutex::new(VecDeque::new()),
                    switches: AtomicU64::new(0),
                    idle: AtomicBool::new(false),
                    post_switch: AtomicU64::new(0),
                    phase: AtomicU64::new(0),
                    tail_target: Mutex::new(None),
                }));
                cpu.self_ptr = cpu as *const Cpu;
                &*cpu
            })
            .collect()
    });
}

pub fn all() -> &'static [&'static Cpu] {
    CPUS.get().expect("percpu not initialized")
}

/// Bind the executing core to `cpu` (sets IA32_GS_BASE).
pub fn bind(cpu: &'static Cpu) {
    unsafe {
        x86_64::registers::model_specific::Msr::new(0xC000_0101).write(cpu.self_ptr as u64);
    }
}

/// The executing core's state. Only valid after `bind` ran on this core.
#[inline]
pub fn cpu() -> &'static Cpu {
    unsafe {
        let p: *const Cpu;
        core::arch::asm!("mov {}, gs:[0]", out(reg) p, options(nostack, preserves_flags, pure, readonly));
        &*p
    }
}

pub fn count() -> usize {
    all().len()
}

/// Sum of a stat across cores (test support).
pub fn total_switches() -> u64 {
    all().iter().map(|c| c.switches.load(Ordering::Relaxed)).sum()
}
