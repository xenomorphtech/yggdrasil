//! AP bring-up via Limine's MP protocol.
//!
//! `percpu::init` already sized the CPU table from the MP response (kmain).
//! Here the BSP hands each AP an index through `MpInfo::bootstrap`; the AP
//! arrives on Limine's page tables + stack, switches to ours, binds its
//! per-CPU state, starts its LAPIC timer and enters the scheduler on an
//! owned stack.

use core::sync::atomic::{AtomicUsize, Ordering};

use limine::mp::MpInfo;
use x86_64::PhysAddr;
use x86_64::registers::control::{Cr3, Cr3Flags};
use x86_64::structures::paging::PhysFrame;

use crate::{boot, percpu};

static ONLINE: AtomicUsize = AtomicUsize::new(1); // BSP

pub fn online() -> usize {
    ONLINE.load(Ordering::Acquire)
}

/// Boot every AP and wait for them to reach their scheduler loops.
pub fn init() {
    let Some(resp) = boot::MP.response() else {
        log::info!("[smp] no MP response; single core");
        return;
    };
    let bsp = resp.bsp_lapic_id;
    let mut next_index = 1u64;
    for info in resp.cpus() {
        if info.lapic_id == bsp {
            continue;
        }
        let index = next_index;
        next_index += 1;
        percpu::all()[index as usize]
            .lapic_id
            .store(info.lapic_id, Ordering::Release);
        info.bootstrap(ap_entry, index);
    }
    let expected = next_index as usize;
    while online() < expected {
        core::hint::spin_loop();
    }
    crate::println!("[smp] {} cpus online", online());
}

/// Flush stale stack-zone TLB entries on every other core before a VA slot is
/// recycled. Caller may hold the MAPPER lock: the receiving handler is
/// lock-free (CR3 reload + atomic ack), so spinning here cannot deadlock.
pub fn flush_broadcast() {
    let others = online().saturating_sub(1);
    if others == 0 {
        return;
    }
    // Serialize shootdowns: the ack counter is shared. Waiters MUST spin with
    // interrupts enabled so they keep servicing the other side's flush IPIs —
    // two cores shooting down concurrently would otherwise deadlock.
    x86_64::instructions::interrupts::enable();
    static SHOOTDOWN: spin::Mutex<()> = spin::Mutex::new(());
    let _guard = SHOOTDOWN.lock();
    crate::irq::TLB_ACKS.store(0, Ordering::Release);
    crate::irq::ipi_all_others(crate::irq::TLB_FLUSH_VECTOR);
    let mut spins = 0u64;
    while crate::irq::TLB_ACKS.load(Ordering::Acquire) < others {
        core::hint::spin_loop();
        spins += 1;
        if spins > 4_000_000_000 {
            panic!("tlb shootdown timed out ({} acks missing)", others);
        }
    }
}

/// Panic path: stop every other core so it can't interleave output or keep
/// mutating state under the panicking CPU.
pub fn halt_others() {
    if online() > 1 {
        crate::irq::ipi_all_others(crate::irq::HALT_VECTOR);
    }
}

unsafe extern "C" fn ap_entry(info: &MpInfo) -> ! {
    // Still on Limine's stack and page tables here.
    unsafe {
        Cr3::write(
            PhysFrame::containing_address(PhysAddr::new(crate::vmm::pml4_phys())),
            Cr3Flags::empty(),
        );
    }
    let index = info.extra_argument() as usize;
    let cpu = percpu::all()[index];
    percpu::bind(cpu);
    crate::gdt::init_cpu();
    crate::idt::load_ap();
    crate::irq::init_ap();

    // Trade Limine's (bootloader-reclaimable) stack for one we own.
    let (_slot, top) = crate::vmm::map_stack();
    unsafe { enter_scheduler(top) }
}

#[unsafe(naked)]
unsafe extern "C" fn enter_scheduler(new_rsp: u64) -> ! {
    core::arch::naked_asm!(
        "mov rsp, rdi",
        "call {main}",
        main = sym ap_main,
    )
}

extern "C" fn ap_main() -> ! {
    ONLINE.fetch_add(1, Ordering::Release);
    x86_64::instructions::interrupts::enable();
    crate::proc::run()
}
