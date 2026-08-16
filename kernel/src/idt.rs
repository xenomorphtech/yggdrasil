//! IDT: all exceptions land in loud panic handlers for now.
//!
//! Breakpoint (#BP) is the exception: it records itself and returns, giving the
//! self-test a recoverable way to prove exception dispatch works. The page
//! fault handler grows a guard-page classifier in M2.

use core::sync::atomic::{AtomicBool, Ordering};

use spin::LazyLock;
use x86_64::instructions::port::Port;
use x86_64::registers::control::Cr2;
use x86_64::structures::idt::{InterruptDescriptorTable, InterruptStackFrame, PageFaultErrorCode};

use crate::gdt;

pub static BREAKPOINT_HIT: AtomicBool = AtomicBool::new(false);

static IDT: LazyLock<InterruptDescriptorTable> = LazyLock::new(|| {
    let mut idt = InterruptDescriptorTable::new();
    idt.divide_error.set_handler_fn(divide_error);
    idt.debug.set_handler_fn(debug);
    idt.breakpoint.set_handler_fn(breakpoint);
    idt.overflow.set_handler_fn(overflow);
    idt.bound_range_exceeded.set_handler_fn(bound_range);
    idt.invalid_opcode.set_handler_fn(invalid_opcode);
    idt.device_not_available.set_handler_fn(device_not_available);
    idt.invalid_tss.set_handler_fn(invalid_tss);
    idt.segment_not_present.set_handler_fn(segment_not_present);
    idt.stack_segment_fault.set_handler_fn(stack_segment_fault);
    idt.general_protection_fault.set_handler_fn(general_protection);
    idt.x87_floating_point.set_handler_fn(x87_fp);
    idt.alignment_check.set_handler_fn(alignment_check);
    idt.simd_floating_point.set_handler_fn(simd_fp);
    unsafe {
        idt.double_fault
            .set_handler_fn(double_fault)
            .set_stack_index(gdt::DOUBLE_FAULT_IST);
        idt.non_maskable_interrupt
            .set_handler_fn(nmi)
            .set_stack_index(gdt::NMI_IST);
        idt.machine_check
            .set_handler_fn(machine_check)
            .set_stack_index(gdt::MACHINE_CHECK_IST);
        idt.page_fault
            .set_handler_fn(page_fault)
            .set_stack_index(gdt::PAGE_FAULT_IST);
    }
    idt[crate::irq::TIMER_VECTOR].set_handler_fn(crate::irq::timer_interrupt);
    idt[crate::irq::SERIAL_VECTOR].set_handler_fn(crate::irq::serial_interrupt);
    idt[crate::irq::SPURIOUS_VECTOR].set_handler_fn(crate::irq::spurious_interrupt);
    idt
});

pub fn init() {
    IDT.load();
    remap_and_mask_pics();
}

/// Remap the legacy PICs away from the exception vectors, then mask everything.
/// Interrupts proper arrive via the LAPIC/IOAPIC in M1.
fn remap_and_mask_pics() {
    unsafe {
        let mut cmd1 = Port::<u8>::new(0x20);
        let mut dat1 = Port::<u8>::new(0x21);
        let mut cmd2 = Port::<u8>::new(0xA0);
        let mut dat2 = Port::<u8>::new(0xA1);
        cmd1.write(0x11u8); // ICW1: init, expect ICW4
        cmd2.write(0x11u8);
        dat1.write(0x20u8); // ICW2: vector offsets 0x20 / 0x28
        dat2.write(0x28u8);
        dat1.write(0x04u8); // ICW3: wiring
        dat2.write(0x02u8);
        dat1.write(0x01u8); // ICW4: 8086 mode
        dat2.write(0x01u8);
        dat1.write(0xFFu8); // mask all
        dat2.write(0xFFu8);
    }
}

macro_rules! fatal_handler {
    ($name:ident, $label:expr) => {
        extern "x86-interrupt" fn $name(frame: InterruptStackFrame) {
            panic!("EXCEPTION: {} at {:?}\n{:#?}", $label, frame.instruction_pointer, frame);
        }
    };
    ($name:ident, $label:expr, err) => {
        extern "x86-interrupt" fn $name(frame: InterruptStackFrame, code: u64) {
            panic!(
                "EXCEPTION: {} (error {:#x}) at {:?}\n{:#?}",
                $label, code, frame.instruction_pointer, frame
            );
        }
    };
}

fatal_handler!(divide_error, "divide error (#DE, vector 0)");
fatal_handler!(debug, "debug (#DB, vector 1)");
fatal_handler!(nmi, "non-maskable interrupt (vector 2)");
fatal_handler!(overflow, "overflow (#OF, vector 4)");
fatal_handler!(bound_range, "bound range exceeded (#BR, vector 5)");
fatal_handler!(invalid_opcode, "invalid opcode (#UD, vector 6)");
fatal_handler!(device_not_available, "device not available (#NM, vector 7)");
fatal_handler!(invalid_tss, "invalid TSS (#TS, vector 10)", err);
fatal_handler!(segment_not_present, "segment not present (#NP, vector 11)", err);
fatal_handler!(stack_segment_fault, "stack segment fault (#SS, vector 12)", err);
fatal_handler!(general_protection, "general protection fault (#GP, vector 13)", err);
fatal_handler!(x87_fp, "x87 floating point (#MF, vector 16)");
fatal_handler!(alignment_check, "alignment check (#AC, vector 17)", err);
fatal_handler!(simd_fp, "SIMD floating point (#XM, vector 19)");

extern "x86-interrupt" fn breakpoint(frame: InterruptStackFrame) {
    BREAKPOINT_HIT.store(true, Ordering::SeqCst);
    crate::println!(
        "[int3] breakpoint (#BP, vector 3) at {:?} — resuming",
        frame.instruction_pointer
    );
}

extern "x86-interrupt" fn page_fault(frame: InterruptStackFrame, code: PageFaultErrorCode) {
    // A fault inside the stack zone is a process blowing through its guard
    // page: kill that process, not the kernel. (Runs on its own IST stack, so
    // this is reachable even when rsp itself is in the guard page.)
    if let Ok(addr) = Cr2::read()
        && crate::vmm::in_stack_zone(addr.as_u64())
        && crate::proc::current() != 0
    {
        crate::proc::fatal_current("stack overflow (guard page)");
    }
    panic!(
        "EXCEPTION: page fault (#PF, vector 14) addr={:?} error={:?}\n{:#?}",
        Cr2::read(),
        code,
        frame
    );
}

extern "x86-interrupt" fn double_fault(frame: InterruptStackFrame, code: u64) -> ! {
    panic!(
        "EXCEPTION: double fault (#DF, vector 8, error {:#x})\n{:#?}",
        code, frame
    );
}

extern "x86-interrupt" fn machine_check(frame: InterruptStackFrame) -> ! {
    panic!("EXCEPTION: machine check (#MC, vector 18)\n{:#?}", frame);
}
