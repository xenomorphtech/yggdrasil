//! Per-CPU GDT, TSS and IST stacks for fatal exceptions.
//!
//! Every core builds and loads its own tables (leaked heap allocations — the
//! kernel heap is a static arena, alive before any of this runs). The IDT is
//! shared; the IST stack *storage* must not be.

use alloc::boxed::Box;

use x86_64::VirtAddr;
use x86_64::instructions::tables::load_tss;
use x86_64::registers::segmentation::{CS, DS, ES, SS, Segment};
use x86_64::structures::gdt::{Descriptor, GlobalDescriptorTable};
use x86_64::structures::tss::TaskStateSegment;

pub const DOUBLE_FAULT_IST: u16 = 0;
pub const NMI_IST: u16 = 1;
pub const MACHINE_CHECK_IST: u16 = 2;
/// #PF gets its own stack so a guard-page hit (rsp in unmapped guard) can be
/// classified and the offending process killed instead of double-faulting.
pub const PAGE_FAULT_IST: u16 = 3;

const IST_STACK_SIZE: usize = 4096 * 8;

#[repr(align(16))]
struct IstStack([u8; IST_STACK_SIZE]);

/// Build and load this core's GDT + TSS (call exactly once per core).
pub fn init_cpu() {
    let tss: &'static mut TaskStateSegment = Box::leak(Box::new(TaskStateSegment::new()));
    for i in 0..4 {
        let stack: &'static mut IstStack = Box::leak(Box::new(IstStack([0; IST_STACK_SIZE])));
        let start = VirtAddr::from_ptr(stack.0.as_ptr());
        // Stacks grow down: point at the top.
        tss.interrupt_stack_table[i] = start + IST_STACK_SIZE as u64;
    }

    let gdt: &'static mut GlobalDescriptorTable = Box::leak(Box::new(GlobalDescriptorTable::new()));
    let code = gdt.append(Descriptor::kernel_code_segment());
    let data = gdt.append(Descriptor::kernel_data_segment());
    let tss_sel = gdt.append(Descriptor::tss_segment(tss));

    gdt.load();
    unsafe {
        CS::set_reg(code);
        DS::set_reg(data);
        ES::set_reg(data);
        SS::set_reg(data);
        load_tss(tss_sel);
    }
}

/// Boot-compat alias for the BSP.
pub fn init() {
    init_cpu();
}
