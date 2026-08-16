//! GDT, TSS and IST stacks for fatal exceptions.

use spin::LazyLock;
use x86_64::VirtAddr;
use x86_64::instructions::tables::load_tss;
use x86_64::registers::segmentation::{CS, DS, ES, SS, Segment};
use x86_64::structures::gdt::{Descriptor, GlobalDescriptorTable, SegmentSelector};
use x86_64::structures::tss::TaskStateSegment;

pub const DOUBLE_FAULT_IST: u16 = 0;
pub const NMI_IST: u16 = 1;
pub const MACHINE_CHECK_IST: u16 = 2;
/// #PF gets its own stack so a guard-page hit (rsp in unmapped guard) can be
/// classified and the offending process killed instead of double-faulting.
pub const PAGE_FAULT_IST: u16 = 3;

const IST_STACK_SIZE: usize = 4096 * 8;

#[repr(align(16))]
struct IstStack(#[allow(dead_code)] [u8; IST_STACK_SIZE]);

static mut IST_STACKS: [IstStack; 4] = [
    IstStack([0; IST_STACK_SIZE]),
    IstStack([0; IST_STACK_SIZE]),
    IstStack([0; IST_STACK_SIZE]),
    IstStack([0; IST_STACK_SIZE]),
];

static TSS: LazyLock<TaskStateSegment> = LazyLock::new(|| {
    let mut tss = TaskStateSegment::new();
    for i in 0..4 {
        let start = VirtAddr::from_ptr(unsafe { (&raw const IST_STACKS[i]).cast::<u8>() });
        // Stacks grow down: point at the top.
        tss.interrupt_stack_table[i] = start + IST_STACK_SIZE as u64;
    }
    tss
});

struct Selectors {
    code: SegmentSelector,
    data: SegmentSelector,
    tss: SegmentSelector,
}

static GDT: LazyLock<(GlobalDescriptorTable, Selectors)> = LazyLock::new(|| {
    let mut gdt = GlobalDescriptorTable::new();
    let code = gdt.append(Descriptor::kernel_code_segment());
    let data = gdt.append(Descriptor::kernel_data_segment());
    let tss = gdt.append(Descriptor::tss_segment(&TSS));
    (gdt, Selectors { code, data, tss })
});

pub fn init() {
    GDT.0.load();
    unsafe {
        CS::set_reg(GDT.1.code);
        DS::set_reg(GDT.1.data);
        ES::set_reg(GDT.1.data);
        SS::set_reg(GDT.1.data);
        load_tss(GDT.1.tss);
    }
}
