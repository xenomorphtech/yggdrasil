//! Kernel-owned page tables.
//!
//! Single address space, rebuilt once at boot (replacing Limine's tables) for:
//! - W^X on the kernel image (text RX, rodata R, data RW+NX)
//! - an HHDM covering all physical memory *and* low MMIO (LAPIC/IOAPIC/ECAM)
//! - a stack zone: per-process native stacks with unmapped guard pages below
//!
//! There are never per-process page tables — isolation is the verifier's job.

use limine::memmap::MEMMAP_USABLE;
use spin::Mutex;
use x86_64::registers::control::{Cr0, Cr0Flags, Cr3, Cr4, Cr4Flags};
use x86_64::registers::model_specific::{Efer, EferFlags};
use x86_64::structures::paging::{
    FrameAllocator, Mapper, OffsetPageTable, Page, PageTable, PageTableFlags as F, PhysFrame,
    Size2MiB, Size4KiB, Translate,
};
use x86_64::{PhysAddr, VirtAddr};

use crate::{boot, mm};

const STACK_ZONE_BASE: u64 = 0xffff_9000_0000_0000;
/// VA stride per stack slot; the mapped stack sits at the top, all below is guard.
const SLOT_STRIDE: u64 = 64 * 4096;
pub const STACK_PAGES: u64 = 56; // 224 KiB usable stack (deep bytecode call chains)
const MAX_SLOTS: u64 = 65536;

unsafe extern "C" {
    static __image_start: u8;
    static __text_start: u8;
    static __rodata_start: u8;
    static __data_start: u8;
    static __kernel_end: u8;
}

struct PmmFrames;

unsafe impl FrameAllocator<Size4KiB> for PmmFrames {
    fn allocate_frame(&mut self) -> Option<PhysFrame> {
        let phys = mm::alloc_frame()?;
        // Page-table frames must start zeroed.
        unsafe { mm::phys_to_virt(phys).write_bytes(0, 4096) };
        Some(PhysFrame::containing_address(PhysAddr::new(phys)))
    }
}

static MAPPER: Mutex<Option<OffsetPageTable<'static>>> = Mutex::new(None);
/// Physical address of our PML4 (APs load it into CR3 on entry).
static PML4_PHYS: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

pub fn pml4_phys() -> u64 {
    PML4_PHYS.load(core::sync::atomic::Ordering::Relaxed)
}
static STACK_SLOTS: Mutex<SlotAllocator> = Mutex::new(SlotAllocator::new());

struct SlotAllocator {
    next: u64,
    freed: alloc::vec::Vec<u64>,
}

impl SlotAllocator {
    const fn new() -> Self {
        Self {
            next: 0,
            freed: alloc::vec::Vec::new(),
        }
    }
    fn alloc(&mut self) -> u64 {
        self.freed.pop().unwrap_or_else(|| {
            let s = self.next;
            self.next += 1;
            assert!(s < MAX_SLOTS, "stack zone exhausted");
            s
        })
    }
}

pub fn init() {
    unsafe {
        Efer::update(|f| f.insert(EferFlags::NO_EXECUTE_ENABLE));
        Cr0::update(|f| f.insert(Cr0Flags::WRITE_PROTECT));
        Cr4::update(|f| f.insert(Cr4Flags::PAGE_GLOBAL));
    }

    let hhdm = mm::hhdm_offset();
    let pml4_phys = mm::alloc_frame().expect("no frame for PML4");
    let pml4 = unsafe { &mut *(mm::phys_to_virt(pml4_phys) as *mut PageTable) };
    pml4.zero();
    let mut mapper = unsafe { OffsetPageTable::new(pml4, VirtAddr::new(hhdm)) };

    // HHDM: 2 MiB pages over all RAM and low MMIO. Cover at least 4 GiB so the
    // LAPIC/IOAPIC/ECAM windows below 4G are reachable.
    let memmap = boot::MEMMAP.response().expect("no memmap");
    let ram_top = memmap
        .entries()
        .iter()
        .filter(|e| e.type_ == MEMMAP_USABLE)
        .map(|e| e.base + e.length)
        .max()
        .unwrap_or(0);
    let top = ram_top
        .max(4 * 1024 * 1024 * 1024)
        .next_multiple_of(2 * 1024 * 1024);
    let hhdm_flags = F::PRESENT | F::WRITABLE | F::NO_EXECUTE | F::GLOBAL;
    for phys in (0..top).step_by(2 * 1024 * 1024) {
        let page = Page::<Size2MiB>::containing_address(VirtAddr::new(hhdm + phys));
        let frame = PhysFrame::<Size2MiB>::containing_address(PhysAddr::new(phys));
        unsafe {
            mapper
                .map_to(page, frame, hhdm_flags, &mut PmmFrames)
                .expect("hhdm map failed")
                .ignore();
        }
    }

    // Kernel image with W^X, 4 KiB granularity.
    let addr = boot::EXECUTABLE_ADDRESS
        .response()
        .expect("no executable address");
    let (vbase, pbase) = (addr.virtual_base, addr.physical_base);
    let sym = |s: &u8| VirtAddr::from_ptr(s).as_u64();
    let (image, text, rodata, data, end) = unsafe {
        (
            sym(&__image_start),
            sym(&__text_start),
            sym(&__rodata_start),
            sym(&__data_start),
            sym(&__kernel_end),
        )
    };
    let ranges = [
        (image, text, F::PRESENT | F::NO_EXECUTE | F::GLOBAL),
        (text, rodata, F::PRESENT | F::GLOBAL),
        (rodata, data, F::PRESENT | F::NO_EXECUTE | F::GLOBAL),
        (
            data,
            end,
            F::PRESENT | F::WRITABLE | F::NO_EXECUTE | F::GLOBAL,
        ),
    ];
    for (start, stop, flags) in ranges {
        for va in (start..stop).step_by(4096) {
            let page = Page::<Size4KiB>::containing_address(VirtAddr::new(va));
            let phys = pbase + (va - vbase);
            let frame = PhysFrame::containing_address(PhysAddr::new(phys));
            unsafe {
                mapper
                    .map_to(page, frame, flags, &mut PmmFrames)
                    .expect("kernel image map failed")
                    .ignore();
            }
        }
    }

    unsafe {
        Cr3::write(
            PhysFrame::containing_address(PhysAddr::new(pml4_phys)),
            x86_64::registers::control::Cr3Flags::empty(),
        );
    }
    PML4_PHYS.store(pml4_phys, core::sync::atomic::Ordering::Relaxed);
    *MAPPER.lock() = Some(mapper);
    log::info!(
        "vmm: own page tables loaded (hhdm 2MiB x {}, image {:#x}..{:#x} W^X)",
        top / (2 * 1024 * 1024),
        image,
        end
    );
}

/// Map a fresh process stack; returns (slot, stack_top_va).
///
/// The backing frames are physically contiguous so DMA-able buffers on a
/// process stack (e.g. virtio request/response structs) never straddle
/// discontiguous frames.
pub fn map_stack() -> (u64, u64) {
    let slot = STACK_SLOTS.lock().alloc();
    let slot_base = STACK_ZONE_BASE + slot * SLOT_STRIDE;
    let stack_base = slot_base + SLOT_STRIDE - STACK_PAGES * 4096;
    let span = mm::alloc_contig(STACK_PAGES as usize, 1).expect("no frames for stack");
    let mut guard = MAPPER.lock();
    let mapper = guard.as_mut().expect("vmm uninitialized");
    for i in 0..STACK_PAGES {
        let frame = PhysFrame::containing_address(PhysAddr::new(span + i * 4096));
        let page = Page::<Size4KiB>::containing_address(VirtAddr::new(stack_base + i * 4096));
        unsafe {
            mapper
                .map_to(
                    page,
                    frame,
                    F::PRESENT | F::WRITABLE | F::NO_EXECUTE,
                    &mut PmmFrames,
                )
                .expect("stack map failed")
                .flush();
        }
    }
    (slot, slot_base + SLOT_STRIDE)
}

/// Translate any mapped virtual address via the live page tables.
pub fn translate(virt: u64) -> Option<u64> {
    let guard = MAPPER.lock();
    guard
        .as_ref()?
        .translate_addr(VirtAddr::new(virt))
        .map(|p| p.as_u64())
}

/// JIT code zone: within ±2 GiB of the kernel image so PC-relative calls
/// between generated code and (potentially) kernel text stay encodable.
const CODE_ZONE_BASE: u64 = 0xffff_ffff_7000_0000;
static CODE_ZONE_NEXT: Mutex<u64> = Mutex::new(CODE_ZONE_BASE);

/// Map `span` (physically contiguous frames) executable + read-only in the
/// code zone. Writes go through the HHDM alias — W^X by aliasing.
pub fn map_code(span_phys: u64, pages: u64) -> u64 {
    let va = {
        let mut next = CODE_ZONE_NEXT.lock();
        let va = *next;
        // Guard gap between allocations.
        *next += (pages + 1) * 4096;
        va
    };
    let mut guard = MAPPER.lock();
    let mapper = guard.as_mut().expect("vmm uninitialized");
    for i in 0..pages {
        let frame = PhysFrame::containing_address(PhysAddr::new(span_phys + i * 4096));
        let page = Page::<Size4KiB>::containing_address(VirtAddr::new(va + i * 4096));
        unsafe {
            mapper
                .map_to(page, frame, F::PRESENT | F::GLOBAL, &mut PmmFrames)
                .expect("code map failed")
                .flush();
        }
    }
    va
}

pub fn unmap_stack(slot: u64) {
    let slot_base = STACK_ZONE_BASE + slot * SLOT_STRIDE;
    let stack_base = slot_base + SLOT_STRIDE - STACK_PAGES * 4096;
    let mut guard = MAPPER.lock();
    let mapper = guard.as_mut().expect("vmm uninitialized");
    let mut frames = [0u64; STACK_PAGES as usize];
    for i in 0..STACK_PAGES {
        let page = Page::<Size4KiB>::containing_address(VirtAddr::new(stack_base + i * 4096));
        let (frame, flush) = mapper.unmap(page).expect("stack unmap failed");
        flush.flush();
        frames[i as usize] = frame.start_address().as_u64();
    }
    // Other cores may hold stale TLB entries for this VA range; flush them
    // before the frames or the slot can be reused.
    crate::smp::flush_broadcast();
    for f in frames {
        mm::free_frames(f, 1);
    }
    STACK_SLOTS.lock().freed.push(slot);
}

/// Does this address fall in the stack zone (i.e. a guard-page hit on overflow)?
pub fn in_stack_zone(addr: u64) -> bool {
    (STACK_ZONE_BASE..STACK_ZONE_BASE + MAX_SLOTS * SLOT_STRIDE).contains(&addr)
}
