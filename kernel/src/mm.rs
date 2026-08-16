//! Memory management: physical frame allocator (PMM) + kernel heap.
//!
//! Single address space: physical memory is reached through Limine's HHDM.
//! The kernel heap is a static talc arena; more spans can be claimed from the
//! PMM later if it runs low.

use limine::memmap::MEMMAP_USABLE;
use spin::Mutex;
use talc::source::Claim;
use talc::{DefaultBinning, TalcLock, min_first_heap_size};
use ygg_alloc::{FRAME_SIZE, FrameBitmap};

use crate::boot;

const KERNEL_HEAP_SIZE: usize = min_first_heap_size::<DefaultBinning>() + 16 * 1024 * 1024;

static mut KERNEL_HEAP: [u8; KERNEL_HEAP_SIZE] = [0; KERNEL_HEAP_SIZE];

#[global_allocator]
static TALC: TalcLock<spinning_top::RawSpinlock, Claim> =
    TalcLock::new(unsafe { Claim::array(&raw mut KERNEL_HEAP) });

static PMM: Mutex<Option<FrameBitmap<'static>>> = Mutex::new(None);

pub fn hhdm_offset() -> u64 {
    boot::HHDM.response().expect("no HHDM").offset
}

pub fn phys_to_virt(phys: u64) -> *mut u8 {
    (phys + hhdm_offset()) as *mut u8
}

/// Reverse mapping for the two kernel address ranges: the HHDM and the kernel
/// image (where the talc heap's static arena lives). Both are physically
/// contiguous, so DMA on any kernel buffer is safe.
pub fn virt_to_phys(virt: u64) -> u64 {
    const IMAGE_BASE: u64 = 0xffff_ffff_8000_0000;
    if virt >= IMAGE_BASE {
        let a = crate::boot::EXECUTABLE_ADDRESS.response().expect("no executable address");
        a.physical_base + (virt - a.virtual_base)
    } else if crate::vmm::in_stack_zone(virt) {
        // Process stacks: walk the page tables (frames per stack are
        // contiguous, so multi-page buffers stay DMA-safe).
        crate::vmm::translate(virt).expect("unmapped stack address")
    } else {
        virt - hhdm_offset()
    }
}

pub fn init() {
    let memmap = boot::MEMMAP.response().expect("no memmap");
    let entries = memmap.entries();

    // Track frames from 0 to the end of the highest usable region.
    let top = entries
        .iter()
        .filter(|e| e.type_ == MEMMAP_USABLE)
        .map(|e| e.base + e.length)
        .max()
        .expect("no usable memory");
    let nframes = (top / FRAME_SIZE) as usize;
    let words = FrameBitmap::storage_words(nframes);
    let storage_bytes = (words * 8) as u64;

    // Carve the bitmap storage itself out of the first usable region that fits.
    let host = entries
        .iter()
        .find(|e| e.type_ == MEMMAP_USABLE && e.length >= storage_bytes)
        .expect("no region can host the frame bitmap");
    let storage: &'static mut [u64] = unsafe {
        core::slice::from_raw_parts_mut(phys_to_virt(host.base).cast::<u64>(), words)
    };

    let mut bitmap = FrameBitmap::new(0, nframes, storage);
    for e in entries.iter().filter(|e| e.type_ == MEMMAP_USABLE) {
        bitmap.mark_free_range(e.base, e.length);
    }
    // The bitmap lives in a usable region: re-reserve it.
    bitmap.mark_used_range(host.base, storage_bytes);
    // Never hand out frame 0 (null-ish physical addresses breed bugs).
    if nframes > 0 {
        bitmap.mark_used_range(0, FRAME_SIZE);
    }

    let free = bitmap.free_count();
    *PMM.lock() = Some(bitmap);
    log::info!(
        "pmm: {} frames tracked, {} free ({} MiB)",
        nframes,
        free,
        free as u64 * FRAME_SIZE / (1024 * 1024)
    );
}

pub fn alloc_frame() -> Option<u64> {
    PMM.lock().as_mut()?.alloc()
}

/// `n` physically contiguous frames, base aligned to `align_frames` frames.
pub fn alloc_contig(n: usize, align_frames: usize) -> Option<u64> {
    PMM.lock().as_mut()?.alloc_contig(n, align_frames)
}

pub fn free_frames(addr: u64, n: usize) {
    PMM.lock().as_mut().expect("pmm uninitialized").free_frames(addr, n);
}

pub fn free_frame_count() -> usize {
    PMM.lock().as_ref().map_or(0, |b| b.free_count())
}
