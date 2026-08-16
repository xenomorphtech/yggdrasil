//! Pure allocation logic, host-testable: physical frame bitmap (later: the
//! stack-zone VA allocator).
//!
//! The kernel feeds this a memory map and a storage slice; nothing here touches
//! hardware or requires an allocator.

#![cfg_attr(not(test), no_std)]

pub const FRAME_SIZE: u64 = 4096;

/// Bitmap physical-frame allocator. One bit per frame, set = used.
/// Next-fit single-frame allocation, first-fit for contiguous runs.
pub struct FrameBitmap<'a> {
    bits: &'a mut [u64],
    /// Physical address of frame 0.
    base: u64,
    nframes: usize,
    cursor: usize,
    free: usize,
}

impl<'a> FrameBitmap<'a> {
    /// Words of storage needed to track `nframes`.
    pub const fn storage_words(nframes: usize) -> usize {
        nframes.div_ceil(64)
    }

    /// All frames start **used**; mark usable ranges free afterwards.
    pub fn new(base: u64, nframes: usize, storage: &'a mut [u64]) -> Self {
        assert!(storage.len() >= Self::storage_words(nframes));
        assert_eq!(base % FRAME_SIZE, 0);
        storage.fill(u64::MAX);
        Self { bits: storage, base, nframes, cursor: 0, free: 0 }
    }

    fn index_of(&self, addr: u64) -> usize {
        debug_assert_eq!(addr % FRAME_SIZE, 0, "unaligned frame address");
        debug_assert!(addr >= self.base);
        ((addr - self.base) / FRAME_SIZE) as usize
    }

    fn addr_of(&self, index: usize) -> u64 {
        self.base + index as u64 * FRAME_SIZE
    }

    fn is_used(&self, i: usize) -> bool {
        self.bits[i / 64] & (1 << (i % 64)) != 0
    }

    fn set_used(&mut self, i: usize, used: bool) {
        let mask = 1 << (i % 64);
        let word = &mut self.bits[i / 64];
        let was = *word & mask != 0;
        if used {
            *word |= mask;
            if !was {
                self.free -= 1;
            }
        } else {
            *word &= !mask;
            if was {
                self.free += 1;
            }
        }
    }

    /// Mark `[addr, addr + len)` free. Clipped to the tracked range.
    pub fn mark_free_range(&mut self, addr: u64, len: u64) {
        let first = self.index_of(addr);
        let n = (len / FRAME_SIZE) as usize;
        for i in first..(first + n).min(self.nframes) {
            self.set_used(i, false);
        }
    }

    /// Mark `[addr, addr + len)` used (len rounded up to whole frames).
    pub fn mark_used_range(&mut self, addr: u64, len: u64) {
        let first = self.index_of(addr);
        let n = (len.div_ceil(FRAME_SIZE)) as usize;
        for i in first..(first + n).min(self.nframes) {
            self.set_used(i, true);
        }
    }

    pub fn alloc(&mut self) -> Option<u64> {
        if self.free == 0 {
            return None;
        }
        let start = self.cursor;
        let mut i = start;
        loop {
            if !self.is_used(i) {
                self.set_used(i, true);
                self.cursor = (i + 1) % self.nframes;
                return Some(self.addr_of(i));
            }
            i = (i + 1) % self.nframes;
            if i == start {
                return None;
            }
        }
    }

    /// Allocate `n` physically contiguous frames whose base is aligned to
    /// `align_frames` frames. First-fit from the bottom.
    pub fn alloc_contig(&mut self, n: usize, align_frames: usize) -> Option<u64> {
        assert!(n > 0 && align_frames.is_power_of_two());
        if self.free < n {
            return None;
        }
        let mut i = 0;
        'outer: while i + n <= self.nframes {
            // Alignment is relative to physical addresses; base is frame-aligned.
            let misalign = (self.base / FRAME_SIZE) as usize + i;
            if misalign % align_frames != 0 {
                i += align_frames - misalign % align_frames;
                continue;
            }
            for j in 0..n {
                if self.is_used(i + j) {
                    i = i + j + 1;
                    continue 'outer;
                }
            }
            for j in 0..n {
                self.set_used(i + j, true);
            }
            return Some(self.addr_of(i));
        }
        None
    }

    pub fn free_frames(&mut self, addr: u64, n: usize) {
        let first = self.index_of(addr);
        for i in first..first + n {
            assert!(self.is_used(i), "double free of frame {:#x}", self.addr_of(i));
            self.set_used(i, false);
        }
    }

    pub fn free_count(&self) -> usize {
        self.free
    }

    pub fn total(&self) -> usize {
        self.nframes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bitmap(storage: &mut Vec<u64>, nframes: usize) -> FrameBitmap<'_> {
        storage.resize(FrameBitmap::storage_words(nframes), 0);
        FrameBitmap::new(0x10_0000, nframes, storage)
    }

    #[test]
    fn alloc_free_roundtrip() {
        let mut s = Vec::new();
        let mut b = bitmap(&mut s, 128);
        b.mark_free_range(0x10_0000, 128 * FRAME_SIZE);
        assert_eq!(b.free_count(), 128);

        let a = b.alloc().unwrap();
        let c = b.alloc().unwrap();
        assert_ne!(a, c);
        assert_eq!(a % FRAME_SIZE, 0);
        assert_eq!(b.free_count(), 126);

        b.free_frames(a, 1);
        b.free_frames(c, 1);
        assert_eq!(b.free_count(), 128);
    }

    #[test]
    fn exhaustion_returns_none() {
        let mut s = Vec::new();
        let mut b = bitmap(&mut s, 4);
        b.mark_free_range(0x10_0000, 4 * FRAME_SIZE);
        for _ in 0..4 {
            assert!(b.alloc().is_some());
        }
        assert!(b.alloc().is_none());
    }

    #[test]
    fn contig_is_aligned_and_contiguous() {
        let mut s = Vec::new();
        let mut b = bitmap(&mut s, 256);
        b.mark_free_range(0x10_0000, 256 * FRAME_SIZE);
        // Poke holes so first-fit has to skip.
        let x = b.alloc().unwrap();
        let _ = b.alloc().unwrap();
        b.free_frames(x, 1);

        let addr = b.alloc_contig(16, 16).unwrap();
        assert_eq!(addr % (16 * FRAME_SIZE), 0);
        // All 16 frames must have been marked used: freeing them must not panic.
        b.free_frames(addr, 16);
    }

    #[test]
    fn partial_free_ranges() {
        let mut s = Vec::new();
        let mut b = bitmap(&mut s, 64);
        b.mark_free_range(0x10_0000, 64 * FRAME_SIZE);
        b.mark_used_range(0x10_0000 + 8 * FRAME_SIZE, 8 * FRAME_SIZE);
        assert_eq!(b.free_count(), 56);
        // Contig run of 16 can't use the used hole.
        let addr = b.alloc_contig(16, 1).unwrap();
        assert_eq!(addr, 0x10_0000 + 16 * FRAME_SIZE);
    }
}
