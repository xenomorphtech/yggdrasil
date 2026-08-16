//! SPSC rings and the port queue-entry formats.
//!
//! Every port is a submission ring + completion ring pair — the shape virtio
//! queues, NVMe queues and io_uring all converged on. The same ring type also
//! serves driver-internal buffers (e.g. the serial rx ring filled from IRQ
//! context and drained by the scheduler bottom-half).
//!
//! Single producer, single consumer, lock-free, power-of-two capacity.

#![cfg_attr(not(test), no_std)]

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicU32, Ordering};

/// Submission queue entry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Sqe {
    pub op: u32,
    /// Completion correlation tag. `tag == SKIP_CQE` suppresses the CQE
    /// (io_uring's "skip success" — e.g. fire-and-forget writes).
    pub tag: i64,
    pub arg0: u64,
    pub arg1: u64,
}

pub const SKIP_CQE: i64 = -1;

/// Completion queue entry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Cqe {
    pub tag: i64,
    pub result: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Full;

/// `Default` usable in const context (for `Ring::new_zeroed` statics).
pub trait ConstDefault: Copy {
    const DEFAULT: Self;
}

impl ConstDefault for u8 {
    const DEFAULT: Self = 0;
}
impl ConstDefault for Sqe {
    const DEFAULT: Self = Sqe { op: 0, tag: 0, arg0: 0, arg1: 0 };
}
impl ConstDefault for Cqe {
    const DEFAULT: Self = Cqe { tag: 0, result: 0 };
}

/// Lock-free SPSC ring. `N` must be a power of two.
pub struct Ring<T: Copy + Default, const N: usize> {
    entries: UnsafeCell<[T; N]>,
    /// Consumer cursor (only the consumer writes it).
    head: AtomicU32,
    /// Producer cursor (only the producer writes it).
    tail: AtomicU32,
}

// Safety: SPSC discipline — one producer thread/context, one consumer.
unsafe impl<T: Copy + Default + Send, const N: usize> Sync for Ring<T, N> {}
unsafe impl<T: Copy + Default + Send, const N: usize> Send for Ring<T, N> {}

impl<T: Copy + Default, const N: usize> Default for Ring<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy + Default, const N: usize> Ring<T, N> {
    pub fn new() -> Self {
        const { assert!(N.is_power_of_two()) };
        Ring {
            entries: UnsafeCell::new([T::default(); N]),
            head: AtomicU32::new(0),
            tail: AtomicU32::new(0),
        }
    }

    /// Const constructor for statics (e.g. IRQ-fed rings).
    pub const fn new_zeroed() -> Self
    where
        T: ConstDefault,
    {
        const { assert!(N.is_power_of_two()) };
        Ring {
            entries: UnsafeCell::new([T::DEFAULT; N]),
            head: AtomicU32::new(0),
            tail: AtomicU32::new(0),
        }
    }

    pub fn len(&self) -> usize {
        self.tail.load(Ordering::Acquire).wrapping_sub(self.head.load(Ordering::Acquire)) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_full(&self) -> bool {
        self.len() == N
    }

    /// Producer side.
    pub fn push(&self, v: T) -> Result<(), Full> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);
        if tail.wrapping_sub(head) as usize == N {
            return Err(Full);
        }
        unsafe {
            (*self.entries.get())[tail as usize % N] = v;
        }
        self.tail.store(tail.wrapping_add(1), Ordering::Release);
        Ok(())
    }

    /// Consumer side.
    pub fn pop(&self) -> Option<T> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        if head == tail {
            return None;
        }
        let v = unsafe { (*self.entries.get())[head as usize % N] };
        self.head.store(head.wrapping_add(1), Ordering::Release);
        Some(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fifo_order_and_wraparound() {
        let r: Ring<u32, 8> = Ring::new();
        for round in 0..100u32 {
            for i in 0..5 {
                r.push(round * 10 + i).unwrap();
            }
            for i in 0..5 {
                assert_eq!(r.pop(), Some(round * 10 + i));
            }
        }
        assert!(r.is_empty());
    }

    #[test]
    fn full_and_empty() {
        let r: Ring<u8, 4> = Ring::new();
        assert_eq!(r.pop(), None);
        for i in 0..4 {
            r.push(i).unwrap();
        }
        assert!(r.is_full());
        assert_eq!(r.push(9), Err(Full));
        assert_eq!(r.pop(), Some(0));
        r.push(9).unwrap();
        assert_eq!(r.len(), 4);
    }

    #[test]
    fn cross_thread_spsc() {
        use std::sync::Arc;
        let r: Arc<Ring<u64, 64>> = Arc::new(Ring::new());
        let p = r.clone();
        let producer = std::thread::spawn(move || {
            for i in 0..100_000u64 {
                while p.push(i).is_err() {
                    std::hint::spin_loop();
                }
            }
        });
        let mut expect = 0u64;
        while expect < 100_000 {
            if let Some(v) = r.pop() {
                assert_eq!(v, expect);
                expect += 1;
            }
        }
        producer.join().unwrap();
    }
}
