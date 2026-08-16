//! Tagged term values and per-process bump heaps.
//!
//! A `Term` is one 64-bit word. Low 3 bits are the tag:
//!
//! | tag | meaning        | payload (word >> 3)          |
//! |-----|----------------|------------------------------|
//! | 000 | boxed pointer  | 8-aligned address (whole word)|
//! | 001 | small integer  | i61, sign-extended           |
//! | 010 | atom           | atom-table index             |
//! | 011 | pid            | process id                   |
//! | 100 | ref            | unique 61-bit counter        |
//! | 101 | special        | 0 = nil (empty list)         |
//! | 110 | port           | port id                      |
//!
//! Boxed objects live in a process's `Heap` and start with a header word:
//! low 3 bits = kind, rest = arity/byte-length.
//!
//! Heaps are bump arenas over caller-provided memory (a physically contiguous
//! span in the kernel, a `Vec` in host tests). There is no reclamation yet:
//! exhausting the heap is a quota breach and kills the process (per-process
//! semispace GC is a later milestone — `copy_term` is already its core).
//!
//! Deliberate BEAM divergences: contiguous immutable arrays are planned as a
//! first-class boxed kind, and large binaries will move to a shared refcounted
//! region (M4+ buffer handles are the first user).

#![cfg_attr(not(test), no_std)]

extern crate alloc;

use alloc::vec::Vec;
use core::fmt;

pub const TAG_MASK: u64 = 0b111;
pub const TAG_BOXED: u64 = 0;
pub const TAG_INT: u64 = 1;
pub const TAG_ATOM: u64 = 2;
pub const TAG_PID: u64 = 3;
pub const TAG_REF: u64 = 4;
pub const TAG_SPECIAL: u64 = 5;
pub const TAG_PORT: u64 = 6;

const KIND_TUPLE: u64 = 0;
const KIND_CONS: u64 = 1;
const KIND_BINARY: u64 = 2;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct Term(pub u64);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Int,
    Atom,
    Pid,
    Ref,
    Nil,
    Port,
    Tuple,
    Cons,
    Binary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeapFull;

impl Term {
    pub const NIL: Term = Term(TAG_SPECIAL);

    pub fn int(v: i64) -> Term {
        debug_assert!((-(1 << 60)..(1 << 60)).contains(&v), "int out of i61 range");
        Term(((v as u64) << 3) | TAG_INT)
    }
    pub fn atom(idx: u32) -> Term {
        Term(((idx as u64) << 3) | TAG_ATOM)
    }
    pub fn pid(p: u64) -> Term {
        Term((p << 3) | TAG_PID)
    }
    pub fn reference(r: u64) -> Term {
        Term((r << 3) | TAG_REF)
    }
    pub fn port(p: u64) -> Term {
        Term((p << 3) | TAG_PORT)
    }

    pub fn tag(self) -> u64 {
        self.0 & TAG_MASK
    }

    pub fn as_int(self) -> Option<i64> {
        (self.tag() == TAG_INT).then(|| (self.0 as i64) >> 3)
    }
    pub fn as_atom(self) -> Option<u32> {
        (self.tag() == TAG_ATOM).then(|| (self.0 >> 3) as u32)
    }
    pub fn as_pid(self) -> Option<u64> {
        (self.tag() == TAG_PID).then(|| self.0 >> 3)
    }
    pub fn as_ref(self) -> Option<u64> {
        (self.tag() == TAG_REF).then(|| self.0 >> 3)
    }
    pub fn as_port(self) -> Option<u64> {
        (self.tag() == TAG_PORT).then(|| self.0 >> 3)
    }
    pub fn is_nil(self) -> bool {
        self == Term::NIL
    }
    pub fn is_boxed(self) -> bool {
        self.tag() == TAG_BOXED
    }

    fn ptr(self) -> *const u64 {
        debug_assert!(self.is_boxed());
        self.0 as *const u64
    }

    /// # Safety
    /// Boxed terms must point into a live heap.
    pub unsafe fn kind(self) -> Kind {
        match self.tag() {
            TAG_INT => Kind::Int,
            TAG_ATOM => Kind::Atom,
            TAG_PID => Kind::Pid,
            TAG_REF => Kind::Ref,
            TAG_SPECIAL => Kind::Nil,
            TAG_PORT => Kind::Port,
            TAG_BOXED => {
                let header = unsafe { *self.ptr() };
                match header & TAG_MASK {
                    KIND_TUPLE => Kind::Tuple,
                    KIND_CONS => Kind::Cons,
                    KIND_BINARY => Kind::Binary,
                    k => unreachable!("bad box kind {k}"),
                }
            }
            t => unreachable!("bad tag {t}"),
        }
    }

    /// # Safety: must be a tuple term into a live heap.
    pub unsafe fn tuple_arity(self) -> usize {
        unsafe { (*self.ptr() >> 3) as usize }
    }
    /// # Safety: must be a tuple term into a live heap; `i` < arity.
    pub unsafe fn tuple_elem(self, i: usize) -> Term {
        debug_assert!(i < unsafe { self.tuple_arity() });
        Term(unsafe { *self.ptr().add(1 + i) })
    }
    /// # Safety: must be a cons term into a live heap.
    pub unsafe fn head(self) -> Term {
        Term(unsafe { *self.ptr().add(1) })
    }
    /// # Safety: must be a cons term into a live heap.
    pub unsafe fn tail(self) -> Term {
        Term(unsafe { *self.ptr().add(2) })
    }
    /// # Safety: must be a binary term into a live heap.
    pub unsafe fn bin_bytes<'a>(self) -> &'a [u8] {
        unsafe {
            let len = (*self.ptr() >> 3) as usize;
            core::slice::from_raw_parts(self.ptr().add(1).cast::<u8>(), len)
        }
    }
}

/// Bump arena over caller-provided memory. Word-granular.
pub struct Heap {
    base: *mut u64,
    cap_words: usize,
    top: usize,
}

// Terms are only touched by the owning process (or under the kernel's process
// table lock during copy-on-send).
unsafe impl Send for Heap {}

impl Heap {
    /// # Safety
    /// `base..base+bytes` must be writable, 8-aligned, and outlive the heap.
    pub unsafe fn new(base: *mut u8, bytes: usize) -> Heap {
        debug_assert_eq!(base as usize % 8, 0);
        Heap { base: base.cast(), cap_words: bytes / 8, top: 0 }
    }

    pub fn used_bytes(&self) -> usize {
        self.top * 8
    }
    pub fn capacity_bytes(&self) -> usize {
        self.cap_words * 8
    }

    fn alloc(&mut self, words: usize) -> Result<*mut u64, HeapFull> {
        if self.top + words > self.cap_words {
            return Err(HeapFull);
        }
        let p = unsafe { self.base.add(self.top) };
        self.top += words;
        Ok(p)
    }

    pub fn tuple(&mut self, elems: &[Term]) -> Result<Term, HeapFull> {
        let p = self.alloc(1 + elems.len())?;
        unsafe {
            p.write(((elems.len() as u64) << 3) | KIND_TUPLE);
            for (i, e) in elems.iter().enumerate() {
                p.add(1 + i).write(e.0);
            }
        }
        Ok(Term(p as u64))
    }

    pub fn cons(&mut self, head: Term, tail: Term) -> Result<Term, HeapFull> {
        let p = self.alloc(3)?;
        unsafe {
            p.write(KIND_CONS);
            p.add(1).write(head.0);
            p.add(2).write(tail.0);
        }
        Ok(Term(p as u64))
    }

    pub fn binary(&mut self, data: &[u8]) -> Result<Term, HeapFull> {
        let words = 1 + data.len().div_ceil(8);
        let p = self.alloc(words)?;
        unsafe {
            p.write(((data.len() as u64) << 3) | KIND_BINARY);
            core::ptr::copy_nonoverlapping(data.as_ptr(), p.add(1).cast::<u8>(), data.len());
        }
        Ok(Term(p as u64))
    }

    /// Build a list from a slice (right to left).
    pub fn list(&mut self, elems: &[Term]) -> Result<Term, HeapFull> {
        let mut t = Term::NIL;
        for e in elems.iter().rev() {
            t = self.cons(*e, t)?;
        }
        Ok(t)
    }
}

/// Deep-copy `src` (whose boxed parts live in some other live heap) into `dst`.
/// This is the copy in copy-on-send, and the seed of the future copying GC.
///
/// List spines are walked iteratively (heads recurse), so deep lists don't
/// blow the native stack; deeply *nested* non-list structures still recurse.
///
/// # Safety
/// All boxed parts of `src` must point into live heap memory.
pub unsafe fn copy_term(src: Term, dst: &mut Heap) -> Result<Term, HeapFull> {
    unsafe {
        match src.kind() {
            Kind::Int | Kind::Atom | Kind::Pid | Kind::Ref | Kind::Nil | Kind::Port => Ok(src),
            Kind::Tuple => {
                let n = src.tuple_arity();
                let p = dst.alloc(1 + n)?;
                p.write(((n as u64) << 3) | KIND_TUPLE);
                for i in 0..n {
                    let c = copy_term(src.tuple_elem(i), dst)?;
                    p.add(1 + i).write(c.0);
                }
                Ok(Term(p as u64))
            }
            Kind::Binary => dst.binary(src.bin_bytes()),
            Kind::Cons => {
                // Iterate the spine; patch each new cell's tail as we go.
                let mut spine: Vec<Term> = Vec::new();
                let mut cur = src;
                while cur.kind() == Kind::Cons {
                    spine.push(cur);
                    cur = cur.tail();
                }
                let mut tail = copy_term(cur, dst)?; // improper tail or NIL
                for cell in spine.into_iter().rev() {
                    let h = copy_term(cell.head(), dst)?;
                    tail = dst.cons(h, tail)?;
                }
                Ok(tail)
            }
        }
    }
}

/// Structural equality.
///
/// # Safety
/// All boxed parts of both terms must point into live heap memory.
pub unsafe fn eq(a: Term, b: Term) -> bool {
    unsafe {
        if a.0 == b.0 {
            return true;
        }
        if !(a.is_boxed() && b.is_boxed()) {
            return false;
        }
        match (a.kind(), b.kind()) {
            (Kind::Tuple, Kind::Tuple) => {
                let n = a.tuple_arity();
                n == b.tuple_arity() && (0..n).all(|i| eq(a.tuple_elem(i), b.tuple_elem(i)))
            }
            (Kind::Binary, Kind::Binary) => a.bin_bytes() == b.bin_bytes(),
            (Kind::Cons, Kind::Cons) => {
                let (mut x, mut y) = (a, b);
                loop {
                    if !eq(x.head(), y.head()) {
                        return false;
                    }
                    x = x.tail();
                    y = y.tail();
                    match (x.kind() == Kind::Cons, y.kind() == Kind::Cons) {
                        (true, true) => continue,
                        (false, false) => return eq(x, y),
                        _ => return false,
                    }
                }
            }
            _ => false,
        }
    }
}

/// Render a term; atoms resolve through `atom_name`.
///
/// # Safety
/// All boxed parts must point into live heap memory.
pub unsafe fn fmt_term(
    t: Term,
    out: &mut dyn fmt::Write,
    atom_name: &dyn Fn(u32) -> &'static str,
) -> fmt::Result {
    unsafe {
        match t.kind() {
            Kind::Int => write!(out, "{}", t.as_int().unwrap()),
            Kind::Atom => write!(out, "{}", atom_name(t.as_atom().unwrap())),
            Kind::Pid => write!(out, "<pid:{}>", t.as_pid().unwrap()),
            Kind::Ref => write!(out, "#ref<{}>", t.as_ref().unwrap()),
            Kind::Port => write!(out, "#port<{}>", t.as_port().unwrap()),
            Kind::Nil => write!(out, "[]"),
            Kind::Binary => write!(out, "<<{} bytes>>", t.bin_bytes().len()),
            Kind::Tuple => {
                write!(out, "{{")?;
                for i in 0..t.tuple_arity() {
                    if i > 0 {
                        write!(out, ", ")?;
                    }
                    fmt_term(t.tuple_elem(i), out, atom_name)?;
                }
                write!(out, "}}")
            }
            Kind::Cons => {
                write!(out, "[")?;
                let mut cur = t;
                let mut first = true;
                while cur.kind() == Kind::Cons {
                    if !first {
                        write!(out, ", ")?;
                    }
                    first = false;
                    fmt_term(cur.head(), out, atom_name)?;
                    cur = cur.tail();
                }
                if !cur.is_nil() {
                    write!(out, " | ")?;
                    fmt_term(cur, out, atom_name)?;
                }
                write!(out, "]")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn heap(buf: &mut Vec<u64>) -> Heap {
        buf.resize(4096, 0);
        unsafe { Heap::new(buf.as_mut_ptr().cast(), buf.len() * 8) }
    }

    #[test]
    fn immediates_roundtrip() {
        assert_eq!(Term::int(-42).as_int(), Some(-42));
        assert_eq!(Term::int(1 << 59).as_int(), Some(1 << 59));
        assert_eq!(Term::atom(7).as_atom(), Some(7));
        assert_eq!(Term::pid(123).as_pid(), Some(123));
        assert!(Term::NIL.is_nil());
        assert_ne!(Term::int(7).0, Term::atom(7).0);
    }

    #[test]
    fn build_and_read() {
        let mut buf = Vec::new();
        let mut h = heap(&mut buf);
        let t = h.tuple(&[Term::int(1), Term::atom(2)]).unwrap();
        let l = h.list(&[t, Term::int(9)]).unwrap();
        let b = h.binary(b"hello").unwrap();
        unsafe {
            assert_eq!(t.tuple_arity(), 2);
            assert_eq!(t.tuple_elem(0).as_int(), Some(1));
            assert_eq!(l.head().0, t.0);
            assert_eq!(l.tail().head().as_int(), Some(9));
            assert!(l.tail().tail().is_nil());
            assert_eq!(b.bin_bytes(), b"hello");
        }
    }

    #[test]
    fn copy_between_heaps_and_eq() {
        let (mut b1, mut b2) = (Vec::new(), Vec::new());
        let mut src = heap(&mut b1);
        let mut dst = heap(&mut b2);

        let inner = src.tuple(&[Term::int(5), Term::atom(1)]).unwrap();
        let list = src.list(&[inner, Term::int(7), Term::pid(3)]).unwrap();
        let bin = src.binary(b"xyz").unwrap();
        let msg = src.tuple(&[Term::atom(0), list, bin]).unwrap();

        let copied = unsafe { copy_term(msg, &mut dst) }.unwrap();
        assert_ne!(copied.0, msg.0);
        unsafe {
            assert!(eq(copied, msg));
            // Mutating nothing — but confirm deep structure landed in dst.
            assert_eq!(copied.tuple_elem(1).head().tuple_elem(0).as_int(), Some(5));
        }
    }

    #[test]
    fn deep_list_copy_no_stack_overflow() {
        let mut b1 = vec![0u64; 400_000];
        let mut b2 = vec![0u64; 400_000];
        let mut src = unsafe { Heap::new(b1.as_mut_ptr().cast(), b1.len() * 8) };
        let mut dst = unsafe { Heap::new(b2.as_mut_ptr().cast(), b2.len() * 8) };
        let mut l = Term::NIL;
        for i in 0..100_000 {
            l = src.cons(Term::int(i), l).unwrap();
        }
        let c = unsafe { copy_term(l, &mut dst) }.unwrap();
        unsafe {
            assert!(eq(c, l));
        }
    }

    #[test]
    fn heap_full_is_reported() {
        let mut buf = vec![0u64; 4];
        let mut h = unsafe { Heap::new(buf.as_mut_ptr().cast(), 32) };
        assert!(h.tuple(&[Term::int(1)]).is_ok()); // 2 words
        assert_eq!(h.tuple(&[Term::int(1), Term::int(2)]), Err(HeapFull)); // needs 3
    }
}
