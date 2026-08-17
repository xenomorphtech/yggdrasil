//! Tier-0 execution engine: the bytecode interpreter.
//!
//! Runs on the calling process's native stack (bytecode calls recurse
//! natively), so the scheduler treats interpreted and future-JIT'd processes
//! identically. All system effects go through `SystemApi`, which the kernel
//! implements over real processes and host tests implement as a mock.
//!
//! Safepoints: every backward jump and every call polls `api.safepoint()` —
//! exactly where the JIT will emit its preempt-flag checks.

#![cfg_attr(not(test), no_std)]

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use ygg_bytecode::{Module, op};
use ygg_term::{Heap, HeapFull, Term};

/// Reserved word (tag 0b111, never a valid term) returned by an engine when
/// the function ended in `TAIL_CALL_EXT`; the invoking trampoline picks the
/// stashed target up via `SystemApi::take` semantics and re-dispatches.
pub const TAIL_SENTINEL: Term = Term(7);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trap {
    /// Type error, bad index, arity mismatch, bad register…
    Badarg,
    /// Unknown opcode / truncated code / bad function index.
    BadCode,
    /// Heap quota exhausted.
    HeapFull,
    /// Process chose to exit with the given global atom as reason.
    Exit(u32),
}

impl From<HeapFull> for Trap {
    fn from(_: HeapFull) -> Trap {
        Trap::HeapFull
    }
}

pub trait SystemApi {
    fn heap(&mut self) -> &mut Heap;
    fn self_pid(&self) -> u64;
    /// `to` must be a pid term. The message is copied by the implementation.
    fn send(&mut self, to: Term, msg: Term) -> Result<(), Trap>;
    fn recv(&mut self) -> Term;
    /// Spawn `fn_idx` of the same module with one immediate argument.
    fn spawn(&mut self, fn_idx: u32, arg: Term) -> Result<u64, Trap>;
    fn safepoint(&mut self);
    /// Map a module-local atom index to a global atom index.
    fn atom_global(&mut self, local: u32) -> u32;
    fn print(&mut self, t: Term);
    /// Open a port of `kind`; returns a port term.
    fn port_open(&mut self, kind: u8) -> Result<Term, Trap>;
    /// Submit to a port's SQ. `arg0`/`tag` must be ints; tag -1 skips the CQE.
    fn port_submit(&mut self, port: Term, op: u8, arg0: Term, tag: Term) -> Result<(), Trap>;
    /// Fully-qualified call: resolve `module:fname/args.len()` in the *current*
    /// module table and run it. Atoms are global indices.
    fn call_ext(&mut self, module_atom: u32, fname_atom: u32, args: &[Term]) -> Result<Term, Trap>;
    /// Consume kernel packet buffer `id`, producing a binary term.
    fn buf_to_bin(&mut self, id: i64) -> Result<Term, Trap>;
    /// Create a kernel packet buffer from a binary; returns its id as an int term.
    fn bin_to_buf(&mut self, bin: Term) -> Result<Term, Trap>;
    /// Try to grow the process heap (segmented, non-moving). False at quota.
    fn heap_grow(&mut self) -> bool {
        false
    }
    /// Stash a tail-call target (global atoms); the engine then returns
    /// `TAIL_SENTINEL` and the trampoline re-dispatches.
    fn tail_call(&mut self, _module_atom: u32, _fname_atom: u32, _args: &[Term]) {}
    /// Stash a *local* tail-call target (function index in the module instance
    /// currently running); same sentinel/trampoline contract as `tail_call`.
    fn tail_call_local(&mut self, _fn_idx: u32, _args: &[Term]) {}
    /// Overwrite bytes of an existing kernel buffer at an offset. Result 0.
    fn buf_write(&mut self, _buf: Term, _off: Term, _bin: Term) -> Result<Term, Trap> {
        Err(Trap::Badarg)
    }
    /// Allocate a fixed-size zero-filled kernel blob; result = its id.
    fn buf_new(&mut self, _size: Term) -> Result<Term, Trap> {
        Err(Trap::Badarg)
    }
    /// Copy a slice of a kernel blob out as a fresh binary term.
    fn buf_read(&mut self, _buf: Term, _off: Term, _len: Term) -> Result<Term, Trap> {
        Err(Trap::Badarg)
    }
    /// Park the process for `ms` without consuming a mailbox message.
    fn sleep_ms(&mut self, _ms: u64) {}
    /// Milliseconds since timer start (1 kHz). Host stubs may return 0.
    fn ticks(&self) -> Term {
        Term::int(0)
    }
    /// All-register submit (`PORT_SUBMIT2`): dynamic op/tag plus `arg1`.
    fn port_submit2(
        &mut self,
        _port: Term,
        _op: Term,
        _arg0: Term,
        _arg1: Term,
        _tag: Term,
    ) -> Result<(), Trap> {
        Err(Trap::Badarg)
    }
}

/// Allocate with grow-on-full retry; `Trap::HeapFull` only at the quota cap.
fn alloc_term(
    api: &mut dyn SystemApi,
    f: impl Fn(&mut Heap) -> Result<Term, HeapFull>,
) -> Result<Term, Trap> {
    loop {
        match f(api.heap()) {
            Ok(t) => return Ok(t),
            Err(HeapFull) => {
                if !api.heap_grow() {
                    return Err(Trap::HeapFull);
                }
            }
        }
    }
}

struct Frame<'m> {
    code: &'m [u8],
    pc: usize,
    regs: Vec<Term>,
}

impl<'m> Frame<'m> {
    fn u8(&mut self) -> Result<u8, Trap> {
        let v = *self.code.get(self.pc).ok_or(Trap::BadCode)?;
        self.pc += 1;
        Ok(v)
    }
    fn u32(&mut self) -> Result<u32, Trap> {
        let s = self.code.get(self.pc..self.pc + 4).ok_or(Trap::BadCode)?;
        self.pc += 4;
        Ok(u32::from_le_bytes(s.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32, Trap> {
        Ok(self.u32()? as i32)
    }
    fn i64(&mut self) -> Result<i64, Trap> {
        let s = self.code.get(self.pc..self.pc + 8).ok_or(Trap::BadCode)?;
        self.pc += 8;
        Ok(i64::from_le_bytes(s.try_into().unwrap()))
    }
    fn bytes(&mut self, n: usize) -> Result<&'m [u8], Trap> {
        let s = self.code.get(self.pc..self.pc + n).ok_or(Trap::BadCode)?;
        self.pc += n;
        Ok(s)
    }
    fn get(&self, r: u8) -> Result<Term, Trap> {
        self.regs.get(r as usize).copied().ok_or(Trap::Badarg)
    }
    fn set(&mut self, r: u8, v: Term) -> Result<(), Trap> {
        *self.regs.get_mut(r as usize).ok_or(Trap::Badarg)? = v;
        Ok(())
    }
}

/// Execute `fn_idx` with `args`; returns the function's result.
pub fn run_function(
    m: &Module,
    fn_idx: usize,
    args: &[Term],
    api: &mut dyn SystemApi,
) -> Result<Term, Trap> {
    let f = m.functions.get(fn_idx).ok_or(Trap::BadCode)?;
    if args.len() != f.arity as usize || (f.nregs as usize) < args.len() {
        return Err(Trap::Badarg);
    }
    let mut fr = Frame {
        code: &f.code,
        pc: 0,
        regs: vec![Term::NIL; f.nregs as usize],
    };
    fr.regs[..args.len()].copy_from_slice(args);

    loop {
        let opcode = fr.u8()?;
        match opcode {
            op::NOP => {}
            op::LOAD_INT => {
                let rd = fr.u8()?;
                let v = fr.i64()?;
                fr.set(rd, Term::int(v))?;
            }
            op::LOAD_ATOM => {
                let rd = fr.u8()?;
                let local = fr.u32()?;
                let g = api.atom_global(local);
                fr.set(rd, Term::atom(g))?;
            }
            op::LOAD_NIL => {
                let rd = fr.u8()?;
                fr.set(rd, Term::NIL)?;
            }
            op::MOVE => {
                let rd = fr.u8()?;
                let rs = fr.u8()?;
                let v = fr.get(rs)?;
                fr.set(rd, v)?;
            }
            op::SELF_PID => {
                let rd = fr.u8()?;
                let p = api.self_pid();
                fr.set(rd, Term::pid(p))?;
            }
            op::TICKS => {
                let rd = fr.u8()?;
                fr.set(rd, api.ticks())?;
            }
            op::MAKE_TUPLE => {
                let rd = fr.u8()?;
                let n = fr.u8()? as usize;
                let mut elems = Vec::with_capacity(n);
                for _ in 0..n {
                    let r = fr.u8()?;
                    elems.push(fr.get(r)?);
                }
                let t = alloc_term(api, |h| h.tuple(&elems))?;
                fr.set(rd, t)?;
            }
            op::GET_ELEM => {
                let rd = fr.u8()?;
                let rs = fr.u8()?;
                let idx = fr.u8()? as usize;
                let t = fr.get(rs)?;
                let v = unsafe {
                    if t.kind() != ygg_term::Kind::Tuple || idx >= t.tuple_arity() {
                        return Err(Trap::Badarg);
                    }
                    t.tuple_elem(idx)
                };
                fr.set(rd, v)?;
            }
            op::CONS => {
                let rd = fr.u8()?;
                let rh = fr.u8()?;
                let rt = fr.u8()?;
                let (h, t) = (fr.get(rh)?, fr.get(rt)?);
                let c = alloc_term(api, |heap| heap.cons(h, t))?;
                fr.set(rd, c)?;
            }
            op::HEAD | op::TAIL => {
                let rd = fr.u8()?;
                let rs = fr.u8()?;
                let t = fr.get(rs)?;
                let v = unsafe {
                    if t.kind() != ygg_term::Kind::Cons {
                        return Err(Trap::Badarg);
                    }
                    if opcode == op::HEAD {
                        t.head()
                    } else {
                        t.tail()
                    }
                };
                fr.set(rd, v)?;
            }
            op::ADD | op::SUB | op::MUL => {
                let rd = fr.u8()?;
                let ra = fr.u8()?;
                let rb = fr.u8()?;
                let a = fr.get(ra)?.as_int().ok_or(Trap::Badarg)?;
                let b = fr.get(rb)?.as_int().ok_or(Trap::Badarg)?;
                let v = match opcode {
                    op::ADD => a.checked_add(b),
                    op::SUB => a.checked_sub(b),
                    _ => a.checked_mul(b),
                }
                .ok_or(Trap::Badarg)?;
                fr.set(rd, Term::int(v))?;
            }
            op::CMP_EQ => {
                let rd = fr.u8()?;
                let ra = fr.u8()?;
                let rb = fr.u8()?;
                let e = unsafe { ygg_term::eq(fr.get(ra)?, fr.get(rb)?) };
                fr.set(rd, Term::int(e as i64))?;
            }
            op::CMP_LT => {
                let rd = fr.u8()?;
                let ra = fr.u8()?;
                let rb = fr.u8()?;
                let a = fr.get(ra)?.as_int().ok_or(Trap::Badarg)?;
                let b = fr.get(rb)?.as_int().ok_or(Trap::Badarg)?;
                fr.set(rd, Term::int((a < b) as i64))?;
            }
            op::JMP => {
                let off = fr.i32()?;
                jump(&mut fr, off, api)?;
            }
            op::JMP_IF => {
                let rc = fr.u8()?;
                let off = fr.i32()?;
                let taken = fr.get(rc)?.as_int().ok_or(Trap::Badarg)? != 0;
                if taken {
                    jump(&mut fr, off, api)?;
                }
            }
            op::CALL => {
                let rd = fr.u8()?;
                let callee = fr.u32()? as usize;
                let n = fr.u8()? as usize;
                let mut args = Vec::with_capacity(n);
                for _ in 0..n {
                    let r = fr.u8()?;
                    args.push(fr.get(r)?);
                }
                api.safepoint();
                let v = run_function(m, callee, &args, api)?;
                if v == TAIL_SENTINEL {
                    // Callee tail-called out: unwind to the trampoline.
                    return Ok(TAIL_SENTINEL);
                }
                fr.set(rd, v)?;
            }
            op::RET => {
                let rs = fr.u8()?;
                return fr.get(rs);
            }
            op::SPAWN => {
                let rd = fr.u8()?;
                let callee = fr.u32()?;
                let ra = fr.u8()?;
                let arg = fr.get(ra)?;
                if arg.is_boxed() {
                    // Spawn args are immediates only (pids, ints, atoms) until
                    // cross-heap spawn-copy lands.
                    return Err(Trap::Badarg);
                }
                let pid = api.spawn(callee, arg)?;
                fr.set(rd, Term::pid(pid))?;
            }
            op::SEND => {
                let rt = fr.u8()?;
                let rm = fr.u8()?;
                let (to, msg) = (fr.get(rt)?, fr.get(rm)?);
                api.send(to, msg)?;
            }
            op::RECV => {
                let rd = fr.u8()?;
                let v = api.recv();
                fr.set(rd, v)?;
            }
            op::PRINT => {
                let rs = fr.u8()?;
                let v = fr.get(rs)?;
                api.print(v);
            }
            op::EXIT_ATOM => {
                let rs = fr.u8()?;
                let a = fr.get(rs)?.as_atom().ok_or(Trap::Badarg)?;
                return Err(Trap::Exit(a));
            }
            op::PORT_OPEN => {
                let rd = fr.u8()?;
                let kind = fr.u8()?;
                let p = api.port_open(kind)?;
                fr.set(rd, p)?;
            }
            op::PORT_SUBMIT => {
                let rp = fr.u8()?;
                let o = fr.u8()?;
                let ra = fr.u8()?;
                let rt = fr.u8()?;
                let (port, arg0, tag) = (fr.get(rp)?, fr.get(ra)?, fr.get(rt)?);
                api.port_submit(port, o, arg0, tag)?;
            }
            op::BUF_WRITE => {
                let rd = fr.u8()?;
                let rb = fr.u8()?;
                let ro = fr.u8()?;
                let rs = fr.u8()?;
                let (buf, off, bin) = (fr.get(rb)?, fr.get(ro)?, fr.get(rs)?);
                let v = api.buf_write(buf, off, bin)?;
                fr.set(rd, v)?;
            }
            op::SLEEP_MS => {
                let rm = fr.u8()?;
                let ms = fr.get(rm)?.as_int().ok_or(Trap::Badarg)?;
                if ms < 0 {
                    return Err(Trap::Badarg);
                }
                api.safepoint();
                api.sleep_ms(ms as u64);
            }
            op::BUF_NEW => {
                let rd = fr.u8()?;
                let rs = fr.u8()?;
                let size = fr.get(rs)?;
                let v = api.buf_new(size)?;
                fr.set(rd, v)?;
            }
            op::BUF_READ => {
                let rd = fr.u8()?;
                let rb = fr.u8()?;
                let ro = fr.u8()?;
                let rl = fr.u8()?;
                let (buf, off, len) = (fr.get(rb)?, fr.get(ro)?, fr.get(rl)?);
                let v = api.buf_read(buf, off, len)?;
                fr.set(rd, v)?;
            }
            op::PORT_SUBMIT2 => {
                let rp = fr.u8()?;
                let ro = fr.u8()?;
                let ra0 = fr.u8()?;
                let ra1 = fr.u8()?;
                let rt = fr.u8()?;
                let (port, o, a0, a1, tag) =
                    (fr.get(rp)?, fr.get(ro)?, fr.get(ra0)?, fr.get(ra1)?, fr.get(rt)?);
                api.port_submit2(port, o, a0, a1, tag)?;
            }
            op::TAIL_CALL => {
                let callee = fr.u32()?;
                let n = fr.u8()? as usize;
                let mut args = Vec::with_capacity(n);
                for _ in 0..n {
                    let r = fr.u8()?;
                    args.push(fr.get(r)?);
                }
                api.safepoint();
                api.tail_call_local(callee, &args);
                return Ok(TAIL_SENTINEL);
            }
            op::TAIL_CALL_EXT => {
                let mlocal = fr.u32()?;
                let flocal = fr.u32()?;
                let n = fr.u8()? as usize;
                let mut args = Vec::with_capacity(n);
                for _ in 0..n {
                    let r = fr.u8()?;
                    args.push(fr.get(r)?);
                }
                let mg = api.atom_global(mlocal);
                let fg = api.atom_global(flocal);
                api.safepoint();
                api.tail_call(mg, fg, &args);
                return Ok(TAIL_SENTINEL);
            }
            op::CALL_EXT => {
                let rd = fr.u8()?;
                let mlocal = fr.u32()?;
                let flocal = fr.u32()?;
                let n = fr.u8()? as usize;
                let mut args = Vec::with_capacity(n);
                for _ in 0..n {
                    let r = fr.u8()?;
                    args.push(fr.get(r)?);
                }
                let mg = api.atom_global(mlocal);
                let fg = api.atom_global(flocal);
                api.safepoint();
                let v = api.call_ext(mg, fg, &args)?;
                fr.set(rd, v)?;
            }
            op::BAND | op::BOR | op::BXOR => {
                let rd = fr.u8()?;
                let ra = fr.u8()?;
                let rb = fr.u8()?;
                let a = fr.get(ra)?.as_int().ok_or(Trap::Badarg)?;
                let b = fr.get(rb)?.as_int().ok_or(Trap::Badarg)?;
                let v = match opcode {
                    op::BAND => a & b,
                    op::BOR => a | b,
                    _ => a ^ b,
                };
                fr.set(rd, Term::int(v))?;
            }
            op::BSL | op::BSR => {
                let rd = fr.u8()?;
                let ra = fr.u8()?;
                let rb = fr.u8()?;
                let a = fr.get(ra)?.as_int().ok_or(Trap::Badarg)?;
                let b = fr.get(rb)?.as_int().ok_or(Trap::Badarg)?;
                if !(0..=60).contains(&b) {
                    return Err(Trap::Badarg);
                }
                let v = if opcode == op::BSL { a.wrapping_shl(b as u32) } else { a >> b };
                fr.set(rd, Term::int((v << 3) >> 3))?; // clamp into i61
            }
            op::BNOT => {
                let rd = fr.u8()?;
                let rs = fr.u8()?;
                let a = fr.get(rs)?.as_int().ok_or(Trap::Badarg)?;
                fr.set(rd, Term::int(!a))?;
            }
            op::BIN_NEW => {
                let rd = fr.u8()?;
                let len = fr.u32()? as usize;
                let bytes = fr.bytes(len)?;
                let b = alloc_term(api, |h| h.binary(bytes))?;
                fr.set(rd, b)?;
            }
            op::BIN_FROM_LIST => {
                let rd = fr.u8()?;
                let rs = fr.u8()?;
                let mut cur = fr.get(rs)?;
                let mut bytes: Vec<u8> = Vec::new();
                unsafe {
                    while cur.kind() == ygg_term::Kind::Cons {
                        let h = cur.head().as_int().ok_or(Trap::Badarg)?;
                        if !(0..=255).contains(&h) {
                            return Err(Trap::Badarg);
                        }
                        bytes.push(h as u8);
                        cur = cur.tail();
                    }
                }
                if !cur.is_nil() {
                    return Err(Trap::Badarg);
                }
                let b = alloc_term(api, |h| h.binary(&bytes))?;
                fr.set(rd, b)?;
            }
            op::BIN_TO_LIST => {
                let rd = fr.u8()?;
                let rs = fr.u8()?;
                let t = fr.get(rs)?;
                let elems: Vec<Term> = unsafe {
                    if t.kind() != ygg_term::Kind::Binary {
                        return Err(Trap::Badarg);
                    }
                    t.bin_bytes().iter().map(|&b| Term::int(b as i64)).collect()
                };
                let l = alloc_term(api, |h| h.list(&elems))?;
                fr.set(rd, l)?;
            }
            op::BIN_SIZE => {
                let rd = fr.u8()?;
                let rs = fr.u8()?;
                let t = fr.get(rs)?;
                let n = unsafe {
                    if t.kind() != ygg_term::Kind::Binary {
                        return Err(Trap::Badarg);
                    }
                    t.bin_bytes().len()
                };
                fr.set(rd, Term::int(n as i64))?;
            }
            op::BUF_TO_BIN => {
                let rd = fr.u8()?;
                let rs = fr.u8()?;
                let id = fr.get(rs)?.as_int().ok_or(Trap::Badarg)?;
                let b = api.buf_to_bin(id)?;
                fr.set(rd, b)?;
            }
            op::BIN_TO_BUF => {
                let rd = fr.u8()?;
                let rs = fr.u8()?;
                let b = fr.get(rs)?;
                let id = api.bin_to_buf(b)?;
                fr.set(rd, id)?;
            }
            op::IS_BINARY => {
                let rd = fr.u8()?;
                let rs = fr.u8()?;
                let t = fr.get(rs)?;
                let is = unsafe { t.is_boxed() && t.kind() == ygg_term::Kind::Binary };
                fr.set(rd, Term::int(is as i64))?;
            }
            op::BIN_CAT => {
                let rd = fr.u8()?;
                let ra = fr.u8()?;
                let rb = fr.u8()?;
                let (a, b) = (fr.get(ra)?, fr.get(rb)?);
                let joined: Vec<u8> = unsafe {
                    if !a.is_boxed()
                        || a.kind() != ygg_term::Kind::Binary
                        || !b.is_boxed()
                        || b.kind() != ygg_term::Kind::Binary
                    {
                        return Err(Trap::Badarg);
                    }
                    let mut v = a.bin_bytes().to_vec();
                    v.extend_from_slice(b.bin_bytes());
                    v
                };
                let out = alloc_term(api, |h| h.binary(&joined))?;
                fr.set(rd, out)?;
            }
            op::LIST_CAT => {
                let rd = fr.u8()?;
                let ra = fr.u8()?;
                let rb = fr.u8()?;
                let (a, b) = (fr.get(ra)?, fr.get(rb)?);
                let mut elems: Vec<Term> = Vec::new();
                let mut cur = a;
                unsafe {
                    while cur.is_boxed() && cur.kind() == ygg_term::Kind::Cons {
                        elems.push(cur.head());
                        cur = cur.tail();
                    }
                }
                if !cur.is_nil() {
                    return Err(Trap::Badarg);
                }
                let out = alloc_term(api, |h| {
                    let mut out = b;
                    for e in elems.iter().rev() {
                        out = h.cons(*e, out)?;
                    }
                    Ok(out)
                })?;
                fr.set(rd, out)?;
            }
            op::BIN_AT => {
                let rd = fr.u8()?;
                let rb = fr.u8()?;
                let ri = fr.u8()?;
                let b = fr.get(rb)?;
                let idx = fr.get(ri)?.as_int().ok_or(Trap::Badarg)?;
                let byte = unsafe {
                    if !b.is_boxed() || b.kind() != ygg_term::Kind::Binary || idx < 0 {
                        return Err(Trap::Badarg);
                    }
                    let bytes = b.bin_bytes();
                    if idx as usize >= bytes.len() {
                        return Err(Trap::Badarg);
                    }
                    bytes[idx as usize]
                };
                fr.set(rd, Term::int(byte as i64))?;
            }
            op::BIN_PART => {
                let rd = fr.u8()?;
                let rb = fr.u8()?;
                let ro = fr.u8()?;
                let rl = fr.u8()?;
                let b = fr.get(rb)?;
                let off = fr.get(ro)?.as_int().ok_or(Trap::Badarg)?;
                let len = fr.get(rl)?.as_int().ok_or(Trap::Badarg)?;
                let part: Vec<u8> = unsafe {
                    if !b.is_boxed() || b.kind() != ygg_term::Kind::Binary || off < 0 || len < 0 {
                        return Err(Trap::Badarg);
                    }
                    let bytes = b.bin_bytes();
                    let (off, len) = (off as usize, len as usize);
                    if off + len > bytes.len() {
                        return Err(Trap::Badarg);
                    }
                    bytes[off..off + len].to_vec()
                };
                let out = alloc_term(api, |h| h.binary(&part))?;
                fr.set(rd, out)?;
            }
            op::MAP_NEW => {
                let rd = fr.u8()?;
                let n = fr.u8()? as usize;
                let mut pairs = Vec::with_capacity(n);
                for _ in 0..n {
                    let rk = fr.u8()?;
                    let rv = fr.u8()?;
                    pairs.push((fr.get(rk)?, fr.get(rv)?));
                }
                let m = loop {
                    match api.heap().map_from_pairs(&mut pairs) {
                        Ok(m) => break m,
                        Err(ygg_term::MapError::NonImmediateKey) => return Err(Trap::Badarg),
                        Err(ygg_term::MapError::Heap(_)) => {
                            if !api.heap_grow() {
                                return Err(Trap::HeapFull);
                            }
                        }
                    }
                };
                fr.set(rd, m)?;
            }
            op::MAP_GET => {
                let rd = fr.u8()?;
                let rm = fr.u8()?;
                let rk = fr.u8()?;
                let m = fr.get(rm)?;
                let k = fr.get(rk)?;
                let v = unsafe {
                    if !m.is_boxed() || m.kind() != ygg_term::Kind::Map {
                        return Err(Trap::Badarg);
                    }
                    m.map_get(k).ok_or(Trap::Badarg)?
                };
                fr.set(rd, v)?;
            }
            op::MAP_PUT => {
                let rd = fr.u8()?;
                let rm = fr.u8()?;
                let rk = fr.u8()?;
                let rv = fr.u8()?;
                let m = fr.get(rm)?;
                let (k, v) = (fr.get(rk)?, fr.get(rv)?);
                unsafe {
                    if !m.is_boxed() || m.kind() != ygg_term::Kind::Map {
                        return Err(Trap::Badarg);
                    }
                    let out = loop {
                        match api.heap().map_put(m, k, v) {
                            Ok(o) => break o,
                            Err(ygg_term::MapError::NonImmediateKey) => return Err(Trap::Badarg),
                            Err(ygg_term::MapError::Heap(_)) => {
                                if !api.heap_grow() {
                                    return Err(Trap::HeapFull);
                                }
                            }
                        }
                    };
                    fr.set(rd, out)?;
                }
            }
            _ => return Err(Trap::BadCode),
        }
    }
}

fn jump(fr: &mut Frame, off: i32, api: &mut dyn SystemApi) -> Result<(), Trap> {
    if off < 0 {
        // Backward jump = loop back-edge = safepoint.
        api.safepoint();
    }
    let pc = fr.pc as i64 + off as i64;
    if pc < 0 || pc as usize > fr.code.len() {
        return Err(Trap::BadCode);
    }
    fr.pc = pc as usize;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::VecDeque;
    use alloc::string::String;
    use ygg_bytecode::{CodeBuilder, Function};

    // `heap` borrows `buf`'s stable allocation; both move together.
    struct MockApi {
        #[allow(dead_code)]
        buf: Vec<u64>,
        heap: Heap,
        mailbox: VecDeque<Term>,
        printed: Vec<String>,
        spawned: Vec<(u32, Term)>,
        safepoints: usize,
        buffers: alloc::collections::BTreeMap<i64, Vec<u8>>,
    }

    impl MockApi {
        fn new() -> Box<MockApi> {
            let mut buf = vec![0u64; 8192];
            let heap = unsafe { Heap::new(buf.as_mut_ptr().cast(), buf.len() * 8) };
            Box::new(MockApi {
                heap,
                buf,
                mailbox: VecDeque::new(),
                printed: Vec::new(),
                spawned: Vec::new(),
                safepoints: 0,
                buffers: alloc::collections::BTreeMap::new(),
            })
        }
    }

    impl SystemApi for MockApi {
        fn heap(&mut self) -> &mut Heap {
            &mut self.heap
        }
        fn self_pid(&self) -> u64 {
            7
        }
        fn send(&mut self, to: Term, msg: Term) -> Result<(), Trap> {
            // Loopback: sends to self land in the mock mailbox.
            assert_eq!(to.as_pid(), Some(7));
            self.mailbox.push_back(msg);
            Ok(())
        }
        fn recv(&mut self) -> Term {
            self.mailbox
                .pop_front()
                .expect("mock recv on empty mailbox")
        }
        fn spawn(&mut self, fn_idx: u32, arg: Term) -> Result<u64, Trap> {
            self.spawned.push((fn_idx, arg));
            Ok(100 + self.spawned.len() as u64)
        }
        fn safepoint(&mut self) {
            self.safepoints += 1;
        }
        fn atom_global(&mut self, local: u32) -> u32 {
            1000 + local
        }
        fn print(&mut self, t: Term) {
            let mut s = String::new();
            unsafe { ygg_term::fmt_term(t, &mut s, &|a| if a == 1000 { "ok" } else { "?" }) }
                .unwrap();
            self.printed.push(s);
        }
        fn port_open(&mut self, kind: u8) -> Result<Term, Trap> {
            Ok(Term::port(kind as u64 + 50))
        }
        fn port_submit(&mut self, port: Term, _op: u8, arg0: Term, tag: Term) -> Result<(), Trap> {
            port.as_port().ok_or(Trap::Badarg)?;
            arg0.as_int().ok_or(Trap::Badarg)?;
            tag.as_int().ok_or(Trap::Badarg)?;
            Ok(())
        }
        fn call_ext(
            &mut self,
            _module_atom: u32,
            _fname_atom: u32,
            _args: &[Term],
        ) -> Result<Term, Trap> {
            Err(Trap::Badarg) // no module table in the mock
        }
        fn buf_to_bin(&mut self, id: i64) -> Result<Term, Trap> {
            let data = self.buffers.remove(&id).ok_or(Trap::Badarg)?;
            Ok(self.heap.binary(&data)?)
        }
        fn bin_to_buf(&mut self, bin: Term) -> Result<Term, Trap> {
            let bytes = unsafe {
                if bin.kind() != ygg_term::Kind::Binary {
                    return Err(Trap::Badarg);
                }
                bin.bin_bytes().to_vec()
            };
            let id = 900 + self.buffers.len() as i64;
            self.buffers.insert(id, bytes);
            Ok(Term::int(id))
        }
    }

    /// fact(n) = n <= 1 ? 1 : n * fact(n - 1)
    fn fact_module() -> Module {
        let mut b = CodeBuilder::new();
        // r0 = n. r1 = 2, r2 = (n < 2)
        b.u8(op::LOAD_INT).u8(1).i64(2);
        b.u8(op::CMP_LT).u8(2).u8(0).u8(1);
        b.u8(op::JMP_IF).u8(2).label_ref(0);
        // r3 = n - 1; r4 = fact(r3); r5 = n * r4; ret r5
        b.u8(op::LOAD_INT).u8(1).i64(1);
        b.u8(op::SUB).u8(3).u8(0).u8(1);
        b.u8(op::CALL).u8(4).u32(0).u8(1).u8(3);
        b.u8(op::MUL).u8(5).u8(0).u8(4);
        b.u8(op::RET).u8(5);
        b.bind(0);
        b.u8(op::LOAD_INT).u8(5).i64(1);
        b.u8(op::RET).u8(5);
        Module {
            atoms: vec!["fact".into()],
            functions: vec![Function {
                name_atom: 0,
                arity: 1,
                nregs: 6,
                code: b.finish().unwrap(),
            }],
        }
    }

    #[test]
    fn factorial() {
        let m = fact_module();
        let mut api = MockApi::new();
        let r = run_function(&m, 0, &[Term::int(10)], &mut *api).unwrap();
        assert_eq!(r.as_int(), Some(3628800));
        assert!(api.safepoints >= 9, "calls must hit safepoints");
    }

    /// Loop 5 times sending self an int, then receive them all into a list.
    #[test]
    fn send_recv_loop_with_backedge_safepoints() {
        let mut b = CodeBuilder::new();
        // r1 = 0 (i), r2 = 5, r3 = self
        b.u8(op::LOAD_INT).u8(1).i64(0);
        b.u8(op::LOAD_INT).u8(2).i64(5);
        b.u8(op::SELF_PID).u8(3);
        b.bind(0); // loop head
        b.u8(op::CMP_LT).u8(4).u8(1).u8(2);
        b.u8(op::JMP_IF).u8(4).label_ref(1);
        b.u8(op::JMP).label_ref(2); // exit loop
        b.bind(1);
        b.u8(op::SEND).u8(3).u8(1);
        b.u8(op::LOAD_INT).u8(5).i64(1);
        b.u8(op::ADD).u8(1).u8(1).u8(5);
        b.u8(op::JMP).label_ref(0); // back edge -> safepoint
        b.bind(2);
        // drain three: r6 = recv; r7 = recv; print both, ret tuple
        b.u8(op::RECV).u8(6);
        b.u8(op::RECV).u8(7);
        b.u8(op::MAKE_TUPLE).u8(8).u8(2).u8(6).u8(7);
        b.u8(op::PRINT).u8(8);
        b.u8(op::RET).u8(8);
        let m = Module {
            atoms: vec!["main".into()],
            functions: vec![Function {
                name_atom: 0,
                arity: 0,
                nregs: 9,
                code: b.finish().unwrap(),
            }],
        };
        let mut api = MockApi::new();
        let r = run_function(&m, 0, &[], &mut *api).unwrap();
        unsafe {
            assert_eq!(r.tuple_elem(0).as_int(), Some(0));
            assert_eq!(r.tuple_elem(1).as_int(), Some(1));
        }
        assert_eq!(api.printed, vec!["{0, 1}"]);
        assert!(api.safepoints >= 5, "back edges must hit safepoints");
        assert_eq!(api.mailbox.len(), 3, "two of five drained");
    }

    #[test]
    fn traps() {
        // HEAD of a non-list traps Badarg.
        let mut b = CodeBuilder::new();
        b.u8(op::LOAD_INT).u8(0).i64(3);
        b.u8(op::HEAD).u8(1).u8(0);
        let m = Module {
            atoms: vec!["t".into()],
            functions: vec![Function {
                name_atom: 0,
                arity: 0,
                nregs: 2,
                code: b.finish().unwrap(),
            }],
        };
        assert_eq!(
            run_function(&m, 0, &[], &mut *MockApi::new()),
            Err(Trap::Badarg)
        );

        // Truncated code traps BadCode.
        let m2 = Module {
            atoms: vec!["t".into()],
            functions: vec![Function {
                name_atom: 0,
                arity: 0,
                nregs: 1,
                code: vec![op::LOAD_INT, 0],
            }],
        };
        assert_eq!(
            run_function(&m2, 0, &[], &mut *MockApi::new()),
            Err(Trap::BadCode)
        );

        // ExitAtom surfaces the global atom.
        let mut b3 = CodeBuilder::new();
        b3.u8(op::LOAD_ATOM).u8(0).u32(4);
        b3.u8(op::EXIT_ATOM).u8(0);
        let m3 = Module {
            atoms: vec!["t".into()],
            functions: vec![Function {
                name_atom: 0,
                arity: 0,
                nregs: 1,
                code: b3.finish().unwrap(),
            }],
        };
        assert_eq!(
            run_function(&m3, 0, &[], &mut *MockApi::new()),
            Err(Trap::Exit(1004))
        );
    }
}
