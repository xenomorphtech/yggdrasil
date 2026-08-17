//! The Yggdrasil bytecode: instruction set and `.yggm` module format.
//!
//! Register-based, function-structured, term-typed — designed to be (a)
//! verifiable at load (M6) and (b) lowered 1:1 to Cranelift IR (M8). Runtime
//! type checks trap today; the verifier will let the JIT elide them later.
//!
//! Registers are per-frame (`nregs` per function), args arrive in r0..arity-1.
//! Jump offsets are byte-relative to the *next* instruction. Backward jumps
//! and calls are safepoints.
//!
//! Module binary layout (little-endian):
//! ```text
//! magic "YGGM1\n"
//! u32 atom_count,   atom_count * { u16 len, bytes }   (module-local atoms)
//! u32 fn_count,     fn_count * { u32 name_atom, u8 arity, u8 nregs,
//!                                u32 code_len, code bytes }
//! ```

#![cfg_attr(not(test), no_std)]

extern crate alloc;

pub mod verify;

use alloc::string::String;
use alloc::vec::Vec;

/// Opcodes. Operands noted as: r = u8 register, i64/u32/u8 immediates, off = i32.
pub mod op {
    pub const NOP: u8 = 0;
    pub const LOAD_INT: u8 = 1; //  rd, i64
    pub const LOAD_ATOM: u8 = 2; // rd, u32 (module-local atom index)
    pub const LOAD_NIL: u8 = 3; //  rd
    pub const MOVE: u8 = 4; //      rd, rs
    pub const SELF_PID: u8 = 5; //  rd
    pub const MAKE_TUPLE: u8 = 6; // rd, n:u8, n * r
    pub const GET_ELEM: u8 = 7; //  rd, rs, idx:u8
    pub const CONS: u8 = 8; //      rd, rhead, rtail
    pub const HEAD: u8 = 9; //      rd, rs
    pub const TAIL: u8 = 10; //     rd, rs
    pub const ADD: u8 = 11; //      rd, ra, rb
    pub const SUB: u8 = 12; //      rd, ra, rb
    pub const MUL: u8 = 13; //      rd, ra, rb
    pub const CMP_EQ: u8 = 14; //   rd, ra, rb   (structural; int 1/0)
    pub const CMP_LT: u8 = 15; //   rd, ra, rb   (ints only)
    pub const JMP: u8 = 16; //      off:i32
    pub const JMP_IF: u8 = 17; //   rc, off:i32  (taken when rc is int != 0)
    pub const CALL: u8 = 18; //     rd, fn:u32, nargs:u8, nargs * r
    pub const RET: u8 = 19; //      rs
    pub const SPAWN: u8 = 20; //    rd, fn:u32, rarg (arg must be an immediate)
    pub const SEND: u8 = 21; //     rtarget, rmsg
    pub const RECV: u8 = 22; //     rd
    pub const PRINT: u8 = 23; //    rs (debug console; real IO is ports)
    pub const EXIT_ATOM: u8 = 24; // rs (atom = exit reason)
    pub const PORT_OPEN: u8 = 25; //   rd, kind:u8
    pub const PORT_SUBMIT: u8 = 26; // rport, op:u8, rarg0, rtag (tag -1 = no CQE)
    /// External call: rd, module_atom:u32, fname_atom:u32, nargs:u8, nargs * r.
    /// Resolves through the module table at call time — the hot-code-loading
    /// migration point (BEAM's fully-qualified call).
    pub const CALL_EXT: u8 = 27;
    // Integer bit operations (ints only; trap otherwise). Shifts trap on
    // amounts outside 0..=60 (i61 world; Erlang's negative-shift form is not
    // supported).
    pub const BAND: u8 = 28; // rd, ra, rb
    pub const BOR: u8 = 29; //  rd, ra, rb
    pub const BXOR: u8 = 30; // rd, ra, rb
    pub const BSL: u8 = 31; //  rd, ra, rb
    pub const BSR: u8 = 32; //  rd, ra, rb (arithmetic)
    pub const BNOT: u8 = 33; // rd, rs
    // Binaries.
    pub const BIN_NEW: u8 = 34; //       rd, len:u32, len * byte (constant)
    pub const BIN_FROM_LIST: u8 = 35; // rd, rs (list of ints 0..=255)
    pub const BIN_TO_LIST: u8 = 36; //   rd, rs
    pub const BIN_SIZE: u8 = 37; //      rd, rs
    // Bridge between port buffer ids (ints) and binary terms.
    pub const BUF_TO_BIN: u8 = 38; //    rd, rs (consumes the kernel buffer)
    pub const BIN_TO_BUF: u8 = 39; //    rd, rs (new kernel buffer, returns id)
    // Maps (flat, immediate keys — the struct/record representation).
    pub const MAP_NEW: u8 = 40; //  rd, n:u8, n * (rkey, rval)
    pub const MAP_GET: u8 = 41; //  rd, rmap, rkey (missing key traps)
    pub const MAP_PUT: u8 = 42; //  rd, rmap, rkey, rval (functional update)
    pub const IS_BINARY: u8 = 43; // rd, rs (int 1/0)
    pub const BIN_CAT: u8 = 44; //  rd, ra, rb (binary append)
    pub const LIST_CAT: u8 = 45; // rd, ra, rb (erlang ++: ra proper list)
    pub const BIN_PART: u8 = 46; // rd, rbin, roff, rlen (bounds-checked slice)
    /// Tail external call: module_atom:u32, fname_atom:u32, nargs:u8, nargs*r.
    /// Terminal (no destination): the engine unwinds to its trampoline, which
    /// re-dispatches — constant native stack, and the GC-safe point.
    pub const TAIL_CALL_EXT: u8 = 47;
    /// All-register submit: rport, rop, rarg0, rarg1, rtag — exposes
    /// `Sqe.arg1` (cmd+aux buffer pairs) and makes the op/tag dynamic, which
    /// is what device protocols written in bytecode need. tag -1 = no CQE.
    pub const PORT_SUBMIT2: u8 = 48;
    /// Tail local call: fn:u32, nargs:u8, nargs*r. Terminal like
    /// `TAIL_CALL_EXT`, but the trampoline re-enters the *same module
    /// instance* by function index — no name resolution, and BEAM local-call
    /// semantics (no version migration). For multi-function modules; Lux
    /// function-modules instead tail-call themselves by their own
    /// content-address (self-references are canonicalized before hashing,
    /// so a module's code can and does name its own hash).
    pub const TAIL_CALL: u8 = 49;
    /// rd, rbuf, roff, rbin: overwrite bytes of an existing kernel buffer at
    /// an offset (bounds-checked, never grows — a pinned buffer's physical
    /// address stays valid). The device-backing update path: how a bytecode
    /// driver animates a framebuffer it has attached. Result int 0.
    pub const BUF_WRITE: u8 = 50;
    /// rms: park the process for that many milliseconds without consuming a
    /// mailbox message (timer-wheel sleep; lowers `receive after` pacing).
    pub const SLEEP_MS: u8 = 51;
    /// rd, rsize: allocate a fixed-size zero-filled kernel blob, result = id.
    /// With BUF_WRITE/BUF_READ this is the mutable-fixed-blob primitive
    /// (BEAM's atomics/ETS niche): off-heap, GC-immune, stable physical
    /// address — the thing framebuffers and DMA backings are made of.
    pub const BUF_NEW: u8 = 52;
    /// rd, rbuf, roff, rlen: copy a bounds-checked slice of a blob out as a
    /// fresh binary term.
    pub const BUF_READ: u8 = 53;
    /// rd: milliseconds since timer start (1 kHz monotonic clock).
    pub const TICKS: u8 = 54;

    /// rd, rbin, ridx — byte at `ridx` of a binary as an int term.
    /// Allocation-free byte indexing (`BIN_PART`+`BIN_TO_LIST`+`HEAD` costs
    /// two heap allocations per byte read); traps badarg out of range.
    pub const BIN_AT: u8 = 55;
}

#[derive(Debug, Clone)]
pub struct Function {
    /// Module-local atom index of the function's name.
    pub name_atom: u32,
    pub arity: u8,
    pub nregs: u8,
    pub code: Vec<u8>,
}

#[derive(Debug, Clone, Default)]
pub struct Module {
    pub atoms: Vec<String>,
    pub functions: Vec<Function>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    BadMagic,
    Truncated,
    BadUtf8,
}

const MAGIC: &[u8; 6] = b"YGGM1\n";

/// Skip one instruction's operands (opcode already consumed). Returns the
/// new offset, or None on a truncated/unknown instruction. Must cover every
/// opcode in `op` — the verifier rejects anything else before it runs.
fn skip_operands(opcode: u8, code: &[u8], at: usize) -> Option<usize> {
    let fixed = |n: usize| at.checked_add(n).filter(|&e| e <= code.len());
    match opcode {
        op::NOP => fixed(0),
        op::LOAD_INT => fixed(9),
        op::LOAD_ATOM => fixed(5),
        op::SPAWN => fixed(6),
        op::LOAD_NIL | op::SELF_PID | op::RECV | op::RET | op::EXIT_ATOM | op::PRINT
        | op::TICKS | op::SLEEP_MS => fixed(1),
        op::MOVE | op::HEAD | op::TAIL | op::SEND | op::BNOT | op::BIN_FROM_LIST
        | op::BIN_TO_LIST | op::BIN_SIZE | op::BUF_TO_BIN | op::BIN_TO_BUF | op::IS_BINARY
        | op::PORT_OPEN | op::BUF_NEW => fixed(2),
        op::GET_ELEM | op::CONS | op::ADD | op::SUB | op::MUL | op::CMP_EQ | op::CMP_LT
        | op::BAND | op::BOR | op::BXOR | op::BSL | op::BSR | op::BIN_CAT | op::LIST_CAT
        | op::MAP_GET | op::BIN_AT => fixed(3),
        op::MAP_PUT | op::BIN_PART | op::PORT_SUBMIT | op::BUF_WRITE | op::BUF_READ => fixed(4),
        op::PORT_SUBMIT2 => fixed(5),
        op::JMP => fixed(4),
        op::JMP_IF => fixed(5),
        op::MAKE_TUPLE => {
            let n = *code.get(at + 1)? as usize;
            fixed(2 + n)
        }
        op::MAP_NEW => {
            let n = *code.get(at + 1)? as usize;
            fixed(2 + 2 * n)
        }
        op::BIN_NEW => {
            let len =
                u32::from_le_bytes(code.get(at + 1..at + 5)?.try_into().ok()?) as usize;
            fixed(5 + len)
        }
        op::CALL => {
            let n = *code.get(at + 5)? as usize;
            fixed(6 + n)
        }
        op::TAIL_CALL => {
            let n = *code.get(at + 4)? as usize;
            fixed(5 + n)
        }
        op::CALL_EXT => {
            let n = *code.get(at + 9)? as usize;
            fixed(10 + n)
        }
        op::TAIL_CALL_EXT => {
            let n = *code.get(at + 8)? as usize;
            fixed(9 + n)
        }
        _ => None,
    }
}

/// Which functions of this verified module can return the tail sentinel to a
/// native caller: those containing `TAIL_CALL`/`TAIL_CALL_EXT`, and
/// transitively those making a sibling `CALL` to such a function (sibling
/// calls propagate the sentinel by unwinding). Callers that bind an external
/// call to a function whose bit is false may skip the sentinel check.
pub fn sentinel_returners(m: &Module) -> Vec<bool> {
    let mut bits = alloc::vec![false; m.functions.len()];
    let mut callees: Vec<Vec<u32>> = alloc::vec![Vec::new(); m.functions.len()];
    for (i, f) in m.functions.iter().enumerate() {
        let mut at = 0usize;
        while at < f.code.len() {
            let opcode = f.code[at];
            match opcode {
                op::TAIL_CALL | op::TAIL_CALL_EXT => bits[i] = true,
                op::CALL => {
                    // Operands: rd:u8, fn:u32 — the callee index is at +2.
                    if let Some(t) = f.code.get(at + 2..at + 6) {
                        callees[i].push(u32::from_le_bytes(t.try_into().unwrap()));
                    }
                }
                _ => {}
            }
            match skip_operands(opcode, &f.code, at + 1) {
                Some(next) => at = next,
                // Unknown/truncated: be conservative — it may return anything.
                None => {
                    bits[i] = true;
                    break;
                }
            }
        }
    }
    // Propagate through sibling calls to a fixpoint.
    loop {
        let mut changed = false;
        for i in 0..bits.len() {
            if !bits[i]
                && callees[i]
                    .iter()
                    .any(|&c| bits.get(c as usize).copied().unwrap_or(true))
            {
                bits[i] = true;
                changed = true;
            }
        }
        if !changed {
            return bits;
        }
    }
}

impl Module {
    pub fn function_named(&self, name: &str) -> Option<usize> {
        self.functions.iter().position(|f| {
            self.atoms
                .get(f.name_atom as usize)
                .is_some_and(|a| a == name)
        })
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(self.atoms.len() as u32).to_le_bytes());
        for a in &self.atoms {
            out.extend_from_slice(&(a.len() as u16).to_le_bytes());
            out.extend_from_slice(a.as_bytes());
        }
        out.extend_from_slice(&(self.functions.len() as u32).to_le_bytes());
        for f in &self.functions {
            out.extend_from_slice(&f.name_atom.to_le_bytes());
            out.push(f.arity);
            out.push(f.nregs);
            out.extend_from_slice(&(f.code.len() as u32).to_le_bytes());
            out.extend_from_slice(&f.code);
        }
        out
    }

    pub fn decode(bytes: &[u8]) -> Result<Module, DecodeError> {
        let mut r = Reader { b: bytes, at: 0 };
        if r.take(6)? != MAGIC {
            return Err(DecodeError::BadMagic);
        }
        let natoms = r.u32()?;
        let mut atoms = Vec::new();
        for _ in 0..natoms {
            let len = r.u16()? as usize;
            let s = core::str::from_utf8(r.take(len)?).map_err(|_| DecodeError::BadUtf8)?;
            atoms.push(String::from(s));
        }
        let nfns = r.u32()?;
        let mut functions = Vec::new();
        for _ in 0..nfns {
            let name_atom = r.u32()?;
            let arity = r.u8()?;
            let nregs = r.u8()?;
            let code_len = r.u32()? as usize;
            functions.push(Function {
                name_atom,
                arity,
                nregs,
                code: r.take(code_len)?.to_vec(),
            });
        }
        Ok(Module { atoms, functions })
    }
}

struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], DecodeError> {
        let end = self.at.checked_add(n).ok_or(DecodeError::Truncated)?;
        if end > self.b.len() {
            return Err(DecodeError::Truncated);
        }
        let s = &self.b[self.at..end];
        self.at = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, DecodeError> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, DecodeError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, DecodeError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
}

/// Builder for emitting code with label fixups (shared by ygg-asm and tests).
#[derive(Default)]
pub struct CodeBuilder {
    pub code: Vec<u8>,
    /// (patch position of the i32, label id)
    fixups: Vec<(usize, u32)>,
    labels: alloc::collections::BTreeMap<u32, usize>,
}

impl CodeBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn u8(&mut self, v: u8) -> &mut Self {
        self.code.push(v);
        self
    }
    pub fn u32(&mut self, v: u32) -> &mut Self {
        self.code.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn i64(&mut self, v: i64) -> &mut Self {
        self.code.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn bytes(&mut self, v: &[u8]) -> &mut Self {
        self.code.extend_from_slice(v);
        self
    }
    /// Emit a jump-offset placeholder pointing at `label`.
    pub fn label_ref(&mut self, label: u32) -> &mut Self {
        self.fixups.push((self.code.len(), label));
        self.code.extend_from_slice(&0i32.to_le_bytes());
        self
    }
    pub fn bind(&mut self, label: u32) {
        self.labels.insert(label, self.code.len());
    }
    /// Resolve all label references; offsets are relative to the byte after
    /// the i32 (i.e. the next instruction, since offsets end instructions).
    pub fn finish(mut self) -> Result<Vec<u8>, u32> {
        for (pos, label) in self.fixups {
            let target = *self.labels.get(&label).ok_or(label)?;
            let next = pos + 4;
            let off = target as i64 - next as i64;
            self.code[pos..pos + 4].copy_from_slice(&(off as i32).to_le_bytes());
        }
        Ok(self.code)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let m = Module {
            atoms: vec!["main".into(), "hello".into()],
            functions: vec![Function {
                name_atom: 0,
                arity: 1,
                nregs: 4,
                code: vec![1, 2, 3],
            }],
        };
        let enc = m.encode();
        let d = Module::decode(&enc).unwrap();
        assert_eq!(d.atoms, m.atoms);
        assert_eq!(d.functions[0].code, vec![1, 2, 3]);
        assert_eq!(d.function_named("main"), Some(0));
        assert_eq!(d.function_named("nope"), None);
    }

    #[test]
    fn truncated_rejected() {
        let m = Module {
            atoms: vec!["x".into()],
            functions: vec![],
        };
        let enc = m.encode();
        for cut in 0..enc.len() {
            assert!(
                Module::decode(&enc[..cut]).is_err(),
                "cut at {cut} accepted"
            );
        }
    }

    #[test]
    fn labels_resolve_backward_and_forward() {
        let mut b = CodeBuilder::new();
        b.bind(0);
        b.u8(op::NOP);
        b.u8(op::JMP).label_ref(1); // forward
        b.u8(op::NOP);
        b.bind(1);
        b.u8(op::JMP).label_ref(0); // backward
        let code = b.finish().unwrap();
        // JMP at 1: i32 at 2..6, next=6, target=7 -> +1
        assert_eq!(i32::from_le_bytes(code[2..6].try_into().unwrap()), 1);
        // JMP at 7: i32 at 8..12, next=12, target=0 -> -12
        assert_eq!(i32::from_le_bytes(code[8..12].try_into().unwrap()), -12);
    }
}
