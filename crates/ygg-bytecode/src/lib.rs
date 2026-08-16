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

impl Module {
    pub fn function_named(&self, name: &str) -> Option<usize> {
        self.functions
            .iter()
            .position(|f| self.atoms.get(f.name_atom as usize).is_some_and(|a| a == name))
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
            functions.push(Function { name_atom, arity, nregs, code: r.take(code_len)?.to_vec() });
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
            functions: vec![Function { name_atom: 0, arity: 1, nregs: 4, code: vec![1, 2, 3] }],
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
        let m = Module { atoms: vec!["x".into()], functions: vec![] };
        let enc = m.encode();
        for cut in 0..enc.len() {
            assert!(Module::decode(&enc[..cut]).is_err(), "cut at {cut} accepted");
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
