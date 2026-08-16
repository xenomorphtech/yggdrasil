//! The bytecode verifier — the isolation boundary of the whole OS.
//!
//! Everything that runs in ring 0 as a "process" is either trusted kernel Rust
//! or bytecode that passed this check. The verifier proves, statically:
//!
//! - every instruction decodes (no truncation, no unknown opcodes)
//! - every register operand is within the function's declared frame
//! - every jump lands on an instruction boundary inside the same function
//! - every call/spawn targets a real function with the right arity
//! - every atom operand indexes the module atom table
//! - control cannot fall off the end of a function
//!
//! Value-level typing (tuple arity, int-ness) stays a runtime check in the
//! interpreter for now; when the Cranelift tier lands, what is proven here is
//! what the JIT is allowed to stop checking.

use alloc::collections::BTreeSet;

use crate::{Function, Module, op};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifyError {
    /// Function's name atom is out of range.
    BadFunctionName {
        fn_idx: usize,
    },
    /// arity > nregs (arguments wouldn't fit the frame).
    ArityExceedsRegs {
        fn_idx: usize,
    },
    UnknownOpcode {
        fn_idx: usize,
        at: usize,
        opcode: u8,
    },
    TruncatedInstruction {
        fn_idx: usize,
        at: usize,
    },
    BadRegister {
        fn_idx: usize,
        at: usize,
        reg: u8,
    },
    BadAtom {
        fn_idx: usize,
        at: usize,
        atom: u32,
    },
    BadJumpTarget {
        fn_idx: usize,
        at: usize,
    },
    BadCallTarget {
        fn_idx: usize,
        at: usize,
        callee: u32,
    },
    BadCallArity {
        fn_idx: usize,
        at: usize,
    },
    /// Execution can run past the last instruction.
    FallsOffEnd {
        fn_idx: usize,
    },
}

pub fn verify(m: &Module) -> Result<(), VerifyError> {
    for (fn_idx, f) in m.functions.iter().enumerate() {
        verify_fn(m, fn_idx, f)?;
    }
    Ok(())
}

struct Scan<'a> {
    code: &'a [u8],
    at: usize,
    start: usize,
    fn_idx: usize,
}

impl<'a> Scan<'a> {
    fn u8(&mut self) -> Result<u8, VerifyError> {
        let v = *self
            .code
            .get(self.at)
            .ok_or(VerifyError::TruncatedInstruction {
                fn_idx: self.fn_idx,
                at: self.start,
            })?;
        self.at += 1;
        Ok(v)
    }
    fn u32(&mut self) -> Result<u32, VerifyError> {
        let s = self
            .code
            .get(self.at..self.at + 4)
            .ok_or(VerifyError::TruncatedInstruction {
                fn_idx: self.fn_idx,
                at: self.start,
            })?;
        self.at += 4;
        Ok(u32::from_le_bytes(s.try_into().unwrap()))
    }
    fn skip(&mut self, n: usize) -> Result<(), VerifyError> {
        if self.at + n > self.code.len() {
            return Err(VerifyError::TruncatedInstruction {
                fn_idx: self.fn_idx,
                at: self.start,
            });
        }
        self.at += n;
        Ok(())
    }
}

fn verify_fn(m: &Module, fn_idx: usize, f: &Function) -> Result<(), VerifyError> {
    if m.atoms.get(f.name_atom as usize).is_none() {
        return Err(VerifyError::BadFunctionName { fn_idx });
    }
    if f.arity > f.nregs {
        return Err(VerifyError::ArityExceedsRegs { fn_idx });
    }

    let mut boundaries: BTreeSet<usize> = BTreeSet::new();
    // (jump-site, absolute target)
    let mut jump_targets: alloc::vec::Vec<(usize, i64)> = alloc::vec::Vec::new();
    let mut s = Scan {
        code: &f.code,
        at: 0,
        start: 0,
        fn_idx,
    };
    // Whether the previous instruction allows falling into the next offset.
    let mut fell_through = true;

    let reg_ok = |r: u8, at: usize| {
        if r >= f.nregs {
            Err(VerifyError::BadRegister { fn_idx, at, reg: r })
        } else {
            Ok(())
        }
    };

    while s.at < f.code.len() {
        s.start = s.at;
        boundaries.insert(s.start);
        let at = s.start;
        let opcode = s.u8()?;
        fell_through = true;
        match opcode {
            op::NOP => {}
            op::LOAD_INT => {
                reg_ok(s.u8()?, at)?;
                s.skip(8)?;
            }
            op::LOAD_ATOM => {
                reg_ok(s.u8()?, at)?;
                let a = s.u32()?;
                if m.atoms.get(a as usize).is_none() {
                    return Err(VerifyError::BadAtom {
                        fn_idx,
                        at,
                        atom: a,
                    });
                }
            }
            op::LOAD_NIL | op::SELF_PID | op::RECV => {
                reg_ok(s.u8()?, at)?;
            }
            op::MOVE | op::HEAD | op::TAIL => {
                reg_ok(s.u8()?, at)?;
                reg_ok(s.u8()?, at)?;
            }
            op::MAKE_TUPLE => {
                reg_ok(s.u8()?, at)?;
                let n = s.u8()?;
                for _ in 0..n {
                    reg_ok(s.u8()?, at)?;
                }
            }
            op::GET_ELEM => {
                reg_ok(s.u8()?, at)?;
                reg_ok(s.u8()?, at)?;
                s.skip(1)?;
            }
            op::CONS | op::ADD | op::SUB | op::MUL | op::CMP_EQ | op::CMP_LT => {
                reg_ok(s.u8()?, at)?;
                reg_ok(s.u8()?, at)?;
                reg_ok(s.u8()?, at)?;
            }
            op::JMP => {
                let off = s.u32()? as i32;
                jump_targets.push((at, s.at as i64 + off as i64));
                fell_through = false;
            }
            op::JMP_IF => {
                reg_ok(s.u8()?, at)?;
                let off = s.u32()? as i32;
                jump_targets.push((at, s.at as i64 + off as i64));
            }
            op::CALL => {
                reg_ok(s.u8()?, at)?;
                let callee = s.u32()?;
                let nargs = s.u8()?;
                let Some(target) = m.functions.get(callee as usize) else {
                    return Err(VerifyError::BadCallTarget { fn_idx, at, callee });
                };
                if target.arity != nargs {
                    return Err(VerifyError::BadCallArity { fn_idx, at });
                }
                for _ in 0..nargs {
                    reg_ok(s.u8()?, at)?;
                }
            }
            op::RET | op::EXIT_ATOM => {
                reg_ok(s.u8()?, at)?;
                fell_through = false;
            }
            op::SPAWN => {
                reg_ok(s.u8()?, at)?;
                let callee = s.u32()?;
                let Some(target) = m.functions.get(callee as usize) else {
                    return Err(VerifyError::BadCallTarget { fn_idx, at, callee });
                };
                // The kernel spawns with exactly one (immediate) argument.
                if target.arity != 1 {
                    return Err(VerifyError::BadCallArity { fn_idx, at });
                }
                reg_ok(s.u8()?, at)?;
            }
            op::SEND => {
                reg_ok(s.u8()?, at)?;
                reg_ok(s.u8()?, at)?;
            }
            op::PRINT => {
                reg_ok(s.u8()?, at)?;
            }
            op::PORT_OPEN => {
                reg_ok(s.u8()?, at)?;
                s.skip(1)?;
            }
            op::PORT_SUBMIT => {
                reg_ok(s.u8()?, at)?;
                s.skip(1)?;
                reg_ok(s.u8()?, at)?;
                reg_ok(s.u8()?, at)?;
            }
            op::CALL_EXT => {
                reg_ok(s.u8()?, at)?;
                for _ in 0..2 {
                    let a = s.u32()?;
                    if m.atoms.get(a as usize).is_none() {
                        return Err(VerifyError::BadAtom {
                            fn_idx,
                            at,
                            atom: a,
                        });
                    }
                }
                // Target module/arity resolve at runtime (hot loading).
                let nargs = s.u8()?;
                for _ in 0..nargs {
                    reg_ok(s.u8()?, at)?;
                }
            }
            op::BAND | op::BOR | op::BXOR | op::BSL | op::BSR => {
                reg_ok(s.u8()?, at)?;
                reg_ok(s.u8()?, at)?;
                reg_ok(s.u8()?, at)?;
            }
            op::BIN_CAT | op::LIST_CAT => {
                reg_ok(s.u8()?, at)?;
                reg_ok(s.u8()?, at)?;
                reg_ok(s.u8()?, at)?;
            }
            op::BNOT | op::BIN_FROM_LIST | op::BIN_TO_LIST | op::BIN_SIZE | op::BUF_TO_BIN
            | op::BIN_TO_BUF | op::IS_BINARY => {
                reg_ok(s.u8()?, at)?;
                reg_ok(s.u8()?, at)?;
            }
            op::BIN_NEW => {
                reg_ok(s.u8()?, at)?;
                let len = s.u32()? as usize;
                s.skip(len)?;
            }
            op::MAP_NEW => {
                reg_ok(s.u8()?, at)?;
                let n = s.u8()?;
                for _ in 0..n {
                    reg_ok(s.u8()?, at)?;
                    reg_ok(s.u8()?, at)?;
                }
            }
            op::MAP_GET => {
                reg_ok(s.u8()?, at)?;
                reg_ok(s.u8()?, at)?;
                reg_ok(s.u8()?, at)?;
            }
            op::MAP_PUT | op::BIN_PART => {
                reg_ok(s.u8()?, at)?;
                reg_ok(s.u8()?, at)?;
                reg_ok(s.u8()?, at)?;
                reg_ok(s.u8()?, at)?;
            }
            other => {
                return Err(VerifyError::UnknownOpcode {
                    fn_idx,
                    at,
                    opcode: other,
                });
            }
        }
    }

    if fell_through {
        return Err(VerifyError::FallsOffEnd { fn_idx });
    }
    for (at, target) in jump_targets {
        if target < 0 || !boundaries.contains(&(target as usize)) {
            return Err(VerifyError::BadJumpTarget { fn_idx, at });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CodeBuilder;
    use alloc::vec;

    fn module_with(code: alloc::vec::Vec<u8>, nregs: u8) -> Module {
        Module {
            atoms: vec!["main".into(), "x".into()],
            functions: vec![Function {
                name_atom: 0,
                arity: 0,
                nregs,
                code,
            }],
        }
    }

    fn ret0() -> alloc::vec::Vec<u8> {
        let mut b = CodeBuilder::new();
        b.u8(op::LOAD_NIL).u8(0);
        b.u8(op::RET).u8(0);
        b.finish().unwrap()
    }

    #[test]
    fn accepts_valid() {
        assert_eq!(verify(&module_with(ret0(), 1)), Ok(()));
    }

    #[test]
    fn rejects_bad_register() {
        let mut b = CodeBuilder::new();
        b.u8(op::LOAD_NIL).u8(9); // r9 with nregs=1
        b.u8(op::RET).u8(0);
        let m = module_with(b.finish().unwrap(), 1);
        assert!(matches!(
            verify(&m),
            Err(VerifyError::BadRegister { reg: 9, .. })
        ));
    }

    #[test]
    fn rejects_jump_into_middle_of_instruction() {
        let mut b = CodeBuilder::new();
        b.u8(op::JMP).u32(1u32); // lands inside the RET encoding below? offset +1 from next
        b.u8(op::RET).u8(0);
        let m = module_with(b.finish().unwrap(), 1);
        assert!(matches!(verify(&m), Err(VerifyError::BadJumpTarget { .. })));
    }

    #[test]
    fn rejects_fall_off_end() {
        let mut b = CodeBuilder::new();
        b.u8(op::LOAD_NIL).u8(0); // no terminator
        let m = module_with(b.finish().unwrap(), 1);
        assert!(matches!(verify(&m), Err(VerifyError::FallsOffEnd { .. })));
    }

    #[test]
    fn rejects_unknown_opcode_and_truncation() {
        let m = module_with(vec![0xEE], 1);
        assert!(matches!(
            verify(&m),
            Err(VerifyError::UnknownOpcode { opcode: 0xEE, .. })
        ));
        let m = module_with(vec![op::LOAD_INT, 0, 1, 2], 1); // i64 cut short
        assert!(matches!(
            verify(&m),
            Err(VerifyError::TruncatedInstruction { .. })
        ));
    }

    #[test]
    fn rejects_bad_call_arity_and_target() {
        let mut b = CodeBuilder::new();
        b.u8(op::CALL).u8(0).u32(0).u8(2).u8(0).u8(0); // self arity 0, called with 2
        b.u8(op::RET).u8(0);
        let m = module_with(b.finish().unwrap(), 1);
        assert!(matches!(verify(&m), Err(VerifyError::BadCallArity { .. })));

        let mut b = CodeBuilder::new();
        b.u8(op::CALL).u8(0).u32(99).u8(0);
        b.u8(op::RET).u8(0);
        let m = module_with(b.finish().unwrap(), 1);
        assert!(matches!(
            verify(&m),
            Err(VerifyError::BadCallTarget { callee: 99, .. })
        ));
    }

    #[test]
    fn rejects_bad_atom() {
        let mut b = CodeBuilder::new();
        b.u8(op::LOAD_ATOM).u8(0).u32(1000);
        b.u8(op::RET).u8(0);
        let m = module_with(b.finish().unwrap(), 1);
        assert!(matches!(
            verify(&m),
            Err(VerifyError::BadAtom { atom: 1000, .. })
        ));
    }

    /// Budget fuzzer: random mutations of a valid encoded module must never
    /// panic decode or verify (they may of course be rejected).
    #[test]
    fn mutation_fuzz_never_panics() {
        let mut b = CodeBuilder::new();
        b.bind(0);
        b.u8(op::LOAD_INT).u8(1).i64(5);
        b.u8(op::CALL).u8(2).u32(0).u8(0);
        b.u8(op::JMP_IF).u8(2).label_ref(0);
        b.u8(op::EXIT_ATOM).u8(1);
        let m = module_with(b.finish().unwrap(), 3);
        let base = m.encode();

        let mut rng: u64 = 0x9E3779B97F4A7C15;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for _ in 0..50_000 {
            let mut bytes = base.clone();
            for _ in 0..(next() % 4 + 1) {
                let i = (next() as usize) % bytes.len();
                bytes[i] = next() as u8;
            }
            if next() % 4 == 0 {
                bytes.truncate((next() as usize) % (bytes.len() + 1));
            }
            if let Ok(m) = Module::decode(&bytes) {
                let _ = verify(&m);
            }
        }
    }
}
