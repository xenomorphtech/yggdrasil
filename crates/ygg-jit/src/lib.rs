//! Tier-1 execution engine: bytecode -> Cranelift -> x86_64 machine code.
//!
//! Terms stay tagged u64 words (i64 in CLIF). Arithmetic and branches compile
//! inline with tag checks (failure jumps to a trap block that exits the
//! process); everything effectful (heap builds, messaging, ports, external
//! calls) calls kernel runtime helpers, which exit the process themselves on
//! error — so generated code never sees a failure sentinel.
//!
//! ABI: every bytecode function becomes `extern "C" fn(arity × u64) -> u64`.
//! Sibling and direct-bound external calls are colocated (PC-rel `call
//! rel32`). Helper calls are PC-rel too when the publisher places code
//! within ±2 GiB of the helpers (the kernel's code zone is sited for this);
//! host loaders keep absolute `movabs`+`call`. Safepoints: a helper call at
//! dynamic call sites and loop back-edges — direct-bound sites need none,
//! since the sealed call graph is acyclic (a cycle would require dynamic
//! dispatch, which keeps its safepoint).
//!
//! Known divergence from the interpreter (documented, benign): integer
//! arithmetic wraps at i61 instead of trapping on overflow.
//!
//! Output is target-independent data: code bytes + relocation list per
//! function. The kernel's publisher lays them out, patches relocs and maps
//! the pages executable.

#![no_std]

extern crate alloc;

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use cranelift_codegen::control::ControlPlane;
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{
    AbiParam, ExtFuncData, ExternalName, FuncRef, InstBuilder, Signature, StackSlotData,
    StackSlotKind, TrapCode, UserExternalName, UserFuncName, types,
};
use cranelift_codegen::isa::{CallConv, OwnedTargetIsa};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::{Context, ir};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use ygg_bytecode::{Module, op};
use ygg_term::Term;

/// Runtime helpers the kernel must provide, indexed by discriminant
/// (`ExternalName` namespace 0). All params/returns are raw u64 terms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Helper {
    SelfPid = 0,     // () -> pid
    Send = 1,        // (to, msg) -> 0
    Recv = 2,        // () -> term
    Spawn = 3,       // (fn_idx, arg) -> pid
    Safepoint = 4,   // ()
    Print = 5,       // (term)
    Eq = 6,          // (a, b) -> 0/1 (raw, untagged)
    MakeTuple = 7,   // (ptr, n) -> term
    GetElem = 8,     // (term, idx) -> term
    Cons = 9,        // (head, tail) -> term
    Head = 10,       // (term) -> term
    Tail = 11,       // (term) -> term
    PortOpen = 12,   // (kind) -> port term
    PortSubmit = 13, // (port, op, arg0, tag) -> 0
    CallExt = 14,    // (module_atom, fname_atom, ptr, n) -> term
    ExitAtom = 15,   // (atom) -> ! (never returns)
    TrapBadarg = 16, // () -> ! (never returns)
    BinConst = 17,   // (ptr, len) -> binary  (ptr into the module's bytecode)
    BinFromList = 18, // (list) -> binary
    BinToList = 19,  // (binary) -> list
    BinSize = 20,    // (binary) -> int term
    BufToBin = 21,   // (int term buffer id) -> binary
    BinToBuf = 22,   // (binary) -> int term buffer id
    MapNew = 23,     // (ptr to k/v pairs, n_pairs) -> map
    MapGet = 24,     // (map, key) -> value
    MapPut = 25,     // (map, key, val) -> map
    IsBinary = 26,   // (term) -> raw 0/1
    BinCat = 27,     // (bin, bin) -> bin
    ListCat = 28,    // (list, term) -> list
    BinPart = 29,    // (bin, off, len) -> bin
    TailCallExt = 30, // (module_atom, fname_atom, ptr, n) — stash only
    TailCallLocal = 31, // (fn_idx, ptr, n) — stash only (same module instance)
    PortSubmit2 = 32, // (port, op, arg0, arg1, tag) -> 0
    BufWrite = 33,    // (buf id term, off term, bin) -> 0
    SleepMs = 34,     // (ms term) — parks the process
    BufNew = 35,      // (size term) -> blob id term
    BufRead = 36,     // (buf id term, off term, len term) -> binary
    Ticks = 37,       // () -> int term (milliseconds)
    ResumeTail = 38,  // (module token) -> term: run the stashed tail chain
    BinAt = 39,       // (binary, idx term) -> int term (no allocation)
}
pub const HELPER_COUNT: usize = 40;

fn helper_nargs(h: u32) -> usize {
    match h {
        x if x == Helper::SelfPid as u32 => 0,
        x if x == Helper::Send as u32 => 2,
        x if x == Helper::Recv as u32 => 0,
        x if x == Helper::Spawn as u32 => 2,
        x if x == Helper::Safepoint as u32 => 0,
        x if x == Helper::Print as u32 => 1,
        x if x == Helper::Eq as u32 => 2,
        x if x == Helper::MakeTuple as u32 => 2,
        x if x == Helper::GetElem as u32 => 2,
        x if x == Helper::Cons as u32 => 2,
        x if x == Helper::Head as u32 => 1,
        x if x == Helper::Tail as u32 => 1,
        x if x == Helper::PortOpen as u32 => 1,
        x if x == Helper::PortSubmit as u32 => 4,
        x if x == Helper::CallExt as u32 => 4,
        x if x == Helper::ExitAtom as u32 => 1,
        x if x == Helper::BinConst as u32 => 2,
        x if x == Helper::BinFromList as u32 => 1,
        x if x == Helper::BinToList as u32 => 1,
        x if x == Helper::BinSize as u32 => 1,
        x if x == Helper::BufToBin as u32 => 1,
        x if x == Helper::BinToBuf as u32 => 1,
        x if x == Helper::MapNew as u32 => 2,
        x if x == Helper::MapGet as u32 => 2,
        x if x == Helper::MapPut as u32 => 3,
        x if x == Helper::IsBinary as u32 => 1,
        x if x == Helper::BinCat as u32 => 2,
        x if x == Helper::ListCat as u32 => 2,
        x if x == Helper::BinPart as u32 => 3,
        x if x == Helper::TailCallExt as u32 => 4,
        x if x == Helper::TailCallLocal as u32 => 3,
        x if x == Helper::PortSubmit2 as u32 => 5,
        x if x == Helper::BufWrite as u32 => 3,
        x if x == Helper::SleepMs as u32 => 1,
        x if x == Helper::BufNew as u32 => 1,
        x if x == Helper::BufRead as u32 => 3,
        x if x == Helper::Ticks as u32 => 0,
        x if x == Helper::ResumeTail as u32 => 1,
        x if x == Helper::BinAt as u32 => 2,
        _ => 0,
    }
}

fn helper_has_ret(h: u32) -> bool {
    !matches!(
        h,
        x if x == Helper::Safepoint as u32
            || x == Helper::Print as u32
            || x == Helper::ExitAtom as u32
            || x == Helper::TrapBadarg as u32
            || x == Helper::TailCallExt as u32
            || x == Helper::TailCallLocal as u32
            || x == Helper::SleepMs as u32
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelocKind {
    Abs8,
    PcRel4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelocTarget {
    Helper(u32),
    Function(u32),
    /// A resolved absolute code address (direct-bound external call).
    Address(u64),
}

#[derive(Debug, Clone)]
pub struct RelocEntry {
    pub offset: u32,
    pub kind: RelocKind,
    pub target: RelocTarget,
    pub addend: i64,
}

/// A statically resolvable external call target: an immutable (sealed)
/// module's already-published function. `addr` is its final executable entry
/// point; `token` is an opaque runtime handle for the callee module, handed
/// to `Helper::ResumeTail` when the callee tail-calls out of the direct call.
#[derive(Debug, Clone, Copy)]
pub struct ExtTarget {
    pub addr: u64,
    pub token: u64,
    pub arity: u8,
    /// Whether the callee can return the tail sentinel (it, or a sibling it
    /// CALLs, ends in a tail call). False lets the call site skip the
    /// sentinel check entirely — the call is then just `call rel32`.
    pub may_tail: bool,
}

/// Resolves `(global module atom, global fname atom)` to a direct-call
/// target, or None to fall back to the dynamic `Helper::CallExt` path.
pub type ExtResolver<'a> = &'a dyn Fn(u32, u32) -> Option<ExtTarget>;

pub struct CompiledFn {
    pub code: Vec<u8>,
    pub relocs: Vec<RelocEntry>,
}

#[derive(Debug)]
pub enum JitError {
    Isa,
    UnsupportedReloc,
    Codegen,
    BadBytecode,
}

fn make_isa() -> Result<OwnedTargetIsa, JitError> {
    let mut fb = settings::builder();
    let _ = fb.set("opt_level", "speed");
    let _ = fb.set("unwind_info", "false");
    let _ = fb.set("is_pic", "false");
    let flags = settings::Flags::new(fb);
    let triple: target_lexicon::Triple =
        "x86_64-unknown-none".parse().map_err(|_| JitError::Isa)?;
    cranelift_codegen::isa::lookup(triple)
        .map_err(|_| JitError::Isa)?
        .finish(flags)
        .map_err(|_| JitError::Isa)
}

fn fn_signature(arity: usize, call_conv: CallConv) -> Signature {
    let mut sig = Signature::new(call_conv);
    for _ in 0..arity {
        sig.params.push(AbiParam::new(types::I64));
    }
    sig.returns.push(AbiParam::new(types::I64));
    sig
}

/// Compile every function of a verified module. `atom_map` maps module-local
/// atom indices to global ones (baked into the code as constants).
pub fn compile_module(m: &Module, atom_map: &[u32]) -> Result<Vec<CompiledFn>, JitError> {
    compile_module_linked(m, atom_map, &|_, _| None, false)
}

/// Like `compile_module`, but external calls the resolver can pin to a
/// published sealed function compile to direct calls instead of the dynamic
/// `Helper::CallExt` dispatch. `pcrel_helpers` emits helper calls as
/// PC-relative `call rel32` — only valid when the publisher guarantees the
/// helpers are within ±2 GiB of the code (the kernel's code zone is placed
/// for exactly this; host loaders must pass false and keep `movabs`).
pub fn compile_module_linked(
    m: &Module,
    atom_map: &[u32],
    resolver: ExtResolver,
    pcrel_helpers: bool,
) -> Result<Vec<CompiledFn>, JitError> {
    let isa = make_isa()?;
    let mut out = Vec::new();
    let mut fb_ctx = FunctionBuilderContext::new();
    for fn_idx in 0..m.functions.len() {
        out.push(compile_fn(m, atom_map, fn_idx, &isa, &mut fb_ctx, resolver, pcrel_helpers)?);
    }
    Ok(out)
}

struct Dec<'a> {
    code: &'a [u8],
    pc: usize,
}

impl<'a> Dec<'a> {
    fn u8(&mut self) -> Result<u8, JitError> {
        let v = *self.code.get(self.pc).ok_or(JitError::BadBytecode)?;
        self.pc += 1;
        Ok(v)
    }
    fn u32(&mut self) -> Result<u32, JitError> {
        let s = self
            .code
            .get(self.pc..self.pc + 4)
            .ok_or(JitError::BadBytecode)?;
        self.pc += 4;
        Ok(u32::from_le_bytes(s.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32, JitError> {
        Ok(self.u32()? as i32)
    }
    fn i64(&mut self) -> Result<i64, JitError> {
        let s = self
            .code
            .get(self.pc..self.pc + 8)
            .ok_or(JitError::BadBytecode)?;
        self.pc += 8;
        Ok(i64::from_le_bytes(s.try_into().unwrap()))
    }
}

/// First pass: block leaders = offset 0, every jump target, and the
/// instruction after every branch.
fn find_leaders(code: &[u8]) -> Result<BTreeSet<usize>, JitError> {
    let mut leaders = BTreeSet::new();
    leaders.insert(0);
    let mut d = Dec { code, pc: 0 };
    while d.pc < code.len() {
        let opcode = d.u8()?;
        match opcode {
            op::NOP => {}
            op::LOAD_INT => {
                d.u8()?;
                d.i64()?;
            }
            op::LOAD_ATOM => {
                d.u8()?;
                d.u32()?;
            }
            op::LOAD_NIL | op::SELF_PID | op::RECV | op::RET | op::EXIT_ATOM | op::PRINT
            | op::TICKS => {
                d.u8()?;
                if opcode == op::RET || opcode == op::EXIT_ATOM {
                    leaders.insert(d.pc);
                }
            }
            op::MOVE | op::HEAD | op::TAIL | op::SEND => {
                d.u8()?;
                d.u8()?;
            }
            op::MAKE_TUPLE => {
                d.u8()?;
                let n = d.u8()?;
                for _ in 0..n {
                    d.u8()?;
                }
            }
            op::GET_ELEM => {
                d.u8()?;
                d.u8()?;
                d.u8()?;
            }
            op::CONS
            | op::ADD
            | op::SUB
            | op::MUL
            | op::CMP_EQ
            | op::CMP_LT
            | op::BAND
            | op::BOR
            | op::BXOR
            | op::BSL
            | op::BSR
            | op::BIN_CAT
            | op::LIST_CAT => {
                d.u8()?;
                d.u8()?;
                d.u8()?;
            }
            op::BNOT
            | op::BIN_FROM_LIST
            | op::BIN_TO_LIST
            | op::BIN_SIZE
            | op::BUF_TO_BIN
            | op::BIN_TO_BUF
            | op::IS_BINARY => {
                d.u8()?;
                d.u8()?;
            }
            op::BIN_NEW => {
                d.u8()?;
                let len = d.u32()? as usize;
                if d.pc + len > d.code.len() {
                    return Err(JitError::BadBytecode);
                }
                d.pc += len;
            }
            op::MAP_NEW => {
                d.u8()?;
                let n = d.u8()?;
                for _ in 0..n {
                    d.u8()?;
                    d.u8()?;
                }
            }
            op::MAP_GET | op::BIN_AT => {
                d.u8()?;
                d.u8()?;
                d.u8()?;
            }
            op::MAP_PUT | op::BIN_PART => {
                d.u8()?;
                d.u8()?;
                d.u8()?;
                d.u8()?;
            }
            op::JMP => {
                let off = d.i32()?;
                leaders.insert((d.pc as i64 + off as i64) as usize);
                leaders.insert(d.pc);
            }
            op::JMP_IF => {
                d.u8()?;
                let off = d.i32()?;
                leaders.insert((d.pc as i64 + off as i64) as usize);
                leaders.insert(d.pc);
            }
            op::CALL => {
                d.u8()?;
                d.u32()?;
                let n = d.u8()?;
                for _ in 0..n {
                    d.u8()?;
                }
            }
            op::SPAWN => {
                d.u8()?;
                d.u32()?;
                d.u8()?;
            }
            op::PORT_OPEN => {
                d.u8()?;
                d.u8()?;
            }
            op::PORT_SUBMIT => {
                d.u8()?;
                d.u8()?;
                d.u8()?;
                d.u8()?;
            }
            op::PORT_SUBMIT2 => {
                for _ in 0..5 {
                    d.u8()?;
                }
            }
            op::BUF_WRITE => {
                for _ in 0..4 {
                    d.u8()?;
                }
            }
            op::SLEEP_MS => {
                d.u8()?;
            }
            op::BUF_NEW => {
                d.u8()?;
                d.u8()?;
            }
            op::BUF_READ => {
                for _ in 0..4 {
                    d.u8()?;
                }
            }
            op::CALL_EXT => {
                d.u8()?;
                d.u32()?;
                d.u32()?;
                let n = d.u8()?;
                for _ in 0..n {
                    d.u8()?;
                }
            }
            op::TAIL_CALL_EXT => {
                d.u32()?;
                d.u32()?;
                let n = d.u8()?;
                for _ in 0..n {
                    d.u8()?;
                }
                leaders.insert(d.pc);
            }
            op::TAIL_CALL => {
                d.u32()?;
                let n = d.u8()?;
                for _ in 0..n {
                    d.u8()?;
                }
                leaders.insert(d.pc);
            }
            _ => return Err(JitError::BadBytecode),
        }
    }
    leaders.remove(&code.len());
    Ok(leaders)
}

struct Tx<'a, 'b> {
    b: FunctionBuilder<'b>,
    m: &'a Module,
    atom_map: &'a [u32],
    call_conv: CallConv,
    regs: Vec<Variable>,
    blocks: BTreeMap<usize, ir::Block>,
    trap_block: ir::Block,
    helpers: BTreeMap<u32, FuncRef>,
    siblings: BTreeMap<u32, FuncRef>,
    /// Direct-bound external callees: resolved address -> imported func.
    /// Namespace 2; the index keys `ext_addrs`, which reloc translation
    /// turns into `RelocTarget::Address`.
    ext_funcs: BTreeMap<u64, FuncRef>,
    ext_addrs: Vec<u64>,
    pcrel_helpers: bool,
}

impl<'a, 'b> Tx<'a, 'b> {
    fn helper(&mut self, h: Helper) -> FuncRef {
        let idx = h as u32;
        if let Some(&r) = self.helpers.get(&idx) {
            return r;
        }
        let name_ref = self
            .b
            .func
            .declare_imported_user_function(UserExternalName {
                namespace: 0,
                index: idx,
            });
        let mut sig = Signature::new(self.call_conv);
        for _ in 0..helper_nargs(idx) {
            sig.params.push(AbiParam::new(types::I64));
        }
        if helper_has_ret(idx) {
            sig.returns.push(AbiParam::new(types::I64));
        }
        let sig = self.b.import_signature(sig);
        let r = self.b.import_function(ExtFuncData {
            name: ExternalName::user(name_ref),
            signature: sig,
            // PC-rel when the publisher keeps helpers within ±2 GiB (the
            // kernel code zone); absolute movabs otherwise (host loaders).
            colocated: self.pcrel_helpers,
            patchable: false,
        });
        self.helpers.insert(idx, r);
        r
    }

    /// Import a direct-bound external callee at a known final address as a
    /// colocated (PC-rel) call target.
    fn ext_target(&mut self, addr: u64, arity: usize) -> FuncRef {
        if let Some(&r) = self.ext_funcs.get(&addr) {
            return r;
        }
        let index = self.ext_addrs.len() as u32;
        self.ext_addrs.push(addr);
        let name_ref = self
            .b
            .func
            .declare_imported_user_function(UserExternalName {
                namespace: 2,
                index,
            });
        let sig = fn_signature(arity, self.call_conv);
        let sig = self.b.import_signature(sig);
        let r = self.b.import_function(ExtFuncData {
            name: ExternalName::user(name_ref),
            signature: sig,
            // Both ends live in the code zone: rel32 always reaches.
            colocated: true,
            patchable: false,
        });
        self.ext_funcs.insert(addr, r);
        r
    }

    fn sibling(&mut self, fn_idx: u32) -> FuncRef {
        if let Some(&r) = self.siblings.get(&fn_idx) {
            return r;
        }
        let name_ref = self
            .b
            .func
            .declare_imported_user_function(UserExternalName {
                namespace: 1,
                index: fn_idx,
            });
        let arity = self.m.functions[fn_idx as usize].arity as usize;
        let sig = fn_signature(arity, self.call_conv);
        let sig = self.b.import_signature(sig);
        let r = self.b.import_function(ExtFuncData {
            name: ExternalName::user(name_ref),
            signature: sig,
            colocated: true,
            patchable: false,
        });
        self.siblings.insert(fn_idx, r);
        r
    }

    fn call_helper(&mut self, h: Helper, args: &[ir::Value]) -> Option<ir::Value> {
        let f = self.helper(h);
        let call = self.b.ins().call(f, args);
        self.b.inst_results(call).first().copied()
    }

    /// Branch to the trap block unless `v` is a small int; returns untagged i64.
    fn expect_int(&mut self, v: ir::Value) -> ir::Value {
        let tag = self.b.ins().band_imm(v, 7);
        let ok = self
            .b
            .ins()
            .icmp_imm(IntCC::Equal, tag, ygg_term::TAG_INT as i64);
        let cont = self.b.create_block();
        self.b.ins().brif(ok, cont, &[], self.trap_block, &[]);
        self.b.switch_to_block(cont);
        self.b.ins().sshr_imm(v, 3)
    }

    fn retag_int(&mut self, raw: ir::Value) -> ir::Value {
        let shifted = self.b.ins().ishl_imm(raw, 3);
        self.b.ins().bor_imm(shifted, ygg_term::TAG_INT as i64)
    }

    fn atom_const(&mut self, local: u32) -> Result<ir::Value, JitError> {
        let global = *self
            .atom_map
            .get(local as usize)
            .ok_or(JitError::BadBytecode)?;
        Ok(self.b.ins().iconst(types::I64, Term::atom(global).0 as i64))
    }

    /// Spill values into a fresh stack slot; returns its address.
    fn spill(&mut self, vals: &[ir::Value]) -> ir::Value {
        let size = (vals.len().max(1) * 8) as u32;
        let ss = self.b.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            size,
            3,
        ));
        for (i, v) in vals.iter().enumerate() {
            self.b.ins().stack_store(types::I64, *v, ss, (i * 8) as i32);
        }
        self.b.ins().stack_addr(types::I64, ss, 0)
    }
}

fn compile_fn(
    m: &Module,
    atom_map: &[u32],
    fn_idx: usize,
    isa: &OwnedTargetIsa,
    fb_ctx: &mut FunctionBuilderContext,
    resolver: ExtResolver,
    pcrel_helpers: bool,
) -> Result<CompiledFn, JitError> {
    let f = &m.functions[fn_idx];
    let call_conv = isa.default_call_conv();
    let mut ctx = Context::new();
    ctx.func.signature = fn_signature(f.arity as usize, call_conv);
    ctx.func.name = UserFuncName::user(1, fn_idx as u32);

    let leaders = find_leaders(&f.code)?;
    let ext_addrs: Vec<u64>;
    {
        let mut b = FunctionBuilder::new(&mut ctx.func, fb_ctx);
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);

        let regs: Vec<Variable> = (0..f.nregs as usize)
            .map(|_| b.declare_var(types::I64))
            .collect();
        let params: Vec<ir::Value> = b.block_params(entry).to_vec();
        let nil = b.ins().iconst(types::I64, Term::NIL.0 as i64);
        for (i, var) in regs.iter().enumerate() {
            match params.get(i) {
                Some(p) => b.def_var(*var, *p),
                None => b.def_var(*var, nil),
            }
        }
        let trap_block = b.create_block();
        let blocks: BTreeMap<usize, ir::Block> =
            leaders.iter().map(|&off| (off, b.create_block())).collect();
        b.ins().jump(blocks[&0], &[]);

        let mut tx = Tx {
            b,
            m,
            atom_map,
            call_conv,
            regs,
            blocks,
            trap_block,
            helpers: BTreeMap::new(),
            siblings: BTreeMap::new(),
            ext_funcs: BTreeMap::new(),
            ext_addrs: Vec::new(),
            pcrel_helpers,
        };

        let mut d = Dec {
            code: &f.code,
            pc: 0,
        };
        let mut filled = true; // entry jump emitted
        let mut in_block = false;
        while d.pc < f.code.len() {
            if let Some(&blk) = tx.blocks.get(&d.pc) {
                if in_block && !filled {
                    tx.b.ins().jump(blk, &[]);
                }
                tx.b.switch_to_block(blk);
                filled = false;
                in_block = true;
            }
            if filled {
                // Unreachable tail after a terminator inside a dead region:
                // decode to advance, emit nothing.
                skip_insn(&mut d)?;
                continue;
            }
            let at_pc = d.pc;
            let opcode = d.u8()?;
            let _ = at_pc;
            match opcode {
                op::NOP => {}
                op::LOAD_INT => {
                    let rd = d.u8()? as usize;
                    let v = d.i64()?;
                    let c = tx.b.ins().iconst(types::I64, Term::int(v).0 as i64);
                    tx.b.def_var(tx.regs[rd], c);
                }
                op::LOAD_ATOM => {
                    let rd = d.u8()? as usize;
                    let a = d.u32()?;
                    let c = tx.atom_const(a)?;
                    tx.b.def_var(tx.regs[rd], c);
                }
                op::LOAD_NIL => {
                    let rd = d.u8()? as usize;
                    let c = tx.b.ins().iconst(types::I64, Term::NIL.0 as i64);
                    tx.b.def_var(tx.regs[rd], c);
                }
                op::MOVE => {
                    let rd = d.u8()? as usize;
                    let rs = d.u8()? as usize;
                    let v = tx.b.use_var(tx.regs[rs]);
                    tx.b.def_var(tx.regs[rd], v);
                }
                op::SELF_PID => {
                    let rd = d.u8()? as usize;
                    let v = tx.call_helper(Helper::SelfPid, &[]).unwrap();
                    tx.b.def_var(tx.regs[rd], v);
                }
                op::TICKS => {
                    let rd = d.u8()? as usize;
                    let v = tx.call_helper(Helper::Ticks, &[]).unwrap();
                    tx.b.def_var(tx.regs[rd], v);
                }
                op::MAKE_TUPLE => {
                    let rd = d.u8()? as usize;
                    let n = d.u8()? as usize;
                    let mut vals = Vec::with_capacity(n);
                    for _ in 0..n {
                        let r = d.u8()? as usize;
                        vals.push(tx.b.use_var(tx.regs[r]));
                    }
                    let ptr = tx.spill(&vals);
                    let nv = tx.b.ins().iconst(types::I64, n as i64);
                    let v = tx.call_helper(Helper::MakeTuple, &[ptr, nv]).unwrap();
                    tx.b.def_var(tx.regs[rd], v);
                }
                op::GET_ELEM => {
                    let rd = d.u8()? as usize;
                    let rs = d.u8()? as usize;
                    let idx = d.u8()? as i64;
                    let t = tx.b.use_var(tx.regs[rs]);
                    let iv = tx.b.ins().iconst(types::I64, idx);
                    let v = tx.call_helper(Helper::GetElem, &[t, iv]).unwrap();
                    tx.b.def_var(tx.regs[rd], v);
                }
                op::CONS => {
                    let rd = d.u8()? as usize;
                    let rh = d.u8()? as usize;
                    let rt = d.u8()? as usize;
                    let h = tx.b.use_var(tx.regs[rh]);
                    let t = tx.b.use_var(tx.regs[rt]);
                    let v = tx.call_helper(Helper::Cons, &[h, t]).unwrap();
                    tx.b.def_var(tx.regs[rd], v);
                }
                op::HEAD | op::TAIL => {
                    let rd = d.u8()? as usize;
                    let rs = d.u8()? as usize;
                    let t = tx.b.use_var(tx.regs[rs]);
                    let h = if opcode == op::HEAD {
                        Helper::Head
                    } else {
                        Helper::Tail
                    };
                    let v = tx.call_helper(h, &[t]).unwrap();
                    tx.b.def_var(tx.regs[rd], v);
                }
                op::ADD | op::SUB | op::MUL => {
                    let rd = d.u8()? as usize;
                    let ra = d.u8()? as usize;
                    let rb = d.u8()? as usize;
                    let a = tx.b.use_var(tx.regs[ra]);
                    let b_ = tx.b.use_var(tx.regs[rb]);
                    let ua = tx.expect_int(a);
                    let ub = tx.expect_int(b_);
                    let raw = match opcode {
                        op::ADD => tx.b.ins().iadd(ua, ub),
                        op::SUB => tx.b.ins().isub(ua, ub),
                        _ => tx.b.ins().imul(ua, ub),
                    };
                    let v = tx.retag_int(raw);
                    tx.b.def_var(tx.regs[rd], v);
                }
                op::CMP_EQ => {
                    let rd = d.u8()? as usize;
                    let ra = d.u8()? as usize;
                    let rb = d.u8()? as usize;
                    let a = tx.b.use_var(tx.regs[ra]);
                    let b_ = tx.b.use_var(tx.regs[rb]);
                    let raw = tx.call_helper(Helper::Eq, &[a, b_]).unwrap();
                    let v = tx.retag_int(raw);
                    tx.b.def_var(tx.regs[rd], v);
                }
                op::CMP_LT => {
                    let rd = d.u8()? as usize;
                    let ra = d.u8()? as usize;
                    let rb = d.u8()? as usize;
                    let a = tx.b.use_var(tx.regs[ra]);
                    let b_ = tx.b.use_var(tx.regs[rb]);
                    let ua = tx.expect_int(a);
                    let ub = tx.expect_int(b_);
                    let c = tx.b.ins().icmp(IntCC::SignedLessThan, ua, ub);
                    let c64 = tx.b.ins().uextend(types::I64, c);
                    let v = tx.retag_int(c64);
                    tx.b.def_var(tx.regs[rd], v);
                }
                op::JMP => {
                    let off = d.i32()?;
                    let target = (d.pc as i64 + off as i64) as usize;
                    if off < 0 {
                        tx.call_helper(Helper::Safepoint, &[]);
                    }
                    let blk = *tx.blocks.get(&target).ok_or(JitError::BadBytecode)?;
                    tx.b.ins().jump(blk, &[]);
                    filled = true;
                }
                op::JMP_IF => {
                    let rc = d.u8()? as usize;
                    let off = d.i32()?;
                    let target = (d.pc as i64 + off as i64) as usize;
                    let fall = d.pc;
                    let v = tx.b.use_var(tx.regs[rc]);
                    let raw = tx.expect_int(v);
                    if off < 0 {
                        tx.call_helper(Helper::Safepoint, &[]);
                    }
                    let cond = tx.b.ins().icmp_imm(IntCC::NotEqual, raw, 0);
                    let tblk = *tx.blocks.get(&target).ok_or(JitError::BadBytecode)?;
                    let fblk = *tx.blocks.get(&fall).ok_or(JitError::BadBytecode)?;
                    tx.b.ins().brif(cond, tblk, &[], fblk, &[]);
                    filled = true;
                }
                op::CALL => {
                    let rd = d.u8()? as usize;
                    let callee = d.u32()?;
                    let n = d.u8()? as usize;
                    let mut args = Vec::with_capacity(n);
                    for _ in 0..n {
                        let r = d.u8()? as usize;
                        args.push(tx.b.use_var(tx.regs[r]));
                    }
                    tx.call_helper(Helper::Safepoint, &[]);
                    let fref = tx.sibling(callee);
                    let call = tx.b.ins().call(fref, &args);
                    let v = tx.b.inst_results(call)[0];
                    // Sentinel propagation: a callee that tail-called out
                    // unwinds this frame too.
                    let is_tail = tx.b.ins().icmp_imm(IntCC::Equal, v, 7);
                    let cont = tx.b.create_block();
                    let unwind = tx.b.create_block();
                    tx.b.ins().brif(is_tail, unwind, &[], cont, &[]);
                    tx.b.switch_to_block(unwind);
                    let sent = tx.b.ins().iconst(types::I64, 7);
                    tx.b.ins().return_(&[sent]);
                    tx.b.switch_to_block(cont);
                    tx.b.def_var(tx.regs[rd], v);
                }
                op::RET => {
                    let rs = d.u8()? as usize;
                    let v = tx.b.use_var(tx.regs[rs]);
                    tx.b.ins().return_(&[v]);
                    filled = true;
                }
                op::SPAWN => {
                    let rd = d.u8()? as usize;
                    let callee = d.u32()?;
                    let ra = d.u8()? as usize;
                    let arg = tx.b.use_var(tx.regs[ra]);
                    let fi = tx.b.ins().iconst(types::I64, callee as i64);
                    let v = tx.call_helper(Helper::Spawn, &[fi, arg]).unwrap();
                    tx.b.def_var(tx.regs[rd], v);
                }
                op::SEND => {
                    let rt = d.u8()? as usize;
                    let rm = d.u8()? as usize;
                    let to = tx.b.use_var(tx.regs[rt]);
                    let msg = tx.b.use_var(tx.regs[rm]);
                    tx.call_helper(Helper::Send, &[to, msg]);
                }
                op::RECV => {
                    let rd = d.u8()? as usize;
                    let v = tx.call_helper(Helper::Recv, &[]).unwrap();
                    tx.b.def_var(tx.regs[rd], v);
                }
                op::PRINT => {
                    let rs = d.u8()? as usize;
                    let v = tx.b.use_var(tx.regs[rs]);
                    tx.call_helper(Helper::Print, &[v]);
                }
                op::EXIT_ATOM => {
                    let rs = d.u8()? as usize;
                    let v = tx.b.use_var(tx.regs[rs]);
                    tx.call_helper(Helper::ExitAtom, &[v]);
                    tx.b.ins().trap(TrapCode::user(1).unwrap());
                    filled = true;
                }
                op::PORT_OPEN => {
                    let rd = d.u8()? as usize;
                    let kind = d.u8()? as i64;
                    let k = tx.b.ins().iconst(types::I64, kind);
                    let v = tx.call_helper(Helper::PortOpen, &[k]).unwrap();
                    tx.b.def_var(tx.regs[rd], v);
                }
                op::PORT_SUBMIT => {
                    let rp = d.u8()? as usize;
                    let o = d.u8()? as i64;
                    let ra = d.u8()? as usize;
                    let rt = d.u8()? as usize;
                    let p = tx.b.use_var(tx.regs[rp]);
                    let ov = tx.b.ins().iconst(types::I64, o);
                    let a0 = tx.b.use_var(tx.regs[ra]);
                    let tg = tx.b.use_var(tx.regs[rt]);
                    tx.call_helper(Helper::PortSubmit, &[p, ov, a0, tg]);
                }
                op::PORT_SUBMIT2 => {
                    let rp = d.u8()? as usize;
                    let ro = d.u8()? as usize;
                    let ra0 = d.u8()? as usize;
                    let ra1 = d.u8()? as usize;
                    let rt = d.u8()? as usize;
                    let p = tx.b.use_var(tx.regs[rp]);
                    let o = tx.b.use_var(tx.regs[ro]);
                    let a0 = tx.b.use_var(tx.regs[ra0]);
                    let a1 = tx.b.use_var(tx.regs[ra1]);
                    let tg = tx.b.use_var(tx.regs[rt]);
                    tx.call_helper(Helper::PortSubmit2, &[p, o, a0, a1, tg]);
                }
                op::BUF_WRITE => {
                    let rd = d.u8()? as usize;
                    let rb = d.u8()? as usize;
                    let ro = d.u8()? as usize;
                    let rs = d.u8()? as usize;
                    let b_ = tx.b.use_var(tx.regs[rb]);
                    let o = tx.b.use_var(tx.regs[ro]);
                    let s = tx.b.use_var(tx.regs[rs]);
                    let v = tx.call_helper(Helper::BufWrite, &[b_, o, s]).unwrap();
                    tx.b.def_var(tx.regs[rd], v);
                }
                op::SLEEP_MS => {
                    let rm = d.u8()? as usize;
                    let m = tx.b.use_var(tx.regs[rm]);
                    tx.call_helper(Helper::Safepoint, &[]);
                    tx.call_helper(Helper::SleepMs, &[m]);
                }
                op::BUF_NEW => {
                    let rd = d.u8()? as usize;
                    let rs = d.u8()? as usize;
                    let s = tx.b.use_var(tx.regs[rs]);
                    let v = tx.call_helper(Helper::BufNew, &[s]).unwrap();
                    tx.b.def_var(tx.regs[rd], v);
                }
                op::BUF_READ => {
                    let rd = d.u8()? as usize;
                    let rb = d.u8()? as usize;
                    let ro = d.u8()? as usize;
                    let rl = d.u8()? as usize;
                    let b_ = tx.b.use_var(tx.regs[rb]);
                    let o = tx.b.use_var(tx.regs[ro]);
                    let l = tx.b.use_var(tx.regs[rl]);
                    let v = tx.call_helper(Helper::BufRead, &[b_, o, l]).unwrap();
                    tx.b.def_var(tx.regs[rd], v);
                }
                op::CALL_EXT => {
                    let rd = d.u8()? as usize;
                    let mlocal = d.u32()?;
                    let flocal = d.u32()?;
                    let n = d.u8()? as usize;
                    let mut args = Vec::with_capacity(n);
                    for _ in 0..n {
                        let r = d.u8()? as usize;
                        args.push(tx.b.use_var(tx.regs[r]));
                    }
                    let gm = *tx.atom_map.get(mlocal as usize).ok_or(JitError::BadBytecode)?;
                    let gf = *tx.atom_map.get(flocal as usize).ok_or(JitError::BadBytecode)?;
                    match resolver(gm, gf) {
                        Some(t) if t.arity as usize == n && n <= 16 => {
                            // Sealed callee at a known final address: a plain
                            // PC-rel call. No safepoint — sealed modules form
                            // an acyclic call graph, so any unbounded work in
                            // the callee crosses its own back-edge/tail-call
                            // safepoints; a cycle would need dynamic dispatch.
                            let fref = tx.ext_target(t.addr, n);
                            let call = tx.b.ins().call(fref, &args);
                            let v = tx.b.inst_results(call)[0];
                            tx.b.def_var(tx.regs[rd], v);
                            // Only callees that can actually tail-call out
                            // need the sentinel check; for the rest the call
                            // really is just `call rel32`.
                            if t.may_tail {
                                let is_sent = tx.b.ins().icmp_imm(IntCC::Equal, v, 7);
                                let resume = tx.b.create_block();
                                let join = tx.b.create_block();
                                tx.b.ins().brif(is_sent, resume, &[], join, &[]);
                                tx.b.switch_to_block(resume);
                                let tok = tx.b.ins().iconst(types::I64, t.token as i64);
                                let v2 = tx.call_helper(Helper::ResumeTail, &[tok]).unwrap();
                                tx.b.def_var(tx.regs[rd], v2);
                                tx.b.ins().jump(join, &[]);
                                tx.b.switch_to_block(join);
                            }
                        }
                        _ => {
                            let ma = tx.atom_const(mlocal)?;
                            let fa = tx.atom_const(flocal)?;
                            let ptr = tx.spill(&args);
                            let nv = tx.b.ins().iconst(types::I64, n as i64);
                            tx.call_helper(Helper::Safepoint, &[]);
                            let v =
                                tx.call_helper(Helper::CallExt, &[ma, fa, ptr, nv]).unwrap();
                            tx.b.def_var(tx.regs[rd], v);
                        }
                    }
                }
                op::TAIL_CALL_EXT => {
                    let mlocal = d.u32()?;
                    let flocal = d.u32()?;
                    let n = d.u8()? as usize;
                    let mut args = Vec::with_capacity(n);
                    for _ in 0..n {
                        let r = d.u8()? as usize;
                        args.push(tx.b.use_var(tx.regs[r]));
                    }
                    let ma = tx.atom_const(mlocal)?;
                    let fa = tx.atom_const(flocal)?;
                    let ptr = tx.spill(&args);
                    let nv = tx.b.ins().iconst(types::I64, n as i64);
                    tx.call_helper(Helper::Safepoint, &[]);
                    tx.call_helper(Helper::TailCallExt, &[ma, fa, ptr, nv]);
                    let sent = tx.b.ins().iconst(types::I64, 7);
                    tx.b.ins().return_(&[sent]);
                    filled = true;
                }
                op::TAIL_CALL => {
                    let callee = d.u32()?;
                    let n = d.u8()? as usize;
                    let mut args = Vec::with_capacity(n);
                    for _ in 0..n {
                        let r = d.u8()? as usize;
                        args.push(tx.b.use_var(tx.regs[r]));
                    }
                    let fi = tx.b.ins().iconst(types::I64, callee as i64);
                    let ptr = tx.spill(&args);
                    let nv = tx.b.ins().iconst(types::I64, n as i64);
                    tx.call_helper(Helper::Safepoint, &[]);
                    tx.call_helper(Helper::TailCallLocal, &[fi, ptr, nv]);
                    let sent = tx.b.ins().iconst(types::I64, 7);
                    tx.b.ins().return_(&[sent]);
                    filled = true;
                }
                op::BAND | op::BOR | op::BXOR => {
                    let rd = d.u8()? as usize;
                    let ra = d.u8()? as usize;
                    let rb = d.u8()? as usize;
                    let a = tx.b.use_var(tx.regs[ra]);
                    let b_ = tx.b.use_var(tx.regs[rb]);
                    let ua = tx.expect_int(a);
                    let ub = tx.expect_int(b_);
                    // Bitops on sign-extended i61 values stay valid i61.
                    let raw = match opcode {
                        op::BAND => tx.b.ins().band(ua, ub),
                        op::BOR => tx.b.ins().bor(ua, ub),
                        _ => tx.b.ins().bxor(ua, ub),
                    };
                    let v = tx.retag_int(raw);
                    tx.b.def_var(tx.regs[rd], v);
                }
                op::BSL | op::BSR => {
                    let rd = d.u8()? as usize;
                    let ra = d.u8()? as usize;
                    let rb = d.u8()? as usize;
                    let a = tx.b.use_var(tx.regs[ra]);
                    let b_ = tx.b.use_var(tx.regs[rb]);
                    let ua = tx.expect_int(a);
                    let ub = tx.expect_int(b_);
                    // Shift amount must be 0..=60 (i61 world), else trap.
                    let in_range =
                        tx.b.ins()
                            .icmp_imm(IntCC::UnsignedLessThanOrEqual, ub, 60);
                    let cont = tx.b.create_block();
                    tx.b.ins().brif(in_range, cont, &[], tx.trap_block, &[]);
                    tx.b.switch_to_block(cont);
                    let raw = if opcode == op::BSL {
                        tx.b.ins().ishl(ua, ub)
                    } else {
                        tx.b.ins().sshr(ua, ub)
                    };
                    let v = tx.retag_int(raw);
                    tx.b.def_var(tx.regs[rd], v);
                }
                op::BNOT => {
                    let rd = d.u8()? as usize;
                    let rs = d.u8()? as usize;
                    let a = tx.b.use_var(tx.regs[rs]);
                    let ua = tx.expect_int(a);
                    let raw = tx.b.ins().bnot(ua);
                    let v = tx.retag_int(raw);
                    tx.b.def_var(tx.regs[rd], v);
                }
                op::BIN_NEW => {
                    let rd = d.u8()? as usize;
                    let len = d.u32()? as usize;
                    if d.pc + len > d.code.len() {
                        return Err(JitError::BadBytecode);
                    }
                    // The constant bytes live inside the module's bytecode,
                    // which the kernel keeps alive as long as this code runs.
                    let ptr = d.code[d.pc..].as_ptr() as i64;
                    d.pc += len;
                    let pv = tx.b.ins().iconst(types::I64, ptr);
                    let lv = tx.b.ins().iconst(types::I64, len as i64);
                    let v = tx.call_helper(Helper::BinConst, &[pv, lv]).unwrap();
                    tx.b.def_var(tx.regs[rd], v);
                }
                op::MAP_NEW => {
                    let rd = d.u8()? as usize;
                    let n = d.u8()? as usize;
                    let mut vals = Vec::with_capacity(2 * n);
                    for _ in 0..n {
                        let rk = d.u8()? as usize;
                        let rv = d.u8()? as usize;
                        vals.push(tx.b.use_var(tx.regs[rk]));
                        vals.push(tx.b.use_var(tx.regs[rv]));
                    }
                    let ptr = tx.spill(&vals);
                    let nv = tx.b.ins().iconst(types::I64, n as i64);
                    let v = tx.call_helper(Helper::MapNew, &[ptr, nv]).unwrap();
                    tx.b.def_var(tx.regs[rd], v);
                }
                op::MAP_GET => {
                    let rd = d.u8()? as usize;
                    let rm = d.u8()? as usize;
                    let rk = d.u8()? as usize;
                    let m = tx.b.use_var(tx.regs[rm]);
                    let k = tx.b.use_var(tx.regs[rk]);
                    let v = tx.call_helper(Helper::MapGet, &[m, k]).unwrap();
                    tx.b.def_var(tx.regs[rd], v);
                }
                op::MAP_PUT => {
                    let rd = d.u8()? as usize;
                    let rm = d.u8()? as usize;
                    let rk = d.u8()? as usize;
                    let rv = d.u8()? as usize;
                    let m = tx.b.use_var(tx.regs[rm]);
                    let k = tx.b.use_var(tx.regs[rk]);
                    let v0 = tx.b.use_var(tx.regs[rv]);
                    let v = tx.call_helper(Helper::MapPut, &[m, k, v0]).unwrap();
                    tx.b.def_var(tx.regs[rd], v);
                }
                op::BIN_FROM_LIST | op::BIN_TO_LIST | op::BIN_SIZE | op::BUF_TO_BIN
                | op::BIN_TO_BUF => {
                    let rd = d.u8()? as usize;
                    let rs = d.u8()? as usize;
                    let a = tx.b.use_var(tx.regs[rs]);
                    let h = match opcode {
                        op::BIN_FROM_LIST => Helper::BinFromList,
                        op::BIN_TO_LIST => Helper::BinToList,
                        op::BIN_SIZE => Helper::BinSize,
                        op::BUF_TO_BIN => Helper::BufToBin,
                        _ => Helper::BinToBuf,
                    };
                    let v = tx.call_helper(h, &[a]).unwrap();
                    tx.b.def_var(tx.regs[rd], v);
                }
                op::BIN_AT => {
                    let rd = d.u8()? as usize;
                    let rb = d.u8()? as usize;
                    let ri = d.u8()? as usize;
                    let b_ = tx.b.use_var(tx.regs[rb]);
                    let i = tx.b.use_var(tx.regs[ri]);
                    let v = tx.call_helper(Helper::BinAt, &[b_, i]).unwrap();
                    tx.b.def_var(tx.regs[rd], v);
                }
                op::BIN_PART => {
                    let rd = d.u8()? as usize;
                    let rb = d.u8()? as usize;
                    let ro = d.u8()? as usize;
                    let rl = d.u8()? as usize;
                    let b_ = tx.b.use_var(tx.regs[rb]);
                    let o = tx.b.use_var(tx.regs[ro]);
                    let l = tx.b.use_var(tx.regs[rl]);
                    let v = tx.call_helper(Helper::BinPart, &[b_, o, l]).unwrap();
                    tx.b.def_var(tx.regs[rd], v);
                }
                op::IS_BINARY => {
                    let rd = d.u8()? as usize;
                    let rs = d.u8()? as usize;
                    let a = tx.b.use_var(tx.regs[rs]);
                    let raw = tx.call_helper(Helper::IsBinary, &[a]).unwrap();
                    let v = tx.retag_int(raw);
                    tx.b.def_var(tx.regs[rd], v);
                }
                op::BIN_CAT | op::LIST_CAT => {
                    let rd = d.u8()? as usize;
                    let ra = d.u8()? as usize;
                    let rb = d.u8()? as usize;
                    let a = tx.b.use_var(tx.regs[ra]);
                    let b_ = tx.b.use_var(tx.regs[rb]);
                    let h = if opcode == op::BIN_CAT { Helper::BinCat } else { Helper::ListCat };
                    let v = tx.call_helper(h, &[a, b_]).unwrap();
                    tx.b.def_var(tx.regs[rd], v);
                }
                _ => return Err(JitError::BadBytecode),
            }
        }

        // Trap block: report badarg and never come back.
        tx.b.switch_to_block(tx.trap_block);
        tx.call_helper(Helper::TrapBadarg, &[]);
        tx.b.ins().trap(TrapCode::user(1).unwrap());

        tx.b.seal_all_blocks();
        ext_addrs = core::mem::take(&mut tx.ext_addrs);
        tx.b.finalize(isa.frontend_config());
    }

    let (code, raw_relocs) = {
        let compiled = ctx
            .compile(isa.as_ref(), &mut ControlPlane::default())
            .map_err(|_| JitError::Codegen)?;
        let code = compiled.buffer.data().to_vec();
        let raw: Vec<_> = compiled
            .buffer
            .relocs()
            .iter()
            .map(|r| (r.offset, r.kind, r.target.clone(), r.addend))
            .collect();
        (code, raw)
    };
    let mut relocs = Vec::new();
    for (offset, rkind, rtarget, addend) in raw_relocs {
        let kind = match rkind {
            cranelift_codegen::binemit::Reloc::Abs8 => RelocKind::Abs8,
            cranelift_codegen::binemit::Reloc::X86CallPCRel4 => RelocKind::PcRel4,
            _ => return Err(JitError::UnsupportedReloc),
        };
        let target = match rtarget {
            cranelift_codegen::FinalizedRelocTarget::ExternalName(ExternalName::User(nref)) => {
                let uen = &ctx.func.params.user_named_funcs()[nref];
                match uen.namespace {
                    0 => RelocTarget::Helper(uen.index),
                    1 => RelocTarget::Function(uen.index),
                    2 => RelocTarget::Address(
                        *ext_addrs
                            .get(uen.index as usize)
                            .ok_or(JitError::UnsupportedReloc)?,
                    ),
                    _ => return Err(JitError::UnsupportedReloc),
                }
            }
            _ => return Err(JitError::UnsupportedReloc),
        };
        relocs.push(RelocEntry {
            offset,
            kind,
            target,
            addend,
        });
    }
    Ok(CompiledFn { code, relocs })
}

fn skip_insn(d: &mut Dec) -> Result<(), JitError> {
    let opcode = d.u8()?;
    match opcode {
        op::NOP => {}
        op::LOAD_INT => {
            d.u8()?;
            d.i64()?;
        }
        op::LOAD_ATOM => {
            d.u8()?;
            d.u32()?;
        }
        op::LOAD_NIL | op::SELF_PID | op::RECV | op::RET | op::EXIT_ATOM | op::PRINT
        | op::TICKS => {
            d.u8()?;
        }
        op::MOVE | op::HEAD | op::TAIL | op::SEND => {
            d.u8()?;
            d.u8()?;
        }
        op::MAKE_TUPLE => {
            d.u8()?;
            let n = d.u8()?;
            for _ in 0..n {
                d.u8()?;
            }
        }
        op::GET_ELEM
        | op::CONS
        | op::ADD
        | op::SUB
        | op::MUL
        | op::CMP_EQ
        | op::CMP_LT
        | op::BAND
        | op::BOR
        | op::BXOR
        | op::BSL
        | op::BSR
        | op::BIN_CAT
        | op::LIST_CAT => {
            d.u8()?;
            d.u8()?;
            d.u8()?;
        }
        op::BNOT
        | op::BIN_FROM_LIST
        | op::BIN_TO_LIST
        | op::BIN_SIZE
        | op::BUF_TO_BIN
        | op::BIN_TO_BUF
        | op::IS_BINARY => {
            d.u8()?;
            d.u8()?;
        }
        op::BIN_NEW => {
            d.u8()?;
            let len = d.u32()? as usize;
            if d.pc + len > d.code.len() {
                return Err(JitError::BadBytecode);
            }
            d.pc += len;
        }
        op::MAP_NEW => {
            d.u8()?;
            let n = d.u8()?;
            for _ in 0..n {
                d.u8()?;
                d.u8()?;
            }
        }
        op::MAP_GET | op::BIN_AT => {
            d.u8()?;
            d.u8()?;
            d.u8()?;
        }
        op::MAP_PUT | op::BIN_PART => {
            d.u8()?;
            d.u8()?;
            d.u8()?;
            d.u8()?;
        }
        op::JMP => {
            d.i32()?;
        }
        op::JMP_IF => {
            d.u8()?;
            d.i32()?;
        }
        op::CALL => {
            d.u8()?;
            d.u32()?;
            let n = d.u8()?;
            for _ in 0..n {
                d.u8()?;
            }
        }
        op::SPAWN => {
            d.u8()?;
            d.u32()?;
            d.u8()?;
        }
        op::PORT_OPEN => {
            d.u8()?;
            d.u8()?;
        }
        op::PORT_SUBMIT => {
            d.u8()?;
            d.u8()?;
            d.u8()?;
            d.u8()?;
        }
        op::PORT_SUBMIT2 => {
            for _ in 0..5 {
                d.u8()?;
            }
        }
        op::BUF_WRITE => {
            for _ in 0..4 {
                d.u8()?;
            }
        }
        op::SLEEP_MS => {
            d.u8()?;
        }
        op::BUF_NEW => {
            d.u8()?;
            d.u8()?;
        }
        op::BUF_READ => {
            for _ in 0..4 {
                d.u8()?;
            }
        }
        op::CALL_EXT => {
            d.u8()?;
            d.u32()?;
            d.u32()?;
            let n = d.u8()?;
            for _ in 0..n {
                d.u8()?;
            }
        }
        op::TAIL_CALL_EXT => {
            d.u32()?;
            d.u32()?;
            let n = d.u8()?;
            for _ in 0..n {
                d.u8()?;
            }
        }
        op::TAIL_CALL => {
            d.u32()?;
            let n = d.u8()?;
            for _ in 0..n {
                d.u8()?;
            }
        }
        _ => return Err(JitError::BadBytecode),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::sync::Mutex as StdMutex;
    use std::vec;
    use ygg_bytecode::{CodeBuilder, Function};

    /// mmap an RWX buffer, copy the functions in, patch relocs, return fn addrs.
    fn load_exec(fns: &[CompiledFn], helpers: &[u64; HELPER_COUNT]) -> Vec<u64> {
        let mut offsets = Vec::new();
        let mut total = 0usize;
        for f in fns {
            total = (total + 15) & !15;
            offsets.push(total);
            total += f.code.len();
        }
        let size = total.max(1).next_multiple_of(4096);
        let base = unsafe {
            libc::mmap(
                core::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        } as *mut u8;
        assert!(!base.is_null() && base as isize != -1);
        let fn_addrs: Vec<u64> = offsets.iter().map(|o| base as u64 + *o as u64).collect();
        unsafe {
            for (f, off) in fns.iter().zip(&offsets) {
                core::ptr::copy_nonoverlapping(f.code.as_ptr(), base.add(*off), f.code.len());
            }
            for (f, off) in fns.iter().zip(&offsets) {
                for r in &f.relocs {
                    let target = match r.target {
                        RelocTarget::Helper(h) => helpers[h as usize],
                        RelocTarget::Function(i) => fn_addrs[i as usize],
                        RelocTarget::Address(a) => a,
                    };
                    let at = base.add(off + r.offset as usize);
                    match r.kind {
                        RelocKind::Abs8 => at
                            .cast::<u64>()
                            .write_unaligned((target as i64 + r.addend) as u64),
                        RelocKind::PcRel4 => {
                            let rel = target as i64 + r.addend - at as i64;
                            at.cast::<i32>().write_unaligned(rel as i32);
                        }
                    }
                }
            }
        }
        fn_addrs
    }

    static PRINTED: StdMutex<Vec<u64>> = StdMutex::new(Vec::new());

    extern "C" fn stub_safepoint() {}
    extern "C" fn stub_print(t: u64) {
        PRINTED.lock().unwrap().push(t);
    }
    extern "C" fn stub_eq(a: u64, b: u64) -> u64 {
        (a == b) as u64
    }
    extern "C" fn stub_self() -> u64 {
        Term::pid(9).0
    }

    fn helpers() -> [u64; HELPER_COUNT] {
        let mut t = [0u64; HELPER_COUNT];
        t[Helper::Safepoint as usize] = stub_safepoint as usize as u64;
        t[Helper::Print as usize] = stub_print as usize as u64;
        t[Helper::Eq as usize] = stub_eq as usize as u64;
        t[Helper::SelfPid as usize] = stub_self as usize as u64;
        t
    }

    fn one_fn_module(code: Vec<u8>, arity: u8, nregs: u8) -> Module {
        Module {
            atoms: vec!["main".into(), "extra".into()],
            functions: vec![Function {
                name_atom: 0,
                arity,
                nregs,
                code,
            }],
        }
    }

    #[test]
    fn const_return_executes() {
        let mut b = CodeBuilder::new();
        b.u8(op::LOAD_INT).u8(0).i64(42);
        b.u8(op::RET).u8(0);
        let m = one_fn_module(b.finish().unwrap(), 0, 1);
        let fns = compile_module(&m, &[100, 101]).unwrap();
        let addrs = load_exec(&fns, &helpers());
        let f: extern "C" fn() -> u64 = unsafe { core::mem::transmute(addrs[0]) };
        assert_eq!(Term(f()).as_int(), Some(42));
    }

    #[test]
    fn arith_branch_loop_executes() {
        // sum 1..=n via loop with back-edge (safepoint helper) and JMP_IF.
        let mut b = CodeBuilder::new();
        // r0 = n; r1 = acc = 0; r2 = one
        b.u8(op::LOAD_INT).u8(1).i64(0);
        b.u8(op::LOAD_INT).u8(2).i64(1);
        b.bind(0); // loop
        b.u8(op::LOAD_INT).u8(3).i64(1);
        b.u8(op::CMP_LT).u8(4).u8(0).u8(3); // n < 1 ?
        b.u8(op::JMP_IF).u8(4).label_ref(1);
        b.u8(op::ADD).u8(1).u8(1).u8(0); // acc += n
        b.u8(op::SUB).u8(0).u8(0).u8(2); // n -= 1
        b.u8(op::JMP).label_ref(0);
        b.bind(1);
        b.u8(op::RET).u8(1);
        let m = one_fn_module(b.finish().unwrap(), 1, 5);
        let fns = compile_module(&m, &[100, 101]).unwrap();
        let addrs = load_exec(&fns, &helpers());
        let f: extern "C" fn(u64) -> u64 = unsafe { core::mem::transmute(addrs[0]) };
        assert_eq!(Term(f(Term::int(100).0)).as_int(), Some(5050));
    }

    #[test]
    fn sibling_call_recursion_executes() {
        // fact(n) with self-call (PcRel4 sibling reloc).
        let mut b = CodeBuilder::new();
        b.u8(op::LOAD_INT).u8(1).i64(2);
        b.u8(op::CMP_LT).u8(2).u8(0).u8(1);
        b.u8(op::JMP_IF).u8(2).label_ref(0);
        b.u8(op::LOAD_INT).u8(1).i64(1);
        b.u8(op::SUB).u8(3).u8(0).u8(1);
        b.u8(op::CALL).u8(4).u32(0).u8(1).u8(3);
        b.u8(op::MUL).u8(5).u8(0).u8(4);
        b.u8(op::RET).u8(5);
        b.bind(0);
        b.u8(op::LOAD_INT).u8(5).i64(1);
        b.u8(op::RET).u8(5);
        let m = one_fn_module(b.finish().unwrap(), 1, 6);
        let fns = compile_module(&m, &[100, 101]).unwrap();
        let addrs = load_exec(&fns, &helpers());
        let f: extern "C" fn(u64) -> u64 = unsafe { core::mem::transmute(addrs[0]) };
        assert_eq!(Term(f(Term::int(10).0)).as_int(), Some(3628800));
    }

    #[test]
    fn bit_ops_execute() {
        let mut b = CodeBuilder::new();
        b.u8(op::LOAD_INT).u8(1).i64(0xFF);
        b.u8(op::BAND).u8(2).u8(0).u8(1); // x & 0xFF
        b.u8(op::LOAD_INT).u8(1).i64(0x100);
        b.u8(op::BOR).u8(2).u8(2).u8(1); // | 0x100
        b.u8(op::LOAD_INT).u8(1).i64(4);
        b.u8(op::LOAD_INT).u8(3).i64(1);
        b.u8(op::BSL).u8(3).u8(3).u8(1); // 1 << 4
        b.u8(op::BXOR).u8(2).u8(2).u8(3);
        b.u8(op::LOAD_INT).u8(1).i64(2);
        b.u8(op::BSR).u8(2).u8(2).u8(1); // >> 2
        b.u8(op::BNOT).u8(4).u8(2);
        b.u8(op::BNOT).u8(4).u8(4); // double-not = identity
        b.u8(op::RET).u8(4);
        let m = one_fn_module(b.finish().unwrap(), 1, 5);
        let fns = compile_module(&m, &[100, 101]).unwrap();
        let addrs = load_exec(&fns, &helpers());
        let f: extern "C" fn(u64) -> u64 = unsafe { core::mem::transmute(addrs[0]) };
        let x: i64 = 0x3AB;
        let expect = (((x & 0xFF) | 0x100) ^ (1 << 4)) >> 2;
        assert_eq!(Term(f(Term::int(x).0)).as_int(), Some(expect));
    }

    #[test]
    fn helper_calls_execute() {
        // print(self()), print(atom), eq check, return eq result
        let mut b = CodeBuilder::new();
        b.u8(op::SELF_PID).u8(0);
        b.u8(op::PRINT).u8(0);
        b.u8(op::LOAD_ATOM).u8(1).u32(1); // local atom 1 -> global 101
        b.u8(op::PRINT).u8(1);
        b.u8(op::CMP_EQ).u8(2).u8(0).u8(0);
        b.u8(op::RET).u8(2);
        let m = one_fn_module(b.finish().unwrap(), 0, 3);
        let fns = compile_module(&m, &[100, 101]).unwrap();
        let addrs = load_exec(&fns, &helpers());
        PRINTED.lock().unwrap().clear();
        let f: extern "C" fn() -> u64 = unsafe { core::mem::transmute(addrs[0]) };
        let r = f();
        assert_eq!(Term(r).as_int(), Some(1));
        let printed = PRINTED.lock().unwrap().clone();
        assert_eq!(printed, vec![Term::pid(9).0, Term::atom(101).0]);
    }
}
