//! ygg-asm: text assembly -> .yggm bytecode module.
//!
//! Syntax (line-based, `;` comments):
//! ```text
//! module <name>
//! fn <name>/<arity> regs=<n> {
//!     load_int r1, 42
//!     load_atom r2, some_atom
//!     make_tuple r3, r1, r2
//!     call r4, helper/1, r3
//! lbl:
//!     jmp lbl
//! }
//! ```
//! Registers are `rN`. Call/spawn targets are `name/arity`. Jump targets are
//! labels bound with `name:`.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use ygg_bytecode::{CodeBuilder, Function, Module, op};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let [_, input, output] = &args[..] else {
        bail!("usage: ygg-asm <input.yasm> <output.yggm>");
    };
    let src = std::fs::read_to_string(input).with_context(|| format!("reading {input}"))?;
    let module = assemble(&src).with_context(|| format!("assembling {input}"))?;
    std::fs::write(output, module.encode()).with_context(|| format!("writing {output}"))?;
    println!(
        "ygg-asm: {} -> {} ({} fns, {} atoms)",
        input,
        output,
        module.functions.len(),
        module.atoms.len()
    );
    Ok(())
}

#[derive(Default)]
struct Atoms {
    names: Vec<String>,
    index: HashMap<String, u32>,
}

impl Atoms {
    fn intern(&mut self, s: &str) -> u32 {
        if let Some(&i) = self.index.get(s) {
            return i;
        }
        let i = self.names.len() as u32;
        self.names.push(s.to_string());
        self.index.insert(s.to_string(), i);
        i
    }
}

struct FnDecl {
    name: String,
    arity: u8,
    nregs: u8,
    body: Vec<(usize, String)>, // (line number, text)
}

fn assemble(src: &str) -> Result<Module> {
    let mut atoms = Atoms::default();
    let mut decls: Vec<FnDecl> = Vec::new();
    let mut cur: Option<FnDecl> = None;

    for (ln, raw) in src.lines().enumerate() {
        let line = raw.split(';').next().unwrap().trim();
        if line.is_empty() || line.starts_with("module ") {
            continue;
        }
        if let Some(rest) = line.strip_prefix("fn ") {
            if cur.is_some() {
                bail!("line {}: nested fn", ln + 1);
            }
            // "name/arity regs=N {"
            let rest = rest.trim_end_matches('{').trim();
            let (sig, regs) = rest.split_once(" ").context("fn needs `regs=N`")?;
            let (name, arity) = sig.split_once('/').context("fn name needs /arity")?;
            let nregs: u8 = regs.strip_prefix("regs=").context("expected regs=N")?.parse()?;
            cur = Some(FnDecl {
                name: name.to_string(),
                arity: arity.parse()?,
                nregs,
                body: Vec::new(),
            });
        } else if line == "}" {
            decls.push(cur.take().context("stray `}`")?);
        } else if let Some(f) = &mut cur {
            f.body.push((ln + 1, line.to_string()));
        } else {
            bail!("line {}: code outside fn: {line}", ln + 1);
        }
    }
    if cur.is_some() {
        bail!("unclosed fn");
    }

    // Function name -> (index, arity) for call/spawn resolution.
    let mut fn_index: HashMap<String, (u32, u8)> = HashMap::new();
    for (i, d) in decls.iter().enumerate() {
        let key = format!("{}/{}", d.name, d.arity);
        if fn_index.insert(key.clone(), (i as u32, d.arity)).is_some() {
            bail!("duplicate function {key}");
        }
    }

    let mut functions = Vec::new();
    for d in &decls {
        let code = assemble_fn(d, &fn_index, &mut atoms)
            .with_context(|| format!("in fn {}/{}", d.name, d.arity))?;
        functions.push(Function {
            name_atom: atoms.intern(&d.name),
            arity: d.arity,
            nregs: d.nregs,
            code,
        });
    }
    Ok(Module { atoms: atoms.names, functions })
}

fn assemble_fn(
    d: &FnDecl,
    fns: &HashMap<String, (u32, u8)>,
    atoms: &mut Atoms,
) -> Result<Vec<u8>> {
    let mut b = CodeBuilder::new();
    let mut labels: HashMap<String, u32> = HashMap::new();
    let mut next_label = 0u32;
    let mut label_id = |name: &str, labels: &mut HashMap<String, u32>| {
        *labels.entry(name.to_string()).or_insert_with(|| {
            next_label += 1;
            next_label - 1
        })
    };

    for (ln, line) in &d.body {
        if let Some(name) = line.strip_suffix(':') {
            let id = label_id(name, &mut labels);
            b.bind(id);
            continue;
        }
        let (mnem, rest) = line.split_once(char::is_whitespace).unwrap_or((line.as_str(), ""));
        let ops: Vec<&str> = rest.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
        let reg = |s: &str| -> Result<u8> {
            s.strip_prefix('r')
                .and_then(|n| n.parse().ok())
                .with_context(|| format!("line {ln}: bad register {s:?}"))
        };
        let fnref = |s: &str| -> Result<u32> {
            fns.get(s).map(|(i, _)| *i).with_context(|| format!("line {ln}: unknown fn {s:?}"))
        };
        match mnem {
            "nop" => {
                b.u8(op::NOP);
            }
            "load_int" => {
                b.u8(op::LOAD_INT).u8(reg(ops[0])?).i64(ops[1].parse()?);
            }
            "load_atom" => {
                let a = atoms.intern(ops[1]);
                b.u8(op::LOAD_ATOM).u8(reg(ops[0])?).u32(a);
            }
            "load_nil" => {
                b.u8(op::LOAD_NIL).u8(reg(ops[0])?);
            }
            "move" => {
                b.u8(op::MOVE).u8(reg(ops[0])?).u8(reg(ops[1])?);
            }
            "self" => {
                b.u8(op::SELF_PID).u8(reg(ops[0])?);
            }
            "make_tuple" => {
                b.u8(op::MAKE_TUPLE).u8(reg(ops[0])?).u8((ops.len() - 1) as u8);
                for r in &ops[1..] {
                    b.u8(reg(r)?);
                }
            }
            "get_elem" => {
                b.u8(op::GET_ELEM).u8(reg(ops[0])?).u8(reg(ops[1])?).u8(ops[2].parse()?);
            }
            "cons" => {
                b.u8(op::CONS).u8(reg(ops[0])?).u8(reg(ops[1])?).u8(reg(ops[2])?);
            }
            "head" => {
                b.u8(op::HEAD).u8(reg(ops[0])?).u8(reg(ops[1])?);
            }
            "tail" => {
                b.u8(op::TAIL).u8(reg(ops[0])?).u8(reg(ops[1])?);
            }
            "add" | "sub" | "mul" | "eq" | "lt" => {
                let o = match mnem {
                    "add" => op::ADD,
                    "sub" => op::SUB,
                    "mul" => op::MUL,
                    "eq" => op::CMP_EQ,
                    _ => op::CMP_LT,
                };
                b.u8(o).u8(reg(ops[0])?).u8(reg(ops[1])?).u8(reg(ops[2])?);
            }
            "jmp" => {
                let id = label_id(ops[0], &mut labels);
                b.u8(op::JMP).label_ref(id);
            }
            "jmp_if" => {
                let id = label_id(ops[1], &mut labels);
                b.u8(op::JMP_IF).u8(reg(ops[0])?).label_ref(id);
            }
            "call" => {
                b.u8(op::CALL).u8(reg(ops[0])?).u32(fnref(ops[1])?).u8((ops.len() - 2) as u8);
                for r in &ops[2..] {
                    b.u8(reg(r)?);
                }
            }
            "ret" => {
                b.u8(op::RET).u8(reg(ops[0])?);
            }
            "spawn" => {
                b.u8(op::SPAWN).u8(reg(ops[0])?).u32(fnref(ops[1])?).u8(reg(ops[2])?);
            }
            "send" => {
                b.u8(op::SEND).u8(reg(ops[0])?).u8(reg(ops[1])?);
            }
            "recv" => {
                b.u8(op::RECV).u8(reg(ops[0])?);
            }
            "print" => {
                b.u8(op::PRINT).u8(reg(ops[0])?);
            }
            "exit_atom" => {
                b.u8(op::EXIT_ATOM).u8(reg(ops[0])?);
            }
            "port_open" => {
                b.u8(op::PORT_OPEN).u8(reg(ops[0])?).u8(ops[1].parse()?);
            }
            "call_ext" => {
                // call_ext rd, module, fname, rArg...
                let m = atoms.intern(ops[1]);
                let f2 = atoms.intern(ops[2]);
                b.u8(op::CALL_EXT).u8(reg(ops[0])?).u32(m).u32(f2).u8((ops.len() - 3) as u8);
                for r in &ops[3..] {
                    b.u8(reg(r)?);
                }
            }
            "port_submit" => {
                b.u8(op::PORT_SUBMIT)
                    .u8(reg(ops[0])?)
                    .u8(ops[1].parse()?)
                    .u8(reg(ops[2])?)
                    .u8(reg(ops[3])?);
            }
            _ => bail!("line {ln}: unknown mnemonic {mnem:?}"),
        }
    }
    b.finish().map_err(|l| anyhow::anyhow!("unbound label id {l}"))
}
