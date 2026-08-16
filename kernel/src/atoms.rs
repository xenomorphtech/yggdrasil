//! Global atom table. Atoms are interned once and live forever.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use spin::Mutex;
use ygg_term::Term;

static ATOMS: Mutex<AtomTable> = Mutex::new(AtomTable { names: Vec::new(), index: BTreeMap::new() });

struct AtomTable {
    names: Vec<&'static str>,
    index: BTreeMap<&'static str, u32>,
}

pub fn intern(name: &str) -> u32 {
    let mut t = ATOMS.lock();
    if let Some(&i) = t.index.get(name) {
        return i;
    }
    let leaked: &'static str = alloc::boxed::Box::leak(name.into());
    let i = t.names.len() as u32;
    t.names.push(leaked);
    t.index.insert(leaked, i);
    i
}

pub fn name(idx: u32) -> &'static str {
    ATOMS.lock().names.get(idx as usize).copied().unwrap_or("?badatom")
}

pub fn atom(s: &str) -> Term {
    Term::atom(intern(s))
}
