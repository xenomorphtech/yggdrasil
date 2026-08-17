//! The port runtime: SQ/CQ ring pairs owned by processes, plus the serial
//! port driver.
//!
//! Submission: the owner pushes an SQE and the driver's submit hook runs
//! immediately (synchronous ops complete right there; async ops park until
//! device interrupts).
//!
//! Completion: drivers push CQEs; `pump()` — the "softirq-lite" bottom half,
//! called from the scheduler loop — drains CQs into owner mailboxes as
//! `{port_reply, Port, Tag, Result}` messages. IRQ top-halves only stash
//! bytes in an SPSC ring and set `PENDING`.

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use spin::Mutex;
use ygg_rings::{Cqe, Ring, SKIP_CQE, Sqe};
use ygg_term::Term;

use crate::atoms;
use crate::proc::{self, Pid};

pub const KIND_SERIAL: u8 = 0;
pub const KIND_BLK: u8 = 1;
pub const KIND_NET: u8 = 2;
pub const KIND_GPU: u8 = 3;
/// Cryptographically secure random bytes supplied by virtio-rng.
pub const KIND_RNG: u8 = 4;

/// Bound allocations made on behalf of bytecode callers.
pub const RNG_MAX_REQUEST: usize = 4096;

/// Common ops. Serial: WRITE arg0 = byte, READ completes with a byte,
/// TRY_READ returns the next byte or -1 immediately (no park).
/// Blk: WRITE/READ arg0 = sector, arg1 = buffer id (READ allocates its own,
/// returning the id as result); FLUSH. Net: WRITE arg1 = buffer id (tx frame),
/// READ parks until a frame arrives, result = buffer id.
pub const OP_WRITE: u32 = 1;
pub const OP_READ: u32 = 2;
pub const OP_FLUSH: u32 = 3;
pub const OP_TRY_READ: u32 = 4;
/// Gpu: CTRL arg0 = command buffer id (consumed), arg1 = response capacity
/// hint; result = response buffer id. CTRL_ATTACH arg0 = command-prefix
/// buffer id (an ATTACH_BACKING header with nr_entries=1), arg1 = backing
/// buffer id: the kernel appends the one `{phys, len}` mem entry — guest
/// physical addresses never reach bytecode — and pins the backing buffer.
pub const OP_CTRL: u32 = 1;
pub const OP_CTRL_ATTACH: u32 = 2;

pub struct Port {
    pub owner: Pid,
    pub kind: u8,
    pub sq: Ring<Sqe, 64>,
    pub cq: Ring<Cqe, 64>,
    /// Serial: READ sqes waiting for input.
    pending_reads: Ring<Sqe, 64>,
}

static PORTS: Mutex<BTreeMap<u64, Box<Port>>> = Mutex::new(BTreeMap::new());
static NEXT_PORT: AtomicU64 = AtomicU64::new(1);
/// Set from IRQ top-halves; consumed by `pump`.
static PENDING: AtomicBool = AtomicBool::new(false);

/// Serial receive ring: producer = IRQ handler, consumer = pump.
static SERIAL_RX: Ring<u8, 256> = Ring::new_zeroed();

pub fn open(kind: u8) -> Option<Term> {
    match kind {
        KIND_SERIAL => {}
        KIND_BLK if crate::virtio::has_blk() => {}
        KIND_NET if crate::virtio::has_net() => {}
        KIND_GPU if crate::virtio::has_gpu() => {}
        KIND_RNG if crate::virtio::has_rng() => {}
        _ => return None,
    }
    let id = NEXT_PORT.fetch_add(1, Ordering::Relaxed);
    PORTS.lock().insert(
        id,
        Box::new(Port {
            owner: proc::current(),
            kind,
            sq: Ring::new(),
            cq: Ring::new(),
            pending_reads: Ring::new(),
        }),
    );
    Some(Term::port(id))
}

/// Push an SQE and run the driver's submit path.
pub fn submit(port_id: u64, sqe: Sqe) -> bool {
    let mut ports = PORTS.lock();
    let Some(p) = ports.get_mut(&port_id) else {
        return false;
    };
    if p.sq.push(sqe).is_err() {
        return false;
    }
    match p.kind {
        KIND_SERIAL => serial_drain_sq(p),
        KIND_BLK => blk_drain_sq(p),
        KIND_NET => net_drain_sq(p),
        KIND_GPU => gpu_drain_sq(p),
        KIND_RNG => rng_drain_sq(p),
        _ => unreachable!(),
    }
    drop(ports);
    // Synchronous completions (e.g. writes) are ready right away.
    pump();
    true
}

fn serial_drain_sq(p: &mut Port) {
    while let Some(sqe) = p.sq.pop() {
        match sqe.op {
            OP_WRITE => {
                crate::serial::raw_write_byte(sqe.arg0 as u8);
                if sqe.tag != SKIP_CQE {
                    let _ = p.cq.push(Cqe {
                        tag: sqe.tag,
                        result: 0,
                    });
                }
            }
            OP_READ => {
                // Parks until rx bytes arrive; completed by the pump.
                let _ = p.pending_reads.push(sqe);
            }
            OP_TRY_READ => {
                let result = SERIAL_RX.pop().map(|b| b as i64).unwrap_or(-1);
                if sqe.tag != SKIP_CQE {
                    let _ = p.cq.push(Cqe {
                        tag: sqe.tag,
                        result,
                    });
                }
            }
            _ => {
                if sqe.tag != SKIP_CQE {
                    let _ = p.cq.push(Cqe {
                        tag: sqe.tag,
                        result: -1,
                    });
                }
            }
        }
    }
}

/// virtio-blk: synchronous (driver polls the virtqueue).
fn blk_drain_sq(p: &mut Port) {
    while let Some(sqe) = p.sq.pop() {
        let result: i64 = match sqe.op {
            OP_WRITE => match buf_get(sqe.arg1) {
                Some(data) => match crate::virtio::blk_write(sqe.arg0, &data) {
                    Ok(()) => 0,
                    Err(()) => -1,
                },
                None => -1,
            },
            OP_READ => match crate::virtio::blk_read(sqe.arg0) {
                Ok(data) => buf_create(data) as i64,
                Err(()) => -1,
            },
            OP_FLUSH => 0, // writes already flush
            _ => -1,
        };
        if sqe.tag != SKIP_CQE {
            let _ = p.cq.push(Cqe {
                tag: sqe.tag,
                result,
            });
        }
    }
}

/// virtio-rng: synchronous, bounded, and all-or-error.
fn rng_drain_sq(p: &mut Port) {
    while let Some(sqe) = p.sq.pop() {
        let result = if sqe.op != OP_READ
            || sqe.arg0 == 0
            || sqe.arg0 > RNG_MAX_REQUEST as u64
        {
            -1
        } else {
            match crate::virtio::random_bytes(sqe.arg0 as usize) {
                Ok(bytes) => buf_create(bytes) as i64,
                Err(()) => -1,
            }
        };
        if sqe.tag != SKIP_CQE {
            let _ = p.cq.push(Cqe {
                tag: sqe.tag,
                result,
            });
        }
    }
}

/// virtio-net: tx synchronous; rx parks until the pump polls a frame in.
fn net_drain_sq(p: &mut Port) {
    while let Some(sqe) = p.sq.pop() {
        match sqe.op {
            OP_WRITE => {
                let result = match buf_take(sqe.arg1) {
                    Some(frame) => match crate::virtio::net_send(&frame) {
                        Ok(()) => 0,
                        Err(()) => -1,
                    },
                    None => -1,
                };
                if sqe.tag != SKIP_CQE {
                    let _ = p.cq.push(Cqe {
                        tag: sqe.tag,
                        result,
                    });
                }
            }
            OP_READ => {
                let _ = p.pending_reads.push(sqe);
            }
            _ => {
                if sqe.tag != SKIP_CQE {
                    let _ = p.cq.push(Cqe {
                        tag: sqe.tag,
                        result: -1,
                    });
                }
            }
        }
    }
}

/// virtio-gpu raw transport: synchronous, command bytes are opaque.
fn gpu_drain_sq(p: &mut Port) {
    while let Some(sqe) = p.sq.pop() {
        let result: i64 = match sqe.op {
            OP_CTRL => match buf_take(sqe.arg0) {
                Some(cmd) => match crate::virtio::gpu_ctrl(&cmd, sqe.arg1 as usize) {
                    Ok(resp) => buf_create(resp) as i64,
                    Err(()) => -1,
                },
                None => -2,
            },
            OP_CTRL_ATTACH => match (buf_take(sqe.arg0), buf_pin(sqe.arg1)) {
                (Some(mut cmd), Some((phys, len))) => {
                    // struct virtio_gpu_mem_entry { le64 addr; le32 length; le32 pad }
                    cmd.extend_from_slice(&phys.to_le_bytes());
                    cmd.extend_from_slice(&(len as u32).to_le_bytes());
                    cmd.extend_from_slice(&0u32.to_le_bytes());
                    match crate::virtio::gpu_ctrl(&cmd, 24) {
                        Ok(resp) => buf_create(resp) as i64,
                        Err(()) => -1,
                    }
                }
                _ => -2,
            },
            _ => -1,
        };
        if sqe.tag != SKIP_CQE {
            let _ = p.cq.push(Cqe { tag: sqe.tag, result });
        }
    }
}

// ---- buffer table: opaque data handles crossing the port boundary ----

static BUFFERS: Mutex<BTreeMap<u64, alloc::vec::Vec<u8>>> = Mutex::new(BTreeMap::new());
static NEXT_BUF: AtomicU64 = AtomicU64::new(1);
/// Buffers attached as device backing stores: the device holds their guest
/// physical address, so `buf_take` must never move them out. They live until
/// the kernel shuts the resource down (for now: forever — a demo-scale leak).
static PINNED: Mutex<alloc::collections::BTreeSet<u64>> = Mutex::new(alloc::collections::BTreeSet::new());

pub fn buf_create(data: alloc::vec::Vec<u8>) -> u64 {
    let id = NEXT_BUF.fetch_add(1, Ordering::Relaxed);
    BUFFERS.lock().insert(id, data);
    id
}

pub fn buf_take(id: u64) -> Option<alloc::vec::Vec<u8>> {
    if PINNED.lock().contains(&id) {
        return None;
    }
    BUFFERS.lock().remove(&id)
}

/// Overwrite part of an existing buffer in place. Never grows it — a pinned
/// backing buffer's physical address must stay valid — so out-of-bounds
/// writes are refused rather than extended.
pub fn buf_write(id: u64, offset: usize, data: &[u8]) -> bool {
    let mut bufs = BUFFERS.lock();
    let Some(buf) = bufs.get_mut(&id) else {
        return false;
    };
    let Some(end) = offset.checked_add(data.len()) else {
        return false;
    };
    if end > buf.len() {
        return false;
    }
    buf[offset..end].copy_from_slice(data);
    true
}

/// Copy a bounds-checked slice of a buffer out.
pub fn buf_read(id: u64, offset: usize, len: usize) -> Option<alloc::vec::Vec<u8>> {
    let bufs = BUFFERS.lock();
    let buf = bufs.get(&id)?;
    let end = offset.checked_add(len)?;
    if end > buf.len() {
        return None;
    }
    Some(buf[offset..end].to_vec())
}

/// Pin a buffer as a device backing store and return its physical address and
/// length. The kernel heap is physically contiguous, so one mem entry covers
/// the whole buffer.
fn buf_pin(id: u64) -> Option<(u64, usize)> {
    let bufs = BUFFERS.lock();
    let data = bufs.get(&id)?;
    let phys = crate::mm::virt_to_phys(data.as_ptr() as u64);
    let len = data.len();
    drop(bufs);
    PINNED.lock().insert(id);
    Some((phys, len))
}

pub fn buf_get(id: u64) -> Option<alloc::vec::Vec<u8>> {
    BUFFERS.lock().get(&id).cloned()
}

/// IRQ top-half for the UART: stash bytes, flag the bottom half.
pub fn serial_rx_irq() {
    while let Some(b) = crate::serial::try_read_byte() {
        let _ = SERIAL_RX.push(b);
    }
    PENDING.store(true, Ordering::Release);
}

/// Bottom half. Runs in the scheduler loop (and after submissions): match
/// buffered input with parked reads, then deliver all CQEs as messages.
pub fn pump() {
    PENDING.store(false, Ordering::Release);
    let mut deliveries: alloc::vec::Vec<(Pid, u64, Cqe)> = alloc::vec::Vec::new();
    {
        let mut ports = PORTS.lock();
        for (id, p) in ports.iter_mut() {
            match p.kind {
                KIND_SERIAL => {
                    while !SERIAL_RX.is_empty() && !p.pending_reads.is_empty() {
                        let sqe = p.pending_reads.pop().unwrap();
                        let b = SERIAL_RX.pop().unwrap();
                        let _ = p.cq.push(Cqe {
                            tag: sqe.tag,
                            result: b as i64,
                        });
                    }
                }
                KIND_NET => {
                    // Polled rx for now (INTx/MSI-X later): at most one frame
                    // per parked read per pump pass.
                    while !p.pending_reads.is_empty() {
                        let Some(frame) = crate::virtio::net_try_recv() else {
                            break;
                        };
                        let sqe = p.pending_reads.pop().unwrap();
                        let _ = p.cq.push(Cqe {
                            tag: sqe.tag,
                            result: buf_create(frame) as i64,
                        });
                    }
                }
                _ => {}
            }
            while let Some(cqe) = p.cq.pop() {
                deliveries.push((p.owner, *id, cqe));
            }
        }
    }
    for (owner, port_id, cqe) in deliveries {
        deliver(owner, port_id, cqe);
    }
}

fn deliver(owner: Pid, port_id: u64, cqe: Cqe) {
    let reply = atoms::intern("port_reply");
    let ok = proc::send_built(owner, |h| {
        h.tuple(&[
            Term::atom(reply),
            Term::port(port_id),
            Term::int(cqe.tag),
            Term::int(cqe.result),
        ])
    });
    if !ok {
        log::warn!("[ports] dropped completion for dead/full pid {owner}");
    }
}

/// Close all ports owned by a dead process.
pub fn close_owned_by(pid: Pid) {
    PORTS.lock().retain(|_, p| p.owner != pid);
}
