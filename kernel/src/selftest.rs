//! In-kernel self tests, run when the cmdline contains `selftest`.
//! The xtask test harness asserts on the `[ok]`/`[selftest]` serial markers.

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

use ygg_alloc::FRAME_SIZE;

use ygg_term::Term;

use crate::{atoms, boot, idt, irq, mm, proc, println};

/// Pre-scheduler tests, run from the boot context.
pub fn early() {
    println!("[selftest] running");

    limine_requests();
    breakpoint();
    pmm();
    heap();
    acpi();
    timer();
}

/// Process-machinery tests; runs *as* the init process under the scheduler.
pub extern "C" fn proc_tests(_arg: u64) {
    ping_pong();
    preemption();
    stack_overflow();
    terms_in_kernel();
    supervisor();
    link_propagation();
    heap_quota();
    receive_after();
    selective_receive();
    blk_port();
    net_port();
    verifier_rejects();
    runtime_module_load();
    hot_code_loading();
    differential_engines();
    bytecode();
    serial_port_echo();
    println!("[selftest] all passed");
    crate::qemu::exit(crate::qemu::ExitCode::Success);
}

fn ping_pong() {
    let b = proc::spawn(pong, 0);
    proc::send(b, Term::pid(proc::current()));
    for i in 1..=5i64 {
        proc::send(b, Term::int(i));
        let r = proc::recv().as_int().expect("expected int reply");
        assert_eq!(r, i + 1, "pong replied {r} to {i}");
        println!("[pingpong] round {i}: sent {i}, got {r}");
    }
    proc::send(b, Term::NIL);
    println!("[ok] process ping-pong (5 rounds)");
}

extern "C" fn pong(_arg: u64) {
    let a = proc::recv().as_pid().expect("first message must be a pid");
    loop {
        let m = proc::recv();
        match m.as_int() {
            Some(v) => proc::send(a, Term::int(v + 1)),
            None => break,
        };
    }
}

static BUSY_A: AtomicU64 = AtomicU64::new(0);
static BUSY_B: AtomicU64 = AtomicU64::new(0);

extern "C" fn busy(which: u64) {
    let counter = if which == 0 { &BUSY_A } else { &BUSY_B };
    loop {
        counter.fetch_add(1, Ordering::Relaxed);
        // Never yields voluntarily — only the safepoint (interpreter back-edge
        // stand-in) can take the CPU away, and only when the timer asked.
        proc::safepoint();
    }
}

/// Two busy loops never yield; this watcher only runs again if the timer
/// preempts them at safepoints. Both counters advancing while the watcher
/// observes them proves preemptive scheduling.
fn preemption() {
    let a = proc::spawn(busy, 0);
    let b = proc::spawn(busy, 1);
    const TARGET: u64 = 200_000;
    loop {
        proc::yield_now();
        if BUSY_A.load(Ordering::Relaxed) > TARGET && BUSY_B.load(Ordering::Relaxed) > TARGET {
            break;
        }
    }
    proc::kill(a, "test finished");
    proc::kill(b, "test finished");
    println!("[ok] preemptive scheduling (two non-yielding busy loops both advanced)");
}

extern "C" fn recurser(_arg: u64) {
    recurse(0);
}

fn recurse(n: u64) -> u64 {
    let mut buf = [0u8; 256];
    unsafe { core::ptr::write_volatile(buf.as_mut_ptr(), n as u8) };
    let sub = recurse(n + 1);
    sub + unsafe { core::ptr::read_volatile(buf.as_ptr()) } as u64
}

fn stack_overflow() {
    let p = proc::spawn(recurser, 0);
    while proc::is_alive(p) {
        proc::yield_now();
    }
    // We're still here, so the kill was surgical.
    println!("[ok] stack overflow killed only the offender");
}

// ---- M3: terms and full process semantics ----

/// Build a compound term, send it to ourselves (forcing a cross-heap copy),
/// and verify structure + rendering.
fn terms_in_kernel() {
    let me = proc::current();
    let msg = proc::build(|h| {
        let inner = h.tuple(&[atoms::atom("hello"), Term::int(42)])?;
        let lst = h.list(&[Term::int(1), Term::int(2), Term::int(3)])?;
        let bin = h.binary(b"ygg")?;
        h.tuple(&[inner, lst, bin])
    });
    proc::send(me, msg);
    let got = proc::recv();
    assert_ne!(got.0, msg.0, "self-send must copy, not alias");
    unsafe {
        assert!(ygg_term::eq(got, msg), "copy is not structurally equal");
        assert_eq!(got.tuple_elem(0).tuple_elem(1).as_int(), Some(42));
        assert_eq!(got.tuple_elem(2).bin_bytes(), b"ygg");
    }
    let mut s = alloc::string::String::new();
    unsafe { ygg_term::fmt_term(got, &mut s, &atoms::name).unwrap() };
    println!("[terms] {s}");
    assert_eq!(s, "{{hello, 42}, [1, 2, 3], <<3 bytes>>}");
    println!("[ok] terms: build/copy-on-send/eq/format");
}

extern "C" fn crashing_child(_arg: u64) {
    // Do a little work so the death is not instant.
    proc::yield_now();
    proc::exit("boom");
}

/// Monitor a child, see it crash, restart it — three times.
fn supervisor() {
    let down = atoms::atom("DOWN");
    for round in 1..=3 {
        let c = proc::spawn(crashing_child, 0);
        let r = proc::monitor(c);
        let msg = proc::recv();
        unsafe {
            assert_eq!(msg.tuple_arity(), 4);
            assert_eq!(msg.tuple_elem(0), down);
            assert_eq!(msg.tuple_elem(1), Term::reference(r));
            assert_eq!(msg.tuple_elem(2), Term::pid(c));
            assert_eq!(msg.tuple_elem(3), atoms::atom("boom"));
        }
        println!("[supervisor] restart {round}: child {c} down with reason boom");
    }
    println!("[ok] supervisor: 3 restarts via DOWN messages");
}

extern "C" fn linked_parent(_arg: u64) {
    let _child = proc::spawn_link(bad_child, 0);
    // Wait forever: only the link's exit signal can take us down.
    proc::recv();
}

extern "C" fn bad_child(_arg: u64) {
    proc::exit("bad");
}

/// A crashing child kills its linked parent; reason propagates.
fn link_propagation() {
    let a = proc::spawn(linked_parent, 0);
    let r = proc::monitor(a);
    let msg = proc::recv();
    unsafe {
        assert_eq!(msg.tuple_elem(1), Term::reference(r));
        assert_eq!(msg.tuple_elem(3), atoms::atom("bad"), "link must propagate the reason");
    }
    println!("[ok] exit propagation over links (parent died with child's reason)");
}

extern "C" fn heap_hog(_arg: u64) {
    // Allocate until the fixed heap (quota) runs out; proc::build exits us.
    let mut n = 0i64;
    loop {
        let _ = proc::build(|h| {
            let mut l = Term::NIL;
            for i in 0..64 {
                l = h.cons(Term::int(i), l)?;
            }
            Ok(l)
        });
        n += 1;
        if n % 1024 == 0 {
            proc::safepoint();
        }
    }
}

fn heap_quota() {
    let p = proc::spawn(heap_hog, 0);
    let r = proc::monitor(p);
    let msg = proc::recv();
    unsafe {
        assert_eq!(msg.tuple_elem(1), Term::reference(r));
        assert_eq!(msg.tuple_elem(3), atoms::atom("heap quota exceeded"));
    }
    println!("[ok] heap quota breach killed the process, not the kernel");
}

fn receive_after() {
    let t0 = irq::ticks();
    let r = proc::recv_timeout(100);
    let dt = irq::ticks() - t0;
    assert!(r.is_none(), "empty mailbox must time out");
    assert!(dt >= 100, "timeout fired early: {dt} ms");
    println!("[ok] receive-after: empty mailbox timed out after {dt} ms");
}

/// Load the assembled selftest module and run its main/1 under the
/// interpreter. Its internal busy loop only ever leaves the CPU via back-edge
/// safepoints, so the ping-pong completing proves interpreter preemption.
fn bytecode() {
    let bytes = crate::modload::boot_module_bytes("selftest.yggm")
        .expect("selftest.yggm boot module missing");
    let module = crate::modload::load("selftest", bytes).expect("selftest.yggm failed to decode");
    let me = proc::current();
    let p = crate::modload::spawn(module, "main", Term::pid(me)).expect("no main/1");
    let msg = proc::recv_timeout(10_000).expect("bytecode main never reported back");
    assert_eq!(msg, atoms::atom("bc_done"));
    while proc::is_alive(p) {
        proc::yield_now();
    }
    println!("[ok] bytecode ping-pong under interpreter (busy loop preempted at back-edges)");
}

// ---- M5: virtio-blk and virtio-net through the port layer ----

/// Submit one SQE on `port` and block for its `{port_reply, _, tag, result}`.
fn port_request(port: Term, op: u32, arg0: u64, arg1: u64, tag: i64) -> i64 {
    use ygg_rings::Sqe;
    let id = port.as_port().unwrap();
    assert!(crate::ports::submit(id, Sqe { op, tag, arg0, arg1 }), "submit failed");
    let reply = atoms::atom("port_reply");
    let msg = proc::recv_where(
        |m| unsafe {
            m.is_boxed()
                && m.kind() == ygg_term::Kind::Tuple
                && m.tuple_arity() == 4
                && m.tuple_elem(0) == reply
                && m.tuple_elem(2) == Term::int(tag)
        },
        Some(10_000),
    )
    .expect("port completion timed out");
    unsafe { msg.tuple_elem(3).as_int().unwrap() }
}

fn blk_pattern() -> Vec<u8> {
    const SEED: &[u8] = b"YGGDRASIL-BLK-PERSISTENCE-";
    (0..512).map(|i| SEED[i % SEED.len()] ^ (i / SEED.len()) as u8).collect()
}

const BLK_TEST_SECTOR: u64 = 1;

fn blk_port() {
    use crate::ports::{OP_READ, OP_WRITE};
    let port = crate::ports::open(crate::ports::KIND_BLK).expect("no virtio-blk device");
    let data = blk_pattern();
    let buf = crate::ports::buf_create(data.clone());
    assert_eq!(port_request(port, OP_WRITE, BLK_TEST_SECTOR, buf, 10), 0, "blk write failed");
    let r = port_request(port, OP_READ, BLK_TEST_SECTOR, 0, 11);
    assert!(r > 0, "blk read failed");
    let back = crate::ports::buf_take(r as u64).unwrap();
    assert_eq!(back, data, "sector read-back mismatch");
    println!("[ok] blk port: wrote and read back sector {BLK_TEST_SECTOR} via rings");
}

/// Second-boot phase (cmdline `verify-disk`): the pattern written by the
/// previous boot must still be on disk.
pub extern "C" fn verify_disk(_arg: u64) {
    use crate::ports::OP_READ;
    let port = crate::ports::open(crate::ports::KIND_BLK).expect("no virtio-blk device");
    let r = port_request(port, OP_READ, BLK_TEST_SECTOR, 0, 1);
    assert!(r > 0, "blk read failed");
    let back = crate::ports::buf_take(r as u64).unwrap();
    assert_eq!(back, blk_pattern(), "pattern did not survive the reboot");
    println!("[ok] blk persistence verified after reboot");
    crate::qemu::exit(crate::qemu::ExitCode::Success);
}

fn ip_checksum(hdr: &[u8]) -> u16 {
    let mut sum = 0u32;
    for w in hdr.chunks(2) {
        sum += u32::from(u16::from_be_bytes([w[0], *w.get(1).unwrap_or(&0)]));
    }
    while sum >> 16 != 0 {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }
    !(sum as u16)
}

fn build_arp_request(src_mac: [u8; 6], spa: [u8; 4], tpa: [u8; 4]) -> Vec<u8> {
    let mut f = Vec::new();
    f.extend_from_slice(&[0xFF; 6]); // broadcast
    f.extend_from_slice(&src_mac);
    f.extend_from_slice(&[0x08, 0x06]); // ARP
    f.extend_from_slice(&[0, 1, 8, 0, 6, 4, 0, 1]); // eth/ipv4, request
    f.extend_from_slice(&src_mac);
    f.extend_from_slice(&spa);
    f.extend_from_slice(&[0; 6]);
    f.extend_from_slice(&tpa);
    f.resize(60, 0);
    f
}

fn build_udp(
    src_mac: [u8; 6],
    dst_mac: [u8; 6],
    src_ip: [u8; 4],
    dst_ip: [u8; 4],
    port: u16,
    payload: &[u8],
) -> Vec<u8> {
    let mut f = Vec::new();
    f.extend_from_slice(&dst_mac);
    f.extend_from_slice(&src_mac);
    f.extend_from_slice(&[0x08, 0x00]); // IPv4
    let ip_len = (20 + 8 + payload.len()) as u16;
    let mut ip = alloc::vec![0x45, 0, 0, 0, 0, 0, 0, 0, 64, 17, 0, 0];
    ip[2..4].copy_from_slice(&ip_len.to_be_bytes());
    ip.extend_from_slice(&src_ip);
    ip.extend_from_slice(&dst_ip);
    let csum = ip_checksum(&ip);
    ip[10..12].copy_from_slice(&csum.to_be_bytes());
    f.extend_from_slice(&ip);
    let udp_len = (8 + payload.len()) as u16;
    f.extend_from_slice(&port.to_be_bytes());
    f.extend_from_slice(&port.to_be_bytes());
    f.extend_from_slice(&udp_len.to_be_bytes());
    f.extend_from_slice(&[0, 0]); // checksum optional for IPv4 UDP
    f.extend_from_slice(payload);
    if f.len() < 60 {
        f.resize(60, 0);
    }
    f
}

/// ARP-resolve the QEMU user-net gateway (proving rx), then send it a UDP
/// packet whose payload the harness greps out of the pcap (proving tx).
fn net_port() {
    use crate::ports::{OP_READ, OP_WRITE};
    let port = crate::ports::open(crate::ports::KIND_NET).expect("no virtio-net device");
    let mac = crate::virtio::net_mac();
    let (our_ip, gw_ip) = ([10, 0, 2, 15], [10, 0, 2, 2]);

    let arp = build_arp_request(mac, our_ip, gw_ip);
    let buf = crate::ports::buf_create(arp);
    assert_eq!(port_request(port, OP_WRITE, 0, buf, 20), 0, "arp tx failed");

    let mut gw_mac = None;
    for _ in 0..20 {
        let r = port_request(port, OP_READ, 0, 0, 21);
        if r <= 0 {
            continue;
        }
        let frame = crate::ports::buf_take(r as u64).unwrap();
        // ARP reply (op 2) answering for the gateway IP?
        if frame.len() >= 42
            && frame[12..14] == [0x08, 0x06]
            && frame[20..22] == [0, 2]
            && frame[28..32] == gw_ip
        {
            gw_mac = Some(<[u8; 6]>::try_from(&frame[22..28]).unwrap());
            break;
        }
    }
    let gw_mac = gw_mac.expect("no ARP reply from user-net gateway");

    let udp = build_udp(mac, gw_mac, our_ip, gw_ip, 7777, b"YGG-NET-OK");
    let buf = crate::ports::buf_create(udp);
    assert_eq!(port_request(port, OP_WRITE, 0, buf, 22), 0, "udp tx failed");
    println!("[ok] net port: ARP resolved gateway, UDP payload sent");
}

// ---- M6: verifier + runtime module loading ----

fn verifier_rejects() {
    use ygg_bytecode::{Function, Module, op};
    // Truncated real module: must be rejected without crashing.
    let good = crate::modload::boot_module_bytes("selftest.yggm").unwrap();
    assert!(crate::modload::load("bad-truncated", &good[..good.len() / 2]).is_err());

    // Structurally valid encoding, but control falls off the end.
    let falls = Module {
        atoms: alloc::vec![alloc::string::String::from("main")],
        functions: alloc::vec![Function {
            name_atom: 0,
            arity: 0,
            nregs: 1,
            code: alloc::vec![op::LOAD_NIL, 0],
        }],
    };
    assert!(crate::modload::load("bad-falloff", &falls.encode()).is_err());

    // Jump to nowhere.
    let badjmp = Module {
        atoms: alloc::vec![alloc::string::String::from("main")],
        functions: alloc::vec![Function {
            name_atom: 0,
            arity: 0,
            nregs: 1,
            code: alloc::vec![op::JMP, 0xFF, 0xFF, 0xFF, 0x7F],
        }],
    };
    assert!(crate::modload::load("bad-jump", &badjmp.encode()).is_err());
    println!("[ok] verifier rejected malformed modules (no crash)");
}

/// A module that is NOT on the boot ISO: fetched sector-by-sector from the
/// storage port at runtime, verified, loaded and spawned.
fn runtime_module_load() {
    use crate::ports::OP_READ;
    const MODULE_SECTOR: u64 = 2048; // [u32 len][bytes], written by xtask
    let port = crate::ports::open(crate::ports::KIND_BLK).expect("no virtio-blk device");

    let r = port_request(port, OP_READ, MODULE_SECTOR, 0, 30);
    assert!(r > 0);
    let first = crate::ports::buf_take(r as u64).unwrap();
    let len = u32::from_le_bytes(first[0..4].try_into().unwrap()) as usize;
    assert!(len > 0 && len < 1024 * 1024, "implausible module length {len}");
    let mut bytes = first[4..].to_vec();
    let mut sector = MODULE_SECTOR + 1;
    while bytes.len() < len {
        let r = port_request(port, OP_READ, sector, 0, 31);
        assert!(r > 0);
        bytes.extend_from_slice(&crate::ports::buf_take(r as u64).unwrap());
        sector += 1;
    }
    bytes.truncate(len);

    let module = crate::modload::load("hotmod", &bytes).expect("hotmod failed to verify/load");
    let me = proc::current();
    crate::modload::spawn(module, "main", Term::pid(me)).expect("no main/1 in hotmod");
    let msg = proc::recv_timeout(10_000).expect("hotmod never greeted us");
    assert_eq!(msg, atoms::atom("hot_hello"));
    println!("[ok] module loaded at runtime from the storage port and spawned");
}

// ---- M7: hot code loading ----

/// Upgrade a live counter server from v1 to v2: the in-flight v1 loop answers
/// one more request, then migrates through its `call_ext` back-edge into v2 —
/// state intact, new reply format. A second server deliberately left on v1 is
/// then killed by `purge`.
fn hot_code_loading() {
    let v1_bytes = crate::modload::boot_module_bytes("counter_v1.yggm").unwrap();
    let v2_bytes = crate::modload::boot_module_bytes("counter_v2.yggm").unwrap();
    let v1 = crate::modload::load("counter", v1_bytes).expect("counter v1 load");
    let me = proc::current();
    let server = crate::modload::spawn(v1.clone(), "main", Term::int(0)).unwrap();

    let ask = || -> (Term, i64) {
        proc::send(server, Term::pid(me));
        let msg = proc::recv_timeout(10_000).expect("counter did not reply");
        unsafe { (msg.tuple_elem(0), msg.tuple_elem(1).as_int().expect("non-int count")) }
    };
    let (count, count2) = (atoms::atom("count"), atoms::atom("count2"));

    assert_eq!(ask(), (count, 0));
    assert_eq!(ask(), (count, 1));
    crate::modload::load("counter", v2_bytes).expect("counter v2 load"); // live upgrade

    // The in-flight v1 loop migrates at its next call_ext. Whether request 3
    // is still served by the parked v1 frame or already by v2 depends on
    // where the scheduler preempted the server — both are correct:
    //   late:  {count,2}  {count2,3}  {count2,13}
    //   early: {count2,2} {count2,12} {count2,22}
    let (f3, n3) = ask();
    assert_eq!(n3, 2, "state lost in upgrade");
    let step = if f3 == count { 1 } else { 10 };
    assert!(f3 == count || f3 == count2, "unknown reply format");
    let (f4, n4) = ask();
    assert_eq!((f4, n4), (count2, n3 + step), "migration did not reach v2");
    let (f5, n5) = ask();
    assert_eq!((f5, n5), (count2, n4 + 10), "v2 must increment by 10");
    println!(
        "[hot] migration at request {}: state carried ({} -> {} -> {})",
        if step == 1 { 4 } else { 3 },
        n3,
        n4,
        n5
    );

    let holdout = crate::modload::spawn(v1, "main", Term::int(0)).unwrap();
    let killed = crate::modload::purge("counter");
    assert!(killed >= 1, "purge found no holdouts");
    assert!(!proc::is_alive(holdout), "holdout survived purge");
    assert!(proc::is_alive(server), "purge must spare migrated processes");
    proc::kill(server, "test finished");
    println!("[ok] hot upgrade v1->v2: state retained, new format, purge killed the holdout");
}

// ---- M8: Cranelift JIT tier ----

/// The same module, run under tier-0 (interpreter) and tier-1 (Cranelift):
/// the result terms must be structurally identical. (The rest of the suite is
/// itself a broad differential test — every module in it runs JIT'd now, with
/// expectations that were recorded under the interpreter.)
fn differential_engines() {
    let bytes = crate::modload::boot_module_bytes("mathmod.yggm").unwrap();
    let me = proc::current();

    let interp = crate::modload::load_with_engine("mathmod_interp", bytes, false).unwrap();
    assert!(interp.jit.is_none());
    crate::modload::spawn(interp, "main", Term::pid(me)).unwrap();
    let r_interp = proc::recv_timeout(10_000).expect("interp mathmod never replied");

    let jitted = crate::modload::load_with_engine("mathmod_jit", bytes, true).unwrap();
    assert!(jitted.jit.is_some(), "JIT compilation failed for mathmod");
    crate::modload::spawn(jitted, "main", Term::pid(me)).unwrap();
    let r_jit = proc::recv_timeout(10_000).expect("JIT mathmod never replied");

    unsafe {
        assert!(ygg_term::eq(r_interp, r_jit), "engines disagree");
        assert_eq!(r_jit.tuple_elem(1).as_int(), Some(3628800));
    }
    println!("[ok] differential: interpreter and Cranelift JIT agree on mathmod");
}

/// Spawn the bytecode echo server: it opens the serial port, prints a ready
/// marker (which cues the host harness to type "PING\n"), echoes every byte
/// through the SQ/CQ rings, and reports back after the newline.
fn serial_port_echo() {
    let bytes = crate::modload::boot_module_bytes("selftest.yggm").unwrap();
    let module = crate::modload::load("selftest-echo", bytes).unwrap();
    let me = proc::current();
    crate::modload::spawn(module, "echo", Term::pid(me)).expect("no echo/1");
    let msg = proc::recv_timeout(30_000).expect("echo never saw a newline — harness didn't type?");
    assert_eq!(msg, atoms::atom("bc_echo_done"));
    println!("[ok] serial port echo via SQ/CQ rings");
}

fn selective_receive() {
    let me = proc::current();
    // Queue three messages, then take the middle one first.
    proc::send(me, Term::int(1));
    proc::send(me, atoms::atom("wanted"));
    proc::send(me, Term::int(2));
    let picked = proc::recv_where(|m| m == atoms::atom("wanted"), None).unwrap();
    assert_eq!(picked, atoms::atom("wanted"));
    assert_eq!(proc::recv().as_int(), Some(1), "skipped messages must stay in order");
    assert_eq!(proc::recv().as_int(), Some(2));
    println!("[ok] selective receive picked the matching message first");
}

fn limine_requests() {
    assert!(boot::HHDM.response().is_some(), "no HHDM response");
    assert!(boot::MEMMAP.response().is_some(), "no memmap response");
    assert!(
        !boot::MEMMAP.response().unwrap().entries().is_empty(),
        "empty memmap"
    );
    println!("[ok] limine requests answered");
}

fn breakpoint() {
    x86_64::instructions::interrupts::int3();
    assert!(
        idt::BREAKPOINT_HIT.load(Ordering::SeqCst),
        "breakpoint handler did not run"
    );
    println!("[ok] int3 dispatched and recovered");
}

fn pmm() {
    let before = mm::free_frame_count();
    assert!(before > 1000, "suspiciously few free frames: {before}");

    let a = mm::alloc_frame().unwrap();
    let b = mm::alloc_frame().unwrap();
    assert_ne!(a, b);
    assert_eq!(a % FRAME_SIZE, 0);

    // The frames are real, distinct memory: write through the HHDM and check.
    unsafe {
        mm::phys_to_virt(a).write_bytes(0xAA, FRAME_SIZE as usize);
        mm::phys_to_virt(b).write_bytes(0x55, FRAME_SIZE as usize);
        assert_eq!(*mm::phys_to_virt(a), 0xAA);
        assert_eq!(*mm::phys_to_virt(b), 0x55);
    }

    let contig = mm::alloc_contig(16, 16).unwrap();
    assert_eq!(contig % (16 * FRAME_SIZE), 0, "contig alloc misaligned");

    mm::free_frames(a, 1);
    mm::free_frames(b, 1);
    mm::free_frames(contig, 16);
    assert_eq!(mm::free_frame_count(), before, "frame leak in pmm test");
    println!("[ok] pmm alloc/free/contig");
}

fn heap() {
    let mut v: Vec<u64> = Vec::new();
    for i in 0..100_000u64 {
        v.push(i);
    }
    assert_eq!(v.iter().sum::<u64>(), 100_000 * 99_999 / 2);

    let mut m = BTreeMap::new();
    for i in 0..1000u32 {
        m.insert(i, i * 2);
    }
    assert_eq!(m.get(&500), Some(&1000));

    let boxed = Box::new([0u8; 4096]);
    drop(boxed);
    println!("[ok] kernel heap (vec/btree/box)");
}

fn acpi() {
    let p = crate::acpi_tables::platform();
    assert!(p.lapic_phys != 0, "no LAPIC address");
    assert!(!p.ioapics.is_empty(), "no IOAPIC");
    assert!(!p.ecam.is_empty(), "no ECAM (MCFG) — q35 should have MMCONFIG");
    println!(
        "[ok] acpi: lapic={:#x} ioapic0={:#x} ecam0={:#x}",
        p.lapic_phys, p.ioapics[0].phys_addr, p.ecam[0].base
    );
}

/// Prove the 1 kHz timer advances monotonically for 3 seconds of uptime.
fn timer() {
    let start = irq::ticks();
    let mut last_report = start;
    let mut prev = start;
    loop {
        x86_64::instructions::hlt();
        let now = irq::ticks();
        assert!(now >= prev, "tick count went backwards");
        prev = now;
        if now - last_report >= 500 {
            last_report = now;
            println!("[time] uptime {} ms", now);
        }
        if now - start >= 3000 {
            break;
        }
    }
    println!("[ok] lapic timer: 3 s of monotonic 1 kHz ticks");
}
