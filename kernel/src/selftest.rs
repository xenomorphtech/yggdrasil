//! In-kernel self tests, run when the cmdline contains `selftest`.
//! The xtask test harness asserts on the `[ok]`/`[selftest]` serial markers.

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

use ygg_alloc::FRAME_SIZE;

use ygg_term::Term;

use crate::{atoms, boot, idt, irq, mm, println, proc};

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
    smp_parallelism();
    smp_stealing();
    smp_churn();
    terms_in_kernel();
    supervisor();
    link_propagation();
    heap_quota();
    receive_after();
    selective_receive();
    blk_port();
    rng_port();
    net_port();
    verifier_rejects();
    runtime_module_load();
    hot_code_loading();
    differential_engines();
    lux_tcp_stack();
    lux_redmagic();
    lux_bounded_loop();
    lux_port_hello();
    lux_gpu_demo();
    lux_font_scene();
    lux_virgl_probe();
    lux_tcp_echo_live();
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

// ---- M9: SMP ----

static SPIN_FLAG: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
static SPIN_PROGRESS: AtomicU64 = AtomicU64::new(0);

extern "C" fn spinner_no_safepoints(_arg: u64) {
    // Deliberately no safepoints: this process can never be preempted. It can
    // only make progress if another core runs it while its parent also runs.
    while !SPIN_FLAG.load(Ordering::Acquire) {
        SPIN_PROGRESS.fetch_add(1, Ordering::Relaxed);
    }
}

/// Proves simultaneous execution: the parent busy-waits (no yielding) while
/// the un-preemptible spinner advances — impossible on one core.
fn smp_parallelism() {
    assert!(crate::percpu::count() >= 2, "suite requires -smp 2");
    let p = proc::spawn(spinner_no_safepoints, 0);
    let start = irq::ticks();
    while SPIN_PROGRESS.load(Ordering::Relaxed) == 0 {
        if irq::ticks() - start >= 10_000 {
            let mut snap = alloc::string::String::new();
            use core::fmt::Write;
            for c in crate::percpu::all() {
                let _ = write!(
                    snap,
                    "cpu{}: cur={} q={} idle={} phase={} sw={} | ",
                    c.id,
                    c.current.load(Ordering::Relaxed),
                    c.runq.lock().len(),
                    c.idle.load(Ordering::Relaxed),
                    c.phase.load(Ordering::Relaxed),
                    c.switches.load(Ordering::Relaxed),
                );
            }
            panic!("spinner never ran concurrently: {snap}");
        }
        core::hint::spin_loop();
    }
    SPIN_FLAG.store(true, Ordering::Release);
    while proc::is_alive(p) {
        core::hint::spin_loop();
    }
    println!("[ok] smp: two cores executed simultaneously");
}

static JOBS_DONE: AtomicU64 = AtomicU64::new(0);

extern "C" fn compute_job(_arg: u64) {
    let mut x = 0u64;
    for i in 0..200_000u64 {
        x = x.wrapping_add(i * i);
        if i % 4096 == 0 {
            proc::safepoint();
        }
    }
    core::hint::black_box(x);
    JOBS_DONE.fetch_add(1, Ordering::Release);
}

/// All jobs are spawned onto this core's queue; the other core only gets work
/// by stealing. Every core's switch-in counter must grow.
fn smp_stealing() {
    let before: Vec<u64> = crate::percpu::all()
        .iter()
        .map(|c| c.switches.load(Ordering::Relaxed))
        .collect();
    JOBS_DONE.store(0, Ordering::Release);
    for _ in 0..8 {
        proc::spawn(compute_job, 0);
    }
    while JOBS_DONE.load(Ordering::Acquire) < 8 {
        proc::yield_now();
    }
    for (c, b) in crate::percpu::all().iter().zip(before) {
        assert!(
            c.switches.load(Ordering::Relaxed) > b,
            "cpu {} never ran a stolen process",
            c.id
        );
    }
    println!("[ok] smp: work stealing spread load across cpus");
}

extern "C" fn churn_worker(_arg: u64) {
    proc::yield_now();
    let _ = proc::build(|h| {
        let l = h.list(&[Term::int(1), Term::int(2), Term::int(3)])?;
        h.tuple(&[atoms::atom("churn"), l])
    });
    proc::yield_now();
}

/// 200 spawn/exit cycles across cores: exercises stack-slot recycling under
/// TLB shootdown and cross-core reaping.
fn smp_churn() {
    for _ in 0..40 {
        let pids: Vec<proc::Pid> = (0..5).map(|_| proc::spawn(churn_worker, 0)).collect();
        for p in pids {
            while proc::is_alive(p) {
                proc::yield_now();
            }
        }
    }
    println!("[ok] smp: 200-process churn with TLB shootdown");
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
        let (c, r) = proc::spawn_monitor(crashing_child, 0);
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
    let (a, r) = proc::spawn_monitor(linked_parent, 0);
    let _ = a;
    let msg = proc::recv();
    unsafe {
        assert_eq!(msg.tuple_elem(1), Term::reference(r));
        assert_eq!(
            msg.tuple_elem(3),
            atoms::atom("bad"),
            "link must propagate the reason"
        );
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
    let (_p, r) = proc::spawn_monitor(heap_hog, 0);
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
    port_request_timeout(port, op, arg0, arg1, tag, Some(10_000))
        .expect("port completion timed out")
}

fn port_request_wait(port: Term, op: u32, arg0: u64, arg1: u64, tag: i64) -> i64 {
    port_request_timeout(port, op, arg0, arg1, tag, None).expect("blocking port request ended")
}

fn port_request_timeout(
    port: Term,
    op: u32,
    arg0: u64,
    arg1: u64,
    tag: i64,
    timeout_ms: Option<u64>,
) -> Option<i64> {
    use ygg_rings::Sqe;
    let id = port.as_port().unwrap();
    assert!(
        crate::ports::submit(
            id,
            Sqe {
                op,
                tag,
                arg0,
                arg1
            }
        ),
        "submit failed"
    );
    let reply = atoms::atom("port_reply");
    let msg = proc::recv_where(
        |m| unsafe {
            m.is_boxed()
                && m.kind() == ygg_term::Kind::Tuple
                && m.tuple_arity() == 4
                && m.tuple_elem(0) == reply
                && m.tuple_elem(2) == Term::int(tag)
        },
        timeout_ms,
    )?;
    Some(unsafe { msg.tuple_elem(3).as_int().unwrap() })
}

fn blk_pattern() -> Vec<u8> {
    const SEED: &[u8] = b"YGGDRASIL-BLK-PERSISTENCE-";
    (0..512)
        .map(|i| SEED[i % SEED.len()] ^ (i / SEED.len()) as u8)
        .collect()
}

const BLK_TEST_SECTOR: u64 = 1;

fn blk_port() {
    use crate::ports::{OP_READ, OP_WRITE};
    let port = crate::ports::open(crate::ports::KIND_BLK).expect("no virtio-blk device");
    let data = blk_pattern();
    let buf = crate::ports::buf_create(data.clone());
    assert_eq!(
        port_request(port, OP_WRITE, BLK_TEST_SECTOR, buf, 10),
        0,
        "blk write failed"
    );
    let r = port_request(port, OP_READ, BLK_TEST_SECTOR, 0, 11);
    assert!(r > 0, "blk read failed");
    let back = crate::ports::buf_take(r as u64).unwrap();
    assert_eq!(back, data, "sector read-back mismatch");
    println!("[ok] blk port: wrote and read back sector {BLK_TEST_SECTOR} via rings");
}

fn rng_port() {
    use crate::ports::OP_READ;
    let port = crate::ports::open(crate::ports::KIND_RNG).expect("no virtio-rng device");
    let first = port_request(port, OP_READ, 64, 0, 12);
    let second = port_request(port, OP_READ, 64, 0, 13);
    assert!(first > 0 && second > 0, "rng request failed");
    let first = crate::ports::buf_take(first as u64).unwrap();
    let second = crate::ports::buf_take(second as u64).unwrap();
    assert_eq!(first.len(), 64, "rng returned a short buffer");
    assert_eq!(second.len(), 64, "rng returned a short buffer");
    assert!(first.iter().any(|byte| *byte != 0), "rng returned all zeroes");
    assert_ne!(first, second, "rng repeated a 512-bit output");
    assert_eq!(
        port_request(
            port,
            OP_READ,
            crate::ports::RNG_MAX_REQUEST as u64 + 1,
            0,
            14,
        ),
        -1,
        "rng accepted an oversized request"
    );
    println!("[ok] rng port: two distinct 512-bit CSPRNG reads");
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
    assert!(
        len > 0 && len < 1024 * 1024,
        "implausible module length {len}"
    );
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
        unsafe {
            (
                msg.tuple_elem(0),
                msg.tuple_elem(1).as_int().expect("non-int count"),
            )
        }
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
    assert!(
        proc::is_alive(server),
        "purge must spare migrated processes"
    );
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

// ---- M10: the Lux TCP/IP stack, compiled by Lux to Yggdrasil bytecode ----

/// Engine tier for the Lux modules (toggle for differential debugging).
const LUX_USE_JIT: bool = true;

/// Parse the luxpack (all content-addressed modules of examples/tcp_ip.lux),
/// load each one, then run the stack's own deterministic protocol suite —
/// checksum vectors, corruption rejection, a full handshake with data
/// transfer, and an orderly close — as a Yggdrasil process. `main/0` returns
/// the atom `true` iff every protocol scenario passed.
fn lux_tcp_stack() {
    let me = proc::current();
    // 16 MiB heap: packet codecs churn lists/binaries and there is no GC yet.
    proc::spawn_with_heap(lux_tcp_runner, Term::pid(me).0, 4096);
    let msg = proc::recv_timeout(120_000).expect("lux tcp stack never reported");
    assert_eq!(msg, atoms::atom("lux_tcp_ok"), "lux tcp/ip self-test failed");
    println!("[ok] lux tcp/ip stack: full protocol suite passed on yggdrasil");
}

fn lux_redmagic() {
    let me = proc::current();
    // TLS exercises the full pure-Lux SHA-256/HMAC/HKDF and AEAD stack. Keep its
    // allocation churn out of the coordinator process and give it room to grow.
    proc::spawn_with_heap(lux_redmagic_runner, Term::pid(me).0, 32768);
    let msg = proc::recv_timeout(180_000).expect("Redmagic self-test never reported");
    assert_eq!(
        msg,
        atoms::atom("redmagic_ok"),
        "Redmagic self-test failed"
    );
    println!("[ok] Redmagic: disk, CSPRNG, and pure-Lux TLS client/server passed");
}

extern "C" fn lux_redmagic_runner(parent_raw: u64) {
    let parent = Term(parent_raw).as_pid().unwrap();
    let (key, value) = proc::alloc_retry(|heap| {
        Ok((
            heap.binary(b"redmagic-selftest")?,
            heap.binary(b"persistent Lux data")?,
        ))
    });
    assert_eq!(
        lux_call_gc("storage_roundtrip", &[Term::int(64), key, value]),
        atoms::atom("true"),
        "Redmagic disk storage roundtrip failed"
    );

    let random = lux_call_gc("secure_random_bytes", &[Term::int(64)]);
    unsafe {
        assert!(random.is_boxed() && random.kind() == ygg_term::Kind::Binary);
        assert_eq!(random.bin_bytes().len(), 64, "Redmagic CSPRNG length mismatch");
    }

    assert_eq!(
        lux_call_gc("tls_profile_smoke", &[]),
        atoms::atom("true"),
        "Redmagic TLS 1.3 profile self-test failed"
    );
    proc::send(parent, atoms::atom("redmagic_ok"));
}

extern "C" fn lux_tcp_runner(parent_raw: u64) {
    let parent = Term(parent_raw).as_pid().unwrap();
    let pack = crate::modload::boot_module_bytes("tcp_ip.luxpack").expect("no luxpack");
    let entry = load_luxpack(pack).expect("bad luxpack");

    let module = crate::modload::current(&entry).expect("entry module missing");
    let fn_idx = module.module.function_named("apply").expect("no apply/0");
    let result = crate::modload::invoke(&module, fn_idx, &[]).expect("lux main trapped");
    assert_eq!(result, atoms::atom("true"), "lux tcp_ip:main() returned false");
    proc::send(parent, atoms::atom("lux_tcp_ok"));
}

/// Lux source_name -> content-addressed module hash, from the luxpack.
static LUX_ALIASES: spin::Mutex<BTreeMap<alloc::string::String, alloc::string::String>> =
    spin::Mutex::new(BTreeMap::new());

/// Resolve a Lux function's module by source name and invoke its `apply`.
fn lux_call(name: &str, args: &[Term]) -> Term {
    let hash = LUX_ALIASES.lock().get(name).cloned().unwrap_or_else(|| panic!("no lux fn {name}"));
    let module = crate::modload::current(&hash).expect("lux module missing");
    let fn_idx = module.module.function_named("apply").expect("no apply");
    crate::modload::invoke(&module, fn_idx, args)
        .unwrap_or_else(|t| panic!("lux {name} trapped: {t:?}"))
}

/// `lux_call` with trampoline-hop compaction. Only for callers holding no
/// process-heap terms across the call (immediate args, no live boxed state).
fn lux_call_gc(name: &str, args: &[Term]) -> Term {
    let hash = LUX_ALIASES.lock().get(name).cloned().unwrap_or_else(|| panic!("no lux fn {name}"));
    let module = crate::modload::current(&hash).expect("lux module missing");
    let fn_idx = module.module.function_named("apply").expect("no apply");
    crate::modload::invoke_gc(&module, fn_idx, args)
        .unwrap_or_else(|t| panic!("lux {name} trapped: {t:?}"))
}

/// `LUXPK1\n [u32 entry_len][entry][u32 count] count*([u32 nlen][name][u32 dlen][data])`
/// then `[u32 alias_count] count*([u32 len][source_name][u32 len][hash])`.
fn load_luxpack(bytes: &[u8]) -> Option<alloc::string::String> {
    let mut at = 0usize;
    let take = |at: &mut usize, n: usize| -> Option<&[u8]> {
        let s = bytes.get(*at..*at + n)?;
        *at += n;
        Some(s)
    };
    let u32at = |at: &mut usize| -> Option<usize> {
        Some(u32::from_le_bytes(take(at, 4)?.try_into().ok()?) as usize)
    };
    let strat = |at: &mut usize| -> Option<alloc::string::String> {
        let n = u32at(at)?;
        core::str::from_utf8(take(at, n)?).ok().map(alloc::string::String::from)
    };
    if take(&mut at, 7)? != b"LUXPK1\n" {
        return None;
    }
    let entry = strat(&mut at)?;
    let count = u32at(&mut at)?;
    for _ in 0..count {
        let name = strat(&mut at)?;
        let dlen = u32at(&mut at)?;
        let data = take(&mut at, dlen)?;
        // Content-addressed: sealed, so callers direct-bind to its code.
        crate::modload::load_sealed(&name, data, LUX_USE_JIT).ok()?;
    }
    let (hits, misses) = crate::modload::ext_bind_stats();
    crate::println!("[lux] direct-bound ext call sites: {hits} bound, {misses} dynamic");
    let alias_count = u32at(&mut at)?;
    let mut aliases = LUX_ALIASES.lock();
    for _ in 0..alias_count {
        let name = strat(&mut at)?;
        let hash = strat(&mut at)?;
        aliases.insert(name, hash);
    }
    Some(entry)
}

// ---- M11: tail calls + trampoline GC ----

/// 100k tail-recursive Lux iterations, each allocating garbage, inside the
/// *default* 256 KiB quota — impossible without both the trampoline (native
/// stack would overflow) and compaction (heap would cap out in ~700 rounds).
fn lux_bounded_loop() {
    let me = proc::current();
    proc::spawn_with_heap(lux_loop_runner, Term::pid(me).0, 64);
    let msg = proc::recv_timeout(120_000).expect("lux loop never finished");
    assert_eq!(msg, atoms::atom("lux_loop_ok"));
    println!("[ok] lux loop: 100k tail-recursive iterations in bounded memory");
}

extern "C" fn lux_loop_runner(parent_raw: u64) {
    let parent = Term(parent_raw).as_pid().unwrap();
    let r = lux_call_gc("loop_test", &[Term::int(100_000)]);
    // sum(1..=100000)
    assert_eq!(r.as_int(), Some(5_000_050_000), "loop result wrong");
    proc::send(parent, atoms::atom("lux_loop_ok"));
}

/// A Lux program owns the serial port and writes `LUX-PORT-OK` through
/// `PORT_SUBMIT2` — the whole Lux→port path with zero kernel driver code.
fn lux_port_hello() {
    let me = proc::current();
    proc::spawn_with_heap(lux_port_runner, Term::pid(me).0, 64);
    let msg = proc::recv_timeout(30_000).expect("lux port hello never finished");
    assert_eq!(msg, atoms::atom("lux_port_ok"));
    println!("[ok] lux port: serial written via PORT_SUBMIT2");
}

extern "C" fn lux_port_runner(parent_raw: u64) {
    let parent = Term(parent_raw).as_pid().unwrap();
    let r = lux_call("hello", &[]);
    assert_eq!(r, atoms::atom("true"), "port hello returned false");
    proc::send(parent, atoms::atom("lux_port_ok"));
}

/// A 2D display driver written entirely in Lux: it encodes every virtio-gpu
/// control command itself and pushes them through the raw transport port.
/// The harness screendumps the scanout and asserts the scene's pixels.
fn lux_gpu_demo() {
    let me = proc::current();
    proc::spawn_with_heap(lux_gpu_runner, Term::pid(me).0, 4096);
    let msg = proc::recv_timeout(120_000).expect("lux gpu demo never finished");
    assert_eq!(msg, atoms::atom("lux_gpu_ok"));
    println!("[ok] lux gpu: scene rendered via virtio-gpu");
    let msg = proc::recv_timeout(120_000).expect("lux gpu animation never finished");
    assert_eq!(msg, atoms::atom("lux_gpu_anim_ok"));
    println!("[ok] lux gpu: animation played via buf_write");
    // The harness screendumps the final animation frame on that marker; hold
    // the scanout so the font scene doesn't replace it mid-dump.
    let _ = proc::recv_timeout(1_000);
}

fn lux_font_scene() {
    let me = proc::current();
    proc::spawn_with_heap(lux_font_runner, Term::pid(me).0, 4096);
    let msg = proc::recv_timeout(120_000).expect("lux font scene never finished");
    assert_eq!(msg, atoms::atom("lux_font_ok"));
    println!("[ok] lux font: grayscale atlas rendered via virtio-gpu");
}

extern "C" fn lux_font_runner(parent_raw: u64) {
    let parent = Term(parent_raw).as_pid().unwrap();
    let r = lux_call_gc("font_scene", &[]);
    assert_eq!(r, atoms::atom("true"), "font_scene returned false");
    proc::send(parent, atoms::atom("lux_font_ok"));
}

fn lux_virgl_probe() {
    let me = proc::current();
    proc::spawn_with_heap(lux_virgl_runner, Term::pid(me).0, 1024);
    let msg = proc::recv_timeout(30_000).expect("lux virgl probe never finished");
    let code = msg.as_int().expect("virgl probe should return an int");
    assert!(code == 0 || code == 1 || code == 2, "unexpected virgl probe {code}");
    if code == 1 {
        println!("[ok] lux virgl: 3d context created");
    } else if code == 0 {
        println!("[ok] lux virgl: 3d not offered");
    } else {
        println!("[ok] lux virgl: capset present, context create failed");
    }
}

extern "C" fn lux_virgl_runner(parent_raw: u64) {
    let parent = Term(parent_raw).as_pid().unwrap();
    let r = lux_call_gc("virgl_probe", &[]);
    proc::send(parent, r);
}

/// `virglprobe` cmdline: load the luxpack and report whether the gpu
/// advertised a virgl capset and accepted CTX_CREATE.
pub extern "C" fn virgl_visual(_arg: u64) {
    let pack = crate::modload::boot_module_bytes("tcp_ip.luxpack").expect("no luxpack");
    load_luxpack(pack).expect("bad luxpack");
    let r = lux_call_gc("virgl_probe", &[]);
    match r.as_int() {
        Some(1) => println!("[virgl] context created"),
        Some(0) => println!("[virgl] 3d not offered"),
        Some(2) => println!("[virgl] capset present, context create failed"),
        other => println!("[virgl] unexpected {other:?}"),
    }
    crate::qemu::exit(crate::qemu::ExitCode::Success);
}

extern "C" fn lux_gpu_runner(parent_raw: u64) {
    let parent = Term(parent_raw).as_pid().unwrap();
    let r = lux_call_gc("drive", &[]);
    assert_eq!(r, atoms::atom("true"), "gpu drive returned false");
    proc::send(parent, atoms::atom("lux_gpu_ok"));
    // Give the harness time to screendump the static scene before the
    // animation replaces the scanout (it dumps within ~1s of the marker).
    let _ = proc::recv_timeout(5_000);
    let r = lux_call_gc("animate", &[Term::int(60)]);
    assert_eq!(r, atoms::atom("true"), "gpu animate returned false");
    proc::send(parent, atoms::atom("lux_gpu_anim_ok"));
}

/// `gpudemo` cmdline (see `cargo xtask watch`): run the Lux driver on a
/// visible display — the static scene, then the bouncing band, indefinitely.
pub extern "C" fn gpu_visual(_arg: u64) {
    let pack = crate::modload::boot_module_bytes("tcp_ip.luxpack").expect("no luxpack");
    load_luxpack(pack).expect("bad luxpack");
    assert_eq!(lux_call_gc("drive", &[]), atoms::atom("true"), "gpu drive failed");
    println!("[gpu] static scene up; animation starts in 2s");
    let _ = proc::recv_timeout(2_000);
    let _ = lux_call_gc("animate", &[Term::int(1_000_000)]);
}

pub extern "C" fn terminal_visual(_arg: u64) {
    let pack = crate::modload::boot_module_bytes("tcp_ip.luxpack").expect("no luxpack");
    load_luxpack(pack).expect("bad luxpack");
    proc::spawn_with_heap(tcp_repl_adapter, 0, 4096);
    println!("[terminal] Lux ANSI terminal starting; serial stdin and TCP REPL are active");
    let result = lux_call_gc("terminal", &[]);
    panic!("Lux ANSI terminal exited unexpectedly: {result:?}");
}

// ---- M11 C1 spike: native fill through the gpu transport port ----
// Throwaway validation that (a) the raw control-queue transport works and
// (b) headless `screendump` sees the scanout. Boot with `gpuspike`.

pub extern "C" fn gpu_spike(_arg: u64) {
    let port = crate::ports::open(crate::ports::KIND_GPU).expect("no gpu port");
    let pid = port.as_port().unwrap();
    let hdr = |ty: u32| -> alloc::vec::Vec<u8> {
        let mut v = alloc::vec::Vec::with_capacity(24);
        v.extend_from_slice(&ty.to_le_bytes());
        v.extend_from_slice(&[0u8; 20]); // flags, fence, ctx, ring+pad
        v
    };
    let rect = |v: &mut alloc::vec::Vec<u8>| {
        for x in [0u32, 0, 320, 200] {
            v.extend_from_slice(&x.to_le_bytes());
        }
    };
    let submit = |op: u32, cmd: alloc::vec::Vec<u8>, arg1: u64| -> alloc::vec::Vec<u8> {
        let id = crate::ports::buf_create(cmd);
        let sqe = ygg_rings::Sqe { op, tag: 7, arg0: id, arg1 };
        assert!(crate::ports::submit(pid, sqe), "gpu submit failed");
        let msg = proc::recv_timeout(5_000).expect("no gpu reply");
        let result = unsafe { msg.tuple_elem(3) }.as_int().expect("bad reply");
        assert!(result > 0, "gpu ctrl error: {result}");
        crate::ports::buf_take(result as u64).expect("resp buffer gone")
    };
    let ok = |resp: &[u8]| u32::from_le_bytes(resp[0..4].try_into().unwrap());

    // RESOURCE_CREATE_2D: id 1, format B8G8R8X8 (2), 320x200.
    let mut c = hdr(0x101);
    for x in [1u32, 2, 320, 200] {
        c.extend_from_slice(&x.to_le_bytes());
    }
    assert_eq!(ok(&submit(crate::ports::OP_CTRL, c, 24)), 0x1100);

    // Backing store: solid red XRGB (bytes B,G,R,X little-endian).
    let mut fb = alloc::vec::Vec::with_capacity(320 * 200 * 4);
    for _ in 0..320 * 200 {
        fb.extend_from_slice(&[0u8, 0, 255, 0]);
    }
    let backing = crate::ports::buf_create(fb);

    // ATTACH_BACKING: kernel appends the {phys, len} entry.
    let mut c = hdr(0x106);
    c.extend_from_slice(&1u32.to_le_bytes()); // resource id
    c.extend_from_slice(&1u32.to_le_bytes()); // nr_entries
    assert_eq!(ok(&submit(crate::ports::OP_CTRL_ATTACH, c, backing)), 0x1100);

    // SET_SCANOUT 0 -> resource 1.
    let mut c = hdr(0x103);
    rect(&mut c);
    c.extend_from_slice(&0u32.to_le_bytes());
    c.extend_from_slice(&1u32.to_le_bytes());
    assert_eq!(ok(&submit(crate::ports::OP_CTRL, c, 24)), 0x1100);

    // TRANSFER_TO_HOST_2D + FLUSH.
    let mut c = hdr(0x105);
    rect(&mut c);
    c.extend_from_slice(&0u64.to_le_bytes());
    c.extend_from_slice(&1u32.to_le_bytes());
    c.extend_from_slice(&0u32.to_le_bytes());
    assert_eq!(ok(&submit(crate::ports::OP_CTRL, c, 24)), 0x1100);
    let mut c = hdr(0x104);
    rect(&mut c);
    c.extend_from_slice(&1u32.to_le_bytes());
    c.extend_from_slice(&0u32.to_le_bytes());
    assert_eq!(ok(&submit(crate::ports::OP_CTRL, c, 24)), 0x1100);

    println!("[ok] gpu spike: scene flushed");
    // Stay alive so the harness can screendump.
    let _ = proc::recv_timeout(600_000);
}

// ---- M10 phase B: the Lux stack serving a real host TCP connection ----

/// The harness forwards host 127.0.0.1:17799 to guest 10.0.2.15:7 (slirp).
/// A native adapter owns the NIC: it answers ARP, unwraps IPv4/TCP packets
/// into the Lux stack's pure `input`, and transmits whatever the stack emits.
/// Data delivered by the stack is echoed back through `tcp_send`. The TCP
/// protocol work — handshake, ACKs, teardown — is all Lux bytecode.
fn lux_tcp_echo_live() {
    let me = proc::current();
    let adapter = proc::spawn_with_heap(tcp_echo_adapter, Term::pid(me).0, 4096);
    let msg = proc::recv_timeout(60_000).expect("live tcp echo never completed");
    assert_eq!(msg, atoms::atom("tcp_echo_served"));
    println!("[ok] lux tcp stack served a real host connection (SYN->echo)");
    proc::kill(adapter, "test finished");
}

const GUEST_IP: [u8; 4] = [10, 0, 2, 15];
const GUEST_TCP_PORT: u16 = 7;

fn tcp_listening(initial_sequence: i64) -> Term {
    let ip_term = proc::build(|h| {
        let mut pairs = [
            (atoms::atom("a"), Term::int(GUEST_IP[0] as i64)),
            (atoms::atom("b"), Term::int(GUEST_IP[1] as i64)),
            (atoms::atom("c"), Term::int(GUEST_IP[2] as i64)),
            (atoms::atom("d"), Term::int(GUEST_IP[3] as i64)),
        ];
        h.map_from_pairs(&mut pairs).map_err(|_| ygg_term::HeapFull)
    });
    lux_call_gc(
        "listening",
        &[ip_term, Term::int(GUEST_TCP_PORT as i64), Term::int(initial_sequence)],
    )
}

fn tcp_repl_send(
    port: Term,
    src_mac: [u8; 6],
    dst_mac: [u8; 6],
    connection: Term,
    payload: Term,
) -> Term {
    let step = lux_call_gc("tcp_send", &[connection, payload]);
    tx_outbound(port, src_mac, dst_mac, step_field(step, "outbound"));
    step_field(step, "connection")
}

fn tcp_repl_responses(delivered: Term, line: &mut Vec<u8>, overflowed: &mut bool) -> Vec<u8> {
    let input = unsafe {
        assert!(delivered.is_boxed() && delivered.kind() == ygg_term::Kind::Binary);
        delivered.bin_bytes().to_vec()
    };
    let mut responses = Vec::new();
    for byte in input {
        match byte {
            b'\r' => {}
            b'\n' => {
                if *overflowed {
                    responses.extend_from_slice(b"\r\nline too long (1024 byte limit)\r\n\r\nlux> ");
                } else {
                    let line_term = proc::build(|h| h.binary(line));
                    let reply = lux_call("repl_eval", &[line_term]);
                    let reply_bytes = unsafe {
                        assert!(reply.is_boxed() && reply.kind() == ygg_term::Kind::Binary);
                        reply.bin_bytes()
                    };
                    responses.extend_from_slice(reply_bytes);
                }
                line.clear();
                *overflowed = false;
            }
            8 | 127 => {
                line.pop();
            }
            _ if line.len() < 1024 => line.push(byte),
            _ => *overflowed = true,
        }
    }
    responses
}

/// Serve a single remote REPL connection at a time. Ethernet and ARP stay in
/// this adapter; the TCP state machine and expression evaluator are Lux.
extern "C" fn tcp_repl_adapter(_arg: u64) {
    let port = crate::ports::open(crate::ports::KIND_NET).expect("no virtio-net");
    let mac = crate::virtio::net_mac();
    let mut initial_sequence = 50_000i64;
    let mut connection = tcp_listening(initial_sequence);
    let mut line = Vec::new();
    let mut overflowed = false;
    println!(
        "[tcp-repl] listening on {}.{}.{}.{}:{} (host 127.0.0.1:17888)",
        GUEST_IP[0], GUEST_IP[1], GUEST_IP[2], GUEST_IP[3], GUEST_TCP_PORT
    );

    loop {
        let frame = net_rx_wait(port);
        if frame.len() >= 42 && frame[12..14] == [0x08, 0x06] && frame[20..22] == [0, 1] {
            if frame[38..42] == GUEST_IP {
                let mut reply = Vec::with_capacity(60);
                reply.extend_from_slice(&frame[6..12]);
                reply.extend_from_slice(&mac);
                reply.extend_from_slice(&[0x08, 0x06, 0, 1, 8, 0, 6, 4, 0, 2]);
                reply.extend_from_slice(&mac);
                reply.extend_from_slice(&GUEST_IP);
                reply.extend_from_slice(&frame[22..28]);
                reply.extend_from_slice(&frame[28..32]);
                reply.resize(60, 0);
                net_tx(port, reply);
            }
            continue;
        }
        if frame.len() < 34 || frame[12..14] != [0x08, 0x00] || frame[23] != 6 {
            continue;
        }
        let peer_mac: [u8; 6] = frame[6..12].try_into().unwrap();
        let ip_len = u16::from_be_bytes([frame[16], frame[17]]) as usize;
        if frame.len() < 14 + ip_len {
            continue;
        }

        let packet_term = proc::build(|h| h.binary(&frame[14..14 + ip_len]));
        let step = lux_call_gc("input", &[connection, packet_term]);
        connection = step_field(step, "connection");
        let event = step_field(step, "event").as_int().unwrap_or(0);
        tx_outbound(port, mac, peer_mac, step_field(step, "outbound"));

        if event == 1 {
            println!("[tcp-repl] client connected");
            line.clear();
            overflowed = false;
            let banner = lux_call("repl_banner", &[]);
            connection = tcp_repl_send(port, mac, peer_mac, connection, banner);
        } else if event == 2 {
            let responses =
                tcp_repl_responses(step_field(step, "delivered"), &mut line, &mut overflowed);
            if !responses.is_empty() {
                let response = proc::build(|h| h.binary(&responses));
                connection = tcp_repl_send(port, mac, peer_mac, connection, response);
            }
        } else if event == 3 {
            let close_step = lux_call_gc("close", &[connection]);
            tx_outbound(port, mac, peer_mac, step_field(close_step, "outbound"));
            // This is a single-client development REPL, so return to Listen as
            // soon as the peer's FIN has been acknowledged. The final ACK for
            // our FIN is harmless in Listen and a missing ACK cannot strand the
            // service in LastAck forever.
            initial_sequence += 10_000;
            connection = tcp_listening(initial_sequence);
            line.clear();
            overflowed = false;
            println!("[tcp-repl] client disconnected; listening");
        } else if event == 4 || event == 5 {
            initial_sequence += 10_000;
            connection = tcp_listening(initial_sequence);
            line.clear();
            overflowed = false;
            println!("[tcp-repl] connection reset; listening");
        }
    }
}

extern "C" fn tcp_echo_adapter(parent_raw: u64) {
    let parent = Term(parent_raw).as_pid().unwrap();
    let port = crate::ports::open(crate::ports::KIND_NET).expect("no virtio-net");
    let mac = crate::virtio::net_mac();

    // conn = tcp_ip:listening(#{a..d}, 7, InitialSeq)
    let ip_term = proc::build(|h| {
        let mut pairs = [
            (atoms::atom("a"), Term::int(GUEST_IP[0] as i64)),
            (atoms::atom("b"), Term::int(GUEST_IP[1] as i64)),
            (atoms::atom("c"), Term::int(GUEST_IP[2] as i64)),
            (atoms::atom("d"), Term::int(GUEST_IP[3] as i64)),
        ];
        h.map_from_pairs(&mut pairs).map_err(|_| ygg_term::HeapFull)
    });
    let mut conn = lux_call(
        "listening",
        &[ip_term, Term::int(GUEST_TCP_PORT as i64), Term::int(12345)],
    );
    println!("[tcp] listening on {}.{}.{}.{}:{}", GUEST_IP[0], GUEST_IP[1], GUEST_IP[2], GUEST_IP[3], GUEST_TCP_PORT);

    let mut served = false;
    loop {
        let frame = net_rx(port);
        // ARP request for our IP?
        if frame.len() >= 42 && frame[12..14] == [0x08, 0x06] && frame[20..22] == [0, 1] {
            if frame[38..42] == GUEST_IP {
                let mut reply = Vec::with_capacity(42);
                reply.extend_from_slice(&frame[6..12]); // dst = requester
                reply.extend_from_slice(&mac);
                reply.extend_from_slice(&[0x08, 0x06, 0, 1, 8, 0, 6, 4, 0, 2]);
                reply.extend_from_slice(&mac);
                reply.extend_from_slice(&GUEST_IP);
                reply.extend_from_slice(&frame[22..28]); // requester mac
                reply.extend_from_slice(&frame[28..32]); // requester ip
                reply.resize(60, 0);
                net_tx(port, reply);
            }
            continue;
        }
        // IPv4/TCP to us?
        if frame.len() >= 34 && frame[12..14] == [0x08, 0x00] && frame[23] == 6 {
            let peer_mac: [u8; 6] = frame[6..12].try_into().unwrap();
            let ip_len = u16::from_be_bytes([frame[16], frame[17]]) as usize;
            if frame.len() < 14 + ip_len {
                continue;
            }
            let packet = &frame[14..14 + ip_len];
            let pkt_term = proc::build(|h| h.binary(packet));
            let step = lux_call("input", &[conn, pkt_term]);
            conn = step_field(step, "connection");
            let event = step_field(step, "event").as_int().unwrap_or(0);
            tx_outbound(port, mac, peer_mac, step_field(step, "outbound"));
            if event == 2 {
                // Data delivered: echo it back through the stack.
                let delivered = step_field(step, "delivered");
                let step2 = lux_call("tcp_send", &[conn, delivered]);
                conn = step_field(step2, "connection");
                tx_outbound(port, mac, peer_mac, step_field(step2, "outbound"));
                if !served {
                    served = true;
                    proc::send(parent, atoms::atom("tcp_echo_served"));
                }
            }
            if event == 3 {
                // Peer closed: close our side too and keep ACKing the rest.
                let step3 = lux_call("close", &[conn]);
                conn = step_field(step3, "connection");
                tx_outbound(port, mac, peer_mac, step_field(step3, "outbound"));
            }
        }
    }
}

fn step_field(step: Term, name: &str) -> Term {
    unsafe {
        assert!(step.is_boxed() && step.kind() == ygg_term::Kind::Map, "TcpStep is not a map");
        step.map_get(atoms::atom(name)).unwrap_or_else(|| panic!("TcpStep missing {name}"))
    }
}

/// Blocking frame receive through the port rings.
fn net_rx(port: Term) -> Vec<u8> {
    use crate::ports::OP_READ;
    let r = port_request(port, OP_READ, 0, 0, 40);
    assert!(r > 0, "net rx failed");
    crate::ports::buf_take(r as u64).unwrap()
}

/// Blocking receive for long-lived network services. Unlike the self-test
/// helper above, an idle connection is not treated as a failure.
fn net_rx_wait(port: Term) -> Vec<u8> {
    use crate::ports::OP_READ;
    let r = port_request_wait(port, OP_READ, 0, 0, 42);
    assert!(r > 0, "net rx failed");
    crate::ports::buf_take(r as u64).unwrap()
}

fn net_tx(port: Term, frame: Vec<u8>) {
    use crate::ports::OP_WRITE;
    let buf = crate::ports::buf_create(frame);
    assert_eq!(port_request(port, OP_WRITE, 0, buf, 41), 0, "net tx failed");
}

/// Wrap a stack-emitted IPv4 packet (binary term) in ethernet and transmit.
fn tx_outbound(port: Term, src_mac: [u8; 6], dst_mac: [u8; 6], outbound: Term) {
    let payload = unsafe {
        assert!(outbound.is_boxed() && outbound.kind() == ygg_term::Kind::Binary);
        outbound.bin_bytes()
    };
    if payload.is_empty() {
        return;
    }
    let mut frame = Vec::with_capacity(14 + payload.len());
    frame.extend_from_slice(&dst_mac);
    frame.extend_from_slice(&src_mac);
    frame.extend_from_slice(&[0x08, 0x00]);
    frame.extend_from_slice(payload);
    if frame.len() < 60 {
        frame.resize(60, 0);
    }
    net_tx(port, frame);
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
    assert_eq!(
        proc::recv().as_int(),
        Some(1),
        "skipped messages must stay in order"
    );
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
    assert!(
        !p.ecam.is_empty(),
        "no ECAM (MCFG) — q35 should have MMCONFIG"
    );
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
