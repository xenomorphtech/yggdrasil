# Yggdrasil

A BEAM-like operating system: the organizing abstractions are **processes and
memory**, not files. Everything runs in ring 0 — isolation comes from a
bytecode verifier and JIT-generated code (Singularity-style software-isolated
processes), not the MMU privilege split.

BEAM where it's proven, modernized where technology moved on:

| | |
|---|---|
| **Processes** | Green threads with isolated term heaps, own native stacks (goroutine model), guard pages. A crashing/overflowing/quota-breaching process dies alone. |
| **Messaging** | Copy-on-send via heap fragments (no core ever writes another core's heap), mailboxes, selective receive, `receive … after` via timer wheel. |
| **Supervision** | Links propagate exit signals (hop-by-hop at reap); monitors deliver `{'DOWN', Ref, Pid, Reason}`. |
| **Preemption** | Per-core LAPIC timers set per-CPU preempt flags; interpreter back-edges and JIT-emitted safepoints yield. Timer interrupts never context-switch. |
| **SMP** | Limine MP bring-up, per-CPU run queues with work stealing, wake IPIs, TLB-shootdown IPIs for stack recycling, panic halts all cores. |
| **Ports** | Every device is an SQ/CQ ring pair owned by a process (the shape virtio/NVMe/io_uring converged on). Completions arrive as ordinary messages. Serial (IRQ-driven), virtio-blk, virtio-net. |
| **Code** | Custom register bytecode, load-time **verifier** (the security boundary), two-version **hot code loading** with `call_ext` migration and `purge`. |
| **Execution** | Tier 0: interpreter. Tier 1: **Cranelift JIT compiled in-kernel** (no_std cranelift 0.134), published into an RX code zone with hand-rolled relocation patching. Differentially tested against the interpreter. |

## Layout

```
kernel/           the kernel binary (boot, mm, vmm, proc, ports, irq, virtio, jit publisher)
crates/
  ygg-alloc       frame bitmap + pure allocators        (host-tested)
  ygg-term        tagged terms, bump heaps, copy-on-send (host-tested)
  ygg-bytecode    instruction set, module format, VERIFIER (host-tested + mutation-fuzzed)
  ygg-interp      tier-0 interpreter behind SystemApi   (host-tested vs mock)
  ygg-jit         tier-1 bytecode -> Cranelift          (host-tested by executing JIT output)
  ygg-rings       SPSC SQ/CQ rings                      (host-tested incl. cross-thread)
modules/          .yasm test/demo programs (assembled at build time)
tools/
  ygg-asm         text assembler -> .yggm
  xtask           ISO build (Limine) + QEMU run/test harness
```

## Running

Requires: nightly Rust (pinned via `rust-toolchain.toml`), `qemu-system-x86_64`,
`xorriso`, `git`, `make`/`cc` (for the Limine host tool, fetched automatically).

```sh
cargo xtask run    # boot in QEMU q35, serial on stdio
cargo xtask test   # full acceptance suite: two boots + pcap assertion
cargo test         # host tests for all pure crates
```

The test suite boots the OS, runs in-kernel selftests (memory, timers,
processes, supervision, quotas, ports, verifier, hot loading, JIT
differential), types `PING` at a bytecode echo server over the emulated
serial line, writes a disk pattern, **reboots** to verify persistence, and
greps the guest's UDP payload out of a pcap of the virtual NIC.

## The Lux TCP/IP stack

Yggdrasil runs a real TCP/IP stack written in [Lux](../lux) (`examples/tcp_ip.lux`),
compiled by Lux's Yggdrasil backend into 54 content-addressed, verified bytecode
modules and JIT-compiled in-kernel. A native adapter owns the NIC (ethernet +
ARP); every TCP protocol decision — handshake, checksums, ACKs, teardown — is
Lux bytecode calling through `CALL_EXT`. The test suite proves it end to end: a
real host TCP client connects through QEMU's user-net (`hostfwd`), and the Lux
stack accepts the connection and echoes the payload back.

Supporting ops added for this: bit ops (`band/bor/bxor/bsl/bsr/bnot`), binaries
(`BIN_NEW/FROM_LIST/TO_LIST/SIZE/CAT/PART`, `IS_BINARY`), flat maps with
immediate keys (`MAP_NEW/GET/PUT` — the struct representation, same shape BEAM
uses below 33 keys), `LIST_CAT`, and the packet-buffer bridge
(`BUF_TO_BIN`/`BIN_TO_BUF`). `tools/ygg-run` executes a luxpack on the host
under either engine (`--jit` mmaps and runs the Cranelift output in userspace)
for fast differential debugging.

## Status

Milestones M0–M10 complete (boot → memory → processes → full semantics →
bytecode/interpreter → ports → virtio → verifier → hot loading → Cranelift
JIT → Lux TCP/IP stack → **SMP**). The suite runs on 4 cores: Limine MP
bring-up, per-CPU state behind a `gs:[0]` accessor (run queues, preempt flags,
scheduler contexts, TSS/IST), BEAM-style heap *fragments* so no core ever
writes another core's process heap, work stealing with wake IPIs, TLB
shootdown for stack-slot recycling, and tests that only pass with true
parallelism (an unpreemptible spinner released by concurrently-running code).
Known deferrals: per-process GC (heaps are fixed-quota bump arenas;
`copy_term` is the seed of the future semispace collector), floats, arrays as
a first-class term kind.
