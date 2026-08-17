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
| **GC + tail calls** | `TAIL_CALL`/`TAIL_CALL_EXT` unwind to an engine trampoline (constant native stack); each outermost hop is an exact GC point where the stashed args are the whole live set — Cheney compaction with forwarding pointers preserves sharing. Heaps grow segmented (non-moving) up to quota between collections. |
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
cargo xtask watch    # Lux ANSI terminal + TCP REPL (host port 17888)
cargo xtask termshot # headless dump of that terminal (asserts AA pixels)
cargo xtask virgl  # virtio-gpu-gl: Lux creates a virgl 3D context
cargo test         # host tests for all pure crates
```

The test suite boots the OS, runs in-kernel selftests (memory, timers,
processes, supervision, quotas, ports, verifier, hot loading, JIT
differential), types `PING` at a bytecode echo server over the emulated
serial line, writes a disk pattern, **reboots** to verify persistence, and
greps the guest's UDP payload out of a pcap of the virtual NIC.

## Redmagic system library

Yggdrasil's Lux programs live in the sibling [Redmagic](../redmagic) project,
the system standard library. The xtask build compiles every program in
`../redmagic/programs/` with the sibling Lux compiler and installs one
content-addressed pack in the boot image. This includes the device examples,
TCP/IP stack, ANSI terminal, TCP REPL, disk storage, kernel-backed secure
random adapter, and the deliberately narrow pure-Lux TLS 1.3 client/server.

## The Lux TCP/IP stack

Yggdrasil runs a real TCP/IP stack written in [Lux](../lux)
([`programs/tcp_ip.lux`](../redmagic/programs/tcp_ip.lux)),
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

## Devices from Lux: ports as the whole driver surface

Lux programs reach hardware through the raw port surface — `ygg::port_open`,
`ygg::port_submit` (`PORT_SUBMIT2`: all-register, exposes both SQE args),
`ygg::buf_to_bin`/`bin_to_buf` — with completions arriving as ordinary
`{port_reply, Port, Tag, Result}` messages. Alongside it sit **mutable
fixed-size kernel blobs** (`ygg::buf_new`/`buf_write`/`buf_read`) — the niche
BEAM fills with atomics/ETS: off-heap, GC-immune, stable physical address.
A framebuffer is just a blob the device is also attached to, so a driver
animates by writing rows in place. `receive { after N => … }` lowers to a
timer-wheel sleep for frame pacing. The suite proves it all:

- `port_hello.lux` owns the serial port and writes `LUX-PORT-OK` byte-by-byte.
- `gpu_demo.lux` is a **complete virtio-gpu 2D display driver in Lux**: it
  encodes every control command itself (display info, `RESOURCE_CREATE_2D`,
  `ATTACH_BACKING`, `SET_SCANOUT`, `TRANSFER_TO_HOST_2D`, `RESOURCE_FLUSH`),
  renders a scene into a framebuffer binary, and pushes it all through the
  kernel's **gpu transport port** (`KIND_GPU`). The kernel owns the control
  virtqueue but never interprets a command: `OP_CTRL` ships an opaque buffer
  and returns the response buffer; `OP_CTRL_ATTACH` is the single
  protocol-shaped assist — it appends the `{phys, len}` mem-entry for a pinned
  backing buffer, because guest physical addresses must never be visible to
  bytecode. The harness screendumps QEMU headlessly and asserts the rendered
  pixels — then the driver **animates**: a paced tail-recursive frame loop
  (`buf_write` band → transfer → flush → sleep) whose final frame is
  screendump-verified too. `cargo xtask watch` shows it live in a GTK window.
  (`cargo xtask spike` runs the equivalent native fill — useful to isolate
  transport bugs from driver bugs.)

## Lux ANSI terminal

`cargo xtask watch` boots Redmagic's `programs/ansi_terminal.lux`, a 40x25 terminal whose
implementation is entirely Lux above the existing raw device-port surface.
The shell running QEMU is the serial input stream and the GTK virtio-gpu
window is the display. The terminal has a reusable Alacritty-style grayscale font (`use font` in
Lux: Adwaita Mono coverage atlas with integer alpha blending), sixteen VGA
colors, a visible cursor, wrapping, tabs, backspace, scrolling, and local
echo. It accepts the common VT100/ANSI sequences:

- cursor movement and position: `CSI A/B/C/D/E/F/G/H/f`
- erase display/line: `CSI J/K` (modes 0, 1, and 2)
- SGR reset, bold, reverse, and 16 colors: `CSI m`
- save/restore cursor: `CSI s/u` and `ESC 7/8`

For example, press `Ctrl-[` to enter ESC, then type `[31mred`; enter another
ESC with `Ctrl-[` and type `[0m` to reset the color.

The same boot also starts a line-oriented TCP REPL backed by the Lux TCP/IP
stack and a Lux expression evaluator. Connect from another shell with:

```sh
nc 127.0.0.1 17888
```

The REPL evaluates integer expressions with `+`, `-`, `*`, and parentheses.
It also accepts `help`, `about`, `colors`, and `clear`. It handles input split
across TCP packets, accepts another client after disconnect, and emits ANSI
formatting suitable for a normal host terminal.

**What this unlocks (virgl):** a 3D/virgl driver is *the same surface*. The
reusable `use virgl` library encodes `CTX_CREATE`, capset queries, 3D
resource creation, `SUBMIT_3D`, and a virgl ccw helper. The kernel negotiates
the VIRGL feature bit when the device offers it. `cargo xtask watch` runs
`-device virtio-gpu-gl-pci` so those commands have a real 3D backend; the
selftest suite stays on 2D `virtio-gpu-pci`. Completions are still
synchronous/poll-based through the port pump — fence-driven async completion
is a later upgrade, not an architectural change.

## Status

Milestones M0–M11 complete (boot → memory → processes → full semantics →
bytecode/interpreter → ports → virtio → verifier → hot loading → Cranelift
JIT → Lux TCP/IP stack → **SMP** → **GC + tail calls + Lux device drivers**).
The suite runs on 4 cores: Limine MP bring-up, per-CPU state behind a `gs:[0]`
accessor (run queues, preempt flags, scheduler contexts, TSS/IST), BEAM-style
heap *fragments* so no core ever writes another core's process heap, work
stealing with wake IPIs, TLB shootdown for stack-slot recycling, and tests
that only pass with true parallelism. M11's acceptance: 100k tail-recursive
Lux iterations in bounded memory, and the Lux gpu driver's scene verified
pixel-by-pixel from a headless screendump.

Known limits/deferrals: compaction runs only at *outermost* trampoline hops
(nested calls and native drivers hold live terms the collector cannot see, so
they only grow — BEAM-style anywhere-GC needs stack maps); floats; arrays as
a first-class term kind; fence/interrupt-driven gpu completion.
