//! Build/run harness: `cargo xtask <build|iso|run|test>`.
//!
//! - build: compile the kernel for x86_64-unknown-none
//! - iso:   build a hybrid BIOS+UEFI bootable ISO with Limine
//! - run:   boot the ISO in QEMU q35, serial on stdio
//! - test:  boot with `selftest` on the cmdline, assert serial markers + exit code

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};

const LIMINE_GIT: &str = "https://github.com/limine-bootloader/limine.git";
const LIMINE_BRANCH: &str = "v11.x-binary";
const QEMU_TIMEOUT: Duration = Duration::from_secs(180);
/// isa-debug-exit: (0x10 << 1) | 1
const EXIT_SUCCESS: i32 = 33;

fn main() -> Result<()> {
    let root = workspace_root();
    std::env::set_current_dir(&root)?;
    let cmd = std::env::args().nth(1).unwrap_or_default();
    match cmd.as_str() {
        "build" => {
            build_kernel()?;
        }
        "iso" => {
            make_iso(&root, "")?;
        }
        "run" => {
            let iso = make_iso(&root, "")?;
            run_qemu(&root, &iso)?;
        }
        "test" => {
            test(&root)?;
        }
        _ => bail!("usage: cargo xtask <build|iso|run|test>"),
    }
    Ok(())
}

fn workspace_root() -> PathBuf {
    // tools/xtask -> workspace root
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap()
        .to_path_buf()
}

fn sh(program: &str, args: &[&str]) -> Result<()> {
    let status = Command::new(program)
        .args(args)
        .status()
        .with_context(|| format!("failed to spawn {program}"))?;
    ensure!(status.success(), "{program} {args:?} failed: {status}");
    Ok(())
}

fn build_kernel() -> Result<PathBuf> {
    sh(
        "cargo",
        &[
            "build",
            "-p",
            "ygg-kernel",
            "--target",
            "x86_64-unknown-none",
            "--release",
        ],
    )?;
    Ok(PathBuf::from(
        "target/x86_64-unknown-none/release/ygg-kernel",
    ))
}

/// Fetch the Limine binary release and build its host `limine` tool once.
fn ensure_limine(root: &Path) -> Result<PathBuf> {
    let dir = root.join("third_party/limine");
    if !dir.join("limine-bios.sys").exists() {
        sh(
            "git",
            &[
                "clone",
                LIMINE_GIT,
                "--branch",
                LIMINE_BRANCH,
                "--depth=1",
                dir.to_str().unwrap(),
            ],
        )?;
    }
    if !dir.join("limine").exists() {
        sh("make", &["-C", dir.to_str().unwrap()])?;
    }
    Ok(dir)
}

/// Assemble every modules/*.yasm into build/modules/*.yggm.
fn build_modules(root: &Path) -> Result<Vec<PathBuf>> {
    let out_dir = root.join("build/modules");
    std::fs::create_dir_all(&out_dir)?;
    let mut outs = Vec::new();
    for entry in std::fs::read_dir(root.join("modules"))? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "yasm") {
            let out = out_dir.join(path.file_stem().unwrap()).with_extension("yggm");
            sh(
                "cargo",
                &["run", "-q", "-p", "ygg-asm", "--", path.to_str().unwrap(), out.to_str().unwrap()],
            )?;
            outs.push(out);
        }
    }
    Ok(outs)
}

fn make_iso(root: &Path, cmdline: &str) -> Result<PathBuf> {
    let kernel = build_kernel()?;
    let modules = build_modules(root)?;
    let limine_dir = ensure_limine(root)?;

    let iso_root = root.join("build/iso_root");
    let boot = iso_root.join("boot");
    let efi = iso_root.join("EFI/BOOT");
    let _ = std::fs::remove_dir_all(&iso_root);
    std::fs::create_dir_all(boot.join("limine"))?;
    std::fs::create_dir_all(&efi)?;

    std::fs::copy(&kernel, boot.join("yggdrasil"))?;

    let mut conf = String::from("timeout: 0\n\n/Yggdrasil\n    protocol: limine\n    path: boot():/boot/yggdrasil\n");
    if !cmdline.is_empty() {
        conf.push_str(&format!("    cmdline: {cmdline}\n"));
    }
    for m in &modules {
        let name = m.file_name().unwrap().to_str().unwrap();
        // hotmod is deliberately NOT a boot module: the kernel fetches it from
        // the storage port at runtime (M6 acceptance).
        if name.starts_with("hotmod") {
            continue;
        }
        std::fs::copy(m, boot.join(name))?;
        conf.push_str(&format!("    module_path: boot():/boot/{name}\n"));
    }
    std::fs::write(boot.join("limine/limine.conf"), conf)?;

    for f in ["limine-bios.sys", "limine-bios-cd.bin", "limine-uefi-cd.bin"] {
        std::fs::copy(limine_dir.join(f), boot.join("limine").join(f))?;
    }
    for f in ["BOOTX64.EFI", "BOOTIA32.EFI"] {
        std::fs::copy(limine_dir.join(f), efi.join(f))?;
    }

    let iso = root.join("build/yggdrasil.iso");
    sh(
        "xorriso",
        &[
            "-as",
            "mkisofs",
            "-quiet",
            "-R",
            "-r",
            "-J",
            "-b",
            "boot/limine/limine-bios-cd.bin",
            "-no-emul-boot",
            "-boot-load-size",
            "4",
            "-boot-info-table",
            "-hfsplus",
            "-apm-block-size",
            "2048",
            "--efi-boot",
            "boot/limine/limine-uefi-cd.bin",
            "-efi-boot-part",
            "--efi-boot-image",
            "--protective-msdos-label",
            iso_root.to_str().unwrap(),
            "-o",
            iso.to_str().unwrap(),
        ],
    )?;
    sh(
        limine_dir.join("limine").to_str().unwrap(),
        &["bios-install", iso.to_str().unwrap()],
    )?;
    Ok(iso)
}

/// Create the (persistent) test disk if it doesn't exist: 8 MiB of zeroes.
fn ensure_disk(root: &Path) -> Result<PathBuf> {
    let disk = root.join("build/disk.img");
    if !disk.exists() {
        std::fs::create_dir_all(disk.parent().unwrap())?;
        std::fs::write(&disk, vec![0u8; 8 * 1024 * 1024])?;
    }
    Ok(disk)
}

/// Write hotmod.yggm into the disk image at sector 2048 as [u32 len][bytes] —
/// the kernel loads it from there at runtime through the storage port.
fn embed_hotmod(root: &Path) -> Result<()> {
    use std::io::{Seek, SeekFrom, Write};
    let module = std::fs::read(root.join("build/modules/hotmod.yggm"))
        .context("hotmod.yggm not built")?;
    let disk = ensure_disk(root)?;
    let mut img = std::fs::OpenOptions::new().write(true).open(&disk)?;
    img.seek(SeekFrom::Start(2048 * 512))?;
    img.write_all(&(module.len() as u32).to_le_bytes())?;
    img.write_all(&module)?;
    Ok(())
}

fn qemu_command(root: &Path, iso: &Path, capture: bool) -> Result<Command> {
    let disk = ensure_disk(root)?;
    let pcap = root.join("build/net.pcap");
    let mut cmd = Command::new("qemu-system-x86_64");
    cmd.args([
        "-M",
        "q35",
        "-cpu",
        "qemu64,+x2apic",
        "-m",
        "512M",
        "-cdrom",
        iso.to_str().unwrap(),
        "-display",
        "none",
        "-device",
        "isa-debug-exit,iobase=0xf4,iosize=0x04",
        "-no-reboot",
        "-serial",
        "stdio",
        "-drive",
        &format!("file={},if=none,id=d0,format=raw", disk.display()),
        "-device",
        "virtio-blk-pci,drive=d0",
        "-netdev",
        "user,id=n0",
        "-device",
        "virtio-net-pci,netdev=n0",
        "-object",
        &format!("filter-dump,id=f0,netdev=n0,file={}", pcap.display()),
    ]);
    if capture {
        cmd.stdout(Stdio::piped()).stdin(Stdio::piped());
    }
    Ok(cmd)
}

fn run_qemu(root: &Path, iso: &Path) -> Result<()> {
    let status = qemu_command(root, iso, false)?.status()?;
    // isa-debug-exit makes any exit code nonzero; don't treat that as failure here.
    println!("qemu exited: {status}");
    Ok(())
}

fn test(root: &Path) -> Result<()> {
    use std::io::Write;
    use std::sync::{Arc, Mutex};

    // Fresh disk + pcap per test run; the disk must survive between the two
    // boot phases below, so it's only recreated here.
    let _ = std::fs::remove_file(root.join("build/disk.img"));
    let _ = std::fs::remove_file(root.join("build/net.pcap"));

    let iso = make_iso(root, "selftest")?;
    embed_hotmod(root)?;

    let mut child = qemu_command(root, &iso, true)?.spawn().context("spawning qemu")?;
    let mut stdout = child.stdout.take().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let captured: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
    let reader = {
        let captured = captured.clone();
        std::thread::spawn(move || {
            let mut chunk = [0u8; 4096];
            loop {
                match stdout.read(&mut chunk) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => captured.lock().unwrap().extend_from_slice(&chunk[..n]),
                }
            }
        })
    };

    // Drive the interactive part: when the bytecode echo server says it's
    // ready, type PING at it over the (virtual) serial line.
    let mut typed = false;
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if !typed {
            let snapshot = captured.lock().unwrap().clone();
            if String::from_utf8_lossy(&snapshot).contains("[bc] echo_ready") {
                stdin.write_all(b"PING\n").context("typing into qemu serial")?;
                stdin.flush()?;
                typed = true;
            }
        }
        if start.elapsed() > QEMU_TIMEOUT {
            child.kill()?;
            child.wait()?;
            let transcript = String::from_utf8_lossy(&captured.lock().unwrap()).into_owned();
            println!("--- serial transcript (timeout) ---\n{transcript}");
            bail!("qemu timed out after {QEMU_TIMEOUT:?}");
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    reader.join().unwrap();
    let transcript = String::from_utf8_lossy(&captured.lock().unwrap()).into_owned();
    println!("--- serial transcript ---\n{transcript}-------------------------");

    let code = status.code().context("qemu killed by signal")?;
    ensure!(
        code == EXIT_SUCCESS,
        "kernel exited with {code}, expected {EXIT_SUCCESS} (success marker)"
    );
    let markers: &[&str] = &[
        "Yggdrasil",
        "[ok] limine requests answered",
        "[ok] int3 dispatched and recovered",
        "[ok] pmm alloc/free/contig",
        "[ok] kernel heap (vec/btree/box)",
        "[ok] acpi:",
        "[ok] lapic timer: 3 s of monotonic 1 kHz ticks",
        "[ok] process ping-pong (5 rounds)",
        "[ok] preemptive scheduling (two non-yielding busy loops both advanced)",
        "[ok] stack overflow killed only the offender",
        "[ok] terms: build/copy-on-send/eq/format",
        "[ok] supervisor: 3 restarts via DOWN messages",
        "[ok] exit propagation over links (parent died with child's reason)",
        "[ok] heap quota breach killed the process, not the kernel",
        "[ok] receive-after: empty mailbox timed out",
        "[ok] selective receive picked the matching message first",
        "[ok] blk port: wrote and read back sector 1 via rings",
        "[ok] net port: ARP resolved gateway, UDP payload sent",
        "[ok] verifier rejected malformed modules (no crash)",
        "[ok] module loaded at runtime from the storage port and spawned",
        "[ok] hot upgrade v1->v2: state retained, new format, purge killed the holdout",
        "[ok] differential: interpreter and Cranelift JIT agree on mathmod",
        "[bc] 101",
        "[bc] 105",
        "[ok] bytecode ping-pong under interpreter (busy loop preempted at back-edges)",
        "PING",
        "[ok] serial port echo via SQ/CQ rings",
        "[selftest] all passed",
    ];
    for marker in markers {
        ensure!(transcript.contains(marker), "missing marker: {marker:?}");
    }

    // The guest's UDP payload must have hit the wire. Check before phase 2,
    // which truncates the pcap when QEMU restarts.
    let pcap = std::fs::read(root.join("build/net.pcap")).context("reading net.pcap")?;
    ensure!(
        pcap.windows(b"YGG-NET-OK".len()).any(|w| w == b"YGG-NET-OK"),
        "UDP payload not found in pcap"
    );

    // Phase 2: reboot with `verify-disk` — the pattern must persist.
    let iso2 = make_iso(root, "verify-disk")?;
    let out = qemu_command(root, &iso2, true)?
        .stdin(Stdio::null())
        .output()
        .context("running verify-disk boot")?;
    let transcript2 = String::from_utf8_lossy(&out.stdout).into_owned();
    println!("--- verify-disk transcript ---\n{transcript2}------------------------------");
    ensure!(
        out.status.code() == Some(EXIT_SUCCESS),
        "verify-disk boot exited with {:?}",
        out.status.code()
    );
    ensure!(
        transcript2.contains("[ok] blk persistence verified after reboot"),
        "missing persistence marker"
    );

    println!("xtask test: PASS");
    Ok(())
}
