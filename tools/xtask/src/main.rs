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

#[derive(Clone, Copy)]
enum QemuIo {
    TcpRepl,
    Full,
}

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
        "spike" => {
            gpu_spike(&root)?;
        }
        "watch" => {
            // Boot the Lux ANSI terminal on a visible display. QEMU's serial
            // stdio is its input stream, the virtio-gpu window is its output,
            // and host port 17888 forwards to the Lux TCP REPL.
            // gtk,gl=on + virtio-gpu-gl opens a window but the scanout stays
            // black on this host. virtio-gpu-pci + gtk is the path that
            // actually paints the GTK window.
            let iso = make_iso(&root, "terminal")?;
            let status = qemu_command(&root, &iso, false, "gtk", QemuIo::TcpRepl, "virtio-gpu-pci")?
                .status()?;
            println!("qemu exited: {status}");
        }
        "termshot" => {
            termshot(&root)?;
        }
        "virgl" => {
            virgl_probe(&root)?;
        }
        _ => bail!("usage: cargo xtask <build|iso|run|test|watch|termshot|spike|virgl>"),
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
            let out = out_dir
                .join(path.file_stem().unwrap())
                .with_extension("yggm");
            sh(
                "cargo",
                &[
                    "run",
                    "-q",
                    "-p",
                    "ygg-asm",
                    "--",
                    path.to_str().unwrap(),
                    out.to_str().unwrap(),
                ],
            )?;
            outs.push(out);
        }
    }
    Ok(outs)
}

/// Compile the Lux TCP/IP stack (sibling checkout) to Yggdrasil modules and
/// pack them into one blob the kernel can load:
/// `LUXPK1\n [u32 entry_len][entry][u32 count] count*([u32 nlen][name][u32 dlen][data])`.
fn build_luxpack(root: &Path) -> Result<PathBuf> {
    let lux_root = root.parent().unwrap().join("lux");
    let redmagic_root = root.parent().unwrap().join("redmagic");
    const PROGRAMS: &[&str] = &[
        "tcp_ip",
        "lux_loop",
        "port_hello",
        "gpu_demo",
        "ansi_terminal",
        "tcp_repl",
        "font",
        "font_demo",
        "virgl",
        "disk_storage",
        "random",
        "tls13_psk",
        "tls_client",
        "tls_server",
    ];
    let artifacts = lux_root.join("target/lux/artifacts");
    let mut entry = String::new();
    let mut module_list: Vec<(String, String)> = Vec::new(); // (name, file)
    let mut aliases: Vec<(String, String)> = Vec::new();
    for stem in PROGRAMS {
        let source = redmagic_root.join("programs").join(format!("{stem}.lux"));
        ensure!(source.is_file(), "missing Redmagic program {}", source.display());
        let status = Command::new("cargo")
            .args(["run", "-q", "--", "--yggdrasil"])
            .arg(&source)
            .current_dir(&lux_root)
            .status()
            .context("running the Lux compiler for Redmagic")?;
        ensure!(status.success(), "lux --yggdrasil failed for {stem}");
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(artifacts.join(format!("{stem}.ygg.json")))?)?;
        if *stem == "tcp_ip" {
            entry = manifest["entry"]["module"]
                .as_str()
                .context("manifest has no entry module")?
                .to_string();
        }
        for m in manifest["modules"].as_array().context("manifest has no modules")? {
            let name = m["name"].as_str().context("module without name")?;
            let file = m["file"].as_str().context("module without file")?;
            if !module_list.iter().any(|(n, _)| n == name) {
                module_list.push((name.to_string(), file.to_string()));
            }
        }
        for f in manifest["aliases"].as_array().context("manifest has no aliases")? {
            if let (Some(name), Some(module)) = (f["name"].as_str(), f["module"].as_str()) {
                if let Some((_, current)) = aliases.iter_mut().find(|(alias, _)| alias == name) {
                    *current = module.to_string();
                } else {
                    aliases.push((name.to_string(), module.to_string()));
                }
            }
        }
    }

    let mut pack: Vec<u8> = b"LUXPK1\n".to_vec();
    pack.extend_from_slice(&(entry.len() as u32).to_le_bytes());
    pack.extend_from_slice(entry.as_bytes());
    pack.extend_from_slice(&(module_list.len() as u32).to_le_bytes());
    for (name, file) in &module_list {
        let data = std::fs::read(artifacts.join(file))?;
        pack.extend_from_slice(&(name.len() as u32).to_le_bytes());
        pack.extend_from_slice(name.as_bytes());
        pack.extend_from_slice(&(data.len() as u32).to_le_bytes());
        pack.extend_from_slice(&data);
    }
    pack.extend_from_slice(&(aliases.len() as u32).to_le_bytes());
    for (name, module) in &aliases {
        pack.extend_from_slice(&(name.len() as u32).to_le_bytes());
        pack.extend_from_slice(name.as_bytes());
        pack.extend_from_slice(&(module.len() as u32).to_le_bytes());
        pack.extend_from_slice(module.as_bytes());
    }

    let out = root.join("build/tcp_ip.luxpack");
    std::fs::create_dir_all(out.parent().unwrap())?;
    std::fs::write(&out, pack)?;
    println!("luxpack: {} modules, {} aliases, entry {entry}", module_list.len(), aliases.len());
    Ok(out)
}

fn make_iso(root: &Path, cmdline: &str) -> Result<PathBuf> {
    let kernel = build_kernel()?;
    let modules = build_modules(root)?;
    let luxpack = build_luxpack(root)?;
    let limine_dir = ensure_limine(root)?;

    let iso_root = root.join("build/iso_root");
    let boot = iso_root.join("boot");
    let efi = iso_root.join("EFI/BOOT");
    let _ = std::fs::remove_dir_all(&iso_root);
    std::fs::create_dir_all(boot.join("limine"))?;
    std::fs::create_dir_all(&efi)?;

    std::fs::copy(&kernel, boot.join("yggdrasil"))?;

    let mut conf = String::from(
        "timeout: 0\n\n/Yggdrasil\n    protocol: limine\n    path: boot():/boot/yggdrasil\n",
    );
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
    std::fs::copy(&luxpack, boot.join("tcp_ip.luxpack"))?;
    conf.push_str("    module_path: boot():/boot/tcp_ip.luxpack\n");
    std::fs::write(boot.join("limine/limine.conf"), conf)?;

    for f in [
        "limine-bios.sys",
        "limine-bios-cd.bin",
        "limine-uefi-cd.bin",
    ] {
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
    let module =
        std::fs::read(root.join("build/modules/hotmod.yggm")).context("hotmod.yggm not built")?;
    let disk = ensure_disk(root)?;
    let mut img = std::fs::OpenOptions::new().write(true).open(&disk)?;
    img.seek(SeekFrom::Start(2048 * 512))?;
    img.write_all(&(module.len() as u32).to_le_bytes())?;
    img.write_all(&module)?;
    Ok(())
}

fn qemu_command(
    root: &Path,
    iso: &Path,
    capture: bool,
    display: &str,
    io: QemuIo,
    gpu_device: &str,
) -> Result<Command> {
    // gtk,gl=on needs an X display. Headless harnesses wrap QEMU in Xvfb so
    // monitor screendump can read the GL scanout.
    let wrap_xvfb = capture && display.contains("gtk");
    let mut cmd = if wrap_xvfb {
        let mut wrapped = Command::new("xvfb-run");
        wrapped.args(["-a", "-s", "-screen 0 1024x768x24", "qemu-system-x86_64"]);
        wrapped.env("LIBGL_ALWAYS_SOFTWARE", "1");
        // New process group so termshot can kill Xvfb + QEMU together.
        // Do not do this for `watch`: a background group stops on SIGTTIN
        // when `-serial stdio` reads the terminal, so no GTK window appears.
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            wrapped.process_group(0);
        }
        wrapped
    } else {
        Command::new("qemu-system-x86_64")
    };
    // KVM when /dev/kvm is usable (near-native guest speed); TCG otherwise.
    let kvm = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/kvm")
        .is_ok();
    cmd.args(["-accel", if kvm { "kvm" } else { "tcg" }]);
    cmd.args([
        "-M",
        "q35",
        "-cpu",
        "qemu64,+x2apic",
        "-smp",
        "4",
        "-m",
        "512M",
        "-cdrom",
        iso.to_str().unwrap(),
        "-display",
        display,
        "-device",
        "isa-debug-exit,iobase=0xf4,iosize=0x04",
        "-no-reboot",
        "-serial",
        "stdio",
        "-object",
        "rng-random,filename=/dev/urandom,id=rng0",
        "-device",
        "virtio-rng-pci,rng=rng0",
    ]);
    if matches!(io, QemuIo::Full) {
        let disk = ensure_disk(root)?;
        cmd.args([
            "-drive",
            &format!("file={},if=none,id=d0,format=raw", disk.display()),
            "-device",
            "virtio-blk-pci,drive=d0",
        ]);
    }
    if matches!(io, QemuIo::Full | QemuIo::TcpRepl) {
        let hostfwd = match io {
            QemuIo::TcpRepl => "user,id=n0,hostfwd=tcp:127.0.0.1:17888-:7",
            _ => "user,id=n0,hostfwd=tcp:127.0.0.1:17799-:7",
        };
        cmd.args([
            "-netdev",
            hostfwd,
            "-device",
            "virtio-net-pci,netdev=n0",
        ]);
    }
    if matches!(io, QemuIo::Full) {
        let pcap = root.join("build/net.pcap");
        cmd.args([
            "-object",
            &format!("filter-dump,id=f0,netdev=n0,file={}", pcap.display()),
        ]);
    }
    cmd.args([
        "-vga",
        "none",
        "-device",
        gpu_device,
        "-monitor",
        &format!(
            "unix:{},server,nowait",
            root.join("build/mon.sock").display()
        ),
    ]);
    if capture {
        cmd.stdout(Stdio::piped()).stdin(Stdio::piped());
    }
    Ok(cmd)
}

fn kill_qemu(child: &mut std::process::Child) {
    let _ = Command::new("kill")
        .args(["-9", "--", &format!("-{}", child.id())])
        .status();
    let _ = child.kill();
    let _ = child.wait();
}

fn run_qemu(root: &Path, iso: &Path) -> Result<()> {
    let status = qemu_command(root, iso, false, "none", QemuIo::Full, "virtio-gpu-pci")?.status()?;
    // isa-debug-exit makes any exit code nonzero; don't treat that as failure here.
    println!("qemu exited: {status}");
    Ok(())
}

/// Issue `screendump` over the QEMU monitor socket and parse the P6 PPM.
/// Returns (width, height, rgb bytes).
fn monitor_screendump(root: &Path) -> Result<(usize, usize, Vec<u8>)> {
    use std::io::{Read, Write};
    use std::os::unix::net::UnixStream;

    let ppm = root.join("build/gpu.ppm");
    let _ = std::fs::remove_file(&ppm);
    let mut sock = UnixStream::connect(root.join("build/mon.sock"))
        .context("connecting to qemu monitor socket")?;
    // QEMU 11 HMP treats '/' in an unquoted absolute path as an expression.
    // A workspace-relative path is what the 2D termshot already used.
    sock.write_all(b"screendump build/gpu.ppm\n")?;
    sock.flush()?;
    // Drain the monitor greeting/echo until the file shows up.
    sock.set_read_timeout(Some(std::time::Duration::from_millis(200)))?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let mut reply = Vec::new();
    let mut sink = [0u8; 1024];
    while std::time::Instant::now() < deadline {
        if let Ok(n) = sock.read(&mut sink) {
            reply.extend_from_slice(&sink[..n]);
        }
        if ppm.exists() && std::fs::metadata(&ppm)?.len() > 0 {
            std::thread::sleep(std::time::Duration::from_millis(200)); // let it finish
            break;
        }
    }
    let data = std::fs::read(&ppm).with_context(|| {
        format!(
            "reading screendump ppm (monitor said: {})",
            String::from_utf8_lossy(&reply)
        )
    })?;
    ensure!(data.starts_with(b"P6"), "screendump is not a P6 ppm");
    // P6\n<width> <height>\n<maxval>\n<binary rgb>
    let mut fields = Vec::new();
    let mut at = 2usize;
    while fields.len() < 3 {
        while at < data.len() && data[at].is_ascii_whitespace() {
            at += 1;
        }
        let start = at;
        while at < data.len() && !data[at].is_ascii_whitespace() {
            at += 1;
        }
        fields.push(std::str::from_utf8(&data[start..at])?.parse::<usize>()?);
    }
    at += 1; // the single whitespace after maxval
    let (w, h) = (fields[0], fields[1]);
    ensure!(data.len() >= at + w * h * 3, "ppm truncated");
    Ok((w, h, data[at..at + w * h * 3].to_vec()))
}

/// C1 spike: boot the native gpu fill and verify headless screendump works.
fn gpu_spike(root: &Path) -> Result<()> {
    use std::sync::{Arc, Mutex};

    let iso = make_iso(root, "gpuspike")?;
    let mut child =
        qemu_command(root, &iso, true, "none", QemuIo::Full, "virtio-gpu-pci")?
            .spawn()
            .context("spawning qemu")?;
    let mut stdout = child.stdout.take().unwrap();
    let captured: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
    {
        let captured = captured.clone();
        std::thread::spawn(move || {
            let mut chunk = [0u8; 4096];
            loop {
                match stdout.read(&mut chunk) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => captured.lock().unwrap().extend_from_slice(&chunk[..n]),
                }
            }
        });
    }
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    loop {
        ensure!(std::time::Instant::now() < deadline, "gpu spike marker never appeared");
        let text = String::from_utf8_lossy(&captured.lock().unwrap()).into_owned();
        if text.contains("[ok] gpu spike: scene flushed") {
            break;
        }
        if text.contains("KERNEL PANIC") {
            let _ = child.kill();
            bail!("kernel panicked:\n{text}");
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
    let dump = monitor_screendump(root);
    let _ = child.kill();
    let (w, h, rgb) = dump?;
    println!("screendump: {w}x{h}");
    let px = |x: usize, y: usize| -> (u8, u8, u8) {
        let i = (y * w + x) * 3;
        (rgb[i], rgb[i + 1], rgb[i + 2])
    };
    println!("center pixel: {:?}", px(w / 2, h / 2));
    ensure!(px(w / 2, h / 2) == (255, 0, 0), "expected solid red at center");
    println!("gpu spike: PASS");
    Ok(())
}

/// Boot with virtio-gpu-gl and prove the Lux virgl library can create a 3D context.
fn virgl_probe(root: &Path) -> Result<()> {
    use std::sync::{Arc, Mutex};

    let iso = make_iso(root, "virglprobe")?;
    let mut child = qemu_command(
        root,
        &iso,
        true,
        "egl-headless,gl=on",
        QemuIo::TcpRepl,
        "virtio-gpu-gl-pci",
    )?
    .spawn()
    .context("spawning qemu for virgl probe")?;
    let mut stdout = child.stdout.take().unwrap();
    let captured: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
    {
        let captured = captured.clone();
        std::thread::spawn(move || {
            let mut chunk = [0u8; 4096];
            loop {
                match stdout.read(&mut chunk) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => captured.lock().unwrap().extend_from_slice(&chunk[..n]),
                }
            }
        });
    }
    let deadline = Instant::now() + Duration::from_secs(90);
    let text = loop {
        if Instant::now() > deadline {
            let _ = child.kill();
            let transcript = String::from_utf8_lossy(&captured.lock().unwrap()).into_owned();
            bail!("virgl probe timed out:\n{transcript}");
        }
        if let Some(status) = child.try_wait()? {
            let transcript = String::from_utf8_lossy(&captured.lock().unwrap()).into_owned();
            ensure!(
                status.code() == Some(EXIT_SUCCESS),
                "virgl qemu exited {status}:\n{transcript}"
            );
            break transcript;
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    println!("{text}");
    ensure!(
        text.contains("[virgl] context created"),
        "virgl probe did not create a 3D context"
    );
    println!("virgl probe: PASS");
    Ok(())
}

/// Headless dump of the Lux terminal scanout. Used to verify watch would
/// show text, not a black framebuffer.
fn termshot(root: &Path) -> Result<()> {
    use std::sync::{Arc, Mutex};

    let iso = make_iso(root, "terminal")?;
    let mut child = qemu_command(root, &iso, true, "gtk,gl=on", QemuIo::TcpRepl, "virtio-gpu-gl-pci")?
        .spawn()
        .context("spawning qemu for termshot")?;
    let mut stdout = child.stdout.take().unwrap();
    let captured: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
    {
        let captured = captured.clone();
        std::thread::spawn(move || {
            let mut chunk = [0u8; 4096];
            loop {
                match stdout.read(&mut chunk) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => captured.lock().unwrap().extend_from_slice(&chunk[..n]),
                }
            }
        });
    }
    let deadline = Instant::now() + Duration::from_secs(90);
    let mut visible = false;
    loop {
        if Instant::now() > deadline {
            kill_qemu(&mut child);
            let transcript = String::from_utf8_lossy(&captured.lock().unwrap()).into_owned();
            bail!("termshot timed out:\n{transcript}");
        }
        let text = String::from_utf8_lossy(&captured.lock().unwrap()).into_owned();
        if text.contains("KERNEL PANIC") || text.contains("exited unexpectedly") {
            kill_qemu(&mut child);
            bail!("terminal crashed:\n{text}");
        }
        // GL scanouts have no HMP surface. The guest reads the colorbuf back
        // and prints whether the banner glyphs actually landed.
        if text.contains("[terminal] gpu text visible") {
            visible = true;
            break;
        }
        if text.contains("[terminal] gpu text blank") || text.contains("[terminal] readback failed") {
            kill_qemu(&mut child);
            bail!("terminal GPU compositor produced no visible text:\n{text}");
        }
        if text.contains("[terminal] gpu setup failed") {
            kill_qemu(&mut child);
            bail!("terminal gpu setup failed:\n{text}");
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    kill_qemu(&mut child);
    let transcript = String::from_utf8_lossy(&captured.lock().unwrap()).into_owned();
    println!("{transcript}");
    ensure!(visible, "termshot never saw GPU text");
    println!("termshot: PASS (virgl compositor drew banner glyphs)");
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

    let mut child = qemu_command(root, &iso, true, "none", QemuIo::Full, "virtio-gpu-pci")?
        .spawn()
        .context("spawning qemu")?;
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

    // Drive the interactive parts:
    //  1. when the Lux TCP adapter listens, connect through slirp's hostfwd
    //     and expect our bytes echoed back by the Lux stack;
    //  2. when the bytecode serial echo server says it's ready, type PING.
    let mut typed = false;
    let mut gpu_dump: Option<Result<(usize, usize, Vec<u8>)>> = None;
    let mut anim_dump: Option<Result<(usize, usize, Vec<u8>)>> = None;
    let mut font_dump: Option<Result<(usize, usize, Vec<u8>)>> = None;
    let mut tcp_thread: Option<std::thread::JoinHandle<Result<()>>> = None;
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        {
            let snapshot = captured.lock().unwrap().clone();
            let text = String::from_utf8_lossy(&snapshot).into_owned();
            // The Lux gpu driver leaves its scene on scanout 0; grab it over
            // the monitor while the guest is still running.
            if gpu_dump.is_none() && text.contains("[ok] lux gpu: scene rendered via virtio-gpu")
            {
                gpu_dump = Some(monitor_screendump(root));
            }
            if anim_dump.is_none() && text.contains("[ok] lux gpu: animation played via buf_write")
            {
                anim_dump = Some(monitor_screendump(root));
            }
            if font_dump.is_none()
                && text.contains("[ok] lux font: grayscale atlas rendered via virtio-gpu")
            {
                font_dump = Some(monitor_screendump(root));
            }
            if tcp_thread.is_none() && text.contains("[tcp] listening") {
                tcp_thread = Some(std::thread::spawn(|| {
                    use std::io::{Read as _, Write as _};
                    let mut s = std::net::TcpStream::connect(("127.0.0.1", 17799))
                        .context("connecting to hostfwd")?;
                    s.set_read_timeout(Some(Duration::from_secs(30)))?;
                    s.write_all(b"PING-TCP")?;
                    let mut buf = [0u8; 8];
                    s.read_exact(&mut buf).context("reading echo")?;
                    ensure!(&buf == b"PING-TCP", "echo mismatch: {buf:?}");
                    println!("host tcp client: echo verified");
                    Ok(())
                }));
            }
            if !typed && text.contains("[bc] echo_ready") {
                stdin
                    .write_all(b"PING\n")
                    .context("typing into qemu serial")?;
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
    match tcp_thread {
        Some(t) => t.join().unwrap().context("host-side tcp echo failed")?,
        None => {
            println!("--- serial transcript ---\n{transcript}-------------------------");
            bail!("the Lux TCP adapter never announced it was listening");
        }
    }
    println!("--- serial transcript ---\n{transcript}-------------------------");

    let code = status.code().context("qemu killed by signal")?;
    ensure!(
        code == EXIT_SUCCESS,
        "kernel exited with {code}, expected {EXIT_SUCCESS} (success marker)"
    );
    let markers: &[&str] = &[
        "Yggdrasil",
        "[smp] 4 cpus online",
        "[ok] limine requests answered",
        "[ok] int3 dispatched and recovered",
        "[ok] pmm alloc/free/contig",
        "[ok] kernel heap (vec/btree/box)",
        "[ok] acpi:",
        "[ok] lapic timer: 3 s of monotonic 1 kHz ticks",
        "[ok] process ping-pong (5 rounds)",
        "[ok] preemptive scheduling (two non-yielding busy loops both advanced)",
        "[ok] stack overflow killed only the offender",
        "[ok] smp: two cores executed simultaneously",
        "[ok] smp: work stealing spread load across cpus",
        "[ok] smp: 200-process churn with TLB shootdown",
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
        "[ok] lux tcp/ip stack: full protocol suite passed on yggdrasil",
        "[ok] lux loop: 100k tail-recursive iterations in bounded memory",
        "LUX-PORT-OK",
        "[ok] lux port: serial written via PORT_SUBMIT2",
        "[ok] lux gpu: scene rendered via virtio-gpu",
        "[ok] lux gpu: animation played via buf_write",
        "[ok] lux font: grayscale atlas rendered via virtio-gpu",
        "[ok] lux virgl:",
        "[ok] lux tcp stack served a real host connection (SYN->echo)",
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

    // Assert the Lux-rendered scene pixel-by-pixel: three bands + the
    // centered white rectangle (B8G8R8X8 written by the driver -> RGB here).
    let (w, h, rgb) = gpu_dump
        .context("gpu marker never appeared before qemu exit")?
        .context("screendump failed")?;
    ensure!((w, h) == (320, 200), "unexpected scanout {w}x{h}");
    let px = |x: usize, y: usize| -> (u8, u8, u8) {
        let i = (y * w + x) * 3;
        (rgb[i], rgb[i + 1], rgb[i + 2])
    };
    ensure!(px(160, 33) == (255, 0, 0), "top band not red: {:?}", px(160, 33));
    ensure!(px(160, 73) == (0, 255, 0), "middle band not green: {:?}", px(160, 73));
    ensure!(px(160, 100) == (255, 255, 255), "center rect not white: {:?}", px(160, 100));
    ensure!(px(10, 100) == (0, 255, 0), "rect margin not green: {:?}", px(10, 100));
    ensure!(px(160, 166) == (0, 0, 255), "bottom band not blue: {:?}", px(160, 166));
    println!("gpu screendump: scene verified at 5 probe points");

    // The animation's last frame: 60 frames starting at y=0, +2/frame, no
    // erase after the final draw -> white band rows 118..157 on the dark
    // blue background (bytes b,g,r = 96,32,32 -> RGB 32,32,96).
    let (w, h, rgb) = anim_dump
        .context("animation marker never appeared before qemu exit")?
        .context("animation screendump failed")?;
    ensure!((w, h) == (320, 200), "unexpected animated scanout {w}x{h}");
    let px = |x: usize, y: usize| -> (u8, u8, u8) {
        let i = (y * w + x) * 3;
        (rgb[i], rgb[i + 1], rgb[i + 2])
    };
    ensure!(px(160, 138) == (255, 255, 255), "band not at rest: {:?}", px(160, 138));
    ensure!(px(160, 60) == (32, 32, 96), "background above band: {:?}", px(160, 60));
    ensure!(px(160, 185) == (32, 32, 96), "background below band: {:?}", px(160, 185));
    println!("gpu screendump: animation final frame verified");

    let (w, h, rgb) = font_dump
        .context("font marker never appeared before qemu exit")?
        .context("font screendump failed")?;
    ensure!((w, h) == (384, 52), "unexpected font scanout {w}x{h}");
    let mut gray = 0usize;
    let mut ink = 0usize;
    for px in rgb.chunks_exact(3) {
        let (r, g, b) = (px[0], px[1], px[2]);
        if (r, g, b) == (18, 18, 22) {
            continue;
        }
        ink += 1;
        if r > 18 && r < 236 && g > 18 && g < 236 && b > 22 && b < 236 {
            gray += 1;
        }
    }
    ensure!(ink > 200, "font scene has too little ink ({ink} lit pixels)");
    ensure!(gray > 20, "font scene has no antialiased coverage ({gray} gray pixels)");
    println!("gpu screendump: font atlas verified ({ink} lit, {gray} gray AA pixels)");

    // The guest's UDP payload must have hit the wire. Check before phase 2,
    // which truncates the pcap when QEMU restarts.
    let pcap = std::fs::read(root.join("build/net.pcap")).context("reading net.pcap")?;
    ensure!(
        pcap.windows(b"YGG-NET-OK".len())
            .any(|w| w == b"YGG-NET-OK"),
        "UDP payload not found in pcap"
    );

    // Phase 2: reboot with `verify-disk` — the pattern must persist.
    let iso2 = make_iso(root, "verify-disk")?;
    let out = qemu_command(root, &iso2, true, "none", QemuIo::Full, "virtio-gpu-pci")?
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
