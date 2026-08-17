#![no_std]
#![no_main]
#![feature(abi_x86_interrupt)]

extern crate alloc;

mod acpi_tables;
mod atoms;
mod boot;
mod gdt;
mod idt;
mod irq;
mod jit;
mod mm;
mod modload;
mod percpu;
mod ports;
mod proc;
mod qemu;
mod selftest;
mod serial;
mod smp;
mod virtio;
mod vmm;

use core::fmt::Write;
use core::panic::PanicInfo;

use limine::memmap::MEMMAP_USABLE;

#[unsafe(no_mangle)]
extern "C" fn kmain() -> ! {
    serial::init();
    println!();
    println!("Yggdrasil 0.1.0 — BEAM-like OS, booting");

    assert!(
        boot::BASE_REVISION.is_supported(),
        "limine base revision {} not supported by bootloader",
        limine::BaseRevision::MAX_SUPPORTED
    );

    // Per-CPU table + gs binding first: exception/interrupt paths use cpu().
    let (cpu_count, bsp_lapic) = boot::MP
        .response()
        .map(|r| (r.cpus().len().max(1), r.bsp_lapic_id))
        .unwrap_or((1, 0));
    percpu::init(cpu_count, bsp_lapic);
    percpu::bind(percpu::all()[0]);
    gdt::init();
    idt::init();

    report_boot_info();

    mm::init();
    acpi_tables::init();
    irq::init();
    vmm::init();
    irq::init_devices();
    virtio::init();
    smp::init();
    x86_64::instructions::interrupts::enable();

    let has_arg = |w: &str| boot::cmdline().split_whitespace().any(|a| a == w);
    if has_arg("selftest") {
        selftest::early();
        proc::spawn(selftest::proc_tests, 0);
    } else if has_arg("gpuspike") {
        proc::spawn(selftest::gpu_spike, 0);
    } else if has_arg("gpudemo") {
        // The Lux driver builds whole-frame binaries; give it a real heap.
        proc::spawn_with_heap(selftest::gpu_visual, 0, 4096);
    } else if has_arg("terminal") {
        // The Lux terminal owns both serial input and the virtio-gpu scanout.
        proc::spawn_with_heap(selftest::terminal_visual, 0, 4096);
    } else if has_arg("virglprobe") {
        proc::spawn_with_heap(selftest::virgl_visual, 0, 1024);
    } else if has_arg("verify-disk") {
        proc::spawn(selftest::verify_disk, 0);
    } else {
        println!("boot complete; starting init process");
        proc::spawn(init_process, 0);
    }
    proc::run()
}

/// Placeholder init: keeps the system visibly alive until real modules exist.
extern "C" fn init_process(_arg: u64) {
    let mut last_s = 0;
    loop {
        x86_64::instructions::hlt();
        proc::safepoint();
        let s = irq::ticks() / 1000;
        if s > last_s {
            last_s = s;
            println!("[time] uptime {s} s");
        }
    }
}

fn report_boot_info() {
    let hhdm = boot::HHDM.response().expect("limine: no HHDM response");
    println!("[boot] hhdm offset      = {:#x}", hhdm.offset);

    if let Some(addr) = boot::EXECUTABLE_ADDRESS.response() {
        println!(
            "[boot] kernel phys/virt = {:#x} / {:#x}",
            addr.physical_base, addr.virtual_base
        );
    }

    let memmap = boot::MEMMAP.response().expect("limine: no memmap response");
    let entries = memmap.entries();
    let usable: u64 = entries
        .iter()
        .filter(|e| e.type_ == MEMMAP_USABLE)
        .map(|e| e.length)
        .sum();
    println!(
        "[boot] memmap           = {} entries, {} MiB usable",
        entries.len(),
        usable / (1024 * 1024)
    );

    match boot::RSDP.response() {
        Some(rsdp) => println!("[boot] rsdp             = {:p}", rsdp.address),
        None => println!("[boot] rsdp             = (none)"),
    }

    let cmdline = boot::cmdline();
    if !cmdline.is_empty() {
        println!("[boot] cmdline          = {:?}", cmdline);
    }
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    smp::halt_others();
    x86_64::instructions::interrupts::disable();
    let mut out = unsafe { serial::force_writer() };
    let _ = writeln!(out, "\nKERNEL PANIC: {info}");
    qemu::exit(qemu::ExitCode::Failure);
}
