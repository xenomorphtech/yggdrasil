//! Local APIC and the 1 kHz system timer.
//!
//! The timer handler only bumps the tick count (and, from M2, sets the preempt
//! flag) — it never context-switches. All suspension happens at safepoints.

use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use x86_64::instructions::port::Port;
use x86_64::structures::idt::InterruptStackFrame;

pub const TIMER_VECTOR: u8 = 0x40;
/// Wake an idle (hlt-parked) core; handler is EOI-only.
pub const WAKE_VECTOR: u8 = 0x41;
/// TLB shootdown: reload CR3, ack, EOI.
pub const TLB_FLUSH_VECTOR: u8 = 0x42;
/// Panic path: stop all other cores.
pub const HALT_VECTOR: u8 = 0x43;
pub const SERIAL_VECTOR: u8 = 0x44;
pub const SPURIOUS_VECTOR: u8 = 0xFF;

/// Acks for an in-flight TLB shootdown.
pub static TLB_ACKS: AtomicUsize = AtomicUsize::new(0);

/// Milliseconds since timer start (1 kHz).
static TICKS: AtomicU64 = AtomicU64::new(0);
/// BSP-calibrated LAPIC timer ticks per millisecond (divide 16).
static TICKS_PER_MS: AtomicUsize = AtomicUsize::new(0);
/// 0 = x2APIC (MSR access); nonzero = xAPIC MMIO base (HHDM-virtual).
///
/// x2APIC is preferred because MSR access needs no page mapping — Limine's HHDM
/// covers only memory-map regions, not the LAPIC MMIO page. The xAPIC path
/// becomes usable on real hardware once M2 builds our own page tables.
static LAPIC_MMIO: AtomicUsize = AtomicUsize::new(0);

pub fn ticks() -> u64 {
    TICKS.load(Ordering::Relaxed)
}

mod reg {
    pub const TPR: usize = 0x80;
    pub const EOI: usize = 0xB0;
    pub const SVR: usize = 0xF0;
    pub const LVT_TIMER: usize = 0x320;
    pub const TIMER_INIT: usize = 0x380;
    pub const TIMER_CURRENT: usize = 0x390;
    pub const TIMER_DIVIDE: usize = 0x3E0;
}

const IA32_APIC_BASE: u32 = 0x1B;
const X2APIC_MSR_BASE: u32 = 0x800;

fn lapic_write(reg: usize, value: u32) {
    let mmio = LAPIC_MMIO.load(Ordering::Relaxed);
    unsafe {
        if mmio == 0 {
            x86_64::registers::model_specific::Msr::new(X2APIC_MSR_BASE + (reg >> 4) as u32)
                .write(value as u64);
        } else {
            ((mmio + reg) as *mut u32).write_volatile(value);
        }
    }
}

fn lapic_read(reg: usize) -> u32 {
    let mmio = LAPIC_MMIO.load(Ordering::Relaxed);
    unsafe {
        if mmio == 0 {
            x86_64::registers::model_specific::Msr::new(X2APIC_MSR_BASE + (reg >> 4) as u32).read()
                as u32
        } else {
            ((mmio + reg) as *const u32).read_volatile()
        }
    }
}

pub fn eoi() {
    lapic_write(reg::EOI, 0);
}

fn has_x2apic() -> bool {
    let leaf = unsafe { core::arch::x86_64::__cpuid(1) };
    leaf.ecx & (1 << 21) != 0
}

pub fn init() {
    let platform = crate::acpi_tables::platform();

    unsafe {
        let mut msr = x86_64::registers::model_specific::Msr::new(IA32_APIC_BASE);
        let v = msr.read();
        if has_x2apic() {
            // EN (bit 11) + EXTD (bit 10): x2APIC mode, registers via MSRs.
            msr.write(v | (1 << 11) | (1 << 10));
            LAPIC_MMIO.store(0, Ordering::Relaxed);
            log::info!("lapic: x2apic mode");
        } else {
            msr.write(v | (1 << 11));
            let base = crate::mm::phys_to_virt(platform.lapic_phys) as usize;
            LAPIC_MMIO.store(base, Ordering::Relaxed);
            // The HHDM does not cover MMIO; until M2 maps the LAPIC page in our
            // own tables, xAPIC access would fault.
            panic!("xAPIC MMIO unsupported until M2 page tables (no x2apic on this CPU)");
        }
    }

    // Software-enable the LAPIC, accept all priorities.
    lapic_write(reg::SVR, 0x100 | SPURIOUS_VECTOR as u32);
    lapic_write(reg::TPR, 0);

    let per_ms = calibrate_timer();
    TICKS_PER_MS.store(per_ms as usize, Ordering::Relaxed);
    log::info!("lapic: timer calibrated, {per_ms} ticks/ms (divide 16)");

    // Periodic (bit 17), 1 kHz.
    lapic_write(reg::LVT_TIMER, (1 << 17) | TIMER_VECTOR as u32);
    lapic_write(reg::TIMER_INIT, per_ms);
}

/// Per-AP LAPIC setup: x2APIC on, timer running with the BSP's calibration.
pub fn init_ap() {
    unsafe {
        let mut msr = x86_64::registers::model_specific::Msr::new(IA32_APIC_BASE);
        let v = msr.read();
        msr.write(v | (1 << 11) | (1 << 10));
    }
    lapic_write(reg::SVR, 0x100 | SPURIOUS_VECTOR as u32);
    lapic_write(reg::TPR, 0);
    lapic_write(reg::TIMER_DIVIDE, 0b0011);
    lapic_write(reg::LVT_TIMER, (1 << 17) | TIMER_VECTOR as u32);
    lapic_write(reg::TIMER_INIT, TICKS_PER_MS.load(Ordering::Relaxed) as u32);
}

/// Device interrupt routing. Requires our own page tables (IOAPIC is MMIO,
/// which Limine's HHDM doesn't cover) — call after `vmm::init`.
pub fn init_devices() {
    ioapic_route_isa(4, SERIAL_VECTOR);
    crate::serial::enable_rx_interrupt();
}

/// Measure LAPIC timer ticks per millisecond against a 10 ms PIT countdown.
fn calibrate_timer() -> u32 {
    lapic_write(reg::TIMER_DIVIDE, 0b0011); // divide by 16
    lapic_write(reg::LVT_TIMER, 1 << 16); // masked while calibrating

    unsafe {
        let mut gate = Port::<u8>::new(0x61);
        let mut cmd = Port::<u8>::new(0x43);
        let mut ch2 = Port::<u8>::new(0x42);

        // Gate on, speaker off.
        let g = gate.read();
        gate.write((g & !0x02) | 0x01);
        // Channel 2, lobyte/hibyte, mode 0 (interrupt on terminal count).
        cmd.write(0xB0u8);
        // 10 ms at 1.193182 MHz = 11932 counts.
        ch2.write((11932u16 & 0xFF) as u8);
        ch2.write((11932u16 >> 8) as u8);

        lapic_write(reg::TIMER_INIT, u32::MAX);
        // Wait for OUT2 (port 0x61 bit 5) to go high.
        while gate.read() & 0x20 == 0 {
            core::hint::spin_loop();
        }
        let elapsed = u32::MAX - lapic_read(reg::TIMER_CURRENT);
        lapic_write(reg::TIMER_INIT, 0); // stop

        elapsed / 10
    }
}

pub extern "x86-interrupt" fn timer_interrupt(_frame: InterruptStackFrame) {
    let cpu = crate::percpu::cpu();
    // Only the BSP advances the wall clock; every core preempts itself.
    if cpu.id == 0 {
        TICKS.fetch_add(1, Ordering::Relaxed);
    }
    cpu.preempt.store(true, Ordering::Relaxed);
    eoi();
}

pub extern "x86-interrupt" fn serial_interrupt(_frame: InterruptStackFrame) {
    crate::ports::serial_rx_irq();
    eoi();
}

pub extern "x86-interrupt" fn spurious_interrupt(_frame: InterruptStackFrame) {
    // No EOI for spurious interrupts.
}

pub extern "x86-interrupt" fn wake_interrupt(_frame: InterruptStackFrame) {
    // Its only job is to break `hlt` on an idle core.
    eoi();
}

pub extern "x86-interrupt" fn tlb_flush_interrupt(_frame: InterruptStackFrame) {
    // Reload CR3: flushes all non-global TLB entries (stack zone is
    // non-global; kernel image/HHDM are GLOBAL and survive).
    unsafe {
        let (frame, flags) = x86_64::registers::control::Cr3::read();
        x86_64::registers::control::Cr3::write(frame, flags);
    }
    TLB_ACKS.fetch_add(1, Ordering::Release);
    eoi();
}

pub extern "x86-interrupt" fn halt_interrupt(_frame: InterruptStackFrame) {
    loop {
        x86_64::instructions::interrupts::disable();
        x86_64::instructions::hlt();
    }
}

/// x2APIC ICR: fixed-delivery IPI to every core except this one.
pub fn ipi_all_others(vector: u8) {
    // Destination shorthand 0b11 = all excluding self.
    unsafe {
        x86_64::registers::model_specific::Msr::new(X2APIC_MSR_BASE + 0x30)
            .write((0b11 << 18) | vector as u64);
    }
}

/// x2APIC ICR: fixed-delivery IPI to one core by lapic id.
pub fn ipi(lapic_id: u32, vector: u8) {
    unsafe {
        x86_64::registers::model_specific::Msr::new(X2APIC_MSR_BASE + 0x30)
            .write(((lapic_id as u64) << 32) | vector as u64);
    }
}

/// Route an ISA IRQ through the IOAPIC to `vector` on the BSP
/// (edge-triggered, active-high, physical destination, unmasked).
pub fn ioapic_route_isa(irq: u8, vector: u8) {
    let p = crate::acpi_tables::platform();
    let gsi = p
        .isa_overrides
        .iter()
        .find(|o| o.irq == irq)
        .map(|o| o.gsi)
        .unwrap_or(irq as u32);
    let io = p
        .ioapics
        .iter()
        .find(|i| gsi >= i.gsi_base)
        .expect("no IOAPIC covers this GSI");
    let base = crate::mm::phys_to_virt(io.phys_addr);
    let index = gsi - io.gsi_base;
    unsafe {
        let sel = base.cast::<u32>();
        let win = base.add(0x10).cast::<u32>();
        sel.write_volatile(0x10 + index * 2 + 1);
        win.write_volatile(0); // destination: APIC id 0 (BSP)
        sel.write_volatile(0x10 + index * 2);
        win.write_volatile(vector as u32); // fixed, physical, edge, high, unmasked
    }
}
