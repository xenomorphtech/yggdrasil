//! ACPI table parsing: MADT (LAPIC/IOAPIC/ISA overrides) and MCFG (PCIe ECAM).
//!
//! Everything is reached through the HHDM, so "mapping" a table is pointer
//! arithmetic. Only static tables are parsed — no AML.

use alloc::vec::Vec;
use core::ptr::NonNull;

use acpi::sdt::madt::{Madt, MadtEntry};
use acpi::sdt::mcfg::Mcfg;
use acpi::{AcpiTables, Handler, PhysicalMapping};
use spin::Once;

use crate::{boot, mm};

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // consumed in M4 (IOAPIC routing) and M5 (ECAM)
pub struct IoApic {
    pub id: u8,
    pub phys_addr: u64,
    pub gsi_base: u32,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // consumed in M4 (IOAPIC routing) and M5 (ECAM)
pub struct IsaOverride {
    pub irq: u8,
    pub gsi: u32,
    pub flags: u16,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // consumed in M4 (IOAPIC routing) and M5 (ECAM)
pub struct EcamRegion {
    pub base: u64,
    pub segment: u16,
    pub bus_start: u8,
    pub bus_end: u8,
}

#[derive(Debug)]
pub struct Platform {
    pub lapic_phys: u64,
    pub ioapics: Vec<IoApic>,
    pub isa_overrides: Vec<IsaOverride>,
    pub ecam: Vec<EcamRegion>,
}

static PLATFORM: Once<Platform> = Once::new();

pub fn platform() -> &'static Platform {
    PLATFORM.get().expect("acpi not initialized")
}

pub fn init() {
    let rsdp = boot::RSDP.response().expect("limine: no RSDP").address as u64;
    // Limine hands back an HHDM-virtual pointer; the acpi crate wants physical.
    let hhdm = mm::hhdm_offset();
    let rsdp_phys = if rsdp >= hhdm { rsdp - hhdm } else { rsdp };

    let tables = unsafe { AcpiTables::from_rsdp(HhdmHandler { hhdm }, rsdp_phys as usize) }
        .expect("failed to parse ACPI tables");

    let madt = tables.find_table::<Madt>().expect("no MADT");
    let madt = madt.get();
    let mut lapic_phys = madt.local_apic_address as u64;
    let mut ioapics = Vec::new();
    let mut isa_overrides = Vec::new();
    for entry in madt.entries() {
        match entry {
            MadtEntry::IoApic(e) => ioapics.push(IoApic {
                id: e.io_apic_id,
                phys_addr: e.io_apic_address as u64,
                gsi_base: e.global_system_interrupt_base,
            }),
            MadtEntry::InterruptSourceOverride(e) => isa_overrides.push(IsaOverride {
                irq: e.irq,
                gsi: e.global_system_interrupt,
                flags: e.flags,
            }),
            MadtEntry::LocalApicAddressOverride(e) => lapic_phys = e.local_apic_address,
            _ => {}
        }
    }

    let mut ecam = Vec::new();
    if let Some(mcfg) = tables.find_table::<Mcfg>() {
        for e in mcfg.get().entries() {
            ecam.push(EcamRegion {
                base: e.base_address,
                segment: e.pci_segment_group,
                bus_start: e.bus_number_start,
                bus_end: e.bus_number_end,
            });
        }
    }

    let p = Platform { lapic_phys, ioapics, isa_overrides, ecam };
    log::info!(
        "acpi: lapic={:#x}, {} ioapic(s), {} isa override(s), {} ecam region(s)",
        p.lapic_phys,
        p.ioapics.len(),
        p.isa_overrides.len(),
        p.ecam.len()
    );
    PLATFORM.call_once(|| p);
}

#[derive(Clone)]
struct HhdmHandler {
    hhdm: u64,
}

impl Handler for HhdmHandler {
    unsafe fn map_physical_region<T>(
        &self,
        physical_address: usize,
        size: usize,
    ) -> PhysicalMapping<Self, T> {
        PhysicalMapping {
            physical_start: physical_address,
            virtual_start: NonNull::new((physical_address as u64 + self.hhdm) as *mut T)
                .expect("null ACPI physical address"),
            region_length: size,
            mapped_length: size,
            handler: self.clone(),
        }
    }

    fn unmap_physical_region<T>(_region: &PhysicalMapping<Self, T>) {}

    fn read_u8(&self, address: usize) -> u8 {
        unsafe { ((address as u64 + self.hhdm) as *const u8).read_volatile() }
    }
    fn read_u16(&self, address: usize) -> u16 {
        unsafe { ((address as u64 + self.hhdm) as *const u16).read_volatile() }
    }
    fn read_u32(&self, address: usize) -> u32 {
        unsafe { ((address as u64 + self.hhdm) as *const u32).read_volatile() }
    }
    fn read_u64(&self, address: usize) -> u64 {
        unsafe { ((address as u64 + self.hhdm) as *const u64).read_volatile() }
    }
    fn write_u8(&self, address: usize, value: u8) {
        unsafe { ((address as u64 + self.hhdm) as *mut u8).write_volatile(value) }
    }
    fn write_u16(&self, address: usize, value: u16) {
        unsafe { ((address as u64 + self.hhdm) as *mut u16).write_volatile(value) }
    }
    fn write_u32(&self, address: usize, value: u32) {
        unsafe { ((address as u64 + self.hhdm) as *mut u32).write_volatile(value) }
    }
    fn write_u64(&self, address: usize, value: u64) {
        unsafe { ((address as u64 + self.hhdm) as *mut u64).write_volatile(value) }
    }

    fn read_io_u8(&self, port: u16) -> u8 {
        unsafe { x86_64::instructions::port::Port::new(port).read() }
    }
    fn read_io_u16(&self, port: u16) -> u16 {
        unsafe { x86_64::instructions::port::Port::new(port).read() }
    }
    fn read_io_u32(&self, port: u16) -> u32 {
        unsafe { x86_64::instructions::port::Port::new(port).read() }
    }
    fn write_io_u8(&self, port: u16, value: u8) {
        unsafe { x86_64::instructions::port::Port::new(port).write(value) }
    }
    fn write_io_u16(&self, port: u16, value: u16) {
        unsafe { x86_64::instructions::port::Port::new(port).write(value) }
    }
    fn write_io_u32(&self, port: u16, value: u32) {
        unsafe { x86_64::instructions::port::Port::new(port).write(value) }
    }

    // PCI config access is only needed by the AML interpreter, which we don't run.
    fn read_pci_u8(&self, _a: acpi::PciAddress, _o: u16) -> u8 {
        unimplemented!("PCI config via acpi::Handler")
    }
    fn read_pci_u16(&self, _a: acpi::PciAddress, _o: u16) -> u16 {
        unimplemented!("PCI config via acpi::Handler")
    }
    fn read_pci_u32(&self, _a: acpi::PciAddress, _o: u16) -> u32 {
        unimplemented!("PCI config via acpi::Handler")
    }
    fn write_pci_u8(&self, _a: acpi::PciAddress, _o: u16, _v: u8) {
        unimplemented!("PCI config via acpi::Handler")
    }
    fn write_pci_u16(&self, _a: acpi::PciAddress, _o: u16, _v: u16) {
        unimplemented!("PCI config via acpi::Handler")
    }
    fn write_pci_u32(&self, _a: acpi::PciAddress, _o: u16, _v: u32) {
        unimplemented!("PCI config via acpi::Handler")
    }

    // AML mutex hooks: required by the trait because the crate's `aml` feature
    // can't be disabled (it doesn't compile without it), but we never run AML.
    fn create_mutex(&self) -> acpi::Handle {
        acpi::Handle(0)
    }
    fn acquire(&self, _mutex: acpi::Handle, _timeout: u16) -> Result<(), acpi::aml::AmlError> {
        Ok(())
    }
    fn release(&self, _mutex: acpi::Handle) {}

    fn nanos_since_boot(&self) -> u64 {
        crate::irq::ticks() * 1_000_000
    }
    fn stall(&self, microseconds: u64) {
        // Rough busy-wait; only used by AML, which we don't run.
        for _ in 0..microseconds * 100 {
            core::hint::spin_loop();
        }
    }
    fn sleep(&self, milliseconds: u64) {
        self.stall(milliseconds * 1000);
    }
}
