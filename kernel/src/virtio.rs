//! virtio-blk and virtio-net over PCIe ECAM, using `virtio-drivers` with a
//! HAL over our PMM + HHDM.
//!
//! Device access is synchronous (the drivers poll their virtqueues); the port
//! layer wraps them in the async SQ/CQ interface. INTx/MSI-X interrupt-driven
//! completion is a later upgrade — net rx is polled from the scheduler's pump.

use core::ptr::NonNull;

use spin::Mutex;
use virtio_drivers::device::blk::{SECTOR_SIZE, VirtIOBlk};
use virtio_drivers::device::net::VirtIONet;
use virtio_drivers::transport::DeviceType;
use virtio_drivers::transport::pci::bus::{Cam, Command, MmioCam, PciRoot};
use virtio_drivers::transport::pci::{PciTransport, virtio_device_type};
use virtio_drivers::{BufferDirection, Hal, PhysAddr as VPhysAddr};

use crate::mm;

const NET_QUEUE_SIZE: usize = 16;
const NET_BUF_LEN: usize = 2048;

type Blk = VirtIOBlk<KernelHal, PciTransport>;
type Net = VirtIONet<KernelHal, PciTransport, NET_QUEUE_SIZE>;

static BLK: Mutex<Option<Blk>> = Mutex::new(None);
static NET: Mutex<Option<Net>> = Mutex::new(None);

pub struct KernelHal;

unsafe impl Hal for KernelHal {
    fn dma_alloc(pages: usize, _direction: BufferDirection) -> (VPhysAddr, NonNull<u8>) {
        let phys = mm::alloc_contig(pages, 1).expect("virtio dma alloc failed");
        let virt = mm::phys_to_virt(phys);
        unsafe { virt.write_bytes(0, pages * 4096) };
        (phys, NonNull::new(virt).unwrap())
    }

    unsafe fn dma_dealloc(paddr: VPhysAddr, _vaddr: NonNull<u8>, pages: usize) -> i32 {
        mm::free_frames(paddr, pages);
        0
    }

    unsafe fn mmio_phys_to_virt(paddr: VPhysAddr, _size: usize) -> NonNull<u8> {
        NonNull::new(mm::phys_to_virt(paddr)).unwrap()
    }

    unsafe fn share(buffer: NonNull<[u8]>, _direction: BufferDirection) -> VPhysAddr {
        // No IOMMU; every kernel address (HHDM or kernel image) has a known
        // physical address, and both regions are physically contiguous.
        mm::virt_to_phys(buffer.as_ptr() as *mut u8 as u64)
    }

    unsafe fn unshare(_paddr: VPhysAddr, _buffer: NonNull<[u8]>, _direction: BufferDirection) {}
}

pub fn init() {
    let platform = crate::acpi_tables::platform();
    let Some(ecam) = platform.ecam.first() else {
        log::warn!("virtio: no ECAM region, skipping PCI scan");
        return;
    };
    let base = mm::phys_to_virt(ecam.base);
    let mut root = PciRoot::new(unsafe { MmioCam::new(base, Cam::Ecam) });
    let mut found = alloc::vec::Vec::new();
    for (df, info) in root.enumerate_bus(0) {
        if let Some(ty) = virtio_device_type(&info) {
            found.push((df, ty));
        }
    }
    for (df, ty) in found {
        root.set_command(df, Command::MEMORY_SPACE | Command::BUS_MASTER);
        match ty {
            DeviceType::Block => {
                let t = PciTransport::new::<KernelHal, _>(&mut root, df)
                    .expect("virtio-blk transport");
                let blk = VirtIOBlk::new(t).expect("virtio-blk init");
                log::info!("virtio: blk at {df}, {} sectors", blk.capacity());
                *BLK.lock() = Some(blk);
            }
            DeviceType::Network => {
                let t = PciTransport::new::<KernelHal, _>(&mut root, df)
                    .expect("virtio-net transport");
                let net = VirtIONet::new(t, NET_BUF_LEN).expect("virtio-net init");
                log::info!("virtio: net at {df}, mac {:02x?}", net.mac_address());
                *NET.lock() = Some(net);
            }
            other => log::info!("virtio: ignoring {other:?} at {df}"),
        }
    }
}

pub fn has_blk() -> bool {
    BLK.lock().is_some()
}

pub fn has_net() -> bool {
    NET.lock().is_some()
}

pub fn blk_read(sector: u64) -> Result<alloc::vec::Vec<u8>, ()> {
    let mut guard = BLK.lock();
    let blk = guard.as_mut().ok_or(())?;
    let mut buf = alloc::vec![0u8; SECTOR_SIZE];
    blk.read_blocks(sector as usize, &mut buf).map_err(|e| log::warn!("blk read: {e:?}"))?;
    Ok(buf)
}

pub fn blk_write(sector: u64, data: &[u8]) -> Result<(), ()> {
    let mut guard = BLK.lock();
    let blk = guard.as_mut().ok_or(())?;
    blk.write_blocks(sector as usize, data).map_err(|e| log::warn!("blk write: {e:?}"))?;
    blk.flush().map_err(|e| log::warn!("blk flush: {e:?}"))
}

pub fn net_mac() -> [u8; 6] {
    NET.lock().as_ref().map(|n| n.mac_address()).unwrap_or_default()
}

pub fn net_send(frame: &[u8]) -> Result<(), ()> {
    let mut guard = NET.lock();
    let net = guard.as_mut().ok_or(())?;
    let mut tx = net.new_tx_buffer(frame.len());
    tx.packet_mut().copy_from_slice(frame);
    net.send(tx).map_err(|e| log::warn!("net send: {e:?}"))
}

/// Non-blocking receive: returns a frame if one is waiting.
pub fn net_try_recv() -> Option<alloc::vec::Vec<u8>> {
    let mut guard = NET.lock();
    let net = guard.as_mut()?;
    if !net.can_recv() {
        return None;
    }
    let rx = net.receive().ok()?;
    let frame = rx.packet().to_vec();
    let _ = net.recycle_rx_buffer(rx);
    Some(frame)
}
