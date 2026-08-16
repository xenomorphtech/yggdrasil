//! QEMU isa-debug-exit device (`-device isa-debug-exit,iobase=0xf4,iosize=0x04`).
//!
//! QEMU exits with status `(value << 1) | 1`: Success => 33, Failure => 35.

use x86_64::instructions::port::Port;

#[derive(Clone, Copy)]
#[repr(u32)]
pub enum ExitCode {
    Success = 0x10,
    Failure = 0x11,
}

pub fn exit(code: ExitCode) -> ! {
    unsafe {
        Port::<u32>::new(0xF4).write(code as u32);
    }
    // Not running under the test harness (no debug-exit device): hang.
    loop {
        x86_64::instructions::hlt();
    }
}
