//! Polled 16550 UART driver (COM1) and the kernel logger.
//!
//! Interrupt-driven rx arrives with the serial *port* in M4; this module stays
//! the low-level console path (including from panic context).

use core::fmt::{self, Write};

use spin::Mutex;
use x86_64::instructions::port::Port;

const COM1: u16 = 0x3F8;

pub struct Uart {
    base: u16,
}

impl Uart {
    const fn new(base: u16) -> Self {
        Self { base }
    }

    fn reg(&self, offset: u16) -> Port<u8> {
        Port::new(self.base + offset)
    }

    fn init(&mut self) {
        unsafe {
            self.reg(1).write(0x00u8); // IER: all interrupts off
            self.reg(3).write(0x80u8); // LCR: DLAB on
            self.reg(0).write(0x01u8); // divisor low: 115200 baud
            self.reg(1).write(0x00u8); // divisor high
            self.reg(3).write(0x03u8); // LCR: 8n1, DLAB off
            self.reg(2).write(0x07u8); // FCR: FIFO on, clear, 1-byte rx trigger
            self.reg(4).write(0x0Bu8); // MCR: DTR | RTS | OUT2
        }
    }

    fn write_byte(&mut self, byte: u8) {
        unsafe {
            // Wait for THR empty (LSR bit 5).
            while self.reg(5).read() & 0x20 == 0 {
                core::hint::spin_loop();
            }
            self.reg(0).write(byte);
        }
    }
}

impl Write for Uart {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for byte in s.bytes() {
            if byte == b'\n' {
                self.write_byte(b'\r');
            }
            self.write_byte(byte);
        }
        Ok(())
    }
}

static SERIAL: Mutex<Uart> = Mutex::new(Uart::new(COM1));

pub fn init() {
    SERIAL.lock().init();
    // Racy variant because no_std; single-threaded at this point anyway.
    unsafe {
        let _ = log::set_logger_racy(&KernelLog);
    }
    // Info: cranelift's trace/debug logging would otherwise flood the console.
    log::set_max_level(log::LevelFilter::Info);
}

pub fn write_fmt(args: fmt::Arguments) {
    let _ = SERIAL.lock().write_fmt(args);
}

/// Regain console access from panic/exception context, even if the lock is
/// held by the interrupted code.
///
/// # Safety
/// Only call when no other CPU/thread can be inside the lock (panic path).
pub unsafe fn force_writer() -> impl Write {
    unsafe { SERIAL.force_unlock() };
    struct Forced;
    impl Write for Forced {
        fn write_str(&mut self, s: &str) -> fmt::Result {
            SERIAL.lock().write_str(s)
        }
    }
    Forced
}

/// TX one byte through the console lock (port-driver write path).
pub fn raw_write_byte(b: u8) {
    SERIAL.lock().write_byte(b);
}

/// RX poll, lock-free — safe from IRQ context even if the console lock is
/// held by interrupted code (LSR/RBR don't clash with THR writes).
pub fn try_read_byte() -> Option<u8> {
    unsafe {
        let mut lsr = Port::<u8>::new(COM1 + 5);
        if lsr.read() & 0x01 != 0 {
            Some(Port::<u8>::new(COM1).read())
        } else {
            None
        }
    }
}

/// Enable the UART "received data available" interrupt (IER bit 0).
pub fn enable_rx_interrupt() {
    unsafe { Port::<u8>::new(COM1 + 1).write(0x01u8) };
}

#[macro_export]
macro_rules! print {
    ($($arg:tt)*) => { $crate::serial::write_fmt(format_args!($($arg)*)) };
}

#[macro_export]
macro_rules! println {
    () => { $crate::print!("\n") };
    ($($arg:tt)*) => { $crate::print!("{}\n", format_args!($($arg)*)) };
}

struct KernelLog;

impl log::Log for KernelLog {
    fn enabled(&self, _metadata: &log::Metadata) -> bool {
        true
    }

    fn log(&self, record: &log::Record) {
        crate::println!("[{:5}] {}", record.level(), record.args());
    }

    fn flush(&self) {}
}
