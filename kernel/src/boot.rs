//! Limine boot protocol requests.
//!
//! Every request lives in the `.limine_requests` section between the start/end
//! markers; the linker script keeps them and the bootloader scans the section.

use limine::request::{
    ExecutableAddressRequest, ExecutableCmdlineRequest, HhdmRequest, MemmapRequest,
    ModulesRequest, RsdpRequest, StackSizeRequest,
};
use limine::{BaseRevision, RequestsEndMarker, RequestsStartMarker};

#[used]
#[unsafe(link_section = ".limine_requests_start")]
static START_MARKER: RequestsStartMarker = RequestsStartMarker::new();

#[used]
#[unsafe(link_section = ".limine_requests")]
pub static BASE_REVISION: BaseRevision = BaseRevision::new();

/// Boot stack. Generous: Cranelift will eventually run on kernel stacks.
#[used]
#[unsafe(link_section = ".limine_requests")]
pub static STACK_SIZE: StackSizeRequest = StackSizeRequest::new(512 * 1024);

#[used]
#[unsafe(link_section = ".limine_requests")]
pub static HHDM: HhdmRequest = HhdmRequest::new();

#[used]
#[unsafe(link_section = ".limine_requests")]
pub static MEMMAP: MemmapRequest = MemmapRequest::new();

#[used]
#[unsafe(link_section = ".limine_requests")]
pub static RSDP: RsdpRequest = RsdpRequest::new();

#[used]
#[unsafe(link_section = ".limine_requests")]
pub static EXECUTABLE_ADDRESS: ExecutableAddressRequest = ExecutableAddressRequest::new();

#[used]
#[unsafe(link_section = ".limine_requests")]
pub static CMDLINE: ExecutableCmdlineRequest = ExecutableCmdlineRequest::new();

#[used]
#[unsafe(link_section = ".limine_requests")]
pub static MODULES: ModulesRequest = ModulesRequest::new();

#[used]
#[unsafe(link_section = ".limine_requests_end")]
static END_MARKER: RequestsEndMarker = RequestsEndMarker::new();

/// The kernel command line, or "" if the bootloader gave none.
pub fn cmdline() -> &'static str {
    CMDLINE.response().map(|r| r.cmdline()).unwrap_or("")
}
