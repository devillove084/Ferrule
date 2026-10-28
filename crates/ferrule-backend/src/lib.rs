#![allow(
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::unnecessary_sort_by,
    unsafe_code
)]
//! Ferrule device runtime and kernel-provider boundary.
//!
//! [`plan`] contains device-neutral executable-plan contracts. Concrete device
//! implementations live in implementation modules such as `cuda`, and must
//! not leak vendor runtime types into the scheduler or I/O protocols.

pub mod cpu;
pub mod cuda_plan;
pub mod error;
pub mod operator_contract;
pub mod physical_placement;
pub use physical_placement as host_kv_placement;
pub mod plan;
pub mod resources;

#[path = "cuda/allocation_layout.rs"]
mod cuda_allocation_layout;
pub use cuda_allocation_layout::cuda_allocation_grant_bytes;

pub use error::{BackendError, BackendResult};
pub use ferrule_common::numeric_fp8;

#[cfg(feature = "cuda")]
pub mod cuda;
