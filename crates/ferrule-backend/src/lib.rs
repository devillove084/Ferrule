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

pub mod error;
pub mod plan;

pub use error::{BackendError, BackendResult};

#[cfg(feature = "cuda")]
pub mod cuda;
