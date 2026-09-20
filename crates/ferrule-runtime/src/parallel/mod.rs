//! Host-side parallel execution and bounded collective services.
//!
//! Transaction decisions and publication remain owned by
//! [`crate::DistributedTransaction`].

pub mod collective;
pub mod data;
pub mod expert;
pub mod pipeline;
#[cfg(unix)]
pub mod process;
pub mod tensor;
