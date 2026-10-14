//! Host-side parallel execution and bounded collective services.
//!
//! Transaction decisions and publication remain owned by
//! [`crate::DistributedTransaction`].

#[cfg(unix)]
pub(crate) mod codec;
#[cfg(unix)]
pub(crate) mod frame_budget;

pub mod collective;
pub mod data;
pub mod expert;
#[cfg(unix)]
pub mod expert_process;
pub mod hybrid_topology;
#[cfg(unix)]
pub mod network;
pub mod pipeline;
#[cfg(unix)]
pub mod process;
pub mod tensor;
