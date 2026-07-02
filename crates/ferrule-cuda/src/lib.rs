#![allow(
    clippy::manual_div_ceil,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::unnecessary_sort_by,
    unsafe_code
)]
//! CUDA backend for Ferrule — kernels, forward pass, memory pool.
//!
//! Note: `graph` in this crate refers to CUDA driver graph capture/replay.
//! Ferrule's device-independent compute graph IR lives in `ferrule-graph`.

pub mod build;
pub mod context;
pub mod forward;
pub mod graph;
pub mod kernels;
pub mod transformer;
pub mod weightpack;
