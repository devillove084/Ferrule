#![cfg(feature = "cuda")]

#[path = "router_normalization/contract.rs"]
mod contract;
#[path = "router_normalization/gpu.rs"]
mod gpu;
