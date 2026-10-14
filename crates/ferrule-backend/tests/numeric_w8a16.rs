//! CPU oracles always run. CUDA metadata/preflight tests need CUDA linking,
//! not a device. All tests that create a CUDA owner are explicitly ignored.
#[path = "numeric_w8a16/cpu.rs"]
mod cpu;
#[cfg(feature = "cuda")]
#[path = "numeric_w8a16/performance/dispatch.rs"]
mod dispatch;
#[cfg(feature = "cuda")]
#[path = "numeric_w8a16/gpu.rs"]
mod gpu;
#[cfg(feature = "cuda")]
#[path = "numeric_w8a16/metadata.rs"]
mod metadata;
#[path = "numeric_w8a16/oracle.rs"]
mod oracle;
#[cfg(all(feature = "cuda", ferrule_cuda_test_oracle))]
#[path = "numeric_w8a16/scalar.rs"]
mod scalar;
