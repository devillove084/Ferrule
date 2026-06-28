//! CUDA backend stub — real implementation uses ferrule-model/kernels.rs.
//! This module is kept for future standalone GPU backend (Phase 5+).

#[cfg(feature = "cuda")]
pub struct CudaBackend;

#[cfg(feature = "cuda")]
impl CudaBackend {
    /// Placeholder — real GPU dispatch goes through ferrule-model.
    pub fn new(_ordinal: usize) -> ferrule_core::Result<Self> {
        Ok(Self)
    }
}
