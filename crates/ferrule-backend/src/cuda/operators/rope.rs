//! Provider-neutral rotary-position operations.

use ferrule_common::Result;

pub use crate::cuda::context::{CudaF32Buffer, CudaI32Buffer, CudaOperators};
pub use crate::cuda::operators::contracts::SplitHalfRopeLayout;

impl CudaOperators {
    /// Apply indexed split-half RoPE to packed device rows.
    pub fn apply_split_half_rope(
        &self,
        values: &mut CudaF32Buffer,
        cosine: &CudaF32Buffer,
        sine: &CudaF32Buffer,
        positions: &CudaI32Buffer,
        layout: SplitHalfRopeLayout,
        inverse: bool,
    ) -> Result<()> {
        self.split_half_rope_rows_indexed_from_device(
            values, cosine, sine, positions, layout, inverse,
        )
    }
}
