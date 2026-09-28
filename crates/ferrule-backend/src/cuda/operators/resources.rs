//! Narrow borrows from context-owned storage. Only this adapter is a context
//! child; family implementations compile outside that privacy boundary.
//! Views neither allocate nor clone device storage and carry no owner ledger.

use super::{
    ARTIFACT_LINEAR_FP8_ACTIVATION_BLOCK_SIZE, CudaArtifactLinearHandle, CudaArtifactLinearShape,
    CudaArtifactLinearWorkspace, CudaF32Buffer, CudaFp8ActivationPack, CudaI32Buffer,
    CudaOperators, CudaPreparedFp8Activation, DeviceBuffer,
};
use ferrule_common::{Error, Result};

pub(crate) struct ArtifactLinearRef<'a> {
    pub(crate) shape: CudaArtifactLinearShape,
    pub(crate) weight: &'a DeviceBuffer<u8>,
    pub(crate) scale: Option<&'a DeviceBuffer<u8>>,
}

impl CudaArtifactLinearHandle {
    pub(crate) fn operator_view(&self) -> ArtifactLinearRef<'_> {
        ArtifactLinearRef {
            shape: self.shape,
            weight: &self.weight,
            scale: self.scale.as_ref(),
        }
    }
}

/// Disjoint mutable pack/scale borrows, plus their existing capacity contract.
/// The generic artifact workspace's unrelated `cloned` storage is not exposed.
pub(crate) struct Fp8StorageMut<'a> {
    pub(crate) x_packed: &'a mut DeviceBuffer<u8>,
    pub(crate) x_scales: &'a mut DeviceBuffer<u8>,
    pub(crate) value_capacity: usize,
    pub(crate) scale_capacity: usize,
}

impl CudaArtifactLinearWorkspace {
    pub(crate) fn operator_storage_mut(&mut self) -> Fp8StorageMut<'_> {
        Fp8StorageMut {
            x_packed: &mut self.x_packed,
            x_scales: &mut self.x_scales,
            value_capacity: self.value_capacity,
            scale_capacity: self.scale_capacity,
        }
    }
}

impl CudaFp8ActivationPack {
    pub(crate) fn operator_storage_mut(&mut self) -> Fp8StorageMut<'_> {
        Fp8StorageMut {
            x_packed: &mut self.x_packed,
            x_scales: &mut self.x_scales,
            value_capacity: self.value_capacity,
            scale_capacity: self.scale_capacity,
        }
    }

    pub(crate) fn prepared_view(
        &self,
        rows: usize,
        row_width: usize,
    ) -> Result<CudaPreparedFp8Activation<'_>> {
        let values = rows.checked_mul(row_width).ok_or_else(|| Error::Internal {
            message: "CUDA prepared FP8 activation size overflow".into(),
        })?;
        let scales = rows
            .checked_mul(row_width.div_ceil(ARTIFACT_LINEAR_FP8_ACTIVATION_BLOCK_SIZE))
            .ok_or_else(|| Error::Internal {
                message: "CUDA prepared FP8 scale size overflow".into(),
            })?;
        if rows == 0
            || row_width == 0
            || !row_width.is_multiple_of(ARTIFACT_LINEAR_FP8_ACTIVATION_BLOCK_SIZE)
            || self.value_capacity != values
            || self.scale_capacity != scales
        {
            return Err(Error::Internal {
                message: format!(
                    "CUDA prepared FP8 activation storage mismatch: rows={rows} width={row_width} values={}/{} scales={}/{}",
                    self.value_capacity, values, self.scale_capacity, scales
                ),
            });
        }
        Ok(CudaPreparedFp8Activation {
            x_packed: &self.x_packed,
            x_scales: &self.x_scales,
            rows,
            row_width,
        })
    }
}

/// A reborrow cannot outlive the prepared activation proof that supplied it.
pub(crate) struct PreparedFp8ActivationRef<'a> {
    pub(crate) x_packed: &'a DeviceBuffer<u8>,
    pub(crate) x_scales: &'a DeviceBuffer<u8>,
    pub(crate) rows: usize,
    pub(crate) row_width: usize,
}

impl CudaPreparedFp8Activation<'_> {
    pub(crate) fn operator_view(&self) -> PreparedFp8ActivationRef<'_> {
        PreparedFp8ActivationRef {
            x_packed: self.x_packed,
            x_scales: self.x_scales,
            rows: self.rows,
            row_width: self.row_width,
        }
    }
}

impl CudaOperators {
    /// Reuse the original packing validation, kernel and accounting in place.
    pub(crate) fn prepare_operator_fp8(
        &self,
        input: &CudaF32Buffer,
        rows: usize,
        row_width: usize,
        storage: &mut Fp8StorageMut<'_>,
    ) -> Result<()> {
        self.pack_fp8_rows_from_f32_preallocated(
            input.as_device_buffer(),
            rows,
            row_width,
            storage.x_packed,
            storage.value_capacity,
            storage.x_scales,
            storage.scale_capacity,
        )
    }

    pub(crate) fn clear_operator_status(&self, status: &mut CudaI32Buffer) -> Result<()> {
        self.zero_i32_buffer_in_place(status)
    }
}
