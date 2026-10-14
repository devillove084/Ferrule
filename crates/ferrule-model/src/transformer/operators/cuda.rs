use std::fmt;
use std::sync::Arc;

use ferrule_backend::cuda::operators::linear::{CudaBf16Buffer, CudaF32Buffer, CudaOperators};
use ferrule_common::Result;

use super::{Rows, RowsArenaId, RowsDType, RowsShape, model_error};

/// CUDA-owned dense rows.
pub struct CudaRows {
    shape: RowsShape,
    dtype: RowsDType,
    arena: Option<RowsArenaId>,
    storage: CudaRowsStorage,
}

enum CudaRowsStorage {
    F32(CudaF32Buffer),
    Bf16(CudaBf16Buffer),
}

impl fmt::Debug for CudaRows {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaRows")
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("arena", &self.arena)
            .finish_non_exhaustive()
    }
}

impl CudaRows {
    pub fn f32(
        shape: RowsShape,
        arena: Option<RowsArenaId>,
        buffer: CudaF32Buffer,
    ) -> Result<Self> {
        if buffer.len() != shape.elements() {
            return Err(model_error(format!(
                "CUDA F32 rows contain {} values for shape [{}, {}]",
                buffer.len(),
                shape.rows(),
                shape.width()
            )));
        }
        Ok(Self {
            shape,
            dtype: RowsDType::F32,
            arena,
            storage: CudaRowsStorage::F32(buffer),
        })
    }

    pub fn bf16(
        shape: RowsShape,
        arena: Option<RowsArenaId>,
        buffer: CudaBf16Buffer,
    ) -> Result<Self> {
        if buffer.len() != shape.elements() {
            return Err(model_error(format!(
                "CUDA BF16 rows contain {} values for shape [{}, {}]",
                buffer.len(),
                shape.rows(),
                shape.width()
            )));
        }
        Ok(Self {
            shape,
            dtype: RowsDType::Bf16,
            arena,
            storage: CudaRowsStorage::Bf16(buffer),
        })
    }

    pub const fn shape(&self) -> RowsShape {
        self.shape
    }

    pub const fn dtype(&self) -> RowsDType {
        self.dtype
    }

    pub const fn arena(&self) -> Option<RowsArenaId> {
        self.arena
    }

    pub fn device_ordinal(&self) -> u32 {
        match &self.storage {
            CudaRowsStorage::F32(buffer) => buffer.as_device_buffer().context().ordinal() as u32,
            CudaRowsStorage::Bf16(buffer) => buffer.as_device_buffer().context().ordinal() as u32,
        }
    }

    /// Ordinals alone do not identify an allocator/stream owner.
    pub fn validate_owner(&self, operators: &CudaOperators) -> Result<()> {
        let stream = operators.stream_clone();
        let matches = match &self.storage {
            CudaRowsStorage::F32(buffer) => {
                Arc::ptr_eq(buffer.as_device_buffer().context(), stream.context())
            }
            CudaRowsStorage::Bf16(buffer) => {
                Arc::ptr_eq(buffer.as_device_buffer().context(), stream.context())
            }
        };
        if !matches {
            return Err(model_error(
                "CUDA rows belong to a different operator owner",
            ));
        }
        Ok(())
    }

    pub fn f32_buffer(&self) -> Option<&CudaF32Buffer> {
        match &self.storage {
            CudaRowsStorage::F32(buffer) => Some(buffer),
            CudaRowsStorage::Bf16(_) => None,
        }
    }

    pub fn f32_buffer_mut(&mut self) -> Option<&mut CudaF32Buffer> {
        match &mut self.storage {
            CudaRowsStorage::F32(buffer) => Some(buffer),
            CudaRowsStorage::Bf16(_) => None,
        }
    }

    pub fn bf16_buffer(&self) -> Option<&CudaBf16Buffer> {
        match &self.storage {
            CudaRowsStorage::F32(_) => None,
            CudaRowsStorage::Bf16(buffer) => Some(buffer),
        }
    }
}

impl Rows {
    pub fn cuda(&self) -> Result<&CudaRows> {
        match self {
            Self::Cuda(rows) => Ok(rows),
            Self::Host(_) => Err(model_error("CUDA operator received host rows")),
        }
    }

    pub fn into_cuda(self) -> Result<CudaRows> {
        match self {
            Self::Cuda(rows) => Ok(rows),
            Self::Host(_) => Err(model_error("CUDA operator received host rows")),
        }
    }
}
