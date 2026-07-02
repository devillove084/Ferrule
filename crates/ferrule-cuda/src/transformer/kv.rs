use cuda_core::{stream::CudaStream, DeviceBuffer};
use ferrule_core::{Error, Result};

use crate::context::cu;

/// Single-sequence contiguous CUDA KV cache used by the current decode path.
///
/// This is intentionally a narrow wrapper around the existing buffers: it owns
/// K/V layers, decode position, max context length, and the attention score
/// workspace without changing kernel layout or launch behavior.
pub(crate) struct CudaContiguousKvCache {
    k_cache: Vec<DeviceBuffer<f32>>, // [num_layers][max_seq × kv_dim]
    v_cache: Vec<DeviceBuffer<f32>>, // [num_layers][max_seq × kv_dim]
    scores_buf: DeviceBuffer<f32>,   // [num_heads × max_seq]
    cur_seq: usize,
    max_seq: usize,
    kv_dim: usize,
    num_layers: usize,
    num_heads: usize,
}

impl CudaContiguousKvCache {
    pub(crate) fn new(
        stream: &CudaStream,
        num_layers: usize,
        max_seq: usize,
        kv_dim: usize,
        num_heads: usize,
    ) -> Result<Self> {
        let k_cache: Vec<_> = (0..num_layers)
            .map(|_| cu(DeviceBuffer::<f32>::zeroed(stream, max_seq * kv_dim)))
            .collect::<Result<_>>()?;
        let v_cache: Vec<_> = (0..num_layers)
            .map(|_| cu(DeviceBuffer::<f32>::zeroed(stream, max_seq * kv_dim)))
            .collect::<Result<_>>()?;
        let scores_buf = cu(DeviceBuffer::<f32>::zeroed(stream, num_heads * max_seq))?;

        Ok(Self {
            k_cache,
            v_cache,
            scores_buf,
            cur_seq: 0,
            max_seq,
            kv_dim,
            num_layers,
            num_heads,
        })
    }

    pub(crate) fn current_position(&self) -> usize {
        self.cur_seq
    }

    pub(crate) fn current_seq_len_after_append(&self) -> usize {
        self.cur_seq + 1
    }

    #[allow(dead_code)]
    pub(crate) fn max_seq(&self) -> usize {
        self.max_seq
    }

    #[allow(dead_code)]
    pub(crate) fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    #[allow(dead_code)]
    pub(crate) fn num_layers(&self) -> usize {
        self.num_layers
    }

    #[allow(dead_code)]
    pub(crate) fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub(crate) fn ensure_token_room(&self) -> Result<()> {
        if self.cur_seq >= self.max_seq {
            return Err(Error::Internal(format!(
                "GPU context length {} exceeds max_seq {}",
                self.cur_seq + 1,
                self.max_seq
            )));
        }
        Ok(())
    }

    pub(crate) fn append_layer(
        &mut self,
        layer: usize,
        pos: usize,
        k: &DeviceBuffer<f32>,
        v: &DeviceBuffer<f32>,
    ) -> Result<()> {
        if layer >= self.num_layers {
            return Err(Error::Internal(format!(
                "CUDA KV layer {layer} out of range {}",
                self.num_layers
            )));
        }
        if pos >= self.max_seq {
            return Err(Error::Internal(format!(
                "CUDA KV position {pos} exceeds capacity {}",
                self.max_seq
            )));
        }

        let byte_offset = pos * self.kv_dim * std::mem::size_of::<f32>();
        let byte_len = self.kv_dim * std::mem::size_of::<f32>();
        let k_copy = unsafe {
            cuda_bindings::cuMemcpyDtoD_v2(
                self.k_cache[layer].cu_deviceptr() + byte_offset as u64,
                k.cu_deviceptr(),
                byte_len,
            )
        };
        if k_copy != cuda_bindings::cudaError_enum_CUDA_SUCCESS {
            return Err(Error::Internal(format!(
                "CUDA K-cache device copy failed for layer {layer}, pos {pos}: {k_copy}"
            )));
        }

        let v_copy = unsafe {
            cuda_bindings::cuMemcpyDtoD_v2(
                self.v_cache[layer].cu_deviceptr() + byte_offset as u64,
                v.cu_deviceptr(),
                byte_len,
            )
        };
        if v_copy != cuda_bindings::cudaError_enum_CUDA_SUCCESS {
            return Err(Error::Internal(format!(
                "CUDA V-cache device copy failed for layer {layer}, pos {pos}: {v_copy}"
            )));
        }
        Ok(())
    }

    pub(crate) fn layer_buffers_mut(
        &mut self,
        layer: usize,
    ) -> Result<(
        &DeviceBuffer<f32>,
        &DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
    )> {
        if layer >= self.num_layers {
            return Err(Error::Internal(format!(
                "CUDA KV layer {layer} out of range {}",
                self.num_layers
            )));
        }
        Ok((
            &self.k_cache[layer],
            &self.v_cache[layer],
            &mut self.scores_buf,
        ))
    }

    pub(crate) fn advance_token(&mut self) {
        self.cur_seq += 1;
    }

    pub(crate) fn reset(&mut self) {
        self.cur_seq = 0;
    }
}
