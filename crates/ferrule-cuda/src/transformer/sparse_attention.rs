//! CUDA sparse-attention backend contracts.
//!
//! Sparse attention over a sliding-window or compressed-KV index set with an
//! optional attention sink. The first CUDA kernel is correctness-oriented but
//! fixes the ABI for a later tiled FlashAttention-style implementation.

use cuda_core::{stream::CudaStream, DeviceBuffer, LaunchConfig};
use ferrule_core::{Error, Result};

use crate::context::cu;
use crate::kernels::kernels::LoadedModule;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CudaSparseAttentionShape {
    pub batch_size: usize,
    pub tokens_per_batch: usize,
    pub kv_len: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub topk: usize,
    pub softmax_scale: f32,
}

impl CudaSparseAttentionShape {
    pub fn tokens(&self) -> usize {
        self.batch_size * self.tokens_per_batch
    }

    pub fn output_elements(&self) -> usize {
        self.tokens() * self.heads * self.head_dim
    }

    pub fn q_elements(&self) -> usize {
        self.output_elements()
    }

    pub fn kv_elements(&self) -> usize {
        self.batch_size * self.kv_len * self.head_dim
    }

    pub fn topk_elements(&self) -> usize {
        self.tokens() * self.topk
    }

    pub fn validate(&self) -> Result<()> {
        if self.batch_size == 0
            || self.tokens_per_batch == 0
            || self.kv_len == 0
            || self.heads == 0
            || self.head_dim == 0
            || self.topk == 0
        {
            return Err(Error::Internal(format!(
                "invalid sparse attention shape: batch={} tokens_per_batch={} kv_len={} heads={} head_dim={} topk={}",
                self.batch_size,
                self.tokens_per_batch,
                self.kv_len,
                self.heads,
                self.head_dim,
                self.topk
            )));
        }
        if !self.softmax_scale.is_finite() || self.softmax_scale <= 0.0 {
            return Err(Error::Internal(format!(
                "invalid sparse attention softmax_scale {}",
                self.softmax_scale
            )));
        }
        checked_u32(
            self.output_elements(),
            "sparse attention",
            "output elements",
        )?;
        checked_u32(
            self.tokens_per_batch,
            "sparse attention",
            "tokens_per_batch",
        )?;
        checked_u32(self.kv_len, "sparse attention", "kv_len")?;
        checked_u32(self.heads, "sparse attention", "heads")?;
        checked_u32(self.head_dim, "sparse attention", "head_dim")?;
        checked_u32(self.topk, "sparse attention", "topk")?;
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub struct CudaSparseAttentionExecutor<'a> {
    pub module: &'a LoadedModule,
    pub stream: &'a CudaStream,
}

impl<'a> CudaSparseAttentionExecutor<'a> {
    pub fn new(module: &'a LoadedModule, stream: &'a CudaStream) -> Self {
        Self { module, stream }
    }

    pub fn sparse_attention_sink_f32(
        &self,
        q: &DeviceBuffer<f32>,
        kv: &DeviceBuffer<f32>,
        topk: &DeviceBuffer<i32>,
        sink: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        shape: CudaSparseAttentionShape,
    ) -> Result<()> {
        shape.validate()?;
        cu(unsafe {
            self.module.sparse_attn_tiled_sink_f32(
                self.stream,
                LaunchConfig::for_num_elems((shape.tokens() * shape.heads) as u32),
                q,
                kv,
                topk,
                sink,
                output,
                checked_u32(
                    shape.tokens() * shape.heads,
                    "sparse attention",
                    "num_pairs",
                )?,
                checked_u32(
                    shape.tokens_per_batch,
                    "sparse attention",
                    "tokens_per_batch",
                )?,
                checked_u32(shape.kv_len, "sparse attention", "kv_len")?,
                checked_u32(shape.heads, "sparse attention", "heads")?,
                checked_u32(shape.head_dim, "sparse attention", "head_dim")?,
                checked_u32(shape.topk, "sparse attention", "topk")?,
                shape.softmax_scale,
            )
        })
    }
}

fn checked_u32(value: usize, label: &str, field: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        Error::Internal(format!(
            "{label} {field} exceeds CUDA u32 launch ABI: {value}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_attention_shape_validates_large_decode() {
        let shape = CudaSparseAttentionShape {
            batch_size: 1,
            tokens_per_batch: 1,
            kv_len: 128 + 1024,
            heads: 64,
            head_dim: 512,
            topk: 128 + 512,
            softmax_scale: 512.0f32.powf(-0.5),
        };
        shape.validate().unwrap();
        assert_eq!(shape.output_elements(), 64 * 512);
    }

    #[test]
    fn sparse_attention_shape_rejects_zero_topk() {
        let shape = CudaSparseAttentionShape {
            batch_size: 1,
            tokens_per_batch: 1,
            kv_len: 1,
            heads: 1,
            head_dim: 1,
            topk: 0,
            softmax_scale: 1.0,
        };
        assert!(shape
            .validate()
            .unwrap_err()
            .to_string()
            .contains("invalid sparse attention"));
    }
}
