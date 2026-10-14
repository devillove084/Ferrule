//! Mixed prefill GDN scratch is part of the admitted image, not its KV journal.
use super::*;
use ferrule_backend::cuda::operators::recurrent_bf16::Bf16GatedDeltaNetBuffers;
use ferrule_backend::cuda::operators::recurrent_bf16_chunk::CudaBf16ChunkWorkspace;
use ferrule_backend::cuda_plan::{Bf16ChunkPlan, GatedDeltaNetLayout};

impl CudaStandardDecoderOperators {
    pub(super) fn prepare_bf16_chunk_workspace(&mut self, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        if !bytes.is_multiple_of(4) || self.bf16_chunk_workspace.is_some() {
            return Err(cuda_error(
                "invalid or repeated BF16 chunk workspace preparation",
            ));
        }
        self.synchronous(|this| {
            this.bf16_chunk_workspace = Some(this.ops.zero_f32_buffer(bytes / 4)?);
            Ok(())
        })
    }

    pub(super) fn bf16_chunk_workspace_bytes(&self) -> usize {
        self.bf16_chunk_workspace
            .as_ref()
            .map_or(0, |buffer| buffer.len() * 4)
    }

    pub(super) fn bf16_chunk_plan(
        &self,
        layout: GatedDeltaNetLayout,
        state_position: usize,
    ) -> Result<Option<Bf16ChunkPlan>> {
        if self.mixed_gqa_block.is_some() {
            // The native rank invocation has its own explicit Mixed boundary;
            // operand selection must not accidentally select cached arithmetic.
            return crate::transformer::NumericFp8Precision::Bf16ModelMixedF32Accumulate
                .gdn_chunk_plan(
                    layout,
                    state_position != 0,
                    self.bf16_chunk_workspace_bytes(),
                );
        }
        self.numeric_fp8_precision().map_or(Ok(None), |precision| {
            precision.gdn_chunk_plan(
                layout,
                state_position != 0,
                self.bf16_chunk_workspace_bytes(),
            )
        })
    }

    pub(super) fn submit_bf16_delta(
        &mut self,
        buffers: Bf16GatedDeltaNetBuffers<'_>,
        layout: GatedDeltaNetLayout,
        plan: Option<Bf16ChunkPlan>,
    ) -> Result<()> {
        if let Some(plan) = plan {
            if plan.layout() != layout {
                return Err(cuda_error("BF16 chunk layout changed after preflight"));
            }
            // Exact-length view of one maximal allocation. The owner retains the
            // root through completion, error/unwind and permanent Unknown quarantine.
            let mut scratch = self
                .bf16_chunk_workspace
                .as_ref()
                .ok_or_else(|| cuda_error("unprepared BF16 chunk workspace"))?
                .prefix(plan.workspace_elements())?;
            self.ops.gated_delta_net_bf16_chunked_into(
                buffers,
                Some(CudaBf16ChunkWorkspace::new(&mut scratch, plan)?),
                plan,
            )?;
            self.trace_buffer(
                "gdn.chunk.workspace",
                &scratch,
                RowsShape::new(1, scratch.len())?,
                None,
                None,
                None,
            )
        } else {
            self.ops.gated_delta_net_bf16_into(buffers, layout)
        }
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/transformer/bf16_chunk/lifecycle.rs"]
mod tests;
