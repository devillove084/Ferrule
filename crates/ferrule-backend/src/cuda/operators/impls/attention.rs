//! Semantic attention execution using borrowed owner resources.

use crate::cuda::operators::OperatorOwner;
use crate::cuda::operators::attention::{
    CudaF32Buffer, CudaHybridMlaAttentionWorkspace, CudaI32Buffer, CudaOperators,
};
use ferrule_common::Result;

impl CudaOperators {
    /// Compute a BF16-compressed two-projection bundle on device.
    /// Run checkpoint-native proposal attention over committed paged context and
    /// one read-only five-row proposal block. All scratch remains caller-owned.
    #[allow(clippy::too_many_arguments)]
    pub fn hybrid_mla_attention_into(
        &self,
        query: &CudaF32Buffer,
        context_plane: &CudaF32Buffer,
        block_kv: &CudaF32Buffer,
        block_slots: &CudaI32Buffer,
        attention_sink: &CudaF32Buffer,
        layout: crate::cuda::operators::HybridMlaAttentionLayout,
        output: &mut CudaF32Buffer,
        workspace: &mut CudaHybridMlaAttentionWorkspace,
    ) -> Result<()> {
        self.clear_operator_status(&mut workspace.status)?;
        self.submit_operator(1, |stream| {
            crate::cuda::operators::attention::hybrid::hybrid_mla_attention(
                stream,
                query.as_device_buffer(),
                context_plane.as_device_buffer(),
                block_kv.as_device_buffer(),
                block_slots.as_device_buffer(),
                attention_sink.as_device_buffer(),
                &mut workspace.query_bf16,
                &mut workspace.gathered_kv_bf16,
                workspace.scores.as_device_buffer_mut(),
                &mut workspace.probabilities_bf16,
                workspace.online_rescales.as_device_buffer_mut(),
                workspace.denominators.as_device_buffer_mut(),
                output.as_device_buffer_mut(),
                workspace.status.as_device_buffer_mut(),
                layout,
            )
        })?;
        Ok(())
    }
}
