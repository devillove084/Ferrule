//! Semantic hc execution using borrowed owner resources.

use crate::cuda::operators::OperatorOwner;
use crate::cuda::operators::linear::{
    CudaF32Buffer, CudaFp8ActivationPack, CudaOperators, CudaPreparedFp8Activation,
};
use ferrule_common::Result;

impl CudaOperators {
    /// Execute the complete HC-pre + layer RMSNorm + FP8 activation producer.
    #[allow(clippy::too_many_arguments)]
    pub fn hc_pre_rmsnorm_fp8_into<'a>(
        &self,
        state: &CudaF32Buffer,
        function_row_major: &CudaF32Buffer,
        hc_scale: &CudaF32Buffer,
        hc_base: &CudaF32Buffer,
        layer_rms_weight: &CudaF32Buffer,
        mix_output: &mut CudaF32Buffer,
        workspace: &mut CudaF32Buffer,
        rows: usize,
        hc: usize,
        hidden_size: usize,
        sinkhorn_iters: usize,
        hc_eps: f32,
        hc_norm_eps: f32,
        layer_rms_eps: f32,
        hidden_output: &mut CudaF32Buffer,
        normalized_output: &mut CudaF32Buffer,
        split_pre: &mut CudaF32Buffer,
        split_post: &mut CudaF32Buffer,
        split_comb: &mut CudaF32Buffer,
        packed_output: &'a mut CudaFp8ActivationPack,
    ) -> Result<CudaPreparedFp8Activation<'a>> {
        self.submit_operator(1, |stream| {
            let packed_output = packed_output.operator_storage_mut();
            crate::cuda::providers::cutlass::submit_hc_producer(
                stream,
                state.as_device_buffer(),
                function_row_major.as_device_buffer(),
                hc_scale.as_device_buffer(),
                hc_base.as_device_buffer(),
                layer_rms_weight.as_device_buffer(),
                mix_output.as_device_buffer_mut(),
                workspace.as_device_buffer_mut(),
                hidden_output.as_device_buffer_mut(),
                normalized_output.as_device_buffer_mut(),
                packed_output.x_packed,
                packed_output.x_scales,
                split_pre.as_device_buffer_mut(),
                split_post.as_device_buffer_mut(),
                split_comb.as_device_buffer_mut(),
                rows,
                hc,
                hidden_size,
                sinkhorn_iters,
                hc_eps,
                hc_norm_eps,
                layer_rms_eps,
            )
        })?;
        self.prepared_fp8_activation_from_storage(packed_output, rows, hidden_size)
    }
}
