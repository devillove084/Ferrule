//! Semantic proposal execution using borrowed owner resources.

use crate::cuda::operators::OperatorOwner;
use crate::cuda::operators::linear::{
    CudaArtifactLinearHandle, CudaArtifactLinearShape, CudaF32Buffer, CudaFp8ActivationPack,
    CudaOperators,
};
use crate::cuda::operators::proposal::CudaProposalHeadWorkspace;
use ferrule_common::{Error, Result};

impl CudaOperators {
    /// Run the checkpoint-native proposal HC/LM/Markov/confidence proposal head.
    #[allow(clippy::too_many_arguments)]
    pub fn artifact_proposal_head_into(
        &self,
        hc_state: &CudaF32Buffer,
        hc_function: &CudaF32Buffer,
        hc_scale: &CudaF32Buffer,
        hc_base: &CudaF32Buffer,
        norm_weight: &CudaF32Buffer,
        lm_head: &CudaArtifactLinearHandle,
        markov_w1: &CudaArtifactLinearHandle,
        markov_w2: &CudaArtifactLinearHandle,
        confidence_weight: &CudaArtifactLinearHandle,
        anchor_token_id: u32,
        layout: crate::cuda::operators::ProposalHeadLayout,
        workspace: &mut CudaProposalHeadWorkspace,
    ) -> Result<()> {
        let lm_head = lm_head.operator_view();
        let markov_w1 = markov_w1.operator_view();
        let markov_w2 = markov_w2.operator_view();
        let confidence_weight = confidence_weight.operator_view();
        let expected = [
            (
                "LM head",
                lm_head.shape,
                CudaArtifactLinearShape::Bf16Bytes {
                    out_features: layout.vocab,
                    in_features: layout.hidden,
                },
            ),
            (
                "Markov W1",
                markov_w1.shape,
                CudaArtifactLinearShape::Bf16Bytes {
                    out_features: layout.vocab,
                    in_features: layout.markov_rank,
                },
            ),
            (
                "Markov W2",
                markov_w2.shape,
                CudaArtifactLinearShape::Bf16Bytes {
                    out_features: layout.vocab,
                    in_features: layout.markov_rank,
                },
            ),
            (
                "confidence",
                confidence_weight.shape,
                CudaArtifactLinearShape::Bf16Bytes {
                    out_features: 1,
                    in_features: layout.hidden + layout.markov_rank,
                },
            ),
        ];
        for (name, actual, required) in expected {
            if actual != required {
                return Err(Error::Internal {
                    message: format!(
                        "proposal-head {name} shape mismatch: actual={actual:?} expected={required:?}"
                    ),
                });
            }
        }
        let anchor = i32::try_from(anchor_token_id).map_err(|_| Error::Internal {
            message: "proposal anchor token exceeds i32 ABI".into(),
        })?;
        let mut token_ids = vec![0i32; layout.rows + 1];
        token_ids[0] = anchor;
        self.update_i32_host_mirror(&token_ids, &mut workspace.token_ids)?;
        self.submit_operator(3, |stream| {
            crate::cuda::operators::proposal::proposal_head(
                stream,
                hc_state.as_device_buffer(),
                hc_function.as_device_buffer(),
                hc_scale.as_device_buffer(),
                hc_base.as_device_buffer(),
                norm_weight.as_device_buffer(),
                lm_head.weight,
                &markov_w1.weight,
                &markov_w2.weight,
                confidence_weight.weight,
                workspace.hidden.as_device_buffer_mut(),
                workspace.normalized.as_device_buffer_mut(),
                workspace.base_logits.as_device_buffer_mut(),
                workspace.partial_values.as_device_buffer_mut(),
                workspace.partial_indices.as_device_buffer_mut(),
                workspace
                    .token_ids
                    .device_mut_invalidate_host()
                    .as_device_buffer_mut(),
                workspace.confidence.as_device_buffer_mut(),
                workspace.status.as_device_buffer_mut(),
                layout,
            )
        })?;
        Ok(())
    }
    /// Execute the checkpoint-native proposal stage-zero target-tap projection and
    /// RMSNorm in one cooperative fused semantic launch.
    #[allow(clippy::too_many_arguments)]
    pub fn artifact_main_project_norm_into(
        &self,
        projection: &CudaArtifactLinearHandle,
        norm_weight: &CudaF32Buffer,
        input: &CudaF32Buffer,
        rows: usize,
        rms_eps: f32,
        activation: &mut CudaFp8ActivationPack,
        inv_rms: &mut CudaF32Buffer,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        let projection = projection.operator_view();
        let activation = activation.operator_storage_mut();
        let CudaArtifactLinearShape::Fp8E4M3WithE8M0Scale {
            out_features,
            in_features,
            block_m,
            block_k,
        } = projection.shape
        else {
            return Err(Error::Internal {
                message: format!(
                    "proposal main projection requires FP8/E8M0 weights, got {:?}",
                    projection.shape
                ),
            });
        };
        if block_m != 128 || block_k != 128 || !out_features.is_multiple_of(128) {
            return Err(Error::Internal {
                message: format!(
                    "proposal main projection requires K128/N128 layout, got {:?}",
                    projection.shape
                ),
            });
        }
        let input_len = rows
            .checked_mul(in_features)
            .ok_or_else(|| Error::Internal {
                message: "proposal main projection input size overflow".into(),
            })?;
        let output_len = rows
            .checked_mul(out_features)
            .ok_or_else(|| Error::Internal {
                message: "proposal main projection output size overflow".into(),
            })?;
        let scale_len = rows
            .checked_mul(in_features / 128)
            .ok_or_else(|| Error::Internal {
                message: "proposal main projection scale size overflow".into(),
            })?;
        if input.len() != input_len
            || norm_weight.len() != out_features
            || activation.value_capacity != input_len
            || activation.scale_capacity != scale_len
            || inv_rms.len() != rows
            || output.len() != output_len
        {
            return Err(Error::Internal {
                message: format!(
                    "fused proposal main-project/norm binding mismatch: input={}/{} norm={}/{} activation={}/{} scales={}/{} inv_rms={}/{} output={}/{}",
                    input.len(),
                    input_len,
                    norm_weight.len(),
                    out_features,
                    activation.value_capacity,
                    input_len,
                    activation.scale_capacity,
                    scale_len,
                    inv_rms.len(),
                    rows,
                    output.len(),
                    output_len
                ),
            });
        }
        let weight_scales = projection.scale.ok_or_else(|| Error::Internal {
            message: "proposal main projection weight scales are missing".into(),
        })?;
        self.submit_operator(1, |stream| {
            crate::cuda::providers::cutlass::submit_main_project_norm(
                stream,
                input.as_device_buffer(),
                activation.x_packed,
                activation.x_scales,
                projection.weight,
                weight_scales,
                norm_weight.as_device_buffer(),
                inv_rms.as_device_buffer_mut(),
                output.as_device_buffer_mut(),
                rows,
                in_features,
                out_features,
                rms_eps,
            )
        })?;
        Ok(())
    }
}
