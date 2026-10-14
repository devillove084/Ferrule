//! Semantic compressor execution using borrowed owner resources.

use crate::cuda::operator_owner::{CudaCompressorRecurrentState, checked_u32, cu};
use crate::cuda::operators::OperatorOwner;
use crate::cuda::operators::linear::{
    CudaArtifactLinearHandle, CudaArtifactLinearShape, CudaF32Buffer, CudaOperators,
};
use crate::cuda::runtime::LaunchConfig;
use ferrule_common::{Error, Result};

impl CudaOperators {
    /// Execute the BF16 compressor dual projection semantic operator.
    pub fn artifact_bf16_compressor_into(
        &self,
        projection1: &CudaArtifactLinearHandle,
        projection2: &CudaArtifactLinearHandle,
        activation: &CudaF32Buffer,
        rows: usize,
        projection1_output: &mut CudaF32Buffer,
        projection2_output: &mut CudaF32Buffer,
    ) -> Result<()> {
        Self::f32_no_alias(&[activation], &[projection1_output, projection2_output])?;
        let projection1 = projection1.operator_view();
        let projection2 = projection2.operator_view();
        let (
            CudaArtifactLinearShape::Bf16Bytes {
                out_features: n1,
                in_features: k1,
            },
            CudaArtifactLinearShape::Bf16Bytes {
                out_features: n2,
                in_features: k2,
            },
        ) = (projection1.shape, projection2.shape)
        else {
            return Err(Error::Internal {
                message: format!(
                    "BF16 compressor requires BF16 weights, got first={:?} second={:?}",
                    projection1.shape, projection2.shape
                ),
            });
        };
        if k1 != k2 || activation.len() != rows * k1 {
            return Err(Error::Internal {
                message: format!(
                    "BF16 compressor input mismatch: rows={rows} first_k={k1} second_k={k2} input={}",
                    activation.len()
                ),
            });
        }
        if projection1_output.len() != rows * n1 || projection2_output.len() != rows * n2 {
            return Err(Error::Internal {
                message: format!(
                    "BF16 compressor output mismatch: first={}/{} second={}/{}",
                    projection1_output.len(),
                    rows * n1,
                    projection2_output.len(),
                    rows * n2
                ),
            });
        }
        self.submit_operator(1, |stream| {
            crate::cuda::providers::cutlass::submit_bf16_compressor(
                stream,
                activation.as_device_buffer(),
                projection1.weight,
                projection2.weight,
                projection1_output.as_device_buffer_mut(),
                projection2_output.as_device_buffer_mut(),
                rows,
                n1,
                n2,
                k1,
            )
        })?;
        Ok(())
    }

    pub fn reset_compressor_recurrent_state(
        &self,
        state: &mut CudaCompressorRecurrentState,
    ) -> Result<()> {
        let state = state.operator_state_mut();
        let state_elements = state.shape.state_elements()?;
        self.submit_operator(1, |stream| {
            cu(unsafe {
                self.core_native().compressor_recurrent_reset_f32(
                    stream,
                    LaunchConfig::for_num_elems(checked_u32(
                        state_elements,
                        "compressor recurrent reset",
                        "state_elements",
                    )?),
                    state.kv_state.as_device_buffer_mut(),
                    state.score_state.as_device_buffer_mut(),
                    checked_u32(
                        state_elements,
                        "compressor recurrent reset",
                        "state_elements",
                    )?,
                )
            })
        })
    }

    pub fn compressor_recurrent_seed_prefill(
        &self,
        state: &mut CudaCompressorRecurrentState,
        projected_kv_rows: &CudaF32Buffer,
        projected_score_rows: &CudaF32Buffer,
        ape: &CudaF32Buffer,
        tokens: usize,
    ) -> Result<usize> {
        let state = state.operator_state_mut();
        Self::f32_no_alias(
            &[projected_kv_rows, projected_score_rows, ape],
            &[&*state.kv_state, &*state.score_state],
        )?;
        state.shape.validate()?;
        let projected_elements =
            tokens
                .checked_mul(state.shape.out_dim)
                .ok_or_else(|| Error::Internal {
                    message: "compressor recurrent seed size overflow".into(),
                })?;
        if projected_kv_rows.len() != projected_elements
            || projected_score_rows.len() != projected_elements
            || ape.len() != state.shape.ape_elements()?
        {
            return Err(Error::Internal {
                message: format!(
                    "compressor recurrent seed length mismatch: kv={} score={} ape={} expected projected={} ape={}",
                    projected_kv_rows.len(),
                    projected_score_rows.len(),
                    ape.len(),
                    projected_elements,
                    state.shape.ape_elements()?
                ),
            });
        }
        let state_elements = state.shape.state_elements()?;
        self.submit_operator(1, |stream| {
            cu(unsafe {
                self.core_native().compressor_recurrent_seed_prefill_f32(
                    stream,
                    LaunchConfig::for_num_elems(checked_u32(
                        state_elements,
                        "compressor recurrent seed",
                        "state_elements",
                    )?),
                    projected_kv_rows.as_device_buffer(),
                    projected_score_rows.as_device_buffer(),
                    ape.as_device_buffer(),
                    state.kv_state.as_device_buffer_mut(),
                    state.score_state.as_device_buffer_mut(),
                    checked_u32(tokens, "compressor recurrent seed", "tokens")?,
                    checked_u32(state.shape.ratio, "compressor recurrent seed", "ratio")?,
                    checked_u32(state.shape.out_dim, "compressor recurrent seed", "out_dim")?,
                    if state.shape.overlap { 1 } else { 0 },
                    checked_u32(
                        state_elements,
                        "compressor recurrent seed",
                        "state_elements",
                    )?,
                )
            })
        })?;
        Ok(state.shape.prefill_groups(tokens))
    }

    pub fn compressor_recurrent_append_projected(
        &self,
        state: &mut CudaCompressorRecurrentState,
        projected_kv: &CudaF32Buffer,
        projected_score: &CudaF32Buffer,
        ape: &CudaF32Buffer,
        position: usize,
    ) -> Result<bool> {
        let state = state.operator_state_mut();
        Self::f32_no_alias(
            &[projected_kv, projected_score, ape],
            &[&*state.kv_state, &*state.score_state],
        )?;
        state.shape.validate()?;
        if projected_kv.len() != state.shape.out_dim
            || projected_score.len() != state.shape.out_dim
            || ape.len() != state.shape.ape_elements()?
        {
            return Err(Error::Internal {
                message: format!(
                    "compressor recurrent append length mismatch: kv={} score={} ape={} expected row={} ape={}",
                    projected_kv.len(),
                    projected_score.len(),
                    ape.len(),
                    state.shape.out_dim,
                    state.shape.ape_elements()?
                ),
            });
        }
        self.submit_operator(1, |stream| {
            cu(unsafe {
                self.core_native()
                    .compressor_recurrent_append_projected_f32(
                        stream,
                        LaunchConfig::for_num_elems(checked_u32(
                            state.shape.out_dim,
                            "compressor recurrent append",
                            "out_dim",
                        )?),
                        projected_kv.as_device_buffer(),
                        projected_score.as_device_buffer(),
                        ape.as_device_buffer(),
                        state.kv_state.as_device_buffer_mut(),
                        state.score_state.as_device_buffer_mut(),
                        checked_u32(position, "compressor recurrent append", "position")?,
                        checked_u32(state.shape.ratio, "compressor recurrent append", "ratio")?,
                        checked_u32(
                            state.shape.out_dim,
                            "compressor recurrent append",
                            "out_dim",
                        )?,
                        if state.shape.overlap { 1 } else { 0 },
                    )
            })
        })?;
        Ok(state.shape.is_boundary(position))
    }

    /// Compresses the current recurrent window into `output` and advances the
    /// overlap state in-place. This method performs no allocation or D2H copy.
    pub fn compressor_recurrent_boundary_into(
        &self,
        state: &mut CudaCompressorRecurrentState,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        let state = state.operator_state_mut();
        Self::f32_no_alias(&[&*state.kv_state, &*state.score_state], &[output])?;
        state.shape.validate()?;
        if output.len() != state.shape.head_dim {
            return Err(Error::Internal {
                message: format!(
                    "compressor recurrent output length mismatch: got {} expected {}",
                    output.len(),
                    state.shape.head_dim
                ),
            });
        }
        self.submit_operator(1, |stream| {
            cu(unsafe {
                self.core_native().compressor_recurrent_softmax_f32(
                    stream,
                    LaunchConfig::for_num_elems(checked_u32(
                        state.shape.head_dim,
                        "compressor recurrent boundary",
                        "head_dim",
                    )?),
                    state.kv_state.as_device_buffer(),
                    state.score_state.as_device_buffer(),
                    output.as_device_buffer_mut(),
                    checked_u32(state.shape.ratio, "compressor recurrent boundary", "ratio")?,
                    checked_u32(
                        state.shape.head_dim,
                        "compressor recurrent boundary",
                        "head_dim",
                    )?,
                    checked_u32(
                        state.shape.out_dim,
                        "compressor recurrent boundary",
                        "out_dim",
                    )?,
                    if state.shape.overlap { 1 } else { 0 },
                )
            })
        })?;
        if state.shape.overlap {
            let half = state
                .shape
                .ratio
                .checked_mul(state.shape.out_dim)
                .ok_or_else(|| Error::Internal {
                    message: "compressor recurrent half overflow".into(),
                })?;
            self.copy_f32_within(state.kv_state, half, 0, half)?;
            self.copy_f32_within(state.score_state, half, 0, half)?;
        }
        Ok(())
    }

    pub fn compressor_prefill_softmax_from_device(
        &self,
        kv_rows: &CudaF32Buffer,
        score_rows: &CudaF32Buffer,
        ape: &[f32],
        groups: usize,
        ratio: usize,
        head_dim: usize,
        out_dim: usize,
        overlap: bool,
    ) -> Result<CudaF32Buffer> {
        let ape_dev = self.upload_f32_buffer(ape)?;
        let mut output = self.zero_f32_buffer(groups.checked_mul(head_dim).ok_or_else(
            || Error::Internal {
                message: "CUDA compressor output size overflow".into(),
            },
        )?)?;
        self.compressor_prefill_softmax_from_device_into(
            kv_rows,
            score_rows,
            &ape_dev,
            groups,
            ratio,
            head_dim,
            out_dim,
            overlap,
            &mut output,
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compressor_prefill_softmax_from_device_into(
        &self,
        kv_rows: &CudaF32Buffer,
        score_rows: &CudaF32Buffer,
        ape: &CudaF32Buffer,
        groups: usize,
        ratio: usize,
        head_dim: usize,
        out_dim: usize,
        overlap: bool,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        Self::f32_no_alias(&[kv_rows, score_rows, ape], &[output])?;
        if ratio == 0 || head_dim == 0 || out_dim == 0 {
            return Err(Error::Internal {
                message: format!(
                    "invalid CUDA compressor shape: groups={groups} ratio={ratio} head_dim={head_dim} out_dim={out_dim}"
                ),
            });
        }
        let consumed_tokens = groups.checked_mul(ratio).ok_or_else(|| Error::Internal {
            message: "CUDA compressor token count overflow".into(),
        })?;
        let min_projected =
            consumed_tokens
                .checked_mul(out_dim)
                .ok_or_else(|| Error::Internal {
                    message: "CUDA compressor projected row size overflow".into(),
                })?;
        if kv_rows.len() < min_projected
            || score_rows.len() < min_projected
            || !kv_rows.len().is_multiple_of(out_dim)
            || !score_rows.len().is_multiple_of(out_dim)
            || kv_rows.len() != score_rows.len()
        {
            return Err(Error::Internal {
                message: format!(
                    "CUDA compressor projected length mismatch: kv={} score={} min_required={min_projected} out_dim={out_dim}",
                    kv_rows.len(),
                    score_rows.len()
                ),
            });
        }
        let expected_ape = ratio.checked_mul(out_dim).ok_or_else(|| Error::Internal {
            message: "CUDA compressor APE size overflow".into(),
        })?;
        if ape.len() != expected_ape {
            return Err(Error::Internal {
                message: format!(
                    "CUDA compressor APE length mismatch: got {} expected {expected_ape}",
                    ape.len()
                ),
            });
        }
        let expected_output = groups
            .checked_mul(head_dim)
            .ok_or_else(|| Error::Internal {
                message: "CUDA compressor output size overflow".into(),
            })?;
        if output.len() != expected_output {
            return Err(Error::Internal {
                message: format!(
                    "CUDA compressor output length mismatch: got {} expected {expected_output}",
                    output.len()
                ),
            });
        }
        if groups == 0 {
            return Ok(());
        }
        self.submit_operator(1, |stream| {
            cu(unsafe {
                self.core_native().dsv4_compressor_prefill_softmax(
                    stream,
                    LaunchConfig::for_num_elems(expected_output as u32),
                    kv_rows.as_device_buffer(),
                    score_rows.as_device_buffer(),
                    ape.as_device_buffer(),
                    output.as_device_buffer_mut(),
                    groups as u32,
                    ratio as u32,
                    head_dim as u32,
                    out_dim as u32,
                    if overlap { 1u32 } else { 0u32 },
                )
            })
        })
    }
}
