//! Semantic projection execution using borrowed owner resources.

use crate::cuda::operators::OperatorOwner;
use crate::cuda::operators::linear::{
    ARTIFACT_LINEAR_FP8_ACTIVATION_BLOCK_SIZE, CudaArtifactLinearHandle, CudaArtifactLinearShape,
    CudaArtifactLinearWorkspace, CudaBf16Buffer, CudaF32Buffer, CudaFp8ActivationPack,
    CudaOperators, CudaPreparedFp8Activation,
};
use ferrule_common::{Error, Result};

impl CudaOperators {
    pub fn artifact_fp8_projection_rows_from_device_into_with_scratch(
        &self,
        handle: &CudaArtifactLinearHandle,
        input: &CudaF32Buffer,
        rows: usize,
        output: &mut CudaF32Buffer,
        scratch: &mut CudaArtifactLinearWorkspace,
    ) -> Result<()> {
        let handle = handle.operator_view();
        let mut scratch = scratch.operator_storage_mut();
        let CudaArtifactLinearShape::Fp8E4M3WithE8M0Scale {
            out_features,
            in_features,
            block_m: 128,
            block_k: 128,
        } = handle.shape
        else {
            return Err(Error::Internal {
                message: "FP8 projection requires an FP8 K128 artifact".into(),
            });
        };
        let input_len = rows
            .checked_mul(in_features)
            .ok_or_else(|| Error::Internal {
                message: "FP8 projection input size overflow".into(),
            })?;
        let output_len = rows
            .checked_mul(out_features)
            .ok_or_else(|| Error::Internal {
                message: "FP8 projection output size overflow".into(),
            })?;
        if rows == 0 || input.len() != input_len || output.len() != output_len {
            return Err(Error::Internal {
                message: format!(
                    "FP8 projection shape mismatch: rows={rows} input={}/{} output={}/{}",
                    input.len(),
                    input_len,
                    output.len(),
                    output_len
                ),
            });
        }
        let scale_cols = in_features / ARTIFACT_LINEAR_FP8_ACTIVATION_BLOCK_SIZE;
        let scale_len = rows
            .checked_mul(scale_cols)
            .ok_or_else(|| Error::Internal {
                message: "FP8 projection scale size overflow".into(),
            })?;
        if input_len > scratch.value_capacity || scale_len > scratch.scale_capacity {
            return Err(Error::Internal {
                message: format!(
                    "FP8 projection scratch too small: packed={input_len}/{} scales={scale_len}/{}",
                    scratch.value_capacity, scratch.scale_capacity
                ),
            });
        }
        let weight_scales = handle.scale.ok_or_else(|| Error::Internal {
            message: "FP8 projection scales are missing".into(),
        })?;
        self.prepare_operator_fp8(input, rows, in_features, &mut scratch)?;
        self.submit_operator(1, |stream| {
            crate::cuda::providers::cutlass::submit_fp8_projection(
                stream,
                scratch.x_packed,
                scratch.x_scales,
                handle.weight,
                weight_scales,
                output.as_device_buffer_mut(),
                rows,
                out_features,
                in_features,
            )
        })?;
        Ok(())
    }

    /// Execute the required one-launch QueryA+KV FP8 projection bundle.
    /// Any shape, binding, or native-provider mismatch is fatal.
    pub fn artifact_fp8_query_a_kv_into(
        &self,
        query_a: &CudaArtifactLinearHandle,
        key_value: &CudaArtifactLinearHandle,
        activation: &CudaPreparedFp8Activation<'_>,
        query_a_output: &mut CudaF32Buffer,
        key_value_output: &mut CudaF32Buffer,
    ) -> Result<()> {
        let query_a = query_a.operator_view();
        let key_value = key_value.operator_view();
        let activation = activation.operator_view();
        let (
            CudaArtifactLinearShape::Fp8E4M3WithE8M0Scale {
                out_features: query_a_out,
                in_features: query_a_in,
                block_m: query_a_block_m,
                block_k: query_a_block_k,
            },
            CudaArtifactLinearShape::Fp8E4M3WithE8M0Scale {
                out_features: kv_out,
                in_features: kv_in,
                block_m: kv_block_m,
                block_k: kv_block_k,
            },
        ) = (query_a.shape, key_value.shape)
        else {
            return Err(Error::Internal {
                message: format!(
                    "fused FP8 QueryA+KV requires FP8 weights, got query_a={:?} kv={:?}",
                    query_a.shape, key_value.shape
                ),
            });
        };
        if query_a_in != kv_in
            || query_a_in != activation.row_width
            || query_a_block_m != 128
            || query_a_block_k != 128
            || kv_block_m != 128
            || kv_block_k != 128
        {
            return Err(Error::Internal {
                message: format!(
                    "fused FP8 QueryA+KV binding mismatch: query_a={:?} kv={:?} activation_width={}",
                    query_a.shape, key_value.shape, activation.row_width
                ),
            });
        }
        let rows = activation.rows;
        if query_a_output.len() != rows * query_a_out || key_value_output.len() != rows * kv_out {
            return Err(Error::Internal {
                message: format!(
                    "CUTLASS FP8 QueryA+KV output mismatch: query_a={}/{} kv={}/{}",
                    query_a_output.len(),
                    rows * query_a_out,
                    key_value_output.len(),
                    rows * kv_out
                ),
            });
        }
        let query_a_scales = query_a.scale.ok_or_else(|| Error::Internal {
            message: "CUTLASS FP8 QueryA weight scales are missing".into(),
        })?;
        let kv_scales = key_value.scale.ok_or_else(|| Error::Internal {
            message: "CUTLASS FP8 KV weight scales are missing".into(),
        })?;

        self.submit_operator(1, |stream| {
            crate::cuda::providers::cutlass::submit_fp8_query_a_kv(
                stream,
                activation.x_packed,
                activation.x_scales,
                query_a.weight,
                query_a_scales,
                key_value.weight,
                kv_scales,
                query_a_output.as_device_buffer_mut(),
                key_value_output.as_device_buffer_mut(),
                rows,
                query_a_out,
                kv_out,
                query_a_in,
            )
        })?;
        Ok(())
    }

    /// Grouped output-A -> BF16 latent -> output-B MLA transaction. Single-row
    /// execution uses three ordered kernels; wider inputs use one cooperative kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn artifact_mla_output_into(
        &self,
        context: &CudaF32Buffer,
        rows: usize,
        output_a: &CudaArtifactLinearHandle,
        output_b: &CudaArtifactLinearHandle,
        groups: usize,
        group_input: usize,
        rank: usize,
        latent: &mut CudaBf16Buffer,
        workspace: &mut CudaArtifactLinearWorkspace,
        output: &mut CudaF32Buffer,
    ) -> Result<()> {
        let output_a = output_a.operator_view();
        let output_b = output_b.operator_view();
        let workspace = workspace.operator_storage_mut();
        let latent_size = groups.checked_mul(rank).ok_or_else(|| Error::Internal {
            message: "fused FP8 MLA latent size overflow".into(),
        })?;
        let context_size = groups
            .checked_mul(group_input)
            .ok_or_else(|| Error::Internal {
                message: "fused FP8 MLA context size overflow".into(),
            })?;
        let hidden_size = match output_b.shape {
            CudaArtifactLinearShape::Fp8E4M3WithE8M0Scale {
                out_features,
                in_features,
                block_m: 128,
                block_k: 128,
            } if in_features == latent_size => out_features,
            _ => {
                return Err(Error::Internal {
                    message: format!(
                        "fused MLA output-B requires FP8/E8M0 [{hidden_size},{latent_size}], got {:?}",
                        output_b.shape,
                        hidden_size = output.len().checked_div(rows).unwrap_or(0)
                    ),
                });
            }
        };
        if !matches!(
            output_a.shape,
            CudaArtifactLinearShape::Fp8E4M3WithE8M0Scale {
                out_features,
                in_features,
                block_m: 128,
                block_k: 128,
            } if out_features == latent_size && in_features == group_input
        ) {
            return Err(Error::Internal {
                message: format!(
                    "fused MLA output-A requires FP8/E8M0 [{latent_size},{group_input}], got {:?}",
                    output_a.shape
                ),
            });
        }
        let output_a_scales = output_a.scale.ok_or_else(|| Error::Internal {
            message: "fused MLA output-A scales are missing".into(),
        })?;
        let output_b_scales = output_b.scale.ok_or_else(|| Error::Internal {
            message: "fused MLA output-B scales are missing".into(),
        })?;
        self.submit_operator(if rows == 1 { 3 } else { 1 }, |stream| {
            crate::cuda::providers::cutlass::submit_mla_output(
                stream,
                context.as_device_buffer(),
                output_a.weight,
                output_a_scales,
                output_b.weight,
                output_b_scales,
                latent.as_device_buffer_mut(),
                workspace.x_packed,
                workspace.x_scales,
                output.as_device_buffer_mut(),
                rows,
                context_size,
                groups,
                group_input,
                rank,
                latent_size,
                hidden_size,
            )
        })?;
        Ok(())
    }
    pub fn prepare_fp8_activation_from_device<'a>(
        &self,
        input: &CudaF32Buffer,
        rows: usize,
        row_width: usize,
        storage: &'a mut CudaFp8ActivationPack,
    ) -> Result<CudaPreparedFp8Activation<'a>> {
        let expected = rows.checked_mul(row_width).ok_or_else(|| Error::Internal {
            message: "CUDA FP8 activation pack input size overflow".into(),
        })?;
        if rows == 0 || row_width == 0 || input.len() != expected {
            return Err(Error::Internal {
                message: format!(
                    "CUDA FP8 activation pack input mismatch: rows={rows} row_width={row_width} input={}",
                    input.len()
                ),
            });
        }
        self.prepare_operator_fp8(input, rows, row_width, &mut storage.operator_storage_mut())?;
        self.prepared_fp8_activation_from_storage(storage, rows, row_width)
    }
    pub fn prepared_fp8_activation_from_storage<'a>(
        &self,
        storage: &'a CudaFp8ActivationPack,
        rows: usize,
        row_width: usize,
    ) -> Result<CudaPreparedFp8Activation<'a>> {
        storage.prepared_view(rows, row_width)
    }
}

#[cfg(test)]
mod resource_tests {
    use super::*;

    #[test]
    #[ignore = "actual CUDA device; independent family borrows, FP8 pack and real capture replay"]
    fn borrowed_resources_preserve_storage_and_pack_capture_contract() {
        let op = CudaOperators::new_on_device(0).expect("required CUDA owner");
        let artifact = op
            .upload_fp8_e4m3_e8m0_linear(&[0x38; 128 * 128], &[127], 128, 128, 128, 128)
            .unwrap();
        let mut workspace = op.artifact_linear_workspace(2, 128).unwrap();
        let mut pack = op.fp8_activation_pack(2, 128).unwrap();
        let values: Vec<f32> = [vec![1.0; 128], vec![-2.0; 128]].concat();
        let mut input = op.upload_f32_buffer(&values).unwrap();
        let before = op.allocator_metrics();
        op.reset_counters();
        let weight = artifact.operator_view();
        assert_eq!(weight.shape, artifact.shape());
        assert_eq!(weight.weight.len(), 128 * 128);
        assert_eq!(weight.scale.unwrap().len(), 1);
        assert!(std::ptr::eq(weight.weight, artifact.operator_view().weight));
        {
            let scratch = workspace.operator_storage_mut();
            assert_eq!(scratch.value_capacity, 256);
            assert_eq!(scratch.scale_capacity, 2);
            assert_eq!(scratch.x_packed.len(), 256);
            assert_eq!(scratch.x_scales.len(), 2);
        }
        assert!(
            op.prepared_fp8_activation_from_storage(&pack, 1, 128)
                .is_err()
        );
        assert!(
            op.prepared_fp8_activation_from_storage(&pack, 2, 64)
                .is_err()
        );
        assert!(
            op.prepared_fp8_activation_from_storage(&pack, usize::MAX, 128)
                .is_err()
        );
        assert_eq!(op.counters(), Default::default());
        assert_eq!(
            op.allocator_metrics().allocation_requests,
            before.allocation_requests
        );

        // Exact native packing for powers of two: 1 -> 256 * 2^-8,
        // -2 -> -256 * 2^-7. No native FP8 MMA instruction is required.
        let (packed_address, scales_address) = {
            let prepared = op
                .prepare_fp8_activation_from_device(&input, 2, 128, &mut pack)
                .unwrap();
            let view = prepared.operator_view();
            assert_eq!((view.rows, view.row_width), (2, 128));
            op.record_compute_event().unwrap().synchronize().unwrap();
            let stream = op.stream_clone();
            assert_eq!(
                view.x_packed.to_host_vec(&stream).unwrap(),
                [vec![0x78; 128], vec![0xf8; 128]].concat()
            );
            assert_eq!(view.x_scales.to_host_vec(&stream).unwrap(), [119, 120]);
            let packed_address = view.x_packed.cu_deviceptr();
            let scales_address = view.x_scales.cu_deviceptr();
            (packed_address, scales_address)
        };

        // The generic linear workspace exposes only its packed subresources;
        // exercise the same owner preparation adapter without native FP8 GEMM.
        let mut scratch = workspace.operator_storage_mut();
        op.prepare_operator_fp8(&input, 2, 128, &mut scratch)
            .unwrap();
        op.record_compute_event().unwrap().synchronize().unwrap();
        let stream = op.stream_clone();
        assert_eq!(
            scratch.x_packed.to_host_vec(&stream).unwrap(),
            [vec![0x78; 128], vec![0xf8; 128]].concat()
        );
        assert_eq!(scratch.x_scales.to_host_vec(&stream).unwrap(), [119, 120]);

        op.reset_counters();
        op.failpoints().arm_allocation();
        let graph = op
            .capture_decode_graph(|| {
                let prepared = op.prepare_fp8_activation_from_device(&input, 2, 128, &mut pack)?;
                let view = prepared.operator_view();
                assert_eq!(view.x_packed.cu_deviceptr(), packed_address);
                assert_eq!(view.x_scales.cu_deviceptr(), scales_address);
                Ok(())
            })
            .unwrap();
        assert_eq!(op.counters().compute_kernel_launches, 1);
        assert_eq!(op.counters().device_allocation_attempts, 0);
        assert_eq!(op.counters().stream_wide_syncs, 0);
        assert_eq!(
            op.allocator_metrics().allocation_requests,
            before.allocation_requests
        );
        assert!(
            op.zero_f32_buffer(1).is_err(),
            "borrow/pack consumed allocation failpoint"
        );
        for sign in [1.0, -1.0] {
            op.overwrite_f32_buffer(
                &values.iter().map(|v| v * sign).collect::<Vec<_>>(),
                &mut input,
            )
            .unwrap();
            op.launch_graph(&graph).unwrap();
            op.record_compute_event().unwrap().synchronize().unwrap();
            let prepared = op
                .prepared_fp8_activation_from_storage(&pack, 2, 128)
                .unwrap();
            let view = prepared.operator_view();
            let codes = if sign > 0.0 {
                [0x78, 0xf8]
            } else {
                [0xf8, 0x78]
            };
            assert_eq!(
                view.x_packed.to_host_vec(&stream).unwrap(),
                [vec![codes[0]; 128], vec![codes[1]; 128]].concat()
            );
            assert_eq!(view.x_scales.to_host_vec(&stream).unwrap(), [119, 120]);
            assert_eq!(view.x_packed.cu_deviceptr(), packed_address);
            assert_eq!(view.x_scales.cu_deviceptr(), scales_address);
        }
    }
}
