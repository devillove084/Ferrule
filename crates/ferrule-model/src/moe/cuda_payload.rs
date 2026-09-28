//! Storage payload to CUDA expert encoding/shape adapter.
//!
//! Grouping/byte validation is independent of read transport and submission.
//! This module has no operation, reservation, lease, or publication authority.

use super::write_debug_artifact;
use crate::moe::streaming::storage::PinnedExpertTensorPayload;
use crate::moe::streaming::{
    ExpertId, ExpertLinearFormat, ExpertLoadSource, ExpertMatrixKind, PinnedExpertArtifactPayload,
    infer_expert_linear_format,
};
use ferrule_backend::cuda::operators::moe::{CudaPinnedU8HostBuffer, CudaRoutedExpertShape};
use ferrule_common::{Error, Result};
use std::collections::BTreeMap;
use std::path::Path;

struct PinnedExpertLinear {
    matrix: ExpertMatrixKind,
    format: ExpertLinearFormat,
    weight: CudaPinnedU8HostBuffer,
    scale: CudaPinnedU8HostBuffer,
}

pub(super) struct PinnedExpertBundle {
    expert: ExpertId,
    gate: PinnedExpertLinear,
    up: PinnedExpertLinear,
    down: PinnedExpertLinear,
    bytes: u64,
}

impl PinnedExpertBundle {
    pub(super) fn expert(&self) -> ExpertId {
        self.expert
    }
    pub(super) fn bytes(&self) -> u64 {
        self.bytes
    }
    pub(super) fn shape(&self) -> Result<CudaRoutedExpertShape> {
        pinned_routed_expert_shape(self)
    }

    /// Move the original six slab views in backend gate/up/down weight/scale order.
    pub(super) fn into_upload_buffers(self) -> [CudaPinnedU8HostBuffer; 6] {
        let Self { gate, up, down, .. } = self;
        [
            gate.weight,
            gate.scale,
            up.weight,
            up.scale,
            down.weight,
            down.scale,
        ]
    }
}

impl PinnedExpertBundle {
    pub(super) fn debug_dump_pinned(&self, directory: &Path) -> Result<()> {
        let prefix = format!("layer{}.expert{}", self.expert.layer, self.expert.expert);
        for (matrix, linear) in [("gate", &self.gate), ("up", &self.up), ("down", &self.down)] {
            write_debug_artifact(
                &directory.join(format!("{prefix}.{matrix}.weight.pinned.bin")),
                linear.weight.as_slice(),
            )?;
            write_debug_artifact(
                &directory.join(format!("{prefix}.{matrix}.scale.pinned.bin")),
                linear.scale.as_slice(),
            )?;
        }
        Ok(())
    }

    pub(super) fn from_payload(payload: PinnedExpertArtifactPayload) -> Result<Self> {
        let expert = payload.expert;
        let mut grouped = BTreeMap::<ExpertMatrixKind, Vec<PinnedExpertTensorPayload>>::new();
        for tensor in payload.tensors {
            if tensor.slice.key.expert != expert {
                return Err(Error::Model {
                    message: "physical pinned expert payload identity mismatch".into(),
                });
            }
            grouped
                .entry(tensor.slice.key.matrix)
                .or_default()
                .push(tensor);
        }
        let gate = PinnedExpertLinear::from_tensors(
            expert,
            ExpertMatrixKind::Gate,
            grouped.remove(&ExpertMatrixKind::Gate).unwrap_or_default(),
        )?;
        let up = PinnedExpertLinear::from_tensors(
            expert,
            ExpertMatrixKind::Up,
            grouped.remove(&ExpertMatrixKind::Up).unwrap_or_default(),
        )?;
        let down = PinnedExpertLinear::from_tensors(
            expert,
            ExpertMatrixKind::Down,
            grouped.remove(&ExpertMatrixKind::Down).unwrap_or_default(),
        )?;
        let bytes = [
            gate.weight.len(),
            gate.scale.len(),
            up.weight.len(),
            up.scale.len(),
            down.weight.len(),
            down.scale.len(),
        ]
        .into_iter()
        .try_fold(0u64, |total, bytes| {
            let bytes = u64::try_from(bytes).map_err(|_| Error::Model {
                message: "physical pinned expert payload component exceeds u64".into(),
            })?;
            total.checked_add(bytes).ok_or_else(|| Error::Model {
                message: "physical pinned expert payload byte total overflow".into(),
            })
        })?;
        Ok(Self {
            expert,
            gate,
            up,
            down,
            bytes,
        })
    }
}

impl PinnedExpertLinear {
    fn from_tensors(
        expert: ExpertId,
        matrix: ExpertMatrixKind,
        tensors: Vec<PinnedExpertTensorPayload>,
    ) -> Result<Self> {
        let mut weight = None;
        let mut scale = None;
        for tensor in tensors {
            if tensor.slice.key.expert != expert || tensor.slice.key.matrix != matrix {
                return Err(Error::Model {
                    message: "physical pinned expert tensor identity mismatch".into(),
                });
            }
            match tensor.slice.component {
                crate::moe::streaming::ExpertTensorComponent::Weight => {
                    if weight.replace(tensor).is_some() {
                        return Err(Error::Model {
                            message: "duplicate physical expert weight".into(),
                        });
                    }
                }
                crate::moe::streaming::ExpertTensorComponent::Scale => {
                    if scale.replace(tensor).is_some() {
                        return Err(Error::Model {
                            message: "duplicate physical expert scale".into(),
                        });
                    }
                }
                crate::moe::streaming::ExpertTensorComponent::Other(component) => {
                    return Err(Error::Model {
                        message: format!(
                            "unsupported physical expert tensor component {component}"
                        ),
                    });
                }
            }
        }
        let weight = weight.ok_or_else(|| Error::Model {
            message: "missing physical expert weight".into(),
        })?;
        let scale = scale.ok_or_else(|| Error::Model {
            message: "missing physical expert scale".into(),
        })?;
        let format = infer_expert_linear_format(
            &weight.slice,
            weight.bytes.len(),
            Some((&scale.slice, scale.bytes.len())),
        )?;
        Ok(Self {
            matrix,
            format,
            weight: weight.bytes,
            scale: scale.bytes,
        })
    }
}

fn pinned_linear_dimensions(linear: &PinnedExpertLinear) -> Result<(usize, usize)> {
    let ExpertLinearFormat::Fp4E2M1PackedWithE8M0Scale {
        out_features,
        in_features,
        block_size: 32,
    } = linear.format
    else {
        return Err(Error::Model {
            message: format!(
                "CUDA physical expert {:?} requires FP4 E2M1/E8M0 block_size=32",
                linear.matrix
            ),
        });
    };
    validate_mxfp4_linear_storage(
        out_features,
        in_features,
        linear.weight.len(),
        linear.scale.len(),
    )?;
    Ok((out_features, in_features))
}

fn pinned_routed_expert_shape(bundle: &PinnedExpertBundle) -> Result<CudaRoutedExpertShape> {
    let (gate_out, gate_in) = pinned_linear_dimensions(&bundle.gate)?;
    let (up_out, up_in) = pinned_linear_dimensions(&bundle.up)?;
    let (down_out, down_in) = pinned_linear_dimensions(&bundle.down)?;
    if (up_out, up_in) != (gate_out, gate_in) || down_in != gate_out {
        return Err(Error::Model {
            message: format!(
                "inconsistent CUDA routed-expert projection dimensions: gate={gate_out}x{gate_in} up={up_out}x{up_in} down={down_out}x{down_in}"
            ),
        });
    }
    CudaRoutedExpertShape::new(gate_in, gate_out, down_out)
}

fn validate_mxfp4_linear_storage(
    out_features: usize,
    in_features: usize,
    weight_bytes: usize,
    scale_bytes: usize,
) -> Result<()> {
    if out_features == 0
        || in_features == 0
        || !in_features.is_multiple_of(32)
        || !in_features.is_multiple_of(2)
    {
        return Err(Error::Model {
            message: format!(
                "invalid CUDA physical expert FP4 shape: out={out_features} in={in_features}"
            ),
        });
    }
    let expected_weight =
        out_features
            .checked_mul(in_features / 2)
            .ok_or_else(|| Error::Model {
                message: "physical expert FP4 weight storage overflow".into(),
            })?;
    let expected_scale =
        out_features
            .checked_mul(in_features / 32)
            .ok_or_else(|| Error::Model {
                message: "physical expert linear scale storage overflow".into(),
            })?;
    if weight_bytes != expected_weight || scale_bytes != expected_scale {
        return Err(Error::Model {
            message: format!(
                "physical expert FP4 storage mismatch: weight={weight_bytes}/{expected_weight} scale={scale_bytes}/{expected_scale}"
            ),
        });
    }
    Ok(())
}

pub(super) fn source_routed_expert_shape(
    source: &ExpertLoadSource,
) -> Result<CudaRoutedExpertShape> {
    let tensors = match source {
        ExpertLoadSource::LocalTensorSet { tensors }
        | ExpertLoadSource::HfLocalTensorSet { tensors, .. } => tensors,
        _ => {
            return Err(Error::Model {
                message: "CUDA physical expert source does not expose tensor shapes".into(),
            });
        }
    };
    let mut dimensions = BTreeMap::new();
    for matrix in [
        ExpertMatrixKind::Gate,
        ExpertMatrixKind::Up,
        ExpertMatrixKind::Down,
    ] {
        let mut weight = None;
        let mut scale = None;
        for tensor in tensors.iter().filter(|tensor| tensor.key.matrix == matrix) {
            match tensor.component {
                crate::moe::streaming::ExpertTensorComponent::Weight => {
                    if weight.replace(tensor).is_some() {
                        return Err(Error::Model {
                            message: "duplicate physical expert source weight".into(),
                        });
                    }
                }
                crate::moe::streaming::ExpertTensorComponent::Scale => {
                    if scale.replace(tensor).is_some() {
                        return Err(Error::Model {
                            message: "duplicate physical expert source scale".into(),
                        });
                    }
                }
                crate::moe::streaming::ExpertTensorComponent::Other(_) => {}
            }
        }
        let weight = weight.ok_or_else(|| Error::Model {
            message: "missing physical expert source weight".into(),
        })?;
        let scale = scale.ok_or_else(|| Error::Model {
            message: "missing physical expert source scale".into(),
        })?;
        let weight_bytes = usize::try_from(weight.bytes).map_err(|_| Error::Model {
            message: "physical expert source weight exceeds usize".into(),
        })?;
        let scale_bytes = usize::try_from(scale.bytes).map_err(|_| Error::Model {
            message: "physical expert source scale exceeds usize".into(),
        })?;
        let format = infer_expert_linear_format(weight, weight_bytes, Some((scale, scale_bytes)))?;
        let ExpertLinearFormat::Fp4E2M1PackedWithE8M0Scale {
            out_features,
            in_features,
            block_size: 32,
        } = format
        else {
            return Err(Error::Model {
                message: "CUDA physical expert source requires FP4 E2M1/E8M0 block_size=32".into(),
            });
        };
        validate_mxfp4_linear_storage(out_features, in_features, weight_bytes, scale_bytes)?;
        dimensions.insert(matrix, (out_features, in_features));
    }
    let (gate_out, gate_in) = dimensions[&ExpertMatrixKind::Gate];
    let (up_out, up_in) = dimensions[&ExpertMatrixKind::Up];
    let (down_out, down_in) = dimensions[&ExpertMatrixKind::Down];
    if (up_out, up_in) != (gate_out, gate_in) || down_in != gate_out {
        return Err(Error::Model {
            message: format!(
                "inconsistent CUDA routed-expert source dimensions: gate={gate_out}x{gate_in} up={up_out}x{up_in} down={down_out}x{down_in}"
            ),
        });
    }
    CudaRoutedExpertShape::new(gate_in, gate_out, down_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moe::streaming::{ExpertTensorComponent, ExpertTensorKey, ExpertTensorSlice};
    use ferrule_backend::cuda::operators::moe::CudaOperators;

    fn slices(expert: ExpertId) -> Vec<ExpertTensorSlice> {
        [
            (ExpertMatrixKind::Gate, 128, 128),
            (ExpertMatrixKind::Up, 128, 128),
            (ExpertMatrixKind::Down, 64, 128),
        ]
        .into_iter()
        .flat_map(|(matrix, out, input)| {
            [
                (ExpertTensorComponent::Weight, "I8", input / 2),
                (ExpertTensorComponent::Scale, "F8_E8M0", input / 32),
            ]
            .map(|(component, dtype, columns)| ExpertTensorSlice {
                key: ExpertTensorKey { expert, matrix },
                component,
                path: "metadata-only-expert.safetensors".into(),
                offset: 0,
                bytes: (out * columns) as u64,
                dtype: dtype.into(),
                shape: vec![out, columns],
            })
        })
        .collect()
    }

    #[test]
    fn source_adapter_validates_metadata_without_reading_or_allocating_device_storage() {
        let shape = source_routed_expert_shape(&ExpertLoadSource::LocalTensorSet {
            tensors: slices(ExpertId::new(0, 0)),
        })
        .unwrap();
        assert_eq!(
            (shape.input, shape.intermediate, shape.output),
            (128, 128, 64)
        );
    }

    #[test]
    fn source_adapter_preserves_duplicate_missing_and_storage_errors() {
        let original = slices(ExpertId::new(0, 0));
        let mut duplicate = original.clone();
        duplicate.push(original[0].clone());
        let mut missing = original.clone();
        missing.remove(1);
        let mut short = original;
        short[0].bytes -= 1;
        for (tensors, message) in [
            (duplicate, "duplicate physical expert source weight"),
            (missing, "missing physical expert source scale"),
            (short, "byte length mismatch"),
        ] {
            let error = source_routed_expert_shape(&ExpertLoadSource::LocalTensorSet { tensors })
                .unwrap_err();
            assert!(error.to_string().contains(message), "{error}");
        }
    }

    #[test]
    fn cuda_storage_validation_rejects_overflow_and_bad_alignment() {
        assert!(
            validate_mxfp4_linear_storage(usize::MAX, 128, 0, 0)
                .unwrap_err()
                .to_string()
                .contains("overflow")
        );
        assert!(
            validate_mxfp4_linear_storage(128, 31, 0, 0)
                .unwrap_err()
                .to_string()
                .contains("invalid CUDA physical expert FP4 shape")
        );
        assert!(
            validate_mxfp4_linear_storage(128, 128, 8192, 511)
                .unwrap_err()
                .to_string()
                .contains("storage mismatch")
        );
    }

    fn pinned_payload(ops: &CudaOperators, expert: ExpertId) -> PinnedExpertArtifactPayload {
        let mut tensors = slices(expert)
            .into_iter()
            .enumerate()
            .map(|(index, slice)| {
                let bytes = ops
                    .pin_u8_host_buffer(&vec![index as u8 + 1; slice.bytes as usize])
                    .expect("pin storage payload");
                PinnedExpertTensorPayload { slice, bytes }
            })
            .collect::<Vec<_>>();
        tensors.reverse();
        PinnedExpertArtifactPayload { expert, tensors }
    }

    #[test]
    #[ignore = "requires a real CUDA device; run this exact pinned-boundary test explicitly"]
    fn gpu_pinned_payload_handoff_preserves_views_and_encoding_errors() {
        let ops = CudaOperators::new_on_device(0).expect("pinned boundary test requires CUDA");
        let expert = ExpertId::new(0, 0);
        let payload = pinned_payload(&ops, expert);
        let original_pointers = payload
            .tensors
            .iter()
            .rev()
            .map(|tensor| tensor.bytes.as_ptr())
            .collect::<Vec<_>>();
        let bundle = PinnedExpertBundle::from_payload(payload).unwrap();
        assert_eq!(bundle.expert(), expert);
        assert_eq!(bundle.bytes(), 21760);
        let buffers = bundle.into_upload_buffers();
        for (index, buffer) in buffers.iter().enumerate() {
            assert_eq!(
                buffer.as_ptr(),
                original_pointers[index],
                "boundary must move, not copy"
            );
            assert!(
                buffer.is_uniquely_owned(),
                "boundary must not create a second slab owner"
            );
            assert!(
                buffer
                    .as_slice()
                    .iter()
                    .all(|byte| *byte == index as u8 + 1)
            );
        }
        drop(buffers);

        let mut wrong_identity = pinned_payload(&ops, expert);
        wrong_identity.tensors[0].slice.key.expert = ExpertId::new(0, 1);
        let mut duplicate = pinned_payload(&ops, expert);
        let first = &duplicate.tensors[0];
        duplicate.tensors.push(PinnedExpertTensorPayload {
            slice: first.slice.clone(),
            bytes: first.bytes.clone(),
        });
        let mut short = pinned_payload(&ops, expert);
        let first = &mut short.tensors[0];
        first.bytes = first.bytes.slice(0, first.bytes.len() - 1).unwrap();
        for (payload, message) in [
            (wrong_identity, "payload identity mismatch"),
            (duplicate, "duplicate physical expert scale"),
            (short, "byte length mismatch"),
        ] {
            let error = match PinnedExpertBundle::from_payload(payload) {
                Ok(_) => panic!("malformed payload accepted: {message}"),
                Err(error) => error,
            };
            assert!(error.to_string().contains(message), "{error}");
        }
    }

    // Diagnostic-only legacy provider access. Production adapters continue to
    // consume opaque backend capabilities rather than native layout/ABI types.
    fn fp4_layout_capability() -> (bool, usize) {
        use ferrule_backend::cuda::providers::{COMPILED_TARGET, compiled_capabilities, cutlass};
        let provider = cutlass::discover_provider().expect("compiled CUTLASS manifest");
        let grouped_fp4 = provider.supports(cutlass::CutlassKernelId::GroupedFp4Moe);
        let prepared_bytes = cutlass::mxfp4_sfb_storage_bytes(128, 128).unwrap();
        eprintln!(
            "PR22 compiled_target={COMPILED_TARGET} capabilities={:?} kernel_mask={:#x}              grouped_fp4={grouped_fp4} native_sfb_bytes_128x128={prepared_bytes}",
            compiled_capabilities(),
            provider.manifest().kernel_mask,
        );
        (grouped_fp4, prepared_bytes)
    }

    #[test]
    #[ignore = "requires a real CUDA device and a build without the SM103 grouped FP4 provider"]
    fn gpu_unsupported_fp4_upload_matches_pre_m2_unknown_contract() {
        use super::super::{CudaExpertFrame, upload};
        use ferrule_common::{CompletionHub, QuiescenceEvidence};

        let ops = CudaOperators::new_on_device(0).expect("unsupported contract requires CUDA");
        let (grouped_fp4, prepared_scale_bytes) = fp4_layout_capability();
        assert!(
            !grouped_fp4,
            "this rejection contract requires an unsupported build; use the success gate on supported hardware"
        );
        assert_eq!(
            prepared_scale_bytes, 0,
            "record the native unsupported sizing contract"
        );
        let expert = ExpertId::new(0, 0);
        let before = PinnedExpertBundle::from_payload(pinned_payload(&ops, expert)).unwrap();
        let after = PinnedExpertBundle::from_payload(pinned_payload(&ops, expert)).unwrap();
        let expected_bytes = before.bytes();
        let mut arena = ops
            .allocate_routed_expert_arena(before.shape().unwrap(), 2)
            .unwrap();
        let mut before_frame = CudaExpertFrame {
            expert: arena.allocate_frame().unwrap(),
        };
        let after_frame = CudaExpertFrame {
            expert: arena.allocate_frame().unwrap(),
        };

        // Reproduce the pre-M2 submit call from cuda_materialization.rs, including
        // its conservative error-custody rule. This bypasses the new adapter.
        let PinnedExpertBundle { gate, up, down, .. } = before;
        ops.reset_counters();
        let baseline_error = match ops.materialize_routed_expert_from_pinned_async(
            &mut before_frame.expert,
            gate.weight,
            gate.scale,
            up.weight,
            up.scale,
            down.weight,
            down.scale,
        ) {
            Ok(ticket) => {
                // Keep even an unexpected success safe until the native fence completes.
                if ticket.synchronize().is_err() {
                    std::mem::forget(before_frame);
                }
                panic!("unsupported baseline unexpectedly accepted the FP4 upload");
            }
            Err(error) => {
                // The backend error does not provide a returned-custody proof,
                // even when the backend attempted its own cleanup synchronization.
                std::mem::forget(before_frame);
                error
            }
        };
        let baseline_counters = ops.counters();
        let baseline_message = baseline_error.to_string();
        assert!(baseline_message.contains("CUDA routed-expert private layout preparation failed"));

        ops.reset_counters();
        let result = upload::submit(&ops, &CompletionHub::new(), after, after_frame);
        let (adapter_error, evidence) = match result {
            Ok((ticket, _)) => {
                drop(ticket);
                panic!("unsupported adapter unexpectedly accepted the FP4 upload");
            }
            Err(failure) => failure,
        };
        let adapter_counters = ops.counters();
        assert_eq!(adapter_error.to_string(), baseline_message);
        assert_eq!(evidence, QuiescenceEvidence::Unknown);
        assert_eq!(baseline_counters.host_to_device_copies, 6);
        assert_eq!(
            adapter_counters.host_to_device_copies,
            baseline_counters.host_to_device_copies
        );
        assert_eq!(baseline_counters.host_to_device_bytes, expected_bytes);
        assert_eq!(adapter_counters.host_to_device_bytes, expected_bytes);
        assert_eq!(
            adapter_counters.upload_kernel_launches,
            baseline_counters.upload_kernel_launches
        );
        assert_eq!(
            adapter_counters.stream_wide_syncs,
            baseline_counters.stream_wide_syncs
        );
        eprintln!(
            "PR22 rejection contract: pre_m2={baseline_message}; adapter={adapter_error}; evidence={evidence:?}"
        );
        eprintln!(
            "PR22 pre_m2_counters={baseline_counters:?}; adapter_counters={adapter_counters:?}"
        );
        eprintln!(
            "PR22 full_fp4_upload_success=PENDING_SUPPORTED_HARDWARE; no install submitted and unknown frames not recycled"
        );
    }

    #[test]
    #[ignore = "requires real CUDA and an unsupported grouped FP4 build; bounded unknown-custody leak"]
    fn gpu_unsupported_upload_cleanup_failure_retains_pinned_sources_and_frame() {
        use super::super::{CudaExpertFrame, upload};
        use ferrule_common::{CompletionHub, QuiescenceEvidence};
        let ops = CudaOperators::new_on_device(0).expect("real CUDA required");
        let (supported, scale_bytes) = fp4_layout_capability();
        assert!(
            !supported && scale_bytes == 0,
            "unsupported-build fault contract only"
        );
        let payload = pinned_payload(&ops, ExpertId::new(0, 0));
        let sources = payload
            .tensors
            .iter()
            .map(|tensor| tensor.bytes.clone())
            .collect::<Vec<_>>();
        let bundle = PinnedExpertBundle::from_payload(payload).unwrap();
        let mut arena = ops
            .allocate_routed_expert_arena(bundle.shape().unwrap(), 1)
            .unwrap();
        let frame = CudaExpertFrame {
            expert: arena.allocate_frame().unwrap(),
        };
        let live_bytes = ops.allocator_metrics().live_requested_bytes;
        ops.reset_counters();
        // Private-layout rejection happens after six real H2D copies. Withhold
        // the cleanup fence using the existing backend sync failpoint.
        ops.failpoints().arm_stream_sync();
        let (error, evidence) = match upload::submit(&ops, &CompletionHub::new(), bundle, frame) {
            Err(failure) => failure,
            Ok(_) => panic!("unsupported upload unexpectedly succeeded"),
        };
        assert_eq!(evidence, QuiescenceEvidence::Unknown);
        let message = error.to_string();
        assert!(
            message.contains("private layout preparation failed"),
            "{message}"
        );
        assert!(
            message.contains("synchronizing the upload stream also failed"),
            "{message}"
        );
        assert_eq!(ops.counters().host_to_device_copies, 6);
        assert_eq!(ops.counters().host_to_device_bytes, 21760);
        assert_eq!(ops.counters().stream_wide_syncs, 0);
        assert!(sources.iter().all(|source| !source.is_uniquely_owned()));
        drop(arena);
        assert_eq!(ops.allocator_metrics().live_requested_bytes, live_bytes);
        // A later out-of-band fence cannot retroactively change the adapter's
        // Unknown return or recover the intentionally retained credentials.
        ops.sync_upload_stream().unwrap();
        assert_eq!(evidence, QuiescenceEvidence::Unknown);
        assert!(sources.iter().all(|source| !source.is_uniquely_owned()));
        assert_eq!(ops.allocator_metrics().live_requested_bytes, live_bytes);
        eprintln!("real H2D=6/21760B; cleanup fence withheld; sources+frame retained; {message}");
    }

    #[test]
    #[ignore = "requires a real CUDA device and working FP4 private-layout preparation"]
    fn gpu_storage_payload_upload_and_slot_install_adapter_roundtrip() {
        use super::super::{CudaExpertFrame, install, upload};
        use ferrule_common::{
            CompletionHub, ExpertKey, ExpertSlotBinding, ExpertSlotGeneration, ExpertSlotId,
        };

        let ops = CudaOperators::new_on_device(0).expect("CUDA adapter smoke requires device 0");
        let (grouped_fp4, prepared_scale_bytes) = fp4_layout_capability();
        assert!(
            grouped_fp4 && prepared_scale_bytes > 0,
            "full FP4 upload success requires the compiled SM103 grouped-FP4 layout provider;              this GPU/build cannot close that success gate. Run the explicit unsupported              contract test instead; do not count it as a successful upload"
        );
        let expert = ExpertId::new(0, 0);
        let bundle = PinnedExpertBundle::from_payload(pinned_payload(&ops, expert)).unwrap();
        let mut arena = ops
            .allocate_routed_expert_arena(bundle.shape().unwrap(), 1)
            .unwrap();
        let frame = CudaExpertFrame {
            expert: arena.allocate_frame().unwrap(),
        };
        let hub = CompletionHub::new();
        let (ticket, notify_error) = upload::submit(&ops, &hub, bundle, frame)
            .unwrap_or_else(|(error, evidence)| panic!("upload failed: {error}; {evidence:?}"));
        assert!(notify_error.is_none());
        let frame = ticket.drain_into_frame().unwrap();
        let pointers = frame.expert_slot_pointers().unwrap();
        let consumer = ops.compute_stream_authority();
        let mut table = ops.expert_slot_table(2, 1).unwrap();
        let binding = ExpertSlotBinding {
            key: ExpertKey::new(1, 0, 0),
            slot: ExpertSlotId::new(0),
            generation: ExpertSlotGeneration::new(1),
        };
        let target = install::target(&consumer, None).unwrap();
        let ticket = install::submit(&ops, &mut table, target, 0, binding, pointers).unwrap();
        install::notify(&ops, &hub);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !ticket.is_complete().unwrap() {
            assert!(
                std::time::Instant::now() < deadline,
                "install completion timeout"
            );
            std::thread::yield_now();
        }
        assert!(
            table.host().binding(0).is_none(),
            "event completion is not host publication"
        );
        let first = ticket.complete(&mut table).unwrap();
        assert_eq!(table.host().binding(0), Some(first));
        let target = install::target(&consumer, Some((expert, binding))).unwrap();
        let next = ExpertSlotBinding {
            key: ExpertKey::new(1, 0, 1),
            generation: ExpertSlotGeneration::new(2),
            ..binding
        };
        let ticket = install::submit(&ops, &mut table, target, 1, next, pointers).unwrap();
        let second = ticket.complete(&mut table).unwrap();
        assert!(table.host().binding(0).is_none());
        assert_eq!(table.host().binding(1), Some(second));
        assert_eq!(
            (second.slot, second.generation),
            (first.slot, first.generation + 1)
        );
        // The installed frame remains alive until both publication events have completed.
        drop(table);
        drop(frame);
    }
}
