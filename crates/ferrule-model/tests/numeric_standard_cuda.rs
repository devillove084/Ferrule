//! Compressed numeric storage through the existing StandardDecoderOperators seam.
//! The independent oracle also contains full hybrid logits for the runner tests.
use ferrule_model::TensorRole;
use ferrule_model::checkpoint::{CheckpointDType, CheckpointTensorSlice};
use ferrule_model::nn::{
    ModulePath, ParameterDType, ParameterId, ParameterResidency, ParameterSpec,
};
use ferrule_model::transformer::*;
use serde_json::Value;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

fn floats(value: &Value) -> Vec<f32> {
    serde_json::from_value(value.clone()).unwrap()
}
pub(crate) fn role(name: &str) -> TensorRole {
    use TensorRole::*;
    if name == "token_embedding.weight" {
        return TokenEmbedding;
    }
    if name == "final_norm.weight" {
        return OutputNorm;
    }
    if name == "output.weight" {
        return OutputHead;
    }
    let suffix = name.split('.').skip(2).collect::<Vec<_>>().join(".");
    if suffix.starts_with("feed_forward.experts.") {
        return match suffix.split('.').nth(3).unwrap() {
            "gate" => RoutedExpertGate,
            "up" => RoutedExpertUp,
            "down" => RoutedExpertDown,
            _ => panic!("unknown expert role"),
        };
    }
    match suffix.as_str() {
        "input_norm.weight" => AttentionNorm,
        "post_attention_norm.weight" => FeedForwardNorm,
        "attention.query.weight" => AttentionQuery,
        "attention.key.weight" => AttentionKey,
        "attention.value.weight" => AttentionValue,
        "attention.output.weight" => AttentionOutput,
        "attention.query_norm.weight" => AttentionQueryNorm,
        "attention.key_norm.weight" => AttentionKeyNorm,
        "attention.qkv.weight" => LinearAttentionQkv,
        "attention.z.weight" => LinearAttentionZ,
        "attention.beta.weight" => LinearAttentionBeta,
        "attention.a.weight" => LinearAttentionA,
        "attention.conv.weight" => LinearAttentionConv,
        "attention.a_log.weight" => LinearAttentionALog,
        "attention.dt_bias.weight" => LinearAttentionDtBias,
        "attention.norm.weight" => LinearAttentionNorm,
        "feed_forward.router.weight" => RouterLogits,
        "feed_forward.shared.gate.weight" => SharedExpertGate,
        "feed_forward.shared.up.weight" => SharedExpertUp,
        "feed_forward.shared.down.weight" => SharedExpertDown,
        "feed_forward.shared_gate.weight" => SharedExpertOutputGate,
        _ => panic!("unexpected fixture parameter {name}"),
    }
}
pub(crate) fn bind(
    pairs: &[(
        CheckpointTensorSlice,
        Option<CheckpointTensorSlice>,
        ParameterResidency,
    )],
) -> BoundStateDict {
    let mut schema = StateDictSchema::builder();
    let mut mapper = ExactNameMapper::new();
    let mut slices = Vec::new();
    for (i, (weight, scale, residency)) in pairs.iter().enumerate() {
        let path = ModulePath::new(&weight.name).unwrap();
        let dtype = if scale.is_some() {
            ParameterDType::F8E4M3
        } else if weight.dtype == CheckpointDType::Bf16 {
            ParameterDType::Bf16
        } else {
            ParameterDType::F32
        };
        let mut spec = ParameterSpec::new(
            ParameterId::new(i as u64 + 1),
            path.clone(),
            dtype,
            weight.shape.clone(),
            residency.clone(),
        )
        .unwrap();
        if let Some(scale) = scale {
            spec = spec
                .with_required_scale(ParameterDType::Bf16, scale.shape.clone())
                .unwrap();
            mapper
                .insert(&scale.name, NameMapping::scale(path.clone()))
                .unwrap();
            slices.push(scale.clone());
        }
        schema
            .register_with_role(spec, weight.role.clone())
            .unwrap();
        mapper
            .insert(&weight.name, NameMapping::weight(path))
            .unwrap();
        slices.push(weight.clone());
    }
    StateDictBinder::new(&schema.build().unwrap(), &mapper)
        .bind_slices(slices)
        .unwrap()
}
pub(crate) struct Fixture {
    pub(crate) dir: PathBuf,
    pub(crate) oracle: Value,
    pub(crate) bound: BoundStateDict,
    materializer: StateDictMaterializer,
}
impl Fixture {
    pub(crate) fn new() -> Self {
        Self::with_precision(NumericFp8Precision::Bf16RneF32Accumulate)
    }
    pub(crate) fn with_precision(precision: NumericFp8Precision) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "ferrule-numeric-standard-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&dir).unwrap();
        let oracle: Value = serde_json::from_str(match precision {
            NumericFp8Precision::Bf16RneF32Accumulate => {
                include_str!("fixtures/numeric_standard_oracle.json")
            }
            NumericFp8Precision::F32Tf32x3 => {
                include_str!("fixtures/numeric_standard_f32_oracle.json")
            }
        })
        .unwrap();
        let mut payload = Vec::new();
        let mut pairs = Vec::new();
        for (name, tensor) in oracle["tensors"].as_object().unwrap() {
            let shape: Vec<usize> = serde_json::from_value(tensor["shape"].clone()).unwrap();
            let numeric = tensor["dtype"] == "F8_E4M3";
            let offset = payload.len();
            if numeric {
                payload.extend(serde_json::from_value::<Vec<u8>>(tensor["raw"].clone()).unwrap());
            } else {
                payload.extend(
                    floats(&tensor["values"])
                        .iter()
                        .flat_map(|v| v.to_le_bytes()),
                );
            }
            let weight = CheckpointTensorSlice {
                name: name.clone(),
                role: role(name),
                path: dir.join("weights.bin"),
                offset: offset as u64,
                bytes: (payload.len() - offset) as u64,
                dtype: if numeric {
                    CheckpointDType::F8E4M3
                } else {
                    CheckpointDType::F32
                },
                shape: shape.clone(),
            };
            let scale = if numeric {
                let offset = payload.len();
                payload.extend(
                    serde_json::from_value::<Vec<u16>>(tensor["scales"].clone())
                        .unwrap()
                        .iter()
                        .flat_map(|v| v.to_le_bytes()),
                );
                Some(CheckpointTensorSlice {
                    name: name.replace(".weight", ".weight_scale_inv"),
                    role: weight.role.clone(),
                    path: weight.path.clone(),
                    offset: offset as u64,
                    bytes: (payload.len() - offset) as u64,
                    dtype: CheckpointDType::Bf16,
                    shape: shape.iter().map(|n| n.div_ceil(128)).collect(),
                })
            } else {
                None
            };
            let parts = name.split('.').collect::<Vec<_>>();
            let residency = if name.contains(".experts.") {
                ParameterResidency::expert(parts[1].parse().unwrap(), parts[4].parse().unwrap())
            } else if name.starts_with("layers.") {
                ParameterResidency::layer(parts[1].parse().unwrap())
            } else {
                ParameterResidency::Static
            };
            pairs.push((weight, scale, residency));
        }
        std::fs::write(dir.join("weights.bin"), payload).unwrap();
        Self {
            dir,
            oracle,
            bound: bind(&pairs),
            materializer: StateDictMaterializer::new(1 << 20).unwrap(),
        }
    }
    fn metadata(&self, layer: usize, expert: usize) -> ferrule_common::Result<ExpertMetadata> {
        let prefix = format!("layers.{layer}.feed_forward.experts.{expert}");
        let parameters = ["gate", "up", "down"].map(|part| {
            self.bound
                .get(&ModulePath::new(format!("{prefix}.{part}.weight")).unwrap())
                .unwrap()
                .clone()
        });
        ExpertMetadata::new(layer, expert, parameters, None)
    }
    fn linear(&self, path: &str) -> PreparedLinear {
        let parameter = self.bound.get(&ModulePath::new(path).unwrap()).unwrap();
        self.materializer
            .prepared_linear(parameter, parameter.role().clone())
            .unwrap()
    }
    #[cfg(feature = "cuda")]
    fn swiglu(&self, prefix: &str) -> PreparedSwiGlu {
        PreparedSwiGlu::new(
            self.linear(&format!("{prefix}.gate.weight")),
            self.linear(&format!("{prefix}.up.weight")),
            self.linear(&format!("{prefix}.down.weight")),
            None,
        )
        .unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

#[test]
fn expert_metadata_is_payload_free_and_matches_exact_paired_source() {
    let f = Fixture::new();
    let plan = f.metadata(0, 0).unwrap();
    assert_eq!(plan.expected_device_bytes(), 294);
    assert_eq!(plan.input_width(), 8);
    assert_eq!(plan.output_width(), 8);
    let make = |fixture: &Fixture, id: usize| {
        let prefix = format!("layers.0.feed_forward.experts.{id}");
        PreparedSwiGlu::new(
            fixture.linear(&format!("{prefix}.gate.weight")),
            fixture.linear(&format!("{prefix}.up.weight")),
            fixture.linear(&format!("{prefix}.down.weight")),
            None,
        )
        .unwrap()
    };
    plan.validate_payload(&make(&f, 0)).unwrap();
    assert!(plan.validate_payload(&make(&f, 1)).is_err());
    let other = Fixture::new();
    assert!(plan.validate_payload(&make(&other, 0)).is_err());
    let [gate, up, down] = plan.parameters().clone();
    assert!(ExpertMetadata::new(0, 0, [up, gate, down], None).is_err());
    assert!(ExpertMetadata::new(1, 0, plan.parameters().clone(), None).is_err());
    assert!(ExpertMetadata::new(0, 0, plan.parameters().clone(), Some(f32::NAN)).is_err());
    // Invalid FP8 payload contents are deliberately not read/decoded by metadata.
    // A replaced file, even with identical bytes, invalidates its source snapshot.
    std::fs::rename(f.dir.join("weights.bin"), f.dir.join("old.bin")).unwrap();
    std::fs::copy(f.dir.join("old.bin"), f.dir.join("weights.bin")).unwrap();
    assert!(plan.validate_sources().is_err());
    assert!(f.metadata(0, 0).is_err());
}

#[test]
fn tiny_numeric_f32_oracle_has_identical_storage_but_distinct_arithmetic() {
    let bf16 = Fixture::new();
    let f32 = Fixture::with_precision(NumericFp8Precision::F32Tf32x3);
    assert_eq!(f32.oracle["precision_profile"], "f32");
    assert_eq!(bf16.oracle["tensors"], f32.oracle["tensors"]);
    assert_eq!(bf16.oracle["tokens"], f32.oracle["tokens"]);
    assert_ne!(bf16.oracle["logits"], f32.oracle["logits"]);
    let linear = f32.linear("layers.0.feed_forward.experts.0.gate.weight");
    assert_eq!(linear.numeric_fp8().unwrap().storage_bytes(), 98);
    assert!(linear.parameter().values_f32().is_err());
}

#[test]
fn tiny_numeric_metadata_retains_compressed_geometry_without_host_expert_cache() {
    let f = Fixture::new();
    assert_eq!(f.oracle["tokens"].as_array().unwrap().len(), 12);
    assert_eq!(f.oracle["ffn_cases"].as_array().unwrap().len(), 8);
    assert_eq!(f.bound.parameters().len(), 50);
    let parameter = f.linear("layers.0.feed_forward.experts.0.gate.weight");
    assert_eq!(parameter.global_shape(), [12, 8]);
    assert_eq!(parameter.numeric_fp8().unwrap().storage_bytes(), 98);
    assert!(parameter.parameter().values_f32().is_err());
    let PreparedLinearStorage::NumericFp8(artifact) = parameter.storage() else {
        panic!()
    };
    let weak = Arc::downgrade(artifact);
    drop(parameter);
    assert!(weak.upgrade().is_none());
    let reloaded = f.linear("layers.0.feed_forward.experts.0.gate.weight");
    assert_eq!(reloaded.numeric_fp8().unwrap().storage_bytes(), 98);
}

#[cfg(feature = "cuda")]
mod actual {
    use super::*;
    use ferrule_backend::cuda::operators::linear::CudaOperators;
    use ferrule_model::checkpoint::NumericFp8Artifact;
    use ferrule_model::execution::ExecutionPrecisionPolicy;
    use std::rc::Rc;
    use std::sync::Weak;
    const SCRATCH: usize = 4096;
    const BUDGET: usize = 8192;
    fn policy() -> ExpertCachePolicy {
        ExpertCachePolicy::Bounded(ExpertCacheLimits {
            max_experts: 1,
            max_bytes: BUDGET,
        })
    }
    fn owner(f: &Fixture) -> CudaStandardDecoderOperators {
        CudaStandardDecoderOperators::new_numeric_fp8(
            Rc::new(CudaOperators::new_on_device(0).unwrap()),
            f.bound.parameters(),
            policy(),
            SCRATCH,
        )
        .unwrap()
    }
    fn ready<T>(value: OperatorProgress<T>) -> T {
        match value {
            OperatorProgress::Ready(v) => v,
            _ => panic!("expected Ready"),
        }
    }
    fn rows(values: Vec<f32>, width: usize) -> Rows {
        Rows::Host(
            HostRows::new(
                RowsShape::new(values.len() / width, width).unwrap(),
                RowsDType::F32,
                None,
                values,
            )
            .unwrap(),
        )
    }
    fn close(actual: &[f32], expected: &[f32], label: &str) {
        assert_eq!(actual.len(), expected.len());
        let mut max_error = 0.0f64;
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            let error = (f64::from(a) - f64::from(e)).abs();
            assert!(
                a.is_finite() && e.is_finite() && error <= 2e-5 + 3e-4 * f64::from(e).abs(),
                "{label}[{i}]: {a} != {e}"
            );
            max_error = max_error.max(error);
        }
        eprintln!("{label}: {} values, max_abs={max_error:.8e}", actual.len());
    }
    struct Provider<'a> {
        fixture: &'a Fixture,
        weak: Vec<Weak<NumericFp8Artifact>>,
        calls: usize,
    }
    impl ExpertProvider for Provider<'_> {
        fn expert_metadata(
            &mut self,
            layer: usize,
            expert: usize,
        ) -> ferrule_common::Result<Option<ExpertMetadata>> {
            self.fixture.metadata(layer, expert).map(Some)
        }
        fn expert(
            &mut self,
            layer: usize,
            expert: usize,
        ) -> ferrule_common::Result<ExpertAvailability> {
            self.calls += 1;
            let value = self
                .fixture
                .swiglu(&format!("layers.{layer}.feed_forward.experts.{expert}"));
            for projection in [value.gate(), value.up(), value.down()] {
                let PreparedLinearStorage::NumericFp8(artifact) = projection.storage() else {
                    panic!()
                };
                self.weak.push(Arc::downgrade(artifact));
            }
            Ok(ExpertAvailability::Ready(Arc::new(value)))
        }
    }
    #[test]
    #[ignore = "requires CUDA GPU; independent Torch BF16-RNE oracle"]
    fn numeric_two_layer_moe_top2_cap1_reloads_and_shared_gate_matches_torch() {
        let f = Fixture::new();
        let mut gpu = owner(&f);
        assert_eq!(
            gpu.resident_parameter_bytes(),
            0,
            "constructor must not upload any expert"
        );
        let router_policy = MoeRouterSpec::new(
            3,
            2,
            RouterScoreFunction::Softmax,
            RouterSelection::TopK,
            true,
            1.0,
        )
        .unwrap();
        let mut provider = Provider {
            fixture: &f,
            weak: Vec::new(),
            calls: 0,
        };
        for (step, case) in f.oracle["ffn_cases"].as_array().unwrap().iter().enumerate() {
            let mut input = gpu.bind_rows(rows(floats(&case["input"]), 8)).unwrap();
            for layer in 0..2 {
                let prefix = format!("layers.{layer}.feed_forward");
                let logits = ready(
                    gpu.linear(&f.linear(&format!("{prefix}.router.weight")), &input, None)
                        .unwrap(),
                );
                let routes = ready(gpu.router(&logits, &router_policy).unwrap());
                let routed = ready(
                    gpu.routed_swiglu(layer, &input, &routes, &mut provider, None)
                        .unwrap(),
                );
                let shared = ready(
                    gpu.dense_swiglu(&f.swiglu(&format!("{prefix}.shared")), &input, None)
                        .unwrap(),
                );
                let gate = ready(
                    gpu.linear(
                        &f.linear(&format!("{prefix}.shared_gate.weight")),
                        &input,
                        None,
                    )
                    .unwrap(),
                );
                let shared = ready(gpu.shared_expert_gate(shared, &gate).unwrap());
                let result = ready(gpu.residual(routed, &shared).unwrap());
                assert_eq!(result.device(), RowsDevice::Cuda { ordinal: 0 });

                let expected = floats(&case["layers"][layer]);
                input = ready(gpu.residual(input, &result).unwrap());
                close(
                    gpu.operators()
                        .download_f32_buffer(result.cuda().unwrap().f32_buffer().unwrap())
                        .unwrap()
                        .as_slice(),
                    &expected,
                    &format!("step {step} layer {layer} FFN"),
                );

                let stats = gpu.expert_cache_stats().unwrap();
                assert_eq!(stats.resident_experts, 1);
                assert_eq!(stats.resident_bytes, 3 * (8 * 12 + 2));
                assert_eq!(stats.pending_upload_bytes, 0);
                assert_eq!(stats.scratch_bytes, SCRATCH);
                assert!(stats.peak_bytes <= BUDGET);
                assert!(!stats.quarantined);
                assert!(
                    provider.weak.iter().all(|w| w.upgrade().is_none()),
                    "neither owner nor materializer may retain host experts"
                );
            }
        }
        let stats = gpu.expert_cache_stats().unwrap();
        assert!(stats.uploads >= 32 && stats.evictions >= 31);
        assert_eq!(
            provider.calls as u64, stats.uploads,
            "one payload materialization per miss, none during preflight"
        );
        let workspace = gpu.numeric_workspace_stats().unwrap();
        assert!(workspace.allocated_bytes <= SCRATCH);
        assert!(workspace.reuses > workspace.allocations);
        assert!(workspace.submissions > 100);
        eprintln!("cap=1 residency: {stats:?}; numeric workspace: {workspace:?}");
        gpu.quiesce().unwrap();
        assert_eq!(gpu.expert_cache_stats().unwrap(), stats);
    }

    #[test]
    #[ignore = "requires CUDA GPU; metadata miss/hit counters in both numeric precisions"]
    fn metadata_payload_calls_cap1_reload_and_large_cache_hits() {
        for precision in [
            NumericFp8Precision::Bf16RneF32Accumulate,
            NumericFp8Precision::F32Tf32x3,
        ] {
            let f = Fixture::with_precision(precision);
            for capacity in [1, 64, 1024] {
                let ops = Rc::new(CudaOperators::new_on_device(0).unwrap());
                let mut gpu = CudaStandardDecoderOperators::new_numeric_fp8_with_precision(
                    Rc::clone(&ops),
                    f.bound.parameters(),
                    ExpertCachePolicy::Bounded(ExpertCacheLimits {
                        max_experts: capacity,
                        max_bytes: BUDGET,
                    }),
                    SCRATCH,
                    precision,
                )
                .unwrap();
                let input = gpu.bind_rows(rows(vec![0.1; 8], 8)).unwrap();
                let mut provider = Provider {
                    fixture: &f,
                    weak: Vec::new(),
                    calls: 0,
                };
                let one = RouterRoutes::new(1, 1, vec![0], vec![1.0]).unwrap();
                for expected_calls in [1, 1] {
                    let output = ready(
                        gpu.routed_swiglu(0, &input, &one, &mut provider, None)
                            .unwrap(),
                    );
                    assert!(
                        gpu.download_rows(output)
                            .unwrap()
                            .values()
                            .iter()
                            .all(|v| v.is_finite())
                    );
                    assert_eq!(provider.calls, expected_calls);
                    assert!(provider.weak.iter().all(|w| w.upgrade().is_none()));
                }
                let top2 = RouterRoutes::new(1, 2, vec![0, 1], vec![0.4, 0.6]).unwrap();
                let first = ready(
                    gpu.routed_swiglu(0, &input, &top2, &mut provider, None)
                        .unwrap(),
                );
                let first = gpu.download_rows(first).unwrap();
                assert_eq!(provider.calls, 2);
                let second = ready(
                    gpu.routed_swiglu(0, &input, &top2, &mut provider, None)
                        .unwrap(),
                );
                let second = gpu.download_rows(second).unwrap();
                assert_eq!(
                    first.values(),
                    second.values(),
                    "reload must not change arithmetic"
                );
                assert_eq!(provider.calls, if capacity == 1 { 4 } else { 2 });
                let stats = gpu.expert_cache_stats().unwrap();
                assert_eq!(provider.calls as u64, stats.uploads);
                assert_eq!(stats.resident_experts, capacity.min(2));
                assert!(stats.peak_bytes <= BUDGET);
                assert!(provider.weak.iter().all(|w| w.upgrade().is_none()));
                gpu.quiesce().unwrap();
                drop(input);
                drop(gpu);
                ops.trim_device_allocator().unwrap();
                assert_eq!(ops.allocator_metrics().live_requested_bytes, 0);
            }
        }
    }

    #[test]
    #[ignore = "requires CUDA GPU; borrowed image plans avoid metadata reconstruction and payload hits"]
    fn borrowed_metadata_reuses_image_and_still_rejects_stale_resident_source() {
        struct Borrowed<'a> {
            inner: Provider<'a>,
            metadata: ExpertMetadata,
        }
        impl ExpertProvider for Borrowed<'_> {
            fn expert_metadata_bindings(
                &self,
                _: usize,
                _: usize,
            ) -> ferrule_common::Result<Option<ExpertMetadataBindings<'_>>> {
                Ok(Some(ExpertMetadataBindings {
                    parameters: self.metadata.parameters(),
                    activation_limit: self.metadata.activation_limit(),
                }))
            }
            fn expert_metadata(
                &mut self,
                _: usize,
                _: usize,
            ) -> ferrule_common::Result<Option<ExpertMetadata>> {
                panic!("image-backed provider must not reconstruct metadata")
            }
            fn expert(
                &mut self,
                layer: usize,
                expert: usize,
            ) -> ferrule_common::Result<ExpertAvailability> {
                self.inner.expert(layer, expert)
            }
        }
        let f = Fixture::new();
        let mut gpu = owner(&f);
        let mut provider = Borrowed {
            inner: Provider {
                fixture: &f,
                weak: Vec::new(),
                calls: 0,
            },
            metadata: f.metadata(0, 0).unwrap(),
        };
        let input = gpu.bind_rows(rows(vec![0.1; 8], 8)).unwrap();
        let routes = RouterRoutes::new(1, 1, vec![0], vec![1.0]).unwrap();
        let metadata_stats = gpu.expert_metadata_preflight_stats();
        assert_eq!(metadata_stats.prepared_experts, 6);
        assert_eq!(metadata_stats.source_snapshots, 1);
        for forward in 1..=4 {
            gpu.preflight_expert_metadata().unwrap();
            drop(ready(
                gpu.routed_swiglu(0, &input, &routes, &mut provider, None)
                    .unwrap(),
            ));
            assert_eq!(provider.inner.calls, 1);
            assert!(provider.inner.weak.iter().all(|w| w.upgrade().is_none()));
            let stats = gpu.expert_metadata_preflight_stats();
            assert_eq!(stats.preflights, forward);
            assert_eq!(stats.source_checks, forward);
            assert_eq!(stats.prepared_experts, 6);
        }
        let cache = gpu.expert_cache_stats().unwrap();
        let alien = Fixture::new();
        provider.metadata = alien.metadata(0, 0).unwrap();
        assert!(
            gpu.routed_swiglu(0, &input, &routes, &mut provider, None)
                .is_err()
        );
        assert_eq!(gpu.expert_cache_stats().unwrap(), cache);
        provider.metadata = f.metadata(0, 0).unwrap();
        std::fs::rename(f.dir.join("weights.bin"), f.dir.join("old.bin")).unwrap();
        std::fs::copy(f.dir.join("old.bin"), f.dir.join("weights.bin")).unwrap();
        assert!(gpu.preflight_expert_metadata().is_err());
        assert!(
            gpu.routed_swiglu(0, &input, &routes, &mut provider, None)
                .is_err()
        );
        assert_eq!(gpu.expert_cache_stats().unwrap(), cache);
        assert_eq!(provider.inner.calls, 1);
    }

    #[test]
    #[ignore = "requires CUDA GPU; bad unselected expert fails image construction"]
    fn metadata_invalid_unrouted_expert_rejected_during_image_prepare() {
        let f = Fixture::new();
        let ops = Rc::new(CudaOperators::new_on_device(0).unwrap());
        let mut parameters = f.bound.parameters().to_vec();
        let index = parameters
            .iter()
            .position(|p| {
                p.residency() == &ParameterResidency::expert(1, 2)
                    && p.role() == &TensorRole::RoutedExpertDown
            })
            .unwrap();
        parameters.remove(index);
        let before = ops.allocator_metrics().live_requested_bytes;
        assert!(
            CudaStandardDecoderOperators::new_numeric_fp8(
                Rc::clone(&ops),
                &parameters,
                policy(),
                SCRATCH
            )
            .is_err()
        );
        assert_eq!(ops.allocator_metrics().live_requested_bytes, before);
    }

    #[test]
    #[ignore = "requires CUDA GPU; metadata validation precedes any expert mutation"]
    fn metadata_invalid_selected_expert_and_stale_hit_do_not_materialize() {
        let f = Fixture::new();
        let alien = Fixture::new();
        let mut gpu = owner(&f);
        let input = gpu.bind_rows(rows(vec![0.1; 8], 8)).unwrap();
        struct Invalid<'a> {
            local: &'a Fixture,
            alien: &'a Fixture,
            calls: usize,
        }
        impl ExpertProvider for Invalid<'_> {
            fn expert_metadata(
                &mut self,
                layer: usize,
                expert: usize,
            ) -> ferrule_common::Result<Option<ExpertMetadata>> {
                if expert == 0 {
                    self.local.metadata(layer, expert).map(Some)
                } else {
                    self.alien.metadata(layer, expert).map(Some)
                }
            }
            fn expert(&mut self, _: usize, _: usize) -> ferrule_common::Result<ExpertAvailability> {
                self.calls += 1;
                panic!("invalid metadata must reject before materialization")
            }
        }
        let mut invalid = Invalid {
            local: &f,
            alien: &alien,
            calls: 0,
        };
        let top2 = RouterRoutes::new(1, 2, vec![0, 1], vec![0.4, 0.6]).unwrap();
        let before = gpu.expert_cache_stats().unwrap();
        let bad_width = gpu.bind_rows(rows(vec![0.1; 7], 7)).unwrap();
        assert!(
            gpu.routed_swiglu(0, &bad_width, &top2, &mut invalid, None)
                .is_err()
        );
        let bad_rows = RouterRoutes::new(2, 1, vec![0, 0], vec![1.0; 2]).unwrap();
        assert!(
            gpu.routed_swiglu(0, &input, &bad_rows, &mut invalid, None)
                .is_err()
        );
        let mut other_owner = owner(&f);
        let other_input = other_owner.bind_rows(rows(vec![0.1; 8], 8)).unwrap();
        assert!(
            gpu.routed_swiglu(0, &other_input, &top2, &mut invalid, None)
                .is_err()
        );
        assert!(
            gpu.routed_swiglu(0, &input, &top2, &mut invalid, None)
                .is_err()
        );
        assert_eq!(invalid.calls, 0);
        assert_eq!(gpu.expert_cache_stats().unwrap(), before);
        assert_eq!(gpu.resident_parameter_bytes(), 0);
        let one = RouterRoutes::new(1, 1, vec![0], vec![1.0]).unwrap();
        let mut provider = Provider {
            fixture: &f,
            weak: Vec::new(),
            calls: 0,
        };
        drop(ready(
            gpu.routed_swiglu(0, &input, &one, &mut provider, None)
                .unwrap(),
        ));
        let before = gpu.expert_cache_stats().unwrap();
        let bytes = gpu.resident_parameter_bytes();
        std::fs::rename(f.dir.join("weights.bin"), f.dir.join("old.bin")).unwrap();
        std::fs::copy(f.dir.join("old.bin"), f.dir.join("weights.bin")).unwrap();
        assert!(gpu.preflight_expert_metadata().is_err());
        assert!(
            gpu.routed_swiglu(0, &input, &one, &mut provider, None)
                .is_err()
        );
        assert_eq!(provider.calls, 1);
        assert_eq!(gpu.expert_cache_stats().unwrap(), before);
        assert_eq!(gpu.resident_parameter_bytes(), bytes);
    }

    #[test]
    #[ignore = "requires CUDA GPU; byte limit independently forces compressed expert eviction"]
    fn numeric_byte_limit_includes_scales_and_reserved_workspace() {
        let f = Fixture::new();
        let ops = Rc::new(CudaOperators::new_on_device(0).unwrap());
        let bytes = SCRATCH + 294;
        let mut gpu = CudaStandardDecoderOperators::new_numeric_fp8(
            Rc::clone(&ops),
            f.bound.parameters(),
            ExpertCachePolicy::Bounded(ExpertCacheLimits {
                max_experts: 64,
                max_bytes: bytes,
            }),
            SCRATCH,
        )
        .unwrap();
        for id in [0, 1, 0, 2, 1] {
            gpu.prepare_expert(&f.swiglu(&format!("layers.0.feed_forward.experts.{id}")))
                .unwrap();
            let stats = gpu.expert_cache_stats().unwrap();
            assert_eq!(stats.resident_experts, 1);
            assert_eq!(stats.resident_bytes, 294);
            assert_eq!(stats.charged_bytes(), bytes);
            assert_eq!(stats.peak_bytes, bytes);
        }
        assert_eq!(gpu.expert_cache_stats().unwrap().uploads, 5);
        assert_eq!(gpu.expert_cache_stats().unwrap().evictions, 4);
        let input = gpu.bind_rows(rows(vec![0.1; 8], 8)).unwrap();
        assert!(
            gpu.dense_swiglu(&f.swiglu("layers.0.feed_forward.experts.0"), &input, None)
                .is_err(),
            "weights fit but activation/result scratch must also be charged"
        );
        assert!(!gpu.needs_quarantine());
        let mut too_small = CudaStandardDecoderOperators::new_numeric_fp8(
            ops,
            f.bound.parameters(),
            ExpertCachePolicy::Bounded(ExpertCacheLimits {
                max_experts: 64,
                max_bytes: bytes - 1,
            }),
            SCRATCH,
        )
        .unwrap();
        assert!(
            too_small
                .prepare_expert(&f.swiglu("layers.0.feed_forward.experts.0"))
                .is_err()
        );
        assert_eq!(too_small.resident_parameter_bytes(), 0);
        assert_eq!(
            too_small.expert_cache_stats().unwrap().charged_bytes(),
            SCRATCH
        );
    }

    #[test]
    #[ignore = "requires CUDA GPU; all numeric non-expert projection roles"]
    fn numeric_attention_router_shared_and_head_use_the_same_linear_binding() {
        let f = Fixture::new();
        let mut gpu = owner(&f);
        for (name, case) in f.oracle["projection_cases"].as_object().unwrap() {
            let projection = f.linear(name);
            let input = gpu
                .bind_rows(rows(floats(&case["input"]), projection.in_features()))
                .unwrap();
            let result = ready(gpu.linear(&projection, &input, None).unwrap());
            close(
                gpu.download_rows(result).unwrap().values(),
                &floats(&case["expected"]),
                name,
            );
            let stats = gpu.expert_cache_stats().unwrap();
            assert_eq!(stats.resident_experts, 0);
            assert_eq!(stats.pending_upload_bytes, 0);
            assert_eq!(stats.scratch_bytes, SCRATCH);
        }
        assert!(gpu.numeric_workspace_stats().unwrap().submissions >= 20);
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn numeric_default_rejects_and_identity_budget_clean_errors_fail_closed() {
        identity_budget_clean_errors(NumericFp8Precision::Bf16RneF32Accumulate);
    }

    #[test]
    #[ignore = "requires CUDA GPU; explicit F32 stale-source/allocation/budget errors"]
    fn numeric_f32_identity_budget_clean_errors_fail_closed() {
        identity_budget_clean_errors(NumericFp8Precision::F32Tf32x3);
    }

    fn identity_budget_clean_errors(precision: NumericFp8Precision) {
        let f = Fixture::with_precision(precision);
        let mut gpu = CudaStandardDecoderOperators::new_numeric_fp8_with_precision(
            Rc::new(CudaOperators::new_on_device(0).unwrap()),
            f.bound.parameters(),
            policy(),
            SCRATCH,
            precision,
        )
        .unwrap();
        let expert = f.swiglu("layers.0.feed_forward.experts.0");
        let input = gpu.bind_rows(rows(vec![0.1; 16], 8)).unwrap();
        let mut legacy = CudaStandardDecoderOperators::new(
            Rc::clone(gpu.operators()),
            ExecutionPrecisionPolicy::f32(),
            f.bound.parameters(),
        )
        .unwrap();
        assert!(legacy.prepare_expert(&expert).is_err());
        assert_eq!(legacy.resident_parameter_bytes(), 0);
        gpu.prepare_expert(&expert).unwrap();
        assert_eq!(gpu.expert_cache_stats().unwrap().resident_bytes, 294);
        assert!(
            gpu.linear(expert.gate(), &input, None).is_err(),
            "cannot bypass expert lease"
        );
        let alien = Fixture::new();
        assert!(
            gpu.prepare_expert(&alien.swiglu("layers.0.feed_forward.experts.0"))
                .is_err()
        );
        let wrong = PreparedSwiGlu::new(
            expert.up().clone(),
            expert.gate().clone(),
            expert.down().clone(),
            None,
        )
        .unwrap();
        assert!(gpu.prepare_expert(&wrong).is_err());
        let shared_gate = f.linear("layers.0.feed_forward.shared_gate.weight");
        let wrong = shared_gate
            .parameter()
            .into_linear(TensorRole::RouterLogits)
            .unwrap();
        assert!(gpu.linear(&wrong, &input, None).is_err());
        let bad_shape = gpu.bind_rows(rows(vec![0.1; 14], 7)).unwrap();
        assert!(gpu.linear(&shared_gate, &bad_shape, None).is_err());
        let next = f.swiglu("layers.1.feed_forward.experts.1");
        gpu.operators().failpoints().arm_allocation();
        assert!(gpu.dense_swiglu(&next, &input, None).is_err());
        assert!(!gpu.needs_quarantine());
        assert_eq!(gpu.expert_cache_stats().unwrap().pending_upload_bytes, 0);
        ready(gpu.dense_swiglu(&next, &input, None).unwrap());
        let mut tiny = CudaStandardDecoderOperators::new_numeric_fp8_with_precision(
            Rc::clone(gpu.operators()),
            f.bound.parameters(),
            policy(),
            1,
            precision,
        )
        .unwrap();
        assert!(tiny.dense_swiglu(&expert, &input, None).is_err());
        assert_eq!(tiny.expert_cache_stats().unwrap().pending_upload_bytes, 0);
        assert_eq!(tiny.expert_cache_stats().unwrap().resident_bytes, 0);
        assert_eq!(tiny.numeric_workspace_stats().unwrap().allocated_bytes, 0);
        assert!(
            CudaStandardDecoderOperators::new_numeric_fp8_with_precision(
                Rc::clone(gpu.operators()),
                &[],
                policy(),
                BUDGET + 1,
                precision,
            )
            .is_err()
        );
        // Invalidate the paired scale source after the expert is resident.
        gpu.prepare_expert(&expert).unwrap();
        use std::io::Write;
        std::fs::OpenOptions::new()
            .append(true)
            .open(f.dir.join("weights.bin"))
            .unwrap()
            .write_all(&[0])
            .unwrap();
        assert!(gpu.prepare_expert(&expert).is_err());
        assert!(gpu.dense_swiglu(&expert, &input, None).is_err());
        assert!(
            !gpu.needs_quarantine(),
            "invalid source is a clean failure, not unknown CUDA completion"
        );
    }

    #[test]
    #[ignore = "requires CUDA, local 35B checkpoint and FERRULE_FP8_PYTHON"]
    fn local_35b_selected_expert_projections_match_independent_torch() {
        let model = std::env::var("FERRULE_NUMERIC_FP8_MODEL_DIR")
            .expect("set FERRULE_NUMERIC_FP8_MODEL_DIR");
        let python = std::env::var("FERRULE_FP8_PYTHON").unwrap_or_else(|_| "python3".into());
        let output = std::process::Command::new(python)
            .arg("-B")
            .arg(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/numeric_standard_oracle.py"
            ))
            .arg("--model")
            .arg(model)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let oracle: Value = serde_json::from_slice(&output.stdout).unwrap();
        for (case, role) in oracle["projections"].as_array().unwrap().iter().zip([
            TensorRole::RoutedExpertGate,
            TensorRole::RoutedExpertUp,
            TensorRole::RoutedExpertDown,
        ]) {
            let tensor = |v: &Value| CheckpointTensorSlice {
                name: v["name"].as_str().unwrap().into(),
                path: v["path"].as_str().unwrap().into(),
                role: role.clone(),
                offset: v["offset"].as_u64().unwrap(),
                bytes: v["bytes"].as_u64().unwrap(),
                shape: serde_json::from_value(v["shape"].clone()).unwrap(),
                dtype: CheckpointDType::from_safetensors_dtype(v["dtype"].as_str().unwrap()),
            };
            let weight = tensor(&case["weight"]);
            let scale = tensor(&case["scale"]);
            let bytes = weight.bytes + scale.bytes;
            assert!(bytes <= 2 * 1024 * 1024);
            // This oracle exercises one standalone projection, not a complete
            // routed image. Routed catalogs now reject incomplete experts at prepare.
            let bound = bind(&[(weight, Some(scale), ParameterResidency::Static)]);
            let parameter = &bound.parameters()[0];
            let materializer = StateDictMaterializer::new(bytes).unwrap();
            let linear = materializer
                .prepared_linear(parameter, role.clone())
                .unwrap();
            // A single projection probe, deliberately not a full MoE/model load.
            let mut gpu = CudaStandardDecoderOperators::new_numeric_fp8(
                Rc::new(CudaOperators::new_on_device(0).unwrap()),
                bound.parameters(),
                ExpertCachePolicy::KeepAll,
                64 * 1024,
            )
            .unwrap();
            let input = gpu
                .bind_rows(rows(floats(&case["input"]), linear.in_features()))
                .unwrap();
            for _ in 0..2 {
                let result = ready(gpu.linear(&linear, &input, None).unwrap());
                close(
                    gpu.download_rows(result).unwrap().values(),
                    &floats(&case["expected"]),
                    &format!("35B layer0 expert0 {role:?}"),
                );
                assert_eq!(gpu.resident_parameter_bytes(), bytes as usize);
            }
            let stats = gpu.numeric_workspace_stats().unwrap();
            assert_eq!(stats.allocations, 1);
            assert_eq!(stats.reuses, 1);
            assert!(stats.allocated_bytes <= 64 * 1024);
        }
    }
}
