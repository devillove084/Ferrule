//! Full generic hybrid runner, compressed MoE checkpoint and independent Torch oracle.
use super::*;
use ferrule_common::execution::KvLayoutSchema;
use ferrule_model::transformer::*;
use std::sync::Arc;
#[path = "external.rs"]
mod external;
#[path = "numeric_f32.rs"]
mod f32_profile;
#[path = "../numeric_standard_cuda.rs"]
mod fixture;

fn spec() -> DecoderModelSpec {
    routed_spec(2, 12)
}
fn routed_spec(top_k: usize, intermediate: usize) -> DecoderModelSpec {
    let norm = |n| RmsNorm::new(n, 1e-6).unwrap().with_one_plus_weight();
    let rope = RotaryEmbedding::new(
        8,
        10000.0,
        RotaryPairing::SplitHalf,
        RotaryRegion::Prefix { dimensions: 4 },
        RotaryScaling::None,
    )
    .unwrap();
    let layers = (0..2)
        .map(|i| {
            let attention = if i == 0 {
                Attention::GatedDeltaNet(
                    GatedDeltaNetAttention::new(8, 1, 2, 3, 2, 3, 1e-6, false).unwrap(),
                )
            } else {
                Attention::Gqa(
                    GqaAttention::new(8, 2, 1, 8, false, rope.clone())
                        .unwrap()
                        .with_gated_query()
                        .unwrap()
                        .with_qk_norms(norm(8), norm(8))
                        .unwrap(),
                )
            };
            let router = MoeRouterSpec::new(
                3,
                top_k,
                RouterScoreFunction::Softmax,
                RouterSelection::TopK,
                true,
                1.0,
            )
            .unwrap();
            let moe = Moe::new(8, intermediate, router, false)
                .unwrap()
                .with_shared_expert(SwiGlu::new(8, 12, false).unwrap())
                .unwrap()
                .with_shared_expert_gate(Linear::new(8, 1, false).unwrap())
                .unwrap();
            DecoderLayer::new(
                i,
                norm(8),
                attention,
                Residual::Add,
                norm(8),
                FeedForward::Moe(moe),
                Residual::Add,
            )
            .unwrap()
        })
        .collect();
    DecoderModelSpec::new(DecoderModelParts {
        architecture: "numeric-hybrid-oracle".into(),
        hidden_size: 8,
        vocab_size: 11,
        max_sequence_length: Some(32),
        token_embedding: Embedding::new(11, 8, None).unwrap(),
        layers,
        final_norm: norm(8),
        output: Linear::new(8, 11, false).unwrap(),
        tie_word_embeddings: false,
    })
    .unwrap()
}
fn resources(f: &fixture::Fixture) -> BoundDecoderResources {
    BoundDecoderResources::new(spec(), Arc::new(f.bound.clone())).unwrap()
}
fn options(resources: &BoundDecoderResources) -> GenericDecoderOptions {
    options_with_rows(resources, 8)
}
fn options_with_rows(resources: &BoundDecoderResources, rows: usize) -> GenericDecoderOptions {
    GenericDecoderOptions::standard_cpu(
        resources.spec(),
        ModelFamily::Unknown("numeric-hybrid-oracle".into()),
        WeightSource::Safetensors,
        2,
        32,
        rows,
        1,
        1 << 20,
        ExecutionPrecisionPolicy::f32(),
    )
    .unwrap()
}
fn numeric(
    options: GenericDecoderOptions,
    max_experts: usize,
    max_bytes: usize,
    scratch: usize,
) -> GenericDecoderOptions {
    options
        .with_hybrid_cuda_numeric_fp8(
            ExpertCacheLimits {
                max_experts,
                max_bytes,
            },
            scratch,
        )
        .unwrap()
}
fn tokenizer(f: &fixture::Fixture) -> TokenizerHandle {
    let mut tokenizer = tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default());
    tokenizer
        .add_tokens((0..11).map(|i| tokenizers::AddedToken::from(format!("t{i}"), false)))
        .unwrap();
    tokenizer.save(f.dir.join("tokenizer.json"), false).unwrap();
    TokenizerHandle::load(&f.dir).unwrap()
}
fn typed(error: ferrule_common::Error) {
    let ferrule_common::Error::ModelSource { source } = error else {
        panic!("expected typed unsupported: {error}")
    };
    assert!(source.downcast_ref::<UnsupportedOperator>().is_some());
}
// Independent byte ledger, deliberately not calling the production helper.
fn bucket_minimum(rows: usize, top_k: usize, intermediate: usize, weights: usize) -> usize {
    let hidden = 8;
    let routes = rows * top_k;
    let route = (2 * rows * hidden + routes * hidden + 2 * routes + 2 * rows) * 4;
    let bucket = rows * (2 * hidden + 3 * intermediate) * 4;
    // Largest non-expert projection is gated Q [32, 8].
    let non_expert = rows * (2 * 32 + hidden + 1) * 4;
    4096 + (weights + route + bucket).max(non_expert)
}

#[test]
fn bucket_admission_exact_cap_and_minus_one_across_rows_topk_storage_and_precision() {
    use ferrule_model::nn::ParameterResidency;
    let f = fixture::Fixture::new();
    for dtype in [
        CheckpointDType::F8E4M3,
        CheckpointDType::F32,
        CheckpointDType::Bf16,
    ] {
        let bound = if dtype == CheckpointDType::F8E4M3 {
            f.bound.clone()
        } else {
            // Metadata-only rebinding: no expert decoding or CUDA owner.
            let path = f.dir.join(format!("admission-{dtype:?}.bin"));
            std::fs::write(&path, vec![0; f.bound.parameters().len() * 12 * 8 * 4]).unwrap();
            let pairs = f
                .bound
                .parameters()
                .iter()
                .enumerate()
                .map(|(index, p)| {
                    let mut weight = p.weight().slice().clone();
                    let mut scale = p.scale().map(|s| s.slice().clone());
                    if matches!(p.residency(), ParameterResidency::Expert { .. }) {
                        weight.path = path.clone();
                        weight.offset = (index * 12 * 8 * 4) as u64;
                        weight.dtype = dtype.clone();
                        weight.bytes =
                            (12 * 8 * if dtype == CheckpointDType::F32 { 4 } else { 2 }) as u64;
                        scale = None;
                    }
                    (weight, scale, p.residency().clone())
                })
                .collect::<Vec<_>>();
            fixture::bind(&pairs)
        };
        for precision in [
            NumericFp8Precision::Bf16RneF32Accumulate,
            NumericFp8Precision::F32Tf32x3,
        ] {
            for rows in [1, 3, 8, 32] {
                for top_k in [1, 2, 3] {
                    let r =
                        BoundDecoderResources::new(routed_spec(top_k, 12), Arc::new(bound.clone()))
                            .unwrap();
                    let weights = if dtype == CheckpointDType::F8E4M3 {
                        294
                    } else {
                        3 * 12 * 8 * 4
                    };
                    let minimum = bucket_minimum(rows, top_k, 12, weights);
                    let configured = |cap| {
                        options_with_rows(&r, rows)
                            .with_hybrid_cuda_numeric_fp8_precision(
                                ExpertCacheLimits {
                                    max_experts: 1,
                                    max_bytes: cap,
                                },
                                4096,
                                precision,
                            )
                            .unwrap()
                    };
                    let exact =
                        HybridCudaMemoryEstimate::for_resources(&r, &configured(minimum), 32)
                            .unwrap();
                    let error =
                        HybridCudaMemoryEstimate::for_resources(&r, &configured(minimum - 1), 32)
                            .unwrap_err();
                    assert!(error.to_string().contains("numeric admission"), "{error}");
                    let extra =
                        HybridCudaMemoryEstimate::for_resources(&r, &configured(minimum + 17), 32)
                            .unwrap();
                    assert_eq!(
                        extra.weight_bytes_upper_bound,
                        exact.weight_bytes_upper_bound + 17
                    );
                    assert_eq!(
                        extra.workspace_bytes_upper_bound,
                        exact.workspace_bytes_upper_bound
                    );
                    assert_eq!(
                        exact.workspace_bytes_upper_bound,
                        (32 * 32 + 2 * 32) * rows * 4 + 4096
                    );
                }
            }
        }
    }
}

#[test]
fn bucket_admission_numeric_workspace_checks_max_bucket_not_one_row() {
    use ferrule_model::nn::ParameterResidency;
    let f = fixture::Fixture::new();
    let path = f.dir.join("wide-expert.bin");
    std::fs::write(&path, vec![0; f.bound.parameters().len() * (8 * 256 + 4)]).unwrap();
    let pairs = f
        .bound
        .parameters()
        .iter()
        .enumerate()
        .map(|(index, p)| {
            let mut weight = p.weight().slice().clone();
            let mut scale = p.scale().map(|s| s.slice().clone());
            if matches!(p.residency(), ParameterResidency::Expert { .. }) {
                weight.path = path.clone();
                weight.offset = (index * (8 * 256 + 4)) as u64;
                weight.bytes = 8 * 256;
                weight.shape = if p.role() == &TensorRole::RoutedExpertDown {
                    vec![8, 256]
                } else {
                    vec![256, 8]
                };
                let scale = scale.as_mut().unwrap();
                scale.path = path.clone();
                scale.offset = weight.offset + 8 * 256;
                scale.bytes = 4;
                scale.shape = weight.shape.iter().map(|n| n.div_ceil(128)).collect();
            }
            (weight, scale, p.residency().clone())
        })
        .collect::<Vec<_>>();
    let r =
        BoundDecoderResources::new(routed_spec(2, 256), Arc::new(fixture::bind(&pairs))).unwrap();
    let configured = |rows, precision, scratch| {
        options_with_rows(&r, rows)
            .with_hybrid_cuda_numeric_fp8_precision(
                ExpertCacheLimits {
                    max_experts: 1,
                    max_bytes: 1 << 20,
                },
                scratch,
                precision,
            )
            .unwrap()
    };
    let bf16 = NumericFp8Precision::Bf16RneF32Accumulate;
    HybridCudaMemoryEstimate::for_resources(&r, &configured(1, bf16, 4096), 32).unwrap();
    assert!(HybridCudaMemoryEstimate::for_resources(&r, &configured(8, bf16, 4096), 32).is_err());
    // Down projection requires eight packed BF16 rows and one decoded weight row.
    HybridCudaMemoryEstimate::for_resources(&r, &configured(8, bf16, 9 * 256 * 2), 32).unwrap();
    assert!(
        HybridCudaMemoryEstimate::for_resources(&r, &configured(8, bf16, 9 * 256 * 2 - 1), 32)
            .is_err()
    );
    HybridCudaMemoryEstimate::for_resources(
        &r,
        &configured(8, NumericFp8Precision::F32Tf32x3, 4096),
        32,
    )
    .unwrap();
}

#[test]
fn bucket_admission_dense_unrouted_keeps_cache_and_workspace_single_counted() {
    let f = reference::Fixture::new();
    let r = f.resources();
    let base = options(&r);
    let legacy = HybridCudaMemoryEstimate::for_resources(&r, &base, 32).unwrap();
    let operation = 8 * (2 * 32 + 8 + 1) * 4;
    let cap = 4096 + operation;
    let exact =
        HybridCudaMemoryEstimate::for_resources(&r, &numeric(base.clone(), 1, cap, 4096), 32)
            .unwrap();
    assert!(
        HybridCudaMemoryEstimate::for_resources(&r, &numeric(base, 1, cap - 1, 4096), 32).is_err()
    );
    assert_eq!(
        exact.weight_bytes_upper_bound,
        legacy.weight_bytes_upper_bound + operation
    );
    assert_eq!(
        exact.workspace_bytes_upper_bound,
        legacy.workspace_bytes_upper_bound + 4096
    );
}

#[test]
fn bucket_admission_dense_and_shared_bias_match_execution_at_cap_and_minus_one() {
    use ferrule_model::nn::ParameterResidency;
    let dense = reference::Fixture::new();
    let routed = fixture::Fixture::new();
    for shared in [false, true] {
        let original = if shared {
            resources(&routed)
        } else {
            dense.resources()
        };
        let spec = original.spec();
        let layers = spec
            .layers()
            .iter()
            .map(|layer| {
                let biased = SwiGlu::new(8, 12, true).unwrap();
                let feed_forward = if let FeedForward::Moe(moe) = layer.feed_forward() {
                    FeedForward::Moe(moe.clone().with_shared_expert(biased).unwrap())
                } else {
                    FeedForward::SwiGlu(biased)
                };
                DecoderLayer::new(
                    layer.index(),
                    layer.input_norm().clone(),
                    layer.attention().clone(),
                    layer.attention_residual().clone(),
                    layer.post_attention_norm().clone(),
                    feed_forward,
                    layer.feed_forward_residual().clone(),
                )
                .unwrap()
            })
            .collect();
        let biased = DecoderModelSpec::new(DecoderModelParts {
            architecture: spec.architecture().into(),
            hidden_size: 8,
            vocab_size: 11,
            max_sequence_length: spec.max_sequence_length(),
            token_embedding: spec.token_embedding().clone(),
            layers,
            final_norm: spec.final_norm().clone(),
            output: spec.output().clone(),
            tie_word_embeddings: false,
        })
        .unwrap();
        let path = dense.dir.join(format!("scratch-bias-{shared}.bin"));
        std::fs::write(&path, vec![0u8; 48]).unwrap();
        let mut pairs = original
            .state_dict()
            .parameters()
            .iter()
            .map(|p| {
                let mut weight = p.weight().slice().clone();
                // A raw checkpoint slice may be Unknown; the bound parameter owns
                // the resolved semantic role needed when rebinding this fixture.
                weight.role = p.role().clone();
                (
                    weight,
                    p.scale().map(|s| s.slice().clone()),
                    p.residency().clone(),
                )
            })
            .collect::<Vec<_>>();
        for layer in spec.layers() {
            let roles = if shared {
                [
                    TensorRole::SharedExpertGate,
                    TensorRole::SharedExpertUp,
                    TensorRole::SharedExpertDown,
                ]
            } else {
                [
                    TensorRole::DenseMlpGate,
                    TensorRole::DenseMlpUp,
                    TensorRole::DenseMlpDown,
                ]
            };
            for (role, width) in roles.into_iter().zip([12, 12, 8]) {
                pairs.push((
                    ferrule_model::checkpoint::CheckpointTensorSlice {
                        name: format!("layers.{}.scratch_bias.{role:?}", layer.index()),
                        role,
                        path: path.clone(),
                        offset: 0,
                        bytes: width as u64 * 4,
                        dtype: CheckpointDType::F32,
                        shape: vec![width],
                    },
                    None,
                    ParameterResidency::layer(layer.index()),
                ));
            }
        }
        let resources =
            BoundDecoderResources::new(biased, Arc::new(fixture::bind(&pairs))).unwrap();
        for rows in [1, 3, 8, 32] {
            for precision in [
                NumericFp8Precision::Bf16RneF32Accumulate,
                NumericFp8Precision::F32Tf32x3,
            ] {
                // Input + gate/up/product + output; then each broadcast bias
                // and its gather IDs. Shared/external excludes remote experts.
                let operation = rows * (8 + 3 * 12 + 8 + 13 + 13 + 9) * 4;
                let cap = 4096 + operation;
                let configured = |cap| {
                    options_with_rows(&resources, rows)
                        .with_hybrid_cuda_numeric_fp8_precision(
                            ExpertCacheLimits {
                                max_experts: 1,
                                max_bytes: cap,
                            },
                            4096,
                            precision,
                        )
                        .unwrap()
                };
                let estimate = |cap| {
                    if shared {
                        HybridCudaMemoryEstimate::for_resources_with_routed_experts(
                            &resources,
                            &configured(cap),
                            32,
                        )
                    } else {
                        HybridCudaMemoryEstimate::for_resources(&resources, &configured(cap), 32)
                    }
                };
                let exact = estimate(cap).unwrap();
                let error = estimate(cap - 1).unwrap_err();
                assert!(error.to_string().contains("numeric admission"), "{error}");
                let extra = estimate(cap + 17).unwrap();
                assert_eq!(
                    extra.workspace_bytes_upper_bound,
                    exact.workspace_bytes_upper_bound
                );
                assert_eq!(
                    extra.weight_bytes_upper_bound,
                    exact.weight_bytes_upper_bound + if shared { 0 } else { 17 }
                );
            }
        }
    }
}

#[test]
#[ignore = "requires CUDA GPU; exact metadata cap must also admit maximum-row prefill"]
fn bucket_admission_exact_cap_tiny_gpu_max_rows_prefill() {
    for precision in [
        NumericFp8Precision::Bf16RneF32Accumulate,
        NumericFp8Precision::F32Tf32x3,
    ] {
        let f = fixture::Fixture::with_precision(precision);
        // Top-3 selects every expert in every row, attaining the worst bucket.
        let r = BoundDecoderResources::new(routed_spec(3, 12), Arc::new(f.bound.clone())).unwrap();
        let cap = bucket_minimum(8, 3, 12, 294);
        let opts = options(&r)
            .with_hybrid_cuda_numeric_fp8_precision(
                ExpertCacheLimits {
                    max_experts: 1,
                    max_bytes: cap,
                },
                4096,
                precision,
            )
            .unwrap();
        let estimate = HybridCudaMemoryEstimate::for_resources(&r, &opts, 32).unwrap();
        let device = HybridCudaDevice::new_on_device(
            0,
            HybridCudaMemoryBudget {
                kv_pages: 32,
                state_bytes: estimate.per_sequence_state_bytes * 6,
                weight_bytes: estimate.weight_bytes_upper_bound,
                workspace_bytes: estimate.workspace_bytes_upper_bound,
            },
        )
        .unwrap();
        let mut runner = GpuRunner::hybrid_cuda(r, tokenizer(&f), opts, device.clone()).unwrap();
        let mut states = vec![runner.create_sequence_state().unwrap()];
        let mut pages = vec![Vec::new()];
        let output = step(
            &mut runner,
            &mut states,
            &mut pages,
            &[(&[1, 2, 3, 4, 5, 6, 7, 8], ForwardPhase::Prefill)],
            1,
            Ending::Publish,
        );
        assert!(output.iter().flatten().all(|x| x.is_finite()));
        let stats = runner
            .forward_executor()
            .module()
            .expert_cache_stats()
            .unwrap();
        assert_eq!(stats.peak_bytes, cap);
        assert_eq!(stats.scratch_bytes, 4096);
        assert_eq!(stats.resident_experts, 1);
        assert_eq!(stats.pending_upload_bytes, 0);
        let usage = runner
            .forward_executor()
            .module()
            .numeric_workspace_usage()
            .unwrap();
        assert_eq!(usage.0, 4096);
        assert!(usage.1 <= usage.0);
        assert!(!device.needs_quarantine());
    }
}

#[test]
fn numeric_hybrid_metadata_estimate_and_strict_old_profile_rejection() {
    let f = fixture::Fixture::new();
    let r = resources(&f);
    let old = options(&r);
    typed(HybridCudaMemoryEstimate::for_resources(&r, &old, 32).unwrap_err());
    assert!(old.hybrid_cuda_numeric_fp8().is_none());
    let bf16 = GenericDecoderOptions::standard_cpu(
        r.spec(),
        ModelFamily::Unknown("numeric-profile-rejection".into()),
        WeightSource::Safetensors,
        2,
        32,
        8,
        1,
        1 << 20,
        ExecutionPrecisionPolicy::bf16_compatibility(),
    )
    .unwrap();
    typed(
        bf16.clone()
            .with_hybrid_cuda_numeric_fp8(
                ExpertCacheLimits {
                    max_experts: 1,
                    max_bytes: 8192,
                },
                4096,
            )
            .unwrap_err(),
    );
    typed(HybridCudaMemoryEstimate::for_resources(&r, &bf16, 32).unwrap_err());
    let configured = numeric(old.clone(), 1, 8192, 4096);
    let estimate = HybridCudaMemoryEstimate::for_resources(&r, &configured, 32).unwrap();
    assert!(HybridCudaMemoryEstimate::for_resources(&r, &configured, usize::MAX).is_err());
    assert!(
        HybridCudaMemoryEstimate::for_resources(&r, &numeric(old.clone(), 1, 4200, 4096), 32)
            .is_err()
    );
    let larger = numeric(old.clone(), 64, 16384, 4096);
    let larger = HybridCudaMemoryEstimate::for_resources(&r, &larger, 32).unwrap();
    assert_eq!(
        larger.weight_bytes_upper_bound - estimate.weight_bytes_upper_bound,
        8192
    );
    let prefix = configured.clone().with_active_layers(1).unwrap();
    assert!(
        HybridCudaMemoryEstimate::for_resources(&r, &prefix, 32).is_err(),
        "a paged hybrid prefix must include a full-attention layer"
    );
    let more_kv = HybridCudaMemoryEstimate::for_resources(&r, &configured, 64).unwrap();
    assert_eq!(more_kv.kv_bytes, estimate.kv_bytes * 2);
    assert_eq!(
        more_kv.per_sequence_state_bytes,
        estimate.per_sequence_state_bytes
    );
    let s = HybridStateSchema::from_spec(r.spec(), 2).unwrap();
    assert_eq!(
        KvLayoutSchema::planes(&s.kv_planes(2, 32).unwrap()).len(),
        2,
        "one full attention K/V pair, not two layers of full KV"
    );
    assert!(estimate.per_sequence_state_bytes > 0);
    typed(
        GenericDecoderRunner::<HybridCpuDecoder>::hybrid_cpu(r, tokenizer(&f), configured)
            .err()
            .expect("numeric must not fall back to CPU"),
    );
    for (entries, bytes, scratch) in [(0, 8192, 4096), (1, 4096, 4096), (1, 8192, 0)] {
        typed(
            old.clone()
                .with_hybrid_cuda_numeric_fp8(
                    ExpertCacheLimits {
                        max_experts: entries,
                        max_bytes: bytes,
                    },
                    scratch,
                )
                .unwrap_err(),
        );
    }
}

#[test]
#[ignore = "requires CUDA GPU; stale expert metadata rejects before any forward kernel or KV write"]
fn hybrid_numeric_stale_expert_preflight_precedes_kv_and_expert_mutation() {
    let f = fixture::Fixture::new();
    let r = resources(&f);
    let opts = numeric(options(&r), 1, 8192, 4096);
    let estimate = HybridCudaMemoryEstimate::for_resources(&r, &opts, 32).unwrap();
    let device = HybridCudaDevice::new_on_device(
        0,
        HybridCudaMemoryBudget {
            kv_pages: 32,
            state_bytes: estimate.per_sequence_state_bytes * 6,
            weight_bytes: estimate.weight_bytes_upper_bound,
            workspace_bytes: estimate.workspace_bytes_upper_bound,
        },
    )
    .unwrap();
    let mut runner = GpuRunner::hybrid_cuda(r, tokenizer(&f), opts, device.clone()).unwrap();
    let mut states = vec![runner.create_sequence_state().unwrap()];
    let before = snapshot(&states);
    let cache = runner
        .forward_executor()
        .module()
        .expert_cache_stats()
        .unwrap();
    let batch = ExecutionBatch::new(
        ForwardMode::Prefill,
        vec![1],
        vec![0],
        vec![Some(KvWriteSlot::new(2))],
        vec![LogitsRequest::Full],
        vec![ExecutionSequence::new(
            StateSlot::new(0),
            ForwardPhase::Prefill,
            0..1,
            0,
            1,
            0..1,
        )],
        vec![KvBlockId::new(1)],
    );
    let tx = ExecutionTransactionId::new(1).unwrap();
    let reservation = KvReservationView {
        state_slot: StateSlot::new(0),
        execution_state_slot: StateSlot::new(0),
        positions: 0..1,
        newly_allocated: vec![KvPageId(1)],
        generation: states[0].core().generation(),
        execution_generation: states[0].core().generation(),
        cow_replacement: None,
    };
    runner
        .prepare_multi_session_batch(tx, &mut states, &batch, &[reservation])
        .unwrap();
    // Preparing a transaction may allocate its working copies; the forward's
    // metadata preflight must precede all activation/expert/KV kernels.
    std::fs::rename(f.dir.join("weights.bin"), f.dir.join("old.bin")).unwrap();
    std::fs::copy(f.dir.join("old.bin"), f.dir.join("weights.bin")).unwrap();
    device.operators().reset_counters();
    assert!(
        runner
            .execute_multi_session_batch_progress(tx, &mut states, &batch)
            .is_err()
    );
    assert_eq!(device.operators().counters().kernel_launches, 0);
    assert_eq!(device.operators().counters().artifact_uploads, 0);
    assert_eq!(
        runner
            .forward_executor()
            .module()
            .expert_cache_stats()
            .unwrap(),
        cache
    );
    assert_eq!(snapshot(&states), before);
    finish(&mut runner, &mut states, tx, TransactionEndIntent::Abort);
    assert!(!device.needs_quarantine());
}

#[test]
#[ignore = "requires CUDA GPU; complete two-layer hybrid numeric MoE versus independent Torch"]
fn hybrid_numeric_tiny_moe_top2_cache1_full_runner_matches_torch() {
    tiny_numeric_runner(NumericFp8Precision::Bf16RneF32Accumulate);
}

#[test]
#[ignore = "requires CUDA GPU; explicit F32 TF32x3 versus independent F32 Torch"]
fn hybrid_numeric_f32_tiny_moe_top2_cache1_full_runner_matches_torch() {
    tiny_numeric_runner(NumericFp8Precision::F32Tf32x3);
}

fn tiny_numeric_runner(precision: NumericFp8Precision) {
    let f = fixture::Fixture::with_precision(precision);
    let r = resources(&f);
    let opts = options(&r)
        .with_hybrid_cuda_numeric_fp8_precision(
            ExpertCacheLimits {
                max_experts: 1,
                max_bytes: 8192,
            },
            4096,
            precision,
        )
        .unwrap();
    let estimate = HybridCudaMemoryEstimate::for_resources(&r, &opts, 32).unwrap();
    let device = HybridCudaDevice::new_on_device(
        0,
        HybridCudaMemoryBudget {
            kv_pages: 32,
            state_bytes: estimate.per_sequence_state_bytes * 6,
            weight_bytes: estimate.weight_bytes_upper_bound,
            workspace_bytes: estimate.workspace_bytes_upper_bound,
        },
    )
    .unwrap();
    let mut runner = GpuRunner::hybrid_cuda(r, tokenizer(&f), opts, device.clone()).unwrap();
    let mut held_states = Vec::new();
    for _ in 0..5 {
        held_states.push(runner.create_sequence_state().unwrap());
    }
    assert!(
        runner.create_sequence_state().is_err(),
        "default/forks/working states all consume the same finite state budget"
    );
    for state in held_states {
        runner.try_release_sequence_state(state).unwrap();
    }
    assert_eq!(device.live_state_bytes(), estimate.per_sequence_state_bytes);
    assert_eq!(
        runner
            .forward_executor()
            .module()
            .expert_cache_stats()
            .unwrap()
            .uploads,
        0,
        "preparation must not upload experts"
    );
    let tokens: Vec<u32> = serde_json::from_value(f.oracle["tokens"].clone()).unwrap();
    let mut states = vec![runner.create_sequence_state().unwrap()];
    let mut pages = vec![Vec::new()];
    let mut tx = 1;
    for (call, (begin, end)) in std::iter::once((0, 3))
        .chain((3..tokens.len()).map(|i| (i, i + 1)))
        .enumerate()
    {
        let phase = if call == 0 {
            ForwardPhase::Prefill
        } else {
            ForwardPhase::Decode
        };
        if call == 2 {
            let before = snapshot(&states);
            let aborted = step(
                &mut runner,
                &mut states,
                &mut pages,
                &[(&tokens[begin..end], phase)],
                tx,
                Ending::Abort,
            );
            close(
                &aborted[0],
                &floats(&f.oracle["logits"][begin]),
                "numeric aborted logits",
            );
            assert_eq!(snapshot(&states), before);
            tx += 1;
        }
        let metadata_before = runner
            .forward_executor()
            .module()
            .expert_metadata_preflight_stats();
        let output = step(
            &mut runner,
            &mut states,
            &mut pages,
            &[(&tokens[begin..end], phase)],
            tx,
            Ending::Publish,
        );
        let metadata_after = runner
            .forward_executor()
            .module()
            .expert_metadata_preflight_stats();
        assert_eq!(metadata_after.prepared_experts, 6);
        assert_eq!(metadata_after.source_snapshots, 1);
        assert_eq!(
            metadata_after.preflights,
            metadata_before.preflights + 1,
            "one full-image preflight per forward, not per layer"
        );
        assert_eq!(
            metadata_after.source_checks,
            metadata_before.source_checks + 1
        );
        tx += 1;
        for (offset, actual) in output.iter().enumerate() {
            close(
                actual,
                &floats(&f.oracle["logits"][begin + offset]),
                &format!("numeric hybrid token {}", begin + offset),
            );
        }
        assert!(
            states[0]
                .kv_state()
                .layers()
                .iter()
                .flatten()
                .all(|s| s.position() == end)
        );
        let stats = runner
            .forward_executor()
            .module()
            .expert_cache_stats()
            .unwrap();
        assert_eq!(stats.resident_experts, 1);
        assert_eq!(stats.resident_bytes, 294);
        assert_eq!(stats.scratch_bytes, 4096);
        assert_eq!(stats.pending_upload_bytes, 0);
        assert!(stats.peak_bytes <= 8192);
        assert!(!stats.quarantined);
    }
    device.operators().reset_counters();
    let fork = runner
        .fork_sequence_state_from(&states[0], tokens.len())
        .unwrap();
    assert_eq!(device.operators().counters().device_to_host_copies, 0);
    assert_eq!(
        device.live_state_bytes(),
        estimate.per_sequence_state_bytes * 3
    );
    runner.try_release_sequence_state(fork).unwrap();
    let module = runner.forward_executor().module();
    let stats = module.expert_cache_stats().unwrap();
    assert!(stats.evictions > 30 && stats.uploads > 30);
    let workspace = module.numeric_workspace_usage().unwrap();
    assert!(workspace.1 <= 4096 && workspace.3 > workspace.2);
    eprintln!("tiny full hybrid: {stats:?}, workspace={workspace:?}");
    runner
        .try_release_sequence_state(states.pop().unwrap())
        .unwrap();
    assert_eq!(device.live_state_bytes(), estimate.per_sequence_state_bytes);
    assert!(!device.needs_quarantine());
}

#[test]
#[ignore = "requires CUDA and the reference generator's native Qwen MoE tiny checkpoint"]
fn hybrid_numeric_native_moe_tiny_checkpoint_matches_full_reference_adapter() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/validation/qwen35-fp8-reference-full40-v1");
    let oracle = Oracle::open_case(&dir, "tiny-lazy");
    let r = native_tiny_resources(&dir.join("tiny-checkpoint"));
    assert_eq!(r.spec().layers().len(), 2);
    let opts = numeric(options(&r), 1, 2 * 1024 * 1024, 1024 * 1024);
    let estimate = HybridCudaMemoryEstimate::for_resources(&r, &opts, 32).unwrap();
    let device = HybridCudaDevice::new_on_device(
        0,
        HybridCudaMemoryBudget {
            kv_pages: 32,
            state_bytes: estimate.per_sequence_state_bytes * 4,
            weight_bytes: estimate.weight_bytes_upper_bound,
            workspace_bytes: estimate.workspace_bytes_upper_bound,
        },
    )
    .unwrap();
    let f = fixture::Fixture::new();
    let mut runner = GpuRunner::hybrid_cuda(r, tokenizer(&f), opts, device.clone()).unwrap();
    let mut states = vec![runner.create_sequence_state().unwrap()];
    let mut pages = vec![Vec::new()];
    let mut mismatches = Vec::new();
    for (i, stage) in ["prefill", "decode.0"].into_iter().enumerate() {
        let tokens: Vec<u32> =
            serde_json::from_value(oracle.report["calls"][i]["input_token_ids"].clone()).unwrap();
        let logits = step(
            &mut runner,
            &mut states,
            &mut pages,
            &[(
                &tokens,
                if i == 0 {
                    ForwardPhase::Prefill
                } else {
                    ForwardPhase::Decode
                },
            )],
            i as u64 + 1,
            Ending::Publish,
        );
        let actual = logits.iter().flatten().copied().collect::<Vec<_>>();
        if !report_diff(
            &actual,
            &oracle.values(&format!("{stage}.logits")),
            stage,
            2e-5,
            3e-4,
        ) {
            mismatches.push(stage.to_owned());
        }
        let (conv, recurrent) = states[0].kv_state().layers()[0]
            .as_ref()
            .unwrap()
            .diagnostic_snapshot()
            .unwrap();
        for (field, actual) in [("conv_states", conv), ("recurrent_states", recurrent)] {
            let label = format!("{stage}.cache.{field}.00");
            if !report_diff(&actual, &oracle.values(&label), &label, 2e-5, 3e-4) {
                mismatches.push(label);
            }
        }
        let stats = runner
            .forward_executor()
            .module()
            .expert_cache_stats()
            .unwrap();
        assert_eq!(stats.resident_experts, 1);
        assert!(stats.evictions > 0 && stats.peak_bytes <= 2 * 1024 * 1024);
    }
    runner
        .try_release_sequence_state(states.pop().unwrap())
        .unwrap();
    assert!(
        mismatches.is_empty(),
        "native tiny adapter mismatch: {mismatches:?}"
    );
}

fn native_tiny_resources(dir: &Path) -> BoundDecoderResources {
    use ferrule_model::nn::ParameterResidency;
    let config: Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
    let c = &config["text_config"];
    let n = |key: &str| c[key].as_u64().unwrap() as usize;
    let h = n("hidden_size");
    let eps = c["rms_norm_eps"].as_f64().unwrap() as f32;
    let norm = |width| RmsNorm::new(width, eps).unwrap().with_one_plus_weight();
    let rope = RotaryEmbedding::new(
        n("head_dim"),
        c["rope_parameters"]["rope_theta"].as_f64().unwrap() as f32,
        RotaryPairing::SplitHalf,
        RotaryRegion::Prefix {
            dimensions: (n("head_dim") as f64
                * c["rope_parameters"]["partial_rotary_factor"]
                    .as_f64()
                    .unwrap()) as usize,
        },
        RotaryScaling::None,
    )
    .unwrap();
    let layers = c["layer_types"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, kind)| {
            let attention = if kind == "linear_attention" {
                Attention::GatedDeltaNet(
                    GatedDeltaNetAttention::new(
                        h,
                        n("linear_num_key_heads"),
                        n("linear_num_value_heads"),
                        n("linear_key_head_dim"),
                        n("linear_value_head_dim"),
                        n("linear_conv_kernel_dim"),
                        eps,
                        false,
                    )
                    .unwrap(),
                )
            } else {
                Attention::Gqa(
                    GqaAttention::new(
                        h,
                        n("num_attention_heads"),
                        n("num_key_value_heads"),
                        n("head_dim"),
                        false,
                        rope.clone(),
                    )
                    .unwrap()
                    .with_gated_query()
                    .unwrap()
                    .with_qk_norms(norm(n("head_dim")), norm(n("head_dim")))
                    .unwrap(),
                )
            };
            let router = MoeRouterSpec::new(
                n("num_experts"),
                n("num_experts_per_tok"),
                RouterScoreFunction::Softmax,
                RouterSelection::TopK,
                true,
                1.0,
            )
            .unwrap();
            let moe = Moe::new(h, n("moe_intermediate_size"), router, false)
                .unwrap()
                .with_shared_expert(
                    SwiGlu::new(h, n("shared_expert_intermediate_size"), false).unwrap(),
                )
                .unwrap()
                .with_shared_expert_gate(Linear::new(h, 1, false).unwrap())
                .unwrap();
            DecoderLayer::new(
                i,
                norm(h),
                attention,
                Residual::Add,
                norm(h),
                FeedForward::Moe(moe),
                Residual::Add,
            )
            .unwrap()
        })
        .collect();
    let spec = DecoderModelSpec::new(DecoderModelParts {
        architecture: "independent-native-moe-fixture".into(),
        hidden_size: h,
        vocab_size: n("vocab_size"),
        max_sequence_length: Some(n("max_position_embeddings")),
        token_embedding: Embedding::new(n("vocab_size"), h, None).unwrap(),
        layers,
        final_norm: norm(h),
        output: Linear::new(h, n("vocab_size"), false).unwrap(),
        tie_word_embeddings: false,
    })
    .unwrap();
    let path = dir.join("model.safetensors");
    let mut file = std::fs::File::open(&path).unwrap();
    let mut len = [0; 8];
    file.read_exact(&mut len).unwrap();
    let len = u64::from_le_bytes(len);
    assert!(len < 16 * 1024 * 1024);
    let mut header = vec![0; len as usize];
    file.read_exact(&mut header).unwrap();
    let header: Value = serde_json::from_slice(&header).unwrap();
    let mut pairs = Vec::new();
    for (name, meta) in header.as_object().unwrap() {
        if name == "__metadata__" || name.ends_with("_scale_inv") {
            continue;
        }
        let mut canonical = name
            .strip_prefix("model.language_model.")
            .unwrap_or(name)
            .to_owned();
        for (a, b) in [
            ("embed_tokens.weight", "token_embedding.weight"),
            ("lm_head.weight", "output.weight"),
            ("input_layernorm", "input_norm"),
            ("post_attention_layernorm", "post_attention_norm"),
            ("linear_attn.in_proj_qkv", "attention.qkv"),
            ("linear_attn.in_proj_z", "attention.z"),
            ("linear_attn.in_proj_a", "attention.a"),
            ("linear_attn.in_proj_b", "attention.beta"),
            ("linear_attn.out_proj", "attention.output"),
            ("linear_attn.conv1d", "attention.conv"),
            ("linear_attn.A_log", "attention.a_log.weight"),
            ("linear_attn.dt_bias", "attention.dt_bias.weight"),
            ("linear_attn.norm", "attention.norm"),
            ("self_attn.q_proj", "attention.query"),
            ("self_attn.k_proj", "attention.key"),
            ("self_attn.v_proj", "attention.value"),
            ("self_attn.o_proj", "attention.output"),
            ("self_attn.q_norm", "attention.query_norm"),
            ("self_attn.k_norm", "attention.key_norm"),
            ("mlp.shared_expert_gate", "feed_forward.shared_gate"),
            ("mlp.shared_expert", "feed_forward.shared"),
            ("mlp.gate", "feed_forward.router"),
            ("mlp.experts", "feed_forward.experts"),
            ("gate_proj", "gate"),
            ("up_proj", "up"),
            ("down_proj", "down"),
        ] {
            canonical = canonical.replace(a, b);
        }
        if canonical == "norm.weight" {
            canonical = "final_norm.weight".into();
        }
        let role = fixture::role(&canonical);
        let slice = |meta: &Value, name: String| {
            let a = meta["data_offsets"][0].as_u64().unwrap();
            let b = meta["data_offsets"][1].as_u64().unwrap();
            CheckpointTensorSlice {
                name,
                role: role.clone(),
                path: path.clone(),
                offset: 8 + len + a,
                bytes: b - a,
                dtype: CheckpointDType::from_safetensors_dtype(meta["dtype"].as_str().unwrap()),
                shape: serde_json::from_value(meta["shape"].clone()).unwrap(),
            }
        };
        let weight = slice(meta, canonical.clone());
        let scale = header
            .get(format!("{name}_scale_inv"))
            .map(|meta| slice(meta, canonical.replace(".weight", ".weight_scale_inv")));
        let parts = canonical.split('.').collect::<Vec<_>>();
        let residency = if canonical.contains(".experts.") {
            ParameterResidency::expert(parts[1].parse().unwrap(), parts[4].parse().unwrap())
        } else if canonical.starts_with("layers.") {
            ParameterResidency::layer(parts[1].parse().unwrap())
        } else {
            ParameterResidency::Static
        };
        pairs.push((weight, scale, residency));
    }
    BoundDecoderResources::new(spec, Arc::new(fixture::bind(&pairs))).unwrap()
}

struct Oracle {
    path: PathBuf,
    header: Value,
    start: u64,
    report: Value,
}
impl Oracle {
    fn open(dir: &Path) -> Self {
        Self::open_case(dir, "hello")
    }
    fn open_case(dir: &Path, case: &str) -> Self {
        Self::open_schema(dir, case, "ferrule.qwen35-fp8-reference.v1")
    }
    fn open_schema(dir: &Path, case: &str, schema: &str) -> Self {
        let manifest: Value =
            serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest["schema"], schema);
        assert_eq!(manifest["full_reference_complete"], true);
        let report: Value =
            serde_json::from_slice(&std::fs::read(dir.join(format!("{case}.json"))).unwrap())
                .unwrap();
        let path = dir.join(report["file"].as_str().unwrap());
        let checksum = Command::new("sha256sum").arg(&path).output().unwrap();
        assert!(checksum.status.success());
        assert_eq!(
            String::from_utf8(checksum.stdout)
                .unwrap()
                .split_whitespace()
                .next()
                .unwrap(),
            report["sha256"].as_str().unwrap()
        );
        let mut file = std::fs::File::open(&path).unwrap();
        let mut len = [0; 8];
        file.read_exact(&mut len).unwrap();
        let len = u64::from_le_bytes(len);
        assert!(len < 16 * 1024 * 1024);
        let mut header = vec![0; len as usize];
        file.read_exact(&mut header).unwrap();
        Self {
            path,
            header: serde_json::from_slice(&header).unwrap(),
            start: 8 + len,
            report,
        }
    }
    fn values(&self, name: &str) -> Vec<f32> {
        let t = &self.header[name];
        assert_eq!(t["dtype"], "F32");
        let a = t["data_offsets"][0].as_u64().unwrap();
        let b = t["data_offsets"][1].as_u64().unwrap();
        let shape = serde_json::from_value(t["shape"].clone()).unwrap();
        let slice = CheckpointTensorSlice {
            name: name.into(),
            role: TensorRole::Unknown,
            path: self.path.clone(),
            offset: self.start + a,
            bytes: b - a,
            dtype: CheckpointDType::F32,
            shape,
        };
        CheckpointTensorReader::new(64 * 1024 * 1024)
            .read_slice(&slice)
            .unwrap()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }
}
#[test]
#[ignore = "diagnostic only: BF16 full35B all-logit numerics are known unaccepted; requires CUDA/NAS and 900s timeout"]
fn qwen35_35b_numeric_full40_hello_prefill_decode_matches_reference() {
    use ferrule_model::models::qwen35::Qwen35Adapter;
    let started = std::time::Instant::now();
    let model = std::env::var_os("FERRULE_NUMERIC_FP8_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| "/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8".into());
    let dir = std::env::var_os("FERRULE_NUMERIC_FP8_REFERENCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/validation/qwen35-fp8-reference-full40-v1")
        });
    let oracle = Oracle::open(&dir);
    let (_, r) = Qwen35Adapter::bind_hf_metadata(&model).unwrap();
    assert_eq!(r.spec().layers().len(), 40);
    assert_eq!(
        r.spec()
            .layers()
            .iter()
            .filter(|l| matches!(l.attention(), Attention::GatedDeltaNet(_)))
            .count(),
        30
    );
    let options = GenericDecoderOptions::standard_cpu(
        r.spec(),
        ModelFamily::Qwen35,
        WeightSource::Safetensors,
        2,
        8,
        1,
        1,
        1 << 30,
        ExecutionPrecisionPolicy::f32(),
    )
    .unwrap();
    let options = numeric(options, 64, 320 * 1024 * 1024, 64 * 1024 * 1024);
    let estimate = HybridCudaMemoryEstimate::for_resources(&r, &options, 8).unwrap();
    eprintln!(
        "40-layer numeric CUDA metadata admission: {estimate:?}, elapsed={:?}",
        started.elapsed()
    );
    let device = HybridCudaDevice::new_on_device(
        0,
        HybridCudaMemoryBudget {
            kv_pages: 8,
            state_bytes: estimate.per_sequence_state_bytes * 4,
            weight_bytes: estimate.weight_bytes_upper_bound,
            workspace_bytes: estimate.workspace_bytes_upper_bound,
        },
    )
    .unwrap();
    eprintln!(
        "device free/total={:?}",
        device.operators().memory_info().unwrap()
    );
    let mut runner = GpuRunner::hybrid_cuda(
        r,
        TokenizerHandle::load(&model).unwrap(),
        options,
        device.clone(),
    )
    .unwrap();
    eprintln!(
        "full40 prepared: resident={} bytes, elapsed={:?}",
        runner
            .forward_executor()
            .module()
            .resident_parameter_bytes(),
        started.elapsed()
    );
    assert_eq!(
        runner
            .forward_executor()
            .module()
            .expert_cache_stats()
            .unwrap()
            .uploads,
        0
    );
    let trace_stage = std::rc::Rc::new(std::cell::Cell::new("prefill"));
    if let Some(dir) = std::env::var_os("FERRULE_HYBRID_TRACE_DIR") {
        use std::io::Write;
        let dir = PathBuf::from(dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut log = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(dir.join("events.jsonl"))
            .unwrap();
        let stage = std::rc::Rc::clone(&trace_stage);
        let mut sequence = 0usize;
        runner.forward_executor().module().set_diagnostic_trace(move |event| {
            let file = format!("{sequence:05}.f32");
            let values = event.operators.download_f32_buffer(event.buffer)?;
            let payload = values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
            std::fs::write(dir.join(&file), payload)?;
            let parameter = event.parameter;
            let tensor = |s: &CheckpointTensorSlice| serde_json::json!({"name":s.name,"path":s.path,"offset":s.offset,"bytes":s.bytes,"shape":s.shape,"dtype":format!("{:?}",s.dtype)});
            let record = serde_json::json!({
                "weight":parameter.map(|p|tensor(p.weight().slice())),
                "scale":parameter.and_then(|p|p.scale()).map(|p|tensor(p.slice())),
                "epsilon":event.norm.map(|p|p.epsilon()),
                "one_plus_weight":event.norm.map(|p|p.one_plus_weight()),
                "sequence":sequence, "stage":stage.get(), "layer":event.layer, "event":event.name,
                "shape":[event.shape.rows(),event.shape.width()], "file":file,
                "parameter":parameter.map(|p|p.path().to_string()),
                "role":parameter.map(|p|format!("{:?}",p.role())),
                "numeric":parameter.is_some_and(|p|p.numeric_fp8_encoding().is_some()),
                "ids":event.routes.map(|r|r.expert_ids()), "weights":event.routes.map(|r|r.weights()),
            });
            writeln!(log,"{record}")?;
            sequence += 1;
            Ok(())
        });
    }
    let mut states = vec![runner.create_sequence_state().unwrap()];
    let mut pages = vec![Vec::new()];
    let mut mismatches = Vec::new();
    for (i, stage) in ["prefill", "decode.0"].into_iter().enumerate() {
        trace_stage.set(stage);
        let tokens: Vec<u32> =
            serde_json::from_value(oracle.report["calls"][i]["input_token_ids"].clone()).unwrap();
        let logits = step(
            &mut runner,
            &mut states,
            &mut pages,
            &[(
                &tokens,
                if i == 0 {
                    ForwardPhase::Prefill
                } else {
                    ForwardPhase::Decode
                },
            )],
            i as u64 + 1,
            Ending::Publish,
        );
        let actual = logits.iter().flatten().copied().collect::<Vec<_>>();
        let expected = oracle.values(&format!("{stage}.logits"));
        let stats = runner
            .forward_executor()
            .module()
            .expert_cache_stats()
            .unwrap();
        eprintln!(
            "full40 Hello/{stage}: elapsed={:?}, argmax={}, cache={stats:?}, workspace={:?}",
            started.elapsed(),
            argmax(logits.last().unwrap()),
            runner.forward_executor().module().numeric_workspace_usage()
        );
        if !report_diff(
            &actual,
            &expected,
            &format!("full40 Hello/{stage} all logits"),
            2e-3,
            2e-3,
        ) {
            mismatches.push(stage.to_owned());
        }
        assert_eq!(
            argmax(logits.last().unwrap()),
            oracle.report["calls"][i]["next_token_id"].as_u64().unwrap() as u32
        );
        assert!(stats.resident_experts <= 64 && stats.peak_bytes <= 320 * 1024 * 1024);
        assert_eq!(stats.pending_upload_bytes, 0);
        assert!(
            states[0]
                .kv_state()
                .layers()
                .iter()
                .flatten()
                .all(|s| s.position() == i + 1)
        );
        assert!(!device.needs_quarantine());
        for (layer, state) in states[0].kv_state().layers().iter().enumerate() {
            if let Some(state) = state {
                let (conv, recurrent) = state.diagnostic_snapshot().unwrap();
                for (field, actual) in [("conv_states", conv), ("recurrent_states", recurrent)] {
                    let label = format!("{stage}.cache.{field}.{layer:02}");
                    if !report_diff(&actual, &oracle.values(&label), &label, 2e-3, 2e-3) {
                        mismatches.push(label);
                    }
                }
            }
        }
    }
    runner
        .try_release_sequence_state(states.pop().unwrap())
        .unwrap();
    assert!(
        mismatches.is_empty(),
        "strict full40 all-logit comparison failed: {mismatches:?}; both calls executed before reporting failure"
    );
}

#[test]
#[ignore = "requires CUDA; observer has no implicit D2H and errors use existing completion custody"]
fn numeric_diagnostic_is_opt_in_and_observer_error_is_recoverable() {
    use ferrule_backend::cuda::operators::linear::CudaOperators;
    use std::{cell::Cell, rc::Rc};
    let f = fixture::Fixture::new();
    let ops = Rc::new(CudaOperators::new_on_device(0).unwrap());
    let mut owner = CudaStandardDecoderOperators::new_numeric_fp8(
        Rc::clone(&ops),
        f.bound.parameters(),
        ExpertCachePolicy::Bounded(ExpertCacheLimits {
            max_experts: 1,
            max_bytes: 8192,
        }),
        4096,
    )
    .unwrap();
    let p = f
        .bound
        .get(
            &ferrule_model::nn::ModulePath::new("layers.0.feed_forward.shared_gate.weight")
                .unwrap(),
        )
        .unwrap();
    let linear = StateDictMaterializer::new(1024)
        .unwrap()
        .prepared_linear(p, p.role().clone())
        .unwrap();
    let input = owner
        .bind_rows(Rows::Host(
            HostRows::new(
                RowsShape::new(1, 8).unwrap(),
                RowsDType::F32,
                None,
                vec![0.5; 8],
            )
            .unwrap(),
        ))
        .unwrap();
    let ready = |p| match p {
        OperatorProgress::Ready(v) => v,
        _ => panic!("expected Ready"),
    };
    let expected = ready(owner.linear(&linear, &input, None).unwrap());
    let expected = owner.download_rows(expected).unwrap();
    let count = Rc::new(Cell::new(0));
    let observed = Rc::clone(&count);
    owner.set_diagnostic_trace(move |e| {
        assert!(e.linear.is_some());
        observed.set(observed.get() + 1);
        Ok(())
    });
    ops.reset_counters();
    let result = ready(owner.linear(&linear, &input, None).unwrap());
    assert_eq!(count.get(), 2);
    assert_eq!(ops.counters().device_to_host_copies, 0);
    assert_eq!(
        owner.download_rows(result).unwrap().values(),
        expected.values()
    );
    owner.clear_diagnostic_trace();
    let result = ready(owner.linear(&linear, &input, None).unwrap());
    assert_eq!(count.get(), 2);
    assert_eq!(
        owner.download_rows(result).unwrap().values(),
        expected.values()
    );
    owner.set_diagnostic_trace(|e| {
        if e.name == "linear.output" {
            return Err(ferrule_common::Error::Model {
                message: "diagnostic observer failure".into(),
            });
        }
        Ok(())
    });
    assert!(
        owner
            .linear(&linear, &input, None)
            .unwrap_err()
            .to_string()
            .contains("diagnostic observer failure")
    );
    assert!(!owner.needs_quarantine());
    owner.clear_diagnostic_trace();
    let result = ready(owner.linear(&linear, &input, None).unwrap());
    assert_eq!(
        owner.download_rows(result).unwrap().values(),
        expected.values()
    );
}

#[test]
#[ignore = "requires CUDA and FERRULE_HYBRID_TRACE_DIR containing a full40 trace; isolates one real QKV block"]
fn numeric_qkv_same_operands_rounding_and_accumulation_probe() {
    use ferrule_backend::cuda::operators::linear::*;
    use ferrule_backend::cuda::providers::DeviceBuffer;
    let dir = PathBuf::from(std::env::var_os("FERRULE_HYBRID_TRACE_DIR").expect("trace directory"));
    let events: Vec<Value> = std::fs::read_to_string(dir.join("events.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let event = |name| {
        events
            .iter()
            .find(|v| {
                v["stage"] == "prefill"
                    && v["layer"] == 0
                    && v["role"] == "LinearAttentionQkv"
                    && v["event"] == name
            })
            .unwrap()
    };
    let input = event("linear.input");
    let original = event("linear.output");
    let read = |path: &Path| {
        std::fs::read(path)
            .unwrap()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>()
    };
    let x = read(&dir.join(input["file"].as_str().unwrap()));
    let k = x.len();
    let start = 5632usize;
    let n = 128usize;
    let meta = &input["weight"];
    let weight = CheckpointTensorSlice {
        name: "probe.weight".into(),
        role: TensorRole::LinearAttentionQkv,
        path: meta["path"].as_str().unwrap().into(),
        offset: meta["offset"].as_u64().unwrap() + (start * k) as u64,
        bytes: (n * k) as u64,
        dtype: CheckpointDType::F8E4M3,
        shape: vec![n, k],
    };
    let meta = &input["scale"];
    let scale = CheckpointTensorSlice {
        name: "probe.weight_scale_inv".into(),
        role: weight.role.clone(),
        path: meta["path"].as_str().unwrap().into(),
        offset: meta["offset"].as_u64().unwrap() + (start / 128 * k.div_ceil(128) * 2) as u64,
        bytes: (k.div_ceil(128) * 2) as u64,
        dtype: CheckpointDType::Bf16,
        shape: vec![1, k.div_ceil(128)],
    };
    let reader = CheckpointTensorReader::new(2 * 1024 * 1024);
    let raw = reader.read_slice(&weight).unwrap().bytes;
    let scales = reader.read_slice(&scale).unwrap().bytes;
    let w: Vec<f32> = raw
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let s = i % k / 128 * 2;
            let scale =
                half::bf16::from_bits(u16::from_le_bytes([scales[s], scales[s + 1]])).to_f32();
            half::bf16::from_f32(ferrule_model::checkpoint::decode_fp8_e4m3fn_byte(v) * scale)
                .to_f32()
        })
        .collect();
    let xb: Vec<u16> = x
        .iter()
        .map(|&v| half::bf16::from_f32(v).to_bits())
        .collect();
    let wb: Vec<u16> = w
        .iter()
        .map(|&v| half::bf16::from_f32(v).to_bits())
        .collect();
    let ops = CudaOperators::new_on_device(0).unwrap();
    let stream = ops.stream_clone();
    let layout = NumericFp8Layout {
        n,
        k,
        row_origin: 0,
        column_origin: 0,
        scale_type: NumericFp8ScaleType::Bf16,
    };
    let artifact = ops
        .upload_numeric_fp8_linear(layout, &raw, &scales)
        .unwrap();
    let launch = |a: &[f32], m: usize, artifact: &CudaNumericFp8Artifact| {
        let l = artifact.layout();
        let plan = NumericFp8LinearPlan::new(
            l,
            m,
            64 * 1024 * 1024,
            NumericFp8Precision::Bf16RneF32Accumulate,
        )
        .unwrap();
        let mut scratch = ops.numeric_fp8_linear_workspace(plan).unwrap();
        let a = ops.upload_f32_buffer(a).unwrap();
        let mut out = ops.zero_f32_buffer(m * l.n).unwrap();
        ops.numeric_fp8_linear_into(artifact, &a, &mut out, &mut scratch, plan, l.k, l.n)
            .unwrap();
        ops.record_compute_event().unwrap().synchronize().unwrap();
        ops.download_f32_buffer(&out).unwrap()
    };
    let numeric = launch(&x, 1, &artifact);
    // Inspect packed operands directly, including signed zero bits. Identity
    // GEMM alone cannot distinguish the sign of a zero weight after summation.
    let plan = NumericFp8LinearPlan::new(
        layout,
        1,
        64 * 1024 * 1024,
        NumericFp8Precision::Bf16RneF32Accumulate,
    )
    .unwrap();
    assert_eq!(plan.tile_rows(), n);
    let bytes = plan.workspace_requirements().bytes as usize;
    let root = DeviceBuffer::<u8>::zeroed(&stream, bytes).unwrap();
    let mut scratch = CudaNumericFp8Workspace::from_buffer(root.slice(0, bytes).unwrap());
    let a = DeviceBuffer::from_host(&stream, &x).unwrap();
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, n).unwrap();
    numeric_fp8_linear(&stream, &artifact, &a, &mut out, &mut scratch, plan, k, n).unwrap();
    ops.record_compute_event().unwrap().synchronize().unwrap();
    let packed = root.to_host_vec(&stream).unwrap();
    let packed: Vec<u16> = packed
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(&packed[..k], &xb, "actual GPU activation BF16 bits");
    assert_eq!(
        &packed[k..k + n * k],
        &wb,
        "actual GPU outlier-block BF16 weight bits"
    );
    eprintln!(
        "packed operands bitwise verified: activation={}, weights={}, full projection rows={}..{}",
        k,
        n * k,
        start,
        start + n
    );
    let mut identity = vec![0.0; k * k];
    for i in 0..k {
        identity[i * k + i] = 1.0;
    }
    let decoded = launch(&identity, k, &artifact);
    for row in 0..n {
        for col in 0..k {
            assert_eq!(
                decoded[col * n + row],
                w[row * k + col],
                "real outlier block dequant {row},{col}"
            );
        }
    }
    let idraw: Vec<u8> = identity
        .iter()
        .map(|&v| if v == 0.0 { 0 } else { 0x38 })
        .collect();
    let idscales: Vec<u8> = (0..k.div_ceil(128).pow(2))
        .flat_map(|_| 0x3f80u16.to_le_bytes())
        .collect();
    let id = ops
        .upload_numeric_fp8_linear(NumericFp8Layout { n: k, ..layout }, &idraw, &idscales)
        .unwrap();
    let rounded = launch(&x, 1, &id);
    for (&a, &b) in rounded.iter().zip(&xb) {
        assert_eq!(a, half::bf16::from_bits(b).to_f32(), "activation RNE");
    }
    let a = DeviceBuffer::from_host(&stream, &xb).unwrap();
    let b = DeviceBuffer::from_host(&stream, &wb).unwrap();
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, n).unwrap();
    bf16_gemm(
        &stream,
        &a,
        &b,
        &mut out,
        Bf16GemmLayout::contiguous(1, n, k),
    )
    .unwrap();
    let direct = out.to_host_vec(&stream).unwrap();
    assert_eq!(
        numeric, direct,
        "numeric decode+GEMM vs independently packed BF16 GEMM"
    );
    let original = read(&dir.join(original["file"].as_str().unwrap()));
    assert_eq!(
        numeric,
        original[start..start + n],
        "isolated rows reproduce full runner projection"
    );
    let f64sum: Vec<f32> = (0..n)
        .map(|r| {
            (0..k)
                .map(|c| f64::from(rounded[c]) * f64::from(w[r * k + c]))
                .sum::<f64>() as f32
        })
        .collect();
    let save = |name: &str, values: &[f32]| {
        std::fs::write(
            dir.join(format!("probe-{name}.f32")),
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap()
    };
    save("numeric", &numeric);
    save("f64sum", &f64sum);
    save("weights", &w);
    save("activation", &rounded);
    let f32_bytes = w.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
    let f32_handle = ops.upload_f32_linear(&f32_bytes, n, k).unwrap();
    let f32_input = ops.upload_f32_buffer(&rounded).unwrap();
    let mut f32_output = ops.zero_f32_buffer(n).unwrap();
    ops.linear_f32_into(&f32_handle, &f32_input, 1, &mut f32_output)
        .unwrap();
    ops.record_compute_event().unwrap().synchronize().unwrap();
    let f32_output = ops.download_f32_buffer(&f32_output).unwrap();
    save("tf32x3-same-bf16-operands", &f32_output);
    report_diff(
        &f32_output,
        &f64sum,
        "diagnostic TF32x3 SAME BF16 operands vs F64",
        0.0,
        0.0,
    );
    report_diff(
        &numeric,
        &f64sum,
        "QKV BF16 TensorOp vs F64 product-sum",
        0.0,
        0.0,
    );
    for chunk in [16, 32, 64, 128, 256, 512] {
        let mut sum = vec![0f64; n];
        for begin in (0..k).step_by(chunk) {
            let a = DeviceBuffer::from_host(&stream, &xb[begin..begin + chunk]).unwrap();
            let chunk_w: Vec<u16> = (0..n)
                .flat_map(|r| wb[r * k + begin..r * k + begin + chunk].iter().copied())
                .collect();
            let b = DeviceBuffer::from_host(&stream, &chunk_w).unwrap();
            let mut out = DeviceBuffer::<f32>::zeroed(&stream, n).unwrap();
            bf16_gemm(
                &stream,
                &a,
                &b,
                &mut out,
                Bf16GemmLayout::contiguous(1, n, chunk),
            )
            .unwrap();
            for (sum, v) in sum.iter_mut().zip(out.to_host_vec(&stream).unwrap()) {
                *sum += f64::from(v);
            }
        }
        let sum: Vec<f32> = sum.into_iter().map(|v| v as f32).collect();
        report_diff(
            &sum,
            &f64sum,
            &format!("diagnostic K{chunk} partials vs F64"),
            0.0,
            0.0,
        );
        save(&format!("k{chunk}"), &sum);
    }
}

fn report_diff(actual: &[f32], expected: &[f32], label: &str, atol: f64, rtol: f64) -> bool {
    assert_eq!(actual.len(), expected.len(), "{label} shape");
    let mut max_abs = 0.0f64;
    let mut squared = 0.0;
    let mut failures = 0;
    for (&a, &e) in actual.iter().zip(expected) {
        let diff = (f64::from(a) - f64::from(e)).abs();
        max_abs = max_abs.max(diff);
        squared += diff * diff;
        if !within_tolerance(a, e, atol, rtol) {
            failures += 1;
        }
    }
    eprintln!(
        "{label}: n={}, max_abs={max_abs:.9e}, rms={:.9e}, outside({atol}+{rtol}*abs(ref))={failures}",
        actual.len(),
        (squared / actual.len() as f64).sqrt()
    );
    failures == 0
}
