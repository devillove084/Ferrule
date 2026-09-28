use super::*;
use crate::{GenerateRequest, InferenceShutdownProgress, RequestId, ResidentDriverStep, SessionId};
use ferrule_model::TensorRole;
use ferrule_model::checkpoint::{CheckpointDType, CheckpointTensorSlice};
use ferrule_model::nn::{ModulePath, ParameterDType, ParameterId, ParameterSpec};
use ferrule_model::transformer::*;
use serde_json::Value;

struct Fixture {
    dir: std::path::PathBuf,
    resources: BoundDecoderResources,
}
impl Fixture {
    fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "qwen35-ep-fixture-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let oracle: Value = serde_json::from_str(include_str!(
            "../../../../ferrule-model/tests/fixtures/numeric_standard_f32_oracle.json"
        ))
        .unwrap();
        let mut tensors = oracle["tensors"].as_object().unwrap().clone();
        // Four experts makes a legal EP2 shape; the fourth has the same immutable
        // weights as expert two. Both local and EP consume this exact image.
        for layer in 0..2 {
            for part in ["gate", "up", "down"] {
                let from = format!("layers.{layer}.feed_forward.experts.2.{part}.weight");
                tensors.insert(
                    from.replace(".experts.2.", ".experts.3."),
                    tensors[&from].clone(),
                );
            }
            let router = tensors
                .get_mut(&format!("layers.{layer}.feed_forward.router.weight"))
                .unwrap();
            router["shape"][0] = serde_json::json!(4);
            let values = router["raw"].as_array_mut().unwrap();
            let row = values[16..24].to_vec();
            values.extend(row);
        }
        let mut schema = StateDictSchema::builder();
        let mut mapper = ExactNameMapper::new();
        let mut slices = Vec::new();
        let mut payload = Vec::new();
        for (index, (name, tensor)) in tensors.iter().enumerate() {
            let numeric = tensor["dtype"] == "F8_E4M3";
            let shape: Vec<usize> = serde_json::from_value(tensor["shape"].clone()).unwrap();
            let start = payload.len();
            if numeric {
                payload.extend(serde_json::from_value::<Vec<u8>>(tensor["raw"].clone()).unwrap());
            } else {
                for v in serde_json::from_value::<Vec<f32>>(tensor["values"].clone()).unwrap() {
                    payload.extend(v.to_le_bytes());
                }
            }
            let role = role(name);
            slices.push(CheckpointTensorSlice {
                name: name.clone(),
                role: role.clone(),
                path: dir.join("weights.bin"),
                offset: start as u64,
                bytes: (payload.len() - start) as u64,
                dtype: if numeric {
                    CheckpointDType::F8E4M3
                } else {
                    CheckpointDType::F32
                },
                shape: shape.clone(),
            });
            let path = ModulePath::new(name).unwrap();
            let parts = name.split('.').collect::<Vec<_>>();
            let residency = if name.contains(".experts.") {
                ParameterResidency::expert(parts[1].parse().unwrap(), parts[4].parse().unwrap())
            } else if name.starts_with("layers.") {
                ParameterResidency::layer(parts[1].parse().unwrap())
            } else {
                ParameterResidency::Static
            };
            let mut parameter = ParameterSpec::new(
                ParameterId::new(index as u64 + 1),
                path.clone(),
                if numeric {
                    ParameterDType::F8E4M3
                } else {
                    ParameterDType::F32
                },
                shape.clone(),
                residency,
            )
            .unwrap();
            if numeric {
                let start = payload.len();
                for scale in serde_json::from_value::<Vec<u16>>(tensor["scales"].clone()).unwrap() {
                    payload.extend(scale.to_le_bytes());
                }
                let scale_name = name.replace(".weight", ".weight_scale_inv");
                let scale_shape = shape.iter().map(|n| n.div_ceil(128)).collect::<Vec<_>>();
                slices.push(CheckpointTensorSlice {
                    name: scale_name.clone(),
                    role: role.clone(),
                    path: dir.join("weights.bin"),
                    offset: start as u64,
                    bytes: (payload.len() - start) as u64,
                    dtype: CheckpointDType::Bf16,
                    shape: scale_shape.clone(),
                });
                parameter = parameter
                    .with_required_scale(ParameterDType::Bf16, scale_shape)
                    .unwrap();
                mapper
                    .insert(scale_name, NameMapping::scale(path.clone()))
                    .unwrap();
            }
            mapper.insert(name, NameMapping::weight(path)).unwrap();
            schema.register_with_role(parameter, role).unwrap();
        }
        std::fs::write(dir.join("weights.bin"), payload).unwrap();
        std::fs::write(dir.join("tokenizer.json"), serde_json::json!({
            "version":"1.0", "truncation":null, "padding":null, "added_tokens":[],
            "normalizer":null, "pre_tokenizer":{"type":"WhitespaceSplit"}, "post_processor":null, "decoder":null,
            "model":{"type":"WordLevel", "vocab":{"t0":0,"t1":1,"t2":2,"t3":3,"t4":4,"t5":5,"t6":6,"t7":7,"t8":8,"t9":9,"t10":10},"unk_token":"t0"}
        }).to_string()).unwrap();
        let bound = StateDictBinder::new(&schema.build().unwrap(), &mapper)
            .bind_slices(slices)
            .unwrap();
        let resources = BoundDecoderResources::new(spec(), Arc::new(bound)).unwrap();
        Self { dir, resources }
    }
    fn request(&self) -> ResolvedModelRequest {
        let mut descriptor =
            super::super::super::tests::descriptor(ModelFamily::Qwen35Moe, "ep-tiny", 2);
        descriptor.path = self.dir.clone();
        let mut options = super::super::super::tests::options();
        options.max_tensor_mebibytes = 1024;
        options.driver_config.ctx_size = 32;
        options.driver_config.enable_native_proposals = false;
        options.scheduler_config.max_active_sequences = 2;
        options.scheduler_config.max_batch_tokens = 4;
        options.scheduler_config.prefill_chunk_size = 2;
        let entry = resolve(&descriptor, BackendSelection::Cuda).unwrap();
        let mut request = configure_request(&descriptor, entry, options).unwrap();
        request
            .family_options
            .qwen35_mut()
            .unwrap()
            .set_expert_placement(Qwen35ExpertPlacement::new(2, vec![0, 1]).unwrap());
        request
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn role(name: &str) -> TensorRole {
    use TensorRole::*;
    match name {
        "token_embedding.weight" => return TokenEmbedding,
        "final_norm.weight" => return OutputNorm,
        "output.weight" => return OutputHead,
        _ => {}
    }
    let suffix = name.split('.').skip(2).collect::<Vec<_>>().join(".");
    if suffix.starts_with("feed_forward.experts.") {
        return match suffix.split('.').nth(3).unwrap() {
            "gate" => RoutedExpertGate,
            "up" => RoutedExpertUp,
            "down" => RoutedExpertDown,
            _ => panic!(),
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
        _ => panic!("{name}"),
    }
}
fn spec() -> DecoderModelSpec {
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
                4,
                2,
                RouterScoreFunction::Softmax,
                RouterSelection::TopK,
                true,
                1.0,
            )
            .unwrap();
            let moe = Moe::new(8, 12, router, false)
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
        architecture: "ep-tiny".into(),
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

#[test]
fn ownership_metadata_has_disjoint_root_and_balanced_exact_payloads() {
    let f = Fixture::new();
    let p = OwnerPlan::new(
        &f.resources,
        &Qwen35ExpertPlacement::new(2, vec![0, 1]).unwrap(),
        4,
    )
    .unwrap();
    assert!(!p.group.members.contains(&p.group.source_rank));
    assert_eq!(p.group.source_rank, ParallelRankId::new(0));
    assert_eq!(
        p.group.members,
        vec![ParallelRankId::new(1), ParallelRankId::new(2)]
    );
    assert_eq!(p.counts, [4, 4]);
    assert_eq!(p.weights, [4 * 294, 4 * 294]);
    assert_eq!(p.group.limits.max_tokens, 8);
    assert_eq!(p.activation_bytes, 8 * (12 * 3 + 8 * 2) * 4);
}

fn local(f: &Fixture, request: &ResolvedModelRequest) -> BoxedSessionInferenceEngine {
    let options = GenericDecoderOptions::standard_cpu(
        f.resources.spec(),
        ModelFamily::Qwen35Moe,
        WeightSource::Safetensors,
        16,
        32,
        4,
        2,
        1 << 20,
        ExecutionPrecisionPolicy::f32(),
    )
    .unwrap()
    .with_hybrid_cuda_numeric_fp8_precision(
        ExpertCacheLimits {
            max_experts: 8,
            max_bytes: 4 << 20,
        },
        64 << 10,
        NumericFp8Precision::F32Tf32x3,
    )
    .unwrap();
    let planes = HybridStateSchema::from_spec(f.resources.spec(), 2)
        .unwrap()
        .kv_planes(16, 32)
        .unwrap();
    let accounting = ResidentKvPageAccounting::TransactionCapacity {
        page_bytes: physical_page_bytes(&planes).unwrap(),
        budget_bytes: Some(1 << 20),
    };
    let pages = plan_resident_kv_pages(
        &planes,
        accounting,
        request.scheduler_config,
        request.driver_config,
    )
    .unwrap();
    let estimate = ferrule_model::decoder::HybridCudaMemoryEstimate::for_resources(
        &f.resources,
        &options,
        pages.configured_pages,
    )
    .unwrap();
    let device = HybridCudaDevice::new_on_device(
        0,
        HybridCudaMemoryBudget {
            kv_pages: pages.configured_pages,
            state_bytes: estimate.per_sequence_state_bytes * 5,
            weight_bytes: estimate.weight_bytes_upper_bound,
            workspace_bytes: estimate.workspace_bytes_upper_bound,
        },
    )
    .unwrap();
    let runner = GenericDecoderRunner::<HybridCudaDecoder>::hybrid_cuda(
        f.resources.clone(),
        TokenizerHandle::load(&f.dir).unwrap(),
        options,
        device,
    )
    .unwrap();
    Box::new(
        super::super::super::super::composition::compose_resident_engine(
            runner,
            Box::new(planes),
            accounting,
            request.scheduler_config,
            request.driver_config,
        )
        .unwrap(),
    )
}
fn generate(engine: &mut BoxedSessionInferenceEngine, id: u64, prompt: Vec<u32>) -> Vec<u32> {
    engine
        .try_submit(GenerateRequest {
            id: RequestId(id),
            session_id: Some(SessionId(1)),
            prompt_tokens: prompt,
            max_new_tokens: 3,
            stop: vec![],
            ignore_eos: true,
        })
        .unwrap();
    let mut tokens = Vec::new();
    for _ in 0..100 {
        let step = engine
            .step(&mut |token| {
                tokens.push(token.token);
                Ok(())
            })
            .unwrap();
        if step == ResidentDriverStep::Idle {
            assert_eq!(tokens.len(), 3);
            return tokens;
        }
    }
    panic!("bounded tiny request did not finish");
}
#[test]
#[ignore = "requires two CUDA GPUs; real eager EP2 full hybrid factory versus local numeric F32"]
fn tiny_ep2_full_hybrid_matches_local_and_lifecycle() {
    ferrule_common::observability::init_tracing();
    let f = Fixture::new();
    let request = f.request();
    let mut reference = local(&f, &request);
    let mut parallel = build_bound(
        request,
        f.resources.clone(),
        TokenizerHandle::load(&f.dir).unwrap(),
    )
    .unwrap();
    for engine in [&mut reference, &mut parallel] {
        engine.retain_session(SessionId(1)).unwrap();
    }
    for (id, prompt) in [(1, vec![1, 3, 2]), (2, vec![4])] {
        let a = generate(&mut reference, id, prompt.clone());
        let b = generate(&mut parallel, id, prompt);
        eprintln!("EP2 request={id} local={a:?} parallel={b:?}");
        assert_eq!(a, b);
        assert_eq!(
            reference.retained_session_position(SessionId(1)),
            parallel.retained_session_position(SessionId(1))
        );
    }
    for engine in [&mut reference, &mut parallel] {
        engine.reset_session(SessionId(1)).unwrap();
        assert_eq!(engine.retained_session_position(SessionId(1)), Some(0));
        engine
            .try_submit(GenerateRequest {
                id: RequestId(3),
                session_id: Some(SessionId(1)),
                prompt_tokens: vec![1, 3, 2],
                max_new_tokens: 16,
                stop: vec![],
                ignore_eos: true,
            })
            .unwrap();
        engine.step(&mut |_| Ok(())).unwrap();
        engine.cancel_request(RequestId(3)).unwrap();
        assert_eq!(engine.drain_cancelled().len(), 1);
        engine.reset_session(SessionId(1)).unwrap();
    }
    assert_eq!(
        generate(&mut reference, 4, vec![1, 3, 2]),
        generate(&mut parallel, 4, vec![1, 3, 2])
    );
    for engine in [&mut reference, &mut parallel] {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if engine.shutdown().unwrap() == InferenceShutdownProgress::Complete {
                break;
            }
            assert!(Instant::now() < deadline, "shutdown deadline");
        }
    }
}
