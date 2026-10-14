use crate::model_factory::*;
use ferrule_model::TensorRole;
use ferrule_model::checkpoint::{CheckpointDType, CheckpointTensorSlice};
use ferrule_model::nn::ParameterResidency;
use ferrule_model::nn::{ModulePath, ParameterDType, ParameterId, ParameterSpec};
use ferrule_model::transformer::*;
use serde_json::Value;
use std::sync::Arc;

pub(in crate::model_factory) struct Fixture {
    pub(in crate::model_factory) dir: std::path::PathBuf,
    pub(in crate::model_factory) resources: BoundDecoderResources,
}
impl Fixture {
    pub(in crate::model_factory) fn new() -> Self {
        Self::with_layers(4)
    }
    pub(in crate::model_factory) fn with_layers(layers: usize) -> Self {
        assert_eq!(layers, 4);
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "qwen35-tp-ep-fixture-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let oracle: Value = serde_json::from_str(include_str!(
            "../../../../../ferrule-model/tests/fixtures/numeric_standard_f32_oracle.json"
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
        if layers == 4 {
            let extra = tensors
                .iter()
                .filter(|(name, _)| name.starts_with("layers."))
                .map(|(name, value)| {
                    let index = name.split('.').nth(1).unwrap().parse::<usize>().unwrap();
                    (
                        name.replacen(
                            &format!("layers.{index}."),
                            &format!("layers.{}.", index + 2),
                            1,
                        ),
                        value.clone(),
                    )
                })
                .collect::<Vec<_>>();
            tensors.extend(extra);
        }
        // Four local key groups allow TP1/2/4. Repeat Q/K/V blocks separately:
        // concatenating whole QKV matrices would change the global head identity.
        for layer in [0, 2] {
            let prefix = format!("layers.{layer}.attention");
            for part in ["qkv", "conv"] {
                expand(
                    tensors.get_mut(&format!("{prefix}.{part}.weight")).unwrap(),
                    &[3, 3, 4],
                    4,
                    false,
                );
            }
            for part in ["z", "beta", "a", "a_log", "dt_bias"] {
                let t = tensors.get_mut(&format!("{prefix}.{part}.weight")).unwrap();
                let n = t["shape"][0].as_u64().unwrap() as usize;
                expand(t, &[n], 4, false);
            }
            expand(
                tensors.get_mut(&format!("{prefix}.output.weight")).unwrap(),
                &[4],
                4,
                true,
            );
        }
        for layer in [1, 3] {
            let prefix = format!("layers.{layer}.attention");
            expand(
                tensors.get_mut(&format!("{prefix}.query.weight")).unwrap(),
                &[32],
                2,
                false,
            );
            expand(
                tensors.get_mut(&format!("{prefix}.output.weight")).unwrap(),
                &[16],
                2,
                true,
            );
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
        let resources =
            BoundDecoderResources::new(spec_with_layers(layers), Arc::new(bound)).unwrap();
        Self { dir, resources }
    }
    pub(in crate::model_factory) fn request(&self) -> ResolvedModelRequest {
        let mut descriptor =
            crate::model_factory::tests::descriptor(ModelFamily::Qwen35Moe, "ep-tiny", 2);
        descriptor.path = self.dir.clone();
        let mut options = crate::model_factory::tests::options();
        options.max_tensor_mebibytes = 1024;
        options.driver_config.ctx_size = 32;
        options.driver_config.enable_native_proposals = false;
        options.scheduler_config.max_active_sequences = 2;
        options.scheduler_config.max_batch_tokens = 4;
        options.scheduler_config.prefill_chunk_size = 2;
        let entry = ResolvedModelBackend {
            family: ModelFamily::Qwen35Moe,
            model_name: "ep-tiny",
            backend: ModelExecutionBackend::Cuda,
            backend_profile: Qwen35MoeCapacityPlan::PLANNED_BACKEND_PROFILE,
            default_chat_template: ChatTemplate::Qwen35,
        };
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
fn spec_with_layers(count: usize) -> DecoderModelSpec {
    let norm = |n| RmsNorm::new(n, 1e-6).unwrap().with_one_plus_weight();
    let rope = RotaryEmbedding::new(
        8,
        10000.0,
        RotaryPairing::SplitHalf,
        RotaryRegion::Prefix { dimensions: 4 },
        RotaryScaling::None,
    )
    .unwrap();
    let layers = (0..count)
        .map(|i| {
            let attention = if i % 2 == 0 {
                Attention::GatedDeltaNet(
                    GatedDeltaNetAttention::new(8, 4, 8, 3, 2, 3, 1e-6, false).unwrap(),
                )
            } else {
                Attention::Gqa(
                    GqaAttention::new(8, 4, 1, 8, false, rope.clone())
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

fn expand(t: &mut Value, blocks: &[usize], copies: usize, columns: bool) {
    let key = if t["dtype"] == "F8_E4M3" {
        "raw"
    } else {
        "values"
    };
    let values = t[key].as_array().unwrap().clone();
    let mut out = Vec::new();
    if columns {
        let width = t["shape"][1].as_u64().unwrap() as usize;
        for row in values.chunks_exact(width) {
            for _ in 0..copies {
                out.extend_from_slice(row);
            }
        }
        t["shape"][1] = serde_json::json!(width * copies);
    } else {
        let rows = t["shape"][0].as_u64().unwrap() as usize;
        let width = values.len() / rows;
        assert_eq!(blocks.iter().sum::<usize>(), rows);
        let mut start = 0;
        for &block in blocks {
            for _ in 0..copies {
                out.extend_from_slice(&values[start..start + block * width]);
            }
            start += block * width;
        }
        t["shape"][0] = serde_json::json!(rows * copies);
    }
    t[key] = Value::Array(out);
    // Tiny rectangles remain inside the original single 128x128 scale block.
    if key == "raw" {
        assert!(
            t["shape"]
                .as_array()
                .unwrap()
                .iter()
                .all(|n| n.as_u64().unwrap() <= 128)
        );
    }
}
