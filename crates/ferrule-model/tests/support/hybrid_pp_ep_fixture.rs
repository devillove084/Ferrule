//! Deterministic MoE extension of the existing hybrid attention fixture.
//! The dense fixture's golden logits do NOT apply: tests use an unsplit MoE
//! segment as the arithmetic oracle. No model forward is copied here.
use ferrule_model::nn::{
    DTypeConstraint, ModulePath, ParameterDType, ParameterId, ParameterResidency, ParameterSpec,
};
use ferrule_model::transformer::*;
use ferrule_model::{ModelFamily, TensorRole};
use serde_json::Value;
use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
fn floats(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}
fn spec(context: usize) -> DecoderModelSpec {
    let norm = |width| RmsNorm::new(width, 1e-6).unwrap().with_one_plus_weight();
    let rope = RotaryEmbedding::new(
        8,
        10000.0,
        RotaryPairing::SplitHalf,
        RotaryRegion::Prefix { dimensions: 4 },
        RotaryScaling::None,
    )
    .unwrap();
    let layers = (0..4)
        .map(|layer| {
            let attention = if layer % 2 == 0 {
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
            DecoderLayer::new(
                layer,
                norm(8),
                attention,
                Residual::Add,
                norm(8),
                FeedForward::Moe(
                    Moe::new(
                        8,
                        12,
                        MoeRouterSpec::new(
                            2,
                            2,
                            RouterScoreFunction::Softmax,
                            RouterSelection::TopK,
                            true,
                            1.0,
                        )
                        .unwrap(),
                        false,
                    )
                    .unwrap()
                    .with_shared_expert(SwiGlu::new(8, 12, false).unwrap())
                    .unwrap()
                    .with_shared_expert_gate(Linear::new(8, 1, false).unwrap())
                    .unwrap(),
                ),
                Residual::Add,
            )
            .unwrap()
        })
        .collect();
    DecoderModelSpec::new(DecoderModelParts {
        architecture: "synthetic-hybrid".into(),
        hidden_size: 8,
        vocab_size: 11,
        max_sequence_length: Some(context),
        token_embedding: Embedding::new(11, 8, None).unwrap(),
        layers,
        final_norm: norm(8),
        output: Linear::new(8, 11, false).unwrap(),
        tie_word_embeddings: false,
    })
    .unwrap()
}
fn role(name: &str) -> TensorRole {
    match name {
        "token_embedding.weight" => return TensorRole::TokenEmbedding,
        "final_norm.weight" => return TensorRole::OutputNorm,
        "output.weight" => return TensorRole::OutputHead,
        _ => (),
    }
    let suffix = name.split('.').skip(2).collect::<Vec<_>>().join(".");
    if suffix.starts_with("feed_forward.experts.") {
        return match name.split('.').nth(5).unwrap() {
            "gate" => TensorRole::RoutedExpertGate,
            "up" => TensorRole::RoutedExpertUp,
            "down" => TensorRole::RoutedExpertDown,
            _ => panic!("unknown expert matrix"),
        };
    }
    match suffix.as_str() {
        "input_norm.weight" => TensorRole::AttentionNorm,
        "post_attention_norm.weight" => TensorRole::FeedForwardNorm,
        "attention.query.weight" => TensorRole::AttentionQuery,
        "attention.key.weight" => TensorRole::AttentionKey,
        "attention.value.weight" => TensorRole::AttentionValue,
        "attention.output.weight" => TensorRole::AttentionOutput,
        "attention.query_norm.weight" => TensorRole::AttentionQueryNorm,
        "attention.key_norm.weight" => TensorRole::AttentionKeyNorm,
        "attention.qkv.weight" => TensorRole::LinearAttentionQkv,
        "attention.z.weight" => TensorRole::LinearAttentionZ,
        "attention.beta.weight" => TensorRole::LinearAttentionBeta,
        "attention.a.weight" => TensorRole::LinearAttentionA,
        "attention.conv.weight" => TensorRole::LinearAttentionConv,
        "attention.a_log.weight" => TensorRole::LinearAttentionALog,
        "attention.dt_bias.weight" => TensorRole::LinearAttentionDtBias,
        "attention.norm.weight" => TensorRole::LinearAttentionNorm,
        "feed_forward.shared.gate.weight" => TensorRole::SharedExpertGate,
        "feed_forward.shared.up.weight" => TensorRole::SharedExpertUp,
        "feed_forward.shared.down.weight" => TensorRole::SharedExpertDown,
        "feed_forward.shared_output_gate.weight" => TensorRole::SharedExpertOutputGate,
        "feed_forward.router.weight" => TensorRole::RouterLogits,
        _ => panic!("unexpected fixture parameter {name}"),
    }
}
pub(crate) struct Fixture {
    pub(crate) dir: PathBuf,
    pub(crate) oracle: Value,
}
impl Fixture {
    pub(crate) fn new() -> Self {
        let mut oracle: Value =
            serde_json::from_str(include_str!("../fixtures/hybrid_cpu/oracle.json")).unwrap();
        let tensors = oracle["tensors"].as_object_mut().unwrap();
        for layer in 0..4 {
            let prefix = format!("layers.{layer}.feed_forward");
            for matrix in ["gate", "up", "down"] {
                let base = tensors
                    .remove(&format!("{prefix}.{matrix}.weight"))
                    .unwrap();
                tensors.insert(format!("{prefix}.shared.{matrix}.weight"), base.clone());
                for expert in 0..2 {
                    let values = floats(&base["values"])
                        .into_iter()
                        .enumerate()
                        .map(|(i, x)| {
                            x * (0.6 + expert as f32 * 0.35 + layer as f32 * 0.07)
                                + ((i + expert * 11 + layer * 3) as f32 * 0.13).sin() * 0.015
                        })
                        .collect::<Vec<_>>();
                    tensors.insert(
                        format!("{prefix}.experts.{expert}.{matrix}.weight"),
                        serde_json::json!({"shape": base["shape"], "values": values}),
                    );
                }
            }
            tensors.insert(format!("{prefix}.router.weight"), serde_json::json!({
                "shape": [2, 8], "values": (0..16).map(|i| ((i + layer * 2) as f32 * 0.41).sin() * 0.3).collect::<Vec<_>>()
            }));
            tensors.insert(format!("{prefix}.shared_output_gate.weight"), serde_json::json!({
                "shape": [1, 8], "values": (0..8).map(|i| ((i + layer) as f32 * 0.29).cos() * 0.2).collect::<Vec<_>>()
            }));
        }
        Self::with_oracle(oracle)
    }
    pub(crate) fn with_oracle(oracle: Value) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "ferrule-hybrid-pp-ep-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for (name, tensor) in oracle["tensors"].as_object().unwrap() {
            let start = payload.len();
            for value in floats(&tensor["values"]) {
                payload.extend_from_slice(&value.to_le_bytes());
            }
            header.insert(name.clone(), serde_json::json!({"shape": tensor["shape"], "dtype": "F32", "data_offsets": [start, payload.len()]}));
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(payload);
        std::fs::write(dir.join("model.safetensors"), bytes).unwrap();
        std::fs::write(dir.join("tokenizer.json"), serde_json::json!({
            "version":"1.0", "truncation":null, "padding":null, "added_tokens":[],
            "normalizer":null, "pre_tokenizer":{"type":"WhitespaceSplit"},
            "post_processor":null, "decoder":null,
            "model":{"type":"WordLevel", "vocab":{"t0":0,"t1":1,"t2":2,"t3":3,"t4":4,"t5":5,"t6":6,"t7":7,"t8":8,"t9":9,"t10":10},"unk_token":"t0"}
        }).to_string()).unwrap();
        Self { dir, oracle }
    }
    pub(crate) fn resources(&self) -> BoundDecoderResources {
        self.resources_with_context(32)
    }
    pub(crate) fn resources_with_context(&self, context: usize) -> BoundDecoderResources {
        self.resources_for(spec(context))
    }
    pub(crate) fn resources_for(&self, spec: DecoderModelSpec) -> BoundDecoderResources {
        let mut schema = StateDictSchema::builder();
        let mut mapper = ExactNameMapper::new();
        for (i, (name, tensor)) in self.oracle["tensors"]
            .as_object()
            .unwrap()
            .iter()
            .enumerate()
        {
            let path = ModulePath::new(name).unwrap();
            let residency = if name.contains(".experts.") {
                ParameterResidency::expert(
                    name.split('.').nth(1).unwrap().parse().unwrap(),
                    name.split('.').nth(4).unwrap().parse().unwrap(),
                )
            } else if name.starts_with("layers.") {
                ParameterResidency::layer(name.split('.').nth(1).unwrap().parse().unwrap())
            } else {
                ParameterResidency::Static
            };
            let shape = tensor["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect::<Vec<_>>();
            schema
                .register_with_role(
                    ParameterSpec::new(
                        ParameterId::new(i as u64 + 1),
                        path.clone(),
                        DTypeConstraint::exact(ParameterDType::F32),
                        shape,
                        residency,
                    )
                    .unwrap(),
                    role(name),
                )
                .unwrap();
            mapper.insert(name, NameMapping::weight(path)).unwrap();
        }
        HFDecoderCheckpoint::open(
            &self.dir,
            ModelFamily::Unknown("synthetic-hybrid".into()),
            spec,
            &schema.build().unwrap(),
            &mapper,
        )
        .unwrap()
        .into_resources()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}
