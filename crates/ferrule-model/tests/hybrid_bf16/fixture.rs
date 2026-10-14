use ferrule_model::nn::{
    DTypeConstraint, ModulePath, ParameterDType, ParameterId, ParameterResidency, ParameterSpec,
};
use ferrule_model::transformer::*;
use ferrule_model::{CheckpointDType, CheckpointTensorSlice, TensorRole};
use serde_json::Value;
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

pub fn spec() -> DecoderModelSpec {
    let norm = |n| RmsNorm::new(n, 1e-6).unwrap().with_one_plus_weight();
    let rope = RotaryEmbedding::new(
        8,
        10000.,
        RotaryPairing::SplitHalf,
        RotaryRegion::Prefix { dimensions: 4 },
        RotaryScaling::None,
    )
    .unwrap();
    let layers = (0..2)
        .map(|i| {
            let attention = if i == 0 {
                Attention::GatedDeltaNet(
                    GatedDeltaNetAttention::new(16, 2, 4, 4, 4, 4, 1e-6, false).unwrap(),
                )
            } else {
                Attention::Gqa(
                    GqaAttention::new(16, 2, 1, 8, false, rope.clone())
                        .unwrap()
                        .with_gated_query()
                        .unwrap()
                        .with_qk_norms(norm(8), norm(8))
                        .unwrap(),
                )
            };
            let router = MoeRouterSpec::new(
                4,
                3,
                RouterScoreFunction::Softmax,
                RouterSelection::TopK,
                true,
                1.,
            )
            .unwrap();
            let moe = Moe::new(16, 16, router, false)
                .unwrap()
                .with_shared_expert(SwiGlu::new(16, 16, false).unwrap())
                .unwrap()
                .with_shared_expert_gate(Linear::new(16, 1, false).unwrap())
                .unwrap();
            DecoderLayer::new(
                i,
                norm(16),
                attention,
                Residual::Add,
                norm(16),
                FeedForward::Moe(moe),
                Residual::Add,
            )
            .unwrap()
        })
        .collect();
    DecoderModelSpec::new(DecoderModelParts {
        architecture: "native-hybrid-bf16-fixture".into(),
        hidden_size: 16,
        vocab_size: 32,
        max_sequence_length: Some(32),
        token_embedding: Embedding::new(32, 16, None).unwrap(),
        layers,
        final_norm: norm(16),
        output: Linear::new(16, 32, false).unwrap(),
        tie_word_embeddings: false,
    })
    .unwrap()
}
pub fn values(v: &Value) -> Vec<f32> {
    serde_json::from_value(v.clone()).unwrap()
}
pub struct Fixture {
    pub dir: PathBuf,
    pub resources: BoundDecoderResources,
    pub oracle: Value,
}
impl Fixture {
    pub fn new() -> Self {
        Self::with_dtype(CheckpointDType::Bf16)
    }
    pub fn numeric() -> Self {
        Self::with_oracle(CheckpointDType::Bf16, include_str!("hf_numeric_model.json"))
    }
    pub fn numeric_profile() -> Self {
        let mut fixture = Self::numeric();
        let reference: Value =
            serde_json::from_str(include_str!("hf_numeric_profile.json")).unwrap();
        assert!(reference.get("tensors").is_none());
        for key in ["torch", "transformers", "hf_source_sha256", "profile"] {
            assert_eq!(reference[key], fixture.oracle[key], "profile oracle {key}");
        }
        fixture.oracle["cases"] = reference["cases"].clone();
        fixture.oracle["router_tie_policy"] = reference["router_tie_policy"].clone();
        fixture.oracle["oracle_contract"] = reference["oracle_contract"].clone();
        fixture
    }
    pub fn with_dtype(dtype: CheckpointDType) -> Self {
        Self::with_oracle(dtype, include_str!("hf_model.json"))
    }
    fn with_oracle(dtype: CheckpointDType, oracle: &str) -> Self {
        assert!(matches!(
            dtype,
            CheckpointDType::Bf16 | CheckpointDType::F32
        ));
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "ferrule-hybrid-bf16-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let oracle: Value = serde_json::from_str(oracle).unwrap();
        let mut schema = StateDictSchema::builder();
        let mut mapper = ExactNameMapper::new();
        let mut slices = Vec::new();
        for (id, (name, t)) in oracle["tensors"].as_object().unwrap().iter().enumerate() {
            let path = ModulePath::new(name).unwrap();
            let shape: Vec<usize> = serde_json::from_value(t["shape"].clone()).unwrap();
            let residency = match (t["layer"].as_u64(), t["expert"].as_u64()) {
                (Some(l), Some(e)) => ParameterResidency::expert(l as usize, e as usize),
                (Some(l), None) => ParameterResidency::layer(l as usize),
                _ => ParameterResidency::Static,
            };
            let role = role(t["role"].as_str().unwrap());
            let numeric = t.get("raw_fp8").is_some();
            let dtype = if numeric {
                CheckpointDType::F8E4M3
            } else if t["source_dtype"] == "float32" {
                CheckpointDType::F32
            } else {
                dtype.clone()
            };
            let mut parameter = ParameterSpec::new(
                ParameterId::new(id as u64 + 1),
                path.clone(),
                DTypeConstraint::exact(if numeric {
                    ParameterDType::F8E4M3
                } else if dtype == CheckpointDType::Bf16 {
                    ParameterDType::Bf16
                } else {
                    ParameterDType::F32
                }),
                shape.clone(),
                residency,
            )
            .unwrap();
            if numeric {
                let scale_shape: Vec<usize> =
                    serde_json::from_value(t["scale_shape"].clone()).unwrap();
                parameter = parameter
                    .with_required_scale(ParameterDType::Bf16, scale_shape.clone())
                    .unwrap();
                let words: Vec<i16> = serde_json::from_value(t["scale_words"].clone()).unwrap();
                let bytes: Vec<u8> = words.iter().flat_map(|v| v.to_le_bytes()).collect();
                let file = dir.join(format!("{id}-scale.bin"));
                std::fs::write(&file, &bytes).unwrap();
                let name = format!("{name}_scale");
                mapper
                    .insert(&name, NameMapping::scale(path.clone()))
                    .unwrap();
                slices.push(CheckpointTensorSlice {
                    name,
                    role: TensorRole::Unknown,
                    path: file,
                    offset: 0,
                    bytes: bytes.len() as u64,
                    dtype: CheckpointDType::Bf16,
                    shape: scale_shape,
                });
            }
            schema.register_with_role(parameter, role).unwrap();
            mapper.insert(name, NameMapping::weight(path)).unwrap();
            let bytes = if numeric {
                serde_json::from_value::<Vec<u8>>(t["raw_fp8"].clone()).unwrap()
            } else if dtype == CheckpointDType::Bf16 {
                values(&t["values"])
                    .iter()
                    .flat_map(|&v| half::bf16::from_f32(v).to_bits().to_le_bytes())
                    .collect::<Vec<_>>()
            } else {
                values(t.get("source_values").unwrap_or(&t["values"]))
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>()
            };
            let file = dir.join(format!("{id}.bin"));
            std::fs::write(&file, &bytes).unwrap();
            slices.push(CheckpointTensorSlice {
                name: name.clone(),
                role: TensorRole::Unknown,
                path: file,
                offset: 0,
                bytes: bytes.len() as u64,
                dtype: dtype.clone(),
                shape,
            });
        }
        let schema = schema.build().unwrap();
        let bound = StateDictBinder::new(&schema, &mapper)
            .bind_slices(slices)
            .unwrap();
        let resources = BoundDecoderResources::new(spec(), Arc::new(bound)).unwrap();
        let mut tokenizer = tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default());
        tokenizer
            .add_tokens((0..32).map(|i| tokenizers::AddedToken::from(format!("t{i}"), false)))
            .unwrap();
        tokenizer.save(dir.join("tokenizer.json"), false).unwrap();
        Self {
            dir,
            resources,
            oracle,
        }
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}
fn role(s: &str) -> TensorRole {
    use TensorRole::*;
    match s {
        "TokenEmbedding" => TokenEmbedding,
        "OutputNorm" => OutputNorm,
        "OutputHead" => OutputHead,
        "AttentionNorm" => AttentionNorm,
        "FeedForwardNorm" => FeedForwardNorm,
        "LinearAttentionQkv" => LinearAttentionQkv,
        "LinearAttentionZ" => LinearAttentionZ,
        "LinearAttentionBeta" => LinearAttentionBeta,
        "LinearAttentionA" => LinearAttentionA,
        "LinearAttentionConv" => LinearAttentionConv,
        "LinearAttentionALog" => LinearAttentionALog,
        "LinearAttentionDtBias" => LinearAttentionDtBias,
        "LinearAttentionNorm" => LinearAttentionNorm,
        "AttentionQuery" => AttentionQuery,
        "AttentionKey" => AttentionKey,
        "AttentionValue" => AttentionValue,
        "AttentionOutput" => AttentionOutput,
        "AttentionQueryNorm" => AttentionQueryNorm,
        "AttentionKeyNorm" => AttentionKeyNorm,
        "RouterLogits" => RouterLogits,
        "SharedExpertOutputGate" => SharedExpertOutputGate,
        "SharedExpertGate" => SharedExpertGate,
        "SharedExpertUp" => SharedExpertUp,
        "SharedExpertDown" => SharedExpertDown,
        "RoutedExpertGate" => RoutedExpertGate,
        "RoutedExpertUp" => RoutedExpertUp,
        "RoutedExpertDown" => RoutedExpertDown,
        _ => panic!("unknown role {s}"),
    }
}
