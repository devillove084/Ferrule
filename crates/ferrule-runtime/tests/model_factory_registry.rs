use std::path::PathBuf;

use ferrule_model::{
    AttentionKind, ChatTemplate, ModelDescriptor, ModelExecutionBackend, ModelFamily, MoeSpec,
    RouterKind, TransformerSemantics, TransformerSpec, WeightSource,
};
use ferrule_runtime::{BackendSelection, BuiltinModelResolver};

fn descriptor(family: ModelFamily, architecture: &str) -> ModelDescriptor {
    ModelDescriptor {
        path: PathBuf::from("model"),
        spec: TransformerSpec {
            family,
            architecture: Some(architecture.into()),
            weight_source: WeightSource::Safetensors,
            hidden_size: Some(64),
            num_layers: Some(48),
            vocab_size: Some(128),
            num_heads: Some(4),
            num_kv_heads: Some(4),
            head_dim: Some(16),
            attention: AttentionKind::MultiLatentAttention,
            moe: MoeSpec::none(),
            semantics: TransformerSemantics::default(),
            tensor_count: None,
            quantization: Vec::new(),
            notes: Vec::new(),
        },
        tensor_classes: Vec::new(),
    }
}

#[test]
fn owner_engine_has_no_send_or_leak_escape_hatch() {
    let source = include_str!("../src/engine/inference.rs");

    assert!(source.contains("pub trait InferenceEngine: 'static"));
    assert!(!source.contains("OwnerLocal"));
    assert!(!source.contains("unsafe impl"));
    assert!(!source.contains("mem::forget"));
}

fn qwen_descriptor() -> ModelDescriptor {
    let mut descriptor = descriptor(ModelFamily::QwenMoe, "qwen3_moe");
    descriptor.spec.attention = AttentionKind::GroupedQuery;
    descriptor.spec.num_kv_heads = Some(1);
    descriptor.spec.moe = MoeSpec {
        num_experts: Some(128),
        num_experts_per_tok: Some(8),
        has_shared_experts: false,
        router: RouterKind::DenseTopK,
    };
    descriptor
}

#[test]
fn qwen_auto_registers_cpu_backend() {
    let entry = BuiltinModelResolver::new()
        .resolve(&qwen_descriptor(), BackendSelection::Auto)
        .unwrap();

    assert_eq!(entry.model_name(), "qwen3-moe");
    assert_eq!(entry.backend(), ModelExecutionBackend::Cpu);
    assert_eq!(entry.backend_profile(), "cpu-standard-decoder");
    assert_eq!(entry.default_chat_template(), ChatTemplate::Qwen3);
}

#[cfg(not(feature = "cuda"))]
#[test]
fn deepseek_auto_is_known_unavailable_without_cuda() {
    let error = BuiltinModelResolver::new()
        .resolve(
            &descriptor(ModelFamily::DeepSeekV4, "deepseek4"),
            BackendSelection::Auto,
        )
        .unwrap_err();

    assert!(error.to_string().contains("requires CUDA"));
    assert!(
        error
            .to_string()
            .contains("compiled without the 'cuda' feature")
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_build_registers_deepseek_auto_backend() {
    let entry = BuiltinModelResolver::new()
        .resolve(
            &descriptor(ModelFamily::DeepSeekV4, "deepseek4"),
            BackendSelection::Auto,
        )
        .unwrap();

    assert_eq!(entry.model_name(), "deepseek-v4");
    assert_eq!(entry.backend(), ModelExecutionBackend::Cuda);
    assert_eq!(entry.backend_profile(), "cuda");
    assert_eq!(entry.default_chat_template(), ChatTemplate::DeepSeekV4);
}
