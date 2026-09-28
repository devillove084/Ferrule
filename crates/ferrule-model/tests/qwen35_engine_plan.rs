//! Header-only public capability reporting; never materializes model weights or CUDA state.
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use ferrule_model::models::qwen35::{
    Qwen35Config, Qwen35HfNameMapper, Qwen35Metadata, Qwen35TensorPartitionKind,
};
use ferrule_model::{
    AutoConfig, EnginePlan, EnginePlanStatus, ModelDescriptor, ModelExecutionBackend, ModelFamily,
    ParallelismPlan, PolicyArea, QuantFormatCount, SpeculationMode, TransformerSemantics,
    WeightSource,
};
use serde_json::{Value, json};

const CUDA_PROFILE: &str = "cuda-hybrid-numeric-fp8-f32-tf32x3-qwen35-35b-a3b";
fn config_value() -> Value {
    serde_json::from_str(include_str!("qwen35_35b_fp8_config.json")).unwrap()
}

struct Artifact(PathBuf);
impl Artifact {
    fn new(visual: bool, mtp: bool) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let artifact = Self(std::env::temp_dir().join(format!(
            "ferrule-qwen35-plan-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        )));
        std::fs::create_dir(&artifact.0).unwrap();
        let value = config_value();
        std::fs::write(
            artifact.0.join("config.json"),
            serde_json::to_vec(&value).unwrap(),
        )
        .unwrap();
        let config = Qwen35Config::from_value(&value).unwrap();
        let mapper = Qwen35HfNameMapper::new(&config);
        let mut header = serde_json::Map::new();
        let mut offset = 0;
        for t in mapper.tensors().filter(|t| match t.partition {
            Qwen35TensorPartitionKind::Text => true,
            Qwen35TensorPartitionKind::Visual => visual,
            Qwen35TensorPartitionKind::Mtp => mtp,
        }) {
            header.insert(
                t.external_name.clone(),
                json!({"dtype":t.dtype.as_str(),"shape":t.shape,
                "data_offsets":[offset,offset+t.bytes()]}),
            );
            offset += t.bytes();
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut f = std::fs::File::create(artifact.0.join("model.safetensors")).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&header).unwrap();
        // Sparse extent: only JSON headers occupy disk. No weight payload reads.
        f.set_len(8 + header.len() as u64 + offset).unwrap();
        artifact
    }
    fn descriptor(&self) -> ModelDescriptor {
        AutoConfig::from_pretrained(&self.0)
            .unwrap()
            .into_descriptor()
    }
}
impl Drop for Artifact {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn assert_backend_plans(d: &ModelDescriptor) {
    assert_eq!(d.spec.family, ModelFamily::Qwen35Moe);
    assert!(d.spec.family.is_supported_runtime_family());
    assert_eq!(d.spec.supports_current_runtime(), cfg!(feature = "cuda"));
    let contract = d.support_contract();
    let default = d.engine_plan();
    let cpu = EnginePlan::from_contract_for_backend(&contract, ModelExecutionBackend::Cpu);
    assert_eq!(default, cpu, "default remains CPU, even in a CUDA build");
    assert_eq!(cpu.status, EnginePlanStatus::Unsupported);
    assert_eq!(cpu.backend_profile, None);
    assert!(
        cpu.missing
            .iter()
            .any(|p| p.area == PolicyArea::Backend && p.reason.contains("no supported CPU"))
    );
    assert!(
        !cpu.missing
            .iter()
            .any(|p| p.reason.contains("no executable model-family policy"))
    );

    let cuda = EnginePlan::from_contract_for_backend(&contract, ModelExecutionBackend::Cuda);
    assert_eq!(cuda.is_executable(), cfg!(feature = "cuda"), "{cuda:?}");
    if cfg!(feature = "cuda") {
        assert_eq!(cuda.status, EnginePlanStatus::Executable);
        assert_eq!(cuda.backend_profile, Some(CUDA_PROFILE));
        assert!(cuda.missing.is_empty());
        assert!(cuda.policies.residency.streaming_allowed);
        assert!(!cuda.policies.residency.all_resident_required);
        assert!(!cuda.policies.validation.supports_cpu_reference);
    } else {
        assert_eq!(cuda.status, EnginePlanStatus::Unsupported);
        assert_eq!(cuda.backend_profile, None);
        assert_eq!(cuda.missing.len(), 1, "{cuda:?}");
        assert_eq!(cuda.missing[0].area, PolicyArea::Backend);
        assert!(cuda.missing[0].reason.contains("requires the cuda feature"));
    }
    let notes = d.spec.notes.join("\n");
    for expected in [
        "FP8 storage",
        "numeric BF16 128x128 block scales",
        "F32Tf32x3 compute",
        "not native FP8 compute",
        "bounded expert cache",
        "full 40-layer text-only",
        "vision/MTP execution",
        "TP/PP/EP",
        "process/rank",
        "attachments are not executed",
    ] {
        assert!(
            notes.contains(expected),
            "missing info wording {expected}: {notes}"
        );
    }
    assert!(!notes.contains("no executable/servable profile"));
    assert!(!notes.contains("strict metadata/binding only"));
    assert_eq!(
        notes.contains("not enabled in this build"),
        !cfg!(feature = "cuda")
    );
}

#[test]
fn strict_header_descriptor_reports_default_cpu_and_feature_gated_cuda_plans() {
    // Both attachment subtrees are optional but, if present, strictly complete.
    for visual in [false, true] {
        for mtp in [false, true] {
            let artifact = Artifact::new(visual, mtp);
            assert_backend_plans(&artifact.descriptor());
        }
    }
}

#[test]
fn family_identity_and_nearby_or_unknown_storage_profiles_are_not_admission() {
    let artifact = Artifact::new(false, false);
    let original = artifact.descriptor();
    let mut cases = Vec::new();
    macro_rules! bad {
        ($field:ident, $value:expr) => {{
            let mut s = original.spec.clone();
            s.$field = $value;
            cases.push(s);
        }};
    }
    bad!(architecture, Some("Qwen3_5MoeForCausalLM".into()));
    bad!(
        architecture,
        Some("Qwen3_8MoeForConditionalGeneration".into())
    );
    bad!(architecture, Some("Qwen4ForConditionalGeneration".into()));
    bad!(hidden_size, Some(4096));
    bad!(num_layers, Some(39));
    bad!(num_heads, Some(32));
    bad!(num_kv_heads, Some(4));
    bad!(head_dim, Some(128));
    bad!(vocab_size, Some(151936));
    bad!(weight_source, WeightSource::Gguf);
    bad!(tensor_count, None);
    bad!(tensor_count, Some(62302));
    bad!(semantics, TransformerSemantics::default());
    bad!(quantization, Vec::new());
    bad!(
        quantization,
        vec![QuantFormatCount {
            format: "BF16".into(),
            tensors: 62303
        }]
    );
    for format in ["unknown", "F8_E8M0", "F8_E5M2", "F32"] {
        let mut s = original.spec.clone();
        s.quantization
            .iter_mut()
            .find(|q| q.format == "F8_E4M3")
            .unwrap()
            .format = format.into();
        cases.push(s);
    }
    let mut s = original.spec.clone();
    s.quantization.push(s.quantization[0].clone());
    cases.push(s);
    let mut s = original.spec.clone();
    s.quantization[0].tensors += 1;
    cases.push(s);
    let mut s = original.spec.clone();
    s.moe.num_experts = Some(128);
    cases.push(s);
    let mut s = original.spec.clone();
    s.moe.num_experts_per_tok = Some(4);
    cases.push(s);
    let mut s = original.spec.clone();
    s.moe.has_shared_experts = false;
    cases.push(s);
    for spec in cases {
        assert!(!spec.supports_current_runtime(), "{spec:?}");
        let contract = ferrule_model::ModelSupportContract::from_spec(&spec, &[]);
        for backend in [ModelExecutionBackend::Cpu, ModelExecutionBackend::Cuda] {
            let plan = EnginePlan::from_contract_for_backend(&contract, backend);
            assert_eq!(plan.status, EnginePlanStatus::Unsupported);
            assert_eq!(plan.backend_profile, None);
            assert!(
                plan.missing
                    .iter()
                    .any(|p| p.area == PolicyArea::ModelFamily)
            );
        }
    }
    // Human-readable notes never authorize execution or affect profile matching.
    let mut spec = original.spec.clone();
    spec.notes.clear();
    assert_eq!(spec.supports_current_runtime(), cfg!(feature = "cuda"));
    spec.quantization.clear();
    spec.notes = original.spec.notes;
    assert!(!spec.supports_current_runtime());
}

#[test]
fn strict_profile_does_not_enable_parallel_process_or_other_policy_variants() {
    let artifact = Artifact::new(false, false);
    let d = artifact.descriptor();
    for degrees in [
        [2, 1, 1, 1, 1, 1],
        [1, 2, 1, 1, 1, 1],
        [1, 1, 2, 1, 1, 1],
        [1, 1, 1, 2, 1, 1],
        [1, 1, 1, 1, 2, 1],
        [1, 1, 1, 1, 1, 2],
        [1, 0, 1, 1, 1, 1],
    ] {
        let [
            data_parallel,
            tensor_parallel,
            expert_parallel,
            sequence_parallel,
            context_parallel,
            pipeline_parallel,
        ] = degrees;
        let mut contract = d.support_contract();
        contract.policies.parallelism = ParallelismPlan {
            data_parallel,
            tensor_parallel,
            expert_parallel,
            sequence_parallel,
            context_parallel,
            pipeline_parallel,
        };
        let plan = EnginePlan::from_contract_for_backend(&contract, ModelExecutionBackend::Cuda);
        assert_eq!(plan.status, EnginePlanStatus::Unsupported);
        assert_eq!(plan.backend_profile, None);
        assert!(
            plan.missing
                .iter()
                .any(|p| p.reason.contains("process/rank execution are unsupported"))
        );
    }
    let mut contract = d.support_contract();
    contract.policies.quant.formats.clear();
    let plan = EnginePlan::from_contract_for_backend(&contract, ModelExecutionBackend::Cuda);
    assert_eq!(plan.status, EnginePlanStatus::Unsupported);
    assert_eq!(plan.backend_profile, None);
    assert!(
        plan.missing
            .iter()
            .any(|p| p.area == PolicyArea::Validation)
    );
    let contract = d
        .support_contract()
        .with_speculation_mode(SpeculationMode::MultiTokenPrediction);
    let plan = EnginePlan::from_contract_for_backend(&contract, ModelExecutionBackend::Cuda);
    assert_eq!(plan.status, EnginePlanStatus::Unsupported);
    assert_eq!(plan.backend_profile, None);
    assert!(
        plan.missing
            .iter()
            .any(|p| p.area == PolicyArea::Speculation)
    );
}

#[test]
fn strict_metadata_entry_still_rejects_config_variants_and_unknown_names() {
    let artifact = Artifact::new(false, false);
    for (pointer, value) in [
        ("/text_config/moe_intermediate_size", json!(1024)),
        ("/text_config/shared_expert_intermediate_size", json!(1024)),
        ("/text_config/linear_num_value_heads", json!(16)),
        (
            "/text_config/rope_parameters/partial_rotary_factor",
            json!(0.5),
        ),
        ("/quantization_config/quant_method", json!("unknown")),
    ] {
        let mut config = config_value();
        *config.pointer_mut(pointer).unwrap() = value;
        std::fs::write(
            artifact.0.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        assert!(
            AutoConfig::from_pretrained(&artifact.0).is_err(),
            "accepted {pointer}"
        );
    }
    let mut bf16 = config_value();
    bf16.as_object_mut().unwrap().remove("quantization_config");
    std::fs::write(
        artifact.0.join("config.json"),
        serde_json::to_vec(&bf16).unwrap(),
    )
    .unwrap();
    assert!(
        AutoConfig::from_pretrained(&artifact.0)
            .unwrap_err()
            .to_string()
            .contains("packed experts")
    );
    std::fs::write(
        artifact.0.join("config.json"),
        serde_json::to_vec(&config_value()).unwrap(),
    )
    .unwrap();
    let metadata = Qwen35Metadata::open_hf(&artifact.0).unwrap();
    let mut inventory = metadata.inventory().clone();
    inventory.tensors[0].name = "model.language_model.unknown.weight".into();
    assert!(
        metadata
            .name_mapper()
            .validate_inventory(&inventory)
            .is_err()
    );
}

#[test]
#[ignore = "requires FERRULE_QWEN35_35B_FP8_DIR; strict config/index/header validation only, no payload or GPU"]
fn nas_35b_fp8_public_backend_plans_match_strict_metadata() {
    let dir = std::env::var("FERRULE_QWEN35_35B_FP8_DIR").expect("set FERRULE_QWEN35_35B_FP8_DIR");
    let path = Path::new(&dir);
    let metadata = Qwen35Metadata::open_hf(path).unwrap();
    assert_eq!(metadata.inventory().tensor_count, 64196);
    assert_eq!(metadata.partition().text().len(), 62303);
    assert_eq!(metadata.partition().visual().len(), 333);
    assert_eq!(metadata.partition().mtp().len(), 1560);
    assert_backend_plans(&metadata.descriptor());
    assert_backend_plans(AutoConfig::from_pretrained(path).unwrap().descriptor());
    let mut bad = metadata.inventory().clone();
    bad.tensors[0].name += ".unknown";
    assert!(metadata.name_mapper().validate_inventory(&bad).is_err());
}
