use super::fixture::*;
use ferrule_model::transformer::{Bf16DenseCudaProfile, Bf16HybridCudaProfile};

#[test]
fn native_complete_hybrid_is_distinct_from_dense_profile() {
    let f = Fixture::new();
    Bf16HybridCudaProfile::validate(&f.resources).unwrap();
    assert!(Bf16DenseCudaProfile::validate(&f.resources).is_err());
    let f32 = Fixture::with_dtype(ferrule_model::CheckpointDType::F32);
    assert!(Bf16HybridCudaProfile::validate(&f32.resources).is_err());
    assert_eq!(f.oracle["conv_dtype"], "bfloat16");
    assert_eq!(f.oracle["ssm_dtype"], "float32");
    assert_eq!(f.oracle["inv_freq_dtype"], "float32");
    assert!(f.dir.join("tokenizer.json").exists());
    assert_eq!(f.oracle["cases"].as_array().unwrap().len(), 5);
    // Exercise the same host preparation/binding functions without executing a
    // CPU forward. The CUDA owner subsequently uploads the native BF16 bytes.
    ferrule_model::transformer::HybridDecoderSegment::prepare(
        &f.resources,
        ferrule_model::transformer::LayerSegmentPlan::new(2, 0..2, true, true).unwrap(),
        ferrule_model::execution::ExecutionPrecisionPolicy::f32(),
        32,
        1 << 20,
    )
    .unwrap();
}

#[test]
fn fp8_35b_is_not_relabelled_as_native_bf16() {
    use ferrule_model::models::qwen35::Qwen35Recipe;
    use ferrule_model::transformer::DecoderRecipe;
    let config = serde_json::from_str(include_str!("../qwen35_35b_fp8_config.json")).unwrap();
    let recipe = Qwen35Recipe::new().build(&config).unwrap();
    // Geometry is separate from weight-encoding admission.
    Bf16HybridCudaProfile::validate_spec(recipe.spec()).unwrap();
    assert_eq!(
        ferrule_model::execution::ExecutionPrecisionPolicy::default(),
        ferrule_model::execution::ExecutionPrecisionPolicy::f32()
    );
}

#[cfg(feature = "cuda")]
#[test]
fn native_bf16_budget_uses_typed_state_kv_and_full_resident_experts() {
    use ferrule_model::decoder::*;
    use ferrule_model::execution::ExecutionPrecisionPolicy;
    use ferrule_model::{ModelFamily, WeightSource};
    let f = Fixture::new();
    let options = GenericDecoderOptions::standard_cpu(
        f.resources.spec(),
        ModelFamily::Unknown("tiny".into()),
        WeightSource::Safetensors,
        2,
        32,
        8,
        1,
        1 << 20,
        ExecutionPrecisionPolicy::bf16_compatibility(),
    )
    .unwrap();
    let plan = HybridCudaMemoryEstimate::for_resources(&f.resources, &options, 32).unwrap();
    assert_eq!(plan.per_sequence_state_bytes, 32 * 4 * 2 + 4 * 4 * 4 * 4);
    assert_eq!(plan.kv_bytes, 32 * 2 * 2 * 8 * 2);
    let words: usize = f.oracle["tensors"]
        .as_object()
        .unwrap()
        .values()
        .map(|t| t["values"].as_array().unwrap().len())
        .sum();
    assert_eq!(plan.weight_bytes_upper_bound, words * 2 + 32 * 4 * 2);
    assert!(plan.workspace_bytes_upper_bound >= 64 << 20);
    assert!(
        HybridCudaMemoryEstimate::for_resources_with_routed_experts(&f.resources, &options, 32)
            .is_err()
    );
    assert!(
        HybridCudaMemoryEstimate::for_resources(
            &f.resources,
            &options.with_active_layers(1).unwrap(),
            32
        )
        .is_err()
    );
}

fn numeric_options(f: &Fixture) -> ferrule_model::decoder::GenericDecoderOptions {
    use ferrule_model::{
        ModelFamily, WeightSource, decoder::GenericDecoderOptions,
        execution::ExecutionPrecisionPolicy,
    };
    GenericDecoderOptions::standard_cpu(
        f.resources.spec(),
        ModelFamily::Unknown("numeric-bf16-tiny".into()),
        WeightSource::Safetensors,
        2,
        32,
        8,
        1,
        1 << 20,
        ExecutionPrecisionPolicy::bf16_compatibility(),
    )
    .unwrap()
    .with_hybrid_cuda_numeric_fp8_bf16(
        ferrule_model::transformer::ExpertCacheLimits {
            max_experts: 1,
            max_bytes: 64 << 10,
        },
        0,
    )
    .unwrap()
}

#[test]
fn raw_numeric_bf16_identity_and_cpu_metadata_preserve_storage() {
    use ferrule_model::transformer::{
        NumericFp8Bf16HybridCudaProfile as Profile, NumericFp8Precision, StateDictMaterializer,
    };
    let f = Fixture::numeric();
    Profile::validate(&f.resources).unwrap();
    assert!(Bf16HybridCudaProfile::validate(&f.resources).is_err());
    assert!(Profile::validate(&Fixture::new().resources).is_err());
    assert_eq!(Profile::NAME, f.oracle["profile"]);
    assert_ne!(
        Profile::NAME,
        NumericFp8Precision::W8A16Bf16F32Accumulate.hybrid_backend_name()
    );
    let options = numeric_options(&f);
    assert_eq!(
        options.hybrid_cuda_numeric_fp8_precision(),
        Some(NumericFp8Precision::Bf16ModelF32Accumulate)
    );
    let plan = ferrule_model::decoder::HybridMemoryPlan::for_resources(&f.resources, &options, 32)
        .unwrap();
    assert_eq!(plan.per_sequence_state_bytes, 512);
    assert_eq!(plan.kv_bytes, 2048);
    assert!(plan.weight_bytes_upper_bound < 128 << 10);
    assert!(
        ferrule_model::decoder::HybridMemoryPlan::for_resources(
            &f.resources,
            &options.clone().with_active_layers(1).unwrap(),
            32
        )
        .is_err()
    );
    let m = StateDictMaterializer::new(1 << 20).unwrap();
    // Preparation cannot call values_f32 on numeric artifacts. Native BF16 is
    // checked word-for-word, not decoded/requantized by the execution loader.
    let mut count = 0;
    for p in f.resources.state_dict().parameters() {
        let prepared = m.parameter(p).unwrap();
        if let Some(a) = prepared.numeric_fp8() {
            count += 1;
            assert_eq!(p.weight().slice().dtype, Profile::WEIGHT_DTYPE);
            assert_eq!(p.scale().unwrap().slice().dtype, Profile::SCALE_DTYPE);
            assert_eq!(
                a.encoding(),
                ferrule_model::checkpoint::NumericFp8Encoding::E4M3FnBlock128Bf16
            );
            assert!(prepared.values_f32().is_err());
            assert_eq!(
                a.weight_bytes(),
                std::fs::read(&p.weight().slice().path).unwrap()
            );
            assert_eq!(
                a.scale_bytes(),
                std::fs::read(&p.scale().unwrap().slice().path).unwrap()
            );
        } else if p.weight().slice().dtype == ferrule_model::CheckpointDType::F32 {
            assert!(matches!(
                p.role(),
                ferrule_model::TensorRole::LinearAttentionALog
                    | ferrule_model::TensorRole::LinearAttentionNorm
            ));
            assert!(
                prepared
                    .values_f32()
                    .unwrap()
                    .iter()
                    .any(|v| v.to_bits() & 0xffff != 0)
            );
        } else {
            assert_eq!(p.weight().slice().dtype, Profile::NATIVE_WEIGHT_DTYPE);
            let bytes = std::fs::read(&p.weight().slice().path).unwrap();
            let expected: Vec<u16> = bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect();
            assert_eq!(prepared.values_bf16_words().unwrap(), expected);
        }
    }
    assert!(count >= 30);
    for (key, expected) in [
        ("conv_dtype", "bfloat16"),
        ("ssm_dtype", "float32"),
        ("scale_dtype", "bfloat16"),
        ("numeric_weight_dtype", "float8_e4m3fn"),
        ("activation_dtype", "bfloat16"),
        ("kv_dtype", "bfloat16"),
        // v1 oracle key means selected route weights, NOT gate parameters.
        ("router_weight_dtype", "float32"),
        ("accumulator_dtype", "float32"),
        ("inv_freq_dtype", "float32"),
    ] {
        assert_eq!(f.oracle[key], expected);
    }
    assert_eq!(f.oracle["cases"].as_array().unwrap().len(), 5);
}

#[test]
fn bound_router_parameters_match_bf16_profile_not_f32_route_weights() {
    use ferrule_model::transformer::NumericFp8Bf16HybridCudaProfile as Numeric;
    use ferrule_model::{CheckpointDType as D, TensorRole};

    for (f, weight_dtype, logits_dtype, probability_dtype, route_dtype) in [
        (
            Fixture::new(),
            Bf16HybridCudaProfile::ROUTER_WEIGHT_DTYPE,
            Bf16HybridCudaProfile::ROUTER_LOGITS_DTYPE,
            Bf16HybridCudaProfile::ROUTER_PROBABILITY_DTYPE,
            Bf16HybridCudaProfile::ROUTE_WEIGHT_DTYPE,
        ),
        (
            Fixture::numeric(),
            Numeric::ROUTER_WEIGHT_DTYPE,
            Numeric::ROUTER_LOGITS_DTYPE,
            Numeric::ROUTER_PROBABILITY_DTYPE,
            Numeric::ROUTE_WEIGHT_DTYPE,
        ),
    ] {
        assert_eq!(weight_dtype, D::Bf16);
        assert_eq!(logits_dtype, D::Bf16);
        assert_eq!(probability_dtype, D::F32);
        assert_eq!(route_dtype, D::F32);
        let routers: Vec<_> = f
            .resources
            .state_dict()
            .parameters()
            .iter()
            .filter(|p| p.role() == &TensorRole::RouterLogits)
            .collect();
        assert_eq!(routers.len(), 2);
        for p in routers {
            assert_eq!(p.weight().slice().dtype, weight_dtype);
            assert!(p.numeric_fp8_encoding().is_none());
            assert!(p.scale().is_none());
        }
    }
    // Serialized v1 oracle compatibility: the old key names route weights.
    let f = Fixture::numeric();
    assert_eq!(f.oracle["router_weight_dtype"], "float32");
    assert_eq!(Numeric::ROUTE_WEIGHT_DTYPE, D::F32);
}

#[test]
fn bound_gdn_sources_round_to_declared_bf16_operands_on_host_only() {
    use ferrule_model::execution::bf16_rne;
    use ferrule_model::transformer::{
        NumericFp8Bf16HybridCudaProfile as Profile, StateDictMaterializer,
    };
    use ferrule_model::{CheckpointDType as D, TensorRole};

    let f = Fixture::numeric();
    Profile::validate(&f.resources).unwrap();
    assert_eq!(
        Profile::GDN_SOURCE_DTYPE,
        Profile::GDN_ALOG_NORM_SOURCE_DTYPE
    );
    assert_eq!(Profile::GDN_ALOG_NORM_SOURCE_DTYPE, D::F32);
    assert_eq!(Profile::GDN_ALOG_NORM_RESIDENT_DTYPE, D::Bf16);
    assert_eq!(Profile::GDN_ALOG_NORM_OPERATOR_DTYPE, D::Bf16);
    assert_eq!(Profile::SSM_DTYPE, D::F32);
    let m = StateDictMaterializer::new(1 << 20).unwrap();
    for role in [
        TensorRole::LinearAttentionALog,
        TensorRole::LinearAttentionNorm,
    ] {
        let parameters: Vec<_> = f
            .resources
            .state_dict()
            .parameters()
            .iter()
            .filter(|p| p.role() == &role)
            .collect();
        assert_eq!(parameters.len(), 1);
        let p = parameters[0];
        let slice = p.weight().slice();
        assert_eq!(slice.dtype, Profile::GDN_ALOG_NORM_SOURCE_DTYPE);
        let prepared = m.parameter(p).unwrap();
        let source = prepared.values_f32().unwrap();
        let tensor = &f.oracle["tensors"][&slice.name];
        assert_eq!(source, values(&tensor["source_values"]));
        let loaded = values(&tensor["values"]);
        assert_eq!(source.len(), loaded.len());
        assert!(source.iter().zip(&loaded).any(|(a, b)| a != b));
        // Host RNE proof against pinned HF values, not an upload/storage test.
        for (&source, expected) in source.iter().zip(loaded) {
            assert_eq!(bf16_rne(source), expected);
        }
    }
}

#[test]
fn numeric_bf16_rope_metadata_separates_calculation_from_resident_policy() {
    use ferrule_model::CheckpointDType as D;
    use ferrule_model::transformer::NumericFp8Bf16HybridCudaProfile as Profile;

    let f = Fixture::numeric();
    assert_eq!(f.oracle["inv_freq_dtype"], "float32");
    assert_eq!(Profile::ROPE_INV_FREQ_DTYPE, D::F32);
    assert_eq!(Profile::ROPE_FREQUENCY_DTYPE, Profile::ROPE_INV_FREQ_DTYPE);
    assert_eq!(Profile::ROPE_TRIG_DTYPE, D::F32);
    // Declared resident/output policy only; no CUDA table or operator is read.
    assert_eq!(Profile::ROPE_COS_SIN_DTYPE, D::Bf16);
    assert_eq!(Profile::ROPE_OUTPUT_DTYPE, D::Bf16);
}

#[test]
fn numeric_bf16_cannot_enter_legacy_f32_or_expert_wire_profiles() {
    use ferrule_model::transformer::{ExpertCacheLimits, NumericFp8Precision};
    let f = Fixture::numeric();
    let options = numeric_options(&f);
    assert!(
        options
            .with_hybrid_cuda_numeric_fp8_precision(
                ExpertCacheLimits {
                    max_experts: 1,
                    max_bytes: 65536
                },
                0,
                NumericFp8Precision::W8A16Bf16F32Accumulate
            )
            .is_err()
    );
    assert_eq!(
        NumericFp8Precision::default(),
        NumericFp8Precision::Bf16RneF32Accumulate
    );
}
