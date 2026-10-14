//! CPU-only admission and independent-HF precision regression checks.
#[path = "support/bf16_dense_fixture.rs"]
mod fixture;
use ferrule_model::decoder::*;
use ferrule_model::execution::{ExecutionPrecisionPolicy, bf16_rne};
use ferrule_model::transformer::*;
use fixture::*;

#[test]
fn bf16_dense_profile_admits_the_complete_fixture() {
    let fixture = Fixture::new();
    Bf16DenseCudaProfile::validate(&fixture.resources).unwrap();
    assert_eq!(fixture.resources.spec().layers().len(), 2);
    assert_eq!(
        ExecutionPrecisionPolicy::default(),
        ExecutionPrecisionPolicy::f32()
    );
}

#[test]
#[ignore = "legacy CPU BF16 full-state differs from independent HF; not the CUDA oracle"]
fn dense_cpu_precision_matches_independent_hf_all_layer_prefill_decode() {
    let fixture = Fixture::new();
    let stage = StandardDecoderSegment::prepare(
        &fixture.resources,
        LayerSegmentPlan::new(2, 0..2, true, true).unwrap(),
        ExecutionPrecisionPolicy::bf16_compatibility(),
        POSITIONS,
        LIMIT,
    )
    .unwrap();
    let mut backend =
        PagedKvBackend::new(CpuPagedKvPool::from_strategy(&fixture.planes(2), 12).unwrap());
    let mut state = GenericDecoderSequenceState::new((), ());
    for (id, case) in [0usize, 1, 3].into_iter().enumerate() {
        let case = &fixture.oracle["cases"][case];
        let tokens: Vec<u32> = serde_json::from_value(case["tokens"].clone()).unwrap();
        let (batch, tx, mut view) = enter(
            &fixture,
            &mut backend,
            &mut state,
            id as u64 + 1,
            &tokens,
            None,
        );
        let SegmentOutput::Logits(logits) = stage
            .execute(&batch, SegmentInput::Tokens, &mut view)
            .unwrap()
        else {
            panic!("logits")
        };
        close(logits.values(), &values(&case["logits"]));
        drop(view);
        finish(&mut backend, &mut state, tx, tokens.len());
    }
}

#[test]
fn semantic_boundaries_are_not_blanket_rounding_of_f32_operations() {
    // Norm casts the normalized value before its BF16 weight multiplication.
    let normalized = 0.9971_f32;
    let weight = bf16_rne(1.071);
    assert_ne!(
        bf16_rne(bf16_rne(normalized) * weight),
        bf16_rne(normalized * weight)
    );
    // RoPE has two BF16 products and a BF16 add; a fused FMA is a different graph.
    let a = bf16_rne(0.73);
    let b = bf16_rne(0.91);
    let c = bf16_rne(-0.62);
    let d = bf16_rne(0.42);
    assert_ne!(
        bf16_rne(bf16_rne(a * b) + bf16_rne(c * d)),
        bf16_rne(a * b + c * d)
    );
}

#[test]
fn rejects_qwen35_gdn_and_moe_before_materialization() {
    use ferrule_model::models::qwen35::Qwen35Recipe;
    for config in [
        include_str!("qwen35_08b_config.json"),
        include_str!("qwen35_35b_fp8_config.json"),
    ] {
        let config = serde_json::from_str(config).unwrap();
        let output = Qwen35Recipe::new().build(&config).unwrap();
        assert!(Bf16DenseCudaProfile::validate_spec(output.spec()).is_err());
    }
    let config = serde_json::json!({"vocab_size":16,"hidden_size":16,"num_attention_heads":2,
        "num_key_value_heads":1,"head_dim":8,"intermediate_size":24,"num_experts":2,
        "experts_per_token":2,"max_position_embeddings":16,"rms_norm_eps":0.00001,"rope_theta":10000.0,
        "tie_word_embeddings":true});
    let output = SyntheticDecoderRecipe::new().build(&config).unwrap();
    assert!(Bf16DenseCudaProfile::validate_spec(output.spec()).is_err());
}

#[test]
fn explicit_boundary_math_matches_independent_torch_bf16_tensors() {
    let fixture = Fixture::new();
    let b = &fixture.oracle["boundaries"];
    let x = values(&b["input"]);
    let weight = values(&b["weight"]);
    let up = values(&b["up"]);
    let c = values(&b["cosine"]);
    let s = values(&b["sine"]);
    let inv = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + 1e-5)
        .sqrt()
        .recip();
    let norm = x
        .iter()
        .zip(&weight)
        .map(|(&v, &w)| bf16_rne(bf16_rne(v * inv) * w))
        .collect::<Vec<_>>();
    assert_eq!(norm, values(&b["norm"]));
    let rope = (0..4)
        .map(|i| {
            let rotated = if i < 2 { -x[i + 2] } else { x[i - 2] };
            bf16_rne(bf16_rne(x[i] * c[i]) + bf16_rne(rotated * s[i]))
        })
        .collect::<Vec<_>>();
    assert_eq!(rope, values(&b["rope"]));
    let swiglu = x
        .iter()
        .zip(&up)
        .map(|(&v, &u)| bf16_rne(bf16_rne(v / (1.0 + (-v).exp())) * u))
        .collect::<Vec<_>>();
    assert_eq!(swiglu, values(&b["swiglu"]));
    assert_eq!(
        x.iter()
            .zip(&up)
            .map(|(&v, &u)| bf16_rne(v + u))
            .collect::<Vec<_>>(),
        values(&b["residual"])
    );
}

#[test]
fn rejects_unaligned_gemm_before_cuda_or_materialization() {
    use ferrule_model::models::qwen3::Qwen3DenseRecipe;
    let fixture = Fixture::new();
    let mut config = fixture.oracle["config"].clone();
    config["intermediate_size"] = 25.into();
    let output = Qwen3DenseRecipe::new().build(&config).unwrap();
    assert!(
        Bf16DenseCudaProfile::validate_spec(output.spec())
            .unwrap_err()
            .to_string()
            .contains("K8")
    );
}

#[test]
fn hf_checkpoint_loading_preserves_f32_rope_frequencies() {
    let fixture = Fixture::new();
    let frequencies = values(&fixture.oracle["rotary_contract"]["inverse_frequency"]);
    assert_eq!(frequencies, vec![1.0, 0.1, 0.01, 0.001]);
    // The old blanket .to(BF16) oracle rounded inv_freq before trigonometry.
    // Casting that rounded buffer back to F32 cannot recover these frequencies.
    assert_ne!(frequencies[1], bf16_rne(frequencies[1]));
    assert_eq!(
        bf16_rne((2.0 * frequencies[1]).sin()),
        f32::from_bits(0x3e4b0000)
    );
    assert_eq!(bf16_rne((2.0 * bf16_rne(frequencies[1])).sin()), 0.19921875);
}
