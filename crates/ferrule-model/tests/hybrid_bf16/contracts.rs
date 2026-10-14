use ferrule_model::execution::{ExecutionPrecisionPolicy, bf16_rne};
use serde_json::Value;

fn evidence() -> Value {
    serde_json::from_str(include_str!("hf_boundaries.json")).unwrap()
}

fn number(value: &Value) -> f32 {
    value.as_f64().unwrap() as f32
}

#[test]
fn native_sigmoid_gates_require_two_bf16_boundaries() {
    let evidence = evidence();
    for name in ["scalar_gate", "attention_gate", "shared_gate"] {
        let case = &evidence[name];
        let x = number(&case["x"]);
        let gate = number(&case["gate"]);
        assert_eq!(bf16_rne(x), x);
        assert_eq!(bf16_rne(gate), gate);
        let sigmoid = 1.0 / (1.0 + (-gate).exp());
        assert_eq!(bf16_rne(sigmoid), number(&case["sigmoid"]));
        assert_eq!(bf16_rne(x * bf16_rne(sigmoid)), number(&case["hf"]));
        assert_eq!(bf16_rne(x * sigmoid), number(&case["f32_gate_then_cast"]));
        assert_ne!(case["hf"], case["f32_gate_then_cast"]);
    }
}

#[test]
fn native_expert_combine_rounds_contributions_and_each_add() {
    let evidence = evidence();
    let case = &evidence["combine"];
    let mut hf = 0.0_f32;
    let mut legacy = 0.0_f32;
    let mut previous_expert = None;
    for term in case["terms_expert_order"].as_array().unwrap() {
        let expert = term["expert"].as_u64().unwrap();
        assert!(previous_expert.is_none_or(|previous| previous < expert));
        previous_expert = Some(expert);
        let value = number(&term["value"]);
        let weight = number(&term["weight"]);
        assert_eq!(value, bf16_rne(value));
        hf = bf16_rne(hf + bf16_rne(value * weight));
        legacy += value * weight;
    }
    assert_eq!(hf, number(&case["hf"]));
    assert_eq!(bf16_rne(legacy), number(&case["f32_sum_then_cast"]));
    assert_ne!(hf, bf16_rne(legacy));
    assert_eq!(case["elements"], 128);
    assert!(case["mismatches"].as_u64().unwrap() > 0);
}

#[test]
fn native_loading_keeps_router_probabilities_and_rope_frequencies_f32() {
    let evidence = evidence();
    assert_eq!(evidence["device"], "cpu");
    assert_eq!(evidence["experts_implementation"], "eager");
    assert_eq!(evidence["attention_implementation"], "eager");
    let parameters = evidence["parameters"].as_object().unwrap();
    assert!(!parameters.is_empty());
    assert!(parameters.values().all(|dtype| dtype == "torch.bfloat16"));
    for name in [
        "model.rotary_emb.inv_freq",
        "model.rotary_emb.original_inv_freq",
    ] {
        assert_eq!(evidence["buffers"][name], "torch.float32");
    }
    let router = &evidence["router"];
    assert_eq!(router["linear_output"], "torch.bfloat16");
    assert_eq!(router["probabilities"], "torch.float32");
    assert_eq!(router["weights"], "torch.float32");
    assert!(
        router["selected_weights"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| {
                let value = number(v);
                bf16_rne(value) != value
            })
    );
    assert!(number(&router["selected_softmax_max_error"]) > 0.0);
    assert!(
        evidence["offset_norm"]["early_cast_mismatches"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert_eq!(
        ExecutionPrecisionPolicy::default(),
        ExecutionPrecisionPolicy::f32()
    );
}

#[test]
fn native_router_weights_come_from_full_f32_softmax_then_renormalization() {
    let evidence = evidence();
    let router = &evidence["router"];
    let probabilities = router["full_probabilities"].as_array().unwrap();
    let weights = router["selected_weights"].as_array().unwrap();
    let indices = router["indices"].as_array().unwrap();
    assert_eq!(probabilities.len(), 4 * indices.len());
    assert_eq!(weights.len(), 3 * indices.len());
    for (row, ids) in indices.iter().enumerate() {
        let selected = ids
            .as_array()
            .unwrap()
            .iter()
            .map(|id| number(&probabilities[row * 4 + id.as_u64().unwrap() as usize]))
            .collect::<Vec<_>>();
        let denominator: f32 = selected.iter().sum();
        for (slot, probability) in selected.iter().enumerate() {
            let expected = number(&weights[row * 3 + slot]);
            assert_eq!(*probability / denominator, expected);
        }
    }
}

#[test]
fn pinned_hf_loader_casts_f32_gdn_but_f32_ssm_is_an_explicit_policy() {
    let e: Value = serde_json::from_str(include_str!("hf_loading.json")).unwrap();
    let tiny: Value = serde_json::from_str(include_str!("hf_numeric_model.json")).unwrap();
    assert_eq!(e["versions"]["transformers"], tiny["transformers"]);
    assert_eq!(
        e["source_pins"]["modeling_qwen3_5_moe"],
        tiny["hf_source_sha256"]
    );
    assert_eq!(e["device"], "cpu");
    assert_eq!(e["native_bf16_words_preserved"], true);
    assert_eq!(e["rope"]["native_f32_exact"], true);
    assert_eq!(e["parameter_load"].as_object().unwrap().len(), 2);
    for name in [
        "model.layers.0.linear_attn.A_log",
        "model.layers.0.linear_attn.norm.weight",
    ] {
        let p = &e["parameter_load"][name];
        assert_eq!(p["source_dtype"], "torch.float32");
        assert_eq!(p["loaded_dtype"], "torch.bfloat16");
        for (source, loaded) in p["source_values"]
            .as_array()
            .unwrap()
            .iter()
            .zip(p["loaded_values"].as_array().unwrap())
        {
            assert_ne!(number(source), number(loaded));
            assert_eq!(bf16_rne(number(source)), number(loaded));
        }
    }
    for (policy, dtype) in [
        ("hf_default", "torch.bfloat16"),
        ("explicit_f32_ssm", "torch.float32"),
    ] {
        let cases = e["cache_runs"][policy].as_array().unwrap();
        assert_eq!(cases.len(), 2);
        for case in cases {
            assert_eq!(case["ssm"], dtype);
            for field in ["conv", "kv", "logits"] {
                assert_eq!(case[field], "torch.bfloat16");
            }
        }
    }
}

#[test]
fn numeric_decode_tie_reference_is_explicit_and_keeps_native_oracle() {
    let native: Value = serde_json::from_str(include_str!("hf_numeric_model.json")).unwrap();
    let profile: Value = serde_json::from_str(include_str!("hf_numeric_profile.json")).unwrap();
    assert!(
        profile.get("tensors").is_none(),
        "no alternative weight artifact"
    );
    assert_eq!(profile["hf_source_sha256"], native["hf_source_sha256"]);
    assert_eq!(
        profile["router_tie_policy"],
        "equal raw logits: ascending expert ID; native F32 probabilities/weights unchanged"
    );
    let tie = &profile["cases"][1]["router_ties"][0];
    assert_eq!(tie["layer"], 1);
    assert_eq!(tie["probabilities"][0], tie["probabilities"][2]);
    assert_eq!(tie["native_ids"], serde_json::json!([[1, 3, 2]]));
    assert_eq!(tie["profile_ids"], serde_json::json!([[1, 3, 0]]));
    assert_eq!(native["cases"][1]["routes"][1]["ids"], tie["native_ids"]);
    assert_eq!(profile["cases"][1]["routes"][1]["ids"], tie["profile_ids"]);
    assert_eq!(number(&native["cases"][1]["residuals"][3][0]), -0.12451172);
    assert_eq!(number(&profile["cases"][1]["residuals"][3][0]), -0.13183594);
    for index in [0, 3] {
        for (key, value) in native["cases"][index].as_object().unwrap() {
            assert_eq!(&profile["cases"][index][key], value, "untied {key}");
        }
        assert_eq!(
            profile["cases"][index]["router_ties"],
            serde_json::json!([])
        );
    }
    let fixture = super::fixture::Fixture::numeric_profile();
    assert_eq!(fixture.oracle["tensors"], native["tensors"]);
    assert_eq!(fixture.oracle["cases"], profile["cases"]);
}
