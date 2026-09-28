//! Native full-depth Qwen3.5 CPU checks against the independently exported HF oracle.
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use ferrule_common::execution::*;
use ferrule_model::decoder::HybridDecoderSequenceState;
use ferrule_model::models::qwen35::{
    Qwen35Adapter, Qwen35CpuRunner, Qwen35PrepareOptions, Qwen35Recipe,
};
use ferrule_model::nn::{ModulePath, ParameterDType};
use ferrule_model::runner::{
    ModelRunner, MultiSessionBatchProgress, MultiSessionRunner, TransactionEndIntent,
    TransactionEndProgress,
};
use ferrule_model::transformer::{Attention, DecoderRecipe, RotaryRegion};
use ferrule_model::{
    CheckpointDType, CheckpointTensorReader, CheckpointTensorSlice, ModelFamily, TensorRole,
    TokenizerHandle,
};
use serde_json::Value;

#[test]
fn qwen35_typed_unsupported_profiles_and_cuda_fail_before_loading() {
    use ferrule_model::ModelExecutionBackend;
    use ferrule_model::models::qwen35::{Qwen35Config, Qwen35Unsupported};
    let assert_source = |error: ferrule_common::Error, expected: Qwen35Unsupported| {
        let ferrule_common::Error::ModelSource { source } = error else {
            panic!("lost typed unsupported source")
        };
        assert_eq!(source.downcast_ref::<Qwen35Unsupported>(), Some(&expected));
    };
    let mut value: Value = serde_json::from_str(include_str!("qwen35_08b_config.json")).unwrap();
    value["model_type"] = "qwen3_5_moe".into();
    assert_source(
        Qwen35Config::from_value(&value).unwrap_err(),
        Qwen35Unsupported::PackedBf16Experts,
    );
    value["model_type"] = "qwen3_5".into();
    value["quantization_config"] = serde_json::json!({"quant_method":"fp8"});
    assert_source(
        Qwen35Config::from_value(&value).unwrap_err(),
        Qwen35Unsupported::Quantization,
    );
    let error = Qwen35Adapter::load_hf_with_options_and_backend(
        Path::new("unused-qwen35-no-io"),
        Qwen35PrepareOptions::default(),
        ModelExecutionBackend::Cuda,
    )
    .err()
    .expect("CUDA must be rejected before any artifact I/O");
    assert_source(
        error,
        Qwen35Unsupported::Backend(ModelExecutionBackend::Cuda),
    );
}

#[test]
fn qwen35_recipe_has_exact_hybrid_descriptors_roles_and_alias() {
    let value = serde_json::from_str(include_str!("qwen35_08b_config.json")).unwrap();
    let recipe = Qwen35Recipe::new().build(&value).unwrap();
    assert_eq!(recipe.spec().layers().len(), 24);
    assert_eq!(recipe.schema().len(), 321);
    assert_eq!(recipe.schema().storage_tensor_count(), 320);
    let get = |path: &str| {
        recipe
            .schema()
            .get(&ModulePath::new(path).unwrap())
            .unwrap()
            .clone()
    };
    let embedding = get("token_embedding.weight");
    assert_eq!(get("output.weight").alias_of(), Some(embedding.id()));
    for (i, layer) in recipe.spec().layers().iter().enumerate() {
        assert!(layer.input_norm().one_plus_weight());
        assert!(layer.post_attention_norm().one_plus_weight());
        if i % 4 == 3 {
            let Attention::Gqa(g) = layer.attention() else {
                panic!("missing full attention")
            };
            assert!(g.gated_query());
            assert_eq!(g.query().weight_shape(), [4096, 1024]);
            assert!(g.query_norm().unwrap().one_plus_weight());
            assert!(g.key_norm().unwrap().one_plus_weight());
            assert_eq!(g.rotary().region(), RotaryRegion::Prefix { dimensions: 64 });
        } else {
            let Attention::GatedDeltaNet(g) = layer.attention() else {
                panic!("missing linear attention")
            };
            assert_eq!(g.qkv().weight_shape(), [6144, 1024]);
            assert_eq!(g.conv_weight_shape(), [6144, 1, 4]);
            assert!(!g.norm().one_plus_weight());
            for (suffix, role, dtype) in [
                (
                    "query_key_value",
                    TensorRole::LinearAttentionQkv,
                    ParameterDType::Bf16,
                ),
                ("gate", TensorRole::LinearAttentionZ, ParameterDType::Bf16),
                ("decay", TensorRole::LinearAttentionA, ParameterDType::Bf16),
                (
                    "beta",
                    TensorRole::LinearAttentionBeta,
                    ParameterDType::Bf16,
                ),
                (
                    "convolution",
                    TensorRole::LinearAttentionConv,
                    ParameterDType::Bf16,
                ),
                (
                    "decay_log",
                    TensorRole::LinearAttentionALog,
                    ParameterDType::F32,
                ),
                (
                    "time_bias",
                    TensorRole::LinearAttentionDtBias,
                    ParameterDType::Bf16,
                ),
                ("norm", TensorRole::LinearAttentionNorm, ParameterDType::F32),
                ("output", TensorRole::AttentionOutput, ParameterDType::Bf16),
            ] {
                let parameter = get(&format!("layers.{i}.linear_attention.{suffix}.weight"));
                assert_eq!(recipe.schema().role(parameter.id()), Some(&role));
                assert_eq!(parameter.dtype().allowed(), [dtype]);
            }
        }
    }
    assert!(recipe.spec().final_norm().one_plus_weight());
}

fn step(
    runner: &mut Qwen35CpuRunner,
    states: &mut [HybridDecoderSequenceState],
    pages: &mut Vec<KvPageId>,
    input: &[u32],
    id: u64,
    base_page: u32,
) -> Vec<f32> {
    let context = states[0].core().position();
    let end = context + input.len();
    let mut new_pages = Vec::new();
    while pages.len() < end.div_ceil(16) {
        let page = KvPageId(base_page + pages.len() as u32);
        pages.push(page);
        new_pages.push(page);
    }
    let phase = if context == 0 {
        ForwardPhase::Prefill
    } else {
        ForwardPhase::Decode
    };
    let batch = ExecutionBatch::new(
        if context == 0 {
            ForwardMode::Prefill
        } else {
            ForwardMode::Decode
        },
        input.to_vec(),
        (context..end).map(|p| p as u32).collect(),
        (context..end)
            .map(|p| Some(KvWriteSlot::new(pages[p / 16].0 * 16 + (p % 16) as u32)))
            .collect(),
        vec![LogitsRequest::Full; input.len()],
        vec![ExecutionSequence::new(
            StateSlot::new(0),
            phase,
            0..input.len() as u32,
            context as u32,
            end as u32,
            0..pages.len() as u32,
        )],
        pages.iter().map(|p| KvBlockId::new(p.0)).collect(),
    );
    let reservations = [KvReservationView {
        state_slot: StateSlot::new(0),
        execution_state_slot: StateSlot::new(0),
        positions: context..end,
        newly_allocated: new_pages,
        generation: states[0].core().generation(),
        execution_generation: states[0].core().generation(),
        cow_replacement: None,
    }];
    let tx = ExecutionTransactionId::new(id).unwrap();
    runner
        .prepare_multi_session_batch(tx, states, &batch, &reservations)
        .unwrap();
    let output = match runner
        .execute_multi_session_batch_progress(tx, states, &batch)
        .unwrap()
    {
        MultiSessionBatchProgress::Complete(output) => output,
        _ => panic!("synchronous CPU hybrid unexpectedly suspended"),
    };
    assert_eq!(
        states[0].core().position(),
        context,
        "unpublished transaction changed committed cursor"
    );
    assert_eq!(
        runner
            .end_transaction(tx, states, TransactionEndIntent::Publish)
            .unwrap(),
        TransactionEndProgress::Complete
    );
    assert_eq!(states[0].core().position(), end);
    assert_eq!(output.logits.len(), input.len());
    output
        .logits
        .into_iter()
        .flat_map(|row| match row.logits {
            LogitsOutput::Full(values) => values,
            _ => panic!("full logits required"),
        })
        .collect()
}

struct Reference {
    path: PathBuf,
    header: Value,
    data_start: u64,
    report: Value,
}
impl Reference {
    fn open(directory: &Path, manifest: &Value, case: &str) -> Self {
        let entry = manifest["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["id"] == case)
            .unwrap();
        assert_eq!(entry["self_checks_passed"], true);
        let report: Value = serde_json::from_slice(
            &std::fs::read(directory.join(entry["report"].as_str().unwrap())).unwrap(),
        )
        .unwrap();
        let path = directory.join(entry["file"].as_str().unwrap());
        assert_eq!(report["sha256"], entry["sha256"]);
        // Test-only integrity check; no new production crypto/serialization dependencies.
        let checksum = Command::new("sha256sum")
            .arg(&path)
            .output()
            .expect("sha256sum is required for the opt-in oracle test");
        assert!(checksum.status.success());
        assert_eq!(
            String::from_utf8(checksum.stdout)
                .unwrap()
                .split_whitespace()
                .next()
                .unwrap(),
            entry["sha256"].as_str().unwrap()
        );
        let mut file = std::fs::File::open(&path).unwrap();
        let mut len = [0u8; 8];
        file.read_exact(&mut len).unwrap();
        let len = u64::from_le_bytes(len);
        assert!(len < 16 * 1024 * 1024);
        let mut header = vec![0u8; len as usize];
        file.read_exact(&mut header).unwrap();
        Self {
            path,
            header: serde_json::from_slice(&header).unwrap(),
            data_start: 8 + len,
            report,
        }
    }
    fn logits(&self, stage: &str, rows: usize) -> Vec<f32> {
        let name = format!("{stage}.logits");
        let t = &self.header[&name];
        assert_eq!(t["dtype"], "F32");
        assert_eq!(t["shape"], serde_json::json!([1, rows, 248320]));
        assert_eq!(t["shape"], self.report["tensors"][&name]["shape"]);
        let start = t["data_offsets"][0].as_u64().unwrap();
        let end = t["data_offsets"][1].as_u64().unwrap();
        let slice = CheckpointTensorSlice {
            name,
            role: TensorRole::OutputHead,
            path: self.path.clone(),
            offset: self.data_start + start,
            bytes: end - start,
            dtype: CheckpointDType::F32,
            shape: vec![1, rows, 248320],
        };
        let payload = CheckpointTensorReader::new(16 * 1024 * 1024)
            .read_slice(&slice)
            .unwrap();
        payload
            .bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }
}
fn argmax(values: &[f32]) -> u32 {
    let mut best = 0;
    for i in 1..values.len() {
        if values[i] > values[best] {
            best = i;
        }
    }
    best as u32
}
fn compare(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len());
    let (mut max_abs, mut sum_abs, mut sum_sq, mut max_rel, mut max_ratio, mut failed) =
        (0f64, 0f64, 0f64, 0f64, 0f64, 0usize);
    for (&a, &r) in actual.iter().zip(expected) {
        assert!(a.is_finite() && r.is_finite());
        let (a, r) = (a as f64, r as f64);
        let d = (a - r).abs();
        let limit = 2e-4 + 2e-4 * r.abs();
        max_abs = max_abs.max(d);
        sum_abs += d;
        sum_sq += d * d;
        max_rel = max_rel.max(d / r.abs().max(1e-8));
        max_ratio = max_ratio.max(d / limit);
        failed += usize::from(d > limit);
    }
    println!(
        "{label}: elements={} max_abs={max_abs:.9e} mean_abs={:.9e} rmse={:.9e} max_relative_floor_1e-8={max_rel:.9e} max_tolerance_ratio={max_ratio:.9e} failed={failed}",
        actual.len(),
        sum_abs / actual.len() as f64,
        (sum_sq / actual.len() as f64).sqrt()
    );
    assert_eq!(failed, 0, "{label} exceeds 2e-4 + 2e-4*abs(reference)");
    for (a, r) in actual
        .as_chunks::<248320>()
        .0
        .iter()
        .zip(expected.as_chunks::<248320>().0)
    {
        assert_eq!(argmax(a), argmax(r), "{label} row argmax");
    }
}

#[test]
#[ignore = "NAS 0.8B full 24-layer CPU F32; requires exported oracle and sha256sum"]
fn nas_08b_native_prefill_and_two_decodes_match_all_oracle_logits() {
    let model = std::env::var_os("FERRULE_QWEN35_08B_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| "/mnt/nas1/hf/Qwen3.5-0.8B".into());
    let reference = std::env::var_os("FERRULE_QWEN35_ORACLE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/validation/qwen35-reference-cpu-f32")
        });
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(reference.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["schema"], "ferrule.qwen35-reference.v1");
    assert_eq!(manifest["status"], "complete");
    assert_eq!(manifest["execution"]["device"], "cpu");
    assert_eq!(manifest["execution"]["dtype"], "F32");
    let start = Instant::now();
    let adapter = Qwen35Adapter::load_hf_with_options(
        &model,
        Qwen35PrepareOptions {
            page_size: 16,
            max_parameter_bytes: 1 << 30,
        },
    )
    .unwrap();
    assert_eq!(adapter.resources().state_dict().len(), 321);
    assert_eq!(adapter.resources().spec().layers().len(), 24);
    assert_eq!(adapter.metadata().partition().text().len(), 320);
    assert_eq!(adapter.metadata().partition().visual().len(), 153);
    assert_eq!(adapter.metadata().partition().mtp().len(), 15);
    let embedding = adapter
        .resources()
        .require_static(TensorRole::TokenEmbedding)
        .unwrap();
    let output = adapter
        .resources()
        .require_static(TensorRole::OutputHead)
        .unwrap();
    assert!(output.shares_storage_with(embedding));
    assert_eq!(embedding.weight().slice().bytes, 508_559_360);
    let mut runner = adapter.into_decoder(32, 16, 1).unwrap();
    runner.configure_kv_page_capacity(8).unwrap();
    assert_eq!(runner.bound_layer_count(), Some(24));
    assert_eq!(runner.model_info().family, ModelFamily::Qwen35);
    println!("Loaded full Qwen3.5-0.8B CPU F32 in {:?}", start.elapsed());
    let tokenizer = TokenizerHandle::load(&model).unwrap();
    for (case_index, case) in ["capital", "hello"].iter().enumerate() {
        let oracle = Reference::open(&reference, &manifest, case);
        let mut states = vec![runner.create_sequence_state().unwrap()];
        let mut pages = Vec::new();
        let mut predictions = Vec::new();
        for (call_index, stage) in ["prefill", "decode.0", "decode.1"].iter().enumerate() {
            let call = &oracle.report["calls"][call_index];
            assert_eq!(call["stage"], *stage);
            assert_eq!(
                call["position_start"].as_u64().unwrap() as usize,
                states[0].core().position()
            );
            let tokens: Vec<u32> = call["input_token_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            if call_index > 0 {
                assert_eq!(tokens, [predictions[call_index - 1]]);
            }
            let started = Instant::now();
            let actual = step(
                &mut runner,
                &mut states,
                &mut pages,
                &tokens,
                (case_index * 3 + call_index + 1) as u64,
                (case_index * 4) as u32,
            );
            let expected = oracle.logits(stage, tokens.len());
            compare(&actual, &expected, &format!("{case}/{stage}"));
            let prediction = argmax(&actual[actual.len() - 248320..]);
            predictions.push(prediction);
            assert_eq!(prediction, call["next_token_id"].as_u64().unwrap() as u32);
            println!(
                "{case}/{stage}: position={} next={prediction} text={:?} elapsed={:?}",
                states[0].core().position(),
                tokenizer.decode(&[prediction]).unwrap(),
                started.elapsed()
            );
        }
        println!(
            "{case}: actual_predictions={predictions:?} actual_text={:?}",
            tokenizer.decode(&predictions).unwrap()
        );
        runner
            .try_release_sequence_state(states.pop().unwrap())
            .unwrap();
    }
}
