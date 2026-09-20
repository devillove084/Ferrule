use std::collections::BTreeMap;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use ferrule_common::execution::{
    ExecutionBatch, ExecutionSequence, ExecutionTransactionId, ForwardMode, ForwardPhase,
    KvBlockId, KvLayoutSchema, KvPageId, KvReservationView, KvWriteSlot, LogitsOutput,
    LogitsRequest, StateSlot,
};
use ferrule_model::models::qwen3::{Qwen3DenseAdapter, Qwen3DensePrepareOptions, Qwen3DenseRecipe};
use ferrule_model::nn::{ModulePath, ParameterDType};
use ferrule_model::runner::{
    ModelRunner, MultiSessionBatchProgress, MultiSessionRunner, ResidentModelRunner,
    TransactionEndIntent, TransactionEndProgress,
};
use ferrule_model::transformer::{
    Attention, DecoderLoadOptions, DecoderRecipe, ExternalTensorMeta, FeedForward, NameMapper,
    RotaryPairing, RotaryRegion,
};
use ferrule_model::{
    AutoConfig, HfSafetensorsInventory, ModelExecutionBackend, ModelFamily, TensorRole,
};

#[test]
fn qwen3_dense_recipe_builds_qwen3_06b_contract() {
    let value = qwen3_06b_config();
    let output = Qwen3DenseRecipe::new().build(&value).unwrap();
    let spec = output.spec();
    let schema = output.schema();

    assert_eq!(spec.architecture(), "Qwen3ForCausalLM");
    assert_eq!(spec.layers().len(), 28);
    assert_eq!(spec.hidden_size(), 1024);
    assert_eq!(spec.vocab_size(), 151_936);
    assert_eq!(spec.max_sequence_length(), Some(40_960));
    assert!(spec.tie_word_embeddings());
    assert_eq!(schema.len(), 3 + 28 * 11);

    for (index, layer) in spec.layers().iter().enumerate() {
        assert_eq!(layer.index(), index);
        let Attention::Gqa(attention) = layer.attention() else {
            panic!("Qwen3 dense layer is not GQA")
        };
        assert_eq!(attention.num_heads(), 16);
        assert_eq!(attention.num_kv_heads(), 8);
        assert_eq!(attention.head_dim(), 128);
        assert_eq!(attention.query().weight_shape(), [2048, 1024]);
        assert_eq!(attention.key().weight_shape(), [1024, 1024]);
        assert_eq!(attention.value().weight_shape(), [1024, 1024]);
        assert_eq!(attention.output().weight_shape(), [1024, 2048]);
        assert_eq!(attention.query_norm().unwrap().hidden_size(), 128);
        assert_eq!(attention.key_norm().unwrap().hidden_size(), 128);
        assert_eq!(attention.rotary().pairing(), RotaryPairing::SplitHalf);
        assert_eq!(
            attention.rotary().region(),
            RotaryRegion::Prefix { dimensions: 128 }
        );
        let FeedForward::SwiGlu(feed_forward) = layer.feed_forward() else {
            panic!("Qwen3 dense layer is not SwiGLU")
        };
        assert_eq!(feed_forward.gate().weight_shape(), [3072, 1024]);
        assert_eq!(feed_forward.up().weight_shape(), [3072, 1024]);
        assert_eq!(feed_forward.down().weight_shape(), [1024, 3072]);
    }

    assert!(schema.parameters().iter().all(|parameter| {
        parameter.dtype().allowed() == [ParameterDType::Bf16]
            && parameter.scale().tensor().is_none()
            && !parameter.optional()
    }));
    let embedding = schema
        .get(&ModulePath::new("token_embedding.weight").unwrap())
        .unwrap();
    let output_head = schema
        .get(&ModulePath::new("output.weight").unwrap())
        .unwrap();
    assert_eq!(output_head.alias_of(), Some(embedding.id()));

    let mapper = output.name_mapper();
    for (external, canonical) in [
        (
            "model.layers.27.input_layernorm.weight",
            "layers.27.input_norm.weight",
        ),
        (
            "model.layers.27.post_attention_layernorm.weight",
            "layers.27.post_attention_norm.weight",
        ),
        (
            "model.layers.27.self_attn.q_proj.weight",
            "layers.27.attention.query.weight",
        ),
        (
            "model.layers.27.self_attn.k_proj.weight",
            "layers.27.attention.key.weight",
        ),
        (
            "model.layers.27.self_attn.v_proj.weight",
            "layers.27.attention.value.weight",
        ),
        (
            "model.layers.27.self_attn.o_proj.weight",
            "layers.27.attention.output.weight",
        ),
        (
            "model.layers.27.self_attn.q_norm.weight",
            "layers.27.attention.query_norm.weight",
        ),
        (
            "model.layers.27.self_attn.k_norm.weight",
            "layers.27.attention.key_norm.weight",
        ),
        (
            "model.layers.27.mlp.gate_proj.weight",
            "layers.27.feed_forward.gate.weight",
        ),
        (
            "model.layers.27.mlp.up_proj.weight",
            "layers.27.feed_forward.up.weight",
        ),
        (
            "model.layers.27.mlp.down_proj.weight",
            "layers.27.feed_forward.down.weight",
        ),
        ("model.embed_tokens.weight", "token_embedding.weight"),
        ("model.norm.weight", "final_norm.weight"),
    ] {
        assert_mapping(mapper, external, canonical);
    }
    let tied_head = mapper.map("lm_head.weight", meta()).unwrap().unwrap();
    assert_eq!(tied_head.path.as_str(), "output.weight");
}

#[test]
fn qwen3_dense_single_file_binds_and_runs_prefill_decode_kv() {
    let fixture = DenseFixture::new();
    let config = fixture.config.clone();

    let descriptor = AutoConfig::from_pretrained(&fixture.dir).unwrap();
    assert_eq!(descriptor.descriptor().spec.family, ModelFamily::Qwen3);
    assert_eq!(descriptor.descriptor().spec.tensor_count, Some(13));
    assert!(descriptor.descriptor().engine_plan().is_executable());
    assert!(
        descriptor
            .descriptor()
            .spec
            .notes
            .iter()
            .any(|note| note.contains("single-file safetensors header inventory"))
    );

    let recipe = Qwen3DenseRecipe::new();
    let checkpoint = DecoderLoadOptions::new(&recipe, &config)
        .open_hf_checkpoint(&fixture.dir, ModelFamily::Qwen3)
        .unwrap();
    assert!(checkpoint.index().is_none());
    assert_eq!(checkpoint.inventory().shard_count, 1);
    assert_eq!(
        checkpoint.inventory().shard_summaries[0].shard,
        "model.safetensors"
    );
    assert_eq!(checkpoint.resources().state_dict().len(), 14);
    assert!(
        checkpoint
            .resources()
            .state_dict()
            .validate_source_identities()
    );

    let adapter = Qwen3DenseAdapter::load_hf_with_options(
        &fixture.dir,
        1024 * 1024,
        Qwen3DensePrepareOptions {
            page_size: 2,
            max_dense_tensor_bytes: 1024 * 1024,
            max_layers: Some(1),
        },
    )
    .unwrap();
    assert_eq!(adapter.backend_name(), "cpu");
    assert_eq!(adapter.kv_schema().planes().len(), 2);
    assert_eq!(adapter.kv_schema().planes()[0].layer_count, 1);
    assert_eq!(
        adapter.resources().spec().architecture(),
        "Qwen3ForCausalLM"
    );

    let mut runner = adapter.into_decoder(8, 3, 1).unwrap();
    assert_eq!(runner.model_info().family, ModelFamily::Qwen3);
    assert_eq!(runner.model_info().num_experts, 0);
    assert_eq!(runner.bound_layer_count(), Some(1));
    assert!(runner.expert_residency_requirements().is_none());
    runner.configure_kv_page_capacity(2).unwrap();
    let mut states = vec![runner.create_sequence_state().unwrap()];
    let state_slot = StateSlot::new(0);
    let first_page = KvPageId(10);
    let second_page = KvPageId(11);

    let prefill = ExecutionBatch::new(
        ForwardMode::Prefill,
        vec![1, 2],
        vec![0, 1],
        vec![
            Some(KvWriteSlot::new(first_page.0 * 2)),
            Some(KvWriteSlot::new(first_page.0 * 2 + 1)),
        ],
        vec![
            LogitsRequest::TopK(NonZeroU32::new(8).unwrap()),
            LogitsRequest::TopK(NonZeroU32::new(8).unwrap()),
        ],
        vec![ExecutionSequence::new(
            state_slot,
            ForwardPhase::Prefill,
            0..2,
            0,
            2,
            0..1,
        )],
        vec![KvBlockId::new(first_page.0)],
    );
    let prefill_reservations = vec![KvReservationView {
        state_slot,
        execution_state_slot: state_slot,
        positions: 0..2,
        newly_allocated: vec![first_page],
        generation: states[0].core().generation(),
        execution_generation: states[0].core().generation(),
        cow_replacement: None,
    }];
    let prefill_output = execute(
        &mut runner,
        &mut states,
        ExecutionTransactionId::new(1).unwrap(),
        &prefill,
        &prefill_reservations,
    );
    assert_eq!(prefill_output.logits.len(), 2);
    publish(
        &mut runner,
        &mut states,
        ExecutionTransactionId::new(1).unwrap(),
    );
    assert_eq!(states[0].core().position(), 2);

    let decode = ExecutionBatch::new(
        ForwardMode::Decode,
        vec![3],
        vec![2],
        vec![Some(KvWriteSlot::new(second_page.0 * 2))],
        vec![LogitsRequest::TopK(NonZeroU32::new(8).unwrap())],
        vec![ExecutionSequence::new(
            state_slot,
            ForwardPhase::Decode,
            0..1,
            2,
            3,
            0..2,
        )],
        vec![KvBlockId::new(first_page.0), KvBlockId::new(second_page.0)],
    );
    let decode_reservations = vec![KvReservationView {
        state_slot,
        execution_state_slot: state_slot,
        positions: 2..3,
        newly_allocated: vec![second_page],
        generation: states[0].core().generation(),
        execution_generation: states[0].core().generation(),
        cow_replacement: None,
    }];
    let decode_output = execute(
        &mut runner,
        &mut states,
        ExecutionTransactionId::new(2).unwrap(),
        &decode,
        &decode_reservations,
    );
    let LogitsOutput::TopK(logits) = &decode_output.logits[0].logits else {
        panic!("dense Qwen3 decode did not return top-k logits")
    };
    assert_eq!(logits.len(), 8);
    assert!(logits.iter().all(|item| item.logit.is_finite()));
    assert!(logits.windows(2).any(|pair| pair[0].logit != pair[1].logit));
    publish(
        &mut runner,
        &mut states,
        ExecutionTransactionId::new(2).unwrap(),
    );
    assert_eq!(states[0].core().position(), 3);
    assert_eq!(runner.observability_snapshot().resident_kv_pages, 2);
    assert_eq!(runner.observability_snapshot().completed_batches, 2);

    // Compare with the same real decoder processing the entire causal prefix.
    // There is no separate reference/toy forward implementation in this test.
    let reference = full_prefill(&fixture);
    assert_same_logits(&decode_output.logits[0].logits, &reference);
    let no_mlp = DenseFixture::with_mlp(false);
    assert_ne!(
        reference,
        full_prefill(&no_mlp),
        "dense SwiGLU must affect logits"
    );
}

fn assert_mapping(mapper: &dyn NameMapper, external: &str, canonical: &str) {
    let mapping = mapper
        .map(external, meta())
        .unwrap_or_else(|error| panic!("failed to map {external}: {error}"))
        .unwrap_or_else(|| panic!("did not recognize {external}"));
    assert_eq!(mapping.path.as_str(), canonical);
}

fn meta() -> ExternalTensorMeta<'static> {
    ExternalTensorMeta {
        dtype: "BF16",
        shape: &[1],
        bytes: 2,
    }
}

fn qwen3_06b_config() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["Qwen3ForCausalLM"],
        "attention_bias": false,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "max_position_embeddings": 40960,
        "max_window_layers": 28,
        "model_type": "qwen3",
        "num_attention_heads": 16,
        "num_hidden_layers": 28,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-6,
        "rope_scaling": null,
        "rope_theta": 1000000.0,
        "sliding_window": null,
        "tie_word_embeddings": true,
        "torch_dtype": "bfloat16",
        "transformers_version": "test",
        "use_cache": true,
        "use_sliding_window": false,
        "vocab_size": 151936
    })
}

struct DenseFixture {
    dir: PathBuf,
    config: serde_json::Value,
}

impl DenseFixture {
    fn new() -> Self {
        Self::with_mlp(true)
    }

    fn with_mlp(mlp: bool) -> Self {
        let dir = unique_temp_dir("ferrule-qwen3-dense");
        std::fs::create_dir_all(&dir).unwrap();
        let config = small_dense_config();
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_vec_pretty(&config).unwrap(),
        )
        .unwrap();
        std::fs::write(dir.join("generation_config.json"), r#"{"eos_token_id": 2}"#).unwrap();
        write_tokenizer(&dir);
        write_safetensors(&dir.join("model.safetensors"), &dense_tensor_shapes(), mlp);
        Self { dir, config }
    }
}

impl Drop for DenseFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn small_dense_config() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["Qwen3ForCausalLM"],
        "attention_bias": false,
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "head_dim": 2,
        "hidden_act": "silu",
        "hidden_size": 4,
        "initializer_range": 0.02,
        "intermediate_size": 6,
        "max_position_embeddings": 8,
        "max_window_layers": 1,
        "model_type": "qwen3",
        "num_attention_heads": 4,
        "num_hidden_layers": 1,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "rope_scaling": null,
        "rope_theta": 1000000.0,
        "sliding_window": null,
        "tie_word_embeddings": true,
        "torch_dtype": "bfloat16",
        "transformers_version": "test",
        "use_cache": true,
        "use_sliding_window": false,
        "vocab_size": 8
    })
}

fn dense_tensor_shapes() -> BTreeMap<String, Vec<usize>> {
    BTreeMap::from([
        ("model.embed_tokens.weight".into(), vec![8, 4]),
        ("model.norm.weight".into(), vec![4]),
        ("model.layers.0.input_layernorm.weight".into(), vec![4]),
        (
            "model.layers.0.post_attention_layernorm.weight".into(),
            vec![4],
        ),
        ("model.layers.0.self_attn.q_proj.weight".into(), vec![8, 4]),
        ("model.layers.0.self_attn.k_proj.weight".into(), vec![4, 4]),
        ("model.layers.0.self_attn.v_proj.weight".into(), vec![4, 4]),
        ("model.layers.0.self_attn.o_proj.weight".into(), vec![4, 8]),
        ("model.layers.0.self_attn.q_norm.weight".into(), vec![2]),
        ("model.layers.0.self_attn.k_norm.weight".into(), vec![2]),
        ("model.layers.0.mlp.gate_proj.weight".into(), vec![6, 4]),
        ("model.layers.0.mlp.up_proj.weight".into(), vec![6, 4]),
        ("model.layers.0.mlp.down_proj.weight".into(), vec![4, 6]),
    ])
}

fn write_safetensors(path: &Path, tensors: &BTreeMap<String, Vec<usize>>, mlp: bool) {
    let mut header = serde_json::Map::new();
    let mut payload_bytes = 0usize;
    for (name, shape) in tensors {
        let bytes = shape.iter().product::<usize>() * 2;
        header.insert(
            name.clone(),
            serde_json::json!({
                "dtype": "BF16",
                "shape": shape,
                "data_offsets": [payload_bytes, payload_bytes + bytes]
            }),
        );
        payload_bytes += bytes;
    }
    let mut header = serde_json::to_vec(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut file = Vec::with_capacity(8 + header.len() + payload_bytes);
    file.extend_from_slice(&(header.len() as u64).to_le_bytes());
    file.extend_from_slice(&header);
    for (tensor_index, (name, shape)) in tensors.iter().enumerate() {
        for index in 0..shape.iter().product::<usize>() {
            let value = if !mlp && name.ends_with("mlp.down_proj.weight") {
                0.0
            } else if shape.len() == 1 {
                1.0 + (index % 2) as f32 * 0.125
            } else {
                ((index * 5 + tensor_index * 7) % 17) as f32 / 16.0 - 0.5
            };
            file.extend_from_slice(&half::bf16::from_f32(value).to_bits().to_le_bytes());
        }
    }
    std::fs::write(path, file).unwrap();
}

fn write_tokenizer(dir: &Path) {
    let mut tokenizer = tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default());
    let tokens = ["a", "b", "c", "d", "e", "f", "g", "h"]
        .into_iter()
        .map(|token| tokenizers::AddedToken::from(token, false));
    assert_eq!(tokenizer.add_tokens(tokens).unwrap(), 8);
    tokenizer.save(dir.join("tokenizer.json"), false).unwrap();
}

fn execute(
    runner: &mut ferrule_model::decoder::GenericDecoderRunner<
        ferrule_model::decoder::CpuPagedKvBackend,
    >,
    states: &mut [ferrule_model::decoder::GenericDecoderSequenceState],
    transaction: ExecutionTransactionId,
    batch: &ExecutionBatch,
    reservations: &[KvReservationView],
) -> ferrule_common::execution::ExecutionOutput {
    runner
        .prepare_multi_session_batch(transaction, states, batch, reservations)
        .unwrap();
    match runner
        .execute_multi_session_batch_progress(transaction, states, batch)
        .unwrap()
    {
        MultiSessionBatchProgress::Complete(output) => output,
        MultiSessionBatchProgress::Waiting(_) => {
            panic!("standard CPU decoder unexpectedly suspended")
        }
    }
}

fn publish(
    runner: &mut ferrule_model::decoder::GenericDecoderRunner<
        ferrule_model::decoder::CpuPagedKvBackend,
    >,
    states: &mut [ferrule_model::decoder::GenericDecoderSequenceState],
    transaction: ExecutionTransactionId,
) {
    assert_eq!(
        runner
            .end_transaction(transaction, states, TransactionEndIntent::Publish)
            .unwrap(),
        TransactionEndProgress::Complete
    );
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{}-{nonce}", std::process::id()))
}

fn full_prefill(fixture: &DenseFixture) -> LogitsOutput {
    let adapter = Qwen3DenseAdapter::load_hf_with_options(
        &fixture.dir,
        1024 * 1024,
        Qwen3DensePrepareOptions {
            page_size: 2,
            ..Default::default()
        },
    )
    .unwrap();
    let mut runner = adapter.into_decoder(8, 3, 1).unwrap();
    runner.configure_kv_page_capacity(2).unwrap();
    let mut states = vec![runner.create_sequence_state().unwrap()];
    let slot = StateSlot::new(0);
    let batch = ExecutionBatch::new(
        ForwardMode::Prefill,
        vec![1, 2, 3],
        vec![0, 1, 2],
        vec![
            Some(KvWriteSlot::new(20)),
            Some(KvWriteSlot::new(21)),
            Some(KvWriteSlot::new(22)),
        ],
        vec![
            LogitsRequest::None,
            LogitsRequest::None,
            LogitsRequest::TopK(NonZeroU32::new(8).unwrap()),
        ],
        vec![ExecutionSequence::new(
            slot,
            ForwardPhase::Prefill,
            0..3,
            0,
            3,
            0..2,
        )],
        vec![KvBlockId::new(10), KvBlockId::new(11)],
    );
    let reservations = vec![KvReservationView {
        state_slot: slot,
        execution_state_slot: slot,
        positions: 0..3,
        newly_allocated: vec![KvPageId(10), KvPageId(11)],
        generation: states[0].core().generation(),
        execution_generation: states[0].core().generation(),
        cow_replacement: None,
    }];
    let transaction = ExecutionTransactionId::new(1).unwrap();
    let result = execute(&mut runner, &mut states, transaction, &batch, &reservations);
    publish(&mut runner, &mut states, transaction);
    result.logits[0].logits.clone()
}

fn assert_same_logits(left: &LogitsOutput, right: &LogitsOutput) {
    let (LogitsOutput::TopK(left), LogitsOutput::TopK(right)) = (left, right) else {
        panic!("expected top-k logits");
    };
    assert_eq!(left.len(), right.len());
    for item in left {
        let expected = right
            .iter()
            .find(|other| item.token_id == other.token_id)
            .unwrap();
        assert!(
            (item.logit - expected.logit).abs() < 1e-5,
            "{item:?} != {expected:?}"
        );
    }
}

#[test]
fn qwen3_dense_tied_head_and_all_dense_roles_bind() {
    let fixture = DenseFixture::new();
    let mut tensors = dense_tensor_shapes();
    tensors.insert("lm_head.weight".into(), vec![8, 4]);
    write_safetensors(&fixture.dir.join("model.safetensors"), &tensors, true);
    let recipe = Qwen3DenseRecipe::new();
    let resources = DecoderLoadOptions::new(&recipe, &fixture.config)
        .open_hf(&fixture.dir, ModelFamily::Qwen3)
        .unwrap();
    assert_eq!(resources.experts().count(), 0);
    let embedding = resources
        .require_static(TensorRole::TokenEmbedding)
        .unwrap();
    let head = resources.require_static(TensorRole::OutputHead).unwrap();
    assert!(head.is_alias());
    assert!(head.shares_storage_with(embedding));
    assert_eq!(head.weight().slice().name, "model.embed_tokens.weight");
    let executable = resources.prepared_executable(1).unwrap();
    let embedding_resource =
        ferrule_model::transformer::parameter_resource_id(embedding.canonical_id());
    let norm_resource = ferrule_model::transformer::parameter_resource_id(
        resources
            .require_static(TensorRole::OutputNorm)
            .unwrap()
            .canonical_id(),
    );
    let output_stage = executable
        .stages()
        .iter()
        .find(|stage| stage.operation() == &ferrule_model::execution::TransformerStage::Output)
        .unwrap();
    assert_eq!(
        executable.stages()[0].resources()[0].resource(),
        embedding_resource
    );
    assert_eq!(output_stage.resources().len(), 2);
    for resource in [embedding_resource, norm_resource] {
        assert_eq!(
            output_stage
                .resources()
                .iter()
                .filter(|usage| usage.resource() == resource)
                .count(),
            1
        );
        assert!(executable.resource(resource).is_some());
    }
    assert!(
        output_stage
            .resources()
            .iter()
            .all(|usage| usage.access() == ferrule_model::execution::ResourceAccess::Read)
    );
    for (role, shape) in [
        (TensorRole::AttentionQuery, vec![8, 4]),
        (TensorRole::AttentionKey, vec![4, 4]),
        (TensorRole::AttentionValue, vec![4, 4]),
        (TensorRole::AttentionOutput, vec![4, 8]),
        (TensorRole::AttentionQueryNorm, vec![2]),
        (TensorRole::AttentionKeyNorm, vec![2]),
        (TensorRole::DenseMlpGate, vec![6, 4]),
        (TensorRole::DenseMlpUp, vec![6, 4]),
        (TensorRole::DenseMlpDown, vec![4, 6]),
    ] {
        resources.require_layer_shape(0, role, &shape).unwrap();
    }
}

#[test]
fn qwen3_dense_header_discovery_does_not_require_payload() {
    let fixture = DenseFixture::new();
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(fixture.dir.join("model.safetensors"))
        .unwrap();
    let inventory = HfSafetensorsInventory::open(&fixture.dir, ModelFamily::Qwen3).unwrap();
    let data_start = inventory
        .tensors
        .iter()
        .map(|tensor| tensor.file_offset - tensor.data_offset)
        .min()
        .unwrap();
    file.set_len(data_start).unwrap();
    // The file now physically contains only the header, not zero-filled weights.
    let inventory = HfSafetensorsInventory::open(&fixture.dir, ModelFamily::Qwen3).unwrap();
    assert_eq!(inventory.tensor_count, 13);
    assert_eq!(inventory.shard_count, 1);
    assert!(inventory.index_only_tensors.is_empty());
    assert!(inventory.header_only_tensors.is_empty());
    assert!(!fixture.dir.join("model.safetensors.index.json").exists());
    // Binding still validates source extents, rather than pretending truncated weights are loadable.
    assert!(
        DecoderLoadOptions::new(&Qwen3DenseRecipe::new(), &fixture.config)
            .open_hf(&fixture.dir, ModelFamily::Qwen3)
            .is_err()
    );
}

#[test]
fn qwen3_dense_existing_index_remains_authoritative_and_strict() {
    let fixture = DenseFixture::new();
    let recipe = Qwen3DenseRecipe::new();
    let load = DecoderLoadOptions::new(&recipe, &fixture.config);
    let index_path = fixture.dir.join("model.safetensors.index.json");
    std::fs::write(&index_path, "not-json").unwrap();
    assert!(load.open_hf(&fixture.dir, ModelFamily::Qwen3).is_err());
    assert!(HfSafetensorsInventory::open(&fixture.dir, ModelFamily::Qwen3).is_err());
    std::fs::write(
        &index_path,
        r#"{"weight_map":{"model.embed_tokens.weight":"missing.safetensors"}}"#,
    )
    .unwrap();
    assert!(
        load.open_hf(&fixture.dir, ModelFamily::Qwen3)
            .unwrap_err()
            .to_string()
            .contains("missing shards")
    );
    std::fs::write(
        &index_path,
        r#"{"weight_map":{"model.embed_tokens.weight":"model.safetensors"}}"#,
    )
    .unwrap();
    assert!(
        load.open_hf(&fixture.dir, ModelFamily::Qwen3)
            .unwrap_err()
            .to_string()
            .contains("index/header mismatch")
    );
    let weight_map = dense_tensor_shapes()
        .into_keys()
        .map(|name| (name, "model.safetensors"))
        .collect::<BTreeMap<_, _>>();
    std::fs::write(
        &index_path,
        serde_json::json!({"weight_map": weight_map}).to_string(),
    )
    .unwrap();
    let checkpoint = load
        .open_hf_checkpoint(&fixture.dir, ModelFamily::Qwen3)
        .unwrap();
    assert!(checkpoint.index().is_some());
    assert_eq!(checkpoint.inventory().tensor_count, 13);
}

#[test]
fn qwen3_dense_rejects_unsupported_configs_names_and_cuda() {
    for (field, value) in [
        ("model_type", serde_json::json!("qwen3_moe")),
        ("architectures", serde_json::json!(["Qwen3MoeForCausalLM"])),
        ("num_experts", serde_json::json!(8)),
        ("torch_dtype", serde_json::json!("float16")),
        ("num_key_value_heads", serde_json::json!(3)),
        ("head_dim", serde_json::json!(3)),
        (
            "rope_scaling",
            serde_json::json!({"rope_type":"linear", "factor":2.0}),
        ),
        ("attention_bias", serde_json::json!(true)),
        ("use_sliding_window", serde_json::json!(true)),
    ] {
        let mut config = small_dense_config();
        config[field] = value;
        assert!(
            Qwen3DenseRecipe::new().build(&config).is_err(),
            "accepted {field}"
        );
    }
    let output = Qwen3DenseRecipe::new()
        .build(&small_dense_config())
        .unwrap();
    for name in [
        "model.layers.1.mlp.up_proj.weight",
        "model.layers.x.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.self_attn.q_norm.bias",
    ] {
        assert!(
            output.name_mapper().map(name, meta()).is_err(),
            "accepted {name}"
        );
    }
    let missing_dir = unique_temp_dir("ferrule-qwen3-dense-not-created");
    let error = Qwen3DenseAdapter::load_hf_with_options_and_backend(
        &missing_dir,
        1024,
        Qwen3DensePrepareOptions::default(),
        ModelExecutionBackend::Cuda,
    )
    .err()
    .unwrap();
    assert!(error.to_string().contains("CUDA is unsupported"));
}

#[test]
#[ignore = "metadata only: set FERRULE_QWEN3_DENSE_DIR explicitly to a local/NAS Qwen3-0.6B directory"]
fn qwen3_dense_real_06b_metadata_only_opt_in() {
    let dir = PathBuf::from(
        std::env::var_os("FERRULE_QWEN3_DENSE_DIR").expect("set FERRULE_QWEN3_DENSE_DIR"),
    );
    let value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
    let checkpoint = DecoderLoadOptions::new(&Qwen3DenseRecipe::new(), &value)
        .open_hf_checkpoint(&dir, ModelFamily::Qwen3)
        .unwrap();
    assert!(checkpoint.index().is_none());
    assert_eq!(checkpoint.inventory().tensor_count, 311);
    assert_eq!(checkpoint.resources().state_dict().len(), 311);
    assert_eq!(checkpoint.resources().spec().layers().len(), 28);
    assert_eq!(checkpoint.resources().spec().hidden_size(), 1024);
    assert!(checkpoint.resources().spec().tie_word_embeddings());
    assert!(
        checkpoint
            .resources()
            .state_dict()
            .validate_source_identities()
    );
}

#[test]
#[ignore = "opt-in NAS run: full 28-layer Qwen3-0.6B CPU prefill/decode"]
fn real_full_dense_prefill_decode() {
    let started = std::time::Instant::now();
    eprintln!("loading full 28-layer Qwen3-0.6B CPU checkpoint");
    let model_dir = PathBuf::from("/mnt/nas1/hf/Qwen3-0.6B");
    assert!(model_dir.join("config.json").is_file());
    assert!(model_dir.join("model.safetensors").is_file());
    assert!(!model_dir.join("model.safetensors.index.json").exists());

    let adapter = Qwen3DenseAdapter::load_hf_with_options(
        &model_dir,
        1024 * 1024 * 1024,
        Qwen3DensePrepareOptions {
            page_size: 16,
            max_dense_tensor_bytes: 1024 * 1024 * 1024,
            max_layers: None,
        },
    )
    .unwrap();
    assert_eq!(adapter.config().num_hidden_layers, 28);
    assert_eq!(adapter.config().vocab_size, 151_936);
    let tokens = vec![9707, 11, 1879];
    assert!(
        tokens
            .iter()
            .all(|id| (*id as usize) < adapter.config().vocab_size)
    );

    let mut runner = adapter.into_decoder(64, 3, 1).unwrap();
    eprintln!("all 28 layers materialized in {:?}", started.elapsed());
    assert_eq!(runner.bound_layer_count(), Some(28));
    assert_eq!(runner.model_info().num_layers, 28);
    runner.configure_kv_page_capacity(1).unwrap();
    let mut states = vec![runner.create_sequence_state().unwrap()];
    let slot = StateSlot::new(0);
    let first_page = KvPageId(20);
    let prefill = ExecutionBatch::new(
        ForwardMode::Prefill,
        tokens,
        vec![0, 1, 2],
        vec![
            Some(KvWriteSlot::new(first_page.0 * 16)),
            Some(KvWriteSlot::new(first_page.0 * 16 + 1)),
            Some(KvWriteSlot::new(first_page.0 * 16 + 2)),
        ],
        vec![
            LogitsRequest::None,
            LogitsRequest::None,
            LogitsRequest::Full,
        ],
        vec![ExecutionSequence::new(
            slot,
            ForwardPhase::Prefill,
            0..3,
            0,
            3,
            0..1,
        )],
        vec![KvBlockId::new(first_page.0)],
    );
    let reservations = vec![KvReservationView {
        state_slot: slot,
        execution_state_slot: slot,
        positions: 0..3,
        newly_allocated: vec![first_page],
        generation: states[0].core().generation(),
        execution_generation: states[0].core().generation(),
        cow_replacement: None,
    }];
    let transaction = ExecutionTransactionId::new(100).unwrap();
    let output = execute(
        &mut runner,
        &mut states,
        transaction,
        &prefill,
        &reservations,
    );
    let mut next_token = finite_real_argmax(&output);
    eprintln!(
        "prefill complete in {:?}; next token {next_token}",
        started.elapsed()
    );
    assert_eq!(states[0].core().position(), 0);
    publish(&mut runner, &mut states, transaction);
    assert_eq!(states[0].core().position(), 3);

    for step in 0usize..2 {
        let position = 3 + step;
        assert!(next_token < 151_936);
        let transaction = ExecutionTransactionId::new(101 + step as u64).unwrap();
        let batch = ExecutionBatch::new(
            ForwardMode::Decode,
            vec![next_token],
            vec![position as u32],
            vec![Some(KvWriteSlot::new(first_page.0 * 16 + position as u32))],
            vec![LogitsRequest::Full],
            vec![ExecutionSequence::new(
                slot,
                ForwardPhase::Decode,
                0..1,
                position as u32,
                (position + 1) as u32,
                0..1,
            )],
            vec![KvBlockId::new(first_page.0)],
        );
        let reservations = vec![KvReservationView {
            state_slot: slot,
            execution_state_slot: slot,
            positions: position..position + 1,
            newly_allocated: Vec::new(),
            generation: states[0].core().generation(),
            execution_generation: states[0].core().generation(),
            cow_replacement: None,
        }];
        let output = execute(&mut runner, &mut states, transaction, &batch, &reservations);
        next_token = finite_real_argmax(&output);
        eprintln!(
            "decode {} complete in {:?}; next token {next_token}",
            step + 1,
            started.elapsed()
        );
        assert_eq!(states[0].core().position(), position);
        publish(&mut runner, &mut states, transaction);
        assert_eq!(states[0].core().position(), position + 1);
        assert_eq!(runner.observability_snapshot().resident_kv_pages, 1);
    }
    let snapshot = runner.observability_snapshot();
    assert_eq!(snapshot.bound_layers, 28);
    assert_eq!(snapshot.completed_batches, 3);
    assert_eq!(snapshot.active_transactions, 0);
}

fn finite_real_argmax(output: &ferrule_common::execution::ExecutionOutput) -> u32 {
    assert_eq!(output.logits.len(), 1);
    let LogitsOutput::Full(logits) = &output.logits[0].logits else {
        panic!("real dense execution must return the complete vocabulary logits")
    };
    assert_eq!(logits.len(), 151_936);
    assert!(logits.iter().all(|logit| logit.is_finite()));
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap()
        .0 as u32
}

#[test]
fn qwen3_dense_untied_head_binds_and_incomplete_artifacts_fail() {
    let mut fixture = DenseFixture::new();
    fixture.config["tie_word_embeddings"] = false.into();
    let mut tensors = dense_tensor_shapes();
    tensors.insert("lm_head.weight".into(), vec![8, 4]);
    write_safetensors(&fixture.dir.join("model.safetensors"), &tensors, true);
    let recipe = Qwen3DenseRecipe::new();
    let load = DecoderLoadOptions::new(&recipe, &fixture.config);
    let resources = load.open_hf(&fixture.dir, ModelFamily::Qwen3).unwrap();
    assert!(
        !resources
            .require_static(TensorRole::OutputHead)
            .unwrap()
            .is_alias()
    );
    for name in [
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ] {
        let mut missing = tensors.clone();
        missing.remove(name);
        write_safetensors(&fixture.dir.join("model.safetensors"), &missing, true);
        assert!(
            load.open_hf(&fixture.dir, ModelFamily::Qwen3).is_err(),
            "accepted missing {name}"
        );
    }
    tensors.insert("model.layers.0.self_attn.q_proj.weight".into(), vec![4, 4]);
    write_safetensors(&fixture.dir.join("model.safetensors"), &tensors, true);
    assert!(load.open_hf(&fixture.dir, ModelFamily::Qwen3).is_err());
}

#[test]
fn qwen3_dense_load_limits_and_layer_bounds_are_enforced() {
    let fixture = DenseFixture::new();
    for max_layers in [Some(0), Some(2)] {
        assert!(
            Qwen3DenseAdapter::load_hf_with_options(
                &fixture.dir,
                1024 * 1024,
                Qwen3DensePrepareOptions {
                    max_layers,
                    ..Default::default()
                }
            )
            .is_err()
        );
    }
    assert!(
        Qwen3DenseAdapter::load_hf_with_options(
            &fixture.dir,
            1,
            Qwen3DensePrepareOptions::default()
        )
        .is_err()
    );
    let adapter = Qwen3DenseAdapter::load_hf_with_options(
        &fixture.dir,
        1024 * 1024,
        Qwen3DensePrepareOptions::default(),
    )
    .unwrap();
    assert!(adapter.into_decoder(9, 1, 1).is_err());
}

#[test]
fn qwen3_dense_redundant_tied_head_is_validated_not_a_canonical_fallback() {
    let fixture = DenseFixture::new();
    let recipe = Qwen3DenseRecipe::new();
    let load = DecoderLoadOptions::new(&recipe, &fixture.config);
    let mut tensors = dense_tensor_shapes();
    tensors.insert("lm_head.weight".into(), vec![8, 4]);
    write_safetensors(&fixture.dir.join("model.safetensors"), &tensors, true);
    let checkpoint = load
        .open_hf_checkpoint(&fixture.dir, ModelFamily::Qwen3)
        .unwrap();
    assert_eq!(checkpoint.inventory().tensor_count, 14);
    assert_eq!(checkpoint.resources().state_dict().len(), 14);
    let slices = checkpoint
        .inventory()
        .tensors
        .iter()
        .map(|tensor| ferrule_model::CheckpointTensorSlice::from_hf_inventory(&fixture.dir, tensor))
        .collect::<Vec<_>>();
    for name in ["lm_head.weight", "model.embed_tokens.weight"] {
        let mut duplicate = slices.clone();
        duplicate.push(
            slices
                .iter()
                .find(|slice| slice.name == name)
                .unwrap()
                .clone(),
        );
        assert!(load.bind_slices(duplicate).is_err());
    }
    for invalid in [
        slices
            .iter()
            .filter(|slice| slice.name != "model.embed_tokens.weight")
            .cloned()
            .collect::<Vec<_>>(),
        slices
            .iter()
            .cloned()
            .map(|mut slice| {
                if slice.name == "lm_head.weight" {
                    slice.dtype = ferrule_model::CheckpointDType::F32;
                }
                slice
            })
            .collect(),
        slices
            .iter()
            .cloned()
            .map(|mut slice| {
                if slice.name == "lm_head.weight" {
                    slice.offset = u64::MAX;
                }
                slice
            })
            .collect(),
    ] {
        assert!(load.bind_slices(invalid).is_err());
    }
    tensors.insert("lm_head.weight".into(), vec![4, 8]);
    write_safetensors(&fixture.dir.join("model.safetensors"), &tensors, true);
    assert!(load.open_hf(&fixture.dir, ModelFamily::Qwen3).is_err());
}
