//! Generic hybrid decoder checked against Transformers 5.2 F32 eager.
use ferrule_common::execution::*;
use ferrule_model::decoder::*;
use ferrule_model::execution::ExecutionPrecisionPolicy;
use ferrule_model::nn::{
    DTypeConstraint, ModulePath, ParameterDType, ParameterId, ParameterResidency, ParameterSpec,
};
use ferrule_model::runner::{
    MultiSessionBatchProgress, MultiSessionRunner, TransactionEndIntent, TransactionEndProgress,
};
use ferrule_model::tokenizer::TokenizerHandle;
use ferrule_model::transformer::*;
use ferrule_model::{ModelFamily, TensorRole, WeightSource};
use serde_json::Value;
use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

type Runner = GenericDecoderRunner<HybridCpuDecoder>;
fn fixture_json() -> Value {
    serde_json::from_str(include_str!("fixtures/hybrid_cpu/oracle.json")).unwrap()
}
pub(crate) fn floats(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}
fn spec() -> DecoderModelSpec {
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
                FeedForward::SwiGlu(SwiGlu::new(8, 12, false).unwrap()),
                Residual::Add,
            )
            .unwrap()
        })
        .collect();
    DecoderModelSpec::new(DecoderModelParts {
        architecture: "synthetic-hybrid".into(),
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
fn role(name: &str) -> TensorRole {
    match name {
        "token_embedding.weight" => return TensorRole::TokenEmbedding,
        "final_norm.weight" => return TensorRole::OutputNorm,
        "output.weight" => return TensorRole::OutputHead,
        _ => (),
    }
    let suffix = name.split('.').skip(2).collect::<Vec<_>>().join(".");
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
        "feed_forward.gate.weight" => TensorRole::DenseMlpGate,
        "feed_forward.up.weight" => TensorRole::DenseMlpUp,
        "feed_forward.down.weight" => TensorRole::DenseMlpDown,
        _ => panic!("unexpected fixture parameter {name}"),
    }
}
pub(crate) struct Fixture {
    pub(crate) dir: PathBuf,
    pub(crate) oracle: Value,
}
impl Fixture {
    pub(crate) fn new() -> Self {
        Self::with_oracle(fixture_json())
    }
    pub(crate) fn with_oracle(oracle: Value) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "ferrule-hybrid-{}-{}",
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
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(payload);
        std::fs::write(dir.join("model.safetensors"), bytes).unwrap();
        let mut tokenizer = tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default());
        tokenizer
            .add_tokens((0..11).map(|i| tokenizers::AddedToken::from(format!("t{i}"), false)))
            .unwrap();
        tokenizer.save(dir.join("tokenizer.json"), false).unwrap();
        Self { dir, oracle }
    }
    pub(crate) fn resources(&self) -> BoundDecoderResources {
        self.resources_for(spec())
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
            let residency = if name.starts_with("layers.") {
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
    pub(crate) fn runner(&self) -> Runner {
        let resources = self.resources();
        let options = GenericDecoderOptions::standard_cpu(
            resources.spec(),
            ModelFamily::Unknown("synthetic-hybrid".into()),
            WeightSource::Safetensors,
            2,
            32,
            32,
            4,
            1 << 20,
            ExecutionPrecisionPolicy::f32(),
        )
        .unwrap();
        let mut runner = Runner::hybrid_cpu(
            resources,
            TokenizerHandle::load(&self.dir).unwrap(),
            options,
        )
        .unwrap();
        runner.configure_kv_page_capacity(64).unwrap();
        runner
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}
pub(crate) fn close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (a - b).abs() <= 2e-5 + 3e-4 * b.abs(),
            "{label}[{i}]: actual {a}, expected {b}"
        );
    }
}
fn assert_states(state: &HybridDecoderSequenceState, expected: &Value) {
    for (i, layer) in state.kv_state().layers().iter().enumerate() {
        if let HybridLayerState::GatedDeltaNet(s) = layer {
            assert_eq!(s.position(), state.core().position());
            close(
                s.conv_history(),
                &floats(&expected[i.to_string()]["conv"]),
                "raw conv history",
            );
            close(
                s.recurrent(),
                &floats(&expected[i.to_string()]["recurrent"]),
                "recurrent",
            );
        }
    }
}

pub(crate) fn step(
    runner: &mut Runner,
    states: &mut [HybridDecoderSequenceState],
    pages: &mut [Vec<KvPageId>],
    inputs: &[(&[u32], ForwardPhase)],
    id: u64,
    publish: bool,
) -> Vec<Vec<f32>> {
    let original = states.to_vec();
    let mut staged_pages = pages.to_vec();
    let mut tokens = Vec::new();
    let mut positions = Vec::new();
    let mut writes = Vec::new();
    let mut sequences = Vec::new();
    let mut blocks = Vec::new();
    let mut reservations = Vec::new();
    for (i, (input, phase)) in inputs.iter().enumerate() {
        let context = states[i].core().position();
        let start = tokens.len();
        let block_start = blocks.len();
        let mut new_pages = Vec::new();
        while staged_pages[i].len() < (context + input.len()).div_ceil(2) {
            let page = KvPageId((id * 64 + i as u64 * 16 + staged_pages[i].len() as u64) as u32);
            staged_pages[i].push(page);
            new_pages.push(page);
        }
        for (offset, &token) in input.iter().enumerate() {
            let position = context + offset;
            tokens.push(token);
            positions.push(position as u32);
            writes.push(Some(KvWriteSlot::new(
                staged_pages[i][position / 2].0 * 2 + (position % 2) as u32,
            )));
        }
        blocks.extend(staged_pages[i].iter().map(|page| KvBlockId::new(page.0)));
        sequences.push(ExecutionSequence::new(
            StateSlot::new(i as u32),
            *phase,
            start as u32..tokens.len() as u32,
            context as u32,
            (context + input.len()) as u32,
            block_start as u32..blocks.len() as u32,
        ));
        reservations.push(KvReservationView {
            state_slot: StateSlot::new(i as u32),
            execution_state_slot: StateSlot::new(i as u32),
            positions: context..(context + input.len()),
            newly_allocated: new_pages,
            generation: states[i].core().generation(),
            execution_generation: states[i].core().generation(),
            cow_replacement: None,
        });
    }
    let prefill = inputs.iter().any(|(_, p)| *p == ForwardPhase::Prefill);
    let decode = inputs.iter().any(|(_, p)| *p == ForwardPhase::Decode);
    let mode = match (prefill, decode) {
        (true, true) => ForwardMode::Mixed,
        (true, false) => ForwardMode::Prefill,
        _ => ForwardMode::Decode,
    };
    let logits = vec![LogitsRequest::Full; tokens.len()];
    let batch = ExecutionBatch::new(mode, tokens, positions, writes, logits, sequences, blocks);
    let tx = ExecutionTransactionId::new(id).unwrap();
    runner
        .prepare_multi_session_batch(tx, states, &batch, &reservations)
        .unwrap();
    let output = match runner
        .execute_multi_session_batch_progress(tx, states, &batch)
        .unwrap()
    {
        MultiSessionBatchProgress::Complete(output) => output,
        _ => panic!("CPU hybrid unexpectedly suspended"),
    };
    assert_eq!(
        states, original,
        "forward must only mutate transaction working copies"
    );
    if !publish {
        let executed = inputs
            .iter()
            .map(|(rows, _)| rows.len())
            .collect::<Vec<_>>();
        let retained = vec![1; executed.len()];
        let error = runner
            .retain_provisional_prefixes(tx, &original, states, &executed, &retained)
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("no provisional proposal executor")
        );
        assert_eq!(
            states, original,
            "unsupported retain must not mutate any branch"
        );
    }
    assert_eq!(
        runner
            .end_transaction(
                tx,
                states,
                if publish {
                    TransactionEndIntent::Publish
                } else {
                    TransactionEndIntent::Abort
                }
            )
            .unwrap(),
        TransactionEndProgress::Complete
    );
    if publish {
        pages.clone_from_slice(&staged_pages);
    } else {
        assert_eq!(states, original, "abort must preserve all committed states");
    }
    output
        .logits
        .into_iter()
        .map(|row| match row.logits {
            LogitsOutput::Full(values) => values,
            _ => panic!("expected full logits"),
        })
        .collect()
}

#[test]
fn hybrid_full_prefill_matches_transformers_f32_logits_and_states() {
    let fixture = Fixture::new();
    assert_eq!(fixture.oracle["transformers"], "5.2.0");
    let schema = HybridStateSchema::from_spec(&spec(), 4).unwrap();
    let planes = schema.kv_planes(2, 32).unwrap();
    assert_eq!(KvLayoutSchema::planes(&planes).len(), 2);
    assert_eq!(
        KvLayoutSchema::planes(&planes)[0].layer_count,
        2,
        "linear layers must not allocate KV"
    );
    assert!(matches!(
        schema.layers()[1],
        HybridLayerSchema::FullAttention { kv_layer: 0, .. }
    ));
    assert!(matches!(
        schema.layers()[3],
        HybridLayerSchema::FullAttention { kv_layer: 1, .. }
    ));
    let mut runner = fixture.runner();
    let mut states = vec![runner.create_sequence_state().unwrap()];
    let output = step(
        &mut runner,
        &mut states,
        &mut [Vec::new()],
        &[(&[1, 3, 2, 5, 7], ForwardPhase::Prefill)],
        1,
        true,
    );
    for (row, expected) in output
        .iter()
        .zip(fixture.oracle["logits"].as_array().unwrap())
    {
        close(row, &floats(expected), "HF full prefill logits");
    }
    assert_states(&states[0], &fixture.oracle["states"]);
}

#[test]
fn hybrid_chunking_decode_abort_fork_reset_and_release_share_the_frontier() {
    let fixture = Fixture::new();
    let mut runner = fixture.runner();
    let mut states = vec![runner.create_sequence_state().unwrap()];
    let mut pages = vec![Vec::new()];
    step(
        &mut runner,
        &mut states,
        &mut pages,
        &[(&[1, 3], ForwardPhase::Prefill)],
        1,
        true,
    );
    step(
        &mut runner,
        &mut states,
        &mut pages,
        &[(&[2], ForwardPhase::Prefill)],
        2,
        true,
    );
    assert_states(&states[0], &fixture.oracle["prefill_states"]);
    step(
        &mut runner,
        &mut states,
        &mut pages,
        &[(&[9, 8], ForwardPhase::Prefill)],
        3,
        false,
    );
    assert_states(&states[0], &fixture.oracle["prefill_states"]);
    let fourth = step(
        &mut runner,
        &mut states,
        &mut pages,
        &[(&[5], ForwardPhase::Decode)],
        4,
        true,
    );
    close(
        &fourth[0],
        &floats(&fixture.oracle["logits"][3]),
        "fourth token",
    );
    let parent = states[0].clone();
    let child = runner.fork_sequence_state_from(&parent, 4).unwrap();
    assert_ne!(parent.topology_id(), child.topology_id());
    assert_eq!(parent.kv_state(), child.kv_state());
    assert!(runner.fork_sequence_state_from(&parent, 3).is_err());
    let mut branches = vec![child];
    let mut branch_pages = pages.clone();
    let fifth = step(
        &mut runner,
        &mut branches,
        &mut branch_pages,
        &[(&[7], ForwardPhase::Decode)],
        5,
        true,
    );
    close(
        &fifth[0],
        &floats(&fixture.oracle["logits"][4]),
        "fork decode",
    );
    assert_states(&branches[0], &fixture.oracle["decoded_states"]);
    assert_eq!(
        states[0], parent,
        "fork must not alias conv or recurrent memory"
    );
    runner.reset_sequence_state(&mut branches[0]).unwrap();
    assert_eq!(branches[0].core().position(), 0);
    let fresh = runner.create_sequence_state().unwrap();
    assert_eq!(branches[0].kv_state(), fresh.kv_state());
    runner
        .try_release_sequence_state(branches.pop().unwrap())
        .unwrap();
    runner.try_release_sequence_state(fresh).unwrap();
    runner
        .try_release_sequence_state(states.pop().unwrap())
        .unwrap();
}

#[test]
fn hybrid_ragged_mixed_batch_matches_independent_sequences() {
    let fixture = Fixture::new();
    let mut runner = fixture.runner();
    let mut states = vec![
        runner.create_sequence_state().unwrap(),
        runner.create_sequence_state().unwrap(),
    ];
    let mut pages = vec![Vec::new(), Vec::new()];
    step(
        &mut runner,
        &mut states,
        &mut pages,
        &[
            (&[1, 3], ForwardPhase::Prefill),
            (&[1], ForwardPhase::Prefill),
        ],
        1,
        true,
    );
    let output = step(
        &mut runner,
        &mut states,
        &mut pages,
        &[
            (&[2], ForwardPhase::Decode),
            (&[3, 2], ForwardPhase::Prefill),
        ],
        2,
        true,
    );
    close(
        &output[0],
        &floats(&fixture.oracle["logits"][2]),
        "mixed decode",
    );
    close(
        &output[1],
        &floats(&fixture.oracle["logits"][1]),
        "mixed prefill 1",
    );
    close(
        &output[2],
        &floats(&fixture.oracle["logits"][2]),
        "mixed prefill 2",
    );
    assert_states(&states[0], &fixture.oracle["prefill_states"]);
    assert_eq!(states[0].kv_state(), states[1].kv_state());
}

#[test]
fn hybrid_rejects_bf16_before_execution_and_invalid_shapes() {
    let fixture = Fixture::new();
    let resources = fixture.resources();
    let options = GenericDecoderOptions::standard_cpu(
        resources.spec(),
        ModelFamily::Unknown("synthetic".into()),
        WeightSource::Safetensors,
        2,
        32,
        32,
        4,
        1 << 20,
        ExecutionPrecisionPolicy::bf16_compatibility(),
    )
    .unwrap();
    let error = Runner::hybrid_cpu(
        resources,
        TokenizerHandle::load(&fixture.dir).unwrap(),
        options,
    )
    .err()
    .unwrap();
    let ferrule_common::Error::ModelSource { source } = error else {
        panic!("expected typed unsupported")
    };
    assert!(source.downcast_ref::<UnsupportedOperator>().is_some());
    assert!(GatedDeltaNetAttention::new(8, 3, 2, 4, 4, 3, 1e-6, false).is_err());
    assert!(GatedDeltaNetAttention::new(8, 1, 2, 4, 4, 0, 1e-6, false).is_err());
    assert!(GatedDeltaNetAttention::new(8, 1, 2, usize::MAX, 4, 3, 1e-6, false).is_err());
    assert!(
        HybridStateSchema::from_spec(&spec(), 1)
            .unwrap()
            .kv_planes(2, 32)
            .is_err()
    );
}

#[test]
fn gated_query_is_head_interleaved_and_partial_rope_preserves_tail() {
    let mut operators = CpuStandardDecoderOperators::default();
    let input = Rows::Host(
        HostRows::new(
            RowsShape::new(1, 1024).unwrap(),
            RowsDType::F32,
            None,
            (0..1024).map(|i| i as f32 * 0.01).collect(),
        )
        .unwrap(),
    );
    let OperatorProgress::Ready((query, gate)) =
        operators.unpack_gated_query(input, 2, 256).unwrap()
    else {
        panic!("unpack unsupported")
    };
    assert_eq!(query.host().unwrap().values()[256], 5.12);
    assert_eq!(gate.host().unwrap().values()[0], 2.56);
    assert_eq!(gate.host().unwrap().values()[256], 7.68);
    let before = query.host().unwrap().values().to_vec();
    let rope = RotaryEmbedding::new(
        256,
        10000.0,
        RotaryPairing::SplitHalf,
        RotaryRegion::Prefix { dimensions: 64 },
        RotaryScaling::None,
    )
    .unwrap();
    let cosine: Vec<f32> = (0..32)
        .map(|pair| (10000.0f32.powf(-(2 * pair) as f32 / 64.0)).cos())
        .collect();
    let sine: Vec<f32> = (0..32)
        .map(|pair| (10000.0f32.powf(-(2 * pair) as f32 / 64.0)).sin())
        .collect();
    let table = PreparedRope::new(1, 64, cosine, sine).unwrap();
    let OperatorProgress::Ready(rotated) = operators.rope(&rope, &table, query, 2, &[0]).unwrap()
    else {
        panic!("rope unsupported")
    };
    let rotated = rotated.host().unwrap().values();
    for head in 0..2 {
        assert_eq!(
            &rotated[head * 256 + 64..(head + 1) * 256],
            &before[head * 256 + 64..(head + 1) * 256]
        );
        close(
            &[rotated[head * 256]],
            &[before[head * 256] * 1.0f32.cos() - before[head * 256 + 32] * 1.0f32.sin()],
            "partial split-half",
        );
    }
    let values = Rows::Host(
        HostRows::new(
            RowsShape::new(1, 512).unwrap(),
            RowsDType::F32,
            None,
            vec![1.0; 512],
        )
        .unwrap(),
    );
    let OperatorProgress::Ready(gated) = operators.sigmoid_gate(values, &gate).unwrap() else {
        panic!("gate unsupported")
    };
    close(
        &[gated.host().unwrap().values()[0]],
        &[1.0 / (1.0 + (-2.56f32).exp())],
        "sigmoid gate",
    );
}

#[test]
fn unsupported_segment_and_empty_state_fail_before_payload_reads() {
    use std::sync::Arc;
    let fixture = Fixture::new();
    let resources = Arc::new(fixture.resources());
    std::fs::remove_file(fixture.dir.join("model.safetensors")).unwrap();
    let error = CpuGqaMoeModule::prepare(
        Arc::clone(&resources),
        Arc::new(StateDictMaterializer::new(1 << 20).unwrap()),
        ExecutionPrecisionPolicy::f32(),
        32,
    )
    .err()
    .unwrap();
    let ferrule_common::Error::ModelSource { source } = error else {
        panic!("expected typed unsupported before payload read")
    };
    assert!(source.downcast_ref::<UnsupportedOperator>().is_some());
    let plan = LayerSegmentPlan::new(4, 0..4, true, true).unwrap();
    let error = StandardDecoderSegment::prepare(
        &resources,
        plan,
        ExecutionPrecisionPolicy::f32(),
        32,
        1 << 20,
    )
    .err()
    .unwrap();
    let SegmentError::Execution { source, .. } = error else {
        panic!("expected segment execution error")
    };
    let ferrule_common::Error::ModelSource { source } = source else {
        panic!("expected typed unsupported segment")
    };
    assert!(source.downcast_ref::<UnsupportedOperator>().is_some());
}
