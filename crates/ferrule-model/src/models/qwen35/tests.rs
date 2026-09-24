//! Small BF16/F32 artifact exercises the same recipe, binder, materializer and runner.
//! The dimension override exists only under cfg(test); production remains strict 0.8B.
use super::*;
use crate::models::qwen35::{Qwen35HfNameMapper, Qwen35TensorPartitionKind};
use crate::nn::ParameterDType;
use crate::runner::{
    MultiSessionBatchProgress, MultiSessionRunner, TransactionEndIntent, TransactionEndProgress,
};
use ferrule_common::execution::*;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

struct Fixture {
    dir: PathBuf,
    config: Qwen35Config,
}
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "ferrule-qwen35-tiny-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&dir).unwrap();
        let config = Qwen35Config::tiny_for_test();
        let mapper = Qwen35HfNameMapper::new(&config);
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for (tensor_id, spec) in mapper
            .tensors()
            .filter(|t| t.partition == Qwen35TensorPartitionKind::Text)
            .enumerate()
        {
            let start = payload.len();
            for i in 0..spec.shape.iter().product::<usize>() {
                let value = if spec.external_name.ends_with("linear_attn.norm.weight") {
                    1.0
                } else if spec.external_name.ends_with("A_log") {
                    -0.5
                } else {
                    (((tensor_id * 13 + i * 7) % 31) as f32 - 15.0) * 0.015625
                };
                match spec.dtype {
                    ParameterDType::Bf16 => {
                        payload.extend(half::bf16::from_f32(value).to_bits().to_le_bytes())
                    }
                    ParameterDType::F32 => payload.extend(value.to_le_bytes()),
                    _ => panic!("invalid fixture dtype"),
                }
            }
            header.insert(spec.external_name.clone(), serde_json::json!({"dtype":spec.dtype.as_str(),"shape":spec.shape,"data_offsets":[start,payload.len()]}));
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while !header.len().is_multiple_of(8) {
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
        Self { dir, config }
    }
    fn bound(&self) -> (Qwen35Metadata, BoundDecoderResources) {
        let metadata = Qwen35Metadata::from_config(&self.dir, self.config.clone()).unwrap();
        Qwen35Adapter::bind_metadata(&self.dir, metadata).unwrap()
    }
    fn runner(&self) -> Qwen35CpuRunner {
        let (metadata, resources) = self.bound();
        let adapter = Qwen35Adapter {
            metadata,
            resources,
            tokenizer: TokenizerHandle::load(&self.dir).unwrap(),
            options: Qwen35PrepareOptions {
                page_size: 2,
                max_parameter_bytes: 1 << 20,
            },
        };
        let planes = adapter.state_schema().unwrap().kv_planes(2, 32).unwrap();
        assert_eq!(KvLayoutSchema::planes(&planes)[0].layer_count, 1);
        let mut runner = adapter.into_decoder(32, 16, 1).unwrap();
        runner.configure_kv_page_capacity(16).unwrap();
        runner
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn step(
    runner: &mut Qwen35CpuRunner,
    states: &mut [crate::decoder::HybridDecoderSequenceState],
    pages: &mut Vec<KvPageId>,
    tokens: &[u32],
    id: u64,
    publish: bool,
) -> Vec<Vec<f32>> {
    let before = states.to_vec();
    let start = states[0].core().position();
    let end = start + tokens.len();
    let mut staged = pages.clone();
    let mut newly_allocated = Vec::new();
    while staged.len() < end.div_ceil(2) {
        let p = KvPageId((id * 32 + staged.len() as u64) as u32);
        staged.push(p);
        newly_allocated.push(p);
    }
    let phase = if start == 0 || tokens.len() > 1 {
        ForwardPhase::Prefill
    } else {
        ForwardPhase::Decode
    };
    let batch = ExecutionBatch::new(
        if phase == ForwardPhase::Prefill {
            ForwardMode::Prefill
        } else {
            ForwardMode::Decode
        },
        tokens.to_vec(),
        (start..end).map(|p| p as u32).collect(),
        (start..end)
            .map(|p| Some(KvWriteSlot::new(staged[p / 2].0 * 2 + (p % 2) as u32)))
            .collect(),
        vec![LogitsRequest::Full; tokens.len()],
        vec![ExecutionSequence::new(
            StateSlot::new(0),
            phase,
            0..tokens.len() as u32,
            start as u32,
            end as u32,
            0..staged.len() as u32,
        )],
        staged.iter().map(|p| KvBlockId::new(p.0)).collect(),
    );
    let reservation = KvReservationView {
        state_slot: StateSlot::new(0),
        execution_state_slot: StateSlot::new(0),
        positions: start..end,
        newly_allocated,
        generation: states[0].core().generation(),
        execution_generation: states[0].core().generation(),
        cow_replacement: None,
    };
    let tx = ExecutionTransactionId::new(id).unwrap();
    runner
        .prepare_multi_session_batch(tx, states, &batch, &[reservation])
        .unwrap();
    let MultiSessionBatchProgress::Complete(output) = runner
        .execute_multi_session_batch_progress(tx, states, &batch)
        .unwrap()
    else {
        panic!("CPU suspended")
    };
    assert_eq!(states, before);
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
        *pages = staged;
        assert_eq!(states[0].core().position(), end);
    } else {
        assert_eq!(states, before);
    }
    output
        .logits
        .into_iter()
        .map(|l| match l.logits {
            LogitsOutput::Full(v) => v,
            _ => panic!("full required"),
        })
        .collect()
}

#[test]
fn tiny_bound_alias_roles_and_read_limits() {
    let f = Fixture::new();
    let (m, r) = f.bound();
    assert_eq!(m.partition().text().len(), 55);
    assert!(m.partition().visual().is_empty());
    assert!(m.partition().mtp().is_empty());
    assert_eq!(r.state_dict().len(), 56);
    assert!(r.state_dict().validate_source_identities());
    let embed = r.require_static(crate::TensorRole::TokenEmbedding).unwrap();
    assert!(
        r.require_static(crate::TensorRole::OutputHead)
            .unwrap()
            .shares_storage_with(embed)
    );
    assert!(r.validate_parameter_limits(1, 1).is_err());
    r.validate_parameter_limits(1 << 20, 1 << 20).unwrap();
    // The small shape is not a new public supported profile.
    let mut value: serde_json::Value =
        serde_json::from_str(include_str!("../../../tests/qwen35_08b_config.json")).unwrap();
    value["text_config"]["hidden_size"] = 8.into();
    assert!(Qwen35Config::from_value(&value).is_err());
}

#[test]
fn tiny_prefill_two_decodes_abort_and_reset_match_full_replay() {
    let f = Fixture::new();
    let mut runner = f.runner();
    let mut states = vec![runner.create_sequence_state().unwrap()];
    let mut pages = Vec::new();
    let mut actual = step(&mut runner, &mut states, &mut pages, &[1, 3, 2], 1, true);
    let committed = states[0].clone();
    step(&mut runner, &mut states, &mut pages, &[9], 2, false);
    assert_eq!(states[0], committed);
    actual.extend(step(&mut runner, &mut states, &mut pages, &[5], 3, true));
    actual.extend(step(&mut runner, &mut states, &mut pages, &[7], 4, true));
    let final_state = states[0].kv_state().clone();
    runner.reset_sequence_state(&mut states[0]).unwrap();
    pages.clear();
    let expected = step(
        &mut runner,
        &mut states,
        &mut pages,
        &[1, 3, 2, 5, 7],
        5,
        true,
    );
    assert_eq!(actual.len(), 5);
    assert_eq!(expected.len(), 5);
    for (a, r) in actual.iter().flatten().zip(expected.iter().flatten()) {
        assert!(a.is_finite());
        assert!((a - r).abs() <= 2e-5 + 2e-4 * r.abs());
    }
    assert!(actual[4].windows(2).any(|p| p[0] != p[1]));
    // Linear state is compared numerically by the generic oracle tests; this
    // exact check also catches stale conv history in this deterministic fixture.
    assert_eq!(final_state, states[0].kv_state().clone());
}
