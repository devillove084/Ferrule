use ferrule_common::execution::*;
use ferrule_model::checkpoint::{CheckpointDType, CheckpointTensorSlice};
use ferrule_model::decoder::*;
use ferrule_model::execution::ExecutionPrecisionPolicy;
use ferrule_model::models::qwen3::Qwen3DenseRecipe;
use ferrule_model::transformer::*;
use ferrule_model::{ModelFamily, TensorRole, WeightSource};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
pub const PAGE: usize = 2;
pub const POSITIONS: usize = 16;
pub const LIMIT: u64 = 1024 * 1024;
pub struct Fixture {
    directory: PathBuf,
    pub resources: BoundDecoderResources,
    pub oracle: serde_json::Value,
}
impl Fixture {
    pub fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let directory = std::env::temp_dir().join(format!(
            "ferrule-bf16-dense-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let oracle: serde_json::Value =
            serde_json::from_str(include_str!("../fixtures/standard_bf16_hf.json")).unwrap();
        assert_eq!(oracle["schema"], "ferrule.standard-bf16-hf-eager.v2");
        assert_eq!(
            oracle["rotary_contract"]["initialization"],
            "from_pretrained"
        );
        assert_eq!(
            oracle["rotary_contract"]["inverse_frequency_dtype"],
            "float32"
        );
        assert_eq!(oracle["rotary_contract"]["table_dtype"], "bfloat16");
        assert_eq!(oracle["device"], "cpu");
        assert_eq!(oracle["dtype"], "bfloat16");
        let slices: Vec<_> = oracle["weights"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(index, weight)| {
                let values = values(&weight["values"]);
                let bytes: Vec<_> = values
                    .iter()
                    .flat_map(|&v| half::bf16::from_f32(v).to_bits().to_le_bytes())
                    .collect();
                let path = directory.join(format!("{index}.bin"));
                std::fs::write(&path, &bytes).unwrap();
                CheckpointTensorSlice {
                    name: weight["name"].as_str().unwrap().into(),
                    role: TensorRole::Unknown,
                    path,
                    offset: 0,
                    bytes: bytes.len() as u64,
                    dtype: CheckpointDType::Bf16,
                    shape: serde_json::from_value(weight["shape"].clone()).unwrap(),
                }
            })
            .collect();
        let resources = DecoderLoadOptions::new(&Qwen3DenseRecipe::new(), &oracle["config"])
            .bind_slices(slices)
            .unwrap();
        Self {
            directory,
            resources,
            oracle,
        }
    }
    pub fn planes(&self, layers: usize) -> StandardGqaPlanes {
        StandardGqaPlanes::new(layers, 1, 8, PAGE, POSITIONS, KvElementType::Bf16).unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}
pub fn values(v: &serde_json::Value) -> Vec<f32> {
    serde_json::from_value(v.clone()).unwrap()
}
pub fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    let max_error = actual
        .iter()
        .zip(expected)
        .map(|(&a, &e)| (a - e).abs())
        .fold(0.0_f32, f32::max);
    eprintln!(
        "BF16_ORACLE elements={} max_abs_error={max_error}",
        actual.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (a - e).abs() <= 0.00025 + 0.012 * e.abs(),
            "element {i}: actual={a}, HF={e}"
        );
        assert_eq!(
            a.to_bits() & 0xffff,
            0,
            "output must have a real BF16 boundary"
        );
    }
}
pub fn enter<B: DecoderKvBackend<SequenceState = GenericDecoderSequenceState>>(
    fixture: &Fixture,
    backend: &mut B,
    state: &mut GenericDecoderSequenceState,
    id: u64,
    tokens: &[u32],
    tail: Option<(KvPageId, Option<KvCowReplacement>)>,
) -> (PackedDecoderBatch, B::Transaction, B::KvView) {
    let start = state.core().position();
    let end = start + tokens.len();
    // Intentionally unrelated to physical GPU slots.
    let mut pages = (0..end.div_ceil(PAGE))
        .map(|p| KvPageId(700 + p as u32 * 11))
        .collect::<Vec<_>>();
    if let Some((tail, _)) = tail {
        pages[1] = tail;
    }
    let slot = StateSlot::new(0);
    let phase = if start == 0 {
        ForwardPhase::Prefill
    } else {
        ForwardPhase::Decode
    };
    let mode = if start == 0 {
        ForwardMode::Prefill
    } else {
        ForwardMode::Decode
    };
    let batch = ExecutionBatch::new(
        mode,
        tokens.to_vec(),
        (start..end).map(|p| p as u32).collect(),
        (start..end)
            .map(|p| {
                Some(KvWriteSlot::new(
                    pages[p / PAGE].0 * PAGE as u32 + (p % PAGE) as u32,
                ))
            })
            .collect(),
        vec![LogitsRequest::Full; tokens.len()],
        vec![ExecutionSequence::new(
            slot,
            phase,
            0..tokens.len() as u32,
            start as u32,
            end as u32,
            0..pages.len() as u32,
        )],
        pages.iter().map(|p| KvBlockId::new(p.0)).collect(),
    );
    let reservations = [KvReservationView {
        state_slot: slot,
        execution_state_slot: slot,
        positions: start..end,
        newly_allocated: pages[start.div_ceil(PAGE)..].to_vec(),
        generation: state.core().generation(),
        execution_generation: state.core().generation(),
        cow_replacement: tail.and_then(|(_, cow)| cow),
    }];
    let options = GenericDecoderOptions::standard_cpu(
        fixture.resources.spec(),
        ModelFamily::Unknown("gpu-test".into()),
        WeightSource::Safetensors,
        PAGE,
        POSITIONS,
        POSITIONS,
        1,
        LIMIT,
        ExecutionPrecisionPolicy::bf16_compatibility(),
    )
    .unwrap();
    let packed = PackedDecoderBatch::lower(
        &batch,
        &reservations,
        std::slice::from_ref(state),
        &options.capabilities(),
        PAGE,
        &|p| backend.page_status(p),
    )
    .unwrap();
    let custody = packed
        .sequences()
        .iter()
        .map(|s| DecoderKvSequenceCustody {
            source_index: s.state_index(),
            topology_id: s.topology_id(),
            page_state_slot: s.page_state_slot(),
            page_generation: s.page_generation(),
            execution_generation: s.execution_generation(),
            context_len: s.context_len(),
            query_len: s.query_len(),
        })
        .collect::<Vec<_>>();
    let statuses = packed
        .protected_pages()
        .iter()
        .map(|&page| DecoderKvPageSnapshot {
            page,
            status: backend.page_status(page),
        })
        .collect::<Vec<_>>();
    let mut transaction = backend
        .prepare(DecoderKvPrepare {
            transaction: ExecutionTransactionId::new(id).unwrap(),
            sequences: &custody,
            new_pages: packed.new_pages(),
            writable_pages: packed.writable_pages(),
            cow_replacements: packed.cow_replacements(),
            protected_pages: packed.protected_pages(),
            capacity: backend.capacity(),
            page_statuses: &statuses,
        })
        .unwrap();
    backend
        .enter(&mut transaction, &packed, std::slice::from_mut(state))
        .unwrap();
    let view = backend.active_view(&mut transaction).unwrap();
    (packed, transaction, view)
}

pub fn finish<B: DecoderKvBackend<SequenceState = GenericDecoderSequenceState>>(
    backend: &mut B,
    state: &mut GenericDecoderSequenceState,
    mut tx: B::Transaction,
    rows: usize,
) {
    let binding = state.core().begin_step().unwrap();
    backend.leave(&mut tx).unwrap();
    assert_eq!(
        backend.commit(&mut Some(tx)).unwrap(),
        KvEndProgress::Complete
    );
    state.core_mut().commit_step(binding, rows).unwrap();
}
