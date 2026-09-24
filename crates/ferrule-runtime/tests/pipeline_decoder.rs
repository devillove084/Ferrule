//! Checkpoint-backed production PP execution. Tests call the runtime; there are
//! no test-owned pipeline threads, command loops, stage math or KV coordinators.
#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, ThreadId};

use ferrule_common::execution::{
    ExecutionBatch, ExecutionOutput, ExecutionSequence, ExecutionTransactionId, ForwardMode,
    ForwardPhase, KvBlockId, KvElementType, KvPageId, KvReservationView, KvWriteSlot, LogitsOutput,
    LogitsRequest, StateSlot,
};
use ferrule_common::{
    ParallelRankId, ParallelTopologyId, ParallelismPlan, Result, ValidatedParallelTopology,
};
use ferrule_model::checkpoint::{CheckpointDType, CheckpointTensorSlice};
use ferrule_model::decoder::{
    CpuKvView, CpuPagedKvBackend, CpuPagedKvPool, DecoderKvBackend, DecoderKvCapacity,
    DecoderKvCommitBackend, DecoderKvPageStatus, DecoderKvPrepare, DenseLogits,
    GenericDecoderOptions, GenericDecoderRunner, GenericDecoderSequenceState, KvCommitBinding,
    KvEndProgress, PackedDecoderBatch, PagedKvBackend, PagedKvTransactionHandle, StandardGqaPlanes,
};
use ferrule_model::execution::ExecutionPrecisionPolicy;
use ferrule_model::runner::{
    MultiSessionBatchProgress, MultiSessionRunner, TransactionEndIntent, TransactionEndProgress,
};
use ferrule_model::transformer::{
    BoundDecoderResources, DecoderLoadOptions, DecoderRecipe, LayerSegmentPlan,
    SyntheticDecoderRecipe,
};
use ferrule_model::{ModelFamily, TensorRole, TokenizerHandle, WeightSource};

use ferrule_runtime::parallel::pipeline::{
    PipelineConfig, PipelineParallelExecutor, PipelineStage,
};
use ferrule_runtime::{Decision, SessionId, TransactionState};

#[path = "support/expert_decoder.rs"]
mod expert_decoder;

#[cfg(feature = "cuda")]
#[path = "support/cuda_pipeline.rs"]
mod cuda_pipeline;

const PAGE: usize = 2;
const MAX: usize = 16;

fn config(precision: ExecutionPrecisionPolicy) -> PipelineConfig {
    PipelineConfig {
        page_size: PAGE,
        max_pages: 16,
        max_positions: MAX,
        max_batch_tokens: MAX,
        session_capacity: 2,
        max_parameter_bytes: 4096,
        precision,
        max_ack_polls: 8,
    }
}
fn tx(id: u64) -> ExecutionTransactionId {
    ExecutionTransactionId::new(id).unwrap()
}
fn rank(id: u32) -> ParallelRankId {
    ParallelRankId::new(id)
}
fn topology(degree: usize) -> ValidatedParallelTopology {
    ValidatedParallelTopology::new(
        ParallelTopologyId::new(71),
        degree as u32,
        rank(0),
        ParallelismPlan::validated(1, 1, 1, 1, 1, degree).unwrap(),
    )
    .unwrap()
}
fn plans(degree: usize) -> Vec<LayerSegmentPlan> {
    match degree {
        1 => vec![LayerSegmentPlan::new(2, 0..2, true, true).unwrap()],
        2 => vec![
            LayerSegmentPlan::new(2, 0..1, true, false).unwrap(),
            LayerSegmentPlan::new(2, 1..2, false, true).unwrap(),
        ],
        _ => unreachable!(),
    }
}

struct Fixture(PathBuf);
impl Fixture {
    fn new(tied: bool) -> Self {
        Self::with_top_k(tied, 1)
    }
    fn with_top_k(tied: bool, top_k: usize) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ferrule-runtime-pp-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        let config = serde_json::json!({
            "vocab_size":8, "hidden_size":4, "num_attention_heads":2,
            "num_key_value_heads":1, "head_dim":2, "intermediate_size":4,
            "num_experts":2, "experts_per_token":top_k, "max_position_embeddings":MAX,
            "rms_norm_eps":0.00001, "rope_theta":10000.0, "tie_word_embeddings":tied
        });
        std::fs::write(
            path.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        std::fs::write(path.join("tokenizer.json"), r#"{
            "version":"1.0", "truncation":null, "padding":null, "added_tokens":[],
            "normalizer":null, "pre_tokenizer":null, "post_processor":null, "decoder":null,
            "model":{"type":"WordLevel", "vocab":{"<unk>":0,"a":1,"b":2,"c":3,"d":4,"e":5,"f":6,"g":7},"unk_token":"<unk>"}
        }"#).unwrap();
        let output = SyntheticDecoderRecipe::new().build(&config).unwrap();
        assert_eq!(output.spec().layers().len(), 2);
        for parameter in output
            .schema()
            .parameters()
            .iter()
            .filter(|p| p.alias_of().is_none())
        {
            let seed = parameter
                .path()
                .as_str()
                .bytes()
                .fold(0usize, |sum, byte| (sum * 31 + byte as usize) % 251);
            let values = (0..parameter.shape().iter().product::<usize>()).map(|index| {
                if parameter.shape().len() == 1 {
                    0.9 + index as f32 * 0.03
                } else {
                    ((seed + index * 17 + index * index * 3) % 41) as f32 * 0.012 - 0.24
                }
            });
            let payload = values.flat_map(f32::to_le_bytes).collect::<Vec<_>>();
            std::fs::write(path.join(format!("{}.bin", parameter.path())), payload).unwrap();
        }
        Self(path)
    }
    fn pipeline(
        &self,
        config: PipelineConfig,
        degree: usize,
        spy: Arc<Spy>,
    ) -> PipelineParallelExecutor {
        let path = self.0.clone();
        PipelineParallelExecutor::new(
            topology(degree),
            plans(degree),
            config,
            move |rank, plan| {
                // The factory is called on the real persistent owner. No resource or
                // CPU pool was constructed/captured on the host coordinator.
                let resources = load(&path)?;
                let stage = PipelineStage::prepare_cpu(&resources, plan, config)?;
                Ok(stage
                    .map_backend(|inner| SpyBackend::new(inner, rank.local.get() as usize, spy)))
            },
        )
        .unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn load(path: &Path) -> Result<BoundDecoderResources> {
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.join("config.json"))?).unwrap();
    let recipe = SyntheticDecoderRecipe::new();
    let output = recipe.build(&config).unwrap();
    let slices = output
        .schema()
        .parameters()
        .iter()
        .filter(|p| p.alias_of().is_none())
        .map(|p| CheckpointTensorSlice {
            name: SyntheticDecoderRecipe::external_name(p.path()),
            role: TensorRole::Unknown,
            path: path.join(format!("{}.bin", p.path())),
            offset: 0,
            bytes: (p.shape().iter().product::<usize>() * 4) as u64,
            dtype: CheckpointDType::F32,
            shape: p.shape().to_vec(),
        })
        .collect::<Vec<_>>();
    DecoderLoadOptions::new(&recipe, &config).bind_slices(slices)
}

struct Oracle {
    runner: GenericDecoderRunner<CpuPagedKvBackend>,
    states: Vec<GenericDecoderSequenceState>,
    next: u64,
}
impl Oracle {
    fn new(fixture: &Fixture, precision: ExecutionPrecisionPolicy) -> Self {
        let resources = load(&fixture.0).unwrap();
        let dtype = if precision == ExecutionPrecisionPolicy::f32() {
            KvElementType::F32
        } else {
            KvElementType::Bf16
        };
        let planes = StandardGqaPlanes::new(2, 1, 2, PAGE, MAX, dtype).unwrap();
        let backend = PagedKvBackend::new(CpuPagedKvPool::from_strategy(&planes, 16).unwrap());
        let options = GenericDecoderOptions::standard_cpu(
            resources.spec(),
            ModelFamily::Unknown("oracle".into()),
            WeightSource::Safetensors,
            PAGE,
            MAX,
            MAX,
            1,
            4096,
            precision,
        )
        .unwrap();
        let mut runner = GenericDecoderRunner::new(
            resources,
            TokenizerHandle::load(&fixture.0).unwrap(),
            backend,
            options,
        )
        .unwrap();
        let states = vec![runner.create_sequence_state().unwrap()];
        Self {
            runner,
            states,
            next: 1,
        }
    }
    fn forward(&mut self, tokens: &[u32], phase: ForwardPhase) -> ExecutionOutput {
        let state = &self.states[0];
        let start = state.core().position();
        let end = start + tokens.len();
        let pages = (0..end.div_ceil(PAGE))
            .map(|index| KvPageId(100 + index as u32))
            .collect::<Vec<_>>();
        let slot = StateSlot::new(0);
        let view = KvReservationView {
            state_slot: slot,
            execution_state_slot: slot,
            positions: start..end,
            newly_allocated: pages[start.div_ceil(PAGE)..].to_vec(),
            generation: state.core().generation(),
            execution_generation: state.core().generation(),
            cow_replacement: None,
        };
        let mode = if phase == ForwardPhase::Prefill {
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
            pages.iter().map(|page| KvBlockId::new(page.0)).collect(),
        );
        let id = tx(self.next);
        self.next += 1;
        self.runner
            .prepare_multi_session_batch(id, &mut self.states, &batch, &[view])
            .unwrap();
        let MultiSessionBatchProgress::Complete(output) = self
            .runner
            .execute_multi_session_batch_progress(id, &mut self.states, &batch)
            .unwrap()
        else {
            panic!("CPU oracle suspended")
        };
        assert_eq!(
            self.runner
                .end_transaction(id, &mut self.states, TransactionEndIntent::Publish)
                .unwrap(),
            TransactionEndProgress::Complete
        );
        output
    }
}
fn values(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && b.is_finite());
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "logit element {index}: {a} vs {b}"
        );
    }
}
fn logits(actual: &DenseLogits, expected: &ExecutionOutput) {
    assert_eq!(actual.rows(), expected.logits.len());
    for (index, row) in expected.logits.iter().enumerate() {
        let LogitsOutput::Full(expected) = &row.logits else {
            panic!("oracle logits absent")
        };
        values(actual.row(index).unwrap(), expected);
    }
}

#[derive(Default)]
struct RankSpy {
    constructed: AtomicUsize,
    dropped: AtomicUsize,
    installs: AtomicUsize,
    polls: AtomicUsize,
    publications: AtomicUsize,
    finishes: AtomicUsize,
    // Only the latest bounded logical projection is retained, never a request log.
    last_pages: Mutex<Vec<KvPageId>>,
    owner: Mutex<Option<ThreadId>>,
}
struct Spy {
    ranks: [RankSpy; 2],
    fail_ready: AtomicBool,
    fail_prepare: AtomicBool,
    hold_install: AtomicBool,
    unknown_install: AtomicBool,
    reject_install: AtomicBool,
    unknown_abort: AtomicBool,
    fail_release: AtomicBool,
    panic_publish: AtomicBool,
    cancel_after_enter: AtomicBool,
    cancellation: AtomicBool,
}
impl Spy {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            ranks: std::array::from_fn(|_| RankSpy::default()),
            fail_ready: AtomicBool::new(false),
            fail_prepare: AtomicBool::new(false),
            hold_install: AtomicBool::new(false),
            unknown_install: AtomicBool::new(false),
            reject_install: AtomicBool::new(false),
            unknown_abort: AtomicBool::new(false),
            fail_release: AtomicBool::new(false),
            panic_publish: AtomicBool::new(false),
            cancel_after_enter: AtomicBool::new(false),
            cancellation: AtomicBool::new(false),
        })
    }
}

// This decorator only observes/interrupts the real CPU backend. It does not
// implement decoder math, manufacture physical success, or drive the pipeline.
struct SpyBackend {
    inner: CpuPagedKvBackend,
    local: usize,
    spy: Arc<Spy>,
    owner: Rc<ThreadId>,
    installed: Option<(ExecutionTransactionId, u64)>,
}
impl SpyBackend {
    fn new(inner: CpuPagedKvBackend, local: usize, spy: Arc<Spy>) -> Self {
        let owner = thread::current().id();
        spy.ranks[local].constructed.fetch_add(1, Ordering::SeqCst);
        *spy.ranks[local].owner.lock().unwrap() = Some(owner);
        Self {
            inner,
            local,
            spy,
            owner: Rc::new(owner),
            installed: None,
        }
    }
    fn here(&self) {
        assert_eq!(*self.owner, thread::current().id());
    }
    fn fault(&self, fault: &AtomicBool) -> bool {
        self.local == 1 && fault.load(Ordering::Acquire)
    }
}
fn fault(message: &str) -> ferrule_common::Error {
    ferrule_common::Error::Execution {
        message: message.into(),
    }
}
impl Drop for SpyBackend {
    fn drop(&mut self) {
        self.here();
        self.spy.ranks[self.local]
            .dropped
            .fetch_add(1, Ordering::SeqCst);
    }
}
impl DecoderKvBackend for SpyBackend {
    type SequenceState = GenericDecoderSequenceState;
    type Transaction = PagedKvTransactionHandle;
    type KvView = CpuKvView;
    fn configure_capacity(&mut self, n: usize) -> Result<()> {
        self.here();
        self.inner.configure_capacity(n)
    }
    fn prepare(&mut self, request: DecoderKvPrepare<'_>) -> Result<Self::Transaction> {
        self.here();
        if self.fault(&self.spy.fail_prepare) {
            return Err(fault("injected prepare failure"));
        }
        self.inner.prepare(request)
    }
    fn enter(
        &mut self,
        t: &mut Self::Transaction,
        b: &PackedDecoderBatch,
        states: &mut [Self::SequenceState],
    ) -> Result<()> {
        self.here();
        self.inner.enter(t, b, states)?;
        *self.spy.ranks[self.local].last_pages.lock().unwrap() =
            b.sequences()[0].block_table().to_vec();
        assert!(b.protected_pages().len() <= 16);
        if self.spy.cancel_after_enter.load(Ordering::Acquire) {
            self.spy.cancellation.store(true, Ordering::Release);
        }
        Ok(())
    }
    fn active_view(&mut self, t: &mut Self::Transaction) -> Result<CpuKvView> {
        self.here();
        self.inner.active_view(t)
    }
    fn leave(&mut self, t: &mut Self::Transaction) -> Result<()> {
        self.here();
        self.inner.leave(t)
    }
    fn commit(&mut self, _: &mut Option<Self::Transaction>) -> Result<KvEndProgress> {
        panic!("pipeline must not call the legacy commit path")
    }
    fn rollback(&mut self, t: &mut Option<Self::Transaction>) -> Result<KvEndProgress> {
        self.here();
        if self.fault(&self.spy.unknown_abort) {
            return Err(fault("injected unknown cleanup"));
        }
        self.inner.rollback(t)
    }
    fn release(&mut self, pages: &[KvPageId]) -> Result<()> {
        self.here();
        if self.fault(&self.spy.fail_release) {
            return Err(fault("injected release failure"));
        }
        self.inner.release(pages)
    }
    fn preempt(&mut self, pages: &[KvPageId]) -> Result<()> {
        self.inner.preempt(pages)
    }
    fn restore(&mut self, pages: &[KvPageId]) -> Result<()> {
        self.inner.restore(pages)
    }
    fn capacity(&self) -> DecoderKvCapacity {
        self.here();
        self.inner.capacity()
    }
    fn page_status(&self, p: KvPageId) -> DecoderKvPageStatus {
        self.here();
        self.inner.page_status(p)
    }
    fn shutdown(&mut self) -> Result<()> {
        self.here();
        self.inner.shutdown()
    }
}
impl DecoderKvCommitBackend for SpyBackend {
    fn preflight_commit_ready(
        &self,
        t: &Self::Transaction,
        b: &KvCommitBinding,
        rank: ParallelRankId,
        states: &[Self::SequenceState],
    ) -> Result<()> {
        self.here();
        self.inner.preflight_commit_ready(t, b, rank, states)?;
        if self.fault(&self.spy.fail_ready) {
            return Err(fault("injected readiness failure"));
        }
        Ok(())
    }
    fn commit_batch(&self, t: &Self::Transaction) -> Result<&PackedDecoderBatch> {
        self.inner.commit_batch(t)
    }
    fn install_commit(
        &mut self,
        t: &mut Self::Transaction,
        generation: u64,
    ) -> Result<KvEndProgress> {
        self.here();
        self.spy.ranks[self.local]
            .installs
            .fetch_add(1, Ordering::SeqCst);
        assert_eq!(
            self.inner.install_commit(t, generation)?,
            KvEndProgress::Complete
        );
        self.installed = Some((t.id(), generation));
        if self.fault(&self.spy.unknown_install) {
            return Err(fault("injected lost install ACK"));
        }
        if self.fault(&self.spy.reject_install) {
            return Ok(KvEndProgress::ConsumedRejected);
        }
        if self.fault(&self.spy.hold_install) {
            return Ok(KvEndProgress::Pending);
        }
        Ok(KvEndProgress::Complete)
    }
    fn poll_install_ack(
        &mut self,
        t: &mut Self::Transaction,
        generation: u64,
    ) -> Result<KvEndProgress> {
        self.here();
        self.spy.ranks[self.local]
            .polls
            .fetch_add(1, Ordering::SeqCst);
        // This is ACK delivery for the synchronous inner operation already
        // observed above, not inference from the absence of a physical handle.
        assert_eq!(self.installed, Some((t.id(), generation)));
        if self.fault(&self.spy.unknown_install) {
            return Err(fault("install ACK still unknown"));
        }
        if self.fault(&self.spy.reject_install) {
            return Ok(KvEndProgress::ConsumedRejected);
        }
        if self.fault(&self.spy.hold_install) {
            return Ok(KvEndProgress::Pending);
        }
        Ok(KvEndProgress::Complete)
    }
    fn abort_prepared(
        &mut self,
        t: &mut Self::Transaction,
        generation: u64,
    ) -> Result<KvEndProgress> {
        self.here();
        if self.fault(&self.spy.unknown_abort) {
            return Err(fault("injected unknown cleanup"));
        }
        self.inner.abort_prepared(t, generation)
    }
    fn publish_committed(&mut self, t: &Self::Transaction) {
        self.here();
        self.inner.publish_committed(t);
        assert!(
            !self.fault(&self.spy.panic_publish),
            "injected owner loss before publication receipt"
        );
        self.spy.ranks[self.local]
            .publications
            .fetch_add(1, Ordering::SeqCst);
    }
    fn preflight_retirement(&self, t: &Self::Transaction, pages: &[KvPageId]) -> Result<()> {
        self.inner.preflight_retirement(t, pages)
    }
    fn retire_prepared(
        &mut self,
        t: &mut Self::Transaction,
        generation: u64,
        pages: &[KvPageId],
    ) -> Result<KvEndProgress> {
        self.here();
        self.inner.retire_prepared(t, generation, pages)
    }
    fn finish_prepared(&mut self, t: Self::Transaction) {
        self.here();
        self.inner.finish_prepared(t);
        self.installed = None;
        self.spy.ranks[self.local]
            .finishes
            .fetch_add(1, Ordering::SeqCst);
    }
}

fn drained(pipeline: &PipelineParallelExecutor) {
    assert_eq!(pipeline.outstanding(), 0);
    assert_eq!(pipeline.coordinator().in_use_credits(), 0);
    assert_eq!(pipeline.coordinator().retained_transaction_count(), 0);
    assert_eq!(pipeline.coordinator().retained_operation_count(), 0);
    assert_eq!(pipeline.page_manager().stats().retiring_pages, 0);
    for owner in pipeline.owner_stats().unwrap() {
        assert_eq!(owner.kv.active_transactions, 0);
        assert_eq!(owner.expert_outstanding, 0);
    }
    assert!(!pipeline.is_quarantined());
}
fn zero(pipeline: &PipelineParallelExecutor) {
    drained(pipeline);
    assert_eq!(pipeline.page_manager().active_sequences(), 0);
    assert_eq!(pipeline.page_manager().allocated_pages(), 0);
    for owner in pipeline.owner_stats().unwrap() {
        assert_eq!(owner.sessions, 0);
        assert_eq!(owner.kv.resident_pages, 0);
        assert_eq!(owner.kv.preempted_pages, 0);
        assert_eq!(owner.kv.free_pages, owner.kv.physical_pages);
    }
}

#[test]
fn checkpoint_pp2_prefill_decode_matches_generic_runner_on_persistent_owners() {
    for tied in [false, true] {
        for precision in [
            ExecutionPrecisionPolicy::f32(),
            ExecutionPrecisionPolicy::bf16_compatibility(),
        ] {
            let fixture = Fixture::new(tied);
            let spy = Spy::new();
            let mut pipeline = fixture.pipeline(config(precision), 2, Arc::clone(&spy));
            let owners = pipeline.owner_stats().unwrap();
            assert_eq!(owners[0].layers, 0..1);
            assert_eq!(owners[1].layers, 1..2);
            assert_ne!(owners[0].thread, owners[1].thread);
            assert!(
                owners
                    .iter()
                    .all(|owner| owner.thread != thread::current().id())
            );
            let mut oracle = Oracle::new(&fixture, precision);
            let session = SessionId(19);
            for tokens in [&[1, 2][..], &[3], &[4], &[5]] {
                let phase = if tokens.len() > 1 {
                    ForwardPhase::Prefill
                } else {
                    ForwardPhase::Decode
                };
                let expected = oracle.forward(tokens, phase);
                let actual = pipeline.forward(session, tokens, phase).unwrap();
                logits(&actual.logits, &expected);
                let slot = pipeline.session_slot(session).unwrap();
                let logical = pipeline.page_manager().block_table(slot).unwrap();
                assert_eq!(
                    logical.committed_tokens(),
                    oracle.states[0].core().position()
                );
                for (index, stats) in pipeline.owner_stats().unwrap().iter().enumerate() {
                    assert_eq!(stats.thread, owners[index].thread);
                    assert_eq!(stats.sessions, 1);
                    assert_eq!(stats.kv.resident_pages, logical.pages().len());
                    assert_eq!(
                        *spy.ranks[index].last_pages.lock().unwrap(),
                        logical.pages()
                    );
                }
                drained(&pipeline);
            }
            pipeline.release_session(session).unwrap();
            zero(&pipeline);
            pipeline.shutdown().unwrap();
            for rank in &spy.ranks {
                assert_eq!(rank.constructed.load(Ordering::Acquire), 1);
                assert_eq!(rank.dropped.load(Ordering::Acquire), 1);
                assert_eq!(rank.installs.load(Ordering::Acquire), 4);
                assert_eq!(rank.finishes.load(Ordering::Acquire), 4);
            }
        }
    }
}

#[test]
fn prepare_failures_and_cancellation_restore_history_without_residual_custody() {
    let fixture = Fixture::new(false);
    let cfg = config(ExecutionPrecisionPolicy::f32());
    for kind in 0..4 {
        let spy = Spy::new();
        let mut pipeline = fixture.pipeline(cfg, 2, Arc::clone(&spy));
        let session = SessionId(7);
        pipeline
            .forward(session, &[1], ForwardPhase::Prefill)
            .unwrap();
        let before = pipeline
            .page_manager()
            .block_table(pipeline.session_slot(session).unwrap())
            .unwrap()
            .pages()
            .to_vec();
        match kind {
            0 => spy.fail_prepare.store(true, Ordering::Release),
            1 => spy.fail_ready.store(true, Ordering::Release),
            2 => spy.cancel_after_enter.store(true, Ordering::Release),
            _ => {}
        }
        let failed = pipeline.forward_observed(
            session,
            &[2],
            ForwardPhase::Decode,
            &spy.cancellation,
            |progress| {
                if kind == 3 && progress.state == TransactionState::Preparing {
                    // Cancellation after all physical ready votes, before decision.
                    spy.cancellation.store(true, Ordering::Release);
                }
            },
        );
        assert!(failed.is_err());
        drained(&pipeline);
        let table = pipeline
            .page_manager()
            .block_table(pipeline.session_slot(session).unwrap())
            .unwrap();
        assert_eq!(table.committed_tokens(), 1);
        assert_eq!(table.pages(), before);
        assert_eq!(pipeline.coordinator().publication_count(), 1);
        assert_eq!(pipeline.page_manager().allocated_pages(), 1);
        for rank in &spy.ranks {
            assert_eq!(rank.installs.load(Ordering::Acquire), 1);
        }
        spy.fail_prepare.store(false, Ordering::Release);
        spy.fail_ready.store(false, Ordering::Release);
        spy.cancel_after_enter.store(false, Ordering::Release);
        spy.cancellation.store(false, Ordering::Release);
        let mut oracle = Oracle::new(&fixture, cfg.precision);
        oracle.forward(&[1], ForwardPhase::Prefill);
        let expected = oracle.forward(&[3], ForwardPhase::Decode);
        let actual = pipeline
            .forward(session, &[3], ForwardPhase::Decode)
            .unwrap();
        logits(&actual.logits, &expected);
        pipeline.release_session(session).unwrap();
        zero(&pipeline);
    }
}

#[test]
fn pending_and_lost_install_ack_gate_logical_publish_and_logits() {
    for lost in [false, true] {
        let fixture = Fixture::new(false);
        let spy = Spy::new();
        let cfg = config(ExecutionPrecisionPolicy::f32());
        let mut pipeline = fixture.pipeline(cfg, 2, Arc::clone(&spy));
        spy.hold_install.store(!lost, Ordering::Release);
        spy.unknown_install.store(lost, Ordering::Release);
        let mut saw_gate = false;
        let mut saw_publication = false;
        let result = pipeline.forward_observed(
            SessionId(1),
            &[1, 2],
            ForwardPhase::Prefill,
            &spy.cancellation,
            |progress| {
                if progress.state == TransactionState::Decided(Decision::Commit) {
                    assert_eq!(progress.committed_tokens, 0);
                    assert_eq!(progress.publications, 0);
                    assert_eq!(progress.allocated_pages, 1);
                    for owner in &spy.ranks {
                        assert_eq!(owner.publications.load(Ordering::Acquire), 0);
                    }
                    if !saw_gate && spy.ranks[1].polls.load(Ordering::Acquire) > 0 {
                        assert!(progress.pending_ranks.contains(&rank(1)));
                        saw_gate = true;
                        spy.hold_install.store(false, Ordering::Release);
                        spy.unknown_install.store(false, Ordering::Release);
                    }
                }
                if progress.state == TransactionState::Published {
                    saw_publication = true;
                    assert!(saw_gate);
                    assert_eq!(progress.committed_tokens, 2);
                    assert!(progress.pending_ranks.is_empty());
                }
            },
        );

        let result = result.unwrap();
        assert!(saw_gate && saw_publication);
        let mut oracle = Oracle::new(&fixture, cfg.precision);
        logits(
            &result.logits,
            &oracle.forward(&[1, 2], ForwardPhase::Prefill),
        );
        for owner in &spy.ranks {
            assert_eq!(owner.installs.load(Ordering::Acquire), 1);
        }
        assert!(spy.ranks[1].polls.load(Ordering::Acquire) >= 2);
        drained(&pipeline);
        pipeline.release_session(SessionId(1)).unwrap();
        zero(&pipeline);
    }
}

#[test]
fn cow_fork_projects_same_page_ids_to_both_local_planes_and_keeps_history() {
    let fixture = Fixture::new(true);
    for precision in [
        ExecutionPrecisionPolicy::f32(),
        ExecutionPrecisionPolicy::bf16_compatibility(),
    ] {
        let cfg = config(precision);
        let spy = Spy::new();
        let mut pipeline = fixture.pipeline(cfg, 2, Arc::clone(&spy));
        let source = SessionId(41);
        let branch = SessionId(42);
        let mut source_oracle = Oracle::new(&fixture, precision);
        let mut branch_oracle = Oracle::new(&fixture, precision);
        logits(
            &pipeline
                .forward(source, &[1, 2, 3], ForwardPhase::Prefill)
                .unwrap()
                .logits,
            &source_oracle.forward(&[1, 2, 3], ForwardPhase::Prefill),
        );
        branch_oracle.forward(&[1, 2, 3], ForwardPhase::Prefill);
        pipeline.fork_session(source, branch).unwrap();
        let source_slot = pipeline.session_slot(source).unwrap();
        let branch_slot = pipeline.session_slot(branch).unwrap();
        assert_ne!(source_slot, branch_slot);
        let original = pipeline
            .page_manager()
            .block_table(source_slot)
            .unwrap()
            .pages()
            .to_vec();
        for &page in &original {
            assert_eq!(pipeline.page_manager().page_refcount(page), 2);
        }

        spy.fail_ready.store(true, Ordering::Release);
        assert!(
            pipeline
                .forward(source, &[6], ForwardPhase::Decode)
                .is_err()
        );
        drained(&pipeline);
        assert_eq!(pipeline.page_manager().allocated_pages(), 2);
        assert_eq!(
            pipeline
                .page_manager()
                .block_table(source_slot)
                .unwrap()
                .pages(),
            original
        );
        assert_eq!(
            pipeline
                .page_manager()
                .block_table(branch_slot)
                .unwrap()
                .pages(),
            original
        );
        spy.fail_ready.store(false, Ordering::Release);
        logits(
            &pipeline
                .forward(source, &[4], ForwardPhase::Decode)
                .unwrap()
                .logits,
            &source_oracle.forward(&[4], ForwardPhase::Decode),
        );
        let changed = pipeline
            .page_manager()
            .block_table(source_slot)
            .unwrap()
            .pages()
            .to_vec();
        assert_eq!(changed[0], original[0]);
        assert_ne!(changed[1], original[1]);
        for observed in &spy.ranks {
            assert_eq!(*observed.last_pages.lock().unwrap(), changed);
        }
        assert_eq!(pipeline.page_manager().page_refcount(original[0]), 2);
        assert_eq!(pipeline.page_manager().page_refcount(original[1]), 1);
        assert_eq!(pipeline.page_manager().page_refcount(changed[1]), 1);
        logits(
            &pipeline
                .forward(branch, &[6], ForwardPhase::Decode)
                .unwrap()
                .logits,
            &branch_oracle.forward(&[6], ForwardPhase::Decode),
        );
        assert_eq!(
            pipeline
                .page_manager()
                .block_table(branch_slot)
                .unwrap()
                .pages(),
            original
        );
        for observed in &spy.ranks {
            assert_eq!(*observed.last_pages.lock().unwrap(), original);
        }
        pipeline.release_session(source).unwrap();
        drained(&pipeline);
        assert_eq!(pipeline.page_manager().allocated_pages(), 2);
        logits(
            &pipeline
                .forward(branch, &[5], ForwardPhase::Decode)
                .unwrap()
                .logits,
            &branch_oracle.forward(&[5], ForwardPhase::Decode),
        );
        pipeline.release_session(branch).unwrap();
        zero(&pipeline);
    }
}

#[test]
fn unresolved_install_or_abort_retains_custody_and_rejects_reuse() {
    for kind in 0..3 {
        let fixture = Fixture::new(false);
        let cfg = config(ExecutionPrecisionPolicy::f32());
        let spy = Spy::new();
        let mut pipeline = fixture.pipeline(cfg, 2, Arc::clone(&spy));
        match kind {
            0 => spy.unknown_install.store(true, Ordering::Release),
            1 => spy.reject_install.store(true, Ordering::Release),
            _ => {
                spy.fail_ready.store(true, Ordering::Release);
                spy.unknown_abort.store(true, Ordering::Release);
            }
        }
        assert!(
            pipeline
                .forward(SessionId(1), &[1, 2], ForwardPhase::Prefill)
                .is_err()
        );
        assert!(pipeline.is_quarantined());
        assert_eq!(pipeline.outstanding(), 0);
        assert_eq!(pipeline.coordinator().retained_transaction_count(), 1);
        assert_eq!(pipeline.coordinator().in_use_credits(), 2);
        assert_eq!(pipeline.coordinator().publication_count(), 0);
        assert_eq!(pipeline.page_manager().stats().committed_tokens, 0);
        assert_eq!(pipeline.page_manager().allocated_pages(), 1);
        assert_eq!(pipeline.page_manager().free_pages(), 0);
        assert_eq!(
            pipeline.coordinator().state(tx(1)).unwrap(),
            TransactionState::Decided(if kind == 2 {
                Decision::Abort
            } else {
                Decision::Commit
            })
        );
        assert!(
            pipeline
                .forward(SessionId(2), &[3], ForwardPhase::Prefill)
                .is_err()
        );
        assert!(pipeline.release_session(SessionId(1)).is_err());
        assert!(pipeline.shutdown().is_err());
        for rank in &spy.ranks {
            assert_eq!(rank.publications.load(Ordering::Acquire), 0);
        }
        // Unknown work intentionally has residual custody. Zero-residual is only
        // required/proven for successful and acknowledged rollback paths.
    }
}

#[test]
fn page_session_and_request_bounds_do_not_leave_transaction_logs() {
    let fixture = Fixture::new(false);
    let spy = Spy::new();
    let mut cfg = config(ExecutionPrecisionPolicy::f32());
    cfg.max_pages = 1;
    cfg.session_capacity = 1;
    let mut pipeline = fixture.pipeline(cfg, 2, Arc::clone(&spy));
    assert!(
        pipeline
            .forward(SessionId(1), &[], ForwardPhase::Prefill)
            .is_err()
    );
    assert!(
        pipeline
            .forward(SessionId(1), &[8], ForwardPhase::Prefill)
            .is_err()
    );
    assert!(
        pipeline
            .forward(SessionId(1), &[1; MAX + 1], ForwardPhase::Prefill)
            .is_err()
    );
    assert!(
        pipeline
            .forward(SessionId(1), &[1], ForwardPhase::Decode)
            .is_err()
    );
    assert!(
        pipeline
            .forward(SessionId(1), &[1, 2, 3], ForwardPhase::Prefill)
            .is_err()
    );
    drained(&pipeline);
    assert_eq!(pipeline.page_manager().allocated_pages(), 0);
    assert!(
        pipeline
            .forward(SessionId(2), &[1], ForwardPhase::Prefill)
            .is_err()
    );
    pipeline.release_session(SessionId(1)).unwrap();
    for _ in 0..24 {
        pipeline
            .forward(SessionId(1), &[1], ForwardPhase::Prefill)
            .unwrap();
        pipeline
            .forward(SessionId(1), &[2], ForwardPhase::Prefill)
            .unwrap();
        assert!(
            pipeline
                .forward(SessionId(1), &[3], ForwardPhase::Decode)
                .is_err()
        );
        drained(&pipeline);
        pipeline.release_session(SessionId(1)).unwrap();
        zero(&pipeline);
    }
}

#[test]
fn one_segment_with_two_complete_layers_uses_the_same_production_path() {
    let fixture = Fixture::new(false);
    let cfg = config(ExecutionPrecisionPolicy::f32());
    let mut pipeline = fixture.pipeline(cfg, 1, Spy::new());
    let mut oracle = Oracle::new(&fixture, cfg.precision);
    for (tokens, phase) in [
        (&[1, 2][..], ForwardPhase::Prefill),
        (&[3][..], ForwardPhase::Decode),
    ] {
        logits(
            &pipeline
                .forward(SessionId(1), tokens, phase)
                .unwrap()
                .logits,
            &oracle.forward(tokens, phase),
        );
    }
    assert_eq!(pipeline.owner_stats().unwrap()[0].layers, 0..2);
    pipeline.release_session(SessionId(1)).unwrap();
    zero(&pipeline);
}

#[test]
fn owner_loss_before_publish_receipt_cannot_publish_logical_kv() {
    let fixture = Fixture::new(false);
    let spy = Spy::new();
    spy.panic_publish.store(true, Ordering::Release);
    let mut pipeline =
        fixture.pipeline(config(ExecutionPrecisionPolicy::f32()), 2, Arc::clone(&spy));
    assert!(
        pipeline
            .forward(SessionId(1), &[1, 2], ForwardPhase::Prefill)
            .is_err()
    );
    assert!(pipeline.is_quarantined());
    assert_eq!(pipeline.page_manager().stats().committed_tokens, 0);
    assert_eq!(pipeline.page_manager().allocated_pages(), 1);
    assert_eq!(pipeline.page_manager().free_pages(), 0);
    assert_eq!(pipeline.coordinator().publication_count(), 0);
    assert_eq!(
        pipeline.coordinator().state(tx(1)).unwrap(),
        TransactionState::Decided(Decision::Commit)
    );
    assert!(
        pipeline
            .coordinator()
            .pending_ranks(tx(1))
            .unwrap()
            .is_empty()
    );
    assert!(pipeline.release_session(SessionId(1)).is_err());
    drop(pipeline);
    // DP deliberately leaked the failed owner with original KV ledger custody.
    assert_eq!(spy.ranks[1].dropped.load(Ordering::Acquire), 0);
}

#[test]
fn release_error_keeps_zero_refcount_pages_quarantined() {
    let fixture = Fixture::new(false);
    let spy = Spy::new();
    let mut cfg = config(ExecutionPrecisionPolicy::f32());
    cfg.max_pages = 1;
    let mut pipeline = fixture.pipeline(cfg, 2, Arc::clone(&spy));
    pipeline
        .forward(SessionId(1), &[1, 2], ForwardPhase::Prefill)
        .unwrap();
    spy.fail_release.store(true, Ordering::Release);
    assert!(pipeline.release_session(SessionId(1)).is_err());
    assert!(pipeline.is_quarantined());
    assert_eq!(pipeline.page_manager().stats().retiring_pages, 1);
    assert_eq!(pipeline.page_manager().allocated_pages(), 1);
    assert_eq!(pipeline.page_manager().free_pages(), 0);
    let stats = pipeline.owner_stats().unwrap();
    assert_eq!(stats[0].kv.resident_pages, 0);
    assert_eq!(stats[1].kv.resident_pages, 1);
    assert!(
        pipeline
            .forward(SessionId(2), &[3], ForwardPhase::Prefill)
            .is_err()
    );
}

#[test]
fn composed_mesh_metadata_does_not_enable_pipeline_execution() {
    for (dp, tp, ep) in [(1, 2, 2), (2, 1, 2), (1, 1, 2)] {
        let topology = ValidatedParallelTopology::new(
            ParallelTopologyId::new(71),
            (dp * 2 * tp) as u32,
            rank(0),
            ParallelismPlan::validated(dp, tp, ep, 1, 1, 2).unwrap(),
        )
        .unwrap();
        assert_eq!(
            topology
                .execution_scopes(0)
                .unwrap()
                .kv_participants()
                .len(),
            2 * tp
        );
        let result = PipelineParallelExecutor::new(
            topology,
            plans(2),
            config(ExecutionPrecisionPolicy::f32()),
            |_, _| -> Result<PipelineStage<CpuPagedKvBackend>> {
                panic!("composed topology must be rejected before any owner factory runs")
            },
        );
        assert!(result.is_err());
    }
}
