//! Two real gated-GQA layers through the ordinary prepared runner. No GPU runs
//! unless the independent device owner explicitly selects the ignored fixture.
use ferrule_common::execution::*;
use ferrule_model::decoder::*;
use ferrule_model::execution::ExecutionPrecisionPolicy;
use ferrule_model::nn::{
    DTypeConstraint, ModulePath, ParameterDType, ParameterId, ParameterResidency, ParameterSpec,
};
use ferrule_model::tokenizer::TokenizerHandle;
use ferrule_model::transformer::*;
use ferrule_model::{
    CheckpointDType, CheckpointTensorSlice, ModelFamily, MultiSessionBatchProgress,
    MultiSessionRunner, TensorRole, TransactionEndIntent, TransactionEndProgress, WeightSource,
};
use ferrule_runtime::{
    KvPageManager, KvReservation, KvReservationCommit, RunnerMultiSessionExecutor,
};
use std::sync::Arc;

fn spec() -> DecoderModelSpec {
    let norm = || RmsNorm::new(8, 1e-6).unwrap().with_one_plus_weight();
    let rope = RotaryEmbedding::new(
        8,
        10000.,
        RotaryPairing::SplitHalf,
        RotaryRegion::Prefix { dimensions: 4 },
        RotaryScaling::None,
    )
    .unwrap();
    let layers = (0..2)
        .map(|i| {
            DecoderLayer::new(
                i,
                norm(),
                Attention::Gqa(
                    GqaAttention::new(8, 2, 1, 8, false, rope.clone())
                        .unwrap()
                        .with_gated_query()
                        .unwrap()
                        .with_qk_norms(norm(), norm())
                        .unwrap(),
                ),
                Residual::Add,
                norm(),
                FeedForward::SwiGlu(SwiGlu::new(8, 12, false).unwrap()),
                Residual::Add,
            )
            .unwrap()
        })
        .collect();
    DecoderModelSpec::new(DecoderModelParts {
        architecture: "host-kv-two-gqa".into(),
        hidden_size: 8,
        vocab_size: 11,
        max_sequence_length: Some(32),
        token_embedding: Embedding::new(11, 8, None).unwrap(),
        layers,
        final_norm: norm(),
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
        _ => {}
    }
    match name
        .split('.')
        .skip(2)
        .collect::<Vec<_>>()
        .join(".")
        .as_str()
    {
        "input_norm.weight" => TensorRole::AttentionNorm,
        "post_attention_norm.weight" => TensorRole::FeedForwardNorm,
        "attention.query.weight" => TensorRole::AttentionQuery,
        "attention.key.weight" => TensorRole::AttentionKey,
        "attention.value.weight" => TensorRole::AttentionValue,
        "attention.output.weight" => TensorRole::AttentionOutput,
        "attention.query_norm.weight" => TensorRole::AttentionQueryNorm,
        "attention.key_norm.weight" => TensorRole::AttentionKeyNorm,
        "feed_forward.gate.weight" => TensorRole::DenseMlpGate,
        "feed_forward.up.weight" => TensorRole::DenseMlpUp,
        "feed_forward.down.weight" => TensorRole::DenseMlpDown,
        _ => panic!("unexpected tensor {name}"),
    }
}
pub(crate) struct Fixture {
    dir: tempfile::TempDir,
}
impl Fixture {
    pub(crate) fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("tokenizer.json"), r#"{
            "version":"1.0", "truncation":null, "padding":null, "added_tokens":[],
            "normalizer":null, "pre_tokenizer":null, "post_processor":null, "decoder":null,
            "model":{"type":"WordLevel","vocab":{"<unk>":0,"a":1,"b":2,"c":3,"d":4,"e":5,"f":6,"g":7,"h":8,"i":9,"j":10},"unk_token":"<unk>"}
        }"#).unwrap();
        Self { dir }
    }
    pub(crate) fn resources(&self) -> BoundDecoderResources {
        let oracle: serde_json::Value = serde_json::from_str(include_str!(
            "../../../ferrule-model/tests/fixtures/hybrid_cpu/oracle.json"
        ))
        .unwrap();
        let path = self.dir.path().join("weights.f32");
        let mut schema = StateDictSchema::builder();
        let mut mapper = ExactNameMapper::new();
        let mut bytes = Vec::new();
        let mut slices = Vec::new();
        // Reuse real fixture parameters for original GQA layers 1 and 3 only.
        for (original, tensor) in oracle["tensors"].as_object().unwrap() {
            let name = if let Some(suffix) = original.strip_prefix("layers.1.") {
                format!("layers.0.{suffix}")
            } else if let Some(suffix) = original.strip_prefix("layers.3.") {
                format!("layers.1.{suffix}")
            } else if original.starts_with("layers.") {
                continue;
            } else {
                original.clone()
            };
            let offset = bytes.len();
            for value in tensor["values"].as_array().unwrap() {
                bytes.extend_from_slice(&(value.as_f64().unwrap() as f32).to_le_bytes());
            }
            let shape = tensor["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| n.as_u64().unwrap() as usize)
                .collect::<Vec<_>>();
            let module = ModulePath::new(&name).unwrap();
            let residency = if name.starts_with("layers.") {
                ParameterResidency::layer(name.split('.').nth(1).unwrap().parse().unwrap())
            } else {
                ParameterResidency::Static
            };
            schema
                .register_with_role(
                    ParameterSpec::new(
                        ParameterId::new(slices.len() as u64 + 1),
                        module.clone(),
                        DTypeConstraint::exact(ParameterDType::F32),
                        shape.clone(),
                        residency,
                    )
                    .unwrap(),
                    role(&name),
                )
                .unwrap();
            mapper.insert(&name, NameMapping::weight(module)).unwrap();
            slices.push(CheckpointTensorSlice {
                role: role(&name),
                name,
                path: path.clone(),
                offset: offset as u64,
                bytes: (bytes.len() - offset) as u64,
                dtype: CheckpointDType::F32,
                shape,
            });
        }
        std::fs::write(path, bytes).unwrap();
        let schema = schema.build().unwrap();
        let state = StateDictBinder::new(&schema, &mapper)
            .bind_slices(slices)
            .unwrap();
        BoundDecoderResources::new(spec(), Arc::new(state)).unwrap()
    }
    pub(crate) fn tokenizer(&self) -> TokenizerHandle {
        TokenizerHandle::load(self.dir.path()).unwrap()
    }
    fn options(&self) -> GenericDecoderOptions {
        GenericDecoderOptions::standard_cpu(
            &spec(),
            ModelFamily::Unknown("host-kv-two-gqa".into()),
            WeightSource::Safetensors,
            2,
            32,
            8,
            1,
            1 << 20,
            ExecutionPrecisionPolicy::f32(),
        )
        .unwrap()
    }
    fn cpu(&self) -> RunnerMultiSessionExecutor<GenericDecoderRunner<HybridCpuDecoder>> {
        let mut r =
            GenericDecoderRunner::hybrid_cpu(self.resources(), self.tokenizer(), self.options())
                .unwrap();
        r.configure_kv_page_capacity(32).unwrap();
        RunnerMultiSessionExecutor::new(r)
    }
}
#[derive(Debug)]
struct Schema;
impl KvLayoutSchema for Schema {
    fn planes(&self) -> &[KvPlaneDescriptor] {
        static P: [KvPlaneDescriptor; 2] = [
            KvPlaneDescriptor::new("standard_gqa.key", 8, 2, KvElementType::F32),
            KvPlaneDescriptor::new("standard_gqa.value", 8, 2, KvElementType::F32),
        ];
        &P
    }
    fn page_size(&self) -> usize {
        2
    }
    fn max_sequence_len(&self) -> usize {
        32
    }
}
fn manager() -> KvPageManager {
    let mut m = KvPageManager::new(Box::new(Schema), 32);
    m.alloc_sequence(StateSlot::new(0), 0).unwrap();
    m
}
fn batch(m: &KvPageManager, r: &KvReservation, tokens: &[u32]) -> ExecutionBatch {
    let start = r.positions.start;
    let bindings = m.reservation_bindings(r.view()).unwrap();
    let (mode, phase) = if start == 0 {
        (ForwardMode::Prefill, ForwardPhase::Prefill)
    } else {
        (ForwardMode::Decode, ForwardPhase::Decode)
    };
    ExecutionBatch::new(
        mode,
        tokens.to_vec(),
        (start as u32..(start + tokens.len()) as u32).collect(),
        bindings.write_slots.into_iter().map(Some).collect(),
        vec![LogitsRequest::Full; tokens.len()],
        vec![ExecutionSequence::new(
            StateSlot::new(0),
            phase,
            0..tokens.len() as u32,
            start as u32,
            (start + tokens.len()) as u32,
            0..bindings.block_ids.len() as u32,
        )],
        bindings.block_ids,
    )
}
fn logits(progress: MultiSessionBatchProgress) -> Vec<Vec<f32>> {
    let MultiSessionBatchProgress::Complete(output) = progress else {
        panic!("sync runner suspended")
    };
    output
        .logits
        .into_iter()
        .map(|r| match r.logits {
            LogitsOutput::Full(v) => v,
            _ => panic!("full logits required"),
        })
        .collect()
}
fn publish<R: MultiSessionRunner>(
    e: &mut RunnerMultiSessionExecutor<R>,
    states: &mut [R::SequenceState],
    m: &mut KvPageManager,
    r: KvReservation,
    tx: ExecutionTransactionId,
) {
    assert_eq!(
        e.end_transaction(tx, states, TransactionEndIntent::Publish)
            .unwrap(),
        TransactionEndProgress::Complete
    );
    let n = r.positions.len();
    let p = m
        .prepare_commit(vec![KvReservationCommit::new(r, n)])
        .unwrap();
    let retired = m.publish_commit(p);
    m.confirm_page_retirement(retired).unwrap();
}
fn resident<R: MultiSessionRunner>(
    e: &mut RunnerMultiSessionExecutor<R>,
    states: &mut [R::SequenceState],
    m: &mut KvPageManager,
    tokens: &[u32],
    id: u64,
) -> Vec<Vec<f32>> {
    let r = m.reserve(StateSlot::new(0), 0, tokens.len()).unwrap();
    let b = batch(m, &r, tokens);
    let tx = ExecutionTransactionId::new(id).unwrap();
    e.prepare_batch_with_kv(tx, states, &b, std::slice::from_ref(r.view()))
        .unwrap();
    let out = logits(e.execute_prepared_batch(tx, states, &b).unwrap());
    publish(e, states, m, r, tx);
    out
}
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (&a, &b) in a.iter().zip(b) {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() < 3e-5 + b.abs() * 5e-4,
            "{a} != {b}"
        );
    }
}
#[test]
fn two_layer_normal_runner_cpu_prefill_and_repeated_decode_agree() {
    let f = Fixture::new();
    let mut e = f.cpu();
    let mut m = manager();
    let mut states = vec![e.create_sequence_state().unwrap()];
    let mut actual = resident(&mut e, &mut states, &mut m, &[1, 3, 2], 1);
    for (i, token) in [4, 5, 6].into_iter().enumerate() {
        actual.extend(resident(
            &mut e,
            &mut states,
            &mut m,
            &[token],
            i as u64 + 2,
        ));
    }
    let mut reference = f.cpu();
    let mut rm = manager();
    let mut rs = vec![reference.create_sequence_state().unwrap()];
    let expected = resident(&mut reference, &mut rs, &mut rm, &[1, 3, 2, 4, 5, 6], 1);
    for (a, b) in actual.iter().zip(&expected) {
        close(a, b);
    }
    assert_eq!(states[0].core().position(), 6);
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use ferrule_common::host_kv::placement::{PlacementBudget, PlacementCredits};
    #[test]
    #[ignore = "sole GPU owner only: real two-layer ordinary runner HostKV forward"]
    fn ordinary_two_layer_append_matches_cpu_across_decode_abort_and_late_unknown() {
        let f = Fixture::new();
        let resources = f.resources();
        let options = f.options();
        let estimate = HybridCudaMemoryEstimate::for_resources(&resources, &options, 32).unwrap();
        let device = HybridCudaDevice::new_on_device(
            0,
            HybridCudaMemoryBudget {
                kv_pages: 32,
                // Pure-GQA models have no recurrent state estimate. The CUDA
                // device budget contract still requires a nonzero state domain;
                // this fixture reserves one byte without changing production
                // admission semantics.
                state_bytes: (estimate.per_sequence_state_bytes * 16).max(1),
                weight_bytes: estimate.weight_bytes_upper_bound,
                workspace_bytes: estimate.workspace_bytes_upper_bound,
            },
        )
        .unwrap();
        let runner = GenericDecoderRunner::<HybridCudaDecoder>::hybrid_cuda(
            resources,
            f.tokenizer(),
            options,
            device.clone(),
        )
        .unwrap();
        let mut e = RunnerMultiSessionExecutor::new(runner);
        let mut m = manager();
        let mut states = vec![e.create_sequence_state().unwrap()];
        let mut cpu = f.cpu();
        let mut cm = manager();
        let mut cs = vec![cpu.create_sequence_state().unwrap()];
        let a = resident(&mut e, &mut states, &mut m, &[1, 3, 2], 1);
        let b = resident(&mut cpu, &mut cs, &mut cm, &[1, 3, 2], 1);
        for (a, b) in a.iter().zip(b) {
            close(a, &b);
        }
        // One shared charge, reused sequentially by both layers, never a sum of
        // independent per-layer managers. No resident eviction is performed.
        let budget = PlacementBudget::new(PlacementCredits {
            pageable_host_bytes: 1 << 20,
            pinned_host_bytes: 1 << 20,
            staging_bytes: 1 << 20,
            staging_slots: 4,
            transfers: 4,
            consumers: 4,
        });
        for (i, token) in [4, 5, 6].into_iter().enumerate() {
            let id = i as u64 + 2;
            let tx = ExecutionTransactionId::new(id).unwrap();
            let snapshot =
                (i == 0).then(|| m.capture_prefix_snapshot(StateSlot::new(0), 3).unwrap());
            let r = m.reserve(StateSlot::new(0), 0, 1).unwrap();
            let b = batch(&m, &r, &[token]);
            let prefix = m.derive_append_prefix(&r, tx, 0).unwrap();
            e.prepare_host_kv_decode(&prefix, &mut states, &b).unwrap();
            let out = logits(
                e.execute_host_kv_decode(&prefix, &mut states, &b, &budget, 2)
                    .unwrap(),
            );
            assert!(
                e.execute_host_kv_decode(&prefix, &mut states, &b, &budget, 2)
                    .is_err(),
                "same transaction cannot replay forward"
            );
            assert_eq!(
                budget.in_use(),
                PlacementCredits::default(),
                "all layers and gate/linear consumers retired"
            );
            drop(prefix);
            publish(&mut e, &mut states, &mut m, r, tx);
            if let Some(snapshot) = snapshot {
                let retired = m.release_prefix_snapshot(snapshot).unwrap();
                e.release_kv_pages(retired.pages()).unwrap();
                m.confirm_page_retirement(retired).unwrap();
            }
            let expected = resident(&mut cpu, &mut cs, &mut cm, &[token], id);
            close(&out[0], &expected[0]);
        }
        // Prepare then cancel before executing: no layer charge, no frontier change.
        let tx = ExecutionTransactionId::new(10).unwrap();
        let before = states[0].core().position();
        let r = m.reserve(StateSlot::new(0), 0, 1).unwrap();
        let b = batch(&m, &r, &[7]);
        let prefix = m.derive_append_prefix(&r, tx, 0).unwrap();
        e.prepare_host_kv_decode(&prefix, &mut states, &b).unwrap();
        drop(prefix);
        assert_eq!(
            e.end_transaction(tx, &mut states, TransactionEndIntent::Abort)
                .unwrap(),
            TransactionEndProgress::Complete
        );
        let retired = m.abort_reservations(vec![r]).unwrap();
        m.confirm_page_retirement(retired).unwrap();
        assert_eq!(states[0].core().position(), before);
        assert_eq!(budget.in_use(), PlacementCredits::default());
        assert!(!device.needs_quarantine());

        // Fail live-authority revalidation only AFTER real Entry resources and
        // tail writes have been installed. This adapter can only deny access;
        // it never creates authority from copied witness metadata.
        struct FailAfterAdmission<'a> {
            live: &'a dyn ferrule_backend::cuda::operators::kv::transaction::append_bridge::AppendPrefixAuthority,
            budget: &'a PlacementBudget,
            failed: std::cell::Cell<bool>,
        }
        use ferrule_backend::cuda::operators::kv::transaction::append_bridge::AppendPrefixAuthority;
        // SAFETY: retain the actual live reservation authority, delegate its
        // stable witness/layer and validate before injecting a denial only.
        #[expect(
            unsafe_code,
            reason = "Fault adapter borrows real live authority and only adds validation failure"
        )]
        unsafe impl AppendPrefixAuthority for FailAfterAdmission<'_> {
            fn witness(&self) -> &ferrule_common::host_kv::HostKvAppendReadWitness {
                self.live.witness()
            }
            fn layer(&self) -> usize {
                self.live.layer()
            }
            fn validate(&self) -> ferrule_common::Result<()> {
                self.live.validate()?;
                if self.failed.get() || self.budget.in_use() != PlacementCredits::default() {
                    self.failed.set(true);
                    return Err(ferrule_common::Error::Execution {
                        message: "injected live-authority loss after admission".into(),
                    });
                }
                Ok(())
            }
        }
        let mut runner = e.into_runner();
        let tx = ExecutionTransactionId::new(11).unwrap();
        let r = m.reserve(StateSlot::new(0), 0, 1).unwrap();
        let b = batch(&m, &r, &[7]);
        let prefix = m.derive_append_prefix(&r, tx, 0).unwrap();
        prefix
            .prepare_cuda_decode(&mut runner, &mut states, &b)
            .unwrap();
        let next = prefix.for_layer(1).unwrap();
        let fault = FailAfterAdmission {
            live: &prefix,
            budget: &budget,
            failed: std::cell::Cell::new(false),
        };
        let ctx = CudaHostKvAppendContext::new(vec![&fault, &next], &budget, 2).unwrap();
        assert!(
            runner
                .execute_host_kv_decode(tx, &mut states, &b, &ctx)
                .is_err()
        );
        assert!(fault.failed.get());
        let held = budget.in_use();
        assert_ne!(held, PlacementCredits::default());
        drop(ctx);
        drop(next);
        drop(prefix);
        for _ in 0..2 {
            // Actual late completion cannot rehabilitate the earlier Unknown.
            device.operators().sync_stream().unwrap();
            assert!(!matches!(
                runner.end_transaction(tx, &mut states, TransactionEndIntent::Abort),
                Ok(TransactionEndProgress::Complete)
            ));
            assert_eq!(budget.in_use(), held);
            assert_eq!(states[0].core().position(), before);
        }
        // Keep the logical reservation pending; never acknowledge physical abort.
        assert!(m.reserve(StateSlot::new(0), 0, 1).is_err());
        drop(r);
    }

    #[test]
    fn host_kv_context_requires_all_layers_of_the_same_live_reservation() {
        let mut m = manager();
        let r = m.reserve(StateSlot::new(0), 0, 3).unwrap();
        let p = m
            .prepare_commit(vec![KvReservationCommit::new(r, 3)])
            .unwrap();
        let retired = m.publish_commit(p);
        m.confirm_page_retirement(retired).unwrap();
        let r = m.reserve(StateSlot::new(0), 0, 1).unwrap();
        let prefix = m
            .derive_append_prefix(&r, ExecutionTransactionId::new(1).unwrap(), 0)
            .unwrap();
        let other = prefix.for_layer(1).unwrap();
        let budget = PlacementBudget::new(PlacementCredits::default());
        assert!(CudaHostKvAppendContext::new(vec![&prefix], &budget, 2).is_err());
        assert!(CudaHostKvAppendContext::new(vec![&prefix, &prefix], &budget, 2).is_err());
        assert!(CudaHostKvAppendContext::new(vec![&prefix, &other], &budget, 0).is_err());
        let context = CudaHostKvAppendContext::new(vec![&prefix, &other], &budget, 2).unwrap();
        drop(context);
        assert_eq!(budget.in_use(), PlacementCredits::default());
    }
}
