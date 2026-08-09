use std::num::NonZeroU32;
use std::path::PathBuf;

use ferrule_common::execution::{
    ExecutionBatch, ExecutionSequence, ExecutionTransactionId, ForwardMode, ForwardPhase,
    KvBlockId, KvElementType, KvLayoutSchema, KvPageId, KvReservationView, KvWriteSlot,
    LogitsOutput, LogitsRequest, StateSlot,
};
use ferrule_model::models::qwen3::{Qwen3MoeAdapter, Qwen3MoePrepareOptions};
use ferrule_model::{
    ModelExecutionBackend, ModelFamily, ModelRunner, MultiSessionBatchProgress, MultiSessionRunner,
    ResidentModelRunner, TransactionEndIntent, TransactionEndProgress,
};

#[test]
fn qwen_adapter_uses_real_standard_kv_and_rejects_cuda() {
    let model_dir = model_dir();
    let adapter =
        Qwen3MoeAdapter::load_hf_with_options(&model_dir, 1024 * 1024 * 1024, one_layer_options())
            .unwrap();

    assert_eq!(adapter.backend_name(), "cpu");
    assert_eq!(adapter.resources().spec().layers().len(), 48);
    assert_eq!(adapter.kv_schema().planes().len(), 2);
    assert_eq!(adapter.kv_schema().planes()[0].name, "standard_gqa.key");
    assert_eq!(adapter.kv_schema().planes()[1].name, "standard_gqa.value");
    assert_eq!(adapter.kv_schema().planes()[0].layer_count, 1);
    assert_eq!(
        adapter.kv_schema().planes()[0].element_type,
        KvElementType::Bf16
    );
    assert_eq!(
        adapter.kv_schema().planes()[1].element_type,
        KvElementType::Bf16
    );
    assert_eq!(adapter.kv_schema().checked_page_bytes(), Some(32_768));

    let error = Qwen3MoeAdapter::load_hf_with_options_and_backend(
        &model_dir,
        1024 * 1024 * 1024,
        one_layer_options(),
        ModelExecutionBackend::Cuda,
    )
    .err()
    .unwrap();
    assert!(error.to_string().contains("CUDA is unsupported"));
}

#[test]
#[ignore = "loads models/Qwen3-30B-A3B and executes one CPU token through GenericDecoderRunner"]
fn real_qwen3_one_layer_one_token_generic_runner_smoke() {
    let adapter = Qwen3MoeAdapter::load_hf_with_options(
        &model_dir(),
        1024 * 1024 * 1024,
        one_layer_options(),
    )
    .unwrap();
    let token = adapter.config().bos_token_id;
    let mut runner = adapter.into_decoder(16, 1, 1).unwrap();

    let info = runner.model_info();
    assert_eq!(info.family, ModelFamily::QwenMoe);
    assert_eq!(info.architecture.as_deref(), Some("Qwen3MoeForCausalLM"));
    assert_eq!(info.num_layers, 1);
    assert_eq!(info.backend, "cpu");
    assert_eq!(runner.bound_layer_count(), Some(1));
    assert_eq!(runner.eos_token_ids(), [151645, 151643]);
    assert_eq!(runner.observability_snapshot().bound_layers, 1);

    runner.configure_kv_page_capacity(2).unwrap();
    let mut states = vec![runner.create_sequence_state().unwrap()];
    let state_slot = StateSlot::new(0);
    let page = KvPageId(900);
    let batch = ExecutionBatch::new(
        ForwardMode::Prefill,
        vec![token],
        vec![0],
        vec![Some(KvWriteSlot::new(page.0 * 16))],
        vec![LogitsRequest::TopK(NonZeroU32::new(1).unwrap())],
        vec![ExecutionSequence::new(
            state_slot,
            ForwardPhase::Prefill,
            0..1,
            0,
            1,
            0..1,
        )],
        vec![KvBlockId::new(page.0)],
    );
    let reservations = vec![KvReservationView {
        state_slot,
        execution_state_slot: state_slot,
        positions: 0..1,
        newly_allocated: vec![page],
        generation: states[0].core().generation(),
        execution_generation: states[0].core().generation(),
        cow_replacement: None,
    }];
    let transaction = ExecutionTransactionId::new(900).unwrap();

    runner
        .prepare_multi_session_batch(transaction, &mut states, &batch, &reservations)
        .unwrap();
    let output = match runner
        .execute_multi_session_batch_progress(transaction, &mut states, &batch)
        .unwrap()
    {
        MultiSessionBatchProgress::Complete(output) => output,
        MultiSessionBatchProgress::Waiting(_) => panic!("CPU standard decoder suspended"),
    };
    assert_eq!(output.logits.len(), 1);
    let LogitsOutput::TopK(logits) = &output.logits[0].logits else {
        panic!("single-token smoke did not return top-k logits")
    };
    assert_eq!(logits.len(), 1);
    assert!(logits[0].logit.is_finite());
    assert_eq!(
        runner
            .end_transaction(transaction, &mut states, TransactionEndIntent::Publish)
            .unwrap(),
        TransactionEndProgress::Complete
    );
    assert_eq!(states[0].core().position(), 1);
    assert_eq!(runner.observability_snapshot().completed_batches, 1);
}

fn one_layer_options() -> Qwen3MoePrepareOptions {
    Qwen3MoePrepareOptions {
        page_size: 16,
        max_dense_tensor_bytes: 1024 * 1024 * 1024,
        max_expert_tensor_bytes: 256 * 1024 * 1024,
        max_layers: Some(1),
    }
}

fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("models/Qwen3-30B-A3B")
}
