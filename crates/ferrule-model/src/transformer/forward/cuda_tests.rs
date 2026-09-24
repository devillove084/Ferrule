//! Unknown-completion regression using the existing tiny hybrid CPU oracle.
use super::*;
use crate::execution::ExecutionPrecisionPolicy;
use crate::nn::{
    DTypeConstraint, ModulePath, ParameterDType, ParameterId, ParameterResidency, ParameterSpec,
};
use crate::transformer::*;
use crate::{CheckpointDType, CheckpointTensorSlice, TensorRole};
use std::sync::Arc;

// Same descriptor and semantic mapping as tests/hybrid_cpu.rs.
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

fn oracle_resources(path: &std::path::Path) -> BoundDecoderResources {
    let oracle: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/hybrid_cpu/oracle.json"
    ))
    .unwrap();
    let mut schema = StateDictSchema::builder();
    let mut mapper = ExactNameMapper::new();
    let mut bytes = Vec::new();
    let mut slices = Vec::new();
    for (i, (name, tensor)) in oracle["tensors"].as_object().unwrap().iter().enumerate() {
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
        let module = ModulePath::new(name).unwrap();
        let residency = if name.starts_with("layers.") {
            ParameterResidency::layer(name.split('.').nth(1).unwrap().parse().unwrap())
        } else {
            ParameterResidency::Static
        };
        schema
            .register_with_role(
                ParameterSpec::new(
                    ParameterId::new(i as u64 + 1),
                    module.clone(),
                    DTypeConstraint::exact(ParameterDType::F32),
                    shape.clone(),
                    residency,
                )
                .unwrap(),
                role(name),
            )
            .unwrap();
        mapper.insert(name, NameMapping::weight(module)).unwrap();
        slices.push(CheckpointTensorSlice {
            name: name.clone(),
            role: role(name),
            path: path.into(),
            offset: offset as u64,
            bytes: (bytes.len() - offset) as u64,
            dtype: CheckpointDType::F32,
            shape,
        });
    }
    std::fs::write(path, bytes).unwrap();
    let schema = schema.build().unwrap();
    let state_dict = StateDictBinder::new(&schema, &mapper)
        .bind_slices(slices)
        .unwrap();
    BoundDecoderResources::new(spec(), Arc::new(state_dict)).unwrap()
}

#[test]
#[ignore = "requires CUDA GPU; tiny non-destructive completion fault"]
fn hybrid_unknown_finish_keeps_transaction_pins_despite_independent_kv_fence() {
    let path = std::env::temp_dir().join(format!(
        "ferrule-forward-unknown-{}.f32",
        std::process::id()
    ));
    let resources = oracle_resources(&path);
    let options = GenericDecoderOptions::standard_cpu(
        resources.spec(),
        ModelFamily::Unknown("hybrid-oracle".into()),
        WeightSource::Safetensors,
        2,
        32,
        32,
        4,
        1 << 20,
        ExecutionPrecisionPolicy::f32(),
    )
    .unwrap();
    let estimate = HybridCudaMemoryEstimate::for_resources(&resources, &options, 4).unwrap();
    let device = HybridCudaDevice::new_on_device(
        0,
        HybridCudaMemoryBudget {
            kv_pages: 4,
            state_bytes: estimate.per_sequence_state_bytes * 16,
            weight_bytes: estimate.weight_bytes_upper_bound,
            workspace_bytes: estimate.workspace_bytes_upper_bound,
        },
    )
    .unwrap();
    let tokenizer = TokenizerHandle::from_parts(
        tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default()),
        None,
    );
    let mut runner = GenericDecoderRunner::<HybridCudaDecoder>::hybrid_cuda(
        resources,
        tokenizer,
        options,
        device.clone(),
    )
    .unwrap();
    let mut states = vec![runner.create_sequence_state().unwrap()];
    let original = (states[0].topology_id(), states[0].core().clone());
    let (batch, reservations) = input(&states[0]);
    runner
        .prepare_multi_session_batch(tx(), &mut states, &batch, &reservations)
        .unwrap();
    let charged = device.live_state_bytes();
    runner
        .forward_executor()
        .module()
        .reject_next_finish_completion();
    let error = runner
        .execute_multi_session_batch_progress(tx(), &mut states, &batch)
        .unwrap_err();
    assert_unknown(&error);
    assert!(device.needs_quarantine());
    for _ in 0..3 {
        // This is the same physical operator/compute stream used by the KV pool.
        // Its independent fence succeeds but cannot discharge module custody.
        device
            .operators()
            .record_compute_event()
            .unwrap()
            .synchronize()
            .unwrap();
        device
            .operators()
            .record_upload_event()
            .unwrap()
            .synchronize()
            .unwrap();
        let error = runner
            .end_transaction(tx(), &mut states, TransactionEndIntent::Abort)
            .unwrap_err();
        assert_unknown(&error);
        assert_eq!(runner.observability_snapshot().active_transactions, 1);
        assert_eq!(runner.backend().capacity().active_transactions, 1);
        assert_eq!(runner.backend().pool().active_transaction_count(), 1);
        assert_eq!(runner.backend().pool().free_slot_count(), 3);
        assert_eq!(device.live_state_bytes(), charged);
        assert_eq!(states[0].topology_id(), original.0);
        assert_eq!(states[0].core(), &original.1);
        assert!(runner.forward_executor().arenas.inner.borrow().is_empty());
        assert!(runner.release_kv_pages(&[KvPageId(1)]).is_err());
        assert!(
            runner
                .backend_mut()
                .pool_mut()
                .release(&[KvPageId(1)])
                .is_err()
        );
        assert!(
            runner
                .backend_mut()
                .pool_mut()
                .preempt(&[KvPageId(1)])
                .is_err()
        );
        assert!(
            runner
                .end_transaction(tx(), &mut states, TransactionEndIntent::Publish)
                .is_err()
        );
        assert!(runner.create_sequence_state().is_err());
        assert!(runner.shutdown().is_err());
    }
    // Preserve the unknown owner, working states, pending rows and arena rather
    // than using destructor behavior to manufacture a successful cleanup.
    std::mem::forget(runner);
    std::mem::forget(states);
    std::mem::forget(device);
    std::fs::remove_file(path).unwrap();
}

fn assert_unknown(error: &Error) {
    let Error::ModelSource { source } = error else {
        panic!("expected typed unknown, got {error:?}")
    };
    assert!(
        source
            .downcast_ref::<HybridCudaCompletionUnknown>()
            .is_some()
    );
}
