use ferrule_backend::cpu::{
    CpuExecutionPrecision, CpuKvView, CpuPagedKvPool, ExpertLinearRef, ExpertSwiGluRef, HostRows,
    KvEndProgress, PagedCausalGqa, PagedKvBatch, PagedKvHistory, PagedKvPrepare, PagedKvSequence,
    RowsDType, RowsShape, execute_reference_expert, paged_causal_gqa,
};
use ferrule_common::execution::{
    ExecutionTransactionId, KvElementType, KvPageId, KvPlaneDescriptor,
};

fn transaction(value: u64) -> ExecutionTransactionId {
    ExecutionTransactionId::new(value).unwrap()
}

fn entered_pool() -> (
    CpuPagedKvPool,
    ferrule_backend::cpu::CpuPagedKvTransaction,
    CpuKvView,
) {
    let planes = [
        KvPlaneDescriptor::new("key", 2, 1, KvElementType::Bf16),
        KvPlaneDescriptor::new("value", 2, 1, KvElementType::Bf16),
    ];
    let mut pool = CpuPagedKvPool::new(planes, 2, 2).unwrap();
    assert_eq!(pool.planes(), planes);
    let mut tx = pool
        .prepare(PagedKvPrepare {
            transaction: transaction(1),
            new_pages: &[KvPageId(7)],
            writable_pages: &[],
            cow_replacements: &[],
            protected_pages: &[],
        })
        .unwrap();
    pool.enter(
        &mut tx,
        PagedKvBatch {
            row_sequence_ids: vec![0, 0].into_boxed_slice(),
            row_positions: vec![0, 1].into_boxed_slice(),
            sequences: vec![PagedKvSequence {
                sequence_len: 2,
                block_table: vec![KvPageId(7)].into_boxed_slice(),
            }]
            .into_boxed_slice(),
        },
    )
    .unwrap();
    let view = pool.active_view(&mut tx).unwrap();
    (pool, tx, view)
}

#[test]
fn typed_paged_kv_appends_bf16_and_serves_causal_gqa() {
    let (mut pool, mut tx, mut view) = entered_pool();
    view.append(
        0,
        &[0, 0],
        &[0, 1],
        1,
        2,
        &[1.0, 0.0, 0.0, 1.0],
        &[2.0, 3.0, 5.0, 7.0],
    )
    .unwrap();
    let query = HostRows::new(
        RowsShape::new(2, 2).unwrap(),
        RowsDType::Bf16,
        None,
        vec![1.0, 0.0, 0.0, 1.0],
    )
    .unwrap();
    let output = paged_causal_gqa(
        &view,
        0,
        PagedCausalGqa {
            query: &query,
            row_sequence_ids: &[0, 0],
            row_positions: &[0, 1],
            query_heads: 1,
            kv_heads: 1,
            head_dim: 2,
            softmax_scale: 1.0,
            arena: None,
        },
        CpuExecutionPrecision::F32,
    )
    .unwrap();
    assert_eq!(&output.values()[..2], &[2.0, 3.0]);
    assert!(output.values()[2] > 3.0 && output.values()[2] < 5.0);
    assert!(output.values()[3] > 5.0 && output.values()[3] < 7.0);

    let history = view.history(0, 0, 1, 1, 2).unwrap();
    assert_eq!(history.key, vec![1.0, 0.0, 0.0, 1.0]);
    pool.leave(&mut tx).unwrap();
    let mut tx = Some(tx);
    assert_eq!(pool.commit(&mut tx).unwrap(), KvEndProgress::Complete);
    assert!(tx.is_none());
    assert_eq!(pool.physical_slot(KvPageId(7)), Some(0));
}

#[test]
fn reference_expert_consumes_packed_fp4_without_model_math() {
    let mut gate_weight = vec![0u8; 16];
    gate_weight[0] = 0x02; // first logical value is 1.0
    let mut up_weight = vec![0u8; 16];
    up_weight[0] = 0x03; // first logical value is 1.5
    let mut down_weight = vec![0u8; 16];
    down_weight[0] = 0x04; // first logical value is 2.0
    let scales = [127u8];
    let gate = ExpertLinearRef::Fp4E2M1E8M0 {
        weight: &gate_weight,
        scales: &scales,
        out_features: 1,
        in_features: 32,
        block_size: 32,
    };
    let up = ExpertLinearRef::Fp4E2M1E8M0 {
        weight: &up_weight,
        scales: &scales,
        out_features: 1,
        in_features: 32,
        block_size: 32,
    };
    let down = ExpertLinearRef::Fp4E2M1E8M0 {
        weight: &down_weight,
        scales: &scales,
        out_features: 1,
        in_features: 32,
        block_size: 32,
    };
    let mut input = vec![0.0; 32];
    input[0] = 2.0;
    let mut padded_hidden = vec![0.0; 32];
    let gate_value = 2.0f32;
    let expected_hidden = gate_value / (1.0 + (-gate_value).exp()) * 3.0 * 0.5;
    padded_hidden[0] = expected_hidden;
    let expected = ferrule_backend::cpu::expert_linear(down, &padded_hidden).unwrap();

    // Gate/up are [1, 32], so the down contract for a realistic expert would be
    // [32, 1]. Use BF16 for the tiny shape-preserving down projection here.
    let down_bf16 = ferrule_backend::cpu::bf16_rne_word(2.0).to_le_bytes();
    let output = execute_reference_expert(
        ExpertSwiGluRef {
            gate,
            up,
            down: ExpertLinearRef::Bf16 {
                weight: &down_bf16,
                out_features: 1,
                in_features: 1,
            },
            activation_limit: None,
        },
        &input,
        0.5,
    )
    .unwrap();
    assert!((output[0] - 2.0 * expected_hidden).abs() < 1e-6);
    assert_eq!(expected.len(), 1);
}
