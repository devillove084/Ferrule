//! Real model/backend numerical fixture. Compile only unless GPU owner opts in.
use super::*;
use ferrule_backend::cpu::{
    StreamingF32GqaConfig, StreamingF32GqaRequest, streaming_f32_gqa_partial,
};
use ferrule_backend::cuda::context::CudaOperators;
use ferrule_backend::cuda::operators::{SplitHalfRopeLayout, kv::committed_gqa_credits};
use ferrule_common::execution::*;
use ferrule_common::host_kv::placement::{PlacementBudget, PlacementCredits};
use ferrule_model::decoder::*;
use ferrule_model::execution::ExecutionPrecisionPolicy;
use ferrule_model::transformer::{
    CudaRows, CudaStandardDecoderOperators, GqaMetadata, GqaRequest, Rows, RowsShape,
};
use ferrule_runtime::io::host_kv_adapters::AppendGqaBinding;
use std::rc::Rc;
type State = DecoderSequenceState<(), ()>;
fn capabilities() -> ExecutionCapabilities {
    ExecutionCapabilities {
        max_batch_tokens: 8,
        max_sequences: 1,
        max_prefill_query_tokens_per_sequence: 8,
        max_decode_query_tokens_per_sequence: 1,
        max_top_k: None,
        supports_prefill: true,
        supports_decode: true,
        supports_mixed: false,
        full_logits_width: None,
        kv_binding_mode: KvBindingMode::Paged,
        logits_row_policy: LogitsRowPolicy::None,
    }
}
fn packed(
    manager: &KvPageManager,
    reservation: &ferrule_runtime::KvReservation,
    states: &[State],
    backend: &PagedKvBackend<TypedCudaPagedKvPool<State>>,
) -> PackedDecoderBatch {
    let start = reservation.positions.start;
    let len = reservation.positions.len();
    let bindings = manager.reservation_bindings(reservation.view()).unwrap();
    let (mode, phase) = if start == 0 {
        (ForwardMode::Prefill, ForwardPhase::Prefill)
    } else {
        (ForwardMode::Decode, ForwardPhase::Decode)
    };
    let batch = ExecutionBatch::new(
        mode,
        vec![1; len],
        (start as u32..(start + len) as u32).collect(),
        bindings.write_slots.into_iter().map(Some).collect(),
        vec![LogitsRequest::None; len],
        vec![ExecutionSequence::new(
            StateSlot::new(0),
            phase,
            0..len as u32,
            start as u32,
            (start + len) as u32,
            0..bindings.block_ids.len() as u32,
        )],
        bindings.block_ids,
    );
    PackedDecoderBatch::lower(
        &batch,
        &[reservation.view().clone()],
        states,
        &capabilities(),
        4,
        backend,
    )
    .unwrap()
}
fn prepare(
    backend: &mut PagedKvBackend<TypedCudaPagedKvPool<State>>,
    states: &[State],
    packed: &PackedDecoderBatch,
    id: u64,
    start: usize,
) -> PagedKvTransactionHandle {
    backend
        .prepare(DecoderKvPrepare {
            transaction: ExecutionTransactionId::new(id).unwrap(),
            sequences: &[DecoderKvSequenceCustody {
                source_index: 0,
                topology_id: states[0].topology_id(),
                page_state_slot: StateSlot::new(0),
                page_generation: 0,
                execution_generation: 0,
                context_len: start,
                query_len: packed.len(),
            }],
            new_pages: packed.new_pages(),
            writable_pages: packed.writable_pages(),
            cow_replacements: packed.cow_replacements(),
            protected_pages: packed.protected_pages(),
            capacity: backend.capacity(),
            page_statuses: &[],
        })
        .unwrap()
}
#[test]
#[ignore = "GPU owner only: actual runtime reservation, model append hook and Q/K/V DMA"]
fn reserved_prefix_model_append_shadow_cow_rope_and_output_custody() {
    for (committed, cow, consumer_failure) in [
        (3, false, false),
        (3, true, false),
        (4, false, false),
        (7, true, false),
        (3, false, true),
    ] {
        let ops = Rc::new(CudaOperators::new().unwrap());
        let mut operators =
            CudaStandardDecoderOperators::new(ops.clone(), ExecutionPrecisionPolicy::f32(), &[])
                .unwrap();
        let mut backend = PagedKvBackend::new(
            TypedCudaPagedKvPool::<State>::new(
                ops.clone(),
                KvLayoutSchema::planes(&Schema).iter().copied(),
                4,
                8,
            )
            .unwrap(),
        );
        let mut manager = KvPageManager::new(Box::new(Schema), 8);
        manager.alloc_sequence(StateSlot::new(0), 0).unwrap();
        let mut states = vec![State::new((), ())];
        let seed = manager.reserve(StateSlot::new(0), 0, committed).unwrap();
        let batch = packed(&manager, &seed, &states, &backend);
        let mut tx = prepare(&mut backend, &states, &batch, 1, 0);
        backend.enter(&mut tx, &batch, &mut states).unwrap();
        let mut keys: Vec<_> = (0..committed * 2).map(|i| i as f32 * 0.1 - 0.5).collect();
        let mut values: Vec<_> = (0..committed * 2).map(|i| (i % 5) as f32 - 2.).collect();
        backend
            .active_view(&mut tx)
            .unwrap()
            .with_f32_gqa_planes(&ops, 0, &batch, batch.protected_pages(), |p| {
                for row in 0..committed {
                    let offset = p
                        .layout
                        .resolve_row_offset(0, row, p.key.len(), p.block_slots, p.block_offsets)
                        .unwrap();
                    ops.overwrite_f32_range(&keys[row * 2..row * 2 + 2], p.key, offset)?;
                    ops.overwrite_f32_range(&values[row * 2..row * 2 + 2], p.value, offset)?;
                }
                Ok(())
            })
            .unwrap();
        backend.leave(&mut tx).unwrap();
        ops.sync_stream().unwrap();
        assert_eq!(
            backend.commit(&mut Some(tx)).unwrap(),
            KvEndProgress::Complete
        );
        let step = states[0].core().begin_step().unwrap();
        states[0].core_mut().commit_step(step, committed).unwrap();
        let p = manager
            .prepare_commit(vec![KvReservationCommit::new(seed, committed)])
            .unwrap();
        let retired = manager.publish_commit(p);
        manager.confirm_page_retirement(retired).unwrap();
        let snapshot = cow.then(|| {
            manager
                .capture_prefix_snapshot(StateSlot::new(0), committed)
                .unwrap()
        });
        let reservation = manager.reserve(StateSlot::new(0), 0, 1).unwrap();
        assert_eq!(reservation.cow_replacement.is_some(), cow);
        let batch = packed(&manager, &reservation, &states, &backend);
        let prefix = manager
            .derive_append_prefix(&reservation, ExecutionTransactionId::new(2).unwrap(), 0)
            .unwrap();
        prefix.bind_cuda_pool(&mut backend).unwrap();
        let mut tx = prepare(&mut backend, &states, &batch, 2, committed);
        backend.enter(&mut tx, &batch, &mut states).unwrap();
        let before = backend.capacity();
        let mut view = backend.active_view(&mut tx).unwrap();
        // Destroy only projected prefix bytes to detect wrong-source DMA. This
        // transaction is aborted after the numerical test; original is immutable.
        view.with_f32_gqa_planes(&ops, 0, &batch, batch.protected_pages(), |p| {
            if committed % 4 != 0 {
                let base = p
                    .layout
                    .resolve_row_offset(0, committed, p.key.len(), p.block_slots, p.block_offsets)
                    .unwrap()
                    - (committed % 4) * 2;
                ops.overwrite_f32_range(&vec![90.; (committed % 4) * 2], p.key, base)?;
                ops.overwrite_f32_range(&vec![-90.; (committed % 4) * 2], p.value, base)?;
            }
            Ok(())
        })
        .unwrap();
        // The actual post-RoPE Q and K used by the standard request feed BOTH
        // original prefix and shadow tail; no caller-supplied CPU Q is substituted.
        let mut q = ops.upload_f32_buffer(&[0.5, 1., 1., -0.5]).unwrap();
        let mut k = ops.upload_f32_buffer(&[0.25, -0.75]).unwrap();
        let cosine = ops.upload_f32_buffer(&[0.6; 16]).unwrap();
        let sine = ops.upload_f32_buffer(&[0.8; 16]).unwrap();
        for (buffer, heads) in [(&mut q, 2), (&mut k, 1)] {
            ops.split_half_rope_f32(
                buffer,
                &cosine,
                &sine,
                &[committed as i32],
                SplitHalfRopeLayout {
                    rows: 1,
                    heads,
                    head_dim: 2,
                    rope_dim: 2,
                    table_positions: 16,
                    restore_bf16_boundary: false,
                },
            )
            .unwrap();
        }
        let q_host = [-0.5, 1., 1., 0.5];
        keys.extend_from_slice(&[0.75, -0.25]);
        values.extend_from_slice(&[3., -2.]);
        let query = Rows::Cuda(CudaRows::f32(RowsShape::new(1, 4).unwrap(), None, q).unwrap());
        let key = Rows::Cuda(CudaRows::f32(RowsShape::new(1, 2).unwrap(), None, k).unwrap());
        let value = Rows::Cuda(
            CudaRows::f32(
                RowsShape::new(1, 2).unwrap(),
                None,
                ops.upload_f32_buffer(&[3., -2.]).unwrap(),
            )
            .unwrap(),
        );
        let metadata = GqaMetadata::new(1, vec![0], vec![committed], vec![committed + 1]).unwrap();
        let config = StreamingF32GqaConfig {
            rows: 1,
            q_heads: 2,
            kv_heads: 1,
            head_dim: 2,
            block_tokens: 2,
            softmax_scale: 0.5,
            sliding_window: None,
        };
        let credits = committed_gqa_credits(config, prefix.witness().source().pages.len()).unwrap();
        let budget = PlacementBudget::new(credits);
        let expected = streaming_f32_gqa_partial(StreamingF32GqaRequest {
            config,
            query: &q_host,
            key: &keys,
            value: &values,
            query_start: committed as u64,
            kv_start: 0,
        })
        .unwrap()
        .output();
        let request = || GqaRequest {
            layer: 0,
            query: &query,
            key: &key,
            value: &value,
            metadata: &metadata,
            query_heads: 2,
            kv_heads: 1,
            head_dim: 2,
            softmax_scale: 0.5,
            arena: None,
        };
        let denied = PlacementBudget::new(PlacementCredits::default());
        assert!(
            prefix
                .with_cuda_append(
                    AppendGqaBinding {
                        view: &mut view,
                        batch: &batch,
                        budget: &denied,
                        block_tokens: 2
                    },
                    &mut operators,
                    request(),
                    |_| panic!("denied admission executed")
                )
                .is_err()
        );
        assert_eq!(denied.in_use(), PlacementCredits::default());
        let result = prefix.with_cuda_append(
            AppendGqaBinding {
                view: &mut view,
                batch: &batch,
                budget: &budget,
                block_tokens: 2,
            },
            &mut operators,
            request(),
            |output| {
                assert_eq!(budget.in_use(), credits);
                assert!(backend.leave(&mut tx).is_err()); // output still owns Entry consumer custody
                let actual = ops.download_f32_buffer(output.cuda()?.f32_buffer().unwrap())?;
                for (a, e) in actual.iter().zip(&expected) {
                    assert!((a - e).abs() < 1e-5);
                }
                if consumer_failure {
                    return Err(ferrule_common::Error::Execution {
                        message: "injected output consumer failure".into(),
                    });
                }
                Ok(())
            },
        );
        if consumer_failure {
            assert!(result.is_err());
            assert_eq!(budget.in_use(), credits);
            assert!(backend.leave(&mut tx).is_err());
            assert_eq!(backend.capacity().free_pages, 0);
            // Fail-stop resources intentionally retained; no test-only reset.
            continue;
        }
        result.unwrap();
        assert_eq!(budget.in_use(), PlacementCredits::default());
        assert_eq!(backend.capacity(), before); // no eviction or VRAM saving
        view.with_f32_gqa_planes(&ops, 0, &batch, batch.protected_pages(), |p| {
            let offset = p
                .layout
                .resolve_row_offset(0, committed, p.key.len(), p.block_slots, p.block_offsets)
                .unwrap();
            let actual = ops.download_f32_range(p.key, offset, 2)?;
            assert!((actual[0] - 0.75).abs() < 1e-6 && (actual[1] + 0.25).abs() < 1e-6);
            Ok(())
        })
        .unwrap();
        drop(prefix);
        backend.leave(&mut tx).unwrap();
        ops.sync_stream().unwrap();
        assert_eq!(
            backend.rollback(&mut Some(tx)).unwrap(),
            KvEndProgress::Complete
        );
        let retired = manager.abort_reservations(vec![reservation]).unwrap();
        manager.confirm_page_retirement(retired).unwrap();
        if let Some(snapshot) = snapshot {
            let retired = manager.release_prefix_snapshot(snapshot).unwrap();
            manager.confirm_page_retirement(retired).unwrap();
        }
        assert_eq!(
            manager
                .block_table(StateSlot::new(0))
                .unwrap()
                .committed_tokens(),
            committed
        );
    }
}
