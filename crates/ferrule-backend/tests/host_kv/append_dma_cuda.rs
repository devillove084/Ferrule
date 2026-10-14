//! Compile-only here; GPU owner must explicitly select an idle device to run.
use super::*;
use crate::cpu::{
    PagedKvSequence, StreamingF32GqaConfig, StreamingF32GqaRequest, streaming_f32_gqa_partial,
};
use ferrule_common::execution::{
    KvCowReplacement, KvElementType, KvLayoutSchemaIdentity, KvReservationView, StateSlot,
};
use ferrule_common::host_kv::placement::{PlacementBudget, PlacementCredits};
use ferrule_common::host_kv::{CommittedKvSource, HostKvAppendReadWitness};

#[test]
#[ignore = "requires an explicitly selected idle CUDA device; no GPU run during implementation"]
fn transaction_prefix_dma_reads_original_and_holds_budget_until_consumer() {
    for cow in [false, true] {
        let ops = Rc::new(CudaOperators::new().unwrap());
        let planes = [
            KvPlaneDescriptor::new("standard_gqa.key", 1, 1, KvElementType::F32),
            KvPlaneDescriptor::new("standard_gqa.value", 1, 1, KvElementType::F32),
        ];
        let mut pool = CudaPagedKvPool::new(ops.clone(), &planes, 4, 3).unwrap();
        let seed = pool
            .prepare(PagedKvPrepare {
                transaction: ExecutionTransactionId::new(1).unwrap(),
                new_pages: &[KvPageId(1)],
                writable_pages: &[],
                cow_replacements: &[],
                protected_pages: &[],
            })
            .unwrap();
        {
            let mut inner = pool.inner.borrow_mut();
            let slot = inner.entries[&seed.id].plan.slot(KvPageId(1)).unwrap() as usize;
            for (plane, values) in [[0., 1., 2., 0.], [2., 4., 8., 0.]].iter().enumerate() {
                ops.overwrite_f32_range(
                    values,
                    inner.storage_mut().plane_storage_mut(plane).unwrap(),
                    slot * 4,
                )
                .unwrap();
            }
        }
        ops.sync_stream().unwrap();
        assert_eq!(
            pool.commit(&mut Some(seed)).unwrap(),
            KvEndProgress::Complete
        );
        pool.bind_committed_logical_owner(7).unwrap();
        let replacement = cow.then_some(KvCowReplacement {
            logical_page: 0,
            source: KvPageId(1),
            replacement: KvPageId(2),
        });
        let source = CommittedKvSource {
            owner: 7,
            state_slot: StateSlot::new(0),
            generation: 1,
            incarnation: 1,
            schema: KvLayoutSchemaIdentity::new(planes.to_vec(), 4, 8),
            committed_tokens: 3,
            positions: 0..3,
            pages: vec![KvPageId(1)],
        };
        let id = ExecutionTransactionId::new(2).unwrap();
        let witness = HostKvAppendReadWitness::new(
            id,
            1,
            source,
            KvReservationView {
                state_slot: StateSlot::new(0),
                execution_state_slot: StateSlot::new(0),
                generation: 1,
                execution_generation: 1,
                positions: 3..4,
                newly_allocated: vec![],
                cow_replacement: replacement,
            },
        )
        .unwrap();
        let cows: Vec<_> = replacement.into_iter().collect();
        let writes = if cow { vec![] } else { vec![KvPageId(1)] };
        let mut tx = pool
            .prepare(PagedKvPrepare {
                transaction: id,
                new_pages: &[],
                writable_pages: &writes,
                cow_replacements: &cows,
                protected_pages: &[],
            })
            .unwrap();
        let batch = PagedKvBatch {
            row_sequence_ids: vec![0].into(),
            row_positions: vec![3].into(),
            sequences: vec![PagedKvSequence {
                sequence_len: 4,
                block_table: witness.pages().into(),
            }]
            .into(),
        };
        pool.enter(&mut tx, batch.clone()).unwrap();
        let mut view = pool.active_view(&mut tx).unwrap();
        // Deliberately poison the projected copy: the DMA must read original,
        // not merely a slot with a matching logical block table.
        view.with_f32_gqa_planes(&ops, 0, &batch, |p| {
            let base = p.block_slots[0] as usize * 4;
            ops.overwrite_f32_range(&[100.; 4], p.key, base)?;
            ops.overwrite_f32_range(&[-100.; 4], p.value, base)
        })
        .unwrap();
        let capacity = pool.capacity();
        let config = StreamingF32GqaConfig {
            rows: 1,
            q_heads: 1,
            kv_heads: 1,
            head_dim: 1,
            block_tokens: 2,
            softmax_scale: 1.,
            sliding_window: None,
        };
        let credits = super::super::committed_gqa_credits(config, 1).unwrap();
        let budget = PlacementBudget::new(credits);
        {
            let mut read = pool.derive_append_prefix(&tx, &witness, 0).unwrap();
            let rejected = PlacementBudget::new(PlacementCredits::default());
            assert!(
                read.start_cpu_prefix(
                    &ops,
                    ops.upload_f32_buffer(&[1.]).unwrap(),
                    config,
                    &rejected
                )
                .is_err()
            );
            assert_eq!(rejected.in_use(), PlacementCredits::default());
            read.start_cpu_prefix(&ops, ops.upload_f32_buffer(&[1.]).unwrap(), config, &budget)
                .unwrap();
            assert_eq!(budget.in_use(), credits);
            assert!(!read.finish().unwrap());
            assert!(
                view.with_f32_gqa_planes(&ops, 0, &batch, |_| Ok(()))
                    .is_err()
            );
            assert!(pool.clone().preempt(&[KvPageId(1)]).is_err());
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while !read.poll_cpu_prefix().unwrap() {
                assert!(std::time::Instant::now() < deadline, "prefix DMA timed out");
                std::thread::yield_now();
            }
            assert!(!read.finish().unwrap());
            assert_eq!(budget.in_use(), credits);
            let expected = streaming_f32_gqa_partial(StreamingF32GqaRequest {
                config,
                query: &[1.],
                key: &[0., 1., 2.],
                value: &[2., 4., 8.],
                query_start: 3,
                kv_start: 0,
            })
            .unwrap()
            .output();
            read.consume_cpu_prefix(|partial| {
                assert!((partial.output()[0] - expected[0]).abs() < 1e-6);
                Ok(())
            })
            .unwrap();
            assert_eq!(budget.in_use(), credits);
            assert!(read.finish().unwrap());
        }
        assert_eq!(budget.in_use(), PlacementCredits::default());
        assert_eq!(pool.capacity(), capacity); // Copying/consuming is not eviction.
        pool.leave(&mut tx).unwrap();
        ops.sync_stream().unwrap();
        assert_eq!(pool.abort(&mut Some(tx)).unwrap(), KvEndProgress::Complete);
        pool.release(&[KvPageId(1)]).unwrap();
        pool.shutdown().unwrap();
    }
}
