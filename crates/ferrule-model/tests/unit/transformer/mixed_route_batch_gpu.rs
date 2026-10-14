//! GPU gates only: never execute without the existing separate sole-owner grant.
use super::*;
use crate::transformer::standard::cuda::diagnostic::{CudaDiagnosticBuffer, CudaDiagnosticLimits};

#[path = "mixed_route_batch_legacy.rs"]
mod legacy;

thread_local! {
    static GATHERS: Cell<usize> = const { Cell::new(0) };
    static UNKNOWN_AFTER_GATHER: Cell<bool> = const { Cell::new(false) };
    static GATHER_PTR: Cell<u64> = const { Cell::new(0) };
}
pub(crate) fn after_gather_submission(gathered: &CudaBf16Buffer) -> Result<()> {
    GATHERS.set(GATHERS.get() + 1);
    GATHER_PTR.set(gathered.as_device_buffer().cu_deviceptr());
    if UNKNOWN_AFTER_GATHER.replace(false) {
        return Err(crate::decoder::CompletionUnknown::wrap(cuda_error(
            "injected Unknown after real batch gather",
        )));
    }
    Ok(())
}
fn raw_words(
    op: &ferrule_backend::cuda::operators::linear::CudaOperators,
    rows: &Rows,
) -> Vec<u16> {
    rows.cuda()
        .unwrap()
        .bf16_buffer()
        .unwrap()
        .as_device_buffer()
        .to_host_vec(&op.stream_clone())
        .unwrap()
}

#[test]
#[ignore = "sole GPU owner; raw BF16 local partial, frozen legacy gather/combine, M1/M5 overlap/decode cold/hot"]
fn raw_partial_legacy_vs_batch_exact_after_source_retirement() {
    for lane in 0..2 {
        for (rows, per_row) in [(1, 2), (5, 2), (5, 1), (1, 1)] {
            for cached in [false, true] {
                let f = Fixture::new(lane);
                let linears = f.prepare();
                let op = authorized_ops(lane);
                let old_op = authorized_ops(lane);
                let block = if cached {
                    cached_block(&f, &linears)
                } else {
                    f.block(&linears)
                };
                let mut batch = owner(op.clone(), &f, block.clone(), &linears);
                let mut old = owner(old_op.clone(), &f, block, &linears);
                let _runtime = retire(linears);
                std::fs::remove_file(&f.path).unwrap();
                let x = input(&op, rows);
                let old_x = input(&old_op, rows);
                let ids = (0..rows)
                    .flat_map(|row| {
                        if per_row == 2 {
                            vec![1, 0]
                        } else {
                            vec![row % 2]
                        }
                    })
                    .collect();
                let weights = (0..rows)
                    .flat_map(|_| {
                        if per_row == 2 {
                            vec![0.100123, 0.899877]
                        } else {
                            vec![0.100123]
                        }
                    })
                    .collect();
                let routes = RouterRoutes::new(rows, per_row, ids, weights).unwrap();
                for _ in 0..2 {
                    GATHERS.set(0);
                    let before = op.counters();
                    let actual = batch
                        .mixed_rank_routed_local(
                            0,
                            &x,
                            &routes,
                            HybridRoutedAccumulation::HfEagerBf16ExpertOrder,
                        )
                        .unwrap();
                    assert_eq!(GATHERS.get(), 1);
                    assert!(
                        batch
                            .mixed_rank_block
                            .as_ref()
                            .unwrap()
                            .pending_routes
                            .is_empty()
                    );
                    let after = op.counters();
                    assert_eq!(
                        after.host_to_device_bytes - before.host_to_device_bytes,
                        routes.expert_ids().len() as u64 * 8
                    );
                    let e = routes
                        .expert_ids()
                        .iter()
                        .copied()
                        .collect::<std::collections::BTreeSet<_>>()
                        .len() as u64;
                    assert_eq!(
                        after.host_to_device_copies - before.host_to_device_copies,
                        1
                    );
                    let old_before = old_op.counters();
                    let expected = legacy::legacy_routed(&mut old, &old_x, &routes).unwrap();
                    let old_after = old_op.counters();
                    assert_eq!(
                        old_after.host_to_device_copies - old_before.host_to_device_copies,
                        e + 2
                    );
                    assert_eq!(
                        old_after.host_to_device_bytes - old_before.host_to_device_bytes,
                        routes.expert_ids().len() as u64 * 12
                    );
                    let new_kernels =
                        after.compute_kernel_launches - before.compute_kernel_launches;
                    let old_kernels =
                        old_after.compute_kernel_launches - old_before.compute_kernel_launches;
                    assert_eq!(old_kernels, new_kernels + e - 1);
                    println!(
                        "ROUTE_BATCH_COUNTER lane={lane} M={rows} cached={cached} E={e} R={} h2d_new=1 h2d_old={} kernels_new={new_kernels} kernels_old={old_kernels}",
                        routes.expert_ids().len(),
                        e + 2
                    );
                    assert_eq!(raw_words(&op, &actual), raw_words(&old_op, &expected));
                }
            }
        }
    }
}

#[test]
#[ignore = "sole GPU owner; Unknown after batch gather must retain opaque host/device metadata and gathered root, cache on/off"]
fn unknown_after_gather_retains_both_roots_and_never_replays() {
    for cached in [false, true] {
        let f = Fixture::new(0);
        let linears = f.prepare();
        let op = authorized_ops(0);
        let block = if cached {
            cached_block(&f, &linears)
        } else {
            f.block(&linears)
        };
        let mut c = owner(op.clone(), &f, block, &linears);
        let _runtime = retire(linears);
        std::fs::remove_file(&f.path).unwrap();
        let x = input(&op, 5);
        let routes = RouterRoutes::new(
            5,
            2,
            (0..5).flat_map(|_| [1, 0]).collect(),
            (0..5).flat_map(|_| [0.100123, 0.899877]).collect(),
        )
        .unwrap();
        GATHERS.set(0);
        UNKNOWN_AFTER_GATHER.set(true);
        assert!(
            c.mixed_rank_routed_local(
                0,
                &x,
                &routes,
                HybridRoutedAccumulation::HfEagerBf16ExpertOrder
            )
            .is_err()
        );
        assert_eq!(GATHERS.get(), 1);
        assert!(c.needs_quarantine());
        let image = c.mixed_rank_block.as_ref().unwrap();
        assert_eq!(image.pending_routes.len(), 1);
        assert!(image.pending_routes[0].is_prepared());
        assert_eq!(
            image.pending_routes[0].row_ids().collect::<Vec<_>>(),
            [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        );
        assert_eq!(
            image.pending_routes[0].weight_bits(),
            (0..5)
                .map(|_| 0.899877f32.to_bits())
                .chain((0..5).map(|_| 0.100123f32.to_bits()))
                .collect::<Vec<_>>()
        );
        assert!(
            c.bf16_hold
                .iter()
                .any(|root| root.cu_deviceptr() == GATHER_PTR.get() && root.len() == 10 * 256)
        );
        let live = op.allocator_metrics().live_granted_bytes;
        let launches = op.counters().compute_kernel_launches;
        assert!(
            c.mixed_rank_routed_local(
                0,
                &x,
                &routes,
                HybridRoutedAccumulation::HfEagerBf16ExpertOrder
            )
            .is_err()
        );
        assert!(c.quiesce().is_err());
        assert_eq!(op.counters().compute_kernel_launches, launches);
        drop(c);
        assert_eq!(op.allocator_metrics().live_granted_bytes, live);
    }
}

#[test]
#[ignore = "sole GPU owner; observer retains full expert input views at batch offsets, no per-expert regather"]
fn observer_input_views_use_the_single_gather_root() {
    let f = Fixture::new(0);
    let linears = f.prepare();
    let op = authorized_ops(0);
    let mut c = owner(op.clone(), &f, f.block(&linears), &linears);
    let seen = Rc::new(Cell::new(0usize));
    let calls = Rc::clone(&seen);
    c.set_native_diagnostic_trace(
        CudaDiagnosticLimits::new(1 << 20).unwrap(),
        |_| true,
        move |event| {
            let Some(parameter) = event.parameter else {
                return Ok(());
            };
            if event.descriptor.name == "mixed.linear.input"
                && matches!(
                    parameter.role(),
                    TensorRole::RoutedExpertGate | TensorRole::RoutedExpertUp
                )
            {
                let ParameterResidency::Expert { expert, .. } = parameter.residency() else {
                    panic!("not an expert")
                };
                let CudaDiagnosticBuffer::Bf16(buffer) = event.buffer else {
                    panic!("not BF16")
                };
                assert_eq!(buffer.len(), 5 * 256);
                assert_eq!(
                    buffer.as_device_buffer().cu_deviceptr(),
                    GATHER_PTR.get() + (*expert * 5 * 256 * 2) as u64
                );
                calls.set(calls.get() + 1);
            }
            Ok(())
        },
    );
    let _runtime = retire(linears);
    std::fs::remove_file(&f.path).unwrap();
    let x = input(&op, 5);
    let routes =
        RouterRoutes::new(5, 2, (0..5).flat_map(|_| [1, 0]).collect(), vec![0.5; 10]).unwrap();
    GATHERS.set(0);
    c.mixed_rank_routed_local(
        0,
        &x,
        &routes,
        HybridRoutedAccumulation::HfEagerBf16ExpertOrder,
    )
    .unwrap();
    assert_eq!(GATHERS.get(), 1);
    assert_eq!(seen.get(), 4);
}

#[test]
#[ignore = "sole GPU owner; actual batch completes submissions then loses existing consumer proof, cache on/off roots retained"]
fn late_consumer_unknown_retains_metadata_gather_and_image() {
    for cached in [false, true] {
        let f = Fixture::new(0);
        let linears = f.prepare();
        let op = authorized_ops(0);
        let block = if cached {
            cached_block(&f, &linears)
        } else {
            f.block(&linears)
        };
        let mut c = owner(op.clone(), &f, block, &linears);
        let _runtime = retire(linears);
        std::fs::remove_file(&f.path).unwrap();
        let x = input(&op, 5);
        let routes =
            RouterRoutes::new(5, 2, (0..5).flat_map(|_| [1, 0]).collect(), vec![0.5; 10]).unwrap();
        GATHERS.set(0);
        c.lose_next_consumer_proof = true;
        let before = op.counters();
        assert!(
            c.mixed_rank_routed_local(
                0,
                &x,
                &routes,
                HybridRoutedAccumulation::HfEagerBf16ExpertOrder
            )
            .is_err()
        );
        assert_eq!(GATHERS.get(), 1);
        assert!(c.needs_quarantine());
        assert_eq!(
            op.counters().host_to_device_bytes - before.host_to_device_bytes,
            80
        );
        let image = c.mixed_rank_block.as_ref().unwrap();
        assert_eq!(image.pending_routes.len(), 1);
        assert!(image.pending_routes[0].is_prepared());
        assert!(
            c.bf16_hold
                .iter()
                .any(|b| b.cu_deviceptr() == GATHER_PTR.get() && b.len() == 10 * 256)
        );
        assert!(c.bf16_hold.iter().any(|b| b.len() == 5 * 256));
        if cached {
            let physical = custody(&c).physical_stats();
            assert_eq!(physical.unknown, physical.arena_backing);
        }
        let live = op.allocator_metrics().live_granted_bytes;
        let launches = op.counters().compute_kernel_launches;
        assert!(
            c.mixed_rank_routed_local(
                0,
                &x,
                &routes,
                HybridRoutedAccumulation::HfEagerBf16ExpertOrder
            )
            .is_err()
        );
        assert!(c.quiesce().is_err());
        assert_eq!(op.counters().compute_kernel_launches, launches);
        assert_eq!(GATHERS.get(), 1);
        drop(c);
        assert_eq!(op.allocator_metrics().live_granted_bytes, live);
    }
}

#[test]
#[ignore = "sole GPU owner; public preparation rejects equal-shape foreign model/source before allocation, H2D or kernel submission"]
fn foreign_model_sources_reject_before_payload_cuda_work() {
    let f = Fixture::new(0);
    let foreign = Fixture::new(0);
    let linears = f.prepare();
    let foreign_linears = foreign.prepare();
    let op = authorized_ops(0);
    let mut c = CudaStandardDecoderOperators::new_numeric_fp8_with_precision(
        op.clone(),
        &f.bindings,
        ExpertCachePolicy::KeepAll,
        0,
        NumericFp8Precision::W8A16Bf16F32Accumulate,
    )
    .unwrap();
    let plan = f.block(&linears);
    let parameters = foreign_linears
        .iter()
        .map(|p| p.hybrid_rank().unwrap().clone())
        .collect::<Vec<_>>();
    let before = op.counters();
    let live = op.allocator_metrics().live_granted_bytes;
    assert!(c.prepare_mixed_rank_block(plan, &parameters).is_err());
    let after = op.counters();
    assert_eq!(after.host_to_device_copies, before.host_to_device_copies);
    assert_eq!(after.host_to_device_bytes, before.host_to_device_bytes);
    assert_eq!(
        after.compute_kernel_launches,
        before.compute_kernel_launches
    );
    assert_eq!(
        after.device_allocation_attempts,
        before.device_allocation_attempts
    );
    assert_eq!(op.allocator_metrics().live_granted_bytes, live);
    assert!(c.mixed_rank_block.is_none());
    assert!(c.bf16_gemmex.is_none());
    assert!(!c.needs_quarantine());
}
