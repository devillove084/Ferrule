//! Real model paths only; GPU execution requires separate sole-owner permission.
use super::*;
use crate::transformer::standard::cuda::diagnostic::{CudaDiagnosticBuffer, CudaDiagnosticLimits};
use crate::transformer::standard::cuda::hybrid_rank::mixed::routed_output::RoutedOutputLayout;

thread_local! {
    static FUSED_SUBMISSIONS: Cell<usize> = const { Cell::new(0) };
    static UNKNOWN_AFTER_FUSED: Cell<bool> = const { Cell::new(false) };
    static DOWN_SUBMISSIONS: Cell<usize> = const { Cell::new(0) };
    static DOWN_FAULT: Cell<u8> = const { Cell::new(0) };
}

pub(crate) fn after_fused_submission(result: Result<()>) -> Result<()> {
    result?;
    FUSED_SUBMISSIONS.set(FUSED_SUBMISSIONS.get() + 1);
    if UNKNOWN_AFTER_FUSED.replace(false) {
        return Err(crate::decoder::CompletionUnknown::wrap(cuda_error(
            "injected Unknown after real routed pair-SwiGLU submission",
        )));
    }
    Ok(())
}

pub(crate) fn after_down_submission() -> Result<()> {
    DOWN_SUBMISSIONS.set(DOWN_SUBMISSIONS.get() + 1);
    match DOWN_FAULT.replace(0) {
        1 => Err(cuda_error("injected error after real routed down")),
        2 => Err(crate::decoder::CompletionUnknown::wrap(cuda_error(
            "injected Unknown after real routed down",
        ))),
        _ => Ok(()),
    }
}

fn observe(owner: &CudaStandardDecoderOperators) -> Rc<RefCell<Vec<&'static str>>> {
    let events = Rc::new(RefCell::new(Vec::new()));
    let seen = events.clone();
    owner.set_native_diagnostic_trace(
        CudaDiagnosticLimits::new(1 << 20).unwrap(),
        |_| true,
        move |event| {
            if matches!(event.descriptor.name, "mixed.pair.gate" | "mixed.pair.up") {
                let CudaDiagnosticBuffer::Bf16(buffer) = event.buffer else {
                    panic!("pair observations must borrow actual BF16 tensors")
                };
                assert_eq!(buffer.len(), event.descriptor.layout.elements());
                seen.borrow_mut().push(event.descriptor.name);
            }
            Ok(())
        },
    );
    events
}

#[test]
#[ignore = "sole GPU owner; actual routed no-observer fusion versus native-observer split, M1/M2/M5 cold/hot"]
fn routed_fusion_noobs_and_obs_exact_with_launch_and_allocation_counts() {
    for lane in 0..2 {
        for rows in [1, 2, 5] {
            for cached in [false, true] {
                let f = Fixture::new(lane);
                let linears = f.prepare();
                let op = authorized_ops(lane);
                let observed_op = authorized_ops(lane);
                let block = if cached {
                    cached_block(&f, &linears)
                } else {
                    f.block(&linears)
                };
                let observed_block = if cached {
                    cached_block(&f, &linears)
                } else {
                    f.block(&linears)
                };
                let mut fused = owner(op.clone(), &f, block, &linears);
                let mut observed = owner(observed_op.clone(), &f, observed_block, &linears);
                let _runtime = retire(linears);
                std::fs::remove_file(&f.path).unwrap();
                let x = input(&op, rows);
                let ox = input(&observed_op, rows);
                // Original IDs are intentionally not in router slot order.
                let routes = RouterRoutes::new(
                    rows,
                    2,
                    (0..rows).flat_map(|_| [1, 0]).collect(),
                    (0..rows).flat_map(|_| [0.100123, 0.899877]).collect(),
                )
                .unwrap();
                let events = observe(&observed);
                for _ in 0..2 {
                    let before = op.counters();
                    let obefore = observed_op.counters();
                    FUSED_SUBMISSIONS.set(0);
                    let actual = fused
                        .mixed_rank_routed_local(
                            0,
                            &x,
                            &routes,
                            HybridRoutedAccumulation::HfEagerBf16ExpertOrder,
                        )
                        .unwrap();
                    let after = op.counters();
                    assert_eq!(FUSED_SUBMISSIONS.get(), 2);
                    FUSED_SUBMISSIONS.set(0);
                    events.borrow_mut().clear();
                    let expected = observed
                        .mixed_rank_routed_local(
                            0,
                            &ox,
                            &routes,
                            HybridRoutedAccumulation::HfEagerBf16ExpertOrder,
                        )
                        .unwrap();
                    let oafter = observed_op.counters();
                    assert_eq!(FUSED_SUBMISSIONS.get(), 0);
                    assert_eq!(
                        &*events.borrow(),
                        &[
                            "mixed.pair.gate",
                            "mixed.pair.up",
                            "mixed.pair.gate",
                            "mixed.pair.up"
                        ]
                    );
                    // Same packing in both paths: only split+SwiGLU versus fused
                    // tail differs. No native diagnostic expansion or readback.
                    assert_eq!(
                        oafter.compute_kernel_launches - obefore.compute_kernel_launches,
                        after.compute_kernel_launches - before.compute_kernel_launches + 2
                    );
                    assert_eq!(
                        oafter.device_allocations - obefore.device_allocations,
                        after.device_allocations - before.device_allocations + 4
                    );
                    assert_eq!(
                        after.host_to_device_bytes - before.host_to_device_bytes,
                        oafter.host_to_device_bytes - obefore.host_to_device_bytes
                    );
                    assert_eq!(after.device_to_host_bytes, before.device_to_host_bytes);
                    assert_eq!(oafter.device_to_host_bytes, obefore.device_to_host_bytes);
                    assert_eq!(
                        after.stream_wide_syncs - before.stream_wide_syncs,
                        oafter.stream_wide_syncs - obefore.stream_wide_syncs
                    );
                    assert_eq!(
                        output_words(&op, &actual),
                        output_words(&observed_op, &expected)
                    );
                    if cached {
                        for owner in [&fused, &observed] {
                            let snapshot = owner.mixed_decoded_cache_snapshot().unwrap();
                            assert_eq!(snapshot.phase, MixedDecodedCachePhase::Idle);
                            assert_eq!(snapshot.pending_lease_count, 0);
                            assert_eq!(snapshot.borrowed_key_count, 0);
                            assert!(!snapshot.quarantined);
                        }
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "sole GPU owner; Unknown after actual fused tail must quarantine and never replay"]
fn routed_fusion_unknown_retains_leases_and_never_replays() {
    let f = Fixture::new(0);
    let linears = f.prepare();
    let op = authorized_ops(0);
    let mut c = owner(op.clone(), &f, cached_block(&f, &linears), &linears);
    let _runtime = retire(linears);
    std::fs::remove_file(&f.path).unwrap();
    let x = input(&op, 1);
    let routes = RouterRoutes::new(1, 2, vec![1, 0], vec![0.25, 0.75]).unwrap();
    FUSED_SUBMISSIONS.set(0);
    UNKNOWN_AFTER_FUSED.set(true);
    assert!(
        c.mixed_rank_routed_local(
            0,
            &x,
            &routes,
            HybridRoutedAccumulation::HfEagerBf16ExpertOrder
        )
        .is_err()
    );
    assert_eq!(FUSED_SUBMISSIONS.get(), 1);
    assert!(c.needs_quarantine());
    let frozen = custody(&c).physical_stats();
    assert_eq!(frozen.unknown, frozen.arena_backing);
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
    assert_eq!(FUSED_SUBMISSIONS.get(), 1);
    assert_eq!(custody(&c).physical_stats(), frozen);
    drop(c);
    assert_eq!(op.allocator_metrics().live_granted_bytes, live);
}

#[test]
#[ignore = "sole GPU owner; actual old packing kernels versus BF16 views, nonfinite words and E+1 launches"]
fn routed_pack_words_nonfinite_contract_and_removed_launches() {
    let op = authorized_ops(0);
    let words = (0..=u16::MAX).collect::<Vec<_>>();
    let direct = op.upload_bf16_words(&words).unwrap();
    let layout = RoutedOutputLayout::new(256, 256).unwrap();
    let mut old_f32 = op.zero_f32_buffer(layout.elements()).unwrap();
    let mut retained = Vec::new();
    let before = op.counters();
    for start in [0, 128] {
        let range = layout.bucket(start, 128).unwrap();
        let view = direct.slice(range.start, range.len()).unwrap();
        let expanded = op.bf16_to_f32(&view).unwrap();
        op.copy_f32_range(&expanded, 0, &mut old_f32, range.start, range.len())
            .unwrap();
        retained.push(expanded);
    }
    let old_bf16 = op
        .f32_rows_to_bf16_rne(&old_f32, F32ToBf16RowsLayout::packed(1, 256, 256).unwrap())
        .unwrap();
    assert_eq!(
        op.counters().compute_kernel_launches - before.compute_kernel_launches,
        3
    );
    let old = old_bf16
        .as_device_buffer()
        .to_host_vec(&op.stream_clone())
        .unwrap();
    let new = direct
        .as_device_buffer()
        .to_host_vec(&op.stream_clone())
        .unwrap();
    assert_eq!(new, words);
    for (&source, &repacked) in new.iter().zip(&old) {
        if source & 0x7fff > 0x7f80 {
            assert_eq!(repacked, source | 0x0040, "old pack quiets NaNs");
        } else {
            assert_eq!(repacked, source, "finite, signed zero and infinity words");
        }
    }
    drop(retained);
}

#[test]
#[ignore = "sole GPU owner; error/Unknown after actual down must not replay pair or remaining buckets"]
fn routed_fusion_after_down_failure_does_not_replay() {
    for fault in [1, 2] {
        let f = Fixture::new(0);
        let linears = f.prepare();
        let op = authorized_ops(0);
        let mut c = owner(op.clone(), &f, cached_block(&f, &linears), &linears);
        let _runtime = retire(linears);
        std::fs::remove_file(&f.path).unwrap();
        let x = input(&op, 1);
        let routes = RouterRoutes::new(1, 2, vec![1, 0], vec![0.25, 0.75]).unwrap();
        FUSED_SUBMISSIONS.set(0);
        DOWN_SUBMISSIONS.set(0);
        DOWN_FAULT.set(fault);
        assert!(
            c.mixed_rank_routed_local(
                0,
                &x,
                &routes,
                HybridRoutedAccumulation::HfEagerBf16ExpertOrder
            )
            .is_err()
        );
        assert_eq!(FUSED_SUBMISSIONS.get(), 1);
        assert_eq!(DOWN_SUBMISSIONS.get(), 1);
        assert_eq!(c.needs_quarantine(), fault == 2);
        if fault == 2 {
            let frozen = custody(&c).physical_stats();
            assert_eq!(frozen.unknown, frozen.arena_backing);
            let live = op.allocator_metrics().live_granted_bytes;
            let launches = op.counters().compute_kernel_launches;
            assert!(execute(&mut c, 0, &x).is_err());
            assert_eq!(op.counters().compute_kernel_launches, launches);
            drop(c);
            assert_eq!(op.allocator_metrics().live_granted_bytes, live);
        } else {
            let snapshot = c.mixed_decoded_cache_snapshot().unwrap();
            assert_eq!(snapshot.phase, MixedDecodedCachePhase::Idle);
            assert_eq!(snapshot.pending_lease_count, 0);
            assert_eq!(snapshot.borrowed_key_count, 0);
        }
    }
}
