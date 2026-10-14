use super::*;
use std::cell::RefCell;

fn precisions() -> [NumericFp8Precision; 3] {
    [
        NumericFp8Precision::Bf16RneF32Accumulate,
        NumericFp8Precision::F32Tf32x3,
        NumericFp8Precision::W8A16Bf16F32Accumulate,
    ]
}

fn prepared(provider: &mut Provider<'_>) -> Arc<PreparedSwiGlu> {
    let ExpertAvailability::Ready(expert) = provider.expert(0, 0).unwrap() else {
        panic!("ready fixture")
    };
    expert
}

fn warm(owner: &mut CudaStandardDecoderOperators, expert: &PreparedSwiGlu, input: &Rows) {
    ready(owner.dense_swiglu(expert, input, None).unwrap());
}

#[test]
#[ignore = "requires CUDA; real proof hook, allocation counters and final D2H consumer"]
fn warm_scope_proofs_and_same_stream_order_match_standalone_reference() {
    for precision in precisions() {
        let fixture = Fixture::new();
        let mut provider = fixture.provider(true);
        let expert = prepared(&mut provider);
        let mut owner = fixture.owner(precision, 32768);
        let input = owner.bind_rows(input(3)).unwrap();
        warm(&mut owner, &expert, &input);
        let allocations = owner.numeric_workspace_stats().unwrap().allocations;
        let baseline = owner.consumer_proofs;
        let expected = owner
            .synchronous(|this| {
                this.expert_operation(&expert, 3, |this| {
                    let projections = [expert.gate(), expert.up(), expert.down()];
                    let output = this.swiglu_compute(&input, None, None, |this, i, input| {
                        this.linear_rows(projections[i], input, None)
                    })?;
                    this.boundary.download(&this.ops, this.f32(&output)?)
                })
            })
            .unwrap();
        let standalone_proofs = owner.consumer_proofs - baseline;
        assert_eq!(standalone_proofs, 8); // six projections, lease, synchronous

        let stages = Rc::new(RefCell::new(Vec::new()));
        let observed = Rc::clone(&stages);
        owner.set_diagnostic_trace(move |event| {
            if event.name == "linear.output" {
                observed.borrow_mut().push((
                    event.parameter.unwrap().role().clone(),
                    event.operators.counters(),
                ));
            }
            Ok(())
        });
        let baseline = owner.consumer_proofs;
        let actual = owner
            .synchronous(|this| {
                this.swiglu_consume(&expert, &input, None, |this, output| {
                    assert_eq!(this.consumer_proofs, baseline, "no projection proof");
                    assert_eq!(this.expert_artifacts.len(), 3);
                    assert_eq!(this.expert_hold.len(), 5);
                    assert!(
                        this.expert_artifacts
                            .iter()
                            .all(|w| Rc::strong_count(w) >= 2)
                    );
                    assert!(this.bindings.expert_admission.is_some());
                    let values = this.boundary.download(&this.ops, this.f32(&output)?)?;
                    assert_eq!(this.consumer_proofs, baseline, "D2H precedes lease proof");
                    Ok(values)
                })
            })
            .unwrap();
        let scope_proofs = owner.consumer_proofs - baseline;
        assert_eq!(scope_proofs, 2); // final-consumer lease, outer synchronous
        eprintln!(
            "numeric scope {precision:?}: proof attempts {standalone_proofs} -> {scope_proofs}"
        );
        assert_eq!(
            owner.numeric_workspace_stats().unwrap().allocations,
            allocations
        );
        assert!(owner.expert_artifacts.is_empty());
        assert!(owner.expert_hold.is_empty());
        let stages = stages.borrow();
        assert_eq!(
            stages
                .iter()
                .map(|(role, _)| role.clone())
                .collect::<Vec<_>>(),
            vec![
                TensorRole::RoutedExpertGate,
                TensorRole::RoutedExpertUp,
                TensorRole::RoutedExpertDown
            ]
        );
        for pair in stages.windows(2) {
            assert_eq!(
                pair[0].1.device_allocation_attempts,
                pair[1].1.device_allocation_attempts
            );
            assert_eq!(pair[0].1.stream_wide_syncs, pair[1].1.stream_wide_syncs);
            assert_eq!(
                pair[0].1.device_to_host_copies,
                pair[1].1.device_to_host_copies
            );
            assert!(pair[0].1.compute_kernel_launches < pair[1].1.compute_kernel_launches);
        }
        // Use the existing launch order and independent synchronous numeric path
        // as the numerical oracle; no precision/math changes are tolerated.
        close(&actual, &expected);
    }
}

#[test]
#[ignore = "requires CUDA; each projection error/panic/Unknown retains owner custody"]
fn each_projection_failure_and_unknown_is_sticky() {
    for precision in precisions() {
        for target in [
            TensorRole::RoutedExpertGate,
            TensorRole::RoutedExpertUp,
            TensorRole::RoutedExpertDown,
        ] {
            for mode in 0..3 {
                let fixture = Fixture::new();
                let mut provider = fixture.provider(true);
                let expert = prepared(&mut provider);
                let mut owner = fixture.owner(precision, 32768);
                let input = owner.bind_rows(input(1)).unwrap();
                warm(&mut owner, &expert, &input);
                let target = target.clone();
                owner.set_diagnostic_trace(move |event| {
                    if event.name == "linear.output" && event.parameter.unwrap().role() == &target {
                        if mode == 1 {
                            panic!("projection panic");
                        }
                        return Err(cuda_error("projection failure"));
                    }
                    Ok(())
                });
                owner.lose_next_consumer_proof = mode == 2;
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    owner.dense_swiglu(&expert, &input, None)
                }));
                assert!(!matches!(result, Ok(Ok(_))));
                if mode == 0 {
                    // A proven operation error may retire operands, not Unknown.
                    assert!(!owner.needs_quarantine());
                    assert!(owner.expert_artifacts.is_empty());
                    assert!(owner.expert_hold.is_empty());
                    continue;
                }
                assert!(owner.needs_quarantine());
                assert_eq!(owner.expert_artifacts.len(), 3);
                assert_eq!(owner.expert_hold.len(), 5);
                let frozen = owner.expert_cache_stats().unwrap();
                let workspace = owner.numeric_workspace_stats().unwrap();
                assert_eq!(frozen.unknown_quarantine_bytes, frozen.charged_bytes());
                assert_eq!(frozen.resident_bytes, 294);
                let held = owner.ops.allocator_metrics().live_requested_bytes;
                owner.ops.sync_stream().unwrap();
                owner.ops.sync_upload_stream().unwrap();
                assert!(owner.quiesce().is_err());
                assert!(owner.finish_cache_operation().is_err());
                assert_eq!(owner.expert_cache_stats().unwrap(), frozen);
                assert_eq!(owner.numeric_workspace_stats().unwrap(), workspace);
                let ops = Rc::clone(&owner.ops);
                drop(input);
                drop(owner);
                ops.trim_device_allocator().unwrap();
                assert!(ops.allocator_metrics().live_requested_bytes >= held);
            }
        }
    }
}

#[test]
#[ignore = "requires CUDA; real allocation failure before the first projection"]
fn scope_allocation_failure_never_submits_and_unknown_keeps_credit() {
    for precision in precisions() {
        for (unknown, allocation) in [false, true]
            .into_iter()
            .flat_map(|v| (0..4).map(move |i| (v, i)))
        {
            let fixture = Fixture::new();
            let mut provider = fixture.provider(true);
            let expert = prepared(&mut provider);
            let mut owner = fixture.owner(precision, 32768);
            let input = owner.bind_rows(input(1)).unwrap();
            warm(&mut owner, &expert, &input);
            let submissions = owner.numeric_workspace_stats().unwrap().submissions;
            owner.numeric_allocation_failure = Some(allocation);
            owner.lose_next_consumer_proof = unknown;
            assert!(owner.dense_swiglu(&expert, &input, None).is_err());
            assert_eq!(
                owner.numeric_workspace_stats().unwrap().submissions,
                submissions
            );
            assert_eq!(owner.needs_quarantine(), unknown);
            if unknown {
                assert_eq!(owner.expert_artifacts.len(), 3);
                assert_eq!(owner.expert_hold.len(), allocation + 1); // input and successful allocations
                let frozen = owner.expert_cache_stats().unwrap();
                assert_eq!(frozen.unknown_quarantine_bytes, frozen.charged_bytes());
                assert!(owner.quiesce().is_err());
                assert_eq!(owner.expert_cache_stats().unwrap(), frozen);
            }
        }
    }
}

#[test]
#[ignore = "requires CUDA; scatter inside admission and actual proof-attempt counter"]
fn routed_scatter_is_final_consumer_not_a_second_bucket_proof() {
    for precision in precisions() {
        let fixture = Fixture::new();
        let mut provider = fixture.provider(true);
        let expert = prepared(&mut provider);
        let mut owner = fixture.owner(precision, 32768);
        let input = owner.bind_rows(input(1)).unwrap();
        warm(&mut owner, &expert, &input);
        let route = RouterRoutes::new(1, 1, vec![0], vec![1.0]).unwrap();
        let calls = provider.calls;
        let start = owner.consumer_proofs;
        let output = ready(
            owner
                .routed_swiglu(0, &input, &route, &mut provider, None)
                .unwrap(),
        );
        assert_eq!(owner.consumer_proofs - start, 2); // scatter lease + final combine
        assert_eq!(
            provider.calls, calls,
            "known residency does not rematerialize"
        );
        let actual = owner.download_rows(output).unwrap();
        let output = ready(owner.dense_swiglu(&expert, &input, None).unwrap());
        let expected = owner.download_rows(output).unwrap();
        close(actual.values(), expected.values());
    }
}

#[test]
#[ignore = "requires CUDA; pending workspace is fenced before growth, Unknown never replaces it"]
fn growth_after_queued_work_requires_a_real_proof() {
    for unknown in [false, true] {
        let fixture = Fixture::new();
        let mut provider = fixture.provider(true);
        let expert = prepared(&mut provider);
        let mut owner = fixture.owner(NumericFp8Precision::Bf16RneF32Accumulate, 32768);
        let small = owner.bind_rows(input(1)).unwrap();
        let large = owner.bind_rows(input(16)).unwrap();
        warm(&mut owner, &expert, &small);
        let parameters = fixture.parameters[0].clone();
        let before = owner.numeric_workspace_stats().unwrap();
        let proofs = owner.consumer_proofs;
        let result = owner.synchronous(|this| {
            this.expert_operation(&expert, 16, |this| {
                let _pending = this
                    .resident_numeric_swiglu(&parameters, None, None, &small, None)?
                    .unwrap();
                assert_eq!(this.consumer_proofs, proofs);
                this.lose_next_consumer_proof = unknown;
                let grown = this.resident_numeric_swiglu(&parameters, None, None, &large, None);
                assert_eq!(
                    this.consumer_proofs,
                    proofs + 1,
                    "growth needs its own proof"
                );
                grown.map(|_| ())
            })
        });
        assert_eq!(result.is_err(), unknown);
        let after = owner.numeric_workspace_stats().unwrap();
        if unknown {
            assert_eq!(after.allocated_bytes, before.allocated_bytes);
            assert_eq!(after.allocations, before.allocations);
            assert_eq!(after.submissions, before.submissions + 3);
            assert_eq!(owner.expert_artifacts.len(), 6);
            assert_eq!(owner.expert_hold.len(), 6); // first scope + second input
            let frozen = owner.expert_cache_stats().unwrap();
            owner.ops.sync_stream().unwrap();
            assert!(owner.quiesce().is_err());
            assert_eq!(owner.expert_cache_stats().unwrap(), frozen);
            assert_eq!(owner.numeric_workspace_stats().unwrap(), after);
        } else {
            assert!(after.allocated_bytes > before.allocated_bytes);
            assert_eq!(after.allocations, before.allocations + 1);
            assert_eq!(after.submissions, before.submissions + 6);
        }
    }
}

#[test]
#[ignore = "requires CUDA; final consumer failure/panic/lost proof retains all roots"]
fn final_consumer_unknown_retains_artifacts_operands_and_credit() {
    for precision in precisions() {
        for mode in 0..3 {
            let fixture = Fixture::new();
            let mut provider = fixture.provider(true);
            let expert = prepared(&mut provider);
            let mut owner = fixture.owner(precision, 32768);
            let input = owner.bind_rows(input(1)).unwrap();
            warm(&mut owner, &expert, &input);
            let ids = owner.ops.upload_i32_buffer(&[0]).unwrap();
            let mut table = owner.ops.zero_f32_buffer(8).unwrap();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                owner.synchronous(|this| {
                    this.hold_route_buffer(&table)?;
                    this.route_ids.push(ids.as_device_buffer().slice(0, 1)?);
                    this.swiglu_consume(&expert, &input, None, |this, output| -> Result<()> {
                        this.ops.scatter_add_f32_rows(
                            this.f32(&output)?,
                            &ids,
                            &mut table,
                            1,
                            8,
                        )?;
                        assert_eq!(this.expert_artifacts.len(), 3);
                        assert_eq!(this.expert_hold.len(), 5);
                        if mode == 1 {
                            panic!("final consumer panic");
                        }
                        this.lose_next_consumer_proof = true;
                        if mode == 0 {
                            Err(cuda_error("final consumer failed"))
                        } else {
                            Ok(())
                        }
                    })
                })
            }));
            assert!(!matches!(result, Ok(Ok(_))));
            assert!(owner.needs_quarantine());
            assert_eq!(owner.expert_artifacts.len(), 3);
            assert_eq!(owner.expert_hold.len(), 5);
            assert_eq!(owner.route_hold.len(), 1);
            assert_eq!(owner.route_ids.len(), 1);
            let frozen = owner.expert_cache_stats().unwrap();
            let bytes = owner.ops.allocator_metrics().live_requested_bytes;
            let ops = Rc::clone(&owner.ops);
            ops.sync_stream().unwrap();
            ops.sync_upload_stream().unwrap();
            assert!(owner.quiesce().is_err());
            assert_eq!(owner.expert_cache_stats().unwrap(), frozen);
            drop(input);
            drop(ids);
            drop(table);
            drop(owner);
            ops.trim_device_allocator().unwrap();
            assert!(ops.allocator_metrics().live_requested_bytes >= bytes);
        }
    }
}

mod boundary_record {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/unit/transformer/boundary_record.rs"
    ));
}

#[test]
#[ignore = "requires CUDA; production W8A16 zero-workspace expert path at exact cache cap"]
fn w8a16_zero_workspace_expert_scopes_keep_compressed_weights_and_operation_credit() {
    let fixture = Fixture::new();
    let mut provider = fixture.provider(true);
    let expert = prepared(&mut provider);
    let mut owner = CudaStandardDecoderOperators::new_numeric_fp8_with_precision(
        Rc::new(CudaOperators::new_on_device(0).unwrap()),
        &fixture
            .parameters
            .iter()
            .flatten()
            .cloned()
            .collect::<Vec<_>>(),
        ExpertCachePolicy::Bounded(ExpertCacheLimits {
            max_experts: 1,
            max_bytes: 294 + 3 * (2 * 8 + 3 * 12) * 4,
        }),
        0,
        NumericFp8Precision::W8A16Bf16F32Accumulate,
    )
    .unwrap();
    assert_eq!(
        owner.backend_name(),
        NumericFp8Precision::W8A16Bf16F32Accumulate.standard_backend_name()
    );
    assert_eq!(owner.precision(), ExecutionPrecisionPolicy::f32());
    let metadata = provider.expert_metadata(0, 0).unwrap().unwrap();
    let prepared_plan = owner.expert_plan(&expert, 3).unwrap();
    let metadata_plan = owner.metadata_plan(&metadata, 0, 0, 3, 8).unwrap();
    assert_eq!(prepared_plan, metadata_plan);
    assert_eq!(
        prepared_plan.0.numeric_precision,
        Some(NumericFp8Precision::W8A16Bf16F32Accumulate)
    );
    assert_eq!(prepared_plan.1, 294);
    owner
        .preflight_bucket_numeric(metadata.parameters(), 3)
        .unwrap();
    for rows in [1, 3, 1] {
        let host = input(rows);
        let input = owner.bind_rows(host).unwrap();
        let actual = ready(owner.dense_swiglu(&expert, &input, None).unwrap());
        let actual = owner.download_rows(actual).unwrap();
        // This fixture has all weights exactly 1/16. Independent scalar oracle
        // follows BF16 operand boundaries, not the fused kernel reduction order.
        let Rows::Host(host) = super::input(rows) else {
            unreachable!()
        };
        let expected = host
            .values()
            .as_chunks::<8>()
            .0
            .iter()
            .flat_map(|row| {
                let gate: f32 = row
                    .iter()
                    .map(|&v| half::bf16::from_f32(v).to_f32() / 16.0)
                    .sum();
                let product = half::bf16::from_f32(gate / (1.0 + (-gate).exp()) * gate).to_f32();
                [product * (12.0 / 16.0); 8]
            })
            .collect::<Vec<_>>();
        close(actual.values(), &expected);
        let workspace = owner.numeric_workspace_stats().unwrap();
        assert_eq!(workspace.reserved_bytes, 0);
        assert_eq!(workspace.allocated_bytes, 0);
        let cache = owner.expert_cache_stats().unwrap();
        assert_eq!(cache.resident_bytes, 294);
        assert!(cache.peak_bytes <= 294 + 3 * (2 * 8 + 3 * 12) * 4);
    }
    assert_eq!(owner.numeric_workspace_stats().unwrap().submissions, 9);
    assert_eq!(owner.expert_cache_stats().unwrap().uploads, 1);
    assert_eq!(owner.expert_cache_stats().unwrap().hits, 2);
}
