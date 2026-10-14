use super::*;
use crate::transformer::standard::cuda::BarrierReason;
use crate::transformer::{RotaryEmbedding, RotaryPairing, RotaryRegion, RotaryScaling};
use ferrule_common::Error;

// Same 208-byte SwiGLU operation envelope already admitted by this fixture:
// one input, three intermediate rows, one output. No additional device budget.
const PINS: usize = (8 + 3 * 12 + 8) * 4;

fn norm(fixture: &mut Fixture) -> crate::transformer::PreparedNorm {
    let module = ModulePath::new("attention.norm").unwrap();
    let role = TensorRole::AttentionNorm;
    let mut schema = StateDictSchema::builder();
    schema
        .register_with_role(
            ParameterSpec::new(
                ParameterId::new(99),
                module.clone(),
                ParameterDType::F32,
                vec![8],
                ParameterResidency::layer(0),
            )
            .unwrap(),
            role.clone(),
        )
        .unwrap();
    let mut mapper = ExactNameMapper::new();
    mapper
        .insert("attention.norm", NameMapping::weight(module))
        .unwrap();
    let bound = StateDictBinder::new(&schema.build().unwrap(), &mapper)
        .bind_slices(vec![CheckpointTensorSlice {
            name: "attention.norm".into(),
            path: fixture.path.clone(),
            role,
            offset: 0,
            bytes: 32,
            shape: vec![8],
            dtype: CheckpointDType::F32,
        }])
        .unwrap();
    let parameter = StateDictMaterializer::new(1024)
        .unwrap()
        .parameter(bound.get_by_id(ParameterId::new(99)).unwrap())
        .unwrap();
    fixture.parameters.extend_from_slice(bound.parameters());
    crate::transformer::PreparedNorm::new(parameter, 1e-6).unwrap()
}

fn rope() -> (RotaryEmbedding, crate::transformer::PreparedRope) {
    let descriptor = RotaryEmbedding::new(
        8,
        10_000.0,
        RotaryPairing::SplitHalf,
        RotaryRegion::Prefix { dimensions: 4 },
        RotaryScaling::None,
    )
    .unwrap();
    let table = crate::transformer::standard::prepare_rope(&descriptor, 8).unwrap();
    (descriptor, table)
}

#[test]
#[ignore = "requires CUDA; compares actual consumer proofs and unchanged F32 arithmetic"]
fn root_submission_projection_residual_rope_chain_proves_once() {
    for precision in precisions() {
        let mut fixture = Fixture::new();
        let norm = norm(&mut fixture);
        let mut owner = fixture.owner(precision, RESERVATION + PINS);
        let input = input(&mut owner, 1);
        warm(&mut owner, &fixture.expert, &input);
        let (descriptor, table) = rope();
        owner
            .synchronous(|this| this.root_rope_binding(&table))
            .unwrap();
        let before = owner.consumer_proofs;
        let normalized = ready(owner.rms_norm(&norm, &input, 1, None).unwrap());
        let gate = owner
            .synchronous(|this| this.linear_rows(fixture.expert.gate(), &normalized, None))
            .unwrap();
        let down = owner
            .synchronous(|this| this.linear_rows(fixture.expert.down(), &gate, None))
            .unwrap();
        let expected = ready(owner.residual(input, &down).unwrap());
        let expected = ready(owner.rope(&descriptor, &table, expected, 1, &[3]).unwrap());
        assert_eq!(
            owner.consumer_proofs - before,
            9,
            "six numeric projection proofs plus norm, residual and rope"
        );
        let expected = owner.download_rows(expected).unwrap();
        drop(normalized);
        drop(gate);
        drop(down);
        let input = super::input(&mut owner, 1);
        let before = owner.consumer_proofs;
        let allocations = owner.numeric_workspace_stats().unwrap().allocations;
        let actual = owner
            .with_root_submission(PINS, |this| {
                let normalized = ready(this.rms_norm(&norm, &input, 1, None)?);
                let gate = ready(this.linear(fixture.expert.gate(), &normalized, None)?);
                let down = ready(this.linear(fixture.expert.down(), &gate, None)?);
                let output = ready(this.residual(input, &down)?);
                let output = ready(this.rope(&descriptor, &table, output, 1, &[3])?);
                assert_eq!(
                    this.consumer_proofs, before,
                    "same-stream operators must not prove again"
                );
                let segment = this.root_submission.as_ref().unwrap();
                assert_eq!(segment.root_count(), 4);
                assert_eq!(segment.bytes(), (8 + 8 + 12 + 8) * 4);
                assert_eq!(
                    this.expert_cache_stats().unwrap().scratch_bytes,
                    RESERVATION + PINS
                );
                assert!(
                    this.expert_hold.is_empty(),
                    "no unbounded duplicate pin vector"
                );
                Ok(output)
            })
            .unwrap();
        assert_eq!(owner.consumer_proofs - before, 1);
        assert!(owner.root_submission.is_none());
        assert_eq!(
            owner.numeric_workspace_stats().unwrap().allocations,
            allocations
        );
        let actual = owner.download_rows(actual).unwrap();
        assert_eq!(actual.values(), expected.values());
        assert_eq!(
            owner.expert_cache_stats().unwrap().scratch_bytes,
            RESERVATION
        );
    }
}

#[test]
#[ignore = "requires CUDA; D2H remains an explicit producer and host boundary"]
fn root_submission_host_read_flushes_before_transfer() {
    let fixture = Fixture::new();
    let mut owner = fixture.owner(precisions()[0], RESERVATION + PINS);
    let input = input(&mut owner, 1);
    warm(&mut owner, &fixture.expert, &input);
    let before = owner.consumer_proofs;
    owner
        .with_root_submission(PINS, |this| {
            let rows = ready(this.linear(fixture.expert.gate(), &input, None)?);
            assert_eq!(this.consumer_proofs, before);
            let host = this.download_rows(rows)?;
            assert_eq!(host.shape().width(), 12);
            assert_eq!(
                this.consumer_proofs - before,
                2,
                "producer flush plus ordinary host-call proof"
            );
            let segment = this.root_submission.as_ref().unwrap();
            assert_eq!(segment.last_barrier, Some(BarrierReason::HostRead));
            assert_eq!(segment.root_count(), 0);
            Ok(())
        })
        .unwrap();
    assert_eq!(
        owner.consumer_proofs - before,
        2,
        "an empty exit is not a new consumer"
    );
}

#[test]
#[ignore = "requires CUDA; fixed budget flushes prior roots rather than extending the pin queue"]
fn root_submission_exact_cap_and_workspace_growth_keep_barriers() {
    for growth in [false, true] {
        let fixture = Fixture::new();
        let mut owner = fixture.owner(precisions()[0], RESERVATION + PINS);
        let input = input(&mut owner, 1);
        warm(&mut owner, &fixture.expert, &input);
        if growth {
            owner.numeric.as_mut().unwrap().workspace = None;
            owner.numeric.as_mut().unwrap().stats.allocated_bytes = 0;
        }
        let before = owner.consumer_proofs;
        owner
            .with_root_submission((8 + 12) * 4, |this| {
                let first = ready(this.linear(fixture.expert.gate(), &input, None)?);
                assert_eq!(this.consumer_proofs - before, usize::from(growth));
                let _second = ready(this.linear(fixture.expert.down(), &first, None)?);
                assert_eq!(this.root_submission.as_ref().unwrap().root_count(), 2);
                assert_eq!(this.root_submission.as_ref().unwrap().bytes(), 80);
                assert_eq!(this.consumer_proofs - before, 1 + usize::from(growth));
                Ok(())
            })
            .unwrap();
        assert_eq!(owner.consumer_proofs - before, 2 + usize::from(growth));
        assert!(owner.expert_cache_stats().unwrap().peak_bytes <= RESERVATION + PINS);
    }
}

#[test]
#[ignore = "requires CUDA; errors, panic, allocation failure and lost proof retain all roots/credits"]
fn root_submission_failure_custody_is_sticky_after_successful_sync() {
    for failure in 0..4 {
        let fixture = Fixture::new();
        let mut owner = fixture.owner(precisions()[0], RESERVATION + PINS);
        let input = input(&mut owner, 1);
        warm(&mut owner, &fixture.expert, &input);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            owner.with_root_submission(PINS, |this| {
                let first = ready(this.linear(fixture.expert.gate(), &input, None)?);
                match failure {
                    0 => return Err(cuda_error("root operation error")),
                    1 => panic!("root operation panic"),
                    2 => this.lose_next_consumer_proof = true,
                    _ => {
                        this.ops.failpoints().arm_allocation();
                        let _ = this.linear(fixture.expert.down(), &first, None)?;
                    }
                }
                Ok(first)
            })
        }));
        assert!(!matches!(result, Ok(Ok(_))));
        assert!(owner.needs_quarantine());
        assert!(owner.root_submission.as_ref().unwrap().is_unknown());
        assert_eq!(owner.root_submission.as_ref().unwrap().root_count(), 2);
        let credit = owner.expert_cache_stats().unwrap();
        assert_eq!(credit.scratch_bytes, RESERVATION + PINS);
        assert_eq!(credit.unknown_quarantine_bytes, credit.charged_bytes());
        owner.ops.sync_stream().unwrap();
        owner.ops.sync_upload_stream().unwrap();
        assert!(owner.quiesce().is_err());
        assert!(owner.root_barrier(BarrierReason::SegmentExit).is_err());
        assert_eq!(owner.expert_cache_stats().unwrap(), credit);
        let ops = Rc::clone(&owner.ops);
        let live = ops.allocator_metrics().live_requested_bytes;
        drop(input);
        drop(owner);
        ops.trim_device_allocator().unwrap();
        assert!(ops.allocator_metrics().live_requested_bytes >= live);
    }
}

#[test]
#[ignore = "requires CUDA; unsupported capture is rejected before the submission closure"]
fn root_submission_rejects_capture_before_any_submission() {
    let fixture = Fixture::new();
    let mut owner = fixture.owner(precisions()[0], RESERVATION + PINS);
    owner.ops.enable_capture_safe();
    assert!(
        owner
            .with_root_submission(PINS, |_| -> Result<()> {
                panic!("capture must not reach submission")
            })
            .is_err()
    );
    owner.ops.disable_capture_safe();
    assert!(owner.root_submission.is_none());
    assert!(!owner.needs_quarantine());
}

#[test]
#[ignore = "requires CUDA; CPU router keeps its producer flush and synchronous readback"]
fn root_submission_cpu_router_is_a_real_barrier() {
    let fixture = Fixture::new();
    let mut owner = fixture.owner(precisions()[0], RESERVATION + PINS);
    let input = input(&mut owner, 1);
    warm(&mut owner, &fixture.expert, &input);
    let policy = MoeRouterSpec::new(
        12,
        2,
        RouterScoreFunction::Softmax,
        RouterSelection::TopK,
        true,
        1.0,
    )
    .unwrap();
    let before = owner.consumer_proofs;
    owner
        .with_root_submission(PINS, |this| {
            let logits = ready(this.linear(fixture.expert.gate(), &input, None)?);
            let result = this.router(&logits, &policy)?;
            assert!(matches!(result, OperatorProgress::Ready(_)));
            assert_eq!(this.consumer_proofs - before, 2);
            assert_eq!(
                this.root_submission.as_ref().unwrap().last_barrier,
                Some(BarrierReason::CpuRouter)
            );
            assert_eq!(this.root_submission.as_ref().unwrap().root_count(), 0);
            assert_eq!(
                this.expert_cache_stats().unwrap().scratch_bytes,
                RESERVATION + PINS
            );
            Ok(())
        })
        .unwrap();
    assert_eq!(owner.consumer_proofs - before, 2);
}

#[test]
#[ignore = "requires CUDA; ordinary resident FFN joins the same bounded semantic segment"]
fn root_submission_dense_swiglu_obeys_the_full_budget_flush() {
    for precision in precisions() {
        let fixture = Fixture::new();
        let mut owner = fixture.owner(precision, RESERVATION + PINS);
        let input = input(&mut owner, 1);
        warm(&mut owner, &fixture.expert, &input);
        let before = owner.consumer_proofs;
        owner
            .with_root_submission(PINS, |this| {
                let update = ready(this.dense_swiglu(&fixture.expert, &input, None)?);
                assert_eq!(this.consumer_proofs, before);
                assert_eq!(this.root_submission.as_ref().unwrap().root_count(), 5);
                // A byte-full FFN must flush before another operation, even though
                // its exact inputs are duplicates. Conservative preflight is bounded.
                let _output = ready(this.residual(input, &update)?);
                assert_eq!(this.consumer_proofs - before, 1);
                Ok(())
            })
            .unwrap();
        assert_eq!(owner.consumer_proofs - before, 2);
    }
}

#[test]
#[ignore = "requires CUDA; external callbacks need a real producer proof, including failure custody"]
fn root_submission_external_callback_barrier_proves_or_quarantines() {
    for lost_proof in [false, true] {
        let fixture = Fixture::new();
        let mut owner = fixture.owner(precisions()[0], RESERVATION + PINS);
        let input = input(&mut owner, 1);
        warm(&mut owner, &fixture.expert, &input);
        let before = owner.consumer_proofs;
        let mut callback_ran = false;
        let result = owner.with_root_submission(PINS, |this| {
            let _rows = ready(this.linear(fixture.expert.gate(), &input, None)?);
            assert_eq!(this.consumer_proofs, before);
            this.lose_next_consumer_proof = lost_proof;
            // Invoke through the exact trait seam used by shared FFN composition.
            let operators: &mut dyn StandardDecoderOperators = this;
            operators.before_external_routed_execution()?;
            callback_ran = true;
            assert_eq!(this.consumer_proofs, before + 1);
            let segment = this.root_submission.as_ref().unwrap();
            assert_eq!(segment.last_barrier, Some(BarrierReason::ExternalExperts));
            assert_eq!(segment.root_count(), 0);
            assert!(!segment.has_pending());
            Ok(())
        });
        assert_eq!(callback_ran, !lost_proof);
        if lost_proof {
            assert!(result.is_err());
            assert!(owner.needs_quarantine());
            assert!(owner.root_submission.as_ref().unwrap().is_unknown());
            assert_eq!(owner.root_submission.as_ref().unwrap().root_count(), 2);
            owner.ops.sync_stream().unwrap();
            owner.ops.sync_upload_stream().unwrap();
            assert!(owner.before_external_routed_execution().is_err());
            assert!(owner.quiesce().is_err());
        } else {
            result.unwrap();
            assert!(owner.root_submission.is_none());
            assert_eq!(owner.consumer_proofs, before + 1);
        }
    }
}

#[test]
#[ignore = "requires CUDA; clean external failure/Waiting preserves independently drainable custody"]
fn root_submission_external_errors_release_only_local_scope_credit() {
    use crate::transformer::{OperatorWaiting, standard};
    for idle in [false, true] {
        for waiting in [false, true] {
            let fixture = Fixture::new();
            let mut owner = fixture.owner(precisions()[0], RESERVATION + PINS);
            let input = input(&mut owner, 1);
            warm(&mut owner, &fixture.expert, &input);
            let mut remote_hold = None;
            let mut proof_at_dispatch = 0;
            let result: Result<()> = owner.with_root_submission(PINS, |this| {
                let rows = ready(this.linear(fixture.expert.gate(), &input, None)?);
                // Model shared_expert_gate's synchronous completion before EP.
                if idle {
                    this.synchronous(|_| Ok(()))?;
                    assert!(!this.root_submission.as_ref().unwrap().has_pending());
                }
                this.before_external_routed_execution()?;
                proof_at_dispatch = this.consumer_proofs;
                assert!(
                    this.root_submission
                        .as_ref()
                        .unwrap()
                        .can_return_external_error()
                );
                remote_hold = Some(
                    this.f32(&rows)?
                        .as_device_buffer()
                        .slice(0, rows.shape().elements())?,
                );
                if waiting {
                    standard::ready::<Rows>(OperatorProgress::Waiting(OperatorWaiting::Experts {
                        layer: 0,
                        experts: vec![0],
                    }))
                    .map(|_| ())
                } else {
                    Err(cuda_error("external executor failure"))
                }
            });
            let error = result.unwrap_err();
            let Error::Model { message } = error else {
                panic!("external error type was replaced: {error}");
            };
            assert!(message.contains(if waiting {
                "suspended unexpectedly"
            } else {
                "external executor failure"
            }));
            assert!(!owner.needs_quarantine());
            assert!(owner.root_submission.is_none());
            assert_eq!(
                owner.consumer_proofs, proof_at_dispatch,
                "scope error must not manufacture a remote completion proof"
            );
            assert_eq!(
                owner.expert_cache_stats().unwrap().scratch_bytes,
                RESERVATION
            );
            let hold = remote_hold.as_ref().unwrap();
            assert_eq!(
                hold.num_bytes(),
                12 * 4,
                "remote holder is independent of lexical scope cleanup"
            );
            owner.quiesce().unwrap();
            assert!(
                remote_hold.is_some(),
                "local quiescence is not remote drain"
            );
            drop(remote_hold); // simulated explicit remote drain
            owner
                .with_root_submission(PINS, |this| {
                    this.linear(fixture.expert.gate(), &input, None).map(|_| ())
                })
                .unwrap();
        }
    }
}

#[test]
#[ignore = "requires CUDA; external error exception must not forgive resumed work, panic, or owner poison"]
fn root_submission_external_handoff_does_not_mask_local_failure_or_panic() {
    for failure in 0..4 {
        let fixture = Fixture::new();
        let mut owner = fixture.owner(precisions()[0], RESERVATION + PINS);
        let input = input(&mut owner, 1);
        warm(&mut owner, &fixture.expert, &input);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            owner.with_root_submission(PINS, |this| -> Result<()> {
                this.linear(fixture.expert.gate(), &input, None)?;
                this.before_external_routed_execution()?;
                match failure {
                    0 => {
                        this.linear(fixture.expert.gate(), &input, None)?;
                    }
                    1 => {
                        this.synchronous(|_| Ok(()))?;
                    }
                    2 => panic!("external callback panic"),
                    _ => this.poisoned = true,
                }
                Err(cuda_error("failure after handoff"))
            })
        }));
        assert!(!matches!(result, Ok(Ok(_))));
        assert!(owner.needs_quarantine());
        assert!(owner.root_submission.as_ref().unwrap().is_unknown());
        assert_eq!(
            owner.expert_cache_stats().unwrap().scratch_bytes,
            RESERVATION + PINS
        );
        owner.ops.sync_stream().unwrap();
        owner.ops.sync_upload_stream().unwrap();
        assert!(owner.before_external_routed_execution().is_err());
        assert!(owner.quiesce().is_err());
    }
}
