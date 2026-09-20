use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::{Error, ParallelRankId, Result};
use ferrule_model::TensorRole;
use ferrule_model::checkpoint::{CheckpointDType, CheckpointTensorSlice};
use ferrule_model::moe::ExpertId;
use ferrule_model::nn::{
    ModulePath, ParameterDType, ParameterId, ParameterResidency, ParameterSpec,
};
use ferrule_model::transformer::expert_parallel::{
    CpuExpertResultExecutor, CpuReferenceExpertWorker, ExpertDispatchContext, ExpertDispatchLimits,
    ExpertDispatchPlan, ExpertParallelRoutedExecutor, ExpertPlacement, ExpertResult,
    ExpertResultExecutor, ExpertToken, ExpertTokenBucket, ExpertWorker, RoutedSwiGluExecutor,
    RoutedSwiGluRequest,
};
use ferrule_model::transformer::{
    CpuStandardDecoderOperators, ExactNameMapper, ExpertAvailability, ExpertProvider, HostRows,
    MoeRouterSpec, NameMapping, OperatorProgress, PreparedLinear, PreparedSwiGlu, RouterRoutes,
    RouterScoreFunction, RouterSelection, Rows, RowsArenaId, RowsDType, RowsShape,
    StandardDecoderOperators, StateDictBinder, StateDictMaterializer, StateDictSchema,
};

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use ferrule_backend::cuda::operators::linear::CudaOperators;
    use ferrule_model::execution::ExecutionPrecisionPolicy;
    use ferrule_model::transformer::{
        CudaExpertParallelRoutedExecutor, CudaExpertWorker, CudaStandardDecoderOperators,
        RowsDevice,
    };
    use std::rc::Rc;

    struct GpuResults {
        placement: ExpertPlacement,
        owners: Vec<(ParallelRankId, LocalProvider, CudaStandardDecoderOperators)>,
        calls: usize,
        corrupt: bool,
    }
    impl ExpertResultExecutor for GpuResults {
        fn execute(&mut self, bucket: ExpertTokenBucket) -> Result<Vec<ExpertResult>> {
            self.calls += 1;
            if bucket.tokens.is_empty() {
                return Ok(Vec::new());
            }
            let (owner, provider, operators) = self
                .owners
                .iter_mut()
                .find(|(owner, _, _)| *owner == bucket.owner_rank)
                .unwrap();
            let mut results = CudaExpertWorker::new(*owner, &self.placement, provider, operators)
                .compute(&bucket.tokens)?;
            results.reverse();
            if self.corrupt {
                results[0].transaction = transaction(42);
            }
            Ok(results)
        }
    }

    #[test]
    #[ignore = "requires CUDA; use an external timeout and --test-threads=1"]
    fn clamped_cuda_experts_and_gpu_combine_match_cpu_and_reject_stale_results() {
        let mut fixture = PreparedFixture::new();
        for expert in &mut fixture.experts {
            *expert = Arc::new(
                PreparedSwiGlu::new(
                    expert.gate().clone(),
                    expert.up().clone(),
                    expert.down().clone(),
                    Some(1.1),
                )
                .unwrap(),
            );
        }
        let host = seam_input();
        let routes = routes();
        let expected = local_prepared_output(&fixture, &host, &routes);
        let ops = Rc::new(CudaOperators::new_on_device(0).unwrap());
        let mut staging = CudaStandardDecoderOperators::new(
            Rc::clone(&ops),
            ExecutionPrecisionPolicy::f32(),
            &[],
        )
        .unwrap();
        let input = staging.bind_rows(host).unwrap();
        let mut results = GpuResults {
            placement: placement(),
            owners: Vec::new(),
            calls: 0,
            corrupt: false,
        };
        for (owner, id) in [(rank(2), 0), (rank(7), 1)] {
            let expert = &fixture.experts[id];
            let parameters = [expert.gate(), expert.up(), expert.down()]
                .map(|linear| linear.parameter().binding().clone());
            let mut operators = CudaStandardDecoderOperators::new(
                Rc::clone(&ops),
                ExecutionPrecisionPolicy::f32(),
                &parameters,
            )
            .unwrap();
            operators.prepare_expert(expert).unwrap();
            results
                .owners
                .push((owner, LocalProvider::new(&fixture, &[id]), operators));
        }
        let placement = placement();
        let output = CudaExpertParallelRoutedExecutor::new(
            &mut staging,
            members(),
            &placement,
            limits(),
            &mut results,
        )
        .routed_swiglu(seam_request(&input, &routes), &mut active)
        .unwrap();
        let OperatorProgress::Ready(output) = output else {
            panic!("CUDA EP suspended")
        };
        assert_eq!(output.device(), RowsDevice::Cuda { ordinal: 0 });
        assert_eq!(output.arena(), Some(RowsArenaId::new(83)));
        let actual = staging.download_rows(output).unwrap();
        for (&a, &e) in actual.values().iter().zip(expected.values()) {
            assert!((a - e).abs() <= 2e-5 * (1.0 + e.abs()), "CUDA={a}, CPU={e}");
        }
        assert_eq!(results.calls, members().len());
        results.calls = 0;
        results.corrupt = true;
        assert_error(
            CudaExpertParallelRoutedExecutor::new(
                &mut staging,
                members(),
                &placement,
                limits(),
                &mut results,
            )
            .routed_swiglu(seam_request(&input, &routes), &mut active),
            "stale transaction",
        );
        assert_eq!(results.calls, 1);
        results.calls = 0;
        assert!(
            CudaExpertParallelRoutedExecutor::new(
                &mut staging,
                members(),
                &placement,
                ExpertDispatchLimits {
                    max_tokens: 1,
                    ..limits()
                },
                &mut results
            )
            .routed_swiglu(seam_request(&input, &routes), &mut active)
            .is_err()
        );
        assert_eq!(results.calls, 0);
        assert!(
            CudaExpertParallelRoutedExecutor::new(
                &mut staging,
                members(),
                &placement,
                limits(),
                &mut results
            )
            .routed_swiglu(seam_request(&input, &routes), &mut |_| Err(rejected(
                "cancelled"
            )))
            .is_err()
        );
        assert_eq!(results.calls, 0);
        assert!(!staging.needs_quarantine());
        staging.quiesce().unwrap();
    }
}

const LAYER: usize = 4;
const INPUT: [f32; 4] = [1.0, 2.0, -2.0, 3.0];

fn rank(value: u32) -> ParallelRankId {
    ParallelRankId::new(value)
}

fn transaction(value: u64) -> ExecutionTransactionId {
    ExecutionTransactionId::new(value).unwrap()
}

fn context() -> ExpertDispatchContext {
    ExpertDispatchContext {
        transaction: transaction(41),
        source_rank: rank(3),
        layer: LAYER,
    }
}

fn members() -> Vec<ParallelRankId> {
    vec![rank(7), rank(3), rank(2)]
}

fn placement() -> ExpertPlacement {
    ExpertPlacement::new([(LAYER, 0, rank(2)), (LAYER, 1, rank(7))]).unwrap()
}

fn routes() -> RouterRoutes {
    // Deliberately neither expert-sorted nor normalized in the first row.
    RouterRoutes::new(2, 2, vec![1, 0, 0, 1], vec![0.25, 1.5, 0.6, 0.4]).unwrap()
}

fn limits() -> ExpertDispatchLimits {
    ExpertDispatchLimits {
        max_tokens: 128,
        max_bytes: 4096,
    }
}

fn plan() -> ExpertDispatchPlan {
    ExpertDispatchPlan::new(
        context(),
        members(),
        &placement(),
        &routes(),
        &[100, 200],
        2,
        limits(),
    )
    .unwrap()
}

fn active(id: ExecutionTransactionId) -> Result<()> {
    if id == context().transaction {
        Ok(())
    } else {
        Err(rejected("unknown transaction"))
    }
}

fn rejected(message: &str) -> Error {
    Error::Execution {
        message: message.into(),
    }
}

fn assert_error<T: std::fmt::Debug>(result: Result<T>, expected: &str) {
    let error = result.unwrap_err().to_string();
    assert!(
        error.contains(expected),
        "expected {expected:?}, got {error:?}"
    );
}

struct EchoWorker {
    owner: ParallelRankId,
    calls: usize,
}

impl ExpertWorker for EchoWorker {
    fn owner(&self) -> ParallelRankId {
        self.owner
    }

    fn compute(&mut self, tokens: &[ExpertToken]) -> Result<Vec<ExpertResult>> {
        self.calls += 1;
        Ok(tokens
            .iter()
            .map(|token| ExpertResult::from_token(token, self.owner, token.payload.clone()))
            .collect())
    }
}

fn echo_results(plan: &ExpertDispatchPlan, input: &[f32]) -> Vec<ExpertResult> {
    plan.dispatch(input, &mut active)
        .unwrap()
        .into_iter()
        .flat_map(|bucket| {
            bucket
                .tokens
                .iter()
                .map(|token| {
                    ExpertResult::from_token(token, bucket.owner_rank, token.payload.clone())
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

#[test]
fn fixed_member_and_route_order_keeps_empty_buckets_and_unweighted_payloads() {
    let plan = plan();
    let buckets = plan.dispatch(&INPUT, &mut active).unwrap();
    assert_eq!(plan.context(), context());
    assert_eq!(plan.members(), members());
    assert_eq!(plan.token_count(), 4);
    assert_eq!(plan.required_bytes(), 32);
    assert_eq!(
        buckets
            .iter()
            .map(|bucket| bucket.owner_rank)
            .collect::<Vec<_>>(),
        members()
    );
    assert!(buckets[1].tokens.is_empty());
    assert_eq!(
        buckets[0]
            .tokens
            .iter()
            .map(|t| (t.source_row, t.route_slot))
            .collect::<Vec<_>>(),
        [(0, 0), (1, 1)]
    );
    assert_eq!(
        buckets[2]
            .tokens
            .iter()
            .map(|t| (t.source_row, t.route_slot))
            .collect::<Vec<_>>(),
        [(0, 1), (1, 0)]
    );
    assert_eq!(buckets[0].tokens[0].payload, INPUT[..2]);
    assert_eq!(buckets[2].tokens[0].payload, INPUT[..2]);
    assert_eq!(buckets[0].tokens[1].sequence, 200);
    let mut empty = EchoWorker {
        owner: rank(3),
        calls: 0,
    };
    assert!(
        plan.execute_bucket(&buckets[1], &mut empty, &mut active)
            .unwrap()
            .is_empty()
    );
    assert_eq!(empty.calls, 0);
    assert_error(plan.dispatch(&INPUT[..3], &mut active), "length mismatch");
    assert_error(plan.dispatch(&[f32::NAN; 4], &mut active), "non-finite");
}

#[test]
fn skew_all_topk2_routes_can_land_on_one_owner() {
    let placement = ExpertPlacement::new([(LAYER, 0, rank(2)), (LAYER, 1, rank(2))]).unwrap();
    let routes = RouterRoutes::new(64, 2, [1, 0].repeat(64), [0.0, 2.0].repeat(64)).unwrap();
    let plan = ExpertDispatchPlan::new(
        context(),
        members(),
        &placement,
        &routes,
        &[100; 64],
        2,
        limits(),
    )
    .unwrap();
    let input = vec![3.0; 128];
    let mut buckets = plan.dispatch(&input, &mut active).unwrap();
    assert!(buckets[0].tokens.is_empty());
    assert!(buckets[1].tokens.is_empty());
    assert_eq!(buckets[2].tokens.len(), 128);
    buckets[2].tokens.reverse();
    let mut worker = EchoWorker {
        owner: rank(2),
        calls: 0,
    };
    let mut results = plan
        .execute_bucket(&buckets[2], &mut worker, &mut active)
        .unwrap();
    results.reverse();
    assert_eq!(
        plan.combine(&results, &mut active).unwrap().values(),
        &[6.0; 128]
    );
    assert_eq!(worker.calls, 1);
}

#[test]
fn placement_members_routes_and_sequences_reject_ambiguous_or_unknown_identities() {
    assert_error(
        ExpertPlacement::new([(LAYER, 0, rank(2)), (LAYER, 0, rank(7))]),
        "duplicate placement",
    );
    assert_eq!(placement().owner(LAYER, 0), Some(rank(2)));
    assert_eq!(placement().owner(LAYER + 1, 0), None);
    for (members, expected) in [
        (vec![], "non-empty"),
        (vec![rank(3), rank(3)], "duplicate dispatch member"),
        (vec![rank(7), rank(2)], "unknown source rank"),
        (vec![rank(3), rank(2)], "unknown expert owner"),
    ] {
        assert_error(
            ExpertDispatchPlan::new(
                context(),
                members,
                &placement(),
                &routes(),
                &[100, 200],
                2,
                limits(),
            ),
            expected,
        );
    }
    for (ids, weights, expected) in [
        (vec![0, 0], vec![0.5, 0.5], "duplicate expert"),
        (vec![0, 99], vec![0.5, 0.5], "unknown expert"),
        (vec![0, 1], vec![f32::NAN, 0.5], "finite"),
        (vec![0, 1], vec![f32::INFINITY, 0.5], "finite"),
        (vec![0, 1], vec![-0.5, 1.5], "non-negative"),
    ] {
        let routes = RouterRoutes::new(1, 2, ids, weights).unwrap();
        assert_error(
            ExpertDispatchPlan::new(
                context(),
                members(),
                &placement(),
                &routes,
                &[100],
                2,
                limits(),
            ),
            expected,
        );
    }
    assert_error(
        ExpertDispatchPlan::new(
            context(),
            members(),
            &placement(),
            &routes(),
            &[100],
            2,
            limits(),
        ),
        "sequence identity",
    );
    assert_error(
        ExpertDispatchPlan::new(
            context(),
            members(),
            &placement(),
            &routes(),
            &[100, 200],
            0,
            limits(),
        ),
        "non-empty",
    );
}

#[test]
fn admission_limits_include_topk_fanout_and_check_arithmetic_overflow() {
    let build = |limits, width| {
        ExpertDispatchPlan::new(
            context(),
            members(),
            &placement(),
            &routes(),
            &[100, 200],
            width,
            limits,
        )
    };
    assert_error(
        build(
            ExpertDispatchLimits {
                max_tokens: 3,
                max_bytes: 32,
            },
            2,
        ),
        "token limit",
    );
    assert_error(
        build(
            ExpertDispatchLimits {
                max_tokens: 4,
                max_bytes: 31,
            },
            2,
        ),
        "byte limit",
    );
    let exact = build(
        ExpertDispatchLimits {
            max_tokens: 4,
            max_bytes: 32,
        },
        2,
    )
    .unwrap();
    assert_eq!(exact.required_bytes(), 32);
    assert_eq!(exact.token_count(), 4);
    assert_error(
        build(
            ExpertDispatchLimits {
                max_tokens: 0,
                max_bytes: 32,
            },
            2,
        ),
        "token limit",
    );
    assert_error(
        build(
            ExpertDispatchLimits {
                max_tokens: 4,
                max_bytes: 0,
            },
            2,
        ),
        "byte limit",
    );
    assert_error(
        build(
            ExpertDispatchLimits {
                max_tokens: usize::MAX,
                max_bytes: usize::MAX,
            },
            usize::MAX,
        ),
        "overflow",
    );
}

#[test]
fn invalid_tokens_are_rejected_before_any_owner_computation() {
    let plan = plan();
    let bucket = plan.dispatch(&INPUT, &mut active).unwrap().remove(0);
    for case in 0..14 {
        let mut invalid = bucket.clone();
        match case {
            0 => invalid.tokens[0].transaction = transaction(42),
            1 => invalid.tokens[0].sequence += 1,
            2 => invalid.tokens[0].source_rank = rank(2),
            3 => invalid.tokens[0].source_row = usize::MAX,
            4 => invalid.tokens[0].route_slot = usize::MAX,
            5 => invalid.tokens[0].expert.expert = 99,
            6 => invalid.tokens[0].expert.layer += 1,
            7 => invalid.tokens[0].weight = 0.3,
            8 => invalid.tokens[0].payload.push(0.0),
            9 => invalid.tokens[0].payload[0] = f32::NAN,
            10 => invalid.tokens[1] = invalid.tokens[0].clone(),
            11 => {
                invalid.tokens.pop();
            }
            12 => invalid.tokens.push(invalid.tokens[0].clone()),
            13 => invalid.owner_rank = rank(99),
            _ => unreachable!(),
        }
        let mut worker = EchoWorker {
            owner: rank(7),
            calls: 0,
        };
        assert!(
            plan.execute_bucket(&invalid, &mut worker, &mut active)
                .is_err(),
            "case {case}"
        );
        assert_eq!(worker.calls, 0);
    }
    let mut wrong_worker = EchoWorker {
        owner: rank(2),
        calls: 0,
    };
    assert_error(
        plan.execute_bucket(&bucket, &mut wrong_worker, &mut active),
        "wrong owner",
    );
    assert_eq!(wrong_worker.calls, 0);
}

#[test]
fn invalid_results_never_combine_including_duplicate_stale_unknown_and_bad_shapes() {
    let plan = plan();
    let results = echo_results(&plan, &INPUT);
    for case in 0..14 {
        let mut invalid = results.clone();
        match case {
            0 => invalid[0].transaction = transaction(42),
            1 => invalid[0].sequence += 1,
            2 => invalid[0].source_rank = rank(2),
            3 => invalid[0].source_row = usize::MAX,
            4 => invalid[0].route_slot = usize::MAX,
            5 => invalid[0].expert.expert = 99,
            6 => invalid[0].expert.layer += 1,
            7 => invalid[0].owner_rank = rank(2),
            8 => invalid[0].output.push(0.0),
            9 => invalid[0].output[0] = f32::INFINITY,
            10 => invalid[1] = invalid[0].clone(),
            11 => {
                invalid.pop();
            }
            12 => invalid.push(invalid[0].clone()),
            13 => invalid[0].owner_rank = rank(99),
            _ => unreachable!(),
        }
        assert!(plan.combine(&invalid, &mut active).is_err(), "case {case}");
    }
    let mut duplicate = results.clone();
    duplicate[1] = duplicate[0].clone();
    assert_error(plan.combine(&duplicate, &mut active), "duplicate result");
    assert_error(plan.combine(&[], &mut active), "incomplete");
    // A late stale identity is rejected before an earlier weighted overflow
    // could be reduced. Failure exposes neither partial output nor mutation.
    let mut invalid = results.clone();
    for result in &mut invalid {
        result.output.fill(f32::MAX);
    }
    invalid.last_mut().unwrap().transaction = transaction(42);
    assert_error(plan.combine(&invalid, &mut active), "stale transaction");
    invalid.last_mut().unwrap().transaction = transaction(41);
    assert_error(
        plan.combine(&invalid, &mut active),
        "combined output is non-finite",
    );
}

#[test]
fn canonical_route_slot_order_not_arrival_or_expert_order_controls_f32_sum() {
    let placement = ExpertPlacement::new([
        (LAYER, 0, rank(2)),
        (LAYER, 1, rank(3)),
        (LAYER, 2, rank(7)),
    ])
    .unwrap();
    let routes = RouterRoutes::new(1, 3, vec![2, 0, 1], vec![1.0; 3]).unwrap();
    let plan = ExpertDispatchPlan::new(
        context(),
        members(),
        &placement,
        &routes,
        &[100],
        1,
        limits(),
    )
    .unwrap();
    let mut results = echo_results(&plan, &[1.0]);
    for result in &mut results {
        result.output[0] = [1e20, -1e20, 1.0][result.route_slot];
    }
    for _ in 0..3 {
        results.rotate_left(1);
        assert_eq!(
            plan.combine(&results, &mut active).unwrap().values(),
            &[1.0]
        );
    }
    results.reverse();
    assert_eq!(
        plan.combine(&results, &mut active).unwrap().values(),
        &[1.0]
    );
}

#[test]
fn external_cancel_or_unknown_transaction_prevents_dispatch_compute_and_combine() {
    let plan = plan();
    let buckets = plan.dispatch(&INPUT, &mut active).unwrap();
    let results = echo_results(&plan, &INPUT);
    for reason in ["cancelled", "unknown transaction"] {
        let mut deny = |id| {
            assert_eq!(id, context().transaction);
            Err(rejected(reason))
        };
        let mut worker = EchoWorker {
            owner: rank(7),
            calls: 0,
        };
        assert_error(plan.dispatch(&INPUT, &mut deny), reason);
        assert_error(
            plan.execute_bucket(&buckets[0], &mut worker, &mut deny),
            reason,
        );
        assert_error(plan.combine(&results, &mut deny), reason);
        // The authority is consulted before even validating a reply set.
        assert_error(plan.combine(&[], &mut deny), reason);
        assert_eq!(worker.calls, 0);
    }
    assert_eq!(results, echo_results(&plan, &INPUT));
}

#[test]
fn cancellation_during_computation_or_reduction_prevents_returning_results() {
    let plan = plan();
    let buckets = plan.dispatch(&INPUT, &mut active).unwrap();
    let mut calls = 0;
    let mut cancel_after_start = |_| {
        calls += 1;
        if calls == 1 {
            Ok(())
        } else {
            Err(rejected("cancelled"))
        }
    };
    let mut worker = EchoWorker {
        owner: rank(7),
        calls: 0,
    };
    assert_error(
        plan.execute_bucket(&buckets[0], &mut worker, &mut cancel_after_start),
        "cancelled",
    );
    assert_eq!(worker.calls, 1);
    let results = echo_results(&plan, &INPUT);
    let mut calls = 0;
    assert_error(
        plan.combine(&results, &mut |_| {
            calls += 1;
            if calls == 1 {
                Ok(())
            } else {
                Err(rejected("cancelled"))
            }
        }),
        "cancelled",
    );
    assert_eq!(calls, 2);
}

#[test]
fn worker_cannot_return_a_valid_route_from_a_different_owner() {
    struct WrongOwner(Vec<ExpertResult>);
    impl ExpertWorker for WrongOwner {
        fn owner(&self) -> ParallelRankId {
            rank(7)
        }
        fn compute(&mut self, _: &[ExpertToken]) -> Result<Vec<ExpertResult>> {
            Ok(self.0.clone())
        }
    }
    let plan = plan();
    let buckets = plan.dispatch(&INPUT, &mut active).unwrap();
    let results = buckets[2]
        .tokens
        .iter()
        .map(|token| ExpertResult::from_token(token, rank(2), token.payload.clone()))
        .collect();
    assert_error(
        plan.execute_bucket(&buckets[0], &mut WrongOwner(results), &mut active),
        "wrong owner",
    );
}

fn seam_input() -> Rows {
    Rows::Host(
        HostRows::new(
            RowsShape::new(2, 2).unwrap(),
            RowsDType::F32,
            None,
            INPUT.to_vec(),
        )
        .unwrap(),
    )
}

fn seam_request<'a>(input: &'a Rows, routes: &'a RouterRoutes) -> RoutedSwiGluRequest<'a> {
    RoutedSwiGluRequest {
        context: context(),
        sequences: &[100, 200],
        input,
        routes,
        arena: Some(RowsArenaId::new(83)),
    }
}

fn seam_execute(
    results: &mut dyn ExpertResultExecutor,
    placement: &ExpertPlacement,
    input: &Rows,
    routes: &RouterRoutes,
    check_active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
) -> Result<OperatorProgress<Rows>> {
    let mut executor = ExpertParallelRoutedExecutor::new(members(), placement, limits(), results);
    let injected: &mut dyn RoutedSwiGluExecutor = &mut executor;
    injected.routed_swiglu(seam_request(input, routes), check_active)
}

fn ready_host(progress: OperatorProgress<Rows>) -> HostRows {
    match progress {
        OperatorProgress::Ready(rows) => rows.into_host().unwrap(),
        other => panic!("expected ready CPU rows, got {other:?}"),
    }
}

fn operator_routes() -> RouterRoutes {
    let logits = Rows::Host(
        HostRows::new(
            RowsShape::new(2, 2).unwrap(),
            RowsDType::F32,
            None,
            vec![0.0, 2.0, 3.0, 1.0],
        )
        .unwrap(),
    );
    let policy = MoeRouterSpec::new(
        2,
        2,
        RouterScoreFunction::Softmax,
        RouterSelection::TopK,
        true,
        1.0,
    )
    .unwrap();
    match CpuStandardDecoderOperators::default()
        .router(&logits, &policy)
        .unwrap()
    {
        OperatorProgress::Ready(routes) => routes,
        other => panic!("expected ready router outputs, got {other:?}"),
    }
}

fn local_prepared_output(
    fixture: &PreparedFixture,
    input: &Rows,
    routes: &RouterRoutes,
) -> HostRows {
    let mut provider = LocalProvider::new(fixture, &[0, 1]);
    ready_host(
        CpuStandardDecoderOperators::default()
            .routed_swiglu(LAYER, input, routes, &mut provider, None)
            .unwrap(),
    )
}

#[test]
fn injected_routed_executor_uses_existing_router_and_prepared_cpu_math() {
    let fixture = PreparedFixture::new();
    let placement = placement();
    let input = seam_input();
    let routes = operator_routes();
    let expected = local_prepared_output(&fixture, &input, &routes);
    let mut provider2 = LocalProvider::new(&fixture, &[0]);
    let mut provider7 = LocalProvider::new(&fixture, &[1]);
    let mut worker2 = CpuReferenceExpertWorker::new(rank(2), &placement, &mut provider2);
    let mut worker7 = CpuReferenceExpertWorker::new(rank(7), &placement, &mut provider7);
    // No worker/provider is needed for source rank 3's empty bucket.
    let mut results = CpuExpertResultExecutor::new(vec![&mut worker2, &mut worker7]).unwrap();
    let output =
        ready_host(seam_execute(&mut results, &placement, &input, &routes, &mut active).unwrap());
    assert_eq!(output.values(), expected.values());
    assert_eq!(output.shape(), expected.shape());
    assert_eq!(output.arena(), Some(RowsArenaId::new(83)));
    assert_eq!(provider2.calls, [ExpertId::new(LAYER, 0)]);
    assert_eq!(provider7.calls, [ExpertId::new(LAYER, 1)]);
}

#[test]
fn cpu_result_executor_rejects_duplicate_and_missing_workers() {
    let mut first = EchoWorker {
        owner: rank(7),
        calls: 0,
    };
    let mut second = EchoWorker {
        owner: rank(7),
        calls: 0,
    };
    assert!(CpuExpertResultExecutor::new(vec![&mut first, &mut second]).is_err());
    let mut executor = CpuExpertResultExecutor::new(vec![]).unwrap();
    assert!(
        executor
            .execute(ExpertTokenBucket {
                owner_rank: rank(3),
                tokens: vec![]
            })
            .unwrap()
            .is_empty()
    );
    let bucket = plan().dispatch(&INPUT, &mut active).unwrap().remove(0);
    assert_error(executor.execute(bucket), "missing worker");
}

#[derive(Default)]
struct RecordingResults {
    calls: Vec<ParallelRankId>,
}

impl ExpertResultExecutor for RecordingResults {
    fn execute(&mut self, bucket: ExpertTokenBucket) -> Result<Vec<ExpertResult>> {
        self.calls.push(bucket.owner_rank);
        Ok(bucket
            .tokens
            .iter()
            .rev()
            .map(|token| ExpertResult::from_token(token, bucket.owner_rank, token.payload.clone()))
            .collect())
    }
}

#[test]
fn routed_seam_preserves_member_order_empty_buckets_and_original_weights() {
    let mut results = RecordingResults::default();
    let output = ready_host(
        seam_execute(
            &mut results,
            &placement(),
            &seam_input(),
            &routes(),
            &mut active,
        )
        .unwrap(),
    );
    assert_eq!(results.calls, members());
    assert_eq!(
        output.values(),
        plan()
            .combine(&echo_results(&plan(), &INPUT), &mut active)
            .unwrap()
            .values()
    );
}

#[test]
fn routed_seam_rejects_bad_admission_or_cancel_before_sending_any_bucket() {
    let placement = placement();
    let input = seam_input();
    let routes = routes();
    for reason in ["cancelled", "unknown transaction"] {
        let mut results = RecordingResults::default();
        assert_error(
            seam_execute(&mut results, &placement, &input, &routes, &mut |_| {
                Err(rejected(reason))
            }),
            reason,
        );
        assert!(results.calls.is_empty());
    }
    for case in 0..5 {
        let mut results = RecordingResults::default();
        let bounds = if case == 0 {
            ExpertDispatchLimits {
                max_tokens: 3,
                max_bytes: 32,
            }
        } else if case == 1 {
            ExpertDispatchLimits {
                max_tokens: 4,
                max_bytes: 31,
            }
        } else {
            limits()
        };
        let invalid_routes = RouterRoutes::new(1, 2, vec![0, 1], vec![0.5, 0.5]).unwrap();
        let mut request = seam_request(&input, &routes);
        match case {
            2 => request.routes = &invalid_routes,
            3 => request.sequences = &[100],
            4 => request.context.layer += 1,
            _ => (),
        }
        let mut executor =
            ExpertParallelRoutedExecutor::new(members(), &placement, bounds, &mut results);
        assert!(executor.routed_swiglu(request, &mut active).is_err());
        assert!(results.calls.is_empty());
    }
    let input = Rows::Host(
        HostRows::new(
            RowsShape::new(2, 2).unwrap(),
            RowsDType::Bf16,
            None,
            INPUT.to_vec(),
        )
        .unwrap(),
    );
    let mut results = RecordingResults::default();
    assert!(matches!(
        seam_execute(&mut results, &placement, &input, &routes, &mut active).unwrap(),
        OperatorProgress::Unsupported(_)
    ));
    assert!(results.calls.is_empty());
}

#[test]
fn routed_seam_validates_owner_reply_before_next_bucket_or_combine() {
    struct Malformed {
        case: usize,
        calls: usize,
    }
    impl ExpertResultExecutor for Malformed {
        fn execute(&mut self, bucket: ExpertTokenBucket) -> Result<Vec<ExpertResult>> {
            self.calls += 1;
            let mut results = bucket
                .tokens
                .iter()
                .map(|token| {
                    ExpertResult::from_token(token, bucket.owner_rank, token.payload.clone())
                })
                .collect::<Vec<_>>();
            match self.case {
                0 => results[1] = results[0].clone(),
                1 => results[0].transaction = transaction(42),
                2 => results[0].sequence += 1,
                3 => results[0].owner_rank = rank(2),
                4 => results[0].expert.expert = 99,
                5 => {
                    results.pop();
                }
                6 => results.push(results[0].clone()),
                7 => results[0].output[0] = f32::NAN,
                8 => results[0].output.push(0.0),
                _ => return Err(rejected("transport disconnected")),
            }
            Ok(results)
        }
    }
    for case in 0..10 {
        let mut executor = Malformed { case, calls: 0 };
        assert!(
            seam_execute(
                &mut executor,
                &placement(),
                &seam_input(),
                &routes(),
                &mut active
            )
            .is_err(),
            "case {case}"
        );
        assert_eq!(executor.calls, 1);
    }
}

#[test]
fn routed_seam_observes_cancellation_after_owner_reply_without_more_dispatch() {
    use std::cell::Cell;
    struct Cancelling<'a> {
        active: &'a Cell<bool>,
        calls: usize,
    }
    impl ExpertResultExecutor for Cancelling<'_> {
        fn execute(&mut self, bucket: ExpertTokenBucket) -> Result<Vec<ExpertResult>> {
            self.calls += 1;
            self.active.set(false);
            Ok(bucket
                .tokens
                .iter()
                .map(|token| {
                    ExpertResult::from_token(token, bucket.owner_rank, token.payload.clone())
                })
                .collect())
        }
    }
    let live = Cell::new(true);
    let mut executor = Cancelling {
        active: &live,
        calls: 0,
    };
    assert_error(
        seam_execute(
            &mut executor,
            &placement(),
            &seam_input(),
            &routes(),
            &mut |_| {
                if live.get() {
                    Ok(())
                } else {
                    Err(rejected("cancelled"))
                }
            },
        ),
        "cancelled",
    );
    assert_eq!(executor.calls, 1);
}

#[test]
fn routed_seam_moves_only_typed_tokens_and_results_across_owner_threads() {
    use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
    use std::time::Duration;

    struct ThreadResults {
        requests: BTreeMap<ParallelRankId, SyncSender<ExpertTokenBucket>>,
        replies: BTreeMap<ParallelRankId, Receiver<Result<Vec<ExpertResult>>>>,
        calls: Vec<ParallelRankId>,
    }
    impl ExpertResultExecutor for ThreadResults {
        fn execute(&mut self, bucket: ExpertTokenBucket) -> Result<Vec<ExpertResult>> {
            let owner = bucket.owner_rank;
            self.calls.push(owner);
            if bucket.tokens.is_empty() {
                return Ok(vec![]);
            }
            self.requests
                .get(&owner)
                .unwrap()
                .send(bucket)
                .map_err(|_| rejected("request disconnected"))?;
            self.replies
                .get(&owner)
                .unwrap()
                .recv_timeout(Duration::from_secs(5))
                .map_err(|_| rejected("reply timeout/disconnected"))?
        }
    }
    fn assert_send<T: Send>() {}
    assert_send::<ExpertTokenBucket>();
    assert_send::<Vec<ExpertResult>>();
    let fixture = PreparedFixture::new();
    let placement = placement();
    let input = seam_input();
    let routes = operator_routes();
    let expected = local_prepared_output(&fixture, &input, &routes);
    let output = std::thread::scope(|scope| {
        let mut remote = ThreadResults {
            requests: BTreeMap::new(),
            replies: BTreeMap::new(),
            calls: vec![],
        };
        for (owner, expert) in [(rank(2), 0), (rank(7), 1)] {
            let (send, receive) = sync_channel::<ExpertTokenBucket>(1);
            let (reply, result) = sync_channel::<Result<Vec<ExpertResult>>>(1);
            remote.requests.insert(owner, send);
            remote.replies.insert(owner, result);
            let fixture = &fixture;
            let placement = &placement;
            scope.spawn(move || {
                // Provision the local owner before receiving activations. No
                // provider or prepared weights travel through either channel.
                let mut provider = LocalProvider::new(fixture, &[expert]);
                let mut worker = CpuReferenceExpertWorker::new(owner, placement, &mut provider);
                let mut executor = CpuExpertResultExecutor::new(vec![&mut worker]).unwrap();
                let bucket = receive.recv_timeout(Duration::from_secs(5)).unwrap();
                assert_eq!(bucket.owner_rank, owner);
                reply.send(executor.execute(bucket)).unwrap();
                assert_eq!(provider.calls, [ExpertId::new(LAYER, expert)]);
            });
        }
        let output = ready_host(
            seam_execute(&mut remote, &placement, &input, &routes, &mut active).unwrap(),
        );
        assert_eq!(remote.calls, members());
        output
    });
    assert_eq!(output.values(), expected.values());
    assert_eq!(output.arena(), Some(RowsArenaId::new(83)));
}

// Real prepared F32 experts, bound through the existing checkpoint/materializer
// APIs. The providers below retain only their local expert, not a remote weight
// lookup or weight-carrying worker reply.
struct PreparedFixture {
    directory: PathBuf,
    experts: Vec<Arc<PreparedSwiGlu>>,
}

impl PreparedFixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "ferrule-ep-{}-{nonce}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).unwrap();
        let mut fixture = Self {
            directory,
            experts: Vec::new(),
        };
        let file = fixture.directory.join("experts.bin");
        let mut schema = StateDictSchema::builder();
        let mut mapper = ExactNameMapper::new();
        let mut slices = Vec::new();
        let mut bytes = Vec::new();
        for expert in 0..2 {
            for (projection, role, values) in [
                (
                    "gate_proj",
                    TensorRole::RoutedExpertGate,
                    [1.0f32, 0.0, 0.0, 1.0],
                ),
                ("up_proj", TensorRole::RoutedExpertUp, [1.0, 0.0, 0.0, 1.0]),
                (
                    "down_proj",
                    TensorRole::RoutedExpertDown,
                    if expert == 0 {
                        [2.0, 0.0, 0.0, 3.0]
                    } else {
                        [-1.0, 0.0, 0.0, 4.0]
                    },
                ),
            ] {
                let name = format!("layers.{LAYER}.experts.{expert}.{projection}.weight");
                let path = ModulePath::new(name.clone()).unwrap();
                let parameter = ParameterSpec::new(
                    ParameterId::new(slices.len() as u64 + 1),
                    path.clone(),
                    ParameterDType::F32,
                    [2, 2],
                    ParameterResidency::expert(LAYER, expert),
                )
                .unwrap();
                schema.register_with_role(parameter, role.clone()).unwrap();
                mapper
                    .insert(name.clone(), NameMapping::weight(path))
                    .unwrap();
                let offset = bytes.len() as u64;
                bytes.extend(values.into_iter().flat_map(f32::to_le_bytes));
                slices.push(CheckpointTensorSlice {
                    name,
                    role,
                    path: file.clone(),
                    offset,
                    bytes: 16,
                    dtype: CheckpointDType::F32,
                    shape: vec![2, 2],
                });
            }
        }
        std::fs::write(&file, bytes).unwrap();
        let schema = schema.build().unwrap();
        let bound = StateDictBinder::new(&schema, &mapper)
            .bind_slices(slices)
            .unwrap();
        let materializer = StateDictMaterializer::new(16).unwrap();
        for expert in 0..2 {
            let linear = |role: TensorRole| {
                let binding = bound
                    .expert(LAYER, expert)
                    .find(|binding| binding.role() == &role)
                    .unwrap();
                PreparedLinear::from_parameter(
                    materializer
                        .expert_parameter(LAYER, expert, binding)
                        .unwrap(),
                    role,
                )
                .unwrap()
            };
            fixture.experts.push(Arc::new(
                PreparedSwiGlu::new(
                    linear(TensorRole::RoutedExpertGate),
                    linear(TensorRole::RoutedExpertUp),
                    linear(TensorRole::RoutedExpertDown),
                    None,
                )
                .unwrap(),
            ));
        }
        fixture
    }
}

impl Drop for PreparedFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

struct LocalProvider {
    experts: BTreeMap<ExpertId, Arc<PreparedSwiGlu>>,
    calls: Vec<ExpertId>,
}

impl LocalProvider {
    fn new(fixture: &PreparedFixture, ids: &[usize]) -> Self {
        Self {
            experts: ids
                .iter()
                .map(|&expert| {
                    (
                        ExpertId::new(LAYER, expert),
                        fixture.experts[expert].clone(),
                    )
                })
                .collect(),
            calls: Vec::new(),
        }
    }
}

impl ExpertProvider for LocalProvider {
    fn expert(&mut self, layer: usize, expert: usize) -> Result<ExpertAvailability> {
        let id = ExpertId::new(layer, expert);
        self.calls.push(id);
        self.experts
            .get(&id)
            .cloned()
            .map(ExpertAvailability::Ready)
            .ok_or_else(|| rejected("attempt to fetch non-local weights"))
    }
}

#[test]
fn topk2_two_owners_actual_prepared_swiglu_matches_local_moe_and_weights_once() {
    let fixture = PreparedFixture::new();
    let placement = placement();
    let plan = plan();
    let buckets = plan.dispatch(&INPUT, &mut active).unwrap();
    let mut owner2 = LocalProvider::new(&fixture, &[0]);
    let mut owner7 = LocalProvider::new(&fixture, &[1]);
    let mut results = plan
        .execute_bucket(
            &buckets[2],
            &mut CpuReferenceExpertWorker::new(rank(2), &placement, &mut owner2),
            &mut active,
        )
        .unwrap();
    results.extend(
        plan.execute_bucket(
            &buckets[0],
            &mut CpuReferenceExpertWorker::new(rank(7), &placement, &mut owner7),
            &mut active,
        )
        .unwrap(),
    );
    assert_eq!(owner2.calls, [ExpertId::new(LAYER, 0)]);
    assert_eq!(owner7.calls, [ExpertId::new(LAYER, 1)]);
    for result in &results {
        let x = INPUT[result.source_row * 2];
        let down = if result.expert.expert == 0 { 2.0 } else { -1.0 };
        let expected_unweighted = down * (x / (1.0 + (-x).exp()) * x);
        assert!((result.output[0] - expected_unweighted).abs() < 1e-6);
    }
    let mut local = LocalProvider::new(&fixture, &[0, 1]);
    let input = Rows::Host(
        HostRows::new(
            RowsShape::new(2, 2).unwrap(),
            RowsDType::F32,
            None,
            INPUT.to_vec(),
        )
        .unwrap(),
    );
    let OperatorProgress::Ready(expected) = CpuStandardDecoderOperators::default()
        .routed_swiglu(LAYER, &input, &routes(), &mut local, None)
        .unwrap()
    else {
        panic!("CPU reference must be ready")
    };
    results.reverse();
    let output = plan.combine(&results, &mut active).unwrap();
    assert_eq!(output.values(), expected.host().unwrap().values());
    results.rotate_left(1);
    assert_eq!(plan.combine(&results, &mut active).unwrap(), output);
    // Route weights sum to 1.75 in row zero; EP must not renormalize them.
    let silu_times_x = 1.0 / (1.0 + (-1.0f32).exp());
    assert!((output.values()[0] - silu_times_x * (-0.25 + 1.5 * 2.0)).abs() < 1e-6);
}

#[test]
fn unavailable_experts_and_non_local_weight_access_fail_without_partial_results() {
    struct Unavailable {
        waiting: bool,
        calls: usize,
    }
    impl ExpertProvider for Unavailable {
        fn expert(&mut self, _: usize, _: usize) -> Result<ExpertAvailability> {
            self.calls += 1;
            Ok(if self.waiting {
                ExpertAvailability::Waiting
            } else {
                ExpertAvailability::Unsupported("fixture".into())
            })
        }
    }
    let plan = plan();
    let placement = placement();
    let buckets = plan.dispatch(&INPUT, &mut active).unwrap();
    for waiting in [true, false] {
        let mut provider = Unavailable { waiting, calls: 0 };
        let mut worker = CpuReferenceExpertWorker::new(rank(7), &placement, &mut provider);
        assert_error(worker.compute(&buckets[2].tokens), "non-local expert");
        assert_error(
            plan.execute_bucket(&buckets[0], &mut worker, &mut active),
            if waiting { "waiting" } else { "unsupported" },
        );
        assert_eq!(provider.calls, 1);
    }
}
