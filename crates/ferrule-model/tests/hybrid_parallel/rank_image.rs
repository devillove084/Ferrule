//! Real prepared-image execution across general TP owners; no copied model math.
use super::{fixture, support::*};
use ferrule_common::{ParallelRankId, ParallelTopologyId};
use ferrule_model::transformer::hybrid_parallel::*;
use ferrule_model::transformer::{ExpertPlacement, LayerSegmentPlan};
use std::sync::{Arc, Barrier, atomic::AtomicBool};

fn layout(
    fixture: &fixture::Fixture,
    members: Vec<ParallelRankId>,
    leader: usize,
    dense: bool,
) -> HybridParallelPlan {
    let source_rank = members[leader];
    HybridParallelPlan::new(
        fixture.resources().spec(),
        0..4,
        HybridExpertScope {
            topology: ParallelTopologyId::new(907),
            replica: 0,
            stage: 0,
        },
        members,
        if dense {
            None
        } else {
            Some(HybridExpertAttachment {
                source_rank,
                members: vec![rank(101), rank(207)],
                placement: ExpertPlacement::new((1..4).flat_map(|l| {
                    (0..2).map(move |e| (l, e, rank(if e == 0 { 101 } else { 207 })))
                }))
                .unwrap(),
            })
        },
    )
    .unwrap()
}
fn compare_degree(degree: usize, dense: bool) {
    let fixture = Arc::new(fixture::Fixture::for_degree(4, dense));
    let steps: &[&[u32]] = &[&[1, 2, 3], &[4], &[6], &[2]];
    let mut oracle = Oracle::new(&fixture.resources());
    let expected = steps
        .iter()
        .enumerate()
        .map(|(i, t)| oracle.step(i as u64 + 1, t))
        .collect::<Vec<_>>();
    oracle.release();
    let members = [rank(7), rank(19), rank(31), rank(63)][..degree].to_vec();
    let leader = degree - 1;
    let plan = layout(&fixture, members.clone(), leader, dense);
    let endpoints = endpoints_for(&members);
    let control = endpoints[0].control();
    let barrier = Arc::new(Barrier::new(degree));
    let handles = endpoints
        .into_iter()
        .enumerate()
        .map(|(lane, endpoint)| {
            let fixture = Arc::clone(&fixture);
            let plan = plan.clone();
            let barrier = Arc::clone(&barrier);
            let control = control.clone();
            let expected = expected.clone();
            std::thread::spawn(move || {
                let mut tensor = Tensor::new(&fixture.resources(), plan.clone(), lane, endpoint);
                assert_eq!(tensor.module.layout().owner(), plan.tensor_members()[lane]);
                for (i, tokens) in steps.iter().enumerate() {
                    let id = i as u64 + 1;
                    if lane == leader && degree > 1 {
                        control
                            .begin(tx(id), Arc::new(AtomicBool::new(false)))
                            .unwrap();
                    }
                    barrier.wait();
                    let mut work = tensor.enter(id, tokens);
                    close(
                        &logits(
                            tensor
                                .execute(
                                    &mut work,
                                    ferrule_model::transformer::SegmentInput::Tokens,
                                )
                                .unwrap(),
                        ),
                        &expected[i].0,
                    );
                    compare_local(&plan, lane, &tensor.snapshot(&work), &expected[i].1);
                    assert_routes(&tensor, id, tokens.len());
                    // Local completion seeds the next numerical step; no global publication is asserted.
                    tensor.commit_local(work);
                    barrier.wait();
                    if lane == leader && degree > 1 {
                        control.finish(tx(id)).unwrap();
                    }
                    barrier.wait();
                }
                assert_eq!(tensor.experts.is_some(), lane == leader && !dense);
                tensor.release();
            })
        })
        .collect::<Vec<_>>();
    for handle in handles {
        handle.join().unwrap();
    }
}
#[test]
fn tp1_sparse_owner_matches_unsplit() {
    compare_degree(1, false);
}
#[test]
fn tp2_sparse_owners_nonzero_leader_match_unsplit() {
    compare_degree(2, false);
}
#[test]
fn tp4_sparse_owners_nonzero_leader_match_unsplit() {
    compare_degree(4, false);
}
#[test]
fn dense_tp4_has_no_expert_attachment_or_dispatch() {
    compare_degree(4, true);
}

#[test]
fn preparation_reads_exact_local_rectangles_and_binding_rejects_foreign_scope() {
    use ferrule_model::TensorRole;
    use ferrule_model::transformer::expert_parallel::{RoutedSwiGluExecutor, RoutedSwiGluRequest};
    struct Never;
    impl RoutedSwiGluExecutor for Never {
        fn routed_swiglu(
            &mut self,
            _: RoutedSwiGluRequest<'_>,
            _: &mut dyn FnMut(
                ferrule_common::execution::ExecutionTransactionId,
            ) -> ferrule_common::Result<()>,
        ) -> ferrule_common::Result<
            ferrule_model::transformer::OperatorProgress<ferrule_model::transformer::Rows>,
        > {
            panic!("binding must not execute")
        }
    }
    let fixture = fixture::Fixture::for_degree(4, false);
    let resources = fixture.resources();
    let plan = layout(
        &fixture,
        vec![rank(7), rank(19), rank(31), rank(63)],
        3,
        false,
    );
    assert!(
        plan.validate_spec(fixture.resources_with_context(16).spec())
            .is_err()
    );
    let mut executor = Never;
    for (lane, endpoint) in endpoints_for(plan.tensor_members()).into_iter().enumerate() {
        let segment = LayerSegmentPlan::new(4, 0..4, true, true).unwrap();
        let budget = plan
            .rank_layout(lane)
            .unwrap()
            .weight_budget(&resources, &segment)
            .unwrap();
        let largest_global = resources
            .state_dict()
            .parameters()
            .iter()
            .map(|p| p.weight().slice().bytes)
            .max()
            .unwrap();
        assert!((budget.max_parameter_bytes as u64) < largest_global);
        let module = HybridTensorCpuSegment::prepare(
            &resources,
            segment,
            plan.clone(),
            lane,
            Box::new(endpoint),
            32,
            budget.max_parameter_bytes as u64,
        )
        .unwrap();
        let reads = module.preparation_reads().unwrap();
        assert!(!reads.is_empty());
        assert_eq!(
            reads.iter().map(|r| r.bytes as usize).sum::<usize>(),
            budget.matrix_bytes
        );
        assert_eq!(
            budget.resident_bytes,
            budget.matrix_bytes + 2 * budget.norm_bytes
        );
        assert!(reads.iter().all(|r| !matches!(
            r.role,
            TensorRole::RoutedExpertGate
                | TensorRole::RoutedExpertUp
                | TensorRole::RoutedExpertDown
        )));
        for binding in resources.state_dict().parameters() {
            let Ok(partition) = module.layout().parameter_partition(binding) else {
                continue;
            };
            let actual = reads
                .iter()
                .filter(|r| r.parameter == binding.id())
                .collect::<Vec<_>>();
            assert_eq!(actual.len(), partition.rows.len());
            assert_eq!(
                actual.iter().map(|r| r.bytes).sum::<u64>(),
                partition.bytes_f32().unwrap() as u64
            );
            for (read, rows) in actual.iter().zip(&partition.rows) {
                let rect = read.rectangle.as_ref().unwrap();
                assert_eq!(rect.rows(), *rows);
                assert_eq!(rect.columns(), partition.columns);
            }
        }
        assert_eq!(module.schema(), module.layout().state_schema());
        assert_eq!(
            module.layout().state_bytes().unwrap() * 4,
            plan.schema().per_sequence_state_bytes().unwrap()
        );
        let attachment = plan.expert_attachment().unwrap();
        assert_eq!(
            module
                .bind_expert_call(plan.scope(), attachment, &mut executor)
                .is_ok(),
            lane == 3
        );
        for axis in 0..3 {
            let mut wrong_scope = plan.scope();
            match axis {
                0 => wrong_scope.topology = ParallelTopologyId::new(908),
                1 => wrong_scope.replica += 1,
                _ => wrong_scope.stage += 1,
            }
            assert!(
                module
                    .bind_expert_call(wrong_scope, attachment, &mut executor)
                    .is_err()
            );
        }
        let mut wrong = attachment.clone();
        wrong.source_rank = rank(7);
        assert!(
            module
                .bind_expert_call(plan.scope(), &wrong, &mut executor)
                .is_err()
        );
        wrong = attachment.clone();
        wrong.placement = ExpertPlacement::new([]).unwrap();
        assert!(
            module
                .bind_expert_call(plan.scope(), &wrong, &mut executor)
                .is_err()
        );
    }
}
