//! Independent old-formula checks for the shared child ledger; no CUDA context.
use super::super::tensor_tests::fixture::Fixture;
use super::*;
use crate::model_factory::qwen35::Qwen35ResolvedAdapter;
use crate::model_factory::*;
use ferrule_model::nn::ParameterResidency;

fn plan(f: &Fixture, pp: usize, tp: usize, ep: usize) -> Plan {
    let mut request = f.request();
    request.max_layers = 4;
    if let Qwen35ResolvedAdapter::Moe {
        expert_placement, ..
    } = request.family_options.qwen35_mut().unwrap()
    {
        *expert_placement = None;
    }
    request.pipeline = Some(PipelineBuildOptions {
        parallelism: ferrule_common::ParallelismPlan {
            pipeline_parallel: pp,
            tensor_parallel: tp,
            expert_parallel: ep,
            ..Default::default()
        },
        devices: Some(vec![0; pp * (tp + ep)]),
        ..Default::default()
    });
    crate::model_factory::qwen35::binding::pipeline_plan(&request, &f.resources).unwrap()
}

#[test]
fn representations_preserve_each_old_formula_owner_count_and_colocated_sum() {
    let f = Fixture::new();
    for (pp, tp, ep) in [(2, 1, 2), (2, 1, 4), (1, 2, 2), (2, 2, 2), (2, 4, 2)] {
        let plan = plan(&f, pp, tp, ep);
        let payloads = expert_budgets(
            &f.resources,
            &plan,
            ExpertBudgetRepresentation::PayloadEnvelope,
        )
        .unwrap();
        let grants = expert_budgets(
            &f.resources,
            &plan,
            ExpertBudgetRepresentation::PhysicalAllocationGrants,
        )
        .unwrap();
        assert_eq!(payloads.len(), pp * ep);
        assert_eq!(grants.len(), pp * ep);
        for stage in 0..pp {
            for slot in 0..ep {
                let owner = &plan.owners[stage];
                let mut payload = 0u64;
                let mut rounded = 0u64;
                for parameter in f.resources.state_dict().parameters() {
                    if let ParameterResidency::Expert { layer, expert } = parameter.residency()
                        && owner.group.layers.contains(layer)
                        && expert % ep == slot
                    {
                        let weight = parameter.weight().slice().bytes;
                        let scale = parameter.scale().unwrap().slice().bytes;
                        payload += weight + scale;
                        // This tiny fixture has separate sub-256-byte weight/scale
                        // allocations: do not round their combined payload once.
                        assert!(weight > 0 && weight <= 256 && scale > 0 && scale <= 256);
                        rounded += 256 + 256;
                    }
                }
                assert_eq!(payload, owner.weights[slot] as u64);
                assert_eq!(payload, owner.counts[slot] as u64 * 294);
                for (budget, weights, workspace) in [
                    (
                        &payloads[stage * ep + slot],
                        payload,
                        (64 << 20) + owner.activation_bytes as u64,
                    ),
                    (
                        &grants[stage * ep + slot],
                        rounded,
                        ferrule_backend::cuda_allocation_grant_bytes(64 << 20).unwrap() as u64
                            + 16 * ferrule_backend::cuda_allocation_grant_bytes(
                                owner.activation_bytes,
                            )
                            .unwrap() as u64,
                    ),
                ] {
                    assert_eq!(budget.device, 0);
                    assert_eq!(budget.weights, weights);
                    assert_eq!(budget.workspace, workspace);
                    assert_eq!((budget.state, budget.kv, budget.margin), (0, 0, 64 << 20));
                    assert_eq!(budget.total().unwrap(), weights + workspace + (64 << 20));
                }
            }
        }
        for budgets in [&payloads, &grants] {
            let total = budgets.iter().map(|b| b.total().unwrap()).sum::<u64>();
            let admitted = HybridDeviceBudget::admit(budgets, &[(0, total)]).unwrap();
            assert_eq!(admitted.len(), 1);
            assert_eq!(admitted[0].required, total);
            assert!(HybridDeviceBudget::admit(budgets, &[(0, total - 1)]).is_err());
        }
    }
}

#[test]
fn workspace_rounding_is_explicit_and_never_changes_payload_cache_limits() {
    let f = Fixture::new();
    let mut plan = plan(&f, 2, 1, 2);
    let payload_weights = plan
        .owners
        .iter()
        .map(|o| o.weights.clone())
        .collect::<Vec<_>>();
    for owner in &mut plan.owners {
        owner.activation_bytes = 257;
    }
    let payloads = expert_budgets(
        &f.resources,
        &plan,
        ExpertBudgetRepresentation::PayloadEnvelope,
    )
    .unwrap();
    let grants = expert_budgets(
        &f.resources,
        &plan,
        ExpertBudgetRepresentation::PhysicalAllocationGrants,
    )
    .unwrap();
    assert!(
        payloads
            .iter()
            .all(|b| b.workspace == (64 << 20) + 257 && b.weights == 4 * 294)
    );
    assert!(
        grants
            .iter()
            .all(|b| b.workspace == (64 << 20) + 16 * 512 && b.weights == 4 * 3 * 2 * 256)
    );
    assert_eq!(
        payload_weights,
        plan.owners
            .iter()
            .map(|o| o.weights.clone())
            .collect::<Vec<_>>()
    );
}

#[test]
fn both_representations_reject_workspace_overflow() {
    let f = Fixture::new();
    let mut plan = plan(&f, 2, 1, 2);
    plan.owners[0].activation_bytes = usize::MAX;
    for representation in [
        ExpertBudgetRepresentation::PayloadEnvelope,
        ExpertBudgetRepresentation::PhysicalAllocationGrants,
    ] {
        assert!(expert_budgets(&f.resources, &plan, representation).is_err());
    }
}
