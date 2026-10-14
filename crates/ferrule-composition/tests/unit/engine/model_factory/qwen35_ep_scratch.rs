use super::*;
use ferrule_model::TensorRole;
use ferrule_model::transformer::{PreparedLinear, PreparedSwiGlu, StateDictMaterializer};

#[test]
fn scoped_descriptor_peak_matches_prepared_experts_and_exact_payload() {
    let fixture = super::tests::Fixture::new();
    let resources = &fixture.resources;
    let placement = Qwen35ExpertPlacement::new(2, vec![0, 1]).unwrap();
    let materializer = StateDictMaterializer::new(4096).unwrap();
    for layers in [0..1, 1..2, 0..2] {
        for rows in [1, 3, 32] {
            let plan = OwnerPlan::for_stage(
                resources,
                &placement,
                rows,
                layers.clone(),
                ParallelRankId::new(7),
                vec![ParallelRankId::new(109), ParallelRankId::new(42)],
            )
            .unwrap();
            let mut prepared_peak = 0;
            for layer in layers.clone() {
                let FeedForward::Moe(moe) = resources.spec().layers()[layer].feed_forward() else {
                    panic!()
                };
                for expert in 0..4 {
                    let linear = |role: TensorRole| {
                        let binding = resources
                            .experts()
                            .require(layer, expert, role.clone())
                            .unwrap();
                        PreparedLinear::from_parameter(
                            materializer
                                .expert_parameter(layer, expert, binding)
                                .unwrap(),
                            role,
                        )
                        .unwrap()
                    };
                    let prepared = PreparedSwiGlu::new(
                        linear(TensorRole::RoutedExpertGate),
                        linear(TensorRole::RoutedExpertUp),
                        linear(TensorRole::RoutedExpertDown),
                        None,
                    )
                    .unwrap();
                    let bytes = prepared
                        .f32_scratch_bytes(plan.group.limits.max_tokens)
                        .unwrap();
                    assert_eq!(
                        bytes,
                        moe.expert()
                            .f32_scratch_bytes(plan.group.limits.max_tokens)
                            .unwrap()
                    );
                    prepared_peak = prepared_peak.max(bytes);
                }
            }
            assert_eq!(plan.activation_bytes, prepared_peak);
            assert_eq!(plan.activation_bytes, rows * 2 * (12 * 3 + 8 * 2) * 4);
            assert_eq!(plan.counts, vec![2 * layers.len(); 2]);
            assert_eq!(plan.weights, vec![2 * layers.len() * 294; 2]);
        }
    }
}

#[test]
fn factory_preserves_zero_rows_and_checked_overflow_rejections() {
    let fixture = super::tests::Fixture::new();
    let placement = Qwen35ExpertPlacement::new(2, vec![0, 1]).unwrap();
    for rows in [0, usize::MAX, usize::MAX / 2, usize::MAX / (2 * 208) + 1] {
        assert!(OwnerPlan::new(&fixture.resources, &placement, rows).is_err());
    }
    let last = usize::MAX / (2 * 208);
    let plan = OwnerPlan::new(&fixture.resources, &placement, last).unwrap();
    assert_eq!(plan.activation_bytes, last * 2 * 208);
}

#[test]
fn physical_card_charges_keep_workspace_and_allocator_reservations_separate() {
    let fixture = super::tests::Fixture::new();
    let placement = Qwen35ExpertPlacement::new(2, vec![0, 1]).unwrap();
    let owners = OwnerPlan::new(&fixture.resources, &placement, 4).unwrap();
    let root = Qwen35ExpertRootMemory {
        weights_bytes: 100,
        state_bytes: 200,
        kv_bytes: 300,
        workspace_bytes: 400,
    };
    let margin = 512;
    let owner_workspace = add(OWNER_WORKSPACE, owners.activation_bytes).unwrap();
    let physical =
        Qwen35ExpertPhysicalPlan::new(&placement, root, &owners.weights, owner_workspace, margin)
            .unwrap();
    let cards = physical.cards();
    assert_eq!(
        cards[0].workspace_bytes,
        owner_workspace + root.workspace_bytes
    );
    assert_eq!(cards[1].workspace_bytes, owner_workspace);
    assert_eq!(cards[0].contexts, 2);
    assert_eq!(cards[1].contexts, 1);
    assert_eq!(cards[0].allocator_margin_bytes, 2 * margin);
    assert_eq!(cards[1].allocator_margin_bytes, margin);
    assert_eq!(
        cards[0].required_bytes,
        100 + 200 + 300 + 400 + owners.weights[0] + owner_workspace + 2 * margin
    );
    assert_eq!(
        cards[1].required_bytes,
        owners.weights[1] + owner_workspace + margin
    );
    let capacities = cards
        .iter()
        .map(|card| Qwen35PhysicalCardCapacity {
            ordinal: card.ordinal,
            free_bytes: card.required_bytes,
            limit_bytes: card.required_bytes,
        })
        .collect::<Vec<_>>();
    physical.check_capacity(&capacities).unwrap();
    for slot in 0..cards.len() {
        let mut short = capacities.clone();
        short[slot].free_bytes -= 1;
        assert!(physical.check_capacity(&short).is_err());
    }
    assert!(add(usize::MAX, OWNER_WORKSPACE).is_err());
}
