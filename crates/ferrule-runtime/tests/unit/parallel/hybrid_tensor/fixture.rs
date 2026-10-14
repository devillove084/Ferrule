pub(super) use super::checkpoint_fixture::Fixture;
use super::*;

pub(super) fn rank(n: u32) -> ParallelRankId {
    ParallelRankId::new(n)
}
pub(super) fn config() -> PipelineConfig {
    PipelineConfig {
        page_size: 2,
        max_pages: 16,
        max_positions: 32,
        max_batch_tokens: 8,
        session_capacity: 3,
        max_parameter_bytes: 1 << 20,
        precision: ExecutionPrecisionPolicy::f32(),
        max_ack_polls: 2,
    }
}
pub(super) fn options() -> HybridCpuTensorConfig {
    HybridCpuTensorConfig {
        collective_limits: HostCollectiveLimits {
            max_ranks: 4,
            max_elements_per_rank: 4096,
            max_host_bytes: 1 << 20,
        },
        collective_timeout: Duration::from_secs(5),
        expert_limits: ExpertDispatchLimits {
            max_tokens: 32,
            max_bytes: 8192,
        },
        expert_timeout: Duration::from_secs(3),
        max_owner_resident_bytes: 1 << 24,
        max_expert_weight_bytes: 1 << 20,
    }
}
pub(super) fn topology(pp: usize, tp: usize) -> ValidatedParallelTopology {
    ValidatedParallelTopology::new(
        ParallelTopologyId::new(907),
        (pp * tp) as u32,
        rank(0),
        ParallelismPlan::validated(1, tp, 1, 1, 1, pp).unwrap(),
    )
    .unwrap()
}
pub(super) fn layout(spec: &DecoderModelSpec, pp: usize, tp: usize) -> HybridExpertTopology {
    HybridExpertTopology::new(
        spec,
        (0..pp)
            .map(|stage| {
                let layers = stage * 4 / pp..(stage + 1) * 4 / pp;
                let members = (0..tp)
                    .map(|lane| rank((stage * tp + lane) as u32))
                    .collect::<Vec<_>>();
                let experts = vec![
                    rank(101 + stage as u32 * 100),
                    rank(117 + stage as u32 * 100),
                ];
                let routed = layers
                    .clone()
                    .filter(|&layer| {
                        matches!(spec.layers()[layer].feed_forward(), FeedForward::Moe(_))
                    })
                    .collect::<Vec<_>>();
                let attachment = (!routed.is_empty()).then(|| HybridExpertAttachment {
                    source_rank: *members.last().unwrap(),
                    members: experts.clone(),
                    placement: ExpertPlacement::new(routed.iter().flat_map(|&layer| {
                        (0..2).map({
                            let experts = &experts;
                            move |e| (layer, e, experts[e])
                        })
                    }))
                    .unwrap(),
                });
                HybridParallelPlan::new(
                    spec,
                    layers,
                    HybridExpertScope {
                        topology: topology(pp, tp).topology_id(),
                        replica: 0,
                        stage: stage as u32,
                    },
                    members,
                    attachment,
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap()
}
