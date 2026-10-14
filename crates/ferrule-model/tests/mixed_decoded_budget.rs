//! Metadata-only admission of the internal candidate, never GPU execution.
use ferrule_common::{ParallelRankId, ParallelTopologyId};
use ferrule_model::transformer::{hybrid_parallel::*, *};
use std::sync::Arc;
#[test]
#[ignore = "requires explicit real checkpoint headers; GPUs must remain hidden"]
fn real_tp2_decoded_cache_descriptor_ledger() {
    assert_eq!(std::env::var("CUDA_VISIBLE_DEVICES").as_deref(), Ok(""));
    let path = std::env::var("FERRULE_MIXED_HEADER_DIR").unwrap();
    let (_, resources) =
        ferrule_model::models::qwen35::Qwen35Adapter::bind_hf_metadata(std::path::Path::new(&path))
            .unwrap();
    let layers = resources.spec().layers().len();
    let topology = HybridParallelPlan::new_with_execution(
        resources.spec(),
        0..layers,
        HybridExpertScope {
            topology: ParallelTopologyId::new(907),
            replica: 0,
            stage: 0,
        },
        vec![ParallelRankId::new(0), ParallelRankId::new(1)],
        HybridRoutedExpertExecution::LocalTensorShards,
        HybridRankPrecision::MixedBf16F32,
    )
    .unwrap();
    for rank in 0..2 {
        let native = HybridMixedRankProgramPlan::for_segment(
            Arc::new(topology.rank_layout(rank).unwrap()),
            LayerSegmentPlan::new(layers, 0..layers, true, true).unwrap(),
            8,
            2,
            128,
            16,
            5,
        )
        .unwrap();
        assert_eq!(native.decoded_weight_cache_budget(), None);
        let base = native.operation_estimate(&resources, 64).unwrap();
        let enabled = native
            .clone()
            .with_decoded_weight_cache(&resources, 64, 3 << 30, 22 << 30, 128 << 20)
            .unwrap();
        let budget = enabled.decoded_weight_cache_budget().unwrap();
        assert_eq!(budget.pinned_grants, 1340 << 20);
        assert_eq!(budget.routed_working_set_grants, 768 << 20);
        assert_eq!(budget.routed_entries, 20_480);
        let estimate = enabled.operation_estimate(&resources, 64).unwrap();
        assert_eq!(estimate.operation_grants, base.operation_grants + (3 << 30));
        assert_eq!(estimate.gemmex_storage_grants, base.gemmex_storage_grants);
        assert_eq!(estimate.gemmex_provider_bytes, base.gemmex_provider_bytes);
        assert_eq!(estimate.storage_grants, base.storage_grants);
        let total = estimate
            .require_device_capacity(22 << 30, 128 << 20)
            .unwrap();
        println!(
            "rank={rank} decoded={budget:?} native_grants={} total_with_margin={total} headroom={}",
            base.total_device_grants().unwrap(),
            (22usize << 30) - total
        );
        assert!(
            budget
                .require_capacity(base.total_device_grants().unwrap(), 128 << 20, total - 1)
                .is_err()
        );
        assert!(enabled.operation_estimate(&resources, 32).is_err());
    }
}

#[path = "hybrid_bf16/fixture.rs"]
#[allow(dead_code)]
mod fixture;

#[test]
fn synthetic_route_batch_and_prepared_slots_are_charged_with_capacity_enforced() {
    let f = fixture::Fixture::new();
    let layers = f.resources.spec().layers().len();
    let topology = HybridParallelPlan::new_with_execution(
        f.resources.spec(),
        0..layers,
        HybridExpertScope {
            topology: ParallelTopologyId::new(907),
            replica: 0,
            stage: 0,
        },
        vec![ParallelRankId::new(0), ParallelRankId::new(1)],
        HybridRoutedExpertExecution::LocalTensorShards,
        HybridRankPrecision::MixedBf16F32,
    )
    .unwrap();
    for rank in 0..2 {
        let program = HybridMixedRankProgramPlan::for_segment(
            Arc::new(topology.rank_layout(rank).unwrap()),
            LayerSegmentPlan::new(layers, 0..layers, true, true).unwrap(),
            8,
            2,
            32,
            16,
            5,
        )
        .unwrap();
        let estimate = program.operation_estimate(&f.resources, 64).unwrap();
        let word = std::mem::size_of::<usize>();
        // Fixture: two local MoE layers, four experts, three routes per row.
        let slots = layers * (7 * word + 3 * word + 4 * 3 * word);
        let batch_host = 8 * 3 * 16 + 4 * 3 * word;
        assert!(estimate.host_staging_bytes >= slots + batch_host);
        assert_eq!(estimate.pinned_host_bytes, 0);
        let total = estimate
            .require_device_capacity(22 << 30, 128 << 20)
            .unwrap();
        assert!(total < 22 << 30);
        assert!(
            estimate
                .require_device_capacity(total - 1, 128 << 20)
                .is_err()
        );
    }
}
