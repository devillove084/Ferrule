//! CPU-only R64 route-root budget floor, not a model partial/numeric fixture.
#[allow(dead_code)]
#[path = "../../ferrule-backend/src/cuda/operators/bf16_route_metadata.rs"]
mod metadata;
#[allow(dead_code)]
#[path = "../src/transformer/standard/cuda/hybrid_rank/mixed_output.rs"]
mod output;
use ferrule_model::transformer::hybrid_parallel::HybridMixedRankOperationEstimate;

#[test]
fn synthetic_r64_gather_metadata_grants_reject_one_byte_underbudget() {
    let rows = 8;
    let width = 4096;
    let ids: Vec<usize> = (0..64).collect();
    let weights = vec![0.125; 64];
    let batch = output::RoutedBatchPlan::new(rows, width, 64, &ids, &weights).unwrap();
    assert!(output::batch_metadata_supported(rows, batch.row_ids.len()));
    let packed = metadata::PackedRoutes::new(&batch.row_ids, &batch.weights, rows).unwrap();
    assert_eq!(packed.routes, 64);
    assert_eq!(packed.words.len() * std::mem::size_of::<u32>(), 512);
    assert_eq!(packed.pointers(0x1000, 128).unwrap(), (0x1000, 0x1100));
    assert_eq!(batch.layout.elements(), 64 * width);
    let gather = ferrule_backend::cuda_allocation_grant_bytes(2 * batch.layout.elements()).unwrap();
    let opaque = ferrule_backend::cuda_allocation_grant_bytes(512).unwrap();
    assert_eq!(gather, 524_288);
    assert_eq!(opaque, 512);
    // Isolate only the incremental route-root floor. The existing real-header
    // gate admits the complete weights/state/provider/cache envelope separately.
    let estimate = HybridMixedRankOperationEstimate {
        storage_grants: 0,
        gemmex_storage_grants: 0,
        gemmex_provider_bytes: 0,
        attention_storage_grants: 0,
        attention_provider_bytes: 0,
        derived_rope_grants: 0,
        attention_metadata_grants: 0,
        recurrent_workspace_grants: 0,
        state_copy_workspace_grants: 0,
        operation_grants: gather + opaque,
        boundary_device_grants: 0,
        pinned_host_bytes: 0,
        host_staging_bytes: 16 * 64 + 3 * std::mem::size_of::<usize>() * batch.groups.len(),
    };
    assert_eq!(estimate.host_staging_bytes, 2560);
    assert_eq!(
        estimate.require_device_capacity(524_800, 0).unwrap(),
        524_800
    );
    assert!(estimate.require_device_capacity(524_799, 0).is_err());
    assert!(estimate.require_device_capacity(524_800, 1).is_err());
    println!(
        "SYNTHETIC_R64_BUDGET gather_grant={gather} opaque_grant={opaque} host_staging={} full_model_claim=false partial_golden=false",
        estimate.host_staging_bytes
    );
}
