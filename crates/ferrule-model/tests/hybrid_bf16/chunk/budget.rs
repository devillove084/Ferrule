use super::fixture::Fixture;
use ferrule_model::{
    ModelFamily, WeightSource,
    decoder::{GenericDecoderOptions, HybridLayerSchema, HybridMemoryPlan, HybridStateSchema},
    execution::ExecutionPrecisionPolicy,
    transformer::NumericFp8Precision as P,
};
fn options(f: &Fixture, rows: usize, profile: P) -> GenericDecoderOptions {
    GenericDecoderOptions::standard_cpu(
        f.resources.spec(),
        ModelFamily::Unknown("chunk-test".into()),
        WeightSource::Safetensors,
        2,
        32,
        rows,
        1,
        1 << 20,
        ExecutionPrecisionPolicy::bf16_compatibility(),
    )
    .unwrap()
    .with_hybrid_cuda_numeric_fp8_bf16_precision(super::tiled::limits(), 4096, profile)
    .unwrap()
}
#[test]
fn chunk_memory_charges_one_maximal_allocator_grant_not_state_or_cache() {
    let f = Fixture::numeric_profile();
    let schema =
        HybridStateSchema::from_spec(f.resources.spec(), f.resources.spec().layers().len())
            .unwrap();
    for rows in [1, 2, 5, 8] {
        let mixed = options(&f, rows, P::Bf16ModelMixedF32Accumulate);
        let tiled = options(&f, rows, P::Bf16ModelTiledF32Accumulate);
        let expected = schema
            .layers()
            .iter()
            .filter_map(|layer| {
                let HybridLayerSchema::GatedDeltaNet(s) = layer else {
                    return None;
                };
                P::Bf16ModelMixedF32Accumulate
                    .gdn_chunk_plan(
                        ferrule_backend::cuda_plan::GatedDeltaNetLayout {
                            rows,
                            key_heads: s.key_heads,
                            value_heads: s.value_heads,
                            key_dim: s.key_dim,
                            value_dim: s.value_dim,
                        },
                        false,
                        usize::MAX,
                    )
                    .unwrap()
                    .map(|p| p.workspace_requirements().bytes as usize)
            })
            .max()
            .unwrap_or(0);
        assert!(
            expected > 0,
            "fresh sequences require chunk scratch at rows={rows}"
        );
        assert_eq!(
            HybridMemoryPlan::bf16_chunk_workspace_bytes(&f.resources, &mixed).unwrap(),
            expected
        );
        assert_eq!(
            HybridMemoryPlan::bf16_chunk_workspace_bytes(&f.resources, &tiled).unwrap(),
            0
        );
        let a = HybridMemoryPlan::for_resources(&f.resources, &mixed, 32).unwrap();
        let mut b = HybridMemoryPlan::for_resources(&f.resources, &tiled, 32).unwrap();
        b.workspace_bytes_upper_bound +=
            ferrule_backend::cuda_allocation_grant_bytes(expected).unwrap();
        b.workspace_bytes_upper_bound +=
            HybridMemoryPlan::bf16_attention_plan(&f.resources, &mixed)
                .unwrap()
                .unwrap()
                .total_charge_bytes()
                + HybridMemoryPlan::bf16_attention_metadata_bytes(&f.resources, &mixed).unwrap();
        let (projection, external) =
            HybridMemoryPlan::bf16_gemmex_bytes(&f.resources, &mixed).unwrap();
        b.workspace_bytes_upper_bound +=
            ferrule_backend::cuda_allocation_grant_bytes(projection).unwrap() + external;
        assert_eq!(a, b);
    }
}
