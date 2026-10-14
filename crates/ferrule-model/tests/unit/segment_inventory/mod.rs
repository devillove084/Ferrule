use super::*;

fn grant_oracle(bytes: usize) -> usize {
    if bytes == 0 {
        return 0;
    }
    if bytes <= 1_048_576 {
        let mut grant = 256;
        while grant < bytes {
            grant *= 2;
        }
        grant
    } else {
        bytes.div_ceil(2_097_152) * 2_097_152
    }
}
fn peak_oracle(rows: usize, channels: usize, heads: usize, width: usize) -> usize {
    2 * [channels, heads, heads, width, channels, width, width, width]
        .into_iter()
        .map(|columns| grant_oracle(rows * columns * 4))
        .sum::<usize>()
}

#[test]
fn recurrent_scratch_exact_lengths_and_growth_peak_are_checked() {
    for rows in [0, 1, 2, 3, 32] {
        let plan = RecurrentScratchPlan::new(rows, 10, 2, 4).unwrap();
        assert_eq!(plan.elements, [10, 2, 2, 4, 10, 4, 4, 4].map(|n| rows * n));
        // Old roots + one replacement set, NOT old + new + an extra capacity charge.
        assert_eq!(
            plan.growth_peak_bytes().unwrap(),
            peak_oracle(rows, 10, 2, 4)
        );
    }
    for (rows, channels, heads, width) in [
        (usize::MAX, 10, 2, 4),
        (2, i32::MAX as usize, 1, 1),
        (0, usize::MAX, 1, 1),
        (1, 10, 0, 4),
        (1, 0, 2, 4),
        (1, 10, 2, 0),
        (1, 10, 2, usize::MAX),
    ] {
        assert!(RecurrentScratchPlan::new(rows, channels, heads, width).is_err());
    }
    assert!(
        RecurrentScratchPlan {
            elements: [usize::MAX; 8]
        }
        .growth_peak_bytes()
        .is_err()
    );
}

#[test]
fn recurrent_capacity_is_componentwise_high_water_not_largest_layer_or_layer_sum() {
    let first = RecurrentScratchPlan::new(3, 300, 2, 4).unwrap();
    let second = RecurrentScratchPlan::new(3, 20, 80, 120).unwrap();
    let mut capacity = first;
    capacity.include(second);
    assert_eq!(capacity.elements, [900, 240, 240, 360, 900, 360, 360, 360]);
    assert_eq!(capacity.growth_peak_bytes().unwrap(), 36_864);
    assert!(
        capacity.growth_peak_bytes().unwrap()
            > first
                .growth_peak_bytes()
                .unwrap()
                .max(second.growth_peak_bytes().unwrap())
    );
    assert!(
        capacity.growth_peak_bytes().unwrap()
            < first.growth_peak_bytes().unwrap() + second.growth_peak_bytes().unwrap()
    );
    capacity.include(RecurrentScratchPlan::new(1, 10, 1, 2).unwrap());
    assert_eq!(capacity.growth_peak_bytes().unwrap(), 36_864);
}

#[test]
fn recurrent_budget_is_one_owner_peak_and_does_not_charge_state_or_numeric_workspace() {
    let f = Fixture::new();
    std::fs::remove_file(&f.path).unwrap();
    let layers = f.resources.spec().layers();
    let whole = HybridSegmentMemoryEstimate::recurrent_workspace_bytes(layers, 2).unwrap();
    let one = HybridSegmentMemoryEstimate::recurrent_workspace_bytes(&layers[..1], 2).unwrap();
    assert_eq!(whole, one, "uniform repeated layers share the same scratch");
    assert_eq!(
        HybridSegmentMemoryEstimate::recurrent_workspace_bytes(&[], 2).unwrap(),
        0
    );
    assert_eq!(
        HybridSegmentMemoryEstimate::recurrent_workspace_bytes(layers, 0).unwrap(),
        0
    );
    for rows in [1, 3, 32] {
        assert_eq!(
            HybridSegmentMemoryEstimate::recurrent_workspace_bytes(layers, rows).unwrap(),
            peak_oracle(rows, 6144, 16, 2048)
        );
    }
    let estimate = f.estimate(&plan(0..24, true, true));
    assert_eq!(
        estimate.device.workspace_bytes - estimate.host.workspace_bytes,
        whole
    );
    assert_eq!(estimate.per_sequence_state_bytes, 2 * 10_321_920);
}
use crate::models::qwen35::{
    Qwen35Config, Qwen35HfNameMapper, Qwen35Recipe, Qwen35TensorPartitionKind,
};
use crate::transformer::StateDictBinder;
use crate::{CheckpointTensorSlice, ModelFamily, WeightSource};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

struct Fixture {
    path: PathBuf,
    resources: BoundDecoderResources,
}
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let config = Qwen35Config::from_value(
            &serde_json::from_str(include_str!("../../qwen35_08b_config.json")).unwrap(),
        )
        .unwrap();
        let spec = Qwen35Recipe::spec(&config).unwrap();
        let schema = Qwen35Recipe::schema(&config).unwrap();
        let mapper = Qwen35HfNameMapper::new(&config);
        let path = std::env::temp_dir().join(format!(
            "ferrule-segment-inventory-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed),
        ));
        let mut offset = 0;
        let slices = mapper
            .tensors()
            .filter(|t| t.partition == Qwen35TensorPartitionKind::Text)
            .map(|t| {
                let slice = CheckpointTensorSlice {
                    name: t.external_name.clone(),
                    role: TensorRole::Unknown,
                    path: path.clone(),
                    offset,
                    bytes: t.bytes(),
                    shape: t.shape.clone(),
                    dtype: CheckpointDType::from_safetensors_dtype(t.dtype.as_str()),
                };
                offset += slice.bytes;
                slice
            })
            .collect::<Vec<_>>();
        // Sparse identity backing only. The tests below never load its payload.
        std::fs::File::create_new(&path)
            .unwrap()
            .set_len(offset)
            .unwrap();
        let bound = StateDictBinder::new(&schema, &mapper)
            .bind_slices(slices)
            .unwrap();
        Self {
            path,
            resources: BoundDecoderResources::new(spec, Arc::new(bound)).unwrap(),
        }
    }
    fn options(&self) -> GenericDecoderOptions {
        GenericDecoderOptions::standard_cpu(
            self.resources.spec(),
            ModelFamily::Qwen35,
            WeightSource::Safetensors,
            16,
            32,
            2,
            3,
            4 << 30,
            ExecutionPrecisionPolicy::f32(),
        )
        .unwrap()
    }
    fn estimate(&self, plan: &LayerSegmentPlan) -> HybridSegmentMemoryEstimate {
        HybridSegmentMemoryEstimate::for_segment(&self.resources, &self.options(), plan).unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}
fn plan(range: std::ops::Range<usize>, embedding: bool, output: bool) -> LayerSegmentPlan {
    LayerSegmentPlan::new(24, range, embedding, output).unwrap()
}

#[test]
fn selection_preserves_logical_head_but_not_the_embedding_operation() {
    let f = Fixture::new();
    let last = plan(12..24, false, true);
    let inventory = HybridSegmentInventory::new(&f.resources, &last).unwrap();
    assert_eq!(inventory.layers().first().unwrap().index(), 12);
    assert_eq!(inventory.layers().last().unwrap().index(), 23);
    assert_eq!(inventory.schema().layer_range(), 12..24);
    let embedding = f
        .resources
        .state_dict()
        .parameters()
        .iter()
        .find(|p| p.role() == &TensorRole::TokenEmbedding)
        .unwrap();
    let head = inventory
        .parameters()
        .find(|p| p.role() == &TensorRole::OutputHead)
        .unwrap();
    assert_eq!(head.canonical_id(), embedding.canonical_id());
    assert!(head.shares_storage_with(embedding));
    for p in inventory.parameters() {
        match p.residency() {
            ParameterResidency::Layer { layer } => assert!((12..24).contains(layer)),
            ParameterResidency::Static => assert!(matches!(
                p.role(),
                TensorRole::OutputNorm | TensorRole::OutputHead
            )),
            other => panic!("unexpected residency: {other:?}"),
        }
    }
}

#[test]
fn model_units_match_strict_schema_and_endpoint_workspace_without_payload_io() {
    let f = Fixture::new();
    // A metadata ledger cannot rely on opening a checkpoint after binding.
    std::fs::remove_file(&f.path).unwrap();
    let first = f.estimate(&plan(0..12, true, false));
    let last = f.estimate(&plan(12..24, false, true));
    for e in [&first, &last] {
        assert_eq!(e.per_sequence_state_bytes, 10_321_920);
        assert_eq!(e.per_page_kv_bytes, 196_608);
        assert_eq!(e.host.rotary_bytes, 3 * 32 * 64 * 4);
        assert_eq!(
            e.host.weight_bytes().unwrap(),
            e.device.weight_bytes().unwrap()
        );
        let recurrent_peak = peak_oracle(2, 6144, 16, 2048);
        assert_eq!(
            e.device.workspace_bytes,
            e.host.workspace_bytes + recurrent_peak
        );
    }
    assert!(last.device.workspace_bytes > first.device.workspace_bytes * 10);
    assert_eq!(first.boundary_rows_bytes, 2 * 1024 * 4);
    assert_eq!(last.boundary_rows_bytes, 2 * 248_320 * 4);
    assert_eq!(last.host.workspace_bytes, (248_320 * 32 + 8 * 32) * 2 * 4);
    let bare = f.estimate(&plan(12..24, false, false));
    assert_eq!(
        last.device.weight_bytes().unwrap() - bare.device.weight_bytes().unwrap(),
        (248_320 * 1024 + 1024) * 8
    );
    assert_eq!(
        last.host.weight_bytes().unwrap() - bare.host.weight_bytes().unwrap(),
        (248_320 * 1024 + 1024) * 8
    );
    let no_embedding = f.estimate(&plan(0..12, false, false));
    assert_eq!(
        first.device.weight_bytes().unwrap() - no_embedding.device.weight_bytes().unwrap(),
        248_320 * 1024 * 8
    );
}

#[test]
fn canonical_device_dedup_never_erases_coexisting_host_prepared_views() {
    let f = Fixture::new();
    let whole = f.estimate(&plan(0..24, true, true));
    let output_only = f.estimate(&plan(0..24, false, true));
    let endpoint = 248_320 * 1024 * 4;
    // Two logical views share a canonical source in this owner, but their host
    // PreparedLinear representations coexist. Device storage remains deduped.
    assert_eq!(whole.device, output_only.device);
    assert_eq!(whole.host.parameter_bytes, output_only.host.parameter_bytes);
    assert_eq!(
        whole.host.binding_bytes - output_only.host.binding_bytes,
        endpoint
    );
    assert_eq!(
        whole.host.weight_bytes().unwrap() - whole.device.weight_bytes().unwrap(),
        endpoint
    );
    let first = f.estimate(&plan(0..12, true, false));
    let last = f.estimate(&plan(12..24, false, true));
    assert_eq!(
        first.device.weight_bytes().unwrap() + last.device.weight_bytes().unwrap()
            - whole.device.weight_bytes().unwrap(),
        2 * endpoint
    );
    assert_eq!(
        first.host.weight_bytes().unwrap() + last.host.weight_bytes().unwrap()
            - whole.host.weight_bytes().unwrap(),
        endpoint
    );
    assert_eq!(
        first.per_page_kv_bytes + last.per_page_kv_bytes,
        whole.per_page_kv_bytes
    );
    assert_eq!(
        first.per_sequence_state_bytes + last.per_sequence_state_bytes,
        whole.per_sequence_state_bytes
    );
}

#[test]
fn bf16_startup_covers_two_f32_conversion_buffers_and_exact_reader_limit() {
    let f = Fixture::new();
    let p = plan(0..12, true, false);
    let e = f.estimate(&p);
    let largest = 248_320 * 1024;
    assert_eq!(e.host_startup_temporary_bytes, largest * 4 * 2);
    let options = |limit| {
        GenericDecoderOptions::standard_cpu(
            f.resources.spec(),
            ModelFamily::Qwen35,
            WeightSource::Safetensors,
            16,
            32,
            2,
            3,
            limit,
            ExecutionPrecisionPolicy::f32(),
        )
        .unwrap()
    };
    assert!(
        HybridSegmentMemoryEstimate::for_segment(&f.resources, &options((largest * 2) as u64), &p)
            .is_ok()
    );
    assert!(
        HybridSegmentMemoryEstimate::for_segment(
            &f.resources,
            &options((largest * 2 - 1) as u64),
            &p
        )
        .is_err()
    );
}

#[test]
fn inventory_rejects_mismatched_depth_recurrent_only_prefix_and_numeric_options() {
    let f = Fixture::new();
    let options = f.options();
    for p in [
        LayerSegmentPlan::new(25, 0..12, true, false).unwrap(),
        plan(0..3, true, false),
    ] {
        assert!(HybridSegmentMemoryEstimate::for_segment(&f.resources, &options, &p).is_err());
    }
    let p = plan(0..12, true, false);
    assert!(
        HybridSegmentMemoryEstimate::for_segment(
            &f.resources,
            &options.clone().with_active_layers(12).unwrap(),
            &p
        )
        .is_err()
    );
    let numeric = options
        .with_hybrid_cuda_numeric_fp8(
            crate::transformer::ExpertCacheLimits {
                max_experts: 1,
                max_bytes: 4096,
            },
            1024,
        )
        .unwrap();
    assert!(HybridSegmentMemoryEstimate::for_segment(&f.resources, &numeric, &p).is_err());
    assert!(add(usize::MAX, 1).is_err());
    assert!(mul(usize::MAX, 4).is_err());
    let mut memory = f.estimate(&p).host;
    memory.parameter_bytes = usize::MAX;
    assert!(memory.weight_bytes().is_err());
}

#[test]
fn cuda_abi_plan_projects_the_same_ledger_without_a_device() {
    use crate::decoder::HybridMemoryPlan;
    let f = Fixture::new();
    for p in [
        plan(0..12, true, false),
        plan(12..24, false, true),
        plan(0..24, true, true),
    ] {
        let ledger =
            HybridSegmentMemoryEstimate::for_cuda_segment(&f.resources, &f.options(), &p).unwrap();
        assert_eq!(ledger, f.estimate(&p));
        let e = HybridMemoryPlan::for_segment(&f.resources, &f.options(), &p, 6).unwrap();
        assert_eq!(
            e.weight_bytes_upper_bound,
            ledger.device.weight_bytes().unwrap()
        );
        assert_eq!(e.workspace_bytes_upper_bound, ledger.device.workspace_bytes);
        assert_eq!(e.per_sequence_state_bytes, ledger.per_sequence_state_bytes);
        assert_eq!(e.kv_bytes, 6 * ledger.per_page_kv_bytes);
        #[cfg(feature = "cuda")]
        assert_eq!(
            crate::decoder::HybridCudaMemoryEstimate::for_segment(
                &f.resources,
                &f.options(),
                &p,
                6
            )
            .unwrap(),
            e.into(),
        );
        assert!(HybridMemoryPlan::for_segment(&f.resources, &f.options(), &p, 0).is_err());
        assert!(HybridMemoryPlan::for_segment(&f.resources, &f.options(), &p, usize::MAX).is_err());
    }
}

#[test]
fn recurrent_small_growth_charges_sixteen_grants_not_640_logical_bytes() {
    let old = RecurrentScratchPlan::new(1, 10, 2, 4).unwrap();
    let new = RecurrentScratchPlan::new(2, 10, 2, 4).unwrap();
    assert_eq!(2 * new.elements.iter().sum::<usize>() * 4, 640);
    assert_eq!(old.growth_peak_bytes().unwrap(), 4096);
    assert_eq!(new.growth_peak_bytes().unwrap(), 4096);
    for (rows, channels, heads, width) in [
        (1, 262_145, 1, 16),
        (3, 1025, 63, 65),
        (2, 4_194_305, 16, 512),
    ] {
        let plan = RecurrentScratchPlan::new(rows, channels, heads, width).unwrap();
        assert_eq!(
            plan.growth_peak_bytes().unwrap(),
            peak_oracle(rows, channels, heads, width)
        );
    }
}
