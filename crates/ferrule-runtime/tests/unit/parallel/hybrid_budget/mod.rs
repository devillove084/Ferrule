use super::*;
use crate::hybrid_fixture as fixture;

fn config() -> PipelineConfig {
    PipelineConfig {
        page_size: 2,
        max_pages: 12,
        max_positions: 32,
        max_batch_tokens: 8,
        session_capacity: 3,
        max_parameter_bytes: 1 << 20,
        precision: ExecutionPrecisionPolicy::f32(),
        max_ack_polls: 8,
    }
}
fn estimate(
    resources: &BoundDecoderResources,
    plan: &LayerSegmentPlan,
) -> HybridSegmentMemoryEstimate {
    let c = config();
    let options = GenericDecoderOptions::standard_cpu(
        resources.spec(),
        ModelFamily::Qwen35,
        WeightSource::Safetensors,
        c.page_size,
        c.max_positions,
        c.max_batch_tokens,
        c.session_capacity,
        c.max_parameter_bytes,
        c.precision,
    )
    .unwrap();
    HybridSegmentMemoryEstimate::for_segment(resources, &options, plan).unwrap()
}

#[test]
fn capacity_composition_uses_model_units_and_never_deduplicates_colocated_owners() {
    let fixture = fixture::Fixture::new();
    let resources = fixture.resources();
    let plans = [
        LayerSegmentPlan::new(4, 0..2, true, false).unwrap(),
        LayerSegmentPlan::new(4, 2..4, false, true).unwrap(),
    ];
    let estimates = plans
        .iter()
        .map(|p| estimate(&resources, p))
        .collect::<Vec<_>>();
    for backend in [ModelExecutionBackend::Cpu, ModelExecutionBackend::Cuda] {
        let budgets = estimates
            .iter()
            .map(|e| {
                let b =
                    HybridStageBudget::from_segment_estimate(config(), 7, e, backend, 17).unwrap();
                let memory = if backend == ModelExecutionBackend::Cpu {
                    e.host
                } else {
                    e.device
                };
                assert_eq!(b.weights, memory.weight_bytes().unwrap() as u64);
                assert_eq!(b.workspace, memory.workspace_bytes as u64);
                assert_eq!(b.state, e.per_sequence_state_bytes as u64 * 4);
                assert_eq!(b.kv, e.per_page_kv_bytes as u64 * 24);
                b
            })
            .collect::<Vec<_>>();
        let total = budgets.iter().map(|b| b.total().unwrap()).sum::<u64>();
        assert_eq!(
            HybridDeviceBudget::admit(&budgets, &[(7, total)]).unwrap()[0].required,
            total
        );
        assert!(HybridDeviceBudget::admit(&budgets, &[(7, total - 1)]).is_err());
        assert!(HybridDeviceBudget::admit(&budgets, &[(7, total), (7, total)]).is_err());
        let mut more = config();
        more.session_capacity += 2;
        more.max_pages += 5;
        let b =
            HybridStageBudget::from_segment_estimate(more, 7, &estimates[0], backend, 17).unwrap();
        assert_eq!(
            b.state - budgets[0].state,
            2 * estimates[0].per_sequence_state_bytes as u64
        );
        assert_eq!(
            b.kv - budgets[0].kv,
            10 * estimates[0].per_page_kv_bytes as u64
        );
        assert_eq!(b.weights, budgets[0].weights);
        assert_eq!(b.workspace, budgets[0].workspace);
    }
}

#[test]
fn limits_mismatch_overflow_and_non_f32_remain_fail_closed() {
    let fixture = fixture::Fixture::new();
    let resources = fixture.resources();
    let e = estimate(
        &resources,
        &LayerSegmentPlan::new(4, 0..2, true, false).unwrap(),
    );
    for changed in [
        PipelineConfig {
            page_size: 4,
            ..config()
        },
        PipelineConfig {
            max_positions: 16,
            ..config()
        },
        PipelineConfig {
            max_batch_tokens: 4,
            ..config()
        },
        PipelineConfig {
            precision: ExecutionPrecisionPolicy::bf16_compatibility(),
            ..config()
        },
    ] {
        assert!(
            HybridStageBudget::from_segment_estimate(changed, 0, &e, ModelExecutionBackend::Cpu, 0)
                .is_err()
        );
    }
    assert!(
        HybridStageBudget::from_segment_estimate(
            config(),
            0,
            &e,
            ModelExecutionBackend::Cpu,
            u64::MAX
        )
        .is_err()
    );
    assert!(
        stage_capacity(
            config(),
            usize::MAX,
            1,
            PhysicalKvLayout::hybrid(ModelExecutionBackend::Cpu)
        )
        .is_err()
    );
    assert!(
        stage_capacity(
            config(),
            1,
            usize::MAX,
            PhysicalKvLayout::hybrid(ModelExecutionBackend::Cpu)
        )
        .is_err()
    );
}
