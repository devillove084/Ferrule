use ferrule_model::ModelExecutionBackend;
use ferrule_model::execution::ExecutionPrecisionPolicy;
use ferrule_runtime::parallel::pipeline::{
    CustodyStrategy, PhysicalKvLayout, PhysicalSlotLimit, PipelineConfig,
};

fn config(max_pages: usize) -> PipelineConfig {
    PipelineConfig {
        page_size: 1,
        max_pages,
        max_positions: 8,
        max_batch_tokens: 2,
        session_capacity: 1,
        max_parameter_bytes: 1,
        precision: ExecutionPrecisionPolicy::f32(),
        max_ack_polls: 1,
    }
}

#[test]
fn cpu_standard_is_in_place_l_and_cpu_hybrid_is_shadowed_2l() {
    let standard = PhysicalKvLayout::standard(ModelExecutionBackend::Cpu);
    let hybrid = PhysicalKvLayout::hybrid(ModelExecutionBackend::Cpu);
    assert_eq!(
        standard,
        PhysicalKvLayout::new(CustodyStrategy::InPlace, PhysicalSlotLimit::U32)
    );
    assert_eq!(
        hybrid,
        PhysicalKvLayout::new(CustodyStrategy::Shadowed, PhysicalSlotLimit::U32)
    );
    assert_eq!(config(7).physical_pages(standard).unwrap(), 7);
    assert_eq!(config(7).physical_pages(hybrid).unwrap(), 14);
}

#[test]
fn layout_and_slot_limit_are_independent() {
    let in_place_i32 = PhysicalKvLayout::new(CustodyStrategy::InPlace, PhysicalSlotLimit::I32);
    let shadowed_u32 = PhysicalKvLayout::new(CustodyStrategy::Shadowed, PhysicalSlotLimit::U32);
    assert_eq!(config(7).physical_pages(in_place_i32).unwrap(), 7);
    assert_eq!(config(7).physical_pages(shadowed_u32).unwrap(), 14);
    assert!(
        config(i32::MAX as usize / 2 + 2)
            .physical_pages(shadowed_u32)
            .is_ok()
    );
    assert!(
        config(i32::MAX as usize / 2 + 2)
            .physical_pages(PhysicalKvLayout::new(
                CustodyStrategy::Shadowed,
                PhysicalSlotLimit::I32
            ))
            .is_err()
    );
}

#[test]
fn overflow_and_pool_mismatch_are_rejected() {
    let layout = PhysicalKvLayout::new(CustodyStrategy::Shadowed, PhysicalSlotLimit::U32);
    assert!(config(usize::MAX).physical_pages(layout).is_err());
    assert!(layout.validate_pages(8, 8).is_err());
    assert!(layout.validate_pages(8, 16).is_ok());
}

#[test]
fn cuda_shape_is_available_without_cuda_feature_or_device() {
    let standard = PhysicalKvLayout::standard(ModelExecutionBackend::Cuda);
    let hybrid = PhysicalKvLayout::hybrid(ModelExecutionBackend::Cuda);
    assert_eq!(standard, hybrid);
    assert_eq!(standard.custody, CustodyStrategy::Shadowed);
    assert_eq!(standard.slot_limit, PhysicalSlotLimit::I32);
    assert_eq!(config(7).physical_pages(standard).unwrap(), 14);
}

#[test]
fn exact_index_limits_and_multiplication_overflow() {
    for limit in [PhysicalSlotLimit::U32, PhysicalSlotLimit::I32] {
        let in_place = PhysicalKvLayout::new(CustodyStrategy::InPlace, limit);
        let shadowed = PhysicalKvLayout::new(CustodyStrategy::Shadowed, limit);
        let max = limit.max_slots();
        assert_eq!(in_place.physical_pages(max).unwrap(), max);
        assert_eq!(shadowed.physical_pages(max / 2).unwrap(), max / 2 * 2);
        assert!(shadowed.physical_pages(max / 2 + 1).is_err());
        assert!(shadowed.physical_pages(usize::MAX).is_err());
        assert!(in_place.physical_pages(0).is_err());
        if let Some(over) = max.checked_add(1) {
            assert!(in_place.physical_pages(over).is_err());
        }
    }
    let mut c = config(2);
    c.page_size = usize::MAX;
    assert!(
        c.physical_pages(PhysicalKvLayout::standard(ModelExecutionBackend::Cpu))
            .is_err()
    );
}

mod pool;
#[cfg(unix)]
mod wire;
