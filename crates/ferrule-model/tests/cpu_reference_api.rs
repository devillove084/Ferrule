use ferrule_common::Result;
use ferrule_model::{
    CpuReferenceExpertExecutor, ExpertExecutor, ExpertLinearPayload, reference_linear,
};

#[test]
fn reference_exports_are_available_to_external_tests() {
    fn executor<T: ExpertExecutor + Default>() {}
    executor::<CpuReferenceExpertExecutor>();
    let _linear: fn(&ExpertLinearPayload, &[f32]) -> Result<Vec<f32>> = reference_linear;
    let _ = ferrule_model::moe::executor::CpuReferenceExpertExecutor::default();
}
