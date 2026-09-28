//! Non-default test access to the existing provider; no synthetic physical state.
use super::*;
use ferrule_backend::cuda::operators::moe::CudaFailpoints;

impl CudaExpertMaterializationProvider {
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn new_for_test(
        placement: MaterializationPlacement,
        limits: MaterializationResourceLimits,
        sources: Arc<MaterializationSourceCatalog<ExpertLoadSource>>,
        reader: ExpertStreamingReader,
        expert_capacity: usize,
        layer_slot_capacities: &[(usize, usize)],
        consumer_compute: CudaComputeStreamAuthority,
        control: Box<dyn ExpertResidencyControl>,
    ) -> Result<Self> {
        let mut owner = CudaExpertMaterializationOwner::create(
            placement,
            limits,
            sources,
            reader,
            expert_capacity,
            layer_slot_capacities,
            consumer_compute,
        )?;
        owner.install_residency_control(control)?;
        Ok(owner
            .provider
            .take()
            .expect("fresh owner holds its provider"))
    }

    #[doc(hidden)]
    pub fn failpoints_for_test(&self) -> &CudaFailpoints {
        self.ops.failpoints()
    }
}
