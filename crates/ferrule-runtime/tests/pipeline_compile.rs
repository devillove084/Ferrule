//! Feature-matrix API checks: no device initialization, even with CUDA enabled.
use ferrule_model::decoder::{
    CpuKvView, CpuPagedKvBackend, DecoderKvCommitBackend, GenericDecoderSequenceState,
};
use ferrule_runtime::parallel::pipeline::{
    CpuPipelineStageProgram, PipelineStageProgram, PipelineStageWorker,
};

fn compatible<B, P>()
where
    B: DecoderKvCommitBackend<SequenceState = GenericDecoderSequenceState> + 'static,
    P: PipelineStageProgram<KvView = B::KvView>,
{
    // No Send bound on program/backend/view: only the factory crosses owners.
    let _ = std::mem::size_of::<PipelineStageWorker<B, P>>();
}
fn view<P: PipelineStageProgram<KvView = CpuKvView>>() {}

#[test]
fn cpu_program_uses_the_shared_owner() {
    view::<CpuPipelineStageProgram>();
    compatible::<CpuPagedKvBackend, CpuPipelineStageProgram>();
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_program_has_a_real_cuda_view_and_uses_the_shared_owner() {
    use ferrule_model::decoder::{CudaKvView, CudaPagedKvBackend};
    use ferrule_runtime::parallel::pipeline::CudaPipelineStageProgram;
    fn cuda_view<P: PipelineStageProgram<KvView = CudaKvView>>() {}
    cuda_view::<CudaPipelineStageProgram>();
    compatible::<CudaPagedKvBackend, CudaPipelineStageProgram>();
    compatible::<CudaPagedKvBackend, Box<dyn PipelineStageProgram<KvView = CudaKvView>>>();
}
