//! Explicit bounded construction of the production CUDA-pinned io_uring reader.
use super::*;
use ferrule_backend::cuda::operators::moe::CudaPinnedHostAllocator;

impl ExpertStreamingReader {
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn with_cuda_pinned_for_test(
        max_slice_bytes: u64,
        queue_depth: usize,
        buffer_bytes: usize,
        slab_count: usize,
        transport: ExpertIoTransport,
        allocator: CudaPinnedHostAllocator,
        completion_hub: CompletionHub,
    ) -> Result<Self> {
        validate_io_uring_transport(transport).map_err(transport_error)?;
        let reader = io_uring_reader::IoUringExpertReader::new_cuda_pinned(
            queue_depth,
            buffer_bytes,
            slab_count,
            &allocator,
            transport,
            completion_hub.clone(),
        )?;
        Ok(Self {
            max_slice_bytes,
            transport,
            completion_hub,
            io_uring: Some(Arc::new(reader)),
            pinned_source_files: Arc::new(Mutex::new(HashMap::new())),
        })
    }
}
