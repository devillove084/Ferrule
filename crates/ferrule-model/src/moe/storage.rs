//! Storage adapter for expert sources: transport selection and bounded reads.
//!
//! Completed reads cross the semantic boundary as artifact payloads. Pinned
//! reads retain the existing plan/ticket/poll boundary and source snapshots;
//! the io_uring reader remains the sole owner of physical operations and slabs.
//! Encoding validation and device publication are not performed here.

use std::sync::Arc;

use super::source::{
    ExpertArtifactPayload, ExpertId, ExpertLoadSource, ExpertTensorPayload, ExpertTensorSlice,
    checkpoint_read_plan_for_expert_source, slices_for_load_source,
};
use crate::checkpoint::CheckpointPositionedReader;
#[cfg(target_os = "linux")]
use crate::moe::io_uring_reader;
use crate::runner::ModelCompletionReactor;
use ferrule_common::{CompletionHub, Error, Result, StaleReason};
use rayon::prelude::*;
use snafu::Snafu;

#[cfg(target_os = "linux")]
#[path = "storage/io_plan.rs"]
mod io_plan;
#[cfg(target_os = "linux")]
#[allow(unused_imports)] // Host-only plans also remain available in CPU builds.
pub(crate) use io_plan::ExpertIoPlan;

#[cfg(all(target_os = "linux", feature = "cuda"))]
#[path = "storage/cuda_pinned.rs"]
mod cuda_pinned;
#[cfg(all(target_os = "linux", feature = "cuda"))]
pub(crate) use cuda_pinned::{
    PinnedExpertArtifactPayload, PinnedExpertLoadPlan, PinnedExpertReadPoll,
    PinnedExpertReadTicket, PinnedExpertTensorPayload, ReservedPinnedExpertLoad,
};

#[cfg(all(feature = "cuda-test-support", target_os = "linux"))]
#[path = "cuda_reader_test_support.rs"]
mod test_support;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExpertIoTransport {
    Positioned,
    BufferedIoUring,
    DirectIoUring,
}

impl ExpertIoTransport {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Positioned => "positioned",
            Self::BufferedIoUring => "buffered_io_uring",
            Self::DirectIoUring => "direct_io_uring",
        }
    }

    fn from_env() -> std::result::Result<Self, ExpertIoTransportError> {
        let value = std::env::var("FERRULE_EXPERT_IO_TRANSPORT")
            .unwrap_or_else(|_| default_expert_io_transport().as_str().to_owned());
        match value.to_ascii_lowercase().as_str() {
            "positioned" => Ok(Self::Positioned),
            "buffered_io_uring" => Ok(Self::BufferedIoUring),
            "direct_io_uring" => Ok(Self::DirectIoUring),
            _ => Err(ExpertIoTransportError::Unsupported { value }),
        }
    }

    const fn is_io_uring(self) -> bool {
        matches!(self, Self::BufferedIoUring | Self::DirectIoUring)
    }
}

#[derive(Debug, Snafu, PartialEq, Eq)]
pub enum ExpertIoTransportError {
    #[snafu(display(
        "unsupported FERRULE_EXPERT_IO_TRANSPORT '{value}'; expected positioned, buffered_io_uring, or direct_io_uring"
    ))]
    Unsupported { value: String },

    #[snafu(display(
        "io_uring expert streaming requires buffered_io_uring or direct_io_uring, got {}",
        transport.as_str()
    ))]
    IoUringReaderRequiresIoUring { transport: ExpertIoTransport },
}

fn transport_error(source: ExpertIoTransportError) -> Error {
    Error::ModelSource {
        source: Box::new(source),
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExpertIoStats {
    pub submitted_extents: u64,
    pub completed_extents: u64,
    pub failed_extents: u64,
    pub requested_bytes: u64,
    pub aligned_bytes: u64,
    pub coalesced_slices: u64,
    pub fixed_file_registrations: u64,
    pub slab_exhaustions: u64,
    pub peak_queue_depth: usize,
    pub read_us: u64,
}

#[derive(Clone)]
pub struct ExpertStreamingReader {
    max_slice_bytes: u64,
    transport: ExpertIoTransport,
    completion_hub: CompletionHub,
    #[cfg(target_os = "linux")]
    io_uring: Option<Arc<io_uring_reader::IoUringExpertReader>>,
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    pinned_source_files: cuda_pinned::PinnedSourceFiles,
}

impl std::fmt::Debug for ExpertStreamingReader {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExpertStreamingReader")
            .field("max_slice_bytes", &self.max_slice_bytes)
            .field("backend", &self.backend_name())
            .finish()
    }
}

impl ExpertStreamingReader {
    pub fn new(max_slice_bytes: u64) -> Self {
        Self::new_with_completion_hub(max_slice_bytes, CompletionHub::new())
    }

    pub(crate) fn new_with_completion_hub(
        max_slice_bytes: u64,
        completion_hub: CompletionHub,
    ) -> Self {
        Self {
            max_slice_bytes,
            transport: ExpertIoTransport::Positioned,
            completion_hub,
            #[cfg(target_os = "linux")]
            io_uring: None,
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            pinned_source_files: cuda_pinned::PinnedSourceFiles::default(),
        }
    }

    pub fn from_env(max_slice_bytes: u64) -> Result<Self> {
        Self::from_env_with_completion_hub(max_slice_bytes, CompletionHub::new())
    }

    pub(crate) fn from_env_with_completion_hub(
        max_slice_bytes: u64,
        completion_hub: CompletionHub,
    ) -> Result<Self> {
        let transport = ExpertIoTransport::from_env().map_err(transport_error)?;
        if transport == ExpertIoTransport::Positioned {
            return Ok(Self::new_with_completion_hub(
                max_slice_bytes,
                completion_hub,
            ));
        }
        let queue_depth = parse_expert_io_usize("FERRULE_EXPERT_IO_QUEUE_DEPTH", 2)?;
        let buffer_mib = parse_expert_io_usize("FERRULE_EXPERT_IO_BUFFER_MIB", 16)?;
        let buffer_bytes = buffer_mib
            .checked_mul(1024 * 1024)
            .ok_or_else(|| Error::Model {
                message: "FERRULE_EXPERT_IO_BUFFER_MIB overflows usize".into(),
            })?;
        Self::with_transport_and_completion_hub(
            max_slice_bytes,
            queue_depth,
            buffer_bytes,
            transport,
            completion_hub,
        )
    }

    #[cfg(target_os = "linux")]
    pub fn with_transport(
        max_slice_bytes: u64,
        queue_depth: usize,
        buffer_bytes: usize,
        transport: ExpertIoTransport,
    ) -> Result<Self> {
        Self::with_transport_and_completion_hub(
            max_slice_bytes,
            queue_depth,
            buffer_bytes,
            transport,
            CompletionHub::new(),
        )
    }

    #[cfg(target_os = "linux")]
    fn with_transport_and_completion_hub(
        max_slice_bytes: u64,
        queue_depth: usize,
        buffer_bytes: usize,
        transport: ExpertIoTransport,
        completion_hub: CompletionHub,
    ) -> Result<Self> {
        validate_io_uring_transport(transport).map_err(transport_error)?;
        Ok(Self {
            max_slice_bytes,
            transport,
            completion_hub: completion_hub.clone(),
            io_uring: Some(Arc::new(io_uring_reader::IoUringExpertReader::new(
                queue_depth,
                buffer_bytes,
                transport,
                completion_hub,
            )?)),
            #[cfg(feature = "cuda")]
            pinned_source_files: cuda_pinned::PinnedSourceFiles::default(),
        })
    }

    #[cfg(not(target_os = "linux"))]
    pub fn with_transport(
        max_slice_bytes: u64,
        queue_depth: usize,
        buffer_bytes: usize,
        transport: ExpertIoTransport,
    ) -> Result<Self> {
        Self::with_transport_and_completion_hub(
            max_slice_bytes,
            queue_depth,
            buffer_bytes,
            transport,
            CompletionHub::new(),
        )
    }

    #[cfg(not(target_os = "linux"))]
    fn with_transport_and_completion_hub(
        _max_slice_bytes: u64,
        _queue_depth: usize,
        _buffer_bytes: usize,
        _transport: ExpertIoTransport,
        _completion_hub: CompletionHub,
    ) -> Result<Self> {
        Err(Error::Model {
            message: "io_uring expert streaming is supported only on Linux".into(),
        })
    }

    pub const fn transport(&self) -> ExpertIoTransport {
        self.transport
    }

    pub const fn backend_name(&self) -> &'static str {
        self.transport.as_str()
    }

    pub fn max_slice_bytes(&self) -> u64 {
        self.max_slice_bytes
    }

    pub(crate) fn completion_hub(&self) -> CompletionHub {
        self.completion_hub.clone()
    }

    /// Transfers physical reader completion reactors to the owning model runner.
    pub fn take_completion_reactors(&self) -> Vec<ModelCompletionReactor> {
        #[cfg(target_os = "linux")]
        if let Some(reader) = self.io_uring.as_ref()
            && let Some(reactor) = reader.take_completion_reactor()
        {
            return vec![reactor];
        }
        Vec::new()
    }

    pub fn io_stats(&self) -> ExpertIoStats {
        #[cfg(target_os = "linux")]
        if let Some(reader) = self.io_uring.as_ref() {
            return reader.stats();
        }
        ExpertIoStats::default()
    }

    pub fn validate_source_identity(
        &self,
        load_source: &ExpertLoadSource,
    ) -> std::result::Result<(), StaleReason> {
        load_source.validate_source_identity()
    }

    pub fn read_load_source(
        &self,
        expert: ExpertId,
        load_source: &ExpertLoadSource,
    ) -> Result<ExpertArtifactPayload> {
        if matches!(load_source, ExpertLoadSource::HfLocalTensorSet { .. }) {
            return self.read_checkpoint_load_source(expert, load_source);
        }
        self.require_current_source_identity(load_source)?;
        let slices = slices_for_load_source(expert, load_source)?;
        let tensors = self.read_slices_with_backend(&slices, false)?;
        self.require_current_source_identity(load_source)?;
        Ok(ExpertArtifactPayload { expert, tensors })
    }

    fn read_checkpoint_load_source(
        &self,
        expert: ExpertId,
        load_source: &ExpertLoadSource,
    ) -> Result<ExpertArtifactPayload> {
        let slices = slices_for_load_source(expert, load_source)?;
        self.validate_bounded_slices(&slices)?;
        let plan = checkpoint_read_plan_for_expert_source(expert, load_source)?;
        if slices.len() != plan.extents().len()
            || slices.iter().zip(plan.extents()).any(|(slice, extent)| {
                slice.path != extent.path()
                    || slice.offset != extent.offset()
                    || slice.bytes != extent.bytes()
            })
        {
            return Err(Error::Model {
                message: format!(
                    "expert source {}:{} does not match its checkpoint read plan",
                    expert.layer, expert.expert
                ),
            });
        }
        let payloads = CheckpointPositionedReader::new(self.max_slice_bytes).read(&plan)?;
        let tensors = slices
            .into_iter()
            .zip(payloads)
            .map(|(slice, bytes)| ExpertTensorPayload { slice, bytes })
            .collect();
        Ok(ExpertArtifactPayload { expert, tensors })
    }

    pub fn read_local_slice(&self, slice: &ExpertTensorSlice) -> Result<ExpertTensorPayload> {
        if slice.bytes > self.max_slice_bytes {
            return Err(Error::Model {
                message: format!(
                    "expert tensor slice exceeds bounded read size: {} > {} bytes",
                    slice.bytes, self.max_slice_bytes
                ),
            });
        }
        Self::read_local_slice_positioned(slice)
    }

    /// Read a single tensor slice using positioned read (pread) with no shared
    /// file cursor or virtual-memory mapping.
    fn read_local_slice_positioned(slice: &ExpertTensorSlice) -> Result<ExpertTensorPayload> {
        use std::os::unix::fs::FileExt;
        let file = std::fs::File::open(&slice.path).map_err(|e| Error::Model {
            message: format!("expert tensor slice open '{}': {e}", slice.path.display()),
        })?;
        let mut bytes = vec![0u8; slice.bytes as usize];
        file.read_exact_at(&mut bytes, slice.offset)
            .map_err(|e| Error::Model {
                message: format!(
                    "expert tensor slice read_at '{}': {e}",
                    slice.path.display()
                ),
            })?;
        Ok(ExpertTensorPayload {
            slice: slice.clone(),
            bytes,
        })
    }

    /// Read all tensor slices for one expert concurrently. The configured
    /// backend is attempted first, with positioned reads as its fallback.
    pub fn read_load_source_concurrent(
        &self,
        expert: ExpertId,
        load_source: &ExpertLoadSource,
    ) -> Result<ExpertArtifactPayload> {
        if matches!(load_source, ExpertLoadSource::HfLocalTensorSet { .. }) {
            return self.read_checkpoint_load_source(expert, load_source);
        }
        self.require_current_source_identity(load_source)?;
        let slices = slices_for_load_source(expert, load_source)?;
        let tensors = self.read_slices_with_backend(&slices, true)?;
        self.require_current_source_identity(load_source)?;
        Ok(ExpertArtifactPayload { expert, tensors })
    }

    fn require_current_source_identity(&self, load_source: &ExpertLoadSource) -> Result<()> {
        self.validate_source_identity(load_source)
            .map_err(stale_source_identity_error)
    }

    fn validate_bounded_slices(&self, slices: &[ExpertTensorSlice]) -> Result<()> {
        for slice in slices {
            if slice.bytes > self.max_slice_bytes {
                return Err(Error::Model {
                    message: format!(
                        "expert tensor slice exceeds bounded read size: {} > {} bytes",
                        slice.bytes, self.max_slice_bytes
                    ),
                });
            }
        }
        Ok(())
    }

    fn read_slices_positioned_unchecked(
        slices: &[ExpertTensorSlice],
        parallel: bool,
    ) -> Result<Vec<ExpertTensorPayload>> {
        if !parallel || slices.len() <= 1 {
            return slices
                .iter()
                .map(Self::read_local_slice_positioned)
                .collect();
        }

        let results = slices
            .par_iter()
            .map(Self::read_local_slice_positioned)
            .collect::<Vec<_>>();
        results.into_iter().collect()
    }

    fn read_slices_with_backend(
        &self,
        slices: &[ExpertTensorSlice],
        parallel_fallback: bool,
    ) -> Result<Vec<ExpertTensorPayload>> {
        self.validate_bounded_slices(slices)?;

        #[cfg(target_os = "linux")]
        if let Some(reader) = self.io_uring.as_ref() {
            match reader.read_slices(slices) {
                Ok(payloads) => return Ok(payloads),
                Err(error) => {
                    tracing::debug!(
                        error = %error,
                        slices = slices.len(),
                        "expert io_uring read failed; falling back to positioned reads"
                    );
                }
            }
        }

        Self::read_slices_positioned_unchecked(slices, parallel_fallback)
    }
}

fn stale_source_identity_error(reason: StaleReason) -> Error {
    Error::Execution {
        message: format!("Stale expert source identity: {reason:?}"),
    }
}

#[cfg(any(target_os = "linux", test))]
fn validate_io_uring_transport(
    transport: ExpertIoTransport,
) -> std::result::Result<(), ExpertIoTransportError> {
    if transport.is_io_uring() {
        return Ok(());
    }
    Err(ExpertIoTransportError::IoUringReaderRequiresIoUring { transport })
}

const fn default_expert_io_transport() -> ExpertIoTransport {
    #[cfg(target_os = "linux")]
    {
        ExpertIoTransport::BufferedIoUring
    }
    #[cfg(not(target_os = "linux"))]
    {
        ExpertIoTransport::Positioned
    }
}

fn parse_expert_io_usize(name: &str, default: usize) -> Result<usize> {
    let Some(value) = std::env::var(name).ok() else {
        return Ok(default);
    };
    let parsed = value.parse::<usize>().map_err(|error| Error::Model {
        message: format!("invalid {name}='{value}': {error}"),
    })?;
    if parsed == 0 {
        return Err(Error::Model {
            message: format!("{name} must be greater than zero"),
        });
    }
    Ok(parsed)
}

/// Read multiple experts concurrently. Each expert's slices are read in
/// parallel, and multiple experts are read in parallel too.
pub fn read_experts_concurrent(
    reader: &ExpertStreamingReader,
    loads: &[(ExpertId, ExpertLoadSource)],
) -> Result<Vec<ExpertArtifactPayload>> {
    if loads.is_empty() {
        return Ok(Vec::new());
    }
    // Fast path: single expert.
    if loads.len() == 1 {
        let (expert, source) = &loads[0];
        return Ok(vec![reader.read_load_source_concurrent(*expert, source)?]);
    }

    // Parallel path: use rayon scope (no runtime creation overhead).
    // The previous tokio-based code created a new runtime per call, which
    // added ~1ms overhead per layer per token.
    let results: Vec<Result<ExpertArtifactPayload>> = loads
        .par_iter()
        .map(|(expert, source)| {
            let reader = reader.clone();
            reader.read_load_source_concurrent(*expert, source)
        })
        .collect();
    results.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::super::source::{ExpertMatrixKind, ExpertTensorComponent, ExpertTensorKey};
    use super::super::tests::unique_temp_dir;
    use super::*;

    #[test]
    fn io_uring_reader_accepts_only_io_uring_transports() {
        validate_io_uring_transport(ExpertIoTransport::BufferedIoUring).unwrap();
        validate_io_uring_transport(ExpertIoTransport::DirectIoUring).unwrap();
        assert_eq!(
            validate_io_uring_transport(ExpertIoTransport::Positioned),
            Err(ExpertIoTransportError::IoUringReaderRequiresIoUring {
                transport: ExpertIoTransport::Positioned,
            })
        );
    }

    #[test]
    fn expert_io_transport_names_are_explicit_and_stable() {
        assert_eq!(ExpertIoTransport::Positioned.as_str(), "positioned");
        assert_eq!(
            ExpertIoTransport::BufferedIoUring.as_str(),
            "buffered_io_uring"
        );
        assert_eq!(ExpertIoTransport::DirectIoUring.as_str(), "direct_io_uring");
    }

    #[test]
    fn reader_rejects_slices_larger_than_bound() {
        let dir = unique_temp_dir("ferrule-expert-streaming-bound-test");
        std::fs::create_dir_all(&dir).unwrap();
        let shard = dir.join("slice.bin");
        std::fs::write(&shard, vec![0u8; 16]).unwrap();
        let slice = ExpertTensorSlice {
            key: ExpertTensorKey::new(0, 0, ExpertMatrixKind::Gate),
            component: ExpertTensorComponent::Weight,
            path: shard,
            offset: 0,
            bytes: 16,
            dtype: "I8".into(),
            shape: vec![16],
        };
        let err = ExpertStreamingReader::new(8)
            .read_local_slice(&slice)
            .unwrap_err();
        assert!(err.to_string().contains("bounded read size"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn positioned_reader_reads_the_requested_bytes() {
        let dir = unique_temp_dir("ferrule-expert-positioned-reader-test");
        std::fs::create_dir_all(&dir).unwrap();
        let shard = dir.join("slice.bin");
        std::fs::write(&shard, (0u8..16).collect::<Vec<_>>()).unwrap();
        let expert = ExpertId::new(0, 0);
        let source = ExpertLoadSource::LocalShard {
            path: shard,
            offset: 4,
            bytes: 6,
        };
        let reader = ExpertStreamingReader::new(16);
        let payload = reader.read_load_source(expert, &source).unwrap();

        assert_eq!(payload.expert, expert);
        assert_eq!(payload.tensors.len(), 1);
        assert_eq!(payload.tensors[0].bytes, vec![4, 5, 6, 7, 8, 9]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn positioned_payloads_preserve_descriptor_order_in_serial_and_parallel_reads() {
        let dir = unique_temp_dir("ferrule-expert-payload-order");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("slices.bin");
        std::fs::write(&path, (0u8..32).collect::<Vec<_>>()).unwrap();
        let expert = ExpertId::new(2, 7);
        let slices = [
            (ExpertMatrixKind::Down, 20, 3),
            (ExpertMatrixKind::Gate, 2, 4),
            (ExpertMatrixKind::Up, 12, 2),
        ]
        .map(|(matrix, offset, bytes)| ExpertTensorSlice {
            key: ExpertTensorKey { expert, matrix },
            component: ExpertTensorComponent::Weight,
            path: path.clone(),
            offset,
            bytes,
            dtype: "opaque".into(),
            shape: vec![bytes as usize],
        });
        let source = ExpertLoadSource::LocalTensorSet {
            tensors: slices.to_vec(),
        };
        let reader = ExpertStreamingReader::new(4);
        let serial = reader.read_load_source(expert, &source).unwrap();
        let parallel = reader.read_load_source_concurrent(expert, &source).unwrap();
        assert_eq!(serial, parallel);
        assert_eq!(serial.expert, expert);
        for (payload, descriptor) in serial.tensors.iter().zip(&slices) {
            assert_eq!(&payload.slice, descriptor);
            assert_eq!(
                payload.bytes,
                (descriptor.offset as u8..descriptor.end_offset() as u8).collect::<Vec<_>>()
            );
        }
        assert_eq!(
            read_experts_concurrent(&reader, &[(expert, source.clone()), (expert, source)])
                .unwrap(),
            vec![serial.clone(), serial]
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn storage_returns_raw_payload_without_semantic_encoding_validation() {
        use super::super::source::ExpertComputeBundle;

        let dir = unique_temp_dir("ferrule-expert-raw-payload");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("malformed-bf16.bin");
        std::fs::write(&path, [0x12, 0x34, 0x56]).unwrap();
        let expert = ExpertId::new(1, 3);
        let source = ExpertLoadSource::LocalTensorSet {
            tensors: [
                ExpertMatrixKind::Gate,
                ExpertMatrixKind::Up,
                ExpertMatrixKind::Down,
            ]
            .into_iter()
            .map(|matrix| ExpertTensorSlice {
                key: ExpertTensorKey { expert, matrix },
                component: ExpertTensorComponent::Weight,
                path: path.clone(),
                offset: 0,
                bytes: 3,
                dtype: "BF16".into(),
                shape: vec![1, 1],
            })
            .collect(),
        };
        let reader = ExpertStreamingReader::new(3);
        for payload in [
            reader.read_load_source(expert, &source).unwrap(),
            reader.read_load_source_concurrent(expert, &source).unwrap(),
        ] {
            assert!(
                payload
                    .tensors
                    .iter()
                    .all(|tensor| tensor.bytes == [0x12, 0x34, 0x56])
            );
            let error = ExpertComputeBundle::from_artifact_payload(payload).unwrap_err();
            assert!(matches!(error, Error::Model { .. }));
            assert!(error.to_string().contains("byte length mismatch"));
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn read_bound_is_checked_before_open_and_short_reads_remain_errors() {
        let dir = unique_temp_dir("ferrule-expert-read-errors");
        let path = dir.join("missing.bin");
        let expert = ExpertId::new(0, 0);
        let source = ExpertLoadSource::LocalShard {
            path: path.clone(),
            offset: 0,
            bytes: 4,
        };
        let reader = ExpertStreamingReader::new(3);
        for error in [
            reader.read_load_source(expert, &source).unwrap_err(),
            reader
                .read_load_source_concurrent(expert, &source)
                .unwrap_err(),
        ] {
            assert!(matches!(error, Error::Model { .. }));
            assert!(error.to_string().contains("bounded read size"));
        }
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&path, [1, 2]).unwrap();
        let error = ExpertStreamingReader::new(4)
            .read_load_source(expert, &source)
            .unwrap_err();
        assert!(matches!(error, Error::Model { .. }));
        assert!(error.to_string().contains("read_at"));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
