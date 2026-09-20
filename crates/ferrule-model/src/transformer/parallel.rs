//! Rank-local dense BF16/F32 linear and SwiGLU shards, independent of collectives.
//!
//! Weights are row-major `[out_features, in_features]`; activations are row-major
//! `[rows, features]`. There is no bias. Column parallelism partitions output
//! features (weight rows), while row parallelism partitions input features
//! (weight columns). Rank numbering is local to a TP group, not a device ordinal.
//! This module only prepares local shards and Column transport layouts. Runtime
//! HostCollective owns all collective execution, including Row output allreduce.

use std::ops::Range;

use ferrule_common::{Error, ParallelRankId, Result};

use crate::checkpoint::tensor::{CheckpointMatrixPayload, CheckpointMatrixRead};
use crate::checkpoint::{
    CheckpointDType, CheckpointSourceFileIdentity, CheckpointTensorReader, CheckpointTensorSlice,
};
use ferrule_backend::cpu::{
    CpuExecutionPrecision, CpuOperatorProvider, HostRows, LinearRef, LinearWeight,
    NativeCpuProvider, RowsDType, RowsShape, SwiGluRef,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorParallelLinearPartition {
    Column,
    Row,
}

/// Validated, reusable geometry. Ragged partitions give one extra feature to
/// each of the first `split_dimension % ranks` ranks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorParallelLinearPlan {
    out_features: usize,
    in_features: usize,
    ranks: usize,
    partition: TensorParallelLinearPartition,
}

impl TensorParallelLinearPlan {
    pub fn new(
        out_features: usize,
        in_features: usize,
        ranks: usize,
        partition: TensorParallelLinearPartition,
    ) -> Result<Self> {
        checked_elements(out_features, in_features)?;
        let split_dimension = match partition {
            TensorParallelLinearPartition::Column => out_features,
            TensorParallelLinearPartition::Row => in_features,
        };
        if ranks == 0 || ranks > split_dimension || u32::try_from(ranks).is_err() {
            return Err(invalid(format!(
                "ranks must be positive, fit u32, and not exceed split dimension {split_dimension}; got {ranks}"
            )));
        }
        Ok(Self {
            out_features,
            in_features,
            ranks,
            partition,
        })
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn ranks(&self) -> usize {
        self.ranks
    }

    pub fn partition(&self) -> TensorParallelLinearPartition {
        self.partition
    }

    /// Global feature range: output features for Column, input features for Row.
    pub fn rank_range(&self, rank: ParallelRankId) -> Result<Range<usize>> {
        let rank = rank.get() as usize;
        if rank >= self.ranks {
            return Err(invalid(format!(
                "rank {rank} is outside TP group of size {}",
                self.ranks
            )));
        }
        let dimension = match self.partition {
            TensorParallelLinearPartition::Column => self.out_features,
            TensorParallelLinearPartition::Row => self.in_features,
        };
        let base = dimension / self.ranks;
        let remainder = dimension % self.ranks;
        let start = rank
            .checked_mul(base)
            .and_then(|start| start.checked_add(rank.min(remainder)))
            .ok_or_else(|| invalid("partition offset overflow"))?;
        let end = start
            .checked_add(base)
            .and_then(|end| end.checked_add(usize::from(rank < remainder)))
            .ok_or_else(|| invalid("partition end overflow"))?;
        Ok(start..end)
    }

    /// Local weight shape `(out_features, in_features)`.
    pub fn local_shape(&self, rank: ParallelRankId) -> Result<(usize, usize)> {
        let width = self.rank_range(rank)?.len();
        Ok(match self.partition {
            TensorParallelLinearPartition::Column => (width, self.in_features),
            TensorParallelLinearPartition::Row => (self.out_features, width),
        })
    }

    /// Host-only preflight for the current CUDA artifact path. BF16 uses native
    /// bytes with K16 input alignment; physical and logical shapes are identical.
    /// No padding or dtype fallback is performed. CPU shards remain unrestricted.
    pub fn validate_cuda(&self, rank: ParallelRankId, dtype: &CheckpointDType) -> Result<()> {
        dense_element_bytes(dtype)?;
        let (out, width) = self.local_shape(rank)?;
        let elements = checked_elements(out, width)?;
        u32::try_from(elements).map_err(|_| invalid("CUDA element indices exceed u32"))?;
        if *dtype == CheckpointDType::Bf16 && !width.is_multiple_of(16) {
            return Err(invalid(format!(
                "CUDA BF16 artifact requires K16 (local in_features % 16 == 0); rank={} partition={:?} local_shape=[{out}, {width}]; no physical padding or F32 fallback",
                rank.get(),
                self.partition,
            )));
        }
        Ok(())
    }

    /// Check the *full* weight length, then pack a contiguous local row-major
    /// weight and return its global feature range. No device pointers are shared.
    pub fn shard_weight(
        &self,
        rank: ParallelRankId,
        full_weight: &[f32],
    ) -> Result<(Vec<f32>, Range<usize>)> {
        let range = self.rank_range(rank)?;
        check_len(
            "full weight",
            full_weight.len(),
            checked_elements(self.out_features, self.in_features)?,
        )?;
        let weight = match self.partition {
            TensorParallelLinearPartition::Column => {
                full_weight[range.start * self.in_features..range.end * self.in_features].to_vec()
            }
            TensorParallelLinearPartition::Row => {
                let mut local =
                    Vec::with_capacity(checked_elements(self.out_features, range.len())?);
                for row in full_weight.chunks_exact(self.in_features) {
                    local.extend_from_slice(&row[range.clone()]);
                }
                local
            }
        };
        Ok((weight, range))
    }

    /// Pack `[rows, local_in]` from a checked full `[rows, in_features]` input.
    /// Column shards keep all inputs; Row shards slice each input row.
    pub fn shard_input(
        &self,
        rank: ParallelRankId,
        input_rows: &[f32],
        rows: usize,
    ) -> Result<Vec<f32>> {
        let range = self.rank_range(rank)?;
        check_len(
            "full input",
            input_rows.len(),
            checked_elements(rows, self.in_features)?,
        )?;
        checked_elements(rows, self.out_features)?;
        match self.partition {
            TensorParallelLinearPartition::Column => Ok(input_rows.to_vec()),
            TensorParallelLinearPartition::Row => {
                let mut local = Vec::with_capacity(checked_elements(rows, range.len())?);
                for row in input_rows.chunks_exact(self.in_features) {
                    local.extend_from_slice(&row[range.clone()]);
                }
                Ok(local)
            }
        }
    }

    /// Fixed per-rank output width used by the Column transport layout.
    ///
    /// Every rank sends the same number of F32 elements per row. Ragged local
    /// output rows are zero-padded to `ceil(out_features / ranks)`, allowing a
    /// runtime collective descriptor to use one count for every rank.
    pub fn column_gather_width(&self) -> Result<usize> {
        self.require_column_partition()?;
        Ok(self.out_features.div_ceil(self.ranks))
    }

    /// Number of F32 elements sent by each rank for `rows` Column output rows.
    /// Checks both the per-rank block and the complete padded gather byte size.
    /// Rejects Row plans and zero rows.
    pub fn column_gather_count(&self, rows: usize) -> Result<usize> {
        let count = checked_elements(rows, self.column_gather_width()?)?;
        checked_elements(self.ranks, count)?;
        Ok(count)
    }

    /// Pack one rank's local Column output `[rows, local_out]` into its fixed
    /// `[rows, ceil(out_features / ranks)]` rank-major transport block.
    /// Padding is always zero and is not part of the logical output.
    pub fn pack_column_output(
        &self,
        rank: ParallelRankId,
        local_rows: &[f32],
        rows: usize,
    ) -> Result<Vec<f32>> {
        let width = self.column_gather_width()?;
        let range = self.rank_range(rank)?;
        check_len(
            "local Column output",
            local_rows.len(),
            checked_elements(rows, range.len())?,
        )?;
        let count = self.column_gather_count(rows)?;
        let mut packed = vec![0.0; count];
        for (target, source) in packed
            .chunks_exact_mut(width)
            .zip(local_rows.chunks_exact(range.len()))
        {
            target[..range.len()].copy_from_slice(source);
        }
        Ok(packed)
    }

    /// Unpack all rank-major padded Column blocks into logical row-major
    /// `[rows, out_features]`. The input must contain exactly
    /// `ranks * column_gather_count(rows)` elements in rank order.
    pub fn unpack_column_gather(
        &self,
        gathered_rank_major: &[f32],
        rows: usize,
    ) -> Result<Vec<f32>> {
        let width = self.column_gather_width()?;
        let per_rank = self.column_gather_count(rows)?;
        let expected = checked_elements(self.ranks, per_rank)?;
        check_len(
            "Column gathered output",
            gathered_rank_major.len(),
            expected,
        )?;
        let mut output = vec![0.0; checked_elements(rows, self.out_features)?];
        for (rank, block) in gathered_rank_major.chunks_exact(per_rank).enumerate() {
            let range = self.rank_range(ParallelRankId::new(rank as u32))?;
            for (target, source) in output
                .chunks_exact_mut(self.out_features)
                .zip(block.chunks_exact(width))
            {
                target[range.clone()].copy_from_slice(&source[..range.len()]);
            }
        }
        Ok(output)
    }

    fn require_column_partition(&self) -> Result<()> {
        if self.partition != TensorParallelLinearPartition::Column {
            return Err(invalid(
                "Column gather layout is unavailable for Row partition; Row output reduction belongs to HostCollective",
            ));
        }
        Ok(())
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Model {
        message: format!("tensor parallel linear: {}", message.into()),
    }
}

fn check_len(name: &str, actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        return Err(invalid(format!(
            "{name} length: expected {expected}, got {actual}"
        )));
    }
    Ok(())
}

/// Validate element arithmetic and Rust's maximum allocation/slice byte size.
fn checked_elements(rows: usize, features: usize) -> Result<usize> {
    if rows == 0 || features == 0 {
        return Err(invalid("dimensions and rows must be greater than zero"));
    }
    let elements = rows
        .checked_mul(features)
        .ok_or_else(|| invalid("shape element count overflow"))?;
    elements
        .checked_mul(std::mem::size_of::<f32>())
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or_else(|| invalid("shape byte size overflow"))?;
    Ok(elements)
}

#[cfg(feature = "cuda")]
pub use super::parallel_transfer::CudaShardError;
#[cfg(feature = "cuda")]
pub use cuda::{CudaLinearShard, CudaSwiGluShard};

#[cfg(feature = "cuda")]
mod cuda {
    use std::cell::RefCell;
    use std::rc::Rc;

    use crate::transformer::parallel_transfer::{CudaShardTransfer, DEFAULT_TRANSFER_CONFIG};
    use ferrule_backend::cuda::{CudaAsyncTransportStats, CudaTransferConfig};

    use ferrule_backend::cuda::operators::linear::{
        CudaArtifactLinearHandle, CudaArtifactLinearShape, CudaOperators,
    };

    use super::*;

    /// Reusable rank-local BF16/F32 artifact weight, bound to one existing owner.
    ///
    /// Private device state and an Rc owner make this !Send/!Sync. Execution
    /// accepts neither foreign operators nor device buffers: every kernel input,
    /// output and weight is created by this exact owner, not merely its ordinal.
    /// Only host partials should cross rank/worker boundaries. Activations use a
    /// fixed pinned pool: by default 64 KiB chunks, one TX and one RX slot (128 KiB
    /// total per shard), allocated once at construction. Execution host-waits;
    /// neither compute/communication overlap nor zero-copy is claimed.
    pub struct CudaLinearShard {
        // Fence transfers/compute before resident weights are dropped.
        transfer: RefCell<CudaShardTransfer>,
        weight: CudaArtifactLinearHandle,
        plan: TensorParallelLinearPlan,
        rank: ParallelRankId,
    }

    impl CudaLinearShard {
        /// Upload only this rank's weight using the caller's existing operators.
        /// Does not create a context or choose a device. H2D completes on return.
        pub fn new(
            ops: Rc<CudaOperators>,
            plan: TensorParallelLinearPlan,
            rank: ParallelRankId,
            full_weight: &[f32],
        ) -> Result<Self> {
            Self::new_with_transfer_config(ops, plan, rank, full_weight, DEFAULT_TRANSFER_CONFIG)
        }

        /// Full-weight constructor with an explicit fixed activation-transfer
        /// budget. The weight is still materialized by the existing path.
        pub fn new_with_transfer_config(
            ops: Rc<CudaOperators>,
            plan: TensorParallelLinearPlan,
            rank: ParallelRankId,
            full_weight: &[f32],
            config: CudaTransferConfig,
        ) -> Result<Self> {
            Self::from_local_with_transfer_config(
                ops,
                plan.shard_weight_f32(rank, full_weight)?,
                config,
            )
        }

        /// Upload only the already-local BF16/F32 payload; no full weight decode
        /// or second partition step. Source snapshots are checked before upload.
        pub fn from_local(
            ops: Rc<CudaOperators>,
            shard: TensorParallelWeightShard,
        ) -> Result<Self> {
            Self::from_local_with_transfer_config(ops, shard, DEFAULT_TRANSFER_CONFIG)
        }

        /// Explicit fixed pinned budget. Extra slots are allowed, but this
        /// synchronous adapter intentionally consumes one chunk at a time.
        /// Weight materialization remains independent of activation transport.
        pub fn from_local_with_transfer_config(
            ops: Rc<CudaOperators>,
            shard: TensorParallelWeightShard,
            config: CudaTransferConfig,
        ) -> Result<Self> {
            shard.validate_source_identity()?;
            shard.plan.validate_cuda(shard.rank, shard.dtype())?;
            let plan = shard.plan.clone();
            let rank = shard.rank;
            let transfer = RefCell::new(CudaShardTransfer::new(Rc::clone(&ops), config)?);
            let weight = upload_local(&ops, shard)?;
            Ok(Self {
                transfer,
                weight,
                plan,
                rank,
            })
        }

        pub fn transfer_config(&self) -> CudaTransferConfig {
            self.transfer.borrow().config()
        }

        /// Direct backend counter snapshot, not model-side accounting.
        pub fn transfer_stats(&self) -> CudaAsyncTransportStats {
            self.transfer.borrow().stats()
        }

        pub fn is_quiescent(&self) -> bool {
            self.transfer.borrow().is_quiescent()
        }

        pub fn needs_quarantine(&self) -> bool {
            self.transfer.borrow().needs_quarantine()
        }

        /// Drain all shard DMA (including control-stream D2H), then attempt both
        /// compute/upload fences. Worker panic and shutdown hooks must call this,
        /// not just synchronize operators' compute/upload streams. A poisoned
        /// transport still returns a typed error after a successful later drain;
        /// do not reuse or ACK that worker as an ordinary completed failure.
        pub fn quiesce(&self) -> Result<()> {
            self.transfer.borrow_mut().quiesce()
        }

        pub fn plan(&self) -> &TensorParallelLinearPlan {
            &self.plan
        }

        pub fn rank(&self) -> ParallelRankId {
            self.rank
        }

        /// Synchronous host boundary from full `[rows, in_features]` input.
        /// Column borrows the full input; Row packs its local columns once.
        /// Returns this rank's `[rows, local_out]` partial, never a collective.
        /// See `execute_local` for transfer custody and failure handling.
        pub fn execute(&self, input_rows: &[f32], rows: usize) -> Result<Vec<f32>> {
            self.transfer.borrow().ensure_ready()?;
            self.validate_rows(rows)?;
            check_len(
                "full input",
                input_rows.len(),
                checked_elements(rows, self.plan.in_features())?,
            )?;
            if self.plan.partition() == TensorParallelLinearPartition::Column {
                self.execute_local(input_rows, rows)
            } else {
                let local = self.plan.shard_input(self.rank, input_rows, rows)?;
                self.execute_local(&local, rows)
            }
        }

        /// Already packed `[rows, local_in]` input. Pinned H2D chunks enqueue
        /// compute waits; owner-local kernels produce a D2H producer event;
        /// exact-event polls return ordered host chunks, including the tail.
        /// Device buffers are per-call; fixed pinned slots are reused.
        ///
        /// Validation submits nothing. Every operation Err first drains all
        /// transfers and attempts compute/upload fences. Inspect `CudaShardError`
        /// via `from_error`, or `needs_quarantine()`, BEFORE erasing error types.
        /// Unknown custody stays retained, including after a panic, until
        /// `quiesce()` proves completion. Backend-internal kernel temporaries
        /// retain the allocator's existing event-retirement protection.
        pub fn execute_local(&self, input_rows: &[f32], rows: usize) -> Result<Vec<f32>> {
            let mut transfer = self.transfer.borrow_mut();
            transfer.ensure_ready()?;
            let (input_len, _) = self.validate_rows(rows)?;
            check_len("local input", input_rows.len(), input_len)?;
            transfer.execute(input_rows, |ops, input| {
                ops.artifact_linear_rows_from_device(&self.weight, input, rows)
            })
        }

        fn validate_rows(&self, rows: usize) -> Result<(usize, usize)> {
            let (out_features, in_features) = self.plan.local_shape(self.rank)?;
            Ok((
                cuda_elements(rows, in_features)?,
                cuda_elements(rows, out_features)?,
            ))
        }
    }

    /// Three resident rank-local linears on one owner. Execution returns only
    /// the down-projection partial; reduction belongs to runtime HostCollective.
    /// Native BF16 requires K16 for both hidden and local intermediate widths;
    /// construction rejects unsupported shapes before any weight upload. The
    /// artifact path rounds each linear's input to BF16 but returns F32 partials,
    /// unlike the CPU shard's F32-activation precision with BF16 weights.
    /// Activations use the same bounded pinned default as `CudaLinearShard`:
    /// 64 KiB chunks, one TX and one RX slot, synchronous host waiting.
    pub struct CudaSwiGluShard {
        transfer: RefCell<CudaShardTransfer>,
        plan: TensorParallelSwiGluPlan,
        rank: ParallelRankId,
        gate: CudaArtifactLinearHandle,
        up: CudaArtifactLinearHandle,
        down: CudaArtifactLinearHandle,
    }
    impl CudaSwiGluShard {
        pub fn new(
            ops: Rc<CudaOperators>,
            plan: TensorParallelSwiGluPlan,
            rank: ParallelRankId,
            gate: &[f32],
            up: &[f32],
            down: &[f32],
        ) -> Result<Self> {
            Self::new_with_transfer_config(ops, plan, rank, gate, up, down, DEFAULT_TRANSFER_CONFIG)
        }

        /// Full-weight constructor with an explicit fixed activation-transfer
        /// budget; gate/up/down materialization remains unchanged.
        pub fn new_with_transfer_config(
            ops: Rc<CudaOperators>,
            plan: TensorParallelSwiGluPlan,
            rank: ParallelRankId,
            gate: &[f32],
            up: &[f32],
            down: &[f32],
            config: CudaTransferConfig,
        ) -> Result<Self> {
            Self::from_local_with_transfer_config(
                ops,
                plan.shard_weights_f32(rank, gate, up, down)?,
                config,
            )
        }
        pub fn from_local(
            ops: Rc<CudaOperators>,
            weights: TensorParallelSwiGluWeights,
        ) -> Result<Self> {
            Self::from_local_with_transfer_config(ops, weights, DEFAULT_TRANSFER_CONFIG)
        }

        /// Set a fixed pinned activation budget; weight uploads remain separate.
        /// Additional slots do not imply concurrent compute/communication.
        pub fn from_local_with_transfer_config(
            ops: Rc<CudaOperators>,
            weights: TensorParallelSwiGluWeights,
            config: CudaTransferConfig,
        ) -> Result<Self> {
            weights.validate_source_identities()?;
            // Reject the entire bundle before the first H2D, including down's K.
            weights
                .plan
                .validate_cuda(weights.rank, weights.gate.dtype())?;
            let transfer = RefCell::new(CudaShardTransfer::new(Rc::clone(&ops), config)?);
            Ok(Self {
                transfer,
                plan: weights.plan,
                rank: weights.rank,
                gate: upload_local(&ops, weights.gate)?,
                up: upload_local(&ops, weights.up)?,
                down: upload_local(&ops, weights.down)?,
            })
        }

        pub fn transfer_config(&self) -> CudaTransferConfig {
            self.transfer.borrow().config()
        }

        /// Direct backend counter snapshot, not model-side accounting.
        pub fn transfer_stats(&self) -> CudaAsyncTransportStats {
            self.transfer.borrow().stats()
        }

        pub fn is_quiescent(&self) -> bool {
            self.transfer.borrow().is_quiescent()
        }

        pub fn needs_quarantine(&self) -> bool {
            self.transfer.borrow().needs_quarantine()
        }

        /// Same control-event drain / worker-hook contract as
        /// `CudaLinearShard::quiesce`; compute/upload sync alone is insufficient.
        pub fn quiesce(&self) -> Result<()> {
            self.transfer.borrow_mut().quiesce()
        }

        pub fn plan(&self) -> &TensorParallelSwiGluPlan {
            &self.plan
        }
        pub fn rank(&self) -> ParallelRankId {
            self.rank
        }

        /// Pinned-chunk H2D -> existing GPU-local SwiGLU kernels -> pinned-chunk
        /// D2H. The synchronous caller host-polls exact transfer events; there is
        /// no claimed compute/communication overlap. Gate/up/hidden stay on GPU.
        /// Errors drain all active transfers before returning, with the same
        /// typed quarantine contract as `CudaLinearShard::execute_local`.
        pub fn execute(&self, input_rows: &[f32], rows: usize) -> Result<Vec<f32>> {
            let mut transfer = self.transfer.borrow_mut();
            transfer.ensure_ready()?;
            let input_len = cuda_elements(rows, self.plan.hidden_size())?;
            cuda_elements(rows, self.plan.rank_range(self.rank)?.len())?;
            check_len("SwiGLU input", input_rows.len(), input_len)?;
            transfer.execute(input_rows, |ops, input| {
                ops.artifact_swiglu_ffn_rows_from_device(
                    &self.gate, &self.up, &self.down, input, rows, 1.0, 0.0,
                )
            })
        }
    }

    fn upload_local(
        ops: &CudaOperators,
        shard: TensorParallelWeightShard,
    ) -> Result<CudaArtifactLinearHandle> {
        shard.validate_source_identity()?;
        shard.plan.validate_cuda(shard.rank, shard.dtype())?;
        let [out_features, in_features] = shard.local_shape();
        cuda_elements(out_features, in_features)?;
        let shape = match shard.dtype() {
            CheckpointDType::F32 => CudaArtifactLinearShape::F32 {
                out_features,
                in_features,
            },
            CheckpointDType::Bf16 => CudaArtifactLinearShape::Bf16Bytes {
                out_features,
                in_features,
            },
            _ => return Err(invalid("unsupported CUDA local storage")),
        };
        let mut source = UploadSource {
            ops,
            shard: Some(shard),
            fenced: false,
        };
        let upload = ops.upload_artifact_linear(
            shape,
            source
                .shard
                .as_ref()
                .expect("upload source retained")
                .bytes(),
            &[],
        );
        let completion = ops.sync_stream();
        source.fenced = completion.is_ok();
        if completion.is_err() {
            source.leak();
        }
        match (upload, completion) {
            (Ok(handle), Ok(())) => Ok(handle),
            (Err(error), Ok(())) => Err(error),
            (upload, Err(cleanup)) => {
                // The host source was leaked above; keep a returned handle too
                // when materialization cannot prove its compute-stream fence.
                let error = match upload {
                    Ok(handle) => {
                        std::mem::forget(handle);
                        invalid("weight upload completion unknown")
                    }
                    Err(error) => error,
                };
                Err(CudaShardError::wrap(error, Some(cleanup), false, true))
            }
        }
    }

    /// Also protects the source if backend upload unwinds before returning.
    /// No host-thread exit or ordinary error is treated as a DMA completion.
    struct UploadSource<'a> {
        ops: &'a CudaOperators,
        shard: Option<TensorParallelWeightShard>,
        fenced: bool,
    }
    impl UploadSource<'_> {
        fn leak(&mut self) {
            if let Some(shard) = self.shard.take() {
                std::mem::forget(shard);
            }
        }
    }
    impl Drop for UploadSource<'_> {
        fn drop(&mut self) {
            if self.shard.is_some() && !self.fenced && self.ops.sync_stream().is_err() {
                self.leak();
            }
        }
    }

    // Artifact kernels and launch APIs use u32 element indices, including BF16.
    fn cuda_elements(rows: usize, features: usize) -> Result<usize> {
        let elements = checked_elements(rows, features)?;
        u32::try_from(elements).map_err(|_| invalid("CUDA element indices exceed u32"))?;
        Ok(elements)
    }
}

/// Exactly one rank's packed, native-endian-independent checkpoint bytes.
/// Shape/rank fields are private so an upload cannot silently accept a full
/// matrix, a shard from another rank, or quantized storage without its scales.
#[derive(Debug)]
pub struct TensorParallelWeightShard {
    plan: TensorParallelLinearPlan,
    rank: ParallelRankId,
    dtype: CheckpointDType,
    bytes: Vec<u8>,
    provenance: Option<CheckpointMatrixRead>,
}

impl TensorParallelWeightShard {
    /// Accept already local row-major BF16/F32 little-endian bytes without copying.
    pub fn from_local_bytes(
        plan: TensorParallelLinearPlan,
        rank: ParallelRankId,
        dtype: CheckpointDType,
        bytes: Vec<u8>,
    ) -> Result<Self> {
        let (out, width) = plan.local_shape(rank)?;
        let element_bytes = dense_element_bytes(&dtype)?;
        let expected = checked_elements(out, width)?
            .checked_mul(element_bytes)
            .ok_or_else(|| invalid("local weight byte size overflow"))?;
        check_len("local weight bytes", bytes.len(), expected)?;
        Ok(Self {
            plan,
            rank,
            dtype,
            bytes,
            provenance: None,
        })
    }

    pub fn from_checkpoint_payload(
        plan: TensorParallelLinearPlan,
        rank: ParallelRankId,
        payload: CheckpointMatrixPayload,
    ) -> Result<Self> {
        let read = payload.provenance();
        let (rows, columns) = plan.matrix_range(rank)?;
        if read.tensor().shape != [plan.out_features, plan.in_features]
            || read.rows() != rows
            || read.columns() != columns
        {
            return Err(invalid("checkpoint rectangle does not match TP plan/rank"));
        }
        let dtype = read.tensor().dtype.clone();
        let (read, bytes) = payload.into_parts();
        let mut shard = Self::from_local_bytes(plan, rank, dtype, bytes)?;
        shard.provenance = Some(read);
        shard.validate_source_identity()?;
        Ok(shard)
    }

    pub fn plan(&self) -> &TensorParallelLinearPlan {
        &self.plan
    }
    pub fn rank(&self) -> ParallelRankId {
        self.rank
    }
    pub fn dtype(&self) -> &CheckpointDType {
        &self.dtype
    }
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn full_shape(&self) -> [usize; 2] {
        [self.plan.out_features, self.plan.in_features]
    }
    pub fn local_shape(&self) -> [usize; 2] {
        let (out, width) = self
            .plan
            .local_shape(self.rank)
            .expect("validated local shard");
        [out, width]
    }
    pub fn provenance(&self) -> Option<&CheckpointMatrixRead> {
        self.provenance.as_ref()
    }
    pub fn validate_source_identity(&self) -> Result<()> {
        if let Some(read) = &self.provenance {
            read.read_plan()
                .validate_source_identity()
                .map_err(|e| invalid(format!("stale checkpoint shard source: {e:?}")))?;
        }
        Ok(())
    }
}

fn dense_element_bytes(dtype: &CheckpointDType) -> Result<usize> {
    match dtype {
        CheckpointDType::F32 => Ok(4),
        CheckpointDType::Bf16 => Ok(2),
        _ => Err(invalid(format!(
            "TP weight slicing supports only dense BF16/F32; {} requires an encoding/scale-aware plan",
            dtype.as_str()
        ))),
    }
}

impl TensorParallelLinearPlan {
    fn matrix_range(&self, rank: ParallelRankId) -> Result<(Range<usize>, Range<usize>)> {
        let range = self.rank_range(rank)?;
        Ok(match self.partition {
            TensorParallelLinearPartition::Column => (range, 0..self.in_features),
            TensorParallelLinearPartition::Row => (0..self.out_features, range),
        })
    }

    /// Capture an immediate source snapshot. Bound state dictionaries should use
    /// `read_weight_shard_with_source` with `BoundTensorPart::source_identity()`.
    pub fn read_weight_shard(
        &self,
        reader: &CheckpointTensorReader,
        tensor: &CheckpointTensorSlice,
        rank: ParallelRankId,
    ) -> Result<TensorParallelWeightShard> {
        self.validate_tensor(tensor, rank)?;
        let source = CheckpointSourceFileIdentity::capture(&tensor.path)?;
        self.read_weight_shard_with_source(reader, tensor, rank, &source)
    }

    pub fn read_weight_shard_with_source(
        &self,
        reader: &CheckpointTensorReader,
        tensor: &CheckpointTensorSlice,
        rank: ParallelRankId,
        source: &CheckpointSourceFileIdentity,
    ) -> Result<TensorParallelWeightShard> {
        self.validate_tensor(tensor, rank)?;
        let (rows, columns) = self.matrix_range(rank)?;
        let read = reader.plan_2d_range(tensor, rows, columns, source)?;
        TensorParallelWeightShard::from_checkpoint_payload(
            self.clone(),
            rank,
            reader.read_matrix(&read)?,
        )
    }

    fn validate_tensor(&self, tensor: &CheckpointTensorSlice, rank: ParallelRankId) -> Result<()> {
        self.rank_range(rank)?;
        dense_element_bytes(&tensor.dtype)?;
        if tensor.shape != [self.out_features, self.in_features] {
            return Err(invalid(format!(
                "checkpoint '{}' shape {:?} does not match full weight [{}, {}]",
                tensor.name, tensor.shape, self.out_features, self.in_features
            )));
        }
        Ok(())
    }

    /// In-memory/test entrypoint; CUDA and CPU consume the same local payload as
    /// checkpoint readers. Only the local matrix is copied/encoded.
    pub fn shard_weight_f32(
        &self,
        rank: ParallelRankId,
        full_weight: &[f32],
    ) -> Result<TensorParallelWeightShard> {
        let (values, _) = self.shard_weight(rank, full_weight)?;
        TensorParallelWeightShard::from_local_bytes(
            self.clone(),
            rank,
            CheckpointDType::F32,
            values.into_iter().flat_map(f32::to_le_bytes).collect(),
        )
    }
}

/// Bias-free TP MLP: gate/up partition the same intermediate channels, down
/// partitions those input channels. Each rank returns a full hidden-width
/// partial; the runtime performs exactly one sum, after the down projection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorParallelSwiGluPlan {
    gate_up: TensorParallelLinearPlan,
    down: TensorParallelLinearPlan,
}

impl TensorParallelSwiGluPlan {
    pub fn new(hidden_size: usize, intermediate_size: usize, ranks: usize) -> Result<Self> {
        Ok(Self {
            gate_up: TensorParallelLinearPlan::new(
                intermediate_size,
                hidden_size,
                ranks,
                TensorParallelLinearPartition::Column,
            )?,
            down: TensorParallelLinearPlan::new(
                hidden_size,
                intermediate_size,
                ranks,
                TensorParallelLinearPartition::Row,
            )?,
        })
    }
    pub fn hidden_size(&self) -> usize {
        self.gate_up.in_features()
    }
    pub fn intermediate_size(&self) -> usize {
        self.gate_up.out_features()
    }
    pub fn ranks(&self) -> usize {
        self.gate_up.ranks()
    }
    pub fn gate_up_plan(&self) -> &TensorParallelLinearPlan {
        &self.gate_up
    }
    pub fn down_plan(&self) -> &TensorParallelLinearPlan {
        &self.down
    }
    pub fn rank_range(&self, rank: ParallelRankId) -> Result<Range<usize>> {
        self.gate_up.rank_range(rank)
    }

    /// Check one rank without creating a CUDA owner or reading/uploading weights.
    /// Native BF16 requires hidden_size and this rank's intermediate width to be
    /// multiples of 16 (gate/up K and down K respectively). F32 permits ragged K.
    /// A full TP group is supported only when every rank passes this preflight.
    pub fn validate_cuda(&self, rank: ParallelRankId, dtype: &CheckpointDType) -> Result<()> {
        self.gate_up.validate_cuda(rank, dtype)?;
        self.down.validate_cuda(rank, dtype)
    }

    /// Tensor order is gate, up, down. Capture all snapshots before any payload
    /// read, then revalidate the whole bundle before returning it.
    pub fn read_weight_shards(
        &self,
        reader: &CheckpointTensorReader,
        tensors: [&CheckpointTensorSlice; 3],
        rank: ParallelRankId,
    ) -> Result<TensorParallelSwiGluWeights> {
        let sources = [
            CheckpointSourceFileIdentity::capture(&tensors[0].path)?,
            CheckpointSourceFileIdentity::capture(&tensors[1].path)?,
            CheckpointSourceFileIdentity::capture(&tensors[2].path)?,
        ];
        self.read_weight_shards_with_sources(reader, tensors, rank, sources.each_ref())
    }

    pub fn read_weight_shards_with_sources(
        &self,
        reader: &CheckpointTensorReader,
        tensors: [&CheckpointTensorSlice; 3],
        rank: ParallelRankId,
        sources: [&CheckpointSourceFileIdentity; 3],
    ) -> Result<TensorParallelSwiGluWeights> {
        self.gate_up.validate_tensor(tensors[0], rank)?;
        self.gate_up.validate_tensor(tensors[1], rank)?;
        self.down.validate_tensor(tensors[2], rank)?;
        if tensors[0].dtype != tensors[1].dtype || tensors[0].dtype != tensors[2].dtype {
            return Err(invalid("SwiGLU gate/up/down dtypes must match"));
        }
        TensorParallelSwiGluWeights::new(
            self.clone(),
            rank,
            self.gate_up
                .read_weight_shard_with_source(reader, tensors[0], rank, sources[0])?,
            self.gate_up
                .read_weight_shard_with_source(reader, tensors[1], rank, sources[1])?,
            self.down
                .read_weight_shard_with_source(reader, tensors[2], rank, sources[2])?,
        )
    }

    pub fn shard_weights_f32(
        &self,
        rank: ParallelRankId,
        gate: &[f32],
        up: &[f32],
        down: &[f32],
    ) -> Result<TensorParallelSwiGluWeights> {
        TensorParallelSwiGluWeights::new(
            self.clone(),
            rank,
            self.gate_up.shard_weight_f32(rank, gate)?,
            self.gate_up.shard_weight_f32(rank, up)?,
            self.down.shard_weight_f32(rank, down)?,
        )
    }
}

#[derive(Debug)]
pub struct TensorParallelSwiGluWeights {
    plan: TensorParallelSwiGluPlan,
    rank: ParallelRankId,
    gate: TensorParallelWeightShard,
    up: TensorParallelWeightShard,
    down: TensorParallelWeightShard,
}

impl TensorParallelSwiGluWeights {
    pub fn new(
        plan: TensorParallelSwiGluPlan,
        rank: ParallelRankId,
        gate: TensorParallelWeightShard,
        up: TensorParallelWeightShard,
        down: TensorParallelWeightShard,
    ) -> Result<Self> {
        plan.rank_range(rank)?;
        if gate.plan() != plan.gate_up_plan()
            || up.plan() != plan.gate_up_plan()
            || down.plan() != plan.down_plan()
            || [gate.rank(), up.rank(), down.rank()]
                .iter()
                .any(|r| *r != rank)
            || gate.dtype() != up.dtype()
            || gate.dtype() != down.dtype()
        {
            return Err(invalid(
                "SwiGLU local weights disagree on plan, rank, or dtype",
            ));
        }
        let weights = Self {
            plan,
            rank,
            gate,
            up,
            down,
        };
        weights.validate_source_identities()?;
        Ok(weights)
    }
    pub fn plan(&self) -> &TensorParallelSwiGluPlan {
        &self.plan
    }
    pub fn rank(&self) -> ParallelRankId {
        self.rank
    }
    pub fn gate(&self) -> &TensorParallelWeightShard {
        &self.gate
    }
    pub fn up(&self) -> &TensorParallelWeightShard {
        &self.up
    }
    pub fn down(&self) -> &TensorParallelWeightShard {
        &self.down
    }
    pub fn validate_source_identities(&self) -> Result<()> {
        self.gate.validate_source_identity()?;
        self.up.validate_source_identity()?;
        self.down.validate_source_identity()
    }
}

/// Transport-neutral collective requirement. Only runtime owns reduction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorParallelCollective {
    AllGather,
    Sum,
}

/// Additive stage API; existing linear callers retain their original plan API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorParallelStagePlan {
    Linear(TensorParallelLinearPlan),
    SwiGlu(TensorParallelSwiGluPlan),
}

impl From<TensorParallelLinearPlan> for TensorParallelStagePlan {
    fn from(plan: TensorParallelLinearPlan) -> Self {
        Self::Linear(plan)
    }
}
impl From<TensorParallelSwiGluPlan> for TensorParallelStagePlan {
    fn from(plan: TensorParallelSwiGluPlan) -> Self {
        Self::SwiGlu(plan)
    }
}
impl TensorParallelStagePlan {
    pub fn ranks(&self) -> usize {
        match self {
            Self::Linear(p) => p.ranks(),
            Self::SwiGlu(p) => p.ranks(),
        }
    }
    pub fn in_features(&self) -> usize {
        match self {
            Self::Linear(p) => p.in_features(),
            Self::SwiGlu(p) => p.hidden_size(),
        }
    }
    pub fn out_features(&self) -> usize {
        match self {
            Self::Linear(p) => p.out_features(),
            Self::SwiGlu(p) => p.hidden_size(),
        }
    }
    pub fn collective(&self) -> TensorParallelCollective {
        match self {
            Self::Linear(p) if p.partition() == TensorParallelLinearPartition::Column => {
                TensorParallelCollective::AllGather
            }
            _ => TensorParallelCollective::Sum,
        }
    }
    /// F32 elements per member, not the concatenated gather size.
    pub fn collective_count(&self, rows: usize) -> Result<usize> {
        match self {
            Self::Linear(p) if self.collective() == TensorParallelCollective::AllGather => {
                p.column_gather_count(rows)
            }
            _ => checked_elements(rows, self.out_features()),
        }
    }
    pub fn pack_local_output(
        &self,
        rank: ParallelRankId,
        values: &[f32],
        rows: usize,
    ) -> Result<Vec<f32>> {
        match self {
            Self::Linear(p) => {
                p.rank_range(rank)?;
            }
            Self::SwiGlu(p) => {
                p.rank_range(rank)?;
            }
        }
        match self {
            Self::Linear(p) if self.collective() == TensorParallelCollective::AllGather => {
                p.pack_column_output(rank, values, rows)
            }
            _ => {
                check_len("sum partial", values.len(), self.collective_count(rows)?)?;
                Ok(values.to_vec())
            }
        }
    }
    /// Decode an already completed runtime collective; this never sums partials.
    pub fn unpack_collective_output(&self, values: &[f32], rows: usize) -> Result<Vec<f32>> {
        match self {
            Self::Linear(p) if self.collective() == TensorParallelCollective::AllGather => {
                p.unpack_column_gather(values, rows)
            }
            _ => {
                check_len("sum result", values.len(), self.collective_count(rows)?)?;
                Ok(values.to_vec())
            }
        }
    }
}

#[derive(Debug)]
enum CpuLocalStorage {
    F32(Vec<f32>),
    Bf16(Vec<u8>),
}
#[derive(Debug)]
struct CpuLocalWeight {
    storage: CpuLocalStorage,
    shape: [usize; 2],
}
impl CpuLocalWeight {
    fn new(shard: TensorParallelWeightShard) -> Result<Self> {
        shard.validate_source_identity()?;
        let shape = shard.local_shape();
        let storage = match shard.dtype {
            CheckpointDType::Bf16 => CpuLocalStorage::Bf16(shard.bytes),
            CheckpointDType::F32 => CpuLocalStorage::F32(
                shard
                    .bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().expect("validated F32 bytes")))
                    .collect(),
            ),
            _ => return Err(invalid("unsupported CPU local storage")),
        };
        Ok(Self { storage, shape })
    }
    fn as_ref(&self) -> LinearRef<'_> {
        LinearRef {
            weight: match &self.storage {
                CpuLocalStorage::F32(v) => LinearWeight::F32(v),
                CpuLocalStorage::Bf16(v) => LinearWeight::Bf16(v),
            },
            out_features: self.shape[0],
            in_features: self.shape[1],
            bias: None,
        }
    }
}

#[derive(Debug)]
pub struct CpuLinearShard {
    plan: TensorParallelLinearPlan,
    rank: ParallelRankId,
    weight: CpuLocalWeight,
}
impl CpuLinearShard {
    pub fn from_local(shard: TensorParallelWeightShard) -> Result<Self> {
        Ok(Self {
            plan: shard.plan.clone(),
            rank: shard.rank,
            weight: CpuLocalWeight::new(shard)?,
        })
    }
    pub fn plan(&self) -> &TensorParallelLinearPlan {
        &self.plan
    }
    pub fn rank(&self) -> ParallelRankId {
        self.rank
    }
    pub fn execute(&self, input: &[f32], rows: usize) -> Result<Vec<f32>> {
        let local = self.plan.shard_input(self.rank, input, rows)?;
        self.execute_local(&local, rows)
    }
    pub fn execute_local(&self, input: &[f32], rows: usize) -> Result<Vec<f32>> {
        let [out, width] = self.weight.shape;
        checked_elements(rows, out)?;
        let input = cpu_input(input, rows, width)?;
        Ok(NativeCpuProvider
            .linear(
                self.weight.as_ref(),
                &input,
                None,
                CpuExecutionPrecision::F32,
            )?
            .into_values())
    }
}

/// Native provider dispatch with F32 activations/partials. BF16 checkpoint
/// weights stay BF16; do not round each partial to BF16 before runtime's sum.
#[derive(Debug)]
pub struct CpuSwiGluShard {
    plan: TensorParallelSwiGluPlan,
    rank: ParallelRankId,
    gate: CpuLocalWeight,
    up: CpuLocalWeight,
    down: CpuLocalWeight,
}
impl CpuSwiGluShard {
    pub fn from_local(weights: TensorParallelSwiGluWeights) -> Result<Self> {
        weights.validate_source_identities()?;
        Ok(Self {
            plan: weights.plan,
            rank: weights.rank,
            gate: CpuLocalWeight::new(weights.gate)?,
            up: CpuLocalWeight::new(weights.up)?,
            down: CpuLocalWeight::new(weights.down)?,
        })
    }
    pub fn plan(&self) -> &TensorParallelSwiGluPlan {
        &self.plan
    }
    pub fn rank(&self) -> ParallelRankId {
        self.rank
    }
    pub fn execute(&self, input: &[f32], rows: usize) -> Result<Vec<f32>> {
        checked_elements(rows, self.plan.rank_range(self.rank)?.len())?;
        let input = cpu_input(input, rows, self.plan.hidden_size())?;
        Ok(NativeCpuProvider
            .swiglu(
                SwiGluRef {
                    gate: self.gate.as_ref(),
                    up: self.up.as_ref(),
                    down: self.down.as_ref(),
                    activation_limit: None,
                },
                &input,
                None,
                CpuExecutionPrecision::F32,
            )?
            .into_values())
    }
}
fn cpu_input(input: &[f32], rows: usize, width: usize) -> Result<HostRows> {
    check_len(
        "local operator input",
        input.len(),
        checked_elements(rows, width)?,
    )?;
    HostRows::new(
        RowsShape::new(rows, width)?,
        RowsDType::F32,
        None,
        input.to_vec(),
    )
}
