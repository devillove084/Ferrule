//! Synchronous, CPU-owned host-staged collectives; no NCCL or device handles.
//!
//! A group is an ordered member vector within a caller-managed topology epoch.
//! The first submission (from any member) fixes the descriptor. All other members
//! must submit that exact descriptor. AllGather concatenates in member-vector
//! order; AllReduce adds in that same order, starting with member zero's value
//! rather than an artificial zero. Arrival order cannot affect the reduction.
//!
//! There is exactly one in-flight operation per group, including ready outputs
//! not yet taken. Sequences may have gaps but must strictly increase between
//! operations. A single watermark rejects completed and aborted sequences; no
//! transaction history or tombstones accumulate. Reconfiguration requires an
//! explicit `new` with a new epoch; callers must retire the old group themselves.
//!
//! Limits describe physical typed host storage, NOT DistributedTransaction credits.
//! The outer transaction owns `communicate_all`/communication accounting. CPU
//! readiness is not transaction completion: callers must upload each result and
//! synchronize the GPU work before completing its outer communication operation.
//! This service neither commits nor publishes transactions. Abort only drops
//! CPU-owned buffers; it does not cancel DMA or reclaim previously taken outputs.
//! Inputs must already be CPU-owned and no longer referenced by any DMA operation.
//!
//! This is a bounded synchronous baseline, not asynchronous or pinned transport.
//! Ordinary std allocation failure follows the process's allocation-error policy.

use std::fmt;
use std::mem::size_of;

use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::{ParallelGroupId, ParallelRankId, ParallelTopologyId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostCollectiveKind {
    AllReduceSumF32,
    AllGatherF32,
    /// Two already-rounded BF16 inputs, F32 scalar addition, one BF16 RNE output.
    /// TP2 only: no claim about multi-hop NCCL reduction rounding for larger groups.
    AllReduceSumBf16,
    /// Unrounded F32 partials summed in member order, then one BF16 RNE output.
    /// Explicit numerical candidate, NOT interchangeable with BF16 partials.
    AllReduceSumF32ToBf16,
    AllGatherBf16,
}
impl HostCollectiveKind {
    fn input_bf16(self) -> bool {
        matches!(self, Self::AllReduceSumBf16 | Self::AllGatherBf16)
    }
    fn output_bf16(self) -> bool {
        self.input_bf16() || self == Self::AllReduceSumF32ToBf16
    }
    fn gather(self) -> bool {
        matches!(self, Self::AllGatherF32 | Self::AllGatherBf16)
    }
}

/// Owned host values after DMA completion, not a device tensor or execution permit.
/// BF16 payloads retain their actual two-byte representation; no F32 mirror.
#[derive(Debug, PartialEq)]
pub enum CollectivePayload {
    F32(Vec<f32>),
    Bf16(Vec<u16>),
}
impl CollectivePayload {
    pub fn len(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::Bf16(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn capacity(&self) -> usize {
        match self {
            Self::F32(v) => v.capacity(),
            Self::Bf16(v) => v.capacity(),
        }
    }
    pub fn owned_bytes(&self) -> usize {
        self.capacity() * if self.is_bf16() { 2 } else { 4 }
    }
    fn is_bf16(&self) -> bool {
        matches!(self, Self::Bf16(_))
    }
    fn zeros(bf16: bool, n: usize) -> Self {
        if bf16 {
            Self::Bf16(vec![0; n])
        } else {
            Self::F32(vec![0.0; n])
        }
    }
    fn scalar(&self, i: usize) -> f32 {
        match self {
            Self::F32(v) => v[i],
            Self::Bf16(v) => f32::from_bits(u32::from(v[i]) << 16),
        }
    }
}

fn bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        return ((bits >> 16) as u16) | 0x0040;
    }
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

#[derive(Debug, PartialEq)]
pub enum TypedCollectivePoll {
    Pending,
    Ready {
        descriptor: HostCollectiveDescriptor,
        payload: CollectivePayload,
    },
}
#[derive(Debug)]
pub struct TypedCollectiveSubmitError {
    pub error: HostCollectiveError,
    pub payload: CollectivePayload,
}

/// `count` is the number of typed input elements from EACH member.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostCollectiveDescriptor {
    pub epoch: ParallelTopologyId,
    pub group: ParallelGroupId,
    pub transaction: ExecutionTransactionId,
    pub sequence: u64,
    pub kind: HostCollectiveKind,
    pub count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostCollectiveLimits {
    pub max_ranks: usize,
    /// Bounds both payload length and Vec capacity, including unused capacity.
    pub max_elements_per_rank: usize,
    /// Bounds all service-owned input and output buffer capacities combined.
    /// Bounded member/slot metadata and allocator bookkeeping are not included.
    pub max_host_bytes: usize,
}

/// Conservative peak reservation, checked before operation allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostCollectiveMemory {
    /// All ranks may supply a Vec with capacity `max_elements_per_rank`.
    pub max_input_bytes: usize,
    pub output_elements_per_rank: usize,
    pub output_bytes: usize,
    pub peak_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostCollectiveStatus {
    Pending,
    Ready,
}

#[derive(Debug, PartialEq)]
pub struct HostCollectiveResult {
    pub descriptor: HostCollectiveDescriptor,
    pub values: Vec<f32>,
}

#[derive(Debug, PartialEq)]
pub enum HostCollectivePoll {
    Pending,
    Ready(HostCollectiveResult),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostCollectiveError {
    InvalidLimits,
    PayloadTypeMismatch,
    UnsupportedDegree,
    EmptyMembers,
    TooManyRanks,
    DuplicateMember(ParallelRankId),
    WrongRank(ParallelRankId),
    WrongEpoch,
    WrongGroup,
    CountLimit,
    CountMismatch { expected: usize, actual: usize },
    InputCapacityLimit,
    ArithmeticOverflow,
    HostCapacity { required: usize, limit: usize },
    StaleSequence { watermark: u64, received: u64 },
    Backpressure,
    DescriptorMismatch,
    DuplicateSubmission(ParallelRankId),
    NoOperation,
    ResultAlreadyTaken(ParallelRankId),
}

impl fmt::Display for HostCollectiveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "host collective: {self:?}")
    }
}

impl std::error::Error for HostCollectiveError {}

/// Rejection returns the original allocation to the caller and changes no state.
#[derive(Debug)]
pub struct HostCollectiveSubmitError {
    pub error: HostCollectiveError,
    pub payload: Vec<f32>,
}

impl fmt::Display for HostCollectiveSubmitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error.fmt(formatter)
    }
}

impl std::error::Error for HostCollectiveSubmitError {}

#[derive(Debug)]
struct RankBuffers {
    input: Option<CollectivePayload>,
    output: Option<CollectivePayload>,
}

#[derive(Debug)]
struct Operation {
    descriptor: HostCollectiveDescriptor,
    memory: HostCollectiveMemory,
    ranks: Vec<RankBuffers>,
    submitted: usize,
    remaining: usize,
}

/// One fixed ordered participant group. Serialize access via `&mut self`.
/// No global group registry or hidden worker threads are created.
#[derive(Debug)]
pub struct HostCollectiveGroup {
    epoch: ParallelTopologyId,
    group: ParallelGroupId,
    members: Vec<ParallelRankId>,
    limits: HostCollectiveLimits,
    watermark: Option<u64>,
    operation: Option<Operation>,
}

impl HostCollectiveGroup {
    /// Includes a ready operation until its final output is consumed.
    pub const MAX_INFLIGHT_OPERATIONS: usize = 1;

    /// Takes members in their canonical collective order, without sorting them.
    /// IDs, including the group ID, are opaque common IDs. Epoch uniqueness and
    /// membership agreement across callers are the caller's responsibility.
    pub fn new(
        epoch: ParallelTopologyId,
        group: ParallelGroupId,
        members: Vec<ParallelRankId>,
        limits: HostCollectiveLimits,
    ) -> Result<Self, HostCollectiveError> {
        if limits.max_ranks == 0 {
            return Err(HostCollectiveError::InvalidLimits);
        }
        if members.is_empty() {
            return Err(HostCollectiveError::EmptyMembers);
        }
        if members.len() > limits.max_ranks || members.capacity() > limits.max_ranks {
            return Err(HostCollectiveError::TooManyRanks);
        }
        allocation_bytes::<RankBuffers>(limits.max_ranks)?;
        allocation_bytes::<ParallelRankId>(limits.max_ranks)?;
        allocation_bytes::<f32>(limits.max_elements_per_rank)?;
        for (index, rank) in members.iter().enumerate() {
            if members[..index].contains(rank) {
                return Err(HostCollectiveError::DuplicateMember(*rank));
            }
        }
        Ok(Self {
            epoch,
            group,
            members,
            limits,
            watermark: None,
            operation: None,
        })
    }

    pub fn members(&self) -> &[ParallelRankId] {
        &self.members
    }

    pub fn limits(&self) -> HostCollectiveLimits {
        self.limits
    }

    /// Highest admitted sequence, retained across drain and abort, even u64::MAX.
    pub fn sequence_watermark(&self) -> Option<u64> {
        self.watermark
    }

    pub fn active_descriptor(&self) -> Option<HostCollectiveDescriptor> {
        self.operation.as_ref().map(|op| op.descriptor)
    }

    /// None means idle, not a ready/empty operation.
    pub fn status(&self) -> Option<HostCollectiveStatus> {
        self.operation.as_ref().map(|op| {
            if op.submitted == self.members.len() {
                HostCollectiveStatus::Ready
            } else {
                HostCollectiveStatus::Pending
            }
        })
    }

    /// Includes reserved output allocations even while inputs are still pending.
    /// Once taken, outputs belong to the caller and are outside this budget.
    pub fn owned_host_bytes(&self) -> usize {
        self.operation.as_ref().map_or(0, |op| {
            op.ranks
                .iter()
                .map(|rank| {
                    rank.input.as_ref().map_or(0, |v| v.owned_bytes())
                        + rank.output.as_ref().map_or(0, |v| v.owned_bytes())
                })
                .sum()
        })
    }

    pub fn reserved_host_bytes(&self) -> usize {
        self.operation.as_ref().map_or(0, |op| op.memory.peak_bytes)
    }

    /// Validates count, individual allocation sizes and aggregate peak storage.
    /// No allocation or mutation occurs. Unused input Vec capacity is budgeted
    /// conservatively so that a valid later rank cannot exhaust this reservation.
    pub fn memory_for(
        &self,
        kind: HostCollectiveKind,
        count: usize,
    ) -> Result<HostCollectiveMemory, HostCollectiveError> {
        if count > self.limits.max_elements_per_rank {
            return Err(HostCollectiveError::CountLimit);
        }
        let ranks = self.members.len();
        if !matches!(
            kind,
            HostCollectiveKind::AllReduceSumF32 | HostCollectiveKind::AllGatherF32
        ) && ranks != 2
        {
            return Err(HostCollectiveError::UnsupportedDegree);
        }
        let input_bytes = if kind.input_bf16() {
            allocation_bytes::<u16>(self.limits.max_elements_per_rank)?
        } else {
            allocation_bytes::<f32>(self.limits.max_elements_per_rank)?
        };
        let max_input_bytes = input_bytes
            .checked_mul(ranks)
            .ok_or(HostCollectiveError::ArithmeticOverflow)?;
        let output_elements_per_rank = if kind.gather() {
            count
                .checked_mul(ranks)
                .ok_or(HostCollectiveError::ArithmeticOverflow)?
        } else {
            count
        };
        let per_output_bytes = if kind.output_bf16() {
            allocation_bytes::<u16>(output_elements_per_rank)?
        } else {
            allocation_bytes::<f32>(output_elements_per_rank)?
        };
        let output_bytes = per_output_bytes
            .checked_mul(ranks)
            .ok_or(HostCollectiveError::ArithmeticOverflow)?;
        let peak_bytes = max_input_bytes
            .checked_add(output_bytes)
            .ok_or(HostCollectiveError::ArithmeticOverflow)?;
        if peak_bytes > self.limits.max_host_bytes {
            return Err(HostCollectiveError::HostCapacity {
                required: peak_bytes,
                limit: self.limits.max_host_bytes,
            });
        }
        Ok(HostCollectiveMemory {
            max_input_bytes,
            output_elements_per_rank,
            output_bytes,
            peak_bytes,
        })
    }

    /// Moves exactly one input per rank. Ready means ALL members submitted.
    /// Every rejection leaves descriptor, buffers and watermark unchanged.
    /// On success the last submission performs the entire collective synchronously.
    pub fn submit(
        &mut self,
        rank: ParallelRankId,
        descriptor: HostCollectiveDescriptor,
        payload: Vec<f32>,
    ) -> Result<HostCollectiveStatus, HostCollectiveSubmitError> {
        if descriptor.kind.input_bf16() || descriptor.kind.output_bf16() {
            return Err(HostCollectiveSubmitError {
                error: HostCollectiveError::PayloadTypeMismatch,
                payload,
            });
        }
        self.submit_typed(rank, descriptor, CollectivePayload::F32(payload))
            .map_err(|e| {
                let CollectivePayload::F32(payload) = e.payload else {
                    unreachable!("F32 input retained")
                };
                HostCollectiveSubmitError {
                    error: e.error,
                    payload,
                }
            })
    }

    pub fn submit_typed(
        &mut self,
        rank: ParallelRankId,
        descriptor: HostCollectiveDescriptor,
        payload: CollectivePayload,
    ) -> Result<HostCollectiveStatus, TypedCollectiveSubmitError> {
        let (index, memory) = match self.validate_submission(rank, descriptor, &payload) {
            Ok(validated) => validated,
            Err(error) => return Err(TypedCollectiveSubmitError { error, payload }),
        };
        if self.operation.is_none() {
            // All size/capacity/identity checks precede these allocations. Fixed
            // length Vec construction gives exactly one output allocation/rank.
            let mut ranks = Vec::with_capacity(self.members.len());
            for _ in &self.members {
                ranks.push(RankBuffers {
                    input: None,
                    output: Some(CollectivePayload::zeros(
                        descriptor.kind.output_bf16(),
                        memory.output_elements_per_rank,
                    )),
                });
            }
            self.operation = Some(Operation {
                descriptor,
                memory,
                ranks,
                submitted: 0,
                remaining: self.members.len(),
            });
            self.watermark = Some(descriptor.sequence);
        }
        let op = self.operation.as_mut().expect("operation was admitted");
        op.ranks[index].input = Some(payload);
        // The unique-rank check bounds both counters by the validated member count.
        op.submitted += 1;
        if op.submitted != self.members.len() {
            return Ok(HostCollectiveStatus::Pending);
        }
        op.compute();
        Ok(HostCollectiveStatus::Ready)
    }

    /// Takes a ready output once, with its full identity. Pending never consumes
    /// anything. Even members that already took a result cannot submit again until
    /// the final member consumes its result. Callers must serialize rank-only polls
    /// with operation changes, using the returned descriptor to route GPU uploads.
    pub fn take_result(
        &mut self,
        rank: ParallelRankId,
    ) -> Result<HostCollectivePoll, HostCollectiveError> {
        if self
            .operation
            .as_ref()
            .is_some_and(|op| op.descriptor.kind.output_bf16())
        {
            return Err(HostCollectiveError::PayloadTypeMismatch);
        }
        match self.take_typed_result(rank)? {
            TypedCollectivePoll::Pending => Ok(HostCollectivePoll::Pending),
            TypedCollectivePoll::Ready {
                descriptor,
                payload: CollectivePayload::F32(values),
            } => Ok(HostCollectivePoll::Ready(HostCollectiveResult {
                descriptor,
                values,
            })),
            _ => unreachable!("F32 output checked before taking"),
        }
    }

    pub fn take_typed_result(
        &mut self,
        rank: ParallelRankId,
    ) -> Result<TypedCollectivePoll, HostCollectiveError> {
        let index = self.rank_index(rank)?;
        let op = self
            .operation
            .as_mut()
            .ok_or(HostCollectiveError::NoOperation)?;
        if op.submitted != self.members.len() {
            return Ok(TypedCollectivePoll::Pending);
        }
        let values = op.ranks[index]
            .output
            .take()
            .ok_or(HostCollectiveError::ResultAlreadyTaken(rank))?;
        let descriptor = op.descriptor;
        op.remaining -= 1;
        if op.remaining == 0 {
            self.operation = None;
        }
        Ok(TypedCollectivePoll::Ready {
            descriptor,
            payload: values,
        })
    }

    /// Full identity matching prevents a delayed abort from clearing a newer op.
    /// Discards pending inputs and untaken outputs, but retains the watermark.
    /// Already returned buffers and any GPU work remain the caller's responsibility.
    pub fn abort(
        &mut self,
        descriptor: HostCollectiveDescriptor,
    ) -> Result<(), HostCollectiveError> {
        let op = self
            .operation
            .as_ref()
            .ok_or(HostCollectiveError::NoOperation)?;
        if op.descriptor != descriptor {
            return Err(HostCollectiveError::DescriptorMismatch);
        }
        self.operation = None;
        Ok(())
    }

    fn rank_index(&self, rank: ParallelRankId) -> Result<usize, HostCollectiveError> {
        self.members
            .iter()
            .position(|member| *member == rank)
            .ok_or(HostCollectiveError::WrongRank(rank))
    }

    fn validate_submission(
        &self,
        rank: ParallelRankId,
        descriptor: HostCollectiveDescriptor,
        payload: &CollectivePayload,
    ) -> Result<(usize, HostCollectiveMemory), HostCollectiveError> {
        let index = self.rank_index(rank)?;
        if descriptor.epoch != self.epoch {
            return Err(HostCollectiveError::WrongEpoch);
        }
        if descriptor.group != self.group {
            return Err(HostCollectiveError::WrongGroup);
        }
        if let Some(op) = &self.operation {
            if descriptor.sequence > op.descriptor.sequence {
                return Err(HostCollectiveError::Backpressure);
            }
            if descriptor.sequence < op.descriptor.sequence {
                return Err(HostCollectiveError::StaleSequence {
                    watermark: op.descriptor.sequence,
                    received: descriptor.sequence,
                });
            }
            if descriptor != op.descriptor {
                return Err(HostCollectiveError::DescriptorMismatch);
            }
            // Inputs are freed on Ready, so the phase must also reject duplicates.
            if op.submitted == self.members.len() || op.ranks[index].input.is_some() {
                return Err(HostCollectiveError::DuplicateSubmission(rank));
            }
        } else if let Some(watermark) = self.watermark
            && descriptor.sequence <= watermark
        {
            return Err(HostCollectiveError::StaleSequence {
                watermark,
                received: descriptor.sequence,
            });
        }
        if payload.is_bf16() != descriptor.kind.input_bf16() {
            return Err(HostCollectiveError::PayloadTypeMismatch);
        }
        if payload.len() != descriptor.count {
            return Err(HostCollectiveError::CountMismatch {
                expected: descriptor.count,
                actual: payload.len(),
            });
        }
        if descriptor.count > self.limits.max_elements_per_rank {
            return Err(HostCollectiveError::CountLimit);
        }
        if payload.capacity() > self.limits.max_elements_per_rank {
            return Err(HostCollectiveError::InputCapacityLimit);
        }
        Ok((index, self.memory_for(descriptor.kind, descriptor.count)?))
    }
}

impl Operation {
    fn compute(&mut self) {
        if self.descriptor.kind.gather() {
            // Copy bits directly, without widening BF16 or allocating scratch.
            for destination in 0..self.ranks.len() {
                let mut output = self.ranks[destination].output.take().unwrap();
                let mut offset = 0;
                for source in &self.ranks {
                    let input = source.input.as_ref().expect("all inputs submitted");
                    let end = offset + input.len();
                    match (&mut output, input) {
                        (CollectivePayload::F32(out), CollectivePayload::F32(input)) => {
                            out[offset..end].copy_from_slice(input)
                        }
                        (CollectivePayload::Bf16(out), CollectivePayload::Bf16(input)) => {
                            out[offset..end].copy_from_slice(input)
                        }
                        _ => unreachable!("admitted payload type"),
                    }
                    offset = end;
                }
                self.ranks[destination].output = Some(output);
            }
        } else {
            let (first, rest) = self.ranks.split_first_mut().expect("nonempty members");
            let output = first.output.as_mut().expect("output reserved");
            for i in 0..self.descriptor.count {
                let mut sum = first.input.as_ref().unwrap().scalar(i);
                for rank in rest.iter() {
                    sum += rank.input.as_ref().unwrap().scalar(i);
                }
                match output {
                    CollectivePayload::F32(v) => v[i] = sum,
                    CollectivePayload::Bf16(v) => v[i] = bf16_rne(sum),
                }
            }
            for rank in rest {
                match (rank.output.as_mut().unwrap(), &*output) {
                    (CollectivePayload::F32(dst), CollectivePayload::F32(src)) => {
                        dst.copy_from_slice(src)
                    }
                    (CollectivePayload::Bf16(dst), CollectivePayload::Bf16(src)) => {
                        dst.copy_from_slice(src)
                    }
                    _ => unreachable!("admitted output type"),
                }
            }
        }
        for rank in &mut self.ranks {
            rank.input = None;
        }
    }
}

fn allocation_bytes<T>(elements: usize) -> Result<usize, HostCollectiveError> {
    elements
        .checked_mul(size_of::<T>())
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or(HostCollectiveError::ArithmeticOverflow)
}
