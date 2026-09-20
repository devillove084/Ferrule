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
//! Limits describe physical host f32 storage, NOT DistributedTransaction credits.
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
}

/// `count` is the number of input f32 elements from EACH member.
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
    /// Bounds all service-owned input and output f32 buffer capacities combined.
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
    input: Option<Vec<f32>>,
    output: Option<Vec<f32>>,
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
                    rank.input
                        .as_ref()
                        .map_or(0, |v| v.capacity() * size_of::<f32>())
                        + rank
                            .output
                            .as_ref()
                            .map_or(0, |v| v.capacity() * size_of::<f32>())
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
        let max_input_bytes = allocation_bytes::<f32>(self.limits.max_elements_per_rank)?
            .checked_mul(ranks)
            .ok_or(HostCollectiveError::ArithmeticOverflow)?;
        let output_elements_per_rank = match kind {
            HostCollectiveKind::AllReduceSumF32 => count,
            HostCollectiveKind::AllGatherF32 => count
                .checked_mul(ranks)
                .ok_or(HostCollectiveError::ArithmeticOverflow)?,
        };
        let output_bytes = allocation_bytes::<f32>(output_elements_per_rank)?
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
        let (index, memory) = match self.validate_submission(rank, descriptor, &payload) {
            Ok(validated) => validated,
            Err(error) => return Err(HostCollectiveSubmitError { error, payload }),
        };
        if self.operation.is_none() {
            // All size/capacity/identity checks precede these allocations. Fixed
            // length Vec construction gives exactly one output allocation/rank.
            let mut ranks = Vec::with_capacity(self.members.len());
            for _ in &self.members {
                ranks.push(RankBuffers {
                    input: None,
                    output: Some(vec![0.0; memory.output_elements_per_rank]),
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
        let index = self.rank_index(rank)?;
        let op = self
            .operation
            .as_mut()
            .ok_or(HostCollectiveError::NoOperation)?;
        if op.submitted != self.members.len() {
            return Ok(HostCollectivePoll::Pending);
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
        Ok(HostCollectivePoll::Ready(HostCollectiveResult {
            descriptor,
            values,
        }))
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
        payload: &Vec<f32>,
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
        } else if let Some(watermark) = self.watermark {
            if descriptor.sequence <= watermark {
                return Err(HostCollectiveError::StaleSequence {
                    watermark,
                    received: descriptor.sequence,
                });
            }
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
        match self.descriptor.kind {
            HostCollectiveKind::AllReduceSumF32 => {
                let (first, rest) = self.ranks.split_first_mut().expect("nonempty members");
                let output = first.output.as_mut().expect("output reserved");
                output.copy_from_slice(first.input.as_ref().expect("all inputs submitted"));
                for rank in rest.iter() {
                    for (sum, value) in output.iter_mut().zip(rank.input.as_ref().unwrap()) {
                        *sum += *value;
                    }
                }
                for rank in rest {
                    rank.output.as_mut().unwrap().copy_from_slice(output);
                }
            }
            HostCollectiveKind::AllGatherF32 => {
                // Assemble directly into each final output: no concatenation scratch
                // buffer and no duplicate rank inputs. Count zero is a valid rendezvous.
                for destination in 0..self.ranks.len() {
                    let mut output = self.ranks[destination].output.take().unwrap();
                    let mut offset = 0;
                    for source in &self.ranks {
                        let input = source.input.as_ref().expect("all inputs submitted");
                        // count * ranks was checked before allocation.
                        let end = offset + input.len();
                        output[offset..end].copy_from_slice(input);
                        offset = end;
                    }
                    self.ranks[destination].output = Some(output);
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
