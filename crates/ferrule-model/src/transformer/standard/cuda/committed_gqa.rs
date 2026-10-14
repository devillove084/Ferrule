//! Isolated read-only GQA over committed F32 KV. Not an append, decoder forward,
//! transaction view, or next-token API. Q is supplied after RoPE by the caller.
use super::{CudaStandardDecoderOperators, cuda_error, device_rows};
use crate::decoder::{
    DecoderKvSequenceCustody, DecoderSequence, PagedKvBackend, TypedCudaPagedKvPool,
};
use crate::transformer::{Rows, RowsArenaId, RowsShape};
use ferrule_backend::cpu::StreamingF32GqaConfig;
use ferrule_backend::cuda::operators::kv::host_placement::{
    CommittedGqaProgress, CommittedGqaReceipt, CudaCommittedGqaOutput, CudaHostPlacement,
};
use ferrule_common::Result;
use ferrule_common::host_kv::{CommittedKvSource, placement::PlacementBudget};

pub struct CommittedGqaRequest {
    pub query: Rows,
    pub layer: usize,
    pub config: StreamingF32GqaConfig,
    pub query_position: u64,
    pub cpu_tokens: usize,
}

#[must_use = "poll or cancel and reap this owner"]
pub struct CommittedGqa {
    placement: CudaHostPlacement,
    // Own the original Q as well as the backend's allocation view.
    _query: Rows,
    shape: RowsShape,
    arena: Option<RowsArenaId>,
}

#[must_use = "release after the last device consumer"]
pub struct CommittedGqaOutput {
    rows: Option<Rows>,
    physical: CudaCommittedGqaOutput,
}
impl CommittedGqaOutput {
    pub fn rows(&self) -> Option<&Rows> {
        self.rows.as_ref()
    }
    pub fn receipt(&self) -> &CommittedGqaReceipt {
        self.physical.receipt()
    }
    pub fn release(&mut self) -> Result<()> {
        self.rows = None;
        self.physical.release()
    }
}
impl CudaStandardDecoderOperators {
    /// Call only under a runtime logical read lease. The physical half validates
    /// the current model ledger, exact owner, schema, resident slots and epochs.
    pub fn committed_gqa<S: DecoderSequence>(
        &mut self,
        backend: &mut PagedKvBackend<TypedCudaPagedKvPool<S>>,
        sequence: &S,
        custody: &DecoderKvSequenceCustody,
        source: &CommittedKvSource,
        request: CommittedGqaRequest,
        budget: &PlacementBudget,
    ) -> Result<CommittedGqa> {
        if self.needs_quarantine()
            || sequence.core().is_poisoned()
            || custody.source_index != 0
            || custody.query_len != 0
            || custody.topology_id != sequence.topology_id()
            || custody.page_state_slot != source.state_slot
            || custody.page_generation != source.generation
            || custody.execution_generation != sequence.core().generation()
            || custody.context_len != sequence.core().position()
            || custody.context_len != source.committed_tokens
        {
            return Err(cuda_error(
                "committed GQA requires exact quiescent sequence custody (query_len=0)",
            ));
        }
        let query = self.f32(&request.query)?;
        if request.query.shape().rows() != 1
            || request.query.shape().elements() != request.config.query_elements()?
        {
            return Err(cuda_error("committed GQA requires one F32 query row"));
        }
        let admission = CudaHostPlacement::admit(source, request.config, budget)?;
        let read = backend.committed_read(source, request.layer)?;
        read.validate_operator(&self.ops)?;
        let query_view = query.prefix(query.len())?;
        let shape = request.query.shape();
        let arena = request.query.arena();
        let placement = CudaHostPlacement::new(
            read,
            query_view,
            request.config,
            request.query_position,
            request.cpu_tokens,
            admission,
        )?;
        Ok(CommittedGqa {
            placement,
            _query: request.query,
            shape,
            arena,
        })
    }
}
impl CommittedGqa {
    pub fn poll(&mut self) -> Result<CommittedGqaProgress> {
        self.placement.poll()
    }
    pub fn cancel(&mut self) {
        self.placement.cancel();
    }
    pub fn reap(&mut self) -> Result<()> {
        self.placement.reap()
    }
    pub fn is_quarantined(&self) -> bool {
        self.placement.is_quarantined()
    }
    pub fn finish(&mut self) -> Result<CommittedGqaOutput> {
        let physical = self.placement.finish()?;
        let rows = physical
            .buffer()
            .map(|buffer| device_rows(self.shape, self.arena, buffer.prefix(buffer.len())?))
            .transpose()?;
        Ok(CommittedGqaOutput { rows, physical })
    }
}
