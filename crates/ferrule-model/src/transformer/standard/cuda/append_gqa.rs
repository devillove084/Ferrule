//! Explicit runtime-authorized append route through the existing KvView hook.
//! No selection flag, global registry, or witness-only constructor.
use super::super::{cuda_error, device_rows};
use crate::decoder::{CudaKvView, PackedDecoderBatch};
use crate::transformer::{
    GqaRequest, KvAppendRequest, KvHistory, KvView, OperatorProgress, Rows, StandardDecoderKvView,
};
use ferrule_backend::cpu::StreamingF32GqaConfig;
use ferrule_backend::cuda::context::CudaOperators;
use ferrule_backend::cuda::operators::kv::transaction::append_bridge::{
    AppendGqaRequest, AppendPrefixAuthority, CudaAppendGqa, CudaAppendGqaOutput,
};
use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::{Result, host_kv::placement::PlacementBudget};

pub struct CudaStandardAppendKvBinding<'a> {
    view: &'a mut CudaKvView,
    batch: &'a PackedDecoderBatch,
    authority: &'a dyn AppendPrefixAuthority,
    budget: &'a PlacementBudget,
    block_tokens: usize,
    operation: Option<CudaAppendGqa<'a>>,
    output: Option<CudaAppendGqaOutput<'a>>,
    canceled: bool,
    returned: bool,
}
impl<'a> CudaStandardAppendKvBinding<'a> {
    pub fn new(
        view: &'a mut CudaKvView,
        batch: &'a PackedDecoderBatch,
        authority: &'a dyn AppendPrefixAuthority,
        budget: &'a PlacementBudget,
        block_tokens: usize,
    ) -> Result<Self> {
        authority.validate()?;
        view.validate_batch(batch)?;
        if view.transaction() != authority.witness().transaction() {
            return Err(cuda_error("append transaction mismatch"));
        }
        if block_tokens == 0 {
            return Err(cuda_error("append block size must be nonzero"));
        }
        Ok(Self {
            view,
            batch,
            authority,
            budget,
            block_tokens,
            operation: None,
            output: None,
            canceled: false,
            returned: false,
        })
    }
    fn attend(
        &mut self,
        ops: &CudaOperators,
        request: GqaRequest<'_>,
    ) -> Result<OperatorProgress<Rows>> {
        self.authority.validate()?;
        let view = &mut *self.view;
        let batch = self.batch;
        let witness = self.authority.witness();
        if self.canceled
            || self.returned
            || self.operation.is_some()
            || self.output.is_some()
            || view.transaction() != witness.transaction()
            || request.layer != self.authority.layer()
            || batch.len() != 1
            || batch.sequences().len() != 1
            || batch.positions() != [witness.source().committed_tokens]
            || batch.sequences()[0].block_table() != witness.pages()
        {
            return Err(cuda_error(
                "append binding mismatched transaction/layer/frontier or unresolved output",
            ));
        }
        let config = StreamingF32GqaConfig {
            rows: 1,
            q_heads: request.query_heads,
            kv_heads: request.kv_heads,
            head_dim: request.head_dim,
            block_tokens: self.block_tokens,
            softmax_scale: request.softmax_scale,
            sliding_window: None,
        };
        let operands = AppendGqaRequest {
            query: request
                .query
                .cuda()?
                .f32_buffer()
                .ok_or_else(|| cuda_error("append Q is not F32"))?,
            key: request
                .key
                .cuda()?
                .f32_buffer()
                .ok_or_else(|| cuda_error("append K is not F32"))?,
            value: request
                .value
                .cuda()?
                .f32_buffer()
                .ok_or_else(|| cuda_error("append V is not F32"))?,
            config,
        };
        // The genuine decoder view authenticates physical pool, nonce, epoch,
        // packed batch and read set. No raw decoder private field is exported.
        let ticket = view.with_f32_gqa_planes(
            ops,
            request.layer,
            batch,
            batch.protected_pages(),
            |planes| Ok(planes.append_transaction),
        )?;
        self.operation = Some(ticket.begin(self.authority, ops, operands, self.budget)?);
        // This operator surface is synchronous already. Each poll uses exact
        // transport evidence and checks the still-borrowed logical authority.
        while !self.operation.as_mut().expect("owned operation").poll()? {
            std::thread::yield_now();
        }
        self.output = Some(self.operation.as_mut().expect("owned operation").finish()?);
        self.operation = None;
        let output = self.output.as_ref().expect("charged output").buffer()?;
        device_rows(
            request.query.shape(),
            request.arena,
            output.prefix(output.len())?,
        )
        .map(OperatorProgress::Ready)
    }
    pub(super) fn ensure_retired(&self) -> Result<()> {
        if self.canceled || !self.returned || self.operation.is_some() || self.output.is_some() {
            return Err(cuda_error("append layer retains unresolved output custody"));
        }
        Ok(())
    }
    pub fn cancel(&mut self) {
        self.canceled = true;
        if let Some(op) = self.operation.as_mut() {
            op.cancel();
        }
        if let Some(output) = self.output.as_mut() {
            output.cancel();
        }
    }
    /// Consumes the exact Rows returned by attend. Call after all downstream
    /// work using that output has been enqueued; never retain buffer aliases.
    pub fn return_output(&mut self, rows: Rows) -> Result<()> {
        if self.canceled {
            return Err(cuda_error("append binding canceled; custody retained"));
        }
        let output = self
            .output
            .as_mut()
            .ok_or_else(|| cuda_error("append output absent"))?;
        let expected = output.buffer()?;
        let returned = rows
            .cuda()?
            .f32_buffer()
            .ok_or_else(|| cuda_error("returned append output is not F32"))?;
        if returned.len() != expected.len()
            || returned.as_device_buffer().cu_deviceptr()
                != expected.as_device_buffer().cu_deviceptr()
        {
            return Err(cuda_error("foreign append output return"));
        }
        drop(rows);
        output.release()?;
        self.output = None;
        self.returned = true;
        Ok(())
    }
}

impl StandardDecoderKvView for CudaStandardAppendKvBinding<'_> {
    fn transaction(&self) -> ExecutionTransactionId {
        self.view.transaction()
    }
    fn validate_batch(&self, batch: &PackedDecoderBatch) -> Result<()> {
        if batch != self.batch {
            return Err(cuda_error("append binding received foreign batch"));
        }
        self.view.validate_batch(batch)
    }
}
impl KvView for CudaStandardAppendKvBinding<'_> {
    #[expect(
        unsafe_code,
        reason = "Implements the audited same-owner GQA output retirement contract"
    )]
    unsafe fn finish_gqa_consumers(
        &mut self,
        operators: &mut dyn crate::transformer::StandardDecoderOperators,
    ) -> Result<()> {
        if self.canceled {
            return Err(cuda_error("append canceled; cannot retire consumer"));
        }
        // Return owner-held operand roots before the allocation charge. Failure
        // leaves the output guard installed; its Drop quarantines the same Entry.
        operators.before_external_routed_execution()?;
        self.output
            .as_mut()
            .ok_or_else(|| cuda_error("append output absent"))?
            .release()?;
        self.output = None;
        self.returned = true;
        Ok(())
    }
    fn append(&mut self, _: KvAppendRequest<'_>) -> Result<()> {
        Err(cuda_error("append binding does not expose host writes"))
    }
    fn history(&self, _: usize, _: usize, _: usize, _: usize, _: usize) -> Result<KvHistory> {
        Err(cuda_error("append binding does not expose host history"))
    }
    fn append_and_attend_cuda(
        &mut self,
        ops: &CudaOperators,
        request: GqaRequest<'_>,
    ) -> Result<OperatorProgress<Rows>> {
        self.authority.validate()?;
        if request.metadata.row_positions() != self.batch.positions()
            || request.metadata.row_sequence_ids() != self.batch.row_to_sequence()
            || request.metadata.sequence_count() != self.batch.sequences().len()
        {
            return Err(cuda_error("append metadata differs from bound batch"));
        }
        for rows in [request.query, request.key, request.value] {
            rows.cuda()?.validate_owner(ops)?;
        }
        if request.metadata.row_kv_lens().len() != self.batch.len()
            || request
                .metadata
                .row_kv_lens()
                .iter()
                .zip(self.batch.positions())
                .any(|(&len, &p)| Some(len) != p.checked_add(1))
            || request.query.shape().rows() != 1
            || request.key.shape().rows() != 1
            || request.value.shape() != request.key.shape()
            || Some(request.query.shape().width())
                != request.query_heads.checked_mul(request.head_dim)
            || Some(request.key.shape().width()) != request.kv_heads.checked_mul(request.head_dim)
        {
            return Err(cuda_error("append request row/geometry mismatch"));
        }
        self.attend(ops, request)
    }
}
