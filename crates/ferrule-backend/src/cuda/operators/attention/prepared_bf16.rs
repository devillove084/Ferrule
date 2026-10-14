//! Prepared eager attention; legacy scratch-free dispatch is unchanged.
//!
//! One group is a complete sequence matrix batch. Query rows are contiguous in
//! sequence order, positions within each group form a contiguous causal range,
//! and KV lengths equal the group's active logical length (not page capacity).
//! Query values must be exact BF16 expansions. Nonfinite/intermediate overflow
//! parity is not promised. Cache mutation authority/COW belongs to the existing
//! transaction owner, not this arithmetic object.
//!
//! Preparation owns one dedicated physical grant and a caller-charged external
//! reserve; it allocates, warms every shape and synchronizes. Execute allocates
//! nothing and never synchronizes. Device status must be checked at the caller's
//! completion boundary before committing cache/output. Capture is rejected before
//! any submission. Retain operands through completion; Unknown is terminal.
use super::{PagedBf16CausalGqaLayout, PagedBf16GqaBuffers};
use crate::cuda::{
    ffi::core::{
        TRANSFORMER_PAGED_BF16_APPEND_CAUSAL_GQA, TRANSFORMER_PAGED_BF16_CAUSAL_GQA,
        TransformerArgs,
    },
    operator_owner::{CudaI32Buffer, CudaOperators, CudaTypedBuffer, cu},
    operators::{
        OperatorOwner,
        recurrent_bf16::{Range, disjoint},
    },
    providers::admission::{Admission, Charge, ProviderFamily},
    runtime::{CudaStream, DeviceBuffer, DeviceCopy},
};
use crate::cuda_plan::bf16_attention::invalid;
pub use crate::cuda_plan::bf16_attention::{
    Bf16AttentionGroup, Bf16AttentionPlan, Bf16AttentionShape, Bf16AttentionView,
};
use ferrule_common::Result;
use std::sync::Arc;

#[repr(C)]
#[derive(Clone, Copy)]
struct NativePlan {
    heads: u32,
    kv_heads: u32,
    dim: u32,
    max_rows: u32,
    max_tokens: u32,
    reserved: u32,
    offsets: [u64; 7],
    bytes: u64,
}
impl From<&Bf16AttentionPlan> for NativePlan {
    fn from(p: &Bf16AttentionPlan) -> Self {
        Self {
            heads: p.heads() as u32,
            kv_heads: p.kv_heads() as u32,
            dim: p.head_dim() as u32,
            max_rows: p.max_rows() as u32,
            max_tokens: p.max_tokens() as u32,
            reserved: 0,
            offsets: p.offsets().map(|x| x as u64),
            bytes: p.storage_bytes() as u64,
        }
    }
}
unsafe extern "C" {
    fn ferrule_bf16_attention_create(
        p: *const NativePlan,
        g: *const Bf16AttentionGroup,
        count: u32,
        stream: u64,
        memory: u64,
        bytes: u64,
        handle: *mut u64,
        growth: *mut u64,
        version: *mut i32,
    ) -> i32;
    fn ferrule_bf16_attention_destroy(handle: u64) -> i32;
    fn ferrule_bf16_attention_begin(stream: u64, status: u64) -> i32;
    fn ferrule_bf16_attention_launch(
        a: *const TransformerArgs,
        p: *const NativePlan,
        g: *const Bf16AttentionGroup,
        count: u32,
        memory: u64,
        bytes: u64,
        handle: u64,
    ) -> i32;
}
const _: () = {
    assert!(size_of::<NativePlan>() == 88);
    assert!(size_of::<Bf16AttentionGroup>() == 16);
};
fn status(code: i32) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(invalid(format!(
            "native status {code}; requires CUDA12.4/sm86/cuBLAS120405; no fallback"
        )))
    }
}

// Native status9 is terminal even if a later fence might succeed. This helper
// is CPU-tested with a forbidden later-fence closure; it never clears Unknown.
fn initialization_completion(
    code: i32,
    poison: impl FnOnce(),
    fence: impl FnOnce() -> Result<()>,
) -> Result<()> {
    if code == 9 {
        poison();
        return Err(invalid("initialization completion Unknown"));
    }
    if let Err(error) = fence() {
        poison();
        return Err(error);
    }
    Ok(())
}

/// Exclusively used, stream-bound arithmetic resources. No raw handle/buffer
/// escape and no graph capture. The plan's total charge must remain in the
/// caller's admission ledger until this object retires, including quarantine.
struct Invocation {
    status: DeviceBuffer<i32>,
}
pub struct CudaPreparedBf16Attention {
    plan: Bf16AttentionPlan,
    stream: Arc<CudaStream>,
    storage: Option<DeviceBuffer<u8>>,
    handle: u64,
    growth: u64,
    version: i32,
    poisoned: bool,
    admission: Option<Admission>,
    invocation: Option<Invocation>,
    default_view: Bf16AttentionView,
}
impl CudaOperators {
    pub fn prepare_bf16_attention(
        &self,
        plan: Bf16AttentionPlan,
    ) -> Result<CudaPreparedBf16Attention> {
        let stream = self.stream_clone();
        cu(stream.context().check_graph_ready())?;
        if stream.context().is_capturing() {
            return Err(invalid("preparation/capture forbidden"));
        }
        if std::env::var("CUBLAS_WORKSPACE_CONFIG").as_deref() != Ok(":0:0") {
            return Err(invalid(
                "launch process with CUBLAS_WORKSPACE_CONFIG=:0:0 before CUDA initialization",
            ));
        }
        Admission::prepare(
            &stream,
            ProviderFamily::Attention,
            Charge {
                storage_grant: plan.storage_grant_bytes(),
                external: plan.provider_bytes(),
            },
            |admission| {
                cu(stream.context().bind_to_thread())?;
                let default_view = plan.view(plan.groups())?;
                // Install allocation custody before any memset/warmup submission.
                let storage = cu(DeviceBuffer::uninitialized_dedicated(
                    &stream,
                    plan.storage_bytes(),
                ))?;
                let mut owner = CudaPreparedBf16Attention {
                    plan,
                    stream: Arc::clone(&stream),
                    storage: Some(storage),
                    handle: 0,
                    growth: 0,
                    version: 0,
                    poisoned: false,
                    admission: Some(admission),
                    invocation: None,
                    default_view,
                };
                let p = NativePlan::from(&owner.plan);
                let b = owner.storage.as_ref().unwrap();
                let code = unsafe {
                    ferrule_bf16_attention_create(
                        &p,
                        owner.plan.warm_groups().as_ptr(),
                        owner.plan.warm_groups().len() as u32,
                        owner.stream.cu_stream() as usize as u64,
                        b.cu_deviceptr(),
                        b.num_bytes() as u64,
                        &mut owner.handle,
                        &mut owner.growth,
                        &mut owner.version,
                    )
                };
                initialization_completion(
                    code,
                    || owner.stream.context().mark_graph_unknown(),
                    || cu(owner.stream.synchronize()),
                )?;
                status(code)?;
                Ok(owner)
            },
        )
    }
}
impl CudaPreparedBf16Attention {
    /// Preparation can lose completion before returning an owner. Retain the
    /// caller's admission charge/image on true; ordinary preflight rejection is
    /// retryable. A later successful fence never clears this terminal state.
    pub fn preparation_completion_unknown(op: &CudaOperators) -> bool {
        op.stream_clone().context().graph_completion_unknown()
    }
    pub fn plan(&self) -> &Bf16AttentionPlan {
        &self.plan
    }
    pub fn provider_version(&self) -> i32 {
        self.version
    }
    /// Diagnostic global free-memory delta, not attributable allocation accounting.
    pub fn provider_observed_growth_bytes(&self) -> u64 {
        self.growth
    }
    pub fn is_poisoned(&self) -> bool {
        self.poisoned || self.stream.context().graph_completion_unknown()
    }
    fn quarantine(&mut self) {
        self.poisoned = true;
        self.stream.context().mark_graph_unknown();
    }
    fn owner(&self, op: &CudaOperators) -> Result<Arc<CudaStream>> {
        let stream = op.stream_clone();
        if !Arc::ptr_eq(&stream, &self.stream)
            || self.is_poisoned()
            || stream.context().is_capturing()
        {
            return Err(invalid("owner mismatch, Unknown, or unsupported capture"));
        }
        cu(stream.context().check_graph_ready())?;
        Ok(stream)
    }
    /// Explicit boundary before an allocation-free invocation/cohort of layers.
    /// Clears the shared sticky status only after the preceding status was
    /// consumed. Ordinary activation/metadata/expert allocations are permitted
    /// during the invocation; global free-memory changes are not failure evidence.
    pub fn begin_invocation(
        &mut self,
        op: &CudaOperators,
        status_word: &mut CudaI32Buffer,
    ) -> Result<()> {
        let stream = self.owner(op)?;
        if self.invocation.is_some() {
            return Err(invalid(
                "previous invocation status must be consumed before clear",
            ));
        }
        let address = ptr(op, status_word, 4)?;
        if status_word.len() != 1 {
            return Err(invalid("one sticky status word required"));
        }
        cu(stream.context().bind_to_thread())?;
        let lease = cu(status_word.as_device_buffer().slice(0, 1))?;
        let result = op.submit_operator(1, |s| {
            status(unsafe { ferrule_bf16_attention_begin(s.cu_stream() as usize as u64, address) })
        });
        if result.is_err() {
            self.quarantine();
        } else {
            self.invocation = Some(Invocation { status: lease });
        }
        result
    }
    /// Owned status download + completion proof. A nonzero device latch is an
    /// invocation error, not successful output/cache publication. Copy/fence
    /// failure quarantines destination/source/provider and blocks further work.
    /// No global-memory-delta policy is applied at this boundary. Provider
    /// reservations are persistent ledger charges, not hard vendor growth caps.
    pub fn finish_invocation(
        &mut self,
        op: &CudaOperators,
        status_word: &CudaI32Buffer,
    ) -> Result<()> {
        let stream = self.owner(op)?;
        let address = ptr(op, status_word, 4)?;
        let invocation = self
            .invocation
            .as_ref()
            .ok_or_else(|| invalid("no active invocation"))?;
        let expected = invocation.status.cu_deviceptr();
        if address != expected || status_word.len() != 1 {
            return Err(invalid("invocation status owner/address mismatch"));
        }
        let completion =
            cu(status_word.as_device_buffer().to_host_vec(&stream)).map(|host| host[0]);
        match completion {
            Err(e) => {
                self.quarantine();
                Err(e)
            }
            Ok(code) => {
                self.invocation = None;
                if code == 0 {
                    Ok(())
                } else {
                    Err(invalid(format!(
                        "sticky device status {code}; invocation not publishable"
                    )))
                }
            }
        }
    }
    pub(crate) fn preflight_view(
        &self,
        op: &CudaOperators,
        status_word: &CudaI32Buffer,
        view: &Bf16AttentionView,
    ) -> Result<()> {
        self.owner(op)?;
        let address = ptr(op, status_word, 4)?;
        if status_word.len() != 1
            || self.invocation.as_ref().map(|v| v.status.cu_deviceptr()) != Some(address)
            || !self.plan.admits(view)
        {
            return Err(invalid("invocation status or composed view mismatch"));
        }
        Ok(())
    }
    /// Submit the original plan's default cohort. Explicit begin/finish required.
    pub fn execute(
        &mut self,
        op: &CudaOperators,
        b: PagedBf16GqaBuffers<'_>,
        layout: PagedBf16CausalGqaLayout,
        append: bool,
    ) -> Result<()> {
        // No clone/allocation of the checked view during submission.
        self.execute_groups(op, b, layout, self.default_view.groups(), append)
    }
    /// Submit a prevalidated, prewarmed shape cohort within the maximum panels.
    /// No allocation/sync/workspace rebinding or matrix padding. Error latches
    /// prevent all cache/output writes; cuBLAS may still write private scratch.
    pub fn execute_view(
        &mut self,
        op: &CudaOperators,
        b: PagedBf16GqaBuffers<'_>,
        layout: PagedBf16CausalGqaLayout,
        view: &Bf16AttentionView,
        append: bool,
    ) -> Result<()> {
        if !self.plan.admits(view) {
            return Err(invalid("unprepared view"));
        }
        self.execute_groups(op, b, layout, view.groups(), append)
    }
    fn execute_groups(
        &self,
        op: &CudaOperators,
        b: PagedBf16GqaBuffers<'_>,
        layout: PagedBf16CausalGqaLayout,
        groups: &[Bf16AttentionGroup],
        append: bool,
    ) -> Result<()> {
        let stream = self.owner(op)?;
        if self.invocation.as_ref().map(|v| v.status.cu_deviceptr())
            != Some(b.status.as_device_buffer().cu_deviceptr())
        {
            return Err(invalid("begin_invocation required with this sticky status"));
        }
        let mut a = arguments(op, &b, layout, &self.plan, groups, append)?;
        let storage = self.storage.as_ref().unwrap();
        a.stream = stream.cu_stream() as usize as u64;
        let p = NativePlan::from(&self.plan);
        let result = op.submit_operator(u64::from(append) + 7 * groups.len() as u64, |s| {
            cu(s.context().bind_to_thread())?;
            status(unsafe {
                ferrule_bf16_attention_launch(
                    &a,
                    &p,
                    groups.as_ptr(),
                    groups.len() as u32,
                    storage.cu_deviceptr(),
                    storage.num_bytes() as u64,
                    self.handle,
                )
            })
        });
        if result.is_err() {
            self.stream.context().mark_graph_unknown();
        }
        result
    }
}
impl Drop for CudaPreparedBf16Attention {
    fn drop(&mut self) {
        if self.is_poisoned()
            || self.stream.context().is_capturing()
            || self.stream.context().bind_to_thread().is_err()
            || self.stream.synchronize().is_err()
        {
            self.quarantine();
            std::mem::forget(self.admission.take());
            std::mem::forget(self.storage.take());
            std::mem::forget(Arc::clone(&self.stream));
            return;
        }
        if self.handle != 0 && unsafe { ferrule_bf16_attention_destroy(self.handle) } != 0 {
            self.quarantine();
            std::mem::forget(self.admission.take());
            std::mem::forget(self.storage.take());
            std::mem::forget(Arc::clone(&self.stream));
        }
        if self.admission.is_some() {
            drop(self.storage.take());
            if self
                .stream
                .context()
                .synchronize_registered_streams()
                .is_err()
                || self.stream.context().trim_allocator().is_err()
            {
                self.quarantine();
                std::mem::forget(self.admission.take());
                std::mem::forget(Arc::clone(&self.stream));
            }
        }
    }
}
fn ptr<T: DeviceCopy>(op: &CudaOperators, b: &CudaTypedBuffer<T>, bytes: usize) -> Result<u64> {
    op.check_buffer_owner(b.as_device_buffer(), "prepared attention")?;
    if b.as_device_buffer().num_bytes() < bytes {
        return Err(invalid("operand extent"));
    }
    Ok(b.as_device_buffer().cu_deviceptr())
}
fn arguments(
    op: &CudaOperators,
    b: &PagedBf16GqaBuffers<'_>,
    l: PagedBf16CausalGqaLayout,
    p: &Bf16AttentionPlan,
    groups: &[Bf16AttentionGroup],
    append: bool,
) -> Result<TransformerArgs> {
    l.validate_metadata_lengths(
        b.block_slots.len(),
        b.block_offsets.len(),
        b.row_sequence_ids.len(),
        b.row_positions.len(),
        b.row_kv_lens.len(),
    )?;
    if l.rows != groups.iter().map(|g| g.rows as usize).sum::<usize>()
        || l.sequences != groups.len()
        || l.q_heads != p.heads()
        || l.kv_heads != p.kv_heads()
        || l.head_dim != p.head_dim()
    {
        return Err(invalid("layout/plan mismatch"));
    }
    for n in [l.page_tokens, l.layer_index, l.layer_count] {
        u32::try_from(n).map_err(|_| invalid("native dimension overflow"))?;
    }
    disjoint(
        &[
            Range::of(b.query)?,
            Range::of(b.append_key)?,
            Range::of(b.append_value)?,
            Range::of(b.block_slots)?,
            Range::of(b.block_offsets)?,
            Range::of(b.row_sequence_ids)?,
            Range::of(b.row_positions)?,
            Range::of(b.row_kv_lens)?,
        ],
        &[
            Range::of(b.key_cache)?,
            Range::of(b.value_cache)?,
            Range::of(b.output)?,
            Range::of(b.status)?,
        ],
    )?;
    Ok(TransformerArgs {
        kind: if append {
            TRANSFORMER_PAGED_BF16_APPEND_CAUSAL_GQA
        } else {
            TRANSFORMER_PAGED_BF16_CAUSAL_GQA
        },
        flags: 1,
        rows: l.rows as u32,
        sequences: l.sequences as u32,
        q_heads: l.q_heads as u32,
        kv_heads: l.kv_heads as u32,
        head_dim: l.head_dim as u32,
        page_tokens: l.page_tokens as u32,
        layer_index: l.layer_index as u32,
        layer_count: l.layer_count as u32,
        softmax_scale: l.softmax_scale,
        query_f32: ptr(op, b.query, l.query.required_bytes()?)?,
        query_bytes: b.query.as_device_buffer().num_bytes() as u64,
        query_row_stride_bytes: l.query.row_stride_bytes as u64,
        query_head_stride_bytes: l.query.head_stride_bytes as u64,
        append_key_bf16: ptr(
            op,
            b.append_key,
            if append {
                l.append_key.required_bytes()?
            } else {
                0
            },
        )?,
        append_key_bytes: b.append_key.as_device_buffer().num_bytes() as u64,
        append_key_row_stride_bytes: l.append_key.row_stride_bytes as u64,
        append_key_head_stride_bytes: l.append_key.head_stride_bytes as u64,
        append_value_bf16: ptr(
            op,
            b.append_value,
            if append {
                l.append_value.required_bytes()?
            } else {
                0
            },
        )?,
        append_value_bytes: b.append_value.as_device_buffer().num_bytes() as u64,
        append_value_row_stride_bytes: l.append_value.row_stride_bytes as u64,
        append_value_head_stride_bytes: l.append_value.head_stride_bytes as u64,
        key_cache_bf16: ptr(op, b.key_cache, l.key_cache.capacity_bytes)?,
        key_cache_bytes: l.key_cache.capacity_bytes as u64,
        key_slot_stride_bytes: l.key_cache.slot_stride_bytes as u64,
        key_layer_stride_bytes: l.key_cache.layer_stride_bytes as u64,
        key_token_stride_bytes: l.key_cache.token_stride_bytes as u64,
        key_head_stride_bytes: l.key_cache.head_stride_bytes as u64,
        value_cache_bf16: ptr(op, b.value_cache, l.value_cache.capacity_bytes)?,
        value_cache_bytes: l.value_cache.capacity_bytes as u64,
        value_slot_stride_bytes: l.value_cache.slot_stride_bytes as u64,
        value_layer_stride_bytes: l.value_cache.layer_stride_bytes as u64,
        value_token_stride_bytes: l.value_cache.token_stride_bytes as u64,
        value_head_stride_bytes: l.value_cache.head_stride_bytes as u64,
        block_slots_i32: ptr(op, b.block_slots, 0)?,
        block_slots_count: b.block_slots.len() as u64,
        block_offsets_i32: ptr(op, b.block_offsets, 0)?,
        block_offsets_count: b.block_offsets.len() as u64,
        row_sequence_ids_i32: ptr(op, b.row_sequence_ids, 0)?,
        row_sequence_ids_count: b.row_sequence_ids.len() as u64,
        row_positions_i32: ptr(op, b.row_positions, 0)?,
        row_positions_count: b.row_positions.len() as u64,
        row_kv_lens_i32: ptr(op, b.row_kv_lens, 0)?,
        row_kv_lens_count: b.row_kv_lens.len() as u64,
        output_f32: ptr(op, b.output, l.output.required_bytes()?)?,
        output_bytes: b.output.as_device_buffer().num_bytes() as u64,
        output_row_stride_bytes: l.output.row_stride_bytes as u64,
        output_head_stride_bytes: l.output.head_stride_bytes as u64,
        status_i32: ptr(op, b.status, 4)?,
        status_count: b.status.len() as u64,
        ..Default::default()
    })
}

#[cfg(test)]
#[path = "../../../../tests/attention_bf16/owner.rs"]
mod tests;
