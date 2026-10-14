//! Opt-in bounded chunk GDN. No model dispatch/default changes.
//!
//! All allocation is caller-owned. Submission may partially update state/output
//! and scratch before failing: retain every operand through the owner's stream
//! retirement/fence, quarantine on Unknown completion. No implicit reset, host
//! copies, allocation, synchronization, or rollback occurs in this operator.
//!
//! Numerical validity requires finite inputs/state and bounded intermediates at
//! every BF16/F32 materialization (including gates, prefix sums and matrix/state
//! products). Shape/ownership admission does not check values. Finite inputs alone
//! do not guarantee finite output/state: exp(A_log), prefix sums or reductions can
//! overflow; infinity differences and zero times NaN can contaminate padded tails
//! and carried state. No clamping, sanitization, numerical-error status, universal
//! finite-result guarantee or exceptional-value bit-parity guarantee is provided.
use super::OperatorOwner;
use super::recurrent_bf16::{Bf16GatedDeltaNetBuffers, Range, disjoint};
use crate::cuda::ffi::core::recurrent_bf16_chunk::{
    Bf16ChunkArgs, ferrule_core_bf16_chunk_prefill_launch,
};
use crate::cuda::operator_owner::{CudaF32Buffer, CudaOperators, CudaTypedBuffer, cu};
use crate::cuda::runtime::DeviceCopy;
use crate::cuda_plan::recurrent_bf16::invalid;
pub use crate::cuda_plan::{BF16_GDN_CHUNK_ROWS, Bf16ChunkExecution, Bf16ChunkPlan};
use ferrule_common::Result;

/// A typed, exclusively borrowed scratch allocation for this exact plan.
/// Creation does not allocate, synchronize, or grant cross-owner reuse.
pub struct CudaBf16ChunkWorkspace<'a> {
    buffer: &'a mut CudaF32Buffer,
    plan: Bf16ChunkPlan,
}
impl<'a> CudaBf16ChunkWorkspace<'a> {
    pub fn new(buffer: &'a mut CudaF32Buffer, plan: Bf16ChunkPlan) -> Result<Self> {
        if plan.workspace_elements() == 0 || buffer.len() != plan.workspace_elements() {
            return Err(invalid(
                "chunk workspace must have the exact planned F32 extent",
            ));
        }
        Ok(Self { buffer, plan })
    }
}
fn buffer<T: DeviceCopy>(
    owner: &CudaOperators,
    b: &CudaTypedBuffer<T>,
    len: usize,
    name: &str,
) -> Result<()> {
    owner.check_buffer_owner(b.as_device_buffer(), name)?;
    if b.len() != len {
        return Err(invalid(format!("{name} extent mismatch")));
    }
    Ok(())
}
impl CudaOperators {
    /// Explicit backend algorithm selection, never a new model precision profile.
    /// CachedRecurrent takes no workspace. PrefillChunk requires full workspace,
    /// including one row; construct it with Bf16ChunkPlan::new_prefill.
    ///
    /// Callers requiring finite results must ensure bounded BF16/F32 intermediates;
    /// finite operands alone are insufficient. Values are not checked or sanitized,
    /// and successful submission does not establish numerical validity. See the
    /// module's arithmetic policy; execution kind, not row count, selects arithmetic.
    pub fn gated_delta_net_bf16_chunked_into(
        &self,
        b: Bf16GatedDeltaNetBuffers<'_>,
        workspace: Option<CudaBf16ChunkWorkspace<'_>>,
        plan: Bf16ChunkPlan,
    ) -> Result<()> {
        let l = plan.layout();
        if plan.execution() == Bf16ChunkExecution::CachedRecurrent {
            if workspace.is_some() {
                return Err(invalid("single-row GDN takes no chunk workspace"));
            }
            return self.gated_delta_net_bf16_into(b, l);
        }
        let workspace = workspace.ok_or_else(|| invalid("prefill chunk workspace missing"))?;
        if workspace.plan != plan {
            return Err(invalid("chunk workspace plan mismatch"));
        }
        for (v, n, name) in [
            (b.qkv, l.qkv_elements()?, "chunk QKV"),
            (b.a, l.gate_elements()?, "chunk a"),
            (b.b, l.gate_elements()?, "chunk b"),
            (b.a_log, l.value_heads, "chunk A_log"),
            (b.dt_bias, l.value_heads, "chunk dt_bias"),
            (&*b.output, l.output_elements()?, "chunk output"),
        ] {
            buffer(self, v, n, name)?;
        }
        buffer(self, b.state, l.state_elements()?, "chunk F32 state")?;
        buffer(
            self,
            workspace.buffer,
            plan.workspace_elements(),
            "chunk F32 workspace",
        )?;
        disjoint(
            &[
                Range::of(b.qkv)?,
                Range::of(b.a)?,
                Range::of(b.b)?,
                Range::of(b.a_log)?,
                Range::of(b.dt_bias)?,
            ],
            &[
                Range::of(b.state)?,
                Range::of(b.output)?,
                Range::of(workspace.buffer)?,
            ],
        )?;
        let mut args = Bf16ChunkArgs {
            rows: l.rows as u32,
            key_heads: l.key_heads as u32,
            value_heads: l.value_heads as u32,
            key_dim: l.key_dim as u32,
            value_dim: l.value_dim as u32,
            qkv_bf16: b.qkv.as_device_buffer().cu_deviceptr(),
            a_bf16: b.a.as_device_buffer().cu_deviceptr(),
            b_bf16: b.b.as_device_buffer().cu_deviceptr(),
            a_log_bf16: b.a_log.as_device_buffer().cu_deviceptr(),
            dt_bias_bf16: b.dt_bias.as_device_buffer().cu_deviceptr(),
            state_f32: b.state.as_device_buffer().cu_deviceptr(),
            output_bf16: b.output.as_device_buffer().cu_deviceptr(),
            workspace_f32: workspace.buffer.as_device_buffer().cu_deviceptr(),
            workspace_elements: plan.workspace_elements() as u64,
            ..Default::default()
        };
        self.submit_operator(plan.launch_count(), |stream| {
            cu(stream.context().bind_to_thread())?;
            args.stream = stream.cu_stream() as usize as u64;
            // SAFETY: typed extents/owners and all writable ranges are validated.
            let status = unsafe { ferrule_core_bf16_chunk_prefill_launch(&args) };
            if status == 0 {
                Ok(())
            } else {
                Err(invalid(format!(
                    "chunk launch failed: {status}; submission may be partial; retain state and workspace until completion is proved"
                )))
            }
        })
    }
}
