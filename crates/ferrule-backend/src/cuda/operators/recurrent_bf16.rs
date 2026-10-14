//! HF native eager BF16 boundaries, not an all-BF16 recurrence or a model profile.
//!
//! One chronological sequence per call: rows=1 decode, rows>1 prefill/continuation.
//! Conv history is BF16; recurrent S is F32. Initialize/copy/checkpoint both with
//! their actual dtypes. No implicit resets, allocations, conversion scratch,
//! copies or synchronizations occur in these operations. Read buffers may alias;
//! all writable byte ranges must be disjoint from every other operand.
//!
//! Submission is not completion or a rollback guarantee. The model owner retains
//! state custody and orders reuse on the compute stream, or proves completion
//! before publishing to another owner. Unknown completion must quarantine state;
//! existing allocator retirement/fencing applies unchanged.
use crate::cuda::ffi::core::recurrent_bf16::{
    Bf16ConvArgs, Bf16DeltaArgs, Bf16GatedNormArgs, ferrule_core_bf16_conv_launch,
    ferrule_core_bf16_delta_launch, ferrule_core_bf16_gated_norm_launch,
};
use crate::cuda::operator_owner::{
    CudaBf16Buffer, CudaF32Buffer, CudaOperators, CudaTypedBuffer, cu,
};
use crate::cuda::operators::OperatorOwner;
use crate::cuda::runtime::DeviceCopy;
pub use crate::cuda_plan::Bf16RecurrentStorage;
use crate::cuda_plan::recurrent_bf16::invalid;
use crate::cuda_plan::{CausalConv1dLayout, F32RowsLayout, GatedDeltaNetLayout};
use ferrule_common::Result;

/// Same packed/channel-major layout as `CausalConv1dLayout`, actual 2-byte words.
pub struct Bf16CausalConv1dBuffers<'a> {
    pub input: &'a CudaBf16Buffer,
    pub weight: &'a CudaBf16Buffer,
    pub bias: Option<&'a CudaBf16Buffer>,
    pub history: &'a mut CudaBf16Buffer,
    pub output: &'a mut CudaBf16Buffer,
}
/// QKV/a/b and native HF BF16-loaded parameters remain BF16 in storage.
/// Gates/decay promote to F32, beta and L2 retain HF eager BF16 boundaries.
/// S is F32 `[value_heads,key_dim,value_dim]`; output is BF16.
pub struct Bf16GatedDeltaNetBuffers<'a> {
    pub qkv: &'a CudaBf16Buffer,
    pub a: &'a CudaBf16Buffer,
    pub b: &'a CudaBf16Buffer,
    pub a_log: &'a CudaBf16Buffer,
    pub dt_bias: &'a CudaBf16Buffer,
    pub state: &'a mut CudaF32Buffer,
    pub output: &'a mut CudaBf16Buffer,
}

#[derive(Clone, Copy)]
pub(super) struct Range {
    start: u64,
    end: u64,
}
impl Range {
    fn new(start: u64, elements: usize, element_bytes: usize) -> Result<Self> {
        let bytes = elements
            .checked_mul(element_bytes)
            .and_then(|v| u64::try_from(v).ok())
            .ok_or_else(|| invalid("byte extent overflow"))?;
        let end = start
            .checked_add(bytes)
            .ok_or_else(|| invalid("address overflow"))?;
        Ok(Self { start, end })
    }
    pub(super) fn of<T: DeviceCopy>(buffer: &CudaTypedBuffer<T>) -> Result<Self> {
        Self::new(
            buffer.as_device_buffer().cu_deviceptr(),
            buffer.len(),
            size_of::<T>(),
        )
    }
}
pub(super) fn disjoint(reads: &[Range], writes: &[Range]) -> Result<()> {
    for (i, w) in writes.iter().enumerate() {
        for r in reads.iter().chain(writes[..i].iter()) {
            if w.start < r.end && r.start < w.end {
                return Err(invalid("overlapping writable buffer ranges"));
            }
        }
    }
    Ok(())
}
impl CudaOperators {
    fn bf16_recurrent_buffer<T: DeviceCopy>(
        &self,
        b: &CudaTypedBuffer<T>,
        len: usize,
        name: &str,
    ) -> Result<()> {
        self.check_buffer_owner(b.as_device_buffer(), name)?;
        if b.len() != len {
            return Err(invalid(format!(
                "{name} length {}, expected {len}",
                b.len()
            )));
        }
        Ok(())
    }
    fn bf16_recurrent_submit(&self, launch: impl FnOnce(u64) -> i32) -> Result<()> {
        self.submit_operator(1, |stream| {
            cu(stream.context().bind_to_thread())?;
            let status = launch(stream.cu_stream() as usize as u64);
            if status == 0 {
                Ok(())
            } else {
                Err(invalid(format!("native launch failed: {status}")))
            }
        })
    }
    /// HF eager Conv1d BF16 output boundary, then SiLU BF16 output boundary.
    /// Updates raw BF16 history after each row, oldest first, including prefill.
    pub fn causal_depthwise_conv1d_silu_bf16_into(
        &self,
        b: Bf16CausalConv1dBuffers<'_>,
        l: CausalConv1dLayout,
    ) -> Result<()> {
        l.validate()?;
        for (v, n, name) in [
            (b.input, l.row_elements()?, "conv input"),
            (b.weight, l.history_elements()?, "conv weight"),
            (&*b.history, l.history_elements()?, "conv history"),
            (&*b.output, l.row_elements()?, "conv output"),
        ] {
            self.bf16_recurrent_buffer(v, n, name)?;
        }
        let writes = [Range::of(b.history)?, Range::of(b.output)?];
        if let Some(bias) = b.bias {
            self.bf16_recurrent_buffer(bias, l.channels, "conv bias")?;
            disjoint(&[Range::of(bias)?], &writes)?;
        }
        disjoint(&[Range::of(b.input)?, Range::of(b.weight)?], &writes)?;
        let mut args = Bf16ConvArgs {
            rows: l.rows as u32,
            channels: l.channels as u32,
            kernel_size: l.kernel_size as u32,
            input_bf16: b.input.as_device_buffer().cu_deviceptr(),
            weight_bf16: b.weight.as_device_buffer().cu_deviceptr(),
            bias_bf16: b.bias.map_or(0, |v| v.as_device_buffer().cu_deviceptr()),
            history_bf16: b.history.as_device_buffer().cu_deviceptr(),
            output_bf16: b.output.as_device_buffer().cu_deviceptr(),
            ..Default::default()
        };
        self.bf16_recurrent_submit(|stream| {
            args.stream = stream;
            // SAFETY: exact typed extents, owner and disjoint write ranges checked above.
            unsafe { ferrule_core_bf16_conv_launch(&args) }
        })
    }
    /// HF eager Q/K normalization and beta BF16 boundaries; F32 decay and S scan.
    /// Prefill uses the chronological recurrent equation, not HF's chunked GEMMs;
    /// floating-point reduction order can differ, but no dtype boundary is omitted.
    pub fn gated_delta_net_bf16_into(
        &self,
        b: Bf16GatedDeltaNetBuffers<'_>,
        l: GatedDeltaNetLayout,
    ) -> Result<()> {
        l.validate()?;
        for (v, n, name) in [
            (b.qkv, l.qkv_elements()?, "DeltaNet QKV"),
            (b.a, l.gate_elements()?, "DeltaNet a"),
            (b.b, l.gate_elements()?, "DeltaNet b"),
            (b.a_log, l.value_heads, "DeltaNet A_log"),
            (b.dt_bias, l.value_heads, "DeltaNet dt_bias"),
            (&*b.output, l.output_elements()?, "DeltaNet output"),
        ] {
            self.bf16_recurrent_buffer(v, n, name)?;
        }
        self.bf16_recurrent_buffer(b.state, l.state_elements()?, "DeltaNet F32 state")?;
        disjoint(
            &[
                Range::of(b.qkv)?,
                Range::of(b.a)?,
                Range::of(b.b)?,
                Range::of(b.a_log)?,
                Range::of(b.dt_bias)?,
            ],
            &[Range::of(b.state)?, Range::of(b.output)?],
        )?;
        let mut args = Bf16DeltaArgs {
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
            ..Default::default()
        };
        self.bf16_recurrent_submit(|stream| {
            args.stream = stream;
            // SAFETY: BF16 operands and F32 state have validated extents/owner/disjoint writes.
            unsafe { ferrule_core_bf16_delta_launch(&args) }
        })
    }
    /// Ordinary (not offset) per-value-head gated RMSNorm. Set rows=tokens*heads.
    /// F32 normalization -> BF16 -> BF16 affine -> F32 SiLU(gate) product -> BF16.
    pub fn gated_rms_norm_bf16_into(
        &self,
        input: &CudaBf16Buffer,
        weight: &CudaBf16Buffer,
        gate: &CudaBf16Buffer,
        output: &mut CudaBf16Buffer,
        l: F32RowsLayout,
        epsilon: f32,
    ) -> Result<()> {
        let n = l.elements()?;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(invalid("epsilon must be finite and positive"));
        }
        for (v, len, name) in [
            (input, n, "gated norm input"),
            (weight, l.width, "gated norm weight"),
            (gate, n, "gated norm gate"),
            (&*output, n, "gated norm output"),
        ] {
            self.bf16_recurrent_buffer(v, len, name)?;
        }
        disjoint(
            &[Range::of(input)?, Range::of(weight)?, Range::of(gate)?],
            &[Range::of(output)?],
        )?;
        let mut args = Bf16GatedNormArgs {
            rows: l.rows as u32,
            width: l.width as u32,
            epsilon,
            input_bf16: input.as_device_buffer().cu_deviceptr(),
            weight_bf16: weight.as_device_buffer().cu_deviceptr(),
            gate_bf16: gate.as_device_buffer().cu_deviceptr(),
            output_bf16: output.as_device_buffer().cu_deviceptr(),
            ..Default::default()
        };
        self.bf16_recurrent_submit(|stream| {
            args.stream = stream;
            // SAFETY: all typed buffers validated before submission.
            unsafe { ferrule_core_bf16_gated_norm_launch(&args) }
        })
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/cuda/recurrent_bf16.rs"]
mod tests;
