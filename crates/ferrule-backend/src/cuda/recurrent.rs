//! F32 GPU-resident recurrence and gating primitives (no model dependency).
//!
//! Conv and DeltaNet consume one sequence in chronological row order; rows=1
//! is decode, rows>1 is prefill/continuation. Loop over sequences with separate
//! buffers (or use existing GPU range copies to/from a packed batch). History
//! and S are updated in place; initialize them explicitly, never implicitly reset.
//!
//! All operations enqueue on the owner's compute stream. Success means submitted,
//! not completed: use the existing compute event/stream completion API before
//! publishing state elsewhere. On uncertain completion do not reuse state as if
//! rolled back. Allocator stream-fenced retirement/quarantine remains unchanged.
//! There are no allocations, CPU tensor copies, or synchronizations here.

use super::{CudaF32Buffer, CudaOperators};
use crate::cuda::ffi::core::{
    ConvArgs, DATA_GATE_ROW_BROADCAST, DATA_QUERY_GATE_SPLIT, DATA_SIGMOID_GATE, DATA_SILU_GATE,
    DataArgs, DeltaArgs, NORM_OFFSET_AFFINE_F32, NormArgs,
};
use ferrule_common::{Error, Result};

fn invalid(message: impl Into<String>) -> Error {
    Error::Internal {
        message: format!("recurrent F32 CUDA: {}", message.into()),
    }
}

fn elements(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |n, &d| {
        let n = n.checked_mul(d).ok_or_else(|| invalid("shape overflow"))?;
        if d == 0 || n > i32::MAX as usize {
            return Err(invalid(
                "dimensions and element counts must be in 1..=i32::MAX",
            ));
        }
        Ok(n)
    })
}

/// Packed input/output `[rows, channels]`, weight/history `[channels, kernel_size]`.
/// History contains the last K **raw** inputs, oldest first; the oldest slot is
/// shifted out before each convolution. Kernel weights use PyTorch cross-correlation
/// order (last weight multiplies the current input). Bias is optional `[channels]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CausalConv1dLayout {
    pub rows: usize,
    pub channels: usize,
    pub kernel_size: usize,
}
impl CausalConv1dLayout {
    pub fn validate(self) -> Result<()> {
        self.row_elements()?;
        self.history_elements()?;
        Ok(())
    }
    pub fn row_elements(self) -> Result<usize> {
        elements(&[self.rows, self.channels])
    }
    pub fn history_elements(self) -> Result<usize> {
        elements(&[self.channels, self.kernel_size])
    }
}

pub struct CausalConv1dBuffers<'a> {
    pub input: &'a CudaF32Buffer,
    pub weight: &'a CudaF32Buffer,
    pub bias: Option<&'a CudaF32Buffer>,
    pub history: &'a mut CudaF32Buffer,
    pub output: &'a mut CudaF32Buffer,
}

/// Packed post-convolution QKV `[rows, 2*key_heads*key_dim + value_heads*value_dim]`:
/// each row is all Q heads, then all K heads, then all V heads (not head-interleaved).
/// a/b are raw projection logits `[rows,value_heads]`, A_log/dt_bias `[value_heads]`.
/// S is `[value_heads,key_dim,value_dim]`, output `[rows,value_heads,value_dim]`.
/// Q/K head h is repeated contiguously `value_heads/key_heads` times.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatedDeltaNetLayout {
    pub rows: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
}
impl GatedDeltaNetLayout {
    pub fn validate(self) -> Result<()> {
        self.qkv_elements()?;
        self.gate_elements()?;
        self.state_elements()?;
        self.output_elements()?;
        if !self.value_heads.is_multiple_of(self.key_heads) {
            return Err(invalid(
                "value_heads must be an integer multiple of key_heads",
            ));
        }
        Ok(())
    }
    pub fn qkv_width(self) -> Result<usize> {
        let qk = elements(&[2, self.key_heads, self.key_dim])?;
        let v = elements(&[self.value_heads, self.value_dim])?;
        elements(&[qk
            .checked_add(v)
            .ok_or_else(|| invalid("QKV width overflow"))?])
    }
    pub fn qkv_elements(self) -> Result<usize> {
        elements(&[self.rows, self.qkv_width()?])
    }
    pub fn gate_elements(self) -> Result<usize> {
        elements(&[self.rows, self.value_heads])
    }
    pub fn state_elements(self) -> Result<usize> {
        elements(&[self.value_heads, self.key_dim, self.value_dim])
    }
    pub fn output_elements(self) -> Result<usize> {
        elements(&[self.rows, self.value_heads, self.value_dim])
    }
}

pub struct GatedDeltaNetBuffers<'a> {
    pub qkv: &'a CudaF32Buffer,
    pub a: &'a CudaF32Buffer,
    pub b: &'a CudaF32Buffer,
    pub a_log: &'a CudaF32Buffer,
    pub dt_bias: &'a CudaF32Buffer,
    pub state: &'a mut CudaF32Buffer,
    pub output: &'a mut CudaF32Buffer,
}

/// Input `[rows,heads,2,head_dim]` -> separate `[rows,heads,head_dim]` Q and gate.
/// The split is per head, **not** between two halves of the entire projected row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryGateLayout {
    pub rows: usize,
    pub heads: usize,
    pub head_dim: usize,
}
impl QueryGateLayout {
    pub fn validate(self) -> Result<()> {
        self.input_elements().map(|_| ())
    }
    pub fn input_elements(self) -> Result<usize> {
        elements(&[self.rows, self.heads, 2, self.head_dim])
    }
    pub fn output_elements(self) -> Result<usize> {
        elements(&[self.rows, self.heads, self.head_dim])
    }
}

/// Packed `[rows,width]`. For per-head norm use rows=tokens*heads, width=head_dim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct F32RowsLayout {
    pub rows: usize,
    pub width: usize,
}
impl F32RowsLayout {
    pub fn validate(self) -> Result<()> {
        self.elements().map(|_| ())
    }
    pub fn elements(self) -> Result<usize> {
        elements(&[self.rows, self.width])
    }
}

/// Gate indexing for packed F32 input/output `[rows,width]`.
/// Read inputs may alias each other; output must be disjoint from both reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F32GateLayout {
    /// One gate logit per input element: gate `[rows,width]`.
    Elementwise(F32RowsLayout),
    /// One gate logit per row: gate `[rows,1]`, broadcast over width.
    RowBroadcast(F32RowsLayout),
}

impl From<F32RowsLayout> for F32GateLayout {
    fn from(rows: F32RowsLayout) -> Self {
        Self::Elementwise(rows)
    }
}

impl F32GateLayout {
    pub const fn rows_layout(self) -> F32RowsLayout {
        match self {
            Self::Elementwise(rows) | Self::RowBroadcast(rows) => rows,
        }
    }

    pub fn validate(self) -> Result<()> {
        self.rows_layout().validate()
    }

    pub fn input_elements(self) -> Result<usize> {
        self.rows_layout().elements()
    }

    pub fn gate_elements(self) -> Result<usize> {
        let len = self.input_elements()?;
        Ok(match self {
            Self::Elementwise(_) => len,
            Self::RowBroadcast(rows) => rows.rows,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateActivation {
    Sigmoid,
    Silu,
}

impl CudaOperators {
    fn recurrent_buffer(&self, buffer: &CudaF32Buffer, len: usize, label: &str) -> Result<()> {
        self.check_buffer_owner(&buffer.buffer, label)?;
        if buffer.len() != len {
            return Err(invalid(format!(
                "{label} length {}, expected {len}",
                buffer.len()
            )));
        }
        Ok(())
    }

    // Reads may alias each other, but every write range must be disjoint from
    // reads and other writes. This also protects future typed allocation views.
    fn recurrent_no_alias(reads: &[&CudaF32Buffer], writes: &[&CudaF32Buffer]) -> Result<()> {
        let range = |b: &CudaF32Buffer| -> Result<(u64, u64)> {
            let start = b.buffer.cu_deviceptr();
            let bytes = (b.len() as u64)
                .checked_mul(4)
                .ok_or_else(|| invalid("byte overflow"))?;
            Ok((
                start,
                start
                    .checked_add(bytes)
                    .ok_or_else(|| invalid("address overflow"))?,
            ))
        };
        for (i, write) in writes.iter().enumerate() {
            let (start, end) = range(write)?;
            for other in reads.iter().chain(writes[..i].iter()) {
                let (other_start, other_end) = range(other)?;
                if start < other_end && other_start < end {
                    return Err(invalid("overlapping writable buffer ranges"));
                }
            }
        }
        Ok(())
    }

    /// Causal depthwise cross-correlation + SiLU, updating raw history in row order.
    pub fn causal_depthwise_conv1d_silu_into(
        &self,
        buffers: CausalConv1dBuffers<'_>,
        layout: CausalConv1dLayout,
    ) -> Result<()> {
        layout.validate()?;
        self.recurrent_buffer(buffers.input, layout.row_elements()?, "conv input")?;
        self.recurrent_buffer(buffers.weight, layout.history_elements()?, "conv weight")?;
        self.recurrent_buffer(buffers.history, layout.history_elements()?, "conv history")?;
        self.recurrent_buffer(buffers.output, layout.row_elements()?, "conv output")?;
        if let Some(bias) = buffers.bias {
            self.recurrent_buffer(bias, layout.channels, "conv bias")?;
            Self::recurrent_no_alias(&[bias], &[buffers.history, buffers.output])?;
        }
        Self::recurrent_no_alias(
            &[buffers.input, buffers.weight],
            &[buffers.history, buffers.output],
        )?;
        let args = ConvArgs {
            rows: layout.rows as u32,
            channels: layout.channels as u32,
            kernel_size: layout.kernel_size as u32,
            input: buffers.input.buffer.cu_deviceptr(),
            weight: buffers.weight.buffer.cu_deviceptr(),
            bias: buffers.bias.map_or(0, |b| b.buffer.cu_deviceptr()),
            history: buffers.history.buffer.cu_deviceptr(),
            output: buffers.output.buffer.cu_deviceptr(),
            ..Default::default()
        };
        self.launched(unsafe { self.module.causal_conv(&self.stream, args) })
    }

    /// Q/K L2 uses rsqrt(sum(x*x)+1e-6), Q scale=1/sqrt(dk).
    /// beta=sigmoid(b), g=-exp(A_log)*softplus(a+dt_bias).
    /// S=exp(g)*S; delta=beta*(v-k^T*S); S+=k*delta^T; out=q^T*S.
    pub fn gated_delta_net_into(
        &self,
        buffers: GatedDeltaNetBuffers<'_>,
        layout: GatedDeltaNetLayout,
    ) -> Result<()> {
        layout.validate()?;
        self.recurrent_buffer(buffers.qkv, layout.qkv_elements()?, "DeltaNet QKV")?;
        self.recurrent_buffer(buffers.a, layout.gate_elements()?, "DeltaNet a")?;
        self.recurrent_buffer(buffers.b, layout.gate_elements()?, "DeltaNet b")?;
        self.recurrent_buffer(buffers.a_log, layout.value_heads, "DeltaNet A_log")?;
        self.recurrent_buffer(buffers.dt_bias, layout.value_heads, "DeltaNet dt_bias")?;
        self.recurrent_buffer(buffers.state, layout.state_elements()?, "DeltaNet state")?;
        self.recurrent_buffer(buffers.output, layout.output_elements()?, "DeltaNet output")?;
        Self::recurrent_no_alias(
            &[
                buffers.qkv,
                buffers.a,
                buffers.b,
                buffers.a_log,
                buffers.dt_bias,
            ],
            &[buffers.state, buffers.output],
        )?;
        let args = DeltaArgs {
            rows: layout.rows as u32,
            key_heads: layout.key_heads as u32,
            value_heads: layout.value_heads as u32,
            key_dim: layout.key_dim as u32,
            value_dim: layout.value_dim as u32,
            qkv: buffers.qkv.buffer.cu_deviceptr(),
            a: buffers.a.buffer.cu_deviceptr(),
            b: buffers.b.buffer.cu_deviceptr(),
            a_log: buffers.a_log.buffer.cu_deviceptr(),
            dt_bias: buffers.dt_bias.buffer.cu_deviceptr(),
            state: buffers.state.buffer.cu_deviceptr(),
            output: buffers.output.buffer.cu_deviceptr(),
            ..Default::default()
        };
        self.launched(unsafe { self.module.gated_delta(&self.stream, args) })
    }

    pub fn split_query_gate_f32_into(
        &self,
        packed: &CudaF32Buffer,
        query: &mut CudaF32Buffer,
        gate: &mut CudaF32Buffer,
        layout: QueryGateLayout,
    ) -> Result<()> {
        layout.validate()?;
        self.recurrent_buffer(packed, layout.input_elements()?, "packed query/gate")?;
        self.recurrent_buffer(query, layout.output_elements()?, "split query")?;
        self.recurrent_buffer(gate, layout.output_elements()?, "split gate")?;
        Self::recurrent_no_alias(&[packed], &[query, gate])?;
        let args = DataArgs {
            kind: DATA_QUERY_GATE_SPLIT,
            count: layout.output_elements()? as u32,
            width: layout.head_dim as u32,
            input0: packed.buffer.cu_deviceptr(),
            output0: query.buffer.cu_deviceptr(),
            output1: gate.buffer.cu_deviceptr(),
            ..Default::default()
        };
        self.launched(unsafe { self.module.recurrent_data(&self.stream, args) })
    }

    /// output = input * activation(gate), all F32; not a fused normalization.
    /// Passing `F32RowsLayout` retains elementwise gating. Use
    /// `F32GateLayout::RowBroadcast(rows)` for `[rows,1]` gate logits.
    ///
    /// Shape/owner/disjoint-write checks precede launch. Exactly one kernel is
    /// enqueued, without allocations, copies, or synchronization. No host-side
    /// finite-value scan or clamping: NaN/Inf follow the same arithmetic as the
    /// original elementwise sigmoid/SiLU kernel (including Inf * 0 -> NaN).
    /// DeltaNet gated norm uses ordinary rms_norm_f32_into then Silu here;
    /// it must NOT use the offset RMSNorm variant.
    pub fn elementwise_gate_f32_into(
        &self,
        input: &CudaF32Buffer,
        gate: &CudaF32Buffer,
        output: &mut CudaF32Buffer,
        layout: impl Into<F32GateLayout>,
        activation: GateActivation,
    ) -> Result<()> {
        let layout = layout.into();
        let len = layout.input_elements()?;
        let gate_len = layout.gate_elements()?;
        let rows = layout.rows_layout();
        self.recurrent_buffer(input, len, "gate input")?;
        self.recurrent_buffer(gate, gate_len, "gate logits")?;
        self.recurrent_buffer(output, len, "gate output")?;
        Self::recurrent_no_alias(&[input, gate], &[output])?;
        let args = DataArgs {
            kind: match activation {
                GateActivation::Sigmoid => DATA_SIGMOID_GATE,
                GateActivation::Silu => DATA_SILU_GATE,
            },
            count: len as u32,
            rows: rows.rows as u32,
            width: rows.width as u32,
            flags: match layout {
                F32GateLayout::Elementwise(_) => 0,
                F32GateLayout::RowBroadcast(_) => DATA_GATE_ROW_BROADCAST,
            },
            input0: input.buffer.cu_deviceptr(),
            input1: gate.buffer.cu_deviceptr(),
            output0: output.buffer.cu_deviceptr(),
            ..Default::default()
        };
        self.launched(unsafe { self.module.recurrent_data(&self.stream, args) })
    }

    /// output = x * rsqrt(mean(x*x)+epsilon) * (weight+1). Ordinary norm is unchanged.
    pub fn offset_rms_norm_f32_into(
        &self,
        input: &CudaF32Buffer,
        weight: &CudaF32Buffer,
        output: &mut CudaF32Buffer,
        layout: F32RowsLayout,
        epsilon: f32,
    ) -> Result<()> {
        let len = layout.elements()?;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(invalid("RMS epsilon must be finite and positive"));
        }
        self.recurrent_buffer(input, len, "offset norm input")?;
        self.recurrent_buffer(weight, layout.width, "offset norm weight")?;
        self.recurrent_buffer(output, len, "offset norm output")?;
        Self::recurrent_no_alias(&[input, weight], &[output])?;
        let args = NormArgs {
            kind: NORM_OFFSET_AFFINE_F32,
            rows: layout.rows as u32,
            width: layout.width as u32,
            epsilon,
            input: input.buffer.cu_deviceptr(),
            weight: weight.buffer.cu_deviceptr(),
            output: output.buffer.cu_deviceptr(),
            ..Default::default()
        };
        self.launched(unsafe { self.module.standard_norm(&self.stream, args) })
    }
}

#[cfg(test)]
mod gate_tests {
    use super::*;

    #[test]
    #[ignore = "requires actual CUDA GPU; checked allocation views, no fabricated Rust aliases"]
    fn row_broadcast_gate_rejects_write_aliases_but_allows_read_aliases() {
        let op = CudaOperators::new_on_device(0).unwrap();
        let storage = op.upload_f32_buffer(&[0.37; 64]).unwrap();
        let layout = F32GateLayout::RowBroadcast(F32RowsLayout { rows: 3, width: 5 });
        // Separate typed handles retain the shared allocation and preserve its
        // owner. This covers future views without creating &/&mut Rust aliases.
        let view = |offset, len| {
            CudaF32Buffer::from_device_buffer(storage.buffer.slice(offset, len).unwrap())
        };
        for (input_offset, gate_offset, output_offset) in
            [(0, 32, 0), (0, 32, 1), (0, 32, 31), (0, 32, 34)]
        {
            let input = view(input_offset, 15);
            let gate = view(gate_offset, 3);
            let mut output = view(output_offset, 15);
            for activation in [GateActivation::Sigmoid, GateActivation::Silu] {
                op.reset_counters();
                let error = op
                    .elementwise_gate_f32_into(&input, &gate, &mut output, layout, activation)
                    .unwrap_err();
                assert!(
                    error
                        .to_string()
                        .contains("overlapping writable buffer ranges")
                );
                assert_eq!(op.counters(), Default::default());
                assert_eq!(op.download_f32_buffer(&storage).unwrap(), vec![0.37; 64]);
            }
        }
        // Read-only input and gate overlap; adjacent output does not overlap.
        let input = view(0, 15);
        let gate = view(2, 3);
        let mut output = view(15, 15);
        op.reset_counters();
        op.elementwise_gate_f32_into(&input, &gate, &mut output, layout, GateActivation::Sigmoid)
            .unwrap();
        let counters = op.counters();
        assert_eq!(counters.kernel_launches, 1);
        assert_eq!(counters.device_allocation_attempts, 0);
        assert_eq!(counters.device_to_host_copies, 0);
        assert_eq!(counters.host_to_device_copies, 0);
        assert_eq!(counters.stream_wide_syncs, 0);
        let expected = 0.37 / (1.0 + (-0.37f32).exp());
        assert!(
            op.download_f32_buffer(&output)
                .unwrap()
                .iter()
                .all(|v| (v - expected).abs() < 1e-6)
        );
        assert_eq!(op.download_f32_buffer(&input).unwrap(), vec![0.37; 15]);
    }

    #[test]
    fn row_broadcast_native_guard_rejects_invalid_layout_without_launch() {
        // All cases must fail in the native prelaunch guard, even without a GPU
        // context. Non-null sentinels must never reach the device kernel.
        let valid = DataArgs {
            kind: DATA_SIGMOID_GATE,
            count: 15,
            rows: 3,
            width: 5,
            flags: DATA_GATE_ROW_BROADCAST,
            input0: 4,
            input1: 8,
            output0: 12,
            ..Default::default()
        };
        for args in [
            DataArgs { rows: 0, ..valid },
            DataArgs { width: 0, ..valid },
            DataArgs { count: 0, ..valid },
            DataArgs { count: 14, ..valid },
            DataArgs {
                rows: u32::MAX,
                width: u32::MAX,
                ..valid
            },
            DataArgs {
                count: i32::MAX as u32 + 1,
                rows: 1,
                width: i32::MAX as u32 + 1,
                ..valid
            },
            DataArgs {
                flags: DATA_GATE_ROW_BROADCAST | 2,
                ..valid
            },
            DataArgs { input0: 0, ..valid },
            DataArgs { input1: 0, ..valid },
            DataArgs {
                output0: 0,
                ..valid
            },
        ] {
            for kind in [DATA_SIGMOID_GATE, DATA_SILU_GATE] {
                let args = DataArgs { kind, ..args };
                // SAFETY: invalid descriptors must be rejected before any pointer
                // access or CUDA launch; this exercises that native ABI contract.
                let status = unsafe { crate::cuda::ffi::core::ferrule_core_data_launch(&args) };
                assert_ne!(status, 0);
            }
        }
    }
}
