//! CPU operator-provider contract, reference fallback, and native dispatch.

use std::sync::OnceLock;

use ferrule_common::Result;

use super::attention::{PagedCausalGqa, PagedKvHistory, paged_causal_gqa};
use super::moe::{RouterRoutes, SwiGluRef, softmax_topk_routes, swiglu_rows, weighted_reduce};
use super::operators::{
    CpuExecutionPrecision, HostRows, LinearRef, LinearWeight, RopeRef, RotaryPairing, RotaryRegion,
    RowsArenaId, RowsShape, cpu_error, embedding_rows, linear_rows, residual_rows, rms_norm_rows,
    rope_rows,
};

const CAP_AVX2: u64 = 1 << 0;
const CAP_AVX512_BF16: u64 = 1 << 1;

/// Runtime CPU ISA and semantic native-operator availability.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CpuCapabilities {
    pub avx2: bool,
    pub avx512_bf16: bool,
    pub native_bf16_embedding: bool,
    pub native_bf16_linear: bool,
    pub native_rms_norm: bool,
    pub native_bf16_swiglu: bool,
    pub native_paged_gqa: bool,
    pub native_routed_moe: bool,
}

/// Model-neutral semantic CPU provider.
pub trait CpuOperatorProvider {
    fn name(&self) -> &'static str;
    fn capabilities(&self) -> CpuCapabilities;

    fn embedding(
        &self,
        embedding: LinearRef<'_>,
        token_ids: &[u32],
        arena: Option<RowsArenaId>,
        precision: CpuExecutionPrecision,
    ) -> Result<HostRows>;

    fn linear(
        &self,
        linear: LinearRef<'_>,
        input: &HostRows,
        arena: Option<RowsArenaId>,
        precision: CpuExecutionPrecision,
    ) -> Result<HostRows>;

    fn rms_norm(
        &self,
        input: &HostRows,
        weight: &[f32],
        epsilon: f32,
        heads: usize,
        arena: Option<RowsArenaId>,
        precision: CpuExecutionPrecision,
    ) -> Result<HostRows>;

    #[allow(clippy::too_many_arguments)]
    fn rope(
        &self,
        input: HostRows,
        table: RopeRef<'_>,
        pairing: RotaryPairing,
        region: RotaryRegion,
        heads: usize,
        head_dim: usize,
        positions: &[usize],
        precision: CpuExecutionPrecision,
    ) -> Result<HostRows>;

    fn paged_gqa(
        &self,
        history: &dyn PagedKvHistory,
        layer: usize,
        request: PagedCausalGqa<'_>,
        precision: CpuExecutionPrecision,
    ) -> Result<HostRows>;

    fn router(
        &self,
        logits: &HostRows,
        top_k: usize,
        route_scale: f32,
        precision: CpuExecutionPrecision,
    ) -> Result<RouterRoutes>;

    fn swiglu(
        &self,
        expert: SwiGluRef<'_>,
        input: &HostRows,
        arena: Option<RowsArenaId>,
        precision: CpuExecutionPrecision,
    ) -> Result<HostRows>;

    fn weighted_reduce(
        &self,
        output: &mut [f32],
        update: &[f32],
        route_weight: f32,
        precision: CpuExecutionPrecision,
    ) -> Result<()>;

    fn residual(
        &self,
        residual: HostRows,
        update: &HostRows,
        precision: CpuExecutionPrecision,
    ) -> Result<HostRows>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ReferenceCpuProvider;

impl CpuOperatorProvider for ReferenceCpuProvider {
    fn name(&self) -> &'static str {
        "reference-cpu"
    }

    fn capabilities(&self) -> CpuCapabilities {
        CpuCapabilities::default()
    }

    fn embedding(
        &self,
        embedding: LinearRef<'_>,
        token_ids: &[u32],
        arena: Option<RowsArenaId>,
        precision: CpuExecutionPrecision,
    ) -> Result<HostRows> {
        embedding_rows(embedding, token_ids, arena, precision)
    }

    fn linear(
        &self,
        linear: LinearRef<'_>,
        input: &HostRows,
        arena: Option<RowsArenaId>,
        precision: CpuExecutionPrecision,
    ) -> Result<HostRows> {
        linear_rows(linear, input, arena, precision)
    }

    fn rms_norm(
        &self,
        input: &HostRows,
        weight: &[f32],
        epsilon: f32,
        heads: usize,
        arena: Option<RowsArenaId>,
        precision: CpuExecutionPrecision,
    ) -> Result<HostRows> {
        rms_norm_rows(input, weight, epsilon, heads, arena, precision)
    }

    fn rope(
        &self,
        input: HostRows,
        table: RopeRef<'_>,
        pairing: RotaryPairing,
        region: RotaryRegion,
        heads: usize,
        head_dim: usize,
        positions: &[usize],
        precision: CpuExecutionPrecision,
    ) -> Result<HostRows> {
        rope_rows(
            input, table, pairing, region, heads, head_dim, positions, precision,
        )
    }

    fn paged_gqa(
        &self,
        history: &dyn PagedKvHistory,
        layer: usize,
        request: PagedCausalGqa<'_>,
        precision: CpuExecutionPrecision,
    ) -> Result<HostRows> {
        paged_causal_gqa(history, layer, request, precision)
    }

    fn router(
        &self,
        logits: &HostRows,
        top_k: usize,
        route_scale: f32,
        precision: CpuExecutionPrecision,
    ) -> Result<RouterRoutes> {
        softmax_topk_routes(logits, top_k, route_scale, precision)
    }

    fn swiglu(
        &self,
        expert: SwiGluRef<'_>,
        input: &HostRows,
        arena: Option<RowsArenaId>,
        precision: CpuExecutionPrecision,
    ) -> Result<HostRows> {
        swiglu_rows(expert, input, arena, precision)
    }

    fn weighted_reduce(
        &self,
        output: &mut [f32],
        update: &[f32],
        route_weight: f32,
        precision: CpuExecutionPrecision,
    ) -> Result<()> {
        weighted_reduce(output, update, route_weight, precision)
    }

    fn residual(
        &self,
        residual: HostRows,
        update: &HostRows,
        precision: CpuExecutionPrecision,
    ) -> Result<HostRows> {
        residual_rows(residual, update, precision)
    }
}

/// Native dense provider with automatic reference fallback per semantic operation.
#[derive(Debug, Clone, Copy, Default)]
pub struct NativeCpuProvider;

impl NativeCpuProvider {
    pub fn runtime_capabilities() -> CpuCapabilities {
        static CAPABILITIES: OnceLock<CpuCapabilities> = OnceLock::new();
        *CAPABILITIES.get_or_init(|| {
            let bits = unsafe { ffi::ferrule_cpu_capabilities() };
            let avx2 = bits & CAP_AVX2 != 0;
            let avx512_bf16 = bits & CAP_AVX512_BF16 != 0;
            CpuCapabilities {
                avx2,
                avx512_bf16,
                native_bf16_embedding: avx2,
                native_bf16_linear: avx2,
                native_rms_norm: avx2,
                native_bf16_swiglu: avx2,
                native_paged_gqa: false,
                native_routed_moe: false,
            }
        })
    }

    fn reference() -> ReferenceCpuProvider {
        ReferenceCpuProvider
    }
}

impl CpuOperatorProvider for NativeCpuProvider {
    fn name(&self) -> &'static str {
        "native-cpu"
    }

    fn capabilities(&self) -> CpuCapabilities {
        Self::runtime_capabilities()
    }

    fn embedding(
        &self,
        embedding: LinearRef<'_>,
        token_ids: &[u32],
        arena: Option<RowsArenaId>,
        precision: CpuExecutionPrecision,
    ) -> Result<HostRows> {
        let LinearWeight::Bf16(weight) = embedding.weight else {
            return Self::reference().embedding(embedding, token_ids, arena, precision);
        };
        if !self.capabilities().native_bf16_embedding {
            return Self::reference().embedding(embedding, token_ids, arena, precision);
        }
        embedding.validate()?;
        if token_ids.is_empty() || embedding.bias.is_some() {
            return Err(cpu_error(
                "embedding requires tokens and an unbiased row-major matrix",
            ));
        }
        let mut output = vec![0.0; token_ids.len() * embedding.in_features];
        let status = unsafe {
            ffi::ferrule_cpu_embedding_bf16(
                weight.as_ptr().cast(),
                token_ids.as_ptr(),
                output.as_mut_ptr(),
                token_ids.len(),
                embedding.out_features,
                embedding.in_features,
            )
        };
        native_status(status, "BF16 embedding")?;
        precision.apply_slice(&mut output);
        HostRows::new(
            RowsShape::new(token_ids.len(), embedding.in_features)?,
            precision.rows_dtype(),
            arena,
            output,
        )
    }

    fn linear(
        &self,
        linear: LinearRef<'_>,
        input: &HostRows,
        arena: Option<RowsArenaId>,
        precision: CpuExecutionPrecision,
    ) -> Result<HostRows> {
        let LinearWeight::Bf16(weight) = linear.weight else {
            return Self::reference().linear(linear, input, arena, precision);
        };
        if !self.capabilities().native_bf16_linear {
            return Self::reference().linear(linear, input, arena, precision);
        }
        linear.validate()?;
        if input.shape().width() != linear.in_features {
            return Err(cpu_error("linear input width mismatch"));
        }
        let mut prepared = input.values().to_vec();
        precision.apply_slice(&mut prepared);
        let mut output = vec![0.0; input.shape().rows() * linear.out_features];
        let status = unsafe {
            ffi::ferrule_cpu_linear_bf16(
                weight.as_ptr().cast(),
                prepared.as_ptr(),
                linear.bias.map_or(std::ptr::null(), <[f32]>::as_ptr),
                output.as_mut_ptr(),
                input.shape().rows(),
                linear.out_features,
                linear.in_features,
            )
        };
        native_status(status, "BF16 linear")?;
        precision.apply_slice(&mut output);
        HostRows::new(
            RowsShape::new(input.shape().rows(), linear.out_features)?,
            precision.rows_dtype(),
            arena,
            output,
        )
    }

    fn rms_norm(
        &self,
        input: &HostRows,
        weight: &[f32],
        epsilon: f32,
        heads: usize,
        arena: Option<RowsArenaId>,
        precision: CpuExecutionPrecision,
    ) -> Result<HostRows> {
        if !self.capabilities().native_rms_norm {
            return Self::reference().rms_norm(input, weight, epsilon, heads, arena, precision);
        }
        if heads == 0
            || weight.is_empty()
            || !input.shape().width().is_multiple_of(heads)
            || weight.len() != input.shape().width() / heads
            || !epsilon.is_finite()
            || epsilon <= 0.0
        {
            return Err(cpu_error("RMSNorm shape mismatch"));
        }
        let native_rows = input
            .shape()
            .rows()
            .checked_mul(heads)
            .ok_or_else(|| cpu_error("RMSNorm row count overflow"))?;
        let mut output = vec![0.0; input.values().len()];
        let status = unsafe {
            ffi::ferrule_cpu_rms_norm(
                input.values().as_ptr(),
                weight.as_ptr(),
                output.as_mut_ptr(),
                native_rows,
                weight.len(),
                epsilon,
                u32::from(precision == CpuExecutionPrecision::Bf16),
            )
        };
        native_status(status, "RMSNorm")?;
        HostRows::new(input.shape(), precision.rows_dtype(), arena, output)
    }

    fn rope(
        &self,
        input: HostRows,
        table: RopeRef<'_>,
        pairing: RotaryPairing,
        region: RotaryRegion,
        heads: usize,
        head_dim: usize,
        positions: &[usize],
        precision: CpuExecutionPrecision,
    ) -> Result<HostRows> {
        Self::reference().rope(
            input, table, pairing, region, heads, head_dim, positions, precision,
        )
    }

    fn paged_gqa(
        &self,
        history: &dyn PagedKvHistory,
        layer: usize,
        request: PagedCausalGqa<'_>,
        precision: CpuExecutionPrecision,
    ) -> Result<HostRows> {
        Self::reference().paged_gqa(history, layer, request, precision)
    }

    fn router(
        &self,
        logits: &HostRows,
        top_k: usize,
        route_scale: f32,
        precision: CpuExecutionPrecision,
    ) -> Result<RouterRoutes> {
        Self::reference().router(logits, top_k, route_scale, precision)
    }

    fn swiglu(
        &self,
        expert: SwiGluRef<'_>,
        input: &HostRows,
        arena: Option<RowsArenaId>,
        precision: CpuExecutionPrecision,
    ) -> Result<HostRows> {
        let (LinearWeight::Bf16(gate), LinearWeight::Bf16(up), LinearWeight::Bf16(down)) =
            (expert.gate.weight, expert.up.weight, expert.down.weight)
        else {
            return Self::reference().swiglu(expert, input, arena, precision);
        };
        if !self.capabilities().native_bf16_swiglu {
            return Self::reference().swiglu(expert, input, arena, precision);
        }
        expert.validate()?;
        if input.shape().width() != expert.gate.in_features {
            return Err(cpu_error("SwiGLU input width mismatch"));
        }
        let mut output = vec![0.0; input.shape().rows() * expert.down.out_features];
        let status = unsafe {
            ffi::ferrule_cpu_swiglu_bf16(
                gate.as_ptr().cast(),
                up.as_ptr().cast(),
                down.as_ptr().cast(),
                expert.gate.bias.map_or(std::ptr::null(), <[f32]>::as_ptr),
                expert.up.bias.map_or(std::ptr::null(), <[f32]>::as_ptr),
                expert.down.bias.map_or(std::ptr::null(), <[f32]>::as_ptr),
                input.values().as_ptr(),
                output.as_mut_ptr(),
                input.shape().rows(),
                expert.gate.in_features,
                expert.gate.out_features,
                expert.down.out_features,
                expert.activation_limit.unwrap_or(0.0),
                u32::from(precision == CpuExecutionPrecision::Bf16),
            )
        };
        native_status(status, "BF16 SwiGLU")?;
        HostRows::new(
            RowsShape::new(input.shape().rows(), expert.down.out_features)?,
            precision.rows_dtype(),
            arena,
            output,
        )
    }

    fn weighted_reduce(
        &self,
        output: &mut [f32],
        update: &[f32],
        route_weight: f32,
        precision: CpuExecutionPrecision,
    ) -> Result<()> {
        Self::reference().weighted_reduce(output, update, route_weight, precision)
    }

    fn residual(
        &self,
        residual: HostRows,
        update: &HostRows,
        precision: CpuExecutionPrecision,
    ) -> Result<HostRows> {
        Self::reference().residual(residual, update, precision)
    }
}

fn native_status(status: i32, operation: &str) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(cpu_error(format!(
            "native {operation} rejected a validated request with status {status}"
        )))
    }
}

mod ffi {
    unsafe extern "C" {
        pub fn ferrule_cpu_capabilities() -> u64;
        pub fn ferrule_cpu_embedding_bf16(
            weight: *const u8,
            token_ids: *const u32,
            output: *mut f32,
            rows: usize,
            vocabulary: usize,
            width: usize,
        ) -> i32;
        pub fn ferrule_cpu_linear_bf16(
            weight: *const u8,
            input: *const f32,
            bias: *const f32,
            output: *mut f32,
            rows: usize,
            out_features: usize,
            in_features: usize,
        ) -> i32;
        pub fn ferrule_cpu_rms_norm(
            input: *const f32,
            weight: *const f32,
            output: *mut f32,
            rows: usize,
            width: usize,
            epsilon: f32,
            bf16_boundary: u32,
        ) -> i32;
        pub fn ferrule_cpu_swiglu_bf16(
            gate_weight: *const u8,
            up_weight: *const u8,
            down_weight: *const u8,
            gate_bias: *const f32,
            up_bias: *const f32,
            down_bias: *const f32,
            input: *const f32,
            output: *mut f32,
            rows: usize,
            input_width: usize,
            intermediate_width: usize,
            output_width: usize,
            activation_limit: f32,
            bf16_boundary: u32,
        ) -> i32;
    }
}
