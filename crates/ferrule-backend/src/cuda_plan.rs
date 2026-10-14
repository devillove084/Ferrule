//! Static CUDA ABI and numeric workspace plans, available without the CUDA runtime.
//! Success describes geometry and bytes, never provider/device executability.

use crate::plan::OperatorWorkspaceRequirements;
pub use ferrule_common::numeric_fp8::{NumericFp8Layout, NumericFp8ScaleType};
use ferrule_common::{Error, Result};

fn invalid(message: impl Into<String>) -> Error {
    Error::Internal {
        message: format!("numeric FP8: {}", message.into()),
    }
}

fn checked(value: Option<usize>) -> Result<usize> {
    value
        .filter(|&v| v <= isize::MAX as usize)
        .ok_or_else(|| invalid("shape/size overflow"))
}

/// Explicit arithmetic profile, independent of numeric FP8 storage encoding.
/// All profiles accumulate and output F32. The multi-row TF32x3 path approximates IEEE F32
/// products; it is not bitwise SGEMM and does not promise a reduction order.
/// A deprecated inherent `kernel()` method remains in the provider's isolated
/// legacy bridge for source/const compatibility. It is a temporary vendor leak,
/// not part of the semantic contract; new callers should submit a plan instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericFp8Precision {
    /// F32 -> BF16 RNE activation and decoded BF16 weights, then BF16 TensorOp.
    Bf16RneF32Accumulate,
    /// Explicit K64-aligned transposed TensorOp with two on-chip K slices.
    /// Same bounded BF16 staging as the tiled profile; F32 output. Matches the
    /// frozen native M1 gate and M5 QKV reductions on sm86, not a universal HF dispatch policy.
    /// Never selected implicitly by the existing model profiles.
    Bf16SlicedK2F32Accumulate,
    /// BF16-input M1/N2048/K4096 GEMV with fixed staging and a persistent provider reserve.
    Bf16M1GemvF32Accumulate,
    /// Weight-only E4M3FN: separately F32-rounded decode*scale, then BF16 RNE.
    /// F32 input is rounded to BF16 inside the kernel; the BF16-input API
    /// reads prepared BF16 directly. Small batches use BF16-rounded vector
    /// operands, larger batches BF16 TensorOp, with F32 accumulation/output.
    /// One fused launch with no global decoded-weight materialization.
    /// Scales retain their original dtype; this is not Marlin equivalence.
    /// As with the legacy BF16 profile, conversion can overflow/underflow;
    /// TensorOp subnormal/reduction behavior is not a bitwise IEEE promise.
    W8A16Bf16F32Accumulate,
    /// F32 activation and decoded F32 weights. M1 uses fused IEEE F32 FMA;
    /// multiple rows retain tiled TF32x3 TensorOp. The historical profile name
    /// is not a kernel identity: inspect `plan.selected_kernel()`.
    /// No BF16 rounding or CPU fallback. F32 accumulation can still lose
    /// accuracy for ill-conditioned/cancelling dot products.
    F32Tf32x3,
}

/// Actual arithmetic implementation, distinct from the historical precision name.
/// Fused M1 has full F32 products/accumulation, not TF32 truncation or BF16
/// rounding. W8A16 uses BF16-rounded vector or TensorOp operands. No implementation promises a
/// serial reduction order or bitwise equivalence; cancellation can lose accuracy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericFp8SelectedKernel {
    Bf16TiledGemm,
    Bf16TiledSlicedK2Gemm,
    Bf16M1Gemv,
    F32TiledTf32x3,
    F32FusedM1,
    W8A16FusedBf16Mma,
    W8A16FusedBf16Vector,
}
impl NumericFp8SelectedKernel {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Bf16M1Gemv => "numeric-bf16-m1-gemv",
            Self::Bf16TiledSlicedK2Gemm => "numeric-bf16-tiled-sliced-k2-gemm",
            Self::Bf16TiledGemm => "numeric-bf16-tiled-gemm",
            Self::F32TiledTf32x3 => "numeric-f32-tiled-tf32x3",
            Self::F32FusedM1 => "numeric-fused-fp32-m1",
            Self::W8A16FusedBf16Mma => "numeric-w8a16-fused-bf16-mma",
            Self::W8A16FusedBf16Vector => "numeric-w8a16-fused-bf16-vector",
        }
    }
}

/// Validated budget plan. BF16 retains a K8-padded activation plus one BF16
/// weight tile. F32 rows > 1 retain one unpadded F32 weight tile, while M1
/// directly reads compressed weights with zero scratch and one launch.
/// W8A16 does so for every supported M, using on-chip BF16 operand tiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NumericFp8LinearPlan {
    layout: NumericFp8Layout,
    rows: usize,
    padded_k: usize,
    tile_rows: usize,
    bytes: usize,
    budget: usize,
    precision: NumericFp8Precision,
}

/// Admission ceiling for provider memory outside the Ferrule allocator.
/// Observed CUDA12.4 growth: 10 MiB handle plus 2/4 MiB module loading. This
/// is not a vendor guarantee: owner initialization measures and rejects excess.
pub const NUMERIC_BF16_GEMV_PROVIDER_BYTES: usize = (10 + 2 + 4) << 20;
pub const NUMERIC_BF16_GEMV_TILE_ROWS: usize = 257;
/// Activation + decoded tile + disjoint F32 initialization output.
pub const NUMERIC_BF16_GEMV_SCRATCH_BYTES: usize =
    (1 + NUMERIC_BF16_GEMV_TILE_ROWS) * 4096 * 2 + 2048 * 4;

impl NumericFp8LinearPlan {
    pub fn new(
        layout: NumericFp8Layout,
        rows: usize,
        scratch_budget_bytes: usize,
        precision: NumericFp8Precision,
    ) -> Result<Self> {
        layout.storage_lengths()?;
        if rows == 0 || rows > i32::MAX as usize - 64 {
            return Err(invalid("invalid activation rows"));
        }
        if precision == NumericFp8Precision::Bf16M1GemvF32Accumulate {
            if rows != 1
                || layout.n != 2048
                || layout.k != 4096
                || layout.row_origin != 0
                || layout.column_origin != 0
                || layout.scale_type != NumericFp8ScaleType::Bf16
            {
                return Err(invalid(
                    "BF16 GEMV requires M1/N2048/K4096, unshifted BF16 scales",
                ));
            }
            if scratch_budget_bytes < NUMERIC_BF16_GEMV_SCRATCH_BYTES {
                return Err(invalid(
                    "BF16 GEMV scratch below fixed tile and initialization output",
                ));
            }
            return Ok(Self {
                layout,
                rows,
                padded_k: layout.k,
                tile_rows: NUMERIC_BF16_GEMV_TILE_ROWS,
                bytes: NUMERIC_BF16_GEMV_SCRATCH_BYTES,
                budget: scratch_budget_bytes,
                precision,
            });
        }
        if (rows == 1 && precision == NumericFp8Precision::F32Tf32x3)
            || precision == NumericFp8Precision::W8A16Bf16F32Accumulate
        {
            checked(rows.checked_mul(layout.k).and_then(|v| v.checked_mul(4)))?;
            checked(rows.checked_mul(layout.n).and_then(|v| v.checked_mul(4)))?;
            return Ok(Self {
                layout,
                rows,
                padded_k: layout.k,
                tile_rows: 0,
                bytes: 0,
                budget: scratch_budget_bytes,
                precision,
            });
        }
        if precision == NumericFp8Precision::Bf16SlicedK2F32Accumulate
            && !layout.k.is_multiple_of(64)
        {
            return Err(invalid("sliced-K2 profile requires K divisible by 64"));
        }
        let (padded_k, row_bytes, activation_bytes) = match precision {
            NumericFp8Precision::Bf16RneF32Accumulate
            | NumericFp8Precision::Bf16SlicedK2F32Accumulate => {
                let padded_k = layout.k.div_ceil(8) * 8;
                let row_bytes = checked(padded_k.checked_mul(2))?;
                let activation = checked(rows.checked_mul(row_bytes))?;
                (padded_k, row_bytes, activation)
            }
            NumericFp8Precision::W8A16Bf16F32Accumulate
            | NumericFp8Precision::Bf16M1GemvF32Accumulate => unreachable!(),
            NumericFp8Precision::F32Tf32x3 => {
                let row_bytes = checked(layout.k.checked_mul(4))?;
                (layout.k, row_bytes, 0)
            }
        };
        let available = scratch_budget_bytes
            .checked_sub(activation_bytes)
            .ok_or_else(|| invalid("scratch budget below activation bytes"))?;
        // Retain the established maximum output-row tile contract. Changing
        // this bound would change budget partitioning and launch counts.
        let tile_rows = (available / row_bytes).min(layout.n).min(64 * 65535);
        if tile_rows == 0 {
            return Err(invalid(
                "scratch budget cannot hold one weight row plus required conversion storage",
            ));
        }
        let bytes = checked(match precision {
            NumericFp8Precision::Bf16RneF32Accumulate
            | NumericFp8Precision::Bf16SlicedK2F32Accumulate => rows
                .checked_add(tile_rows)
                .and_then(|r| r.checked_mul(row_bytes)),
            NumericFp8Precision::F32Tf32x3 => tile_rows.checked_mul(row_bytes),
            NumericFp8Precision::W8A16Bf16F32Accumulate
            | NumericFp8Precision::Bf16M1GemvF32Accumulate => unreachable!(),
        })?;
        Ok(Self {
            layout,
            rows,
            padded_k,
            tile_rows,
            bytes,
            budget: scratch_budget_bytes,
            precision,
        })
    }

    pub fn rows(self) -> usize {
        self.rows
    }

    pub fn layout(self) -> NumericFp8Layout {
        self.layout
    }

    pub fn workspace_requirements(self) -> OperatorWorkspaceRequirements {
        OperatorWorkspaceRequirements {
            bytes: self.bytes as u64,
            alignment: 16,
        }
    }

    pub const fn precision(self) -> NumericFp8Precision {
        self.precision
    }

    /// Deterministic shape selection; provider availability is checked at submission.
    pub const fn selected_kernel(self) -> NumericFp8SelectedKernel {
        match self.precision {
            NumericFp8Precision::Bf16M1GemvF32Accumulate => NumericFp8SelectedKernel::Bf16M1Gemv,
            NumericFp8Precision::Bf16SlicedK2F32Accumulate => {
                NumericFp8SelectedKernel::Bf16TiledSlicedK2Gemm
            }
            NumericFp8Precision::Bf16RneF32Accumulate => NumericFp8SelectedKernel::Bf16TiledGemm,
            // Keep this boundary in sync with native w8a16_dispatch.
            NumericFp8Precision::W8A16Bf16F32Accumulate if self.rows <= 8 => {
                NumericFp8SelectedKernel::W8A16FusedBf16Vector
            }
            NumericFp8Precision::W8A16Bf16F32Accumulate => {
                NumericFp8SelectedKernel::W8A16FusedBf16Mma
            }
            NumericFp8Precision::F32Tf32x3 if self.rows == 1 => {
                NumericFp8SelectedKernel::F32FusedM1
            }
            NumericFp8Precision::F32Tf32x3 => NumericFp8SelectedKernel::F32TiledTf32x3,
        }
    }

    /// Persistent external-provider reserve, charged once per CUDA operator owner.
    /// Additional to numeric scratch, not subtracted from tile capacity.
    pub const fn provider_bytes(self) -> usize {
        if matches!(self.precision, NumericFp8Precision::Bf16M1GemvF32Accumulate) {
            NUMERIC_BF16_GEMV_PROVIDER_BYTES
        } else {
            0
        }
    }

    /// CPU-visible admission for external provider memory, separate from scratch.
    pub fn validate_provider_budget(self, bytes: usize) -> Result<()> {
        if bytes < self.provider_bytes() {
            return Err(invalid(format!(
                "provider reserve requires {} bytes, admitted {bytes}",
                self.provider_bytes()
            )));
        }
        Ok(())
    }

    pub fn scratch_budget_bytes(self) -> usize {
        self.budget
    }

    /// Additional activation scratch, not caller-owned input storage. F32 uses zero.
    pub fn activation_bytes(self) -> usize {
        match self.precision {
            NumericFp8Precision::Bf16RneF32Accumulate
            | NumericFp8Precision::Bf16SlicedK2F32Accumulate
            | NumericFp8Precision::Bf16M1GemvF32Accumulate => self.rows * self.padded_k * 2,
            NumericFp8Precision::F32Tf32x3 | NumericFp8Precision::W8A16Bf16F32Accumulate => 0,
        }
    }

    pub fn weight_tile_bytes(self) -> usize {
        match self.precision {
            NumericFp8Precision::Bf16RneF32Accumulate
            | NumericFp8Precision::Bf16SlicedK2F32Accumulate
            | NumericFp8Precision::Bf16M1GemvF32Accumulate => self.tile_rows * self.padded_k * 2,
            NumericFp8Precision::F32Tf32x3 => self.tile_rows * self.padded_k * 4,
            NumericFp8Precision::W8A16Bf16F32Accumulate => 0,
        }
    }

    /// Zero for fused kernels: there is no materialized weight tile.
    pub fn tile_rows(self) -> usize {
        self.tile_rows
    }

    pub fn tile_count(self) -> usize {
        if self.tile_rows == 0 {
            0
        } else {
            self.layout.n.div_ceil(self.tile_rows)
        }
    }

    pub fn kernel_launches(self) -> usize {
        match self.selected_kernel() {
            NumericFp8SelectedKernel::Bf16M1Gemv
            | NumericFp8SelectedKernel::Bf16TiledGemm
            | NumericFp8SelectedKernel::Bf16TiledSlicedK2Gemm => 1 + 2 * self.tile_count(),
            NumericFp8SelectedKernel::F32TiledTf32x3 => 2 * self.tile_count(),
            NumericFp8SelectedKernel::F32FusedM1
            | NumericFp8SelectedKernel::W8A16FusedBf16Mma
            | NumericFp8SelectedKernel::W8A16FusedBf16Vector => 1,
        }
    }

    /// Static on-chip shared storage per CTA, not global scratch or allocator
    /// residency. Legacy tiled implementations do not report shared memory here.
    pub fn fused_shared_bytes_per_cta(self) -> Option<usize> {
        match self.selected_kernel() {
            NumericFp8SelectedKernel::W8A16FusedBf16Mma => {
                // BF16 operand/epilogue tile plus bounded F32 reduction levels.
                let shared_levels = if self.layout.k <= 2048 {
                    5
                } else if self.layout.k <= 16384 {
                    8
                } else {
                    22
                };
                Some(3072 + shared_levels * 16 * 32 * 4)
            }
            NumericFp8SelectedKernel::F32FusedM1
            | NumericFp8SelectedKernel::W8A16FusedBf16Vector => Some(0),
            _ => None,
        }
    }

    /// Physical allocator size class for global workspace. This excludes
    /// pooled segment slack, capture reservations, and shared memory.
    pub fn workspace_allocation_grant_bytes(self) -> Result<usize> {
        crate::cuda_allocation_grant_bytes(self.bytes)
            .ok_or_else(|| invalid("workspace allocation class overflow"))
    }

    /// Physical size classes of the two separately uploaded encoded buffers.
    /// No decoded/repacked weight allocation. Segment slack and any old/new
    /// residency overlap during replacement remain the caller's budget items.
    pub fn artifact_allocation_grant_bytes(self) -> Result<usize> {
        let (weights, scales) = self.layout.storage_lengths()?;
        let grant = crate::cuda_allocation_grant_bytes;
        checked(grant(weights).and_then(|w| grant(scales).and_then(|s| w.checked_add(s))))
    }

    pub fn padded_k(self) -> usize {
        self.padded_k
    }
}

pub(crate) mod recurrent {
    use ferrule_common::{Error, Result};
    pub(crate) fn invalid(message: impl Into<String>) -> Error {
        Error::Internal {
            message: format!("recurrent F32 CUDA: {}", message.into()),
        }
    }

    pub(crate) fn elements(dims: &[usize]) -> Result<usize> {
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
}
pub use recurrent::{CausalConv1dLayout, F32RowsLayout, GatedDeltaNetLayout};

// Pure BF16 geometry shares the same validation with CUDA submission.
pub(crate) mod standard_bf16 {
    use ferrule_common::{Error, Result};
    pub(crate) fn invalid(message: &str) -> Error {
        Error::Internal {
            message: format!("BF16 boundary: {message}"),
        }
    }
    pub(crate) fn shape(rows: usize, width: usize) -> Result<usize> {
        rows.checked_mul(width)
            .filter(|&n| rows > 0 && width > 0 && n <= i32::MAX as usize)
            .ok_or_else(|| invalid("invalid shape/extent"))
    }
    /// Packed BF16 input/output `[rows,width]`; gate logits are BF16, never F32.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Bf16GateLayout {
        Elementwise { rows: usize, width: usize },
        RowBroadcast { rows: usize, width: usize },
    }
    impl Bf16GateLayout {
        pub fn input_elements(self) -> Result<usize> {
            let (rows, width) = self.dimensions();
            shape(rows, width)
        }
        pub fn gate_elements(self) -> Result<usize> {
            let n = self.input_elements()?;
            Ok(match self {
                Self::Elementwise { .. } => n,
                Self::RowBroadcast { rows, .. } => rows,
            })
        }
        pub(crate) fn dimensions(self) -> (usize, usize) {
            match self {
                Self::Elementwise { rows, width } | Self::RowBroadcast { rows, width } => {
                    (rows, width)
                }
            }
        }
    }
}
pub use standard_bf16::Bf16GateLayout;

pub(crate) mod recurrent_bf16 {
    use super::{CausalConv1dLayout, GatedDeltaNetLayout};
    use ferrule_common::{Error, Result};
    pub(crate) fn invalid(message: impl Into<String>) -> Error {
        Error::Internal {
            message: format!("recurrent HF BF16 CUDA: {}", message.into()),
        }
    }

    /// Logical byte budget (allocator alignment/retirement overhead is additional).
    /// No hidden F32 expansion of BF16 tensors, and no device scratch allocations.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Bf16RecurrentStorage {
        pub conv_history_bytes: usize,
        pub recurrent_state_bytes: usize,
    }
    impl Bf16RecurrentStorage {
        pub fn for_layouts(conv: CausalConv1dLayout, delta: GatedDeltaNetLayout) -> Result<Self> {
            conv.validate()?;
            delta.validate()?;
            if conv.rows != delta.rows || conv.channels != delta.qkv_width()? {
                return Err(invalid("conv and DeltaNet sequence/QKV extents differ"));
            }
            Ok(Self {
                conv_history_bytes: conv
                    .history_elements()?
                    .checked_mul(2)
                    .ok_or_else(|| invalid("history bytes overflow"))?,
                recurrent_state_bytes: delta
                    .state_elements()?
                    .checked_mul(4)
                    .ok_or_else(|| invalid("state bytes overflow"))?,
            })
        }
        pub fn total_bytes(self) -> Result<usize> {
            self.conv_history_bytes
                .checked_add(self.recurrent_state_bytes)
                .ok_or_else(|| invalid("state bytes overflow"))
        }
    }
}
pub use recurrent_bf16::Bf16RecurrentStorage;

#[path = "recurrent_bf16_chunk_plan.rs"]
mod recurrent_bf16_chunk;
pub use recurrent_bf16_chunk::{BF16_GDN_CHUNK_ROWS, Bf16ChunkExecution, Bf16ChunkPlan};

#[cfg(test)]
#[path = "../tests/numeric_bf16_gemv/metadata.rs"]
mod gemv_tests;

#[path = "bf16_gemmex_plan.rs"]
pub mod bf16_gemmex;

#[path = "bf16_attention_plan.rs"]
pub mod bf16_attention;
