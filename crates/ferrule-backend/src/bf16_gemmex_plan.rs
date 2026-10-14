//! Explicit native BF16-output GemmEx admission, independent of model precision.
use ferrule_common::{
    Error, Result,
    numeric_fp8::{NumericFp8Layout, NumericFp8ScaleType},
};

pub const BF16_GEMMEX_WORKSPACE_BYTES: usize = 32 << 20;
pub const BF16_GEMMEX_PROVIDER_BYTES: usize = 16 << 20;

/// Packed full-width numeric FP8 decode followed by BF16 A/B/C, F32-compute
/// GemmEx. Admission describes arithmetic, not universal HF bit equivalence.
/// No model selects this backend identity implicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bf16GemmExPlan {
    layout: NumericFp8Layout,
    rows: usize,
    resident: bool,
    staging: usize,
    bytes: usize,
}
impl Bf16GemmExPlan {
    pub fn new(
        layout: NumericFp8Layout,
        rows: usize,
        storage_budget: usize,
        provider_budget: usize,
    ) -> Result<Self> {
        let invalid = || Error::Internal {
            message: "BF16 GemmEx: invalid geometry, overflow, or explicit budget".into(),
        };
        layout.storage_lengths()?;
        // Aligned packed operands, complete 128x128 scale blocks; no submatrix
        // origin or hidden padding that could change the native GEMM geometry.
        if !(1..=32).contains(&rows)
            || !layout.n.is_multiple_of(128)
            || !layout.k.is_multiple_of(128)
            || layout.row_origin != 0
            || layout.column_origin != 0
            || layout.scale_type != NumericFp8ScaleType::Bf16
        {
            return Err(invalid());
        }
        Self::admit(layout, rows, false, storage_budget, provider_budget)
    }
    /// Packed resident BF16 operands have no FP8 scale-block restriction.
    /// Require positive native i32 N/K and M1..32. Packed leading dimensions
    /// are the logical K/N; no tensor-core tile padding is imposed. Operand
    /// pointers require BF16 element alignment. Numeric FP8 decode is forbidden.
    pub fn new_resident(
        n: usize,
        k: usize,
        rows: usize,
        storage_budget: usize,
        provider_budget: usize,
    ) -> Result<Self> {
        if !(1..=32).contains(&rows)
            || n == 0
            || k == 0
            || n > i32::MAX as usize
            || k > i32::MAX as usize
        {
            return Err(Error::Internal {
                message: "resident BF16 GemmEx requires M1..32 and positive native i32 N/K".into(),
            });
        }
        Self::admit(
            NumericFp8Layout {
                n,
                k,
                row_origin: 0,
                column_origin: 0,
                scale_type: NumericFp8ScaleType::Bf16,
            },
            rows,
            true,
            storage_budget,
            provider_budget,
        )
    }
    fn admit(
        layout: NumericFp8Layout,
        rows: usize,
        resident: bool,
        storage_budget: usize,
        provider_budget: usize,
    ) -> Result<Self> {
        let invalid = || Error::Internal {
            message: "BF16 GemmEx: overflow or explicit budget".into(),
        };
        let staging = rows
            .checked_add(layout.n)
            .and_then(|v| v.checked_mul(layout.k))
            .and_then(|v| v.checked_add(rows.checked_mul(layout.n)?))
            .and_then(|v| v.checked_mul(2))
            // Keep the once-bound cuBLAS workspace 256-byte aligned even for
            // resident tails. No matrix padding changes native arithmetic.
            .and_then(|v| v.checked_add(255))
            .map(|v| v & !255)
            .ok_or_else(invalid)?;
        let bytes = staging
            .checked_add(BF16_GEMMEX_WORKSPACE_BYTES)
            .ok_or_else(invalid)?;
        if bytes > isize::MAX as usize
            || crate::cuda_allocation_grant_bytes(bytes).is_none_or(|grant| grant > storage_budget)
            || provider_budget < BF16_GEMMEX_PROVIDER_BYTES
        {
            return Err(invalid());
        }
        Ok(Self {
            layout,
            rows,
            resident,
            staging,
            bytes,
        })
    }
    pub const fn is_resident(self) -> bool {
        self.resident
    }
    pub fn storage_grant_bytes(self) -> usize {
        crate::cuda_allocation_grant_bytes(self.bytes).expect("admitted allocation grant")
    }
    pub const fn layout(self) -> NumericFp8Layout {
        self.layout
    }
    pub const fn rows(self) -> usize {
        self.rows
    }
    /// Activation, full decoded weight and initialization BF16 output storage.
    pub const fn staging_bytes(self) -> usize {
        self.staging
    }
    pub const fn cublas_workspace_bytes(self) -> usize {
        BF16_GEMMEX_WORKSPACE_BYTES
    }
    /// Caller allocation, including staging and the disjoint cuBLAS workspace.
    pub const fn storage_bytes(self) -> usize {
        self.bytes
    }
    /// Separately charged external-library growth ceiling, not device scratch.
    pub const fn provider_bytes(self) -> usize {
        BF16_GEMMEX_PROVIDER_BYTES
    }
}
