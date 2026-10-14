//! Explicit, opt-in chunk workspace geometry, available without CUDA.
use crate::cuda_plan::{GatedDeltaNetLayout, recurrent_bf16::invalid};
use crate::plan::OperatorWorkspaceRequirements;
use ferrule_common::Result;

pub const BF16_GDN_CHUNK_ROWS: usize = 64;

/// Explicit arithmetic schedule, not a precision profile or a state-content guess.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bf16ChunkExecution {
    CachedRecurrent,
    PrefillChunk,
}

/// No implicit selection by model profile. `new` preserves one-row delegation;
/// `new_prefill` explicitly uses padded chunk arithmetic even for one row.
/// Chunk execution uses IEEE F32 matrix products but does
/// not promise the reduction tree of a particular external BLAS implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bf16ChunkPlan {
    layout: GatedDeltaNetLayout,
    elements: usize,
    launches: u64,
    execution: Bf16ChunkExecution,
}
impl Bf16ChunkPlan {
    /// Logical scratch budget, excluding existing operands and allocator overhead.
    /// Multi-row capability: Dk/Dv in 1..=512 and at most 65535 value heads.
    pub fn new(layout: GatedDeltaNetLayout, budget_bytes: usize) -> Result<Self> {
        layout.validate()?;
        if layout.rows == 1 {
            return Ok(Self {
                layout,
                elements: 0,
                launches: 1,
                execution: Bf16ChunkExecution::CachedRecurrent,
            });
        }
        Self::new_prefill(layout, budget_bytes)
    }

    /// Prefill/chunk semantics, including a one-row padded 64-token chunk.
    /// Requires the full checked workspace even for one row; does not infer
    /// phase from state values or reset the caller-owned F32 state.
    pub fn new_prefill(layout: GatedDeltaNetLayout, budget_bytes: usize) -> Result<Self> {
        layout.validate()?;
        if layout.key_dim > 512 || layout.value_dim > 512 || layout.value_heads > 65535 {
            return Err(invalid("chunk dimensions exceed native capability"));
        }
        let c = BF16_GDN_CHUNK_ROWS;
        let checked = |v: Option<usize>| v.ok_or_else(|| invalid("chunk workspace overflow"));
        let keys = checked(
            6usize
                .checked_mul(c)
                .and_then(|n| n.checked_mul(layout.key_dim)),
        )?;
        let values = checked(
            7usize
                .checked_mul(c)
                .and_then(|n| n.checked_mul(layout.value_dim)),
        )?;
        let state = checked(layout.key_dim.checked_mul(layout.value_dim))?;
        let per_head = checked(
            keys.checked_add(values)
                .and_then(|n| n.checked_add(4 * c * c + 3 * c))
                .and_then(|n| n.checked_add(state)),
        )?;
        let elements = checked(per_head.checked_mul(layout.value_heads))?;
        let bytes = checked(elements.checked_mul(size_of::<f32>()))?;
        if bytes > i32::MAX as usize || bytes > budget_bytes {
            return Err(invalid(
                "chunk workspace exceeds native extent or caller budget",
            ));
        }
        let chunks = checked(layout.rows.checked_add(c - 1))? / c;
        let launches = (chunks as u64)
            .checked_mul(15)
            .ok_or_else(|| invalid("chunk launch count overflow"))?;
        Ok(Self {
            layout,
            elements,
            launches,
            execution: Bf16ChunkExecution::PrefillChunk,
        })
    }
    pub const fn execution(self) -> Bf16ChunkExecution {
        self.execution
    }
    pub const fn layout(self) -> GatedDeltaNetLayout {
        self.layout
    }
    pub const fn workspace_elements(self) -> usize {
        self.elements
    }
    pub const fn workspace_requirements(self) -> OperatorWorkspaceRequirements {
        OperatorWorkspaceRequirements {
            bytes: self.elements as u64 * 4,
            alignment: 4,
        }
    }
    /// Accepted native kernel launches, not completion or a rollback guarantee.
    pub const fn launch_count(self) -> u64 {
        self.launches
    }
}
