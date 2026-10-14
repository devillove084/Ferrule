//! Explicit eager BF16 attention geometry and caller admission; no CUDA required.
use ferrule_common::{Error, Result};

pub const BF16_ATTENTION_WORKSPACE_BYTES: usize = 32 << 20;
pub const BF16_ATTENTION_PROVIDER_BYTES: usize = 16 << 20;

pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::Internal {
        message: format!("prepared BF16 attention: {}", message.into()),
    }
}
/// One complete HF matrix batch: all query heads and all rows of one sequence.
/// Groups partition packed query rows. Never split a group to change GEMM shape.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bf16AttentionGroup {
    pub row_start: u32,
    pub rows: u32,
    pub sequence: u32,
    pub tokens: u32,
}
/// One exact per-sequence matrix geometry; H/Hkv/D belong to the plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Bf16AttentionShape {
    pub rows: u32,
    pub tokens: u32,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bf16AttentionPlan {
    groups: Vec<Bf16AttentionGroup>,
    max_total_rows: usize,
    max_sequences: usize,
    warm: Vec<Bf16AttentionGroup>,
    heads: usize,
    kv_heads: usize,
    dim: usize,
    rows: usize,
    max_rows: usize,
    max_tokens: usize,
    offsets: [usize; 7],
    bytes: usize,
}
impl Bf16AttentionPlan {
    /// The persistent softmax contract currently covers logical KV lengths
    /// 1..=1024. Larger lengths are rejected, never silently sent to another tree.
    /// Budgets include the allocator's physical grant and an external provider
    /// reserve separately. The reserve is a charge, not an attributable CUDA
    /// allocation measurement. The caller must account for every live owner.
    pub fn new(
        groups: &[Bf16AttentionGroup],
        heads: usize,
        kv_heads: usize,
        dim: usize,
        storage_budget: usize,
        provider_budget: usize,
    ) -> Result<Self> {
        if groups.is_empty()
            || groups.len() > i32::MAX as usize
            || heads == 0
            || kv_heads == 0
            || !heads.is_multiple_of(kv_heads)
            || heads > i32::MAX as usize
            || dim == 0
            || dim > i32::MAX as usize
        {
            return Err(invalid("geometry"));
        }
        let mut rows = 0usize;
        let mut max_rows = 0usize;
        let mut max_tokens = 0usize;
        for (sequence, g) in groups.iter().enumerate() {
            if g.row_start as usize != rows
                || g.sequence as usize != sequence
                || g.rows == 0
                || g.rows > i32::MAX as u32
                || !(1..=1024).contains(&g.tokens)
                || g.rows > g.tokens
            {
                return Err(invalid(
                    "groups must partition rows in sequence order, with positive M <= KV <= 1024",
                ));
            }
            rows = rows
                .checked_add(g.rows as usize)
                .filter(|&n| n <= i32::MAX as usize)
                .ok_or_else(|| invalid("rows overflow"))?;
            max_rows = max_rows.max(g.rows as usize);
            max_tokens = max_tokens.max(g.tokens as usize);
        }
        let size = |a: usize, b: usize| {
            heads
                .checked_mul(a)
                .and_then(|n| n.checked_mul(b))
                .and_then(|n| n.checked_mul(2))
                .filter(|&n| n <= i32::MAX as usize)
                .ok_or_else(|| invalid("matrix extent/grid overflow"))
        };
        let q = size(max_rows, dim)?;
        let kv = size(max_tokens, dim)?;
        let score = size(max_rows, max_tokens)?;
        // Q, duplicated K/V, raw QK, probability, BF16 PV, aligned cuBLAS workspace.
        let mut offsets = [0; 7];
        let mut bytes = 0usize;
        for (i, n) in [q, kv, kv, score, score, q, BF16_ATTENTION_WORKSPACE_BYTES]
            .into_iter()
            .enumerate()
        {
            bytes = bytes
                .checked_add(255)
                .map(|n| n & !255)
                .ok_or_else(|| invalid("alignment overflow"))?;
            offsets[i] = bytes;
            bytes = bytes
                .checked_add(n)
                .ok_or_else(|| invalid("workspace overflow"))?;
        }
        if crate::cuda_allocation_grant_bytes(bytes).is_none_or(|n| n > storage_budget)
            || provider_budget < BF16_ATTENTION_PROVIDER_BYTES
        {
            return Err(invalid(
                "physical storage grant or external reserve exceeds budget",
            ));
        }
        let mut warm = groups
            .iter()
            .map(|g| Bf16AttentionGroup {
                row_start: 0,
                rows: g.rows,
                sequence: 0,
                tokens: g.tokens,
            })
            .collect::<Vec<_>>();
        warm.sort_unstable_by_key(|g| (g.rows, g.tokens));
        warm.dedup();
        Ok(Self {
            groups: groups.to_vec(),
            max_total_rows: rows,
            max_sequences: groups.len(),
            warm,
            heads,
            kv_heads,
            dim,
            rows,
            max_rows,
            max_tokens,
            offsets,
            bytes,
        })
    }
    /// Admit a bounded composition of independently warmed sequence shapes.
    /// No whole-cohort Cartesian product is enumerated. Only declared M/N pairs
    /// are warmed; repeated pairs across sequences share the same warmup.
    pub fn from_shapes(
        shapes: &[Bf16AttentionShape],
        max_total_rows: usize,
        max_sequences: usize,
        heads: usize,
        kv_heads: usize,
        dim: usize,
        storage_budget: usize,
        provider_budget: usize,
    ) -> Result<Self> {
        if shapes.is_empty()
            || max_total_rows == 0
            || max_total_rows > i32::MAX as usize
            || max_sequences == 0
            || max_sequences > max_total_rows
        {
            return Err(invalid("positive native row/sequence capacities required"));
        }
        let shapes = shapes
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        for s in &shapes {
            if s.rows == 0
                || s.rows as usize > max_total_rows
                || s.rows > s.tokens
                || !(1..=1024).contains(&s.tokens)
            {
                return Err(invalid("per-sequence M/KV shape"));
            }
        }
        let max_rows = shapes.iter().map(|s| s.rows).max().unwrap();
        let max_tokens = shapes.iter().map(|s| s.tokens).max().unwrap();
        let mut p = Self::new(
            &[Bf16AttentionGroup {
                row_start: 0,
                rows: max_rows,
                sequence: 0,
                tokens: max_tokens,
            }],
            heads,
            kv_heads,
            dim,
            storage_budget,
            provider_budget,
        )?;
        p.warm = shapes
            .iter()
            .map(|s| Bf16AttentionGroup {
                row_start: 0,
                rows: s.rows,
                sequence: 0,
                tokens: s.tokens,
            })
            .collect();
        p.groups = vec![p.warm[0]];
        p.rows = p.groups[0].rows as usize;
        p.max_total_rows = max_total_rows;
        p.max_sequences = max_sequences;
        Ok(p)
    }
    /// Compatibility convenience: extract the distinct per-sequence shapes and
    /// maximum packed row/sequence bounds. Other combinations within these
    /// bounds are also admitted; whole-cohort enumeration is not required.
    pub fn with_views(
        cohorts: &[Vec<Bf16AttentionGroup>],
        heads: usize,
        kv_heads: usize,
        dim: usize,
        storage_budget: usize,
        provider_budget: usize,
    ) -> Result<Self> {
        if cohorts.is_empty() {
            return Err(invalid("empty view set"));
        }
        let mut shapes = Vec::new();
        let mut max_rows = 0;
        let mut max_sequences = 0;
        for g in cohorts {
            let p = Self::new(g, heads, kv_heads, dim, storage_budget, provider_budget)?;
            max_rows = max_rows.max(p.rows);
            max_sequences = max_sequences.max(g.len());
            shapes.extend(g.iter().map(|g| Bf16AttentionShape {
                rows: g.rows,
                tokens: g.tokens,
            }));
        }
        let mut p = Self::from_shapes(
            &shapes,
            max_rows,
            max_sequences,
            heads,
            kv_heads,
            dim,
            storage_budget,
            provider_budget,
        )?;
        p.groups = cohorts[0].clone();
        p.rows = p.groups.iter().map(|g| g.rows as usize).sum();
        Ok(p)
    }
    fn admits_groups(&self, groups: &[Bf16AttentionGroup]) -> bool {
        if groups.is_empty() || groups.len() > self.max_sequences {
            return false;
        }
        let mut end = 0usize;
        for (seq, g) in groups.iter().enumerate() {
            if g.sequence as usize != seq
                || g.row_start as usize != end
                || self
                    .warm
                    .binary_search_by_key(&(g.rows, g.tokens), |g| (g.rows, g.tokens))
                    .is_err()
            {
                return false;
            }
            let Some(next) = end.checked_add(g.rows as usize) else {
                return false;
            };
            if next > self.max_total_rows {
                return false;
            }
            end = next;
        }
        true
    }
    pub fn view(&self, groups: &[Bf16AttentionGroup]) -> Result<Bf16AttentionView> {
        if !self.admits_groups(groups) {
            return Err(invalid(
                "unwarmed shape, invalid partition, or packed capacity exceeded",
            ));
        }
        Ok(Bf16AttentionView {
            groups: groups.to_vec(),
            heads: self.heads,
            kv_heads: self.kv_heads,
            dim: self.dim,
        })
    }
    pub fn admits(&self, view: &Bf16AttentionView) -> bool {
        view.heads == self.heads
            && view.kv_heads == self.kv_heads
            && view.dim == self.dim
            && self.admits_groups(&view.groups)
    }
    pub const fn max_total_rows(&self) -> usize {
        self.max_total_rows
    }
    pub const fn max_sequences(&self) -> usize {
        self.max_sequences
    }
    pub fn warm_groups(&self) -> &[Bf16AttentionGroup] {
        &self.warm
    }
    pub fn groups(&self) -> &[Bf16AttentionGroup] {
        &self.groups
    }
    pub const fn heads(&self) -> usize {
        self.heads
    }
    pub const fn kv_heads(&self) -> usize {
        self.kv_heads
    }
    pub const fn head_dim(&self) -> usize {
        self.dim
    }
    pub const fn rows(&self) -> usize {
        self.rows
    }
    pub const fn max_rows(&self) -> usize {
        self.max_rows
    }
    pub const fn max_tokens(&self) -> usize {
        self.max_tokens
    }
    pub const fn offsets(&self) -> [usize; 7] {
        self.offsets
    }
    pub const fn storage_bytes(&self) -> usize {
        self.bytes
    }
    pub fn storage_grant_bytes(&self) -> usize {
        crate::cuda_allocation_grant_bytes(self.bytes).expect("admitted physical grant")
    }
    pub const fn provider_bytes(&self) -> usize {
        BF16_ATTENTION_PROVIDER_BYTES
    }
    pub fn total_charge_bytes(&self) -> usize {
        self.storage_grant_bytes() + self.provider_bytes()
    }
    /// CUDA operations submitted: metadata gate per group, optional append,
    /// and six operations/group (pack, gather, QK, softmax, PV, scatter).
    pub fn submission_count(&self, append: bool) -> u64 {
        u64::from(append) + 7 * self.groups.len() as u64
    }
}

/// Checked geometry view, created before execution. No device resources or raw ABI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bf16AttentionView {
    groups: Vec<Bf16AttentionGroup>,
    heads: usize,
    kv_heads: usize,
    dim: usize,
}
impl Bf16AttentionView {
    pub fn groups(&self) -> &[Bf16AttentionGroup] {
        &self.groups
    }
    pub fn rows(&self) -> usize {
        self.groups.iter().map(|g| g.rows as usize).sum()
    }
    pub fn submission_count(&self, append: bool) -> u64 {
        u64::from(append) + 7 * self.groups.len() as u64
    }
}
