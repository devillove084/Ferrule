//! Bounded F32 partial GQA, reconstructed from the post-HEAD delivery contract.
//! Q/K are already RoPE-transformed. This is not a KV placement authority.

use ferrule_common::{Error, Result};

pub(crate) fn streaming_error(message: &str) -> Error {
    Error::Internal {
        message: format!("streaming F32 GQA: {message}"),
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StreamingF32GqaConfig {
    pub rows: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub block_tokens: usize,
    pub softmax_scale: f32,
    /// Causal window including the current token; None means unlimited history.
    pub sliding_window: Option<u64>,
}

impl StreamingF32GqaConfig {
    pub fn validate(self) -> Result<()> {
        if self.rows == 0
            || self.q_heads == 0
            || self.kv_heads == 0
            || !self.q_heads.is_multiple_of(self.kv_heads)
            || self.head_dim == 0
            || self.block_tokens == 0
            || !self.softmax_scale.is_finite()
            || self.softmax_scale <= 0.0
            || self.sliding_window == Some(0)
        {
            return Err(streaming_error("invalid geometry, scale or window"));
        }
        self.query_elements()?;
        self.kv_elements(self.block_tokens)?;
        self.partial_elements()?;
        Ok(())
    }
    pub fn query_elements(self) -> Result<usize> {
        checked_product(&[self.rows, self.q_heads, self.head_dim])
    }
    pub fn kv_elements(self, tokens: usize) -> Result<usize> {
        checked_product(&[tokens, self.kv_heads, self.head_dim])
    }
    pub(crate) fn partial_elements(self) -> Result<usize> {
        checked_product(&[
            self.rows,
            self.q_heads,
            self.head_dim
                .checked_add(2)
                .ok_or_else(|| streaming_error("head dimension overflow"))?,
        ])
    }
}

fn checked_product(values: &[usize]) -> Result<usize> {
    let n = values
        .iter()
        .try_fold(1usize, |n, v| n.checked_mul(*v))
        .ok_or_else(|| streaming_error("element count overflow"))?;
    if n > isize::MAX as usize / size_of::<f32>() {
        return Err(streaming_error("byte extent overflow"));
    }
    Ok(n)
}

#[derive(Debug, Clone, Copy)]
pub struct StreamingF32GqaRequest<'a> {
    pub config: StreamingF32GqaConfig,
    /// Row-major [rows, q_heads, head_dim]. Rows have consecutive positions.
    pub query: &'a [f32],
    /// Token-major [tokens, kv_heads, head_dim]. Empty history is valid.
    pub key: &'a [f32],
    pub value: &'a [f32],
    pub query_start: u64,
    pub kv_start: u64,
}

impl StreamingF32GqaRequest<'_> {
    pub fn validate(self) -> Result<usize> {
        self.config.validate()?;
        let width = self.config.kv_elements(1)?;
        if self.query.len() != self.config.query_elements()?
            || self.key.len() != self.value.len()
            || !self.key.len().is_multiple_of(width)
        {
            return Err(streaming_error("tensor shape mismatch"));
        }
        let tokens = self.key.len() / width;
        validate_positions(self.query_start, self.config.rows)?;
        validate_positions(self.kv_start, tokens)?;
        Ok(tokens)
    }
    pub fn token_range(self, start: usize, end: usize) -> Result<Self> {
        let tokens = self.validate()?;
        if start > end || end > tokens {
            return Err(streaming_error("invalid token range"));
        }
        let width = self.config.kv_elements(1)?;
        Ok(Self {
            key: &self.key[start * width..end * width],
            value: &self.value[start * width..end * width],
            kv_start: self
                .kv_start
                .checked_add(start as u64)
                .ok_or_else(|| streaming_error("position overflow"))?,
            ..self
        })
    }
}

pub(crate) fn validate_positions(start: u64, count: usize) -> Result<()> {
    // Check the exclusive endpoint too, so empty tail partitions are representable.
    start
        .checked_add(count as u64)
        .ok_or_else(|| streaming_error("position overflow"))?;
    Ok(())
}

/// Packed [m, L, U...] per query/head. Storage is O(rows * heads * dimension),
/// independent of history length. Merge only disjoint partitions of the same Q.
#[derive(Debug, Clone)]
pub struct StreamingF32GqaPartial {
    config: StreamingF32GqaConfig,
    query_start: u64,
    state: Vec<f32>,
}

impl StreamingF32GqaPartial {
    pub fn empty(config: StreamingF32GqaConfig, query_start: u64) -> Result<Self> {
        config.validate()?;
        validate_positions(query_start, config.rows)?;
        let mut state = vec![0.0; config.partial_elements()?];
        for row in state.chunks_exact_mut(config.head_dim + 2) {
            row[0] = f32::NEG_INFINITY;
        }
        Ok(Self {
            config,
            query_start,
            state,
        })
    }
    pub fn state(&self) -> &[f32] {
        &self.state
    }
    pub fn storage_bytes(&self) -> usize {
        self.state.len() * size_of::<f32>()
    }
    #[cfg(feature = "cuda")]
    pub(crate) fn from_state(
        config: StreamingF32GqaConfig,
        query_start: u64,
        state: Vec<f32>,
    ) -> Result<Self> {
        if state.len() != config.partial_elements()? {
            return Err(streaming_error("partial extent mismatch"));
        }
        for row in state.chunks_exact(config.head_dim + 2) {
            if !row[1].is_finite()
                || row[1] < 0.0
                || row[2..].iter().any(|v| !v.is_finite())
                || (row[1] > 0.0 && !row[0].is_finite())
                || (row[1] == 0.0
                    && (row[0] != f32::NEG_INFINITY || row[2..].iter().any(|v| *v != 0.0)))
            {
                return Err(streaming_error("non-finite or invalid device partial"));
            }
        }
        Ok(Self {
            config,
            query_start,
            state,
        })
    }
    pub fn merge(&mut self, other: &Self) -> Result<()> {
        // Block size is an execution choice, not a mathematical identity.
        let mut compatible = other.config;
        compatible.block_tokens = self.config.block_tokens;
        if self.config != compatible || self.query_start != other.query_start {
            return Err(streaming_error("partial query geometry/mask mismatch"));
        }
        let mut merged = self.state.clone();
        for (a, b) in merged
            .chunks_exact_mut(self.config.head_dim + 2)
            .zip(other.state.chunks_exact(self.config.head_dim + 2))
        {
            if b[1] == 0.0 {
                continue;
            }
            if a[1] == 0.0 {
                a.copy_from_slice(b);
                continue;
            }
            let m = a[0].max(b[0]);
            let x = (a[0] - m).exp();
            let y = (b[0] - m).exp();
            for d in 1..a.len() {
                a[d] = a[d] * x + b[d] * y;
                if !a[d].is_finite() {
                    return Err(streaming_error("partial merge overflow"));
                }
            }
            a[0] = m;
        }
        self.state = merged;
        Ok(())
    }
    pub fn output(&self) -> Vec<f32> {
        let mut output =
            Vec::with_capacity(self.config.rows * self.config.q_heads * self.config.head_dim);
        for row in self.state.chunks_exact(self.config.head_dim + 2) {
            output.extend(
                row[2..]
                    .iter()
                    .map(|u| if row[1] == 0.0 { 0.0 } else { u / row[1] }),
            );
        }
        output
    }
}

pub fn streaming_f32_gqa_partial(
    request: StreamingF32GqaRequest<'_>,
) -> Result<StreamingF32GqaPartial> {
    let tokens = request.validate()?;
    let c = request.config;
    let mut partial = StreamingF32GqaPartial::empty(c, request.query_start)?;
    for row in 0..c.rows {
        let position = request.query_start + row as u64;
        for head in 0..c.q_heads {
            let q = (row * c.q_heads + head) * c.head_dim;
            let kv_head = head / (c.q_heads / c.kv_heads);
            let state =
                &mut partial.state[(row * c.q_heads + head) * (c.head_dim + 2)..][..c.head_dim + 2];
            for token in 0..tokens {
                let kp = request.kv_start + token as u64;
                if kp > position || c.sliding_window.is_some_and(|w| position - kp >= w) {
                    continue;
                }
                let k = (token * c.kv_heads + kv_head) * c.head_dim;
                let mut score = 0.0f32;
                for d in 0..c.head_dim {
                    score += request.query[q + d] * request.key[k + d];
                }
                score *= c.softmax_scale;
                if !score.is_finite() {
                    return Err(streaming_error("non-finite visible logit"));
                }
                let rescale = state[1] == 0.0 || score > state[0];
                let weight = if rescale {
                    if state[1] == 0.0 {
                        0.0
                    } else {
                        (state[0] - score).exp()
                    }
                } else {
                    (score - state[0]).exp()
                };
                for d in 0..c.head_dim {
                    let v = request.value[k + d];
                    if !v.is_finite() {
                        return Err(streaming_error("non-finite visible value"));
                    }
                    state[d + 2] = if rescale {
                        state[d + 2] * weight + v
                    } else {
                        state[d + 2] + weight * v
                    };
                    if !state[d + 2].is_finite() {
                        return Err(streaming_error("accumulator overflow"));
                    }
                }
                if rescale {
                    state[0] = score;
                    state[1] = state[1] * weight + 1.0;
                } else {
                    state[1] += weight;
                }
                if !state[1].is_finite() {
                    return Err(streaming_error("denominator overflow"));
                }
            }
        }
    }
    Ok(partial)
}

/// A call-scoped read protection, checked against the actual borrowed K/V.
/// Implementors for live caches must retain the existing read lease throughout
/// the call and reject stale generation, transaction, COW and retirement conflicts.
/// Validation alone does not acquire a lease or establish placement authority.
pub trait StreamingF32GqaProtection {
    fn validate(&self, key: &[f32], value: &[f32]) -> Result<()>;
}

/// Protection for independent Rust-owned vectors only, never a live-cache lease.
pub struct StreamingF32GqaOwnedKv<'a> {
    key: &'a Vec<f32>,
    value: &'a Vec<f32>,
}
impl<'a> StreamingF32GqaOwnedKv<'a> {
    pub fn new(key: &'a Vec<f32>, value: &'a Vec<f32>) -> Self {
        Self { key, value }
    }
}
impl StreamingF32GqaProtection for StreamingF32GqaOwnedKv<'_> {
    fn validate(&self, key: &[f32], value: &[f32]) -> Result<()> {
        if key.as_ptr() != self.key.as_ptr()
            || key.len() != self.key.len()
            || value.as_ptr() != self.value.as_ptr()
            || value.len() != self.value.len()
        {
            return Err(streaming_error(
                "protection does not cover these K/V addresses and extents",
            ));
        }
        validate_kv_nonoverlap(key, value)
    }
}

pub(crate) fn validate_kv_nonoverlap(key: &[f32], value: &[f32]) -> Result<()> {
    if key.is_empty() || value.is_empty() {
        return Ok(());
    }
    let a = key.as_ptr() as usize;
    let b = value.as_ptr() as usize;
    if a < b + size_of_val(value) && b < a + size_of_val(key) {
        return Err(streaming_error("overlapping host K/V"));
    }
    Ok(())
}
