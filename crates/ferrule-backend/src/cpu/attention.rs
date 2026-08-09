//! Reference paged causal grouped-query attention.

use ferrule_common::Result;

use super::operators::{CpuExecutionPrecision, HostRows, RowsArenaId, RowsShape, cpu_error};

#[derive(Debug, Clone, Copy)]
pub struct PagedCausalGqa<'a> {
    pub query: &'a HostRows,
    pub row_sequence_ids: &'a [usize],
    pub row_positions: &'a [usize],
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub softmax_scale: f32,
    pub arena: Option<RowsArenaId>,
}

/// Read access to typed causal KV histories.
pub trait PagedKvHistory {
    fn history(
        &self,
        layer: usize,
        sequence: usize,
        through_position: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<KvHistory>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct KvHistory {
    pub tokens: usize,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
}

pub fn paged_causal_gqa(
    history: &dyn PagedKvHistory,
    layer: usize,
    request: PagedCausalGqa<'_>,
    precision: CpuExecutionPrecision,
) -> Result<HostRows> {
    let rows = request.query.shape().rows();
    let query_width = request
        .query_heads
        .checked_mul(request.head_dim)
        .ok_or_else(|| cpu_error("GQA query width overflow"))?;
    if request.query_heads == 0
        || request.kv_heads == 0
        || !request.query_heads.is_multiple_of(request.kv_heads)
        || request.head_dim == 0
        || request.query.shape() != RowsShape::new(rows, query_width)?
        || request.row_sequence_ids.len() != rows
        || request.row_positions.len() != rows
        || !request.softmax_scale.is_finite()
        || request.softmax_scale <= 0.0
    {
        return Err(cpu_error("invalid paged causal GQA request"));
    }

    let queries_per_kv_head = request.query_heads / request.kv_heads;
    let mut output = vec![0.0; rows * query_width];
    for row in 0..rows {
        let sequence = request.row_sequence_ids[row];
        let position = request.row_positions[row];
        let history = history.history(
            layer,
            sequence,
            position,
            request.kv_heads,
            request.head_dim,
        )?;
        let history_elements = history
            .tokens
            .checked_mul(request.kv_heads)
            .and_then(|elements| elements.checked_mul(request.head_dim))
            .ok_or_else(|| cpu_error("GQA history size overflow"))?;
        if history.tokens == 0
            || history.key.len() != history_elements
            || history.value.len() != history_elements
        {
            return Err(cpu_error("paged GQA history shape mismatch"));
        }
        for query_head in 0..request.query_heads {
            let kv_head = query_head / queries_per_kv_head;
            let query_start = (row * request.query_heads + query_head) * request.head_dim;
            let mut scores = Vec::with_capacity(history.tokens);
            let mut maximum = f32::NEG_INFINITY;
            for token in 0..history.tokens {
                let key_start = (token * request.kv_heads + kv_head) * request.head_dim;
                let mut dot = 0.0f32;
                for dimension in 0..request.head_dim {
                    dot += precision.apply(request.query.values()[query_start + dimension])
                        * history.key[key_start + dimension];
                }
                let score = precision.apply(precision.apply(dot) * request.softmax_scale);
                maximum = maximum.max(score);
                scores.push(score);
            }
            let mut probabilities = scores
                .iter()
                .map(|score| (*score - maximum).exp())
                .collect::<Vec<_>>();
            let denominator = probabilities.iter().sum::<f32>();
            if !denominator.is_finite() || denominator <= 0.0 {
                return Err(cpu_error("paged GQA softmax denominator is invalid"));
            }
            probabilities
                .iter_mut()
                .for_each(|probability| *probability = precision.apply(*probability / denominator));
            for dimension in 0..request.head_dim {
                let mut sum = 0.0f32;
                for (token, probability) in probabilities.iter().enumerate() {
                    let value_start = (token * request.kv_heads + kv_head) * request.head_dim;
                    sum += *probability * history.value[value_start + dimension];
                }
                output[query_start + dimension] = precision.apply(sum);
            }
        }
    }
    HostRows::new(
        request.query.shape(),
        precision.rows_dtype(),
        request.arena,
        output,
    )
}
