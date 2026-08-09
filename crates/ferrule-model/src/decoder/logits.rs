use super::LogitsPlan;
use ferrule_common::execution::{
    ExecutionOutput, LogitsOutput, LogitsRequest, LogitsRow, TokenLogit,
};
use ferrule_common::{Error, Result};
use std::collections::BTreeSet;
/// One forward-produced sparse top-k source row.
#[derive(Debug, Clone, PartialEq)]
pub struct DecoderTopKRow {
    input_row: usize,
    candidates: Vec<TokenLogit>,
}
impl DecoderTopKRow {
    pub fn new(input_row: usize, candidates: Vec<TokenLogit>) -> Self {
        Self {
            input_row,
            candidates,
        }
    }
    pub const fn input_row(&self) -> usize {
        self.input_row
    }
    pub fn candidates(&self) -> &[TokenLogit] {
        &self.candidates
    }
}
/// Contiguous row-major dense logits produced by decoder forward execution.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseLogits {
    values: Box<[f32]>,
    rows: usize,
    width: usize,
}
impl DenseLogits {
    pub fn new(rows: usize, width: usize, values: impl Into<Box<[f32]>>) -> Result<Self> {
        if width == 0 {
            return Err(execution_error(
                "dense decoder logits width must be non-zero",
            ));
        }
        let expected = rows
            .checked_mul(width)
            .ok_or_else(|| execution_error("dense decoder logits shape overflows usize"))?;
        let values = values.into();
        if values.len() != expected {
            return Err(execution_error(format!(
                "dense decoder logits contain {} values, expected {rows}x{width}={expected}",
                values.len()
            )));
        }
        Ok(Self {
            values,
            rows,
            width,
        })
    }
    /// Convenience boundary for callers that still materialize independent rows.
    pub fn from_rows(rows: Vec<Vec<f32>>) -> Result<Self> {
        let row_count = rows.len();
        let width = rows.first().map(Vec::len).unwrap_or_default();
        if rows.iter().any(|row| row.len() != width) {
            return Err(execution_error(
                "dense decoder logits rows must have one rectangular width",
            ));
        }
        Self::new(
            row_count,
            width,
            rows.into_iter().flatten().collect::<Vec<_>>(),
        )
    }
    pub const fn rows(&self) -> usize {
        self.rows
    }
    pub const fn width(&self) -> usize {
        self.width
    }
    pub fn values(&self) -> &[f32] {
        &self.values
    }
    pub fn row(&self, row: usize) -> Option<&[f32]> {
        let start = row.checked_mul(self.width)?;
        self.values.get(start..start.checked_add(self.width)?)
    }
}
/// Backend-neutral forward logits before execution-protocol projection.
#[derive(Debug, Clone, PartialEq)]
pub enum DecoderLogits {
    /// One dense vocabulary row for every packed input row.
    Dense(DenseLogits),
    /// Sparse candidates for exactly the requested top-k input rows.
    TopKRows(Vec<DecoderTopKRow>),
    /// No forward logits. Valid only when the plan requests no rows.
    None,
}
impl DecoderLogits {
    pub fn lower(self, plan: &LogitsPlan) -> Result<ExecutionOutput> {
        match self {
            Self::Dense(rows) => lower_dense(rows, plan),
            Self::TopKRows(rows) => lower_top_k_rows(rows, plan),
            Self::None if plan.is_empty() => Ok(ExecutionOutput::default()),
            Self::None => Err(execution_error(format!(
                "decoder forward execution returned no logits for {} requested rows",
                plan.rows().len()
            ))),
        }
    }
}
fn lower_dense(rows: DenseLogits, plan: &LogitsPlan) -> Result<ExecutionOutput> {
    if rows.rows() != plan.packed_row_count() {
        return Err(execution_error(format!(
            "dense decoder logits contain {} rows, expected one for each of {} packed rows",
            rows.rows(),
            plan.packed_row_count()
        )));
    }
    let dense_width = rows.width();
    if let Some(expected_width) = plan.full_logits_width()
        && dense_width != expected_width
    {
        return Err(execution_error(format!(
            "dense decoder logits have width {dense_width}, expected {expected_width}"
        )));
    }
    for (input_row, row) in rows.values().chunks_exact(dense_width).enumerate() {
        if let Some(token_id) = row.iter().position(|logit| !logit.is_finite()) {
            return Err(execution_error(format!(
                "dense decoder logits row {input_row} token {token_id} is non-finite ({})",
                row[token_id]
            )));
        }
    }
    let mut output = Vec::with_capacity(plan.rows().len());
    for requested in plan.rows() {
        let input_row = requested.input_row();
        let source = rows.row(input_row).ok_or_else(|| {
            execution_error(format!(
                "logits plan row {input_row} is outside {} dense rows",
                rows.rows()
            ))
        })?;
        let logits = match requested.request() {
            LogitsRequest::None => {
                return Err(execution_error(format!(
                    "logits plan unexpectedly contains None request at row {input_row}"
                )));
            }
            LogitsRequest::Full => {
                if let Some(expected_width) = plan.full_logits_width()
                    && source.len() != expected_width
                {
                    return Err(execution_error(format!(
                        "dense decoder logits row {input_row} has width {}, expected {expected_width}",
                        source.len()
                    )));
                }
                LogitsOutput::Full(source.to_vec())
            }
            LogitsRequest::TopK(k) => {
                let requested = usize::try_from(k.get())
                    .map_err(|_| execution_error("decoder top-k width exceeds usize"))?;
                let mut candidates = source
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(token_id, logit)| {
                        u32::try_from(token_id)
                            .map(|token_id| TokenLogit::new(token_id, logit))
                            .map_err(|_| {
                                execution_error("decoder vocabulary token ID exceeds the u32 ABI")
                            })
                    })
                    .collect::<Result<Vec<_>>>()?;
                sort_and_validate_candidates(input_row, &mut candidates)?;
                candidates.truncate(requested);
                LogitsOutput::TopK(candidates)
            }
        };
        output.push(LogitsRow::new(input_row_u32(input_row)?, logits));
    }
    Ok(ExecutionOutput::new(output))
}
fn lower_top_k_rows(rows: Vec<DecoderTopKRow>, plan: &LogitsPlan) -> Result<ExecutionOutput> {
    if plan
        .rows()
        .iter()
        .any(|row| matches!(row.request(), LogitsRequest::Full))
    {
        return Err(execution_error(
            "sparse decoder top-k rows cannot satisfy a full-logits request",
        ));
    }
    if rows.len() != plan.rows().len() {
        return Err(execution_error(format!(
            "decoder supplied {} sparse top-k rows for {} requested rows",
            rows.len(),
            plan.rows().len()
        )));
    }
    let mut output = Vec::with_capacity(rows.len());
    for (source, requested) in rows.into_iter().zip(plan.rows()) {
        let input_row = requested.input_row();
        if source.input_row != input_row {
            return Err(execution_error(format!(
                "sparse decoder logits row {} is bound to input {}, expected {input_row}",
                output.len(),
                source.input_row
            )));
        }
        let LogitsRequest::TopK(k) = requested.request() else {
            return Err(execution_error(format!(
                "sparse decoder logits unexpectedly target non-top-k row {input_row}"
            )));
        };
        let mut candidates = source.candidates;
        sort_and_validate_candidates(input_row, &mut candidates)?;
        let requested = usize::try_from(k.get())
            .map_err(|_| execution_error("decoder top-k width exceeds usize"))?;
        candidates.truncate(requested);
        output.push(LogitsRow::new(
            input_row_u32(input_row)?,
            LogitsOutput::TopK(candidates),
        ));
    }
    Ok(ExecutionOutput::new(output))
}
fn sort_and_validate_candidates(input_row: usize, candidates: &mut [TokenLogit]) -> Result<()> {
    let mut token_ids = BTreeSet::new();
    for (candidate_index, candidate) in candidates.iter().enumerate() {
        if !candidate.logit.is_finite() {
            return Err(execution_error(format!(
                "decoder logits row {input_row} candidate {candidate_index} is non-finite ({})",
                candidate.logit
            )));
        }
        if !token_ids.insert(candidate.token_id) {
            return Err(execution_error(format!(
                "decoder logits row {input_row} repeats token ID {}",
                candidate.token_id
            )));
        }
    }
    candidates.sort_by(|first, second| {
        if first.logit == second.logit {
            first.token_id.cmp(&second.token_id)
        } else {
            second.logit.total_cmp(&first.logit)
        }
    });
    Ok(())
}
fn input_row_u32(input_row: usize) -> Result<u32> {
    u32::try_from(input_row)
        .map_err(|_| execution_error("decoder output input row exceeds the u32 ABI"))
}
fn execution_error(message: impl Into<String>) -> Error {
    Error::Execution {
        message: message.into(),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use ferrule_common::execution::{
        ExecutionBatch, ExecutionSequence, ForwardMode, ForwardPhase, StateSlot,
    };
    use std::num::NonZeroU32;
    fn plan(requests: Vec<LogitsRequest>) -> LogitsPlan {
        let rows = requests.len();
        LogitsPlan::from_batch(&ExecutionBatch::new(
            ForwardMode::Prefill,
            vec![0; rows],
            (0..u32::try_from(rows).unwrap()).collect(),
            vec![None; rows],
            requests,
            vec![ExecutionSequence::new(
                StateSlot::new(0),
                ForwardPhase::Prefill,
                0..u32::try_from(rows).unwrap(),
                0,
                u32::try_from(rows).unwrap(),
                0..0,
            )],
            vec![],
        ))
    }
    #[test]
    fn dense_projection_is_requested_row_only_and_has_stable_ties() {
        let output_plan = plan(vec![
            LogitsRequest::None,
            LogitsRequest::TopK(NonZeroU32::new(3).unwrap()),
            LogitsRequest::Full,
        ]);
        let output = DecoderLogits::Dense(
            DenseLogits::from_rows(vec![
                vec![0.0, 1.0, -2.0, -3.0],
                vec![1.0, 2.0, 2.0, -1.0],
                vec![3.0, 4.0, 5.0, 6.0],
            ])
            .unwrap(),
        )
        .lower(&output_plan)
        .unwrap();
        assert_eq!(output.logits.len(), 2);
        assert_eq!(output.logits[0].input_row, 1);
        assert_eq!(
            output.logits[0].logits,
            LogitsOutput::TopK(vec![
                TokenLogit::new(1, 2.0),
                TokenLogit::new(2, 2.0),
                TokenLogit::new(0, 1.0),
            ])
        );
        assert_eq!(
            output.logits[1].logits,
            LogitsOutput::Full(vec![3.0, 4.0, 5.0, 6.0])
        );
        let ragged = DenseLogits::from_rows(vec![vec![0.0], vec![1.0, 2.0], vec![3.0]]);
        assert!(ragged.is_err());
        let signed_zero_plan = plan(vec![LogitsRequest::TopK(NonZeroU32::new(2).unwrap())]);
        let signed_zero =
            DecoderLogits::Dense(DenseLogits::from_rows(vec![vec![-0.0, 0.0]]).unwrap())
                .lower(&signed_zero_plan)
                .unwrap();
        assert_eq!(
            signed_zero.logits[0].logits,
            LogitsOutput::TopK(vec![TokenLogit::new(0, -0.0), TokenLogit::new(1, 0.0)])
        );
    }
    #[test]
    fn sparse_rows_require_exact_requested_rows_and_reject_non_finite_values() {
        let plan = plan(vec![LogitsRequest::TopK(NonZeroU32::new(1).unwrap())]);
        let wrong_row =
            DecoderLogits::TopKRows(vec![DecoderTopKRow::new(1, vec![TokenLogit::new(0, 1.0)])]);
        assert!(wrong_row.lower(&plan).is_err());
        let non_finite = DecoderLogits::TopKRows(vec![DecoderTopKRow::new(
            0,
            vec![TokenLogit::new(0, f32::NAN)],
        )]);
        assert!(non_finite.lower(&plan).is_err());
    }
}
