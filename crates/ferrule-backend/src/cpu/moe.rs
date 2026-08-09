//! CPU MoE routing, SwiGLU, and weighted reduction.

use ferrule_common::Result;

use super::operators::{
    CpuExecutionPrecision, HostRows, LinearRef, LinearWeight, RowsArenaId, RowsShape, cpu_error,
    reference_linear_rows_into,
};

#[derive(Debug, Clone, PartialEq)]
pub struct RouterRoutes {
    rows: usize,
    top_k: usize,
    expert_ids: Vec<usize>,
    weights: Vec<f32>,
}

impl RouterRoutes {
    pub fn new(
        rows: usize,
        top_k: usize,
        expert_ids: Vec<usize>,
        weights: Vec<f32>,
    ) -> Result<Self> {
        let expected = rows
            .checked_mul(top_k)
            .ok_or_else(|| cpu_error("router output size overflow"))?;
        if rows == 0 || top_k == 0 || expert_ids.len() != expected || weights.len() != expected {
            return Err(cpu_error("invalid packed router output"));
        }
        Ok(Self {
            rows,
            top_k,
            expert_ids,
            weights,
        })
    }

    pub const fn rows(&self) -> usize {
        self.rows
    }

    pub const fn top_k(&self) -> usize {
        self.top_k
    }

    pub fn expert_ids(&self) -> &[usize] {
        &self.expert_ids
    }

    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    pub fn row(&self, row: usize) -> Result<(&[usize], &[f32])> {
        if row >= self.rows {
            return Err(cpu_error(format!(
                "router row {row} exceeds {} rows",
                self.rows
            )));
        }
        let start = row * self.top_k;
        Ok((
            &self.expert_ids[start..start + self.top_k],
            &self.weights[start..start + self.top_k],
        ))
    }
}

pub fn softmax_topk_routes(
    logits: &HostRows,
    top_k: usize,
    route_scale: f32,
    precision: CpuExecutionPrecision,
) -> Result<RouterRoutes> {
    if top_k == 0
        || top_k > logits.shape().width()
        || !route_scale.is_finite()
        || route_scale <= 0.0
    {
        return Err(cpu_error("invalid softmax top-k router policy"));
    }
    let mut expert_ids = Vec::with_capacity(logits.shape().rows() * top_k);
    let mut weights = Vec::with_capacity(logits.shape().rows() * top_k);
    for row in logits.values().chunks_exact(logits.shape().width()) {
        let row = row
            .iter()
            .map(|&value| precision.apply(value))
            .collect::<Vec<_>>();
        if row.iter().any(|value| !value.is_finite()) {
            return Err(cpu_error("router logits must be finite"));
        }
        let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probabilities = row
            .iter()
            .map(|value| (*value - maximum).exp())
            .collect::<Vec<_>>();
        let denominator = probabilities.iter().sum::<f32>();
        probabilities
            .iter_mut()
            .for_each(|probability| *probability /= denominator);
        let mut selected = probabilities.into_iter().enumerate().collect::<Vec<_>>();
        selected.sort_by(|(first_id, first), (second_id, second)| {
            second
                .total_cmp(first)
                .then_with(|| first_id.cmp(second_id))
        });
        selected.truncate(top_k);
        let selected_sum = selected.iter().map(|(_, weight)| *weight).sum::<f32>();
        for (expert, weight) in selected {
            expert_ids.push(expert);
            weights.push(precision.apply(weight / selected_sum * route_scale));
        }
    }
    RouterRoutes::new(logits.shape().rows(), top_k, expert_ids, weights)
}

#[derive(Debug, Clone, Copy)]
pub struct SwiGluRef<'a> {
    pub gate: LinearRef<'a>,
    pub up: LinearRef<'a>,
    pub down: LinearRef<'a>,
    pub activation_limit: Option<f32>,
}

impl SwiGluRef<'_> {
    pub fn validate(self) -> Result<()> {
        self.gate.validate()?;
        self.up.validate()?;
        self.down.validate()?;
        if self.gate.in_features != self.up.in_features
            || self.gate.out_features != self.up.out_features
            || self.down.in_features != self.gate.out_features
            || self
                .activation_limit
                .is_some_and(|limit| !limit.is_finite() || limit <= 0.0)
        {
            return Err(cpu_error("invalid SwiGLU linear bundle"));
        }
        Ok(())
    }
}

pub fn swiglu_rows(
    expert: SwiGluRef<'_>,
    input: &HostRows,
    arena: Option<RowsArenaId>,
    precision: CpuExecutionPrecision,
) -> Result<HostRows> {
    expert.validate()?;
    if input.shape().width() != expert.gate.in_features {
        return Err(cpu_error("SwiGLU input width mismatch"));
    }
    let rows = input.shape().rows();
    let mut prepared = input.values().to_vec();
    precision.apply_slice(&mut prepared);
    let mut gate = vec![0.0; rows * expert.gate.out_features];
    let mut up = vec![0.0; gate.len()];
    reference_linear_rows_into(expert.gate, &prepared, rows, &mut gate)?;
    reference_linear_rows_into(expert.up, &prepared, rows, &mut up)?;
    precision.apply_slice(&mut gate);
    precision.apply_slice(&mut up);
    let mut hidden = Vec::with_capacity(gate.len());
    for (mut gate, mut up) in gate.into_iter().zip(up) {
        if let Some(limit) = expert.activation_limit {
            gate = gate.min(limit);
            up = up.clamp(-limit, limit);
        }
        let activated = precision.apply(gate / (1.0 + (-gate).exp()));
        hidden.push(precision.apply(activated * up));
    }
    let mut output = vec![0.0; rows * expert.down.out_features];
    reference_linear_rows_into(expert.down, &hidden, rows, &mut output)?;
    precision.apply_slice(&mut output);
    HostRows::new(
        RowsShape::new(rows, expert.down.out_features)?,
        precision.rows_dtype(),
        arena,
        output,
    )
}

/// Artifact-preserved expert matrix accepted by the reference MoE provider.
#[derive(Debug, Clone, Copy)]
pub enum ExpertLinearRef<'a> {
    Bf16 {
        weight: &'a [u8],
        out_features: usize,
        in_features: usize,
    },
    Fp4E2M1E8M0 {
        weight: &'a [u8],
        scales: &'a [u8],
        out_features: usize,
        in_features: usize,
        block_size: usize,
    },
}

impl ExpertLinearRef<'_> {
    pub const fn out_features(self) -> usize {
        match self {
            Self::Bf16 { out_features, .. } | Self::Fp4E2M1E8M0 { out_features, .. } => {
                out_features
            }
        }
    }

    pub const fn in_features(self) -> usize {
        match self {
            Self::Bf16 { in_features, .. } | Self::Fp4E2M1E8M0 { in_features, .. } => in_features,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ExpertSwiGluRef<'a> {
    pub gate: ExpertLinearRef<'a>,
    pub up: ExpertLinearRef<'a>,
    pub down: ExpertLinearRef<'a>,
    pub activation_limit: Option<f32>,
}

pub fn execute_reference_expert(
    expert: ExpertSwiGluRef<'_>,
    input: &[f32],
    route_weight: f32,
) -> Result<Vec<f32>> {
    execute_reference_expert_with_hidden_transform(expert, input, route_weight, |_| Ok(()))
}

/// Execute one artifact-preserved expert while allowing the model adapter to
/// apply a declared activation-format boundary before the down projection.
pub fn execute_reference_expert_with_hidden_transform(
    expert: ExpertSwiGluRef<'_>,
    input: &[f32],
    route_weight: f32,
    transform_hidden: impl FnOnce(&mut [f32]) -> Result<()>,
) -> Result<Vec<f32>> {
    if expert.gate.in_features() != input.len()
        || expert.up.in_features() != input.len()
        || expert.gate.out_features() != expert.up.out_features()
        || expert.down.in_features() != expert.gate.out_features()
        || !route_weight.is_finite()
        || expert
            .activation_limit
            .is_some_and(|limit| !limit.is_finite() || limit <= 0.0)
    {
        return Err(cpu_error("invalid reference expert request"));
    }
    let mut gate = expert_linear(expert.gate, input)?;
    let mut up = expert_linear(expert.up, input)?;
    if let Some(limit) = expert.activation_limit {
        gate.iter_mut().for_each(|value| *value = value.min(limit));
        up.iter_mut()
            .for_each(|value| *value = value.clamp(-limit, limit));
    }
    let mut hidden = gate
        .into_iter()
        .zip(up)
        .map(|(gate, up)| gate / (1.0 + (-gate).exp()) * up * route_weight)
        .collect::<Vec<_>>();
    transform_hidden(&mut hidden)?;
    expert_linear(expert.down, &hidden)
}

pub fn expert_linear(linear: ExpertLinearRef<'_>, input: &[f32]) -> Result<Vec<f32>> {
    if input.len() != linear.in_features() {
        return Err(cpu_error(format!(
            "expert input length mismatch: expected {}, got {}",
            linear.in_features(),
            input.len()
        )));
    }
    match linear {
        ExpertLinearRef::Bf16 {
            weight,
            out_features,
            in_features,
        } => {
            let mut output = vec![0.0; out_features];
            reference_linear_rows_into(
                LinearRef {
                    weight: LinearWeight::Bf16(weight),
                    out_features,
                    in_features,
                    bias: None,
                },
                input,
                1,
                &mut output,
            )?;
            Ok(output)
        }
        ExpertLinearRef::Fp4E2M1E8M0 {
            weight,
            scales,
            out_features,
            in_features,
            block_size,
        } => fp4_linear(weight, scales, out_features, in_features, block_size, input),
    }
}

fn fp4_linear(
    weight: &[u8],
    scales: &[u8],
    out_features: usize,
    in_features: usize,
    block_size: usize,
    input: &[f32],
) -> Result<Vec<f32>> {
    if block_size == 0 || !in_features.is_multiple_of(block_size) || !in_features.is_multiple_of(2)
    {
        return Err(cpu_error("invalid FP4 expert shape"));
    }
    let packed_columns = in_features / 2;
    let scale_columns = in_features / block_size;
    let expected_weight = out_features
        .checked_mul(packed_columns)
        .ok_or_else(|| cpu_error("FP4 expert weight size overflow"))?;
    let expected_scales = out_features
        .checked_mul(scale_columns)
        .ok_or_else(|| cpu_error("FP4 expert scale size overflow"))?;
    if weight.len() != expected_weight || scales.len() != expected_scales {
        return Err(cpu_error("FP4 expert payload size mismatch"));
    }
    let mut output = vec![0.0; out_features];
    for row in 0..out_features {
        let mut accumulator = 0.0f32;
        for column in 0..in_features {
            let packed = weight[row * packed_columns + column / 2];
            let nibble = if column % 2 == 0 {
                packed & 0x0f
            } else {
                packed >> 4
            };
            let scale = 2.0f32.powi(scales[row * scale_columns + column / block_size] as i32 - 127);
            accumulator += decode_fp4(nibble) * scale * input[column];
        }
        output[row] = accumulator;
    }
    Ok(output)
}

fn decode_fp4(nibble: u8) -> f32 {
    let magnitude = match nibble & 0x07 {
        0 => 0.0,
        1 => 0.5,
        2 => 1.0,
        3 => 1.5,
        4 => 2.0,
        5 => 3.0,
        6 => 4.0,
        _ => 6.0,
    };
    if nibble & 0x08 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

pub fn weighted_reduce(
    output: &mut [f32],
    update: &[f32],
    route_weight: f32,
    precision: CpuExecutionPrecision,
) -> Result<()> {
    if output.len() != update.len() || !route_weight.is_finite() {
        return Err(cpu_error("weighted expert reduction shape mismatch"));
    }
    for (destination, &value) in output.iter_mut().zip(update) {
        let weighted = precision.apply(value * route_weight);
        *destination = precision.apply(*destination + weighted);
    }
    Ok(())
}
