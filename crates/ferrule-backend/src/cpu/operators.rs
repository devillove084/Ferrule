//! Dense CPU row buffers and reference transformer operators.

use ferrule_common::Result;

use crate::BackendError;

/// Logical dtype at a CPU semantic boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RowsDType {
    F32,
    Bf16,
}

/// Optional reusable-arena identity carried with host rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RowsArenaId(u64);

impl RowsArenaId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Dense row-major shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RowsShape {
    rows: usize,
    width: usize,
}

impl RowsShape {
    pub fn new(rows: usize, width: usize) -> Result<Self> {
        if rows == 0 || width == 0 {
            return Err(cpu_error(format!(
                "row shape must be non-empty, got [{rows}, {width}]"
            )));
        }
        rows.checked_mul(width)
            .ok_or_else(|| cpu_error("row shape element count overflow"))?;
        Ok(Self { rows, width })
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn width(self) -> usize {
        self.width
    }

    pub const fn elements(self) -> usize {
        self.rows * self.width
    }
}

/// Host-owned dense rows. BF16 rows retain exactly rounded values in F32 storage.
#[derive(Debug, Clone, PartialEq)]
pub struct HostRows {
    shape: RowsShape,
    dtype: RowsDType,
    arena: Option<RowsArenaId>,
    values: Vec<f32>,
}

impl HostRows {
    pub fn new(
        shape: RowsShape,
        dtype: RowsDType,
        arena: Option<RowsArenaId>,
        values: Vec<f32>,
    ) -> Result<Self> {
        if values.len() != shape.elements() {
            return Err(cpu_error(format!(
                "host rows contain {} values for shape [{}, {}]",
                values.len(),
                shape.rows(),
                shape.width()
            )));
        }
        Ok(Self {
            shape,
            dtype,
            arena,
            values,
        })
    }

    pub fn zeros(shape: RowsShape, dtype: RowsDType, arena: Option<RowsArenaId>) -> Self {
        Self {
            shape,
            dtype,
            arena,
            values: vec![0.0; shape.elements()],
        }
    }

    pub const fn shape(&self) -> RowsShape {
        self.shape
    }

    pub const fn dtype(&self) -> RowsDType {
        self.dtype
    }

    pub const fn arena(&self) -> Option<RowsArenaId> {
        self.arena
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn values_mut(&mut self) -> &mut [f32] {
        &mut self.values
    }

    pub fn into_values(self) -> Vec<f32> {
        self.values
    }
}

/// Observable CPU execution precision contract.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum CpuExecutionPrecision {
    #[default]
    F32,
    Bf16,
}

impl CpuExecutionPrecision {
    pub const fn rows_dtype(self) -> RowsDType {
        match self {
            Self::F32 => RowsDType::F32,
            Self::Bf16 => RowsDType::Bf16,
        }
    }

    pub fn apply(self, value: f32) -> f32 {
        match self {
            Self::F32 => value,
            Self::Bf16 => bf16_rne(value),
        }
    }

    pub fn apply_slice(self, values: &mut [f32]) {
        if self == Self::Bf16 {
            values
                .iter_mut()
                .for_each(|value| *value = bf16_rne(*value));
        }
    }
}

/// Borrowed row-major matrix storage.
#[derive(Debug, Clone, Copy)]
pub enum LinearWeight<'a> {
    F32(&'a [f32]),
    Bf16(&'a [u8]),
}

/// Borrowed linear projection contract.
#[derive(Debug, Clone, Copy)]
pub struct LinearRef<'a> {
    pub weight: LinearWeight<'a>,
    pub out_features: usize,
    pub in_features: usize,
    pub bias: Option<&'a [f32]>,
}

impl LinearRef<'_> {
    pub fn validate(self) -> Result<()> {
        if self.out_features == 0 || self.in_features == 0 {
            return Err(cpu_error("linear dimensions must be non-zero"));
        }
        let elements = self
            .out_features
            .checked_mul(self.in_features)
            .ok_or_else(|| cpu_error("linear weight element count overflow"))?;
        let valid_weight = match self.weight {
            LinearWeight::F32(values) => values.len() == elements,
            LinearWeight::Bf16(bytes) => elements
                .checked_mul(2)
                .is_some_and(|expected| bytes.len() == expected),
        };
        if !valid_weight {
            return Err(cpu_error(format!(
                "linear weight storage does not match [{}, {}]",
                self.out_features, self.in_features
            )));
        }
        if self
            .bias
            .is_some_and(|bias| bias.len() != self.out_features)
        {
            return Err(cpu_error("linear bias length does not match output width"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotaryPairing {
    SplitHalf,
    Interleaved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotaryRegion {
    Prefix,
    Tail,
}

#[derive(Debug, Clone, Copy)]
pub struct RopeRef<'a> {
    pub positions: usize,
    pub dimensions: usize,
    pub cosine: &'a [f32],
    pub sine: &'a [f32],
}

impl RopeRef<'_> {
    pub fn validate(self) -> Result<()> {
        let expected = self
            .positions
            .checked_mul(self.dimensions / 2)
            .ok_or_else(|| cpu_error("RoPE table size overflow"))?;
        if self.positions == 0
            || self.dimensions == 0
            || !self.dimensions.is_multiple_of(2)
            || self.cosine.len() != expected
            || self.sine.len() != expected
        {
            return Err(cpu_error("invalid RoPE table"));
        }
        Ok(())
    }
}

pub fn embedding_rows(
    embedding: LinearRef<'_>,
    token_ids: &[u32],
    arena: Option<RowsArenaId>,
    precision: CpuExecutionPrecision,
) -> Result<HostRows> {
    embedding.validate()?;
    if token_ids.is_empty() || embedding.bias.is_some() {
        return Err(cpu_error(
            "embedding requires tokens and an unbiased row-major matrix",
        ));
    }
    let mut output = Vec::with_capacity(
        token_ids
            .len()
            .checked_mul(embedding.in_features)
            .ok_or_else(|| cpu_error("embedding output size overflow"))?,
    );
    for &token in token_ids {
        let token = usize::try_from(token)
            .map_err(|_| cpu_error("embedding token does not fit host address space"))?;
        if token >= embedding.out_features {
            return Err(cpu_error(format!(
                "embedding token {token} exceeds vocabulary {}",
                embedding.out_features
            )));
        }
        match embedding.weight {
            LinearWeight::F32(values) => {
                let start = token * embedding.in_features;
                output.extend_from_slice(&values[start..start + embedding.in_features]);
            }
            LinearWeight::Bf16(bytes) => {
                let start = token
                    .checked_mul(embedding.in_features)
                    .and_then(|element| element.checked_mul(2))
                    .ok_or_else(|| cpu_error("embedding BF16 row offset overflow"))?;
                output.extend(
                    bytes[start..start + embedding.in_features * 2]
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|bytes| bf16_word_value(u16::from_le_bytes(*bytes))),
                );
            }
        }
    }
    precision.apply_slice(&mut output);
    HostRows::new(
        RowsShape::new(token_ids.len(), embedding.in_features)?,
        precision.rows_dtype(),
        arena,
        output,
    )
}

pub fn linear_rows(
    linear: LinearRef<'_>,
    input: &HostRows,
    arena: Option<RowsArenaId>,
    precision: CpuExecutionPrecision,
) -> Result<HostRows> {
    if input.shape().width() != linear.in_features {
        return Err(cpu_error(format!(
            "linear input width {} does not match {}",
            input.shape().width(),
            linear.in_features
        )));
    }
    let mut prepared = input.values().to_vec();
    precision.apply_slice(&mut prepared);
    let mut output = vec![0.0; input.shape().rows() * linear.out_features];
    reference_linear_rows_into(linear, &prepared, input.shape().rows(), &mut output)?;
    precision.apply_slice(&mut output);
    HostRows::new(
        RowsShape::new(input.shape().rows(), linear.out_features)?,
        precision.rows_dtype(),
        arena,
        output,
    )
}

pub fn reference_linear_rows_into(
    linear: LinearRef<'_>,
    input: &[f32],
    rows: usize,
    output: &mut [f32],
) -> Result<()> {
    linear.validate()?;
    let expected_input = rows
        .checked_mul(linear.in_features)
        .ok_or_else(|| cpu_error("linear input size overflow"))?;
    let expected_output = rows
        .checked_mul(linear.out_features)
        .ok_or_else(|| cpu_error("linear output size overflow"))?;
    if input.len() != expected_input || output.len() != expected_output {
        return Err(cpu_error(format!(
            "linear row shape mismatch: input={}/{} output={}/{}",
            input.len(),
            expected_input,
            output.len(),
            expected_output
        )));
    }
    for input_row in 0..rows {
        let input_start = input_row * linear.in_features;
        let output_start = input_row * linear.out_features;
        for weight_row in 0..linear.out_features {
            let mut accumulator = 0.0f32;
            match linear.weight {
                LinearWeight::F32(weight) => {
                    let weight_start = weight_row * linear.in_features;
                    for column in 0..linear.in_features {
                        accumulator += weight[weight_start + column] * input[input_start + column];
                    }
                }
                LinearWeight::Bf16(weight) => {
                    let weight_start = weight_row * linear.in_features * 2;
                    let row = &weight[weight_start..weight_start + linear.in_features * 2];
                    for (column, bytes) in row.as_chunks::<2>().0.iter().enumerate() {
                        accumulator += bf16_word_value(u16::from_le_bytes(*bytes))
                            * input[input_start + column];
                    }
                }
            }
            output[output_start + weight_row] =
                accumulator + linear.bias.map_or(0.0, |bias| bias[weight_row]);
        }
    }
    Ok(())
}

pub fn rms_norm_rows(
    input: &HostRows,
    weight: &[f32],
    epsilon: f32,
    heads: usize,
    arena: Option<RowsArenaId>,
    precision: CpuExecutionPrecision,
) -> Result<HostRows> {
    if heads == 0
        || weight.is_empty()
        || !input.shape().width().is_multiple_of(heads)
        || weight.len() != input.shape().width() / heads
        || !epsilon.is_finite()
        || epsilon <= 0.0
    {
        return Err(cpu_error(format!(
            "RMSNorm shape mismatch: input=[{},{}] heads={heads} weight={}",
            input.shape().rows(),
            input.shape().width(),
            weight.len()
        )));
    }
    let width = weight.len();
    let mut output = vec![0.0; input.values().len()];
    for row in 0..input.shape().rows() * heads {
        let start = row * width;
        let source = &input.values()[start..start + width];
        let sum = source
            .iter()
            .map(|&value| {
                let value = precision.apply(value);
                value * value
            })
            .sum::<f32>();
        let inverse_rms = (sum / width as f32 + epsilon).sqrt().recip();
        for column in 0..width {
            let value = precision.apply(source[column]);
            let normalized = precision.apply(value * inverse_rms);
            output[start + column] = precision.apply(normalized * weight[column]);
        }
    }
    HostRows::new(input.shape(), precision.rows_dtype(), arena, output)
}

#[allow(clippy::too_many_arguments)]
pub fn rope_rows(
    mut input: HostRows,
    table: RopeRef<'_>,
    pairing: RotaryPairing,
    region: RotaryRegion,
    heads: usize,
    head_dim: usize,
    positions: &[usize],
    precision: CpuExecutionPrecision,
) -> Result<HostRows> {
    table.validate()?;
    let shape = input.shape();
    if heads == 0
        || shape.width() != heads * head_dim
        || positions.len() != shape.rows()
        || table.dimensions > head_dim
        || positions
            .iter()
            .any(|&position| position >= table.positions)
    {
        return Err(cpu_error("RoPE row shape or position mismatch"));
    }
    precision.apply_slice(input.values_mut());
    let table_width = table.dimensions / 2;
    for (row, &position) in positions.iter().enumerate() {
        let table_start = position * table_width;
        for head in 0..heads {
            let head_start = (row * heads + head) * head_dim;
            let region_start = match region {
                RotaryRegion::Prefix => head_start,
                RotaryRegion::Tail => head_start + head_dim - table.dimensions,
            };
            for pair in 0..table_width {
                let (first_index, second_index) = match pairing {
                    RotaryPairing::SplitHalf => {
                        (region_start + pair, region_start + table_width + pair)
                    }
                    RotaryPairing::Interleaved => {
                        (region_start + pair * 2, region_start + pair * 2 + 1)
                    }
                };
                let first = input.values()[first_index];
                let second = input.values()[second_index];
                let cosine = table.cosine[table_start + pair];
                let sine = table.sine[table_start + pair];
                input.values_mut()[first_index] = first * cosine - second * sine;
                input.values_mut()[second_index] = first * sine + second * cosine;
            }
        }
    }
    precision.apply_slice(input.values_mut());
    Ok(input)
}

/// Row-wise RMSNorm for one dense vector.
pub fn rms_norm(input: &[f32], weight: &[f32], epsilon: f32) -> Result<Vec<f32>> {
    if input.is_empty() || input.len() != weight.len() || !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(cpu_error(format!(
            "RMSNorm length mismatch: input={} weight={}",
            input.len(),
            weight.len()
        )));
    }
    let inverse_rms = (input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32
        + epsilon)
        .sqrt()
        .recip();
    Ok(input
        .iter()
        .zip(weight)
        .map(|(value, weight)| value * inverse_rms * weight)
        .collect())
}

/// In-place per-head RMSNorm without an affine weight.
pub fn rms_norm_heads_in_place(
    values: &mut [f32],
    heads: usize,
    head_dim: usize,
    epsilon: f32,
) -> Result<()> {
    let expected = heads
        .checked_mul(head_dim)
        .ok_or_else(|| cpu_error("per-head RMSNorm shape overflow"))?;
    if heads == 0
        || head_dim == 0
        || values.len() != expected
        || !epsilon.is_finite()
        || epsilon <= 0.0
    {
        return Err(cpu_error(format!(
            "per-head RMSNorm length mismatch: expected {expected}, got {}",
            values.len()
        )));
    }
    for row in values.chunks_exact_mut(head_dim) {
        let inverse_rms = (row.iter().map(|value| value * value).sum::<f32>() / head_dim as f32
            + epsilon)
            .sqrt()
            .recip();
        row.iter_mut().for_each(|value| *value *= inverse_rms);
    }
    Ok(())
}

pub fn dot(first: &[f32], second: &[f32]) -> Result<f32> {
    if first.len() != second.len() {
        return Err(cpu_error("dot-product length mismatch"));
    }
    Ok(first
        .iter()
        .zip(second)
        .map(|(first, second)| first * second)
        .sum())
}

pub fn residual_rows(
    mut residual: HostRows,
    update: &HostRows,
    precision: CpuExecutionPrecision,
) -> Result<HostRows> {
    if residual.shape() != update.shape() {
        return Err(cpu_error("residual row shape mismatch"));
    }
    for (destination, &source) in residual.values_mut().iter_mut().zip(update.values()) {
        *destination = precision.apply(precision.apply(*destination) + precision.apply(source));
    }
    Ok(residual)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotaryFrequencyParams {
    pub theta: f32,
    pub original_sequence_length: usize,
    pub factor: f32,
    pub beta_fast: usize,
    pub beta_slow: usize,
}

impl RotaryFrequencyParams {
    pub const fn plain(theta: f32) -> Self {
        Self {
            theta,
            original_sequence_length: 0,
            factor: 1.0,
            beta_fast: 32,
            beta_slow: 1,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn apply_rotary_tail_scaled(
    values: &mut [f32],
    heads: usize,
    head_dim: usize,
    rope_dim: usize,
    position: usize,
    parameters: RotaryFrequencyParams,
    inverse: bool,
) -> Result<()> {
    if rope_dim == 0 {
        return Ok(());
    }
    if rope_dim > head_dim
        || !rope_dim.is_multiple_of(2)
        || values.len() != heads * head_dim
        || !parameters.theta.is_finite()
        || parameters.theta <= 0.0
        || !parameters.factor.is_finite()
        || parameters.factor <= 0.0
    {
        return Err(cpu_error(format!(
            "rotary tail shape mismatch: values={} heads={heads} head_dim={head_dim} rope_dim={rope_dim}",
            values.len()
        )));
    }
    let tail_start = head_dim - rope_dim;
    for head in 0..heads {
        let base = head * head_dim + tail_start;
        for pair in 0..rope_dim / 2 {
            let frequency = rotary_frequency(pair, rope_dim, parameters);
            let angle = position as f32 * frequency;
            let (sin, cos) = angle.sin_cos();
            let sin = if inverse { -sin } else { sin };
            let first = values[base + 2 * pair];
            let second = values[base + 2 * pair + 1];
            values[base + 2 * pair] = first * cos - second * sin;
            values[base + 2 * pair + 1] = first * sin + second * cos;
        }
    }
    Ok(())
}

pub fn apply_rotary_split_half_indexed(
    values: &mut [f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    positions: &[usize],
    theta: f32,
) -> Result<()> {
    if heads == 0 || head_dim == 0 {
        return Err(cpu_error(
            "split-half rotary requires non-zero heads and head_dim",
        ));
    }
    if rotary_dim == 0 || rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
        return Err(cpu_error(
            "split-half rotary_dim must be non-zero, even, and <= head_dim",
        ));
    }
    if !theta.is_finite() || theta <= 0.0 {
        return Err(cpu_error(
            "split-half rotary theta must be finite and positive",
        ));
    }
    let expected = rows
        .checked_mul(heads)
        .and_then(|count| count.checked_mul(head_dim))
        .ok_or_else(|| cpu_error("split-half rotary shape overflows"))?;
    if values.len() != expected {
        return Err(cpu_error(format!(
            "split-half rotary values length mismatch: expected {expected}, got {}",
            values.len()
        )));
    }
    if positions.len() != rows {
        return Err(cpu_error(format!(
            "split-half rotary positions length mismatch: expected {rows}, got {}",
            positions.len()
        )));
    }
    if rows == 0 {
        return Ok(());
    }
    let half = rotary_dim / 2;
    let frequencies = (0..half)
        .map(|pair| theta.powf(-((2 * pair) as f32 / rotary_dim as f32)))
        .collect::<Vec<_>>();
    for (row, &position) in positions.iter().enumerate() {
        for (pair, &frequency) in frequencies.iter().enumerate() {
            if !(position as f32 * frequency).is_finite() {
                return Err(cpu_error(format!(
                    "split-half rotary angle overflow at row {row}, pair {pair}, position {position}"
                )));
            }
        }
    }
    for (row, &position) in positions.iter().enumerate() {
        for head in 0..heads {
            let base = (row * heads + head) * head_dim;
            for (pair, &frequency) in frequencies.iter().enumerate() {
                let (sin, cos) = (position as f32 * frequency).sin_cos();
                let first = values[base + pair];
                let second = values[base + half + pair];
                values[base + pair] = first * cos - second * sin;
                values[base + half + pair] = first * sin + second * cos;
            }
        }
    }
    Ok(())
}

pub fn rotary_frequency(pair: usize, rope_dim: usize, parameters: RotaryFrequencyParams) -> f32 {
    let base_frequency = 1.0 / parameters.theta.powf((2 * pair) as f32 / rope_dim as f32);
    if parameters.original_sequence_length == 0 || parameters.factor == 1.0 {
        return base_frequency;
    }
    let (low, high) = rotary_correction_range(
        parameters.beta_fast as f32,
        parameters.beta_slow as f32,
        rope_dim,
        parameters.theta,
        parameters.original_sequence_length as f32,
    );
    let ramp = rotary_linear_ramp(pair as f32, low as f32, high as f32);
    let smooth = 1.0 - ramp;
    base_frequency / parameters.factor * (1.0 - smooth) + base_frequency * smooth
}

pub fn rotary_correction_range(
    low_rotations: f32,
    high_rotations: f32,
    dimensions: usize,
    theta: f32,
    maximum_position: f32,
) -> (usize, usize) {
    let low = rotary_correction_dimension(low_rotations, dimensions, theta, maximum_position)
        .floor() as isize;
    let high = rotary_correction_dimension(high_rotations, dimensions, theta, maximum_position)
        .ceil() as isize;
    (
        low.max(0) as usize,
        high.min(dimensions as isize - 1).max(0) as usize,
    )
}

pub fn rotary_correction_dimension(
    rotations: f32,
    dimensions: usize,
    theta: f32,
    maximum_position: f32,
) -> f32 {
    dimensions as f32 * (maximum_position / (rotations * 2.0 * std::f32::consts::PI)).ln()
        / (2.0 * theta.ln())
}

pub fn rotary_linear_ramp(value: f32, minimum: f32, mut maximum: f32) -> f32 {
    if (minimum - maximum).abs() < f32::EPSILON {
        maximum += 0.001;
    }
    ((value - minimum) / (maximum - minimum)).clamp(0.0, 1.0)
}

/// Round to BF16 with round-to-nearest, ties-to-even and quiet-NaN preservation.
pub fn bf16_rne(value: f32) -> f32 {
    bf16_word_value(bf16_rne_word(value))
}

pub fn bf16_rne_word(value: f32) -> u16 {
    let bits = value.to_bits();
    if bits & 0x7fff_ffff > 0x7f80_0000 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let tie_to_even_bias = 0x7fff + ((bits >> 16) & 1);
    bits.wrapping_add(tie_to_even_bias).wrapping_shr(16) as u16
}

pub fn bf16_word_value(word: u16) -> f32 {
    f32::from_bits(u32::from(word) << 16)
}

pub(crate) fn cpu_error(message: impl Into<String>) -> ferrule_common::Error {
    BackendError::Invariant {
        message: format!("CPU backend: {}", message.into()),
    }
    .into()
}
