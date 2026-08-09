use thiserror::Error;

/// Inference linear projection descriptor. Matrix storage is `[out, in]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Linear {
    in_features: usize,
    out_features: usize,
    bias: bool,
}

impl Linear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        bias: bool,
    ) -> Result<Self, DescriptorError> {
        non_zero("linear.in_features", in_features)?;
        non_zero("linear.out_features", out_features)?;
        Ok(Self {
            in_features,
            out_features,
            bias,
        })
    }

    pub const fn in_features(&self) -> usize {
        self.in_features
    }

    pub const fn out_features(&self) -> usize {
        self.out_features
    }

    pub const fn has_bias(&self) -> bool {
        self.bias
    }

    pub fn weight_shape(&self) -> [usize; 2] {
        [self.out_features, self.in_features]
    }

    pub fn bias_shape(&self) -> Option<[usize; 1]> {
        self.bias.then_some([self.out_features])
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Embedding {
    num_embeddings: usize,
    embedding_dim: usize,
    padding_index: Option<usize>,
}

impl Embedding {
    pub fn new(
        num_embeddings: usize,
        embedding_dim: usize,
        padding_index: Option<usize>,
    ) -> Result<Self, DescriptorError> {
        non_zero("embedding.num_embeddings", num_embeddings)?;
        non_zero("embedding.embedding_dim", embedding_dim)?;
        if padding_index.is_some_and(|index| index >= num_embeddings) {
            return Err(DescriptorError::Inconsistent {
                component: "embedding",
                message: format!(
                    "padding index {} is outside vocabulary size {num_embeddings}",
                    padding_index.expect("padding index was present")
                ),
            });
        }
        Ok(Self {
            num_embeddings,
            embedding_dim,
            padding_index,
        })
    }

    pub const fn num_embeddings(&self) -> usize {
        self.num_embeddings
    }

    pub const fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    pub const fn padding_index(&self) -> Option<usize> {
        self.padding_index
    }

    pub fn weight_shape(&self) -> [usize; 2] {
        [self.num_embeddings, self.embedding_dim]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RmsNorm {
    hidden_size: usize,
    epsilon: f32,
}

impl RmsNorm {
    pub fn new(hidden_size: usize, epsilon: f32) -> Result<Self, DescriptorError> {
        non_zero("rms_norm.hidden_size", hidden_size)?;
        positive_finite("rms_norm.epsilon", epsilon)?;
        Ok(Self {
            hidden_size,
            epsilon,
        })
    }

    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub const fn epsilon(&self) -> f32 {
        self.epsilon
    }

    pub fn weight_shape(&self) -> [usize; 1] {
        [self.hidden_size]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotaryPairing {
    SplitHalf,
    Interleaved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotaryRegion {
    Prefix { dimensions: usize },
    Tail { dimensions: usize },
}

impl RotaryRegion {
    pub const fn dimensions(self) -> usize {
        match self {
            Self::Prefix { dimensions } | Self::Tail { dimensions } => dimensions,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum RotaryScaling {
    None,
    Linear {
        factor: f32,
    },
    YaRN {
        factor: f32,
        original_max_position_embeddings: usize,
        beta_fast: f32,
        beta_slow: f32,
        attention_factor: Option<f32>,
    },
}

impl RotaryScaling {
    fn validate(&self) -> Result<(), DescriptorError> {
        match self {
            Self::None => Ok(()),
            Self::Linear { factor } => positive_finite("rotary.linear.factor", *factor),
            Self::YaRN {
                factor,
                original_max_position_embeddings,
                beta_fast,
                beta_slow,
                attention_factor,
            } => {
                positive_finite("rotary.yarn.factor", *factor)?;
                non_zero(
                    "rotary.yarn.original_max_position_embeddings",
                    *original_max_position_embeddings,
                )?;
                positive_finite("rotary.yarn.beta_fast", *beta_fast)?;
                positive_finite("rotary.yarn.beta_slow", *beta_slow)?;
                if beta_fast <= beta_slow {
                    return Err(DescriptorError::Inconsistent {
                        component: "rotary.yarn",
                        message: format!(
                            "beta_fast ({beta_fast}) must be greater than beta_slow ({beta_slow})"
                        ),
                    });
                }
                if let Some(attention_factor) = attention_factor {
                    positive_finite("rotary.yarn.attention_factor", *attention_factor)?;
                }
                Ok(())
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RotaryEmbedding {
    head_dim: usize,
    theta: f32,
    pairing: RotaryPairing,
    region: RotaryRegion,
    scaling: RotaryScaling,
}

impl RotaryEmbedding {
    pub fn new(
        head_dim: usize,
        theta: f32,
        pairing: RotaryPairing,
        region: RotaryRegion,
        scaling: RotaryScaling,
    ) -> Result<Self, DescriptorError> {
        non_zero("rotary.head_dim", head_dim)?;
        positive_finite("rotary.theta", theta)?;
        let dimensions = region.dimensions();
        if dimensions == 0 || dimensions > head_dim || !dimensions.is_multiple_of(2) {
            return Err(DescriptorError::Inconsistent {
                component: "rotary",
                message: format!(
                    "rotary dimensions must be non-zero, even, and <= head_dim: dimensions={dimensions}, head_dim={head_dim}"
                ),
            });
        }
        scaling.validate()?;
        Ok(Self {
            head_dim,
            theta,
            pairing,
            region,
            scaling,
        })
    }

    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub const fn theta(&self) -> f32 {
        self.theta
    }

    pub const fn pairing(&self) -> RotaryPairing {
        self.pairing
    }

    pub const fn region(&self) -> RotaryRegion {
        self.region
    }

    pub fn scaling(&self) -> &RotaryScaling {
        &self.scaling
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GqaAttention {
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    query_norm: Option<RmsNorm>,
    key_norm: Option<RmsNorm>,
    rotary: RotaryEmbedding,
}

impl GqaAttention {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        bias: bool,
        rotary: RotaryEmbedding,
    ) -> Result<Self, DescriptorError> {
        non_zero("gqa.hidden_size", hidden_size)?;
        non_zero("gqa.num_heads", num_heads)?;
        non_zero("gqa.num_kv_heads", num_kv_heads)?;
        non_zero("gqa.head_dim", head_dim)?;
        if num_kv_heads > num_heads || !num_heads.is_multiple_of(num_kv_heads) {
            return Err(DescriptorError::Inconsistent {
                component: "gqa",
                message: format!(
                    "num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
                ),
            });
        }
        let query_width = checked_mul("gqa query width", num_heads, head_dim)?;
        if rotary.head_dim() != head_dim {
            return Err(DescriptorError::Inconsistent {
                component: "gqa",
                message: format!(
                    "rotary head_dim {} differs from attention head_dim {head_dim}",
                    rotary.head_dim()
                ),
            });
        }
        let kv_width = checked_mul("gqa key/value width", num_kv_heads, head_dim)?;
        Ok(Self {
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            query: Linear::new(hidden_size, query_width, bias)?,
            key: Linear::new(hidden_size, kv_width, bias)?,
            value: Linear::new(hidden_size, kv_width, bias)?,
            output: Linear::new(query_width, hidden_size, bias)?,
            query_norm: None,
            key_norm: None,
            rotary,
        })
    }

    pub fn with_qk_norms(
        mut self,
        query_norm: RmsNorm,
        key_norm: RmsNorm,
    ) -> Result<Self, DescriptorError> {
        if query_norm.hidden_size() != self.head_dim || key_norm.hidden_size() != self.head_dim {
            return Err(DescriptorError::Inconsistent {
                component: "gqa",
                message: format!(
                    "Q/K norm width must equal head_dim {}: q={}, k={}",
                    self.head_dim,
                    query_norm.hidden_size(),
                    key_norm.hidden_size()
                ),
            });
        }
        self.query_norm = Some(query_norm);
        self.key_norm = Some(key_norm);
        Ok(self)
    }

    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub const fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub const fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn query(&self) -> &Linear {
        &self.query
    }

    pub fn key(&self) -> &Linear {
        &self.key
    }

    pub fn value(&self) -> &Linear {
        &self.value
    }

    pub fn output(&self) -> &Linear {
        &self.output
    }

    pub fn query_norm(&self) -> Option<&RmsNorm> {
        self.query_norm.as_ref()
    }

    pub fn key_norm(&self) -> Option<&RmsNorm> {
        self.key_norm.as_ref()
    }

    pub fn rotary(&self) -> &RotaryEmbedding {
        &self.rotary
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MlaQueryProjection {
    Direct {
        projection: Linear,
    },
    LowRank {
        rank: usize,
        down: Linear,
        norm: RmsNorm,
        up: Linear,
    },
}

impl MlaQueryProjection {
    pub fn direct(projection: Linear) -> Self {
        Self::Direct { projection }
    }

    pub fn low_rank(
        hidden_size: usize,
        output_size: usize,
        rank: usize,
        norm_epsilon: f32,
        bias: bool,
    ) -> Result<Self, DescriptorError> {
        non_zero("mla.q_lora_rank", rank)?;
        Ok(Self::LowRank {
            rank,
            down: Linear::new(hidden_size, rank, bias)?,
            norm: RmsNorm::new(rank, norm_epsilon)?,
            up: Linear::new(rank, output_size, bias)?,
        })
    }

    pub fn input_features(&self) -> usize {
        match self {
            Self::Direct { projection } => projection.in_features(),
            Self::LowRank { down, .. } => down.in_features(),
        }
    }

    pub fn output_features(&self) -> usize {
        match self {
            Self::Direct { projection } => projection.out_features(),
            Self::LowRank { up, .. } => up.out_features(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MlaAttention {
    hidden_size: usize,
    num_heads: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    value_head_dim: usize,
    query: MlaQueryProjection,
    layout: MlaAttentionLayout,
    rotary: RotaryEmbedding,
}

/// Shape parameters for conventional low-rank multi-latent attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlaDimensions {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub value_head_dim: usize,
    pub kv_lora_rank: usize,
}

/// Exact dimensions for shared-KV MLA with grouped low-rank output projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedKvMlaDimensions {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub q_lora_rank: usize,
    pub rope_head_dim: usize,
    pub output_groups: usize,
    pub output_rank: usize,
    pub window_size: usize,
    pub compress_ratio: usize,
    pub index_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MlaAttentionLayout {
    LatentKeyValue {
        kv_lora_rank: usize,
        kv_down: Linear,
        kv_norm: RmsNorm,
        kv_up: Linear,
        output: Linear,
    },
    SharedKeyValueGroupedOutput(SharedKvMlaLayout),
}

#[derive(Debug, Clone, PartialEq)]
pub struct SharedKvMlaLayout {
    key_value: Linear,
    key_value_norm: RmsNorm,
    output_a: Linear,
    output_b: Linear,
    output_groups: usize,
    output_rank: usize,
    window_size: usize,
    attention_sink_heads: usize,
    compressor: Option<MlaCompressorSpec>,
    indexer: Option<MlaIndexerSpec>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MlaCompressorSpec {
    ratio: usize,
    head_dim: usize,
    overlap: bool,
    rotate: bool,
    ape_shape: [usize; 2],
    norm: RmsNorm,
    key_value: Linear,
    gate: Linear,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MlaIndexerSpec {
    heads: usize,
    head_dim: usize,
    topk: usize,
    query: Linear,
    weights: Linear,
    compressor: MlaCompressorSpec,
}

impl MlaAttention {
    pub fn new(
        dimensions: MlaDimensions,
        query: MlaQueryProjection,
        norm_epsilon: f32,
        bias: bool,
        rotary: RotaryEmbedding,
    ) -> Result<Self, DescriptorError> {
        let MlaDimensions {
            hidden_size,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            value_head_dim,
            kv_lora_rank,
        } = dimensions;
        for (name, value) in [
            ("mla.hidden_size", hidden_size),
            ("mla.num_heads", num_heads),
            ("mla.qk_nope_head_dim", qk_nope_head_dim),
            ("mla.qk_rope_head_dim", qk_rope_head_dim),
            ("mla.value_head_dim", value_head_dim),
            ("mla.kv_lora_rank", kv_lora_rank),
        ] {
            non_zero(name, value)?;
        }
        let q_head_dim = qk_nope_head_dim.checked_add(qk_rope_head_dim).ok_or(
            DescriptorError::DimensionOverflow {
                operation: "mla query head width",
            },
        )?;
        let query_width = checked_mul("mla query projection width", num_heads, q_head_dim)?;
        if query.input_features() != hidden_size || query.output_features() != query_width {
            return Err(DescriptorError::Inconsistent {
                component: "mla",
                message: format!(
                    "query projection must be [{query_width}, {hidden_size}], got [{}, {}]",
                    query.output_features(),
                    query.input_features()
                ),
            });
        }
        validate_mla_rotary(&rotary, q_head_dim, qk_rope_head_dim)?;
        let kv_down_width = kv_lora_rank.checked_add(qk_rope_head_dim).ok_or(
            DescriptorError::DimensionOverflow {
                operation: "mla KV down projection width",
            },
        )?;
        let kv_head_width = qk_nope_head_dim.checked_add(value_head_dim).ok_or(
            DescriptorError::DimensionOverflow {
                operation: "mla KV head width",
            },
        )?;
        let kv_up_width = checked_mul("mla KV up projection width", num_heads, kv_head_width)?;
        let output_input = checked_mul("mla output projection width", num_heads, value_head_dim)?;
        Ok(Self {
            hidden_size,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            value_head_dim,
            query,
            layout: MlaAttentionLayout::LatentKeyValue {
                kv_lora_rank,
                kv_down: Linear::new(hidden_size, kv_down_width, bias)?,
                kv_norm: RmsNorm::new(kv_lora_rank, norm_epsilon)?,
                kv_up: Linear::new(kv_lora_rank, kv_up_width, bias)?,
                output: Linear::new(output_input, hidden_size, bias)?,
            },
            rotary,
        })
    }

    pub fn shared_kv_grouped_output(
        dimensions: SharedKvMlaDimensions,
        norm_epsilon: f32,
        bias: bool,
        rotary: RotaryEmbedding,
    ) -> Result<Self, DescriptorError> {
        let SharedKvMlaDimensions {
            hidden_size,
            num_heads,
            head_dim,
            q_lora_rank,
            rope_head_dim,
            output_groups,
            output_rank,
            window_size,
            compress_ratio,
            index_heads,
            index_head_dim,
            index_topk,
        } = dimensions;
        for (name, value) in [
            ("shared_kv_mla.hidden_size", hidden_size),
            ("shared_kv_mla.num_heads", num_heads),
            ("shared_kv_mla.head_dim", head_dim),
            ("shared_kv_mla.q_lora_rank", q_lora_rank),
            ("shared_kv_mla.rope_head_dim", rope_head_dim),
            ("shared_kv_mla.output_groups", output_groups),
            ("shared_kv_mla.output_rank", output_rank),
            ("shared_kv_mla.window_size", window_size),
            ("shared_kv_mla.index_heads", index_heads),
            ("shared_kv_mla.index_head_dim", index_head_dim),
            ("shared_kv_mla.index_topk", index_topk),
        ] {
            non_zero(name, value)?;
        }
        if rope_head_dim > head_dim || !rope_head_dim.is_multiple_of(2) {
            return Err(DescriptorError::Inconsistent {
                component: "shared_kv_mla",
                message: format!(
                    "rope width {rope_head_dim} must be even and no greater than head width {head_dim}"
                ),
            });
        }
        if !num_heads.is_multiple_of(output_groups) {
            return Err(DescriptorError::Inconsistent {
                component: "shared_kv_mla",
                message: format!(
                    "head count {num_heads} must be divisible by output groups {output_groups}"
                ),
            });
        }
        validate_mla_rotary(&rotary, head_dim, rope_head_dim)?;
        let query_width = checked_mul("shared KV MLA query width", num_heads, head_dim)?;
        let query = MlaQueryProjection::low_rank(
            hidden_size,
            query_width,
            q_lora_rank,
            norm_epsilon,
            bias,
        )?;
        let grouped_input = query_width / output_groups;
        let grouped_output = checked_mul(
            "shared KV MLA grouped output width",
            output_groups,
            output_rank,
        )?;
        let compressor = (compress_ratio != 0)
            .then(|| {
                MlaCompressorSpec::new(
                    hidden_size,
                    head_dim,
                    compress_ratio,
                    false,
                    norm_epsilon,
                    bias,
                )
            })
            .transpose()?;
        let indexer = (compress_ratio == 4)
            .then(|| {
                let compressor = MlaCompressorSpec::new(
                    hidden_size,
                    index_head_dim,
                    compress_ratio,
                    true,
                    norm_epsilon,
                    bias,
                )?;
                Ok::<_, DescriptorError>(MlaIndexerSpec {
                    heads: index_heads,
                    head_dim: index_head_dim,
                    topk: index_topk,
                    query: Linear::new(
                        q_lora_rank,
                        checked_mul("MLA index query width", index_heads, index_head_dim)?,
                        bias,
                    )?,
                    weights: Linear::new(hidden_size, index_heads, bias)?,
                    compressor,
                })
            })
            .transpose()?;
        Ok(Self {
            hidden_size,
            num_heads,
            qk_nope_head_dim: head_dim - rope_head_dim,
            qk_rope_head_dim: rope_head_dim,
            value_head_dim: head_dim,
            query,
            layout: MlaAttentionLayout::SharedKeyValueGroupedOutput(SharedKvMlaLayout {
                key_value: Linear::new(hidden_size, head_dim, bias)?,
                key_value_norm: RmsNorm::new(head_dim, norm_epsilon)?,
                output_a: Linear::new(grouped_input, grouped_output, bias)?,
                output_b: Linear::new(grouped_output, hidden_size, bias)?,
                output_groups,
                output_rank,
                window_size,
                attention_sink_heads: num_heads,
                compressor,
                indexer,
            }),
            rotary,
        })
    }

    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub const fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub const fn qk_nope_head_dim(&self) -> usize {
        self.qk_nope_head_dim
    }

    pub const fn qk_rope_head_dim(&self) -> usize {
        self.qk_rope_head_dim
    }

    pub const fn value_head_dim(&self) -> usize {
        self.value_head_dim
    }

    pub fn kv_lora_rank(&self) -> usize {
        match &self.layout {
            MlaAttentionLayout::LatentKeyValue { kv_lora_rank, .. } => *kv_lora_rank,
            MlaAttentionLayout::SharedKeyValueGroupedOutput(layout) => {
                layout.key_value.out_features()
            }
        }
    }

    pub fn query(&self) -> &MlaQueryProjection {
        &self.query
    }

    pub fn layout(&self) -> &MlaAttentionLayout {
        &self.layout
    }

    pub fn shared_kv_layout(&self) -> Option<&SharedKvMlaLayout> {
        match &self.layout {
            MlaAttentionLayout::SharedKeyValueGroupedOutput(layout) => Some(layout),
            MlaAttentionLayout::LatentKeyValue { .. } => None,
        }
    }

    pub fn kv_down(&self) -> &Linear {
        match &self.layout {
            MlaAttentionLayout::LatentKeyValue { kv_down, .. } => kv_down,
            MlaAttentionLayout::SharedKeyValueGroupedOutput(layout) => &layout.key_value,
        }
    }

    pub fn kv_norm(&self) -> &RmsNorm {
        match &self.layout {
            MlaAttentionLayout::LatentKeyValue { kv_norm, .. } => kv_norm,
            MlaAttentionLayout::SharedKeyValueGroupedOutput(layout) => &layout.key_value_norm,
        }
    }

    /// Conventional latent-KV up projection.
    ///
    /// This preserves the original MLA API. Shared-KV layouts have no such
    /// projection; use the optional layout getter for arbitrary MLA layouts.
    pub fn kv_up(&self) -> &Linear {
        self.latent_kv_up()
            .expect("shared-KV MLA has no latent KV up projection")
    }

    pub fn latent_kv_up(&self) -> Option<&Linear> {
        match &self.layout {
            MlaAttentionLayout::LatentKeyValue { kv_up, .. } => Some(kv_up),
            MlaAttentionLayout::SharedKeyValueGroupedOutput(_) => None,
        }
    }

    pub fn output(&self) -> &Linear {
        match &self.layout {
            MlaAttentionLayout::LatentKeyValue { output, .. } => output,
            MlaAttentionLayout::SharedKeyValueGroupedOutput(layout) => &layout.output_b,
        }
    }

    pub fn rotary(&self) -> &RotaryEmbedding {
        &self.rotary
    }
}

impl SharedKvMlaLayout {
    pub fn key_value(&self) -> &Linear {
        &self.key_value
    }

    pub fn key_value_norm(&self) -> &RmsNorm {
        &self.key_value_norm
    }

    pub fn output_a(&self) -> &Linear {
        &self.output_a
    }

    pub fn output_b(&self) -> &Linear {
        &self.output_b
    }

    pub const fn output_groups(&self) -> usize {
        self.output_groups
    }

    pub const fn output_rank(&self) -> usize {
        self.output_rank
    }

    pub const fn window_size(&self) -> usize {
        self.window_size
    }

    pub const fn attention_sink_heads(&self) -> usize {
        self.attention_sink_heads
    }

    pub fn compressor(&self) -> Option<&MlaCompressorSpec> {
        self.compressor.as_ref()
    }

    pub fn indexer(&self) -> Option<&MlaIndexerSpec> {
        self.indexer.as_ref()
    }
}

impl MlaCompressorSpec {
    fn new(
        hidden_size: usize,
        head_dim: usize,
        ratio: usize,
        rotate: bool,
        norm_epsilon: f32,
        bias: bool,
    ) -> Result<Self, DescriptorError> {
        non_zero("mla_compressor.ratio", ratio)?;
        let overlap = ratio == 4;
        let coefficient = if overlap { 2 } else { 1 };
        let output = checked_mul("MLA compressor output width", coefficient, head_dim)?;
        Ok(Self {
            ratio,
            head_dim,
            overlap,
            rotate,
            ape_shape: [ratio, output],
            norm: RmsNorm::new(head_dim, norm_epsilon)?,
            key_value: Linear::new(hidden_size, output, bias)?,
            gate: Linear::new(hidden_size, output, bias)?,
        })
    }

    pub const fn ratio(&self) -> usize {
        self.ratio
    }

    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub const fn overlap(&self) -> bool {
        self.overlap
    }

    pub const fn rotate(&self) -> bool {
        self.rotate
    }

    pub const fn ape_shape(&self) -> [usize; 2] {
        self.ape_shape
    }

    pub fn norm(&self) -> &RmsNorm {
        &self.norm
    }

    pub fn key_value(&self) -> &Linear {
        &self.key_value
    }

    pub fn gate(&self) -> &Linear {
        &self.gate
    }
}

impl MlaIndexerSpec {
    pub const fn heads(&self) -> usize {
        self.heads
    }

    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub const fn topk(&self) -> usize {
        self.topk
    }

    pub fn query(&self) -> &Linear {
        &self.query
    }

    pub fn weights(&self) -> &Linear {
        &self.weights
    }

    pub fn compressor(&self) -> &MlaCompressorSpec {
        &self.compressor
    }
}

fn validate_mla_rotary(
    rotary: &RotaryEmbedding,
    head_dim: usize,
    rope_head_dim: usize,
) -> Result<(), DescriptorError> {
    if rotary.head_dim() != head_dim || rotary.region().dimensions() != rope_head_dim {
        return Err(DescriptorError::Inconsistent {
            component: "mla",
            message: format!(
                "rotary descriptor must use head_dim {head_dim} and rope width {rope_head_dim}"
            ),
        });
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub enum Attention {
    Gqa(GqaAttention),
    Mla(MlaAttention),
}

impl Attention {
    pub fn hidden_size(&self) -> usize {
        match self {
            Self::Gqa(attention) => attention.hidden_size(),
            Self::Mla(attention) => attention.hidden_size(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SwiGlu {
    hidden_size: usize,
    intermediate_size: usize,
    activation_limit: Option<f32>,
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl SwiGlu {
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        bias: bool,
    ) -> Result<Self, DescriptorError> {
        non_zero("swiglu.hidden_size", hidden_size)?;
        non_zero("swiglu.intermediate_size", intermediate_size)?;
        Ok(Self {
            hidden_size,
            intermediate_size,
            activation_limit: None,
            gate: Linear::new(hidden_size, intermediate_size, bias)?,
            up: Linear::new(hidden_size, intermediate_size, bias)?,
            down: Linear::new(intermediate_size, hidden_size, bias)?,
        })
    }

    pub fn with_activation_limit(mut self, limit: f32) -> Result<Self, DescriptorError> {
        positive_finite("swiglu.activation_limit", limit)?;
        self.activation_limit = Some(limit);
        Ok(self)
    }

    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub const fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }

    pub const fn activation_limit(&self) -> Option<f32> {
        self.activation_limit
    }

    pub fn gate(&self) -> &Linear {
        &self.gate
    }

    pub fn up(&self) -> &Linear {
        &self.up
    }

    pub fn down(&self) -> &Linear {
        &self.down
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterScoreFunction {
    Softmax,
    Sigmoid,
    SqrtSoftplus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouterSelection {
    TopK,
    GroupLimitedTopK {
        groups: usize,
        selected_groups: usize,
    },
    HashAssistedTopK {
        hash_layers: usize,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct MoeRouterSpec {
    num_experts: usize,
    experts_per_token: usize,
    score_function: RouterScoreFunction,
    selection: RouterSelection,
    normalize_selected: bool,
    selection_bias: bool,
    route_scale: f32,
}

impl MoeRouterSpec {
    pub fn new(
        num_experts: usize,
        experts_per_token: usize,
        score_function: RouterScoreFunction,
        selection: RouterSelection,
        normalize_selected: bool,
        route_scale: f32,
    ) -> Result<Self, DescriptorError> {
        non_zero("router.num_experts", num_experts)?;
        non_zero("router.experts_per_token", experts_per_token)?;
        if experts_per_token > num_experts {
            return Err(DescriptorError::Inconsistent {
                component: "router",
                message: format!(
                    "experts_per_token ({experts_per_token}) exceeds num_experts ({num_experts})"
                ),
            });
        }
        match selection {
            RouterSelection::TopK => {}
            RouterSelection::GroupLimitedTopK {
                groups,
                selected_groups,
            } => {
                non_zero("router.groups", groups)?;
                non_zero("router.selected_groups", selected_groups)?;
                if !num_experts.is_multiple_of(groups) || selected_groups > groups {
                    return Err(DescriptorError::Inconsistent {
                        component: "router",
                        message: format!(
                            "group-limited routing requires num_experts divisible by groups and selected_groups <= groups: experts={num_experts}, groups={groups}, selected={selected_groups}"
                        ),
                    });
                }
                let candidates = selected_groups * (num_experts / groups);
                if experts_per_token > candidates {
                    return Err(DescriptorError::Inconsistent {
                        component: "router",
                        message: format!(
                            "experts_per_token ({experts_per_token}) exceeds the {candidates} experts available in selected groups"
                        ),
                    });
                }
            }
            RouterSelection::HashAssistedTopK { hash_layers } => {
                non_zero("router.hash_layers", hash_layers)?;
            }
        }
        positive_finite("router.route_scale", route_scale)?;
        Ok(Self {
            num_experts,
            experts_per_token,
            score_function,
            selection,
            normalize_selected,
            selection_bias: false,
            route_scale,
        })
    }

    pub const fn num_experts(&self) -> usize {
        self.num_experts
    }

    pub const fn experts_per_token(&self) -> usize {
        self.experts_per_token
    }

    pub const fn score_function(&self) -> RouterScoreFunction {
        self.score_function
    }

    pub fn selection(&self) -> &RouterSelection {
        &self.selection
    }

    pub fn with_selection_bias(mut self, selection_bias: bool) -> Self {
        self.selection_bias = selection_bias;
        self
    }

    pub const fn normalize_selected(&self) -> bool {
        self.normalize_selected
    }

    pub const fn selection_bias(&self) -> bool {
        self.selection_bias
    }

    pub const fn route_scale(&self) -> f32 {
        self.route_scale
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Moe {
    hidden_size: usize,
    router: Linear,
    router_spec: MoeRouterSpec,
    expert: SwiGlu,
    shared_expert: Option<SwiGlu>,
}

impl Moe {
    pub fn new(
        hidden_size: usize,
        expert_intermediate_size: usize,
        router_spec: MoeRouterSpec,
        bias: bool,
    ) -> Result<Self, DescriptorError> {
        non_zero("moe.hidden_size", hidden_size)?;
        Ok(Self {
            hidden_size,
            router: Linear::new(hidden_size, router_spec.num_experts(), bias)?,
            router_spec,
            expert: SwiGlu::new(hidden_size, expert_intermediate_size, bias)?,
            shared_expert: None,
        })
    }

    pub fn with_activation_limit(mut self, limit: f32) -> Result<Self, DescriptorError> {
        self.expert = self.expert.with_activation_limit(limit)?;
        if let Some(shared) = self.shared_expert.take() {
            self.shared_expert = Some(shared.with_activation_limit(limit)?);
        }
        Ok(self)
    }

    pub fn with_shared_expert(mut self, expert: SwiGlu) -> Result<Self, DescriptorError> {
        if expert.hidden_size() != self.hidden_size {
            return Err(DescriptorError::Inconsistent {
                component: "moe",
                message: format!(
                    "shared expert hidden size {} differs from model hidden size {}",
                    expert.hidden_size(),
                    self.hidden_size
                ),
            });
        }
        self.shared_expert = Some(expert);
        Ok(self)
    }

    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn router(&self) -> &Linear {
        &self.router
    }

    pub fn router_spec(&self) -> &MoeRouterSpec {
        &self.router_spec
    }

    pub fn expert(&self) -> &SwiGlu {
        &self.expert
    }

    pub fn shared_expert(&self) -> Option<&SwiGlu> {
        self.shared_expert.as_ref()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum FeedForward {
    SwiGlu(SwiGlu),
    Moe(Moe),
}

impl FeedForward {
    pub fn hidden_size(&self) -> usize {
        match self {
            Self::SwiGlu(feed_forward) => feed_forward.hidden_size(),
            Self::Moe(feed_forward) => feed_forward.hidden_size(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HyperResidual {
    streams: usize,
    epsilon: f32,
    sinkhorn_iterations: usize,
}

impl HyperResidual {
    pub fn new(
        streams: usize,
        epsilon: f32,
        sinkhorn_iterations: usize,
    ) -> Result<Self, DescriptorError> {
        non_zero("hyper_residual.streams", streams)?;
        positive_finite("hyper_residual.epsilon", epsilon)?;
        non_zero("hyper_residual.sinkhorn_iterations", sinkhorn_iterations)?;
        Ok(Self {
            streams,
            epsilon,
            sinkhorn_iterations,
        })
    }

    pub const fn streams(&self) -> usize {
        self.streams
    }

    pub const fn epsilon(&self) -> f32 {
        self.epsilon
    }

    pub const fn sinkhorn_iterations(&self) -> usize {
        self.sinkhorn_iterations
    }

    pub fn function_shape(&self, hidden_size: usize) -> [usize; 2] {
        [
            (2 + self.streams) * self.streams,
            self.streams * hidden_size,
        ]
    }

    pub const fn scale_shape(&self) -> [usize; 1] {
        [3]
    }

    pub fn base_shape(&self) -> [usize; 1] {
        [(2 + self.streams) * self.streams]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Residual {
    Add,
    Hyper(HyperResidual),
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecoderLayer {
    index: usize,
    input_norm: RmsNorm,
    attention: Attention,
    attention_residual: Residual,
    post_attention_norm: RmsNorm,
    feed_forward: FeedForward,
    feed_forward_residual: Residual,
}

impl DecoderLayer {
    pub fn new(
        index: usize,
        input_norm: RmsNorm,
        attention: Attention,
        attention_residual: Residual,
        post_attention_norm: RmsNorm,
        feed_forward: FeedForward,
        feed_forward_residual: Residual,
    ) -> Result<Self, DescriptorError> {
        let hidden_size = attention.hidden_size();
        if input_norm.hidden_size() != hidden_size
            || post_attention_norm.hidden_size() != hidden_size
            || feed_forward.hidden_size() != hidden_size
        {
            return Err(DescriptorError::Inconsistent {
                component: "decoder_layer",
                message: format!(
                    "layer {index} components disagree on hidden size {hidden_size}: input_norm={}, post_attention_norm={}, feed_forward={}",
                    input_norm.hidden_size(),
                    post_attention_norm.hidden_size(),
                    feed_forward.hidden_size()
                ),
            });
        }
        Ok(Self {
            index,
            input_norm,
            attention,
            attention_residual,
            post_attention_norm,
            feed_forward,
            feed_forward_residual,
        })
    }

    pub const fn index(&self) -> usize {
        self.index
    }

    pub fn hidden_size(&self) -> usize {
        self.attention.hidden_size()
    }

    pub fn input_norm(&self) -> &RmsNorm {
        &self.input_norm
    }

    pub fn attention(&self) -> &Attention {
        &self.attention
    }

    pub fn attention_residual(&self) -> &Residual {
        &self.attention_residual
    }

    pub fn post_attention_norm(&self) -> &RmsNorm {
        &self.post_attention_norm
    }

    pub fn feed_forward(&self) -> &FeedForward {
        &self.feed_forward
    }

    pub fn feed_forward_residual(&self) -> &Residual {
        &self.feed_forward_residual
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HyperConnectionHeadSpec {
    streams: usize,
    hidden_size: usize,
    epsilon: f32,
}

impl HyperConnectionHeadSpec {
    pub fn new(streams: usize, hidden_size: usize, epsilon: f32) -> Result<Self, DescriptorError> {
        non_zero("hyper_connection_head.streams", streams)?;
        non_zero("hyper_connection_head.hidden_size", hidden_size)?;
        positive_finite("hyper_connection_head.epsilon", epsilon)?;
        streams
            .checked_mul(hidden_size)
            .ok_or(DescriptorError::DimensionOverflow {
                operation: "hyper-connection head function width",
            })?;
        Ok(Self {
            streams,
            hidden_size,
            epsilon,
        })
    }

    pub const fn streams(&self) -> usize {
        self.streams
    }

    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub const fn epsilon(&self) -> f32 {
        self.epsilon
    }

    pub fn function_shape(&self) -> [usize; 2] {
        [
            self.streams,
            self.streams
                .checked_mul(self.hidden_size)
                .expect("validated hyper-connection head shape overflow"),
        ]
    }

    pub const fn scale_shape(&self) -> [usize; 1] {
        [1]
    }

    pub const fn base_shape(&self) -> [usize; 1] {
        [self.streams]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum DecoderAttachmentSpec {
    Proposal(ProposalAttachmentSpec),
}

impl DecoderAttachmentSpec {
    pub fn path(&self) -> &str {
        match self {
            Self::Proposal(attachment) => attachment.path(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProposalAttachmentSpec {
    path: String,
    hidden_size: usize,
    block_size: usize,
    noise_token_id: Option<u32>,
    target_layer_ids: Vec<usize>,
    stages: Vec<DecoderLayer>,
    main_projection: Option<Linear>,
    main_norm: Option<RmsNorm>,
    heads: Option<ProposalHeadsSpec>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProposalAttachmentParts {
    pub path: String,
    pub hidden_size: usize,
    pub block_size: usize,
    pub noise_token_id: Option<u32>,
    pub target_layer_ids: Vec<usize>,
    pub stages: Vec<DecoderLayer>,
    pub main_projection: Option<Linear>,
    pub main_norm: Option<RmsNorm>,
    pub heads: Option<ProposalHeadsSpec>,
}

impl ProposalAttachmentSpec {
    pub fn new(parts: ProposalAttachmentParts) -> Result<Self, DescriptorError> {
        let ProposalAttachmentParts {
            path,
            hidden_size,
            block_size,
            noise_token_id,
            target_layer_ids,
            stages,
            main_projection,
            main_norm,
            heads,
        } = parts;
        if path.trim().is_empty() {
            return Err(DescriptorError::EmptyName {
                field: "proposal_attachment.path",
            });
        }
        non_zero("proposal_attachment.hidden_size", hidden_size)?;
        non_zero("proposal_attachment.block_size", block_size)?;
        if target_layer_ids.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(DescriptorError::Inconsistent {
                component: "proposal_attachment",
                message: "target layer IDs must be strictly increasing".into(),
            });
        }
        for (index, stage) in stages.iter().enumerate() {
            if stage.index() != index || stage.hidden_size() != hidden_size {
                return Err(DescriptorError::Inconsistent {
                    component: "proposal_attachment",
                    message: format!(
                        "stage {index} reports index {} and hidden size {}, expected {index} and {hidden_size}",
                        stage.index(),
                        stage.hidden_size()
                    ),
                });
            }
        }
        match (&main_projection, &main_norm) {
            (Some(projection), Some(norm)) => {
                let input = hidden_size.checked_mul(target_layer_ids.len()).ok_or(
                    DescriptorError::DimensionOverflow {
                        operation: "proposal main projection input width",
                    },
                )?;
                if projection.in_features() != input
                    || projection.out_features() != hidden_size
                    || norm.hidden_size() != hidden_size
                {
                    return Err(DescriptorError::Inconsistent {
                        component: "proposal_attachment",
                        message: "stage-zero projection/norm shapes do not match target taps"
                            .into(),
                    });
                }
            }
            (None, None) => {}
            _ => {
                return Err(DescriptorError::Inconsistent {
                    component: "proposal_attachment",
                    message: "stage-zero projection and norm must be present together".into(),
                });
            }
        }
        if let Some(heads) = &heads {
            heads.validate(hidden_size)?;
        }
        Ok(Self {
            path,
            hidden_size,
            block_size,
            noise_token_id,
            target_layer_ids,
            stages,
            main_projection,
            main_norm,
            heads,
        })
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub const fn block_size(&self) -> usize {
        self.block_size
    }

    pub const fn noise_token_id(&self) -> Option<u32> {
        self.noise_token_id
    }

    pub fn target_layer_ids(&self) -> &[usize] {
        &self.target_layer_ids
    }

    pub fn stages(&self) -> &[DecoderLayer] {
        &self.stages
    }

    pub fn main_projection(&self) -> Option<&Linear> {
        self.main_projection.as_ref()
    }

    pub fn main_norm(&self) -> Option<&RmsNorm> {
        self.main_norm.as_ref()
    }

    pub fn heads(&self) -> Option<&ProposalHeadsSpec> {
        self.heads.as_ref()
    }

    /// Proposal stages consume the decoder's canonical token embedding.
    pub const fn shares_model_token_embedding(&self) -> bool {
        true
    }

    /// Proposal logits use the decoder's canonical output head.
    pub const fn shares_model_output_head(&self) -> bool {
        true
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProposalHeadsSpec {
    hyper_connection: HyperConnectionHeadSpec,
    norm: RmsNorm,
    markov_embedding: Embedding,
    markov_output: Linear,
    confidence: Linear,
}

impl ProposalHeadsSpec {
    pub fn new(
        hyper_connection: HyperConnectionHeadSpec,
        norm: RmsNorm,
        vocab_size: usize,
        markov_rank: usize,
        hidden_size: usize,
    ) -> Result<Self, DescriptorError> {
        non_zero("proposal_heads.markov_rank", markov_rank)?;
        Ok(Self {
            hyper_connection,
            norm,
            markov_embedding: Embedding::new(vocab_size, markov_rank, None)?,
            markov_output: Linear::new(markov_rank, vocab_size, false)?,
            confidence: Linear::new(
                hidden_size
                    .checked_add(markov_rank)
                    .ok_or(DescriptorError::DimensionOverflow {
                        operation: "proposal confidence input width",
                    })?,
                1,
                false,
            )?,
        })
    }

    fn validate(&self, hidden_size: usize) -> Result<(), DescriptorError> {
        if self.hyper_connection.hidden_size() != hidden_size
            || self.norm.hidden_size() != hidden_size
            || self.confidence.in_features() != hidden_size + self.markov_embedding.embedding_dim()
            || self.markov_output.in_features() != self.markov_embedding.embedding_dim()
            || self.markov_output.out_features() != self.markov_embedding.num_embeddings()
        {
            return Err(DescriptorError::Inconsistent {
                component: "proposal_heads",
                message: "head dimensions are inconsistent".into(),
            });
        }
        Ok(())
    }

    pub fn hyper_connection(&self) -> &HyperConnectionHeadSpec {
        &self.hyper_connection
    }

    pub fn norm(&self) -> &RmsNorm {
        &self.norm
    }

    pub fn markov_embedding(&self) -> &Embedding {
        &self.markov_embedding
    }

    pub fn markov_output(&self) -> &Linear {
        &self.markov_output
    }

    pub fn confidence(&self) -> &Linear {
        &self.confidence
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecoderModelSpec {
    architecture: String,
    hidden_size: usize,
    vocab_size: usize,
    max_sequence_length: Option<usize>,
    token_embedding: Embedding,
    layers: Vec<DecoderLayer>,
    final_norm: RmsNorm,
    output: Linear,
    tie_word_embeddings: bool,
    output_hyper_connection: Option<HyperConnectionHeadSpec>,
    attachments: Vec<DecoderAttachmentSpec>,
}

/// Inputs used to construct a validated decoder model descriptor.
#[derive(Debug, Clone, PartialEq)]
pub struct DecoderModelParts {
    pub architecture: String,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub max_sequence_length: Option<usize>,
    pub token_embedding: Embedding,
    pub layers: Vec<DecoderLayer>,
    pub final_norm: RmsNorm,
    pub output: Linear,
    pub tie_word_embeddings: bool,
}

impl DecoderModelSpec {
    pub fn new(parts: DecoderModelParts) -> Result<Self, DescriptorError> {
        let DecoderModelParts {
            architecture,
            hidden_size,
            vocab_size,
            max_sequence_length,
            token_embedding,
            layers,
            final_norm,
            output,
            tie_word_embeddings,
        } = parts;
        if architecture.trim().is_empty() {
            return Err(DescriptorError::EmptyName {
                field: "decoder.architecture",
            });
        }
        non_zero("decoder.hidden_size", hidden_size)?;
        non_zero("decoder.vocab_size", vocab_size)?;
        if max_sequence_length == Some(0) {
            return Err(DescriptorError::ZeroDimension {
                field: "decoder.max_sequence_length",
            });
        }
        if token_embedding.num_embeddings() != vocab_size
            || token_embedding.embedding_dim() != hidden_size
        {
            return Err(DescriptorError::Inconsistent {
                component: "decoder",
                message: format!(
                    "embedding shape {:?} must be [{vocab_size}, {hidden_size}]",
                    token_embedding.weight_shape()
                ),
            });
        }
        if layers.is_empty() {
            return Err(DescriptorError::Inconsistent {
                component: "decoder",
                message: "at least one decoder layer is required".into(),
            });
        }
        for (expected_index, layer) in layers.iter().enumerate() {
            if layer.index() != expected_index || layer.hidden_size() != hidden_size {
                return Err(DescriptorError::Inconsistent {
                    component: "decoder",
                    message: format!(
                        "layer at position {expected_index} reports index {} and hidden size {}, expected index {expected_index} and hidden size {hidden_size}",
                        layer.index(),
                        layer.hidden_size()
                    ),
                });
            }
        }
        if final_norm.hidden_size() != hidden_size
            || output.in_features() != hidden_size
            || output.out_features() != vocab_size
        {
            return Err(DescriptorError::Inconsistent {
                component: "decoder",
                message: format!(
                    "final norm/output must map hidden size {hidden_size} to vocab size {vocab_size}"
                ),
            });
        }
        Ok(Self {
            architecture,
            hidden_size,
            vocab_size,
            max_sequence_length,
            token_embedding,
            layers,
            final_norm,
            output,
            tie_word_embeddings,
            output_hyper_connection: None,
            attachments: Vec::new(),
        })
    }

    pub fn with_output_hyper_connection(
        mut self,
        head: HyperConnectionHeadSpec,
    ) -> Result<Self, DescriptorError> {
        if head.hidden_size() != self.hidden_size {
            return Err(DescriptorError::Inconsistent {
                component: "decoder",
                message: format!(
                    "output hyper-connection hidden size {} differs from decoder hidden size {}",
                    head.hidden_size(),
                    self.hidden_size
                ),
            });
        }
        self.output_hyper_connection = Some(head);
        Ok(self)
    }

    pub fn with_attachment(
        mut self,
        attachment: DecoderAttachmentSpec,
    ) -> Result<Self, DescriptorError> {
        if self
            .attachments
            .iter()
            .any(|existing| existing.path() == attachment.path())
        {
            return Err(DescriptorError::Inconsistent {
                component: "decoder",
                message: format!("duplicate attachment path '{}'", attachment.path()),
            });
        }
        self.attachments.push(attachment);
        Ok(self)
    }

    pub fn architecture(&self) -> &str {
        &self.architecture
    }

    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub const fn max_sequence_length(&self) -> Option<usize> {
        self.max_sequence_length
    }

    pub fn token_embedding(&self) -> &Embedding {
        &self.token_embedding
    }

    pub fn layers(&self) -> &[DecoderLayer] {
        &self.layers
    }

    pub fn final_norm(&self) -> &RmsNorm {
        &self.final_norm
    }

    pub fn output(&self) -> &Linear {
        &self.output
    }

    pub const fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings
    }

    pub fn output_hyper_connection(&self) -> Option<&HyperConnectionHeadSpec> {
        self.output_hyper_connection.as_ref()
    }

    pub fn attachments(&self) -> &[DecoderAttachmentSpec] {
        &self.attachments
    }
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum DescriptorError {
    #[error("{field} must be non-zero")]
    ZeroDimension { field: &'static str },
    #[error("{field} must be finite and positive, got {value}")]
    InvalidPositiveFloat { field: &'static str, value: f32 },
    #[error("{field} cannot be empty")]
    EmptyName { field: &'static str },
    #[error("dimension overflow while computing {operation}")]
    DimensionOverflow { operation: &'static str },
    #[error("invalid {component} descriptor: {message}")]
    Inconsistent {
        component: &'static str,
        message: String,
    },
}

fn non_zero(field: &'static str, value: usize) -> Result<(), DescriptorError> {
    if value == 0 {
        Err(DescriptorError::ZeroDimension { field })
    } else {
        Ok(())
    }
}

fn positive_finite(field: &'static str, value: f32) -> Result<(), DescriptorError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(DescriptorError::InvalidPositiveFloat { field, value })
    }
}

fn checked_mul(
    operation: &'static str,
    left: usize,
    right: usize,
) -> Result<usize, DescriptorError> {
    left.checked_mul(right)
        .ok_or(DescriptorError::DimensionOverflow { operation })
}
