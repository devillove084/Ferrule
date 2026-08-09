//! Model-independent HyperConnection residual mixing.
//!
//! A HyperConnection carries multiple hidden streams, reduces them before a
//! transformer sub-layer, and mixes the sub-layer output back into the stream
//! state. This module owns the component's configuration, source and prepared
//! weights, reference math, and narrow CUDA semantic calls. Decoder arenas,
//! continuations, transactions, caches, resolvers, and allocators remain owned by
//! the physical forward executor.

use ferrule_common::{Error, Result};

use super::super::{
    BoundParameter, DescriptorError, HyperConnectionHeadSpec, HyperResidual, StateDictMaterializer,
};

/// Model-independent HyperConnection dimensions and numerical policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HyperConnectionConfig {
    pub hc_mult: usize,
    pub hidden_size: usize,
    pub sinkhorn_iters: usize,
    pub eps: f32,
    pub norm_eps: f32,
}

impl HyperConnectionConfig {
    pub fn mix_hc(self) -> usize {
        (2 + self.hc_mult) * self.hc_mult
    }

    pub fn hc_hidden_size(self) -> usize {
        self.hc_mult * self.hidden_size
    }

    pub fn function_shape(self) -> [usize; 2] {
        [self.mix_hc(), self.hc_hidden_size()]
    }

    pub const fn scale_shape(self) -> [usize; 1] {
        [3]
    }

    pub fn base_shape(self) -> [usize; 1] {
        [self.mix_hc()]
    }

    pub fn head_function_shape(self) -> [usize; 2] {
        [self.hc_mult, self.hc_hidden_size()]
    }

    pub const fn head_scale_shape(self) -> [usize; 1] {
        [1]
    }

    pub const fn head_base_shape(self) -> [usize; 1] {
        [self.hc_mult]
    }

    /// Builds the generic transformer descriptor selected by a model recipe.
    pub fn residual_spec(self) -> std::result::Result<HyperResidual, DescriptorError> {
        HyperResidual::new(self.hc_mult, self.eps, self.sinkhorn_iters)
    }

    /// Builds the generic output-head descriptor selected by a model recipe.
    pub fn head_spec(self) -> std::result::Result<HyperConnectionHeadSpec, DescriptorError> {
        HyperConnectionHeadSpec::new(self.hc_mult, self.hidden_size, self.eps)
    }

    pub fn validate(self) -> Result<()> {
        if self.hc_mult == 0 || self.hidden_size == 0 {
            return Err(model_error(format!(
                "invalid shape: streams={}, hidden_size={}",
                self.hc_mult, self.hidden_size
            )));
        }
        if self.sinkhorn_iters == 0 {
            return Err(model_error("sinkhorn iterations must be positive"));
        }
        if !self.eps.is_finite()
            || self.eps <= 0.0
            || !self.norm_eps.is_finite()
            || self.norm_eps <= 0.0
        {
            return Err(model_error(format!(
                "epsilon values must be positive and finite: residual={} norm={}",
                self.eps, self.norm_eps
            )));
        }
        self.mix_hc()
            .checked_mul(self.hc_hidden_size())
            .ok_or_else(|| model_error("function shape overflows usize"))?;
        Ok(())
    }
}

/// Transformer sub-layer using one HyperConnection phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HyperConnectionStage {
    Attention,
    FeedForward,
}

#[cfg(feature = "cuda")]
impl HyperConnectionStage {
    pub const fn kernel_operation(self) -> ferrule_backend::plan::KernelOperation {
        match self {
            Self::Attention => ferrule_backend::plan::KernelOperation::AttentionHcPre,
            Self::FeedForward => ferrule_backend::plan::KernelOperation::FeedForwardHcPre,
        }
    }
}

/// Host-side parameters for one HyperConnection phase.
#[derive(Debug, Clone, PartialEq)]
pub struct HyperConnectionWeights {
    /// Row-major `[mix_hc, hc_mult * hidden_size]`.
    pub function: Vec<f32>,
    /// `[3]`: pre, post, and combination scales.
    pub scale: Vec<f32>,
    /// `[mix_hc]`: pre, post, and combination base values.
    pub base: Vec<f32>,
}

impl HyperConnectionWeights {
    pub fn materialize(
        materializer: &StateDictMaterializer,
        bindings: [&BoundParameter; 3],
        config: HyperConnectionConfig,
    ) -> Result<Self> {
        validate_same_residency(bindings)?;
        let weights = Self {
            function: materializer.parameter(bindings[0])?.values_f32()?,
            scale: materializer.parameter(bindings[1])?.values_f32()?,
            base: materializer.parameter(bindings[2])?.values_f32()?,
        };
        weights.validate(config)?;
        Ok(weights)
    }

    pub fn validate(&self, config: HyperConnectionConfig) -> Result<()> {
        config.validate()?;
        let expected_fn = config
            .mix_hc()
            .checked_mul(config.hc_hidden_size())
            .ok_or_else(|| model_error("function length overflows usize"))?;
        if self.function.len() != expected_fn {
            return Err(model_error(format!(
                "function length mismatch: expected {expected_fn}, got {}",
                self.function.len()
            )));
        }
        if self.scale.len() != 3 {
            return Err(model_error(format!(
                "scale length mismatch: expected 3, got {}",
                self.scale.len()
            )));
        }
        if self.base.len() != config.mix_hc() {
            return Err(model_error(format!(
                "base length mismatch: expected {}, got {}",
                config.mix_hc(),
                self.base.len()
            )));
        }
        Ok(())
    }
}

/// Source payload for one transformer sub-layer phase.
#[derive(Debug, Clone, PartialEq)]
pub struct HyperConnectionPhase<W = HyperConnectionWeights, N = Vec<f32>> {
    weights: W,
    norm: N,
}

impl<W, N> HyperConnectionPhase<W, N> {
    pub const fn new(weights: W, norm: N) -> Self {
        Self { weights, norm }
    }

    pub const fn weights(&self) -> &W {
        &self.weights
    }

    pub const fn norm(&self) -> &N {
        &self.norm
    }

    pub fn into_parts(self) -> (W, N) {
        (self.weights, self.norm)
    }
}

/// Concrete, composable source HyperConnection for one transformer layer.
#[derive(Debug, Clone, PartialEq)]
pub struct HyperConnection {
    config: HyperConnectionConfig,
    attention: HyperConnectionPhase,
    feed_forward: HyperConnectionPhase,
}

impl HyperConnection {
    pub fn new(
        config: HyperConnectionConfig,
        attention: HyperConnectionPhase,
        feed_forward: HyperConnectionPhase,
    ) -> Result<Self> {
        config.validate()?;
        for (stage, phase) in [
            (HyperConnectionStage::Attention, &attention),
            (HyperConnectionStage::FeedForward, &feed_forward),
        ] {
            phase.weights.validate(config)?;
            if phase.norm.len() != config.hidden_size {
                return Err(model_error(format!(
                    "{stage:?} norm length mismatch: expected {}, got {}",
                    config.hidden_size,
                    phase.norm.len()
                )));
            }
        }
        Ok(Self {
            config,
            attention,
            feed_forward,
        })
    }

    pub const fn config(&self) -> HyperConnectionConfig {
        self.config
    }

    pub const fn phase(&self, stage: HyperConnectionStage) -> &HyperConnectionPhase {
        match stage {
            HyperConnectionStage::Attention => &self.attention,
            HyperConnectionStage::FeedForward => &self.feed_forward,
        }
    }

    pub const fn attention(&self) -> &HyperConnectionPhase {
        &self.attention
    }

    pub const fn feed_forward(&self) -> &HyperConnectionPhase {
        &self.feed_forward
    }
}

/// Host-side weights for the final HyperConnection stream reduction.
#[derive(Debug, Clone, PartialEq)]
pub struct HyperConnectionHeadWeights {
    /// Row-major `[hc_mult, hc_mult * hidden_size]`.
    pub function: Vec<f32>,
    /// `[1]` head scale.
    pub scale: Vec<f32>,
    /// `[hc_mult]` head base.
    pub base: Vec<f32>,
}

impl HyperConnectionHeadWeights {
    pub fn materialize(
        materializer: &StateDictMaterializer,
        bindings: [&BoundParameter; 3],
        config: HyperConnectionConfig,
    ) -> Result<Self> {
        validate_same_residency(bindings)?;
        let weights = Self {
            function: materializer.parameter(bindings[0])?.values_f32()?,
            scale: materializer.parameter(bindings[1])?.values_f32()?,
            base: materializer.parameter(bindings[2])?.values_f32()?,
        };
        weights.validate(config)?;
        Ok(weights)
    }

    pub fn validate(&self, config: HyperConnectionConfig) -> Result<()> {
        config.validate()?;
        let expected_fn = config
            .hc_mult
            .checked_mul(config.hc_hidden_size())
            .ok_or_else(|| model_error("head function length overflows usize"))?;
        if self.function.len() != expected_fn {
            return Err(model_error(format!(
                "head function length mismatch: expected {expected_fn}, got {}",
                self.function.len()
            )));
        }
        if self.scale.len() != 1 {
            return Err(model_error(format!(
                "head scale length mismatch: expected 1, got {}",
                self.scale.len()
            )));
        }
        if self.base.len() != config.hc_mult {
            return Err(model_error(format!(
                "head base length mismatch: expected {}, got {}",
                config.hc_mult,
                self.base.len()
            )));
        }
        Ok(())
    }
}

/// Concrete source component shared by model output and proposal heads.
#[derive(Debug, Clone, PartialEq)]
pub struct HyperConnectionHead {
    config: HyperConnectionConfig,
    weights: HyperConnectionHeadWeights,
}

impl HyperConnectionHead {
    pub fn new(config: HyperConnectionConfig, weights: HyperConnectionHeadWeights) -> Result<Self> {
        weights.validate(config)?;
        Ok(Self { config, weights })
    }

    pub fn materialize(
        materializer: &StateDictMaterializer,
        bindings: [&BoundParameter; 3],
        config: HyperConnectionConfig,
    ) -> Result<Self> {
        Self::new(
            config,
            HyperConnectionHeadWeights::materialize(materializer, bindings, config)?,
        )
    }

    pub const fn config(&self) -> HyperConnectionConfig {
        self.config
    }

    pub const fn weights(&self) -> &HyperConnectionHeadWeights {
        &self.weights
    }

    pub fn reference(&self, state: &[f32], tokens: usize) -> Result<Vec<f32>> {
        hc_head_reference(state, tokens, self.config, &self.weights)
    }
}

/// Backend-ready handles for the three HyperConnection parameter tensors.
#[derive(Debug)]
pub struct PreparedHyperConnectionWeights<W> {
    function_row_major: W,
    scale: W,
    base: W,
}

impl<W> PreparedHyperConnectionWeights<W> {
    pub const fn function_row_major(&self) -> &W {
        &self.function_row_major
    }

    pub const fn scale(&self) -> &W {
        &self.scale
    }

    pub const fn base(&self) -> &W {
        &self.base
    }
}

/// Backend-ready HyperConnection assembled independently of any model family.
#[derive(Debug)]
pub struct PreparedHyperConnection<W, N> {
    config: HyperConnectionConfig,
    attention: HyperConnectionPhase<PreparedHyperConnectionWeights<W>, N>,
    feed_forward: HyperConnectionPhase<PreparedHyperConnectionWeights<W>, N>,
}

impl<W, N> PreparedHyperConnection<W, N> {
    pub const fn config(&self) -> HyperConnectionConfig {
        self.config
    }

    pub const fn phase(
        &self,
        stage: HyperConnectionStage,
    ) -> &HyperConnectionPhase<PreparedHyperConnectionWeights<W>, N> {
        match stage {
            HyperConnectionStage::Attention => &self.attention,
            HyperConnectionStage::FeedForward => &self.feed_forward,
        }
    }
}

/// Backend-ready final reduction component shared by decoder and MTP heads.
#[derive(Debug)]
pub struct PreparedHyperConnectionHead<W> {
    config: HyperConnectionConfig,
    weights: PreparedHyperConnectionWeights<W>,
}

impl<W> PreparedHyperConnectionHead<W> {
    pub const fn config(&self) -> HyperConnectionConfig {
        self.config
    }

    pub const fn weights(&self) -> &PreparedHyperConnectionWeights<W> {
        &self.weights
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use ferrule_backend::cuda::operators::linear::{
        CudaF32Buffer, CudaFp8ActivationPack, CudaOperators, CudaPreparedFp8Activation,
    };
    use ferrule_common::Result;

    use super::{
        HyperConnection, HyperConnectionPhase, HyperConnectionStage, HyperConnectionWeights,
        PreparedHyperConnection, PreparedHyperConnectionHead, PreparedHyperConnectionWeights,
    };

    impl PreparedHyperConnection<CudaF32Buffer, CudaF32Buffer> {
        pub fn prepare(operators: &CudaOperators, source: &HyperConnection) -> Result<Self> {
            let prepare_phase = |phase: &HyperConnectionPhase| -> Result<
                HyperConnectionPhase<PreparedHyperConnectionWeights<CudaF32Buffer>, CudaF32Buffer>,
            > {
                Ok(HyperConnectionPhase::new(
                    prepare_weights(operators, phase.weights())?,
                    operators.upload_norm_weight(phase.norm())?,
                ))
            };
            Ok(Self {
                config: source.config(),
                attention: prepare_phase(source.attention())?,
                feed_forward: prepare_phase(source.feed_forward())?,
            })
        }

        pub fn pre<'packed>(
            &self,
            operators: &CudaOperators,
            stage: HyperConnectionStage,
            state: &CudaF32Buffer,
            rows: usize,
            buffers: HyperConnectionPreBuffers<'_, 'packed>,
        ) -> Result<CudaPreparedFp8Activation<'packed>> {
            let phase = self.phase(stage);
            let weights = phase.weights();
            operators.hc_pre_rmsnorm_fp8_into(
                state,
                weights.function_row_major(),
                weights.scale(),
                weights.base(),
                phase.norm(),
                buffers.mix,
                buffers.workspace,
                rows,
                self.config.hc_mult,
                self.config.hidden_size,
                self.config.sinkhorn_iters,
                self.config.eps,
                self.config.norm_eps,
                self.config.norm_eps,
                buffers.hidden,
                buffers.normalized,
                buffers.split_pre,
                buffers.split_post,
                buffers.split_comb,
                buffers.packed,
            )
        }

        pub fn post(
            &self,
            operators: &CudaOperators,
            rows: usize,
            buffers: HyperConnectionPostBuffers<'_>,
        ) -> Result<()> {
            operators.hc_post_from_device_into(
                buffers.hidden,
                buffers.residual,
                buffers.split_post,
                buffers.split_comb,
                rows,
                self.config.hc_mult,
                self.config.hidden_size,
                buffers.output,
            )
        }
    }

    impl PreparedHyperConnectionHead<CudaF32Buffer> {
        pub fn prepare(
            operators: &CudaOperators,
            source: &super::HyperConnectionHead,
        ) -> Result<Self> {
            Ok(Self {
                config: source.config(),
                weights: prepare_head_weights(operators, source.weights())?,
            })
        }

        pub fn reduce(
            &self,
            operators: &CudaOperators,
            state: &CudaF32Buffer,
            rows: usize,
            output: &mut CudaF32Buffer,
        ) -> Result<()> {
            operators.hc_head_from_device_into(
                state,
                self.weights.function_row_major(),
                self.weights.scale(),
                self.weights.base(),
                rows,
                self.config.hc_mult,
                self.config.hidden_size,
                self.config.eps,
                self.config.norm_eps,
                output,
            )
        }
    }

    fn prepare_weights(
        operators: &CudaOperators,
        weights: &HyperConnectionWeights,
    ) -> Result<PreparedHyperConnectionWeights<CudaF32Buffer>> {
        Ok(PreparedHyperConnectionWeights {
            function_row_major: operators.upload_f32_buffer(&weights.function)?,
            scale: operators.upload_f32_buffer(&weights.scale)?,
            base: operators.upload_f32_buffer(&weights.base)?,
        })
    }

    fn prepare_head_weights(
        operators: &CudaOperators,
        weights: &super::HyperConnectionHeadWeights,
    ) -> Result<PreparedHyperConnectionWeights<CudaF32Buffer>> {
        Ok(PreparedHyperConnectionWeights {
            function_row_major: operators.upload_f32_buffer(&weights.function)?,
            scale: operators.upload_f32_buffer(&weights.scale)?,
            base: operators.upload_f32_buffer(&weights.base)?,
        })
    }

    /// Caller-owned scratch for one fused pre-mix, norm, and FP8 producer.
    pub struct HyperConnectionPreBuffers<'scratch, 'packed> {
        pub hidden: &'scratch mut CudaF32Buffer,
        pub normalized: &'scratch mut CudaF32Buffer,
        pub mix: &'scratch mut CudaF32Buffer,
        pub workspace: &'scratch mut CudaF32Buffer,
        pub split_pre: &'scratch mut CudaF32Buffer,
        pub split_post: &'scratch mut CudaF32Buffer,
        pub split_comb: &'scratch mut CudaF32Buffer,
        pub packed: &'packed mut CudaFp8ActivationPack,
    }

    /// Caller-owned inputs and output for one post-sub-layer residual mix.
    pub struct HyperConnectionPostBuffers<'a> {
        pub hidden: &'a CudaF32Buffer,
        pub residual: &'a CudaF32Buffer,
        pub split_post: &'a CudaF32Buffer,
        pub split_comb: &'a CudaF32Buffer,
        pub output: &'a mut CudaF32Buffer,
    }
}

#[cfg(feature = "cuda")]
pub use cuda::{HyperConnectionPostBuffers, HyperConnectionPreBuffers};

/// Reference split tensors produced before a transformer sub-layer.
#[derive(Debug, Clone, PartialEq)]
pub struct HyperConnectionSplit {
    pub tokens: usize,
    pub hc_mult: usize,
    /// `[tokens, hc_mult]`.
    pub pre: Vec<f32>,
    /// `[tokens, hc_mult]`.
    pub post: Vec<f32>,
    /// `[tokens, hc_mult, hc_mult]`.
    pub comb: Vec<f32>,
}

impl HyperConnectionSplit {
    fn validate(&self) -> Result<()> {
        if self.pre.len() != self.tokens * self.hc_mult
            || self.post.len() != self.tokens * self.hc_mult
            || self.comb.len() != self.tokens * self.hc_mult * self.hc_mult
        {
            return Err(model_error("split tensor length mismatch"));
        }
        Ok(())
    }
}

/// Reference pre-mix output.
#[derive(Debug, Clone, PartialEq)]
pub struct HyperConnectionPreOutput {
    /// Reduced hidden `[tokens, hidden_size]`.
    pub hidden: Vec<f32>,
    pub split: HyperConnectionSplit,
}

pub fn hc_split_sinkhorn_reference(
    mixes: &[f32],
    tokens: usize,
    config: HyperConnectionConfig,
    scale: &[f32],
    base: &[f32],
) -> Result<HyperConnectionSplit> {
    config.validate()?;
    let hc = config.hc_mult;
    let mix_hc = config.mix_hc();
    if mixes.len() != tokens * mix_hc {
        return Err(model_error(format!(
            "mixes length mismatch: expected {}, got {}",
            tokens * mix_hc,
            mixes.len()
        )));
    }
    if scale.len() != 3 || base.len() != mix_hc {
        return Err(model_error("split scale/base length mismatch"));
    }

    let mut pre = vec![0.0f32; tokens * hc];
    let mut post = vec![0.0f32; tokens * hc];
    let mut comb = vec![0.0f32; tokens * hc * hc];
    for token in 0..tokens {
        let input = &mixes[token * mix_hc..(token + 1) * mix_hc];
        for copy in 0..hc {
            pre[token * hc + copy] = sigmoid(input[copy] * scale[0] + base[copy]) + config.eps;
            post[token * hc + copy] = 2.0 * sigmoid(input[hc + copy] * scale[1] + base[hc + copy]);
        }
        let comb_offset = token * hc * hc;
        for row in 0..hc {
            for col in 0..hc {
                let source = 2 * hc + row * hc + col;
                comb[comb_offset + row * hc + col] =
                    (input[source] * scale[2] + base[source]).exp();
            }
        }
        normalize_comb_cols(
            &mut comb[comb_offset..comb_offset + hc * hc],
            hc,
            config.eps,
        );
        for _ in 1..config.sinkhorn_iters {
            normalize_comb_rows(
                &mut comb[comb_offset..comb_offset + hc * hc],
                hc,
                config.eps,
            );
            normalize_comb_cols(
                &mut comb[comb_offset..comb_offset + hc * hc],
                hc,
                config.eps,
            );
        }
    }
    Ok(HyperConnectionSplit {
        tokens,
        hc_mult: hc,
        pre,
        post,
        comb,
    })
}

/// Reference pre-sub-layer stream reduction.
pub fn hc_pre_reference(
    state: &[f32],
    tokens: usize,
    config: HyperConnectionConfig,
    weights: &HyperConnectionWeights,
) -> Result<HyperConnectionPreOutput> {
    weights.validate(config)?;
    let hc = config.hc_mult;
    let dim = config.hidden_size;
    let hc_dim = config.hc_hidden_size();
    if state.len() != tokens * hc_dim {
        return Err(model_error(format!(
            "state length mismatch: expected {}, got {}",
            tokens * hc_dim,
            state.len()
        )));
    }

    let mut mixes = vec![0.0f32; tokens * config.mix_hc()];
    for token in 0..tokens {
        let x = &state[token * hc_dim..(token + 1) * hc_dim];
        let rms = rms_factor(x, config.norm_eps);
        for row in 0..config.mix_hc() {
            let w = &weights.function[row * hc_dim..(row + 1) * hc_dim];
            mixes[token * config.mix_hc() + row] = dot(w, x) * rms;
        }
    }
    let split = hc_split_sinkhorn_reference(&mixes, tokens, config, &weights.scale, &weights.base)?;

    let mut hidden = vec![0.0f32; tokens * dim];
    for token in 0..tokens {
        for copy in 0..hc {
            let weight = split.pre[token * hc + copy];
            for d in 0..dim {
                hidden[token * dim + d] += weight * state[(token * hc + copy) * dim + d];
            }
        }
    }
    Ok(HyperConnectionPreOutput { hidden, split })
}

/// Reference post-sub-layer residual mixing.
///
/// Output stream `j` receives `sum_i comb[..., i, j] * residual[..., i, :]`.
pub fn hc_post_reference(
    hidden: &[f32],
    residual: &[f32],
    config: HyperConnectionConfig,
    split: &HyperConnectionSplit,
) -> Result<Vec<f32>> {
    config.validate()?;
    split.validate()?;
    let tokens = split.tokens;
    let hc = config.hc_mult;
    let dim = config.hidden_size;
    if split.hc_mult != hc {
        return Err(model_error(format!(
            "split stream mismatch: split={}, config={hc}",
            split.hc_mult
        )));
    }
    if hidden.len() != tokens * dim || residual.len() != tokens * hc * dim {
        return Err(model_error(format!(
            "post length mismatch: hidden={} residual={}, expected hidden={} residual={}",
            hidden.len(),
            residual.len(),
            tokens * dim,
            tokens * hc * dim
        )));
    }

    let mut output = vec![0.0f32; tokens * hc * dim];
    for token in 0..tokens {
        for out_copy in 0..hc {
            let post = split.post[token * hc + out_copy];
            for d in 0..dim {
                let residual_mix = (0..hc)
                    .map(|in_copy| {
                        split.comb[(token * hc + in_copy) * hc + out_copy]
                            * residual[(token * hc + in_copy) * dim + d]
                    })
                    .sum::<f32>();
                output[(token * hc + out_copy) * dim + d] =
                    post * hidden[token * dim + d] + residual_mix;
            }
        }
    }
    Ok(output)
}

/// Reference final stream reduction.
pub fn hc_head_reference(
    state: &[f32],
    tokens: usize,
    config: HyperConnectionConfig,
    weights: &HyperConnectionHeadWeights,
) -> Result<Vec<f32>> {
    weights.validate(config)?;
    let hc = config.hc_mult;
    let dim = config.hidden_size;
    let hc_dim = config.hc_hidden_size();
    if state.len() != tokens * hc_dim {
        return Err(model_error(format!(
            "head state length mismatch: expected {}, got {}",
            tokens * hc_dim,
            state.len()
        )));
    }
    let mut output = vec![0.0f32; tokens * dim];
    for token in 0..tokens {
        let x = &state[token * hc_dim..(token + 1) * hc_dim];
        let rms = rms_factor(x, config.norm_eps);
        for copy in 0..hc {
            let w = &weights.function[copy * hc_dim..(copy + 1) * hc_dim];
            let mix = dot(w, x) * rms;
            let pre = sigmoid(mix * weights.scale[0] + weights.base[copy]) + config.eps;
            for d in 0..dim {
                output[token * dim + d] += pre * state[(token * hc + copy) * dim + d];
            }
        }
    }
    Ok(output)
}

fn validate_same_residency<const N: usize>(bindings: [&BoundParameter; N]) -> Result<()> {
    let first = bindings
        .first()
        .ok_or_else(|| model_error("empty parameter binding group"))?;
    if bindings
        .iter()
        .skip(1)
        .any(|binding| binding.residency() != first.residency())
    {
        return Err(model_error("parameter binding group mixes residencies"));
    }
    Ok(())
}

fn normalize_comb_rows(comb: &mut [f32], hc: usize, eps: f32) {
    for row in 0..hc {
        let sum = (0..hc).map(|col| comb[row * hc + col]).sum::<f32>();
        for col in 0..hc {
            comb[row * hc + col] /= sum + eps;
        }
    }
}

fn normalize_comb_cols(comb: &mut [f32], hc: usize, eps: f32) {
    for col in 0..hc {
        let sum = (0..hc).map(|row| comb[row * hc + col]).sum::<f32>();
        for row in 0..hc {
            comb[row * hc + col] /= sum + eps;
        }
    }
}

fn rms_factor(x: &[f32], eps: f32) -> f32 {
    let mean = x.iter().map(|value| value * value).sum::<f32>() / x.len() as f32;
    1.0 / (mean + eps).sqrt()
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn model_error(message: impl Into<String>) -> Error {
    Error::Model {
        message: format!("HyperConnection: {}", message.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_sinkhorn_produces_normalized_columns() {
        let config = tiny_config();
        let split = hc_split_sinkhorn_reference(
            &vec![0.0; config.mix_hc()],
            1,
            config,
            &[1.0, 1.0, 1.0],
            &vec![0.0; config.mix_hc()],
        )
        .unwrap();
        assert_eq!(split.pre, vec![0.5 + config.eps; 2]);
        assert_eq!(split.post, vec![1.0; 2]);
        for col in 0..config.hc_mult {
            let sum = (0..config.hc_mult)
                .map(|row| split.comb[row * config.hc_mult + col])
                .sum::<f32>();
            assert!((sum - 1.0).abs() < 1e-5, "column sum {sum}");
        }
    }

    #[test]
    fn post_mix_uses_combination_columns() {
        let config = tiny_config();
        let split = HyperConnectionSplit {
            tokens: 1,
            hc_mult: 2,
            pre: vec![0.0, 0.0],
            post: vec![0.0, 0.0],
            comb: vec![0.0, 1.0, 2.0, 3.0],
        };
        let output =
            hc_post_reference(&[10.0, 20.0], &[5.0, 7.0, 11.0, 13.0], config, &split).unwrap();
        assert_eq!(output, vec![22.0, 26.0, 38.0, 46.0]);
    }

    #[test]
    fn concrete_component_selects_both_phases() {
        let config = tiny_config();
        let weights = zero_weights(config);
        let connection = HyperConnection::new(
            config,
            HyperConnectionPhase::new(weights.clone(), vec![1.0; config.hidden_size]),
            HyperConnectionPhase::new(weights, vec![2.0; config.hidden_size]),
        )
        .unwrap();
        assert_eq!(
            connection.phase(HyperConnectionStage::Attention).norm(),
            &[1.0, 1.0]
        );
        assert_eq!(
            connection.phase(HyperConnectionStage::FeedForward).norm(),
            &[2.0, 2.0]
        );
    }

    #[test]
    fn pre_and_post_follow_reference_shapes() {
        let config = tiny_config();
        let state = vec![1.0, 2.0, 3.0, 4.0];
        let pre = hc_pre_reference(&state, 1, config, &zero_weights(config)).unwrap();
        assert_eq!(pre.hidden.len(), 2);
        let post = hc_post_reference(&pre.hidden, &state, config, &pre.split).unwrap();
        assert_eq!(post.len(), state.len());
        assert!(post[0] > state[0]);
    }

    #[test]
    fn shared_head_reduces_stream_state() {
        let config = tiny_config();
        let head = HyperConnectionHead::new(
            config,
            HyperConnectionHeadWeights {
                function: vec![0.0; config.hc_mult * config.hc_hidden_size()],
                scale: vec![1.0],
                base: vec![0.0; config.hc_mult],
            },
        )
        .unwrap();
        let output = head.reference(&[1.0, 2.0, 3.0, 4.0], 1).unwrap();
        assert_eq!(
            output,
            vec![(0.5 + config.eps) * 4.0, (0.5 + config.eps) * 6.0]
        );
    }

    fn zero_weights(config: HyperConnectionConfig) -> HyperConnectionWeights {
        HyperConnectionWeights {
            function: vec![0.0; config.mix_hc() * config.hc_hidden_size()],
            scale: vec![1.0, 1.0, 1.0],
            base: vec![0.0; config.mix_hc()],
        }
    }

    fn tiny_config() -> HyperConnectionConfig {
        HyperConnectionConfig {
            hc_mult: 2,
            hidden_size: 2,
            sinkhorn_iters: 3,
            eps: 1e-6,
            norm_eps: 1e-6,
        }
    }
}
