//! Hyper-Connection (HC) reference primitives.
//!
//! HC-style models carry `hc_mult` hidden copies, reduce them before attention/FFN,
//! then expand sub-layer output back into HC state instead of using a plain residual
//! stream. This module implements the math without model-family tensor names so
//! future families can attach similar residual mixers through the same runtime
//! shape.

use ferrule_core::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HyperConnectionConfig {
    pub hc_mult: usize,
    pub hidden_size: usize,
    pub sinkhorn_iters: usize,
    pub eps: f32,
    pub norm_eps: f32,
}

impl HyperConnectionConfig {
    pub fn mix_hc(&self) -> usize {
        (2 + self.hc_mult) * self.hc_mult
    }

    pub fn hc_hidden_size(&self) -> usize {
        self.hc_mult * self.hidden_size
    }

    pub fn validate(&self) -> Result<()> {
        if self.hc_mult == 0 || self.hidden_size == 0 {
            return Err(Error::Model(format!(
                "invalid HC shape: hc_mult={}, hidden_size={}",
                self.hc_mult, self.hidden_size
            )));
        }
        if self.sinkhorn_iters == 0 {
            return Err(Error::Model("HC sinkhorn_iters must be > 0".into()));
        }
        if self.eps <= 0.0 || self.norm_eps <= 0.0 {
            return Err(Error::Model(format!(
                "HC eps values must be positive: eps={}, norm_eps={}",
                self.eps, self.norm_eps
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HyperConnectionWeights {
    /// Row-major `[mix_hc, hc_mult * hidden_size]` for `hc_pre`.
    pub function: Vec<f32>,
    /// `[3]`: pre, post, comb scales.
    pub scale: Vec<f32>,
    /// `[mix_hc]`: pre, post, comb base values.
    pub base: Vec<f32>,
}

impl HyperConnectionWeights {
    pub fn validate(&self, config: HyperConnectionConfig) -> Result<()> {
        config.validate()?;
        let expected_fn = config.mix_hc() * config.hc_hidden_size();
        if self.function.len() != expected_fn {
            return Err(Error::Model(format!(
                "HC function length mismatch: expected {expected_fn}, got {}",
                self.function.len()
            )));
        }
        if self.scale.len() != 3 {
            return Err(Error::Model(format!(
                "HC scale length mismatch: expected 3, got {}",
                self.scale.len()
            )));
        }
        if self.base.len() != config.mix_hc() {
            return Err(Error::Model(format!(
                "HC base length mismatch: expected {}, got {}",
                config.mix_hc(),
                self.base.len()
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HyperConnectionHeadWeights {
    /// Row-major `[hc_mult, hc_mult * hidden_size]` for `hc_head`.
    pub function: Vec<f32>,
    /// `[1]` head scale.
    pub scale: Vec<f32>,
    /// `[hc_mult]` head base.
    pub base: Vec<f32>,
}

impl HyperConnectionHeadWeights {
    pub fn validate(&self, config: HyperConnectionConfig) -> Result<()> {
        config.validate()?;
        let expected_fn = config.hc_mult * config.hc_hidden_size();
        if self.function.len() != expected_fn {
            return Err(Error::Model(format!(
                "HC head function length mismatch: expected {expected_fn}, got {}",
                self.function.len()
            )));
        }
        if self.scale.len() != 1 {
            return Err(Error::Model(format!(
                "HC head scale length mismatch: expected 1, got {}",
                self.scale.len()
            )));
        }
        if self.base.len() != config.hc_mult {
            return Err(Error::Model(format!(
                "HC head base length mismatch: expected {}, got {}",
                config.hc_mult,
                self.base.len()
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HyperConnectionSplit {
    pub tokens: usize,
    pub hc_mult: usize,
    /// `[tokens, hc_mult]`
    pub pre: Vec<f32>,
    /// `[tokens, hc_mult]`
    pub post: Vec<f32>,
    /// `[tokens, hc_mult, hc_mult]`
    pub comb: Vec<f32>,
}

impl HyperConnectionSplit {
    fn validate(&self) -> Result<()> {
        if self.pre.len() != self.tokens * self.hc_mult
            || self.post.len() != self.tokens * self.hc_mult
            || self.comb.len() != self.tokens * self.hc_mult * self.hc_mult
        {
            return Err(Error::Model("HC split tensor length mismatch".into()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HyperConnectionPreOutput {
    /// Reduced hidden `[tokens, hidden_size]`.
    pub hidden: Vec<f32>,
    pub split: HyperConnectionSplit,
}

/// Reference `hc_split_sinkhorn` semantics used by HC-style artifact-bound models.
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
        return Err(Error::Model(format!(
            "HC mixes length mismatch: expected {}, got {}",
            tokens * mix_hc,
            mixes.len()
        )));
    }
    if scale.len() != 3 || base.len() != mix_hc {
        return Err(Error::Model(format!(
            "HC split scale/base mismatch: scale={}, base={}, mix_hc={mix_hc}",
            scale.len(),
            base.len()
        )));
    }

    let mut pre = vec![0.0f32; tokens * hc];
    let mut post = vec![0.0f32; tokens * hc];
    let mut comb = vec![0.0f32; tokens * hc * hc];
    for token in 0..tokens {
        let mix_offset = token * mix_hc;
        for j in 0..hc {
            pre[token * hc + j] = sigmoid(mixes[mix_offset + j] * scale[0] + base[j]) + config.eps;
            post[token * hc + j] =
                2.0 * sigmoid(mixes[mix_offset + hc + j] * scale[1] + base[hc + j]);
        }

        let comb_offset = token * hc * hc;
        for row in 0..hc {
            let mut row_max = f32::NEG_INFINITY;
            for col in 0..hc {
                let idx = mix_offset + 2 * hc + row * hc + col;
                let value = mixes[idx] * scale[2] + base[2 * hc + row * hc + col];
                comb[comb_offset + row * hc + col] = value;
                row_max = row_max.max(value);
            }
            let mut row_sum = 0.0f32;
            for col in 0..hc {
                let idx = comb_offset + row * hc + col;
                comb[idx] = (comb[idx] - row_max).exp();
                row_sum += comb[idx];
            }
            for col in 0..hc {
                let idx = comb_offset + row * hc + col;
                comb[idx] = comb[idx] / row_sum + config.eps;
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

/// Official `Block.hc_pre`: reduce `[tokens, hc, dim]` to `[tokens, dim]`.
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
        return Err(Error::Model(format!(
            "HC state length mismatch: expected {}, got {}",
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

/// Official `Block.hc_post` semantics.
///
/// Note: the Python expression is
/// `post[..., j] * x + sum(comb[..., j, k] * residual[..., j, :], k)`. That is,
/// `residual.unsqueeze(-2)` repeats the same HC copy across the comb row. We keep
/// this exact behavior for reference parity instead of "fixing" it into a full
/// residual-copy matrix multiply.
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
        return Err(Error::Model(format!(
            "HC split multiplier mismatch: split={}, config={hc}",
            split.hc_mult
        )));
    }
    if hidden.len() != tokens * dim || residual.len() != tokens * hc * dim {
        return Err(Error::Model(format!(
            "HC post length mismatch: hidden={}, residual={}, expected hidden={} residual={}",
            hidden.len(),
            residual.len(),
            tokens * dim,
            tokens * hc * dim
        )));
    }

    let mut out = vec![0.0f32; tokens * hc * dim];
    for token in 0..tokens {
        for copy in 0..hc {
            let post = split.post[token * hc + copy];
            let comb_row_sum = (0..hc)
                .map(|k| split.comb[(token * hc + copy) * hc + k])
                .sum::<f32>();
            for d in 0..dim {
                let idx = (token * hc + copy) * dim + d;
                out[idx] = post * hidden[token * dim + d] + comb_row_sum * residual[idx];
            }
        }
    }
    Ok(out)
}

/// Official `Block.hc_head`: reduce final `[tokens, hc, dim]` to `[tokens, dim]`.
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
        return Err(Error::Model(format!(
            "HC head state length mismatch: expected {}, got {}",
            tokens * hc_dim,
            state.len()
        )));
    }
    let mut out = vec![0.0f32; tokens * dim];
    for token in 0..tokens {
        let x = &state[token * hc_dim..(token + 1) * hc_dim];
        let rms = rms_factor(x, config.norm_eps);
        for copy in 0..hc {
            let w = &weights.function[copy * hc_dim..(copy + 1) * hc_dim];
            let mix = dot(w, x) * rms;
            let pre = sigmoid(mix * weights.scale[0] + weights.base[copy]) + config.eps;
            for d in 0..dim {
                out[token * dim + d] += pre * state[(token * hc + copy) * dim + d];
            }
        }
    }
    Ok(out)
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
    let mean = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    1.0 / (mean + eps).sqrt()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hc_split_sinkhorn_produces_expected_shapes_and_normalized_columns() {
        let config = tiny_config();
        let mixes = vec![0.0; config.mix_hc()];
        let split = hc_split_sinkhorn_reference(
            &mixes,
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
            assert!((sum - 1.0).abs() < 1e-5, "col sum {sum}");
        }
    }

    #[test]
    fn hc_pre_and_post_follow_reference_shapes() {
        let config = tiny_config();
        let weights = HyperConnectionWeights {
            function: vec![0.0; config.mix_hc() * config.hc_hidden_size()],
            scale: vec![1.0, 1.0, 1.0],
            base: vec![0.0; config.mix_hc()],
        };
        let state = vec![1.0, 2.0, 3.0, 4.0]; // [tokens=1, hc=2, dim=2]
        let pre = hc_pre_reference(&state, 1, config, &weights).unwrap();
        assert_eq!(pre.hidden.len(), 2);
        let post = hc_post_reference(&pre.hidden, &state, config, &pre.split).unwrap();
        assert_eq!(post.len(), state.len());
        assert!(post[0] > state[0]);
    }

    #[test]
    fn hc_head_reduces_hc_state() {
        let config = tiny_config();
        let weights = HyperConnectionHeadWeights {
            function: vec![0.0; config.hc_mult * config.hc_hidden_size()],
            scale: vec![1.0],
            base: vec![0.0; config.hc_mult],
        };
        let out = hc_head_reference(&[1.0, 2.0, 3.0, 4.0], 1, config, &weights).unwrap();
        assert_eq!(
            out,
            vec![(0.5 + config.eps) * 4.0, (0.5 + config.eps) * 6.0]
        );
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
