//! Scalar F32 gated delta recurrence. State stores raw pre-convolution QKV.
use super::operators::cpu_error;
use ferrule_common::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatedDeltaShape {
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub kernel: usize,
}
impl GatedDeltaShape {
    pub fn sizes(self) -> Result<(usize, usize, usize)> {
        if [
            self.key_heads,
            self.value_heads,
            self.key_dim,
            self.value_dim,
            self.kernel,
        ]
        .contains(&0)
            || !self.value_heads.is_multiple_of(self.key_heads)
        {
            return Err(cpu_error("invalid gated delta shape"));
        }
        let key = self.key_heads.checked_mul(self.key_dim);
        let value = self.value_heads.checked_mul(self.value_dim);
        let qkv = key
            .and_then(|k| k.checked_mul(2))
            .and_then(|k| value.and_then(|v| k.checked_add(v)));
        let conv = qkv.and_then(|n| n.checked_mul(self.kernel));
        let recurrent = value.and_then(|n| n.checked_mul(self.key_dim));
        match (qkv, conv, recurrent) {
            (Some(q), Some(c), Some(r)) => Ok((q, c, r)),
            _ => Err(cpu_error("gated delta shape overflow")),
        }
    }
}

#[derive(Clone, Copy)]
pub struct GatedDeltaWeights<'a> {
    pub conv: &'a [f32],
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    pub norm: &'a [f32],
    pub norm_epsilon: f32,
}
pub struct GatedDeltaInput<'a> {
    pub qkv: &'a [f32],
    pub z: &'a [f32],
    pub a: &'a [f32],
    pub b: &'a [f32],
}
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

/// One token, channel-major convolution history and [value_head,key,value] state.
pub fn gated_delta_step(
    shape: GatedDeltaShape,
    weights: GatedDeltaWeights<'_>,
    input: GatedDeltaInput<'_>,
    conv_history: &mut [f32],
    recurrent: &mut [f32],
) -> Result<Vec<f32>> {
    let (channels, conv_len, recurrent_len) = shape.sizes()?;
    let value_width = shape.value_heads * shape.value_dim;
    if input.qkv.len() != channels
        || input.z.len() != value_width
        || input.a.len() != shape.value_heads
        || input.b.len() != shape.value_heads
        || weights.conv.len() != conv_len
        || conv_history.len() != conv_len
        || recurrent.len() != recurrent_len
        || weights.a_log.len() != shape.value_heads
        || weights.dt_bias.len() != shape.value_heads
        || weights.norm.len() != shape.value_dim
        || !weights.norm_epsilon.is_finite()
        || weights.norm_epsilon <= 0.0
    {
        return Err(cpu_error("gated delta input/weight/state shape mismatch"));
    }
    let mut qkv = vec![0.0; channels];
    for c in 0..channels {
        let history = &mut conv_history[c * shape.kernel..(c + 1) * shape.kernel];
        history.copy_within(1.., 0);
        history[shape.kernel - 1] = input.qkv[c];
        qkv[c] = silu(
            history
                .iter()
                .zip(&weights.conv[c * shape.kernel..(c + 1) * shape.kernel])
                .map(|(x, w)| x * w)
                .sum(),
        );
    }
    let key_width = shape.key_heads * shape.key_dim;
    let mut output = vec![0.0; value_width];
    for h in 0..shape.value_heads {
        let kh = h / (shape.value_heads / shape.key_heads);
        let q = &qkv[kh * shape.key_dim..(kh + 1) * shape.key_dim];
        let k = &qkv[key_width + kh * shape.key_dim..key_width + (kh + 1) * shape.key_dim];
        let qi = (q.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt().recip();
        let qscale = (shape.key_dim as f32).sqrt().recip();
        let ki = (k.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt().recip();
        let g = -weights.a_log[h].exp() * softplus(input.a[h] + weights.dt_bias[h]);
        let decay = g.exp();
        let beta = sigmoid(input.b[h]);
        let state = &mut recurrent
            [h * shape.key_dim * shape.value_dim..(h + 1) * shape.key_dim * shape.value_dim];
        for x in state.iter_mut() {
            *x *= decay;
        }
        for v in 0..shape.value_dim {
            let memory = (0..shape.key_dim)
                .map(|d| state[d * shape.value_dim + v] * (k[d] * ki))
                .sum::<f32>();
            let delta = (qkv[2 * key_width + h * shape.value_dim + v] - memory) * beta;
            for d in 0..shape.key_dim {
                state[d * shape.value_dim + v] += (k[d] * ki) * delta;
            }
            output[h * shape.value_dim + v] = (0..shape.key_dim)
                .map(|d| state[d * shape.value_dim + v] * ((q[d] * qi) * qscale))
                .sum();
        }
        let row = &mut output[h * shape.value_dim..(h + 1) * shape.value_dim];
        let inv = (row.iter().map(|x| x * x).sum::<f32>() / shape.value_dim as f32
            + weights.norm_epsilon)
            .sqrt()
            .recip();
        for (v, x) in row.iter_mut().enumerate() {
            *x = (*x * inv) * weights.norm[v] * silu(input.z[h * shape.value_dim + v]);
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_step_matches_causal_state_contract() {
        let shape = GatedDeltaShape {
            key_heads: 1,
            value_heads: 1,
            key_dim: 2,
            value_dim: 2,
            kernel: 2,
        };
        let (_, conv_len, recurrent_len) = shape.sizes().unwrap();
        let mut conv = vec![0.0; conv_len];
        let mut recurrent = vec![0.0; recurrent_len];
        let conv_weights = vec![1.0; conv_len];
        let weights = GatedDeltaWeights {
            conv: &conv_weights,
            a_log: &[0.0],
            dt_bias: &[0.0],
            norm: &[1.0, 1.0],
            norm_epsilon: 1e-6,
        };
        let first = gated_delta_step(
            shape,
            weights,
            GatedDeltaInput {
                qkv: &[1.0, 0.0, 0.0, 1.0, 1.0, 2.0],
                z: &[0.0, 0.0],
                a: &[0.0],
                b: &[0.0],
            },
            &mut conv,
            &mut recurrent,
        )
        .unwrap();
        assert!(first.iter().all(|value| value.is_finite()));
        assert_eq!(conv[shape.kernel - 1], 1.0);
        let before = recurrent.clone();
        let second = gated_delta_step(
            shape,
            weights,
            GatedDeltaInput {
                qkv: &[0.0, 1.0, 1.0, 0.0, 2.0, 1.0],
                z: &[0.0, 0.0],
                a: &[0.0],
                b: &[0.0],
            },
            &mut conv,
            &mut recurrent,
        )
        .unwrap();
        assert!(second.iter().all(|value| value.is_finite()));
        assert_ne!(
            before, recurrent,
            "delta update must mutate recurrent state"
        );
    }

    #[test]
    fn shape_rejects_non_integral_head_replication() {
        let error = (GatedDeltaShape {
            key_heads: 3,
            value_heads: 2,
            key_dim: 4,
            value_dim: 4,
            kernel: 4,
        })
        .sizes()
        .unwrap_err();
        assert!(error.to_string().contains("invalid gated delta shape"));
    }
}
