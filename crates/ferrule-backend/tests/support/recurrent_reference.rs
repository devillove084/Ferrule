//! Independent scalar f64 oracles. Deliberately no backend/model math calls.

pub fn conv(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    initial: &[f32],
    channels: usize,
    kernel: usize,
) -> (Vec<f32>, Vec<f32>) {
    let rows = input.len() / channels;
    let mut output = vec![0.0; input.len()];
    let mut final_history = vec![0.0; initial.len()];
    for c in 0..channels {
        let mut timeline: Vec<_> = initial[c * kernel..(c + 1) * kernel].to_vec();
        timeline.extend((0..rows).map(|r| input[r * channels + c]));
        for r in 0..rows {
            let sum = (0..kernel).fold(bias.map_or(0.0, |b| f64::from(b[c])), |sum, k| {
                sum + f64::from(timeline[r + 1 + k]) * f64::from(weight[c * kernel + k])
            });
            output[r * channels + c] = (sum / (1.0 + (-sum).exp())) as f32;
        }
        final_history[c * kernel..(c + 1) * kernel].copy_from_slice(&timeline[rows..rows + kernel]);
    }
    (output, final_history)
}

pub struct DeltaInput<'a> {
    pub qkv: &'a [f32],
    pub a: &'a [f32],
    pub b: &'a [f32],
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    pub initial: &'a [f32],
    pub key_heads: usize,
    pub value_heads: usize,
    pub dk: usize,
    pub dv: usize,
}

pub fn delta(x: DeltaInput<'_>) -> (Vec<f32>, Vec<f32>) {
    let kw = x.key_heads * x.dk;
    let vw = x.value_heads * x.dv;
    let width = 2 * kw + vw;
    let rows = x.qkv.len() / width;
    let mut state: Vec<f64> = x.initial.iter().map(|&v| f64::from(v)).collect();
    let mut output = Vec::with_capacity(rows * vw);
    for t in 0..rows {
        for h in 0..x.value_heads {
            let kh = h / (x.value_heads / x.key_heads);
            let q: Vec<_> = (0..x.dk)
                .map(|d| f64::from(x.qkv[t * width + kh * x.dk + d]))
                .collect();
            let k: Vec<_> = (0..x.dk)
                .map(|d| f64::from(x.qkv[t * width + kw + kh * x.dk + d]))
                .collect();
            let qnorm = (q.iter().map(|q| q * q).sum::<f64>() + 1e-6).sqrt();
            let knorm = (k.iter().map(|k| k * k).sum::<f64>() + 1e-6).sqrt();
            let q: Vec<_> = q
                .into_iter()
                .map(|q| q / qnorm / (x.dk as f64).sqrt())
                .collect();
            let k: Vec<_> = k.into_iter().map(|k| k / knorm).collect();
            let idx = t * x.value_heads + h;
            let a = f64::from(x.a[idx]) + f64::from(x.dt_bias[h]);
            let softplus = if a > 20.0 { a } else { a.exp().ln_1p() };
            let decay = (-f64::from(x.a_log[h]).exp() * softplus).exp();
            let beta = 1.0 / (1.0 + (-f64::from(x.b[idx])).exp());
            let s = &mut state[h * x.dk * x.dv..(h + 1) * x.dk * x.dv];
            for v in s.iter_mut() {
                *v *= decay;
            }
            let correction: Vec<_> = (0..x.dv)
                .map(|v| {
                    let memory = (0..x.dk).map(|d| k[d] * s[d * x.dv + v]).sum::<f64>();
                    beta * (f64::from(x.qkv[t * width + 2 * kw + h * x.dv + v]) - memory)
                })
                .collect();
            for d in 0..x.dk {
                for v in 0..x.dv {
                    s[d * x.dv + v] += k[d] * correction[v];
                }
            }
            for v in 0..x.dv {
                output.push((0..x.dk).map(|d| q[d] * s[d * x.dv + v]).sum::<f64>() as f32);
            }
        }
    }
    (output, state.into_iter().map(|s| s as f32).collect())
}
