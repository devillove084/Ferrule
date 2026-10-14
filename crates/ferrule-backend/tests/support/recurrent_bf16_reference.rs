//! Independent test-only scalar arithmetic and externally generated HF captures.
//! No backend math, native production headers, or production reference helpers.
use std::collections::BTreeMap;

pub struct Case {
    pub rows: usize,
    pub kh: usize,
    pub vh: usize,
    pub dk: usize,
    pub dv: usize,
    pub kernel: usize,
    pub bias: bool,
    pub fields: BTreeMap<String, Vec<f32>>,
}
impl Case {
    pub fn get(&self, key: &str) -> &[f32] {
        &self.fields[key]
    }
    pub fn width(&self) -> usize {
        2 * self.kh * self.dk + self.vh * self.dv
    }
}
pub fn cases() -> Vec<Case> {
    let mut lines = include_str!("recurrent_bf16_hf.txt").lines();
    let mut cases = Vec::new();
    while let Some(header) = lines.next() {
        let dims: Vec<usize> = header
            .strip_prefix("case ")
            .unwrap()
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        let mut fields = BTreeMap::new();
        for line in lines.by_ref() {
            if line == "end" {
                break;
            }
            let mut words = line.split_whitespace();
            let name = words.next().unwrap().to_string();
            fields.insert(name, words.map(|v| v.parse().unwrap()).collect());
        }
        cases.push(Case {
            rows: dims[0],
            kh: dims[1],
            vh: dims[2],
            dk: dims[3],
            dv: dims[4],
            kernel: dims[5],
            bias: dims[6] != 0,
            fields,
        });
    }
    cases
}
// Select between adjacent representable values by distance, not the CUDA bit-bias algorithm.
pub fn round(x: f32) -> f32 {
    assert!(x.is_finite());
    let word = (x.abs().to_bits() >> 16) as u16;
    let lo = f32::from_bits(u32::from(word) << 16);
    let hi = f32::from_bits(u32::from(word + 1) << 16);
    let down = f64::from(x.abs()) - f64::from(lo);
    let up = f64::from(hi) - f64::from(x.abs());
    let r = if down < up || (down == up && word.is_multiple_of(2)) {
        lo
    } else {
        hi
    };
    r.copysign(x)
}
pub fn conv(c: &Case, intermediate_round: bool) -> (Vec<f32>, Vec<f32>) {
    let mut output = vec![0.0; c.rows * c.width()];
    let mut histories = vec![0.0; c.rows * c.width() * c.kernel];
    for channel in 0..c.width() {
        let mut timeline = c.get("history")[channel * c.kernel..(channel + 1) * c.kernel].to_vec();
        timeline.extend((0..c.rows).map(|r| c.get("raw")[r * c.width() + channel]));
        for r in 0..c.rows {
            let mut sum = (0..c.kernel)
                .map(|tap| {
                    f64::from(timeline[r + 1 + tap])
                        * f64::from(c.get("weight")[channel * c.kernel + tap])
                })
                .sum::<f64>() as f32;
            if c.bias {
                sum += c.get("bias")[channel];
            }
            if intermediate_round {
                sum = round(sum);
            }
            output[r * c.width() + channel] =
                round((f64::from(sum) / (1.0 + (-f64::from(sum)).exp())) as f32);
            let base = (r * c.width() + channel) * c.kernel;
            histories[base..base + c.kernel].copy_from_slice(&timeline[r + 1..r + 1 + c.kernel]);
        }
    }
    (output, histories)
}
pub fn normalized(x: &[f32], boundary: bool) -> Vec<f32> {
    if boundary {
        let squares: f32 = x.iter().map(|v| round(v * v)).sum();
        let inv = round(1.0 / round(round(squares) + 1e-6).sqrt());
        x.iter().map(|v| round(v * inv)).collect()
    } else {
        let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt();
        x.iter().map(|v| v * inv).collect()
    }
}
pub fn delta(c: &Case, boundaries: bool, round_state: bool) -> (Vec<f32>, Vec<f32>) {
    let mut s = c.get("state").to_vec();
    let mut states = Vec::new();
    let mut out = Vec::new();
    for t in 0..c.rows {
        for h in 0..c.vh {
            let khead = h / (c.vh / c.kh);
            let qi = t * c.width() + khead * c.dk;
            let ki = qi + c.kh * c.dk;
            let q = normalized(&c.get("conv")[qi..qi + c.dk], boundaries);
            let k = normalized(&c.get("conv")[ki..ki + c.dk], boundaries);
            let x = c.get("a")[t * c.vh + h] + c.get("dt_bias")[h];
            let softplus = if x > 20.0 { x } else { x.exp().ln_1p() };
            let decay = (-c.get("a_log")[h].exp() * softplus).exp();
            let beta = 1.0 / (1.0 + (-c.get("b")[t * c.vh + h]).exp());
            let beta = if boundaries { round(beta) } else { beta };
            let state = &mut s[h * c.dk * c.dv..(h + 1) * c.dk * c.dv];
            for v in state.iter_mut() {
                *v *= decay;
            }
            for v in 0..c.dv {
                let memory: f32 = (0..c.dk).map(|d| state[d * c.dv + v] * k[d]).sum();
                let value = c.get("conv")[t * c.width() + 2 * c.kh * c.dk + h * c.dv + v];
                let correction = (value - memory) * beta;
                for d in 0..c.dk {
                    state[d * c.dv + v] += k[d] * correction;
                    if round_state {
                        state[d * c.dv + v] = round(state[d * c.dv + v]);
                    }
                }
                let scale = 1.0 / (c.dk as f32).sqrt();
                out.push(round(
                    (0..c.dk)
                        .map(|d| state[d * c.dv + v] * (q[d] * scale))
                        .sum(),
                ));
            }
        }
        states.extend_from_slice(&s);
    }
    (out, states)
}
pub fn norm(c: &Case, boundaries: bool) -> Vec<f32> {
    let mut out = Vec::new();
    for (r, x) in c.get("output").chunks_exact(c.dv).enumerate() {
        let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() / c.dv as f32 + 1e-6).sqrt();
        for (d, &value) in x.iter().enumerate().take(c.dv) {
            let v = value * inv;
            let weighted = if boundaries {
                round(round(v) * c.get("norm_weight")[d])
            } else {
                v * c.get("norm_weight")[d]
            };
            let z = c.get("z")[r * c.dv + d];
            out.push(round(weighted * (z / (1.0 + (-z).exp()))));
        }
    }
    out
}
pub fn close(actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (a - b).abs() <= atol + rtol * b.abs(),
            "[{i}] {a} != {b} (atol={atol}, rtol={rtol})"
        );
    }
}

// Hello/decode.0 L0, output 2603 = head 20 / column 43, Dk=128.
// Frozen HF CUDA payload SHA256:
// 6752faa07d618e884c6f4be51694aa071866fa88f4e1d0b7860e3423c9b21b07
// Pinned modeling_qwen3_5_moe.py SHA256:
// 0f8ca77e875c31d448137c9ac291bed505c6b9927a85b5b8460af29d1e1ee647
// Q and final S are captured CUDA operands; this entire S column is also
// bit-exact in production. QNORM is a CPU evaluation of pinned HF l2norm,
// NOT an observed CUDA intermediate. No checkpoint/model load is needed.
pub const CACHED_Q: [u32; 128] = [
    0xbc320000, 0x3bc30000, 0xbc1f0000, 0xbadc0000, 0x3c130000, 0x390c0000, 0x3c070000, 0x3afd0000,
    0xbb4a0000, 0x3cef0000, 0x3af00000, 0xbbbd0000, 0xbbdd0000, 0x3a770000, 0x3bca0000, 0x3f180000,
    0x3cc00000, 0xbcb70000, 0x39e90000, 0xbac20000, 0x3a840000, 0xbb7f0000, 0xbb850000, 0x3bf30000,
    0x3af50000, 0x3b560000, 0xbb350000, 0xbe8d0000, 0x3c680000, 0x3c0c0000, 0x3dbd0000, 0xbc0a0000,
    0xba8c0000, 0x3ab70000, 0xbb0c0000, 0xbbe60000, 0xbc1e0000, 0xbc6e0000, 0x3d230000, 0x3d8e0000,
    0xbc960000, 0xbb870000, 0x3cdd0000, 0xbbaf0000, 0x3b550000, 0x3ca20000, 0xbb600000, 0x3b820000,
    0x3d910000, 0xbdd90000, 0xbc9b0000, 0xb9bd0000, 0xba1e0000, 0xbd410000, 0x3c510000, 0x3d8e0000,
    0xbb4e0000, 0x3dc70000, 0x3d6a0000, 0xbdf60000, 0xbd2d0000, 0x3bf30000, 0x3b4a0000, 0x3aca0000,
    0xbb250000, 0xbb9f0000, 0xbb780000, 0xbcf70000, 0x3c680000, 0x3c440000, 0x3c930000, 0xbcd90000,
    0x3c3d0000, 0xbc520000, 0xbb3e0000, 0xbb060000, 0x3bde0000, 0x3bc20000, 0x3bd30000, 0x3d820000,
    0xbc6b0000, 0x3c530000, 0xbe8d0000, 0x3d520000, 0x3c0c0000, 0xbb790000, 0xbc050000, 0x3e1f0000,
    0x3b190000, 0xbaa40000, 0x3c300000, 0x3b5a0000, 0x3a9a0000, 0x3b580000, 0x3bf50000, 0x3a740000,
    0x3b420000, 0xbd770000, 0x3d000000, 0xbd290000, 0x3d2f0000, 0xbc950000, 0xbd5c0000, 0xbc750000,
    0xbe150000, 0x38410000, 0xbc160000, 0xbc8a0000, 0xbbaa0000, 0x3aac0000, 0x3c250000, 0xbc0f0000,
    0xbb8b0000, 0xbd2a0000, 0x3dce0000, 0xb9290000, 0xbe8c0000, 0x3d480000, 0x3ca10000, 0xbbaa0000,
    0x3bce0000, 0x3c320000, 0xbb670000, 0x3dbd0000, 0xbac70000, 0x3caa0000, 0xbbb10000, 0x3b3e0000,
];
pub const CACHED_QNORM: [u32; 128] = [
    0xbc4f0000, 0x3be30000, 0xbc390000, 0xbb000000, 0x3c2b0000, 0x39230000, 0x3c1d0000, 0x3b130000,
    0xbb6b0000, 0x3d0b0000, 0x3b0c0000, 0xbbdc0000, 0xbc010000, 0x3a900000, 0x3beb0000, 0x3f310000,
    0x3ce00000, 0xbcd50000, 0x3a080000, 0xbae20000, 0x3a9a0000, 0xbb940000, 0xbb9b0000, 0x3c0d0000,
    0x3b0f0000, 0x3b790000, 0xbb530000, 0xbea40000, 0x3c870000, 0x3c230000, 0x3ddc0000, 0xbc210000,
    0xbaa30000, 0x3ad50000, 0xbb230000, 0xbc060000, 0xbc380000, 0xbc8b0000, 0x3d3e0000, 0x3da50000,
    0xbcaf0000, 0xbb9d0000, 0x3d010000, 0xbbcc0000, 0x3b780000, 0x3cbd0000, 0xbb820000, 0x3b970000,
    0x3da90000, 0xbdfd0000, 0xbcb40000, 0xb9dc0000, 0xba380000, 0xbd610000, 0x3c730000, 0x3da50000,
    0xbb700000, 0x3de80000, 0x3d880000, 0xbe0f0000, 0xbd490000, 0x3c0d0000, 0x3b6b0000, 0x3aeb0000,
    0xbb400000, 0xbbb90000, 0xbb900000, 0xbd100000, 0x3c870000, 0x3c640000, 0x3cab0000, 0xbcfd0000,
    0x3c5c0000, 0xbc740000, 0xbb5d0000, 0xbb1c0000, 0x3c010000, 0x3be20000, 0x3bf60000, 0x3d970000,
    0xbc890000, 0x3c760000, 0xbea40000, 0x3d740000, 0x3c230000, 0xbb910000, 0xbc1b0000, 0x3e390000,
    0x3b320000, 0xbabf0000, 0x3c4d0000, 0x3b7e0000, 0x3ab30000, 0x3b7b0000, 0x3c0f0000, 0x3a8e0000,
    0x3b620000, 0xbd900000, 0x3d150000, 0xbd450000, 0x3d4c0000, 0xbcad0000, 0xbd800000, 0xbc8f0000,
    0xbe2d0000, 0x38610000, 0xbc2f0000, 0xbca10000, 0xbbc60000, 0x3ac80000, 0x3c400000, 0xbc260000,
    0xbba20000, 0xbd460000, 0x3df00000, 0xb9450000, 0xbea30000, 0x3d690000, 0x3cbb0000, 0xbbc60000,
    0x3bf00000, 0x3c4f0000, 0xbb860000, 0x3ddc0000, 0xbae80000, 0x3cc60000, 0xbbce0000, 0x3b5d0000,
];
pub const CACHED_FINAL_STATE: [u32; 128] = [
    0xb9398b1a, 0x38b54a0a, 0xb9b0c3be, 0xb982de84, 0x36e8d088, 0xb91e8e24, 0x38c69175, 0xb9a20ced,
    0xb8d97f90, 0x3a6dad14, 0xb9b3c744, 0x3a417468, 0xba7945da, 0x396f5fce, 0x3a315f3c, 0x3a9698f8,
    0x3a27b40a, 0xb7a4cdd1, 0xb9b52b58, 0x36e2a4a6, 0xb9a4daac, 0xb9177f86, 0xb9b7feac, 0xb9177875,
    0xb82eb9a6, 0x38b2a876, 0xba09885e, 0x37560d00, 0x39e67467, 0x3986911c, 0x39423bc1, 0x3983400c,
    0x39d2c25a, 0x39d766b9, 0xb9d14912, 0xb9a6d769, 0x38dec1c6, 0xb8c6cda8, 0x39ac1077, 0x3a510ef8,
    0x391c781a, 0x381d4569, 0xb93142f8, 0xb98534b2, 0xb94b495b, 0xb9e881e6, 0xb7bbf470, 0xb9be3a44,
    0xb9afaf78, 0x383f2cbe, 0xb889e81e, 0xb9408961, 0xb9778c0f, 0xb98f2da2, 0x39ad068a, 0xb8636f44,
    0xba0accd7, 0x3901643a, 0x3a5d13ee, 0xb9f23f0e, 0xb816971f, 0x38158d24, 0xb873335d, 0x39e2b370,
    0x38ab9cb0, 0x399d538a, 0xba901fd3, 0x39958122, 0xba20ebbd, 0x3997124d, 0xb981d3e0, 0x39318291,
    0x38117712, 0xba693d20, 0xb91f78a2, 0xb8b477d2, 0x386391d8, 0x3a17754d, 0xb92878aa, 0x38e82bcf,
    0xba6732f7, 0xba05db08, 0xb96fcd87, 0xb66acadd, 0x368abcc0, 0xb9b14650, 0xb790752a, 0x399736ea,
    0xb9af269e, 0x3903e1a3, 0x38ae7012, 0x397b6a07, 0x38879b0e, 0xb99ece7f, 0xb911c71f, 0xb953f5b8,
    0xb9c01910, 0xba810370, 0x39a8693c, 0xb8d7e086, 0x3a1a0b74, 0xba72ca20, 0xb968e353, 0x38f1c394,
    0x393c0d5a, 0xb9d3e58a, 0x39c5e853, 0xba685130, 0x3a5097dc, 0xb9542bd2, 0x39934968, 0xb9afeaf8,
    0xba2cfed4, 0xb92e9230, 0x39a78082, 0xba24fea7, 0xb9dbb11e, 0x3844b088, 0x3a1e489c, 0xb9b8d7da,
    0xba2847f4, 0xb9bacff4, 0x3850e05d, 0x39cf398e, 0x395097dc, 0x37a86170, 0x38615436, 0x39d34b34,
];
