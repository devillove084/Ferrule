//! Weight structures for OLMoE layers (attention, experts, norms).
//! All weights are stored as flat f32 vectors.

pub struct LinearWeight {
    pub w: Vec<f32>,
    pub out_f: usize,
    pub in_f: usize,
}

impl LinearWeight {
    pub(crate) fn forward(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(
            out.len(),
            self.out_f,
            "LinearWeight::forward: out.len()={} != out_f={}",
            out.len(),
            self.out_f
        );
        for j in 0..self.out_f {
            let row = &self.w[j * self.in_f..(j + 1) * self.in_f];
            out[j] = row.iter().zip(x).map(|(r, xi)| r * xi).sum();
        }
    }
}

pub struct ExpertWeights {
    pub gate: LinearWeight,
    pub up: LinearWeight,
    pub down: LinearWeight,
}

pub struct AttnWeights {
    pub q_proj: LinearWeight,
    pub k_proj: LinearWeight,
    pub v_proj: LinearWeight,
    pub o_proj: LinearWeight,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
}

pub struct LayerWeights {
    pub attn_norm: Vec<f32>,
    pub attn: AttnWeights,
    pub ffn_norm: Vec<f32>,
    pub router: LinearWeight,
    pub experts: Vec<ExpertWeights>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_forward_2x2() {
        // w = [[1, 2], [3, 4]] stored row-major as [1, 2, 3, 4]
        let lw = LinearWeight {
            w: vec![1.0, 2.0, 3.0, 4.0],
            out_f: 2,
            in_f: 2,
        };
        let x = vec![1.0, 1.0];
        let mut out = vec![0.0; 2];
        lw.forward(&x, &mut out);
        // [1*1 + 2*1, 3*1 + 4*1] = [3, 7]
        assert!((out[0] - 3.0).abs() < 1e-6);
        assert!((out[1] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_forward_identity() {
        // Identity matrix: w = [[1, 0], [0, 1]]
        let lw = LinearWeight {
            w: vec![1.0, 0.0, 0.0, 1.0],
            out_f: 2,
            in_f: 2,
        };
        let x = vec![5.0, -3.0];
        let mut out = vec![0.0; 2];
        lw.forward(&x, &mut out);
        assert!((out[0] - 5.0).abs() < 1e-6);
        assert!((out[1] - (-3.0)).abs() < 1e-6);
    }

    #[test]
    fn test_linear_forward_3x2() {
        // 3 outputs, 2 inputs
        let lw = LinearWeight {
            w: vec![2.0, 0.0, 0.0, 3.0, 1.0, 1.0],
            out_f: 3,
            in_f: 2,
        };
        let x = vec![4.0, 5.0];
        let mut out = vec![0.0; 3];
        lw.forward(&x, &mut out);
        assert!((out[0] - 8.0).abs() < 1e-6); // 2*4 + 0*5 = 8
        assert!((out[1] - 15.0).abs() < 1e-6); // 0*4 + 3*5 = 15
        assert!((out[2] - 9.0).abs() < 1e-6); // 1*4 + 1*5 = 9
    }
}
