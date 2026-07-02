//! Weight structures for OLMoE model loading.
//! Used by OlmoeModel and GpuOlmoeModel for the GPU backend.
//! All weights are stored as flat f32 vectors.

pub struct LinearWeight {
    pub w: Vec<f32>,
    pub out_f: usize,
    pub in_f: usize,
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
