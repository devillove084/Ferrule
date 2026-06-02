//! DeepSeek-V2/V3 Architecture Implementation
//!
//! Implements MLA (Multi-head Latent Attention) with KV cache compression
//! and MoE (Mixture of Experts) with auxiliary-loss-free load balancing.
//!
//! See docs/DEEPSEEK_ARCHITECTURE.md for detailed documentation.

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Linear, RmsNorm, VarBuilder};
use ferrule_core::{FerruleError, FerruleResult};
use std::sync::{Arc, Mutex};

// ═══════════════════════════════════════════════════════════
// MLA (Multi-head Latent Attention)
// ═══════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct MlaConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub kv_lora_rank: usize,
    pub qk_rope_head_dim: usize,
    pub q_lora_rank: Option<usize>,
    pub qk_nope_head_dim: usize,
    pub v_head_dim: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub dtype: DType,
}

impl Default for MlaConfig {
    fn default() -> Self {
        Self {
            d_model: 2048,
            n_heads: 16,
            kv_lora_rank: 512,
            qk_rope_head_dim: 64,
            q_lora_rank: Some(1536),
            qk_nope_head_dim: 128,
            v_head_dim: 128,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            dtype: DType::F32,
        }
    }
}

impl MlaConfig {
    pub fn head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }
}

/// Compressed KV cache for MLA: stores c_kv = [c_kv_main | k_rope_raw]
/// Shape: [batch, seq, kv_lora_rank + qk_rope_head_dim]
#[derive(Debug, Clone)]
pub struct MlaKvCache(pub Tensor);

pub struct MlaAttention {
    w_q_a: Option<Linear>, // optional Q compression
    q_a_norm: RmsNorm,
    w_q_b: Linear, // q_input -> n_heads x qk_nope_head_dim
    w_qr: Linear,  // q_input -> n_heads x qk_rope_head_dim
    w_dkv: Linear, // d_model -> kv_lora_rank + qk_rope_head_dim
    w_uk: Linear,  // kv_lora_rank -> n_heads x qk_nope_head_dim
    w_uv: Linear,  // kv_lora_rank -> n_heads x v_head_dim
    w_o: Linear,   // n_heads x v_head_dim -> d_model
    config: MlaConfig,
    rope: RoPE,
}

impl MlaAttention {
    pub fn load(vb: VarBuilder, config: &MlaConfig) -> FerruleResult<Self> {
        let (w_q_a, q_input_dim) = match config.q_lora_rank {
            Some(r) => (Some(linear(config.d_model, r, vb.pp("q_a_proj"))?), r),
            None => (None, config.d_model),
        };
        let q_a_norm = rms_norm(q_input_dim, config.rms_norm_eps, vb.pp("q_a_layernorm"))?;
        let w_q_b = linear(
            q_input_dim,
            config.n_heads * config.qk_nope_head_dim,
            vb.pp("q_b_nope_proj"),
        )?;
        let w_qr = linear(
            q_input_dim,
            config.n_heads * config.qk_rope_head_dim,
            vb.pp("q_b_rope_proj"),
        )?;
        let kv_dim = config.kv_lora_rank + config.qk_rope_head_dim;
        let w_dkv = linear(config.d_model, kv_dim, vb.pp("kv_a_proj_with_mqa"))?;
        let w_uk = linear(
            config.kv_lora_rank,
            config.n_heads * config.qk_nope_head_dim,
            vb.pp("kv_b_nope_proj"),
        )?;
        let w_uv = linear(
            config.kv_lora_rank,
            config.n_heads * config.v_head_dim,
            vb.pp("kv_b_v_proj"),
        )?;
        let w_o = linear(
            config.n_heads * config.v_head_dim,
            config.d_model,
            vb.pp("o_proj"),
        )?;
        let rope = RoPE::new(config.qk_rope_head_dim, config.rope_theta)?;
        Ok(Self {
            w_q_a,
            q_a_norm,
            w_q_b,
            w_qr,
            w_dkv,
            w_uk,
            w_uv,
            w_o,
            config: config.clone(),
            rope,
        })
    }

    /// Forward with optional KV cache.
    /// - past_kv: [batch, past_seq, kv_lora_rank + qk_rope_head_dim]
    /// - Returns (output [b, seq, d_model], new_c_kv [b, total_seq, kv_dim])
    pub fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        past_kv: Option<&Tensor>,
    ) -> FerruleResult<(Tensor, Tensor)> {
        let (b_sz, seq_len, _) = x
            .dims3()
            .map_err(|e| FerruleError::Model(format!("dims3: {e}")))?;
        let cfg = &self.config;

        // ── Q (only for current input x) ──
        let q_input = match &self.w_q_a {
            Some(w) => self
                .q_a_norm
                .forward(
                    &w.forward(x)
                        .map_err(|e| FerruleError::Model(format!("qa: {e}")))?,
                )
                .map_err(|e| FerruleError::Model(format!("qan: {e}")))?,
            None => self
                .q_a_norm
                .forward(x)
                .map_err(|e| FerruleError::Model(format!("qan2: {e}")))?,
        };
        let q_c = self
            .w_q_b
            .forward(&q_input)
            .map_err(|e| FerruleError::Model(format!("qb: {e}")))?;
        let qr = self
            .w_qr
            .forward(&q_input)
            .map_err(|e| FerruleError::Model(format!("qr: {e}")))?;
        let fqr = qr
            .reshape((b_sz * cfg.n_heads, seq_len, cfg.qk_rope_head_dim))
            .map_err(|e| FerruleError::Model(format!("qrf: {e}")))?;
        let fqr = self.rope.forward(&fqr)?;
        let q_rope = fqr
            .reshape((b_sz, cfg.n_heads, seq_len, cfg.qk_rope_head_dim))
            .map_err(|e| FerruleError::Model(format!("qru: {e}")))?;
        let q_c = flat_to_heads(&q_c, b_sz, seq_len, cfg.n_heads, cfg.qk_nope_head_dim)?;
        let q = Tensor::cat(&[&q_c, &q_rope], 3)
            .map_err(|e| FerruleError::Model(format!("qcat: {e}")))?;

        // ── KV (compute for x, concat with cache) ──
        let c_kv_new = self
            .w_dkv
            .forward(x)
            .map_err(|e| FerruleError::Model(format!("dkv: {e}")))?;
        let c_kv = match past_kv {
            Some(p) => Tensor::cat(&[p, &c_kv_new], 2)
                .map_err(|e| FerruleError::Model(format!("kvcat: {e}")))?,
            None => c_kv_new,
        };
        let past_len = past_kv.map(|p| p.dim(1).unwrap_or(0)).unwrap_or(0);
        let total_seq = past_len + seq_len;

        let c_kv_main = c_kv
            .narrow(2, 0, cfg.kv_lora_rank)
            .map_err(|e| FerruleError::Model(format!("ckv: {e}")))?;
        let kr = c_kv
            .narrow(2, cfg.kv_lora_rank, cfg.qk_rope_head_dim)
            .map_err(|e| FerruleError::Model(format!("kr: {e}")))?;

        let k_nope = self
            .w_uk
            .forward(&c_kv_main)
            .map_err(|e| FerruleError::Model(format!("uk: {e}")))?;
        let k_rope = self.rope.forward(&kr)?;
        let k_nope = flat_to_heads(&k_nope, b_sz, total_seq, cfg.n_heads, cfg.qk_nope_head_dim)?;
        let kr_h = k_rope
            .unsqueeze(1)
            .map_err(|e| FerruleError::Model(format!("kru: {e}")))?
            .broadcast_as((b_sz, cfg.n_heads, total_seq, cfg.qk_rope_head_dim))
            .map_err(|e| FerruleError::Model(format!("krb: {e}")))?;
        let k = Tensor::cat(&[&k_nope, &kr_h], 3)
            .map_err(|e| FerruleError::Model(format!("kcat: {e}")))?;

        let v = self
            .w_uv
            .forward(&c_kv_main)
            .map_err(|e| FerruleError::Model(format!("uv: {e}")))?;
        let v = flat_to_heads(&v, b_sz, total_seq, cfg.n_heads, cfg.v_head_dim)?;

        // ── Scaled Dot-Product Attention ──
        let scale = (cfg.head_dim() as f64).sqrt();
        let k_t = k
            .transpose(2, 3)
            .map_err(|e| FerruleError::Model(format!("kt: {e}")))?;
        let scores = q
            .matmul(&k_t)
            .map_err(|e| FerruleError::Model(format!("qk: {e}")))?;
        let scores = (&scores / scale).map_err(|e| FerruleError::Model(format!("sc: {e}")))?;
        let scores = match mask {
            Some(m) => scores
                .broadcast_add(m)
                .map_err(|e| FerruleError::Model(format!("mask: {e}")))?,
            None => scores,
        };
        let attn = candle_nn::ops::softmax(&scores, 3)
            .map_err(|e| FerruleError::Model(format!("sm: {e}")))?;
        let attn_out = attn
            .matmul(&v)
            .map_err(|e| FerruleError::Model(format!("av: {e}")))?;
        let attn_out = heads_to_flat(&attn_out, b_sz, seq_len, cfg.n_heads)?;
        let output = self
            .w_o
            .forward(&attn_out)
            .map_err(|e| FerruleError::Model(format!("wo: {e}")))?;

        Ok((output, c_kv))
    }
}

// ═══════════════════════════════════════════════════════════
// MoE (Mixture of Experts) FFN
// ═══════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct MoeConfig {
    pub d_model: usize,
    pub moe_intermediate_size: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub n_activated_experts: usize,
    pub rms_norm_eps: f64,
}

impl Default for MoeConfig {
    fn default() -> Self {
        Self {
            d_model: 2048,
            moe_intermediate_size: 1024,
            n_routed_experts: 8,
            n_shared_experts: 1,
            n_activated_experts: 2,
            rms_norm_eps: 1e-6,
        }
    }
}

/// Single Expert: SwiGLU FFN (gate = SiLU(x @ w_gate), out = (gate * (x @ w_up)) @ w_down)
pub struct Expert {
    w_gate: Linear,
    w_up: Linear,
    w_down: Linear,
}

impl Expert {
    fn load(vb: VarBuilder, d_model: usize, im: usize) -> FerruleResult<Self> {
        Ok(Self {
            w_gate: linear(d_model, im, vb.pp("gate_proj"))?,
            w_up: linear(d_model, im, vb.pp("up_proj"))?,
            w_down: linear(im, d_model, vb.pp("down_proj"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> FerruleResult<Tensor> {
        let gate = candle_nn::ops::silu(
            &self
                .w_gate
                .forward(x)
                .map_err(|e| FerruleError::Model(format!("eg: {e}")))?,
        )
        .map_err(|e| FerruleError::Model(format!("es: {e}")))?;
        let up = self
            .w_up
            .forward(x)
            .map_err(|e| FerruleError::Model(format!("eu: {e}")))?;
        let gated = (&gate * &up).map_err(|e| FerruleError::Model(format!("em: {e}")))?;
        self.w_down
            .forward(&gated)
            .map_err(|e| FerruleError::Model(format!("ed: {e}")))
    }
}

/// Router with Auxiliary-Loss-Free Load Balancing (DeepSeek-V3)
pub struct Router {
    w_router: Linear,
    n_experts: usize,
    top_k: usize,
    bias: Mutex<Vec<f32>>,
    bias_scale: f32,
}

impl Router {
    fn load(vb: VarBuilder, d: usize, n: usize, k: usize) -> FerruleResult<Self> {
        Ok(Self {
            w_router: linear(d, n, vb.pp("router"))?,
            n_experts: n,
            top_k: k,
            bias: Mutex::new(vec![0.0; n]),
            bias_scale: 0.001,
        })
    }

    fn forward(&self, x: &Tensor, device: &Device) -> FerruleResult<(Tensor, Tensor)> {
        let (bs, sl, _) = x
            .dims3()
            .map_err(|e| FerruleError::Model(format!("rd: {e}")))?;
        let nt = bs * sl;
        let xf = x
            .reshape((nt, self.w_router.weight().dims()[1]))
            .map_err(|e| FerruleError::Model(format!("rf: {e}")))?;
        let mut logits = self
            .w_router
            .forward(&xf)
            .map_err(|e| FerruleError::Model(format!("rl: {e}")))?;
        let bt = Tensor::from_vec(
            self.bias.lock().unwrap().clone(),
            (1, self.n_experts),
            device,
        )
        .map_err(|e| FerruleError::Model(format!("rb: {e}")))?;
        logits = logits
            .broadcast_add(&bt)
            .map_err(|e| FerruleError::Model(format!("ra: {e}")))?;
        let (indices, topk_vals) = topk_cpu(&logits, self.top_k)?;
        let weights = candle_nn::ops::softmax(&topk_vals, 1)
            .map_err(|e| FerruleError::Model(format!("rs: {e}")))?;
        // Update load-balancing bias
        let iv: Vec<u32> = indices
            .reshape(((),))
            .map_err(|e| FerruleError::Model(format!("rb1: {e}")))?
            .to_vec1()
            .map_err(|e| FerruleError::Model(format!("rb2: {e}")))?;
        let mut hits = vec![0usize; self.n_experts];
        for &i in &iv {
            hits[i as usize] += 1;
        }
        let mut bias = self.bias.lock().unwrap();
        for e in 0..self.n_experts {
            if hits[e] > 0 {
                bias[e] -= self.bias_scale;
            } else {
                bias[e] += self.bias_scale;
            }
        }
        Ok((indices, weights))
    }
}

/// MoE FFN layer = Shared Experts + Routed Experts
pub struct MoeFfn {
    shared: Vec<Expert>,
    routed: Vec<Expert>,
    router: Router,
    norm: RmsNorm,
    config: MoeConfig,
}

impl MoeFfn {
    pub fn load(vb: VarBuilder, config: &MoeConfig) -> FerruleResult<Self> {
        let mut sh = Vec::new();
        for i in 0..config.n_shared_experts {
            sh.push(Expert::load(
                vb.pp(format!("shared_experts.{i}")),
                config.d_model,
                config.moe_intermediate_size,
            )?);
        }
        let mut rt = Vec::new();
        for i in 0..config.n_routed_experts {
            rt.push(Expert::load(
                vb.pp(format!("experts.{i}")),
                config.d_model,
                config.moe_intermediate_size,
            )?);
        }
        Ok(Self {
            shared: sh,
            routed: rt,
            router: Router::load(
                vb.pp("gate"),
                config.d_model,
                config.n_routed_experts,
                config.n_activated_experts,
            )?,
            norm: rms_norm(
                config.d_model,
                config.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            config: config.clone(),
        })
    }

    pub fn forward(&self, x: &Tensor) -> FerruleResult<Tensor> {
        let (bs, sl, _) = x
            .dims3()
            .map_err(|e| FerruleError::Model(format!("md: {e}")))?;
        let nt = bs * sl;
        let dev = x.device();
        let c = &self.config;
        let normed = self
            .norm
            .forward(x)
            .map_err(|e| FerruleError::Model(format!("mn: {e}")))?;

        // Shared experts
        let mut sh_out = Tensor::zeros((bs, sl, c.d_model), DType::F32, dev)
            .map_err(|e| FerruleError::Model(format!("ms0: {e}")))?;
        for e in &self.shared {
            sh_out = (&sh_out + &e.forward(&normed)?)
                .map_err(|e| FerruleError::Model(format!("ms: {e}")))?;
        }

        // Routed experts
        let (indices, weights) = self.router.forward(&normed, dev)?;
        let iv: Vec<u32> = indices
            .reshape(((),))
            .map_err(|e| FerruleError::Model(format!("mif: {e}")))?
            .to_vec1()
            .map_err(|e| FerruleError::Model(format!("miv: {e}")))?;
        let wv: Vec<f32> = weights
            .reshape(((),))
            .map_err(|e| FerruleError::Model(format!("mwf: {e}")))?
            .to_vec1()
            .map_err(|e| FerruleError::Model(format!("mwv: {e}")))?;
        let tk = c.n_activated_experts;
        let xf = normed
            .reshape((nt, c.d_model))
            .map_err(|e| FerruleError::Model(format!("mxf: {e}")))?;

        let mut tout: Vec<Tensor> = Vec::with_capacity(nt);
        for t in 0..nt {
            let tx = xf
                .get(t)
                .map_err(|e| FerruleError::Model(format!("mg: {e}")))?
                .unsqueeze(0)
                .map_err(|e| FerruleError::Model(format!("mu: {e}")))?;
            let mut acc = Tensor::zeros((1, c.d_model), DType::F32, dev)
                .map_err(|e| FerruleError::Model(format!("ma0: {e}")))?;
            for k in 0..tk {
                let ei = iv[t * tk + k] as usize;
                let w = wv[t * tk + k];
                if w <= 0.0 {
                    continue;
                }
                let eo = self.routed[ei].forward(&tx)?;
                let ws =
                    Tensor::new(&[w], dev).map_err(|e| FerruleError::Model(format!("mws: {e}")))?;
                let wtd = eo
                    .broadcast_mul(&ws)
                    .map_err(|e| FerruleError::Model(format!("mw: {e}")))?;
                acc = (&acc + &wtd).map_err(|e| FerruleError::Model(format!("ma: {e}")))?;
            }
            tout.push(acc);
        }
        let refs: Vec<&Tensor> = tout.iter().collect();
        let rt_out = Tensor::cat(&refs, 0).map_err(|e| FerruleError::Model(format!("mc: {e}")))?;
        let rt_out = rt_out
            .reshape((bs, sl, c.d_model))
            .map_err(|e| FerruleError::Model(format!("mr: {e}")))?;
        let out = (x + &sh_out).map_err(|e| FerruleError::Model(format!("mx1: {e}")))?;
        (&out + &rt_out).map_err(|e| FerruleError::Model(format!("mx2: {e}")))
    }
}

/// CPU-based Top-K: for each row, sort and take top-k values and indices
fn topk_cpu(logits: &Tensor, k: usize) -> FerruleResult<(Tensor, Tensor)> {
    let (nr, _) = logits
        .dims2()
        .map_err(|e| FerruleError::Model(format!("tkd: {e}")))?;
    let data = logits
        .to_vec2::<f32>()
        .map_err(|e| FerruleError::Model(format!("tkv: {e}")))?;
    let dev = logits.device();
    let mut iv = Vec::with_capacity(nr * k);
    let mut vv = Vec::with_capacity(nr * k);
    for row in data.iter() {
        let mut pairs: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for i in 0..k {
            iv.push(pairs[i].0 as u32);
            vv.push(pairs[i].1);
        }
    }
    let idx =
        Tensor::from_vec(iv, (nr, k), dev).map_err(|e| FerruleError::Model(format!("tki: {e}")))?;
    let val = Tensor::from_vec(vv, (nr, k), dev)
        .map_err(|e| FerruleError::Model(format!("tkv2: {e}")))?;
    Ok((idx, val))
}

// ═══════════════════════════════════════════════════════════
// DeepSeek Transformer Block
// ═══════════════════════════════════════════════════════════
//
// Pre-LN: x = x + MLA(attn_norm(x)),  x = x + MoE(ffn_norm(x))

#[derive(Debug, Clone)]
pub struct BlockConfig {
    pub mla: MlaConfig,
    pub moe: MoeConfig,
}

pub struct DeepSeekBlock {
    attn_norm: RmsNorm,
    mla: MlaAttention,
    ffn_norm: RmsNorm,
    moe: MoeFfn,
}

impl DeepSeekBlock {
    pub fn load(vb: VarBuilder, config: &BlockConfig) -> FerruleResult<Self> {
        Ok(Self {
            attn_norm: rms_norm(
                config.mla.d_model,
                config.mla.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            mla: MlaAttention::load(vb.pp("self_attn"), &config.mla)?,
            ffn_norm: rms_norm(
                config.moe.d_model,
                config.moe.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            moe: MoeFfn::load(vb.pp("mlp"), &config.moe)?,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        past_kv: Option<&Tensor>,
    ) -> FerruleResult<(Tensor, Tensor)> {
        let (attn_out, new_kv) = self.mla.forward(
            &self
                .attn_norm
                .forward(x)
                .map_err(|e| FerruleError::Model(format!("ba: {e}")))?,
            mask,
            past_kv,
        )?;
        let x = (x + &attn_out).map_err(|e| FerruleError::Model(format!("br1: {e}")))?;
        let ffn_out = self.moe.forward(
            &self
                .ffn_norm
                .forward(&x)
                .map_err(|e| FerruleError::Model(format!("bf: {e}")))?,
        )?;
        Ok((
            (&x + &ffn_out).map_err(|e| FerruleError::Model(format!("br2: {e}")))?,
            new_kv,
        ))
    }
}

// ═══════════════════════════════════════════════════════════
// DeepSeek Model (full model with KV cache)
// ═══════════════════════════════════════════════════════════

/// Model configuration parsed from config.json
#[derive(Debug, Clone, serde::Deserialize)]
pub struct DeepSeekV2Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    #[serde(default)]
    pub intermediate_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    // MLA-specific
    pub q_lora_rank: Option<usize>,
    pub kv_lora_rank: usize,
    pub qk_rope_head_dim: usize,
    pub qk_nope_head_dim: usize,
    pub v_head_dim: usize,
    // MoE-specific
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub n_activated_experts: usize,
    pub moe_intermediate_size: usize,
    #[serde(default)]
    pub first_k_dense_replace: usize,
    #[serde(default = "default_one")]
    pub moe_layer_freq: usize,
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}
fn default_rope_theta() -> f64 {
    10000.0
}
fn default_one() -> usize {
    1
}

impl DeepSeekV2Config {
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> FerruleResult<Self> {
        let raw = std::fs::read_to_string(path)?;
        serde_json::from_str(&raw)
            .map_err(|e| FerruleError::Config(format!("parse config.json: {e}")))
    }
    pub fn to_mla_config(&self) -> MlaConfig {
        MlaConfig {
            d_model: self.hidden_size,
            n_heads: self.num_attention_heads,
            kv_lora_rank: self.kv_lora_rank,
            qk_rope_head_dim: self.qk_rope_head_dim,
            q_lora_rank: self.q_lora_rank,
            qk_nope_head_dim: self.qk_nope_head_dim,
            v_head_dim: self.v_head_dim,
            rope_theta: self.rope_theta,
            rms_norm_eps: self.rms_norm_eps,
            dtype: DType::F32,
        }
    }
    pub fn to_moe_config(&self) -> MoeConfig {
        MoeConfig {
            d_model: self.hidden_size,
            moe_intermediate_size: self.moe_intermediate_size,
            n_routed_experts: self.n_routed_experts,
            n_shared_experts: self.n_shared_experts,
            n_activated_experts: self.n_activated_experts,
            rms_norm_eps: self.rms_norm_eps,
        }
    }
}

/// Full DeepSeek-V2/V3 model
/// ```
pub struct DeepSeekModel {
    config: DeepSeekV2Config,
    embed: candle_nn::Embedding,
    layers: Vec<DeepSeekBlock>,
    norm: RmsNorm,
    lm_head: Linear,
    device: Arc<Device>,
    eos_token_id: Option<u32>,
}

impl DeepSeekModel {
    pub fn load(
        vb: VarBuilder,
        config: &DeepSeekV2Config,
        device: Arc<Device>,
        eos_token_id: Option<u32>,
    ) -> FerruleResult<Self> {
        let embed = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("model.embed_tokens"),
        )
        .map_err(|e| FerruleError::Model(format!("embed: {e}")))?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let bc = BlockConfig {
                mla: config.to_mla_config(),
                moe: config.to_moe_config(),
            };
            layers.push(DeepSeekBlock::load(
                vb.pp(format!("model.layers.{i}")),
                &bc,
            )?);
        }
        let norm = rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("model.norm"))?;
        let lm_head = linear(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?;
        Ok(Self {
            config: config.clone(),
            embed,
            layers,
            norm,
            lm_head,
            device,
            eos_token_id,
        })
    }

    /// Full forward (no KV cache, for training or single-batch prefill)
    pub fn forward(&self, input_ids: &Tensor, _pos: usize) -> FerruleResult<Tensor> {
        let mut h = self
            .embed
            .forward(input_ids)
            .map_err(|e| FerruleError::Model(format!("ef: {e}")))?;
        for layer in &self.layers {
            let (ho, _) = layer.forward(&h, None, None)?;
            h = ho;
        }
        let h = self
            .norm
            .forward(&h)
            .map_err(|e| FerruleError::Model(format!("nf: {e}")))?;
        self.lm_head
            .forward(&h)
            .map_err(|e| FerruleError::Model(format!("lhf: {e}")))
    }

    /// Incremental forward with per-layer KV cache (for autoregressive decoding)
    ///
    /// - `input_ids`: [batch, 1] for decode, or [batch, N] for initial prefill
    /// - `kv_cache`: per-layer compressed KV, initialized as `vec![None; n_layers]`
    /// - Returns: logits [batch, seq, vocab_size]
    pub fn forward_incremental(
        &self,
        input_ids: &Tensor,
        kv_cache: &mut Vec<Option<Tensor>>,
    ) -> FerruleResult<Tensor> {
        let mut h = self
            .embed
            .forward(input_ids)
            .map_err(|e| FerruleError::Model(format!("eif: {e}")))?;
        for (i, layer) in self.layers.iter().enumerate() {
            let (ho, new_kv) = layer.forward(&h, None, kv_cache[i].as_ref())?;
            kv_cache[i] = Some(new_kv);
            h = ho;
        }
        let h = self
            .norm
            .forward(&h)
            .map_err(|e| FerruleError::Model(format!("nif: {e}")))?;
        self.lm_head
            .forward(&h)
            .map_err(|e| FerruleError::Model(format!("lhif: {e}")))
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn config(&self) -> &DeepSeekV2Config {
        &self.config
    }
}

// ═══════════════════════════════════════════════════════════
// RoPE (Rotary Position Embedding)
// ═══════════════════════════════════════════════════════════

struct RoPE {
    cos: Tensor,
    sin: Tensor,
}

impl RoPE {
    fn new(dim: usize, theta: f64) -> FerruleResult<Self> {
        let max_seq = 8192usize;
        let hd = dim / 2;
        let dev = Device::Cpu;
        let mut freqs = Vec::with_capacity(hd);
        for i in 0..hd {
            freqs.push((1.0 / theta.powf(2.0 * i as f64 / dim as f64)) as f32);
        }
        let freqs = Tensor::from_vec(freqs, hd, &dev)
            .map_err(|e| FerruleError::Model(format!("rpf: {e}")))?;
        let pos: Vec<f32> = (0..max_seq).map(|i| i as f32).collect();
        let pos = Tensor::from_vec(pos, max_seq, &dev)
            .map_err(|e| FerruleError::Model(format!("rpp: {e}")))?;
        let angles = pos
            .unsqueeze(1)
            .map_err(|e| FerruleError::Model(format!("rpu1: {e}")))?
            .broadcast_mul(
                &freqs
                    .unsqueeze(0)
                    .map_err(|e| FerruleError::Model(format!("rpu2: {e}")))?,
            )
            .map_err(|e| FerruleError::Model(format!("rpa: {e}")))?;
        Ok(Self {
            cos: angles
                .cos()
                .map_err(|e| FerruleError::Model(format!("rpc: {e}")))?,
            sin: angles
                .sin()
                .map_err(|e| FerruleError::Model(format!("rps: {e}")))?,
        })
    }

    fn forward(&self, x: &Tensor) -> FerruleResult<Tensor> {
        let r = x.rank();
        let sl = x
            .dim(r - 2)
            .map_err(|e| FerruleError::Model(format!("rds: {e}")))?;
        let cos = self
            .cos
            .narrow(0, 0, sl)
            .map_err(|e| FerruleError::Model(format!("rcn: {e}")))?;
        let sin = self
            .sin
            .narrow(0, 0, sl)
            .map_err(|e| FerruleError::Model(format!("rsn: {e}")))?;
        let cos = if r == 3 {
            cos.unsqueeze(0)
                .map_err(|e| FerruleError::Model(format!("rcu: {e}")))?
        } else {
            cos
        };
        let sin = if r == 3 {
            sin.unsqueeze(0)
                .map_err(|e| FerruleError::Model(format!("rsu: {e}")))?
        } else {
            sin
        };
        let half = x
            .dim(r - 1)
            .map_err(|e| FerruleError::Model(format!("rdd: {e}")))?
            / 2;
        let xe = x
            .narrow(r - 1, 0, half)
            .map_err(|e| FerruleError::Model(format!("rne: {e}")))?;
        let xo = x
            .narrow(r - 1, half, half)
            .map_err(|e| FerruleError::Model(format!("rno: {e}")))?;
        let t1 = xe
            .broadcast_mul(&cos)
            .map_err(|e| FerruleError::Model(format!("rm1: {e}")))?;
        let t2 = xo
            .broadcast_mul(&sin)
            .map_err(|e| FerruleError::Model(format!("rm2: {e}")))?;
        let re = (&t1 - &t2).map_err(|e| FerruleError::Model(format!("rsub: {e}")))?;
        let t3 = xo
            .broadcast_mul(&cos)
            .map_err(|e| FerruleError::Model(format!("rm3: {e}")))?;
        let t4 = xe
            .broadcast_mul(&sin)
            .map_err(|e| FerruleError::Model(format!("rm4: {e}")))?;
        let ro = (&t3 + &t4).map_err(|e| FerruleError::Model(format!("radd: {e}")))?;
        Tensor::cat(&[&re, &ro], r - 1).map_err(|e| FerruleError::Model(format!("rcat: {e}")))
    }
}

// ═══════════════════════════════════════════════════════════
// Utility Functions
// ═══════════════════════════════════════════════════════════

fn linear(a: usize, b: usize, vb: VarBuilder) -> FerruleResult<Linear> {
    candle_nn::linear(a, b, vb).map_err(|e| FerruleError::Model(format!("lin({a}->{b}): {e}")))
}

fn rms_norm(s: usize, e: f64, vb: VarBuilder) -> FerruleResult<RmsNorm> {
    candle_nn::rms_norm(s, e, vb).map_err(|e| FerruleError::Model(format!("rms({s}): {e}")))
}

/// [b, s, nxd] -> [b, n, s, d]
fn flat_to_heads(x: &Tensor, b: usize, s: usize, n: usize, d: usize) -> FerruleResult<Tensor> {
    x.reshape((b, s, n, d))
        .map_err(|e| FerruleError::Model(format!("f2hr: {e}")))?
        .permute((0, 2, 1, 3))
        .map_err(|e| FerruleError::Model(format!("f2hp: {e}")))
}

/// [b, n, s, d] -> [b, s, nxd]
fn heads_to_flat(x: &Tensor, b: usize, s: usize, n: usize) -> FerruleResult<Tensor> {
    let d = x
        .dim(3)
        .map_err(|e| FerruleError::Model(format!("h2fd: {e}")))?;
    x.permute((0, 2, 1, 3))
        .map_err(|e| FerruleError::Model(format!("h2fp: {e}")))?
        .reshape((b, s, n * d))
        .map_err(|e| FerruleError::Model(format!("h2fr: {e}")))
}

// ═══════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mla_shapes() -> FerruleResult<()> {
        let config = MlaConfig {
            d_model: 512,
            n_heads: 8,
            kv_lora_rank: 128,
            qk_rope_head_dim: 32,
            q_lora_rank: Some(256),
            qk_nope_head_dim: 64,
            v_head_dim: 64,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            dtype: DType::F32,
        };
        let dev = Device::Cpu;
        let vm = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mla = MlaAttention::load(vb, &config)?;
        let (bs, sl) = (1usize, 4usize);
        let x = Tensor::randn(0f32, 1f32, (bs, sl, config.d_model), &dev)
            .map_err(|e| FerruleError::Model(format!("rn: {e}")))?;
        let (out, kv) = mla.forward(&x, None, None)?;
        assert_eq!(out.dims(), &[bs, sl, config.d_model]);
        assert_eq!(
            kv.dims(),
            &[bs, sl, config.kv_lora_rank + config.qk_rope_head_dim]
        );
        let std_kv = 2 * sl * config.n_heads * config.head_dim();
        let mla_kv: usize = kv.dims().iter().product();
        println!("=== MLA ===");
        println!(
            "  standard KV: {}, MLA KV: {} ({:.1}%)",
            std_kv,
            mla_kv,
            100.0 * mla_kv as f64 / std_kv as f64
        );
        Ok(())
    }

    #[test]
    fn test_rope() -> FerruleResult<()> {
        let rope = RoPE::new(64, 10000.0)?;
        let dev = Device::Cpu;
        let x = Tensor::randn(0f32, 1f32, (1, 4, 64), &dev)
            .map_err(|e| FerruleError::Model(format!("rn: {e}")))?;
        let p0 = rope.forward(
            &x.narrow(1, 0, 1)
                .map_err(|e| FerruleError::Model(format!("n0: {e}")))?,
        )?;
        let p3 = rope.forward(
            &x.narrow(1, 3, 1)
                .map_err(|e| FerruleError::Model(format!("n3: {e}")))?,
        )?;
        let diff = (&p0 - &p3)
            .map_err(|e| FerruleError::Model(format!("ds: {e}")))?
            .abs()
            .map_err(|e| FerruleError::Model(format!("da: {e}")))?
            .sum_all()
            .map_err(|e| FerruleError::Model(format!("dsum: {e}")))?
            .to_scalar::<f32>()
            .map_err(|e| FerruleError::Model(format!("dsc: {e}")))?;
        assert!(diff > 0.01);
        println!("=== RoPE ===");
        println!("  pos0 vs pos3 diff: {diff:.4}");
        Ok(())
    }

    #[test]
    fn test_moe_shapes() -> FerruleResult<()> {
        let config = MoeConfig {
            d_model: 256,
            moe_intermediate_size: 512,
            n_routed_experts: 4,
            n_shared_experts: 1,
            n_activated_experts: 2,
            rms_norm_eps: 1e-6,
        };
        let dev = Device::Cpu;
        let vm = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let moe = MoeFfn::load(vb, &config)?;
        let (bs, sl) = (1usize, 8usize);
        let x = Tensor::randn(0f32, 1f32, (bs, sl, config.d_model), &dev)
            .map_err(|e| FerruleError::Model(format!("rn: {e}")))?;
        let out = moe.forward(&x)?;
        assert_eq!(out.dims(), &[bs, sl, config.d_model]);
        println!("=== MoE ===");
        println!("  input={:?}, output={:?}", x.dims(), out.dims());
        Ok(())
    }

    #[test]
    fn test_block_shapes() -> FerruleResult<()> {
        let bc = BlockConfig {
            mla: MlaConfig {
                d_model: 256,
                n_heads: 4,
                kv_lora_rank: 64,
                qk_rope_head_dim: 16,
                q_lora_rank: Some(128),
                qk_nope_head_dim: 32,
                v_head_dim: 32,
                rope_theta: 10000.0,
                rms_norm_eps: 1e-6,
                dtype: DType::F32,
            },
            moe: MoeConfig {
                d_model: 256,
                moe_intermediate_size: 512,
                n_routed_experts: 4,
                n_shared_experts: 1,
                n_activated_experts: 2,
                rms_norm_eps: 1e-6,
            },
        };
        let dev = Device::Cpu;
        let vm = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let block = DeepSeekBlock::load(vb, &bc)?;
        let (bs, sl) = (1usize, 8usize);
        let x = Tensor::randn(0f32, 1f32, (bs, sl, bc.mla.d_model), &dev)
            .map_err(|e| FerruleError::Model(format!("rn: {e}")))?;
        let (out, _kv) = block.forward(&x, None, None)?;
        assert_eq!(out.dims(), &[bs, sl, bc.mla.d_model]);
        println!("=== Block ===");
        println!("  input={:?}, output={:?}", x.dims(), out.dims());
        Ok(())
    }

    #[test]
    fn test_model_shapes() -> FerruleResult<()> {
        let config = DeepSeekV2Config {
            hidden_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            vocab_size: 1000,
            intermediate_size: 256,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            q_lora_rank: Some(64),
            kv_lora_rank: 32,
            qk_rope_head_dim: 16,
            qk_nope_head_dim: 32,
            v_head_dim: 32,
            n_routed_experts: 4,
            n_shared_experts: 1,
            n_activated_experts: 2,
            moe_intermediate_size: 128,
            first_k_dense_replace: 0,
            moe_layer_freq: 1,
        };
        let dev = Device::Cpu;
        let vm = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let model = DeepSeekModel::load(vb, &config, Arc::new(dev), None)?;
        let input = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), model.device())
            .map_err(|e| FerruleError::Model(format!("inp: {e}")))?;
        let logits = model.forward(&input, 0)?;
        assert_eq!(logits.dims(), &[1, 4, config.vocab_size]);
        println!("=== Model ===");
        println!(
            "  layers={}, d_model={}, vocab={}",
            config.num_hidden_layers, config.hidden_size, config.vocab_size
        );
        Ok(())
    }
}
