//! OLMoE — f32 weights, tokenizer.
use ferrule_core::{Error, Result};
use ferrule_gguf::safetensors::SafeTensorsFile;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct OlmoeConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub kv_dim: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub norm_topk_prob: bool,
    pub eos_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
}

impl OlmoeConfig {
    pub fn from_json(json: &serde_json::Value) -> Result<Self> {
        fn token_id(v: Option<&serde_json::Value>) -> Option<u32> {
            match v? {
                serde_json::Value::Number(n) => n.as_u64().map(|id| id as u32),
                serde_json::Value::Array(ids) => ids.first()?.as_u64().map(|id| id as u32),
                _ => None,
            }
        }

        let h = json["hidden_size"].as_u64().unwrap_or(2048) as usize;
        let nh = json["num_attention_heads"].as_u64().unwrap_or(16) as usize;
        // Try multiple config key names for num KV heads
        let nkv = json["num_key_value_heads"]
            .as_u64()
            .or_else(|| json["num_kv_heads"].as_u64())
            .or_else(|| json["kv_heads"].as_u64())
            .map(|v| v as usize)
            .unwrap_or(nh); // fallback to nh; will be corrected from weights after loading
        let hd = h / nh;
        Ok(Self {
            hidden_size: h,
            num_layers: json["num_hidden_layers"].as_u64().unwrap_or(16) as usize,
            num_experts: json["num_experts"].as_u64().unwrap_or(64) as usize,
            num_experts_per_tok: json["num_experts_per_tok"].as_u64().unwrap_or(8) as usize,
            intermediate_size: json["intermediate_size"].as_u64().unwrap_or(1024) as usize,
            vocab_size: json["vocab_size"].as_u64().unwrap_or(50304) as usize,
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
            kv_dim: nkv * hd,
            rope_theta: json["rope_theta"].as_f64().unwrap_or(10000.0) as f32,
            rms_norm_eps: json
                .get("rms_norm_eps")
                .and_then(|v| v.as_f64())
                .unwrap_or(1e-6) as f32,
            norm_topk_prob: json
                .get("norm_topk_prob")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            eos_token_id: token_id(json.get("eos_token_id")),
            pad_token_id: token_id(json.get("pad_token_id")),
        })
    }

    /// Correct kv_dim/nkv from actual weight dimensions (call after loading layer 0).
    pub fn correct_kv_heads(&mut self, actual_kv_dim: usize) {
        if actual_kv_dim != self.kv_dim && actual_kv_dim > 0 {
            eprintln!(
                "  Correcting kv_dim: {} → {} (from weights)",
                self.kv_dim, actual_kv_dim
            );
            self.kv_dim = actual_kv_dim;
            self.num_kv_heads = actual_kv_dim / self.head_dim;
        }
    }
}

pub struct LinearWeight {
    pub w: Vec<f32>,
    pub out_f: usize,
    pub in_f: usize,
}
impl LinearWeight {
    fn forward(&self, x: &[f32], out: &mut [f32]) {
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

pub struct OlmoeModel {
    pub config: OlmoeConfig,
    pub embed: Vec<f32>,
    pub lm_head: Vec<f32>,
    pub final_norm: Vec<f32>,
    pub layers: Vec<LayerWeights>,
    pub model_dir: PathBuf,
    tokenizer: Tokenizer,
}

impl OlmoeModel {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(model_dir.join("config.json"))
            .map_err(|e| Error::Model(format!("config: {e}")))?;
        let cj: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| Error::Model(format!("json: {e}")))?;
        let mut config = OlmoeConfig::from_json(&cj)?;
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| Error::Model(format!("tok: {e}")))?;

        let mut sf_files: Vec<_> = std::fs::read_dir(model_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map(|e| e == "safetensors").unwrap_or(false))
            .collect();
        sf_files.sort();
        eprintln!("Loading {} shards...", sf_files.len());
        let t0 = std::time::Instant::now();

        // Open all shards (mmap, no MAP_POPULATE — deferred page faults)
        let shards: Vec<_> = sf_files
            .iter()
            .map(|p| SafeTensorsFile::open(p))
            .collect::<Result<Vec<_>>>()?;

        // Collect all tensor references from all shards for flat parallel processing
        struct TensorRef<'a> {
            shard: &'a SafeTensorsFile,
            info: &'a ferrule_gguf::safetensors::SafeTensorInfo,
        }
        let all_tensors: Vec<TensorRef> = shards
            .iter()
            .flat_map(|sf| {
                sf.tensors
                    .iter()
                    .map(move |t| TensorRef { shard: sf, info: t })
            })
            .collect();

        // Parallel BF16/F16 → F32 conversion across ALL tensors (not just per-shard)
        let converted: Vec<(String, Vec<f32>, Vec<usize>)> = all_tensors
            .par_iter()
            .filter_map(|tr| {
                tr.shard
                    .tensor_f32(tr.info)
                    .ok()
                    .map(|d| (tr.info.name.clone(), d, tr.info.shape.clone()))
            })
            .collect();

        let mut tmap = HashMap::with_capacity(converted.len());
        for (name, data, shape) in converted {
            tmap.insert(name, (data, shape));
        }
        eprintln!(
            "  {} tensors in {:.1}s",
            tmap.len(),
            t0.elapsed().as_secs_f64()
        );

        let d = config.hidden_size;
        let embed = get(&mut tmap, "model.embed_tokens.weight");
        let lm_head = get(&mut tmap, "lm_head.weight");
        let lm_head = if lm_head.is_empty() {
            // Tied embeddings: use embed as lm_head
            embed.clone()
        } else {
            lm_head
        };
        let final_norm = get(&mut tmap, "model.norm.weight");
        let mut layers = Vec::new();
        for i in 0..config.num_layers {
            let p = format!("model.layers.{i}");
            let attn_norm = get(&mut tmap, &format!("{p}.input_layernorm.weight"));
            let ffn_norm = get(&mut tmap, &format!("{p}.post_attention_layernorm.weight"));
            // Use config.kv_dim as fallback; get_lin will use actual tensor shape
            let attn = AttnWeights {
                q_proj: get_lin(&mut tmap, &format!("{p}.self_attn.q_proj.weight"), d, d),
                k_proj: get_lin(
                    &mut tmap,
                    &format!("{p}.self_attn.k_proj.weight"),
                    config.kv_dim,
                    d,
                ),
                v_proj: get_lin(
                    &mut tmap,
                    &format!("{p}.self_attn.v_proj.weight"),
                    config.kv_dim,
                    d,
                ),
                o_proj: get_lin(&mut tmap, &format!("{p}.self_attn.o_proj.weight"), d, d),
                q_norm: get(&mut tmap, &format!("{p}.self_attn.q_norm.weight")),
                k_norm: get(&mut tmap, &format!("{p}.self_attn.k_norm.weight")),
            };
            // After loading first layer, correct config from actual weight dims
            if i == 0 {
                config.correct_kv_heads(attn.k_proj.out_f);
            }
            let router = get_lin(
                &mut tmap,
                &format!("{p}.mlp.gate.weight"),
                config.num_experts,
                d,
            );
            let mid = config.intermediate_size;
            let mut experts = Vec::new();
            for e in 0..config.num_experts {
                let ep = format!("{p}.mlp.experts.{e}");
                experts.push(ExpertWeights {
                    gate: get_lin(&mut tmap, &format!("{ep}.gate_proj.weight"), mid, d),
                    up: get_lin(&mut tmap, &format!("{ep}.up_proj.weight"), mid, d),
                    down: get_lin(&mut tmap, &format!("{ep}.down_proj.weight"), d, mid),
                });
            }
            layers.push(LayerWeights {
                attn_norm,
                attn,
                ffn_norm,
                router,
                experts,
            });
        }
        eprintln!(
            "  {} layers in {:.1}s",
            layers.len(),
            t0.elapsed().as_secs_f64()
        );
        Ok(Self {
            config,
            embed,
            lm_head,
            final_norm,
            layers,
            model_dir: model_dir.to_path_buf(),
            tokenizer,
        })
    }

    /// Lightweight load: only essential small tensors (norms, embed, router).
    /// Skips all large attention projection and expert FFN weights.
    /// Used when QCache provides the quantized weights (~1s vs 30s full load).
    pub fn load_lightweight(model_dir: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(model_dir.join("config.json"))
            .map_err(|e| Error::Model(format!("config: {e}")))?;
        let cj: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| Error::Model(format!("json: {e}")))?;
        let config = OlmoeConfig::from_json(&cj)?;
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| Error::Model(format!("tok: {e}")))?;

        // Collect names of tensors we actually need
        let mut needed: Vec<String> = vec![
            "model.embed_tokens.weight".into(),
            "lm_head.weight".into(),
            "model.norm.weight".into(),
        ];
        for i in 0..config.num_layers {
            let p = format!("model.layers.{i}");
            needed.push(format!("{p}.input_layernorm.weight"));
            needed.push(format!("{p}.self_attn.q_norm.weight"));
            needed.push(format!("{p}.self_attn.k_norm.weight"));
            needed.push(format!("{p}.post_attention_layernorm.weight"));
            needed.push(format!("{p}.mlp.gate.weight"));
        }
        let needed_set: std::collections::HashSet<_> = needed.iter().collect();

        let mut sf_files: Vec<_> = std::fs::read_dir(model_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map(|e| e == "safetensors").unwrap_or(false))
            .collect();
        sf_files.sort();
        eprintln!(
            "Loading {} shards (lightweight: {} tensors)...",
            sf_files.len(),
            needed.len()
        );
        let t0 = std::time::Instant::now();

        let shards: Vec<_> = sf_files
            .iter()
            .map(|p| SafeTensorsFile::open(p))
            .collect::<Result<Vec<_>>>()?;

        // Only load the needed tensors
        struct TensorRef<'a> {
            shard: &'a SafeTensorsFile,
            info: &'a ferrule_gguf::safetensors::SafeTensorInfo,
        }
        let mut to_load: Vec<TensorRef> = Vec::new();
        for sf in &shards {
            for t in &sf.tensors {
                if needed_set.contains(&t.name) {
                    to_load.push(TensorRef { shard: sf, info: t });
                }
            }
        }

        let converted: Vec<(String, Vec<f32>, Vec<usize>)> = to_load
            .par_iter()
            .filter_map(|tr| {
                tr.shard
                    .tensor_f32(tr.info)
                    .ok()
                    .map(|d| (tr.info.name.clone(), d, tr.info.shape.clone()))
            })
            .collect();

        let mut tmap = HashMap::with_capacity(converted.len());
        for (name, data, shape) in converted {
            tmap.insert(name, (data, shape));
        }
        eprintln!(
            "  {} tensors in {:.1}s",
            tmap.len(),
            t0.elapsed().as_secs_f64()
        );

        let d = config.hidden_size;
        let embed = get(&mut tmap, "model.embed_tokens.weight");
        let lm_head = get(&mut tmap, "lm_head.weight");
        let lm_head = if lm_head.is_empty() {
            embed.clone()
        } else {
            lm_head
        };
        let final_norm = get(&mut tmap, "model.norm.weight");
        let mut layers = Vec::new();
        for i in 0..config.num_layers {
            let p = format!("model.layers.{i}");
            let attn_norm = get(&mut tmap, &format!("{p}.input_layernorm.weight"));
            let ffn_norm = get(&mut tmap, &format!("{p}.post_attention_layernorm.weight"));
            // Attention projection weights are NOT loaded — QCache provides them
            let attn = AttnWeights {
                q_proj: LinearWeight {
                    w: Vec::new(),
                    out_f: d,
                    in_f: d,
                },
                k_proj: LinearWeight {
                    w: Vec::new(),
                    out_f: config.kv_dim,
                    in_f: d,
                },
                v_proj: LinearWeight {
                    w: Vec::new(),
                    out_f: config.kv_dim,
                    in_f: d,
                },
                o_proj: LinearWeight {
                    w: Vec::new(),
                    out_f: d,
                    in_f: d,
                },
                q_norm: get(&mut tmap, &format!("{p}.self_attn.q_norm.weight")),
                k_norm: get(&mut tmap, &format!("{p}.self_attn.k_norm.weight")),
            };
            let router = get_lin(
                &mut tmap,
                &format!("{p}.mlp.gate.weight"),
                config.num_experts,
                d,
            );
            // Expert weights are NOT loaded — QCache provides them
            let experts = Vec::new();
            layers.push(LayerWeights {
                attn_norm,
                attn,
                ffn_norm,
                router,
                experts,
            });
        }
        eprintln!(
            "  {} layers in {:.1}s",
            layers.len(),
            t0.elapsed().as_secs_f64()
        );
        Ok(Self {
            config,
            embed,
            lm_head,
            final_norm,
            layers,
            model_dir: model_dir.to_path_buf(),
            tokenizer,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| Error::Model(format!("encode: {e}")))
    }

    pub fn eos_token_id(&self) -> Option<u32> {
        self.config.eos_token_id
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| Error::Model(format!("decode: {e}")))
    }

    /// CPU forward with real attention + RoPE + KV cache (single token).
    /// Supports GQA: k/v cache entries are kv_dim per position.
    pub fn forward(
        &self,
        token_ids: &[u32],
        k_cache: &mut [Vec<f32>],
        v_cache: &mut [Vec<f32>],
        pos: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let d = self.config.hidden_size;
        let nh = self.config.num_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let kv_dim = self.config.kv_dim;
        let mut hidden = vec![0f32; d];
        // Embedding lookup (last token only — prefill handled by caller)
        if let Some(&tid) = token_ids.last() {
            let s = tid as usize * d;
            if s + d <= self.embed.len() {
                hidden.copy_from_slice(&self.embed[s..s + d]);
            }
        }
        // Precompute RoPE cos/sin for this position
        let mut rope_cos = vec![1.0f32; hd / 2];
        let mut rope_sin = vec![0.0f32; hd / 2];
        let theta = self.config.rope_theta;
        for i in 0..hd / 2 {
            let freq = 1.0 / theta.powf(2.0 * i as f32 / hd as f32);
            let angle = pos as f32 * freq;
            rope_cos[i] = angle.cos();
            rope_sin[i] = angle.sin();
        }

        for (li, layer) in self.layers.iter().enumerate() {
            // Attention norm
            let mut normed = hidden.clone();
            rms_norm(&mut normed, &layer.attn_norm, self.config.rms_norm_eps);
            // Q/K/V projections — k,v use kv_dim outputs (GQA)
            let mut q = vec![0f32; d];
            let mut k = vec![0f32; kv_dim];
            let mut v = vec![0f32; kv_dim];
            layer.attn.q_proj.forward(&normed, &mut q);
            layer.attn.k_proj.forward(&normed, &mut k);
            layer.attn.v_proj.forward(&normed, &mut v);
            // Q/K head norms
            rms_norm(&mut q, &layer.attn.q_norm, self.config.rms_norm_eps);
            rms_norm(&mut k, &layer.attn.k_norm, self.config.rms_norm_eps);
            // RoPE on Q and K (k has nkv heads, q has nh heads)
            rope_cpu(&mut q, &rope_cos, &rope_sin);
            rope_cpu(&mut k, &rope_cos, &rope_sin);
            // Append K,V to cache (kv_dim per position)
            k_cache[li].extend_from_slice(&k);
            v_cache[li].extend_from_slice(&v);
            let seq_len = k_cache[li].len() / kv_dim;
            // Attention with GQA support
            let attn_out = cpu_attention_gqa(&q, &k_cache[li], &v_cache[li], seq_len, nh, nkv, hd);
            // O projection
            let mut ao = vec![0f32; d];
            layer.attn.o_proj.forward(&attn_out, &mut ao);
            for j in 0..d {
                hidden[j] += ao[j];
            }

            // FFN
            let mut ffn_in = hidden.clone();
            rms_norm(&mut ffn_in, &layer.ffn_norm, self.config.rms_norm_eps);
            let mut rl = vec![0f32; self.config.num_experts];
            layer.router.forward(&ffn_in, &mut rl);
            let mut idx: Vec<(usize, f32)> = rl.iter().copied().enumerate().collect();
            idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            idx.truncate(self.config.num_experts_per_tok);
            let max_l = rl.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let all_sum: f32 = rl.iter().map(|v| (*v - max_l).exp()).sum();
            let exps: Vec<f32> = idx.iter().map(|(_, v)| (v - max_l).exp()).collect();
            let topk_sum: f32 = exps.iter().sum();
            let sum = if self.config.norm_topk_prob {
                topk_sum
            } else {
                all_sum
            };
            let mut fo = vec![0f32; d];
            let mid = self.config.intermediate_size;
            for (k, &(eid, _)) in idx.iter().enumerate() {
                let w = exps[k] / sum;
                let ex = &layer.experts[eid];
                let (mut g, mut u) = (vec![0f32; mid], vec![0f32; mid]);
                ex.gate.forward(&ffn_in, &mut g);
                ex.up.forward(&ffn_in, &mut u);
                for j in 0..mid {
                    g[j] = silu(g[j]) * u[j];
                }
                let mut dn = vec![0f32; d];
                ex.down.forward(&g, &mut dn);
                for j in 0..d {
                    fo[j] += w * dn[j];
                }
            }
            for j in 0..d {
                hidden[j] += fo[j];
            }
        }

        // Final layer norm
        rms_norm(&mut hidden, &self.final_norm, self.config.rms_norm_eps);
        let mut logits = vec![0f32; self.config.vocab_size];
        for j in 0..self.config.vocab_size {
            let row = &self.lm_head[j * d..(j + 1) * d];
            logits[j] = row.iter().zip(hidden.iter()).map(|(r, h)| r * h).sum();
        }
        // Debug: print top-8 token ids
        if std::env::var_os("FERRULE_DEBUG_TOPK").is_some() {
            let mut top5: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
            top5.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            top5.truncate(8);
            eprintln!(
                "CPU top8 ids: {:?}",
                top5.iter().map(|(i, _)| i).collect::<Vec<_>>()
            );
        }

        Ok((hidden, logits))
    }
}

fn rope_cpu(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let d = x.len();
    let hd2 = cos.len();
    for h in 0..d / (hd2 * 2) {
        let off = h * hd2 * 2;
        for i in 0..hd2 {
            let c = cos[i];
            let s = sin[i];
            let x0 = x[off + i];
            let x1 = x[off + hd2 + i];
            x[off + i] = x0 * c - x1 * s;
            x[off + hd2 + i] = x0 * s + x1 * c;
        }
    }
}

/// GQA attention: q has nh heads, k/v have nkv heads.
/// For each q head h, use kv head h * nkv / nh.
fn cpu_attention_gqa(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    seq_len: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
) -> Vec<f32> {
    let d_out = nh * hd;
    let kv_dim = nkv * hd;
    let sm_scale = 1.0 / (hd as f32).sqrt();
    let n_rep = nh / nkv; // how many q heads share one kv head
    let mut out = vec![0f32; d_out];
    for h in 0..nh {
        let kv_h = h / n_rep; // which kv head this q head maps to
        let mut scores = vec![0f32; seq_len];
        let mut max_s = f32::NEG_INFINITY;
        for p in 0..seq_len {
            let mut dot = 0.0;
            for j in 0..hd {
                dot += q[h * hd + j] * k_cache[p * kv_dim + kv_h * hd + j];
            }
            scores[p] = dot * sm_scale;
            if scores[p] > max_s {
                max_s = scores[p];
            }
        }
        let mut sum = 0.0;
        for p in 0..seq_len {
            scores[p] = (scores[p] - max_s).exp();
            sum += scores[p];
        }
        for j in 0..hd {
            let mut val = 0.0;
            for p in 0..seq_len {
                val += scores[p] * v_cache[p * kv_dim + kv_h * hd + j];
            }
            out[h * hd + j] = val / sum;
        }
    }
    out
}

fn get(map: &mut HashMap<String, (Vec<f32>, Vec<usize>)>, key: &str) -> Vec<f32> {
    map.remove(key).map(|(v, _)| v).unwrap_or_default()
}

/// Get a LinearWeight, deriving out_f from the actual tensor data.
/// The `out_f` and `in_f` params serve as fallback if the tensor is missing.
fn get_lin(
    map: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
    key: &str,
    out_f: usize,
    in_f: usize,
) -> LinearWeight {
    if let Some((v, shape)) = map.remove(key) {
        // Derive out_f from actual tensor shape: shape[0] = out_features
        let actual_out = if shape.len() >= 2 {
            shape[0]
        } else {
            v.len() / in_f
        };
        LinearWeight {
            w: v,
            out_f: actual_out,
            in_f,
        }
    } else {
        LinearWeight {
            w: vec![0f32; out_f * in_f],
            out_f,
            in_f,
        }
    }
}
pub fn rms_norm(x: &mut [f32], weight: &[f32], eps: f32) {
    let n = x.len() as f32;
    let ms = x.iter().map(|v| v * v).sum::<f32>() / n;
    let r = 1.0 / (ms + eps).sqrt();
    for (xi, wi) in x.iter_mut().zip(weight) {
        *xi *= r * *wi;
    }
}
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
