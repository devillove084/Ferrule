//! Model loading from safetensors shards (full and lightweight paths).
use std::collections::HashMap;
use std::path::Path;

use ferrule_core::{Error, Result};
use ferrule_gguf::safetensors::{SafeTensorInfo, SafeTensorsFile};
use rayon::prelude::*;
use tokenizers::Tokenizer;

use crate::config::OlmoeConfig;
use crate::weights::{AttnWeights, ExpertWeights, LayerWeights, LinearWeight};
use crate::OlmoeModel;

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
        tracing::info!("Loading {} shards...", sf_files.len());
        let t0 = std::time::Instant::now();

        // Open all shards (mmap, no MAP_POPULATE — deferred page faults)
        let shards: Vec<_> = sf_files
            .iter()
            .map(SafeTensorsFile::open)
            .collect::<Result<Vec<_>>>()?;

        // Collect all tensor references from all shards for flat parallel processing
        struct TensorRef<'a> {
            shard: &'a SafeTensorsFile,
            info: &'a SafeTensorInfo,
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
            .map(|tr| {
                tr.shard
                    .tensor_f32(tr.info)
                    .map(|d| (tr.info.name.clone(), d, tr.info.shape.clone()))
            })
            .collect::<Result<Vec<_>>>()?;

        let mut tmap = HashMap::with_capacity(converted.len());
        for (name, data, shape) in converted {
            tmap.insert(name, (data, shape));
        }
        tracing::info!(
            "  {} tensors in {:.1}s",
            tmap.len(),
            t0.elapsed().as_secs_f64()
        );

        let d = config.hidden_size;
        let embed = get_required(&mut tmap, "model.embed_tokens.weight")?;
        let lm_head = get_optional(&mut tmap, "lm_head.weight").unwrap_or_else(|| {
            // Tied embeddings: use embed as lm_head
            embed.clone()
        });
        let final_norm = get_required(&mut tmap, "model.norm.weight")?;
        let mut layers = Vec::new();
        for i in 0..config.num_layers {
            let p = format!("model.layers.{i}");
            let attn_norm = get_required(&mut tmap, &format!("{p}.input_layernorm.weight"))?;
            let ffn_norm =
                get_required(&mut tmap, &format!("{p}.post_attention_layernorm.weight"))?;
            let attn = AttnWeights {
                q_proj: get_lin_required(&mut tmap, &format!("{p}.self_attn.q_proj.weight"), d)?,
                k_proj: get_lin_required(&mut tmap, &format!("{p}.self_attn.k_proj.weight"), d)?,
                v_proj: get_lin_required(&mut tmap, &format!("{p}.self_attn.v_proj.weight"), d)?,
                o_proj: get_lin_required(&mut tmap, &format!("{p}.self_attn.o_proj.weight"), d)?,
                q_norm: get_required(&mut tmap, &format!("{p}.self_attn.q_norm.weight"))?,
                k_norm: get_required(&mut tmap, &format!("{p}.self_attn.k_norm.weight"))?,
            };
            // After loading first layer, correct config from actual weight dims
            if i == 0 {
                config.correct_kv_heads(attn.k_proj.out_f);
            }
            let router = get_lin_required(&mut tmap, &format!("{p}.mlp.gate.weight"), d)?;
            let mid = config.intermediate_size;
            let mut experts = Vec::new();
            for e in 0..config.num_experts {
                let ep = format!("{p}.mlp.experts.{e}");
                experts.push(ExpertWeights {
                    gate: get_lin_required(&mut tmap, &format!("{ep}.gate_proj.weight"), d)?,
                    up: get_lin_required(&mut tmap, &format!("{ep}.up_proj.weight"), d)?,
                    down: get_lin_required(&mut tmap, &format!("{ep}.down_proj.weight"), mid)?,
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
        tracing::info!(
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
    /// Used when WeightPack provides the quantized weights (~1s vs 30s full load).
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
        tracing::info!(
            "Loading {} shards (lightweight: {} tensors)...",
            sf_files.len(),
            needed.len()
        );
        let t0 = std::time::Instant::now();

        let shards: Vec<_> = sf_files
            .iter()
            .map(SafeTensorsFile::open)
            .collect::<Result<Vec<_>>>()?;

        // Only load the needed tensors
        struct TensorRef<'a> {
            shard: &'a SafeTensorsFile,
            info: &'a SafeTensorInfo,
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
            .map(|tr| {
                tr.shard
                    .tensor_f32(tr.info)
                    .map(|d| (tr.info.name.clone(), d, tr.info.shape.clone()))
            })
            .collect::<Result<Vec<_>>>()?;

        let mut tmap = HashMap::with_capacity(converted.len());
        for (name, data, shape) in converted {
            tmap.insert(name, (data, shape));
        }
        tracing::info!(
            "  {} tensors in {:.1}s",
            tmap.len(),
            t0.elapsed().as_secs_f64()
        );

        let d = config.hidden_size;
        let embed = get_required(&mut tmap, "model.embed_tokens.weight")?;
        let lm_head = get_optional(&mut tmap, "lm_head.weight").unwrap_or_else(|| embed.clone());
        let final_norm = get_required(&mut tmap, "model.norm.weight")?;
        let mut layers = Vec::new();
        for i in 0..config.num_layers {
            let p = format!("model.layers.{i}");
            let attn_norm = get_required(&mut tmap, &format!("{p}.input_layernorm.weight"))?;
            let ffn_norm =
                get_required(&mut tmap, &format!("{p}.post_attention_layernorm.weight"))?;
            // Attention projection weights are NOT loaded — WeightPack provides them
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
                q_norm: get_required(&mut tmap, &format!("{p}.self_attn.q_norm.weight"))?,
                k_norm: get_required(&mut tmap, &format!("{p}.self_attn.k_norm.weight"))?,
            };
            let router = get_lin_required(&mut tmap, &format!("{p}.mlp.gate.weight"), d)?;
            // Expert weights are NOT loaded — WeightPack provides them
            let experts = Vec::new();
            layers.push(LayerWeights {
                attn_norm,
                attn,
                ffn_norm,
                router,
                experts,
            });
        }
        tracing::info!(
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
}

fn get_required(map: &mut HashMap<String, (Vec<f32>, Vec<usize>)>, key: &str) -> Result<Vec<f32>> {
    map.remove(key)
        .map(|(v, _)| v)
        .ok_or_else(|| Error::Model(format!("missing required tensor {key}")))
}

fn get_optional(map: &mut HashMap<String, (Vec<f32>, Vec<usize>)>, key: &str) -> Option<Vec<f32>> {
    map.remove(key).map(|(v, _)| v)
}

/// Get a required LinearWeight, deriving out_f from the actual tensor shape.
fn get_lin_required(
    map: &mut HashMap<String, (Vec<f32>, Vec<usize>)>,
    key: &str,
    in_f: usize,
) -> Result<LinearWeight> {
    let (v, shape) = map
        .remove(key)
        .ok_or_else(|| Error::Model(format!("missing required tensor {key}")))?;
    let actual_out = if shape.len() >= 2 {
        shape[0]
    } else {
        v.len().checked_div(in_f).unwrap_or(0)
    };
    Ok(LinearWeight {
        w: v,
        out_f: actual_out,
        in_f,
    })
}
