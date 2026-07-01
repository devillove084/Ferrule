//! GPU model construction — upload, quantize, cache.

use cuda_core::{CudaContext, DeviceBuffer};
use ferrule_core::{Error, Result};
use ferrule_quant::{f16_to_f32, QMatrix, QuantType};
use rayon::prelude::*;

use crate::context::cu;
use crate::forward::{GpuOlmoeModel, QLayer, Scratch};
use crate::transformer::CudaContiguousKvCache;
use crate::weightpack::{self, WeightPackLayerData};

struct QData {
    qp: QMatrix,
    kp: QMatrix,
    vp: QMatrix,
    op: QMatrix,
    gate_q_packed: Vec<u8>,
    gate_q_scales: Vec<u16>,
    up_q_packed: Vec<u8>,
    up_q_scales: Vec<u16>,
    down_q_packed: Vec<u8>,
    down_q_scales: Vec<u16>,
}

fn f16_slice_to_f32(raw: &[u16]) -> Vec<f32> {
    raw.iter().map(|&b| f16_to_f32(b)).collect()
}

impl GpuOlmoeModel {
    pub(crate) fn build_from_cpu(model: &ferrule_model::OlmoeModel, qt: QuantType) -> Result<Self> {
        let ctx = cu(CudaContext::new(0))?;
        cu(ctx.bind_to_thread())?;
        let module = cu(crate::kernels::kernels::load(&ctx))?;
        let s = ctx.default_stream();
        let c = &model.config;
        let d = c.hidden_size;
        let kv_dim = c.kv_dim;
        let ne = c.num_experts;
        let mid = c.intermediate_size;

        tracing::info!(
            "Uploading & quantizing weights to GPU (GQA: nh={}, nkv={})...",
            c.num_heads,
            c.num_kv_heads
        );
        let t0 = std::time::Instant::now();

        // Embedding stays f32 (lookup, not matmul)
        let emb = cu(DeviceBuffer::from_host(&s, &model.embed))?;
        let lm_head = cu(DeviceBuffer::from_host(&s, &model.lm_head))?;
        let final_norm = cu(DeviceBuffer::from_host(&s, &model.final_norm))?;
        tracing::info!(
            "  embed ({:.1} MB f32) in {:.1}s",
            model.embed.len() as f64 * 4.0 / 1e6,
            t0.elapsed().as_secs_f64()
        );

        let cache_suffix = weightpack::quant_suffix(qt).to_string();
        let cache_file = weightpack::weightpack_path(&model.model_dir, &cache_suffix);
        let layers: Vec<QLayer> = if cache_file.exists() {
            ferrule_core::observability::METRICS
                .cache_hits
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            tracing::info!("Loading from WeightPack: {}", cache_file.display());
            let cache = weightpack::WeightPackReader::open(&cache_file)?;
            if cache.num_layers() != model.layers.len() {
                return Err(Error::Internal(format!(
                    "weight pack layers {} != model layers {}",
                    cache.num_layers(),
                    model.layers.len()
                )));
            }
            if let Some(m) = cache.manifest() {
                validate_weightpack_manifest(m, model, qt, &cache_suffix)?;
            }
            tracing::info!(
                "  {} layers from WeightPack in {:.1}s",
                cache.num_layers(),
                t0.elapsed().as_secs_f64()
            );
            // Upload cached layers directly to GPU
            let mut layers = Vec::new();
            for li in 0..model.layers.len() {
                let cq = cache.layer(li)?;
                let l = &model.layers[li];
                layers.push(QLayer {
                    an: cu(DeviceBuffer::from_host(&s, &l.attn_norm))?,
                    qn: cu(DeviceBuffer::from_host(&s, &l.attn.q_norm))?,
                    kn: cu(DeviceBuffer::from_host(&s, &l.attn.k_norm))?,
                    fn_: cu(DeviceBuffer::from_host(&s, &l.ffn_norm))?,
                    rt: cu(DeviceBuffer::from_host(&s, &l.router.w))?,
                    qp_packed: cu(DeviceBuffer::from_host(&s, cq.qp_packed))?,
                    qp_scales: cu(DeviceBuffer::from_host(
                        &s,
                        &weightpack::bytes_to_f32_vec(cq.qp_scales)?,
                    ))?,
                    kp_packed: cu(DeviceBuffer::from_host(&s, cq.kp_packed))?,
                    kp_scales: cu(DeviceBuffer::from_host(
                        &s,
                        &weightpack::bytes_to_f32_vec(cq.kp_scales)?,
                    ))?,
                    vp_packed: cu(DeviceBuffer::from_host(&s, cq.vp_packed))?,
                    vp_scales: cu(DeviceBuffer::from_host(
                        &s,
                        &weightpack::bytes_to_f32_vec(cq.vp_scales)?,
                    ))?,
                    op_packed: cu(DeviceBuffer::from_host(&s, cq.op_packed))?,
                    op_scales: cu(DeviceBuffer::from_host(
                        &s,
                        &weightpack::bytes_to_f32_vec(cq.op_scales)?,
                    ))?,
                    ex_gate_packed: cu(DeviceBuffer::from_host(&s, cq.gate_q_packed))?,
                    ex_gate_scales: cu(DeviceBuffer::from_host(
                        &s,
                        &weightpack::bytes_to_f32_vec(cq.gate_q_scales)?,
                    ))?,
                    ex_up_packed: cu(DeviceBuffer::from_host(&s, cq.up_q_packed))?,
                    ex_up_scales: cu(DeviceBuffer::from_host(
                        &s,
                        &weightpack::bytes_to_f32_vec(cq.up_q_scales)?,
                    ))?,
                    ex_down_packed: cu(DeviceBuffer::from_host(&s, cq.down_q_packed))?,
                    ex_down_scales: cu(DeviceBuffer::from_host(
                        &s,
                        &weightpack::bytes_to_f32_vec(cq.down_q_scales)?,
                    ))?,
                });
            }
            layers
        } else {
            // ── Quantize weights + upload to GPU + write WeightPack ──
            ferrule_core::observability::METRICS
                .cache_misses
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            tracing::info!(
                "Quantizing & uploading {} layers (pipelined)...",
                model.layers.len()
            );
            let (tx, rx) = std::sync::mpsc::sync_channel::<(usize, QData, WeightPackLayerData)>(2);
            std::thread::scope(|scope| -> ferrule_core::Result<Vec<QLayer>> {
                // Spawn quantization thread (rayon-parallel across layers, streamed via channel)
                scope.spawn(move || {
                    model.layers.par_iter().enumerate().for_each(|(li, l)| {
                        let qp = QMatrix::quantize(&l.attn.q_proj.w, d, d, qt);
                        let kp = QMatrix::quantize(&l.attn.k_proj.w, l.attn.k_proj.out_f, d, qt);
                        let vp = QMatrix::quantize(&l.attn.v_proj.w, l.attn.v_proj.out_f, d, qt);
                        let op = QMatrix::quantize(&l.attn.o_proj.w, d, d, qt);
                        let mut gate_q_packed = Vec::new();
                        let mut gate_q_scales = Vec::new();
                        let mut up_q_packed = Vec::new();
                        let mut up_q_scales = Vec::new();
                        let mut down_q_packed = Vec::new();
                        let mut down_q_scales = Vec::new();
                        for e in &l.experts {
                            let gq = QMatrix::quantize(&e.gate.w, mid, d, qt);
                            gate_q_packed.extend_from_slice(&gq.packed);
                            gate_q_scales.extend_from_slice(&gq.scales);
                            let uq = QMatrix::quantize(&e.up.w, mid, d, qt);
                            up_q_packed.extend_from_slice(&uq.packed);
                            up_q_scales.extend_from_slice(&uq.scales);
                            let dq = QMatrix::quantize(&e.down.w, d, mid, qt);
                            down_q_packed.extend_from_slice(&dq.packed);
                            down_q_scales.extend_from_slice(&dq.scales);
                        }
                        let cache = WeightPackLayerData::from_qmatrix(
                            &qp,
                            &kp,
                            &vp,
                            &op,
                            &gate_q_packed,
                            &gate_q_scales,
                            &up_q_packed,
                            &up_q_scales,
                            &down_q_packed,
                            &down_q_scales,
                        );
                        let _ = tx.send((
                            li,
                            QData {
                                qp,
                                kp,
                                vp,
                                op,
                                gate_q_packed,
                                gate_q_scales,
                                up_q_packed,
                                up_q_scales,
                                down_q_packed,
                                down_q_scales,
                            },
                            cache,
                        ));
                    });
                });

                // Main thread: receive quantized layers in order, upload to GPU
                let mut pending: std::collections::BTreeMap<usize, (QData, WeightPackLayerData)> =
                    std::collections::BTreeMap::new();
                let mut next_layer = 0usize;
                let mut layers = Vec::new();
                let mut cache_layers: Vec<WeightPackLayerData> = Vec::new();
                for (li, q, cq) in rx {
                    pending.insert(li, (q, cq));
                    while let Some((q, cq)) = pending.remove(&next_layer) {
                        let l = &model.layers[next_layer];
                        let tl = std::time::Instant::now();
                        layers.push(QLayer {
                            an: cu(DeviceBuffer::from_host(&s, &l.attn_norm))?,
                            qn: cu(DeviceBuffer::from_host(&s, &l.attn.q_norm))?,
                            kn: cu(DeviceBuffer::from_host(&s, &l.attn.k_norm))?,
                            fn_: cu(DeviceBuffer::from_host(&s, &l.ffn_norm))?,
                            rt: cu(DeviceBuffer::from_host(&s, &l.router.w))?,
                            qp_packed: cu(DeviceBuffer::from_host(&s, &q.qp.packed))?,
                            qp_scales: cu(DeviceBuffer::from_host(
                                &s,
                                &f16_slice_to_f32(&q.qp.scales),
                            ))?,
                            kp_packed: cu(DeviceBuffer::from_host(&s, &q.kp.packed))?,
                            kp_scales: cu(DeviceBuffer::from_host(
                                &s,
                                &f16_slice_to_f32(&q.kp.scales),
                            ))?,
                            vp_packed: cu(DeviceBuffer::from_host(&s, &q.vp.packed))?,
                            vp_scales: cu(DeviceBuffer::from_host(
                                &s,
                                &f16_slice_to_f32(&q.vp.scales),
                            ))?,
                            op_packed: cu(DeviceBuffer::from_host(&s, &q.op.packed))?,
                            op_scales: cu(DeviceBuffer::from_host(
                                &s,
                                &f16_slice_to_f32(&q.op.scales),
                            ))?,
                            ex_gate_packed: cu(DeviceBuffer::from_host(&s, &q.gate_q_packed))?,
                            ex_gate_scales: cu(DeviceBuffer::from_host(
                                &s,
                                &f16_slice_to_f32(&q.gate_q_scales),
                            ))?,
                            ex_up_packed: cu(DeviceBuffer::from_host(&s, &q.up_q_packed))?,
                            ex_up_scales: cu(DeviceBuffer::from_host(
                                &s,
                                &f16_slice_to_f32(&q.up_q_scales),
                            ))?,
                            ex_down_packed: cu(DeviceBuffer::from_host(&s, &q.down_q_packed))?,
                            ex_down_scales: cu(DeviceBuffer::from_host(
                                &s,
                                &f16_slice_to_f32(&q.down_q_scales),
                            ))?,
                        });
                        tracing::info!(
                            "  layer {next_layer:>2} in {:.1}s",
                            tl.elapsed().as_secs_f64()
                        );
                        cache_layers.push(cq);
                        next_layer += 1;
                    }
                }
                if !cache_layers.is_empty() {
                    tracing::info!("Writing WeightPack: {}", cache_file.display());
                    let qt_suffix = weightpack::quant_suffix(qt);
                    let config_hash = weightpack::model_config_hash(&model.model_dir)?;
                    let mut manifest = weightpack::WeightPackManifest::new(
                        &format!("{qt:?}"),
                        qt_suffix,
                        cache_layers.len(),
                        &config_hash,
                        qt_suffix,
                    );
                    manifest.tensor_shapes = weightpack_tensor_shapes(model);
                    weightpack::write_weightpack(&cache_file, &cache_layers, Some(&manifest))?;
                }
                Ok(layers)
            })?
        };

        tracing::info!("  Done ({:.1}s total).", t0.elapsed().as_secs_f64());

        // ── Common setup (GPU free check, scratch buffers, RoPE, KV cache) ──
        // Free VMEM check
        let mut free: usize = 0;
        let mut total: usize = 0;
        unsafe {
            cuda_bindings::cuMemGetInfo_v2(&mut free, &mut total);
        }
        tracing::info!(
            "  GPU free: {:.1} MB / {:.1} MB",
            free as f64 / 1e6,
            total as f64 / 1e6
        );

        tracing::debug!("Allocating scratch buffers...");
        let scratch = Scratch {
            hidden: cu(DeviceBuffer::<f32>::zeroed(&s, d))?,
            normed: cu(DeviceBuffer::<f32>::zeroed(&s, d))?,
            q: cu(DeviceBuffer::<f32>::zeroed(&s, d))?,
            k: cu(DeviceBuffer::<f32>::zeroed(&s, kv_dim))?,
            v: cu(DeviceBuffer::<f32>::zeroed(&s, kv_dim))?,
            q_tmp: cu(DeviceBuffer::<f32>::zeroed(&s, d))?,
            k_tmp: cu(DeviceBuffer::<f32>::zeroed(&s, kv_dim))?,
            ao: cu(DeviceBuffer::<f32>::zeroed(&s, d))?,
            h_tmp1: cu(DeviceBuffer::<f32>::zeroed(&s, d))?,
            h_tmp2: cu(DeviceBuffer::<f32>::zeroed(&s, d))?,
            ffn_in: cu(DeviceBuffer::<f32>::zeroed(&s, d))?,
            router_out: cu(DeviceBuffer::<f32>::zeroed(&s, ne))?,
            fo: cu(DeviceBuffer::<f32>::zeroed(&s, d))?,
            gb: cu(DeviceBuffer::<f32>::zeroed(&s, mid))?,
            ub: cu(DeviceBuffer::<f32>::zeroed(&s, mid))?,
            gb2: cu(DeviceBuffer::<f32>::zeroed(&s, mid))?,
            db: cu(DeviceBuffer::<f32>::zeroed(&s, d))?,
            logits: cu(DeviceBuffer::<f32>::zeroed(&s, c.vocab_size))?,
            topk_idx: cu(DeviceBuffer::<f32>::zeroed(&s, c.num_experts_per_tok))?,
            topk_w: cu(DeviceBuffer::<f32>::zeroed(&s, c.num_experts_per_tok))?,
            topk_vocab_idx: cu(DeviceBuffer::<f32>::zeroed(&s, 40))?,
            topk_vocab_val: cu(DeviceBuffer::<f32>::zeroed(&s, 40))?,
        };

        let nh = c.num_heads;
        let nkv = c.num_kv_heads;
        let hd = d / nh;
        let max_seq = 4096usize;
        // Precompute RoPE cos/sin on CPU, upload to GPU
        let mut cos_cpu = vec![0f32; max_seq * hd / 2];
        let mut sin_cpu = vec![0f32; max_seq * hd / 2];
        let theta = c.rope_theta;
        for pos in 0..max_seq {
            for i in 0..hd / 2 {
                let freq = 1.0 / theta.powf(2.0 * i as f32 / hd as f32);
                let angle = pos as f32 * freq;
                cos_cpu[pos * hd / 2 + i] = angle.cos();
                sin_cpu[pos * hd / 2 + i] = angle.sin();
            }
        }
        let rope_cos = cu(DeviceBuffer::from_host(&s, &cos_cpu))?;
        let rope_sin = cu(DeviceBuffer::from_host(&s, &sin_cpu))?;
        let kv = CudaContiguousKvCache::new(&s, c.num_layers, max_seq, kv_dim, nh)?;

        let mut m = Self {
            ctx,
            s,
            module,
            emb,
            lm_head,
            final_norm,
            layers,
            d,
            kv_dim,
            ne,
            na: c.num_experts_per_tok,
            mid,
            vocab: c.vocab_size,
            eps: c.rms_norm_eps,
            norm_topk_prob: c.norm_topk_prob,
            qt,
            scratch,
            nh,
            nkv,
            hd,
            rope_cos,
            rope_sin,
            kv,
            expert_hits: vec![vec![0usize; ne]; c.num_layers],
            total_tokens: 0,
        };
        m.capture_decode_graph()?;
        Ok(m)
    }

    /// Build GPU model from a lightweight CPU model + pre-existing WeightPack.
    /// Skips full FP32 weight loading — norms, router, embed, lm_head come
    /// from the lightweight model; quantized attention/expert weights come
    /// from the WeightPack.
    pub(crate) fn build_from_lightweight(
        model: &ferrule_model::OlmoeModel,
        cache: &weightpack::WeightPackReader,
        qt: QuantType,
    ) -> Result<Self> {
        let ctx = cu(CudaContext::new(0))?;
        cu(ctx.bind_to_thread())?;
        let module = cu(crate::kernels::kernels::load(&ctx))?;
        let s = ctx.default_stream();
        let c = &model.config;
        let d = c.hidden_size;
        let kv_dim = c.kv_dim;
        let ne = c.num_experts;
        let mid = c.intermediate_size;

        tracing::info!("Uploading lightweight model + weight pack to GPU...");
        let t0 = std::time::Instant::now();

        let emb = cu(DeviceBuffer::from_host(&s, &model.embed))?;
        let lm_head = cu(DeviceBuffer::from_host(&s, &model.lm_head))?;
        let final_norm = cu(DeviceBuffer::from_host(&s, &model.final_norm))?;
        tracing::info!(
            "  embed ({:.1} MB f32) in {:.1}s",
            model.embed.len() as f64 * 4.0 / 1e6,
            t0.elapsed().as_secs_f64()
        );

        // Validate WeightPack
        if let Some(m) = cache.manifest() {
            validate_weightpack_manifest(m, model, qt, weightpack::quant_suffix(qt))?;
        }
        if cache.num_layers() != model.layers.len() {
            return Err(Error::Internal(format!(
                "weight pack layers {} != model layers {}",
                cache.num_layers(),
                model.layers.len()
            )));
        }

        let mut layers = Vec::new();
        for li in 0..model.layers.len() {
            let cq = cache.layer(li)?;
            let l = &model.layers[li];
            layers.push(QLayer {
                an: cu(DeviceBuffer::from_host(&s, &l.attn_norm))?,
                qn: cu(DeviceBuffer::from_host(&s, &l.attn.q_norm))?,
                kn: cu(DeviceBuffer::from_host(&s, &l.attn.k_norm))?,
                fn_: cu(DeviceBuffer::from_host(&s, &l.ffn_norm))?,
                rt: cu(DeviceBuffer::from_host(&s, &l.router.w))?,
                qp_packed: cu(DeviceBuffer::from_host(&s, cq.qp_packed))?,
                qp_scales: cu(DeviceBuffer::from_host(
                    &s,
                    &weightpack::bytes_to_f32_vec(cq.qp_scales)?,
                ))?,
                kp_packed: cu(DeviceBuffer::from_host(&s, cq.kp_packed))?,
                kp_scales: cu(DeviceBuffer::from_host(
                    &s,
                    &weightpack::bytes_to_f32_vec(cq.kp_scales)?,
                ))?,
                vp_packed: cu(DeviceBuffer::from_host(&s, cq.vp_packed))?,
                vp_scales: cu(DeviceBuffer::from_host(
                    &s,
                    &weightpack::bytes_to_f32_vec(cq.vp_scales)?,
                ))?,
                op_packed: cu(DeviceBuffer::from_host(&s, cq.op_packed))?,
                op_scales: cu(DeviceBuffer::from_host(
                    &s,
                    &weightpack::bytes_to_f32_vec(cq.op_scales)?,
                ))?,
                ex_gate_packed: cu(DeviceBuffer::from_host(&s, cq.gate_q_packed))?,
                ex_gate_scales: cu(DeviceBuffer::from_host(
                    &s,
                    &weightpack::bytes_to_f32_vec(cq.gate_q_scales)?,
                ))?,
                ex_up_packed: cu(DeviceBuffer::from_host(&s, cq.up_q_packed))?,
                ex_up_scales: cu(DeviceBuffer::from_host(
                    &s,
                    &weightpack::bytes_to_f32_vec(cq.up_q_scales)?,
                ))?,
                ex_down_packed: cu(DeviceBuffer::from_host(&s, cq.down_q_packed))?,
                ex_down_scales: cu(DeviceBuffer::from_host(
                    &s,
                    &weightpack::bytes_to_f32_vec(cq.down_q_scales)?,
                ))?,
            });
        }
        tracing::info!(
            "  {} layers from weight pack in {:.1}s",
            layers.len(),
            t0.elapsed().as_secs_f64()
        );

        // Build remaining state (same as from_cpu)
        let nl = c.num_layers;
        let na = c.num_experts_per_tok;
        let nh = c.num_heads;
        let nkv = c.num_kv_heads;
        let hd = c.head_dim;
        let max_seq = 4096;
        let vocab = c.vocab_size;

        // RoPE cos/sin
        let theta = c.rope_theta;
        let mut cos = vec![1.0f32; max_seq * hd / 2];
        let mut sin = vec![0.0f32; max_seq * hd / 2];
        for p in 0..max_seq {
            let off = p * hd / 2;
            for i in 0..hd / 2 {
                let freq = 1.0 / theta.powf(2.0 * i as f32 / hd as f32);
                let angle = p as f32 * freq;
                cos[off + i] = angle.cos();
                sin[off + i] = angle.sin();
            }
        }
        let rope_cos = cu(DeviceBuffer::from_host(&s, &cos))?;
        let rope_sin = cu(DeviceBuffer::from_host(&s, &sin))?;
        let kv = CudaContiguousKvCache::new(&s, nl, max_seq, kv_dim, nh)?;

        let scratch = Scratch {
            hidden: cu(DeviceBuffer::<f32>::zeroed(&s, d))?,
            normed: cu(DeviceBuffer::<f32>::zeroed(&s, d))?,
            q: cu(DeviceBuffer::<f32>::zeroed(&s, d))?,
            k: cu(DeviceBuffer::<f32>::zeroed(&s, kv_dim))?,
            v: cu(DeviceBuffer::<f32>::zeroed(&s, kv_dim))?,
            q_tmp: cu(DeviceBuffer::<f32>::zeroed(&s, d))?,
            k_tmp: cu(DeviceBuffer::<f32>::zeroed(&s, kv_dim))?,
            ao: cu(DeviceBuffer::<f32>::zeroed(&s, d))?,
            h_tmp1: cu(DeviceBuffer::<f32>::zeroed(&s, d))?,
            h_tmp2: cu(DeviceBuffer::<f32>::zeroed(&s, d))?,
            ffn_in: cu(DeviceBuffer::<f32>::zeroed(&s, d))?,
            router_out: cu(DeviceBuffer::<f32>::zeroed(&s, ne))?,
            fo: cu(DeviceBuffer::<f32>::zeroed(&s, d))?,
            gb: cu(DeviceBuffer::<f32>::zeroed(&s, mid))?,
            ub: cu(DeviceBuffer::<f32>::zeroed(&s, mid))?,
            gb2: cu(DeviceBuffer::<f32>::zeroed(&s, mid))?,
            db: cu(DeviceBuffer::<f32>::zeroed(&s, d))?,
            logits: cu(DeviceBuffer::<f32>::zeroed(&s, vocab))?,
            topk_idx: cu(DeviceBuffer::<f32>::zeroed(&s, na))?,
            topk_vocab_idx: cu(DeviceBuffer::<f32>::zeroed(&s, 40))?,
            topk_vocab_val: cu(DeviceBuffer::<f32>::zeroed(&s, 40))?,
            topk_w: cu(DeviceBuffer::<f32>::zeroed(&s, na))?,
        };

        let mut m = Self {
            ctx,
            s,
            module,
            emb,
            lm_head,
            final_norm,
            layers,
            d,
            kv_dim,
            ne,
            na,
            mid,
            vocab,
            eps: c.rms_norm_eps,
            norm_topk_prob: c.norm_topk_prob,
            qt,
            scratch,
            nh,
            nkv,
            hd,
            rope_cos,
            rope_sin,
            kv,
            expert_hits: vec![vec![0usize; ne]; nl],
            total_tokens: 0,
        };
        m.capture_decode_graph()?;
        Ok(m)
    }
}

fn validate_weightpack_manifest(
    manifest: &weightpack::WeightPackManifest,
    model: &ferrule_model::OlmoeModel,
    qt: QuantType,
    expected_suffix: &str,
) -> Result<()> {
    if manifest.num_layers != model.layers.len() {
        return Err(Error::Internal(format!(
            "weight pack has {} layers, model expects {}",
            manifest.num_layers,
            model.layers.len()
        )));
    }
    if manifest.quant_suffix != expected_suffix {
        return Err(Error::Internal(format!(
            "weight pack quant suffix {} does not match requested {}",
            manifest.quant_suffix, expected_suffix
        )));
    }
    if manifest.quant_type != format!("{qt:?}") {
        return Err(Error::Internal(format!(
            "weight pack quant type {} does not match requested {qt:?}",
            manifest.quant_type
        )));
    }

    let expected_hash = weightpack::model_config_hash(&model.model_dir)?;
    if manifest.model_config_hash != expected_hash {
        return Err(Error::Internal(format!(
            "weight pack model fingerprint {} does not match current {}",
            manifest.model_config_hash, expected_hash
        )));
    }

    let expected_shapes = weightpack_tensor_shapes(model);
    if !manifest.tensor_shapes.is_empty() && manifest.tensor_shapes != expected_shapes {
        return Err(Error::Internal(
            "weight pack tensor shapes do not match current model config".into(),
        ));
    }

    Ok(())
}

fn weightpack_tensor_shapes(
    model: &ferrule_model::OlmoeModel,
) -> Vec<weightpack::WeightPackTensorInfo> {
    let c = &model.config;
    let d = c.hidden_size;
    let kv = c.kv_dim;
    let mid = c.intermediate_size;
    let mut out = Vec::new();

    for li in 0..c.num_layers {
        let p = format!("model.layers.{li}");
        out.push(tensor_shape(format!("{p}.self_attn.q_proj.weight"), [d, d]));
        out.push(tensor_shape(
            format!("{p}.self_attn.k_proj.weight"),
            [kv, d],
        ));
        out.push(tensor_shape(
            format!("{p}.self_attn.v_proj.weight"),
            [kv, d],
        ));
        out.push(tensor_shape(format!("{p}.self_attn.o_proj.weight"), [d, d]));
        for ei in 0..c.num_experts {
            let ep = format!("{p}.mlp.experts.{ei}");
            out.push(tensor_shape(format!("{ep}.gate_proj.weight"), [mid, d]));
            out.push(tensor_shape(format!("{ep}.up_proj.weight"), [mid, d]));
            out.push(tensor_shape(format!("{ep}.down_proj.weight"), [d, mid]));
        }
    }

    out
}

fn tensor_shape<const N: usize>(
    name: String,
    shape: [usize; N],
) -> weightpack::WeightPackTensorInfo {
    weightpack::WeightPackTensorInfo {
        name,
        shape: shape.to_vec(),
    }
}
