//! GPU OLMoE forward pass — quantized weights, scratch pool, zero CPU roundtrips.
#![allow(unsafe_code)]

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use ferrule_core::{Error, Result};
use ferrule_quant::{f16_to_f32, QuantType};
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;

fn cu<T, E: std::fmt::Debug>(r: std::result::Result<T, E>) -> Result<T> {
    r.map_err(|e| Error::Internal(format!("CUDA {e:?}")))
}

use crate::qcache::{self, QCacheData};

// ── Kernel dispatch (selects Q4_0 vs Q2S at runtime) ─────────────────

use crate::kernels::kernels::LoadedModule;
use cuda_core::stream::CudaStream;

fn gemv_quant(
    m: &LoadedModule,
    s: &CudaStream,
    qt: QuantType,
    cfg: LaunchConfig,
    x: &DeviceBuffer<f32>,
    packed: &DeviceBuffer<u8>,
    scales: &DeviceBuffer<f32>,
    y: &mut DeviceBuffer<f32>,
    k: u32,
) -> Result<()> {
    match qt {
        QuantType::Q4_0 => cu(m.gemv_q4(s, cfg, x, packed, scales, y, k)),
        QuantType::Q2S => cu(m.gemv_q2(s, cfg, x, packed, scales, y, k)),
        QuantType::T1S => cu(m.gemv_t1(s, cfg, x, packed, scales, y, k)),
    }
}

fn gemv_quant_off(
    m: &LoadedModule,
    s: &CudaStream,
    qt: QuantType,
    cfg: LaunchConfig,
    x: &DeviceBuffer<f32>,
    packed: &DeviceBuffer<u8>,
    scales: &DeviceBuffer<f32>,
    y: &mut DeviceBuffer<f32>,
    k: u32,
    packed_off: u32,
    scales_off: u32,
) -> Result<()> {
    match qt {
        QuantType::Q4_0 => {
            cu(m.gemv_q4_off(s, cfg, x, packed, scales, y, k, packed_off, scales_off))
        }
        QuantType::Q2S => {
            cu(m.gemv_q2_off(s, cfg, x, packed, scales, y, k, packed_off, scales_off))
        }
        QuantType::T1S => {
            cu(m.gemv_t1_off(s, cfg, x, packed, scales, y, k, packed_off, scales_off))
        }
    }
}

// ── Device probe ──────────────────────────────────────────────────────

pub fn cuda_probe() -> Result<()> {
    let ctx = cu(CudaContext::new(0))?;
    let name = cu(ctx.device_name())?;
    cu(ctx.bind_to_thread())?;
    let mut free: usize = 0;
    let mut total: usize = 0;
    unsafe {
        cuda_bindings::cuMemGetInfo_v2(&mut free, &mut total);
    }
    println!(
        "  Device: {name}\n  Memory: {:.1} GB free / {:.1} GB total",
        free as f64 / 1e9,
        total as f64 / 1e9
    );
    Ok(())
}

// ── Standalone GEMV (benchmark) ───────────────────────────────────────

pub fn cuda_gemv(x: &[f32], w: &[f32], out_f: usize) -> Result<Vec<f32>> {
    let ctx = cu(CudaContext::new(0))?;
    cu(ctx.bind_to_thread())?;
    let module = cu(crate::kernels::kernels::load(&ctx))?;
    let s = ctx.default_stream();
    let xd = cu(DeviceBuffer::from_host(&s, x))?;
    let wd = cu(DeviceBuffer::from_host(&s, w))?;
    let mut yd = cu(DeviceBuffer::<f32>::zeroed(&s, out_f))?;
    cu(module.gemv_f32(
        &s,
        LaunchConfig::for_num_elems(out_f as u32),
        &xd,
        &wd,
        &mut yd,
        x.len() as u32,
    ))?;
    cu(yd.to_host_vec(&s))
}

// ── Quantized weight storage ──────────────────────────────────────────

/// Per-layer weights stored in quantized format on GPU.
/// K/V projections use kv_dim outputs (GQA), Q uses d.
#[allow(dead_code)]
struct QLayer {
    // Attention norm weights (f32 — small, precision matters)
    an: DeviceBuffer<f32>,  // d
    qn: DeviceBuffer<f32>,  // d (Q norm)
    kn: DeviceBuffer<f32>,  // kv_dim (K norm)
    fn_: DeviceBuffer<f32>, // d

    // Quantized attention projection weights
    qp_packed: DeviceBuffer<u8>,
    qp_scales: DeviceBuffer<f32>,
    kp_packed: DeviceBuffer<u8>,
    kp_scales: DeviceBuffer<f32>,
    vp_packed: DeviceBuffer<u8>,
    vp_scales: DeviceBuffer<f32>,
    op_packed: DeviceBuffer<u8>,
    op_scales: DeviceBuffer<f32>,

    // Router (f32 — small)
    rt: DeviceBuffer<f32>, // ne × d

    // Expert weights, concatenated across all experts
    ex_gate_packed: DeviceBuffer<u8>,
    ex_gate_scales: DeviceBuffer<f32>,
    ex_up_packed: DeviceBuffer<u8>,
    ex_up_scales: DeviceBuffer<f32>,
    ex_down_packed: DeviceBuffer<u8>,
    ex_down_scales: DeviceBuffer<f32>,
}

pub struct GpuOlmoeModel {
    ctx: Arc<CudaContext>,
    s: Arc<cuda_core::stream::CudaStream>,
    module: crate::kernels::kernels::LoadedModule,
    emb: DeviceBuffer<f32>,
    lm_head: DeviceBuffer<f32>,
    final_norm: DeviceBuffer<f32>,
    layers: Vec<QLayer>,
    d: usize,
    kv_dim: usize,
    ne: usize,
    na: usize,
    mid: usize,
    vocab: usize,
    eps: f32,
    qt: QuantType,
    scratch: Scratch,
    // ── KV cache (GPU) + RoPE (GPU) ──
    nh: usize,
    nkv: usize,
    hd: usize,
    #[allow(dead_code)]
    max_seq: usize,
    k_cache: Vec<DeviceBuffer<f32>>, // [num_layers][max_seq × kv_dim]
    v_cache: Vec<DeviceBuffer<f32>>,
    rope_cos: DeviceBuffer<f32>,
    rope_sin: DeviceBuffer<f32>,
    cur_seq: usize,
    scores_buf: DeviceBuffer<f32>, // [nh × max_seq]
}

/// Pre-allocated GPU buffers — allocated once, reused every token.
struct Scratch {
    hidden: DeviceBuffer<f32>,     // d
    normed: DeviceBuffer<f32>,     // d
    q: DeviceBuffer<f32>,          // d
    k: DeviceBuffer<f32>,          // kv_dim
    v: DeviceBuffer<f32>,          // kv_dim
    q_tmp: DeviceBuffer<f32>,      // d
    k_tmp: DeviceBuffer<f32>,      // kv_dim
    ao: DeviceBuffer<f32>,         // d
    h_tmp1: DeviceBuffer<f32>,     // d
    h_tmp2: DeviceBuffer<f32>,     // d (zero-filled, used as zero operand)
    ffn_in: DeviceBuffer<f32>,     // d
    router_out: DeviceBuffer<f32>, // ne
    fo: DeviceBuffer<f32>,         // d
    gb: DeviceBuffer<f32>,         // mid
    ub: DeviceBuffer<f32>,         // mid
    gb2: DeviceBuffer<f32>,        // mid
    db: DeviceBuffer<f32>,         // d
    rms_buf: DeviceBuffer<f32>,    // 1
    logits: DeviceBuffer<f32>,     // vocab
}

impl GpuOlmoeModel {
    pub fn from_cpu(model: &ferrule_model::OlmoeModel, qt: QuantType) -> Result<Self> {
        let ctx = cu(CudaContext::new(0))?;
        cu(ctx.bind_to_thread())?;
        let module = cu(crate::kernels::kernels::load(&ctx))?;
        let s = ctx.default_stream();
        let c = &model.config;
        let d = c.hidden_size;
        let kv_dim = c.kv_dim;
        let ne = c.num_experts;
        let mid = c.intermediate_size;

        eprintln!(
            "Uploading & quantizing weights to GPU (GQA: nh={}, nkv={})...",
            c.num_heads, c.num_kv_heads
        );
        let t0 = std::time::Instant::now();

        // Embedding stays f32 (lookup, not matmul)
        let emb = cu(DeviceBuffer::from_host(&s, &model.embed))?;
        let lm_head = cu(DeviceBuffer::from_host(&s, &model.lm_head))?;
        let final_norm = cu(DeviceBuffer::from_host(&s, &model.final_norm))?;
        eprintln!(
            "  embed ({:.1} MB f32) in {:.1}s",
            model.embed.len() as f64 * 4.0 / 1e6,
            t0.elapsed().as_secs_f64()
        );

        struct QData {
            qp: ferrule_quant::QMatrix,
            kp: ferrule_quant::QMatrix,
            vp: ferrule_quant::QMatrix,
            op: ferrule_quant::QMatrix,
            gate_q_packed: Vec<u8>,
            gate_q_scales: Vec<u16>,
            up_q_packed: Vec<u8>,
            up_q_scales: Vec<u16>,
            down_q_packed: Vec<u8>,
            down_q_scales: Vec<u16>,
        }

        let cache_file = qcache::cache_path(&model.model_dir, &format!("{:?}", qt).to_lowercase());
        if cache_file.exists() {
            eprintln!("Loading from quantized cache: {}", cache_file.display());
            let cached = qcache::read_cache(&cache_file)?;
            eprintln!(
                "  {} layers from cache in {:.1}s",
                cached.len(),
                t0.elapsed().as_secs_f64()
            );
            // Upload cached layers directly to GPU
            let mut layers = Vec::new();
            for (li, c) in cached.iter().enumerate() {
                let l = &model.layers[li];
                layers.push(QLayer {
                    an: cu(DeviceBuffer::from_host(&s, &l.attn_norm))?,
                    qn: cu(DeviceBuffer::from_host(&s, &l.attn.q_norm))?,
                    kn: cu(DeviceBuffer::from_host(&s, &l.attn.k_norm))?,
                    fn_: cu(DeviceBuffer::from_host(&s, &l.ffn_norm))?,
                    rt: cu(DeviceBuffer::from_host(&s, &l.router.w))?,
                    qp_packed: cu(DeviceBuffer::from_host(&s, &c.qp_packed))?,
                    qp_scales: cu(DeviceBuffer::from_host(
                        &s,
                        qcache::bytes_to_f32_slice(&c.qp_scales),
                    ))?,
                    kp_packed: cu(DeviceBuffer::from_host(&s, &c.kp_packed))?,
                    kp_scales: cu(DeviceBuffer::from_host(
                        &s,
                        qcache::bytes_to_f32_slice(&c.kp_scales),
                    ))?,
                    vp_packed: cu(DeviceBuffer::from_host(&s, &c.vp_packed))?,
                    vp_scales: cu(DeviceBuffer::from_host(
                        &s,
                        qcache::bytes_to_f32_slice(&c.vp_scales),
                    ))?,
                    op_packed: cu(DeviceBuffer::from_host(&s, &c.op_packed))?,
                    op_scales: cu(DeviceBuffer::from_host(
                        &s,
                        qcache::bytes_to_f32_slice(&c.op_scales),
                    ))?,
                    ex_gate_packed: cu(DeviceBuffer::from_host(&s, &c.gate_q_packed))?,
                    ex_gate_scales: cu(DeviceBuffer::from_host(
                        &s,
                        qcache::bytes_to_f32_slice(&c.gate_q_scales),
                    ))?,
                    ex_up_packed: cu(DeviceBuffer::from_host(&s, &c.up_q_packed))?,
                    ex_up_scales: cu(DeviceBuffer::from_host(
                        &s,
                        qcache::bytes_to_f32_slice(&c.up_q_scales),
                    ))?,
                    ex_down_packed: cu(DeviceBuffer::from_host(&s, &c.down_q_packed))?,
                    ex_down_scales: cu(DeviceBuffer::from_host(
                        &s,
                        qcache::bytes_to_f32_slice(&c.down_q_scales),
                    ))?,
                });
            }
            eprintln!("  Done ({:.1}s total).", t0.elapsed().as_secs_f64());

        // Free VMEM check
        let mut free: usize = 0;
        let mut total: usize = 0;
        unsafe {
            cuda_bindings::cuMemGetInfo_v2(&mut free, &mut total);
        }
        eprintln!(
            "  GPU free: {:.1} MB / {:.1} MB",
            free as f64 / 1e6,
            total as f64 / 1e6
        );

        eprintln!("Allocating scratch buffers...");
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
            rms_buf: cu(DeviceBuffer::<f32>::zeroed(&s, 1))?,
            logits: cu(DeviceBuffer::<f32>::zeroed(&s, c.vocab_size))?,
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
        let scores_buf = cu(DeviceBuffer::<f32>::zeroed(&s, nh * max_seq))?;
        // KV cache: store kv_dim elements per position (not d)
        let k_cache: Vec<_> = (0..c.num_layers)
            .map(|_| cu(DeviceBuffer::<f32>::zeroed(&s, max_seq * kv_dim)))
            .collect::<Result<_>>()?;
        let v_cache: Vec<_> = (0..c.num_layers)
            .map(|_| cu(DeviceBuffer::<f32>::zeroed(&s, max_seq * kv_dim)))
            .collect::<Result<_>>()?;

        return Ok(Self {
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
            qt,
            scratch,
            nh,
            nkv,
            hd,
            max_seq,
            k_cache,
            v_cache,
            rope_cos,
            rope_sin,
            cur_seq: 0,
            scores_buf,
        })
    }       }
        eprintln!(
            "Quantizing & uploading {} layers (pipelined)...",
            model.layers.len()
        );
        let tq = std::time::Instant::now();
        let (tx, rx) = std::sync::mpsc::sync_channel::<(usize, QData, QCacheData)>(2);
        let layers: Vec<QLayer> =
            std::thread::scope(|scope| -> ferrule_core::Result<Vec<QLayer>> {
                // Spawn quantization thread (rayon-parallel across layers, streamed via channel)
                scope.spawn(move || {
                    model.layers.par_iter().enumerate().for_each(|(li, l)| {
                        let qp = ferrule_quant::QMatrix::quantize(&l.attn.q_proj.w, d, d, qt);
                        let kp = ferrule_quant::QMatrix::quantize(
                            &l.attn.k_proj.w,
                            l.attn.k_proj.out_f,
                            d,
                            qt,
                        );
                        let vp = ferrule_quant::QMatrix::quantize(
                            &l.attn.v_proj.w,
                            l.attn.v_proj.out_f,
                            d,
                            qt,
                        );
                        let op = ferrule_quant::QMatrix::quantize(&l.attn.o_proj.w, d, d, qt);
                        let mut gate_q_packed = Vec::new();
                        let mut gate_q_scales = Vec::new();
                        let mut up_q_packed = Vec::new();
                        let mut up_q_scales = Vec::new();
                        let mut down_q_packed = Vec::new();
                        let mut down_q_scales = Vec::new();
                        for e in &l.experts {
                            let gq = ferrule_quant::QMatrix::quantize(&e.gate.w, mid, d, qt);
                            gate_q_packed.extend_from_slice(&gq.packed);
                            gate_q_scales.extend_from_slice(&gq.scales);
                            let uq = ferrule_quant::QMatrix::quantize(&e.up.w, mid, d, qt);
                            up_q_packed.extend_from_slice(&uq.packed);
                            up_q_scales.extend_from_slice(&uq.scales);
                            let dq = ferrule_quant::QMatrix::quantize(&e.down.w, d, mid, qt);
                            down_q_packed.extend_from_slice(&dq.packed);
                            down_q_scales.extend_from_slice(&dq.scales);
                        }
                        let cache = QCacheData::from_qmatrix(
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
                let mut pending: std::collections::BTreeMap<usize, (QData, QCacheData)> =
                    std::collections::BTreeMap::new();
                let mut next_layer = 0usize;
                let mut layers = Vec::new();
                let to_f32 =
                    |raw: &[u16]| -> Vec<f32> { raw.iter().map(|&b| f16_to_f32(b)).collect() };
                let mut cache_layers: Vec<QCacheData> = Vec::new();
                for (li, q, c) in rx {
                    pending.insert(li, (q, c));
                    while let Some((q, c)) = pending.remove(&next_layer) {
                        let l = &model.layers[next_layer];
                        let tl = std::time::Instant::now();
                        layers.push(QLayer {
                            an: cu(DeviceBuffer::from_host(&s, &l.attn_norm))?,
                            qn: cu(DeviceBuffer::from_host(&s, &l.attn.q_norm))?,
                            kn: cu(DeviceBuffer::from_host(&s, &l.attn.k_norm))?,
                            fn_: cu(DeviceBuffer::from_host(&s, &l.ffn_norm))?,
                            rt: cu(DeviceBuffer::from_host(&s, &l.router.w))?,
                            qp_packed: cu(DeviceBuffer::from_host(&s, &q.qp.packed))?,
                            qp_scales: cu(DeviceBuffer::from_host(&s, &to_f32(&q.qp.scales)))?,
                            kp_packed: cu(DeviceBuffer::from_host(&s, &q.kp.packed))?,
                            kp_scales: cu(DeviceBuffer::from_host(&s, &to_f32(&q.kp.scales)))?,
                            vp_packed: cu(DeviceBuffer::from_host(&s, &q.vp.packed))?,
                            vp_scales: cu(DeviceBuffer::from_host(&s, &to_f32(&q.vp.scales)))?,
                            op_packed: cu(DeviceBuffer::from_host(&s, &q.op.packed))?,
                            op_scales: cu(DeviceBuffer::from_host(&s, &to_f32(&q.op.scales)))?,
                            ex_gate_packed: cu(DeviceBuffer::from_host(&s, &q.gate_q_packed))?,
                            ex_gate_scales: cu(DeviceBuffer::from_host(
                                &s,
                                &to_f32(&q.gate_q_scales),
                            ))?,
                            ex_up_packed: cu(DeviceBuffer::from_host(&s, &q.up_q_packed))?,
                            ex_up_scales: cu(DeviceBuffer::from_host(&s, &to_f32(&q.up_q_scales)))?,
                            ex_down_packed: cu(DeviceBuffer::from_host(&s, &q.down_q_packed))?,
                            ex_down_scales: cu(DeviceBuffer::from_host(
                                &s,
                                &to_f32(&q.down_q_scales),
                            ))?,
                        });
                        eprintln!(
                            "  layer {next_layer:>2} in {:.1}s",
                            tl.elapsed().as_secs_f64()
                        );
                        cache_layers.push(c);
                        next_layer += 1;
                    }
                }
                if !cache_layers.is_empty() {
                    eprintln!("Writing quantized cache: {}", cache_file.display());
                    qcache::write_cache(&cache_file, &cache_layers)?;
                }
                Ok(layers)
            })?;

        eprintln!("  Done ({:.1}s total).", t0.elapsed().as_secs_f64());

        // Free VMEM check
        let mut free: usize = 0;
        let mut total: usize = 0;
        unsafe {
            cuda_bindings::cuMemGetInfo_v2(&mut free, &mut total);
        }
        eprintln!(
            "  GPU free: {:.1} MB / {:.1} MB",
            free as f64 / 1e6,
            total as f64 / 1e6
        );

        eprintln!("Allocating scratch buffers...");
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
            rms_buf: cu(DeviceBuffer::<f32>::zeroed(&s, 1))?,
            logits: cu(DeviceBuffer::<f32>::zeroed(&s, c.vocab_size))?,
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
        let scores_buf = cu(DeviceBuffer::<f32>::zeroed(&s, nh * max_seq))?;
        // KV cache: store kv_dim elements per position (not d)
        let k_cache: Vec<_> = (0..c.num_layers)
            .map(|_| cu(DeviceBuffer::<f32>::zeroed(&s, max_seq * kv_dim)))
            .collect::<Result<_>>()?;
        let v_cache: Vec<_> = (0..c.num_layers)
            .map(|_| cu(DeviceBuffer::<f32>::zeroed(&s, max_seq * kv_dim)))
            .collect::<Result<_>>()?;

        Ok(Self {
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
            qt,
            scratch,
            nh,
            nkv,
            hd,
            max_seq,
            k_cache,
            v_cache,
            rope_cos,
            rope_sin,
            cur_seq: 0,
            scores_buf,
        })

    pub fn forward(&mut self, tid: u32) -> Result<Vec<f32>> {
        cu(self.ctx.bind_to_thread())?;
        let m = &self.module;
        let s = &self.s;
        let d = self.d;
        let kv_dim = self.kv_dim;
        let nkv = self.nkv;
        let cfg = |n: usize| LaunchConfig::for_num_elems(n as u32);

        let Scratch {
            hidden,
            normed,
            q,
            k,
            v,
            q_tmp,
            k_tmp,
            ao,
            h_tmp1,
            h_tmp2,
            ffn_in,
            router_out,
            fo,
            gb,
            ub,
            gb2,
            db,
            rms_buf,
            logits,
        } = &mut self.scratch;

        // ── Embedding lookup ──
        cu(m.embed_lookup(s, cfg(d), &self.emb, hidden, tid, d as u32))?;

        // Expert stride constants for quantized packed/scales offsets
        let mid = self.mid;
        let (gate_bytes_per_exp, gate_scales_per_exp, down_bytes_per_exp, down_scales_per_exp) =
            match self.qt {
                QuantType::Q4_0 => (
                    mid * d / 2,
                    mid * d.div_ceil(32),
                    d * mid / 2,
                    d * mid.div_ceil(32),
                ),
                QuantType::Q2S => (
                    mid * (d + 3) / 4,
                    mid * d.div_ceil(64),
                    d * (mid + 3) / 4,
                    d * mid.div_ceil(64),
                ),
                QuantType::T1S => (
                    mid * (d + 3) / 4,
                    mid * d.div_ceil(64),
                    d * (mid + 3) / 4,
                    d * mid.div_ceil(64),
                ),
            };

        for (li, layer) in self.layers.iter().enumerate() {
            let qt = self.qt;
            // ── Attention norm ──
            cu(m.compute_rms(s, cfg(1), hidden, rms_buf, d as u32, self.eps))?;
            cu(m.rms_norm_apply(s, cfg(d), hidden, &layer.an, normed, rms_buf))?;

            // ── Q/K/V projections (GQA: k/v have kv_dim outputs, q has d) ──
            if qt == QuantType::Q4_0 && kv_dim == d {
                // MHA: all three outputs same size, use fused triple kernel
                cu(m.gemv_triple_q4(
                    s,
                    cfg(d),
                    normed,
                    &layer.qp_packed,
                    &layer.qp_scales,
                    q,
                    &layer.kp_packed,
                    &layer.kp_scales,
                    k,
                    &layer.vp_packed,
                    &layer.vp_scales,
                    v,
                    d as u32,
                ))?;
            } else {
                // Q projection (d outputs)
                gemv_quant(
                    m,
                    s,
                    qt,
                    cfg(d),
                    normed,
                    &layer.qp_packed,
                    &layer.qp_scales,
                    q,
                    d as u32,
                )?;
                // K projection (kv_dim outputs)
                gemv_quant(
                    m,
                    s,
                    qt,
                    cfg(kv_dim),
                    normed,
                    &layer.kp_packed,
                    &layer.kp_scales,
                    k,
                    d as u32,
                )?;
                // V projection (kv_dim outputs)
                gemv_quant(
                    m,
                    s,
                    qt,
                    cfg(kv_dim),
                    normed,
                    &layer.vp_packed,
                    &layer.vp_scales,
                    v,
                    d as u32,
                )?;
            }

            // ── Q/K head norms (GQA: k norm has kv_dim elements) ──
            cu(m.compute_rms(s, cfg(1), q, rms_buf, d as u32, self.eps))?;
            cu(m.rms_norm_apply(s, cfg(d), q, &layer.qn, q_tmp, rms_buf))?;
            cu(m.compute_rms(s, cfg(1), k, rms_buf, kv_dim as u32, self.eps))?;
            cu(m.rms_norm_apply(s, cfg(kv_dim), k, &layer.kn, k_tmp, rms_buf))?;

            // ── RoPE on GPU: q_tmp→q (nh heads), k_tmp→k (nkv heads) ──
            let pos = self.cur_seq;
            cu(m.rope(
                s,
                cfg(d),
                q_tmp,
                &self.rope_cos,
                &self.rope_sin,
                q,
                pos as u32,
                self.nh as u32,
                self.hd as u32,
            ))?;
            cu(m.rope(
                s,
                cfg(kv_dim),
                k_tmp,
                &self.rope_cos,
                &self.rope_sin,
                k,
                pos as u32,
                nkv as u32,
                self.hd as u32,
            ))?;

            // Copy K_rot, V into GPU KV cache (kv_dim elements per position)
            let offset = (pos * kv_dim * 4) as usize;
            let k_size = kv_dim * 4;
            unsafe {
                cuda_bindings::cuMemcpyDtoD_v2(
                    self.k_cache[li].cu_deviceptr() + offset as u64,
                    k.cu_deviceptr(),
                    k_size,
                );
                cuda_bindings::cuMemcpyDtoD_v2(
                    self.v_cache[li].cu_deviceptr() + offset as u64,
                    v.cu_deviceptr(),
                    k_size,
                );
            }
            let seq_len = pos + 1;

            // GPU: attention scores (GQA-aware) — stays on GPU
            let sm_scale = 1.0 / (self.hd as f32).sqrt();
            cu(m.attn_scores(
                s,
                LaunchConfig::for_num_elems((self.nh * seq_len) as u32),
                q,
                &self.k_cache[li],
                &mut self.scores_buf,
                seq_len as u32,
                self.nh as u32,
                nkv as u32,
                self.hd as u32,
                sm_scale,
            ))?;

            // GPU: inline softmax + V combine (fused, no CPU round-trip, GQA-aware)
            cu(m.attn_combine_softmax(
                s,
                cfg(d),
                &self.scores_buf,
                &self.v_cache[li],
                ao,
                seq_len as u32,
                self.nh as u32,
                nkv as u32,
                self.hd as u32,
            ))?;

            // ── O projection ──
            gemv_quant(
                m,
                s,
                qt,
                cfg(d),
                ao,
                &layer.op_packed,
                &layer.op_scales,
                q,
                d as u32,
            )?;
            cu(m.add(s, cfg(d), hidden, q, h_tmp1))?;
            cu(m.add(s, cfg(d), h_tmp1, h_tmp2, hidden))?;

            // ── FFN norm ──
            cu(m.compute_rms(s, cfg(1), hidden, rms_buf, d as u32, self.eps))?;
            cu(m.rms_norm_apply(s, cfg(d), hidden, &layer.fn_, ffn_in, rms_buf))?;

            // ── Router (f32, small) ──
            cu(m.gemv_f32(s, cfg(self.ne), ffn_in, &layer.rt, router_out, d as u32))?;
            let rl = cu(router_out.to_host_vec(s))?;

            // ── Top-k expert selection (CPU) ──
            let mut idx: Vec<(usize, f32)> = rl.iter().copied().enumerate().collect();
            idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            idx.truncate(self.na);
            let max_l = idx.iter().fold(f32::NEG_INFINITY, |a, (_, v)| a.max(*v));
            let exps: Vec<f32> = idx.iter().map(|(_, v)| (v - max_l).exp()).collect();
            let sum: f32 = exps.iter().sum();

            // ── Expert FFN ──
            cu(m.mul(s, cfg(d), hidden, h_tmp2, fo))?; // fo = 0

            for (k, &(eid, _)) in idx.iter().enumerate() {
                let w = exps[k] / sum;
                let gate_packed_off = eid as u32 * gate_bytes_per_exp as u32;
                let gate_scales_off = eid as u32 * gate_scales_per_exp as u32;
                let down_packed_off = eid as u32 * down_bytes_per_exp as u32;
                let down_scales_off = eid as u32 * down_scales_per_exp as u32;

                // gate + up (fused dual for Q4_0)
                if qt == QuantType::Q4_0 {
                    cu(m.gemv_dual_q4_off(
                        s,
                        cfg(self.mid),
                        ffn_in,
                        &layer.ex_gate_packed,
                        &layer.ex_gate_scales,
                        gb,
                        gate_packed_off as u32,
                        gate_scales_off as u32,
                        &layer.ex_up_packed,
                        &layer.ex_up_scales,
                        ub,
                        gate_packed_off as u32,
                        gate_scales_off as u32,
                        d as u32,
                    ))?;
                } else {
                    gemv_quant_off(
                        m,
                        s,
                        qt,
                        cfg(self.mid),
                        ffn_in,
                        &layer.ex_gate_packed,
                        &layer.ex_gate_scales,
                        gb,
                        d as u32,
                        gate_packed_off as u32,
                        gate_scales_off as u32,
                    )?;
                    gemv_quant_off(
                        m,
                        s,
                        qt,
                        cfg(self.mid),
                        ffn_in,
                        &layer.ex_up_packed,
                        &layer.ex_up_scales,
                        ub,
                        d as u32,
                        gate_packed_off as u32,
                        gate_scales_off as u32,
                    )?;
                }

                // SiLU(gate) * up (fused silu+mul → saves 1 launch)
                cu(m.silu_mul(s, cfg(self.mid), gb, ub, gb2))?;

                // down
                gemv_quant_off(
                    m,
                    s,
                    qt,
                    cfg(d),
                    gb2,
                    &layer.ex_down_packed,
                    &layer.ex_down_scales,
                    db,
                    self.mid as u32,
                    down_packed_off as u32,
                    down_scales_off as u32,
                )?;

                // fo += w * db
                let w_buf = cu(DeviceBuffer::from_host(s, &[w]))?;
                cu(m.saxpy(s, cfg(d), &w_buf, db, fo))?;
            }

            // ── Residual ──
            cu(m.add(s, cfg(d), hidden, fo, h_tmp1))?;
            cu(m.add(s, cfg(d), h_tmp1, h_tmp2, hidden))?;
        }

        self.cur_seq += 1;

        // ── Final layer norm (model.norm.weight) ──
        cu(m.compute_rms(s, cfg(1), hidden, rms_buf, d as u32, self.eps))?;
        cu(m.rms_norm_apply(s, cfg(d), hidden, &self.final_norm, normed, rms_buf))?;

        // ── lm_head on GPU (uses normed hidden state) ──
        let vocab = self.vocab;
        cu(m.gemv_f32(
            s,
            LaunchConfig::for_num_elems(vocab as u32),
            normed,
            &self.lm_head,
            logits,
            d as u32,
        ))?;
        let result = cu(logits.to_host_vec(s));
        if let Ok(ref top) = result {
            let mut top5: Vec<(usize, f32)> = top.iter().copied().enumerate().collect();
            top5.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            top5.truncate(8);
            eprintln!(
                "GPU top8 ids: {:?}",
                top5.iter().map(|(i, _)| i).collect::<Vec<_>>()
            );
        }
        result
    }
}
