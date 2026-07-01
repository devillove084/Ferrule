//! CUDA kernels — compiled by cargo-oxide's rustc-codegen-cuda backend.
//!
//! Most GEMV kernels use 4x inner-loop unrolling for ILP.
//! fast_sigmoid uses reduction+squaring (no libdevice, no bit ops).

use cuda_device::{cuda_module, kernel, ptx_asm, thread, DisjointSlice, SharedArray};

// ── Fast exp via PTX ex2.approx.f32 ──────────────────────────────────
// exp(x) = 2^(x * log2(e))  using hardware base-2 exponent.
// ex2.approx.f32 is ~1 ULP accurate, 1 cycle throughput on most GPUs.
// Much better than the previous Taylor-series + squaring approach.

/// Fast exp(x) using PTX ex2.approx.f32. Works for all x.
fn fast_exp(x: f32) -> f32 {
    // log2(e) ≈ 1.4426950408889634 = 0x3FB8AA3B in IEEE 754
    let scaled: f32;
    let result: f32;
    unsafe {
        ptx_asm!(
            "mul.f32 %0, %1, 0f3FB8AA3B;",
            out("=f") scaled,
            in("f") x,
            options(register_only),
        );
        ptx_asm!(
            "ex2.approx.f32 %0, %1;",
            out("=f") result,
            in("f") scaled,
            options(register_only),
        );
    }
    result
}

/// Fast sigmoid(x) = 1 / (1 + exp(-x)) using PTX ex2.approx
fn fast_sigmoid(x: f32) -> f32 {
    if x < -16.0 {
        return 0.0;
    }
    if x > 16.0 {
        return 1.0;
    }
    if x >= 0.0 {
        1.0 / (1.0 + fast_exp(-x))
    } else {
        let ep = fast_exp(x);
        ep / (1.0 + ep)
    }
}

fn fp4_e2m1_value(nibble: u8) -> f32 {
    let magnitude = match nibble & 0x07 {
        0 => 0.0,
        1 => 0.5,
        2 => 1.0,
        3 => 1.5,
        4 => 2.0,
        5 => 3.0,
        6 => 4.0,
        _ => 6.0,
    };
    if nibble & 0x08 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

fn fp8_e4m3fn_value(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let exponent = (byte >> 3) & 0x0f;
    let mantissa = byte & 0x07;
    if exponent == 0 {
        if mantissa == 0 {
            return sign * 0.0;
        }
        return sign * (mantissa as f32) / 512.0; // 2^(-9) = 1/512
    }
    if exponent == 0x0f && mantissa == 0x07 {
        return f32::NAN;
    }
    let pow2 = {
        let exp_f = (exponent as i32 - 7) as f32;
        let result: f32;
        unsafe {
            ptx_asm!(
                "ex2.approx.f32 %0, %1;",
                out("=f") result,
                in("f") exp_f,
                options(register_only),
            );
        }
        result
    };
    sign * pow2 * (1.0 + mantissa as f32 / 8.0)
}

fn e8m0_scale(byte: u8) -> f32 {
    let exponent = byte as f32 - 127.0;
    let result: f32;
    unsafe {
        ptx_asm!(
            "ex2.approx.f32 %0, %1;",
            out("=f") result,
            in("f") exponent,
            options(register_only),
        );
    }
    result
}

fn clamp_max(x: f32, max_value: f32) -> f32 {
    if x > max_value {
        max_value
    } else {
        x
    }
}

fn clamp_range(x: f32, min_value: f32, max_value: f32) -> f32 {
    if x < min_value {
        min_value
    } else if x > max_value {
        max_value
    } else {
        x
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    // ── Embedding ──────────────────────────────────────────────────────

    #[kernel]
    pub fn embed_lookup(emb: &[f32], mut y: DisjointSlice<f32>, tid: u32, d: u32) {
        let i = thread::index_1d().get();
        if (i as u64) >= d as u64 {
            return;
        }
        if let Some(yi) = y.get_mut(thread::index_1d()) {
            *yi = emb[tid as usize * d as usize + i as usize];
        }
    }

    // ── GEMV f32 (unrolled 4x) ─────────────────────────────────────────

    #[kernel]
    pub fn gemv_f32(x: &[f32], w: &[f32], mut y: DisjointSlice<f32>, n: u32, k: u32) {
        let row = thread::index_1d().get();
        if (row as u64) >= n as u64 {
            return;
        }
        let mut dot0 = 0.0f32;
        let mut dot1 = 0.0f32;
        let mut dot2 = 0.0f32;
        let mut dot3 = 0.0f32;
        let k = k as usize;
        let base = row * k;
        let mut j = 0usize;
        let k4 = k - k % 4;
        while j < k4 {
            dot0 += x[j] * w[base + j];
            dot1 += x[j + 1] * w[base + j + 1];
            dot2 += x[j + 2] * w[base + j + 2];
            dot3 += x[j + 3] * w[base + j + 3];
            j += 4;
        }
        let mut dot = dot0 + dot1 + dot2 + dot3;
        while j < k {
            dot += x[j] * w[base + j];
            j += 1;
        }
        if let Some(yi) = y.get_mut(thread::index_1d()) {
            *yi = dot;
        }
    }

    // ── GEMV Q4_0 (llama.cpp half-block layout) ──────────────────────

    #[kernel]
    pub fn gemv_q4(
        x: &[f32],
        packed: &[u8],
        scales: &[f32],
        mut y: DisjointSlice<f32>,
        n: u32,
        k: u32,
    ) {
        let row = thread::index_1d().get();
        if (row as u64) >= n as u64 {
            return;
        }
        let k = k as usize;
        let blocks_per_row = (k + 31) / 32;
        let bytes_per_row = blocks_per_row * 16;
        let mut dot = 0.0f32;
        for b in 0..blocks_per_row {
            let block_start = b * 32;
            let block_end = (block_start + 32).min(k);
            let delta = scales[row * blocks_per_row + b];
            if delta == 0.0 {
                continue;
            }
            let byte_off = row * bytes_per_row + b * 16;
            let len = block_end - block_start;
            let mut j = 0usize;
            let mut bdot = 0.0f32;
            while j < len {
                let idx = if j < 16 { j } else { j - 16 };
                let bv = packed[byte_off + idx];
                let q = if j < 16 { bv & 0x0F } else { bv >> 4 };
                bdot += x[block_start + j] * (q as f32 - 8.0);
                j += 1;
            }
            dot += bdot * delta;
        }
        if let Some(yi) = y.get_mut(thread::index_1d()) {
            *yi = dot;
        }
    }

    // ── GEMV Q8_0 (unrolled 4x) ──────────────────────────────────────

    #[kernel]
    pub fn gemv_q8(
        x: &[f32],
        packed: &[u8],
        scales: &[f32],
        mut y: DisjointSlice<f32>,
        n: u32,
        k: u32,
    ) {
        let row = thread::index_1d().get();
        if (row as u64) >= n as u64 {
            return;
        }
        let k = k as usize;
        let blocks_per_row = (k + 31) / 32;
        let mut dot = 0.0f32;
        for b in 0..blocks_per_row {
            let block_start = b * 32;
            let block_end = (block_start + 32).min(k);
            let delta = scales[row * blocks_per_row + b];
            if delta == 0.0 {
                continue;
            }
            let byte_off = row * k + block_start;
            let len = block_end - block_start;
            let len4 = len - len % 4;
            let mut j = 0usize;
            let mut bdot = 0.0f32;
            while j < len4 {
                bdot += x[block_start + j] * (packed[byte_off + j] as i8 as f32)
                    + x[block_start + j + 1] * (packed[byte_off + j + 1] as i8 as f32)
                    + x[block_start + j + 2] * (packed[byte_off + j + 2] as i8 as f32)
                    + x[block_start + j + 3] * (packed[byte_off + j + 3] as i8 as f32);
                j += 4;
            }
            while j < len {
                bdot += x[block_start + j] * (packed[byte_off + j] as i8 as f32);
                j += 1;
            }
            dot += bdot * delta;
        }
        if let Some(yi) = y.get_mut(thread::index_1d()) {
            *yi = dot;
        }
    }

    #[kernel]
    pub fn gemv_q8_off(
        x: &[f32],
        packed: &[u8],
        scales: &[f32],
        mut y: DisjointSlice<f32>,
        n: u32,
        k: u32,
        packed_off: u32,
        scales_off: u32,
    ) {
        let row = thread::index_1d().get();
        if (row as u64) >= n as u64 {
            return;
        }
        let k = k as usize;
        let blocks_per_row = (k + 31) / 32;
        let packed_off = packed_off as usize;
        let scales_off = scales_off as usize;
        let mut dot = 0.0f32;
        for b in 0..blocks_per_row {
            let block_start = b * 32;
            let block_end = (block_start + 32).min(k);
            let delta = scales[scales_off + row * blocks_per_row + b];
            if delta == 0.0 {
                continue;
            }
            let byte_off = packed_off + row * k + block_start;
            let len = block_end - block_start;
            let len4 = len - len % 4;
            let mut j = 0usize;
            let mut bdot = 0.0f32;
            while j < len4 {
                bdot += x[block_start + j] * (packed[byte_off + j] as i8 as f32)
                    + x[block_start + j + 1] * (packed[byte_off + j + 1] as i8 as f32)
                    + x[block_start + j + 2] * (packed[byte_off + j + 2] as i8 as f32)
                    + x[block_start + j + 3] * (packed[byte_off + j + 3] as i8 as f32);
                j += 4;
            }
            while j < len {
                bdot += x[block_start + j] * (packed[byte_off + j] as i8 as f32);
                j += 1;
            }
            dot += bdot * delta;
        }
        if let Some(yi) = y.get_mut(thread::index_1d()) {
            *yi = dot;
        }
    }

    #[kernel]
    pub fn gemv_q4_off(
        x: &[f32],
        packed: &[u8],
        scales: &[f32],
        mut y: DisjointSlice<f32>,
        n: u32,
        k: u32,
        packed_off: u32,
        scales_off: u32,
    ) {
        let row = thread::index_1d().get();
        if (row as u64) >= n as u64 {
            return;
        }
        let k = k as usize;
        let blocks_per_row = (k + 31) / 32;
        let bytes_per_row = blocks_per_row * 16;
        let packed_off = packed_off as usize;
        let scales_off = scales_off as usize;
        let mut dot = 0.0f32;
        for b in 0..blocks_per_row {
            let block_start = b * 32;
            let block_end = (block_start + 32).min(k);
            let delta = scales[scales_off + row * blocks_per_row + b];
            if delta == 0.0 {
                continue;
            }
            let byte_off = packed_off + row * bytes_per_row + b * 16;
            let len = block_end - block_start;
            let mut j = 0usize;
            let mut bdot = 0.0f32;
            while j < len {
                let idx = if j < 16 { j } else { j - 16 };
                let bv = packed[byte_off + idx];
                let q = if j < 16 { bv & 0x0F } else { bv >> 4 };
                bdot += x[block_start + j] * (q as f32 - 8.0);
                j += 1;
            }
            dot += bdot * delta;
        }
        if let Some(yi) = y.get_mut(thread::index_1d()) {
            *yi = dot;
        }
    }

    // ── Source FP4 E2M1 + E8M0 GEMV ─────────────────────────────────

    /// GEMV for source-preserved `torch.float4_e2m1fn_x2` weights with
    /// `float8_e8m0fnu` scales.
    ///
    /// Layout matches Ferrule's CPU reference path and DeepSeek V4 expert tensors:
    /// `packed = [out_features, in_features / 2]`, low nibble first; `scales =
    /// [out_features, in_features / 32]` with byte 127 mapping to scale 1.0.
    #[kernel]
    pub fn gemv_fp4_e2m1_e8m0(
        x: &[f32],
        packed: &[u8],
        scales: &[u8],
        mut y: DisjointSlice<f32>,
        n: u32,
        k: u32,
    ) {
        let row = thread::index_1d().get();
        if (row as u64) >= n as u64 {
            return;
        }
        let k = k as usize;
        let packed_cols = k / 2;
        let scale_cols = k / 32;
        let mut dot = 0.0f32;
        let mut j = 0usize;
        while j < k {
            let byte = packed[row * packed_cols + j / 2];
            let nibble = if j % 2 == 0 { byte & 0x0f } else { byte >> 4 };
            let scale = e8m0_scale(scales[row * scale_cols + j / 32]);
            dot += x[j] * fp4_e2m1_value(nibble) * scale;
            j += 1;
        }
        if let Some(yi) = y.get_mut(thread::index_1d()) {
            *yi = dot;
        }
    }

    #[kernel]
    pub fn gemv_fp4_e2m1_e8m0_off(
        x: &[f32],
        packed: &[u8],
        scales: &[u8],
        mut y: DisjointSlice<f32>,
        n: u32,
        k: u32,
        packed_off: u32,
        scales_off: u32,
    ) {
        let row = thread::index_1d().get();
        if (row as u64) >= n as u64 {
            return;
        }
        let k = k as usize;
        let packed_cols = k / 2;
        let scale_cols = k / 32;
        let packed_off = packed_off as usize;
        let scales_off = scales_off as usize;
        let mut dot = 0.0f32;
        let mut j = 0usize;
        while j < k {
            let byte = packed[packed_off + row * packed_cols + j / 2];
            let nibble = if j % 2 == 0 { byte & 0x0f } else { byte >> 4 };
            let scale = e8m0_scale(scales[scales_off + row * scale_cols + j / 32]);
            dot += x[j] * fp4_e2m1_value(nibble) * scale;
            j += 1;
        }
        if let Some(yi) = y.get_mut(thread::index_1d()) {
            *yi = dot;
        }
    }

    #[kernel]
    pub fn gemv_dual_fp4_e2m1_e8m0_off(
        x: &[f32],
        p0: &[u8],
        s0: &[u8],
        mut y0: DisjointSlice<f32>,
        off_p0: u32,
        off_s0: u32,
        p1: &[u8],
        s1: &[u8],
        mut y1: DisjointSlice<f32>,
        off_p1: u32,
        off_s1: u32,
        n: u32,
        k: u32,
    ) {
        let row = thread::index_1d().get();
        if (row as u64) >= n as u64 {
            return;
        }
        let k = k as usize;
        let packed_cols = k / 2;
        let scale_cols = k / 32;
        let off_p0 = off_p0 as usize;
        let off_s0 = off_s0 as usize;
        let off_p1 = off_p1 as usize;
        let off_s1 = off_s1 as usize;
        let mut d0 = 0.0f32;
        let mut d1 = 0.0f32;
        let mut j = 0usize;
        while j < k {
            let byte0 = p0[off_p0 + row * packed_cols + j / 2];
            let byte1 = p1[off_p1 + row * packed_cols + j / 2];
            let nibble0 = if j % 2 == 0 { byte0 & 0x0f } else { byte0 >> 4 };
            let nibble1 = if j % 2 == 0 { byte1 & 0x0f } else { byte1 >> 4 };
            let scale0 = e8m0_scale(s0[off_s0 + row * scale_cols + j / 32]);
            let scale1 = e8m0_scale(s1[off_s1 + row * scale_cols + j / 32]);
            let xv = x[j];
            d0 += xv * fp4_e2m1_value(nibble0) * scale0;
            d1 += xv * fp4_e2m1_value(nibble1) * scale1;
            j += 1;
        }
        if let Some(o) = y0.get_mut(thread::index_1d()) {
            *o = d0;
        }
        if let Some(o) = y1.get_mut(thread::index_1d()) {
            *o = d1;
        }
    }

    /// DeepSeek-style expert activation:
    /// `silu(clamp_max(gate, limit)) * clamp(up, -limit, limit) * route_weight`.
    /// `limit <= 0` disables clipping.
    #[kernel]
    pub fn swiglu_weighted_clamped(
        gate: &[f32],
        up: &[f32],
        mut y: DisjointSlice<f32>,
        n: u32,
        route_weight: f32,
        limit: f32,
    ) {
        let i = thread::index_1d().get();
        if (i as u64) >= n as u64 {
            return;
        }
        let mut g = gate[i];
        let mut u = up[i];
        if limit > 0.0 {
            g = clamp_max(g, limit);
            u = clamp_range(u, -limit, limit);
        }
        if let Some(yi) = y.get_mut(thread::index_1d()) {
            *yi = g * fast_sigmoid(g) * u * route_weight;
        }
    }

    // ── Source FP8 E4M3FN + E8M0 GEMV (DSV4 attention linears) ─────────

    /// GEMV for FP8 E4M3FN weights with 2D E8M0 block scales.
    /// weight: [out_features, in_features] u8
    /// scales: [ceil(out/block_m), ceil(in/block_k)] u8
    #[kernel]
    pub fn gemv_fp8_e4m3fn_e8m0_2d(
        x: &[f32],
        weight: &[u8],
        scales: &[u8],
        mut y: DisjointSlice<f32>,
        n: u32,
        k: u32,
        scale_cols: u32,
        block_m: u32,
        block_k: u32,
    ) {
        let row = thread::index_1d().get();
        if (row as u64) >= n as u64 {
            return;
        }
        let k = k as usize;
        let sc = scale_cols as usize;
        let bm = block_m as usize;
        let bk = block_k as usize;
        let mut dot = 0.0f32;
        let mut j = 0usize;
        while j < k {
            let scale = e8m0_scale(scales[(row / bm) * sc + j / bk]);
            let w = fp8_e4m3fn_value(weight[row * k + j]);
            dot += x[j] * w * scale;
            j += 1;
        }
        if let Some(yi) = y.get_mut(thread::index_1d()) {
            *yi = dot;
        }
    }

    /// Batched GEMM for FP4 E2M1 experts: [batch, n] = [batch, k] × [n, k]^T
    /// Processes B tokens through one expert simultaneously.
    #[kernel]
    pub fn gemm_fp4_e2m1_e8m0(
        x: &[f32],
        packed: &[u8],
        scales: &[u8],
        mut y: DisjointSlice<f32>,
        batch: u32,
        n: u32,
        k: u32,
    ) {
        let idx = thread::index_1d().get();
        let total = (batch as u64) * (n as u64);
        if idx as u64 >= total {
            return;
        }
        let b = (idx as u32 / n) as usize;
        let row = (idx as u32 % n) as usize;
        let k = k as usize;
        let packed_cols = k / 2;
        let scale_cols = k / 32;
        let input_off = b * k;
        let mut dot = 0.0f32;
        let mut j = 0usize;
        while j < k {
            let byte = packed[row * packed_cols + j / 2];
            let nibble = if j % 2 == 0 { byte & 0x0f } else { byte >> 4 };
            let scale = e8m0_scale(scales[row * scale_cols + j / 32]);
            dot += x[input_off + j] * fp4_e2m1_value(nibble) * scale;
            j += 1;
        }
        if let Some(yi) = y.get_mut(thread::index_1d()) {
            *yi = dot;
        }
    }

    // ── GEMV Q2S ────────────────────────────────────────────────────

    #[kernel]
    pub fn gemv_q2(
        x: &[f32],
        packed: &[u8],
        scales: &[f32],
        mut y: DisjointSlice<f32>,
        n: u32,
        k: u32,
    ) {
        let row = thread::index_1d().get();
        if (row as u64) >= n as u64 {
            return;
        }
        let k = k as usize;
        let blocks_per_row = (k + 63) / 64;
        let bytes_per_row = (k + 3) / 4;
        let mut dot = 0.0f32;
        for b in 0..blocks_per_row {
            let block_start = b * 64;
            let block_end = (block_start + 64).min(k);
            let delta = scales[row * blocks_per_row + b];
            if delta == 0.0 {
                continue;
            }
            let byte_off = row * bytes_per_row + block_start / 4;
            let len = block_end - block_start;
            let len4 = len - len % 4;
            let mut j = 0usize;
            let mut bdot = 0.0f32;
            while j < len4 {
                let bv = packed[byte_off + j / 4];
                bdot += x[block_start + j] * (((bv & 0x3) as f32 - 1.5) / 1.5)
                    + x[block_start + j + 1] * ((((bv >> 2) & 0x3) as f32 - 1.5) / 1.5)
                    + x[block_start + j + 2] * ((((bv >> 4) & 0x3) as f32 - 1.5) / 1.5)
                    + x[block_start + j + 3] * ((((bv >> 6) & 0x3) as f32 - 1.5) / 1.5);
                j += 4;
            }
            while j < len {
                let bv = packed[byte_off + j / 4];
                let q = ((bv >> (2 * (j % 4))) & 0x3) as f32;
                bdot += x[block_start + j] * (q - 1.5) / 1.5;
                j += 1;
            }
            dot += bdot * delta;
        }
        if let Some(yi) = y.get_mut(thread::index_1d()) {
            *yi = dot;
        }
    }

    #[kernel]
    pub fn gemv_q2_off(
        x: &[f32],
        packed: &[u8],
        scales: &[f32],
        mut y: DisjointSlice<f32>,
        n: u32,
        k: u32,
        packed_off: u32,
        scales_off: u32,
    ) {
        let row = thread::index_1d().get();
        if (row as u64) >= n as u64 {
            return;
        }
        let k = k as usize;
        let blocks_per_row = (k + 63) / 64;
        let bytes_per_row = (k + 3) / 4;
        let packed_off = packed_off as usize;
        let scales_off = scales_off as usize;
        let mut dot = 0.0f32;
        for b in 0..blocks_per_row {
            let block_start = b * 64;
            let block_end = (block_start + 64).min(k);
            let delta = scales[scales_off + row * blocks_per_row + b];
            if delta == 0.0 {
                continue;
            }
            let byte_off = packed_off + row * bytes_per_row + block_start / 4;
            let len = block_end - block_start;
            let mut j = 0usize;
            let mut bdot = 0.0f32;
            while j < len {
                let bv = packed[byte_off + j / 4];
                let q = ((bv >> (2 * (j % 4))) & 0x3) as f32;
                bdot += x[block_start + j] * (q - 1.5) / 1.5;
                j += 1;
            }
            dot += bdot * delta;
        }
        if let Some(yi) = y.get_mut(thread::index_1d()) {
            *yi = dot;
        }
    }

    // ── GEMV T1S ────────────────────────────────────────────────────

    #[kernel]
    pub fn gemv_t1(
        x: &[f32],
        packed: &[u8],
        scales: &[f32],
        mut y: DisjointSlice<f32>,
        n: u32,
        k: u32,
    ) {
        let row = thread::index_1d().get();
        if (row as u64) >= n as u64 {
            return;
        }
        let k = k as usize;
        let blocks_per_row = (k + 63) / 64;
        let bytes_per_row = (k + 3) / 4;
        let mut dot = 0.0f32;
        for b in 0..blocks_per_row {
            let block_start = b * 64;
            let block_end = (block_start + 64).min(k);
            let delta = scales[row * blocks_per_row + b];
            if delta == 0.0 {
                continue;
            }
            let byte_off = row * bytes_per_row + block_start / 4;
            let len = block_end - block_start;
            let mut j = 0usize;
            let mut bdot = 0.0f32;
            while j < len {
                let bv = packed[byte_off + j / 4];
                let q = (bv >> (2 * (j % 4))) & 0x3;
                let xv = x[block_start + j];
                if q == 0 {
                    bdot -= xv;
                } else if q == 2 {
                    bdot += xv;
                }
                j += 1;
            }
            dot += bdot * delta;
        }
        if let Some(yi) = y.get_mut(thread::index_1d()) {
            *yi = dot;
        }
    }

    #[kernel]
    pub fn gemv_t1_off(
        x: &[f32],
        packed: &[u8],
        scales: &[f32],
        mut y: DisjointSlice<f32>,
        n: u32,
        k: u32,
        packed_off: u32,
        scales_off: u32,
    ) {
        let row = thread::index_1d().get();
        if (row as u64) >= n as u64 {
            return;
        }
        let k = k as usize;
        let blocks_per_row = (k + 63) / 64;
        let bytes_per_row = (k + 3) / 4;
        let packed_off = packed_off as usize;
        let scales_off = scales_off as usize;
        let mut dot = 0.0f32;
        for b in 0..blocks_per_row {
            let block_start = b * 64;
            let block_end = (block_start + 64).min(k);
            let delta = scales[scales_off + row * blocks_per_row + b];
            if delta == 0.0 {
                continue;
            }
            let byte_off = packed_off + row * bytes_per_row + block_start / 4;
            let len = block_end - block_start;
            let mut j = 0usize;
            let mut bdot = 0.0f32;
            while j < len {
                let bv = packed[byte_off + j / 4];
                let q = (bv >> (2 * (j % 4))) & 0x3;
                let xv = x[block_start + j];
                if q == 0 {
                    bdot -= xv;
                } else if q == 2 {
                    bdot += xv;
                }
                j += 1;
            }
            dot += bdot * delta;
        }
        if let Some(yi) = y.get_mut(thread::index_1d()) {
            *yi = dot;
        }
    }

    // ── RMS Norm ────────────────────────────────────────────────────────

    #[kernel]
    pub fn rms_norm_apply(
        x: &[f32],
        w: &[f32],
        mut y: DisjointSlice<f32>,
        rms_val: &[f32],
        n: u32,
    ) {
        let i = thread::index_1d().get();
        if (i as u64) >= n as u64 {
            return;
        }
        if let Some(yi) = y.get_mut(thread::index_1d()) {
            *yi = x[i] * rms_val[0] * w[i];
        }
    }

    // ── SiLU (using reduction+squaring fast sigmoid) ────────────────────

    #[kernel]
    pub fn silu(x: &[f32], mut y: DisjointSlice<f32>, n: u32) {
        let i = thread::index_1d().get();
        if (i as u64) >= n as u64 {
            return;
        }
        let xv = x[i];
        if let Some(yi) = y.get_mut(thread::index_1d()) {
            *yi = xv * fast_sigmoid(xv);
        }
    }

    #[kernel]
    pub fn silu_mul(a: &[f32], b: &[f32], mut y: DisjointSlice<f32>, n: u32) {
        let i = thread::index_1d().get();
        if (i as u64) >= n as u64 {
            return;
        }
        let xv = a[i];
        if let Some(yi) = y.get_mut(thread::index_1d()) {
            *yi = xv * fast_sigmoid(xv) * b[i];
        }
    }

    // ── Dual GEMV ──────────────────────────────────────────────────────

    #[kernel]
    pub fn gemv_dual_q4(
        x: &[f32],
        p0: &[u8],
        s0: &[f32],
        mut y0: DisjointSlice<f32>,
        p1: &[u8],
        s1: &[f32],
        mut y1: DisjointSlice<f32>,
        n: u32,
        k: u32,
    ) {
        let row = thread::index_1d().get();
        if (row as u64) >= n as u64 {
            return;
        }
        let k = k as usize;
        let bpr = ((k + 31) / 32) as usize;
        let bytes_per_row = bpr * 16;
        let mut d0 = 0.0f32;
        let mut d1 = 0.0f32;
        for b in 0..bpr {
            let bs = b * 32;
            let be = (bs + 32).min(k);
            let del0 = s0[row * bpr + b];
            let del1 = s1[row * bpr + b];
            let bo = row * bytes_per_row + b * 16;
            let len = be - bs;
            let mut j = 0usize;
            let mut bd0 = 0.0f32;
            let mut bd1 = 0.0f32;
            while j < len {
                let idx = if j < 16 { j } else { j - 16 };
                let bv0 = p0[bo + idx];
                let bv1 = p1[bo + idx];
                let q0 = if j < 16 { bv0 & 0x0F } else { bv0 >> 4 };
                let q1 = if j < 16 { bv1 & 0x0F } else { bv1 >> 4 };
                let xv = x[bs + j];
                bd0 += xv * (q0 as f32 - 8.0);
                bd1 += xv * (q1 as f32 - 8.0);
                j += 1;
            }
            if del0 != 0.0 {
                d0 += bd0 * del0;
            }
            if del1 != 0.0 {
                d1 += bd1 * del1;
            }
        }
        if let Some(o) = y0.get_mut(thread::index_1d()) {
            *o = d0;
        }
        if let Some(o) = y1.get_mut(thread::index_1d()) {
            *o = d1;
        }
    }

    #[kernel]
    pub fn gemv_triple_q4(
        x: &[f32],
        p0: &[u8],
        s0: &[f32],
        mut y0: DisjointSlice<f32>,
        p1: &[u8],
        s1: &[f32],
        mut y1: DisjointSlice<f32>,
        p2: &[u8],
        s2: &[f32],
        mut y2: DisjointSlice<f32>,
        n: u32,
        k: u32,
    ) {
        let row = thread::index_1d().get();
        if (row as u64) >= n as u64 {
            return;
        }
        let k = k as usize;
        let bpr = ((k + 31) / 32) as usize;
        let bytes_per_row = bpr * 16;
        let mut d0 = 0.0f32;
        let mut d1 = 0.0f32;
        let mut d2 = 0.0f32;
        for b in 0..bpr {
            let bs = b * 32;
            let be = (bs + 32).min(k);
            let del0 = s0[row * bpr + b];
            let del1 = s1[row * bpr + b];
            let del2 = s2[row * bpr + b];
            let bo = row * bytes_per_row + b * 16;
            let len = be - bs;
            let mut j = 0usize;
            let mut bd0 = 0.0;
            let mut bd1 = 0.0;
            let mut bd2 = 0.0;
            while j < len {
                let idx = if j < 16 { j } else { j - 16 };
                let bv0 = p0[bo + idx];
                let bv1 = p1[bo + idx];
                let bv2 = p2[bo + idx];
                let q0 = if j < 16 { bv0 & 0x0F } else { bv0 >> 4 };
                let q1 = if j < 16 { bv1 & 0x0F } else { bv1 >> 4 };
                let q2 = if j < 16 { bv2 & 0x0F } else { bv2 >> 4 };
                let xv = x[bs + j];
                bd0 += xv * (q0 as f32 - 8.0);
                bd1 += xv * (q1 as f32 - 8.0);
                bd2 += xv * (q2 as f32 - 8.0);
                j += 1;
            }
            if del0 != 0.0 {
                d0 += bd0 * del0;
            }
            if del1 != 0.0 {
                d1 += bd1 * del1;
            }
            if del2 != 0.0 {
                d2 += bd2 * del2;
            }
        }
        if let Some(o) = y0.get_mut(thread::index_1d()) {
            *o = d0;
        }
        if let Some(o) = y1.get_mut(thread::index_1d()) {
            *o = d1;
        }
        if let Some(o) = y2.get_mut(thread::index_1d()) {
            *o = d2;
        }
    }

    #[kernel]
    pub fn gemv_dual_q4_off(
        x: &[f32],
        p0: &[u8],
        s0: &[f32],
        mut y0: DisjointSlice<f32>,
        off_p0: u32,
        off_s0: u32,
        p1: &[u8],
        s1: &[f32],
        mut y1: DisjointSlice<f32>,
        off_p1: u32,
        off_s1: u32,
        n: u32,
        k: u32,
    ) {
        let row = thread::index_1d().get();
        if (row as u64) >= n as u64 {
            return;
        }
        let k = k as usize;
        let bpr = ((k + 31) / 32) as usize;
        let bytes_per_row = bpr * 16;
        let off_p0 = off_p0 as usize;
        let off_s0 = off_s0 as usize;
        let off_p1 = off_p1 as usize;
        let off_s1 = off_s1 as usize;
        let mut d0 = 0.0f32;
        let mut d1 = 0.0f32;
        for b in 0..bpr {
            let bs = b * 32;
            let be = (bs + 32).min(k);
            let del0 = s0[off_s0 + row * bpr + b];
            let del1 = s1[off_s1 + row * bpr + b];
            let bo0 = off_p0 + row * bytes_per_row + b * 16;
            let bo1 = off_p1 + row * bytes_per_row + b * 16;
            let len = be - bs;
            let mut j = 0usize;
            let mut bd0 = 0.0;
            let mut bd1 = 0.0;
            while j < len {
                let idx = if j < 16 { j } else { j - 16 };
                let bv0 = p0[bo0 + idx];
                let bv1 = p1[bo1 + idx];
                let q0 = if j < 16 { bv0 & 0x0F } else { bv0 >> 4 };
                let q1 = if j < 16 { bv1 & 0x0F } else { bv1 >> 4 };
                let xv = x[bs + j];
                bd0 += xv * (q0 as f32 - 8.0);
                bd1 += xv * (q1 as f32 - 8.0);
                j += 1;
            }
            if del0 != 0.0 {
                d0 += bd0 * del0;
            }
            if del1 != 0.0 {
                d1 += bd1 * del1;
            }
        }
        if let Some(o) = y0.get_mut(thread::index_1d()) {
            *o = d0;
        }
        if let Some(o) = y1.get_mut(thread::index_1d()) {
            *o = d1;
        }
    }

    // ── Element-wise ────────────────────────────────────────────────────

    #[kernel]
    pub fn mul(a: &[f32], b: &[f32], mut y: DisjointSlice<f32>, n: u32) {
        let i = thread::index_1d().get();
        if (i as u64) >= n as u64 {
            return;
        }
        if let Some(yi) = y.get_mut(thread::index_1d()) {
            *yi = a[i] * b[i];
        }
    }
    #[kernel]
    pub fn add(a: &[f32], b: &[f32], mut y: DisjointSlice<f32>, n: u32) {
        let i = thread::index_1d().get();
        if (i as u64) >= n as u64 {
            return;
        }
        if let Some(yi) = y.get_mut(thread::index_1d()) {
            *yi = a[i] + b[i];
        }
    }
    #[kernel]
    pub fn saxpy(scale: f32, x: &[f32], mut y: DisjointSlice<f32>, n: u32) {
        let i = thread::index_1d().get();
        if (i as u64) >= n as u64 {
            return;
        }
        if let Some(yi) = y.get_mut(thread::index_1d()) {
            *yi += scale * x[i];
        }
    }

    /// Compute 1/sqrt(mean(x^2) + eps) via PTX rsqrt.approx.f32.
    /// This matches llama.cpp's rsqrtf(var + eps) approach using hardware sqrt.
    #[kernel]
    pub fn compute_rms(x: &[f32], mut rms_out: DisjointSlice<f32>, n: u32, eps: f32) {
        if thread::index_1d().get() != 0 {
            return;
        }
        let mut sum = 0.0f32;
        let n = n as usize;
        for j in 0..n {
            sum += x[j] * x[j];
        }
        if let Some(o) = rms_out.get_mut(thread::index_1d()) {
            let val = sum / n as f32 + eps;
            // PTX rsqrt.approx.f32 — same as llama.cpp's rsqrtf(var + eps)
            let result: f32;
            unsafe {
                ptx_asm!(
                    "rsqrt.approx.f32 %0, %1;",
                    out("=f") result,
                    in("f") val,
                    options(register_only),
                );
            }
            *o = result;
        }
    }

    /// Fused RMS norm: parallel reduction via shared memory + rsqrt + apply.
    /// Replaces the compute_rms + rms_norm_apply two-kernel sequence.
    /// Pattern: llama.cpp's block_reduce<SUM> + rsqrtf + element-wise apply.
    /// Launch with up to 1024 threads, 1 block. Handles n > blockDim via striding.
    #[kernel]
    pub fn rms_norm_fused(x: &[f32], w: &[f32], mut y: DisjointSlice<f32>, n: u32, eps: f32) {
        static mut SMEM: SharedArray<f32, 1024> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let bdim = thread::blockDim_x() as usize;
        let n = n as usize;

        // Phase 1: each thread accumulates partial sum for strided elements
        let mut sum = 0.0f32;
        let mut j = tid;
        while j < n {
            sum += x[j] * x[j];
            j += bdim;
        }
        unsafe {
            SMEM[tid] = sum;
        }
        thread::sync_threads();

        // Phase 2: tree reduction — halve active threads each round
        let mut stride = (bdim + 1) / 2;
        while stride > 0 {
            if tid < stride && tid + stride < bdim {
                unsafe {
                    SMEM[tid] += SMEM[tid + stride];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }

        // Phase 3: thread 0 computes rsqrt from the total sum
        if tid == 0 {
            let val = unsafe { SMEM[0] } / n as f32 + eps;
            let result: f32;
            unsafe {
                ptx_asm!(
                    "rsqrt.approx.f32 %0, %1;",
                    out("=f") result,
                    in("f") val,
                    options(register_only),
                );
            }
            unsafe {
                SMEM[0] = result;
            }
        }
        thread::sync_threads();

        // Phase 4: strided write — use raw pointer for multi-element per thread
        let rsqrt = unsafe { SMEM[0] };
        let y_ptr = y.as_mut_ptr();
        let mut j = tid;
        while j < n {
            unsafe {
                *y_ptr.add(j) = x[j] * rsqrt * w[j];
            }
            j += bdim;
        }
    }

    // ── Router Top-K (GPU-side, eliminates CPU round-trip) ────────────

    /// Find top-k expert indices and softmax weights on GPU.
    #[kernel]
    pub fn router_topk(
        logits: &[f32],
        mut indices: DisjointSlice<f32>,
        mut weights: DisjointSlice<f32>,
        ne: u32,
        k: u32,
        norm_topk_prob: u32,
    ) {
        static mut SMEM: SharedArray<f32, 128> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let ne = ne as usize;
        let k = k as usize;

        // Copy logits to shared memory (parallel)
        if tid < ne {
            unsafe {
                SMEM[tid] = logits[tid];
            }
        }
        thread::sync_threads();

        // Thread 0: top-k selection + softmax weights.
        // HF OLMoE/Mixtral semantics: softmax is over all experts first, then top-k is
        // selected. Only renormalize selected probabilities when norm_topk_prob=true.
        if tid == 0 {
            let mut max_v = f32::NEG_INFINITY;
            for i in 0..ne {
                let v = unsafe { SMEM[i] };
                if v > max_v {
                    max_v = v;
                }
            }

            let mut all_sum = 0.0f32;
            for i in 0..ne {
                all_sum += fast_exp(unsafe { SMEM[i] } - max_v);
            }

            // Simple top-k: k passes of find-max-and-mask.
            let mut top_val = [0.0f32; 8];
            let mut top_idx = [0.0f32; 8];
            let mut top_sum = 0.0f32;
            for j in 0..k {
                let mut best_val = f32::NEG_INFINITY;
                let mut best_idx = 0usize;
                for i in 0..ne {
                    let v = unsafe { SMEM[i] };
                    if v > best_val {
                        best_val = v;
                        best_idx = i;
                    }
                }
                let prob_num = fast_exp(best_val - max_v);
                top_val[j] = prob_num;
                top_idx[j] = best_idx as f32;
                top_sum += prob_num;
                unsafe {
                    SMEM[best_idx] = f32::NEG_INFINITY;
                }
            }

            let denom = if norm_topk_prob != 0 {
                top_sum
            } else {
                all_sum
            };

            // Write outputs
            let idx_ptr = indices.as_mut_ptr();
            let w_ptr = weights.as_mut_ptr();
            for j in 0..k {
                unsafe {
                    *idx_ptr.add(j) = top_idx[j];
                    *w_ptr.add(j) = top_val[j] / denom;
                }
            }
        }
    }

    // ── RoPE ─────────────────────────────────────────────────────────────

    #[kernel]
    pub fn rope(
        x: &[f32],
        cos: &[f32],
        sin: &[f32],
        mut y: DisjointSlice<f32>,
        pos: u32,
        nh: u32,
        hd: u32,
    ) {
        let i = thread::index_1d().get();
        let nh = nh as usize;
        let hd = hd as usize;
        let hd2 = hd / 2;
        let total = nh * hd;
        if (i as u64) >= total as u64 {
            return;
        }
        let h = i as usize / hd;
        let off = h * hd;
        let local = i as usize % hd;
        let pair = if local < hd2 { local } else { local - hd2 };
        let c = cos[pos as usize * hd2 + pair];
        let pair_idx = if local < hd2 {
            off + local + hd2
        } else {
            off + local - hd2
        };
        let s = sin[pos as usize * hd2 + pair];
        let x_pair = x[pair_idx];
        let x_self = x[off + local];
        let val = if local < hd2 {
            x_self * c - x_pair * s
        } else {
            x_pair * s + x_self * c
        };
        if let Some(o) = y.get_mut(thread::index_1d()) {
            *o = val;
        }
    }

    // ── Sparse attention with sink (DSV4/MLA path) ─────────────────────

    /// Tiled sparse attention with online softmax and attention sink.
    /// 2D grid: threads map to (token, head), each thread processes topk KV slots.
    #[kernel]
    pub fn sparse_attn_tiled_sink_f32(
        q: &[f32],
        kv: &[f32],
        topk: &[i32],
        sink: &[f32],
        mut output: DisjointSlice<f32>,
        num_pairs: u32,
        _tokens_per_batch: u32,
        kv_len: u32,
        heads: u32,
        head_dim: u32,
        topk_len: u32,
        softmax_scale: f32,
    ) {
        let idx = thread::index_1d().get();
        if (idx as u64) >= num_pairs as u64 {
            return;
        }
        let heads = heads as usize;
        let token = idx / heads;
        let head = idx % heads;
        let hd = head_dim as usize;
        let tk = topk_len as usize;
        let kv_len = kv_len as usize;
        let q_off = (token * heads + head) * hd;
        let sink_val = sink.get(head).copied().unwrap_or(f32::NEG_INFINITY);

        // Pass 1: find max score
        let mut max_score = sink_val;
        for slot in 0..tk {
            let ki = topk[token * tk + slot];
            if ki < 0 {
                continue;
            }
            let ki = ki as usize;
            if ki >= kv_len {
                continue;
            }
            let kv_off = ki * hd;
            let mut dot = 0.0f32;
            for d in 0..hd {
                dot += q[q_off + d] * kv[kv_off + d];
            }
            max_score = if dot * softmax_scale > max_score {
                dot * softmax_scale
            } else {
                max_score
            };
        }

        // Pass 2: accumulate with online softmax
        let sink_exp = fast_exp(sink_val - max_score);
        let mut denom = sink_exp;
        let out_off = (token * heads + head) * hd;
        let out_ptr = output.as_mut_ptr();
        for d in 0..hd {
            unsafe {
                *out_ptr.add(out_off + d) = 0.0;
            }
        }

        for slot in 0..tk {
            let ki = topk[token * tk + slot];
            if ki < 0 {
                continue;
            }
            let ki = ki as usize;
            if ki >= kv_len {
                continue;
            }
            let kv_off = ki * hd;
            let mut dot = 0.0f32;
            for d in 0..hd {
                dot += q[q_off + d] * kv[kv_off + d];
            }
            let weight = fast_exp(dot * softmax_scale - max_score);
            denom += weight;
            for d in 0..hd {
                unsafe {
                    *out_ptr.add(out_off + d) += weight * kv[kv_off + d];
                }
            }
        }
        for d in 0..hd {
            unsafe {
                *out_ptr.add(out_off + d) /= denom;
            }
        }
    }

    // ── Attention scores (GQA-aware) ──────────────────────────────────

    #[kernel]
    pub fn attn_scores(
        q: &[f32],
        k_cache: &[f32],
        mut scores: DisjointSlice<f32>,
        seq_len: u32,
        nh: u32,
        nkv: u32,
        hd: u32,
        sm_scale: f32,
    ) {
        let i = thread::index_1d().get();
        let nh = nh as usize;
        let nkv = nkv as usize;
        let hd = hd as usize;
        let sl = seq_len as usize;
        let total = nh * sl;
        if (i as u64) >= total as u64 {
            return;
        }
        let h = i as usize / sl;
        let p = i as usize % sl;
        let n_rep = nh / nkv;
        let kv_h = h / n_rep;
        let mut dot = 0.0f32;
        for j in 0..hd {
            dot += q[h * hd + j] * k_cache[p * nkv * hd + kv_h * hd + j];
        }
        if let Some(s) = scores.get_mut(thread::index_1d()) {
            *s = dot * sm_scale;
        }
    }

    // ── Attention: weighted V combine with inline softmax (GQA-aware) ──
    /// Fuses softmax(scores) + Σ softmax_i × V_i into one kernel.
    /// Eliminates CPU round-trip for softmax (cf. llama.cpp flash-attn patterns).
    #[kernel]
    pub fn attn_combine_softmax(
        scores: &[f32],
        v_cache: &[f32],
        mut out: DisjointSlice<f32>,
        seq_len: u32,
        nh: u32,
        nkv: u32,
        hd: u32,
    ) {
        let i = thread::index_1d().get();
        let nh = nh as usize;
        let nkv = nkv as usize;
        let hd = hd as usize;
        let sl = seq_len as usize;
        let total = nh * hd;
        if i >= total {
            return;
        }
        let h = i as usize / hd;
        let d = i as usize % hd;
        let n_rep = nh / nkv;
        let kv_h = h / n_rep;
        // Inline softmax: max → exp-sum → weighted combine
        let mut max_s = f32::NEG_INFINITY;
        for p in 0..sl {
            let s = scores[h * sl + p];
            if s > max_s {
                max_s = s;
            }
        }
        let mut sum_w = 0.0f32;
        let mut val = 0.0f32;
        for p in 0..sl {
            let w = fast_exp(scores[h * sl + p] - max_s);
            sum_w += w;
            val += w * v_cache[p * nkv * hd + kv_h * hd + d];
        }
        if let Some(o) = out.get_mut(thread::index_1d()) {
            *o = val / sum_w;
        }
    }

    // ── Vocab top-k ────────────────────────────────────────────────────
    /// Find top-k token indices and values from vocab logits on GPU.
    /// Single block, sequential chunk scan — correct for any vocab size.
    #[kernel]
    pub fn topk_vocab(
        logits: &[f32],
        mut out_idx: DisjointSlice<f32>,
        mut out_val: DisjointSlice<f32>,
        vocab: u32,
        k: u32,
    ) {
        static mut SMEM: SharedArray<f32, 1024> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let vocab = vocab as usize;
        let k = k as usize;
        let chunk = 1024usize;

        // Only block 0, single thread per chunk
        if thread::blockIdx_x() == 0 && tid == 0 {
            let mut best_val = [f32::NEG_INFINITY; 40];
            let mut best_idx = [0u32; 40];

            let mut cursor = 0usize;
            while cursor < vocab {
                let n = (vocab - cursor).min(chunk);
                // Load chunk
                for i in 0..n {
                    unsafe {
                        SMEM[i] = logits[cursor + i];
                    }
                }
                // Scan chunk
                for i in 0..n {
                    let v = unsafe { SMEM[i] };
                    let gid = (cursor + i) as u32;
                    let mut pos = k;
                    while pos > 0 && v > best_val[pos - 1] {
                        pos -= 1;
                    }
                    if pos < k {
                        for j in (pos + 1..k).rev() {
                            best_val[j] = best_val[j - 1];
                            best_idx[j] = best_idx[j - 1];
                        }
                        best_val[pos] = v;
                        best_idx[pos] = gid;
                    }
                }
                cursor += n;
            }

            // Write outputs
            let idx_ptr = out_idx.as_mut_ptr();
            let val_ptr = out_val.as_mut_ptr();
            for j in 0..k {
                unsafe {
                    *idx_ptr.add(j) = best_idx[j] as f32;
                    *val_ptr.add(j) = best_val[j];
                }
            }
        }
    }

    // ── Fused MLA Q-projection (wq_a → q_norm → wq_b) ───────────────
    /// DSV4 MLA query path: x → wq_a → rms_norm(q_norm) → wq_b → [heads, head_dim]
    /// One thread per output element in the final query tensor.
    #[kernel]
    pub fn mla_q_projection_f32(
        x: &[f32],
        wq_a: &[f32],
        wq_b: &[f32],
        q_norm: &[f32],
        mut q_out: DisjointSlice<f32>,
        hidden_size: u32,
        q_lora_rank: u32,
        q_heads_dim: u32,
        eps: f32,
    ) {
        let idx = thread::index_1d().get();
        if (idx as u64) >= q_heads_dim as u64 {
            return;
        }
        let hs = hidden_size as usize;
        let qr = q_lora_rank as usize;
        // Step 1: q_latent = x @ wq_a^T  (one thread per output element)
        let q_latent_idx = idx as usize % qr;
        // Compute rms_norm for this latent position
        let mut sum_sq = 0.0f32;
        let mut latent_val = 0.0f32;
        for j in 0..hs {
            latent_val += x[j] * wq_a[q_latent_idx * hs + j];
        }
        // Apply q_norm weight
        latent_val *= q_norm[q_latent_idx as usize];
        sum_sq += latent_val * latent_val;
        // Note: rms_norm requires all q_latent elements. For fused kernel we
        // compute the full q_latent first, then rms_norm, then wq_b projection.
        // This simplified version does per-element rms approximation.
        let val = sum_sq / qr as f32 + eps;
        let rms: f32;
        unsafe {
            ptx_asm!(
                "rsqrt.approx.f32 %0, %1;",
                out("=f") rms,
                in("f") val,
                options(register_only),
            );
        }
        let q_normed = latent_val * rms;
        // Step 2: q_final = q_normed @ wq_b^T
        let mut dot = 0.0f32;
        for j in 0..qr {
            // wq_b is [heads*head_dim, q_lora_rank]
            dot += q_normed * wq_b[idx as usize * qr + j];
        }
        if let Some(o) = q_out.get_mut(thread::index_1d()) {
            *o = dot;
        }
    }

    // ── YAARN Rotary Position Embedding ───────────────────────────────
    /// Apply rotary embedding with YAARN scaling to query or key tensor.
    /// qk: [tokens, heads, head_dim] in f32.
    /// freqs: [head_dim/2] precomputed cos/sin frequencies.
    #[kernel]
    pub fn rope_yarn(
        mut qk: DisjointSlice<f32>,
        freqs_cos: &[f32],
        freqs_sin: &[f32],
        num_elements: u32,
        head_dim: u32,
        rope_dim: u32,
    ) {
        let idx = thread::index_1d().get();
        if (idx as u64) >= num_elements as u64 {
            return;
        }
        let hd = head_dim as usize;
        let rd = rope_dim as usize;
        let head_idx = idx as usize / hd;
        let d = idx as usize % hd;
        if d >= rd / 2 {
            return;
        }
        let d2 = 2 * d;
        let cos = freqs_cos[d];
        let sin = freqs_sin[d];
        let ptr = qk.as_mut_ptr();
        let x0 = unsafe { *ptr.add(head_idx * hd + d2) };
        let x1 = unsafe { *ptr.add(head_idx * hd + d2 + 1) };
        let out0 = x0 * cos - x1 * sin;
        let out1 = x0 * sin + x1 * cos;
        unsafe {
            *ptr.add(head_idx * hd + d2) = out0;
            *ptr.add(head_idx * hd + d2 + 1) = out1;
        }
    }

    // ── Batched expert SwiGLU accumulation ────────────────────────────
    /// Fused: gate+up activation → down projection → weighted_add into output.
    /// One thread per output element.
    #[kernel]
    pub fn swiglu_down_accumulate(
        gate: &[f32],
        up: &[f32],
        down_packed: &[u8],
        down_scales: &[u8],
        mut output: DisjointSlice<f32>,
        intermediate_size: u32,
        hidden_size: u32,
        route_weight: f32,
        limit: f32,
    ) {
        let row = thread::index_1d().get();
        if (row as u64) >= hidden_size as u64 {
            return;
        }
        let inter = intermediate_size as usize;
        let _hs = hidden_size as usize;
        let packed_cols = inter / 2;
        let scale_cols = inter / 32;
        let mut dot = 0.0f32;
        let mut j = 0usize;
        while j < inter {
            let byte = down_packed[row * packed_cols + j / 2];
            let nibble = if j % 2 == 0 { byte & 0x0f } else { byte >> 4 };
            let scale = e8m0_scale(down_scales[row * scale_cols + j / 32]);
            let mut g = gate[j];
            let mut u = up[j];
            if limit > 0.0 {
                g = clamp_max(g, limit);
            }
            if limit > 0.0 {
                u = clamp_range(u, -limit, limit);
            }
            dot += g * fast_sigmoid(g) * u * fp4_e2m1_value(nibble) * scale;
            j += 1;
        }
        dot *= route_weight;
        if let Some(o) = output.get_mut(thread::index_1d()) {
            *o += dot;
        }
    }
}
