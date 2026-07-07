//! CUDA kernels — compiled by cargo-oxide's rustc-codegen-cuda backend.
//!
//! Most GEMV kernels use 4x inner-loop unrolling for ILP.
//! Device math uses Rust float intrinsics so cuda-oxide lowers them through
//! libdevice/NVVM, matching the upstream examples and avoiding arch-specific
//! inline-PTX JIT issues on newer GPUs.

use cuda_device::{
    atomic::{AtomicOrdering, DeviceAtomicF32},
    cuda_module, kernel, ptx_asm, thread,
    wmma::mma_m16n8k64_mxf4_f32_e2m1_e2m1_b0_t0_b0_t0,
    DisjointSlice, SharedArray,
};

const LN_2_F32: f32 = core::f32::consts::LN_2;

/// Device exp(x). cuda-oxide lowers this to libdevice (`__nv_expf`) on GPU.
fn fast_exp(x: f32) -> f32 {
    libm::expf(x)
}

/// Fast reciprocal sqrt. This stays as inline PTX because libm::sqrtf uses
/// host-arch inline asm before cuda-oxide can route it to libdevice on aarch64.
fn fast_rsqrt(x: f32) -> f32 {
    let result: f32;
    unsafe {
        ptx_asm!(
            "rsqrt.approx.f32 %0, %1;",
            out("=f") result,
            in("f") x,
            options(register_only),
        );
    }
    result
}

/// Fast sigmoid(x) = 1 / (1 + exp(-x)).
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

#[inline(always)]
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

#[inline(always)]
fn fp8_e4m3fn_value(byte: u8) -> f32 {
    let sign_bits = ((byte & 0x80) as u32) << 24;
    let exponent = (byte >> 3) & 0x0f;
    let mantissa = byte & 0x07;
    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign_bits);
        }
        let value = (mantissa as f32) * (1.0 / 512.0); // E4M3FN subnormal step: 2^-9.
        return if sign_bits != 0 { -value } else { value };
    }
    if exponent == 0x0f && mantissa == 0x07 {
        return f32::NAN;
    }
    let f32_exponent = (exponent as u32) + 120; // exponent - bias(7) + f32 bias(127).
    f32::from_bits(sign_bits | (f32_exponent << 23) | ((mantissa as u32) << 20))
}

#[inline(always)]
fn e8m0_scale(byte: u8) -> f32 {
    // E8M0 scales are powers of two: scale = 2^(byte - 127).
    // Build the f32 exponent bits directly instead of calling device expf in
    // every FP4/FP8 inner-loop iteration.
    let bits = if byte == 0 {
        1u32 << 22 // 2^-127, matching the previous expf-based behavior.
    } else {
        (byte as u32) << 23
    };
    f32::from_bits(bits)
}

fn pow2f(exp: f32) -> f32 {
    fast_exp(exp * LN_2_F32)
}

fn rounded_power_of_two_scale(amax: f32, quant_max: f32) -> f32 {
    pow2f(libm::ceilf(libm::log2f(amax / quant_max)))
}

fn quantize_fp8_e4m3fn_to_f32(value: f32) -> f32 {
    if !value.is_finite() || value == 0.0 {
        return value;
    }
    let sign = if value < 0.0 { -1.0 } else { 1.0 };
    let abs_value = if value < 0.0 { -value } else { value };
    let magnitude = if abs_value > 448.0 { 448.0 } else { abs_value };
    sign * nearest_fp8_e4m3fn_positive(magnitude)
}

fn e8m0_scale_byte_for_amax(amax: f32, quant_max: f32) -> u8 {
    if !amax.is_finite() || amax <= 0.0 || !quant_max.is_finite() || quant_max <= 0.0 {
        return 127;
    }
    let exp = libm::ceilf(libm::log2f(amax / quant_max)) as i32;
    let byte = exp + 127;
    if byte < 0 {
        0
    } else if byte > 255 {
        255
    } else {
        byte as u8
    }
}

fn quantize_fp4_e2m1_nibble(value: f32) -> u8 {
    if !value.is_finite() || value == 0.0 {
        return 0;
    }
    let sign = if value < 0.0 { 0x08 } else { 0x00 };
    let magnitude = if value < 0.0 { -value } else { value };
    let clamped = if magnitude > 6.0 { 6.0 } else { magnitude };
    let mut best_idx = 0u8;
    let mut best_err = clamped;
    let mut idx = 1u8;
    while idx < 8 {
        let candidate = fp4_e2m1_value(idx);
        let err = if candidate > clamped {
            candidate - clamped
        } else {
            clamped - candidate
        };
        if err < best_err {
            best_err = err;
            best_idx = idx;
        }
        idx += 1;
    }
    sign | best_idx
}

fn nearest_fp8_e4m3fn_positive(magnitude: f32) -> f32 {
    let mut best = nearest_fp8_subnormal_positive(magnitude);
    let mut best_err = if best > magnitude {
        best - magnitude
    } else {
        magnitude - best
    };
    let exp_floor = libm::floorf(libm::log2f(magnitude)) as i32;
    let mut exp = exp_floor - 1;
    while exp <= exp_floor + 1 {
        if (-6..=8).contains(&exp) {
            let scale = pow2f(exp as f32);
            let mut mantissa = libm::roundf((magnitude / scale - 1.0) * 8.0) as i32;
            let mut candidate_exp = exp;
            if mantissa >= 0 {
                if mantissa > 7 {
                    candidate_exp += 1;
                    mantissa = 0;
                }
                if candidate_exp > 8 {
                    candidate_exp = 8;
                    mantissa = 6;
                }
                if candidate_exp == 8 && mantissa > 6 {
                    mantissa = 6;
                }
                let candidate = pow2f(candidate_exp as f32) * (1.0 + mantissa as f32 / 8.0);
                let err = if candidate > magnitude {
                    candidate - magnitude
                } else {
                    magnitude - candidate
                };
                if err < best_err {
                    best = candidate;
                    best_err = err;
                }
            }
        }
        exp += 1;
    }
    best
}

fn nearest_fp8_subnormal_positive(magnitude: f32) -> f32 {
    let step = pow2f(-9.0);
    let mantissa = libm::roundf(magnitude / step).clamp(0.0, 7.0);
    mantissa * step
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

    // ── GEMV f32/BF16 (unrolled 4x) ───────────────────────────────────

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

    /// Grouped matvec for block-diagonal weight layout.
    ///
    /// `context` is `[o_groups * group_in]` (the full concatenated context).
    /// `weight` is `[output_latent_dim, group_in]` f32, stored row-major:
    ///   row `r = group * o_lora_rank + rank` only touches `context[group * group_in ..]`.
    /// `output` is `[output_latent_dim]` = `[o_groups * o_lora_rank]`.
    ///
    /// One thread per output row.
    #[kernel]
    pub fn grouped_matvec_f32(
        context: &[f32],
        weight: &[f32],
        mut output: DisjointSlice<f32>,
        output_latent_dim: u32,
        group_in: u32,
        o_lora_rank: u32,
    ) {
        let row = thread::index_1d().get();
        if (row as u64) >= output_latent_dim as u64 {
            return;
        }
        let group_in = group_in as usize;
        let o_lora_rank = o_lora_rank as usize;
        let group = row / o_lora_rank;
        let context_start = group * group_in;
        let weight_start = row * group_in;
        let mut dot0 = 0.0f32;
        let mut dot1 = 0.0f32;
        let mut dot2 = 0.0f32;
        let mut dot3 = 0.0f32;
        let mut j = 0usize;
        let k4 = group_in - group_in % 4;
        while j < k4 {
            dot0 += context[context_start + j] * weight[weight_start + j];
            dot1 += context[context_start + j + 1] * weight[weight_start + j + 1];
            dot2 += context[context_start + j + 2] * weight[weight_start + j + 2];
            dot3 += context[context_start + j + 3] * weight[weight_start + j + 3];
            j += 4;
        }
        let mut dot = dot0 + dot1 + dot2 + dot3;
        while j < group_in {
            dot += context[context_start + j] * weight[weight_start + j];
            j += 1;
        }
        if let Some(yi) = output.get_mut(thread::index_1d()) {
            *yi = dot;
        }
    }

    fn f32_bytes_to_f32(b0: u8, b1: u8, b2: u8, b3: u8) -> f32 {
        f32::from_bits((b0 as u32) | ((b1 as u32) << 8) | ((b2 as u32) << 16) | ((b3 as u32) << 24))
    }

    fn bf16_pair_to_f32(lo: u8, hi: u8) -> f32 {
        f32::from_bits((((hi as u32) << 8) | lo as u32) << 16)
    }

    #[kernel]
    pub fn gemv_f32_bytes(x: &[f32], w: &[u8], mut y: DisjointSlice<f32>, n: u32, k: u32) {
        let row = thread::index_1d().get();
        if (row as u64) >= n as u64 {
            return;
        }
        let k = k as usize;
        let base = row * k * 4;
        let mut dot0 = 0.0f32;
        let mut dot1 = 0.0f32;
        let mut dot2 = 0.0f32;
        let mut dot3 = 0.0f32;
        let mut j = 0usize;
        let k4 = k - k % 4;
        while j < k4 {
            let o0 = base + 4 * j;
            let o1 = base + 4 * (j + 1);
            let o2 = base + 4 * (j + 2);
            let o3 = base + 4 * (j + 3);
            dot0 += x[j] * f32_bytes_to_f32(w[o0], w[o0 + 1], w[o0 + 2], w[o0 + 3]);
            dot1 += x[j + 1] * f32_bytes_to_f32(w[o1], w[o1 + 1], w[o1 + 2], w[o1 + 3]);
            dot2 += x[j + 2] * f32_bytes_to_f32(w[o2], w[o2 + 1], w[o2 + 2], w[o2 + 3]);
            dot3 += x[j + 3] * f32_bytes_to_f32(w[o3], w[o3 + 1], w[o3 + 2], w[o3 + 3]);
            j += 4;
        }
        let mut dot = dot0 + dot1 + dot2 + dot3;
        while j < k {
            let off = base + 4 * j;
            dot += x[j] * f32_bytes_to_f32(w[off], w[off + 1], w[off + 2], w[off + 3]);
            j += 1;
        }
        if let Some(yi) = y.get_mut(thread::index_1d()) {
            *yi = dot;
        }
    }

    #[kernel]
    pub fn gemv_bf16_bytes(x: &[f32], w: &[u8], mut y: DisjointSlice<f32>, n: u32, k: u32) {
        let row = thread::index_1d().get();
        if (row as u64) >= n as u64 {
            return;
        }
        let k = k as usize;
        let base = row * k * 2;
        let mut dot0 = 0.0f32;
        let mut dot1 = 0.0f32;
        let mut dot2 = 0.0f32;
        let mut dot3 = 0.0f32;
        let mut j = 0usize;
        let k4 = k - k % 4;
        while j < k4 {
            let o0 = base + 2 * j;
            let o1 = base + 2 * (j + 1);
            let o2 = base + 2 * (j + 2);
            let o3 = base + 2 * (j + 3);
            dot0 += x[j] * bf16_pair_to_f32(w[o0], w[o0 + 1]);
            dot1 += x[j + 1] * bf16_pair_to_f32(w[o1], w[o1 + 1]);
            dot2 += x[j + 2] * bf16_pair_to_f32(w[o2], w[o2 + 1]);
            dot3 += x[j + 3] * bf16_pair_to_f32(w[o3], w[o3 + 1]);
            j += 4;
        }
        let mut dot = dot0 + dot1 + dot2 + dot3;
        while j < k {
            let off = base + 2 * j;
            dot += x[j] * bf16_pair_to_f32(w[off], w[off + 1]);
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

    // ── Artifact FP4 E2M1 + E8M0 GEMV ─────────────────────────────────

    /// GEMV for artifact-preserved `torch.float4_e2m1fn_x2` weights with
    /// `float8_e8m0fnu` scales.
    ///
    /// Layout matches Ferrule's CPU reference path for artifact-format packed experts:
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
        let row_packed = row * packed_cols;
        let row_scales = row * scale_cols;
        let mut block = 0usize;
        while block < scale_cols {
            let scale = e8m0_scale(scales[row_scales + block]);
            let packed_base = row_packed + block * 16;
            let x_base = block * 32;
            let mut b = 0usize;
            while b < 16 {
                let byte = packed[packed_base + b];
                let j = x_base + b * 2;
                dot += (x[j] * fp4_e2m1_value(byte & 0x0f) + x[j + 1] * fp4_e2m1_value(byte >> 4))
                    * scale;
                b += 1;
            }
            block += 1;
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
        let row_packed = packed_off + row * packed_cols;
        let row_scales = scales_off + row * scale_cols;
        let mut block = 0usize;
        while block < scale_cols {
            let scale = e8m0_scale(scales[row_scales + block]);
            let packed_base = row_packed + block * 16;
            let x_base = block * 32;
            let mut b = 0usize;
            while b < 16 {
                let byte = packed[packed_base + b];
                let j = x_base + b * 2;
                dot += (x[j] * fp4_e2m1_value(byte & 0x0f) + x[j + 1] * fp4_e2m1_value(byte >> 4))
                    * scale;
                b += 1;
            }
            block += 1;
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
        let row_p0 = off_p0 + row * packed_cols;
        let row_p1 = off_p1 + row * packed_cols;
        let row_s0 = off_s0 + row * scale_cols;
        let row_s1 = off_s1 + row * scale_cols;
        let mut block = 0usize;
        while block < scale_cols {
            let scale0 = e8m0_scale(s0[row_s0 + block]);
            let scale1 = e8m0_scale(s1[row_s1 + block]);
            let packed_base0 = row_p0 + block * 16;
            let packed_base1 = row_p1 + block * 16;
            let x_base = block * 32;
            let mut b = 0usize;
            while b < 16 {
                let byte0 = p0[packed_base0 + b];
                let byte1 = p1[packed_base1 + b];
                let j = x_base + b * 2;
                let x0 = x[j];
                let x1 = x[j + 1];
                d0 +=
                    (x0 * fp4_e2m1_value(byte0 & 0x0f) + x1 * fp4_e2m1_value(byte0 >> 4)) * scale0;
                d1 +=
                    (x0 * fp4_e2m1_value(byte1 & 0x0f) + x1 * fp4_e2m1_value(byte1 >> 4)) * scale1;
                b += 1;
            }
            block += 1;
        }
        if let Some(o) = y0.get_mut(thread::index_1d()) {
            *o = d0;
        }
        if let Some(o) = y1.get_mut(thread::index_1d()) {
            *o = d1;
        }
    }

    /// Clipped SwiGLU expert activation:
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

    /// Simulate block-wise FP8 E4M3FN + E8M0 activation quantization in-place.
    ///
    /// One CUDA thread handles one activation block. This is intentionally simple
    /// and correctness-oriented; it removes host round-trips before later fusion.
    #[kernel]
    pub fn fp8_e4m3fn_e8m0_quantize_f32_inplace(
        mut values: DisjointSlice<f32>,
        value_len: u32,
        row_width: u32,
        block_size: u32,
    ) {
        let block_idx = thread::index_1d().get() as usize;
        let value_len = value_len as usize;
        let row_width = row_width as usize;
        let block_size = block_size as usize;
        if row_width == 0 || block_size == 0 || !row_width.is_multiple_of(block_size) {
            return;
        }
        let blocks_per_row = row_width / block_size;
        let row = block_idx / blocks_per_row;
        let block = block_idx % blocks_per_row;
        let start = row * row_width + block * block_size;
        if start >= value_len {
            return;
        }
        let end = (start + block_size).min(value_len);
        let ptr = values.as_mut_ptr();
        let mut amax = 1e-4f32;
        let mut i = start;
        while i < end {
            let value = unsafe { *ptr.add(i) };
            let abs_value = if value < 0.0 { -value } else { value };
            if abs_value > amax {
                amax = abs_value;
            }
            i += 1;
        }
        let scale = rounded_power_of_two_scale(amax, 448.0);
        let mut i = start;
        while i < end {
            let value = unsafe { *ptr.add(i) };
            let scaled = clamp_range(value / scale, -448.0, 448.0);
            let quantized = quantize_fp8_e4m3fn_to_f32(scaled) * scale;
            unsafe {
                *ptr.add(i) = quantized;
            }
            i += 1;
        }
    }

    /// Block-wise FP4 E2M1 + E8M0 activation quantization into packed bytes.
    ///
    /// `values` is row-major `[rows, row_width]`; `packed` is
    /// `[rows, row_width / 2]` low-nibble-first; `scales` is
    /// `[rows, row_width / block_size]`. For mxf4 Tensor Core paths the block
    /// size should be 32.
    #[kernel]
    pub fn fp4_e2m1_e8m0_quantize_f32_packed(
        values: &[f32],
        mut packed: DisjointSlice<u8>,
        mut scales: DisjointSlice<u8>,
        value_len: u32,
        row_width: u32,
        block_size: u32,
    ) {
        let block_idx = thread::index_1d().get() as usize;
        let value_len = value_len as usize;
        let row_width = row_width as usize;
        let block_size = block_size as usize;
        if row_width == 0
            || block_size == 0
            || !row_width.is_multiple_of(block_size)
            || !block_size.is_multiple_of(2)
        {
            return;
        }
        if value_len == 0 || !value_len.is_multiple_of(row_width) {
            return;
        }
        let blocks_per_row = row_width / block_size;
        let total_blocks = value_len / block_size;
        if block_idx >= total_blocks {
            return;
        }
        let row = block_idx / blocks_per_row;
        let block = block_idx % blocks_per_row;
        let start = row * row_width + block * block_size;
        let end = start + block_size;

        let mut amax = 0.0f32;
        let mut i = start;
        while i < end {
            let value = values[i];
            let abs_value = if value < 0.0 { -value } else { value };
            if abs_value > amax {
                amax = abs_value;
            }
            i += 1;
        }
        let scale_byte = e8m0_scale_byte_for_amax(amax, 6.0);
        let scale = e8m0_scale(scale_byte);
        unsafe {
            *scales.as_mut_ptr().add(row * blocks_per_row + block) = scale_byte;
        }

        let packed_row = row * (row_width / 2);
        let packed_block = block * (block_size / 2);
        let packed_ptr = packed.as_mut_ptr();
        let mut j = 0usize;
        while j < block_size {
            let v0 = values[start + j] / scale;
            let v1 = values[start + j + 1] / scale;
            let n0 = quantize_fp4_e2m1_nibble(v0);
            let n1 = quantize_fp4_e2m1_nibble(v1);
            unsafe {
                *packed_ptr.add(packed_row + packed_block + j / 2) = n0 | (n1 << 4);
            }
            j += 2;
        }
    }

    // ── Artifact FP8 E4M3FN + E8M0 GEMV ────────────────────────────────

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
        let row_weight = row * k;
        let row_scales = (row / bm) * sc;
        let mut block = 0usize;
        while block < sc {
            let scale = e8m0_scale(scales[row_scales + block]);
            let start = block * bk;
            let end = (start + bk).min(k);
            let mut j = start;
            while j < end {
                let w = fp8_e4m3fn_value(weight[row_weight + j]);
                dot += x[j] * w * scale;
                j += 1;
            }
            block += 1;
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
        let row_packed = row * packed_cols;
        let row_scales = row * scale_cols;
        let mut block = 0usize;
        while block < scale_cols {
            let scale = e8m0_scale(scales[row_scales + block]);
            let packed_base = row_packed + block * 16;
            let x_base = input_off + block * 32;
            let mut p = 0usize;
            while p < 16 {
                let byte = packed[packed_base + p];
                let j = x_base + p * 2;
                dot += (x[j] * fp4_e2m1_value(byte & 0x0f) + x[j + 1] * fp4_e2m1_value(byte >> 4))
                    * scale;
                p += 1;
            }
            block += 1;
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

    /// Copy `n` f32 elements from `src` into `dst` starting at element
    /// `dst_offset`. Used for device-resident KV cache slot appends without
    /// a host round-trip.
    #[kernel]
    pub fn copy_f32_slot(src: &[f32], mut dst: DisjointSlice<f32>, dst_offset: u32, n: u32) {
        let i = thread::index_1d().get();
        if (i as u64) >= n as u64 {
            return;
        }
        let dst_idx = dst_offset as usize + i;
        let dst_ptr = dst.as_mut_ptr();
        unsafe {
            *dst_ptr.add(dst_idx) = src[i];
        }
    }

    /// Compute 1/sqrt(mean(x^2) + eps) through libdevice-backed float math.
    /// This matches llama.cpp's rsqrtf(var + eps) semantics without inline PTX.
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
            *o = fast_rsqrt(val);
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

        // Phase 3: thread 0 computes reciprocal sqrt from the total sum.
        if tid == 0 {
            let val = unsafe { SMEM[0] } / n as f32 + eps;
            unsafe {
                SMEM[0] = fast_rsqrt(val);
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

    /// Per-head RMS normalize `[heads, head_dim]` without an affine weight.
    /// One CUDA block owns one head; each block reduces its row and writes the
    /// normalized row to `y`.
    #[kernel]
    pub fn rms_norm_heads_fused(
        x: &[f32],
        mut y: DisjointSlice<f32>,
        heads: u32,
        head_dim: u32,
        eps: f32,
    ) {
        static mut SMEM: SharedArray<f32, 1024> = SharedArray::UNINIT;
        let head = thread::blockIdx_x() as usize;
        let heads = heads as usize;
        if head >= heads {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let bdim = thread::blockDim_x() as usize;
        let hd = head_dim as usize;
        let base = head * hd;

        let mut sum = 0.0f32;
        let mut j = tid;
        while j < hd {
            let v = x[base + j];
            sum += v * v;
            j += bdim;
        }
        unsafe {
            SMEM[tid] = sum;
        }
        thread::sync_threads();

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

        if tid == 0 {
            let val = unsafe { SMEM[0] } / hd as f32 + eps;
            unsafe {
                SMEM[0] = fast_rsqrt(val);
            }
        }
        thread::sync_threads();

        let rsqrt = unsafe { SMEM[0] };
        let y_ptr = y.as_mut_ptr();
        let mut j = tid;
        while j < hd {
            unsafe {
                *y_ptr.add(base + j) = x[base + j] * rsqrt;
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

    // ── Sparse attention with sink ────────────────────────────────────

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

    // ── Correctness-first MLA query projection ───────────────────────
    /// MLA query path: `x → query_a → rms_norm(query_norm) → query_b`.
    ///
    /// This kernel intentionally recomputes the low-rank latent vector per output
    /// element so it can stay scratch-free and easy to validate. It is a semantic
    /// reference kernel, not the optimized production form needed for full-model
    /// decode.
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
        if hs == 0 || qr == 0 {
            return;
        }

        // Pass 1: compute the RMS over the whole low-rank latent vector.
        let mut sum_sq = 0.0f32;
        for r in 0..qr {
            let mut latent = 0.0f32;
            for j in 0..hs {
                latent += x[j] * wq_a[r * hs + j];
            }
            sum_sq += latent * latent;
        }
        let val = sum_sq / qr as f32 + eps;
        let rms = fast_rsqrt(val);

        // Pass 2: apply q_norm and project through query_b.
        let row = idx as usize;
        let mut dot = 0.0f32;
        for r in 0..qr {
            let mut latent = 0.0f32;
            for j in 0..hs {
                latent += x[j] * wq_a[r * hs + j];
            }
            let normed = latent * rms * q_norm[r];
            dot += normed * wq_b[row * qr + r];
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

    // ── DSV4 Tail Rotary (YAARN-scaled, interleaved pairs) ────────────
    /// Apply rotary embedding to the *last* `rope_dim` elements of each head.
    /// `cos_table` / `sin_table` are precomputed `[max_positions, rope_dim/2]`.
    /// Interleaved pair layout: `[x0, x1] → [x0*cos - x1*sin, x0*sin + x1*cos]`.
    #[kernel]
    pub fn rope_tail_yaarn(
        mut qk: DisjointSlice<f32>,
        cos_table: &[f32],
        sin_table: &[f32],
        num_elements: u32,
        position: u32,
        _heads: u32,
        head_dim: u32,
        rope_dim: u32,
        inverse: u32,
    ) {
        let idx = thread::index_1d().get();
        if (idx as u64) >= num_elements as u64 {
            return;
        }
        let hd = head_dim as usize;
        let rd = rope_dim as usize;
        if rd == 0 || rd > hd {
            return;
        }
        let tail_start = hd - rd;
        let head_idx = idx as usize / hd;
        let local = idx as usize % hd;
        if local < tail_start {
            return;
        }
        let tail_local = local - tail_start;
        let pair = tail_local / 2;
        let rd2 = rd / 2;
        let cos = cos_table[position as usize * rd2 + pair];
        let sin = sin_table[position as usize * rd2 + pair];
        let (s, c) = if inverse != 0 {
            (-sin, cos)
        } else {
            (sin, cos)
        };
        let ptr = qk.as_mut_ptr();
        let base = head_idx * hd + tail_start + pair * 2;
        let x0 = unsafe { *ptr.add(base) };
        let x1 = unsafe { *ptr.add(base + 1) };
        let is_even = (tail_local & 1) == 0;
        let val = if is_even {
            x0 * c - x1 * s
        } else {
            x0 * s + x1 * c
        };
        if let Some(o) = qk.get_mut(thread::index_1d()) {
            *o = val;
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
        let row_packed = row * packed_cols;
        let row_scales = row * scale_cols;
        let mut block = 0usize;
        while block < scale_cols {
            let scale = e8m0_scale(down_scales[row_scales + block]);
            let packed_base = row_packed + block * 16;
            let base = block * 32;
            let mut b = 0usize;
            while b < 16 {
                let byte = down_packed[packed_base + b];
                let j0 = base + b * 2;
                let j1 = j0 + 1;
                let mut g0 = gate[j0];
                let mut u0 = up[j0];
                let mut g1 = gate[j1];
                let mut u1 = up[j1];
                if limit > 0.0 {
                    g0 = clamp_max(g0, limit);
                    u0 = clamp_range(u0, -limit, limit);
                    g1 = clamp_max(g1, limit);
                    u1 = clamp_range(u1, -limit, limit);
                }
                dot += (g0 * fast_sigmoid(g0) * u0 * fp4_e2m1_value(byte & 0x0f)
                    + g1 * fast_sigmoid(g1) * u1 * fp4_e2m1_value(byte >> 4))
                    * scale;
                b += 1;
            }
            block += 1;
        }
        dot *= route_weight;
        if let Some(o) = output.get_mut(thread::index_1d()) {
            *o += dot;
        }
    }

    // ── Batched MoE expert kernels (single launch for all selected experts) ─
    //
    // These kernels process all `num_experts` selected experts in a single
    // launch using a 2D grid: `gridDim = (ceil(output_rows / block), num_experts)`.
    // blockIdx.y = expert index. Expert weight pointers are passed as device
    // address arrays (`&[u64]`) and dereferenced inside the kernel.

    /// Batched gate+up FP4 GEMV for all selected experts.
    ///
    /// Grid: `(ceil(intermediate_size / 256), num_experts)`.
    /// Each thread computes one output row for one expert's gate and up.
    /// `gate_ptrs` / `gate_scale_ptrs` / `up_ptrs` / `up_scale_ptrs` are
    /// `[num_experts]` device address arrays.
    /// `route_weights` is `[num_experts]` f32 (applied later in swiglu).
    /// Output: `y_gate` and `y_up` are `[num_experts * intermediate_size]`.
    #[kernel]
    pub fn moe_gemv_dual_fp4_batched(
        x: &[f32],
        gate_ptrs: &[u64],
        gate_scale_ptrs: &[u64],
        up_ptrs: &[u64],
        up_scale_ptrs: &[u64],
        mut y_gate: DisjointSlice<f32>,
        mut y_up: DisjointSlice<f32>,
        n: u32,
        k: u32,
        num_experts: u32,
    ) {
        let expert = thread::blockIdx_y() as usize;
        let row = thread::blockIdx_x() as usize * thread::blockDim_x() as usize
            + thread::threadIdx_x() as usize;
        if row >= n as usize || expert >= num_experts as usize {
            return;
        }
        let k = k as usize;
        let packed_cols = k / 2;
        let scale_cols = k / 32;

        let gate_ptr = gate_ptrs[expert] as *const u8;
        let gate_scale_ptr = gate_scale_ptrs[expert] as *const u8;
        let up_ptr = up_ptrs[expert] as *const u8;
        let up_scale_ptr = up_scale_ptrs[expert] as *const u8;

        let mut d_gate = 0.0f32;
        let mut d_up = 0.0f32;
        let row_packed = row * packed_cols;
        let row_scales = row * scale_cols;
        let mut block = 0usize;
        while block < scale_cols {
            let scale_g = e8m0_scale(unsafe { *gate_scale_ptr.add(row_scales + block) });
            let scale_u = e8m0_scale(unsafe { *up_scale_ptr.add(row_scales + block) });
            let packed_base = row_packed + block * 16;
            let x_base = block * 32;
            let mut b = 0usize;
            while b < 16 {
                let byte_g = unsafe { *gate_ptr.add(packed_base + b) };
                let byte_u = unsafe { *up_ptr.add(packed_base + b) };
                let j = x_base + b * 2;
                let x0 = x[j];
                let x1 = x[j + 1];
                d_gate += (x0 * fp4_e2m1_value(byte_g & 0x0f) + x1 * fp4_e2m1_value(byte_g >> 4))
                    * scale_g;
                d_up += (x0 * fp4_e2m1_value(byte_u & 0x0f) + x1 * fp4_e2m1_value(byte_u >> 4))
                    * scale_u;
                b += 1;
            }
            block += 1;
        }
        let out_off = expert * n as usize + row;
        let y_ptr = y_gate.as_mut_ptr();
        unsafe {
            *y_ptr.add(out_off) = d_gate;
        }
        let y_up_ptr = y_up.as_mut_ptr();
        unsafe {
            *y_up_ptr.add(out_off) = d_up;
        }
    }

    /// Batched gate+up FP4 GEMM using Blackwell mxf4 Tensor Cores.
    ///
    /// Grid: `(ceil(intermediate_size / 16), num_experts)`, blockDim.x = 32.
    /// One warp computes a `16×min(batch_cols, 8)` output tile for one expert.
    /// Outputs are laid out as `[expert, batch_col, output_row]`; with
    /// `batch_cols=1` this is identical to the historical `[expert, row]` GEMV
    /// layout used by the rest of the decode path.
    #[kernel]
    pub unsafe fn moe_gemm_dual_fp4_mxf4_batched(
        x_packed: &[u8],
        x_scales: &[u8],
        gate_ptrs: &[u64],
        gate_scale_ptrs: &[u64],
        up_ptrs: &[u64],
        up_scale_ptrs: &[u64],
        mut y_gate: DisjointSlice<f32>,
        mut y_up: DisjointSlice<f32>,
        n: u32,
        k: u32,
        batch_cols: u32,
        num_experts: u32,
    ) {
        static mut SMEM_A: SharedArray<u8, 512, 32> = SharedArray::UNINIT;
        static mut SMEM_B: SharedArray<u8, 256, 32> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let expert = thread::blockIdx_y() as usize;
        let row_base = thread::blockIdx_x() as usize * 16;
        let n = n as usize;
        let k = k as usize;
        let batch_cols = batch_cols as usize;
        let num_experts = num_experts as usize;
        if expert >= num_experts
            || batch_cols == 0
            || batch_cols > 8
            || k == 0
            || !k.is_multiple_of(64)
        {
            return;
        }

        let packed_cols = k / 2;
        let scale_cols = k / 32;
        let gate_ptr = gate_ptrs[expert] as *const u8;
        let gate_scale_ptr = gate_scale_ptrs[expert] as *const u8;
        let up_ptr = up_ptrs[expert] as *const u8;
        let up_scale_ptr = up_scale_ptrs[expert] as *const u8;
        let mut acc_gate = [0.0f32; 4];
        let mut acc_up = [0.0f32; 4];

        let mut kt = 0usize;
        while kt < k {
            unsafe {
                let a_dst = &raw mut SMEM_A as *mut u8;
                let b_dst = &raw mut SMEM_B as *mut u8;
                let mut i = tid;
                while i < 512 {
                    *a_dst.add(i) = 0;
                    i += 32;
                }
                let mut i = tid;
                while i < 128 {
                    let k4 = i / 8;
                    let col = i & 7;
                    let dst = k4 * 16 + col * 2;
                    if col < batch_cols {
                        let src = col * packed_cols + kt / 2 + k4 * 2;
                        *b_dst.add(dst) = x_packed[src];
                        *b_dst.add(dst + 1) = x_packed[src + 1];
                    } else {
                        *b_dst.add(dst) = 0;
                        *b_dst.add(dst + 1) = 0;
                    }
                    i += 32;
                }
                let mut i = tid;
                while i < 512 {
                    let row_local = i / 32;
                    let byte = i & 31;
                    let row = row_base + row_local;
                    if row < n {
                        *a_dst.add(row_local * 32 + byte) =
                            *gate_ptr.add(row * packed_cols + kt / 2 + byte);
                    }
                    i += 32;
                }
            }
            thread::sync_threads();

            let b_frag: [u32; 2] = unsafe {
                let row = tid & 0x0F;
                let addr = (&raw const SMEM_B as *const u8).add(row * 16) as *const u32;
                cuda_device::wmma::ldmatrix_x2_trans(addr)
            };
            let a_frag_gate: [u32; 4] = unsafe {
                let q = tid / 8;
                let row = (tid & 7) + if (q & 1) != 0 { 8 } else { 0 };
                let col_b16 = if q >= 2 { 8 } else { 0 };
                let addr =
                    (&raw const SMEM_A as *const u8).add(row * 32 + col_b16 * 2) as *const u32;
                cuda_device::wmma::ldmatrix_x4(addr)
            };
            let group = tid / 4;
            let scale_row = group + if (tid & 1) != 0 { 8 } else { 0 };
            let logical_row = row_base + scale_row;
            let col_for_scale = group;
            let k_block = kt / 32;
            let scale_a_gate = if logical_row < n {
                unsafe {
                    (*gate_scale_ptr.add(logical_row * scale_cols + k_block + 1) as u32) << 8
                        | (*gate_scale_ptr.add(logical_row * scale_cols + k_block) as u32)
                }
            } else {
                (127u32 << 8) | 127u32
            };
            let scale_b = if col_for_scale < batch_cols {
                (x_scales[col_for_scale * scale_cols + k_block + 1] as u32) << 8
                    | (x_scales[col_for_scale * scale_cols + k_block] as u32)
            } else {
                (127u32 << 8) | 127u32
            };
            let d_gate = unsafe {
                mma_m16n8k64_mxf4_f32_e2m1_e2m1_b0_t0_b0_t0(
                    [0.0f32; 4],
                    a_frag_gate,
                    b_frag,
                    scale_a_gate,
                    scale_b,
                )
            };
            let mut j = 0usize;
            while j < 4 {
                acc_gate[j] += d_gate[j];
                j += 1;
            }
            thread::sync_threads();

            unsafe {
                let a_dst = &raw mut SMEM_A as *mut u8;
                let mut i = tid;
                while i < 512 {
                    *a_dst.add(i) = 0;
                    i += 32;
                }
                let mut i = tid;
                while i < 512 {
                    let row_local = i / 32;
                    let byte = i & 31;
                    let row = row_base + row_local;
                    if row < n {
                        *a_dst.add(row_local * 32 + byte) =
                            *up_ptr.add(row * packed_cols + kt / 2 + byte);
                    }
                    i += 32;
                }
            }
            thread::sync_threads();

            let a_frag_up: [u32; 4] = unsafe {
                let q = tid / 8;
                let row = (tid & 7) + if (q & 1) != 0 { 8 } else { 0 };
                let col_b16 = if q >= 2 { 8 } else { 0 };
                let addr =
                    (&raw const SMEM_A as *const u8).add(row * 32 + col_b16 * 2) as *const u32;
                cuda_device::wmma::ldmatrix_x4(addr)
            };
            let scale_a_up = if logical_row < n {
                unsafe {
                    (*up_scale_ptr.add(logical_row * scale_cols + k_block + 1) as u32) << 8
                        | (*up_scale_ptr.add(logical_row * scale_cols + k_block) as u32)
                }
            } else {
                (127u32 << 8) | 127u32
            };
            let d_up = unsafe {
                mma_m16n8k64_mxf4_f32_e2m1_e2m1_b0_t0_b0_t0(
                    [0.0f32; 4],
                    a_frag_up,
                    b_frag,
                    scale_a_up,
                    scale_b,
                )
            };
            let mut j = 0usize;
            while j < 4 {
                acc_up[j] += d_up[j];
                j += 1;
            }
            thread::sync_threads();
            kt += 64;
        }

        let lane = tid;
        let group = lane / 4;
        let thr = lane % 4;
        let gate_ptr_out = y_gate.as_mut_ptr();
        let up_ptr_out = y_up.as_mut_ptr();
        let mut j = 0usize;
        while j < 4 {
            let row = row_base + group + if j >= 2 { 8 } else { 0 };
            let col = thr * 2 + (j & 1);
            if row < n && col < batch_cols {
                let out_off = expert * batch_cols * n + col * n + row;
                unsafe {
                    *gate_ptr_out.add(out_off) = acc_gate[j];
                    *up_ptr_out.add(out_off) = acc_up[j];
                }
            }
            j += 1;
        }
    }

    /// Batched SwiGLU + FP8 quantize for all selected experts.
    ///
    /// Grid: `(ceil(intermediate_size / 256), num_experts)`.
    /// Reads `y_gate` and `y_up`, writes swiglu + fp8-quantized hidden to
    /// `y_hidden`. Also applies route_weight and swiglu_limit.
    #[kernel]
    pub fn moe_swiglu_fp8_batched(
        y_gate: &[f32],
        y_up: &[f32],
        route_weights: &[f32],
        mut y_hidden: DisjointSlice<f32>,
        n: u32,
        num_experts: u32,
        swiglu_limit: f32,
        block_size: u32,
    ) {
        let expert = thread::blockIdx_y() as usize;
        let row = thread::blockIdx_x() as usize * thread::blockDim_x() as usize
            + thread::threadIdx_x() as usize;
        if row >= n as usize || expert >= num_experts as usize {
            return;
        }
        let idx = expert * n as usize + row;
        let mut g = y_gate[idx];
        let mut u = y_up[idx];
        if swiglu_limit > 0.0 {
            g = clamp_max(g, swiglu_limit);
            u = clamp_range(u, -swiglu_limit, swiglu_limit);
        }
        let rw = route_weights[expert];
        let val = g * fast_sigmoid(g) * u * rw;
        let qval = quantize_fp8_e4m3fn_to_f32(val);
        let h_ptr = y_hidden.as_mut_ptr();
        unsafe {
            *h_ptr.add(idx) = qval;
        }
        let _ = block_size;
    }

    /// Batched SwiGLU directly into packed FP4 E2M1 + E8M0 activations for the
    /// Tensor Core down projection.
    ///
    /// Grid: `(intermediate_size / 32, num_experts, batch_cols)`, blockDim.x = 32.
    /// Reads `y_gate`/`y_up` as `[expert, batch_col, row]` and writes
    /// `y_hidden_packed` as `[expert, batch_col, row / 2]` plus one E8M0 scale per
    /// 32-value block. The FP8 activation quantization step from the scalar path
    /// is preserved before FP4 packing so this remains behavior-compatible with
    /// the previous TC path (`SwiGLU -> f32 FP8 values -> FP4 pack`).
    #[kernel]
    pub fn moe_swiglu_fp4_packed_batched(
        y_gate: &[f32],
        y_up: &[f32],
        route_weights: &[f32],
        mut y_hidden_packed: DisjointSlice<u8>,
        mut y_hidden_scales: DisjointSlice<u8>,
        n: u32,
        batch_cols: u32,
        num_experts: u32,
        swiglu_limit: f32,
    ) {
        static mut VALUES: SharedArray<f32, 32> = SharedArray::UNINIT;
        static mut NIBBLES: SharedArray<u8, 32> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let block = thread::blockIdx_x() as usize;
        let expert = thread::blockIdx_y() as usize;
        let batch_col = thread::blockIdx_z() as usize;
        let n = n as usize;
        let batch_cols = batch_cols as usize;
        let num_experts = num_experts as usize;
        if tid >= 32
            || n == 0
            || !n.is_multiple_of(32)
            || batch_cols == 0
            || batch_cols > 8
            || expert >= num_experts
            || batch_col >= batch_cols
            || block >= n / 32
        {
            return;
        }

        let row = block * 32 + tid;
        let idx = (expert * batch_cols + batch_col) * n + row;
        let mut g = y_gate[idx];
        let mut u = y_up[idx];
        if swiglu_limit > 0.0 {
            g = clamp_max(g, swiglu_limit);
            u = clamp_range(u, -swiglu_limit, swiglu_limit);
        }
        let rw = route_weights[expert];
        let val = quantize_fp8_e4m3fn_to_f32(g * fast_sigmoid(g) * u * rw);
        let abs_value = if val < 0.0 { -val } else { val };
        unsafe {
            VALUES[tid] = abs_value;
        }
        thread::sync_threads();

        let mut stride = 16usize;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    let current = VALUES[tid];
                    let other = VALUES[tid + stride];
                    VALUES[tid] = if other > current { other } else { current };
                }
            }
            thread::sync_threads();
            stride /= 2;
        }

        let scale_byte = unsafe { e8m0_scale_byte_for_amax(VALUES[0], 6.0) };
        let scale = e8m0_scale(scale_byte);
        unsafe {
            NIBBLES[tid] = quantize_fp4_e2m1_nibble(val / scale);
        }
        if tid == 0 {
            let scale_cols = n / 32;
            let scale_off = (expert * batch_cols + batch_col) * scale_cols + block;
            unsafe {
                *y_hidden_scales.as_mut_ptr().add(scale_off) = scale_byte;
            }
        }
        thread::sync_threads();

        if tid < 16 {
            let packed_cols = n / 2;
            let packed_off = (expert * batch_cols + batch_col) * packed_cols + block * 16 + tid;
            let lo = unsafe { NIBBLES[tid * 2] };
            let hi = unsafe { NIBBLES[tid * 2 + 1] };
            unsafe {
                *y_hidden_packed.as_mut_ptr().add(packed_off) = lo | (hi << 4);
            }
        }
    }

    /// Batched down FP4 GEMV + accumulate for all selected experts.
    ///
    /// Grid: `(ceil(hidden_size / 256), num_experts)`.
    /// Each thread computes one output element for one expert, reading from
    /// the expert's down weights and the shared `y_hidden` buffer.
    /// Accumulates into `output` (which must be pre-zeroed).
    #[kernel]
    pub fn moe_gemv_down_fp4_batched(
        y_hidden: &[f32],
        down_ptrs: &[u64],
        down_scale_ptrs: &[u64],
        mut output: DisjointSlice<f32>,
        intermediate_size: u32,
        hidden_size: u32,
        num_experts: u32,
    ) {
        let expert = thread::blockIdx_y() as usize;
        let row = thread::blockIdx_x() as usize * thread::blockDim_x() as usize
            + thread::threadIdx_x() as usize;
        if row >= hidden_size as usize || expert >= num_experts as usize {
            return;
        }
        let inter = intermediate_size as usize;
        let packed_cols = inter / 2;
        let scale_cols = inter / 32;

        let down_ptr = down_ptrs[expert] as *const u8;
        let down_scale_ptr = down_scale_ptrs[expert] as *const u8;

        let mut dot = 0.0f32;
        let row_packed = row * packed_cols;
        let row_scales = row * scale_cols;
        let hidden_base = expert * inter;
        let mut block = 0usize;
        while block < scale_cols {
            let scale = e8m0_scale(unsafe { *down_scale_ptr.add(row_scales + block) });
            let packed_base = row_packed + block * 16;
            let h_base = hidden_base + block * 32;
            let mut b = 0usize;
            while b < 16 {
                let byte = unsafe { *down_ptr.add(packed_base + b) };
                let j = h_base + b * 2;
                dot += (y_hidden[j] * fp4_e2m1_value(byte & 0x0f)
                    + y_hidden[j + 1] * fp4_e2m1_value(byte >> 4))
                    * scale;
                b += 1;
            }
            block += 1;
        }
        // Multiple experts can update the same output row concurrently, so this
        // accumulation must be atomic in the batched expert grid.
        let o_ptr = output.as_mut_ptr();
        let o = unsafe { DeviceAtomicF32::from_ptr(o_ptr.add(row)) };
        let _ = o.fetch_add(dot, AtomicOrdering::Relaxed);
    }

    /// Batched down FP4 GEMM using Blackwell mxf4 Tensor Cores.
    ///
    /// Grid: `(ceil(hidden_size / 16), num_experts)`, blockDim.x = 32.
    /// `y_hidden_packed` is laid out as `[expert, batch_col, intermediate/2]`;
    /// output is `[batch_col, hidden_row]`. Multiple experts atomically
    /// accumulate into the shared routed MoE output.
    #[kernel]
    pub unsafe fn moe_gemm_down_fp4_mxf4_batched(
        y_hidden_packed: &[u8],
        y_hidden_scales: &[u8],
        down_ptrs: &[u64],
        down_scale_ptrs: &[u64],
        mut output: DisjointSlice<f32>,
        intermediate_size: u32,
        hidden_size: u32,
        batch_cols: u32,
        num_experts: u32,
    ) {
        static mut SMEM_A: SharedArray<u8, 512, 32> = SharedArray::UNINIT;
        static mut SMEM_B: SharedArray<u8, 256, 32> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let expert = thread::blockIdx_y() as usize;
        let row_base = thread::blockIdx_x() as usize * 16;
        let inter = intermediate_size as usize;
        let hidden = hidden_size as usize;
        let batch_cols = batch_cols as usize;
        let num_experts = num_experts as usize;
        if expert >= num_experts
            || batch_cols == 0
            || batch_cols > 8
            || inter == 0
            || !inter.is_multiple_of(64)
        {
            return;
        }

        let packed_cols = inter / 2;
        let scale_cols = inter / 32;
        let down_ptr = down_ptrs[expert] as *const u8;
        let down_scale_ptr = down_scale_ptrs[expert] as *const u8;
        let mut acc = [0.0f32; 4];

        let mut kt = 0usize;
        while kt < inter {
            unsafe {
                let a_dst = &raw mut SMEM_A as *mut u8;
                let b_dst = &raw mut SMEM_B as *mut u8;
                let mut i = tid;
                while i < 512 {
                    *a_dst.add(i) = 0;
                    i += 32;
                }
                let mut i = tid;
                while i < 128 {
                    let k4 = i / 8;
                    let col = i & 7;
                    let dst = k4 * 16 + col * 2;
                    if col < batch_cols {
                        let src = (expert * batch_cols + col) * packed_cols + kt / 2 + k4 * 2;
                        *b_dst.add(dst) = y_hidden_packed[src];
                        *b_dst.add(dst + 1) = y_hidden_packed[src + 1];
                    } else {
                        *b_dst.add(dst) = 0;
                        *b_dst.add(dst + 1) = 0;
                    }
                    i += 32;
                }
                let mut i = tid;
                while i < 512 {
                    let row_local = i / 32;
                    let byte = i & 31;
                    let row = row_base + row_local;
                    if row < hidden {
                        *a_dst.add(row_local * 32 + byte) =
                            *down_ptr.add(row * packed_cols + kt / 2 + byte);
                    }
                    i += 32;
                }
            }
            thread::sync_threads();

            let a_frag: [u32; 4] = unsafe {
                let q = tid / 8;
                let row = (tid & 7) + if (q & 1) != 0 { 8 } else { 0 };
                let col_b16 = if q >= 2 { 8 } else { 0 };
                let addr =
                    (&raw const SMEM_A as *const u8).add(row * 32 + col_b16 * 2) as *const u32;
                cuda_device::wmma::ldmatrix_x4(addr)
            };
            let b_frag: [u32; 2] = unsafe {
                let row = tid & 0x0F;
                let addr = (&raw const SMEM_B as *const u8).add(row * 16) as *const u32;
                cuda_device::wmma::ldmatrix_x2_trans(addr)
            };
            let group = tid / 4;
            let scale_row = group + if (tid & 1) != 0 { 8 } else { 0 };
            let logical_row = row_base + scale_row;
            let col_for_scale = group;
            let k_block = kt / 32;
            let scale_a = if logical_row < hidden {
                unsafe {
                    (*down_scale_ptr.add(logical_row * scale_cols + k_block + 1) as u32) << 8
                        | (*down_scale_ptr.add(logical_row * scale_cols + k_block) as u32)
                }
            } else {
                (127u32 << 8) | 127u32
            };
            let scale_b = if col_for_scale < batch_cols {
                let scale_base = (expert * batch_cols + col_for_scale) * scale_cols + k_block;
                (y_hidden_scales[scale_base + 1] as u32) << 8 | (y_hidden_scales[scale_base] as u32)
            } else {
                (127u32 << 8) | 127u32
            };
            let d = unsafe {
                mma_m16n8k64_mxf4_f32_e2m1_e2m1_b0_t0_b0_t0(
                    [0.0f32; 4],
                    a_frag,
                    b_frag,
                    scale_a,
                    scale_b,
                )
            };
            let mut j = 0usize;
            while j < 4 {
                acc[j] += d[j];
                j += 1;
            }
            thread::sync_threads();
            kt += 64;
        }

        let lane = tid;
        let group = lane / 4;
        let thr = lane % 4;
        let out_ptr = output.as_mut_ptr();
        let mut j = 0usize;
        while j < 4 {
            let row = row_base + group + if j >= 2 { 8 } else { 0 };
            let col = thr * 2 + (j & 1);
            if row < hidden && col < batch_cols {
                let o = unsafe { DeviceAtomicF32::from_ptr(out_ptr.add(col * hidden + row)) };
                let _ = o.fetch_add(acc[j], AtomicOrdering::Relaxed);
            }
            j += 1;
        }
    }

    /// Experimental block-reduction gate+up FP4 GEMV.
    ///
    /// Grid: `(intermediate_size, num_experts)`. One block owns one
    /// `(expert, row)` and threads split K by FP4 scale block. This is gated
    /// by `FERRULE_CUDA_MOE_REDUCE=1` on the host side for A/B testing.
    #[kernel]
    pub fn moe_gemv_dual_fp4_batched_reduce(
        x: &[f32],
        gate_ptrs: &[u64],
        gate_scale_ptrs: &[u64],
        up_ptrs: &[u64],
        up_scale_ptrs: &[u64],
        mut y_gate: DisjointSlice<f32>,
        mut y_up: DisjointSlice<f32>,
        n: u32,
        k: u32,
        num_experts: u32,
    ) {
        static mut SUM_GATE: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut SUM_UP: SharedArray<f32, 256> = SharedArray::UNINIT;

        let row = thread::blockIdx_x() as usize;
        let expert = thread::blockIdx_y() as usize;
        if row >= n as usize || expert >= num_experts as usize {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let bdim = thread::blockDim_x() as usize;
        let k = k as usize;
        let packed_cols = k / 2;
        let scale_cols = k / 32;

        let gate_ptr = gate_ptrs[expert] as *const u8;
        let gate_scale_ptr = gate_scale_ptrs[expert] as *const u8;
        let up_ptr = up_ptrs[expert] as *const u8;
        let up_scale_ptr = up_scale_ptrs[expert] as *const u8;

        let row_packed = row * packed_cols;
        let row_scales = row * scale_cols;
        let mut d_gate = 0.0f32;
        let mut d_up = 0.0f32;
        let mut block = tid;
        while block < scale_cols {
            let scale_g = e8m0_scale(unsafe { *gate_scale_ptr.add(row_scales + block) });
            let scale_u = e8m0_scale(unsafe { *up_scale_ptr.add(row_scales + block) });
            let packed_base = row_packed + block * 16;
            let x_base = block * 32;
            let mut b = 0usize;
            while b < 16 {
                let byte_g = unsafe { *gate_ptr.add(packed_base + b) };
                let byte_u = unsafe { *up_ptr.add(packed_base + b) };
                let j = x_base + b * 2;
                let x0 = x[j];
                let x1 = x[j + 1];
                d_gate += (x0 * fp4_e2m1_value(byte_g & 0x0f) + x1 * fp4_e2m1_value(byte_g >> 4))
                    * scale_g;
                d_up += (x0 * fp4_e2m1_value(byte_u & 0x0f) + x1 * fp4_e2m1_value(byte_u >> 4))
                    * scale_u;
                b += 1;
            }
            block += bdim;
        }

        unsafe {
            SUM_GATE[tid] = d_gate;
            SUM_UP[tid] = d_up;
        }
        thread::sync_threads();

        let mut stride = bdim / 2;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    SUM_GATE[tid] += SUM_GATE[tid + stride];
                    SUM_UP[tid] += SUM_UP[tid + stride];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }

        if tid == 0 {
            let out_off = expert * n as usize + row;
            let y_ptr = y_gate.as_mut_ptr();
            let y_up_ptr = y_up.as_mut_ptr();
            unsafe {
                *y_ptr.add(out_off) = SUM_GATE[0];
                *y_up_ptr.add(out_off) = SUM_UP[0];
            }
        }
    }

    /// Experimental block-reduction down FP4 GEMV with one atomic add per
    /// `(expert, output row)`.
    ///
    /// Grid: `(hidden_size, num_experts)`. Gated by
    /// `FERRULE_CUDA_MOE_REDUCE=1` on the host side for A/B testing.
    #[kernel]
    pub fn moe_gemv_down_fp4_batched_reduce(
        y_hidden: &[f32],
        down_ptrs: &[u64],
        down_scale_ptrs: &[u64],
        mut output: DisjointSlice<f32>,
        intermediate_size: u32,
        hidden_size: u32,
        num_experts: u32,
    ) {
        static mut SUM: SharedArray<f32, 256> = SharedArray::UNINIT;

        let row = thread::blockIdx_x() as usize;
        let expert = thread::blockIdx_y() as usize;
        if row >= hidden_size as usize || expert >= num_experts as usize {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let bdim = thread::blockDim_x() as usize;
        let inter = intermediate_size as usize;
        let packed_cols = inter / 2;
        let scale_cols = inter / 32;

        let down_ptr = down_ptrs[expert] as *const u8;
        let down_scale_ptr = down_scale_ptrs[expert] as *const u8;

        let row_packed = row * packed_cols;
        let row_scales = row * scale_cols;
        let hidden_base = expert * inter;
        let mut dot = 0.0f32;
        let mut block = tid;
        while block < scale_cols {
            let scale = e8m0_scale(unsafe { *down_scale_ptr.add(row_scales + block) });
            let packed_base = row_packed + block * 16;
            let h_base = hidden_base + block * 32;
            let mut b = 0usize;
            while b < 16 {
                let byte = unsafe { *down_ptr.add(packed_base + b) };
                let j = h_base + b * 2;
                dot += (y_hidden[j] * fp4_e2m1_value(byte & 0x0f)
                    + y_hidden[j + 1] * fp4_e2m1_value(byte >> 4))
                    * scale;
                b += 1;
            }
            block += bdim;
        }

        unsafe {
            SUM[tid] = dot;
        }
        thread::sync_threads();

        let mut stride = bdim / 2;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    SUM[tid] += SUM[tid + stride];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }

        if tid == 0 {
            let o_ptr = output.as_mut_ptr();
            let o = unsafe { DeviceAtomicF32::from_ptr(o_ptr.add(row)) };
            let _ = o.fetch_add(unsafe { SUM[0] }, AtomicOrdering::Relaxed);
        }
    }

    // ── Generic Hyper-Connection helpers ───────────────────────────────

    #[kernel]
    pub fn hc_pre_f32(
        state: &[f32],
        function: &[f32],
        scale: &[f32],
        base: &[f32],
        mut hidden: DisjointSlice<f32>,
        mut split_pre: DisjointSlice<f32>,
        mut split_post: DisjointSlice<f32>,
        mut split_comb: DisjointSlice<f32>,
        tokens: u32,
        hc_mult: u32,
        hidden_size: u32,
        mix_hc: u32,
        sinkhorn_iters: u32,
        eps: f32,
        norm_eps: f32,
    ) {
        static mut SUM: SharedArray<f32, 1024> = SharedArray::UNINIT;
        static mut MIX: SharedArray<f32, 128> = SharedArray::UNINIT;
        static mut PRE: SharedArray<f32, 16> = SharedArray::UNINIT;
        static mut COMB: SharedArray<f32, 256> = SharedArray::UNINIT;
        let token = thread::blockIdx_x() as usize;
        if token >= tokens as usize {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let bdim = thread::blockDim_x() as usize;
        let hc = hc_mult as usize;
        let dim = hidden_size as usize;
        let mix = mix_hc as usize;
        let hc_dim = hc * dim;
        if hc == 0 || hc > 16 || mix > 128 || hc * hc > 256 {
            return;
        }
        let state_base = token * hc_dim;

        let mut sum = 0.0f32;
        let mut j = tid;
        while j < hc_dim {
            let v = state[state_base + j];
            sum += v * v;
            j += bdim;
        }
        unsafe {
            SUM[tid] = sum;
        }
        thread::sync_threads();
        let mut stride = (bdim + 1) / 2;
        while stride > 0 {
            if tid < stride && tid + stride < bdim {
                unsafe {
                    SUM[tid] += SUM[tid + stride];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        if tid == 0 {
            let val = unsafe { SUM[0] } / hc_dim as f32 + norm_eps;
            unsafe {
                SUM[0] = fast_rsqrt(val);
            }
        }
        thread::sync_threads();
        let rms = unsafe { SUM[0] };

        if tid < mix {
            let mut dot = 0.0f32;
            let row = tid;
            for col in 0..hc_dim {
                dot += function[row * hc_dim + col] * state[state_base + col];
            }
            unsafe {
                MIX[row] = dot * rms;
            }
        }
        thread::sync_threads();

        if tid == 0 {
            let pre_ptr = split_pre.as_mut_ptr();
            let post_ptr = split_post.as_mut_ptr();
            let comb_ptr = split_comb.as_mut_ptr();
            let pre_base = token * hc;
            let comb_base = token * hc * hc;
            for copy in 0..hc {
                let pre = fast_sigmoid(unsafe { MIX[copy] } * scale[0] + base[copy]) + eps;
                let post =
                    2.0 * fast_sigmoid(unsafe { MIX[hc + copy] } * scale[1] + base[hc + copy]);
                unsafe {
                    PRE[copy] = pre;
                    *pre_ptr.add(pre_base + copy) = pre;
                    *post_ptr.add(pre_base + copy) = post;
                }
            }

            for row in 0..hc {
                let mut row_max = f32::NEG_INFINITY;
                for col in 0..hc {
                    let idx = row * hc + col;
                    let v = unsafe { MIX[2 * hc + idx] } * scale[2] + base[2 * hc + idx];
                    unsafe {
                        COMB[idx] = v;
                    }
                    if v > row_max {
                        row_max = v;
                    }
                }
                let mut row_sum = 0.0f32;
                for col in 0..hc {
                    let idx = row * hc + col;
                    let v = fast_exp(unsafe { COMB[idx] } - row_max);
                    unsafe {
                        COMB[idx] = v;
                    }
                    row_sum += v;
                }
                for col in 0..hc {
                    let idx = row * hc + col;
                    unsafe {
                        COMB[idx] /= row_sum;
                        COMB[idx] += eps;
                    }
                }
            }

            for col in 0..hc {
                let mut col_sum = 0.0f32;
                for row in 0..hc {
                    col_sum += unsafe { COMB[row * hc + col] };
                }
                for row in 0..hc {
                    let idx = row * hc + col;
                    unsafe {
                        COMB[idx] /= col_sum + eps;
                    }
                }
            }
            let mut iter = 1u32;
            while iter < sinkhorn_iters {
                for row in 0..hc {
                    let mut row_sum = 0.0f32;
                    for col in 0..hc {
                        row_sum += unsafe { COMB[row * hc + col] };
                    }
                    for col in 0..hc {
                        let idx = row * hc + col;
                        unsafe {
                            COMB[idx] /= row_sum + eps;
                        }
                    }
                }
                for col in 0..hc {
                    let mut col_sum = 0.0f32;
                    for row in 0..hc {
                        col_sum += unsafe { COMB[row * hc + col] };
                    }
                    for row in 0..hc {
                        let idx = row * hc + col;
                        unsafe {
                            COMB[idx] /= col_sum + eps;
                        }
                    }
                }
                iter += 1;
            }

            for idx in 0..hc * hc {
                unsafe {
                    *comb_ptr.add(comb_base + idx) = COMB[idx];
                }
            }
        }
        thread::sync_threads();

        let hidden_ptr = hidden.as_mut_ptr();
        let hidden_base = token * dim;
        let mut d = tid;
        while d < dim {
            let mut out = 0.0f32;
            for copy in 0..hc {
                out += unsafe { PRE[copy] } * state[state_base + copy * dim + d];
            }
            unsafe {
                *hidden_ptr.add(hidden_base + d) = out;
            }
            d += bdim;
        }
    }

    #[kernel]
    pub fn hc_post_f32(
        hidden: &[f32],
        residual: &[f32],
        split_post: &[f32],
        split_comb: &[f32],
        mut output: DisjointSlice<f32>,
        tokens: u32,
        hc_mult: u32,
        hidden_size: u32,
    ) {
        let idx = thread::index_1d().get();
        let hc = hc_mult as usize;
        let dim = hidden_size as usize;
        let hc_dim = hc * dim;
        let total = tokens as usize * hc_dim;
        if (idx as u64) >= total as u64 {
            return;
        }
        let idx = idx as usize;
        let token = idx / hc_dim;
        let rem = idx % hc_dim;
        let out_copy = rem / dim;
        let d = rem % dim;
        let mut residual_mix = 0.0f32;
        let comb_base = token * hc * hc;
        let residual_base = token * hc_dim;
        for in_copy in 0..hc {
            let comb = split_comb[comb_base + in_copy * hc + out_copy];
            residual_mix += comb * residual[residual_base + in_copy * dim + d];
        }
        if let Some(o) = output.get_mut(thread::index_1d()) {
            *o = split_post[token * hc + out_copy] * hidden[token * dim + d] + residual_mix;
        }
    }

    #[kernel]
    pub fn hc_head_f32(
        state: &[f32],
        function: &[f32],
        scale: &[f32],
        base: &[f32],
        mut hidden: DisjointSlice<f32>,
        tokens: u32,
        hc_mult: u32,
        hidden_size: u32,
        eps: f32,
        norm_eps: f32,
    ) {
        static mut SUM: SharedArray<f32, 1024> = SharedArray::UNINIT;
        static mut PRE: SharedArray<f32, 16> = SharedArray::UNINIT;
        let token = thread::blockIdx_x() as usize;
        if token >= tokens as usize {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let bdim = thread::blockDim_x() as usize;
        let hc = hc_mult as usize;
        let dim = hidden_size as usize;
        let hc_dim = hc * dim;
        if hc == 0 || hc > 16 {
            return;
        }
        let state_base = token * hc_dim;

        let mut sum = 0.0f32;
        let mut j = tid;
        while j < hc_dim {
            let v = state[state_base + j];
            sum += v * v;
            j += bdim;
        }
        unsafe {
            SUM[tid] = sum;
        }
        thread::sync_threads();
        let mut stride = (bdim + 1) / 2;
        while stride > 0 {
            if tid < stride && tid + stride < bdim {
                unsafe {
                    SUM[tid] += SUM[tid + stride];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        if tid == 0 {
            let val = unsafe { SUM[0] } / hc_dim as f32 + norm_eps;
            unsafe {
                SUM[0] = fast_rsqrt(val);
            }
        }
        thread::sync_threads();
        let rms = unsafe { SUM[0] };

        if tid < hc {
            let mut dot = 0.0f32;
            let row = tid;
            for col in 0..hc_dim {
                dot += function[row * hc_dim + col] * state[state_base + col];
            }
            unsafe {
                PRE[row] = fast_sigmoid(dot * rms * scale[0] + base[row]) + eps;
            }
        }
        thread::sync_threads();

        let hidden_ptr = hidden.as_mut_ptr();
        let hidden_base = token * dim;
        let mut d = tid;
        while d < dim {
            let mut out = 0.0f32;
            for copy in 0..hc {
                out += unsafe { PRE[copy] } * state[state_base + copy * dim + d];
            }
            unsafe {
                *hidden_ptr.add(hidden_base + d) = out;
            }
            d += bdim;
        }
    }

    #[kernel]
    pub fn hc_pre_single_f32(
        state: &[f32],
        function: &[f32],
        scale: &[f32],
        base: &[f32],
        mut hidden: DisjointSlice<f32>,
        mut split_pre: DisjointSlice<f32>,
        mut split_post: DisjointSlice<f32>,
        mut split_comb: DisjointSlice<f32>,
        hc_mult: u32,
        hidden_size: u32,
        mix_hc: u32,
        sinkhorn_iters: u32,
        eps: f32,
        norm_eps: f32,
    ) {
        static mut SUM: SharedArray<f32, 1024> = SharedArray::UNINIT;
        static mut MIX: SharedArray<f32, 128> = SharedArray::UNINIT;
        static mut PRE: SharedArray<f32, 16> = SharedArray::UNINIT;
        static mut COMB: SharedArray<f32, 256> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let bdim = thread::blockDim_x() as usize;
        let hc = hc_mult as usize;
        let dim = hidden_size as usize;
        let mix = mix_hc as usize;
        let hc_dim = hc * dim;
        if hc == 0 || hc > 16 || mix > 128 || hc * hc > 256 {
            return;
        }

        let mut sum = 0.0f32;
        let mut j = tid;
        while j < hc_dim {
            let v = state[j];
            sum += v * v;
            j += bdim;
        }
        unsafe {
            SUM[tid] = sum;
        }
        thread::sync_threads();
        let mut stride = (bdim + 1) / 2;
        while stride > 0 {
            if tid < stride && tid + stride < bdim {
                unsafe {
                    SUM[tid] += SUM[tid + stride];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        if tid == 0 {
            let val = unsafe { SUM[0] } / hc_dim as f32 + norm_eps;
            unsafe {
                SUM[0] = fast_rsqrt(val);
            }
        }
        thread::sync_threads();
        let rms = unsafe { SUM[0] };

        if tid < mix {
            let mut dot = 0.0f32;
            let row = tid;
            for col in 0..hc_dim {
                dot += function[row * hc_dim + col] * state[col];
            }
            unsafe {
                MIX[row] = dot * rms;
            }
        }
        thread::sync_threads();

        if tid == 0 {
            for copy in 0..hc {
                let pre = fast_sigmoid(unsafe { MIX[copy] } * scale[0] + base[copy]) + eps;
                let post =
                    2.0 * fast_sigmoid(unsafe { MIX[hc + copy] } * scale[1] + base[hc + copy]);
                unsafe {
                    PRE[copy] = pre;
                }
                let pre_ptr = split_pre.as_mut_ptr();
                let post_ptr = split_post.as_mut_ptr();
                unsafe {
                    *pre_ptr.add(copy) = pre;
                    *post_ptr.add(copy) = post;
                }
            }

            for row in 0..hc {
                let mut row_max = f32::NEG_INFINITY;
                for col in 0..hc {
                    let idx = row * hc + col;
                    let v = unsafe { MIX[2 * hc + idx] } * scale[2] + base[2 * hc + idx];
                    unsafe {
                        COMB[idx] = v;
                    }
                    if v > row_max {
                        row_max = v;
                    }
                }
                let mut row_sum = 0.0f32;
                for col in 0..hc {
                    let idx = row * hc + col;
                    let v = fast_exp(unsafe { COMB[idx] } - row_max);
                    unsafe {
                        COMB[idx] = v;
                    }
                    row_sum += v;
                }
                for col in 0..hc {
                    let idx = row * hc + col;
                    unsafe {
                        COMB[idx] /= row_sum;
                        COMB[idx] += eps;
                    }
                }
            }

            for col in 0..hc {
                let mut col_sum = 0.0f32;
                for row in 0..hc {
                    col_sum += unsafe { COMB[row * hc + col] };
                }
                for row in 0..hc {
                    let idx = row * hc + col;
                    unsafe {
                        COMB[idx] /= col_sum + eps;
                    }
                }
            }
            let mut iter = 1u32;
            while iter < sinkhorn_iters {
                for row in 0..hc {
                    let mut row_sum = 0.0f32;
                    for col in 0..hc {
                        row_sum += unsafe { COMB[row * hc + col] };
                    }
                    for col in 0..hc {
                        let idx = row * hc + col;
                        unsafe {
                            COMB[idx] /= row_sum + eps;
                        }
                    }
                }
                for col in 0..hc {
                    let mut col_sum = 0.0f32;
                    for row in 0..hc {
                        col_sum += unsafe { COMB[row * hc + col] };
                    }
                    for row in 0..hc {
                        let idx = row * hc + col;
                        unsafe {
                            COMB[idx] /= col_sum + eps;
                        }
                    }
                }
                iter += 1;
            }

            let comb_ptr = split_comb.as_mut_ptr();
            for idx in 0..hc * hc {
                unsafe {
                    *comb_ptr.add(idx) = COMB[idx];
                }
            }
        }
        thread::sync_threads();

        let hidden_ptr = hidden.as_mut_ptr();
        let mut d = tid;
        while d < dim {
            let mut out = 0.0f32;
            for copy in 0..hc {
                out += unsafe { PRE[copy] } * state[copy * dim + d];
            }
            unsafe {
                *hidden_ptr.add(d) = out;
            }
            d += bdim;
        }
    }

    #[kernel]
    pub fn hc_post_single_f32(
        hidden: &[f32],
        residual: &[f32],
        split_post: &[f32],
        split_comb: &[f32],
        mut output: DisjointSlice<f32>,
        hc_mult: u32,
        hidden_size: u32,
    ) {
        let idx = thread::index_1d().get();
        let hc = hc_mult as usize;
        let dim = hidden_size as usize;
        let total = hc * dim;
        if (idx as u64) >= total as u64 {
            return;
        }
        let idx = idx as usize;
        let copy = idx / dim;
        let d = idx % dim;
        let mut comb_row_sum = 0.0f32;
        for k in 0..hc {
            comb_row_sum += split_comb[copy * hc + k];
        }
        if let Some(o) = output.get_mut(thread::index_1d()) {
            *o = split_post[copy] * hidden[d] + comb_row_sum * residual[idx];
        }
    }

    #[kernel]
    pub fn hc_head_single_f32(
        state: &[f32],
        function: &[f32],
        scale: &[f32],
        base: &[f32],
        mut hidden: DisjointSlice<f32>,
        hc_mult: u32,
        hidden_size: u32,
        eps: f32,
        norm_eps: f32,
    ) {
        static mut SUM: SharedArray<f32, 1024> = SharedArray::UNINIT;
        static mut PRE: SharedArray<f32, 16> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let bdim = thread::blockDim_x() as usize;
        let hc = hc_mult as usize;
        let dim = hidden_size as usize;
        let hc_dim = hc * dim;
        if hc == 0 || hc > 16 {
            return;
        }

        let mut sum = 0.0f32;
        let mut j = tid;
        while j < hc_dim {
            let v = state[j];
            sum += v * v;
            j += bdim;
        }
        unsafe {
            SUM[tid] = sum;
        }
        thread::sync_threads();
        let mut stride = (bdim + 1) / 2;
        while stride > 0 {
            if tid < stride && tid + stride < bdim {
                unsafe {
                    SUM[tid] += SUM[tid + stride];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        if tid == 0 {
            let val = unsafe { SUM[0] } / hc_dim as f32 + norm_eps;
            unsafe {
                SUM[0] = fast_rsqrt(val);
            }
        }
        thread::sync_threads();
        let rms = unsafe { SUM[0] };

        if tid < hc {
            let mut dot = 0.0f32;
            let row = tid;
            for col in 0..hc_dim {
                dot += function[row * hc_dim + col] * state[col];
            }
            unsafe {
                PRE[row] = fast_sigmoid(dot * rms * scale[0] + base[row]) + eps;
            }
        }
        thread::sync_threads();

        let hidden_ptr = hidden.as_mut_ptr();
        let mut d = tid;
        while d < dim {
            let mut out = 0.0f32;
            for copy in 0..hc {
                out += unsafe { PRE[copy] } * state[copy * dim + d];
            }
            unsafe {
                *hidden_ptr.add(d) = out;
            }
            d += bdim;
        }
    }

    // ── P6.2: microscaled FP4 (mxf4) warp MMA smoke (sm_120a+) ───────────
    //
    // GB10 FP4 tensor-core path. Single-tile: C[16×8] = A[16×64] × B[64×8]
    // in FP4 (E2M1) with E8M0 block scales via one
    // `mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.f32.e2m1.e2m1.f32.ue8m0`.
    //
    // Block size = 32 FP4 elements → K=64 = 2 scale blocks.
    //
    // Host packs:
    //   a_packed: 16 rows × 32 bytes (64 FP4 elts/row, 2 nibbles/byte, low first).
    //   b_packed:  8 cols × 32 bytes (64 FP4 elts/col), col-major.
    //   a_scales: 16 rows × 2 bytes (one E8M0 per 32-elts block).
    //   b_scales:  8 cols × 2 bytes.
    //
    // One CTA = one warp (32 threads). Grid: (1,1,1).
    #[kernel]
    pub unsafe fn fp4_mxf4_smoke(
        a_packed: &[u8],             // 512 bytes: 16 × 32
        b_packed: &[u8],             // 256 bytes:  8 × 32
        a_scales: &[u8],             //  32 bytes: 16 × 2
        b_scales: &[u8],             //  16 bytes:  8 × 2
        mut out: DisjointSlice<f32>, // 16×8 = 128 f32
    ) {
        static mut SMEM_A: SharedArray<u8, 512, 32> = SharedArray::UNINIT;
        static mut SMEM_B: SharedArray<u8, 256, 32> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;

        // Stage A (512 B) and B (256 B) into shared memory.
        unsafe {
            let a_dst = &raw mut SMEM_A as *mut u8;
            let b_dst = &raw mut SMEM_B as *mut u8;
            let mut i = tid;
            while i < 512 {
                *a_dst.add(i) = a_packed[i];
                i += 32;
            }
            // Host B is packed by logical column: 8 columns × 32 bytes/column.
            // ldmatrix consumes an 8-column b16 tile, so stage it as
            // [k / 4, n] = 16 rows × 8 b16 columns in shared memory.
            let mut i = tid;
            while i < 128 {
                let k4 = i / 8;
                let col = i & 7;
                let src = col * 32 + k4 * 2;
                let dst = k4 * 16 + col * 2;
                *b_dst.add(dst) = b_packed[src];
                *b_dst.add(dst + 1) = b_packed[src + 1];
                i += 32;
            }
        }
        thread::sync_threads();

        // Load A as four 8×8 b16 sub-tiles covering [m, k/4]:
        //   q0 rows 0..7  cols 0..7,  q1 rows 8..15 cols 0..7,
        //   q2 rows 0..7  cols 8..15, q3 rows 8..15 cols 8..15.
        let a_frag: [u32; 4] = unsafe {
            let q = tid / 8;
            let row = (tid & 7) + if (q & 1) != 0 { 8 } else { 0 };
            let col_b16 = if q >= 2 { 8 } else { 0 };
            let addr = (&raw const SMEM_A as *const u8).add(row * 32 + col_b16 * 2) as *const u32;
            cuda_device::wmma::ldmatrix_x4(addr)
        };

        // Load B as two 8×8 b16 sub-tiles covering [k/4, n]. The .trans
        // form supplies the column-major B fragment expected by row.col MMA.
        let b_frag: [u32; 2] = unsafe {
            let row = tid & 0x0F;
            let addr = (&raw const SMEM_B as *const u8).add(row * 16) as *const u32;
            cuda_device::wmma::ldmatrix_x2_trans(addr)
        };

        // Scale factors: u32 packing two E8M0 block scales (block 0 low byte, block 1 high byte).
        // All lanes supply the same value. Use row 0 / col 0 scales.
        let scale_a: u32 = (a_scales[1] as u32) << 8 | (a_scales[0] as u32);
        let scale_b: u32 = (b_scales[1] as u32) << 8 | (b_scales[0] as u32);

        // D = A × B (accumulator starts at zero).
        let d = unsafe {
            mma_m16n8k64_mxf4_f32_e2m1_e2m1_b0_t0_b0_t0(
                [0.0f32; 4],
                a_frag,
                b_frag,
                scale_a,
                scale_b,
            )
        };

        // Write 16×8 result. m16n8k64 C fragment layout (same as m16n8k*):
        //   lane: row = (lane/4) + (j>=2 ? 8 : 0), col = (lane%4)*2 + (j&1).
        let lane = tid;
        let group = lane / 4;
        let thr = lane % 4;
        let out_ptr = out.as_mut_ptr();
        let mut j = 0usize;
        while j < 4 {
            let row = group + if j >= 2 { 8 } else { 0 };
            let col = thr * 2 + (j & 1);
            unsafe {
                *out_ptr.add(row * 8 + col) = d[j];
            }
            j += 1;
        }
    }

    /// Full 16×8×64 mxf4 tile with per-row/per-column E8M0 scales.
    ///
    /// For `.scale_vec::2X`, selector `{byte-id=0, thread-id=0}` reads the lower
    /// two scale bytes from the selected lanes. A uses lanes `%4 == 0 || 1`
    /// within each quad, giving two row scale vectors per quad; B uses lane
    /// `%4 == 0`, giving one column scale vector per quad. Each scale register
    /// packs K-block 0 in byte 0 and K-block 1 in byte 1.
    #[kernel]
    pub unsafe fn fp4_mxf4_full_tile(
        a_packed: &[u8],             // 512 bytes: 16 × 32
        b_packed: &[u8],             // 256 bytes:  8 × 32
        a_scales: &[u8],             //  32 bytes: 16 × 2
        b_scales: &[u8],             //  16 bytes:  8 × 2
        mut out: DisjointSlice<f32>, // 16×8 = 128 f32
    ) {
        static mut SMEM_A: SharedArray<u8, 512, 32> = SharedArray::UNINIT;
        static mut SMEM_B: SharedArray<u8, 256, 32> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        unsafe {
            let a_dst = &raw mut SMEM_A as *mut u8;
            let b_dst = &raw mut SMEM_B as *mut u8;
            let mut i = tid;
            while i < 512 {
                *a_dst.add(i) = a_packed[i];
                i += 32;
            }
            let mut i = tid;
            while i < 128 {
                let k4 = i / 8;
                let col = i & 7;
                let src = col * 32 + k4 * 2;
                let dst = k4 * 16 + col * 2;
                *b_dst.add(dst) = b_packed[src];
                *b_dst.add(dst + 1) = b_packed[src + 1];
                i += 32;
            }
        }
        thread::sync_threads();

        let a_frag: [u32; 4] = unsafe {
            let q = tid / 8;
            let row = (tid & 7) + if (q & 1) != 0 { 8 } else { 0 };
            let col_b16 = if q >= 2 { 8 } else { 0 };
            let addr = (&raw const SMEM_A as *const u8).add(row * 32 + col_b16 * 2) as *const u32;
            cuda_device::wmma::ldmatrix_x4(addr)
        };
        let b_frag: [u32; 2] = unsafe {
            let row = tid & 0x0F;
            let addr = (&raw const SMEM_B as *const u8).add(row * 16) as *const u32;
            cuda_device::wmma::ldmatrix_x2_trans(addr)
        };

        let group = tid / 4;
        let row_for_scale = group + if (tid & 1) != 0 { 8 } else { 0 };
        let col_for_scale = group;
        let scale_a: u32 =
            (a_scales[row_for_scale * 2 + 1] as u32) << 8 | (a_scales[row_for_scale * 2] as u32);
        let scale_b: u32 =
            (b_scales[col_for_scale * 2 + 1] as u32) << 8 | (b_scales[col_for_scale * 2] as u32);

        let d = unsafe {
            mma_m16n8k64_mxf4_f32_e2m1_e2m1_b0_t0_b0_t0(
                [0.0f32; 4],
                a_frag,
                b_frag,
                scale_a,
                scale_b,
            )
        };

        let lane = tid;
        let group = lane / 4;
        let thr = lane % 4;
        let out_ptr = out.as_mut_ptr();
        let mut j = 0usize;
        while j < 4 {
            let row = group + if j >= 2 { 8 } else { 0 };
            let col = thr * 2 + (j & 1);
            unsafe {
                *out_ptr.add(row * 8 + col) = d[j];
            }
            j += 1;
        }
    }

    /// One 8-row FP4 GEMV tile using mxf4 MMA.
    ///
    /// Computes `out[0..8] = A[8×64] × x[64]` using the lower 8 rows and col0
    /// of one `m16n8k64` MMA. This is the decode-batch=1 building block for the
    /// first MoE tensor-core path; it intentionally wastes the upper 8 rows and
    /// N=7 columns until a wider batching/speculation path can use them.
    #[kernel]
    pub unsafe fn fp4_mxf4_gemv8_tile(
        a_packed: &[u8],             // 256 bytes: 8 rows × 32
        x_packed: &[u8],             //  32 bytes: one 64-element vector
        a_scales: &[u8],             //  16 bytes: 8 rows × 2 K-block scales
        x_scales: &[u8],             //   2 bytes: one vector × 2 K-block scales
        mut out: DisjointSlice<f32>, // 8 f32
    ) {
        static mut SMEM_A: SharedArray<u8, 512, 32> = SharedArray::UNINIT;
        static mut SMEM_B: SharedArray<u8, 256, 32> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;

        unsafe {
            let a_dst = &raw mut SMEM_A as *mut u8;
            let b_dst = &raw mut SMEM_B as *mut u8;
            let mut i = tid;
            while i < 512 {
                *a_dst.add(i) = if i < 256 { a_packed[i] } else { 0 };
                i += 32;
            }
            let mut i = tid;
            while i < 128 {
                let k4 = i / 8;
                let col = i & 7;
                let src = k4 * 2;
                let dst = k4 * 16 + col * 2;
                *b_dst.add(dst) = x_packed[src];
                *b_dst.add(dst + 1) = x_packed[src + 1];
                i += 32;
            }
        }
        thread::sync_threads();

        let a_frag: [u32; 4] = unsafe {
            let q = tid / 8;
            let row = (tid & 7) + if (q & 1) != 0 { 8 } else { 0 };
            let col_b16 = if q >= 2 { 8 } else { 0 };
            let addr = (&raw const SMEM_A as *const u8).add(row * 32 + col_b16 * 2) as *const u32;
            cuda_device::wmma::ldmatrix_x4(addr)
        };
        let b_frag: [u32; 2] = unsafe {
            let row = tid & 0x0F;
            let addr = (&raw const SMEM_B as *const u8).add(row * 16) as *const u32;
            cuda_device::wmma::ldmatrix_x2_trans(addr)
        };

        // Probe result: with the b0_t0 selector, lower output row r consumes
        // scale data from lane r*4. Give every lane in the row's 4-lane group
        // the same A scale so the mapping is robust to future minor variations.
        let scale_row = tid / 4;
        let scale_a: u32 =
            (a_scales[scale_row * 2 + 1] as u32) << 8 | (a_scales[scale_row * 2] as u32);
        let scale_b: u32 = (x_scales[1] as u32) << 8 | (x_scales[0] as u32);
        let d = unsafe {
            mma_m16n8k64_mxf4_f32_e2m1_e2m1_b0_t0_b0_t0(
                [0.0f32; 4],
                a_frag,
                b_frag,
                scale_a,
                scale_b,
            )
        };

        if (tid & 3) == 0 {
            let row = tid / 4;
            unsafe {
                *out.as_mut_ptr().add(row) = d[0];
            }
        }
    }

    /// Scale-selector ABI probe for `mma_m16n8k64_mxf4...b0_t0_b0_t0`.
    ///
    /// `lane_a_scales` and `lane_b_scales` are `[32, 2]` E8M0 bytes. Each lane
    /// supplies its own packed two-block scale register; the output reveals which
    /// lane/byte the fixed `{byte=0, thread=0}` selectors actually consume.
    #[kernel]
    pub unsafe fn fp4_mxf4_scale_lane_probe(
        a_packed: &[u8],             // 512 bytes: 16 × 32
        b_packed: &[u8],             // 256 bytes:  8 × 32
        lane_a_scales: &[u8],        //  64 bytes: 32 lanes × 2 K-block scales
        lane_b_scales: &[u8],        //  64 bytes: 32 lanes × 2 K-block scales
        mut out: DisjointSlice<f32>, // 16×8 = 128 f32
    ) {
        static mut SMEM_A: SharedArray<u8, 512, 32> = SharedArray::UNINIT;
        static mut SMEM_B: SharedArray<u8, 256, 32> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;

        unsafe {
            let a_dst = &raw mut SMEM_A as *mut u8;
            let b_dst = &raw mut SMEM_B as *mut u8;
            let mut i = tid;
            while i < 512 {
                *a_dst.add(i) = a_packed[i];
                i += 32;
            }
            let mut i = tid;
            while i < 128 {
                let k4 = i / 8;
                let col = i & 7;
                let src = col * 32 + k4 * 2;
                let dst = k4 * 16 + col * 2;
                *b_dst.add(dst) = b_packed[src];
                *b_dst.add(dst + 1) = b_packed[src + 1];
                i += 32;
            }
        }
        thread::sync_threads();

        let a_frag: [u32; 4] = unsafe {
            let q = tid / 8;
            let row = (tid & 7) + if (q & 1) != 0 { 8 } else { 0 };
            let col_b16 = if q >= 2 { 8 } else { 0 };
            let addr = (&raw const SMEM_A as *const u8).add(row * 32 + col_b16 * 2) as *const u32;
            cuda_device::wmma::ldmatrix_x4(addr)
        };
        let b_frag: [u32; 2] = unsafe {
            let row = tid & 0x0F;
            let addr = (&raw const SMEM_B as *const u8).add(row * 16) as *const u32;
            cuda_device::wmma::ldmatrix_x2_trans(addr)
        };

        let scale_a: u32 =
            (lane_a_scales[tid * 2 + 1] as u32) << 8 | (lane_a_scales[tid * 2] as u32);
        let scale_b: u32 =
            (lane_b_scales[tid * 2 + 1] as u32) << 8 | (lane_b_scales[tid * 2] as u32);
        let d = unsafe {
            mma_m16n8k64_mxf4_f32_e2m1_e2m1_b0_t0_b0_t0(
                [0.0f32; 4],
                a_frag,
                b_frag,
                scale_a,
                scale_b,
            )
        };

        let lane = tid;
        let group = lane / 4;
        let thr = lane % 4;
        let out_ptr = out.as_mut_ptr();
        let mut j = 0usize;
        while j < 4 {
            let row = group + if j >= 2 { 8 } else { 0 };
            let col = thr * 2 + (j & 1);
            unsafe {
                *out_ptr.add(row * 8 + col) = d[j];
            }
            j += 1;
        }
    }
}
