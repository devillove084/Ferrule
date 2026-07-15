//! CUDA kernels — compiled by cargo-oxide's rustc-codegen-cuda backend.
//!
//! Most GEMV kernels use 4x inner-loop unrolling for ILP.
//! Device math uses Rust float intrinsics so cuda-oxide lowers them through
//! libdevice/NVVM, matching the upstream examples and avoiding arch-specific
//! inline-PTX JIT issues on newer GPUs.

#[cfg(ferrule_cuda_cuda_oxide_bf16_mma)]
use cuda_device::wmma::mma_m16n8k16_f32_bf16;
#[cfg(ferrule_cuda_blackwell_mma_sync_fp8)]
use cuda_device::wmma::mma_m16n8k32_f32_e4m3_e4m3;
#[cfg(ferrule_cuda_blackwell_mma_sync_mxfp4)]
use cuda_device::wmma::mma_m16n8k64_mxf4_f32_e2m1_e2m1_b0_t0_b0_t0;
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, ptx_asm, thread};

// Keep the generated module's host API stable on portable targets. Provider
// selection never launches these kernels unless the corresponding codegen
// path is compiled; these helpers remove unavailable intrinsics from device IR.
#[cfg(not(ferrule_cuda_blackwell_mma_sync_fp8))]
unsafe fn mma_m16n8k32_f32_e4m3_e4m3(c: [f32; 4], _a: [u32; 4], _b: [u32; 2]) -> [f32; 4] {
    c
}

#[cfg(not(ferrule_cuda_cuda_oxide_bf16_mma))]
unsafe fn mma_m16n8k16_f32_bf16(c: [f32; 4], _a: [u32; 4], _b: [u32; 2]) -> [f32; 4] {
    c
}

#[cfg(not(ferrule_cuda_blackwell_mma_sync_mxfp4))]
unsafe fn mma_m16n8k64_mxf4_f32_e2m1_e2m1_b0_t0_b0_t0(
    c: [f32; 4],
    _a: [u32; 4],
    _b: [u32; 2],
    _scale_a: u32,
    _scale_b: u32,
) -> [f32; 4] {
    c
}

const LN_2_F32: f32 = core::f32::consts::LN_2;
pub(crate) const DSV4_DECODE_INDEX_QUERY_SHARED_ELEMENTS: usize = 8192;

/// Device exp(x). cuda-oxide lowers this to libdevice (`__nv_expf`) on GPU.
fn fast_exp(x: f32) -> f32 {
    libm::expf(x)
}

#[allow(clippy::too_many_arguments)]
fn paged_plane_row_offset(
    plane_elements: usize,
    block_slots: &[i32],
    block_offsets: &[i32],
    sequence: usize,
    logical_row: usize,
    page_tokens: usize,
    elements_per_token: usize,
    layer_index: usize,
    layer_count: usize,
) -> usize {
    if page_tokens == 0 || elements_per_token == 0 || layer_index >= layer_count {
        return usize::MAX;
    }
    let start = match block_offsets.get(sequence) {
        Some(value) if *value >= 0 => *value as usize,
        _ => return usize::MAX,
    };
    let end = match block_offsets.get(sequence + 1) {
        Some(value) if *value >= 0 => *value as usize,
        _ => return usize::MAX,
    };
    let entry = match start.checked_add(logical_row / page_tokens) {
        Some(value) if value < end => value,
        _ => return usize::MAX,
    };
    let slot = match block_slots.get(entry) {
        Some(value) if *value >= 0 => *value as u64,
        _ => return usize::MAX,
    };
    let page_tokens = page_tokens as u64;
    let width = elements_per_token as u64;
    let slot_stride = match (layer_count as u64)
        .checked_mul(page_tokens)
        .and_then(|value| value.checked_mul(width))
    {
        Some(value) => value,
        None => return usize::MAX,
    };
    let layer_stride = match page_tokens.checked_mul(width) {
        Some(value) => value,
        None => return usize::MAX,
    };
    let offset = match slot
        .checked_mul(slot_stride)
        .and_then(|value| value.checked_add(layer_index as u64 * layer_stride))
        .and_then(|value| value.checked_add((logical_row % page_tokens as usize) as u64 * width))
    {
        Some(value) => value,
        None => return usize::MAX,
    };
    match offset.checked_add(width) {
        Some(end) if end <= plane_elements as u64 => offset as usize,
        _ => usize::MAX,
    }
}

#[allow(clippy::too_many_arguments)]
fn paged_row_sequence(
    row_sequence_ids: &[i32],
    row: usize,
    default_sequence: usize,
    use_row_sequence_ids: u32,
) -> usize {
    if use_row_sequence_ids == 0 {
        default_sequence
    } else {
        row_sequence_ids
            .get(row)
            .and_then(|value| usize::try_from(*value).ok())
            .unwrap_or(usize::MAX)
    }
}

fn paged_row_visible_token(
    logical_token: i32,
    row_kv_lens: &[i32],
    row: usize,
    use_row_kv_lens: u32,
) -> i32 {
    if use_row_kv_lens == 0 {
        return logical_token;
    }
    match row_kv_lens.get(row) {
        Some(visible_len) if logical_token >= 0 && logical_token < *visible_len => logical_token,
        _ => -1,
    }
}

#[allow(clippy::too_many_arguments)]
fn paged_sparse_kv_offset(
    block_slots: &[i32],
    sequence_block_offsets: &[i32],
    sequence_kv_lens: &[i32],
    sequence: usize,
    logical_token: i32,
    page_tokens: usize,
    elements_per_token: usize,
    head_dim: usize,
    layer_index: usize,
    layer_count: usize,
    plane_elements: usize,
) -> usize {
    if logical_token < 0
        || page_tokens == 0
        || elements_per_token < head_dim
        || layer_index >= layer_count
    {
        return usize::MAX;
    }
    let kv_len = match sequence_kv_lens.get(sequence) {
        Some(value) if *value >= 0 => *value as usize,
        _ => return usize::MAX,
    };
    let logical_token = logical_token as usize;
    if logical_token >= kv_len {
        return usize::MAX;
    }
    let block_start = match sequence_block_offsets.get(sequence) {
        Some(value) if *value >= 0 => *value as usize,
        _ => return usize::MAX,
    };
    let block_end = match sequence_block_offsets.get(sequence + 1) {
        Some(value) if *value >= 0 => *value as usize,
        _ => return usize::MAX,
    };
    let block_entry = match block_start.checked_add(logical_token / page_tokens) {
        Some(value) => value,
        None => return usize::MAX,
    };
    if block_entry >= block_end {
        return usize::MAX;
    }
    let physical_slot = match block_slots.get(block_entry) {
        Some(value) if *value >= 0 => *value as u64,
        _ => return usize::MAX,
    };
    let page_tokens = page_tokens as u64;
    let elements_per_token = elements_per_token as u64;
    let slot_stride = match (layer_count as u64)
        .checked_mul(page_tokens)
        .and_then(|value| value.checked_mul(elements_per_token))
    {
        Some(value) => value,
        None => return usize::MAX,
    };
    let layer_stride = match page_tokens.checked_mul(elements_per_token) {
        Some(value) => value,
        None => return usize::MAX,
    };
    let token_in_page = (logical_token % page_tokens as usize) as u64;
    let offset = match physical_slot
        .checked_mul(slot_stride)
        .and_then(|value| value.checked_add(layer_index as u64 * layer_stride))
        .and_then(|value| value.checked_add(token_in_page * elements_per_token))
    {
        Some(value) => value,
        None => return usize::MAX,
    };
    let end = match offset.checked_add(head_dim as u64) {
        Some(value) => value,
        None => return usize::MAX,
    };
    if end > plane_elements as u64 {
        usize::MAX
    } else {
        offset as usize
    }
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

/// Fast sqrt. Use PTX directly to avoid host-arch inline asm emitted by some
/// libm sqrt paths before cuda-oxide lowers math to device code.
fn fast_sqrt(x: f32) -> f32 {
    let result: f32;
    unsafe {
        ptx_asm!(
            "sqrt.rn.f32 %0, %1;",
            out("=f") result,
            in("f") x,
            options(register_only),
        );
    }
    result
}

/// Global i32 atomic add expressed as PTX so it remains legal in both the
/// legacy pre-Blackwell NVVM dialect and the modern Blackwell path.
fn atomic_fetch_add_i32(ptr: *mut i32, value: i32) -> i32 {
    let old: u32;
    unsafe {
        ptx_asm!(
            "atom.global.add.u32 %0, [%1], %2;",
            out("=r") old,
            in("l") ptr as u64,
            in("r") value as u32,
            clobber("memory"),
        );
    }
    old as i32
}

/// Global i32 atomic OR with the same cross-architecture lowering contract as
/// [`atomic_fetch_add_i32`].
fn atomic_fetch_or_i32(ptr: *mut i32, value: i32) -> i32 {
    let old: u32;
    unsafe {
        ptx_asm!(
            "atom.global.or.b32 %0, [%1], %2;",
            out("=r") old,
            in("l") ptr as u64,
            in("r") value as u32,
            clobber("memory"),
        );
    }
    old as i32
}

fn stable_softplus_f32(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        fast_exp(x)
    } else {
        libm::log1pf(fast_exp(x))
    }
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

#[inline(always)]
fn quantize_fp8_e4m3fn_byte(value: f32) -> u8 {
    let sign = if value.to_bits() & 0x8000_0000 != 0 {
        0x80
    } else {
        0
    };
    if value == 0.0 {
        return sign;
    }
    if !value.is_finite() {
        return sign | 0x7f;
    }
    let abs_value = if value < 0.0 { -value } else { value };
    let magnitude = if abs_value > 448.0 { 448.0 } else { abs_value };
    let quantized = nearest_fp8_e4m3fn_positive(magnitude);
    if quantized < 1.0 / 64.0 {
        let mantissa = libm::roundf(quantized * 512.0) as u8;
        return sign | if mantissa > 7 { 7 } else { mantissa };
    }
    let bits = quantized.to_bits();
    let exponent = (((bits >> 23) & 0xff) as i32 - 127 + 7) as u8;
    let mantissa = ((bits >> 20) & 0x07) as u8;
    sign | (exponent << 3) | mantissa
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
    if x > max_value { max_value } else { x }
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

    #[kernel]
    pub fn fill_i32_sequence(mut output: DisjointSlice<i32>, start: i32, len: u32) {
        let index = thread::index_1d().get();
        if index >= len as usize {
            return;
        }
        if let Some(value) = output.get_mut(thread::index_1d()) {
            *value = start.saturating_add(index as i32);
        }
    }

    #[kernel]
    pub fn pack_i32_f32_pairs(
        indices: &[i32],
        weights: &[f32],
        mut output: DisjointSlice<i32>,
        pair_count: u32,
    ) {
        let output_index = thread::index_1d().get();
        if output_index >= pair_count as usize * 2 {
            return;
        }
        let pair = output_index / 2;
        let value = if output_index % 2 == 0 {
            indices[pair]
        } else {
            weights[pair].to_bits() as i32
        };
        if let Some(output) = output.get_mut(thread::index_1d()) {
            *output = value;
        }
    }

    #[kernel]
    pub fn fill_dsv4_paged_window_topk(
        mut output: DisjointSlice<i32>,
        start: u32,
        valid_len: u32,
        output_len: u32,
    ) {
        let index = thread::index_1d().get();
        if index >= output_len as usize {
            return;
        }
        let value = if index < valid_len as usize {
            start.saturating_add(index as u32) as i32
        } else {
            -1
        };
        if let Some(output) = output.get_mut(thread::index_1d()) {
            *output = value;
        }
    }

    #[kernel]
    pub fn fill_dsv4_decode_attention_topk(
        mut output: DisjointSlice<i32>,
        position: u32,
        window_size: u32,
        window_len: u32,
        compressed_len: u32,
        output_len: u32,
    ) {
        let index = thread::index_1d().get();
        if index >= output_len as usize || window_size == 0 {
            return;
        }
        let window_size_usize = window_size as usize;
        let value = if index < window_size_usize {
            if index >= window_len as usize {
                -1
            } else if window_len < window_size {
                index as i32
            } else {
                ((position as usize % window_size_usize + 1 + index) % window_size_usize) as i32
            }
        } else {
            let compressed_index = index - window_size_usize;
            if compressed_index < compressed_len as usize {
                index as i32
            } else {
                -1
            }
        };
        if let Some(output) = output.get_mut(thread::index_1d()) {
            *output = value;
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

    /// Batched grouped matvec for block-diagonal output-A layouts.
    ///
    /// `context` is `[rows, o_groups * group_in]`, `weight` is
    /// `[output_latent_dim, group_in]`, and `output` is
    /// `[rows, output_latent_dim]`. This is the prefill projection shape: all
    /// token rows share the same grouped WO-A weights and differ only in the
    /// context row.
    #[kernel]
    pub fn grouped_matvec_f32_rows(
        context: &[f32],
        weight: &[f32],
        mut output: DisjointSlice<f32>,
        rows: u32,
        output_latent_dim: u32,
        group_in: u32,
        o_lora_rank: u32,
    ) {
        let i = thread::index_1d().get();
        let total = rows as u64 * output_latent_dim as u64;
        if (i as u64) >= total {
            return;
        }
        let rows = rows as usize;
        let output_latent_dim = output_latent_dim as usize;
        let group_in = group_in as usize;
        let o_lora_rank = o_lora_rank as usize;
        if rows == 0 || o_lora_rank == 0 || output_latent_dim == 0 {
            return;
        }
        let groups = output_latent_dim / o_lora_rank;
        let token = i as usize / output_latent_dim;
        let out_row = i as usize - token * output_latent_dim;
        let group = out_row / o_lora_rank;
        let context_start = token * groups * group_in + group * group_in;
        let weight_start = out_row * group_in;
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

    /// DSV4 grouped WO-A using the official BF16 einsum semantics and a
    /// `m16n8k16` Tensor Core tile. Checkpoint weights stay in FP8+E8M0 form;
    /// each weight tile is dequantized directly to BF16 in shared memory.
    ///
    /// Grid: `(ceil(output_latent_dim / 16), ceil(rows / 8))`, one warp per CTA.
    #[kernel]
    pub unsafe fn grouped_output_a_bf16_mma_from_fp8(
        context: &[f32],
        weight: &[u8],
        weight_scales: &[u8],
        mut output: DisjointSlice<f32>,
        rows: u32,
        output_latent_dim: u32,
        group_in: u32,
        o_lora_rank: u32,
        scale_cols: u32,
    ) {
        static mut SMEM_A: SharedArray<u8, 512, 32> = SharedArray::UNINIT;
        static mut SMEM_B: SharedArray<u8, 256, 32> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let out_base = thread::blockIdx_x() as usize * 16;
        let token_base = thread::blockIdx_y() as usize * 8;
        let rows = rows as usize;
        let output_latent_dim = output_latent_dim as usize;
        let group_in = group_in as usize;
        let o_lora_rank = o_lora_rank as usize;
        let scale_cols = scale_cols as usize;
        if rows == 0
            || output_latent_dim == 0
            || group_in == 0
            || o_lora_rank == 0
            || scale_cols == 0
            || !group_in.is_multiple_of(16)
        {
            return;
        }
        let groups = output_latent_dim / o_lora_rank;
        let group = out_base / o_lora_rank;
        let mut acc = [0.0f32; 4];
        let mut k_base = 0usize;
        while k_base < group_in {
            let weight_scale = if out_base < output_latent_dim {
                e8m0_scale(weight_scales[(out_base / 128) * scale_cols + k_base / 128])
            } else {
                1.0
            };
            unsafe {
                let a_dst = &raw mut SMEM_A as *mut u16;
                let b_dst = &raw mut SMEM_B as *mut u16;
                let mut i = tid;
                while i < 256 {
                    let row_local = i / 16;
                    let k_local = i & 15;
                    let out_row = out_base + row_local;
                    let value = if out_row < output_latent_dim {
                        fp8_e4m3fn_value(weight[out_row * group_in + k_base + k_local])
                            * weight_scale
                    } else {
                        0.0
                    };
                    *a_dst.add(i) = cuda_device::tcgen05::f32_to_bf16_rne(value);
                    i += 32;
                }
                let mut i = tid;
                while i < 128 {
                    let k_local = i / 8;
                    let col = i & 7;
                    let value = if token_base + col < rows {
                        context[(token_base + col) * groups * group_in
                            + group * group_in
                            + k_base
                            + k_local]
                    } else {
                        0.0
                    };
                    *b_dst.add(i) = cuda_device::tcgen05::f32_to_bf16_rne(value);
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
                let row = tid & 0x0f;
                let addr = (&raw const SMEM_B as *const u8).add(row * 16) as *const u32;
                cuda_device::wmma::ldmatrix_x2_trans(addr)
            };
            acc = unsafe { mma_m16n8k16_f32_bf16(acc, a_frag, b_frag) };
            thread::sync_threads();
            k_base += 16;
        }

        let lane_group = tid / 4;
        let lane_col = tid & 3;
        let out_ptr = output.as_mut_ptr();
        let mut j = 0usize;
        while j < 4 {
            let out_row = out_base + lane_group + if j >= 2 { 8 } else { 0 };
            let col = lane_col * 2 + (j & 1);
            if out_row < output_latent_dim && token_base + col < rows {
                let rounded = cuda_device::tcgen05::f32_to_bf16_rne(acc[j]);
                unsafe {
                    *out_ptr.add((token_base + col) * output_latent_dim + out_row) =
                        f32::from_bits((rounded as u32) << 16);
                }
            }
            j += 1;
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

    /// Two independent legacy-order BF16 GEMVs over the same activation in one
    /// launch. Each output thread preserves `gemv_bf16_bytes` accumulation order.
    #[kernel]
    pub fn gemv_bf16_bytes_pair(
        x: &[f32],
        first_w: &[u8],
        second_w: &[u8],
        mut first_y: DisjointSlice<f32>,
        mut second_y: DisjointSlice<f32>,
        first_n: u32,
        second_n: u32,
        k: u32,
    ) {
        let combined_row = thread::index_1d().get();
        let first_n = first_n as usize;
        let second_n = second_n as usize;
        if combined_row >= first_n + second_n {
            return;
        }
        let (w, row) = if combined_row < first_n {
            (first_w, combined_row)
        } else {
            (second_w, combined_row - first_n)
        };
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
        unsafe {
            if combined_row < first_n {
                *first_y.as_mut_ptr().add(row) = dot;
            } else {
                *second_y.as_mut_ptr().add(row) = dot;
            }
        }
    }

    /// BF16 GEMV with one 256-thread CTA per output row. The legacy kernel
    /// assigns one complete K reduction to a single thread; this variant
    /// parallelizes that reduction while preserving f32 accumulation.
    #[kernel]
    pub fn gemv_bf16_bytes_block(x: &[f32], w: &[u8], mut y: DisjointSlice<f32>, n: u32, k: u32) {
        static mut SUM: SharedArray<f32, 256> = SharedArray::UNINIT;

        let row = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let bdim = thread::blockDim_x() as usize;
        if row >= n as usize || tid >= 256 {
            return;
        }
        let k = k as usize;
        let base = row * k * 2;
        let mut dot = 0.0f32;
        let mut j = tid;
        while j < k {
            let off = base + j * 2;
            dot += x[j] * bf16_pair_to_f32(w[off], w[off + 1]);
            j += bdim;
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
            unsafe {
                *y.as_mut_ptr().add(row) = SUM[0];
            }
        }
    }

    /// Two independent BF16 GEMVs over the same activation in one launch.
    /// Each output row keeps the exact reduction schedule used by
    /// `gemv_bf16_bytes_block`; the combined grid only removes the second
    /// dispatch and lets both projections share the activation cache footprint.
    #[kernel]
    pub fn gemv_bf16_bytes_block_pair(
        x: &[f32],
        first_w: &[u8],
        second_w: &[u8],
        mut first_y: DisjointSlice<f32>,
        mut second_y: DisjointSlice<f32>,
        first_n: u32,
        second_n: u32,
        k: u32,
    ) {
        static mut SUM: SharedArray<f32, 256> = SharedArray::UNINIT;

        let combined_row = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let bdim = thread::blockDim_x() as usize;
        let first_n = first_n as usize;
        let second_n = second_n as usize;
        if combined_row >= first_n + second_n || tid >= 256 {
            return;
        }
        let (w, row) = if combined_row < first_n {
            (first_w, combined_row)
        } else {
            (second_w, combined_row - first_n)
        };
        let k = k as usize;
        let base = row * k * 2;
        let mut dot = 0.0f32;
        let mut j = tid;
        while j < k {
            let off = base + j * 2;
            dot += x[j] * bf16_pair_to_f32(w[off], w[off + 1]);
            j += bdim;
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
            unsafe {
                if combined_row < first_n {
                    *first_y.as_mut_ptr().add(row) = SUM[0];
                } else {
                    *second_y.as_mut_ptr().add(row) = SUM[0];
                }
            }
        }
    }

    #[kernel]
    pub fn gemm_f32_bytes(
        x: &[f32],
        w: &[u8],
        mut y: DisjointSlice<f32>,
        batch: u32,
        n: u32,
        k: u32,
    ) {
        let idx = thread::index_1d().get();
        let total = batch as u64 * n as u64;
        if (idx as u64) >= total {
            return;
        }
        let b = (idx as u32 / n) as usize;
        let row = (idx as u32 % n) as usize;
        let k = k as usize;
        let x_base = b * k;
        let w_base = row * k * 4;
        let mut dot0 = 0.0f32;
        let mut dot1 = 0.0f32;
        let mut dot2 = 0.0f32;
        let mut dot3 = 0.0f32;
        let mut j = 0usize;
        let k4 = k - k % 4;
        while j < k4 {
            let o0 = w_base + 4 * j;
            let o1 = w_base + 4 * (j + 1);
            let o2 = w_base + 4 * (j + 2);
            let o3 = w_base + 4 * (j + 3);
            dot0 += x[x_base + j] * f32_bytes_to_f32(w[o0], w[o0 + 1], w[o0 + 2], w[o0 + 3]);
            dot1 += x[x_base + j + 1] * f32_bytes_to_f32(w[o1], w[o1 + 1], w[o1 + 2], w[o1 + 3]);
            dot2 += x[x_base + j + 2] * f32_bytes_to_f32(w[o2], w[o2 + 1], w[o2 + 2], w[o2 + 3]);
            dot3 += x[x_base + j + 3] * f32_bytes_to_f32(w[o3], w[o3 + 1], w[o3 + 2], w[o3 + 3]);
            j += 4;
        }
        let mut dot = dot0 + dot1 + dot2 + dot3;
        while j < k {
            let off = w_base + 4 * j;
            dot += x[x_base + j] * f32_bytes_to_f32(w[off], w[off + 1], w[off + 2], w[off + 3]);
            j += 1;
        }
        if let Some(yi) = y.get_mut(thread::index_1d()) {
            *yi = dot;
        }
    }

    #[kernel]
    pub fn gemm_bf16_bytes(
        x: &[f32],
        w: &[u8],
        mut y: DisjointSlice<f32>,
        batch: u32,
        n: u32,
        k: u32,
    ) {
        let idx = thread::index_1d().get();
        let total = batch as u64 * n as u64;
        if (idx as u64) >= total {
            return;
        }
        let b = (idx as u32 / n) as usize;
        let row = (idx as u32 % n) as usize;
        let k = k as usize;
        let x_base = b * k;
        let w_base = row * k * 2;
        let mut dot0 = 0.0f32;
        let mut dot1 = 0.0f32;
        let mut dot2 = 0.0f32;
        let mut dot3 = 0.0f32;
        let mut j = 0usize;
        let k4 = k - k % 4;
        while j < k4 {
            let o0 = w_base + 2 * j;
            let o1 = w_base + 2 * (j + 1);
            let o2 = w_base + 2 * (j + 2);
            let o3 = w_base + 2 * (j + 3);
            dot0 += x[x_base + j] * bf16_pair_to_f32(w[o0], w[o0 + 1]);
            dot1 += x[x_base + j + 1] * bf16_pair_to_f32(w[o1], w[o1 + 1]);
            dot2 += x[x_base + j + 2] * bf16_pair_to_f32(w[o2], w[o2 + 1]);
            dot3 += x[x_base + j + 3] * bf16_pair_to_f32(w[o3], w[o3 + 1]);
            j += 4;
        }
        let mut dot = dot0 + dot1 + dot2 + dot3;
        while j < k {
            let off = w_base + 2 * j;
            dot += x[x_base + j] * bf16_pair_to_f32(w[off], w[off + 1]);
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

    /// Simulate DSV4 attention KV QAT quantization in-place: each row is
    /// `[non_rope | rope_tail]`; only the non-rope prefix is FP8-quantized.
    #[kernel]
    pub fn fp8_e4m3fn_e8m0_quantize_non_rope_f32_inplace(
        mut values: DisjointSlice<f32>,
        value_len: u32,
        head_dim: u32,
        rope_dim: u32,
        block_size: u32,
    ) {
        let block_idx = thread::index_1d().get() as usize;
        let value_len = value_len as usize;
        let head_dim = head_dim as usize;
        let rope_dim = rope_dim as usize;
        let block_size = block_size as usize;
        if head_dim == 0 || rope_dim > head_dim || block_size == 0 {
            return;
        }
        let non_rope = head_dim - rope_dim;
        if non_rope == 0 || value_len == 0 || !value_len.is_multiple_of(head_dim) {
            return;
        }
        let effective_block_size = if non_rope.is_multiple_of(block_size) {
            block_size
        } else {
            non_rope
        };
        let blocks_per_row = (non_rope + effective_block_size - 1) / effective_block_size;
        let row = block_idx / blocks_per_row;
        let rows = value_len / head_dim;
        if row >= rows {
            return;
        }
        let block = block_idx % blocks_per_row;
        let row_base = row * head_dim;
        let start = row_base + block * effective_block_size;
        let end = (start + effective_block_size).min(row_base + non_rope);
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

    /// Apply a normalized Walsh-Hadamard transform row-wise, then simulate
    /// official block-wise FP4 E2M1 + E8M0 activation quantization in-place.
    ///
    /// This is used by the DSV4 indexer QAT path. One CUDA thread owns one row;
    /// row widths are tiny powers of two (index head dim), so the serial row
    /// transform is good enough and avoids global synchronization complexity.
    #[kernel]
    pub fn hadamard_fp4_e2m1_e8m0_quantize_f32_inplace(
        mut values: DisjointSlice<f32>,
        value_len: u32,
        row_width: u32,
        block_size: u32,
    ) {
        let row = thread::index_1d().get() as usize;
        let value_len = value_len as usize;
        let row_width = row_width as usize;
        let block_size = block_size as usize;
        if row_width == 0
            || block_size == 0
            || !row_width.is_power_of_two()
            || !row_width.is_multiple_of(block_size)
            || value_len == 0
            || !value_len.is_multiple_of(row_width)
        {
            return;
        }
        let rows = value_len / row_width;
        if row >= rows {
            return;
        }
        let base = row * row_width;
        let ptr = values.as_mut_ptr();

        let mut span = 1usize;
        while span < row_width {
            let step = span * 2;
            let mut start = 0usize;
            while start < row_width {
                let mut offset = 0usize;
                while offset < span {
                    let left = base + start + offset;
                    let right = left + span;
                    let a = unsafe { *ptr.add(left) };
                    let b = unsafe { *ptr.add(right) };
                    unsafe {
                        *ptr.add(left) = a + b;
                        *ptr.add(right) = a - b;
                    }
                    offset += 1;
                }
                start += step;
            }
            span = step;
        }
        let scale_h = fast_rsqrt(row_width as f32);
        let mut i = 0usize;
        while i < row_width {
            unsafe {
                *ptr.add(base + i) *= scale_h;
            }
            i += 1;
        }

        let blocks_per_row = row_width / block_size;
        let mut block = 0usize;
        while block < blocks_per_row {
            let start = base + block * block_size;
            let end = start + block_size;
            let mut amax = 6.0 * pow2f(-126.0);
            let mut j = start;
            while j < end {
                let value = unsafe { *ptr.add(j) };
                let abs_value = if value < 0.0 { -value } else { value };
                if abs_value > amax {
                    amax = abs_value;
                }
                j += 1;
            }
            let scale_byte = e8m0_scale_byte_for_amax(amax, 6.0);
            let scale = e8m0_scale(scale_byte);
            let mut j = start;
            while j < end {
                let value = unsafe { *ptr.add(j) };
                let scaled = clamp_range(value / scale, -6.0, 6.0);
                let quantized = fp4_e2m1_value(quantize_fp4_e2m1_nibble(scaled)) * scale;
                unsafe {
                    *ptr.add(j) = quantized;
                }
                j += 1;
            }
            block += 1;
        }
    }

    #[kernel]
    pub fn dsv4_prefill_topk_indices_paged_indexer(
        query: &[f32],
        weights: &[f32],
        indexer_plane: &[f32],
        block_slots: &[i32],
        block_offsets: &[i32],
        out: DisjointSlice<i32>,
        tokens: u32,
        window_size: u32,
        window_cols: u32,
        extra_cols: u32,
        value_offset: u32,
        compress_ratio: u32,
        compressed_len: u32,
        index_heads: u32,
        index_head_dim: u32,
        page_tokens: u32,
        layer_index: u32,
        layer_count: u32,
        weight_scale: f32,
    ) {
        dsv4_prefill_topk_indices_impl(
            query,
            weights,
            indexer_plane,
            block_slots,
            block_offsets,
            out,
            tokens,
            window_size,
            window_cols,
            extra_cols,
            value_offset,
            compress_ratio,
            compressed_len,
            index_heads,
            index_head_dim,
            1,
            page_tokens,
            layer_index,
            layer_count,
            1,
            weight_scale,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn dsv4_prefill_topk_indices_impl(
        query: &[f32],
        weights: &[f32],
        indexer_kv: &[f32],
        block_slots: &[i32],
        block_offsets: &[i32],
        mut out: DisjointSlice<i32>,
        tokens: u32,
        window_size: u32,
        window_cols: u32,
        extra_cols: u32,
        value_offset: u32,
        compress_ratio: u32,
        compressed_len: u32,
        index_heads: u32,
        index_head_dim: u32,
        indexer_enabled: u32,
        page_tokens: u32,
        layer_index: u32,
        layer_count: u32,
        paged_indexer: u32,
        weight_scale: f32,
    ) {
        static mut CANDIDATE_SCORES: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut BEST_SCORES: SharedArray<f32, 512> = SharedArray::UNINIT;
        static mut BEST_INDICES: SharedArray<i32, 512> = SharedArray::UNINIT;

        let token = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let bdim = thread::blockDim_x() as usize;
        let tokens = tokens as usize;
        if token >= tokens || bdim == 0 || bdim > 256 {
            return;
        }
        let window_size = window_size as usize;
        let window_cols = window_cols as usize;
        let extra_cols = extra_cols as usize;
        let value_offset = value_offset as usize;
        let ratio = compress_ratio as usize;
        let compressed_len = compressed_len as usize;
        let heads = index_heads as usize;
        let hd = index_head_dim as usize;
        let total_cols = window_cols + extra_cols;
        let out_ptr = out.as_mut_ptr();

        let first = (token + 1).saturating_sub(window_size);
        let mut col = tid;
        while col < window_cols {
            let idx = first + col;
            let value = if idx <= token { idx as i32 } else { -1 };
            unsafe {
                *out_ptr.add(token * total_cols + col) = value;
            }
            col += bdim;
        }
        if extra_cols == 0 || ratio == 0 {
            return;
        }

        if indexer_enabled == 0 {
            let visible = ((token + 1) / ratio).min(compressed_len);
            let mut slot = tid;
            while slot < extra_cols {
                let value = if slot < visible {
                    (value_offset + slot) as i32
                } else {
                    -1
                };
                unsafe {
                    *out_ptr.add(token * total_cols + window_cols + slot) = value;
                }
                slot += bdim;
            }
            return;
        }

        let visible = ((token + 1) / ratio).min(compressed_len);
        let take = extra_cols.min(512);
        let mut slot = tid;
        while slot < take {
            unsafe {
                BEST_SCORES[slot] = f32::NEG_INFINITY;
                BEST_INDICES[slot] = -1;
            }
            slot += bdim;
        }
        thread::sync_threads();

        let mut chunk_start = 0usize;
        while chunk_start < visible {
            let idx = chunk_start + tid;
            let mut score = f32::NEG_INFINITY;
            if idx < visible {
                let kv_base = if paged_indexer != 0 {
                    paged_plane_row_offset(
                        indexer_kv.len(),
                        block_slots,
                        block_offsets,
                        0,
                        idx,
                        page_tokens as usize,
                        hd,
                        layer_index as usize,
                        layer_count as usize,
                    )
                } else {
                    idx * hd
                };
                if kv_base != usize::MAX {
                    score = 0.0;
                }
                let mut head = 0usize;
                while head < heads && kv_base != usize::MAX {
                    let q_base = token * heads * hd + head * hd;
                    let mut dot = 0.0f32;
                    let mut d = 0usize;
                    while d < hd {
                        dot += query[q_base + d] * indexer_kv[kv_base + d];
                        d += 1;
                    }
                    if dot < 0.0 {
                        dot = 0.0;
                    }
                    score += dot * weights[token * heads + head] * weight_scale;
                    head += 1;
                }
            }
            unsafe {
                CANDIDATE_SCORES[tid] = score;
            }
            thread::sync_threads();

            if tid == 0 {
                let chunk_len = (visible - chunk_start).min(bdim);
                let mut candidate = 0usize;
                while candidate < chunk_len {
                    let candidate_idx = (chunk_start + candidate) as i32;
                    let candidate_score = unsafe { CANDIDATE_SCORES[candidate] };
                    let mut pos = take;
                    while pos > 0 {
                        let prev = pos - 1;
                        let prev_score = unsafe { BEST_SCORES[prev] };
                        let prev_idx = unsafe { BEST_INDICES[prev] };
                        let better = candidate_score > prev_score
                            || (candidate_score == prev_score
                                && (prev_idx < 0 || candidate_idx < prev_idx));
                        if !better {
                            break;
                        }
                        pos -= 1;
                    }
                    if pos < take {
                        let mut move_pos = take - 1;
                        while move_pos > pos {
                            unsafe {
                                BEST_SCORES[move_pos] = BEST_SCORES[move_pos - 1];
                                BEST_INDICES[move_pos] = BEST_INDICES[move_pos - 1];
                            }
                            move_pos -= 1;
                        }
                        unsafe {
                            BEST_SCORES[pos] = candidate_score;
                            BEST_INDICES[pos] = candidate_idx;
                        }
                    }
                    candidate += 1;
                }
            }
            thread::sync_threads();
            chunk_start += bdim;
        }

        slot = tid;
        while slot < extra_cols {
            let (best_idx, best_score) = if slot < take {
                unsafe { (BEST_INDICES[slot], BEST_SCORES[slot]) }
            } else {
                (-1, f32::NEG_INFINITY)
            };
            let value = if best_idx >= 0 && best_score.is_finite() {
                (value_offset + best_idx as usize) as i32
            } else {
                -1
            };
            unsafe {
                *out_ptr.add(token * total_cols + window_cols + slot) = value;
            }
            slot += bdim;
        }
    }

    /// Fused variant for the indexer path: consumes untransformed index query
    /// rows and applies RoPE + Hadamard/FP4-QAT in-register before top-k scoring.
    /// This removes two separate query transform kernels and the intermediate
    /// transformed-query global write/read from DSV4 prefill.
    #[kernel]
    pub fn dsv4_prefill_topk_indices_fused_index_query(
        query: &[f32],
        weights: &[f32],
        indexer_kv: &[f32],
        cos_table: &[f32],
        sin_table: &[f32],
        out: DisjointSlice<i32>,
        tokens: u32,
        window_size: u32,
        window_cols: u32,
        extra_cols: u32,
        value_offset: u32,
        compress_ratio: u32,
        compressed_len: u32,
        index_heads: u32,
        index_head_dim: u32,
        rope_dim: u32,
        start_position: u32,
        weight_scale: f32,
    ) {
        let token = thread::index_1d().get() as usize;
        dsv4_prefill_topk_indices_fused_index_query_impl(
            token,
            query,
            weights,
            indexer_kv,
            &[],
            &[],
            cos_table,
            sin_table,
            out,
            tokens,
            window_size,
            window_cols,
            extra_cols,
            value_offset,
            compress_ratio,
            compressed_len,
            index_heads,
            index_head_dim,
            rope_dim,
            start_position,
            1,
            0,
            1,
            0,
            weight_scale,
        )
    }

    #[kernel]
    pub fn dsv4_prefill_topk_indices_fused_index_query_paged_indexer(
        query: &[f32],
        weights: &[f32],
        indexer_plane: &[f32],
        block_slots: &[i32],
        block_offsets: &[i32],
        cos_table: &[f32],
        sin_table: &[f32],
        out: DisjointSlice<i32>,
        tokens: u32,
        window_size: u32,
        window_cols: u32,
        extra_cols: u32,
        value_offset: u32,
        compress_ratio: u32,
        compressed_len: u32,
        index_heads: u32,
        index_head_dim: u32,
        rope_dim: u32,
        start_position: u32,
        page_tokens: u32,
        layer_index: u32,
        layer_count: u32,
        weight_scale: f32,
    ) {
        let token = thread::index_1d().get() as usize;
        dsv4_prefill_topk_indices_fused_index_query_impl(
            token,
            query,
            weights,
            indexer_plane,
            block_slots,
            block_offsets,
            cos_table,
            sin_table,
            out,
            tokens,
            window_size,
            window_cols,
            extra_cols,
            value_offset,
            compress_ratio,
            compressed_len,
            index_heads,
            index_head_dim,
            rope_dim,
            start_position,
            page_tokens,
            layer_index,
            layer_count,
            1,
            weight_scale,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn dsv4_prefill_topk_indices_fused_index_query_impl(
        token: usize,
        query: &[f32],
        weights: &[f32],
        indexer_kv: &[f32],
        block_slots: &[i32],
        block_offsets: &[i32],
        cos_table: &[f32],
        sin_table: &[f32],
        mut out: DisjointSlice<i32>,
        tokens: u32,
        window_size: u32,
        window_cols: u32,
        extra_cols: u32,
        value_offset: u32,
        compress_ratio: u32,
        compressed_len: u32,
        index_heads: u32,
        index_head_dim: u32,
        rope_dim: u32,
        start_position: u32,
        page_tokens: u32,
        layer_index: u32,
        layer_count: u32,
        paged_indexer: u32,
        weight_scale: f32,
    ) {
        let tokens = tokens as usize;
        if token >= tokens {
            return;
        }
        let window_size = window_size as usize;
        let window_cols = window_cols as usize;
        let extra_cols = extra_cols as usize;
        let value_offset = value_offset as usize;
        let ratio = compress_ratio as usize;
        let compressed_len = compressed_len as usize;
        let heads = index_heads as usize;
        let hd = index_head_dim as usize;
        let rd = rope_dim as usize;
        let total_cols = window_cols + extra_cols;
        let out_ptr = out.as_mut_ptr();

        let first = (token + 1).saturating_sub(window_size);
        let mut col = 0usize;
        while col < window_cols {
            let idx = first + col;
            let value = if idx <= token { idx as i32 } else { -1 };
            unsafe {
                *out_ptr.add(token * total_cols + col) = value;
            }
            col += 1;
        }
        if extra_cols == 0 || ratio == 0 || hd == 0 || hd > 256 || !hd.is_power_of_two() {
            return;
        }

        let visible = ((token + 1) / ratio).min(compressed_len);
        let mut best_val = [f32::NEG_INFINITY; 512];
        let mut best_idx = [-1i32; 512];
        let take = extra_cols.min(512);
        let mut idx = 0usize;
        while idx < visible {
            let mut score = 0.0f32;
            let mut head = 0usize;
            while head < heads {
                let mut q = [0.0f32; 256];
                let q_base = token * heads * hd + head * hd;
                let mut d = 0usize;
                while d < hd {
                    q[d] = query[q_base + d];
                    d += 1;
                }

                if rd > 0 && rd <= hd && (rd & 1) == 0 {
                    let tail_start = hd - rd;
                    let rd2 = rd / 2;
                    let position = start_position as usize + token;
                    let mut pair = 0usize;
                    while pair < rd2 {
                        let base = tail_start + pair * 2;
                        let x0 = q[base];
                        let x1 = q[base + 1];
                        let c = cos_table[position * rd2 + pair];
                        let s = sin_table[position * rd2 + pair];
                        q[base] = x0 * c - x1 * s;
                        q[base + 1] = x0 * s + x1 * c;
                        pair += 1;
                    }
                }

                let mut span = 1usize;
                while span < hd {
                    let step = span * 2;
                    let mut start = 0usize;
                    while start < hd {
                        let mut offset = 0usize;
                        while offset < span {
                            let left = start + offset;
                            let right = left + span;
                            let a = q[left];
                            let b = q[right];
                            q[left] = a + b;
                            q[right] = a - b;
                            offset += 1;
                        }
                        start += step;
                    }
                    span = step;
                }
                let scale_h = fast_rsqrt(hd as f32);
                d = 0;
                while d < hd {
                    q[d] *= scale_h;
                    d += 1;
                }

                let block_size = 32usize;
                let blocks = hd / block_size;
                let mut block = 0usize;
                while block < blocks {
                    let start = block * block_size;
                    let end = start + block_size;
                    let mut amax = 6.0 * pow2f(-126.0);
                    let mut j = start;
                    while j < end {
                        let value = q[j];
                        let abs_value = if value < 0.0 { -value } else { value };
                        if abs_value > amax {
                            amax = abs_value;
                        }
                        j += 1;
                    }
                    let scale_byte = e8m0_scale_byte_for_amax(amax, 6.0);
                    let scale = e8m0_scale(scale_byte);
                    j = start;
                    while j < end {
                        let value = q[j];
                        let scaled = clamp_range(value / scale, -6.0, 6.0);
                        q[j] = fp4_e2m1_value(quantize_fp4_e2m1_nibble(scaled)) * scale;
                        j += 1;
                    }
                    block += 1;
                }

                let kv_base = if paged_indexer != 0 {
                    paged_plane_row_offset(
                        indexer_kv.len(),
                        block_slots,
                        block_offsets,
                        0,
                        idx,
                        page_tokens as usize,
                        hd,
                        layer_index as usize,
                        layer_count as usize,
                    )
                } else {
                    idx * hd
                };
                if kv_base == usize::MAX {
                    score = f32::NEG_INFINITY;
                    break;
                }
                let mut dot = 0.0f32;
                d = 0;
                while d < hd {
                    dot += q[d] * indexer_kv[kv_base + d];
                    d += 1;
                }
                if dot < 0.0 {
                    dot = 0.0;
                }
                score += dot * weights[token * heads + head] * weight_scale;
                head += 1;
            }
            let mut pos = take;
            while pos > 0 {
                let prev = pos - 1;
                let better = score > best_val[prev]
                    || (score == best_val[prev]
                        && (best_idx[prev] < 0 || (idx as i32) < best_idx[prev]));
                if !better {
                    break;
                }
                pos -= 1;
            }
            if pos < take {
                let mut move_pos = take - 1;
                while move_pos > pos {
                    best_val[move_pos] = best_val[move_pos - 1];
                    best_idx[move_pos] = best_idx[move_pos - 1];
                    move_pos -= 1;
                }
                best_val[pos] = score;
                best_idx[pos] = idx as i32;
            }
            idx += 1;
        }
        let mut slot = 0usize;
        while slot < extra_cols {
            let value = if slot < take && best_idx[slot] >= 0 && best_val[slot].is_finite() {
                (value_offset + best_idx[slot] as usize) as i32
            } else {
                -1
            };
            unsafe {
                *out_ptr.add(token * total_cols + window_cols + slot) = value;
            }
            slot += 1;
        }
    }

    #[kernel]
    pub fn dsv4_decode_topk_indices_paged_indexer(
        query: &[f32],
        weights: &[f32],
        indexer_plane: &[f32],
        block_slots: &[i32],
        block_offsets: &[i32],
        out: DisjointSlice<i32>,
        position: u32,
        window_len: u32,
        window_size: u32,
        extra_cols: u32,
        value_offset: u32,
        compressed_len: u32,
        index_heads: u32,
        index_head_dim: u32,
        page_tokens: u32,
        layer_index: u32,
        layer_count: u32,
        weight_scale: f32,
    ) {
        dsv4_decode_topk_indices_impl(
            query,
            weights,
            indexer_plane,
            block_slots,
            block_offsets,
            out,
            core::ptr::null_mut(),
            0,
            0,
            0,
            position,
            window_len,
            window_size,
            extra_cols,
            value_offset,
            compressed_len,
            index_heads,
            index_head_dim,
            1,
            page_tokens,
            layer_index,
            layer_count,
            1,
            0,
            weight_scale,
        )
    }

    /// Multi-sequence paged decode indexer. Each CUDA block handles one row
    /// and emits logical indices plus the corresponding KV-plane selector.
    #[kernel]
    pub fn dsv4_decode_topk_indices_paged_indexer_rows(
        query: &[f32],
        weights: &[f32],
        indexer_plane: &[f32],
        block_slots: &[i32],
        block_offsets: &[i32],
        row_sequence_ids: &[i32],
        positions: &[i32],
        window_lens: &[i32],
        compressed_lens: &[i32],
        logical_indices: DisjointSlice<i32>,
        mut plane_selectors: DisjointSlice<i32>,
        rows: u32,
        window_size: u32,
        index_topk: u32,
        index_heads: u32,
        index_head_dim: u32,
        page_tokens: u32,
        layer_index: u32,
        layer_count: u32,
        weight_scale: f32,
    ) {
        let row = thread::blockIdx_x() as usize;
        if row >= rows as usize {
            return;
        }
        let position = positions.get(row).copied().unwrap_or(-1);
        let window_len = window_lens.get(row).copied().unwrap_or(-1);
        let compressed_len = compressed_lens.get(row).copied().unwrap_or(-1);
        let sequence = row_sequence_ids
            .get(row)
            .and_then(|value| usize::try_from(*value).ok())
            .unwrap_or(usize::MAX);
        let valid_metadata =
            sequence != usize::MAX && position >= 0 && window_len >= 0 && compressed_len >= 0;
        dsv4_decode_topk_indices_impl(
            query,
            weights,
            indexer_plane,
            block_slots,
            block_offsets,
            logical_indices,
            plane_selectors.as_mut_ptr(),
            row,
            row,
            sequence,
            position.max(0) as u32,
            if valid_metadata { window_len as u32 } else { 0 },
            window_size,
            index_topk,
            0,
            if valid_metadata {
                compressed_len as u32
            } else {
                0
            },
            index_heads,
            index_head_dim,
            1,
            page_tokens,
            layer_index,
            layer_count,
            1,
            if valid_metadata { 1 } else { 2 },
            weight_scale,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn dsv4_decode_topk_indices_impl(
        query: &[f32],
        weights: &[f32],
        indexer_kv: &[f32],
        block_slots: &[i32],
        block_offsets: &[i32],
        mut out: DisjointSlice<i32>,
        selector_ptr: *mut i32,
        output_row: usize,
        query_row: usize,
        sequence: usize,
        position: u32,
        window_len: u32,
        window_size: u32,
        extra_cols: u32,
        value_offset: u32,
        compressed_len: u32,
        index_heads: u32,
        index_head_dim: u32,
        indexer_enabled: u32,
        page_tokens: u32,
        layer_index: u32,
        layer_count: u32,
        paged_indexer: u32,
        logical_output: u32,
        weight_scale: f32,
    ) {
        static mut CANDIDATE_SCORES: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut BEST_SCORES: SharedArray<f32, 512> = SharedArray::UNINIT;
        static mut BEST_INDICES: SharedArray<i32, 512> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let bdim = thread::blockDim_x() as usize;
        if bdim == 0 || bdim > 256 {
            return;
        }
        let total_cols = window_size as usize + extra_cols as usize;
        if total_cols == 0 {
            return;
        }
        let out_ptr = out.as_mut_ptr();
        let output_base = output_row * total_cols;
        let window_len = window_len as usize;
        let window_size = window_size as usize;
        let position = position as usize;
        let extra_cols = extra_cols as usize;
        let value_offset = value_offset as usize;
        let compressed_len = compressed_len as usize;
        let heads = index_heads as usize;
        let hd = index_head_dim as usize;
        let query_base = query_row * heads * hd;
        let weight_base = query_row * heads;

        let mut col = tid;
        while col < window_size {
            let mut selector = -1i32;
            let value = if logical_output == 2 {
                -1
            } else if logical_output != 0 {
                if window_len <= window_size
                    && window_len <= position.saturating_add(1)
                    && col < window_len
                {
                    selector = 0;
                    (position + 1 - window_len + col) as i32
                } else {
                    -1
                }
            } else if window_len < window_size {
                if col < window_len { col as i32 } else { -1 }
            } else {
                let current_slot = position % window_size;
                let first_slot = if current_slot + 1 == window_size {
                    0
                } else {
                    current_slot + 1
                };
                ((first_slot + col) % window_size) as i32
            };
            unsafe {
                *out_ptr.add(output_base + col) = value;
                if logical_output != 0 {
                    *selector_ptr.add(output_base + col) = selector;
                }
            }
            col += bdim;
        }

        if extra_cols == 0 {
            return;
        }
        if indexer_enabled == 0 {
            let mut slot = tid;
            while slot < extra_cols {
                let value = if slot < compressed_len {
                    (value_offset + slot) as i32
                } else {
                    -1
                };
                unsafe {
                    *out_ptr.add(output_base + window_size + slot) = value;
                }
                slot += bdim;
            }
            return;
        }

        let take = extra_cols.min(512);
        let mut slot = tid;
        while slot < take {
            unsafe {
                BEST_SCORES[slot] = f32::NEG_INFINITY;
                BEST_INDICES[slot] = -1;
            }
            slot += bdim;
        }
        thread::sync_threads();

        let mut chunk_start = 0usize;
        while chunk_start < compressed_len {
            let idx = chunk_start + tid;
            let mut score = f32::NEG_INFINITY;
            if idx < compressed_len {
                let kv_base = if paged_indexer != 0 {
                    paged_plane_row_offset(
                        indexer_kv.len(),
                        block_slots,
                        block_offsets,
                        sequence,
                        idx,
                        page_tokens as usize,
                        hd,
                        layer_index as usize,
                        layer_count as usize,
                    )
                } else {
                    idx * hd
                };
                if kv_base != usize::MAX {
                    score = 0.0;
                }
                let mut head = 0usize;
                while head < heads && kv_base != usize::MAX {
                    let q_base = query_base + head * hd;
                    let mut dot = 0.0f32;
                    let mut d = 0usize;
                    while d < hd {
                        dot += query[q_base + d] * indexer_kv[kv_base + d];
                        d += 1;
                    }
                    if dot < 0.0 {
                        dot = 0.0;
                    }
                    score += dot * weights[weight_base + head] * weight_scale;
                    head += 1;
                }
            }
            unsafe {
                CANDIDATE_SCORES[tid] = score;
            }
            thread::sync_threads();

            if tid == 0 {
                let chunk_len = (compressed_len - chunk_start).min(bdim);
                let mut candidate = 0usize;
                while candidate < chunk_len {
                    let candidate_idx = (chunk_start + candidate) as i32;
                    let candidate_score = unsafe { CANDIDATE_SCORES[candidate] };
                    let mut pos = take;
                    while pos > 0 {
                        let prev = pos - 1;
                        let prev_score = unsafe { BEST_SCORES[prev] };
                        let prev_idx = unsafe { BEST_INDICES[prev] };
                        let better = candidate_score > prev_score
                            || (candidate_score == prev_score
                                && (prev_idx < 0 || candidate_idx < prev_idx));
                        if !better {
                            break;
                        }
                        pos -= 1;
                    }
                    if pos < take {
                        let mut move_pos = take - 1;
                        while move_pos > pos {
                            unsafe {
                                BEST_SCORES[move_pos] = BEST_SCORES[move_pos - 1];
                                BEST_INDICES[move_pos] = BEST_INDICES[move_pos - 1];
                            }
                            move_pos -= 1;
                        }
                        unsafe {
                            BEST_SCORES[pos] = candidate_score;
                            BEST_INDICES[pos] = candidate_idx;
                        }
                    }
                    candidate += 1;
                }
            }
            thread::sync_threads();
            chunk_start += bdim;
        }

        slot = tid;
        while slot < extra_cols {
            let (best_idx, best_score) = if slot < take {
                unsafe { (BEST_INDICES[slot], BEST_SCORES[slot]) }
            } else {
                (-1, f32::NEG_INFINITY)
            };
            let valid = best_idx >= 0 && best_score.is_finite();
            let value = if valid {
                if logical_output != 0 {
                    best_idx
                } else {
                    (value_offset + best_idx as usize) as i32
                }
            } else {
                -1
            };
            unsafe {
                *out_ptr.add(output_base + window_size + slot) = value;
                if logical_output != 0 {
                    *selector_ptr.add(output_base + window_size + slot) =
                        if valid { 1 } else { -1 };
                }
            }
            slot += bdim;
        }
    }

    #[kernel]
    pub fn dsv4_decode_topk_indices_fused_index_query_paged_indexer(
        query: &[f32],
        weights: &[f32],
        indexer_plane: &[f32],
        block_slots: &[i32],
        block_offsets: &[i32],
        cos_table: &[f32],
        sin_table: &[f32],
        out: DisjointSlice<i32>,
        position: u32,
        window_len: u32,
        window_size: u32,
        extra_cols: u32,
        value_offset: u32,
        compressed_len: u32,
        index_heads: u32,
        index_head_dim: u32,
        rope_dim: u32,
        page_tokens: u32,
        layer_index: u32,
        layer_count: u32,
        weight_scale: f32,
    ) {
        dsv4_decode_topk_indices_fused_index_query_impl(
            query,
            weights,
            indexer_plane,
            block_slots,
            block_offsets,
            cos_table,
            sin_table,
            out,
            position,
            window_len,
            window_size,
            extra_cols,
            value_offset,
            compressed_len,
            index_heads,
            index_head_dim,
            rope_dim,
            page_tokens,
            layer_index,
            layer_count,
            1,
            weight_scale,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn dsv4_decode_topk_indices_fused_index_query_impl(
        query: &[f32],
        weights: &[f32],
        indexer_kv: &[f32],
        block_slots: &[i32],
        block_offsets: &[i32],
        cos_table: &[f32],
        sin_table: &[f32],
        mut out: DisjointSlice<i32>,
        position: u32,
        window_len: u32,
        window_size: u32,
        extra_cols: u32,
        value_offset: u32,
        compressed_len: u32,
        index_heads: u32,
        index_head_dim: u32,
        rope_dim: u32,
        page_tokens: u32,
        layer_index: u32,
        layer_count: u32,
        paged_indexer: u32,
        weight_scale: f32,
    ) {
        static mut QUERY_SCRATCH: SharedArray<f32, DSV4_DECODE_INDEX_QUERY_SHARED_ELEMENTS> =
            SharedArray::UNINIT;
        static mut CANDIDATE_SCORES: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut BEST_SCORES: SharedArray<f32, 512> = SharedArray::UNINIT;
        static mut BEST_INDICES: SharedArray<i32, 512> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let bdim = thread::blockDim_x() as usize;
        if bdim == 0 || bdim > 256 {
            return;
        }
        let total_cols = window_size as usize + extra_cols as usize;
        if total_cols == 0 {
            return;
        }
        let out_ptr = out.as_mut_ptr();
        let window_len = window_len as usize;
        let window_size = window_size as usize;
        let position = position as usize;
        let extra_cols = extra_cols as usize;
        let value_offset = value_offset as usize;
        let compressed_len = compressed_len as usize;
        let heads = index_heads as usize;
        let hd = index_head_dim as usize;
        let rd = rope_dim as usize;

        let mut col = tid;
        while col < window_size {
            let value = if window_len < window_size {
                if col < window_len { col as i32 } else { -1 }
            } else {
                let current_slot = position % window_size;
                let first_slot = if current_slot + 1 == window_size {
                    0
                } else {
                    current_slot + 1
                };
                ((first_slot + col) % window_size) as i32
            };
            unsafe {
                *out_ptr.add(col) = value;
            }
            col += bdim;
        }

        if extra_cols == 0
            || heads == 0
            || hd == 0
            || hd > 256
            || !hd.is_power_of_two()
            || rd > hd
            || !rd.is_multiple_of(2)
            || heads > DSV4_DECODE_INDEX_QUERY_SHARED_ELEMENTS / hd
        {
            return;
        }

        let query_len = heads * hd;
        let mut query_index = tid;
        while query_index < query_len {
            unsafe {
                QUERY_SCRATCH[query_index] = query[query_index];
            }
            query_index += bdim;
        }
        thread::sync_threads();

        if rd > 0 {
            let rd2 = rd / 2;
            let pairs = heads * rd2;
            let tail_start = hd - rd;
            let mut pair_index = tid;
            while pair_index < pairs {
                let head = pair_index / rd2;
                let pair = pair_index % rd2;
                let base = head * hd + tail_start + pair * 2;
                let x0 = unsafe { QUERY_SCRATCH[base] };
                let x1 = unsafe { QUERY_SCRATCH[base + 1] };
                let c = cos_table[position * rd2 + pair];
                let s = sin_table[position * rd2 + pair];
                unsafe {
                    QUERY_SCRATCH[base] = x0 * c - x1 * s;
                    QUERY_SCRATCH[base + 1] = x0 * s + x1 * c;
                }
                pair_index += bdim;
            }
            thread::sync_threads();
        }

        let butterflies_per_head = hd / 2;
        let total_butterflies = heads * butterflies_per_head;
        let mut span = 1usize;
        while span < hd {
            let mut butterfly = tid;
            while butterfly < total_butterflies {
                let head = butterfly / butterflies_per_head;
                let within_head = butterfly % butterflies_per_head;
                let group = within_head / span;
                let offset = within_head % span;
                let left = head * hd + group * span * 2 + offset;
                let right = left + span;
                let a = unsafe { QUERY_SCRATCH[left] };
                let b = unsafe { QUERY_SCRATCH[right] };
                unsafe {
                    QUERY_SCRATCH[left] = a + b;
                    QUERY_SCRATCH[right] = a - b;
                }
                butterfly += bdim;
            }
            thread::sync_threads();
            span *= 2;
        }

        let block_size = 32usize;
        let blocks_per_head = hd / block_size;
        let total_blocks = heads * blocks_per_head;
        let scale_h = fast_rsqrt(hd as f32);
        let mut block = tid;
        while block < total_blocks {
            let start = block * block_size;
            let end = start + block_size;
            let mut amax = 6.0 * pow2f(-126.0);
            let mut j = start;
            while j < end {
                let value = unsafe { QUERY_SCRATCH[j] } * scale_h;
                unsafe {
                    QUERY_SCRATCH[j] = value;
                }
                let abs_value = if value < 0.0 { -value } else { value };
                if abs_value > amax {
                    amax = abs_value;
                }
                j += 1;
            }
            let scale_byte = e8m0_scale_byte_for_amax(amax, 6.0);
            let scale = e8m0_scale(scale_byte);
            j = start;
            while j < end {
                let value = unsafe { QUERY_SCRATCH[j] };
                let scaled = clamp_range(value / scale, -6.0, 6.0);
                unsafe {
                    QUERY_SCRATCH[j] = fp4_e2m1_value(quantize_fp4_e2m1_nibble(scaled)) * scale;
                }
                j += 1;
            }
            block += bdim;
        }
        thread::sync_threads();

        let take = extra_cols.min(512);
        let mut slot = tid;
        while slot < take {
            unsafe {
                BEST_SCORES[slot] = f32::NEG_INFINITY;
                BEST_INDICES[slot] = -1;
            }
            slot += bdim;
        }
        thread::sync_threads();

        let mut chunk_start = 0usize;
        while chunk_start < compressed_len {
            let idx = chunk_start + tid;
            let mut score = f32::NEG_INFINITY;
            if idx < compressed_len {
                score = 0.0;
                let mut head = 0usize;
                while head < heads {
                    let q_base = head * hd;
                    let kv_base = if paged_indexer != 0 {
                        paged_plane_row_offset(
                            indexer_kv.len(),
                            block_slots,
                            block_offsets,
                            0,
                            idx,
                            page_tokens as usize,
                            hd,
                            layer_index as usize,
                            layer_count as usize,
                        )
                    } else {
                        idx * hd
                    };
                    if kv_base == usize::MAX {
                        score = f32::NEG_INFINITY;
                        break;
                    }
                    let mut dot = 0.0f32;
                    let mut d = 0usize;
                    while d < hd {
                        dot += unsafe { QUERY_SCRATCH[q_base + d] } * indexer_kv[kv_base + d];
                        d += 1;
                    }
                    if dot < 0.0 {
                        dot = 0.0;
                    }
                    score += dot * weights[head] * weight_scale;
                    head += 1;
                }
            }
            unsafe {
                CANDIDATE_SCORES[tid] = score;
            }
            thread::sync_threads();

            if tid == 0 {
                let chunk_len = (compressed_len - chunk_start).min(bdim);
                let mut candidate = 0usize;
                while candidate < chunk_len {
                    let candidate_idx = (chunk_start + candidate) as i32;
                    let candidate_score = unsafe { CANDIDATE_SCORES[candidate] };
                    let mut pos = take;
                    while pos > 0 {
                        let prev = pos - 1;
                        let prev_score = unsafe { BEST_SCORES[prev] };
                        let prev_idx = unsafe { BEST_INDICES[prev] };
                        let better = candidate_score > prev_score
                            || (candidate_score == prev_score
                                && (prev_idx < 0 || candidate_idx < prev_idx));
                        if !better {
                            break;
                        }
                        pos -= 1;
                    }
                    if pos < take {
                        let mut move_pos = take - 1;
                        while move_pos > pos {
                            unsafe {
                                BEST_SCORES[move_pos] = BEST_SCORES[move_pos - 1];
                                BEST_INDICES[move_pos] = BEST_INDICES[move_pos - 1];
                            }
                            move_pos -= 1;
                        }
                        unsafe {
                            BEST_SCORES[pos] = candidate_score;
                            BEST_INDICES[pos] = candidate_idx;
                        }
                    }
                    candidate += 1;
                }
            }
            thread::sync_threads();
            chunk_start += bdim;
        }

        slot = tid;
        while slot < extra_cols {
            let (best_idx, best_score) = if slot < take {
                unsafe { (BEST_INDICES[slot], BEST_SCORES[slot]) }
            } else {
                (-1, f32::NEG_INFINITY)
            };
            let value = if best_idx >= 0 && best_score.is_finite() {
                (value_offset + best_idx as usize) as i32
            } else {
                -1
            };
            unsafe {
                *out_ptr.add(window_size + slot) = value;
            }
            slot += bdim;
        }
    }

    /// DSV4 compressor prefill softmax reduction.
    ///
    /// Inputs are row-major projected rows:
    /// - non-overlap: `[tokens, head_dim]`
    /// - overlap: `[tokens, 2 * head_dim]`, where previous/current halves are
    ///   combined exactly like the CPU reference compressor.
    ///
    /// Output is `[groups=tokens/ratio, head_dim]` before compressor RMS/RoPE/QAT.
    #[kernel]
    pub fn dsv4_compressor_prefill_softmax(
        kv_rows: &[f32],
        score_rows: &[f32],
        ape: &[f32],
        mut output: DisjointSlice<f32>,
        groups: u32,
        ratio: u32,
        head_dim: u32,
        out_dim: u32,
        overlap: u32,
    ) {
        let idx = thread::index_1d().get() as usize;
        let groups = groups as usize;
        let ratio = ratio as usize;
        let head_dim = head_dim as usize;
        let out_dim = out_dim as usize;
        if groups == 0 || ratio == 0 || head_dim == 0 || out_dim == 0 {
            return;
        }
        let total = groups * head_dim;
        if idx >= total {
            return;
        }
        let group = idx / head_dim;
        let dim = idx - group * head_dim;
        let rows = if overlap != 0 { 2 * ratio } else { ratio };

        let mut max_score = f32::NEG_INFINITY;
        let mut row = 0usize;
        while row < rows {
            let mut valid = true;
            let (src_token, src_dim, ape_dim) = if overlap != 0 {
                if row < ratio {
                    if group == 0 {
                        valid = false;
                    }
                    ((group.saturating_sub(1)) * ratio + row, dim, dim)
                } else {
                    let local = row - ratio;
                    (group * ratio + local, head_dim + dim, head_dim + dim)
                }
            } else {
                (group * ratio + row, dim, dim)
            };
            if valid {
                let score = score_rows[src_token * out_dim + src_dim]
                    + ape[(row % ratio) * out_dim + ape_dim];
                if score > max_score {
                    max_score = score;
                }
            }
            row += 1;
        }

        let mut denom = 0.0f32;
        row = 0;
        while row < rows {
            let mut valid = true;
            let (src_token, src_dim, ape_dim) = if overlap != 0 {
                if row < ratio {
                    if group == 0 {
                        valid = false;
                    }
                    ((group.saturating_sub(1)) * ratio + row, dim, dim)
                } else {
                    let local = row - ratio;
                    (group * ratio + local, head_dim + dim, head_dim + dim)
                }
            } else {
                (group * ratio + row, dim, dim)
            };
            if valid {
                let score = score_rows[src_token * out_dim + src_dim]
                    + ape[(row % ratio) * out_dim + ape_dim];
                denom += fast_exp(score - max_score);
            }
            row += 1;
        }

        let mut out = 0.0f32;
        if denom > 0.0 && denom.is_finite() {
            row = 0;
            while row < rows {
                let mut valid = true;
                let (src_token, src_dim, ape_dim) = if overlap != 0 {
                    if row < ratio {
                        if group == 0 {
                            valid = false;
                        }
                        ((group.saturating_sub(1)) * ratio + row, dim, dim)
                    } else {
                        let local = row - ratio;
                        (group * ratio + local, head_dim + dim, head_dim + dim)
                    }
                } else {
                    (group * ratio + row, dim, dim)
                };
                if valid {
                    let score = score_rows[src_token * out_dim + src_dim]
                        + ape[(row % ratio) * out_dim + ape_dim];
                    let weight = fast_exp(score - max_score) / denom;
                    out += weight * kv_rows[src_token * out_dim + src_dim];
                }
                row += 1;
            }
        }
        if let Some(o) = output.get_mut(thread::index_1d()) {
            *o = out;
        }
    }

    #[kernel]
    pub fn compressor_recurrent_reset_f32(
        mut kv_state: DisjointSlice<f32>,
        mut score_state: DisjointSlice<f32>,
        state_elements: u32,
    ) {
        let index = thread::index_1d().get() as usize;
        if index >= state_elements as usize {
            return;
        }
        if let Some(value) = kv_state.get_mut(thread::index_1d()) {
            *value = 0.0;
        }
        if let Some(value) = score_state.get_mut(thread::index_1d()) {
            *value = f32::NEG_INFINITY;
        }
    }

    #[kernel]
    pub fn compressor_recurrent_append_projected_f32(
        projected_kv: &[f32],
        projected_score: &[f32],
        ape: &[f32],
        mut kv_state: DisjointSlice<f32>,
        mut score_state: DisjointSlice<f32>,
        position: u32,
        ratio: u32,
        out_dim: u32,
        overlap: u32,
    ) {
        let dim = thread::index_1d().get() as usize;
        let ratio = ratio as usize;
        let out_dim = out_dim as usize;
        if dim >= out_dim || ratio == 0 {
            return;
        }
        let position = position as usize;
        let local_row = position % ratio;
        let state_row = if overlap != 0 {
            ratio + local_row
        } else {
            local_row
        };
        let state_index = state_row * out_dim + dim;
        let ape_index = local_row * out_dim + dim;
        let kv_ptr = kv_state.as_mut_ptr();
        let score_ptr = score_state.as_mut_ptr();
        unsafe {
            *kv_ptr.add(state_index) = projected_kv[dim];
            *score_ptr.add(state_index) = projected_score[dim] + ape[ape_index];
        }
    }

    #[kernel]
    pub fn compressor_recurrent_seed_prefill_f32(
        projected_kv_rows: &[f32],
        projected_score_rows: &[f32],
        ape: &[f32],
        mut kv_state: DisjointSlice<f32>,
        mut score_state: DisjointSlice<f32>,
        tokens: u32,
        ratio: u32,
        out_dim: u32,
        overlap: u32,
        state_elements: u32,
    ) {
        let index = thread::index_1d().get() as usize;
        let ratio = ratio as usize;
        let out_dim = out_dim as usize;
        if index >= state_elements as usize || ratio == 0 || out_dim == 0 {
            return;
        }
        let tokens = tokens as usize;
        let state_row = index / out_dim;
        let dim = index % out_dim;
        let remainder = tokens % ratio;
        let cutoff = tokens - remainder;
        let mut source_token = usize::MAX;
        let mut ape_row = 0usize;
        if overlap != 0 && cutoff >= ratio && state_row < ratio {
            source_token = cutoff - ratio + state_row;
            ape_row = state_row;
        } else {
            let state_offset = if overlap != 0 { ratio } else { 0 };
            if state_row >= state_offset && state_row < state_offset + remainder {
                let local = state_row - state_offset;
                source_token = cutoff + local;
                ape_row = local;
            }
        }
        let kv_ptr = kv_state.as_mut_ptr();
        let score_ptr = score_state.as_mut_ptr();
        unsafe {
            if source_token == usize::MAX {
                *kv_ptr.add(index) = 0.0;
                *score_ptr.add(index) = f32::NEG_INFINITY;
            } else {
                let source = source_token * out_dim + dim;
                *kv_ptr.add(index) = projected_kv_rows[source];
                *score_ptr.add(index) = projected_score_rows[source] + ape[ape_row * out_dim + dim];
            }
        }
    }

    #[kernel]
    pub fn compressor_recurrent_softmax_f32(
        kv_state: &[f32],
        score_state: &[f32],
        mut output: DisjointSlice<f32>,
        ratio: u32,
        head_dim: u32,
        out_dim: u32,
        overlap: u32,
    ) {
        let dim = thread::index_1d().get() as usize;
        let ratio = ratio as usize;
        let head_dim = head_dim as usize;
        let out_dim = out_dim as usize;
        if dim >= head_dim || ratio == 0 {
            return;
        }
        let rows = if overlap != 0 { 2 * ratio } else { ratio };
        let mut max_score = f32::NEG_INFINITY;
        let mut row = 0usize;
        while row < rows {
            let src_dim = if overlap != 0 && row >= ratio {
                head_dim + dim
            } else {
                dim
            };
            let score = score_state[row * out_dim + src_dim];
            if score > max_score {
                max_score = score;
            }
            row += 1;
        }
        let mut denominator = 0.0f32;
        row = 0;
        while row < rows {
            let src_dim = if overlap != 0 && row >= ratio {
                head_dim + dim
            } else {
                dim
            };
            denominator += fast_exp(score_state[row * out_dim + src_dim] - max_score);
            row += 1;
        }
        let mut compressed = 0.0f32;
        if denominator > 0.0 && denominator.is_finite() {
            row = 0;
            while row < rows {
                let src_dim = if overlap != 0 && row >= ratio {
                    head_dim + dim
                } else {
                    dim
                };
                let index = row * out_dim + src_dim;
                let weight = fast_exp(score_state[index] - max_score) / denominator;
                compressed += weight * kv_state[index];
                row += 1;
            }
        }
        if let Some(value) = output.get_mut(thread::index_1d()) {
            *value = compressed;
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
        value_offset: u32,
        value_len: u32,
        row_width: u32,
        block_size: u32,
    ) {
        let block_idx = thread::index_1d().get() as usize;
        let value_offset = value_offset as usize;
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
        if value_len == 0
            || !value_len.is_multiple_of(row_width)
            || value_offset.saturating_add(value_len) > values.len()
        {
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
            let value = values[value_offset + i];
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
            let v0 = values[value_offset + start + j] / scale;
            let v1 = values[value_offset + start + j + 1] / scale;
            let n0 = quantize_fp4_e2m1_nibble(v0);
            let n1 = quantize_fp4_e2m1_nibble(v1);
            unsafe {
                *packed_ptr.add(packed_row + packed_block + j / 2) = n0 | (n1 << 4);
            }
            j += 2;
        }
    }

    // ── Artifact FP8 E4M3FN + E8M0 GEMV ────────────────────────────────

    /// Block-wise FP8 E4M3FN + E8M0 activation quantization into raw bytes.
    ///
    /// `values` and `packed` are row-major `[rows, row_width]`; `scales` is
    /// `[rows, row_width / block_size]`. The production artifact MMA path uses
    /// 128-value blocks to match the model's activation quantization contract.
    #[kernel]
    pub fn fp8_e4m3fn_e8m0_quantize_f32_packed(
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
        if value_len == 0
            || row_width == 0
            || block_size == 0
            || !value_len.is_multiple_of(row_width)
            || !row_width.is_multiple_of(block_size)
        {
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

        let mut amax = 1e-4f32;
        let mut i = start;
        while i < end {
            let value = values[i];
            let abs_value = if value < 0.0 { -value } else { value };
            if abs_value > amax {
                amax = abs_value;
            }
            i += 1;
        }
        let scale_byte = e8m0_scale_byte_for_amax(amax, 448.0);
        let scale = e8m0_scale(scale_byte);
        unsafe {
            *scales.as_mut_ptr().add(row * blocks_per_row + block) = scale_byte;
        }

        let packed_ptr = packed.as_mut_ptr();
        let mut i = start;
        while i < end {
            let scaled = clamp_range(values[i] / scale, -448.0, 448.0);
            unsafe {
                *packed_ptr.add(i) = quantize_fp8_e4m3fn_byte(scaled);
            }
            i += 1;
        }
    }

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

    /// Batched GEMM for FP8 E4M3FN weights with 2D E8M0 block scales.
    /// Output layout is row-major `[batch, n]`.
    #[kernel]
    pub fn gemm_fp8_e4m3fn_e8m0_2d(
        x: &[f32],
        weight: &[u8],
        scales: &[u8],
        mut y: DisjointSlice<f32>,
        batch: u32,
        n: u32,
        k: u32,
        scale_cols: u32,
        block_m: u32,
        block_k: u32,
    ) {
        let idx = thread::index_1d().get();
        let total = batch as u64 * n as u64;
        if (idx as u64) >= total {
            return;
        }
        let b = (idx as u32 / n) as usize;
        let row = (idx as u32 % n) as usize;
        let k = k as usize;
        let sc = scale_cols as usize;
        let bm = block_m as usize;
        let bk = block_k as usize;
        let input_off = b * k;
        let row_weight = row * k;
        let row_scales = (row / bm) * sc;
        let mut dot = 0.0f32;
        let mut block = 0usize;
        while block < sc {
            let scale = e8m0_scale(scales[row_scales + block]);
            let start = block * bk;
            let end = (start + bk).min(k);
            let mut j = start;
            while j < end {
                let w = fp8_e4m3fn_value(weight[row_weight + j]);
                dot += x[input_off + j] * w * scale;
                j += 1;
            }
            block += 1;
        }
        if let Some(yi) = y.get_mut(thread::index_1d()) {
            *yi = dot;
        }
    }

    /// Batched FP8 artifact GEMM using Blackwell `m16n8k32` Tensor Cores.
    ///
    /// Grid: `(ceil(n / 16), ceil(batch / 8))`, one warp per CTA. Weight rows
    /// form MMA A; eight activation rows form MMA B columns. Four K32 MMA
    /// operations are folded before applying the model's K128 E8M0 scales.
    #[kernel]
    pub unsafe fn gemm_fp8_e4m3fn_e8m0_2d_mma(
        x_packed: &[u8],
        x_scales: &[u8],
        weight: &[u8],
        weight_scales: &[u8],
        mut y: DisjointSlice<f32>,
        batch: u32,
        n: u32,
        k: u32,
        scale_cols: u32,
    ) {
        static mut SMEM_A: SharedArray<u8, 512, 32> = SharedArray::UNINIT;
        static mut SMEM_B: SharedArray<u8, 256, 32> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let row_base = thread::blockIdx_x() as usize * 16;
        let batch_base = thread::blockIdx_y() as usize * 8;
        let batch = batch as usize;
        let n = n as usize;
        let k = k as usize;
        let scale_cols = scale_cols as usize;
        if batch == 0 || n == 0 || k == 0 || scale_cols == 0 || !k.is_multiple_of(128) {
            return;
        }

        let mut acc = [0.0f32; 4];
        let mut scale_block = 0usize;
        while scale_block < scale_cols {
            let k_base = scale_block * 128;
            let mut block_acc = [0.0f32; 4];
            let mut k_sub = 0usize;
            while k_sub < 128 {
                unsafe {
                    let a_dst = &raw mut SMEM_A as *mut u8;
                    let b_dst = &raw mut SMEM_B as *mut u8;
                    let mut i = tid;
                    while i < 512 {
                        let row_local = i / 32;
                        let byte = i & 31;
                        let row = row_base + row_local;
                        *a_dst.add(i) = if row < n {
                            weight[row * k + k_base + k_sub + byte]
                        } else {
                            0
                        };
                        i += 32;
                    }
                    let mut i = tid;
                    while i < 128 {
                        let k_pair = i / 8;
                        let col = i & 7;
                        let dst = k_pair * 16 + col * 2;
                        if batch_base + col < batch {
                            let src = (batch_base + col) * k + k_base + k_sub + k_pair * 2;
                            *b_dst.add(dst) = x_packed[src];
                            *b_dst.add(dst + 1) = x_packed[src + 1];
                        } else {
                            *b_dst.add(dst) = 0;
                            *b_dst.add(dst + 1) = 0;
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
                    let row = tid & 0x0f;
                    let addr = (&raw const SMEM_B as *const u8).add(row * 16) as *const u32;
                    cuda_device::wmma::ldmatrix_x2_trans(addr)
                };
                block_acc = unsafe { mma_m16n8k32_f32_e4m3_e4m3(block_acc, a_frag, b_frag) };
                thread::sync_threads();
                k_sub += 32;
            }

            let weight_scale = if row_base < n {
                e8m0_scale(weight_scales[(row_base / 128) * scale_cols + scale_block])
            } else {
                1.0
            };
            let lane_col = tid & 3;
            let mut j = 0usize;
            while j < 4 {
                let col = lane_col * 2 + (j & 1);
                if batch_base + col < batch {
                    let x_scale =
                        e8m0_scale(x_scales[(batch_base + col) * scale_cols + scale_block]);
                    acc[j] += block_acc[j] * weight_scale * x_scale;
                }
                j += 1;
            }
            scale_block += 1;
        }

        let group = tid / 4;
        let lane_col = tid & 3;
        let out_ptr = y.as_mut_ptr();
        let mut j = 0usize;
        while j < 4 {
            let row = row_base + group + if j >= 2 { 8 } else { 0 };
            let col = lane_col * 2 + (j & 1);
            if row < n && batch_base + col < batch {
                unsafe {
                    *out_ptr.add((batch_base + col) * n + row) = acc[j];
                }
            }
            j += 1;
        }
    }

    /// Allocation-free single-row variant used by decode and CUDA graphs.
    /// It quantizes each K128 activation block cooperatively into shared memory
    /// and uses the exact same MMA/scale fold as the batched rows kernel.
    #[kernel]
    pub unsafe fn gemv_fp8_e4m3fn_e8m0_2d_mma_from_f32(
        x: &[f32],
        weight: &[u8],
        weight_scales: &[u8],
        mut y: DisjointSlice<f32>,
        n: u32,
        k: u32,
        scale_cols: u32,
    ) {
        static mut SMEM_A: SharedArray<u8, 512, 32> = SharedArray::UNINIT;
        static mut SMEM_B: SharedArray<u8, 256, 32> = SharedArray::UNINIT;
        static mut SMEM_X_SCALE: SharedArray<u8, 1> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let row_base = thread::blockIdx_x() as usize * 16;
        let n = n as usize;
        let k = k as usize;
        let scale_cols = scale_cols as usize;
        if n == 0 || k == 0 || scale_cols == 0 || !k.is_multiple_of(128) {
            return;
        }

        let mut acc = [0.0f32; 4];
        let mut scale_block = 0usize;
        while scale_block < scale_cols {
            let k_base = scale_block * 128;
            if tid == 0 {
                let mut amax = 1e-4f32;
                let mut i = 0usize;
                while i < 128 {
                    let value = x[k_base + i];
                    let abs_value = if value < 0.0 { -value } else { value };
                    if abs_value > amax {
                        amax = abs_value;
                    }
                    i += 1;
                }
                unsafe {
                    *(&raw mut SMEM_X_SCALE as *mut u8) = e8m0_scale_byte_for_amax(amax, 448.0);
                }
            }
            thread::sync_threads();
            let x_scale_byte = unsafe { *(&raw const SMEM_X_SCALE as *const u8) };
            let x_scale = e8m0_scale(x_scale_byte);
            let mut block_acc = [0.0f32; 4];
            let mut k_sub = 0usize;
            while k_sub < 128 {
                unsafe {
                    let a_dst = &raw mut SMEM_A as *mut u8;
                    let b_dst = &raw mut SMEM_B as *mut u8;
                    let mut i = tid;
                    while i < 512 {
                        let row_local = i / 32;
                        let byte = i & 31;
                        let row = row_base + row_local;
                        *a_dst.add(i) = if row < n {
                            weight[row * k + k_base + k_sub + byte]
                        } else {
                            0
                        };
                        i += 32;
                    }
                    let mut i = tid;
                    while i < 128 {
                        let k_pair = i / 8;
                        let col = i & 7;
                        let dst = k_pair * 16 + col * 2;
                        if col == 0 {
                            let src = k_base + k_sub + k_pair * 2;
                            let v0 = clamp_range(x[src] / x_scale, -448.0, 448.0);
                            let v1 = clamp_range(x[src + 1] / x_scale, -448.0, 448.0);
                            *b_dst.add(dst) = quantize_fp8_e4m3fn_byte(v0);
                            *b_dst.add(dst + 1) = quantize_fp8_e4m3fn_byte(v1);
                        } else {
                            *b_dst.add(dst) = 0;
                            *b_dst.add(dst + 1) = 0;
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
                    let row = tid & 0x0f;
                    let addr = (&raw const SMEM_B as *const u8).add(row * 16) as *const u32;
                    cuda_device::wmma::ldmatrix_x2_trans(addr)
                };
                block_acc = unsafe { mma_m16n8k32_f32_e4m3_e4m3(block_acc, a_frag, b_frag) };
                thread::sync_threads();
                k_sub += 32;
            }

            let weight_scale = if row_base < n {
                e8m0_scale(weight_scales[(row_base / 128) * scale_cols + scale_block])
            } else {
                1.0
            };
            let mut j = 0usize;
            while j < 4 {
                acc[j] += block_acc[j] * weight_scale * x_scale;
                j += 1;
            }
            scale_block += 1;
        }

        if tid & 3 == 0 {
            let group = tid / 4;
            let out_ptr = y.as_mut_ptr();
            if row_base + group < n {
                unsafe {
                    *out_ptr.add(row_base + group) = acc[0];
                }
            }
            if row_base + group + 8 < n {
                unsafe {
                    *out_ptr.add(row_base + group + 8) = acc[2];
                }
            }
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

    /// Gather row-major f32 rows from `src` into compact `dst`.
    ///
    /// `row_indices` has `rows` entries. `src` is `[src_rows, row_width]` and
    /// `dst` is `[rows, row_width]`. This is the packing primitive for grouped
    /// MoE prefill: tokens routed to one expert become contiguous columns for a
    /// single batched expert execution.
    #[kernel]
    pub fn gather_f32_rows(
        src: &[f32],
        row_indices: &[i32],
        mut dst: DisjointSlice<f32>,
        rows: u32,
        row_width: u32,
    ) {
        let i = thread::index_1d().get();
        let total = rows as u64 * row_width as u64;
        if (i as u64) >= total {
            return;
        }
        let row_width = row_width as usize;
        let dst_row = i as usize / row_width;
        let col = i as usize - dst_row * row_width;
        let src_row = row_indices[dst_row];
        if src_row < 0 {
            return;
        }
        let src_idx = src_row as usize * row_width + col;
        let dst_ptr = dst.as_mut_ptr();
        unsafe {
            *dst_ptr.add(i as usize) = src[src_idx];
        }
    }

    /// Scatter-add compact row-major f32 rows into `dst`.
    ///
    /// `src` is `[rows, row_width]`, `row_indices` maps each compact row to a
    /// destination row in `dst`. Launches on one stream, so different expert
    /// groups are ordered; within one launch each `(row,col)` is unique.
    #[kernel]
    pub fn scatter_add_f32_rows(
        src: &[f32],
        row_indices: &[i32],
        mut dst: DisjointSlice<f32>,
        rows: u32,
        row_width: u32,
    ) {
        let i = thread::index_1d().get();
        let total = rows as u64 * row_width as u64;
        if (i as u64) >= total {
            return;
        }
        let row_width = row_width as usize;
        let src_row = i as usize / row_width;
        let col = i as usize - src_row * row_width;
        let dst_row = row_indices[src_row];
        if dst_row < 0 {
            return;
        }
        let dst_idx = dst_row as usize * row_width + col;
        let dst_ptr = dst.as_mut_ptr();
        unsafe {
            *dst_ptr.add(dst_idx) += src[i as usize];
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

    /// Batched affine RMS normalize `[rows, row_dim]` with shared weight `[row_dim]`.
    /// One CUDA block owns one row; this is the prefill shape for attention/FFN
    /// RMS norms and avoids launching one single-row RMS kernel per token.
    #[kernel]
    pub fn rms_norm_rows_fused(
        x: &[f32],
        w: &[f32],
        mut y: DisjointSlice<f32>,
        rows: u32,
        row_dim: u32,
        eps: f32,
    ) {
        static mut SMEM: SharedArray<f32, 1024> = SharedArray::UNINIT;
        let row = thread::blockIdx_x() as usize;
        let rows = rows as usize;
        if row >= rows {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let bdim = thread::blockDim_x() as usize;
        let dim = row_dim as usize;
        let base = row * dim;

        let mut sum = 0.0f32;
        let mut j = tid;
        while j < dim {
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
            let val = unsafe { SMEM[0] } / dim as f32 + eps;
            unsafe {
                SMEM[0] = fast_rsqrt(val);
            }
        }
        thread::sync_threads();

        let rsqrt = unsafe { SMEM[0] };
        let y_ptr = y.as_mut_ptr();
        let mut j = tid;
        while j < dim {
            unsafe {
                *y_ptr.add(base + j) = x[base + j] * rsqrt * w[j];
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

    /// DeepSeek-V4 router top-k for score-top-k layers.
    ///
    /// Input logits are `[tokens, experts]`. For each token this computes
    /// `score = sqrt(softplus(logit))`, selects top-k by `score + bias`, then
    /// writes selected expert ids as i32 and normalized route weights as f32,
    /// both shaped `[tokens, k]`. Host-side residency only needs ids/weights;
    /// diagnostic route scores stay CPU-reference-only for now.
    #[kernel]
    pub fn dsv4_router_topk_sqrt_softplus_rows(
        logits: &[f32],
        bias: &[f32],
        mut indices: DisjointSlice<i32>,
        mut weights: DisjointSlice<f32>,
        tokens: u32,
        experts: u32,
        k: u32,
        bias_enabled: u32,
        route_scale: f32,
    ) {
        let row = thread::index_1d().get();
        if row >= tokens as usize {
            return;
        }
        let experts = experts as usize;
        let k = k as usize;
        if k == 0 || k > 64 {
            return;
        }
        let row_offset = row * experts;
        let out_offset = row * k;
        let mut top_idx = [0i32; 64];
        let mut top_score = [0.0f32; 64];
        let mut selected = [false; 512];
        let mut sum = 0.0f32;

        for slot in 0..k {
            let mut best_idx = 0usize;
            let mut best_score = 0.0f32;
            let mut best_selection = f32::NEG_INFINITY;
            for expert in 0..experts {
                if expert < 512 && selected[expert] {
                    continue;
                }
                let logit = logits[row_offset + expert];
                let softplus = stable_softplus_f32(logit);
                let score = if softplus > 0.0 {
                    fast_sqrt(softplus)
                } else {
                    0.0
                };
                let selection = if bias_enabled != 0 {
                    score + bias[expert]
                } else {
                    score
                };
                if selection > best_selection
                    || (selection == best_selection && expert < best_idx)
                    || slot == 0 && best_selection == f32::NEG_INFINITY
                {
                    best_idx = expert;
                    best_score = score;
                    best_selection = selection;
                }
            }
            if best_idx < 512 {
                selected[best_idx] = true;
            }
            top_idx[slot] = best_idx as i32;
            top_score[slot] = best_score;
            sum += best_score;
        }

        let idx_ptr = indices.as_mut_ptr();
        let w_ptr = weights.as_mut_ptr();
        for slot in 0..k {
            let weight = if sum > 0.0 && sum.is_finite() {
                top_score[slot] / sum * route_scale
            } else {
                0.0
            };
            unsafe {
                *idx_ptr.add(out_offset + slot) = top_idx[slot];
                *w_ptr.add(out_offset + slot) = weight;
            }
        }
    }

    /// DeepSeek-V4 hash router over device-resident token ids and hash rows.
    ///
    /// Hash selection deliberately ignores router bias. The selected experts'
    /// original `sqrt(softplus(logit))` scores are normalized and scaled.
    #[kernel]
    pub fn dsv4_router_hash_sqrt_softplus_rows(
        logits: &[f32],
        token_ids: &[i32],
        hash_table: &[i32],
        mut indices: DisjointSlice<i32>,
        mut weights: DisjointSlice<f32>,
        tokens: u32,
        experts: u32,
        hash_rows: u32,
        hash_cols: u32,
        k: u32,
        route_scale: f32,
    ) {
        let row = thread::index_1d().get();
        if row >= tokens as usize {
            return;
        }
        let experts = experts as usize;
        let hash_rows = hash_rows as usize;
        let hash_cols = hash_cols as usize;
        let k = k as usize;
        if k == 0 || k > 64 || k > hash_cols {
            return;
        }
        let token_id = token_ids[row];
        if token_id < 0 || token_id as usize >= hash_rows {
            return;
        }
        let logits_offset = row * experts;
        let hash_offset = token_id as usize * hash_cols;
        let out_offset = row * k;
        let mut selected_score = [0.0f32; 64];
        let mut sum = 0.0f32;

        for slot in 0..k {
            let expert = hash_table[hash_offset + slot];
            if expert < 0 || expert as usize >= experts {
                return;
            }
            let logit = logits[logits_offset + expert as usize];
            let softplus = stable_softplus_f32(logit);
            let score = if softplus > 0.0 {
                fast_sqrt(softplus)
            } else {
                0.0
            };
            selected_score[slot] = score;
            sum += score;
            unsafe {
                *indices.as_mut_ptr().add(out_offset + slot) = expert;
            }
        }

        for slot in 0..k {
            let weight = if sum > 0.0 && sum.is_finite() {
                selected_score[slot] / sum * route_scale
            } else {
                0.0
            };
            unsafe {
                *weights.as_mut_ptr().add(out_offset + slot) = weight;
            }
        }
    }

    /// Publish one exact stable expert binding. Residency policy and generation
    /// validation remain host/runtime-owned; this kernel only applies the
    /// already-validated mapping in compute-stream order.
    #[kernel]
    pub fn install_expert_slot_binding(
        mut gate_weight: DisjointSlice<u64>,
        mut gate_scale: DisjointSlice<u64>,
        mut up_weight: DisjointSlice<u64>,
        mut up_scale: DisjointSlice<u64>,
        mut down_weight: DisjointSlice<u64>,
        mut down_scale: DisjointSlice<u64>,
        mut expert_to_slot: DisjointSlice<i32>,
        mut expert_generation: DisjointSlice<i32>,
        mut slot_generation: DisjointSlice<i32>,
        expert: u32,
        slot: u32,
        generation: i32,
        gate_weight_ptr: u64,
        gate_scale_ptr: u64,
        up_weight_ptr: u64,
        up_scale_ptr: u64,
        down_weight_ptr: u64,
        down_scale_ptr: u64,
    ) {
        if thread::index_1d().get() != 0 {
            return;
        }
        let expert = expert as usize;
        let slot = slot as usize;
        unsafe {
            *gate_weight.as_mut_ptr().add(slot) = gate_weight_ptr;
            *gate_scale.as_mut_ptr().add(slot) = gate_scale_ptr;
            *up_weight.as_mut_ptr().add(slot) = up_weight_ptr;
            *up_scale.as_mut_ptr().add(slot) = up_scale_ptr;
            *down_weight.as_mut_ptr().add(slot) = down_weight_ptr;
            *down_scale.as_mut_ptr().add(slot) = down_scale_ptr;
            *slot_generation.as_mut_ptr().add(slot) = generation;
            *expert_to_slot.as_mut_ptr().add(expert) = slot as i32;
            *expert_generation.as_mut_ptr().add(expert) = generation;
        }
    }

    /// Remove one exact stable expert binding in compute-stream order. The slot
    /// generation is advanced by the host-validated state transition so stale
    /// routes cannot observe a replacement as the previous payload.
    #[kernel]
    pub fn evict_expert_slot_binding(
        mut gate_weight: DisjointSlice<u64>,
        mut gate_scale: DisjointSlice<u64>,
        mut up_weight: DisjointSlice<u64>,
        mut up_scale: DisjointSlice<u64>,
        mut down_weight: DisjointSlice<u64>,
        mut down_scale: DisjointSlice<u64>,
        mut expert_to_slot: DisjointSlice<i32>,
        mut expert_generation: DisjointSlice<i32>,
        mut slot_generation: DisjointSlice<i32>,
        expert: u32,
        slot: u32,
        next_generation: i32,
    ) {
        if thread::index_1d().get() != 0 {
            return;
        }
        let expert = expert as usize;
        let slot = slot as usize;
        unsafe {
            *expert_to_slot.as_mut_ptr().add(expert) = -1;
            *expert_generation.as_mut_ptr().add(expert) = 0;
            *gate_weight.as_mut_ptr().add(slot) = 0;
            *gate_scale.as_mut_ptr().add(slot) = 0;
            *up_weight.as_mut_ptr().add(slot) = 0;
            *up_scale.as_mut_ptr().add(slot) = 0;
            *down_weight.as_mut_ptr().add(slot) = 0;
            *down_scale.as_mut_ptr().add(slot) = 0;
            *slot_generation.as_mut_ptr().add(slot) = next_generation;
        }
    }

    #[kernel]
    pub fn initialize_expert_slot_resolve(
        mut miss_control: DisjointSlice<i32>,
        miss_capacity: u32,
        route_capacity: u32,
    ) {
        let index = thread::index_1d().get() as usize;
        let control_len = (miss_capacity as usize)
            .saturating_add(route_capacity as usize)
            .saturating_add(2);
        if index >= control_len {
            return;
        }
        unsafe {
            *miss_control.as_mut_ptr().add(index) = if index < 2 { 0 } else { -1 };
        }
    }

    /// Resolve every device-produced route independently through the
    /// generation-stamped stable slot table. Miss compaction is atomic; no
    /// thread serially traverses the route array.
    #[kernel]
    pub fn resolve_expert_slots(
        expert_ids: &[i32],
        expert_to_slot: &[i32],
        expert_generation: &[i32],
        slot_generation: &[i32],
        mut route_slots: DisjointSlice<i32>,
        mut route_generations: DisjointSlice<i32>,
        mut miss_markers: DisjointSlice<i32>,
        mut miss_control: DisjointSlice<i32>,
        route_count: u32,
        expert_capacity: u32,
        slot_capacity: u32,
        miss_capacity: u32,
    ) {
        let route = thread::index_1d().get() as usize;
        let route_count = (route_count as usize).min(expert_ids.len());
        if route >= route_count {
            return;
        }
        let expert_capacity = (expert_capacity as usize)
            .min(expert_to_slot.len())
            .min(expert_generation.len());
        let slot_capacity = (slot_capacity as usize).min(slot_generation.len());
        let expert = expert_ids[route];
        let mut slot = -1i32;
        let mut generation = 0i32;
        if expert >= 0 && (expert as usize) < expert_capacity {
            let expert_index = expert as usize;
            let mapped_slot = expert_to_slot[expert_index];
            let expert_gen = expert_generation[expert_index];
            if mapped_slot >= 0
                && (mapped_slot as usize) < slot_capacity
                && expert_gen > 0
                && slot_generation[mapped_slot as usize] == expert_gen
            {
                slot = mapped_slot;
                generation = expert_gen;
            }
        }
        unsafe {
            *route_slots.as_mut_ptr().add(route) = slot;
            *route_generations.as_mut_ptr().add(route) = generation;
            *miss_markers.as_mut_ptr().add(route) = if slot < 0 { 1 } else { 0 };
            *miss_control
                .as_mut_ptr()
                .add(2 + miss_capacity as usize + route) = expert;
        }
        if slot < 0 {
            let control = miss_control.as_mut_ptr();
            let miss = atomic_fetch_add_i32(control, 1);
            if miss >= 0 && (miss as u32) < miss_capacity {
                unsafe {
                    *control.add(2 + miss as usize) = expert;
                }
            } else {
                atomic_fetch_or_i32(unsafe { control.add(1) }, 1);
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

    /// Converts combined `[window ring | compressed]` indices into logical
    /// token indices plus plane selectors without reading any KV values.
    #[kernel]
    pub fn convert_combined_ring_topk_indices(
        combined: &[i32],
        row_window_lens: &[i32],
        mut logical_indices: DisjointSlice<i32>,
        mut plane_selectors: DisjointSlice<i32>,
        elements: u32,
        rows: u32,
        topk: u32,
        start_position: u32,
        position_stride: u32,
        window_size: u32,
        explicit_window_lens: u32,
    ) {
        let index = thread::index_1d().get() as usize;
        if index >= elements as usize || topk == 0 || window_size == 0 {
            return;
        }
        let logical_ptr = logical_indices.as_mut_ptr();
        let selector_ptr = plane_selectors.as_mut_ptr();
        let row = index / topk as usize;
        if row >= rows as usize {
            return;
        }
        let position = start_position as u64 + row as u64 * position_stride as u64;
        let window_size = window_size as u64;
        let maximum_visible = (position + 1).min(window_size);
        let valid_window_len = if explicit_window_lens != 0 {
            match row_window_lens.get(row) {
                Some(value) if *value >= 0 && (*value as u64) <= maximum_visible => *value as u64,
                _ => 0,
            }
        } else {
            maximum_visible
        };
        let combined_index = combined.get(index).copied().unwrap_or(-1);
        let mut logical = -1i32;
        let mut selector = -1i32;
        if combined_index >= 0 {
            let combined_index = combined_index as u64;
            if combined_index >= window_size {
                let compressed = combined_index - window_size;
                if compressed <= i32::MAX as u64 {
                    logical = compressed as i32;
                    selector = 1;
                }
            } else if position < window_size {
                if combined_index < valid_window_len && combined_index <= position {
                    logical = combined_index as i32;
                    selector = 0;
                }
            } else {
                let current_slot = position % window_size;
                let age = (current_slot + window_size - combined_index) % window_size;
                if age < valid_window_len {
                    let absolute = position - age;
                    if absolute <= i32::MAX as u64 {
                        logical = absolute as i32;
                        selector = 0;
                    }
                }
            }
        }
        unsafe {
            *logical_ptr.add(index) = logical;
            *selector_ptr.add(index) = selector;
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

    /// DSV4-specialized sparse attention: one warp owns one `(token, head)`
    /// pair and each lane owns 16 of the 512 output dimensions. Dot products
    /// use a warp butterfly reduction; output accumulators stay in registers
    /// across the full top-k loop.
    #[kernel]
    pub fn sparse_attn_warp_sink_f32_d512(
        q: &[f32],
        kv: &[f32],
        topk: &[i32],
        sink: &[f32],
        mut output: DisjointSlice<f32>,
        num_pairs: u32,
        kv_len: u32,
        heads: u32,
        topk_len: u32,
        softmax_scale: f32,
    ) {
        let pair = thread::blockIdx_x() as usize;
        let lane = thread::threadIdx_x() as usize;
        if pair >= num_pairs as usize || lane >= 32 {
            return;
        }
        let heads = heads as usize;
        let token = pair / heads;
        let head = pair % heads;
        let topk_len = topk_len as usize;
        let kv_len = kv_len as usize;
        let q_off = pair * 512;
        let topk_off = token * topk_len;

        let mut q_values = [0.0f32; 16];
        let mut output_values = [0.0f32; 16];
        let mut local = 0usize;
        while local < 16 {
            q_values[local] = q[q_off + lane + local * 32];
            local += 1;
        }

        let sink_value = sink.get(head).copied().unwrap_or(f32::NEG_INFINITY);
        let mut max_score = sink_value;
        let mut slot = 0usize;
        while slot < topk_len {
            let kv_index = topk[topk_off + slot];
            if kv_index >= 0 && (kv_index as usize) < kv_len {
                let kv_off = kv_index as usize * 512;
                let mut partial = 0.0f32;
                let mut local = 0usize;
                while local < 16 {
                    partial += q_values[local] * kv[kv_off + lane + local * 32];
                    local += 1;
                }
                partial += cuda_device::warp::shuffle_xor_f32(partial, 16);
                partial += cuda_device::warp::shuffle_xor_f32(partial, 8);
                partial += cuda_device::warp::shuffle_xor_f32(partial, 4);
                partial += cuda_device::warp::shuffle_xor_f32(partial, 2);
                partial += cuda_device::warp::shuffle_xor_f32(partial, 1);
                let score = partial * softmax_scale;
                if score > max_score {
                    max_score = score;
                }
            }
            slot += 1;
        }

        let mut denominator = fast_exp(sink_value - max_score);
        let mut slot = 0usize;
        while slot < topk_len {
            let kv_index = topk[topk_off + slot];
            if kv_index >= 0 && (kv_index as usize) < kv_len {
                let kv_off = kv_index as usize * 512;
                let mut kv_values = [0.0f32; 16];
                let mut partial = 0.0f32;
                let mut local = 0usize;
                while local < 16 {
                    let value = kv[kv_off + lane + local * 32];
                    kv_values[local] = value;
                    partial += q_values[local] * value;
                    local += 1;
                }
                partial += cuda_device::warp::shuffle_xor_f32(partial, 16);
                partial += cuda_device::warp::shuffle_xor_f32(partial, 8);
                partial += cuda_device::warp::shuffle_xor_f32(partial, 4);
                partial += cuda_device::warp::shuffle_xor_f32(partial, 2);
                partial += cuda_device::warp::shuffle_xor_f32(partial, 1);
                let weight = fast_exp(partial * softmax_scale - max_score);
                denominator += weight;
                let mut local = 0usize;
                while local < 16 {
                    output_values[local] += weight * kv_values[local];
                    local += 1;
                }
            }
            slot += 1;
        }

        let out_off = pair * 512;
        let out_ptr = output.as_mut_ptr();
        let mut local = 0usize;
        while local < 16 {
            unsafe {
                *out_ptr.add(out_off + lane + local * 32) = output_values[local] / denominator;
            }
            local += 1;
        }
    }

    /// Paged sparse attention over one contiguous physical plane. Each thread
    /// owns one `(query_token, head)` pair and resolves logical top-k tokens
    /// through the sequence's packed physical block table.
    #[kernel]
    pub fn paged_sparse_attn_tiled_sink_f32(
        q: &[f32],
        plane: &[f32],
        block_slots: &[i32],
        sequence_block_offsets: &[i32],
        sequence_kv_lens: &[i32],
        row_sequence_ids: &[i32],
        row_kv_lens: &[i32],
        topk: &[i32],
        sink: &[f32],
        mut output: DisjointSlice<f32>,
        num_pairs: u32,
        tokens_per_sequence: u32,
        heads: u32,
        head_dim: u32,
        topk_len: u32,
        page_tokens: u32,
        elements_per_token: u32,
        layer_index: u32,
        layer_count: u32,
        use_row_sequence_ids: u32,
        use_row_kv_lens: u32,
        softmax_scale: f32,
    ) {
        let pair = thread::index_1d().get() as usize;
        if pair >= num_pairs as usize {
            return;
        }
        let heads = heads as usize;
        let token = pair / heads;
        let head = pair % heads;
        let sequence = paged_row_sequence(
            row_sequence_ids,
            token,
            token / tokens_per_sequence as usize,
            use_row_sequence_ids,
        );
        let head_dim = head_dim as usize;
        let topk_len = topk_len as usize;
        let q_offset = pair * head_dim;
        let topk_offset = token * topk_len;
        let sink_value = sink.get(head).copied().unwrap_or(f32::NEG_INFINITY);

        let mut max_score = sink_value;
        let mut topk_slot = 0usize;
        while topk_slot < topk_len {
            let logical_token = paged_row_visible_token(
                topk.get(topk_offset + topk_slot).copied().unwrap_or(-1),
                row_kv_lens,
                token,
                use_row_kv_lens,
            );
            let kv_offset = paged_sparse_kv_offset(
                block_slots,
                sequence_block_offsets,
                sequence_kv_lens,
                sequence,
                logical_token,
                page_tokens as usize,
                elements_per_token as usize,
                head_dim,
                layer_index as usize,
                layer_count as usize,
                plane.len(),
            );
            if kv_offset != usize::MAX {
                let mut dot = 0.0f32;
                let mut dimension = 0usize;
                while dimension < head_dim {
                    dot += q[q_offset + dimension] * plane[kv_offset + dimension];
                    dimension += 1;
                }
                let score = dot * softmax_scale;
                if score > max_score {
                    max_score = score;
                }
            }
            topk_slot += 1;
        }

        let output_offset = pair * head_dim;
        let output_ptr = output.as_mut_ptr();
        let mut dimension = 0usize;
        while dimension < head_dim {
            unsafe {
                *output_ptr.add(output_offset + dimension) = 0.0;
            }
            dimension += 1;
        }
        let mut denominator = fast_exp(sink_value - max_score);
        let mut topk_slot = 0usize;
        while topk_slot < topk_len {
            let logical_token = paged_row_visible_token(
                topk.get(topk_offset + topk_slot).copied().unwrap_or(-1),
                row_kv_lens,
                token,
                use_row_kv_lens,
            );
            let kv_offset = paged_sparse_kv_offset(
                block_slots,
                sequence_block_offsets,
                sequence_kv_lens,
                sequence,
                logical_token,
                page_tokens as usize,
                elements_per_token as usize,
                head_dim,
                layer_index as usize,
                layer_count as usize,
                plane.len(),
            );
            if kv_offset != usize::MAX {
                let mut dot = 0.0f32;
                let mut dimension = 0usize;
                while dimension < head_dim {
                    dot += q[q_offset + dimension] * plane[kv_offset + dimension];
                    dimension += 1;
                }
                let weight = fast_exp(dot * softmax_scale - max_score);
                denominator += weight;
                let mut dimension = 0usize;
                while dimension < head_dim {
                    unsafe {
                        *output_ptr.add(output_offset + dimension) +=
                            weight * plane[kv_offset + dimension];
                    }
                    dimension += 1;
                }
            }
            topk_slot += 1;
        }
        let mut dimension = 0usize;
        while dimension < head_dim {
            unsafe {
                *output_ptr.add(output_offset + dimension) /= denominator;
            }
            dimension += 1;
        }
    }

    /// Warp-specialized paged sparse attention for a 512-element value vector.
    #[kernel]
    pub fn paged_sparse_attn_warp_sink_f32_d512(
        q: &[f32],
        plane: &[f32],
        block_slots: &[i32],
        sequence_block_offsets: &[i32],
        sequence_kv_lens: &[i32],
        row_sequence_ids: &[i32],
        row_kv_lens: &[i32],
        topk: &[i32],
        sink: &[f32],
        mut output: DisjointSlice<f32>,
        num_pairs: u32,
        tokens_per_sequence: u32,
        heads: u32,
        topk_len: u32,
        page_tokens: u32,
        elements_per_token: u32,
        layer_index: u32,
        layer_count: u32,
        use_row_sequence_ids: u32,
        use_row_kv_lens: u32,
        softmax_scale: f32,
    ) {
        let pair = thread::blockIdx_x() as usize;
        let lane = thread::threadIdx_x() as usize;
        if pair >= num_pairs as usize || lane >= 32 {
            return;
        }
        let heads = heads as usize;
        let token = pair / heads;
        let head = pair % heads;
        let sequence = paged_row_sequence(
            row_sequence_ids,
            token,
            token / tokens_per_sequence as usize,
            use_row_sequence_ids,
        );
        let topk_len = topk_len as usize;
        let topk_offset = token * topk_len;
        let q_offset = pair * 512;
        let mut q_values = [0.0f32; 16];
        let mut output_values = [0.0f32; 16];
        let mut local = 0usize;
        while local < 16 {
            q_values[local] = q[q_offset + lane + local * 32];
            local += 1;
        }

        let sink_value = sink.get(head).copied().unwrap_or(f32::NEG_INFINITY);
        let mut max_score = sink_value;
        let mut topk_slot = 0usize;
        while topk_slot < topk_len {
            let logical_token = paged_row_visible_token(
                topk.get(topk_offset + topk_slot).copied().unwrap_or(-1),
                row_kv_lens,
                token,
                use_row_kv_lens,
            );
            let kv_offset = paged_sparse_kv_offset(
                block_slots,
                sequence_block_offsets,
                sequence_kv_lens,
                sequence,
                logical_token,
                page_tokens as usize,
                elements_per_token as usize,
                512,
                layer_index as usize,
                layer_count as usize,
                plane.len(),
            );
            if kv_offset != usize::MAX {
                let mut partial = 0.0f32;
                let mut local = 0usize;
                while local < 16 {
                    partial += q_values[local] * plane[kv_offset + lane + local * 32];
                    local += 1;
                }
                partial += cuda_device::warp::shuffle_xor_f32(partial, 16);
                partial += cuda_device::warp::shuffle_xor_f32(partial, 8);
                partial += cuda_device::warp::shuffle_xor_f32(partial, 4);
                partial += cuda_device::warp::shuffle_xor_f32(partial, 2);
                partial += cuda_device::warp::shuffle_xor_f32(partial, 1);
                let score = partial * softmax_scale;
                if score > max_score {
                    max_score = score;
                }
            }
            topk_slot += 1;
        }

        let mut denominator = fast_exp(sink_value - max_score);
        let mut topk_slot = 0usize;
        while topk_slot < topk_len {
            let logical_token = paged_row_visible_token(
                topk.get(topk_offset + topk_slot).copied().unwrap_or(-1),
                row_kv_lens,
                token,
                use_row_kv_lens,
            );
            let kv_offset = paged_sparse_kv_offset(
                block_slots,
                sequence_block_offsets,
                sequence_kv_lens,
                sequence,
                logical_token,
                page_tokens as usize,
                elements_per_token as usize,
                512,
                layer_index as usize,
                layer_count as usize,
                plane.len(),
            );
            if kv_offset != usize::MAX {
                let mut kv_values = [0.0f32; 16];
                let mut partial = 0.0f32;
                let mut local = 0usize;
                while local < 16 {
                    let value = plane[kv_offset + lane + local * 32];
                    kv_values[local] = value;
                    partial += q_values[local] * value;
                    local += 1;
                }
                partial += cuda_device::warp::shuffle_xor_f32(partial, 16);
                partial += cuda_device::warp::shuffle_xor_f32(partial, 8);
                partial += cuda_device::warp::shuffle_xor_f32(partial, 4);
                partial += cuda_device::warp::shuffle_xor_f32(partial, 2);
                partial += cuda_device::warp::shuffle_xor_f32(partial, 1);
                let weight = fast_exp(partial * softmax_scale - max_score);
                denominator += weight;
                let mut local = 0usize;
                while local < 16 {
                    output_values[local] += weight * kv_values[local];
                    local += 1;
                }
            }
            topk_slot += 1;
        }

        let output_offset = pair * 512;
        let output_ptr = output.as_mut_ptr();
        let mut local = 0usize;
        while local < 16 {
            unsafe {
                *output_ptr.add(output_offset + lane + local * 32) =
                    output_values[local] / denominator;
            }
            local += 1;
        }
    }

    /// Dual-plane paged sparse attention. A selector beside each top-k entry
    /// chooses plane zero or one; both planes share the physical block table.
    #[kernel]
    pub fn dual_plane_paged_sparse_attn_tiled_sink_f32(
        q: &[f32],
        first_plane: &[f32],
        second_plane: &[f32],
        block_slots: &[i32],
        sequence_block_offsets: &[i32],
        sequence_kv_lens: &[i32],
        row_sequence_ids: &[i32],
        row_kv_lens: &[i32],
        topk: &[i32],
        selectors: &[i32],
        sink: &[f32],
        mut output: DisjointSlice<f32>,
        num_pairs: u32,
        tokens_per_sequence: u32,
        heads: u32,
        head_dim: u32,
        topk_len: u32,
        page_tokens: u32,
        first_elements_per_token: u32,
        second_elements_per_token: u32,
        layer_index: u32,
        layer_count: u32,
        use_row_sequence_ids: u32,
        use_row_kv_lens: u32,
        softmax_scale: f32,
    ) {
        let pair = thread::index_1d().get() as usize;
        if pair >= num_pairs as usize {
            return;
        }
        let heads = heads as usize;
        let token = pair / heads;
        let head = pair % heads;
        let sequence = paged_row_sequence(
            row_sequence_ids,
            token,
            token / tokens_per_sequence as usize,
            use_row_sequence_ids,
        );
        let head_dim = head_dim as usize;
        let topk_len = topk_len as usize;
        let q_offset = pair * head_dim;
        let topk_offset = token * topk_len;
        let sink_value = sink.get(head).copied().unwrap_or(f32::NEG_INFINITY);
        let mut max_score = sink_value;
        let mut item = 0usize;
        while item < topk_len {
            let entry = topk_offset + item;
            let selector = selectors.get(entry).copied().unwrap_or(-1);
            let logical_token = paged_row_visible_token(
                topk.get(entry).copied().unwrap_or(-1),
                row_kv_lens,
                token,
                use_row_kv_lens,
            );
            let kv_offset = if selector == 0 {
                paged_sparse_kv_offset(
                    block_slots,
                    sequence_block_offsets,
                    sequence_kv_lens,
                    sequence,
                    logical_token,
                    page_tokens as usize,
                    first_elements_per_token as usize,
                    head_dim,
                    layer_index as usize,
                    layer_count as usize,
                    first_plane.len(),
                )
            } else if selector == 1 {
                paged_sparse_kv_offset(
                    block_slots,
                    sequence_block_offsets,
                    sequence_kv_lens,
                    sequence,
                    logical_token,
                    page_tokens as usize,
                    second_elements_per_token as usize,
                    head_dim,
                    layer_index as usize,
                    layer_count as usize,
                    second_plane.len(),
                )
            } else {
                usize::MAX
            };
            if kv_offset != usize::MAX {
                let mut dot = 0.0f32;
                let mut d = 0usize;
                while d < head_dim {
                    let value = if selector == 0 {
                        first_plane[kv_offset + d]
                    } else {
                        second_plane[kv_offset + d]
                    };
                    dot += q[q_offset + d] * value;
                    d += 1;
                }
                let score = dot * softmax_scale;
                if score > max_score {
                    max_score = score;
                }
            }
            item += 1;
        }

        let out_offset = pair * head_dim;
        let out_ptr = output.as_mut_ptr();
        let mut d = 0usize;
        while d < head_dim {
            unsafe { *out_ptr.add(out_offset + d) = 0.0 };
            d += 1;
        }
        let mut denominator = fast_exp(sink_value - max_score);
        item = 0;
        while item < topk_len {
            let entry = topk_offset + item;
            let selector = selectors.get(entry).copied().unwrap_or(-1);
            let logical_token = paged_row_visible_token(
                topk.get(entry).copied().unwrap_or(-1),
                row_kv_lens,
                token,
                use_row_kv_lens,
            );
            let kv_offset = if selector == 0 {
                paged_sparse_kv_offset(
                    block_slots,
                    sequence_block_offsets,
                    sequence_kv_lens,
                    sequence,
                    logical_token,
                    page_tokens as usize,
                    first_elements_per_token as usize,
                    head_dim,
                    layer_index as usize,
                    layer_count as usize,
                    first_plane.len(),
                )
            } else if selector == 1 {
                paged_sparse_kv_offset(
                    block_slots,
                    sequence_block_offsets,
                    sequence_kv_lens,
                    sequence,
                    logical_token,
                    page_tokens as usize,
                    second_elements_per_token as usize,
                    head_dim,
                    layer_index as usize,
                    layer_count as usize,
                    second_plane.len(),
                )
            } else {
                usize::MAX
            };
            if kv_offset != usize::MAX {
                let mut dot = 0.0f32;
                let mut d = 0usize;
                while d < head_dim {
                    let value = if selector == 0 {
                        first_plane[kv_offset + d]
                    } else {
                        second_plane[kv_offset + d]
                    };
                    dot += q[q_offset + d] * value;
                    d += 1;
                }
                let weight = fast_exp(dot * softmax_scale - max_score);
                denominator += weight;
                d = 0;
                while d < head_dim {
                    let value = if selector == 0 {
                        first_plane[kv_offset + d]
                    } else {
                        second_plane[kv_offset + d]
                    };
                    unsafe { *out_ptr.add(out_offset + d) += weight * value };
                    d += 1;
                }
            }
            item += 1;
        }
        d = 0;
        while d < head_dim {
            unsafe { *out_ptr.add(out_offset + d) /= denominator };
            d += 1;
        }
    }

    /// Warp-specialized dual-plane paged sparse attention for 512-element rows.
    #[kernel]
    pub fn dual_plane_paged_sparse_attn_warp_sink_f32_d512(
        q: &[f32],
        first_plane: &[f32],
        second_plane: &[f32],
        block_slots: &[i32],
        sequence_block_offsets: &[i32],
        sequence_kv_lens: &[i32],
        row_sequence_ids: &[i32],
        row_kv_lens: &[i32],
        topk: &[i32],
        selectors: &[i32],
        sink: &[f32],
        mut output: DisjointSlice<f32>,
        num_pairs: u32,
        tokens_per_sequence: u32,
        heads: u32,
        topk_len: u32,
        page_tokens: u32,
        first_elements_per_token: u32,
        second_elements_per_token: u32,
        layer_index: u32,
        layer_count: u32,
        use_row_sequence_ids: u32,
        use_row_kv_lens: u32,
        softmax_scale: f32,
    ) {
        let pair = thread::blockIdx_x() as usize;
        let lane = thread::threadIdx_x() as usize;
        if pair >= num_pairs as usize || lane >= 32 {
            return;
        }
        let heads = heads as usize;
        let token = pair / heads;
        let head = pair % heads;
        let sequence = paged_row_sequence(
            row_sequence_ids,
            token,
            token / tokens_per_sequence as usize,
            use_row_sequence_ids,
        );
        let topk_len = topk_len as usize;
        let topk_offset = token * topk_len;
        let mut q_values = [0.0f32; 16];
        let mut output_values = [0.0f32; 16];
        let mut local = 0usize;
        while local < 16 {
            q_values[local] = q[pair * 512 + lane + local * 32];
            local += 1;
        }
        let sink_value = sink.get(head).copied().unwrap_or(f32::NEG_INFINITY);
        let mut max_score = sink_value;
        let mut item = 0usize;
        while item < topk_len {
            let entry = topk_offset + item;
            let selector = selectors.get(entry).copied().unwrap_or(-1);
            let logical_token = paged_row_visible_token(
                topk.get(entry).copied().unwrap_or(-1),
                row_kv_lens,
                token,
                use_row_kv_lens,
            );
            let kv_offset = if selector == 0 {
                paged_sparse_kv_offset(
                    block_slots,
                    sequence_block_offsets,
                    sequence_kv_lens,
                    sequence,
                    logical_token,
                    page_tokens as usize,
                    first_elements_per_token as usize,
                    512,
                    layer_index as usize,
                    layer_count as usize,
                    first_plane.len(),
                )
            } else if selector == 1 {
                paged_sparse_kv_offset(
                    block_slots,
                    sequence_block_offsets,
                    sequence_kv_lens,
                    sequence,
                    logical_token,
                    page_tokens as usize,
                    second_elements_per_token as usize,
                    512,
                    layer_index as usize,
                    layer_count as usize,
                    second_plane.len(),
                )
            } else {
                usize::MAX
            };
            if kv_offset != usize::MAX {
                let mut partial = 0.0f32;
                local = 0;
                while local < 16 {
                    let value = if selector == 0 {
                        first_plane[kv_offset + lane + local * 32]
                    } else {
                        second_plane[kv_offset + lane + local * 32]
                    };
                    partial += q_values[local] * value;
                    local += 1;
                }
                partial += cuda_device::warp::shuffle_xor_f32(partial, 16);
                partial += cuda_device::warp::shuffle_xor_f32(partial, 8);
                partial += cuda_device::warp::shuffle_xor_f32(partial, 4);
                partial += cuda_device::warp::shuffle_xor_f32(partial, 2);
                partial += cuda_device::warp::shuffle_xor_f32(partial, 1);
                let score = partial * softmax_scale;
                if score > max_score {
                    max_score = score;
                }
            }
            item += 1;
        }
        let mut denominator = fast_exp(sink_value - max_score);
        item = 0;
        while item < topk_len {
            let entry = topk_offset + item;
            let selector = selectors.get(entry).copied().unwrap_or(-1);
            let logical_token = paged_row_visible_token(
                topk.get(entry).copied().unwrap_or(-1),
                row_kv_lens,
                token,
                use_row_kv_lens,
            );
            let kv_offset = if selector == 0 {
                paged_sparse_kv_offset(
                    block_slots,
                    sequence_block_offsets,
                    sequence_kv_lens,
                    sequence,
                    logical_token,
                    page_tokens as usize,
                    first_elements_per_token as usize,
                    512,
                    layer_index as usize,
                    layer_count as usize,
                    first_plane.len(),
                )
            } else if selector == 1 {
                paged_sparse_kv_offset(
                    block_slots,
                    sequence_block_offsets,
                    sequence_kv_lens,
                    sequence,
                    logical_token,
                    page_tokens as usize,
                    second_elements_per_token as usize,
                    512,
                    layer_index as usize,
                    layer_count as usize,
                    second_plane.len(),
                )
            } else {
                usize::MAX
            };
            if kv_offset != usize::MAX {
                let mut values = [0.0f32; 16];
                let mut partial = 0.0f32;
                local = 0;
                while local < 16 {
                    let value = if selector == 0 {
                        first_plane[kv_offset + lane + local * 32]
                    } else {
                        second_plane[kv_offset + lane + local * 32]
                    };
                    values[local] = value;
                    partial += q_values[local] * value;
                    local += 1;
                }
                partial += cuda_device::warp::shuffle_xor_f32(partial, 16);
                partial += cuda_device::warp::shuffle_xor_f32(partial, 8);
                partial += cuda_device::warp::shuffle_xor_f32(partial, 4);
                partial += cuda_device::warp::shuffle_xor_f32(partial, 2);
                partial += cuda_device::warp::shuffle_xor_f32(partial, 1);
                let weight = fast_exp(partial * softmax_scale - max_score);
                denominator += weight;
                local = 0;
                while local < 16 {
                    output_values[local] += weight * values[local];
                    local += 1;
                }
            }
            item += 1;
        }
        let out_ptr = output.as_mut_ptr();
        local = 0;
        while local < 16 {
            unsafe {
                *out_ptr.add(pair * 512 + lane + local * 32) = output_values[local] / denominator
            };
            local += 1;
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

    /// Find per-row top-k token indices and values from row-major vocab logits.
    /// One block owns each row and uses the same scan/insertion semantics as
    /// `topk_vocab`, including lower-token-id precedence for equal logits.
    #[kernel]
    pub fn topk_vocab_rows(
        logits: &[f32],
        mut out_idx: DisjointSlice<f32>,
        mut out_val: DisjointSlice<f32>,
        rows: u32,
        vocab: u32,
        k: u32,
    ) {
        static mut SMEM: SharedArray<f32, 1024> = SharedArray::UNINIT;
        let row = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let rows = rows as usize;
        let vocab = vocab as usize;
        let k = k as usize;
        let chunk = 1024usize;

        if row < rows && tid == 0 {
            let mut best_val = [f32::NEG_INFINITY; 40];
            let mut best_idx = [0u32; 40];
            let row_offset = row * vocab;

            let mut cursor = 0usize;
            while cursor < vocab {
                let n = (vocab - cursor).min(chunk);
                for i in 0..n {
                    unsafe {
                        SMEM[i] = logits[row_offset + cursor + i];
                    }
                }
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

            let output_offset = row * k;
            let idx_ptr = out_idx.as_mut_ptr();
            let val_ptr = out_val.as_mut_ptr();
            for j in 0..k {
                unsafe {
                    *idx_ptr.add(output_offset + j) = best_idx[j] as f32;
                    *val_ptr.add(output_offset + j) = best_val[j];
                }
            }
        }
    }

    /// Merge one chunk's row-wise top-k into persistent global row-wise top-k.
    /// Token IDs remain encoded as exact f32 integers because the vocabulary is
    /// well below the 24-bit exact-integer range of f32.
    #[kernel]
    pub fn merge_topk_rows_in_place(
        chunk_idx: &[f32],
        chunk_val: &[f32],
        mut global_idx: DisjointSlice<f32>,
        mut global_val: DisjointSlice<f32>,
        rows: u32,
        global_k: u32,
        chunk_k: u32,
        token_offset: u32,
        has_existing: u32,
    ) {
        let row = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let rows = rows as usize;
        let global_k = global_k as usize;
        let chunk_k = chunk_k as usize;
        if row >= rows || tid != 0 {
            return;
        }

        let mut best_val = [f32::NEG_INFINITY; 40];
        let mut best_idx = [u32::MAX; 40];
        let global_offset = row * global_k;
        let global_idx_ptr = global_idx.as_mut_ptr();
        let global_val_ptr = global_val.as_mut_ptr();
        if has_existing != 0 {
            let mut slot = 0usize;
            while slot < global_k {
                unsafe {
                    best_idx[slot] = *global_idx_ptr.add(global_offset + slot) as u32;
                    best_val[slot] = *global_val_ptr.add(global_offset + slot);
                }
                slot += 1;
            }
        }

        let chunk_offset = row * chunk_k;
        let mut candidate = 0usize;
        while candidate < chunk_k {
            let value = chunk_val[chunk_offset + candidate];
            let token = chunk_idx[chunk_offset + candidate] as u32 + token_offset;
            let mut position = global_k;
            while position > 0 {
                let previous = position - 1;
                let better = value > best_val[previous]
                    || (value == best_val[previous] && token < best_idx[previous]);
                if !better {
                    break;
                }
                position -= 1;
            }
            if position < global_k {
                let mut shift = global_k - 1;
                while shift > position {
                    best_val[shift] = best_val[shift - 1];
                    best_idx[shift] = best_idx[shift - 1];
                    shift -= 1;
                }
                best_val[position] = value;
                best_idx[position] = token;
            }
            candidate += 1;
        }

        let mut slot = 0usize;
        while slot < global_k {
            unsafe {
                *global_idx_ptr.add(global_offset + slot) = best_idx[slot] as f32;
                *global_val_ptr.add(global_offset + slot) = best_val[slot];
            }
            slot += 1;
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

    /// Scatter dense rows into one layer of contiguous paged-plane storage.
    ///
    /// `row_sequence_ids[row]` selects a sequence, while `positions[row]` selects
    /// its logical row within the packed block table. One thread owns
    /// one value, so the complete operation uses a single kernel launch.
    #[kernel]
    pub fn paged_plane_scatter_rows_f32(
        values: &[f32],
        positions: &[i32],
        block_slots: &[i32],
        block_offsets: &[i32],
        row_sequence_ids: &[i32],
        mask: &[i32],
        mut plane: DisjointSlice<f32>,
        num_elements: u32,
        plane_elements: u32,
        rows: u32,
        row_dim: u32,
        page_tokens: u32,
        layer_index: u32,
        layer_count: u32,
        use_row_sequence_ids: u32,
        use_mask: u32,
    ) {
        let idx = thread::index_1d().get();
        if (idx as u64) >= num_elements as u64 || row_dim == 0 {
            return;
        }
        let row_dim = row_dim as usize;
        let row = idx as usize / row_dim;
        if row >= rows as usize {
            return;
        }
        if use_mask != 0 {
            match mask.get(row) {
                Some(value) if *value != 0 => {}
                _ => return,
            }
        }
        let logical_row = match positions.get(row) {
            Some(value) if *value >= 0 => *value as usize,
            _ => return,
        };
        let sequence = paged_row_sequence(row_sequence_ids, row, row, use_row_sequence_ids);
        let base = paged_plane_row_offset(
            plane_elements as usize,
            block_slots,
            block_offsets,
            sequence,
            logical_row,
            page_tokens as usize,
            row_dim,
            layer_index as usize,
            layer_count as usize,
        );
        if base == usize::MAX {
            return;
        }
        let column = idx as usize - row * row_dim;
        let value = match values.get(idx as usize) {
            Some(value) => *value,
            None => return,
        };
        unsafe {
            *plane.as_mut_ptr().add(base + column) = value;
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

    /// Batched tail rotary for `[rows, heads, head_dim]` device rows.
    ///
    /// One thread owns one rotary pair and writes both elements, avoiding the
    /// read/write race that would happen if even/odd lanes updated a pair
    /// independently. Position is `start_position + row`.
    #[kernel]
    pub fn rope_tail_yaarn_rows(
        mut qk: DisjointSlice<f32>,
        cos_table: &[f32],
        sin_table: &[f32],
        num_pairs: u32,
        start_position: u32,
        rows: u32,
        heads: u32,
        head_dim: u32,
        rope_dim: u32,
        inverse: u32,
    ) {
        let idx = thread::index_1d().get();
        if (idx as u64) >= num_pairs as u64 {
            return;
        }
        let hd = head_dim as usize;
        let rd = rope_dim as usize;
        let heads = heads as usize;
        let rows = rows as usize;
        if rows == 0 || heads == 0 || rd == 0 || rd > hd {
            return;
        }
        let rd2 = rd / 2;
        if rd2 == 0 {
            return;
        }
        let pair_idx = idx as usize;
        let pairs_per_row = heads * rd2;
        let row = pair_idx / pairs_per_row;
        if row >= rows {
            return;
        }
        let rem = pair_idx - row * pairs_per_row;
        let head = rem / rd2;
        let pair = rem - head * rd2;
        let position = start_position as usize + row;
        let cos = cos_table[position * rd2 + pair];
        let sin = sin_table[position * rd2 + pair];
        let (s, c) = if inverse != 0 {
            (-sin, cos)
        } else {
            (sin, cos)
        };
        let row_stride = heads * hd;
        let tail_start = hd - rd;
        let base = row * row_stride + head * hd + tail_start + pair * 2;
        let ptr = qk.as_mut_ptr();
        let x0 = unsafe { *ptr.add(base) };
        let x1 = unsafe { *ptr.add(base + 1) };
        unsafe {
            *ptr.add(base) = x0 * c - x1 * s;
            *ptr.add(base + 1) = x0 * s + x1 * c;
        }
    }

    /// Batched tail rotary for `[rows, heads, head_dim]` where row position is
    /// `start_position + row * position_stride`. This is needed by compressed
    /// prefill, whose compressed rows correspond to source positions separated
    /// by `compress_ratio`.
    #[kernel]
    pub fn rope_tail_yaarn_rows_strided(
        mut qk: DisjointSlice<f32>,
        cos_table: &[f32],
        sin_table: &[f32],
        num_pairs: u32,
        start_position: u32,
        position_stride: u32,
        rows: u32,
        heads: u32,
        head_dim: u32,
        rope_dim: u32,
        inverse: u32,
    ) {
        let idx = thread::index_1d().get();
        if (idx as u64) >= num_pairs as u64 {
            return;
        }
        let hd = head_dim as usize;
        let rd = rope_dim as usize;
        let heads = heads as usize;
        let rows = rows as usize;
        if rows == 0 || heads == 0 || rd == 0 || rd > hd {
            return;
        }
        let rd2 = rd / 2;
        if rd2 == 0 {
            return;
        }
        let pair_idx = idx as usize;
        let pairs_per_row = heads * rd2;
        let row = pair_idx / pairs_per_row;
        if row >= rows {
            return;
        }
        let rem = pair_idx - row * pairs_per_row;
        let head = rem / rd2;
        let pair = rem - head * rd2;
        let position = start_position as usize + row * position_stride as usize;
        let cos = cos_table[position * rd2 + pair];
        let sin = sin_table[position * rd2 + pair];
        let (s, c) = if inverse != 0 {
            (-sin, cos)
        } else {
            (sin, cos)
        };
        let row_stride = heads * hd;
        let tail_start = hd - rd;
        let base = row * row_stride + head * hd + tail_start + pair * 2;
        let ptr = qk.as_mut_ptr();
        let x0 = unsafe { *ptr.add(base) };
        let x1 = unsafe { *ptr.add(base + 1) };
        unsafe {
            *ptr.add(base) = x0 * c - x1 * s;
            *ptr.add(base + 1) = x0 * s + x1 * c;
        }
    }

    /// Batched tail rotary for `[rows, heads, head_dim]` with an arbitrary
    /// device-resident position for every row.
    #[kernel]
    pub fn rope_tail_yaarn_rows_indexed(
        mut qk: DisjointSlice<f32>,
        cos_table: &[f32],
        sin_table: &[f32],
        positions: &[i32],
        num_pairs: u32,
        rows: u32,
        heads: u32,
        head_dim: u32,
        rope_dim: u32,
        inverse: u32,
    ) {
        let idx = thread::index_1d().get();
        if (idx as u64) >= num_pairs as u64 {
            return;
        }
        let hd = head_dim as usize;
        let rd = rope_dim as usize;
        let heads = heads as usize;
        let rows = rows as usize;
        if rows == 0 || heads == 0 || rd == 0 || rd > hd {
            return;
        }
        let rd2 = rd / 2;
        if rd2 == 0 {
            return;
        }
        let pair_idx = idx as usize;
        let pairs_per_row = heads * rd2;
        let row = pair_idx / pairs_per_row;
        if row >= rows {
            return;
        }
        let position = match positions.get(row) {
            Some(value) if *value >= 0 => *value as usize,
            _ => return,
        };
        let rem = pair_idx - row * pairs_per_row;
        let head = rem / rd2;
        let pair = rem - head * rd2;
        let table_offset = match position
            .checked_mul(rd2)
            .and_then(|base| base.checked_add(pair))
        {
            Some(value) => value,
            None => return,
        };
        let cos = match cos_table.get(table_offset) {
            Some(value) => *value,
            None => return,
        };
        let sin = match sin_table.get(table_offset) {
            Some(value) => *value,
            None => return,
        };
        let (s, c) = if inverse != 0 {
            (-sin, cos)
        } else {
            (sin, cos)
        };
        let row_stride = heads * hd;
        let tail_start = hd - rd;
        let base = row * row_stride + head * hd + tail_start + pair * 2;
        let ptr = qk.as_mut_ptr();
        let x0 = unsafe { *ptr.add(base) };
        let x1 = unsafe { *ptr.add(base + 1) };
        unsafe {
            *ptr.add(base) = x0 * c - x1 * s;
            *ptr.add(base + 1) = x0 * s + x1 * c;
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

    /// Gather compact decode dispatch metadata from a generation-stamped stable
    /// expert slot table. The route-sized launch is parallel; invalid bindings
    /// atomically mark `dispatch_error` and are made harmless to later kernels.
    #[kernel]
    pub fn gather_stable_moe_dispatch(
        table_gate_ptrs: &[u64],
        table_gate_scale_ptrs: &[u64],
        table_up_ptrs: &[u64],
        table_up_scale_ptrs: &[u64],
        table_down_ptrs: &[u64],
        table_down_scale_ptrs: &[u64],
        slot_generations: &[i32],
        resolved_slots: &[i32],
        resolved_generations: &[i32],
        router_weights: &[f32],
        active_markers: &[i32],
        active_value: i32,
        mut gate_ptrs: DisjointSlice<u64>,
        mut gate_scale_ptrs: DisjointSlice<u64>,
        mut up_ptrs: DisjointSlice<u64>,
        mut up_scale_ptrs: DisjointSlice<u64>,
        mut down_ptrs: DisjointSlice<u64>,
        mut down_scale_ptrs: DisjointSlice<u64>,
        mut route_weights: DisjointSlice<f32>,
        mut route_slots: DisjointSlice<i32>,
        mut dispatch_error: DisjointSlice<i32>,
        route_count: u32,
        slot_capacity: u32,
    ) {
        if thread::blockIdx_x() != 0 {
            return;
        }
        let rank = thread::threadIdx_x() as usize;
        if rank == 0 {
            unsafe {
                *dispatch_error.as_mut_ptr() = 0;
            }
        }
        thread::sync_threads();
        let route_count = (route_count as usize)
            .min(resolved_slots.len())
            .min(resolved_generations.len())
            .min(router_weights.len());
        if rank >= route_count {
            return;
        }
        let slot_capacity = (slot_capacity as usize)
            .min(slot_generations.len())
            .min(table_gate_ptrs.len())
            .min(table_gate_scale_ptrs.len())
            .min(table_up_ptrs.len())
            .min(table_up_scale_ptrs.len())
            .min(table_down_ptrs.len())
            .min(table_down_scale_ptrs.len());
        let slot = resolved_slots[rank];
        let generation = resolved_generations[rank];
        let slot_index = if slot >= 0 {
            slot as usize
        } else {
            slot_capacity
        };
        let active = rank < active_markers.len() && active_markers[rank] == active_value;
        let current = slot_index < slot_capacity
            && generation > 0
            && slot_generations[slot_index] == generation;
        let gate = if current {
            table_gate_ptrs[slot_index]
        } else {
            0
        };
        let gate_scale = if current {
            table_gate_scale_ptrs[slot_index]
        } else {
            0
        };
        let up = if current {
            table_up_ptrs[slot_index]
        } else {
            0
        };
        let up_scale = if current {
            table_up_scale_ptrs[slot_index]
        } else {
            0
        };
        let down = if current {
            table_down_ptrs[slot_index]
        } else {
            0
        };
        let down_scale = if current {
            table_down_scale_ptrs[slot_index]
        } else {
            0
        };
        let valid = active
            && current
            && gate != 0
            && gate_scale != 0
            && up != 0
            && up_scale != 0
            && down != 0
            && down_scale != 0;
        if active && !valid {
            atomic_fetch_or_i32(dispatch_error.as_mut_ptr(), 1);
        }
        unsafe {
            *gate_ptrs.as_mut_ptr().add(rank) = gate;
            *gate_scale_ptrs.as_mut_ptr().add(rank) = gate_scale;
            *up_ptrs.as_mut_ptr().add(rank) = up;
            *up_scale_ptrs.as_mut_ptr().add(rank) = up_scale;
            *down_ptrs.as_mut_ptr().add(rank) = down;
            *down_scale_ptrs.as_mut_ptr().add(rank) = down_scale;
            *route_weights.as_mut_ptr().add(rank) = if valid { router_weights[rank] } else { 0.0 };
            *route_slots.as_mut_ptr().add(rank) = if valid { rank as i32 } else { -1 };
        }
    }

    #[kernel]
    pub fn initialize_moe_segment_invocation(
        mut route_output: DisjointSlice<f32>,
        mut route_written: DisjointSlice<i32>,
        mut route_error: DisjointSlice<i32>,
        output_elements: u32,
        num_routes: u32,
    ) {
        let index = thread::index_1d().get() as usize;
        unsafe {
            if index < output_elements as usize {
                *route_output.as_mut_ptr().add(index) = 0.0;
            }
            if index < num_routes as usize {
                *route_written.as_mut_ptr().add(index) = 0;
            }
            if index == 0 {
                *route_error.as_mut_ptr() = 0;
            }
        }
    }

    #[kernel]
    pub fn initialize_moe_segment_grouping(
        mut slot_counts: DisjointSlice<i32>,
        mut slot_offsets: DisjointSlice<i32>,
        mut slot_cursors: DisjointSlice<i32>,
        mut segment_slots: DisjointSlice<i32>,
        mut segment_generations: DisjointSlice<i32>,
        mut segment_tokens: DisjointSlice<i32>,
        mut segment_routes: DisjointSlice<i32>,
        mut segment_weights: DisjointSlice<f32>,
        slot_capacity: u32,
        max_segments: u32,
    ) {
        let index = thread::index_1d().get() as usize;
        let slot_capacity = slot_capacity as usize;
        let max_segments = max_segments as usize;
        let metadata_capacity = max_segments * 8;
        unsafe {
            if index < slot_capacity {
                *slot_counts.as_mut_ptr().add(index) = 0;
            }
            if index < slot_capacity {
                *slot_offsets.as_mut_ptr().add(index) = 0;
            }
            if index < slot_capacity {
                *slot_cursors.as_mut_ptr().add(index) = 0;
            }
            if index < max_segments {
                *segment_slots.as_mut_ptr().add(index) = -1;
            }
            if index < max_segments {
                *segment_generations.as_mut_ptr().add(index) = 0;
            }
            if index < metadata_capacity {
                *segment_tokens.as_mut_ptr().add(index) = -1;
            }
            if index < metadata_capacity {
                *segment_routes.as_mut_ptr().add(index) = -1;
            }
            if index < metadata_capacity {
                *segment_weights.as_mut_ptr().add(index) = 0.0;
            }
        }
    }

    #[kernel]
    pub fn count_moe_routes_by_slot(
        route_slots: &[i32],
        route_generations: &[i32],
        slot_generations: &[i32],
        mut slot_counts: DisjointSlice<i32>,
        route_count: u32,
        slot_capacity: u32,
    ) {
        let route = thread::index_1d().get() as usize;
        if route >= route_count as usize
            || route >= route_slots.len()
            || route >= route_generations.len()
        {
            return;
        }
        let slot = route_slots[route];
        let generation = route_generations[route];
        if slot >= 0
            && (slot as u32) < slot_capacity
            && (slot as usize) < slot_generations.len()
            && generation > 0
            && slot_generations[slot as usize] == generation
        {
            atomic_fetch_add_i32(unsafe { slot_counts.as_mut_ptr().add(slot as usize) }, 1);
        }
    }

    /// Scan only the bounded stable-slot table. Route arrays are never traversed
    /// by this control kernel.
    #[kernel]
    pub fn scan_moe_slot_segments(
        slot_counts: &[i32],
        slot_generations: &[i32],
        mut slot_offsets: DisjointSlice<i32>,
        mut segment_slots: DisjointSlice<i32>,
        mut segment_generations: DisjointSlice<i32>,
        mut dispatch_error: DisjointSlice<i32>,
        slot_capacity: u32,
        max_segments: u32,
    ) {
        if thread::index_1d().get() != 0 {
            return;
        }
        let slots = (slot_capacity as usize)
            .min(512)
            .min(slot_counts.len())
            .min(slot_generations.len());
        let max_segments = max_segments as usize;
        let mut segment_base = 0usize;
        for slot in 0..slots {
            unsafe {
                *slot_offsets.as_mut_ptr().add(slot) = segment_base as i32;
            }
            let count = slot_counts[slot].max(0) as usize;
            let segments = (count + 7) / 8;
            let end = segment_base.saturating_add(segments);
            if end > max_segments {
                unsafe {
                    *dispatch_error.as_mut_ptr() = 1;
                }
                return;
            }
            for segment in segment_base..end {
                unsafe {
                    *segment_slots.as_mut_ptr().add(segment) = slot as i32;
                    *segment_generations.as_mut_ptr().add(segment) = slot_generations[slot];
                }
            }
            segment_base = end;
        }
    }

    #[kernel]
    pub fn scatter_moe_routes_to_segments(
        route_slots: &[i32],
        route_generations: &[i32],
        router_weights: &[f32],
        slot_generations: &[i32],
        slot_offsets: &[i32],
        mut slot_cursors: DisjointSlice<i32>,
        mut segment_tokens: DisjointSlice<i32>,
        mut segment_routes: DisjointSlice<i32>,
        mut segment_weights: DisjointSlice<f32>,
        mut dispatch_error: DisjointSlice<i32>,
        route_count: u32,
        routes_per_token: u32,
        slot_capacity: u32,
        max_segments: u32,
    ) {
        let route = thread::index_1d().get() as usize;
        if route >= route_count as usize
            || route >= route_slots.len()
            || route >= route_generations.len()
            || route >= router_weights.len()
            || routes_per_token == 0
        {
            return;
        }
        let slot = route_slots[route];
        let generation = route_generations[route];
        if slot < 0
            || (slot as u32) >= slot_capacity
            || (slot as usize) >= slot_generations.len()
            || generation <= 0
            || slot_generations[slot as usize] != generation
        {
            return;
        }
        let position =
            atomic_fetch_add_i32(unsafe { slot_cursors.as_mut_ptr().add(slot as usize) }, 1);
        let segment = slot_offsets[slot as usize] + position / 8;
        let column = position % 8;
        if position < 0 || segment < 0 || segment as u32 >= max_segments {
            atomic_fetch_or_i32(dispatch_error.as_mut_ptr(), 1);
            return;
        }
        let metadata = segment as usize * 8 + column as usize;
        unsafe {
            *segment_tokens.as_mut_ptr().add(metadata) = (route as u32 / routes_per_token) as i32;
            *segment_routes.as_mut_ptr().add(metadata) = route as i32;
            *segment_weights.as_mut_ptr().add(metadata) = router_weights[route];
        }
    }

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
        if gate_ptr.is_null()
            || gate_scale_ptr.is_null()
            || up_ptr.is_null()
            || up_scale_ptr.is_null()
        {
            let out_off = expert * n as usize + row;
            unsafe {
                *y_gate.as_mut_ptr().add(out_off) = 0.0;
                *y_up.as_mut_ptr().add(out_off) = 0.0;
            }
            return;
        }

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
        if gate_ptr.is_null()
            || gate_scale_ptr.is_null()
            || up_ptr.is_null()
            || up_scale_ptr.is_null()
        {
            let group = tid / 4;
            let thr = tid % 4;
            let mut j = 0usize;
            while j < 4 {
                let row = row_base + group + if j >= 2 { 8 } else { 0 };
                let col = thr * 2 + (j & 1);
                if row < n && col < batch_cols {
                    let out_off = expert * batch_cols * n + col * n + row;
                    unsafe {
                        *y_gate.as_mut_ptr().add(out_off) = 0.0;
                        *y_up.as_mut_ptr().add(out_off) = 0.0;
                    }
                }
                j += 1;
            }
            return;
        }
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
        let rw = route_weights[expert * batch_cols + batch_col];
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

    /// Batched down FP4 GEMV for all selected experts.
    ///
    /// Grid: `(ceil(hidden_size / 256), num_experts)`.
    /// Each thread computes one output element for one expert, reading from
    /// the expert's down weights and the shared `y_hidden` buffer. Results
    /// overwrite expert-major scratch `[expert, 0, hidden_row]`.
    #[kernel]
    pub fn moe_gemv_down_fp4_batched(
        y_hidden: &[f32],
        down_ptrs: &[u64],
        down_scale_ptrs: &[u64],
        mut expert_output: DisjointSlice<f32>,
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
        if down_ptr.is_null() || down_scale_ptr.is_null() {
            let out_off = expert * hidden_size as usize + row;
            unsafe {
                *expert_output.as_mut_ptr().add(out_off) = 0.0;
            }
            return;
        }

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
        let out_off = expert * hidden_size as usize + row;
        unsafe {
            *expert_output.as_mut_ptr().add(out_off) = dot;
        }
    }

    /// Batched down FP4 GEMM using Blackwell mxf4 Tensor Cores.
    ///
    /// Grid: `(ceil(hidden_size / 16), num_experts)`, blockDim.x = 32.
    /// `y_hidden_packed` is laid out as `[expert, batch_col, intermediate/2]`;
    /// results overwrite expert-major scratch `[expert, batch_col, hidden_row]`.
    #[kernel]
    pub unsafe fn moe_gemm_down_fp4_mxf4_batched(
        y_hidden_packed: &[u8],
        y_hidden_scales: &[u8],
        down_ptrs: &[u64],
        down_scale_ptrs: &[u64],
        mut expert_output: DisjointSlice<f32>,
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
        if down_ptr.is_null() || down_scale_ptr.is_null() {
            let group = tid / 4;
            let thr = tid % 4;
            let mut j = 0usize;
            while j < 4 {
                let row = row_base + group + if j >= 2 { 8 } else { 0 };
                let col = thr * 2 + (j & 1);
                if row < hidden && col < batch_cols {
                    let out_off = (expert * batch_cols + col) * hidden + row;
                    unsafe {
                        *expert_output.as_mut_ptr().add(out_off) = 0.0;
                    }
                }
                j += 1;
            }
            return;
        }
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
        let out_ptr = expert_output.as_mut_ptr();
        let mut j = 0usize;
        while j < 4 {
            let row = row_base + group + if j >= 2 { 8 } else { 0 };
            let col = thr * 2 + (j & 1);
            if row < hidden && col < batch_cols {
                let out_off = (expert * batch_cols + col) * hidden + row;
                unsafe {
                    *out_ptr.add(out_off) = acc[j];
                }
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
        if gate_ptr.is_null()
            || gate_scale_ptr.is_null()
            || up_ptr.is_null()
            || up_scale_ptr.is_null()
        {
            if tid == 0 {
                let out_off = expert * n as usize + row;
                unsafe {
                    *y_gate.as_mut_ptr().add(out_off) = 0.0;
                    *y_up.as_mut_ptr().add(out_off) = 0.0;
                }
            }
            return;
        }

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

    /// Experimental block-reduction down FP4 GEMV.
    ///
    /// Grid: `(hidden_size, num_experts)`. Each block overwrites one element in
    /// expert-major scratch `[expert, 0, hidden_row]`. Gated by
    /// `FERRULE_CUDA_MOE_REDUCE=1` on the host side for A/B testing.
    #[kernel]
    pub fn moe_gemv_down_fp4_batched_reduce(
        y_hidden: &[f32],
        down_ptrs: &[u64],
        down_scale_ptrs: &[u64],
        mut expert_output: DisjointSlice<f32>,
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
        if down_ptr.is_null() || down_scale_ptr.is_null() {
            if tid == 0 {
                let out_off = expert * hidden_size as usize + row;
                unsafe {
                    *expert_output.as_mut_ptr().add(out_off) = 0.0;
                }
            }
            return;
        }

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
            let out_off = expert * hidden_size as usize + row;
            unsafe {
                *expert_output.as_mut_ptr().add(out_off) = SUM[0];
            }
        }
    }

    /// Deterministically reduce routed expert outputs into an existing output.
    ///
    /// `expert_output` is `[expert, batch_col, hidden]`, `route_slots` is
    /// `[batch_col, rank]`, and `output` is `[batch_col, hidden]`. One thread
    /// exclusively owns each output element and folds contributions in rank
    /// order, starting from the existing output value.
    #[kernel]
    pub fn moe_reduce_expert_outputs_ranked(
        expert_output: &[f32],
        route_slots: &[i32],
        mut output: DisjointSlice<f32>,
        output_offset: u32,
        hidden_size: u32,
        batch_cols: u32,
        routes_per_col: u32,
        num_experts: u32,
    ) {
        let index = thread::index_1d().get();
        let total = hidden_size as u64 * batch_cols as u64;
        if (index as u64) >= total
            || hidden_size == 0
            || batch_cols == 0
            || routes_per_col == 0
            || routes_per_col > num_experts
        {
            return;
        }

        let hidden = hidden_size as usize;
        let batch_cols = batch_cols as usize;
        let routes_per_col = routes_per_col as usize;
        let num_experts = num_experts as usize;
        let col = index / hidden;
        let row = index - col * hidden;
        let output_off = output_offset as usize + col * hidden + row;
        let output_ptr = output.as_mut_ptr();
        let mut acc = unsafe { *output_ptr.add(output_off) };
        let mut rank = 0usize;
        while rank < routes_per_col {
            let slot = route_slots[col * routes_per_col + rank];
            if slot >= 0 && (slot as usize) < num_experts {
                let expert_off = (slot as usize * batch_cols + col) * hidden + row;
                acc += expert_output[expert_off];
            }
            rank += 1;
        }
        unsafe {
            *output_ptr.add(output_off) = acc;
        }
    }

    /// Deterministically merge resident-hit and newly-materialized route outputs
    /// in the original router rank order. Each route appears in exactly one input.
    #[kernel]
    pub fn moe_reduce_split_expert_outputs_ranked(
        resident_output: &[f32],
        materialized_output: &[f32],
        resident_route_slots: &[i32],
        materialized_route_slots: &[i32],
        miss_markers: &[i32],
        mut output: DisjointSlice<f32>,
        output_offset: u32,
        hidden_size: u32,
        batch_cols: u32,
        routes_per_col: u32,
        num_experts: u32,
    ) {
        let index = thread::index_1d().get();
        let total = hidden_size as u64 * batch_cols as u64;
        if (index as u64) >= total
            || hidden_size == 0
            || batch_cols == 0
            || routes_per_col == 0
            || routes_per_col > num_experts
        {
            return;
        }

        let hidden = hidden_size as usize;
        let routes_per_col = routes_per_col as usize;
        let num_experts = num_experts as usize;
        let col = index / hidden;
        let row = index - col * hidden;
        let output_off = output_offset as usize + col * hidden + row;
        let output_ptr = output.as_mut_ptr();
        let mut acc = unsafe { *output_ptr.add(output_off) };
        let mut rank = 0usize;
        while rank < routes_per_col {
            let route = col * routes_per_col + rank;
            let is_miss = route < miss_markers.len() && miss_markers[route] != 0;
            let slots = if is_miss {
                materialized_route_slots
            } else {
                resident_route_slots
            };
            let values = if is_miss {
                materialized_output
            } else {
                resident_output
            };
            let slot = slots[route];
            if slot >= 0 && (slot as usize) < num_experts {
                let expert_off = (slot as usize * batch_cols as usize + col) * hidden + row;
                acc += values[expert_off];
            }
            rank += 1;
        }
        unsafe {
            *output_ptr.add(output_off) = acc;
        }
    }

    /// Expert-major segment gate+up FP4 GEMM using Blackwell mxf4 Tensor Cores.
    ///
    /// Grid: `(ceil(intermediate_size / 16), num_segments)`, blockDim.x = 32.
    /// Each segment owns eight columns and one resident expert slot. Input columns
    /// gather from the full-token packed input through `segment_token_indices`;
    /// `-1` padding columns are materialized as zero.
    #[kernel]
    pub unsafe fn moe_gemm_dual_fp4_mxf4_segmented(
        x_packed: &[u8],
        x_scales: &[u8],
        gate_ptrs: &[u64],
        gate_scale_ptrs: &[u64],
        up_ptrs: &[u64],
        up_scale_ptrs: &[u64],
        down_ptrs: &[u64],
        down_scale_ptrs: &[u64],
        slot_generations: &[i32],
        segment_expert_slots: &[i32],
        segment_generations: &[i32],
        segment_token_indices: &[i32],
        mut route_error: DisjointSlice<i32>,
        mut y_gate: DisjointSlice<f32>,
        mut y_up: DisjointSlice<f32>,
        n: u32,
        k: u32,
        num_tokens: u32,
        slot_capacity: u32,
        num_segments: u32,
    ) {
        static mut SMEM_A: SharedArray<u8, 512, 32> = SharedArray::UNINIT;
        static mut SMEM_B: SharedArray<u8, 256, 32> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let segment = thread::blockIdx_y() as usize;
        let row_base = thread::blockIdx_x() as usize * 16;
        let n = n as usize;
        let k = k as usize;
        let num_tokens = num_tokens as usize;
        let slot_capacity = slot_capacity as usize;
        let num_segments = num_segments as usize;
        if tid >= 32
            || segment >= num_segments
            || num_tokens == 0
            || slot_capacity == 0
            || k == 0
            || !k.is_multiple_of(64)
        {
            return;
        }
        let expert_slot = segment_expert_slots[segment];
        if expert_slot == -1 {
            return;
        }
        let expert = if expert_slot >= 0 {
            expert_slot as usize
        } else {
            slot_capacity
        };
        let valid_binding = expert < slot_capacity
            && expert < slot_generations.len()
            && expert < gate_ptrs.len()
            && expert < gate_scale_ptrs.len()
            && expert < up_ptrs.len()
            && expert < up_scale_ptrs.len()
            && expert < down_ptrs.len()
            && expert < down_scale_ptrs.len()
            && segment < segment_generations.len()
            && segment_generations[segment] > 0
            && slot_generations[expert] == segment_generations[segment]
            && gate_ptrs[expert] != 0
            && gate_scale_ptrs[expert] != 0
            && up_ptrs[expert] != 0
            && up_scale_ptrs[expert] != 0
            && down_ptrs[expert] != 0
            && down_scale_ptrs[expert] != 0;
        let mut valid_tokens = segment * 8 + 7 < segment_token_indices.len();
        if valid_tokens {
            let mut column = 0usize;
            while column < 8 {
                let token = segment_token_indices[segment * 8 + column];
                if token < -1 || (token != -1 && token as usize >= num_tokens) {
                    valid_tokens = false;
                }
                column += 1;
            }
        }
        if !valid_binding || !valid_tokens {
            atomic_fetch_or_i32(route_error.as_mut_ptr(), 1);
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
                    let token = segment_token_indices[segment * 8 + col];
                    if token >= 0 && (token as usize) < num_tokens {
                        let src = token as usize * packed_cols + kt / 2 + k4 * 2;
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
                let row = tid & 0x0f;
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
            let k_block = kt / 32;
            let scale_a_gate = if logical_row < n {
                unsafe {
                    (*gate_scale_ptr.add(logical_row * scale_cols + k_block + 1) as u32) << 8
                        | (*gate_scale_ptr.add(logical_row * scale_cols + k_block) as u32)
                }
            } else {
                (127u32 << 8) | 127u32
            };
            let token = segment_token_indices[segment * 8 + group];
            let scale_b = if token >= 0 && (token as usize) < num_tokens {
                let scale_base = token as usize * scale_cols + k_block;
                (x_scales[scale_base + 1] as u32) << 8 | (x_scales[scale_base] as u32)
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

        let group = tid / 4;
        let thread_in_group = tid % 4;
        let gate_ptr_out = y_gate.as_mut_ptr();
        let up_ptr_out = y_up.as_mut_ptr();
        let mut j = 0usize;
        while j < 4 {
            let row = row_base + group + if j >= 2 { 8 } else { 0 };
            let col = thread_in_group * 2 + (j & 1);
            if row < n {
                let out_off = (segment * 8 + col) * n + row;
                unsafe {
                    *gate_ptr_out.add(out_off) = acc_gate[j];
                    *up_ptr_out.add(out_off) = acc_up[j];
                }
            }
            j += 1;
        }
    }

    /// Expert-major segment down FP4 GEMM using Blackwell mxf4 Tensor Cores.
    ///
    /// Grid: `(ceil(hidden_size / 16), num_segments)`, blockDim.x = 32. Hidden
    /// activations are `[segment, 8, intermediate]`. Each real column scatters
    /// directly to `route_output[route_index, hidden]`; `-1` padding is skipped.
    #[kernel]
    pub unsafe fn moe_gemm_down_fp4_mxf4_segmented(
        y_hidden_packed: &[u8],
        y_hidden_scales: &[u8],
        gate_ptrs: &[u64],
        gate_scale_ptrs: &[u64],
        up_ptrs: &[u64],
        up_scale_ptrs: &[u64],
        down_ptrs: &[u64],
        down_scale_ptrs: &[u64],
        slot_generations: &[i32],
        segment_expert_slots: &[i32],
        segment_generations: &[i32],
        segment_route_indices: &[i32],
        mut route_written: DisjointSlice<i32>,
        mut route_error: DisjointSlice<i32>,
        mut route_output: DisjointSlice<f32>,
        intermediate_size: u32,
        hidden_size: u32,
        slot_capacity: u32,
        num_segments: u32,
        num_routes: u32,
    ) {
        static mut SMEM_A: SharedArray<u8, 512, 32> = SharedArray::UNINIT;
        static mut SMEM_B: SharedArray<u8, 256, 32> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let segment = thread::blockIdx_y() as usize;
        let row_base = thread::blockIdx_x() as usize * 16;
        let inter = intermediate_size as usize;
        let hidden = hidden_size as usize;
        let slot_capacity = slot_capacity as usize;
        let num_segments = num_segments as usize;
        let num_routes = num_routes as usize;
        if tid >= 32
            || segment >= num_segments
            || slot_capacity == 0
            || num_routes == 0
            || inter == 0
            || !inter.is_multiple_of(64)
        {
            return;
        }
        let expert_slot = segment_expert_slots[segment];
        if expert_slot == -1 {
            return;
        }
        let expert = if expert_slot >= 0 {
            expert_slot as usize
        } else {
            slot_capacity
        };
        let valid_binding = expert < slot_capacity
            && expert < slot_generations.len()
            && expert < gate_ptrs.len()
            && expert < gate_scale_ptrs.len()
            && expert < up_ptrs.len()
            && expert < up_scale_ptrs.len()
            && expert < down_ptrs.len()
            && expert < down_scale_ptrs.len()
            && segment < segment_generations.len()
            && segment_generations[segment] > 0
            && slot_generations[expert] == segment_generations[segment]
            && gate_ptrs[expert] != 0
            && gate_scale_ptrs[expert] != 0
            && up_ptrs[expert] != 0
            && up_scale_ptrs[expert] != 0
            && down_ptrs[expert] != 0
            && down_scale_ptrs[expert] != 0;
        let mut valid_routes = segment * 8 + 7 < segment_route_indices.len();
        if valid_routes {
            let mut column = 0usize;
            while column < 8 {
                let route = segment_route_indices[segment * 8 + column];
                if route < -1 || (route != -1 && route as usize >= num_routes) {
                    valid_routes = false;
                }
                column += 1;
            }
        }
        if !valid_binding || !valid_routes {
            atomic_fetch_or_i32(route_error.as_mut_ptr(), 1);
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
                    let src = (segment * 8 + col) * packed_cols + kt / 2 + k4 * 2;
                    *b_dst.add(dst) = y_hidden_packed[src];
                    *b_dst.add(dst + 1) = y_hidden_packed[src + 1];
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
                let row = tid & 0x0f;
                let addr = (&raw const SMEM_B as *const u8).add(row * 16) as *const u32;
                cuda_device::wmma::ldmatrix_x2_trans(addr)
            };
            let group = tid / 4;
            let scale_row = group + if (tid & 1) != 0 { 8 } else { 0 };
            let logical_row = row_base + scale_row;
            let k_block = kt / 32;
            let scale_a = if logical_row < hidden {
                unsafe {
                    (*down_scale_ptr.add(logical_row * scale_cols + k_block + 1) as u32) << 8
                        | (*down_scale_ptr.add(logical_row * scale_cols + k_block) as u32)
                }
            } else {
                (127u32 << 8) | 127u32
            };
            let scale_base = (segment * 8 + group) * scale_cols + k_block;
            let scale_b = (y_hidden_scales[scale_base + 1] as u32) << 8
                | (y_hidden_scales[scale_base] as u32);
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

        let group = tid / 4;
        let thread_in_group = tid % 4;
        let output_ptr = route_output.as_mut_ptr();
        let mut j = 0usize;
        while j < 4 {
            let row = row_base + group + if j >= 2 { 8 } else { 0 };
            let col = thread_in_group * 2 + (j & 1);
            let route = segment_route_indices[segment * 8 + col];
            if row < hidden && route >= 0 {
                let output_off = route as usize * hidden + row;
                unsafe {
                    *output_ptr.add(output_off) = acc[j];
                }
            }
            j += 1;
        }
        if row_base == 0 && tid < 8 {
            let route = segment_route_indices[segment * 8 + tid];
            if route >= 0 {
                atomic_fetch_or_i32(unsafe { route_written.as_mut_ptr().add(route as usize) }, 1);
            }
        }
    }

    /// Deterministically reduce route-major outputs into an existing token output.
    /// One thread owns one `(token, hidden-row)` and performs `rank=0..K` in order.
    #[kernel]
    pub fn moe_reduce_route_outputs_ranked(
        route_output: &[f32],
        mut output: DisjointSlice<f32>,
        tokens: u32,
        routes_per_token: u32,
        hidden_size: u32,
    ) {
        let index = thread::index_1d().get();
        let total = tokens as u64 * hidden_size as u64;
        if (index as u64) >= total || tokens == 0 || routes_per_token == 0 || hidden_size == 0 {
            return;
        }

        let hidden = hidden_size as usize;
        let routes_per_token = routes_per_token as usize;
        let token = index / hidden;
        let row = index - token * hidden;
        let output_ptr = output.as_mut_ptr();
        let mut acc = unsafe { *output_ptr.add(index) };
        let mut rank = 0usize;
        while rank < routes_per_token {
            let route = token * routes_per_token + rank;
            acc += route_output[route * hidden + row];
            rank += 1;
        }
        unsafe {
            *output_ptr.add(index) = acc;
        }
    }

    /// Ranked reducer for cumulative segmented execution. Completion is checked
    /// before any route output is read. Failure writes one deterministic quiet NaN.
    #[kernel]
    pub fn moe_reduce_segment_route_outputs_ranked(
        route_output: &[f32],
        route_written: &[i32],
        route_error: &[i32],
        mut output: DisjointSlice<f32>,
        tokens: u32,
        routes_per_token: u32,
        hidden_size: u32,
    ) {
        let index = thread::index_1d().get();
        let total = tokens as u64 * hidden_size as u64;
        if (index as u64) >= total || tokens == 0 || routes_per_token == 0 || hidden_size == 0 {
            return;
        }

        let hidden = hidden_size as usize;
        let routes_per_token = routes_per_token as usize;
        let token = index / hidden;
        let row = index - token * hidden;
        let mut complete = route_error.first().copied().unwrap_or(1) == 0;
        let mut rank = 0usize;
        while rank < routes_per_token {
            let route = token * routes_per_token + rank;
            if route_written.get(route).copied().unwrap_or(0) == 0 {
                complete = false;
            }
            rank += 1;
        }

        let output_ptr = output.as_mut_ptr();
        if !complete {
            unsafe {
                *output_ptr.add(index) = f32::from_bits(0x7fc0_0000);
            }
            return;
        }

        let mut acc = unsafe { *output_ptr.add(index) };
        rank = 0;
        while rank < routes_per_token {
            let route = token * routes_per_token + rank;
            acc += route_output[route * hidden + row];
            rank += 1;
        }
        unsafe {
            *output_ptr.add(index) = acc;
        }
    }

    // ── Generic Hyper-Connection helpers ───────────────────────────────

    #[kernel]
    pub fn hc_pre_f32(
        state: &[f32],
        function_col_major: &[f32],
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
        // Decode uses hc=4/mix=24. The full CTA cooperatively stages 128
        // transposed columns so global loads use all eight warps, while each
        // row thread still consumes columns in the original arithmetic order.
        static mut FUNCTION_TILE: SharedArray<f32, 3072> = SharedArray::UNINIT;
        static mut STATE_TILE: SharedArray<f32, 128> = SharedArray::UNINIT;
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

        if mix <= 24 {
            let mut dot = 0.0f32;
            let mut tile_base = 0usize;
            while tile_base < hc_dim {
                let tile_cols = (hc_dim - tile_base).min(128);
                let tile_elements = tile_cols * mix;
                let mut index = tid;
                while index < tile_elements {
                    let col = index / mix;
                    let row = index - col * mix;
                    unsafe {
                        FUNCTION_TILE[index] = function_col_major[(tile_base + col) * mix + row];
                    }
                    index += bdim;
                }
                let mut col = tid;
                while col < tile_cols {
                    unsafe {
                        STATE_TILE[col] = state[state_base + tile_base + col];
                    }
                    col += bdim;
                }
                thread::sync_threads();

                if tid < mix {
                    let row = tid;
                    for col in 0..tile_cols {
                        dot += unsafe { FUNCTION_TILE[col * mix + row] * STATE_TILE[col] };
                    }
                }
                thread::sync_threads();
                tile_base += tile_cols;
            }
            if tid < mix {
                unsafe {
                    MIX[tid] = dot * rms;
                }
            }
        } else if tid < mix {
            let mut dot = 0.0f32;
            let row = tid;
            for col in 0..hc_dim {
                dot += function_col_major[col * mix + row] * state[state_base + col];
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

#[cfg(test)]
mod tests {
    use super::paged_plane_row_offset;

    #[test]
    fn paged_plane_row_offset_addresses_flattened_sequences() {
        let block_slots = [4, 1, 3];
        let block_offsets = [0, 2, 3];

        assert_eq!(
            paged_plane_row_offset(60, &block_slots, &block_offsets, 0, 0, 2, 3, 1, 2),
            54
        );
        assert_eq!(
            paged_plane_row_offset(60, &block_slots, &block_offsets, 0, 3, 2, 3, 1, 2),
            21
        );
        assert_eq!(
            paged_plane_row_offset(60, &block_slots, &block_offsets, 1, 1, 2, 3, 1, 2),
            45
        );
        assert_eq!(
            paged_plane_row_offset(60, &block_slots, &block_offsets, 1, 2, 2, 3, 1, 2),
            usize::MAX
        );
    }
}
