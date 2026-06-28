//! Quantization framework — llama.cpp style block-wise quantization.
//!
//! Supported types:
//!   Q4_0 — 4-bit symmetric,  block 32,  4.5 bpw
//!   Q2S — 2-bit symmetric,  block 64,  2.25 bpw
//!   T1S — ternary 1.58b,    block 64,  ~2.1 bpw, add/sub replaces mul
//!
//! Usage:
//!   let m = QMatrix::quantize(&w, out_f, in_f, QuantType::Q4_0);
//!   m.scales_f32()          → Vec<f32> for GPU upload
//!   m.packed()              → &[u8] for GPU upload
//!   m.blocks_per_row()      → used by kernel for indexing

// ── f16 conversion (no external dependency) ───────────────────────────

pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as i32;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mant = (bits & 0x3FF) as i32;
    if exp == 0 {
        (if sign != 0 { -1.0 } else { 1.0 }) * (mant as f32) * 2.0f32.powi(-24)
    } else if exp == 31 {
        if mant == 0 {
            if sign != 0 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        }
    } else {
        (if sign != 0 { -1.0 } else { 1.0 }) * (1.0 + mant as f32 / 1024.0) * 2.0f32.powi(exp - 15)
    }
}

fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = (bits >> 16) as u16 & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = (bits & 0x7F_FFFF) as u32;
    if exp == 0 {
        sign
    } else if exp == 0xFF {
        if mant == 0 {
            sign | 0x7C00
        } else {
            sign | 0x7E00
        }
    } else {
        let e = exp - 127 + 15;
        if e <= 0 {
            sign
        } else if e >= 31 {
            sign | 0x7C00
        } else {
            sign | ((e as u16) << 10) | ((mant >> 13) as u16)
        }
    }
}

// ── Quantization type ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantType {
    /// 4-bit symmetric, block 32, llama.cpp Q4_0 half-block layout
    Q4_0,
    /// 8-bit symmetric, block 32
    Q8_0,
    /// 2-bit symmetric, block 64
    Q2S,
    /// Ternary {-1,0,1}, block 64, add/sub in kernel
    T1S,
}

impl QuantType {
    pub fn block_size(self) -> usize {
        match self {
            Self::Q4_0 | Self::Q8_0 => 32,
            Self::Q2S | Self::T1S => 64,
        }
    }

    pub fn bits(self) -> u32 {
        match self {
            Self::Q4_0 => 4,
            Self::Q8_0 => 8,
            Self::Q2S | Self::T1S => 2,
        }
    }
}

// ── Quantized matrix ──────────────────────────────────────────────────

/// A quantized weight matrix stored as (packed_u8, scales_f16).
#[derive(Clone)]
pub struct QMatrix {
    pub quant: QuantType,
    /// Packed quantized weights (u8 array).
    pub packed: Vec<u8>,
    /// Block scales stored as f16 (u16).
    pub scales: Vec<u16>,
    pub out_f: usize,
    pub in_f: usize,
}

impl QMatrix {
    /// Quantize a f32 weight matrix [out_f × in_f] row-major.
    pub fn quantize(w: &[f32], out_f: usize, in_f: usize, qtype: QuantType) -> Self {
        match qtype {
            QuantType::Q4_0 => Self::quantize_q4_0(w, out_f, in_f),
            QuantType::Q8_0 => Self::quantize_q8_0(w, out_f, in_f),
            QuantType::Q2S => Self::quantize_q2_s(w, out_f, in_f),
            QuantType::T1S => Self::quantize_t1_s(w, out_f, in_f),
        }
    }

    pub fn blocks_per_row(&self) -> usize {
        self.in_f.div_ceil(self.quant.block_size())
    }

    pub fn total_blocks(&self) -> usize {
        self.out_f * self.blocks_per_row()
    }

    /// Convert f16 scales to f32 for GPU upload.
    pub fn scales_f32(&self) -> Vec<f32> {
        self.scales.iter().map(|&b| f16_to_f32(b)).collect()
    }

    // ── Q4_0 (llama.cpp compatible) ──────────────────────────────────

    fn quantize_q4_0(w: &[f32], out_f: usize, in_f: usize) -> Self {
        let bs = QuantType::Q4_0.block_size();
        let bpr = in_f.div_ceil(bs);
        let mut scales = Vec::with_capacity(out_f * bpr);
        // llama.cpp Q4_0 stores each 32-value block as 16 bytes:
        // byte j: low nibble = value j, high nibble = value j + 16.
        let bytes_per_row = bpr * 16;
        let mut packed = vec![0u8; out_f * bytes_per_row];

        for row in 0..out_f {
            let row_off = row * in_f;
            let row_pack_off = row * bytes_per_row;
            for b in 0..bpr {
                let start = b * bs;
                let end = (start + bs).min(in_f);
                let block_pack_off = row_pack_off + b * 16;

                // Find signed max (value at position of max absolute), matching llama.cpp.
                let mut amax = 0.0f32;
                let mut max_val = 0.0f32;
                for j in start..end {
                    let v = w[row_off + j];
                    let abs = v.abs();
                    if abs > amax {
                        amax = abs;
                        max_val = v;
                    }
                }
                // llama.cpp: d = max / -8, id = 1/d.
                let d = max_val / -8.0f32;
                let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                scales.push(f32_to_f16(d));

                for j in start..end {
                    let val = w[row_off + j];
                    // llama.cpp: (int8_t)(x * id + 8.5), clamped to [0,15].
                    let q = if d != 0.0 {
                        let v = (val * id + 8.5) as i32;
                        v.clamp(0, 15) as u8
                    } else {
                        8
                    };
                    let local = j - start;
                    let byte_idx = block_pack_off + if local < 16 { local } else { local - 16 };
                    if local < 16 {
                        packed[byte_idx] = (packed[byte_idx] & 0xF0) | (q & 0x0F);
                    } else {
                        packed[byte_idx] = (packed[byte_idx] & 0x0F) | ((q & 0x0F) << 4);
                    }
                }
            }
        }
        Self {
            quant: QuantType::Q4_0,
            packed,
            scales,
            out_f,
            in_f,
        }
    }

    // ── Q2S ────────────────────────────────────────────────────────

    fn quantize_q2_s(w: &[f32], out_f: usize, in_f: usize) -> Self {
        let bs = QuantType::Q2S.block_size();
        let bpr = in_f.div_ceil(bs);
        let mut scales = Vec::with_capacity(out_f * bpr);
        // 4 values per u8 → ceil(in_f/4) u8 per row
        let packed_per_row = (in_f + 3) / 4;
        let mut packed = vec![0u8; out_f * packed_per_row];

        for row in 0..out_f {
            let row_off = row * in_f;
            for b in 0..bpr {
                let start = b * bs;
                let end = (start + bs).min(in_f);

                let mut max_abs = 0.0f32;
                for j in start..end {
                    let abs = w[row_off + j].abs();
                    if abs > max_abs {
                        max_abs = abs;
                    }
                }
                let delta = max_abs / 1.0f32; // symmetric 2-bit: values are -1, -1/3, 1/3, 1
                scales.push(f32_to_f16(delta));

                for j in start..end {
                    let val = w[row_off + j];
                    let q_f = if delta > 0.0 {
                        // Map to [0,3]: 0→-1, 1→-1/3, 2→1/3, 3→1
                        (val / delta * 1.5 + 1.5).round().clamp(0.0, 3.0)
                    } else {
                        1.5 // center
                    };
                    let q = q_f as u8;
                    let local = j - start;
                    let byte_idx = row * packed_per_row + (start + local) / 4;
                    let shift = 2 * (local % 4);
                    packed[byte_idx] |= q << shift;
                }
            }
        }
        Self {
            quant: QuantType::Q2S,
            packed,
            scales,
            out_f,
            in_f,
        }
    }

    // ── T1S ────────────────────────────────────────────────────────

    fn quantize_t1_s(w: &[f32], out_f: usize, in_f: usize) -> Self {
        let bs = 64;
        let bpr = in_f.div_ceil(bs);
        let mut scales = Vec::with_capacity(out_f * bpr);
        let packed_per_row = (in_f + 3) / 4;
        let mut packed = vec![0u8; out_f * packed_per_row];

        for row in 0..out_f {
            let row_off = row * in_f;
            for b in 0..bpr {
                let start = b * bs;
                let end = (start + bs).min(in_f);

                let mut max_abs = 0.0f32;
                for j in start..end {
                    let abs = w[row_off + j].abs();
                    if abs > max_abs {
                        max_abs = abs;
                    }
                }
                let delta = max_abs; // scale = max |w|, ternary values are -1,0,1
                scales.push(f32_to_f16(delta));

                for j in start..end {
                    let val = w[row_off + j];
                    // Quantize to {-1, 0, 1}: threshold at ±0.5 * delta
                    let q = if delta > 0.0 {
                        let norm = val / delta;
                        if norm < -0.5 {
                            0u8
                        }
                        // -1
                        else if norm > 0.5 {
                            2u8
                        }
                        // +1
                        else {
                            1u8
                        } // 0
                    } else {
                        1u8 // 0
                    };
                    let local = j - start;
                    let byte_idx = row * packed_per_row + (start + local) / 4;
                    let shift = 2 * (local % 4);
                    packed[byte_idx] |= q << shift;
                }
            }
        }
        Self {
            quant: QuantType::T1S,
            packed,
            scales,
            out_f,
            in_f,
        }
    }

    // ── Q8_0 ────────────────────────────────────────────────────────

    fn quantize_q8_0(w: &[f32], out_f: usize, in_f: usize) -> Self {
        let bs = QuantType::Q8_0.block_size(); // 32
        let bpr = in_f.div_ceil(bs);
        let mut scales = Vec::with_capacity(out_f * bpr);
        // 1 byte per value → out_f * in_f bytes
        let mut packed = vec![0u8; out_f * in_f];

        for row in 0..out_f {
            let row_off = row * in_f;
            for b in 0..bpr {
                let start = b * bs;
                let end = (start + bs).min(in_f);

                let mut max_abs = 0.0f32;
                for j in start..end {
                    let abs = w[row_off + j].abs();
                    if abs > max_abs {
                        max_abs = abs;
                    }
                }
                // Q8_0: symmetric, values mapped to [-127, 127]
                let delta = max_abs / 127.0f32;
                scales.push(f32_to_f16(delta));

                for j in start..end {
                    let val = w[row_off + j];
                    let q_f = if delta > 0.0 {
                        (val / delta).round().clamp(-127.0, 127.0)
                    } else {
                        0.0
                    };
                    let q = q_f as i8;
                    packed[row * in_f + j] = q as u8;
                }
            }
        }
        Self {
            quant: QuantType::Q8_0,
            packed,
            scales,
            out_f,
            in_f,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn llama_q4_0_quant(v: f32, d: f32) -> u8 {
        if d == 0.0 {
            8
        } else {
            ((v * (1.0 / d) + 8.5) as i32).clamp(0, 15) as u8
        }
    }

    #[test]
    fn q4_0_uses_llama_half_block_layout() {
        let w: Vec<f32> = (-16..16).map(|v| v as f32).collect();
        let q = QMatrix::quantize(&w, 1, 32, QuantType::Q4_0);

        assert_eq!(q.packed.len(), 16);
        assert_eq!(q.scales.len(), 1);
        let d = f16_to_f32(q.scales[0]);
        assert_eq!(d, 2.0);

        for j in 0..16 {
            let lo = q.packed[j] & 0x0F;
            let hi = q.packed[j] >> 4;
            assert_eq!(lo, llama_q4_0_quant(w[j], d), "low nibble {j}");
            assert_eq!(hi, llama_q4_0_quant(w[j + 16], d), "high nibble {j}");
        }
    }

    #[test]
    fn q4_0_pads_rows_by_block_storage() {
        let w = vec![0.0f32; 2 * 33];
        let q = QMatrix::quantize(&w, 2, 33, QuantType::Q4_0);

        assert_eq!(q.blocks_per_row(), 2);
        assert_eq!(q.packed.len(), 2 * 2 * 16);
        assert_eq!(q.scales.len(), 2 * 2);
    }
}
