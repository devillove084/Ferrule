//! Quantization framework — llama.cpp style block-wise quantization.
#![allow(clippy::needless_range_loop)]

pub mod gguf_quant;

use ferrule_common::QuantType;

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

pub fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = (bits >> 16) as u16 & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;
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
            _ => panic!("unsupported quant type for QMatrix: {qtype:?}"),
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

    /// Dequantize a single row back to f32 (used for validation/testing).
    pub fn dequantize_row(&self, row: usize) -> Vec<f32> {
        let bs = self.quant.block_size();
        let bpr = self.blocks_per_row();
        let mut result = vec![0.0f32; self.in_f];
        match self.quant {
            QuantType::Q8_0 => {
                for b in 0..bpr {
                    let scale = f16_to_f32(self.scales[row * bpr + b]);
                    for j in b * bs..((b + 1) * bs).min(self.in_f) {
                        let q = self.packed[row * self.in_f + j] as i8;
                        result[j] = q as f32 * scale;
                    }
                }
            }
            QuantType::Q4_0 => {
                let bytes_per_row = bpr * 16;
                for b in 0..bpr {
                    let d = f16_to_f32(self.scales[row * bpr + b]);
                    let row_pack_off = row * bytes_per_row;
                    for j in b * bs..((b + 1) * bs).min(self.in_f) {
                        let local = j - b * bs;
                        let byte = self.packed
                            [row_pack_off + b * 16 + if local < 16 { local } else { local - 16 }];
                        let q = if local < 16 { byte & 0x0F } else { byte >> 4 };
                        result[j] = d * (q as f32 - 8.0);
                    }
                }
            }
            _ => {
                // For other quant types, return approximate values using scale only
                for b in 0..bpr {
                    let scale = f16_to_f32(self.scales[row * bpr + b]);
                    for j in b * bs..((b + 1) * bs).min(self.in_f) {
                        result[j] = scale; // crude approximation
                    }
                }
            }
        }
        result
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

    #[test]
    fn q8_0_roundtrip_error_within_tolerance() {
        // Quantize a simple vector and check reconstruction error
        let src: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let q = QMatrix::quantize(&src, 1, 64, QuantType::Q8_0);
        // Q8_0 should have very low error
        let reconstructed = q.dequantize_row(0);
        let max_err = src
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 0.1, "Q8_0 max error {max_err} exceeds tolerance");
    }
}
