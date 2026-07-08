//! GGUF / ggml quantized block decoders.
//!
//! This module contains small CPU decoders used for format validation and
//! correctness fixtures. Production inference should use vectorized/CUDA kernels,
//! but the byte-level semantics live here so model bring-up can fail precisely for
//! unsupported quant classes.

use super::f16_to_f32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufQuantType {
    Q4K,
}

impl GgufQuantType {
    pub fn block_values(self) -> usize {
        match self {
            Self::Q4K => QK_K,
        }
    }

    pub fn block_bytes(self) -> usize {
        match self {
            Self::Q4K => Q4_K_BLOCK_BYTES,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GgufQuantError {
    InvalidBlockSize {
        quant: GgufQuantType,
        expected: usize,
        actual: usize,
    },
    InvalidRowSize {
        quant: GgufQuantType,
        row_values: usize,
    },
}

pub type GgufQuantResult<T> = std::result::Result<T, GgufQuantError>;

pub const QK_K: usize = 256;
pub const Q4_K_SCALE_BYTES: usize = 12;
pub const Q4_K_QUANT_BYTES: usize = QK_K / 2;
pub const Q4_K_BLOCK_BYTES: usize = 2 + 2 + Q4_K_SCALE_BYTES + Q4_K_QUANT_BYTES;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Q4KBlock<'a> {
    bytes: &'a [u8],
}

impl<'a> Q4KBlock<'a> {
    pub fn from_bytes(bytes: &'a [u8]) -> GgufQuantResult<Self> {
        if bytes.len() != Q4_K_BLOCK_BYTES {
            return Err(GgufQuantError::InvalidBlockSize {
                quant: GgufQuantType::Q4K,
                expected: Q4_K_BLOCK_BYTES,
                actual: bytes.len(),
            });
        }
        Ok(Self { bytes })
    }

    pub fn dequantize(&self) -> [f32; QK_K] {
        let d = f16_to_f32(u16::from_le_bytes([self.bytes[0], self.bytes[1]]));
        let min = f16_to_f32(u16::from_le_bytes([self.bytes[2], self.bytes[3]]));
        let scales = &self.bytes[4..4 + Q4_K_SCALE_BYTES];
        let qs = &self.bytes[4 + Q4_K_SCALE_BYTES..];

        let mut out = [0.0f32; QK_K];
        let mut scale_index = 0usize;
        let mut q_offset = 0usize;
        for out_offset in (0..QK_K).step_by(64) {
            let (scale_1, min_1) = scale_min_k4(scale_index, scales);
            let d1 = d * scale_1 as f32;
            let m1 = min * min_1 as f32;
            let (scale_2, min_2) = scale_min_k4(scale_index + 1, scales);
            let d2 = d * scale_2 as f32;
            let m2 = min * min_2 as f32;
            for lane in 0..32 {
                let packed = qs[q_offset + lane];
                out[out_offset + lane] = d1 * (packed & 0x0f) as f32 - m1;
                out[out_offset + 32 + lane] = d2 * (packed >> 4) as f32 - m2;
            }
            q_offset += 32;
            scale_index += 2;
        }
        out
    }
}

pub fn dequantize_q4_k_row(bytes: &[u8], row_values: usize) -> GgufQuantResult<Vec<f32>> {
    if row_values == 0 || !row_values.is_multiple_of(QK_K) {
        return Err(GgufQuantError::InvalidRowSize {
            quant: GgufQuantType::Q4K,
            row_values,
        });
    }
    let blocks = row_values / QK_K;
    let expected = blocks * Q4_K_BLOCK_BYTES;
    if bytes.len() != expected {
        return Err(GgufQuantError::InvalidBlockSize {
            quant: GgufQuantType::Q4K,
            expected,
            actual: bytes.len(),
        });
    }
    let mut out = Vec::with_capacity(row_values);
    for chunk in bytes.chunks_exact(Q4_K_BLOCK_BYTES) {
        out.extend(Q4KBlock::from_bytes(chunk)?.dequantize());
    }
    Ok(out)
}

fn scale_min_k4(index: usize, scales: &[u8]) -> (u8, u8) {
    debug_assert_eq!(scales.len(), Q4_K_SCALE_BYTES);
    if index < 4 {
        (scales[index] & 63, scales[index + 4] & 63)
    } else {
        (
            (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4),
            (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4_k_block_decodes_scales_mins_and_nibbles_like_ggml() {
        let mut block = vec![0u8; Q4_K_BLOCK_BYTES];
        block[0..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1.0
        block[2..4].copy_from_slice(&0x0000u16.to_le_bytes()); // dmin = 0.0
        let scales_offset = 4;
        block[scales_offset] = 1; // sub-block 0 scale
        block[scales_offset + 1] = 2; // sub-block 1 scale
        block[scales_offset + 8] = 4; // sub-block 4 low-scale packing
        let qs_offset = 4 + Q4_K_SCALE_BYTES;
        block[qs_offset] = 0x32; // y[0] = 2, y[32] = 3 * 2
        block[qs_offset + 64] = 0x05; // y[128] = 5 * 4

        let decoded = Q4KBlock::from_bytes(&block).unwrap().dequantize();
        assert_eq!(decoded[0], 2.0);
        assert_eq!(decoded[32], 6.0);
        assert_eq!(decoded[128], 20.0);
    }

    #[test]
    fn q4_k_row_decoder_validates_sizes() {
        let err = dequantize_q4_k_row(&[], 128).unwrap_err();
        assert!(matches!(err, GgufQuantError::InvalidRowSize { .. }));
        let err = dequantize_q4_k_row(&[0; 8], QK_K).unwrap_err();
        assert!(matches!(
            err,
            GgufQuantError::InvalidBlockSize {
                expected: Q4_K_BLOCK_BYTES,
                actual: 8,
                ..
            }
        ));
    }
}
