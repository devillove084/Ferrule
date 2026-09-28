//! CPU-only, source-independent validation of packed numeric E4M3FN storage.
//! This is not E8M0, a source freshness guard, or a device-owner capability.

use std::sync::Arc;

use crate::{Error, Result};

fn invalid(message: impl Into<String>) -> Error {
    Error::Internal {
        message: format!("numeric FP8: {}", message.into()),
    }
}
fn checked(value: Option<usize>) -> Result<usize> {
    value
        .filter(|&v| v <= isize::MAX as usize)
        .ok_or_else(|| invalid("shape/size overflow"))
}

/// Numeric little-endian scale encoding. W = E4M3FN * positive finite scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericFp8ScaleType {
    Bf16,
    F32,
}
impl NumericFp8ScaleType {
    pub const fn element_bytes(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::F32 => 4,
        }
    }
    fn value(self, bytes: &[u8]) -> f32 {
        match self {
            Self::Bf16 => f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16),
            Self::F32 => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        }
    }
}

/// Packed local W[N,K] and its minimal intersecting row-major 128x128 scale grid.
/// The scale grid starts at [row_origin/128, column_origin/128]. Origins need
/// not be block aligned. Bytes from a different grid must not be relabelled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NumericFp8Layout {
    pub n: usize,
    pub k: usize,
    pub row_origin: usize,
    pub column_origin: usize,
    pub scale_type: NumericFp8ScaleType,
}
impl NumericFp8Layout {
    pub fn scale_shape(self) -> Result<[usize; 2]> {
        if self.n == 0
            || self.n > i32::MAX as usize - 128
            || self.k == 0
            || self.k > i32::MAX as usize - 32
        {
            return Err(invalid("invalid numeric FP8 shape"));
        }
        checked(self.row_origin.checked_add(self.n))?;
        checked(self.column_origin.checked_add(self.k))?;
        Ok([
            (self.row_origin % 128 + self.n).div_ceil(128),
            (self.column_origin % 128 + self.k).div_ceil(128),
        ])
    }
    pub fn storage_lengths(self) -> Result<(usize, usize)> {
        let [r, c] = self.scale_shape()?;
        let weight = checked(self.n.checked_mul(self.k))?;
        let scales = checked(
            r.checked_mul(c)
                .and_then(|v| v.checked_mul(self.scale_type.element_bytes())),
        )?;
        checked(weight.checked_add(scales))?;
        Ok((weight, scales))
    }

    /// Constant-time structural check, also used by proof-consuming uploads.
    /// Does not establish validity of raw byte contents; use `validate_payload`
    /// or `ImmutableValidatedNumericFp8Payload::new` for that.
    pub fn validate_lengths(self, weight: usize, scales: usize) -> Result<()> {
        if self.storage_lengths()? != (weight, scales) {
            return Err(invalid("weight/scale byte lengths mismatch"));
        }
        Ok(())
    }

    /// Full validation for raw-byte callers. Checks every weight and scale,
    /// including finite F32 dequantization of each selected block.
    pub fn validate_payload(self, weight: &[u8], scales: &[u8]) -> Result<()> {
        self.validate_lengths(weight.len(), scales.len())?;
        if contains_nan(weight) {
            return Err(invalid("NaN E4M3FN weight (0x7f/0xff)"));
        }
        let scale_columns = self.scale_shape()?[1];
        for (index, bytes) in scales
            .chunks_exact(self.scale_type.element_bytes())
            .enumerate()
        {
            let value = self.scale_type.value(bytes);
            if !value.is_finite() || value <= 0.0 {
                return Err(invalid(format!(
                    "numeric scale {index} must be finite and strictly positive"
                )));
            }
            // Most scales cannot overflow even at the format maximum. Only
            // exceptional scales need a second pass over their actual block.
            // Using 448 unconditionally would reject legal tiny/zero weights.
            if !(448.0 * value).is_finite() {
                let block_row = index / scale_columns;
                let block_column = index % scale_columns;
                let row_offset = self.row_origin % 128;
                let column_offset = self.column_origin % 128;
                let rows = (block_row * 128).saturating_sub(row_offset)
                    ..((block_row + 1) * 128 - row_offset).min(self.n);
                let columns = (block_column * 128).saturating_sub(column_offset)
                    ..((block_column + 1) * 128 - column_offset).min(self.k);
                let mut maximum = 0;
                for row in rows {
                    let start = row * self.k;
                    maximum = maximum.max(
                        weight[start + columns.start..start + columns.end]
                            .iter()
                            .map(|b| b & 0x7f)
                            .max()
                            .unwrap_or(0),
                    );
                }
                if !(decode_abs(maximum) * value).is_finite() {
                    return Err(invalid(format!(
                        "dequantized F32 weight overflow in scale block {index}"
                    )));
                }
            }
        }
        Ok(())
    }
}

fn contains_nan(weight: &[u8]) -> bool {
    const LOW: u64 = 0x7f7f_7f7f_7f7f_7f7f;
    const ONES: u64 = 0x0101_0101_0101_0101;
    const HIGH: u64 = 0x8080_8080_8080_8080;
    let (chunks, remainder) = weight.as_chunks::<8>();
    // Each masked lane is <=127, so +1 cannot carry into its neighbour.
    // Its high bit is set iff the byte is 0x7f or 0xff. The OR reduction
    // permits auto-vectorization and needs neither alignment nor unsafe loads.
    let flags = chunks.iter().fold(0, |flags, bytes| {
        flags | ((u64::from_ne_bytes(*bytes) & LOW) + ONES)
    });
    flags & HIGH != 0 || remainder.iter().any(|b| b & 0x7f == 0x7f)
}

fn decode_abs(code: u8) -> f32 {
    let exponent = (code >> 3) & 15;
    let mantissa = code & 7;
    if exponent == 0 {
        f32::from(mantissa) / 512.0
    } else {
        f32::from_bits((u32::from(exponent) + 120) << 23) * (1.0 + f32::from(mantissa) / 8.0)
    }
}

/// An immutable proof tied to the exact packed bytes, layout and scale type.
/// Construction validates once, without CUDA or checkpoint/source dependencies.
/// Cloning shares storage; no mutable access or unchecked constructor exists.
/// A retained `Arc` alias cannot mutate this storage through safe Rust: making
/// it mutable copies it. Source freshness and GPU ownership remain separate.
///
/// External callers cannot forge or modify the validated fields:
/// ```compile_fail
/// use ferrule_common::numeric_fp8::ImmutableValidatedNumericFp8Payload;
/// fn corrupt(proof: &mut ImmutableValidatedNumericFp8Payload) {
///     proof.layout.row_origin = 127;
/// }
/// ```
/// Nor can they change its bytes through the public accessors:
/// ```compile_fail
/// use ferrule_common::numeric_fp8::ImmutableValidatedNumericFp8Payload;
/// fn corrupt(proof: &mut ImmutableValidatedNumericFp8Payload) {
///     proof.weight_bytes()[0] = 0xff;
/// }
/// ```
#[derive(Debug, Clone)]
pub struct ImmutableValidatedNumericFp8Payload {
    layout: NumericFp8Layout,
    weight: Arc<[u8]>,
    scales: Arc<[u8]>,
}
impl ImmutableValidatedNumericFp8Payload {
    /// Pass existing `Arc<[u8]>` values to avoid copying payload bytes. Vec and
    /// slice inputs are also accepted via `Into<Arc<[u8]>>`.
    pub fn new(
        layout: NumericFp8Layout,
        weight: impl Into<Arc<[u8]>>,
        scales: impl Into<Arc<[u8]>>,
    ) -> Result<Self> {
        let weight = weight.into();
        let scales = scales.into();
        layout.validate_payload(&weight, &scales)?;
        Ok(Self {
            layout,
            weight,
            scales,
        })
    }
    pub const fn layout(&self) -> NumericFp8Layout {
        self.layout
    }
    pub fn weight_bytes(&self) -> &[u8] {
        &self.weight
    }
    pub fn scale_bytes(&self) -> &[u8] {
        &self.scales
    }
    pub fn storage_bytes(&self) -> usize {
        self.weight.len() + self.scales.len()
    }
}
