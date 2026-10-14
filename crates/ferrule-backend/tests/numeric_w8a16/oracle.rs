use ferrule_common::numeric_fp8::{NumericFp8Layout, NumericFp8ScaleType};

// Arithmetic decode, independent of the native bit-field implementation.
pub fn decode(code: u8) -> f32 {
    if code & 127 == 127 {
        return f32::NAN;
    }
    let exponent = (code >> 3) & 15;
    let mantissa = code & 7;
    let magnitude = if exponent == 0 {
        f32::from(mantissa) / 512.0
    } else {
        f32::from(8 + mantissa) * 2.0f32.powi(i32::from(exponent) - 10)
    };
    if code & 128 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

// Compare discarded bits rather than adding the native rounding bias.
pub fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let upper = (bits >> 16) as u16;
    if value.is_nan() {
        return upper | 0x40;
    }
    let lower = bits & 65535;
    upper.wrapping_add(u16::from(
        lower > 32768 || (lower == 32768 && upper & 1 != 0),
    ))
}
pub fn bf16(value: f32) -> f32 {
    f32::from_bits(u32::from(bf16_bits(value)) << 16)
}

pub fn layout(n: usize, k: usize, ty: NumericFp8ScaleType) -> NumericFp8Layout {
    NumericFp8Layout {
        n,
        k,
        row_origin: 255,
        column_origin: 253,
        scale_type: ty,
    }
}

pub fn scales(l: NumericFp8Layout) -> (Vec<u8>, Vec<f32>) {
    let [nr, nc] = l.scale_shape().unwrap();
    let values: Vec<_> = (0..nr * nc)
        .map(|i| {
            let value = 0.01317 + (i % 17) as f32 * 0.00319;
            if l.scale_type == NumericFp8ScaleType::Bf16 {
                bf16(value)
            } else {
                value
            }
        })
        .collect();
    let bytes = values
        .iter()
        .flat_map(|&value| match l.scale_type {
            NumericFp8ScaleType::Bf16 => bf16_bits(value).to_le_bytes().to_vec(),
            NumericFp8ScaleType::F32 => value.to_le_bytes().to_vec(),
        })
        .collect();
    (bytes, values)
}

pub fn weight(l: NumericFp8Layout) -> Vec<u8> {
    let codes = [0, 128, 1, 0x87, 0x38, 0xbc, 0x7e, 0xfe, 0x21];
    (0..l.n * l.k)
        .map(|i| codes[(i * 7 + i / l.k) % codes.len()])
        .collect()
}

pub fn decoded(l: NumericFp8Layout, weights: &[u8], scales: &[f32], n: usize, k: usize) -> f32 {
    // Absolute coordinates, unlike native residue addressing.
    let stride = (l.column_origin + l.k - 1) / 128 - l.column_origin / 128 + 1;
    let sr = (l.row_origin + n) / 128 - l.row_origin / 128;
    let sc = (l.column_origin + k) / 128 - l.column_origin / 128;
    decode(weights[n * l.k + k]) * scales[sr * stride + sc]
}

#[allow(clippy::too_many_arguments)]
pub fn dot(
    l: NumericFp8Layout,
    weights: &[u8],
    scales: &[f32],
    a: &[f32],
    lda: usize,
    m: usize,
    n: usize,
) -> (f64, f64, f32) {
    let mut sum = 0.0f64;
    let mut norm = 0.0f64;
    let mut f32_serial = 0.0f32;
    for k in 0..l.k {
        let av = bf16(a[m * lda + k]);
        let wv = bf16(decoded(l, weights, scales, n, k));
        let product = f64::from(av) * f64::from(wv);
        sum += product;
        norm += product.abs();
        f32_serial = av.mul_add(wv, f32_serial);
    }
    (sum, norm, f32_serial)
}
