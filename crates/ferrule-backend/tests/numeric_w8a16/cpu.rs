use super::oracle::*;
use ferrule_common::numeric_fp8::NumericFp8ScaleType as Scale;

#[test]
fn exhaustive_decode_signed_zero_subnormal_max_and_nan() {
    for code in 0..128u8 {
        if code == 127 {
            assert!(decode(code).is_nan() && decode(code | 128).is_nan());
        } else {
            assert_eq!(
                decode(code | 128).to_bits(),
                decode(code).to_bits() | 0x8000_0000
            );
            if code > 0 {
                assert!(decode(code) > decode(code - 1));
            }
        }
    }
    for (code, value) in [
        (0, 0.0),
        (1, 1.0 / 512.0),
        (7, 7.0 / 512.0),
        (8, 1.0 / 64.0),
        (0x38, 1.0),
        (0x7e, 448.0),
    ] {
        assert_eq!(decode(code), value);
    }
}

#[test]
fn bf16_rne_ties_subnormal_overflow_nan_and_f32_boundary() {
    for (bits, expected) in [
        (0x0000_0000, 0x0000),
        (0x8000_0000, 0x8000),
        (0x3f80_8000, 0x3f80),
        (0x3f81_8000, 0x3f82),
        (0xbf80_8000, 0xbf80),
        (0xbf81_8000, 0xbf82),
        (0x3f80_8001, 0x3f81),
        (0x0000_8000, 0),
        (0x0000_8001, 1),
        (0x007f_ffff, 0x0080),
        (0x7f7f_7fff, 0x7f7f),
        (0x7f7f_8000, 0x7f80),
        (0xff7f_8000, 0xff80),
        (0x7f80_0000, 0x7f80),
        (0x7f80_0001, 0x7fc0),
    ] {
        assert_eq!(bf16_bits(f32::from_bits(bits)), expected);
    }
    assert_eq!(bf16(1.0001) - bf16(1.0), 0.0);
    assert!(1.0001f32.mul_add(1.0, -1.0) > 0.0);
    // F32 multiply then BF16 rounding, not BF16(scale) first.
    let scale = 1.003;
    assert_ne!(bf16(1.75 * scale), bf16(1.75 * bf16(scale)));
}

#[test]
fn positive_scales_lengths_block_offsets_and_overflow_validation() {
    for ty in [Scale::Bf16, Scale::F32] {
        let l = layout(131, 259, ty);
        assert_eq!(l.scale_shape().unwrap(), [3, 3]);
        let w = weight(l);
        let (sb, sv) = scales(l);
        l.validate_payload(&w, &sb).unwrap();
        for (n, k) in [(0usize, 0usize), (1, 3), (129, 131), (130, 258)] {
            let si = n.div_ceil(128) * 3 + (125 + k) / 128;
            assert_eq!(decoded(l, &w, &sv, n, k), decode(w[n * l.k + k]) * sv[si]);
        }
        for bad in [0.0f32, -0.0, -1.0, f32::INFINITY, f32::NAN] {
            let mut invalid = sb.clone();
            let bytes = match ty {
                Scale::Bf16 => bf16_bits(bad).to_le_bytes().to_vec(),
                Scale::F32 => bad.to_le_bytes().to_vec(),
            };
            invalid[..bytes.len()].copy_from_slice(&bytes);
            assert!(l.validate_payload(&w, &invalid).is_err());
        }
        assert!(l.validate_payload(&w[..w.len() - 1], &sb).is_err());
        let mut nan = w.clone();
        nan[7] = 0xff;
        assert!(l.validate_payload(&nan, &sb).is_err());
    }
    let l = layout(1, 1, Scale::F32);
    assert!(
        l.validate_payload(&[0x7e], &f32::MAX.to_le_bytes())
            .is_err()
    );
    l.validate_payload(&[0], &f32::MAX.to_le_bytes()).unwrap();
}

#[test]
fn independent_multirow_tiled_operands_match_f64_and_serial_f32_oracles() {
    for m in [1, 2, 4, 8, 32] {
        for (n, k) in [(1, 1), (7, 3), (33, 35), (131, 129), (17, 259)] {
            for ty in [Scale::F32, Scale::Bf16] {
                let l = layout(n, k, ty);
                let w = weight(l);
                let (_, scales) = scales(l);
                let a: Vec<_> = (0..m * k)
                    .map(|i| (i as i32 % 37 - 18) as f32 * 0.03173)
                    .collect();
                for row in 0..m {
                    for col in 0..n {
                        let (expected, norm, serial) = dot(l, &w, &scales, &a, k, row, col);
                        let tiled = (0..k.div_ceil(32)).fold(0.0f32, |mut sum, tile| {
                            for kk in 0..32 {
                                let idx = tile * 32 + kk;
                                let (av, wv) = if idx < k {
                                    (
                                        bf16(a[row * k + idx]),
                                        bf16(decoded(l, &w, &scales, col, idx)),
                                    )
                                } else {
                                    (0.0, 0.0)
                                };
                                sum = av.mul_add(wv, sum);
                            }
                            sum
                        });
                        assert_eq!(tiled, serial);
                        assert!((f64::from(tiled) - expected).abs() <= 1e-6 + 2e-6 * norm);
                    }
                }
            }
        }
    }
}
