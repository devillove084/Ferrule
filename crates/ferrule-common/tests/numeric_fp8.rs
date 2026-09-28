use ferrule_common::numeric_fp8::{
    ImmutableValidatedNumericFp8Payload as Payload, NumericFp8Layout as Layout,
    NumericFp8ScaleType as ScaleType,
};
use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

fn layout(n: usize, k: usize, scale_type: ScaleType) -> Layout {
    Layout {
        n,
        k,
        row_origin: 0,
        column_origin: 0,
        scale_type,
    }
}
fn scales(ty: ScaleType, values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| match ty {
            ScaleType::Bf16 => ((v.to_bits() >> 16) as u16).to_le_bytes().to_vec(),
            ScaleType::F32 => v.to_le_bytes().to_vec(),
        })
        .collect()
}
fn oracle_abs(byte: u8) -> f32 {
    let e = (byte >> 3) & 15;
    let m = byte & 7;
    if e == 0 {
        f32::from(m) / 512.0
    } else {
        2.0f32.powi(i32::from(e) - 7) * (1.0 + f32::from(m) / 8.0)
    }
}

#[test]
fn all_254_finite_codes_and_signed_zeros_are_valid() {
    for ty in [ScaleType::Bf16, ScaleType::F32] {
        for code in 0..=255u8 {
            // Exercise every lane, not just the scalar remainder.
            let result = Payload::new(layout(1, 17, ty), vec![code; 17], scales(ty, &[1.0]));
            assert_eq!(
                result.is_ok(),
                code != 0x7f && code != 0xff,
                "{ty:?}: {code:02x}"
            );
        }
        let weight: Vec<_> = (0..=255u8).filter(|b| b & 0x7f != 0x7f).collect();
        assert!(Payload::new(layout(1, 254, ty), weight, scales(ty, &[1.0, 1.0])).is_ok());
    }
}

#[test]
fn nan_detection_at_every_lane_tail_and_block_boundary() {
    for length in [1, 7, 8, 9, 15, 16, 17, 127, 128, 129, 257] {
        let l = layout(1, length, ScaleType::F32);
        let sb = scales(l.scale_type, &vec![1.0; length.div_ceil(128)]);
        for position in 0..length {
            for nan in [0x7f, 0xff] {
                let mut w = vec![0x7e; length];
                w[position] = nan;
                let raw_error = l.validate_payload(&w, &sb).unwrap_err().to_string();
                let proof_error = Payload::new(l, w, sb.clone()).unwrap_err().to_string();
                assert_eq!(raw_error, proof_error);
                assert!(raw_error.contains("NaN E4M3FN"), "{length}/{position}");
            }
        }
    }
}

#[test]
fn invalid_scales_and_lengths_are_rejected() {
    for ty in [ScaleType::Bf16, ScaleType::F32] {
        let l = layout(1, 129, ty);
        let weight = vec![0x38; 129];
        for bad in [
            0.0,
            -0.0,
            -1.0,
            f32::NAN,
            -f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ] {
            for values in [[bad, 1.0], [1.0, bad]] {
                let error = Payload::new(l, weight.clone(), scales(ty, &values)).unwrap_err();
                assert!(error.to_string().contains("finite and strictly positive"));
            }
        }
        let sb = scales(ty, &[1.0, 1.0]);
        for length in [0, 128, 130] {
            assert!(Payload::new(l, vec![0x38; length], sb.clone()).is_err());
        }
        for length in [0, sb.len() - 1, sb.len() + 1] {
            assert!(Payload::new(l, weight.clone(), vec![0; length]).is_err());
        }
        let tiny = match ty {
            ScaleType::Bf16 => f32::from_bits(1 << 16),
            ScaleType::F32 => f32::from_bits(1),
        };
        assert!(Payload::new(l, weight, scales(ty, &[tiny, tiny])).is_ok());
    }
}

#[test]
fn rejects_invalid_dimensions_and_origin_overflow_without_allocating() {
    let l = layout(2, 2, ScaleType::F32);
    for bad in [
        Layout { n: 0, ..l },
        Layout { k: 0, ..l },
        Layout { n: usize::MAX, ..l },
        Layout { k: usize::MAX, ..l },
        Layout {
            row_origin: usize::MAX,
            ..l
        },
        Layout {
            column_origin: usize::MAX,
            ..l
        },
        Layout {
            row_origin: isize::MAX as usize,
            ..l
        },
        Layout {
            column_origin: isize::MAX as usize,
            ..l
        },
    ] {
        assert!(bad.storage_lengths().is_err());
        assert!(Payload::new(bad, Vec::new(), Vec::new()).is_err());
    }
    let ragged = Layout {
        row_origin: 127,
        column_origin: 255,
        ..l
    };
    assert_eq!(ragged.scale_shape().unwrap(), [2, 2]);
    assert!(Payload::new(ragged, vec![0; 4], scales(ScaleType::F32, &[1.0])).is_err());
}

#[test]
fn immutable_proof_shares_bytes_but_cannot_be_changed_via_aliases() {
    let mut l = layout(1, 4, ScaleType::F32);
    let mut weight: Arc<[u8]> = Arc::from([0x38; 4]);
    let mut sb: Arc<[u8]> = Arc::from(1.0f32.to_le_bytes());
    let proof = Payload::new(l, weight.clone(), sb.clone()).unwrap();
    let clone = proof.clone();
    assert_eq!(weight.as_ptr(), proof.weight_bytes().as_ptr());
    assert_eq!(sb.as_ptr(), proof.scale_bytes().as_ptr());
    assert_eq!(clone.weight_bytes().as_ptr(), proof.weight_bytes().as_ptr());
    assert!(Arc::get_mut(&mut weight).is_none());
    assert!(Arc::get_mut(&mut sb).is_none());
    Arc::make_mut(&mut weight)[3] = 0xff;
    Arc::make_mut(&mut sb).copy_from_slice(&f32::NAN.to_le_bytes());
    l.row_origin = 127;
    assert_eq!(proof.layout().row_origin, 0);
    assert_eq!(proof.weight_bytes(), &[0x38; 4]);
    assert_eq!(proof.scale_bytes(), &1.0f32.to_le_bytes());
    assert_eq!(proof.storage_bytes(), 8);
    proof
        .layout()
        .validate_payload(proof.weight_bytes(), proof.scale_bytes())
        .unwrap();
    assert!(Payload::new(l, weight, sb).is_err());
}

#[test]
fn overflow_uses_actual_block_maximum_not_format_maximum() {
    for ty in [ScaleType::Bf16, ScaleType::F32] {
        let largest = match ty {
            ScaleType::Bf16 => f32::from_bits(0x7f7f_0000),
            ScaleType::F32 => f32::MAX,
        };
        let l = layout(1, 1, ty);
        // Exhaustive comparison to actual F32 multiplication, around the
        // overflow threshold and for both signs, zeros and subnormals.
        for scale in [
            largest,
            largest / 2.0,
            largest / 448.0,
            f32::from_bits((largest / 448.0).to_bits() + 1),
            1.0,
        ] {
            let sb = scales(ty, &[scale]);
            let scale = match ty {
                ScaleType::Bf16 => f32::from_bits(scale.to_bits() & 0xffff_0000),
                ScaleType::F32 => scale,
            };
            for code in 0..=255u8 {
                if code & 0x7f == 0x7f {
                    continue;
                }
                let expected = (oracle_abs(code) * scale).is_finite();
                assert_eq!(
                    Payload::new(l, vec![code], sb.clone()).is_ok(),
                    expected,
                    "{ty:?} code={code:02x}, scale={scale}"
                );
            }
        }
    }
}

#[test]
fn ragged_origins_map_extreme_scales_to_the_correct_local_block() {
    // One weight in each of four intersecting global blocks. A global max
    // (448) combined with the top-left huge scale would falsely reject this.
    let l = Layout {
        row_origin: 255,
        column_origin: 383,
        ..layout(2, 2, ScaleType::F32)
    };
    let sb = scales(l.scale_type, &[f32::MAX, 1.0, f32::MAX, 1.0]);
    let good = [0x38, 0x7e, 0x80, 0xfe];
    Payload::new(l, good.to_vec(), sb.clone()).unwrap();
    for position in [0, 2] {
        let mut bad = good;
        bad[position] = 0x40; // 2 * MAX is infinite.
        let error = Payload::new(l, bad.to_vec(), sb.clone()).unwrap_err();
        assert!(error.to_string().contains("overflow"));
    }
    // Uneven rows and columns cross several complete/partial blocks. Compare
    // an independent per-element oracle, including an overflowing final byte.
    for (ro, co, n, k) in [(127, 125, 131, 259), (256, 129, 129, 131), (0, 0, 129, 129)] {
        let l = Layout {
            row_origin: ro,
            column_origin: co,
            ..layout(n, k, ScaleType::F32)
        };
        let [sr, sc] = l.scale_shape().unwrap();
        let sv: Vec<_> = (0..sr * sc)
            .map(|i| if i % 2 == 0 { f32::MAX } else { 1.0 })
            .collect();
        let mut w = vec![0; n * k];
        for row in 0..n {
            for col in 0..k {
                let index = ((ro + row) / 128 - ro / 128) * sc + (co + col) / 128 - co / 128;
                w[row * k + col] = if sv[index] == 1.0 { 0xfe } else { 0xb8 };
            }
        }
        Payload::new(l, w.clone(), scales(l.scale_type, &sv)).unwrap();
        let last = n * k - 1;
        w.fill(0xb8);
        let sb = scales(l.scale_type, &vec![f32::MAX; sr * sc]);
        Payload::new(l, w.clone(), sb.clone()).unwrap();
        w[last] = 0x7e;
        let error = Payload::new(l, w, sb).unwrap_err().to_string();
        assert!(error.contains(&format!("overflow in scale block {}", sr * sc - 1)));
    }
}

// Baseline preserves the pre-change backend algorithm, including the short-
// circuit per-byte scan and numeric scale checks. No CUDA or copies are timed.
#[inline(never)]
fn original_validation(l: Layout, weight: &[u8], sb: &[u8]) -> bool {
    if l.storage_lengths().ok() != Some((weight.len(), sb.len())) {
        return false;
    }
    if weight.iter().any(|b| b & 0x7f == 0x7f) {
        return false;
    }
    for bytes in sb.chunks_exact(l.scale_type.element_bytes()) {
        let value = match l.scale_type {
            ScaleType::Bf16 => {
                f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16)
            }
            ScaleType::F32 => f32::from_le_bytes(bytes.try_into().unwrap()),
        };
        if !value.is_finite() || value <= 0.0 {
            return false;
        }
    }
    true
}
fn median(mut run: impl FnMut(), iterations: usize) -> Duration {
    for _ in 0..3 {
        run();
    }
    let mut samples: Vec<_> = (0..iterations)
        .map(|_| {
            let start = Instant::now();
            run();
            start.elapsed()
        })
        .collect();
    samples.sort();
    samples[samples.len() / 2]
}

#[test]
#[ignore = "CPU-only 32 MiB benchmark; run --release --ignored --nocapture"]
fn benchmark_validation_32_mib() {
    let l = layout(4096, 8192, ScaleType::F32);
    let weight: Arc<[u8]> = (0..l.n * l.k)
        .map(|i| {
            let code = (i % 254) as u8;
            if code >= 127 { code + 1 } else { code }
        })
        .collect();
    let sb: Arc<[u8]> = scales(l.scale_type, &vec![0.125; 32 * 64]).into();
    let original = median(
        || {
            assert!(black_box(original_validation(
                black_box(l),
                black_box(&weight),
                black_box(&sb)
            )))
        },
        21,
    );
    let raw = median(
        || black_box(l.validate_payload(black_box(&weight), black_box(&sb))).unwrap(),
        21,
    );
    let constructor = median(
        || {
            black_box(
                Payload::new(
                    black_box(l),
                    black_box(weight.clone()),
                    black_box(sb.clone()),
                )
                .unwrap(),
            );
        },
        21,
    );
    let proof = Payload::new(l, weight, sb).unwrap();
    let reuse = median(
        || {
            for _ in 0..100_000 {
                let p = black_box(&proof);
                black_box(
                    p.layout()
                        .validate_lengths(p.weight_bytes().len(), p.scale_bytes().len()),
                )
                .unwrap();
            }
        },
        21,
    )
    .div_f64(100_000.0);
    eprintln!(
        "32 MiB, 21-sample medians: original={original:?}, shared_raw={raw:?}, Arc_constructor={constructor:?}, proof_structure={reuse:?}; constructor reduction={:.1}%, former two scans -> constructor reduction={:.1}%",
        100.0 * (1.0 - constructor.as_secs_f64() / original.as_secs_f64()),
        100.0 * (1.0 - constructor.as_secs_f64() / (2.0 * original.as_secs_f64()))
    );
    // Intentionally not a timing assertion: noisy/shared CI is not a correctness oracle.
}
