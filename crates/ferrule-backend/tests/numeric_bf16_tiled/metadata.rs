use ferrule_backend::cuda_plan::{
    NumericFp8Layout, NumericFp8LinearPlan, NumericFp8Precision as P, NumericFp8ScaleType,
    NumericFp8SelectedKernel as K,
};

pub(super) fn qkv_layout() -> NumericFp8Layout {
    NumericFp8Layout {
        n: 8192,
        k: 2048,
        row_origin: 0,
        column_origin: 0,
        scale_type: NumericFp8ScaleType::Bf16,
    }
}

#[test]
fn qkv_candidate_exact_bounded_scratch_and_physical_grants() {
    let l = qkv_layout();
    for (tile, tiles, bytes, grant) in [
        (64, 128, 266_240, 524_288),
        (257, 32, 1_056_768, 2_097_152),
        (1024, 8, 4_198_400, 6_291_456),
    ] {
        let p = NumericFp8LinearPlan::new(l, 1, bytes, P::Bf16RneF32Accumulate).unwrap();
        assert_eq!(p.selected_kernel(), K::Bf16TiledGemm);
        assert_eq!(p.padded_k(), 2048);
        assert_eq!(p.activation_bytes(), 4096);
        assert_eq!(p.tile_rows(), tile);
        assert_eq!(p.tile_count(), tiles);
        assert_eq!(p.weight_tile_bytes(), tile * 4096);
        assert_eq!(p.workspace_requirements().bytes, bytes as u64);
        assert_eq!(p.workspace_allocation_grant_bytes().unwrap(), grant);
        assert_eq!(p.kernel_launches(), 1 + 2 * tiles);
        assert_eq!(p.artifact_allocation_grant_bytes().unwrap(), 16_779_264);
        assert!(p.weight_tile_bytes() < l.n * l.k * 2);
    }
}

#[test]
fn candidate_requires_scratch_and_preserves_fused_profiles() {
    let l = qkv_layout();
    for budget in [0, 4096, 8191] {
        assert!(NumericFp8LinearPlan::new(l, 1, budget, P::Bf16RneF32Accumulate).is_err());
    }
    for (precision, kernel) in [
        (P::W8A16Bf16F32Accumulate, K::W8A16FusedBf16Vector),
        (P::F32Tf32x3, K::F32FusedM1),
    ] {
        let p = NumericFp8LinearPlan::new(l, 1, 0, precision).unwrap();
        assert_eq!(p.selected_kernel(), kernel);
        assert_eq!(p.activation_bytes(), 0);
        assert_eq!(p.weight_tile_bytes(), 0);
        assert_eq!(p.kernel_launches(), 1);
    }
}

#[test]
fn typed_staging_counts_k_padding_and_partial_last_tile() {
    let l = NumericFp8Layout {
        n: 259,
        k: 259,
        ..qkv_layout()
    };
    let bytes = (3 + 17) * 264 * 2;
    let p = NumericFp8LinearPlan::new(l, 3, bytes, P::Bf16RneF32Accumulate).unwrap();
    assert_eq!(p.activation_bytes(), 3 * 264 * 2);
    assert_eq!(p.weight_tile_bytes(), 17 * 264 * 2);
    assert_eq!(p.tile_count(), 16);
    assert_eq!(p.kernel_launches(), 33);
    assert_eq!(p.workspace_requirements().bytes as usize, bytes);
}

// Distance-based oracle, independent of the production bit-bias RNE conversion.
pub(super) fn rounded_bf16(value: f32) -> f32 {
    assert!(value.is_finite());
    let bits = value.abs().to_bits();
    let lower = f32::from_bits(bits & 0xffff_0000);
    let upper = f32::from_bits((bits & 0xffff_0000) + 0x1_0000);
    let down = f64::from(value.abs()) - f64::from(lower);
    let up = f64::from(upper) - f64::from(value.abs());
    let rounded = if down < up || (down == up && (bits >> 16) & 1 == 0) {
        lower
    } else {
        upper
    };
    rounded.copysign(value)
}

pub(super) struct SlicedRoundingCase {
    pub layout: NumericFp8Layout,
    pub input: Vec<f32>,
    pub weights: Vec<u8>,
    pub scales: Vec<f32>,
    pub expected: Vec<f32>,
}

pub(super) fn sliced_rounding_case() -> SlicedRoundingCase {
    let layout = NumericFp8Layout {
        n: 259,
        k: 256,
        row_origin: 127,
        column_origin: 125,
        scale_type: NumericFp8ScaleType::F32,
    };
    let input: Vec<_> = (0..layout.k + 3)
        .map(|i| {
            let bits = [
                0x3f80_7fff,
                0x3f80_8000,
                0x3f80_8001,
                0x3f81_7fff,
                0x3f81_8000,
                0x3f81_8001,
            ][i % 6];
            f32::from_bits(bits | if i % 12 >= 6 { 0x8000_0000 } else { 0 })
        })
        .collect();
    let [sr, sc] = layout.scale_shape().unwrap();
    let scales: Vec<_> = (0..sr * sc)
        .map(|i| f32::from_bits([0x3f80_8001, 0x3f81_8000, 0x3f7f_8000, 0x3f81_0001][i % 4]))
        .collect();
    let mut weights = vec![0; layout.n * layout.k];
    let expected = (0..layout.n)
        .map(|row| {
            let col = row % layout.k;
            let negative = row % 3 == 0;
            weights[row * layout.k + col] = if negative { 0xbc } else { 0x3c };
            let decoded = if negative { -1.5_f32 } else { 1.5_f32 };
            let scale =
                scales[((layout.row_origin + row) / 128) * sc + (layout.column_origin + col) / 128];
            let scaled: f32 = decoded * scale;
            // One nonzero term per output avoids any reduction-order assumption.
            rounded_bf16(input[col]) * rounded_bf16(scaled)
        })
        .collect();
    SlicedRoundingCase {
        layout,
        input,
        weights,
        scales,
        expected,
    }
}

#[test]
fn sliced_rounding_fixture_distinguishes_conversion_order_and_f32_output() {
    let c = sliced_rounding_case();
    c.layout
        .validate_payload(
            &c.weights,
            &c.scales
                .iter()
                .flat_map(|s| s.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    for (bits, expected) in [
        (0x3f80_7fff, 0x3f80_0000),
        (0x3f80_8000, 0x3f80_0000),
        (0x3f80_8001, 0x3f81_0000),
        (0x3f81_7fff, 0x3f81_0000),
        (0x3f81_8000, 0x3f82_0000),
        (0x3f81_8001, 0x3f82_0000),
    ] {
        for sign in [0, 0x8000_0000] {
            assert_eq!(
                rounded_bf16(f32::from_bits(bits | sign)).to_bits(),
                expected | sign
            );
        }
    }
    assert!(c.scales.iter().all(|&s| s != rounded_bf16(s)));
    let sc = c.layout.scale_shape().unwrap()[1];
    let mut missing_activation_round = 0;
    let mut premature_scale_round = 0;
    let mut missing_weight_round = 0;
    for row in 0..c.layout.n {
        let col = row % c.layout.k;
        let decoded = if row % 3 == 0 { -1.5_f32 } else { 1.5_f32 };
        let scale = c.scales
            [((c.layout.row_origin + row) / 128) * sc + (c.layout.column_origin + col) / 128];
        let a = rounded_bf16(c.input[col]);
        let w = rounded_bf16(decoded * scale);
        missing_activation_round += usize::from(c.input[col] * w != c.expected[row]);
        premature_scale_round +=
            usize::from(a * rounded_bf16(decoded * rounded_bf16(scale)) != c.expected[row]);
        missing_weight_round += usize::from(a * (decoded * scale) != c.expected[row]);
    }
    assert!(missing_activation_round > 0);
    assert!(premature_scale_round > 0);
    assert!(missing_weight_round > 0);
    assert!(c.expected.iter().any(|&v| v != rounded_bf16(v)));
}

pub(super) fn gate_layout() -> NumericFp8Layout {
    NumericFp8Layout {
        n: 4096,
        ..qkv_layout()
    }
}

#[test]
fn gate_explicit_profile_tile_boundaries_and_production_budget() {
    let l = gate_layout();
    // Includes the model's actual 8 MiB reservation, not just the replay sizes.
    for (tile, tiles, last, bytes) in [
        (64, 64, 64, 266_240),
        (257, 16, 241, 1_056_768),
        (1024, 4, 1024, 4_198_400),
        (2047, 3, 2, 8 << 20),
    ] {
        let p = NumericFp8LinearPlan::new(l, 1, bytes, P::Bf16RneF32Accumulate).unwrap();
        assert_eq!(p.selected_kernel(), K::Bf16TiledGemm);
        assert_eq!(p.padded_k(), 2048);
        assert_eq!(p.activation_bytes(), 4096);
        assert_eq!(p.tile_rows(), tile);
        assert_eq!(p.tile_count(), tiles);
        assert_eq!(l.n - (tiles - 1) * tile, last);
        assert_eq!(p.weight_tile_bytes(), tile * 4096);
        assert_eq!(p.workspace_requirements().bytes, bytes as u64);
        assert_eq!(p.kernel_launches(), 1 + 2 * tiles);
        for index in [3471, 4065] {
            let first = index / tile * tile;
            let count = tile.min(l.n - first);
            assert!(index >= first && index < first + count);
            // Scale addressing must use the global output row, not the local
            // row in the decoded tile (notably for unaligned 257/2047 tiles).
            assert_eq!((first + index % tile) / 128, index / 128);
        }
    }
    let fused = NumericFp8LinearPlan::new(l, 1, 0, P::W8A16Bf16F32Accumulate).unwrap();
    assert_eq!(fused.selected_kernel(), K::W8A16FusedBf16Vector);
    assert_eq!(fused.kernel_launches(), 1);
}

#[test]
fn sliced_k2_is_explicit_multirow_and_preserves_tiled_workspace_contract() {
    for rows in [1, 2, 5, 9, 17, 65] {
        for tile in [64, 257, 1024, 2047] {
            let bytes = (rows + tile) * 4096;
            let base = NumericFp8LinearPlan::new(
                NumericFp8Layout {
                    n: 8192,
                    ..gate_layout()
                },
                rows,
                bytes,
                P::Bf16RneF32Accumulate,
            )
            .unwrap();
            let sliced = NumericFp8LinearPlan::new(
                NumericFp8Layout {
                    n: 8192,
                    ..gate_layout()
                },
                rows,
                bytes,
                P::Bf16SlicedK2F32Accumulate,
            )
            .unwrap();
            assert_eq!(sliced.selected_kernel(), K::Bf16TiledSlicedK2Gemm);
            assert_eq!(base.selected_kernel(), K::Bf16TiledGemm);
            assert_eq!(
                sliced.workspace_requirements(),
                base.workspace_requirements()
            );
            assert_eq!(sliced.tile_rows(), base.tile_rows());
            assert_eq!(sliced.kernel_launches(), base.kernel_launches());
            assert_eq!(
                sliced.artifact_allocation_grant_bytes().unwrap(),
                base.artifact_allocation_grant_bytes().unwrap()
            );
        }
    }
    assert!(
        NumericFp8LinearPlan::new(
            NumericFp8Layout {
                n: 8192,
                ..gate_layout()
            },
            5,
            5 * 4096 + 257 * 4096,
            P::Bf16SlicedK2F32Accumulate
        )
        .is_ok()
    );
    assert!(
        NumericFp8LinearPlan::new(
            NumericFp8Layout {
                k: 2047,
                ..gate_layout()
            },
            1,
            8 << 20,
            P::Bf16SlicedK2F32Accumulate
        )
        .is_err()
    );
    assert!(
        NumericFp8LinearPlan::new(gate_layout(), 1, 8191, P::Bf16SlicedK2F32Accumulate).is_err()
    );
    for (rows, bytes) in [(0, 8 << 20), (5, 5 * 4096), (5, 6 * 4096 - 1)] {
        assert!(
            NumericFp8LinearPlan::new(gate_layout(), rows, bytes, P::Bf16SlicedK2F32Accumulate)
                .is_err()
        );
    }
}
