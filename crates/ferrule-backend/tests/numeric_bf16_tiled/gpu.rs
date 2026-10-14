use super::metadata::qkv_layout;
use ferrule_backend::cuda::operators::linear::*;
use ferrule_backend::cuda::providers::DeviceBuffer;
use std::io::Write;
use std::path::Path;
use std::process::Command;

const TILED: NumericFp8Precision = NumericFp8Precision::Bf16RneF32Accumulate;
const FIXTURE_PIN: &str = "a26fb38acfa8f3d05a8bc8923d99b51b9e0cedf535f8c5a456e3f0b9309a8936";

fn bytes(dir: &Path, name: &str, len: usize) -> Vec<u8> {
    let path = dir.join(name);
    assert_eq!(std::fs::metadata(&path).unwrap().len(), len as u64);
    std::fs::read(path).unwrap()
}

fn words(data: &[u8]) -> Vec<u16> {
    assert!(data.len().is_multiple_of(2));
    data.as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes(*b))
        .collect()
}

fn values(words: &[u16]) -> Vec<f32> {
    words
        .iter()
        .map(|&w| f32::from_bits(u32::from(w) << 16))
        .collect()
}

#[test]
#[ignore = "GPU owner only; synthetic sliced-K2 F32 scale and BF16 rounding oracle"]
fn sliced_k2_synthetic_f32_scale_and_bf16_rounding() {
    let case = super::metadata::sliced_rounding_case();
    let scale_bytes: Vec<u8> = case.scales.iter().flat_map(|s| s.to_le_bytes()).collect();
    let input_bf16: Vec<u16> = case
        .input
        .iter()
        .map(|&value| (super::metadata::rounded_bf16(value).to_bits() >> 16) as u16)
        .collect();
    let op = CudaOperators::new_on_device(0).unwrap();
    let artifact = op
        .upload_numeric_fp8_linear(case.layout, &case.weights, &scale_bytes)
        .unwrap();
    let f32_input = op.upload_f32_buffer(&case.input).unwrap();
    let bf16_input = op.upload_bf16_words(&input_bf16).unwrap();
    let plan = NumericFp8LinearPlan::new(
        case.layout,
        1,
        (1 + 17) * case.layout.k * 2,
        NumericFp8Precision::Bf16SlicedK2F32Accumulate,
    )
    .unwrap();
    assert_eq!(plan.tile_rows(), 17);
    for typed in [false, true] {
        let mut output = op
            .upload_f32_buffer(&vec![7331.; case.layout.n + 5])
            .unwrap();
        let mut workspace = op.numeric_fp8_linear_workspace(plan).unwrap();
        if typed {
            op.numeric_fp8_bf16_linear_into(
                &artifact,
                &bf16_input,
                &mut output,
                &mut workspace,
                plan,
                case.layout.k + 3,
                case.layout.n + 5,
            )
            .unwrap();
        } else {
            op.numeric_fp8_linear_into(
                &artifact,
                &f32_input,
                &mut output,
                &mut workspace,
                plan,
                case.layout.k + 3,
                case.layout.n + 5,
            )
            .unwrap();
        }
        let observed = op.download_f32_buffer(&output).unwrap();
        assert_eq!(&observed[..case.layout.n], case.expected.as_slice());
        assert!(observed[case.layout.n..].iter().all(|&v| v == 7331.));
        assert!(!workspace.is_poisoned());
    }
}

#[test]
#[ignore = "GPU owner only; pinned Hello L0 QKV SAMEINPUT, no full model/reference regeneration"]
fn fixed_input_qkv_native_tiled_compatibility() {
    let dir = std::path::PathBuf::from(
        std::env::var_os("FERRULE_BF16_TILED_QKV_DIR").expect("pinned fixture directory"),
    );
    let report_path = std::path::PathBuf::from(
        std::env::var_os("FERRULE_BF16_TILED_QKV_REPORT").expect("fresh JSONL report path"),
    );
    let sha = Command::new("sha256sum")
        .arg(dir.join("SHA256SUMS"))
        .output()
        .unwrap();
    assert!(sha.status.success());
    assert_eq!(
        String::from_utf8(sha.stdout)
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap(),
        FIXTURE_PIN
    );
    assert!(
        Command::new("sha256sum")
            .args(["--check", "SHA256SUMS"])
            .current_dir(&dir)
            .status()
            .unwrap()
            .success()
    );
    let layout = qkv_layout();
    let input_words = words(&bytes(&dir, "input.bf16", layout.k * 2));
    let expected = values(&words(&bytes(&dir, "hf.bf16", layout.n * 2)));
    let observed = values(&words(&bytes(&dir, "observed.bf16", layout.n * 2)));
    let weights = bytes(&dir, "weight.fp8", layout.n * layout.k);
    let scales = bytes(&dir, "scale.bf16", 2048);
    assert_eq!(
        expected
            .iter()
            .zip(&observed)
            .filter(|(a, b)| a != b)
            .count(),
        11
    );
    let mut report = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&report_path)
        .unwrap();
    writeln!(report, "{{\"event\":\"start\",\"fixture_sha256\":\"{FIXTURE_PIN}\",\"shape\":[1,8192,2048],\"model_profile_changed\":false,\"native_layout\":\"CTA64x64x32 warp32x32x32 MMA16x8x16 stages3\",\"expected_source\":\"pinned HF, never production replay\",\"atol\":0.00025,\"rtol\":0.012,\"tolerance_source\":\"existing hybrid_bf16 tiny, diagnostic counts only; compatibility requires exact\"}}").unwrap();
    report.sync_all().unwrap();
    let op = CudaOperators::new_on_device(0).unwrap();
    let artifact = op
        .upload_numeric_fp8_linear(layout, &weights, &scales)
        .unwrap();
    let input = op.upload_bf16_words(&input_words).unwrap();
    let mut output = op.zero_f32_buffer(layout.n).unwrap();
    let mut rounded = op.zero_bf16_buffer(layout.n).unwrap();
    let mut fused_matches_capture = false;
    let mut all_candidates_exact = true;
    for (precision, tile_rows) in [
        (NumericFp8Precision::W8A16Bf16F32Accumulate, 0),
        (TILED, 64),
        (TILED, 257),
        (TILED, 1024),
    ] {
        let budget = if tile_rows == 0 {
            0
        } else {
            (1 + tile_rows) * layout.k * 2
        };
        let plan = NumericFp8LinearPlan::new(layout, 1, budget, precision).unwrap();
        let mut workspace = op.numeric_fp8_linear_workspace(plan).unwrap();
        op.reset_counters();
        let before = op.allocator_metrics();
        op.numeric_fp8_bf16_linear_into(
            &artifact,
            &input,
            &mut output,
            &mut workspace,
            plan,
            layout.k,
            layout.n,
        )
        .unwrap();
        op.f32_rows_to_bf16_rne_into(
            &output,
            &mut rounded,
            F32ToBf16RowsLayout::packed(1, 1, layout.n).unwrap(),
        )
        .unwrap();
        assert_eq!(
            op.counters().compute_kernel_launches,
            plan.kernel_launches() as u64 + 1
        );
        assert_eq!(op.counters().device_allocation_attempts, 0);
        assert_eq!(op.counters().stream_wide_syncs, 0);
        let after = op.allocator_metrics();
        assert_eq!(after.live_granted_bytes, before.live_granted_bytes);
        assert_eq!(after.driver_allocations, before.driver_allocations);
        let raw = op.download_f32_buffer(&output).unwrap();
        let actual = op.download_bf16_buffer(&rounded).unwrap();
        let mut unequal = 0;
        let mut outside = 0;
        let mut versus_capture = 0;
        let mut max_abs = 0.0_f64;
        for (i, ((&a, &e), &c)) in actual.iter().zip(&expected).zip(&observed).enumerate() {
            assert!(a.is_finite() && raw[i].is_finite());
            let error = (f64::from(a) - f64::from(e)).abs();
            max_abs = max_abs.max(error);
            outside += usize::from(error > 0.00025 + 0.012 * f64::from(e).abs());
            versus_capture += usize::from(a.to_bits() != c.to_bits());
            if a.to_bits() != e.to_bits() {
                unequal += 1;
                writeln!(report, "{{\"event\":\"difference\",\"tile_rows\":{tile_rows},\"index\":{i},\"f32_accumulator\":{},\"actual_bf16\":{a},\"hf_bf16\":{e},\"observed_bf16\":{c}}}", raw[i]).unwrap();
            }
        }
        writeln!(report, "{{\"event\":\"comparison\",\"kernel\":\"{}\",\"tile_rows\":{tile_rows},\"elements\":8192,\"unequal_hf\":{unequal},\"outside_existing_bound\":{outside},\"unequal_capture\":{versus_capture},\"max_abs\":{max_abs},\"activation_bytes\":{},\"weight_tile_bytes\":{},\"workspace_bytes\":{},\"workspace_grant_bytes\":{},\"artifact_grant_bytes\":{},\"live_granted_bytes\":{},\"gemm_and_staging_launches\":{},\"bf16_writeback_launches\":1}}", plan.selected_kernel().name(), plan.activation_bytes(), plan.weight_tile_bytes(), workspace.allocated_bytes(), plan.workspace_allocation_grant_bytes().unwrap(), plan.artifact_allocation_grant_bytes().unwrap(), after.live_granted_bytes, plan.kernel_launches()).unwrap();
        report.sync_all().unwrap();
        if tile_rows == 0 {
            fused_matches_capture = versus_capture == 0;
        } else {
            all_candidates_exact &= unequal == 0;
        }
        assert!(!workspace.is_poisoned());
    }
    writeln!(report, "{{\"event\":\"complete\",\"fused_matches_capture\":{fused_matches_capture},\"all_candidates_exact_hf\":{all_candidates_exact}}}").unwrap();
    report.sync_all().unwrap();
    op.record_compute_event().unwrap().synchronize().unwrap();
    assert!(
        fused_matches_capture,
        "baseline changed; do not reinterpret compatibility evidence"
    );
    assert!(
        all_candidates_exact,
        "candidate is NOT pinned-HF compatible; full report: {}",
        report_path.display()
    );
}

#[test]
#[ignore = "GPU owner only; typed tiled padding/strides, partial tile, native GEMM and independent exact arithmetic"]
fn typed_tiled_padding_strides_and_graph_replay() {
    let op = CudaOperators::new_on_device(0).unwrap();
    for (m, n, k, tile, precision) in [
        (1, 7, 3, 1, TILED),
        (3, 259, 259, 17, TILED),
        (9, 33, 129, 7, TILED),
        (
            1,
            259,
            2048,
            17,
            NumericFp8Precision::Bf16SlicedK2F32Accumulate,
        ),
        (2, 7, 64, 3, NumericFp8Precision::Bf16SlicedK2F32Accumulate),
        (
            5,
            259,
            256,
            17,
            NumericFp8Precision::Bf16SlicedK2F32Accumulate,
        ),
        (
            9,
            33,
            2048,
            7,
            NumericFp8Precision::Bf16SlicedK2F32Accumulate,
        ),
        (
            17,
            129,
            64,
            65,
            NumericFp8Precision::Bf16SlicedK2F32Accumulate,
        ),
        (
            65,
            129,
            256,
            65,
            NumericFp8Precision::Bf16SlicedK2F32Accumulate,
        ),
    ] {
        let layout = NumericFp8Layout {
            n,
            k,
            row_origin: 127,
            column_origin: 125,
            scale_type: NumericFp8ScaleType::Bf16,
        };
        let [sr, sc] = layout.scale_shape().unwrap();
        let weights: Vec<u8> = (0..n * k)
            .map(|i| if i % 3 == 0 { 0xb8 } else { 0x38 })
            .collect();
        let scale_words: Vec<u16> = (0..sr * sc)
            .map(|i| if i % 2 == 0 { 0x3f00 } else { 0x3f80 })
            .collect();
        let scales: Vec<u8> = scale_words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let artifact = op
            .upload_numeric_fp8_linear(layout, &weights, &scales)
            .unwrap();
        let lda = k + 3;
        let ldd = n + 5;
        let input_words: Vec<u16> = (0..m * lda)
            .map(|i| if i % 5 == 0 { 0xbf00 } else { 0x3e80 })
            .collect();
        let input = op.upload_bf16_words(&input_words).unwrap();
        let f32_input = op.upload_f32_buffer(&values(&input_words)).unwrap();
        let mut output = op.upload_f32_buffer(&vec![7331.; m * ldd + 7]).unwrap();
        let mut legacy = op.upload_f32_buffer(&vec![7331.; m * ldd + 7]).unwrap();
        let padded = k.div_ceil(8) * 8;
        let plan =
            NumericFp8LinearPlan::new(layout, m, (m + tile) * padded * 2, precision).unwrap();
        let mut workspace = op.numeric_fp8_linear_workspace(plan).unwrap();
        op.numeric_fp8_linear_into(
            &artifact,
            &f32_input,
            &mut legacy,
            &mut workspace,
            plan,
            lda,
            ldd,
        )
        .unwrap();
        op.record_compute_event().unwrap().synchronize().unwrap();
        op.reset_counters();
        let before = op.allocator_metrics();
        let graph = op
            .capture_decode_graph(|| {
                op.numeric_fp8_bf16_linear_into(
                    &artifact,
                    &input,
                    &mut output,
                    &mut workspace,
                    plan,
                    lda,
                    ldd,
                )
            })
            .unwrap();
        assert_eq!(
            op.counters().compute_kernel_launches,
            plan.kernel_launches() as u64
        );
        for _ in 0..2 {
            op.launch_graph(&graph).unwrap();
        }
        assert_eq!(op.counters().device_allocation_attempts, 0);
        assert_eq!(op.counters().stream_wide_syncs, 0);
        assert_eq!(
            op.allocator_metrics().live_granted_bytes,
            before.live_granted_bytes
        );
        let consumer = op.stream_clone().context().new_stream().unwrap();
        consumer
            .wait(&op.stream_clone().record_event(None).unwrap())
            .unwrap();
        let actual = output.as_device_buffer().to_host_vec(&consumer).unwrap();
        assert_eq!(actual, op.download_f32_buffer(&legacy).unwrap());
        let a = values(&input_words);
        let s = values(&scale_words);
        for row in 0..m {
            for col in 0..n {
                let expected: f32 = (0..k)
                    .map(|inner| {
                        let sign = if weights[col * k + inner] == 0xb8 {
                            -1.
                        } else {
                            1.
                        };
                        a[row * lda + inner]
                            * sign
                            * s[((layout.row_origin + col) / 128) * sc
                                + (layout.column_origin + inner) / 128]
                    })
                    .sum();
                assert_eq!(actual[row * ldd + col], expected);
            }
            assert!(
                actual[row * ldd + n..(row + 1) * ldd]
                    .iter()
                    .all(|&v| v == 7331.)
            );
        }
        assert!(actual[m * ldd..].iter().all(|&v| v == 7331.));
        assert!(!workspace.is_poisoned());
    }
}

#[test]
#[ignore = "GPU owner only; typed tiled owner/profile/budget failure atomicity"]
fn typed_tiled_rejects_foreign_owner_wrong_profile_and_workspace_budget() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let other = CudaOperators::new_on_device(0).unwrap();
    for (m, k, budget, precision) in [
        (2, 3, 64, TILED),
        (5, 64, 1024, NumericFp8Precision::Bf16SlicedK2F32Accumulate),
    ] {
        let l = NumericFp8Layout {
            n: 3,
            k,
            ..qkv_layout()
        };
        let artifact = op
            .upload_numeric_fp8_linear(l, &vec![0x38; 3 * k], &0x3f80_u16.to_le_bytes())
            .unwrap();
        let input = op.upload_bf16_words(&vec![0x3f80; m * k]).unwrap();
        let foreign = other.upload_bf16_words(&vec![0x3f80; m * k]).unwrap();
        let mut output = op.upload_f32_buffer(&vec![73.; m * 3]).unwrap();
        let plan = NumericFp8LinearPlan::new(l, m, budget, precision).unwrap();
        let mut workspace = op.numeric_fp8_linear_workspace(plan).unwrap();
        let mut oversized = CudaNumericFp8Workspace::from_buffer_with_precision(
            DeviceBuffer::<u8>::zeroed(&op.stream_clone(), budget + 1).unwrap(),
            precision,
        );
        let w8 = NumericFp8LinearPlan::new(l, m, 0, NumericFp8Precision::W8A16Bf16F32Accumulate)
            .unwrap();
        let mut wrong = op.numeric_fp8_linear_workspace(w8).unwrap();
        op.reset_counters();
        assert!(
            op.numeric_fp8_bf16_linear_into(
                &artifact,
                &foreign,
                &mut output,
                &mut workspace,
                plan,
                k,
                3
            )
            .is_err()
        );
        assert_eq!(op.counters().compute_kernel_launches, 0);
        assert_eq!(op.counters().device_allocation_attempts, 0);
        let mut undersized = CudaNumericFp8Workspace::from_buffer_with_precision(
            DeviceBuffer::<u8>::zeroed(&op.stream_clone(), budget - 1).unwrap(),
            precision,
        );
        let other_precision = if precision == TILED {
            NumericFp8Precision::F32Tf32x3
        } else {
            TILED
        };
        let other_plan = NumericFp8LinearPlan::new(l, m, budget, other_precision).unwrap();
        let mut wrong_tiled = op.numeric_fp8_linear_workspace(other_plan).unwrap();
        op.reset_counters();
        for ws in [
            &mut oversized,
            &mut undersized,
            &mut wrong,
            &mut wrong_tiled,
        ] {
            assert!(
                op.numeric_fp8_bf16_linear_into(&artifact, &input, &mut output, ws, plan, k, 3)
                    .is_err()
            );
            assert!(!ws.is_poisoned());
        }
        for (lda, ldd) in [(k - 1, 3), (k, 2)] {
            assert!(
                op.numeric_fp8_bf16_linear_into(
                    &artifact,
                    &input,
                    &mut output,
                    &mut workspace,
                    plan,
                    lda,
                    ldd
                )
                .is_err()
            );
        }
        assert_eq!(op.counters().compute_kernel_launches, 0);
        let fp32 = NumericFp8LinearPlan::new(l, m, budget, NumericFp8Precision::F32Tf32x3).unwrap();
        let mut fws = op.numeric_fp8_linear_workspace(fp32).unwrap();
        op.reset_counters();
        assert!(
            op.numeric_fp8_bf16_linear_into(&artifact, &input, &mut output, &mut fws, fp32, k, 3)
                .is_err()
        );
        assert_eq!(op.counters().compute_kernel_launches, 0);
        assert!(!workspace.is_poisoned());
        assert_eq!(op.download_f32_buffer(&output).unwrap(), vec![73.; m * 3]);
    }
}

#[test]
#[ignore = "GPU owner only; pinned Hello L0 in_proj_z, no full40 or reference regeneration"]
fn fixed_input_gate_native_tiled_reduction_profile() {
    const PIN: &str = "c6a90487a76ab30d5babb4e0154886c44024969d372a60833a3ff9c9a10190c5";
    let dir = std::path::PathBuf::from(
        std::env::var_os("FERRULE_BF16_TILED_GATE_DIR").expect("pinned gate fixture directory"),
    );
    let report_path = std::path::PathBuf::from(
        std::env::var_os("FERRULE_BF16_TILED_GATE_REPORT").expect("fresh gate JSONL report path"),
    );
    let sha = Command::new("sha256sum")
        .arg(dir.join("SHA256SUMS"))
        .output()
        .unwrap();
    assert!(sha.status.success());
    assert_eq!(
        String::from_utf8(sha.stdout)
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap(),
        PIN
    );
    assert!(
        Command::new("sha256sum")
            .args(["--check", "SHA256SUMS"])
            .current_dir(&dir)
            .status()
            .unwrap()
            .success()
    );
    let layout = super::metadata::gate_layout();
    let input_words = words(&bytes(&dir, "input.bf16", 4096));
    let hf = words(&bytes(&dir, "hf.bf16", 8192));
    let captured_fused = words(&bytes(&dir, "fused.bf16", 8192));
    let captured_tiled = words(&bytes(&dir, "observed.bf16", 8192));
    let weights = bytes(&dir, "weight.fp8", layout.n * layout.k);
    let scales = bytes(&dir, "scale.bf16", 1024);
    let mut report = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&report_path)
        .unwrap();
    writeln!(report, "{{\"event\":\"start\",\"fixture_sha256\":\"{PIN}\",\"shape\":[1,4096,2048],\"production_tile_rows\":2047,\"acceptance\":\"fused and production capture reproducibility ONLY; tile invariance diagnostic; NOT HF compatibility or model acceptance\",\"reference_or_tolerance_changed\":false}}").unwrap();
    report.sync_all().unwrap();
    let op = CudaOperators::new_on_device(0).unwrap();
    let artifact = op
        .upload_numeric_fp8_linear(layout, &weights, &scales)
        .unwrap();
    let input = op.upload_bf16_words(&input_words).unwrap();
    let mut output = op.zero_f32_buffer(layout.n).unwrap();
    let mut rounded = op.zero_bf16_buffer(layout.n).unwrap();
    let mut fused_words = Vec::new();
    let mut first_tiled_raw = Vec::new();
    let mut reproducible = true;
    let mut invariant = true;
    for (precision, tile_rows) in [
        (NumericFp8Precision::W8A16Bf16F32Accumulate, 0),
        (TILED, 64),
        (TILED, 257),
        (TILED, 1024),
        (TILED, 2047),
    ] {
        let budget = if tile_rows == 0 {
            0
        } else {
            (1 + tile_rows) * layout.k * 2
        };
        let plan = NumericFp8LinearPlan::new(layout, 1, budget, precision).unwrap();
        assert_eq!(plan.tile_rows(), tile_rows);
        let mut workspace = op.numeric_fp8_linear_workspace(plan).unwrap();
        op.reset_counters();
        op.numeric_fp8_bf16_linear_into(
            &artifact,
            &input,
            &mut output,
            &mut workspace,
            plan,
            layout.k,
            layout.n,
        )
        .unwrap();
        op.f32_rows_to_bf16_rne_into(
            &output,
            &mut rounded,
            F32ToBf16RowsLayout::packed(1, 1, layout.n).unwrap(),
        )
        .unwrap();
        assert_eq!(
            op.counters().compute_kernel_launches,
            plan.kernel_launches() as u64 + 1
        );
        assert_eq!(op.counters().device_allocation_attempts, 0);
        assert_eq!(op.counters().stream_wide_syncs, 0);
        let raw = op.download_f32_buffer(&output).unwrap();
        let actual = op.download_bf16_buffer(&rounded).unwrap();
        let actual_words: Vec<u16> = actual.iter().map(|x| (x.to_bits() >> 16) as u16).collect();
        let raw_bits: Vec<u32> = raw.iter().map(|x| x.to_bits()).collect();
        if tile_rows == 0 {
            fused_words = actual_words.clone();
        }
        if tile_rows == 64 {
            first_tiled_raw = raw_bits.clone();
        }
        let mut unequal_hf = 0;
        let mut unequal_fused = 0;
        let mut unequal_capture = 0;
        let mut unequal_raw_64 = 0;
        let mut max_abs = 0.0_f64;
        for i in 0..layout.n {
            assert!(raw[i].is_finite() && actual[i].is_finite());
            let rounded_bits = raw_bits[i].wrapping_add(0x7fff + ((raw_bits[i] >> 16) & 1));
            assert_eq!(
                actual_words[i],
                (rounded_bits >> 16) as u16,
                "BF16 RNE writeback differs at tile_rows={tile_rows}, index={i}"
            );
            let expected = f32::from_bits(u32::from(hf[i]) << 16);
            max_abs = max_abs.max((f64::from(actual[i]) - f64::from(expected)).abs());
            unequal_hf += usize::from(actual_words[i] != hf[i]);
            unequal_fused += usize::from(actual_words[i] != fused_words[i]);
            let capture = if tile_rows == 0 {
                captured_fused[i]
            } else {
                captured_tiled[i]
            };
            unequal_capture += usize::from(actual_words[i] != capture);
            let raw_changed = tile_rows != 0 && raw_bits[i] != first_tiled_raw[i];
            unequal_raw_64 += usize::from(raw_changed);
            // Include the union of all comparisons, plus the two original
            // divergence rows even if a replay happens to match HF there.
            if actual_words[i] != hf[i]
                || actual_words[i] != fused_words[i]
                || actual_words[i] != capture
                || raw_changed
                || [3471, 4065].contains(&i)
            {
                let first = i.checked_div(tile_rows).unwrap_or(0) * tile_rows;
                let count = if tile_rows == 0 {
                    layout.n
                } else {
                    tile_rows.min(layout.n - first)
                };
                writeln!(report, "{{\"event\":\"word\",\"tile_rows\":{tile_rows},\"index\":{i},\"tile_first\":{first},\"tile_count\":{count},\"tile_local\":{},\"scale_row\":{},\"actual_bf16\":\"0x{:04x}\",\"hf_bf16\":\"0x{:04x}\",\"fused_bf16\":\"0x{:04x}\",\"capture_bf16\":\"0x{capture:04x}\",\"actual_f32_bits\":\"0x{:08x}\",\"hf_f32_bits\":\"0x{:08x}\",\"accumulator\":{},\"accumulator_bits\":\"0x{:08x}\"}}",
                    i - first, i / 128, actual_words[i], hf[i], fused_words[i], actual[i].to_bits(), expected.to_bits(), raw[i], raw_bits[i]).unwrap();
            }
        }
        let mut raw_file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(report_path.with_extension(format!("tile-{tile_rows}.f32")))
            .unwrap();
        for bits in &raw_bits {
            raw_file.write_all(&bits.to_le_bytes()).unwrap();
        }
        raw_file.sync_all().unwrap();
        if tile_rows == 0 || tile_rows == 2047 {
            reproducible &= unequal_capture == 0;
        }
        invariant &= unequal_raw_64 == 0;
        writeln!(report, "{{\"event\":\"comparison\",\"kernel\":\"{}\",\"tile_rows\":{tile_rows},\"tile_count\":{},\"unequal_hf\":{unequal_hf},\"unequal_fused\":{unequal_fused},\"unequal_capture\":{unequal_capture},\"unequal_raw_64\":{unequal_raw_64},\"max_abs\":{max_abs},\"workspace_bytes\":{},\"staging_and_gemm_launches\":{},\"writeback_launches\":1}}", plan.selected_kernel().name(), plan.tile_count(), workspace.allocated_bytes(), plan.kernel_launches()).unwrap();
        report.sync_all().unwrap();
        assert!(!workspace.is_poisoned());
    }
    op.record_compute_event().unwrap().synchronize().unwrap();
    writeln!(report, "{{\"event\":\"complete\",\"captures_reproduced\":{reproducible},\"raw_tile_invariant\":{invariant},\"hf_compatibility_claimed\":false}}").unwrap();
    report.sync_all().unwrap();
    assert!(
        reproducible,
        "bounded gate differs; inspect {}",
        report_path.display()
    );
}

#[test]
#[ignore = "GPU owner only; exact pinned gate regression for explicit sliced-K2 profile"]
fn fixed_input_gate_sliced_k2_exact_hf() {
    let dir = std::path::PathBuf::from(std::env::var_os("FERRULE_BF16_TILED_GATE_DIR").unwrap());
    let report_path =
        std::path::PathBuf::from(std::env::var_os("FERRULE_BF16_TILED_GATE_REPORT").unwrap());
    let native_path =
        std::path::PathBuf::from(std::env::var_os("FERRULE_BF16_GATE_NATIVE_F32").unwrap());
    for (path, pin) in [
        (
            dir.join("SHA256SUMS"),
            "c6a90487a76ab30d5babb4e0154886c44024969d372a60833a3ff9c9a10190c5",
        ),
        (
            native_path.clone(),
            "0a4c26c497f737540a287029015531091296d5a45e4a677bba0d7f6d474c1243",
        ),
    ] {
        let hash = Command::new("sha256sum").arg(path).output().unwrap();
        assert!(hash.status.success());
        assert_eq!(
            String::from_utf8(hash.stdout)
                .unwrap()
                .split_whitespace()
                .next()
                .unwrap(),
            pin
        );
    }
    assert!(
        Command::new("sha256sum")
            .args(["--check", "SHA256SUMS"])
            .current_dir(&dir)
            .status()
            .unwrap()
            .success()
    );
    let layout = super::metadata::gate_layout();
    let input_words = words(&bytes(&dir, "input.bf16", 4096));
    let hf = words(&bytes(&dir, "hf.bf16", 8192));
    let native_bytes = std::fs::read(native_path).unwrap();
    assert_eq!(native_bytes.len(), layout.n * 4);
    let native: Vec<u32> = native_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect();
    let op = CudaOperators::new_on_device(0).unwrap();
    let artifact = op
        .upload_numeric_fp8_linear(
            layout,
            &bytes(&dir, "weight.fp8", 8388608),
            &bytes(&dir, "scale.bf16", 1024),
        )
        .unwrap();
    let input = op.upload_bf16_words(&input_words).unwrap();
    let f32_input = op.upload_f32_buffer(&values(&input_words)).unwrap();
    let mut output = op.zero_f32_buffer(layout.n).unwrap();
    let mut rounded = op.zero_bf16_buffer(layout.n).unwrap();
    let mut report = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(report_path)
        .unwrap();
    for tile in [64, 257, 1024, 2047] {
        let budget = (1 + tile) * 4096;
        let plan = NumericFp8LinearPlan::new(
            layout,
            1,
            budget,
            NumericFp8Precision::Bf16SlicedK2F32Accumulate,
        )
        .unwrap();
        let mut ws = op.numeric_fp8_linear_workspace(plan).unwrap();
        let old = NumericFp8LinearPlan::new(layout, 1, budget, TILED).unwrap();
        let mut wrong_profile = op.numeric_fp8_linear_workspace(old).unwrap();
        op.reset_counters();
        assert!(
            op.numeric_fp8_bf16_linear_into(
                &artifact,
                &input,
                &mut output,
                &mut wrong_profile,
                plan,
                layout.k,
                layout.n
            )
            .is_err()
        );
        assert_eq!(op.counters().compute_kernel_launches, 0);
        assert!(!wrong_profile.is_poisoned());
        for typed in [true, false] {
            op.reset_counters();
            let before = op.allocator_metrics();
            if typed {
                op.numeric_fp8_bf16_linear_into(
                    &artifact,
                    &input,
                    &mut output,
                    &mut ws,
                    plan,
                    layout.k,
                    layout.n,
                )
                .unwrap();
            } else {
                op.numeric_fp8_linear_into(
                    &artifact,
                    &f32_input,
                    &mut output,
                    &mut ws,
                    plan,
                    layout.k,
                    layout.n,
                )
                .unwrap();
            }
            op.f32_rows_to_bf16_rne_into(
                &output,
                &mut rounded,
                F32ToBf16RowsLayout::packed(1, 1, layout.n).unwrap(),
            )
            .unwrap();
            assert_eq!(
                op.counters().compute_kernel_launches,
                plan.kernel_launches() as u64 + 1
            );
            assert_eq!(op.counters().device_allocation_attempts, 0);
            assert_eq!(op.counters().stream_wide_syncs, 0);
            let after = op.allocator_metrics();
            assert_eq!(before.live_granted_bytes, after.live_granted_bytes);
            assert_eq!(before.driver_allocations, after.driver_allocations);
            let raw = op.download_f32_buffer(&output).unwrap();
            let actual = op.download_bf16_buffer(&rounded).unwrap();
            let mut raw_diff = 0;
            let mut bf_diff = 0;
            for i in 0..layout.n {
                raw_diff += usize::from(raw[i].to_bits() != native[i]);
                bf_diff += usize::from(actual[i].to_bits() != u32::from(hf[i]) << 16);
                if raw[i].to_bits() != native[i]
                    || actual[i].to_bits() != u32::from(hf[i]) << 16
                    || [3471, 4065].contains(&i)
                {
                    writeln!(report,"{{\"event\":\"word\",\"tile\":{tile},\"typed\":{typed},\"index\":{i},\"raw\":\"0x{:08x}\",\"hf_raw\":\"0x{:08x}\",\"actual\":\"0x{:04x}\",\"hf\":\"0x{:04x}\"}}",raw[i].to_bits(),native[i],actual[i].to_bits()>>16,hf[i]).unwrap();
                }
            }
            writeln!(report,"{{\"event\":\"comparison\",\"tile\":{tile},\"typed\":{typed},\"raw_unequal\":{raw_diff},\"bf16_unequal\":{bf_diff},\"workspace_bytes\":{},\"launches\":{}}}",ws.allocated_bytes(),plan.kernel_launches()+1).unwrap();
            report.sync_all().unwrap();
            assert_eq!(
                (raw_diff, bf_diff),
                (0, 0),
                "exact native gate compatibility required"
            );
            assert!(!ws.is_poisoned());
        }
    }
    op.record_compute_event().unwrap().synchronize().unwrap();
}

#[test]
#[ignore = "GPU owner only; frozen Capital M5 QKV native reduction comparison"]
fn fixed_input_capital_m5_qkv_exact_sliced_k2() {
    let dir = std::path::PathBuf::from(std::env::var_os("FERRULE_BF16_M5_DIR").unwrap());
    let native_path =
        std::path::PathBuf::from(std::env::var_os("FERRULE_BF16_M5_NATIVE_F32").unwrap());
    let report_path = std::path::PathBuf::from(std::env::var_os("FERRULE_BF16_M5_REPORT").unwrap());
    for (path, pin) in [
        (
            dir.join("SHA256SUMS"),
            "a586606e89ea99537b601a2abefd9d0636be3a8085e4f0bc30ee3c4b640a31c7",
        ),
        (
            native_path.clone(),
            "b7832ea6f3d2db55bf1463d9f70ec9084194d0106cfee93e70de021d937bef80",
        ),
    ] {
        let hash = Command::new("sha256sum").arg(path).output().unwrap();
        assert!(hash.status.success());
        assert_eq!(
            String::from_utf8(hash.stdout)
                .unwrap()
                .split_whitespace()
                .next()
                .unwrap(),
            pin
        );
    }
    assert!(
        Command::new("sha256sum")
            .args(["--check", "SHA256SUMS"])
            .current_dir(&dir)
            .status()
            .unwrap()
            .success()
    );
    let layout = qkv_layout();
    let m = 5;
    let elements = m * layout.n;
    let x = words(&bytes(&dir, "input.bf16", m * layout.k * 2));
    let hf = words(&bytes(&dir, "hf.bf16", elements * 2));
    let capture = words(&bytes(&dir, "observed.bf16", elements * 2));
    let raw_hf = std::fs::read(native_path).unwrap();
    assert_eq!(raw_hf.len(), elements * 4);
    let native: Vec<_> = raw_hf
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect();
    let op = CudaOperators::new_on_device(0).unwrap();
    let artifact = op
        .upload_numeric_fp8_linear(
            layout,
            &bytes(&dir, "weight.fp8", layout.n * layout.k),
            &bytes(&dir, "scale.bf16", 2048),
        )
        .unwrap();
    let input = op.upload_bf16_words(&x).unwrap();
    let f32_input = op.upload_f32_buffer(&values(&x)).unwrap();
    let mut output = op.zero_f32_buffer(elements).unwrap();
    let mut rounded = op.zero_bf16_buffer(elements).unwrap();
    let mut report = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(report_path)
        .unwrap();
    let sliced = NumericFp8Precision::Bf16SlicedK2F32Accumulate;
    for (precision, tile) in [
        (NumericFp8Precision::W8A16Bf16F32Accumulate, 0),
        (TILED, 64),
        (TILED, 257),
        (TILED, 1024),
        (TILED, 2043),
        (sliced, 64),
        (sliced, 257),
        (sliced, 1024),
        (sliced, 2043),
    ] {
        let budget = if tile == 0 {
            0
        } else {
            (m + tile) * layout.k * 2
        };
        let plan = NumericFp8LinearPlan::new(layout, m, budget, precision).unwrap();
        let mut ws = op.numeric_fp8_linear_workspace(plan).unwrap();
        assert_eq!(plan.tile_rows(), tile);
        for typed in [true, false] {
            op.reset_counters();
            let before = op.allocator_metrics();
            if typed {
                op.numeric_fp8_bf16_linear_into(
                    &artifact,
                    &input,
                    &mut output,
                    &mut ws,
                    plan,
                    layout.k,
                    layout.n,
                )
                .unwrap();
            } else {
                op.numeric_fp8_linear_into(
                    &artifact,
                    &f32_input,
                    &mut output,
                    &mut ws,
                    plan,
                    layout.k,
                    layout.n,
                )
                .unwrap();
            }
            op.f32_rows_to_bf16_rne_into(
                &output,
                &mut rounded,
                F32ToBf16RowsLayout::packed(1, m, layout.n).unwrap(),
            )
            .unwrap();
            assert_eq!(
                op.counters().compute_kernel_launches,
                plan.kernel_launches() as u64 + 1
            );
            assert_eq!(op.counters().device_allocation_attempts, 0);
            assert_eq!(op.counters().stream_wide_syncs, 0);
            let after = op.allocator_metrics();
            assert_eq!(after.live_granted_bytes, before.live_granted_bytes);
            assert_eq!(after.driver_allocations, before.driver_allocations);
            let raw = op.download_f32_buffer(&output).unwrap();
            let actual = op.download_bf16_buffer(&rounded).unwrap();
            let mut bf_diff = 0;
            let mut raw_diff = 0;
            let mut capture_diff = 0;
            let mut max_abs = 0.0f64;
            for i in 0..elements {
                assert!(raw[i].is_finite() && actual[i].is_finite());
                let b = raw[i].to_bits();
                let word = (actual[i].to_bits() >> 16) as u16;
                assert_eq!(
                    word,
                    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
                );
                bf_diff += usize::from(word != hf[i]);
                raw_diff += usize::from(b != native[i]);
                capture_diff += usize::from(word != capture[i]);
                max_abs = max_abs.max(
                    (f64::from(actual[i]) - f64::from(f32::from_bits(u32::from(hf[i]) << 16)))
                        .abs(),
                );
                if word != hf[i] || i == 750 {
                    writeln!(report,"{{\"event\":\"word\",\"kernel\":\"{}\",\"tile\":{tile},\"typed\":{typed},\"index\":{i},\"m_row\":{},\"n_col\":{},\"actual\":\"0x{word:04x}\",\"hf\":\"0x{:04x}\",\"raw\":\"0x{b:08x}\",\"native_raw\":\"0x{:08x}\"}}",plan.selected_kernel().name(),i/layout.n,i%layout.n,hf[i],native[i]).unwrap();
                }
            }
            writeln!(report,"{{\"event\":\"comparison\",\"kernel\":\"{}\",\"tile\":{tile},\"typed\":{typed},\"bf16_unequal\":{bf_diff},\"raw_unequal\":{raw_diff},\"capture_unequal\":{capture_diff},\"max_abs\":{max_abs},\"scratch\":{},\"launches\":{}}}",plan.selected_kernel().name(),ws.allocated_bytes(),plan.kernel_launches()+1).unwrap();
            report.sync_all().unwrap();
            if precision == sliced {
                assert_eq!((bf_diff, raw_diff), (0, 0));
            }
            if precision == TILED {
                assert_eq!((bf_diff, capture_diff), (26, 0));
            }
            assert!(!ws.is_poisoned());
        }
    }
    op.record_compute_event().unwrap().synchronize().unwrap();
}
