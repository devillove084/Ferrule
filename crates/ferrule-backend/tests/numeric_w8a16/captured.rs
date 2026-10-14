//! Frozen real operands through existing production and independent scalar entries.
//! This is an ignored operator experiment, not a new model harness or quality gate.
use super::{ScalarLayout, ferrule_test_w8a16_scalar, within_bound};
use crate::oracle::{bf16, dot};
use ferrule_backend::cuda::operators::linear::*;
use std::io::Write;
use std::path::Path;
use std::process::Command;

fn payload(directory: &Path, name: &str, bytes: usize) -> Vec<u8> {
    assert!(bytes <= 128 * 1024 * 1024, "bounded operator fixture");
    let path = directory.join(name);
    assert_eq!(std::fs::metadata(&path).unwrap().len(), bytes as u64);
    std::fs::read(path).unwrap()
}

#[test]
#[ignore = "GPU owner only + FERRULE_CUDA_TEST_ORACLE=1 + pinned real-operand fixture; no model run"]
fn captured_same_bf16_operands_vector_mma_scalar_f64() {
    let directory = std::path::PathBuf::from(
        std::env::var_os("FERRULE_W8A16_CAPTURED_CASE_DIR").expect("frozen fixture directory"),
    );
    let pin = std::env::var("FERRULE_W8A16_CAPTURED_MANIFEST_SHA256").expect("SHA256SUMS pin");
    let report = std::env::var_os("FERRULE_W8A16_CAPTURED_REPORT").expect("fresh report path");
    run(directory, &pin, report);
}

pub(super) fn run(directory: std::path::PathBuf, pin: &str, report_path: std::ffi::OsString) {
    let hash = Command::new("sha256sum")
        .arg(directory.join("SHA256SUMS"))
        .output()
        .unwrap();
    assert!(hash.status.success());
    assert_eq!(
        String::from_utf8(hash.stdout)
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap(),
        pin
    );
    assert!(
        Command::new("sha256sum")
            .args(["--check", "SHA256SUMS"])
            .current_dir(&directory)
            .status()
            .unwrap()
            .success()
    );
    let shape = std::fs::read_to_string(directory.join("shape.txt"))
        .unwrap()
        .split_whitespace()
        .map(|v| v.parse::<usize>().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(shape.len(), 3);
    let (observed_m, n, k) = (shape[0], shape[1], shape[2]);
    assert!((1..=8).contains(&observed_m) && (1..=8192).contains(&n));
    assert!((128..=4096).contains(&k) && k % 128 == 0);
    let input_words = payload(&directory, "input.bf16", observed_m * k * 2)
        .as_chunks::<2>()
        .0
        .iter()
        .map(|v| u16::from_le_bytes(*v))
        .collect::<Vec<_>>();
    let weights = payload(&directory, "weight.fp8", n * k);
    let scale_bytes = payload(
        &directory,
        "scale.bf16",
        n.div_ceil(128) * k.div_ceil(128) * 2,
    );
    let scales = scale_bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|v| f32::from_bits(u32::from(u16::from_le_bytes(*v)) << 16))
        .collect::<Vec<_>>();
    let captured = payload(&directory, "captured.f32", observed_m * n * 4)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect::<Vec<_>>();
    let serial_cpu = payload(&directory, "serial.f32", observed_m * n * 4)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect::<Vec<_>>();
    let truth = payload(&directory, "truth.f64", observed_m * n * 8)
        .as_chunks::<8>()
        .0
        .iter()
        .map(|v| f64::from_le_bytes(*v))
        .collect::<Vec<_>>();
    let norms = payload(&directory, "sumabs.f64", observed_m * n * 8)
        .as_chunks::<8>()
        .0
        .iter()
        .map(|v| f64::from_le_bytes(*v))
        .collect::<Vec<_>>();
    let input_f32 = input_words
        .iter()
        .map(|&v| f32::from_bits(u32::from(v) << 16))
        .collect::<Vec<_>>();
    let layout = NumericFp8Layout {
        n,
        k,
        row_origin: 0,
        column_origin: 0,
        scale_type: NumericFp8ScaleType::Bf16,
    };
    // Spot-check fixture F64/serial with a second host implementation before CUDA.
    for row in 0..observed_m {
        for col in [0, n / 2, n - 1] {
            let (expected, norm, serial) = dot(layout, &weights, &scales, &input_f32, k, row, col);
            assert_eq!(expected, truth[row * n + col]);
            assert_eq!(norm, norms[row * n + col]);
            assert_eq!(serial.to_bits(), serial_cpu[row * n + col].to_bits());
        }
    }
    let mut report = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(report_path)
        .unwrap();
    writeln!(report, "{{\"schema\":\"ferrule.w8a16-captured-triangle.v1\",\"fixture_sha256\":\"{pin}\",\"observed_m\":{observed_m},\"n\":{n},\"k\":{k},\"pairwise_acceptance\":false,\"bound\":{{\"absolute\":0.00001,\"sum_abs_products_factor\":0.000002}},\"observations\":[").unwrap();
    let op = CudaOperators::new_on_device(0).unwrap();
    let artifact = op
        .upload_numeric_fp8_linear(layout, &weights, &scale_bytes)
        .unwrap();
    let mut total_failures = 0;
    let mut total_serial_mismatches = 0;
    let mut runtime_version = 0;
    let mut first = true;
    // Four controlled probes repeat EXACTLY row zero, including M8/M9 dispatch
    // transition. The observed case uses ALL captured rows without modification.
    for (kind, m) in [
        ("captured", observed_m),
        ("repeat_row0", 1),
        ("repeat_row0", 8),
        ("repeat_row0", 9),
        ("repeat_row0", 32),
    ] {
        let lda = k + 3;
        let ldd = n + 5;
        let mut a = vec![7331.0; m * lda];
        for row in 0..m {
            let source = if kind == "captured" { row } else { 0 };
            a[row * lda..row * lda + k].copy_from_slice(&input_f32[source * k..(source + 1) * k]);
        }
        let input = op.upload_f32_buffer(&a).unwrap();
        let words = a
            .iter()
            .map(|&v| (bf16(v).to_bits() >> 16) as u16)
            .collect::<Vec<_>>();
        let bfinput = op.upload_bf16_words(&words).unwrap();
        let mut output = op.upload_f32_buffer(&vec![7331.; m * ldd + 7]).unwrap();
        let mut bfoutput = op.upload_f32_buffer(&vec![7331.; m * ldd + 7]).unwrap();
        let plan =
            NumericFp8LinearPlan::new(layout, m, 0, NumericFp8Precision::W8A16Bf16F32Accumulate)
                .unwrap();
        let mut workspace = op.numeric_fp8_linear_workspace(plan).unwrap();
        op.numeric_fp8_linear_into(
            &artifact,
            &input,
            &mut output,
            &mut workspace,
            plan,
            lda,
            ldd,
        )
        .unwrap();
        op.numeric_fp8_bf16_linear_into(
            &artifact,
            &bfinput,
            &mut bfoutput,
            &mut workspace,
            plan,
            lda,
            ldd,
        )
        .unwrap();
        op.record_compute_event().unwrap().synchronize().unwrap();
        let actual = op.download_f32_buffer(&output).unwrap();
        let bfactual = op.download_f32_buffer(&bfoutput).unwrap();
        assert!(
            actual
                .iter()
                .zip(&bfactual)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "F32-input and frozen BF16-input APIs differ"
        );
        op.stream_clone().context().bind_to_thread().unwrap();
        let mut scalar = vec![7331.; m * ldd + 7];
        let scalar_layout = ScalarLayout {
            m: m as u64,
            n: n as u64,
            k: k as u64,
            row_origin: 0,
            column_origin: 0,
            lda: lda as u64,
            ldd: ldd as u64,
            scale_bytes: 2,
        };
        #[allow(
            unsafe_code,
            reason = "test-only host ABI; fixed spans, completed production work, bound CUDA context"
        )]
        let status = unsafe {
            ferrule_test_w8a16_scalar(
                &scalar_layout,
                a.as_ptr(),
                weights.as_ptr(),
                scale_bytes.as_ptr().cast(),
                scalar.as_mut_ptr(),
                &mut runtime_version,
            )
        };
        assert_eq!(status, 0);
        let (
            mut production_failures,
            mut scalar_failures,
            mut pair_bits,
            mut serial_bits,
            mut captured_bits,
            mut bin_changes,
        ) = (0, 0, 0, 0, 0, 0);
        let (
            mut production_error,
            mut scalar_error,
            mut pair_error,
            mut production_fraction,
            mut scalar_fraction,
        ) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for row in 0..m {
            let source = if kind == "captured" { row } else { 0 };
            for col in 0..n {
                let t = truth[source * n + col];
                let norm = norms[source * n + col];
                let p = actual[row * ldd + col];
                let s = scalar[row * ldd + col];
                production_failures += usize::from(!within_bound(f64::from(p), t, norm));
                scalar_failures += usize::from(!within_bound(f64::from(s), t, norm));
                pair_bits += usize::from(p.to_bits() != s.to_bits());
                serial_bits += usize::from(s.to_bits() != serial_cpu[source * n + col].to_bits());
                captured_bits += usize::from(p.to_bits() != captured[source * n + col].to_bits());
                bin_changes += usize::from(bf16(p).to_bits() != bf16(s).to_bits());
                production_error = production_error.max((f64::from(p) - t).abs());
                scalar_error = scalar_error.max((f64::from(s) - t).abs());
                pair_error = pair_error.max((f64::from(p) - f64::from(s)).abs());
                production_fraction =
                    production_fraction.max((f64::from(p) - t).abs() / (1e-5 + 2e-6 * norm));
                scalar_fraction =
                    scalar_fraction.max((f64::from(s) - t).abs() / (1e-5 + 2e-6 * norm));
            }
            assert!(
                actual[row * ldd + n..(row + 1) * ldd]
                    .iter()
                    .all(|&v| v == 7331.)
            );
            assert!(
                scalar[row * ldd + n..(row + 1) * ldd]
                    .iter()
                    .all(|&v| v == 7331.)
            );
        }
        assert!(actual[m * ldd..].iter().all(|&v| v == 7331.));
        assert!(scalar[m * ldd..].iter().all(|&v| v == 7331.));
        assert_eq!(op.download_f32_buffer(&input).unwrap(), a);
        if !first {
            writeln!(report, ",").unwrap();
        }
        first = false;
        writeln!(report,"{{\"kind\":\"{kind}\",\"m\":{m},\"kernel\":\"{:?}\",\"production_failures\":{production_failures},\"scalar_failures\":{scalar_failures},\"production_max_abs\":{production_error},\"scalar_max_abs\":{scalar_error},\"pair_max_abs\":{pair_error},\"production_max_fraction\":{production_fraction},\"scalar_max_fraction\":{scalar_fraction},\"pair_bit_mismatches\":{pair_bits},\"pair_bf16_bin_changes\":{bin_changes},\"scalar_vs_cpu_serial_bit_mismatches\":{serial_bits},\"production_vs_saved_vector_bit_mismatches\":{captured_bits}}}",plan.selected_kernel()).unwrap();
        total_failures += production_failures + scalar_failures;
        total_serial_mismatches += serial_bits;
    }
    writeln!(report,"],\"cuda_runtime\":{runtime_version},\"failures\":{total_failures},\"scalar_cpu_mismatches\":{total_serial_mismatches},\"integrity_passed\":true,\"certifies_full_model\":false}}").unwrap();
    assert_eq!(
        total_failures, 0,
        "existing operator bound violated; retain report"
    );
    assert_eq!(
        total_serial_mismatches, 0,
        "independent CPU serial/CUDA scalar disagree"
    );
}
