//! Triangle test, not production-reduction emulation or a full-model oracle.
use super::oracle::*;
use ferrule_backend::cuda::operators::linear::*;
use std::ffi::c_void;
use std::io::Write;

#[repr(C)]
struct ScalarLayout {
    m: u64,
    n: u64,
    k: u64,
    row_origin: u64,
    column_origin: u64,
    lda: u64,
    ldd: u64,
    scale_bytes: u64,
}

#[allow(
    unsafe_code,
    reason = "test-only host-array ABI; no production provider linkage"
)]
unsafe extern "C" {
    fn ferrule_test_w8a16_scalar(
        layout: &ScalarLayout,
        input: *const f32,
        weight: *const u8,
        scales: *const c_void,
        output: *mut f32,
        runtime_version: &mut i32,
    ) -> i32;
}

// Exactly the pre-existing gpu.rs operator bound, not the full40 logit bound.
fn within_bound(actual: f64, expected: f64, norm: f64) -> bool {
    actual.is_finite()
        && expected.is_finite()
        && norm.is_finite()
        && (actual - expected).abs() <= 1e-5 + 2e-6 * norm
}

#[test]
fn triangle_bound_does_not_require_identical_reductions_or_hide_wrong_results() {
    assert!(within_bound(0., 1., 33554433.));
    assert!(within_bound(1., 1., 33554433.));
    assert!(!within_bound(2., 1., 1.));
    assert!(!within_bound(f64::NAN, 1., 1.));
    assert!(!within_bound(f64::INFINITY, 1., 1.));
    assert!(!within_bound(1., 1., f64::INFINITY));
}

#[test]
fn scalar_source_has_no_production_includes_or_reduction_replay() {
    let source = include_str!("reference.cu");
    let includes: Vec<_> = source
        .lines()
        .filter(|line| line.starts_with("#include"))
        .collect();
    assert_eq!(
        includes,
        [
            "#include <cuda_runtime.h>",
            "#include <cmath>",
            "#include <cstdint>"
        ]
    );
    for forbidden in [
        "__shfl",
        "wmma",
        "cutlass/",
        "numeric_w8a16.cuh",
        "cuda_bf16.h",
    ] {
        assert!(!source.contains(forbidden), "{forbidden}");
    }
}

#[test]
#[ignore = "GPU owner + FERRULE_CUDA_TEST_ORACLE=1; independent scalar CUDA/host/production triangle"]
fn independent_scalar_cuda_host_production_triangle() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let mut records = Vec::new();
    let mut total_elements = 0usize;
    let mut total_failures = 0usize;
    let mut runtime_version = 0;
    for m in [1, 2, 4, 8, 32] {
        for (n, k) in [
            (1, 1),
            (7, 3),
            (33, 35),
            (131, 129),
            (17, 259),
            (7, 8193),
            (33, 128),
            (17, 256),
        ] {
            for (row_origin, column_origin) in [(0, 0), (127, 125), (255, 253), (128, 128)] {
                for ty in [NumericFp8ScaleType::F32, NumericFp8ScaleType::Bf16] {
                    let l = NumericFp8Layout {
                        n,
                        k,
                        row_origin,
                        column_origin,
                        scale_type: ty,
                    };
                    // All 254 finite E4M3FN codes, including signed zero, subnormals,
                    // and +/-448. Avoid production decode and scale addressing.
                    let w: Vec<u8> = (0..n * k)
                        .map(|i| {
                            let code = (i * 73 + i / k) % 254;
                            (if code >= 127 { code + 1 } else { code }) as u8
                        })
                        .collect();
                    let (sb, sv) = scales(l);
                    let lda = k + 3;
                    let ldd = n + 5;
                    let a: Vec<f32> = (0..m * lda)
                        .map(|i| {
                            let ties = [
                                0x3f808000, 0x3f818000, 0xbf808000, 0xbf818000, 0x3f807fff,
                                0x3f808001, 0x00008000, 0x00008001,
                            ];
                            if i % 11 < ties.len() {
                                f32::from_bits(ties[i % 11])
                            } else {
                                (i as i32 % 37 - 18) as f32 * 0.03173
                            }
                        })
                        .collect();
                    let artifact = op.upload_numeric_fp8_linear(l, &w, &sb).unwrap();
                    let input = op.upload_f32_buffer(&a).unwrap();
                    let bfinput = op
                        .upload_bf16_words(&a.iter().copied().map(bf16_bits).collect::<Vec<_>>())
                        .unwrap();
                    let mut output = op.upload_f32_buffer(&vec![7331.; m * ldd + 7]).unwrap();
                    let mut bfoutput = op.upload_f32_buffer(&vec![7331.; m * ldd + 7]).unwrap();
                    let plan = NumericFp8LinearPlan::new(
                        l,
                        m,
                        0,
                        NumericFp8Precision::W8A16Bf16F32Accumulate,
                    )
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
                    assert_eq!(actual, bfactual, "two production input APIs");
                    op.stream_clone().context().bind_to_thread().unwrap();
                    let mut scalar = vec![7331.; m * ldd + 7];
                    let layout = ScalarLayout {
                        m: m as u64,
                        n: n as u64,
                        k: k as u64,
                        row_origin: row_origin as u64,
                        column_origin: column_origin as u64,
                        lda: lda as u64,
                        ldd: ldd as u64,
                        scale_bytes: ty.element_bytes() as u64,
                    };
                    #[allow(
                        unsafe_code,
                        reason = "exact host spans, validated layout, bound owner and completed CUDA work"
                    )]
                    let status = unsafe {
                        ferrule_test_w8a16_scalar(
                            &layout,
                            a.as_ptr(),
                            w.as_ptr(),
                            sb.as_ptr().cast(),
                            scalar.as_mut_ptr(),
                            &mut runtime_version,
                        )
                    };
                    assert_eq!(status, 0, "independent CUDA reference launch/copy failed");
                    let mut production_failures = 0;
                    let mut scalar_failures = 0;
                    let mut pair_unequal = 0;
                    let mut serial_unequal = 0;
                    let mut production_max_abs = 0.0f64;
                    let mut scalar_max_abs = 0.0f64;
                    let mut pair_max_abs = 0.0f64;
                    let mut production_max_fraction = 0.0f64;
                    let mut scalar_max_fraction = 0.0f64;
                    for row in 0..m {
                        for col in 0..n {
                            let (expected, norm, serial) = dot(l, &w, &sv, &a, lda, row, col);
                            let p = actual[row * ldd + col];
                            let s = scalar[row * ldd + col];
                            let pe = (f64::from(p) - expected).abs();
                            let se = (f64::from(s) - expected).abs();
                            production_failures +=
                                usize::from(!within_bound(f64::from(p), expected, norm));
                            scalar_failures +=
                                usize::from(!within_bound(f64::from(s), expected, norm));
                            pair_unequal += usize::from(p.to_bits() != s.to_bits());
                            serial_unequal += usize::from(s.to_bits() != serial.to_bits());
                            production_max_abs = production_max_abs.max(pe);
                            scalar_max_abs = scalar_max_abs.max(se);
                            pair_max_abs = pair_max_abs.max((f64::from(p) - f64::from(s)).abs());
                            production_max_fraction =
                                production_max_fraction.max(pe / (1e-5 + 2e-6 * norm));
                            scalar_max_fraction =
                                scalar_max_fraction.max(se / (1e-5 + 2e-6 * norm));
                        }
                        assert!(
                            actual[row * ldd + n..(row + 1) * ldd]
                                .iter()
                                .all(|v| *v == 7331.)
                        );
                        assert!(
                            scalar[row * ldd + n..(row + 1) * ldd]
                                .iter()
                                .all(|v| *v == 7331.)
                        );
                    }
                    assert!(actual[m * ldd..].iter().all(|v| *v == 7331.));
                    assert!(scalar[m * ldd..].iter().all(|v| *v == 7331.));
                    assert_eq!(op.download_f32_buffer(&input).unwrap(), a);
                    total_elements += m * n;
                    total_failures += production_failures + scalar_failures;
                    records.push(format!(r#"{{"m":{m},"n":{n},"k":{k},"row_origin":{row_origin},"column_origin":{column_origin},"scale_dtype":"{ty:?}","elements":{},"production_failures":{production_failures},"scalar_failures":{scalar_failures},"production_max_abs":{production_max_abs},"scalar_max_abs":{scalar_max_abs},"production_max_tolerance_fraction":{production_max_fraction},"scalar_max_tolerance_fraction":{scalar_max_fraction},"production_vs_scalar_max_abs":{pair_max_abs},"production_vs_scalar_bit_mismatches":{pair_unequal},"scalar_vs_host_serial_bit_mismatches":{serial_unequal}}}"#, m*n));
                }
            }
        }
    }
    let status = if total_failures == 0 {
        "passed"
    } else {
        "failed"
    };
    let report = format!(
        r#"{{"schema":"ferrule.w8a16-operator-triangle.v1","status":"{status}","scope":"synthetic operator matrix; not full40, routes, KV, GDN, task quality or vLLM equivalence","reference":"independent scalar CUDA E4M3FN/BF16-RNE serial F32 FMA + host F64","bound":{{"absolute":0.00001,"sum_abs_products_factor":0.000002}},"pairwise_acceptance":false,"runtime_version":{runtime_version},"cases":{},"elements":{total_elements},"failures":{total_failures},"canaries_and_input_unchanged":true,"production_input_api_equal":true,"observations":[{}]}}"#,
        records.len(),
        records.join(",\n")
    );
    if let Some(path) = std::env::var_os("FERRULE_W8A16_TRIANGLE_REPORT") {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap();
        writeln!(file, "{report}").unwrap();
    }
    eprintln!(
        "W8A16 triangle: {status}, {} cases, {total_elements} elements, {total_failures} violations, CUDA runtime {runtime_version}; pairwise bits diagnostic only",
        records.len()
    );
    assert_eq!(
        total_failures, 0,
        "operator bound violation: inspect triangle report before attributing to production"
    );
}

#[path = "captured.rs"]
mod captured;

#[path = "reduction/capital.rs"]
mod capital;
#[path = "reduction/wmma.rs"]
mod reduction;
