#![cfg(feature = "cuda")]

//! Short, ignored RTX 3090 microbenchmark for the explicit FP8 block-32 baseline.
//!
//! This is deliberately a backend test, not a production benchmark harness. It
//! measures the current two-stage E4M3/UE8M0 decode plus existing BF16 GEMM,
//! the explicit persistent decoded-BF16 path, and the existing BF16 GEMM facade
//! with a pre-decoded BF16 weight. Upload and initialization happen once per
//! shape and are not included in steady samples.

use ferrule_backend::cuda::operators::linear::{
    Bf16GemmLayout, CudaBf16Buffer, CudaDecodedFp8Block32Artifact, CudaF32Buffer,
    CudaFp8Block32ActivationWorkspace, CudaFp8Block32Artifact, CudaFp8Block32Workspace,
    CudaOperators, bf16_gemm_workspace_requirements, fp8_block32_activation_workspace_requirements,
    fp8_block32_workspace_requirements,
};
use ferrule_backend::operator_contract::{Fp8Block32ScaleLayout, Fp8E4m3Ue8m0LinearPlan};
use std::time::Instant;

const WARMUP_ITERS: usize = 8;
const MEASUREMENT_ITERS: usize = 32;
const SHAPES: &[(usize, usize, usize)] = &[(1, 4096, 4032), (32, 4096, 4032), (128, 4096, 4032)];

fn bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16
}

fn fp8_weight(plan: Fp8E4m3Ue8m0LinearPlan) -> Vec<u8> {
    (0..plan.weight_bytes())
        .map(|column| match column % 8 {
            0 => 0x38, // 1.0
            1 => 0x40, // 2.0
            2 => 0xb8, // -1.0
            3 => 0x30, // 0.5
            4 => 0x3c, // 1.5
            5 => 0x34, // 0.75
            6 => 0x44, // 3.0
            _ => 0xbc, // -1.5
        })
        .collect()
}

fn fp8_scales(plan: Fp8E4m3Ue8m0LinearPlan) -> Vec<u8> {
    vec![127; plan.scale_bytes()]
}

fn bf16_activation(m: usize, k: usize) -> Vec<u16> {
    let values = [0x3f80, 0x3f00, 0xbf80, 0x4000]; // 1.0, 0.5, -1.0, 2.0
    (0..m * k)
        .map(|index| values[index % values.len()])
        .collect()
}

fn percentile(samples: &[f32], percentage: usize) -> f32 {
    assert!(!samples.is_empty());
    assert!(percentage > 0 && percentage <= 100);
    let mut ordered = samples.to_vec();
    ordered.sort_unstable_by(f32::total_cmp);
    let rank = (ordered.len() * percentage).div_ceil(100).max(1) - 1;
    ordered[rank]
}

fn steady_samples(
    op: &CudaOperators,
    mut submit: impl FnMut() -> ferrule_common::Result<()>,
) -> Vec<f32> {
    for _ in 0..WARMUP_ITERS {
        submit().expect("warmup submission");
    }
    op.record_compute_event()
        .expect("record warmup completion")
        .synchronize()
        .expect("wait for warmup completion");

    let stream = op.stream_clone();
    let start = stream
        .context()
        .new_event(true)
        .expect("create CUDA timing start event");
    let end = stream
        .context()
        .new_event(true)
        .expect("create CUDA timing end event");
    let mut samples = Vec::with_capacity(MEASUREMENT_ITERS);
    for _ in 0..MEASUREMENT_ITERS {
        start.record(&stream).expect("record CUDA timing start");
        submit().expect("measurement submission");
        end.record(&stream).expect("record CUDA timing end");
        end.synchronize().expect("wait for CUDA timing end");
        samples.push(start.elapsed(&end).expect("read CUDA event elapsed time"));
    }
    samples
}

struct VariantRun {
    samples: Vec<f32>,
    launches: u64,
}

fn run_variant(
    op: &CudaOperators,
    expected_launches: u64,
    submit: impl FnMut() -> ferrule_common::Result<()>,
) -> VariantRun {
    let before = op.counters();
    let samples = steady_samples(op, submit);
    let after = op.counters();
    let calls = (WARMUP_ITERS + MEASUREMENT_ITERS) as u64;

    let total_kernel_launches = after.kernel_launches - before.kernel_launches;
    let total_compute_launches = after.compute_kernel_launches - before.compute_kernel_launches;
    assert_eq!(
        total_kernel_launches, total_compute_launches,
        "kernel/compute launch counter divergence"
    );
    assert_eq!(
        total_kernel_launches,
        calls * expected_launches,
        "unexpected total launch count"
    );
    assert_eq!(
        total_kernel_launches % calls,
        0,
        "launch count has a non-zero per-call remainder"
    );
    assert_eq!(
        total_compute_launches % calls,
        0,
        "compute launch count has a non-zero per-call remainder"
    );
    assert_eq!(
        total_compute_launches / calls,
        expected_launches,
        "unexpected average compute launches per call"
    );

    VariantRun {
        samples,
        launches: total_compute_launches / calls,
    }
}

struct OutputCheck {
    baseline_checksum: f64,
    cached_checksum: f64,
    reference_checksum: f64,
    max_abs_baseline_error: f32,
    max_abs_cached_error: f32,
    max_abs_cached_vs_baseline: f32,
    baseline_checksum_delta: f64,
    cached_checksum_delta: f64,
}

fn checksum(values: &[f32]) -> f64 {
    values.iter().map(|&value| f64::from(value)).sum()
}

fn max_abs_error(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max)
}

fn check_outputs(
    op: &CudaOperators,
    baseline_output: &CudaF32Buffer,
    cached_output: &CudaF32Buffer,
    reference_output: &CudaF32Buffer,
) -> OutputCheck {
    let baseline = op
        .download_f32_buffer(baseline_output)
        .expect("download baseline output");
    let cached = op
        .download_f32_buffer(cached_output)
        .expect("download cached output");
    let reference = op
        .download_f32_buffer(reference_output)
        .expect("download BF16 reference output");
    assert_eq!(
        baseline.len(),
        cached.len(),
        "baseline/cached output length mismatch"
    );
    assert_eq!(baseline.len(), reference.len(), "output length mismatch");
    for (label, values) in [
        ("baseline", &baseline),
        ("cached", &cached),
        ("BF16 reference", &reference),
    ] {
        assert!(
            values.iter().all(|value| value.is_finite()),
            "{label} output contains NaN or infinity"
        );
    }

    let baseline_checksum = checksum(&baseline);
    let cached_checksum = checksum(&cached);
    let reference_checksum = checksum(&reference);
    assert!(
        baseline_checksum.is_finite(),
        "baseline checksum is not finite"
    );
    assert!(cached_checksum.is_finite(), "cached checksum is not finite");
    assert!(
        reference_checksum.is_finite(),
        "BF16 reference checksum is not finite"
    );
    let max_abs_baseline_error = max_abs_error(&baseline, &reference);
    let max_abs_cached_error = max_abs_error(&cached, &reference);
    let max_abs_cached_vs_baseline = max_abs_error(&cached, &baseline);
    assert!(
        max_abs_baseline_error.is_finite(),
        "baseline/reference error is not finite"
    );
    assert!(
        max_abs_cached_error.is_finite(),
        "cached/reference error is not finite"
    );
    assert!(
        max_abs_cached_vs_baseline.is_finite(),
        "baseline/cached error is not finite"
    );
    assert!(
        max_abs_baseline_error <= 1e-3,
        "baseline/reference mismatch: max_abs_error={max_abs_baseline_error}"
    );
    assert!(
        max_abs_cached_error <= 1e-3,
        "cached/reference mismatch: max_abs_error={max_abs_cached_error}"
    );
    assert!(
        max_abs_cached_vs_baseline <= 1e-3,
        "baseline/cached mismatch: max_abs_error={max_abs_cached_vs_baseline}"
    );
    let baseline_checksum_delta = (baseline_checksum - reference_checksum).abs();
    let cached_checksum_delta = (cached_checksum - reference_checksum).abs();
    assert!(
        baseline_checksum_delta <= 1e-3,
        "baseline/reference checksum mismatch: checksum_delta={baseline_checksum_delta}"
    );
    assert!(
        cached_checksum_delta <= 1e-3,
        "cached/reference checksum mismatch: checksum_delta={cached_checksum_delta}"
    );

    OutputCheck {
        baseline_checksum,
        cached_checksum,
        reference_checksum,
        max_abs_baseline_error,
        max_abs_cached_error,
        max_abs_cached_vs_baseline,
        baseline_checksum_delta,
        cached_checksum_delta,
    }
}

struct OrderRun {
    baseline: VariantRun,
    cached: VariantRun,
    reference: VariantRun,
    output: OutputCheck,
}

#[derive(Clone, Copy)]
enum BenchmarkVariant {
    Uncached,
    Cached,
    Bf16,
}

#[allow(clippy::too_many_arguments)]
fn run_order(
    op: &CudaOperators,
    order: [BenchmarkVariant; 3],
    artifact: &CudaFp8Block32Artifact,
    decoded: &CudaDecodedFp8Block32Artifact,
    input: &CudaBf16Buffer,
    reference_weight: &CudaBf16Buffer,
    baseline_output: &mut CudaF32Buffer,
    cached_output: &mut CudaF32Buffer,
    reference_output: &mut CudaF32Buffer,
    workspace: &mut CudaFp8Block32Workspace,
    activation_workspace: &mut CudaFp8Block32ActivationWorkspace,
    plan: Fp8E4m3Ue8m0LinearPlan,
    reference_layout: Bf16GemmLayout,
) -> OrderRun {
    let mut baseline = None;
    let mut cached = None;
    let mut reference = None;
    for variant in order {
        match variant {
            BenchmarkVariant::Uncached => {
                baseline = Some(run_variant(op, 2, || {
                    op.fp8_block32_linear_into(artifact, input, baseline_output, workspace, plan)
                }));
            }
            BenchmarkVariant::Cached => {
                cached = Some(run_variant(op, 1, || {
                    op.fp8_block32_linear_from_decoded(
                        decoded,
                        input,
                        cached_output,
                        activation_workspace,
                        plan,
                    )
                }));
            }
            BenchmarkVariant::Bf16 => {
                reference = Some(run_variant(op, 1, || {
                    op.bf16_linear_into(reference_weight, input, reference_output, reference_layout)
                }));
            }
        }
    }
    let output = check_outputs(op, baseline_output, cached_output, reference_output);
    OrderRun {
        baseline: baseline.expect("benchmark order omitted uncached variant"),
        cached: cached.expect("benchmark order omitted cached variant"),
        reference: reference.expect("benchmark order omitted BF16 variant"),
        output,
    }
}

#[allow(clippy::too_many_arguments)]
fn report_line(
    order: &str,
    label: &str,
    m: usize,
    n: usize,
    k: usize,
    samples: &[f32],
    launches: u64,
    checksum: f64,
    orders: usize,
) {
    println!(
        "RESULT order={order} label={label} m={m} n={n} k={k} warmup_per_order={WARMUP_ITERS} orders={orders} measurements={measurements} launches={launches} p50_ms={:.6} p95_ms={:.6} checksum={checksum:.6}",
        percentile(samples, 50),
        percentile(samples, 95),
        measurements = samples.len(),
    );
}

fn report_order(order: &str, m: usize, n: usize, k: usize, run: &OrderRun) {
    report_line(
        order,
        "fp8_decode_plus_bf16_gemm",
        m,
        n,
        k,
        &run.baseline.samples,
        run.baseline.launches,
        run.output.baseline_checksum,
        1,
    );
    report_line(
        order,
        "fp8_cached_decoded_bf16_gemm",
        m,
        n,
        k,
        &run.cached.samples,
        run.cached.launches,
        run.output.cached_checksum,
        1,
    );
    report_line(
        order,
        "bf16_native_reference",
        m,
        n,
        k,
        &run.reference.samples,
        run.reference.launches,
        run.output.reference_checksum,
        1,
    );
    println!(
        "ERROR order={order} m={m} n={n} k={k} max_abs_baseline_vs_bf16_reference={:.8e} max_abs_cached_vs_bf16_reference={:.8e} max_abs_cached_vs_baseline={:.8e} checksum_delta_baseline={:.8e} checksum_delta_cached={:.8e}",
        run.output.max_abs_baseline_error,
        run.output.max_abs_cached_error,
        run.output.max_abs_cached_vs_baseline,
        run.output.baseline_checksum_delta,
        run.output.cached_checksum_delta,
    );
}

fn combined_samples(groups: &[&[f32]]) -> Vec<f32> {
    let mut combined = Vec::new();
    for group in groups {
        combined.extend_from_slice(group);
    }
    combined
}

fn average_order_metric(runs: &[&OrderRun], metric: fn(&OrderRun) -> f64) -> f64 {
    runs.iter().map(|run| metric(run)).sum::<f64>() / runs.len() as f64
}

#[test]
#[ignore = "RTX3090 SM86 only; CUDA12.4.99; short CUDA-event microbenchmark"]
fn fp8_block32_decode_bf16_gemm_baseline_vs_bf16_native() {
    let op = CudaOperators::new_on_device(0).expect("required CUDA device 0");
    let stream = op.stream_clone();
    println!(
        "CONFIG baseline=Ferrule fp8-e4m3-ue8m0-block32-decode-bf16-gemm-baseline cached=explicit-persistent-decoded-bf16 reference=existing-BF16-native device_ordinal={} warmup={} measurements_per_order={} shapes={:?}",
        op.device_ordinal(),
        WARMUP_ITERS,
        MEASUREMENT_ITERS,
        SHAPES,
    );
    println!(
        "CONFIG comparison=Ferrule uncached baseline vs explicit cached decoded BF16 vs BF16 native; no DeepInfer result; timing=CUDA events; sync=event completion only; three fixed permutations; estimate=3-permutation order-balanced estimate"
    );

    for &(m, n, k) in SHAPES {
        let layout = Fp8Block32ScaleLayout::contiguous(n, k, 0, 0)
            .expect("construct contiguous FP8 block-32 layout");
        let plan =
            Fp8E4m3Ue8m0LinearPlan::new(m, n, k, layout).expect("construct FP8 block-32 plan");
        let encoded_weight = fp8_weight(plan);
        let scales = fp8_scales(plan);
        let activation_host = bf16_activation(m, k);
        let decoded_weight = plan
            .reference_dequantize(&encoded_weight, &scales)
            .expect("CPU-decode FP8 weight for BF16 reference")
            .into_iter()
            .map(bf16_rne)
            .collect::<Vec<_>>();
        let workspace_requirements = fp8_block32_workspace_requirements(plan)
            .expect("query FP8 block-32 workspace requirements");
        let reference_layout = Bf16GemmLayout::contiguous(m, n, k);
        let reference_workspace_bytes = bf16_gemm_workspace_requirements(reference_layout)
            .expect("query BF16 reference workspace requirements")
            .bytes;

        let init_start = stream
            .context()
            .new_event(true)
            .expect("create initialization start event");
        init_start
            .record(&stream)
            .expect("record initialization start");
        let enqueue_host_start = Instant::now();
        let artifact = op
            .upload_fp8_e4m3_ue8m0_block32(plan, &encoded_weight, &scales)
            .expect("upload FP8 baseline artifact");
        let cached_source = op
            .upload_fp8_e4m3_ue8m0_block32(plan, &encoded_weight, &scales)
            .expect("upload FP8 cached source artifact");
        let input = op
            .upload_bf16_words(&activation_host)
            .expect("upload BF16 activation");
        let reference_weight = op
            .upload_bf16_words(&decoded_weight)
            .expect("upload decoded BF16 reference weight");
        let mut baseline_output = op.zero_f32_buffer(m * n).expect("allocate baseline output");
        let mut cached_output = op.zero_f32_buffer(m * n).expect("allocate cached output");
        let mut reference_output = op
            .zero_f32_buffer(m * n)
            .expect("allocate BF16 reference output");
        let mut workspace = op
            .fp8_block32_workspace(plan)
            .expect("allocate FP8 block-32 workspace");
        let mut activation_workspace = op
            .fp8_block32_activation_workspace(plan)
            .expect("allocate cached activation workspace");
        let enqueue_host_ms = enqueue_host_start.elapsed().as_secs_f64() * 1_000.0;
        let init_end = stream
            .context()
            .new_event(true)
            .expect("create initialization end event");
        init_end.record(&stream).expect("record initialization end");
        init_end
            .synchronize()
            .expect("wait for upload and initialization");
        let init_device_ms = init_start
            .elapsed(&init_end)
            .expect("measure upload and initialization device time");

        let cache_prepare_start = stream
            .context()
            .new_event(true)
            .expect("create cache prepare start event");
        cache_prepare_start
            .record(&stream)
            .expect("record cache prepare start");
        let cache_prepare_host_start = Instant::now();
        let decoded = op
            .prepare_fp8_block32_decoded_weight(cached_source, plan)
            .expect("prepare persistent decoded weight");
        let cache_prepare_enqueue_host_ms =
            cache_prepare_host_start.elapsed().as_secs_f64() * 1_000.0;
        let cache_prepare_end = stream
            .context()
            .new_event(true)
            .expect("create cache prepare end event");
        cache_prepare_end
            .record(&stream)
            .expect("record cache prepare end");
        cache_prepare_end
            .synchronize()
            .expect("wait for cache preparation");
        let cache_prepare_device_ms = cache_prepare_start
            .elapsed(&cache_prepare_end)
            .expect("measure cache preparation device time");

        let activation_workspace_requirements = fp8_block32_activation_workspace_requirements(plan)
            .expect("query cached activation workspace requirements");
        assert_eq!(
            workspace.workspace_bytes(),
            workspace_requirements.bytes as usize
        );
        assert_eq!(
            activation_workspace.workspace_bytes(),
            activation_workspace_requirements.bytes as usize
        );
        println!(
            "INIT m={m} n={n} k={k} init_device_ms={init_device_ms:.6} enqueue_host_ms={enqueue_host_ms:.3} artifact_bytes={} fp8_workspace_bytes={} cached_activation_workspace_bytes={} bf16_reference_workspace_bytes={} decoded_weight_bytes={} activation_pad_bytes={} activation_pad_launches={}",
            plan.weight_bytes() + plan.scale_bytes(),
            workspace.workspace_bytes(),
            activation_workspace.workspace_bytes(),
            reference_workspace_bytes,
            workspace.decoded_weight_bytes(),
            workspace.activation_padding_bytes(),
            u64::from(workspace.activation_padding_bytes() != 0),
        );
        println!(
            "CACHE_PREPARE m={m} n={n} k={k} prepare_device_ms={cache_prepare_device_ms:.6} enqueue_host_ms={cache_prepare_enqueue_host_ms:.3} decoded_weight_bytes={} activation_workspace_bytes={} source_artifact_bytes={}",
            decoded.decoded_weight_bytes(),
            activation_workspace.workspace_bytes(),
            plan.weight_bytes() + plan.scale_bytes(),
        );

        let uncached_cached_bf16 = run_order(
            &op,
            [
                BenchmarkVariant::Uncached,
                BenchmarkVariant::Cached,
                BenchmarkVariant::Bf16,
            ],
            &artifact,
            &decoded,
            &input,
            &reference_weight,
            &mut baseline_output,
            &mut cached_output,
            &mut reference_output,
            &mut workspace,
            &mut activation_workspace,
            plan,
            reference_layout,
        );
        report_order(
            "uncached_then_cached_then_bf16",
            m,
            n,
            k,
            &uncached_cached_bf16,
        );

        let cached_bf16_uncached = run_order(
            &op,
            [
                BenchmarkVariant::Cached,
                BenchmarkVariant::Bf16,
                BenchmarkVariant::Uncached,
            ],
            &artifact,
            &decoded,
            &input,
            &reference_weight,
            &mut baseline_output,
            &mut cached_output,
            &mut reference_output,
            &mut workspace,
            &mut activation_workspace,
            plan,
            reference_layout,
        );
        report_order(
            "cached_then_bf16_then_uncached",
            m,
            n,
            k,
            &cached_bf16_uncached,
        );

        let bf16_uncached_cached = run_order(
            &op,
            [
                BenchmarkVariant::Bf16,
                BenchmarkVariant::Uncached,
                BenchmarkVariant::Cached,
            ],
            &artifact,
            &decoded,
            &input,
            &reference_weight,
            &mut baseline_output,
            &mut cached_output,
            &mut reference_output,
            &mut workspace,
            &mut activation_workspace,
            plan,
            reference_layout,
        );
        report_order(
            "bf16_then_uncached_then_cached",
            m,
            n,
            k,
            &bf16_uncached_cached,
        );

        let runs = [
            &uncached_cached_bf16,
            &cached_bf16_uncached,
            &bf16_uncached_cached,
        ];
        let baseline_samples = combined_samples(&[
            &runs[0].baseline.samples,
            &runs[1].baseline.samples,
            &runs[2].baseline.samples,
        ]);
        let cached_samples = combined_samples(&[
            &runs[0].cached.samples,
            &runs[1].cached.samples,
            &runs[2].cached.samples,
        ]);
        let reference_samples = combined_samples(&[
            &runs[0].reference.samples,
            &runs[1].reference.samples,
            &runs[2].reference.samples,
        ]);
        let baseline_checksum = average_order_metric(&runs, |run| run.output.baseline_checksum);
        let cached_checksum = average_order_metric(&runs, |run| run.output.cached_checksum);
        let reference_checksum = average_order_metric(&runs, |run| run.output.reference_checksum);
        let max_abs_baseline_error = runs
            .iter()
            .map(|run| run.output.max_abs_baseline_error)
            .fold(0.0, f32::max);
        let max_abs_cached_error = runs
            .iter()
            .map(|run| run.output.max_abs_cached_error)
            .fold(0.0, f32::max);
        let max_abs_cached_vs_baseline = runs
            .iter()
            .map(|run| run.output.max_abs_cached_vs_baseline)
            .fold(0.0, f32::max);
        let baseline_checksum_delta = runs
            .iter()
            .map(|run| run.output.baseline_checksum_delta)
            .fold(0.0, f64::max);
        let cached_checksum_delta = runs
            .iter()
            .map(|run| run.output.cached_checksum_delta)
            .fold(0.0, f64::max);
        for (label, value) in [
            ("baseline", baseline_checksum),
            ("cached", cached_checksum),
            ("BF16 reference", reference_checksum),
            (
                "baseline/reference error",
                f64::from(max_abs_baseline_error),
            ),
            ("cached/reference error", f64::from(max_abs_cached_error)),
            (
                "cached/baseline error",
                f64::from(max_abs_cached_vs_baseline),
            ),
        ] {
            assert!(value.is_finite(), "order-neutral {label} is not finite");
        }
        report_line(
            "three_fixed_permutations_order_balanced_estimate",
            "fp8_decode_plus_bf16_gemm",
            m,
            n,
            k,
            &baseline_samples,
            runs[0].baseline.launches,
            baseline_checksum,
            3,
        );
        report_line(
            "three_fixed_permutations_order_balanced_estimate",
            "fp8_cached_decoded_bf16_gemm",
            m,
            n,
            k,
            &cached_samples,
            runs[0].cached.launches,
            cached_checksum,
            3,
        );
        report_line(
            "three_fixed_permutations_order_balanced_estimate",
            "bf16_native_reference",
            m,
            n,
            k,
            &reference_samples,
            runs[0].reference.launches,
            reference_checksum,
            3,
        );
        println!(
            "ORDER_NEUTRAL_ERROR_3_PERMUTATIONS m={m} n={n} k={k} max_abs_baseline_vs_bf16_reference={max_abs_baseline_error:.8e} max_abs_cached_vs_bf16_reference={max_abs_cached_error:.8e} max_abs_cached_vs_baseline={max_abs_cached_vs_baseline:.8e} checksum_delta_baseline={baseline_checksum_delta:.8e} checksum_delta_cached={cached_checksum_delta:.8e}"
        );
    }
}
