#![cfg(feature = "cuda")]

//! Short, ignored RTX 3090 microbenchmark for the explicit FP8 block-32 baseline.
//!
//! This is deliberately a backend test, not a production benchmark harness. It
//! measures the current two-stage E4M3/UE8M0 decode plus existing BF16 GEMM
//! against the existing BF16 GEMM facade with a pre-decoded BF16 weight. Upload
//! and initialization happen once per shape and are not included in steady
//! samples.

use ferrule_backend::cuda::operators::linear::{
    Bf16GemmLayout, CudaBf16Buffer, CudaF32Buffer, CudaFp8Block32Artifact, CudaFp8Block32Workspace,
    CudaOperators, bf16_gemm_workspace_requirements, fp8_block32_workspace_requirements,
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
    reference_checksum: f64,
    max_abs_error: f32,
    checksum_delta: f64,
}

fn check_outputs(
    op: &CudaOperators,
    baseline_output: &CudaF32Buffer,
    reference_output: &CudaF32Buffer,
) -> OutputCheck {
    let baseline = op
        .download_f32_buffer(baseline_output)
        .expect("download baseline output");
    let reference = op
        .download_f32_buffer(reference_output)
        .expect("download BF16 reference output");
    assert_eq!(baseline.len(), reference.len(), "output length mismatch");
    assert!(
        baseline.iter().all(|value| value.is_finite()),
        "baseline output contains NaN or infinity"
    );
    assert!(
        reference.iter().all(|value| value.is_finite()),
        "BF16 reference output contains NaN or infinity"
    );

    let baseline_checksum: f64 = baseline.iter().map(|&value| f64::from(value)).sum();
    let reference_checksum: f64 = reference.iter().map(|&value| f64::from(value)).sum();
    assert!(
        baseline_checksum.is_finite(),
        "baseline checksum is not finite"
    );
    assert!(
        reference_checksum.is_finite(),
        "BF16 reference checksum is not finite"
    );
    let max_abs_error = baseline
        .iter()
        .zip(&reference)
        .map(|(baseline, reference)| (baseline - reference).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs_error.is_finite(),
        "baseline/reference error is not finite"
    );
    assert!(
        max_abs_error <= 1e-3,
        "baseline/reference mismatch: max_abs_error={max_abs_error}"
    );
    let checksum_delta = (baseline_checksum - reference_checksum).abs();
    assert!(checksum_delta.is_finite(), "checksum delta is not finite");
    assert!(
        checksum_delta <= 1e-3,
        "baseline/reference checksum mismatch: checksum_delta={checksum_delta}"
    );

    OutputCheck {
        baseline_checksum,
        reference_checksum,
        max_abs_error,
        checksum_delta,
    }
}

struct OrderRun {
    baseline: VariantRun,
    reference: VariantRun,
    output: OutputCheck,
}

fn run_order(
    op: &CudaOperators,
    baseline_first: bool,
    artifact: &CudaFp8Block32Artifact,
    input: &CudaBf16Buffer,
    reference_weight: &CudaBf16Buffer,
    baseline_output: &mut CudaF32Buffer,
    reference_output: &mut CudaF32Buffer,
    workspace: &mut CudaFp8Block32Workspace,
    plan: Fp8E4m3Ue8m0LinearPlan,
    reference_layout: Bf16GemmLayout,
) -> OrderRun {
    let (baseline, reference) = if baseline_first {
        let baseline = run_variant(op, 2, || {
            op.fp8_block32_linear_into(artifact, input, baseline_output, workspace, plan)
        });
        let reference = run_variant(op, 1, || {
            op.bf16_linear_into(reference_weight, input, reference_output, reference_layout)
        });
        (baseline, reference)
    } else {
        let reference = run_variant(op, 1, || {
            op.bf16_linear_into(reference_weight, input, reference_output, reference_layout)
        });
        let baseline = run_variant(op, 2, || {
            op.fp8_block32_linear_into(artifact, input, baseline_output, workspace, plan)
        });
        (baseline, reference)
    };
    let output = check_outputs(op, baseline_output, reference_output);
    OrderRun {
        baseline,
        reference,
        output,
    }
}

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
        "ERROR order={order} m={m} n={n} k={k} max_abs_baseline_vs_bf16_reference={:.8e} checksum_delta={:.8e}",
        run.output.max_abs_error, run.output.checksum_delta,
    );
}

fn combined_samples(first: &[f32], second: &[f32]) -> Vec<f32> {
    let mut combined = Vec::with_capacity(first.len() + second.len());
    combined.extend_from_slice(first);
    combined.extend_from_slice(second);
    combined
}

#[test]
#[ignore = "RTX3090 SM86 only; CUDA12.4.99; short CUDA-event microbenchmark"]
fn fp8_block32_decode_bf16_gemm_baseline_vs_bf16_native() {
    let op = CudaOperators::new_on_device(0).expect("required CUDA device 0");
    let stream = op.stream_clone();
    println!(
        "CONFIG baseline=Ferrule fp8-e4m3-ue8m0-block32-decode-bf16-gemm-baseline reference=existing BF16 native only device_ordinal={} warmup={} measurements_per_order={} shapes={:?}",
        op.device_ordinal(),
        WARMUP_ITERS,
        MEASUREMENT_ITERS,
        SHAPES,
    );
    println!(
        "CONFIG comparison=Ferrule baseline vs BF16 native only; no DeepInfer result; timing=CUDA events; sync=event completion only; estimate=order-neutral aggregate of both orders"
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
            .expect("upload FP8 artifact");
        let input = op
            .upload_bf16_words(&activation_host)
            .expect("upload BF16 activation");
        let reference_weight = op
            .upload_bf16_words(&decoded_weight)
            .expect("upload decoded BF16 reference weight");
        let mut baseline_output = op.zero_f32_buffer(m * n).expect("allocate baseline output");
        let mut reference_output = op
            .zero_f32_buffer(m * n)
            .expect("allocate BF16 reference output");
        let mut workspace = op
            .fp8_block32_workspace(plan)
            .expect("allocate FP8 block-32 workspace");
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

        assert_eq!(
            workspace.workspace_bytes(),
            workspace_requirements.bytes as usize
        );
        println!(
            "INIT m={m} n={n} k={k} init_device_ms={init_device_ms:.6} enqueue_host_ms={enqueue_host_ms:.3} artifact_bytes={} fp8_workspace_bytes={} bf16_reference_workspace_bytes={} decoded_weight_bytes={} activation_pad_bytes={} activation_pad_launches={}",
            plan.weight_bytes() + plan.scale_bytes(),
            workspace.workspace_bytes(),
            reference_workspace_bytes,
            workspace.decoded_weight_bytes(),
            workspace.activation_padding_bytes(),
            u64::from(workspace.activation_padding_bytes() != 0),
        );

        let baseline_first = run_order(
            &op,
            true,
            &artifact,
            &input,
            &reference_weight,
            &mut baseline_output,
            &mut reference_output,
            &mut workspace,
            plan,
            reference_layout,
        );
        report_order("baseline_then_bf16", m, n, k, &baseline_first);

        let bf16_first = run_order(
            &op,
            false,
            &artifact,
            &input,
            &reference_weight,
            &mut baseline_output,
            &mut reference_output,
            &mut workspace,
            plan,
            reference_layout,
        );
        report_order("bf16_then_baseline", m, n, k, &bf16_first);

        let baseline_samples = combined_samples(
            &baseline_first.baseline.samples,
            &bf16_first.baseline.samples,
        );
        let reference_samples = combined_samples(
            &baseline_first.reference.samples,
            &bf16_first.reference.samples,
        );
        let baseline_checksum =
            (baseline_first.output.baseline_checksum + bf16_first.output.baseline_checksum) / 2.0;
        let reference_checksum =
            (baseline_first.output.reference_checksum + bf16_first.output.reference_checksum) / 2.0;
        let max_abs_error = baseline_first
            .output
            .max_abs_error
            .max(bf16_first.output.max_abs_error);
        let checksum_delta = baseline_first
            .output
            .checksum_delta
            .max(bf16_first.output.checksum_delta);
        assert!(
            baseline_checksum.is_finite(),
            "order-neutral baseline checksum is not finite"
        );
        assert!(
            reference_checksum.is_finite(),
            "order-neutral BF16 checksum is not finite"
        );
        assert!(
            max_abs_error.is_finite(),
            "order-neutral error is not finite"
        );
        assert!(
            checksum_delta.is_finite(),
            "order-neutral checksum delta is not finite"
        );
        report_line(
            "order_neutral_estimate",
            "fp8_decode_plus_bf16_gemm",
            m,
            n,
            k,
            &baseline_samples,
            baseline_first.baseline.launches,
            baseline_checksum,
            2,
        );
        report_line(
            "order_neutral_estimate",
            "bf16_native_reference",
            m,
            n,
            k,
            &reference_samples,
            baseline_first.reference.launches,
            reference_checksum,
            2,
        );
        println!(
            "ORDER_NEUTRAL_ERROR m={m} n={n} k={k} max_abs_baseline_vs_bf16_reference={max_abs_error:.8e} checksum_delta={checksum_delta:.8e}"
        );
    }
}
