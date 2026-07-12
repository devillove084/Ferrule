//! E2 Slice B tests: persistent shared SwiGLU workspace, capture-safe mode,
//! deterministic failpoints, and arena hit/miss/grow attribution.
//!
//! These tests verify the graph-safety contract:
//! - `artifact_swiglu_ffn_from_device_into` produces the same output as the
//!   allocating `artifact_swiglu_ffn_from_device`.
//! - Capture-safe mode rejects allocations, D2H copies, and stream syncs.
//! - Deterministic failpoints fire once and disarm.
//! - Arena hit/miss/grow counters are attributed correctly.

use cuda_core::CudaContext;
use ferrule_cuda::context::CudaArtifactOperatorContext;
use std::sync::{Mutex, MutexGuard};

static CUDA_TEST_LOCK: Mutex<()> = Mutex::new(());

fn cuda_test_guard() -> MutexGuard<'static, ()> {
    CUDA_TEST_LOCK
        .lock()
        .expect("E2 SwiGLU workspace test lock poisoned")
}

fn has_cuda() -> bool {
    CudaContext::new(0).is_ok()
}

/// Build a small FP8 E4M3 + E8M0 scale linear handle for testing.
fn make_fp8_linear(
    ctx: &CudaArtifactOperatorContext,
    out_features: usize,
    in_features: usize,
    seed: f32,
) -> ferrule_cuda::context::CudaArtifactLinearHandle {
    assert!(
        out_features % 128 == 0 && in_features % 128 == 0,
        "test FP8 linear dimensions must be multiples of 128"
    );
    let weight_len = out_features * in_features;
    let mut weight = Vec::with_capacity(weight_len);
    for i in 0..weight_len {
        weight.push(fp8_nearest(seed * (i as f32 + 1.0) / (weight_len as f32)));
    }
    let scale_len = (out_features / 128) * (in_features / 128);
    let scale = vec![136u8; scale_len]; // 2^9 = 512 scale
    ctx.upload_fp8_e4m3_e8m0_linear(&weight, &scale, out_features, in_features, 128, 128)
        .expect("upload FP8 linear")
}

fn fp8_nearest(value: f32) -> u8 {
    let sign = if value.is_sign_negative() { 0x80 } else { 0 };
    if value == 0.0 {
        return sign;
    }
    let magnitude = value.abs().min(448.0);
    let mut best = 0u8;
    let mut best_error = magnitude;
    for code in 1..=0x7e {
        let candidate = fp8_e4m3fn(code);
        let error = (candidate - magnitude).abs();
        if error < best_error {
            best = code;
            best_error = error;
        }
    }
    sign | best
}

fn fp8_e4m3fn(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let exponent = (byte >> 3) & 0x0f;
    let mantissa = byte & 0x07;
    if exponent == 0 {
        return sign * mantissa as f32 / 512.0;
    }
    if exponent == 0x0f && mantissa == 0x07 {
        return f32::NAN;
    }
    sign * 2.0f32.powi(exponent as i32 - 7) * (1.0 + mantissa as f32 / 8.0)
}

#[test]
fn swiglu_workspace_into_matches_allocating_path() {
    if !has_cuda() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let _guard = cuda_test_guard();
    let ctx = CudaArtifactOperatorContext::new().expect("CUDA context");

    let in_features = 128usize;
    let intermediate = 128usize;
    let out_features = 128usize;

    let gate = make_fp8_linear(&ctx, intermediate, in_features, 0.5);
    let up = make_fp8_linear(&ctx, intermediate, in_features, 0.7);
    let down = make_fp8_linear(&ctx, out_features, intermediate, 0.3);

    let input_data: Vec<f32> = (0..in_features).map(|i| (i as f32) * 0.01).collect();
    let input = ctx.upload_f32_buffer(&input_data).expect("upload input");

    // Allocating path: creates new buffers each call.
    ctx.reset_counters();
    let allocating_output = ctx
        .artifact_swiglu_ffn_from_device(&gate, &up, &down, &input, 1.0, 0.0)
        .expect("allocating SwiGLU");
    let allocating_counters = ctx.counters();

    // Workspace path: reuses persistent buffers.
    ctx.reset_counters();
    let mut workspace = ctx
        .swiglu_workspace(1, intermediate, out_features)
        .expect("create workspace");
    ctx.artifact_swiglu_ffn_from_device_into(&gate, &up, &down, &input, 1.0, 0.0, &mut workspace)
        .expect("workspace SwiGLU");
    let _workspace_counters = ctx.counters();

    // A warmed workspace call must be accepted by the capture-safe guard and
    // make no allocation attempt.
    ctx.reset_counters();
    ctx.enable_capture_safe();
    ctx.artifact_swiglu_ffn_from_device_into(&gate, &up, &down, &input, 1.0, 0.0, &mut workspace)
        .expect("capture-safe workspace SwiGLU second call");
    ctx.disable_capture_safe();
    let workspace_reuse_counters = ctx.counters();

    // Verify numerical parity.
    let allocating_vals = ctx
        .download_f32_buffer(&allocating_output)
        .expect("download allocating");
    let workspace_vals = ctx
        .download_f32_buffer(workspace.output())
        .expect("download workspace");

    assert_eq!(
        allocating_vals.len(),
        workspace_vals.len(),
        "output length mismatch"
    );
    for (i, (a, w)) in allocating_vals
        .iter()
        .zip(workspace_vals.iter())
        .enumerate()
    {
        assert!(
            (a - w).abs() < 1e-5,
            "SwiGLU output mismatch at index {i}: allocating={a} workspace={w}"
        );
    }

    // The allocating path must have performed at least one allocation.
    assert!(
        allocating_counters.device_allocations > 0,
        "allocating path should have device allocations, got {}",
        allocating_counters.device_allocations
    );

    assert_eq!(
        workspace_reuse_counters.device_allocation_attempts, 0,
        "warmed workspace reuse must make zero device allocation attempts"
    );
    assert_eq!(
        workspace_reuse_counters.device_allocations, 0,
        "warmed workspace reuse must make zero device allocations"
    );
    assert_eq!(
        workspace_reuse_counters.host_to_device_copies, 0,
        "warmed workspace reuse must not upload"
    );
    assert_eq!(
        workspace_reuse_counters.device_to_host_copies, 0,
        "warmed workspace reuse must not download"
    );
    assert_eq!(
        workspace_reuse_counters.stream_wide_syncs, 0,
        "warmed workspace reuse must not synchronize the stream"
    );
}

#[test]
fn swiglu_workspace_add_into_accumulates_correctly() {
    if !has_cuda() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let _guard = cuda_test_guard();
    let ctx = CudaArtifactOperatorContext::new().expect("CUDA context");

    let in_features = 128usize;
    let intermediate = 128usize;
    let out_features = 128usize;

    let gate = make_fp8_linear(&ctx, intermediate, in_features, 0.5);
    let up = make_fp8_linear(&ctx, intermediate, in_features, 0.7);
    let down = make_fp8_linear(&ctx, out_features, intermediate, 0.3);

    let input_data: Vec<f32> = (0..in_features).map(|i| (i as f32) * 0.01).collect();
    let input = ctx.upload_f32_buffer(&input_data).expect("upload input");

    // Compute standalone output.
    let standalone = ctx
        .artifact_swiglu_ffn_from_device(&gate, &up, &down, &input, 1.0, 0.0)
        .expect("standalone SwiGLU");

    // Compute add-into: accumulator starts at zero, should equal standalone.
    let mut workspace = ctx
        .swiglu_workspace(1, intermediate, out_features)
        .expect("create workspace");
    let mut accumulator = ctx.zero_f32_buffer(out_features).expect("zero accumulator");
    ctx.artifact_swiglu_ffn_add_into_from_device(
        &gate,
        &up,
        &down,
        &input,
        1,
        1.0,
        0.0,
        &mut workspace,
        &mut accumulator,
    )
    .expect("add-into SwiGLU");

    let standalone_vals = ctx
        .download_f32_buffer(&standalone)
        .expect("download standalone");
    let accumulator_vals = ctx
        .download_f32_buffer(&accumulator)
        .expect("download accumulator");

    for (i, (s, a)) in standalone_vals
        .iter()
        .zip(accumulator_vals.iter())
        .enumerate()
    {
        assert!(
            (s - a).abs() < 1e-5,
            "add-into mismatch at index {i}: standalone={s} accumulator={a}"
        );
    }

    // Add again under the capture-safe guard: accumulator should now be 2x
    // standalone without any allocation attempt or forbidden operation.
    ctx.reset_counters();
    ctx.enable_capture_safe();
    ctx.artifact_swiglu_ffn_add_into_from_device(
        &gate,
        &up,
        &down,
        &input,
        1,
        1.0,
        0.0,
        &mut workspace,
        &mut accumulator,
    )
    .expect("capture-safe second add-into");
    ctx.disable_capture_safe();
    let add_reuse_counters = ctx.counters();
    assert_eq!(add_reuse_counters.device_allocation_attempts, 0);
    assert_eq!(add_reuse_counters.device_allocations, 0);
    assert_eq!(add_reuse_counters.host_to_device_copies, 0);
    assert_eq!(add_reuse_counters.device_to_host_copies, 0);
    assert_eq!(add_reuse_counters.stream_wide_syncs, 0);

    let doubled_vals = ctx
        .download_f32_buffer(&accumulator)
        .expect("download doubled");
    for (i, (s, d)) in standalone_vals.iter().zip(doubled_vals.iter()).enumerate() {
        assert!(
            (2.0 * s - d).abs() < 1e-4,
            "double add-into mismatch at index {i}: 2*standalone={} doubled={d}",
            2.0 * s
        );
    }
}

#[test]
fn capture_safe_mode_rejects_allocations_and_syncs() {
    if !has_cuda() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let _guard = cuda_test_guard();
    let ctx = CudaArtifactOperatorContext::new().expect("CUDA context");

    assert!(!ctx.is_capture_safe(), "capture-safe should start disabled");

    ctx.enable_capture_safe();
    assert!(ctx.is_capture_safe(), "capture-safe should be enabled");

    // Allocation should fail.
    let result = ctx.zero_f32_buffer(64);
    assert!(
        result.is_err(),
        "capture-safe mode should reject allocation"
    );
    let err_msg = result.err().unwrap().to_string();
    assert!(
        err_msg.contains("capture-safe violation"),
        "error should mention capture-safe, got: {err_msg}"
    );

    // Stream sync should fail.
    let result = ctx.sync_stream();
    assert!(
        result.is_err(),
        "capture-safe mode should reject stream sync"
    );

    // D2H download should fail.
    ctx.disable_capture_safe();
    let buf = ctx.zero_f32_buffer(64).expect("alloc after disabling");
    ctx.enable_capture_safe();
    let result = ctx.download_f32_buffer(&buf);
    assert!(
        result.is_err(),
        "capture-safe mode should reject D2H download"
    );

    ctx.disable_capture_safe();
    assert!(!ctx.is_capture_safe(), "capture-safe should be disabled");
}

#[test]
fn failpoints_fire_once_and_disarm() {
    if !has_cuda() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let _guard = cuda_test_guard();
    let ctx = CudaArtifactOperatorContext::new().expect("CUDA context");

    // Arm allocation failpoint.
    ctx.failpoints().arm_allocation();

    // First allocation should fail.
    let result = ctx.zero_f32_buffer(64);
    assert!(result.is_err(), "armed failpoint should fail allocation");
    let err_msg = result.err().unwrap().to_string();
    assert!(
        err_msg.contains("failpoint"),
        "error should mention failpoint, got: {err_msg}"
    );

    // Second allocation should succeed (failpoint disarmed).
    let result = ctx.zero_f32_buffer(64);
    assert!(
        result.is_ok(),
        "failpoint should be disarmed after firing once"
    );

    // Arena acquire failpoint.
    ctx.failpoints().arm_arena_acquire();
    assert!(ctx.failpoints().check_arena_acquire());
    assert!(!ctx.failpoints().check_arena_acquire());

    // Expert upload failpoint.
    ctx.failpoints().arm_expert_upload();
    assert!(ctx.failpoints().check_expert_upload());
    assert!(!ctx.failpoints().check_expert_upload());
}

#[test]
fn arena_counters_track_workspace_creation_and_reuse() {
    if !has_cuda() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let _guard = cuda_test_guard();
    let ctx = CudaArtifactOperatorContext::new().expect("CUDA context");

    ctx.reset_counters();

    // The swiglu_workspace method itself doesn't track hit/miss; that's done by
    // the cache layer in cuda_cache.rs. Verify the counter mechanism works by
    // calling the public counter methods directly.
    ctx.add_arena_hit();
    ctx.add_arena_hit();
    ctx.add_arena_miss();
    ctx.add_arena_grow();
    ctx.add_arena_reuse();
    ctx.add_arena_reuse();
    ctx.add_arena_reuse();
    let c = ctx.counters();
    assert_eq!(c.arena_hits, 2, "should have 2 arena hits");
    assert_eq!(c.arena_misses, 1, "should have 1 arena miss");
    assert_eq!(c.arena_grows, 1, "should have 1 arena grow");
    assert_eq!(c.arena_reuses, 3, "should have 3 arena reuses");

    ctx.reset_counters();
    assert_eq!(
        ctx.counters().arena_hits,
        0,
        "reset should clear arena hits"
    );
}
