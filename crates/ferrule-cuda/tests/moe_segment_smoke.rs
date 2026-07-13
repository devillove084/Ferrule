//! CUDA smoke coverage for route-major MoE segment outputs.

use cuda_core::CudaContext;
use ferrule_cuda::context::{CudaArtifactLinearShape, CudaArtifactOperatorContext};
use std::sync::{Mutex, MutexGuard};

static CUDA_TEST_LOCK: Mutex<()> = Mutex::new(());

fn cuda_test_guard() -> MutexGuard<'static, ()> {
    CUDA_TEST_LOCK
        .lock()
        .expect("CUDA smoke test lock poisoned")
}

fn has_cuda() -> bool {
    CudaContext::new(0).is_ok()
}

fn assert_close_slice(actual: &[f32], expected: &[f32], tolerance: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label}[{index}]: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn route_ranked_reducer_preserves_prefix_and_uses_token_major_routes() {
    let _guard = cuda_test_guard();
    if !has_cuda() {
        eprintln!("skipping: no CUDA device");
        return;
    }

    const TOKENS: usize = 3;
    const ROUTES_PER_TOKEN: usize = 4;
    const HIDDEN_SIZE: usize = 5;

    let context = CudaArtifactOperatorContext::new().expect("CUDA artifact context");
    let allocated = context
        .allocate_moe_route_output(TOKENS, ROUTES_PER_TOKEN, HIDDEN_SIZE)
        .expect("allocate route output");
    assert_eq!(allocated.len(), TOKENS * ROUTES_PER_TOKEN * HIDDEN_SIZE);
    drop(allocated);

    let route_count = TOKENS * ROUTES_PER_TOKEN;
    let mut routes = vec![0.0f32; route_count * HIDDEN_SIZE];
    for route in 0..route_count {
        for row in 0..HIDDEN_SIZE {
            let magnitude = (route * 16 + row + 1) as f32;
            routes[route * HIDDEN_SIZE + row] = if route % 2 == 0 {
                magnitude
            } else {
                -magnitude
            };
        }
    }

    // This is the shared-expert prefix. The reducer must start from, rather
    // than overwrite, every existing token/row value.
    let prefix = (0..TOKENS * HIDDEN_SIZE)
        .map(|index| (1_000 + index * 3) as f32)
        .collect::<Vec<_>>();
    let mut expected = prefix.clone();
    for token in 0..TOKENS {
        for row in 0..HIDDEN_SIZE {
            let output_index = token * HIDDEN_SIZE + row;
            let mut acc = expected[output_index];
            for rank in 0..ROUTES_PER_TOKEN {
                let route = token * ROUTES_PER_TOKEN + rank;
                acc += routes[route * HIDDEN_SIZE + row];
            }
            expected[output_index] = acc;
        }
    }

    let route_output = context
        .upload_f32_buffer(&routes)
        .expect("upload route-major outputs");
    let mut output = context
        .upload_f32_buffer(&prefix)
        .expect("upload shared-expert prefix");
    context
        .reduce_moe_route_outputs_ranked(
            &route_output,
            TOKENS,
            ROUTES_PER_TOKEN,
            HIDDEN_SIZE,
            &mut output,
        )
        .expect("ranked route reduction");
    context.sync_stream().expect("synchronize reducer");
    let actual = context
        .download_f32_buffer(&output)
        .expect("download reduced output");

    assert_eq!(actual, expected);
}

#[test]
fn expert_major_segments_gather_scatter_and_reduce() {
    let _guard = cuda_test_guard();
    if !has_cuda() {
        eprintln!("skipping: no CUDA device");
        return;
    }

    const TOKENS: usize = 3;
    const ROUTES_PER_TOKEN: usize = 2;
    const INPUT_SIZE: usize = 64;
    const INTERMEDIATE_SIZE: usize = 64;
    const HIDDEN_SIZE: usize = 16;
    const ROUTE_WEIGHT: f32 = 1.0 / 1024.0;

    let context = CudaArtifactOperatorContext::new().expect("CUDA artifact context");
    let gate_up_shape = CudaArtifactLinearShape::Fp4E2M1PackedWithE8M0Scale {
        out_features: INTERMEDIATE_SIZE,
        in_features: INPUT_SIZE,
    };
    let down_shape = CudaArtifactLinearShape::Fp4E2M1PackedWithE8M0Scale {
        out_features: HIDDEN_SIZE,
        in_features: INTERMEDIATE_SIZE,
    };

    // E2M1 nibble 0x2 is +1 and nibble 0x4 is +2. E8M0 byte 127 is scale 1.
    let gate_up_weight = vec![0x22u8; INTERMEDIATE_SIZE * INPUT_SIZE / 2];
    let gate_up_scale = vec![127u8; INTERMEDIATE_SIZE * INPUT_SIZE / 32];
    let down_one_weight = vec![0x22u8; HIDDEN_SIZE * INTERMEDIATE_SIZE / 2];
    let down_two_weight = vec![0x44u8; HIDDEN_SIZE * INTERMEDIATE_SIZE / 2];
    let down_scale = vec![127u8; HIDDEN_SIZE * INTERMEDIATE_SIZE / 32];

    let gate = context
        .upload_artifact_linear(gate_up_shape, &gate_up_weight, &gate_up_scale)
        .expect("upload gate");
    let up = context
        .upload_artifact_linear(gate_up_shape, &gate_up_weight, &gate_up_scale)
        .expect("upload up");
    let down_one = context
        .upload_artifact_linear(down_shape, &down_one_weight, &down_scale)
        .expect("upload expert-0 down");
    let down_two = context
        .upload_artifact_linear(down_shape, &down_two_weight, &down_scale)
        .expect("upload expert-1 down");

    let mut table = context.expert_slot_table(2, 1).expect("slot table");
    let expert_zero = context
        .expert_slot_pointers(&gate, &up, &down_one)
        .expect("expert 0 pointers");
    let expert_one = context
        .expert_slot_pointers(&gate, &up, &down_two)
        .expect("expert 1 pointers");
    context
        .install_expert_slot(&mut table, 0, expert_zero)
        .expect("install expert 0");
    let mut workspace = context
        .moe_segment_workspace(1, 1, TOKENS, INPUT_SIZE, INTERMEDIATE_SIZE, HIDDEN_SIZE)
        .expect("segment workspace");
    assert!(workspace.matches(1, 1, TOKENS, INPUT_SIZE, INTERMEDIATE_SIZE, HIDDEN_SIZE));

    let input = context
        .upload_f32_buffer(&vec![1.0f32; TOKENS * INPUT_SIZE])
        .expect("upload layer input");
    context
        .prepare_moe_segment_input_from_device(&input, TOKENS, INPUT_SIZE, &mut workspace)
        .expect("prepare full layer input");
    let mut route_output = context
        .allocate_moe_route_output(TOKENS, ROUTES_PER_TOKEN, HIDDEN_SIZE)
        .expect("allocate route output");
    context
        .begin_moe_segment_invocation(ROUTES_PER_TOKEN, &mut workspace, &mut route_output)
        .expect("begin cumulative segment invocation");

    // Token-major/rank-major routes map expert 0 to route indices 0, 3, 4 and
    // expert 1 to the complementary indices 1, 2, 5.
    let expert_ids = context
        .upload_i32_buffer(&[0, 1, 1, 0, 0, 1])
        .expect("upload route experts");
    let route_weights = context
        .upload_f32_buffer(&[ROUTE_WEIGHT; TOKENS * ROUTES_PER_TOKEN])
        .expect("upload route weights");
    context
        .prepare_moe_segment_grouping_stable(
            &table,
            &expert_ids,
            &route_weights,
            TOKENS * ROUTES_PER_TOKEN,
            ROUTES_PER_TOKEN,
            &mut workspace,
        )
        .expect("group expert-0 resident window");
    context
        .moe_expert_segments_stable_from_prepared(
            &table,
            ROUTES_PER_TOKEN,
            0.0,
            &mut workspace,
            &mut route_output,
        )
        .expect("execute expert-0 resident window");

    context
        .evict_expert_slot(&mut table, 0)
        .expect("evict expert 0 between windows");
    context
        .install_expert_slot(&mut table, 1, expert_one)
        .expect("install expert 1 window");
    context
        .prepare_moe_segment_grouping_stable(
            &table,
            &expert_ids,
            &route_weights,
            TOKENS * ROUTES_PER_TOKEN,
            ROUTES_PER_TOKEN,
            &mut workspace,
        )
        .expect("group expert-1 resident window");
    context
        .moe_expert_segments_stable_from_prepared(
            &table,
            ROUTES_PER_TOKEN,
            0.0,
            &mut workspace,
            &mut route_output,
        )
        .expect("execute expert-1 resident window");
    context.sync_stream().expect("synchronize segments");

    // Quantized input remains exactly 1. Gate/up each produce 64, sigmoid(64)=1,
    // and route weight 1/1024 yields hidden value 4 exactly. The two down experts
    // therefore produce 64*4*1=256 and 64*4*2=512 respectively.
    let route_values = [256.0f32, 512.0, 512.0, 256.0, 256.0, 512.0];
    let expected_routes = route_values
        .iter()
        .flat_map(|value| std::iter::repeat_n(*value, HIDDEN_SIZE))
        .collect::<Vec<_>>();
    let actual_routes = context
        .download_f32_buffer(&route_output)
        .expect("download route outputs");
    assert_close_slice(
        &actual_routes,
        &expected_routes,
        1e-3,
        "segment route output",
    );

    let prefix = (0..TOKENS * HIDDEN_SIZE)
        .map(|index| (10 + index) as f32)
        .collect::<Vec<_>>();
    let expected = prefix
        .iter()
        .map(|value| value + 256.0 + 512.0)
        .collect::<Vec<_>>();
    let mut output = context
        .upload_f32_buffer(&prefix)
        .expect("upload shared-expert prefix");
    context
        .reduce_moe_segment_route_outputs_ranked(
            &route_output,
            TOKENS,
            ROUTES_PER_TOKEN,
            HIDDEN_SIZE,
            &mut workspace,
            &mut output,
        )
        .expect("reduce complete cumulative segment routes");
    context.sync_stream().expect("synchronize reducer");
    let actual = context
        .download_f32_buffer(&output)
        .expect("download reduced output");
    assert_close_slice(&actual, &expected, 1e-3, "segment route reduction");

    // A generation change after grouping invalidates all six pointers for the
    // resident segment. Execution must accumulate an error, mark no routes
    // complete, and final reduction must not read the zeroed route output.
    context
        .begin_moe_segment_invocation(ROUTES_PER_TOKEN, &mut workspace, &mut route_output)
        .expect("begin failing segment invocation");
    context
        .prepare_moe_segment_grouping_stable(
            &table,
            &expert_ids,
            &route_weights,
            TOKENS * ROUTES_PER_TOKEN,
            ROUTES_PER_TOKEN,
            &mut workspace,
        )
        .expect("group resident expert before generation change");
    context
        .evict_expert_slot(&mut table, 1)
        .expect("invalidate grouped generation");
    context
        .moe_expert_segments_stable_from_prepared(
            &table,
            ROUTES_PER_TOKEN,
            0.0,
            &mut workspace,
            &mut route_output,
        )
        .expect("launch invalidated segment safely");
    let mut failure_output = context
        .zero_f32_buffer(TOKENS * HIDDEN_SIZE)
        .expect("failure output");
    context
        .reduce_moe_segment_route_outputs_ranked(
            &route_output,
            TOKENS,
            ROUTES_PER_TOKEN,
            HIDDEN_SIZE,
            &mut workspace,
            &mut failure_output,
        )
        .expect("reduce invalidated segment invocation");
    context.sync_stream().expect("synchronize failure reducer");
    let failure = context
        .download_f32_buffer(&failure_output)
        .expect("download failure marker");
    assert!(failure.iter().all(|value| value.is_nan()));
}
