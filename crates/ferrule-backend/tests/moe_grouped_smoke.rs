#![cfg(feature = "cuda")]

//! CUDA smoke coverage for compact grouped MoE route outputs.

use ferrule_backend::cuda::CudaContext;
use ferrule_backend::cuda::context::{CudaArtifactOperatorContext, CudaRoutedExpertShape};
use ferrule_backend::cuda::cutlass::{self, CutlassKernelId};
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

fn has_grouped_fp4_moe() -> bool {
    cutlass::discover_provider()
        .is_ok_and(|provider| provider.supports(CutlassKernelId::GroupedFp4Moe))
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

fn bf16_round(value: f32) -> f32 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    f32::from_bits(rounded & 0xffff_0000)
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
            routes[route * HIDDEN_SIZE + row] = if route.is_multiple_of(2) {
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
            expected[output_index] = bf16_round(acc);
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
fn expert_major_groups_gather_scatter_and_reduce() {
    let _guard = cuda_test_guard();
    if !has_cuda() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    if !has_grouped_fp4_moe() {
        eprintln!("skipping: compiled CUDA target has no grouped FP4 MoE provider");
        return;
    }

    const TOKENS: usize = 3;
    const ROUTES_PER_TOKEN: usize = 2;
    const ROUTE_COUNT: usize = TOKENS * ROUTES_PER_TOKEN;
    const INPUT_SIZE: usize = 128;
    const INTERMEDIATE_SIZE: usize = 128;
    const HIDDEN_SIZE: usize = 64;
    const ROUTE_WEIGHT: f32 = 1.0 / 1024.0;

    let context = CudaArtifactOperatorContext::new().expect("CUDA artifact context");
    let shape = CudaRoutedExpertShape::new(INPUT_SIZE, INTERMEDIATE_SIZE, HIDDEN_SIZE)
        .expect("routed expert shape");
    let mut arena = context
        .allocate_routed_expert_arena(shape, 2)
        .expect("routed expert arena");
    let mut expert_zero_frame = arena.allocate_frame().expect("expert 0 frame");
    let mut expert_one_frame = arena.allocate_frame().expect("expert 1 frame");

    // E2M1 nibble 0x2 is +1 and nibble 0x4 is +2. E8M0 byte 127 is scale 1.
    let gate_up_weight = vec![0x22u8; INTERMEDIATE_SIZE * INPUT_SIZE / 2];
    let gate_up_scale = vec![127u8; INTERMEDIATE_SIZE * INPUT_SIZE / 32];
    let down_one_weight = vec![0x22u8; HIDDEN_SIZE * INTERMEDIATE_SIZE / 2];
    let down_two_weight = vec![0x44u8; HIDDEN_SIZE * INTERMEDIATE_SIZE / 2];
    let down_scale = vec![127u8; HIDDEN_SIZE * INTERMEDIATE_SIZE / 32];
    let pin = |values: &[u8]| {
        context
            .pin_u8_host_buffer(values)
            .expect("pin expert tensor")
    };
    let expert_zero_upload = context
        .materialize_routed_expert_from_pinned_async(
            &mut expert_zero_frame,
            pin(&gate_up_weight),
            pin(&gate_up_scale),
            pin(&gate_up_weight),
            pin(&gate_up_scale),
            pin(&down_one_weight),
            pin(&down_scale),
        )
        .expect("materialize expert 0");
    let expert_one_upload = context
        .materialize_routed_expert_from_pinned_async(
            &mut expert_one_frame,
            pin(&gate_up_weight),
            pin(&gate_up_scale),
            pin(&gate_up_weight),
            pin(&gate_up_scale),
            pin(&down_two_weight),
            pin(&down_scale),
        )
        .expect("materialize expert 1");
    expert_zero_upload
        .synchronize()
        .expect("complete expert 0 materialization");
    expert_one_upload
        .synchronize()
        .expect("complete expert 1 materialization");

    let mut table = context.expert_slot_table(2, 1).expect("slot table");
    let expert_zero = expert_zero_frame
        .expert_slot_pointers()
        .expect("expert 0 pointers");
    let expert_one = expert_one_frame
        .expert_slot_pointers()
        .expect("expert 1 pointers");
    context
        .install_expert_slot(&mut table, 0, expert_zero)
        .expect("install expert 0");
    let mut plan = context
        .expert_group_route_plan(
            1,
            ROUTE_COUNT,
            TOKENS,
            INPUT_SIZE,
            INTERMEDIATE_SIZE,
            HIDDEN_SIZE,
        )
        .expect("expert group route plan");
    assert!(plan.matches(
        1,
        ROUTE_COUNT,
        TOKENS,
        INPUT_SIZE,
        INTERMEDIATE_SIZE,
        HIDDEN_SIZE
    ));

    let input = context
        .upload_f32_buffer(&vec![1.0f32; TOKENS * INPUT_SIZE])
        .expect("upload layer input");
    context
        .prepare_expert_group_route_input_from_device(&input, TOKENS, INPUT_SIZE, &mut plan)
        .expect("prepare full layer input");
    let mut route_output = context
        .allocate_moe_route_output(TOKENS, ROUTES_PER_TOKEN, HIDDEN_SIZE)
        .expect("allocate route output");
    context
        .begin_expert_group_route_invocation(ROUTES_PER_TOKEN, &mut plan, &mut route_output)
        .expect("begin cumulative grouped invocation");

    // Token-major/rank-major routes map expert 0 to route indices 0, 3, 4 and
    // expert 1 to the complementary indices 1, 2, 5.
    let expert_ids = context
        .upload_i32_buffer(&[0, 1, 1, 0, 0, 1])
        .expect("upload route experts");
    let route_weights = context
        .upload_f32_buffer(&[ROUTE_WEIGHT; ROUTE_COUNT])
        .expect("upload route weights");
    let first = context
        .prepare_expert_group_route_plan(
            &table,
            &expert_ids,
            &route_weights,
            ROUTE_COUNT,
            ROUTES_PER_TOKEN,
            &mut plan,
        )
        .expect("plan expert-0 resident window");
    assert_eq!(first.active_group_count, 1);
    assert_eq!(first.total_routed_rows, 3);
    context.reset_counters();
    context
        .grouped_fp4_moe_from_prepared_plan(
            &table,
            ROUTES_PER_TOKEN,
            0.0,
            &mut plan,
            &mut route_output,
        )
        .expect("execute expert-0 resident window");
    assert_eq!(context.counters().kernel_launches, 1);

    context
        .evict_expert_slot(&mut table, 0)
        .expect("evict expert 0 between windows");
    context
        .install_expert_slot(&mut table, 1, expert_one)
        .expect("install expert 1 window");
    let second = context
        .prepare_expert_group_route_plan(
            &table,
            &expert_ids,
            &route_weights,
            ROUTE_COUNT,
            ROUTES_PER_TOKEN,
            &mut plan,
        )
        .expect("plan expert-1 resident window");
    assert_eq!(second.active_group_count, 1);
    assert_eq!(second.total_routed_rows, 3);
    context.reset_counters();
    context
        .grouped_fp4_moe_from_prepared_plan(
            &table,
            ROUTES_PER_TOKEN,
            0.0,
            &mut plan,
            &mut route_output,
        )
        .expect("execute expert-1 resident window");
    assert_eq!(context.counters().kernel_launches, 1);
    context.sync_stream().expect("synchronize grouped MoE");

    // The official grouped FP4 contract quantizes BF16 activations to FP8/K128.
    // Input 1 stays exact, gate/up each produce BF16 128, and applying the route
    // weight before the down-input quantization yields hidden value 16. The two
    // down experts therefore produce 128*16*1=2048 and 128*16*2=4096.
    let route_values = [2048.0f32, 4096.0, 4096.0, 2048.0, 2048.0, 4096.0];
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
        "grouped route output",
    );

    let prefix = (0..TOKENS * HIDDEN_SIZE)
        .map(|index| (10 + index) as f32)
        .collect::<Vec<_>>();
    let expected = prefix
        .iter()
        .map(|value| bf16_round(value + 2048.0 + 4096.0))
        .collect::<Vec<_>>();
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
        .expect("reduce complete grouped routes");
    context.sync_stream().expect("synchronize reducer");
    let actual = context
        .download_f32_buffer(&output)
        .expect("download reduced output");
    assert_close_slice(&actual, &expected, 1e-3, "grouped route reduction");
}
