//! CUDA coverage for stable expert slots and device-side route resolution.

use cuda_core::CudaContext;
use ferrule_cuda::context::{
    CudaArtifactLinearShape, CudaArtifactOperatorContext, CudaExpertSlotPointers,
};
use std::sync::{Mutex, MutexGuard};

static CUDA_TEST_LOCK: Mutex<()> = Mutex::new(());

fn cuda_test_guard() -> MutexGuard<'static, ()> {
    CUDA_TEST_LOCK
        .lock()
        .expect("CUDA expert slot test lock poisoned")
}

fn has_cuda() -> bool {
    CudaContext::new(0).is_ok()
}

#[test]
fn stable_slot_table_resolves_residents_misses_and_reuse_generation() {
    let _guard = cuda_test_guard();
    if !has_cuda() {
        eprintln!("skipping: no CUDA device");
        return;
    }

    const WIDTH: usize = 32;
    let context = CudaArtifactOperatorContext::new().expect("CUDA artifact context");
    let shape = CudaArtifactLinearShape::Fp4E2M1PackedWithE8M0Scale {
        out_features: WIDTH,
        in_features: WIDTH,
    };
    let weight = vec![0x22; WIDTH * WIDTH / 2];
    let scale = vec![127; WIDTH * WIDTH / 32];
    let gate = context
        .upload_artifact_linear(shape, &weight, &scale)
        .expect("upload gate");
    let up = context
        .upload_artifact_linear(shape, &weight, &scale)
        .expect("upload up");
    let down = context
        .upload_artifact_linear(shape, &weight, &scale)
        .expect("upload down");
    let pointers = context
        .expert_slot_pointers(&gate, &up, &down)
        .expect("expert pointers");

    let mut table = context.expert_slot_table(4, 1).expect("slot table");
    context.reset_counters();
    let first = context
        .install_expert_slot(&mut table, 2, pointers)
        .expect("install expert 2");
    let publication = context.counters();
    assert_eq!(publication.host_to_device_copies, 9);
    assert_eq!(publication.host_to_device_bytes, 84);
    assert_eq!(publication.stream_wide_syncs, 1);
    assert!(!table.is_poisoned());
    let ids = context
        .upload_i32_buffer(&[2, 1, -1, 4])
        .expect("upload route ids");
    let mut workspace = context
        .expert_route_resolve_workspace(4, 2)
        .expect("resolve workspace");
    context
        .resolve_expert_routes(&table, &ids, 4, &mut workspace)
        .expect("resolve routes");
    let resolved = context
        .download_expert_route_resolve(&workspace, 4)
        .expect("download resolve result");
    assert_eq!(resolved.route_slots, vec![first.slot, -1, -1, -1]);
    assert_eq!(resolved.route_generations, vec![first.generation, 0, 0, 0]);
    assert_eq!(resolved.miss_markers, vec![0, 1, 1, 1]);
    assert_eq!(resolved.miss_ids.len(), 2);
    assert!(resolved.miss_ids.iter().all(|id| [1, -1, 4].contains(id)));
    assert_ne!(resolved.miss_ids[0], resolved.miss_ids[1]);
    assert!(resolved.miss_overflow);

    assert!(
        context
            .evict_expert_slot(&mut table, 2)
            .expect("evict expert 2")
    );
    let second = context
        .install_expert_slot(&mut table, 1, pointers)
        .expect("install expert 1");
    assert_eq!(second.slot, first.slot);
    assert_ne!(second.generation, first.generation);
    assert!(!table.host().is_current(first));
    assert!(table.host().is_current(second));

    let ids = context
        .upload_i32_buffer(&[2, 1])
        .expect("upload reused route ids");
    context
        .resolve_expert_routes(&table, &ids, 2, &mut workspace)
        .expect("resolve reused slot");
    let resolved = context
        .download_expert_route_resolve(&workspace, 2)
        .expect("download reused resolve result");
    assert_eq!(resolved.route_slots, vec![-1, second.slot]);
    assert_eq!(resolved.route_generations, vec![0, second.generation]);
    assert_eq!(resolved.miss_ids, vec![2]);
    assert!(!resolved.miss_overflow);
}

#[test]
fn failed_update_and_rollback_poison_every_stable_entry_point() {
    let _guard = cuda_test_guard();
    if !has_cuda() {
        eprintln!("skipping: no CUDA device");
        return;
    }

    let context = CudaArtifactOperatorContext::new().expect("CUDA artifact context");
    let pointers = CudaExpertSlotPointers {
        gate_weight: 1,
        gate_scale: 2,
        up_weight: 3,
        up_scale: 4,
        down_weight: 5,
        down_scale: 6,
    };
    let mut table = context.expert_slot_table(1, 1).expect("slot table");
    context.enable_capture_safe();
    let error = context
        .install_expert_slot(&mut table, 0, pointers)
        .expect_err("publication and rollback must fail in capture-safe mode");
    context.disable_capture_safe();
    assert!(error.to_string().contains("table poisoned"));
    assert!(table.is_poisoned());

    assert!(
        context
            .install_expert_slot(&mut table, 0, pointers)
            .is_err()
    );
    assert!(context.evict_expert_slot(&mut table, 0).is_err());
    assert!(context.clear_expert_slot_table(&mut table).is_err());

    let expert_ids = context.upload_i32_buffer(&[0]).expect("expert ids");
    let router_weights = context.upload_f32_buffer(&[1.0]).expect("router weights");
    let mut resolve = context
        .expert_route_resolve_workspace(1, 1)
        .expect("resolve workspace");
    assert!(
        context
            .resolve_expert_routes(&table, &expert_ids, 1, &mut resolve)
            .is_err()
    );

    let mut batched = context
        .moe_batched_workspace(1, 32, 32, 32)
        .expect("batched workspace");
    assert!(
        context
            .prepare_moe_experts_batched_workspace_stable(
                &table,
                &[0],
                &expert_ids,
                &router_weights,
                1,
                32,
                32,
                32,
                &mut resolve,
                &mut batched,
            )
            .is_err()
    );

    let mut segments = context
        .moe_segment_workspace(1, 1, 1, 64, 64, 16)
        .expect("segment workspace");
    assert!(
        context
            .prepare_moe_segment_grouping_stable(
                &table,
                &expert_ids,
                &router_weights,
                1,
                1,
                &mut segments,
            )
            .is_err()
    );
    let mut route_output = context
        .allocate_moe_route_output(1, 1, 16)
        .expect("route output");
    assert!(
        context
            .moe_expert_segments_stable_from_prepared(
                &table,
                1,
                0.0,
                &mut segments,
                &mut route_output,
            )
            .is_err()
    );
}

#[test]
fn stable_dispatch_matches_cpu_reference_and_is_transfer_free_when_warm() {
    let _guard = cuda_test_guard();
    if !has_cuda() {
        eprintln!("skipping: no CUDA device");
        return;
    }

    const WIDTH: usize = 64;
    const ROUTES: usize = 2;
    let context = CudaArtifactOperatorContext::new().expect("CUDA artifact context");
    let shape = CudaArtifactLinearShape::Fp4E2M1PackedWithE8M0Scale {
        out_features: WIDTH,
        in_features: WIDTH,
    };
    let weight = vec![0x22; WIDTH * WIDTH / 2];
    let scale = vec![127; WIDTH * WIDTH / 32];
    let gate = context
        .upload_artifact_linear(shape, &weight, &scale)
        .expect("upload gate");
    let up = context
        .upload_artifact_linear(shape, &weight, &scale)
        .expect("upload up");
    let down = context
        .upload_artifact_linear(shape, &weight, &scale)
        .expect("upload down");
    let pointers = context
        .expert_slot_pointers(&gate, &up, &down)
        .expect("expert pointers");
    let mut table = context.expert_slot_table(8, 8).expect("slot table");
    context
        .install_expert_slot(&mut table, 1, pointers)
        .expect("install expert 1");
    context
        .install_expert_slot(&mut table, 3, pointers)
        .expect("install expert 3");

    let input = context
        .upload_f32_buffer(&vec![0.125; WIDTH])
        .expect("upload input");
    let expert_ids = context
        .upload_i32_buffer(&[1, 3])
        .expect("upload expert ids");
    let router_weights = context
        .upload_f32_buffer(&[0.25, 0.75])
        .expect("upload router weights");
    let mut stable_workspace = context
        .moe_batched_workspace(6, WIDTH, WIDTH, WIDTH)
        .expect("stable dispatch workspace");
    let mut resolve = context
        .expert_route_resolve_workspace(6, 6)
        .expect("resolve workspace");
    let mut stable_output = context.zero_f32_buffer(WIDTH).expect("stable output");
    context
        .prepare_moe_experts_batched_workspace_stable(
            &table,
            &[1, 3],
            &expert_ids,
            &router_weights,
            ROUTES,
            WIDTH,
            WIDTH,
            WIDTH,
            &mut resolve,
            &mut stable_workspace,
        )
        .expect("stable device dispatch");
    context
        .moe_experts_batched_add_into_from_device_prepared(
            &input,
            0.0,
            ROUTES,
            WIDTH,
            WIDTH,
            &mut stable_workspace,
            &mut stable_output,
        )
        .expect("stable prepared compute");
    let stable_values = context
        .download_f32_buffer(&stable_output)
        .expect("download stable dispatch output");
    // Every packed weight is +1 with scale 1, and 64 copies of input 0.125
    // produce gate/up values of 8. The two route weights quantize the hidden
    // activations to 16 and 48, so each down-projection element is
    // 64 * (16 + 48) = 4096.
    let expected = vec![4096.0; WIDTH];
    assert_eq!(stable_values, expected);

    context.reset_counters();
    context.enable_capture_safe();
    context
        .prepare_moe_experts_batched_workspace_stable(
            &table,
            &[1, 3],
            &expert_ids,
            &router_weights,
            ROUTES,
            WIDTH,
            WIDTH,
            WIDTH,
            &mut resolve,
            &mut stable_workspace,
        )
        .expect("warm stable device dispatch");
    context
        .moe_experts_batched_add_into_from_device_prepared(
            &input,
            0.0,
            ROUTES,
            WIDTH,
            WIDTH,
            &mut stable_workspace,
            &mut stable_output,
        )
        .expect("warm stable prepared compute");
    context.disable_capture_safe();
    let counters = context.counters();
    assert_eq!(counters.host_to_device_copies, 0, "warm stable H2D");
    assert_eq!(counters.device_to_host_copies, 0, "warm stable D2H");
    assert_eq!(counters.stream_wide_syncs, 0, "warm stable syncs");
    assert_eq!(
        counters.device_allocation_attempts, 0,
        "warm stable allocation attempts"
    );
    assert_eq!(counters.device_allocations, 0, "warm stable allocations");
}

#[test]
fn device_segment_grouping_preserves_route_identity_boundaries_and_padding() {
    let _guard = cuda_test_guard();
    if !has_cuda() {
        eprintln!("skipping: no CUDA device");
        return;
    }

    const WIDTH: usize = 64;
    const TOKENS: usize = 7;
    const ROUTES_PER_TOKEN: usize = 3;
    const ROUTES: usize = TOKENS * ROUTES_PER_TOKEN;
    let context = CudaArtifactOperatorContext::new().expect("CUDA artifact context");
    let shape = CudaArtifactLinearShape::Fp4E2M1PackedWithE8M0Scale {
        out_features: WIDTH,
        in_features: WIDTH,
    };
    let weight = vec![0x22; WIDTH * WIDTH / 2];
    let scale = vec![127; WIDTH * WIDTH / 32];
    let gate = context
        .upload_artifact_linear(shape, &weight, &scale)
        .expect("upload gate");
    let up = context
        .upload_artifact_linear(shape, &weight, &scale)
        .expect("upload up");
    let down = context
        .upload_artifact_linear(shape, &weight, &scale)
        .expect("upload down");
    let pointers = context
        .expert_slot_pointers(&gate, &up, &down)
        .expect("expert pointers");
    let mut table = context.expert_slot_table(16, 16).expect("slot table");
    let expert_one = context
        .install_expert_slot(&mut table, 1, pointers)
        .expect("install expert 1");
    let expert_two = context
        .install_expert_slot(&mut table, 2, pointers)
        .expect("install expert 2");
    context
        .install_expert_slot(&mut table, 5, pointers)
        .expect("install stale expert");
    context
        .evict_expert_slot(&mut table, 5)
        .expect("evict stale expert");

    // Tokens 0..3 and 3..7 represent two packed sequences. Expert 1 has four
    // routes in each sequence and must form one cross-sequence segment of 8.
    // Expert 2 has 9 routes and therefore forms two segments with 7 padding
    // columns. Expert 4 is missing and expert 5 has a stale/evicted binding.
    let expert_ids = [
        1, 2, 4, // token 0
        1, 2, 5, // token 1
        1, 2, 1, // token 2
        1, 2, 2, // token 3
        1, 2, 4, // token 4
        1, 2, 5, // token 5
        1, 2, 2, // token 6
    ];
    let weights = (0..ROUTES)
        .map(|route| route as f32 + 0.25)
        .collect::<Vec<_>>();
    let expert_ids_dev = context
        .upload_i32_buffer(&expert_ids)
        .expect("upload resident routes");
    let weights_dev = context
        .upload_f32_buffer(&weights)
        .expect("upload resident weights");
    let mut workspace = context
        .moe_segment_workspace(16, 8, TOKENS, WIDTH, WIDTH, WIDTH)
        .expect("segment workspace");

    context
        .prepare_moe_segment_grouping_stable(
            &table,
            &expert_ids_dev,
            &weights_dev,
            ROUTES,
            ROUTES_PER_TOKEN,
            &mut workspace,
        )
        .expect("device segment grouping");
    let grouping = context
        .download_moe_segment_grouping(&workspace)
        .expect("download segment grouping");
    assert!(!grouping.dispatch_error);

    let valid_segments = grouping
        .segment_expert_slots
        .iter()
        .copied()
        .take_while(|slot| *slot >= 0)
        .collect::<Vec<_>>();
    assert_eq!(
        valid_segments,
        vec![expert_one.slot, expert_two.slot, expert_two.slot]
    );
    assert!(
        grouping.segment_expert_slots[3..]
            .iter()
            .all(|slot| *slot == -1)
    );

    let resident_routes = expert_ids
        .iter()
        .enumerate()
        .filter_map(|(route, expert)| [1, 2].contains(expert).then_some(route))
        .collect::<Vec<_>>();
    let mut scattered_routes = Vec::new();
    let mut padding = 0;
    for segment in 0..valid_segments.len() {
        for column in 0..8 {
            let metadata = segment * 8 + column;
            let route = grouping.segment_route_indices[metadata];
            let token = grouping.segment_token_indices[metadata];
            if route < 0 {
                assert_eq!(token, -1);
                assert_eq!(grouping.segment_route_weights[metadata], 0.0);
                padding += 1;
                continue;
            }
            let route = route as usize;
            assert_eq!(token as usize, route / ROUTES_PER_TOKEN);
            assert_eq!(grouping.segment_route_weights[metadata], weights[route]);
            let expected_slot = if expert_ids[route] == 1 {
                expert_one.slot
            } else {
                expert_two.slot
            };
            assert_eq!(valid_segments[segment], expected_slot);
            scattered_routes.push(route);
        }
    }
    scattered_routes.sort_unstable();
    assert_eq!(scattered_routes, resident_routes);
    assert_eq!(padding, 7);

    context.reset_counters();
    context.enable_capture_safe();
    context
        .prepare_moe_segment_grouping_stable(
            &table,
            &expert_ids_dev,
            &weights_dev,
            ROUTES,
            ROUTES_PER_TOKEN,
            &mut workspace,
        )
        .expect("warm device segment grouping");
    context.disable_capture_safe();
    let counters = context.counters();
    assert_eq!(counters.host_to_device_copies, 0, "warm grouping H2D");
    assert_eq!(counters.device_to_host_copies, 0, "warm grouping D2H");
    assert_eq!(counters.stream_wide_syncs, 0, "warm grouping syncs");
    assert_eq!(
        counters.device_allocation_attempts, 0,
        "warm grouping allocation attempts"
    );
    assert_eq!(counters.device_allocations, 0, "warm grouping allocations");
}
