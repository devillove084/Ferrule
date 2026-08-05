#![cfg(feature = "cuda")]

//! CUDA coverage for expert slots and compact device-side group route plans.

use ferrule_backend::cuda::CudaContext;
use ferrule_backend::cuda::context::{
    CudaArtifactLinearShape, CudaArtifactOperatorContext, CudaExpertSlotInstallTarget,
    CudaExpertSlotPointers,
};
use std::sync::{Mutex, MutexGuard, mpsc};

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
    assert_eq!(publication.kernel_launches, 1);
    assert_eq!(publication.host_to_device_copies, 0);
    assert_eq!(publication.host_to_device_bytes, 0);
    assert_eq!(publication.stream_wide_syncs, 0);
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
        .download_expert_route_resolve(&mut workspace, 4)
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
        .download_expert_route_resolve(&mut workspace, 2)
        .expect("download reused resolve result");
    assert_eq!(resolved.route_slots, vec![-1, second.slot]);
    assert_eq!(resolved.route_generations, vec![0, second.generation]);
    assert_eq!(resolved.miss_ids, vec![2]);
    assert!(!resolved.miss_overflow);
}

#[test]
fn async_slot_install_uses_exact_cross_context_consumer_fence() {
    let _guard = cuda_test_guard();
    if !has_cuda() {
        eprintln!("skipping: no CUDA device");
        return;
    }

    let consumer = CudaArtifactOperatorContext::new().expect("CUDA consumer context");
    let provider = CudaArtifactOperatorContext::new().expect("CUDA provider context");
    let (compute_priority, upload_priority, control_priority) = provider
        .stream_priorities()
        .expect("CUDA stream priorities");
    assert!(
        compute_priority <= upload_priority,
        "CUDA uses lower numeric values for higher stream priority"
    );
    assert_eq!(control_priority, compute_priority);
    let pointers = CudaExpertSlotPointers {
        gate_weight: 1,
        gate_scale: 2,
        up_weight: 3,
        up_scale: 4,
        down_weight: 5,
        down_scale: 6,
    };
    let mut table = provider.expert_slot_table(3, 1).expect("slot table");

    provider.reset_counters();
    let first = provider
        .submit_expert_slot_install(
            &mut table,
            CudaExpertSlotInstallTarget::Empty,
            0,
            0,
            1,
            pointers,
        )
        .expect("submit first async install");
    assert_eq!(table.host().binding(0), None);
    first
        .synchronize()
        .expect("complete first physical install");
    assert_eq!(
        table.host().binding(0),
        None,
        "physical event completion must not publish the host mirror"
    );
    let first = first.complete(&mut table).expect("publish first install");
    assert_eq!(table.host().binding(0), Some(first));
    let counters = provider.counters();
    assert_eq!(counters.kernel_launches, 1);
    assert_eq!(counters.compute_kernel_launches, 0);
    assert_eq!(counters.upload_kernel_launches, 1);

    let (consumer_entered_tx, consumer_entered_rx) = mpsc::channel();
    let (consumer_release_tx, consumer_release_rx) = mpsc::channel();
    consumer
        .notify_compute_stream(move || {
            consumer_entered_tx
                .send(())
                .expect("report blocked consumer callback");
            consumer_release_rx
                .recv()
                .expect("wait for consumer callback release");
        })
        .expect("queue blocked consumer marker");
    consumer_entered_rx
        .recv()
        .expect("wait for blocked consumer callback");
    let replacement_quiescence = consumer
        .compute_stream_authority()
        .record_event()
        .expect("record actual consumer fence");

    provider.reset_counters();
    let second = provider
        .submit_expert_slot_install(
            &mut table,
            CudaExpertSlotInstallTarget::Replacement {
                previous_expert: 0,
                previous_binding: first,
                consumer_quiescence: replacement_quiescence,
            },
            1,
            0,
            2,
            pointers,
        )
        .expect("submit replacement install");
    assert_eq!(table.host().binding(0), Some(first));
    assert_eq!(table.host().binding(1), None);
    assert!(
        !second.is_complete().expect("poll blocked replacement"),
        "replacement bypassed the distinct consumer stream fence"
    );
    consumer_release_tx
        .send(())
        .expect("release consumer callback");
    second
        .synchronize()
        .expect("complete replacement physical install");
    assert_eq!(table.host().binding(0), Some(first));
    assert_eq!(table.host().binding(1), None);
    let second = second.complete(&mut table).expect("publish replacement");
    assert_eq!(table.host().binding(0), None);
    assert_eq!(table.host().binding(1), Some(second));
    assert_eq!(second.slot, first.slot);
    assert_eq!(second.generation, first.generation + 1);
    let counters = provider.counters();
    assert_eq!(counters.kernel_launches, 2);
    assert_eq!(counters.compute_kernel_launches, 0);
    assert_eq!(counters.upload_kernel_launches, 2);
}

#[test]
fn exact_slot_publication_rejects_stale_and_mismatches_atomically() {
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
    let mut table = context.expert_slot_table(3, 2).expect("slot table");
    let first = context
        .install_expert_slot_at(&mut table, 0, 1, 1, pointers)
        .expect("exact first install");
    assert_eq!(first.slot, 1);
    assert_eq!(first.generation, 1);
    assert!(table.host().is_current(first));

    let installed = table.host().clone();
    assert!(
        context
            .install_expert_slot_at(&mut table, 1, 1, 2, pointers)
            .is_err()
    );
    assert_eq!(
        table.host(),
        &installed,
        "occupied slot mismatch mutated table"
    );
    assert!(
        context
            .install_expert_slot_at(&mut table, 0, 0, 1, pointers)
            .is_err()
    );
    assert_eq!(
        table.host(),
        &installed,
        "expert binding mismatch mutated table"
    );
    assert!(
        context
            .install_expert_slot_at(
                &mut table,
                0,
                1,
                1,
                CudaExpertSlotPointers {
                    gate_weight: 7,
                    ..pointers
                },
            )
            .is_err()
    );
    assert_eq!(table.host(), &installed, "pointer mismatch mutated table");
    assert!(
        context
            .evict_expert_slot_binding(&mut table, 0, 0, 1)
            .is_err()
    );
    assert_eq!(
        table.host(),
        &installed,
        "stale slot eviction mutated table"
    );
    assert!(
        context
            .evict_expert_slot_binding(&mut table, 0, 1, 2)
            .is_err()
    );
    assert_eq!(
        table.host(),
        &installed,
        "stale generation eviction mutated table"
    );

    let ids = context
        .upload_i32_buffer(&[0, 1])
        .expect("upload exact route ids");
    let mut workspace = context
        .expert_route_resolve_workspace(2, 2)
        .expect("resolve workspace");
    context
        .resolve_expert_routes(&table, &ids, 2, &mut workspace)
        .expect("resolve after rejected updates");
    let resolved = context
        .download_expert_route_resolve(&mut workspace, 2)
        .expect("download exact resolve result");
    assert_eq!(resolved.route_slots, vec![1, -1]);
    assert_eq!(resolved.route_generations, vec![1, 0]);

    context
        .evict_expert_slot_binding(&mut table, 0, 1, 1)
        .expect("exact eviction");
    assert!(!table.host().is_current(first));
    let evicted = table.host().clone();
    assert!(
        context
            .evict_expert_slot_binding(&mut table, 0, 1, 1)
            .is_err()
    );
    assert_eq!(table.host(), &evicted, "stale eviction mutated free slot");
    assert!(
        context
            .install_expert_slot_at(&mut table, 1, 1, 3, pointers)
            .is_err()
    );
    assert_eq!(
        table.host(),
        &evicted,
        "generation mismatch mutated free slot"
    );

    let second = context
        .install_expert_slot_at(&mut table, 1, 1, 2, pointers)
        .expect("exact reused install");
    assert_eq!(second.slot, first.slot);
    assert_eq!(second.generation, first.generation + 1);
    context
        .resolve_expert_routes(&table, &ids, 2, &mut workspace)
        .expect("resolve reused exact slot");
    let resolved = context
        .download_expert_route_resolve(&mut workspace, 2)
        .expect("download reused exact resolve result");
    assert_eq!(resolved.route_slots, vec![-1, 1]);
    assert_eq!(resolved.route_generations, vec![0, 2]);
}

#[test]
fn capture_safe_rejection_is_atomic_and_keeps_stable_table_usable() {
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
    let before = table.host().clone();
    let error = context
        .install_expert_slot_at(&mut table, 0, 0, 1, pointers)
        .expect_err("publication must be rejected in capture-safe mode");
    context.disable_capture_safe();
    assert!(error.to_string().contains("capture-safe"));
    assert_eq!(table.host(), &before);
    assert!(!table.is_poisoned());

    let binding = context
        .install_expert_slot_at(&mut table, 0, 0, 1, pointers)
        .expect("table remains usable after atomic rejection");
    assert_eq!(binding.slot, 0);
    assert_eq!(binding.generation, 1);

    let expert_ids = context.upload_i32_buffer(&[0]).expect("expert ids");
    let mut resolve = context
        .expert_route_resolve_workspace(1, 1)
        .expect("resolve workspace");
    context
        .resolve_expert_routes(&table, &expert_ids, 1, &mut resolve)
        .expect("usable table resolves installed expert");
    let resolved = context
        .download_expert_route_resolve(&mut resolve, 1)
        .expect("download resolved binding");
    assert_eq!(resolved.route_slots, vec![0]);
    assert_eq!(resolved.route_generations, vec![1]);
}

#[test]
fn device_group_route_plan_compacts_active_groups_indptr_and_routes() {
    let _guard = cuda_test_guard();
    if !has_cuda() {
        eprintln!("skipping: no CUDA device");
        return;
    }

    const WIDTH: usize = 64;
    const TOKENS: usize = 7;
    const ROUTES_PER_TOKEN: usize = 3;
    const ROUTES: usize = TOKENS * ROUTES_PER_TOKEN;
    const RESIDENT_SLOT_CAPACITY: usize = 16;
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
    let mut table = context
        .expert_slot_table(RESIDENT_SLOT_CAPACITY, RESIDENT_SLOT_CAPACITY)
        .expect("slot table");
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

    // Resident routes compact into two active groups with no fixed-width padding.
    // Expert 4 is missing and expert 5 has an evicted binding, so neither appears.
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
    let mut plan = context
        .expert_group_route_plan(RESIDENT_SLOT_CAPACITY, ROUTES, TOKENS, WIDTH, WIDTH, WIDTH)
        .expect("expert group route plan");

    let host = context
        .prepare_expert_group_route_plan(
            &table,
            &expert_ids_dev,
            &weights_dev,
            ROUTES,
            ROUTES_PER_TOKEN,
            &mut plan,
        )
        .expect("prepare compact group route plan");
    assert_eq!(host.active_group_count, 2);
    assert_eq!(host.small_group_count, 2);
    assert_eq!(host.max_group_rows, 9);
    assert_eq!(host.total_routed_rows, 17);

    let grouping = context
        .download_expert_group_route_plan(&plan)
        .expect("download compact group route plan");
    assert_eq!(grouping.host, host);
    assert!(!grouping.dispatch_error);
    assert_eq!(
        grouping.active_expert_slots,
        vec![expert_one.slot, expert_two.slot]
    );
    assert_eq!(
        grouping.active_group_generations,
        vec![expert_one.generation, expert_two.generation]
    );
    assert_eq!(grouping.expert_route_indptr, vec![0, 8, 17]);
    assert_eq!(grouping.expert_route_counts, vec![8, 9]);
    assert_eq!(grouping.route_indices.len(), host.total_routed_rows);
    assert_eq!(grouping.route_token_indices.len(), host.total_routed_rows);
    assert_eq!(grouping.route_weights.len(), host.total_routed_rows);

    for (group, &slot) in grouping.active_expert_slots.iter().enumerate() {
        let start = grouping.expert_route_indptr[group] as usize;
        let end = grouping.expert_route_indptr[group + 1] as usize;
        let mut compact_routes = Vec::with_capacity(end - start);
        for metadata in start..end {
            let route = grouping.route_indices[metadata] as usize;
            assert_eq!(
                grouping.route_token_indices[metadata] as usize,
                route / ROUTES_PER_TOKEN
            );
            assert_eq!(grouping.route_weights[metadata], weights[route]);
            compact_routes.push(route);
        }
        compact_routes.sort_unstable();
        let expected_routes = if slot == expert_one.slot {
            vec![0, 3, 6, 8, 9, 12, 15, 18]
        } else {
            assert_eq!(slot, expert_two.slot);
            vec![1, 4, 7, 10, 11, 13, 16, 19, 20]
        };
        assert_eq!(compact_routes, expected_routes);
    }
}
