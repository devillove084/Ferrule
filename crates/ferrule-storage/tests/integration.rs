//! Integration test: catalog + transfer engine + residency request flow.
//!
//! Verifies that the vocabulary types compose correctly without any backend
//! dependency. This is a Phase 0 smoke test — no real I/O, no CUDA, no runtime.

use ferrule_storage::{
    Budgets, EvictionWeights, InMemoryStorageCatalog, MockTransferEngine, ObjectLocator,
    ObjectReplica, Placement, ReplicaState, ResidencyPriority, ResidencyReason, ResidencyRequest,
    ResidencyScore, StorageCatalog, StorageLayout, StorageMutability, StorageObjectDescriptor,
    StorageObjectId, StorageObjectKind, StorageResidencyPolicy, TransferEngine, TransferOutcome,
};
use std::path::PathBuf;

fn model_rev(n: u64) -> ferrule_storage::ModelRevision {
    ferrule_storage::ModelRevision(n)
}

fn expert_bundle_id(layer: u32, expert: u32) -> StorageObjectId {
    StorageObjectId::ExpertBundle {
        model_revision: model_rev(0xABCD),
        layer,
        expert,
        layout_version: 2,
    }
}

fn expert_descriptor(id: &StorageObjectId) -> StorageObjectDescriptor {
    StorageObjectDescriptor {
        id: id.clone(),
        kind: StorageObjectKind::ExpertBundle,
        bytes: 2 * 1024 * 1024, // 2 MiB
        layout: StorageLayout::Bytes,
        mutability: StorageMutability::Immutable,
    }
}

fn file_locator() -> ObjectLocator {
    ObjectLocator::LocalFile {
        path: PathBuf::from("/data/model.safetensors"),
        offset: 0,
        bytes: 2 * 1024 * 1024,
    }
}

fn device_placement() -> Placement {
    Placement::Local(ferrule_storage::LocalPlacement::Device {
        device_id: 0,
        memory: ferrule_storage::DeviceMemoryKind::Vram,
    })
}

fn host_placement() -> Placement {
    Placement::Local(ferrule_storage::LocalPlacement::Host { pinned: true })
}

// ── Test: register experts in catalog, look them up ──────────────────

#[test]
fn catalog_registers_and_looks_up_expert_bundles() {
    let mut catalog = InMemoryStorageCatalog::new();

    for layer in 0..3u32 {
        for expert in 0..8u32 {
            let id = expert_bundle_id(layer, expert);
            catalog.insert(expert_descriptor(&id), vec![file_locator()]);
        }
    }

    assert_eq!(catalog.len(), 24);

    let id = expert_bundle_id(1, 4);
    let desc = catalog.descriptor(&id).unwrap();
    assert_eq!(desc.kind, StorageObjectKind::ExpertBundle);
    assert_eq!(desc.bytes, 2 * 1024 * 1024);

    let locators = catalog.locators(&id);
    assert_eq!(locators.len(), 1);
    assert!(locators[0].is_local());
}

// ── Test: transfer engine ensure returns a handle ────────────────────

#[test]
fn transfer_engine_ensure_produces_handle() {
    let mut engine = MockTransferEngine::new();
    let id = expert_bundle_id(0, 0);

    let req = ResidencyRequest::new(
        id.clone(),
        device_placement(),
        ResidencyPriority::Critical,
        ResidencyReason::ExecuteNow,
    );

    let handle = engine.ensure(req).unwrap();
    assert_eq!(handle.backend, "mock");
    assert!(handle.matches_generation(&ObjectReplica {
        object: id,
        placement: device_placement(),
        bytes: 2 * 1024 * 1024,
        state: ReplicaState::Ready,
        generation: handle.generation,
        handle: handle.clone(),
    }));
}

// ── Test: prefetch + poll flow ────────────────────────────────────────

#[test]
fn prefetch_and_poll_round_trip() {
    let mut engine = MockTransferEngine::new();

    let ids: Vec<_> = (0..4).map(|i| expert_bundle_id(0, i)).collect();
    let requests: Vec<_> = ids
        .iter()
        .map(|id| {
            ResidencyRequest::new(
                id.clone(),
                host_placement(),
                ResidencyPriority::Background,
                ResidencyReason::Prefetch,
            )
        })
        .collect();

    let tickets = engine.prefetch(&requests).unwrap();
    assert_eq!(tickets.len(), 4);

    // Queue completions for all tickets.
    for (i, ticket) in tickets.iter().enumerate() {
        engine.queue_completion(ticket.id, ids[i].clone());
    }

    let events = engine.poll().unwrap();
    assert_eq!(events.len(), 4);
    for event in &events {
        assert!(matches!(event.outcome, TransferOutcome::Completed(_)));
    }

    // Second poll is empty.
    let events2 = engine.poll().unwrap();
    assert!(events2.is_empty());
}

// ── Test: policy eviction score ranks cold over hot ──────────────────

#[test]
fn policy_evicts_cold_expert_before_hot() {
    let weights = EvictionWeights::default();
    let current_step = 100u64;

    let hot = ResidencyScore {
        execute_now: false,
        predicted: false,
        last_used_step: 95,
        activation_count: 50,
        load_cost_bytes: 2 * 1024 * 1024,
        object_bytes: 2 * 1024 * 1024,
    };

    let cold = ResidencyScore {
        execute_now: false,
        predicted: false,
        last_used_step: 10,
        activation_count: 1,
        load_cost_bytes: 2 * 1024 * 1024,
        object_bytes: 2 * 1024 * 1024,
    };

    assert!(
        cold.eviction_score(&weights, current_step) > hot.eviction_score(&weights, current_step),
        "cold expert should be evicted first"
    );
}

// ── Test: execute_now objects are never evicted ──────────────────────

#[test]
fn execute_now_never_evicted() {
    let weights = EvictionWeights::default();
    let score = ResidencyScore {
        execute_now: true,
        predicted: false,
        last_used_step: 0,
        activation_count: 0,
        load_cost_bytes: 0,
        object_bytes: 4096,
    };
    assert_eq!(score.eviction_score(&weights, 1000), f64::MIN);
}

// ── Test: budgets carry both slots and bytes ─────────────────────────

#[test]
fn budgets_have_slots_and_bytes() {
    let policy = StorageResidencyPolicy {
        budgets: Budgets {
            device_slots_per_layer: 6,                       // DSV4 top-6
            device_budget_bytes: 6 * 2 * 1024 * 1024 * 1024, // 12 GiB
            host_staging_budget_bytes: 8 * 1024 * 1024 * 1024,
            local_disk_cache_budget_bytes: None,
        },
        retain_hot: true,
        prefetch_window: 4,
        eviction_weights: EvictionWeights::default(),
    };

    assert_eq!(policy.budgets.device_slots_per_layer, 6);
    assert!(policy.budgets.device_budget_bytes > 0);
}

// ── Test: placement tier ordering ────────────────────────────────────

#[test]
fn placement_tier_rank_orders_correctly() {
    let device = device_placement();
    let host = host_placement();
    let disk = Placement::Local(ferrule_storage::LocalPlacement::Disk { volume: None });
    let remote = Placement::Remote(ferrule_storage::RemotePlacement {
        endpoint: ferrule_storage::RemoteEndpoint {
            scheme: ferrule_storage::RemoteScheme::Rdma,
            host: "peer".into(),
            port: None,
        },
        region: None,
    });

    assert!(device.tier_rank() < host.tier_rank());
    assert!(host.tier_rank() < disk.tier_rank());
    assert!(disk.tier_rank() < remote.tier_rank());
}

// ── Test: full catalog + transfer + replica composition ──────────────

#[test]
fn full_flow_catalog_to_replica() {
    let mut catalog = InMemoryStorageCatalog::new();
    let mut engine = MockTransferEngine::new();

    // Register expert in catalog.
    let id = expert_bundle_id(5, 3);
    catalog.insert(expert_descriptor(&id), vec![file_locator()]);

    // Verify catalog has it.
    let desc = catalog.descriptor(&id).unwrap();
    assert_eq!(desc.bytes, 2 * 1024 * 1024);

    // Issue residency request.
    let req = ResidencyRequest::new(
        id.clone(),
        device_placement(),
        ResidencyPriority::Critical,
        ResidencyReason::ExecuteNow,
    );
    let handle = engine.ensure(req).unwrap();

    // Construct replica metadata.
    let replica = ObjectReplica {
        object: id.clone(),
        placement: device_placement(),
        bytes: desc.bytes,
        state: ReplicaState::Ready,
        generation: handle.generation,
        handle: handle.clone(),
    };

    assert!(replica.is_ready());
    assert!(handle.matches_generation(&replica));
    assert!(replica.placement.is_device());
    assert_eq!(replica.object.to_string(), id.to_string());
}
