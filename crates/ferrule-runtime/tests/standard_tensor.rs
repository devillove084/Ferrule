//! Host-only TP admission and bounded collective protocol regression tests.
use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::{ParallelGroupId, ParallelRankId, ParallelTopologyId};
use ferrule_model::models::qwen3::Qwen3DenseRecipe;
use ferrule_model::transformer::parallel::TensorParallelCollective::{AllGather, Sum};
use ferrule_model::transformer::{
    DecoderRecipe, LayerSegmentPlan, StandardTensorCollective, StandardTensorPlacement,
    StandardTensorPlan, SyntheticDecoderRecipe,
};
use ferrule_runtime::parallel::collective::HostCollectiveLimits;
use ferrule_runtime::parallel::tensor::decoder_collective::DecoderTensorCollective;
use std::time::{Duration, Instant};

fn config() -> serde_json::Value {
    serde_json::json!({
        "architectures":["Qwen3ForCausalLM"], "model_type":"qwen3", "torch_dtype":"bfloat16",
        "hidden_act":"silu", "vocab_size":11, "hidden_size":8, "num_hidden_layers":2,
        "num_attention_heads":8, "num_key_value_heads":4, "head_dim":2, "intermediate_size":13,
        "max_position_embeddings":16, "rms_norm_eps":0.00001, "rope_theta":10000.0,
        "tie_word_embeddings":true, "attention_bias":false, "use_sliding_window":false,
        "attention_dropout":0.0, "use_cache":true, "max_window_layers":2,
        "initializer_range":0.02, "bos_token_id":1, "eos_token_id":2
    })
}
fn placements(degree: usize) -> Vec<StandardTensorPlacement> {
    (0..degree)
        .map(|device| StandardTensorPlacement {
            owner: ParallelRankId::new(100 - device as u32 * 7),
            device,
        })
        .collect()
}
fn plan(degree: usize) -> StandardTensorPlan {
    StandardTensorPlan::new(
        &Qwen3DenseRecipe::new().build_spec(&config()).unwrap(),
        placements(degree),
    )
    .unwrap()
}
fn endpoints(timeout: Duration, max: usize) -> Vec<DecoderTensorCollective> {
    DecoderTensorCollective::new_group(
        &plan(2),
        ParallelTopologyId::new(2),
        ParallelGroupId::new(7),
        HostCollectiveLimits {
            max_ranks: 2,
            max_elements_per_rank: max,
            max_host_bytes: 4096,
        },
        timeout,
    )
    .unwrap()
}
fn tx(id: u64) -> ExecutionTransactionId {
    ExecutionTransactionId::new(id).unwrap()
}

#[test]
fn dense_admission_validates_geometry_placement_and_pp_segments() {
    let spec = Qwen3DenseRecipe::new().build_spec(&config()).unwrap();
    for degree in [1, 2, 4] {
        let plan = StandardTensorPlan::new(&spec, placements(degree)).unwrap();
        assert_eq!(plan.ranks(), degree);
        assert_eq!(
            plan.placement(ParallelRankId::new(0)).unwrap().owner.get(),
            100
        );
        assert!(plan.placement(ParallelRankId::new(degree as u32)).is_err());
        assert!(plan.kv_planes(2, 16).is_ok());
        assert!(plan.kv_planes(0, 16).is_err());
        assert!(
            plan.validate_segment(&spec, &LayerSegmentPlan::new(2, 0..2, true, true).unwrap())
                .is_ok()
        );
        assert!(
            plan.validate_segment(&spec, &LayerSegmentPlan::new(2, 0..1, true, false).unwrap())
                .is_ok()
        );
        assert!(
            plan.validate_segment(&spec, &LayerSegmentPlan::new(2, 0..2, false, true).unwrap())
                .is_ok()
        );
        assert!(
            plan.kv_planes_for_segment(
                &LayerSegmentPlan::new(2, 0..1, true, false).unwrap(),
                2,
                16,
            )
            .is_ok()
        );
    }
    for degree in [0, 3, 8] {
        assert!(StandardTensorPlan::new(&spec, placements(degree)).is_err());
    }
    let mut duplicate = placements(2);
    duplicate[1].owner = duplicate[0].owner;
    assert!(
        StandardTensorPlan::new(&spec, duplicate)
            .unwrap_err()
            .to_string()
            .contains("distinct KV owner")
    );
    let mut duplicate = placements(2);
    duplicate[1].device = duplicate[0].device;
    assert!(StandardTensorPlan::new(&spec, duplicate).is_err());
    for (field, value) in [
        ("num_key_value_heads", 2),
        ("num_attention_heads", 6),
        ("intermediate_size", 3),
        ("vocab_size", 3),
    ] {
        let mut config = config();
        config[field] = value.into();
        if field == "num_attention_heads" {
            config["num_key_value_heads"] = 2.into();
        }
        let spec = Qwen3DenseRecipe::new().build_spec(&config).unwrap();
        assert!(
            StandardTensorPlan::new(&spec, placements(4)).is_err(),
            "{field}"
        );
    }
    let mut changed = config();
    changed["vocab_size"] = 12.into();
    assert!(
        plan(2)
            .validate_segment(
                &Qwen3DenseRecipe::new().build_spec(&changed).unwrap(),
                &LayerSegmentPlan::new(2, 0..2, true, true).unwrap()
            )
            .is_err()
    );
    let moe = serde_json::json!({
        "vocab_size": 11, "hidden_size": 8, "num_attention_heads": 8,
        "num_key_value_heads": 4, "head_dim": 2, "intermediate_size": 13,
        "num_experts": 2, "experts_per_token": 2, "max_position_embeddings": 16,
        "rms_norm_eps": 0.00001, "rope_theta": 10000.0, "tie_word_embeddings": true
    });
    let spec = SyntheticDecoderRecipe::new().build_spec(&moe).unwrap();
    assert!(
        StandardTensorPlan::new(&spec, placements(2))
            .unwrap_err()
            .to_string()
            .contains("MoE TP is unsupported")
    );
}

#[test]
fn collective_orders_members_and_reuses_one_group_across_sites_and_transactions() {
    let workers = endpoints(Duration::from_secs(2), 2)
        .into_iter()
        .enumerate()
        .map(|(rank, mut endpoint)| {
            std::thread::spawn(move || {
                assert_eq!(endpoint.owner().get(), 100 - rank as u32 * 7);
                for step in 1..=16 {
                    assert_eq!(
                        endpoint
                            .exchange(tx(step), 10, Sum, vec![rank as f32 + 1.0, 2.0])
                            .unwrap(),
                        vec![3.0, 4.0]
                    );
                    assert_eq!(
                        endpoint
                            .exchange(tx(step), 20, AllGather, vec![rank as f32, step as f32])
                            .unwrap(),
                        vec![0.0, step as f32, 1.0, step as f32]
                    );
                }
            })
        })
        .collect::<Vec<_>>();
    for worker in workers {
        worker.join().unwrap();
    }
}

#[test]
fn mismatched_transaction_site_kind_count_and_budget_wake_all_peers() {
    for case in 0..5 {
        let mut group = endpoints(Duration::from_secs(2), 2);
        let mut second = group.pop().unwrap();
        let mut first = group.pop().unwrap();
        let worker = std::thread::spawn(move || first.exchange(tx(1), 10, Sum, vec![1.0]));
        let transaction = if case == 0 { tx(2) } else { tx(1) };
        let site = if case == 1 { 11 } else { 10 };
        let kind = if case == 2 { AllGather } else { Sum };
        let values = match case {
            3 => vec![1.0, 2.0],
            4 => vec![0.0; 3],
            _ => vec![1.0],
        };
        assert!(second.exchange(transaction, site, kind, values).is_err());
        assert!(worker.join().unwrap().is_err());
        assert!(second.exchange(tx(3), 10, Sum, vec![1.0]).is_err());
    }
}

#[test]
fn timeout_abort_and_dropped_peer_fail_closed_without_unbounded_wait() {
    for case in 0..3 {
        let mut group = endpoints(Duration::from_millis(25), 2);
        let mut second = group.pop().unwrap();
        let mut first = group.pop().unwrap();
        let start = Instant::now();
        if case == 0 {
            second.abort();
        }
        if case == 1 {
            drop(second);
        }
        assert!(first.exchange(tx(1), 10, Sum, vec![1.0]).is_err());
        assert!(start.elapsed() < Duration::from_secs(1));
        assert!(first.exchange(tx(2), 10, Sum, vec![1.0]).is_err());
    }
    assert!(
        DecoderTensorCollective::new_group(
            &plan(2),
            ParallelTopologyId::new(2),
            ParallelGroupId::new(7),
            HostCollectiveLimits {
                max_ranks: 2,
                max_elements_per_rank: 2,
                max_host_bytes: 4096
            },
            Duration::ZERO
        )
        .is_err()
    );
}

#[test]
fn pipeline_collective_control_checks_lifetime_and_cancellation_without_a_peer() {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    let mut peers = endpoints(Duration::from_secs(30), 2);
    let control = peers[0].control();
    assert!(control.finish(tx(1)).is_err());
    control
        .begin(tx(1), Arc::new(AtomicBool::new(false)))
        .unwrap();
    assert!(
        control
            .begin(tx(2), Arc::new(AtomicBool::new(false)))
            .is_err()
    );
    assert!(control.finish(tx(2)).is_err());
    control.finish(tx(1)).unwrap();
    assert!(
        control
            .begin(tx(1), Arc::new(AtomicBool::new(false)))
            .is_err()
    );
    control
        .begin(tx(2), Arc::new(AtomicBool::new(true)))
        .unwrap();
    let before = Instant::now();
    assert!(peers[0].exchange(tx(2), 1, Sum, vec![1.0]).is_err());
    assert!(before.elapsed() < Duration::from_secs(1));
    assert!(control.is_failed());
    assert!(control.finish(tx(2)).is_err());
    assert!(
        control
            .begin(tx(3), Arc::new(AtomicBool::new(false)))
            .is_err()
    );
}

#[test]
fn shared_abort_peer_error_and_peer_drop_wake_an_already_blocked_collective() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    for mode in 0..3 {
        let mut peers = endpoints(Duration::from_secs(30), 2);
        let mut second = Some(peers.pop().unwrap());
        let mut first = peers.pop().unwrap();
        let control = first.control();
        let cancelled = Arc::new(AtomicBool::new(false));
        control.begin(tx(1), Arc::clone(&cancelled)).unwrap();
        let waiter = std::thread::spawn(move || first.exchange(tx(1), 10, Sum, vec![1.0]));
        let deadline = Instant::now() + Duration::from_secs(2);
        while control.waiting_members() != 1 {
            assert!(
                Instant::now() < deadline,
                "collective never entered its condvar wait"
            );
            std::thread::yield_now();
        }
        let started = Instant::now();
        if mode == 1 {
            assert!(
                second
                    .as_mut()
                    .unwrap()
                    .exchange(tx(1), 11, Sum, vec![2.0])
                    .is_err()
            );
        } else if mode == 2 {
            drop(second.take());
        } else {
            assert!(control.abort(tx(99), "stale cancellation").is_err());
            assert_eq!(control.waiting_members(), 1);
            assert!(!cancelled.load(Ordering::Acquire));
            control.abort(tx(1), "cancel blocked execution").unwrap();
        }
        assert!(waiter.join().unwrap().is_err());
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "must wake, not expire the 30s timeout"
        );
        assert!(cancelled.load(Ordering::Acquire));
        assert!(control.is_failed());
        assert_eq!(control.waiting_members(), 0);
        // Cancel and peer-error modes do not rely on dropping an endpoint.
        if let Some(mut second) = second {
            assert!(second.exchange(tx(2), 10, Sum, vec![1.0]).is_err());
        }
    }
}

// CUDA-feature preflight regression without CUDA initialization: a valid standard
// TP mesh must reach its owner loader, unlike the rejected generic TP constructor.
#[cfg(feature = "cuda")]
#[test]
fn standard_cuda_tensor_admits_tp2_before_owner_resource_loading() {
    use ferrule_common::{ParallelismPlan, ValidatedParallelTopology};
    use ferrule_model::execution::ExecutionPrecisionPolicy;
    use ferrule_runtime::parallel::pipeline::{
        PipelineConfig, PipelineParallelExecutor, StandardCudaTensorConfig,
    };
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    let spec = Qwen3DenseRecipe::new().build_spec(&config()).unwrap();
    for pp in [1, 2] {
        let topology = ValidatedParallelTopology::new(
            ParallelTopologyId::new(501),
            (pp * 2) as u32,
            ParallelRankId::new(0),
            ParallelismPlan::validated(1, 2, 1, 1, 1, pp).unwrap(),
        )
        .unwrap();
        let plans = (0..pp)
            .map(|stage| {
                LayerSegmentPlan::new(
                    2,
                    stage * 2 / pp..(stage + 1) * 2 / pp,
                    stage == 0,
                    stage + 1 == pp,
                )
                .unwrap()
            })
            .collect();
        let loads = Arc::new(AtomicUsize::new(0));
        let owner_loads = Arc::clone(&loads);
        let result = PipelineParallelExecutor::new_standard_cuda_tensor(
            topology,
            plans,
            PipelineConfig {
                page_size: 2,
                max_pages: 8,
                max_positions: 16,
                max_batch_tokens: 4,
                session_capacity: 2,
                max_parameter_bytes: 1 << 30,
                precision: ExecutionPrecisionPolicy::f32(),
                max_ack_polls: 2,
            },
            &spec,
            StandardCudaTensorConfig {
                devices: (0..pp * 2).collect(),
                collective_limits: HostCollectiveLimits {
                    max_ranks: 2,
                    max_elements_per_rank: 32,
                    max_host_bytes: 4096,
                },
                collective_timeout: Duration::from_secs(2),
            },
            move |_| {
                owner_loads.fetch_add(1, Ordering::AcqRel);
                Err(ferrule_common::Error::Execution {
                    message: "injected loader stop before CUDA initialization".into(),
                })
            },
        );
        let message = result.err().expect("injected load failure").to_string();
        assert!(
            loads.load(Ordering::Acquire) > 0,
            "standard TP was rejected before loading: {message}"
        );
        assert!(
            message.contains("injected loader stop before CUDA initialization"),
            "{message}"
        );
        assert!(!message.contains("generic tensor program requires TP=1"));
    }
}
