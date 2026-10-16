//! Exact production owner/lease identity gates, no CUDA context or DMA.
use super::*;
use crate::transformer::expert_reads::tests::{Fixture, IMAGE};
use crate::transformer::standard::expert_cache::Cache;
use crate::transformer::{ExpertCacheLimits, NumericFp8Precision};

fn cache_and_key() -> (Cache<ExpertCacheKey>, ExpertCacheKey) {
    let f = Fixture::new(1);
    let key = ExpertCacheKey {
        image_generation: IMAGE,
        numeric_precision: Some(NumericFp8Precision::Bf16RneF32Accumulate),
        layer: 0,
        expert: 0,
        parameters: f.metadata[0]
            .parameters()
            .each_ref()
            .map(|p| (p.canonical_id(), p.role().clone())),
    };
    (
        Cache::new(ExpertCacheLimits {
            max_experts: 1,
            max_bytes: 64,
        })
        .unwrap(),
        key,
    )
}

#[test]
fn exact_ready_identity_rejects_stale_generation_lease_image_and_foreign_owner() {
    let (mut cache, key) = cache_and_key();
    let lease = cache.acquire(key.clone(), 18, |_| {}).unwrap();
    let identity = Identity {
        lease: lease.clone(),
        ticket_generation: 7,
    };
    assert!(
        identity
            .validate(Some(&lease), Some(&key), IMAGE, 7)
            .is_ok()
    );
    for generation in [0, 6, 8] {
        assert!(
            identity
                .validate(Some(&lease), Some(&key), IMAGE, generation)
                .is_err()
        );
    }
    assert!(
        identity
            .validate(Some(&lease), Some(&key), IMAGE + 1, 7)
            .is_err()
    );
    assert!(identity.validate(None, Some(&key), IMAGE, 7).is_err());
    assert!(identity.validate(Some(&lease), None, IMAGE, 7).is_err());
    cache.complete(&lease, true).unwrap();
    let replacement = cache.acquire(key.clone(), 18, |_| {}).unwrap();
    assert!(
        identity
            .validate(Some(&replacement), Some(&key), IMAGE, 7)
            .is_err()
    );
    let (mut other, _) = cache_and_key();
    let foreign = other.acquire(key.clone(), 18, |_| {}).unwrap();
    assert!(
        identity
            .validate(Some(&foreign), Some(&key), IMAGE, 7)
            .is_err()
    );
}

#[test]
fn one_owner_rejects_duplicate_lease_and_unknown_never_reopens() {
    let (mut cache, key) = cache_and_key();
    let lease = cache.acquire(key, 18, |_| {}).unwrap();
    let mut state = State::new(4096, Duration::from_secs(1));
    state.begin_lease(&lease).unwrap();
    assert!(state.begin_lease(&lease).is_err());
    state.finish_lease().unwrap();
    state.begin_lease(&lease).unwrap();
    state.quarantine();
    assert!(state.finish_lease().is_err());
    assert!(state.begin_lease(&lease).is_err());
}

#[test]
#[ignore = "requires GPU-owner authorization; actual production miss/consume/evict lifecycle"]
fn real_owner_opt_in_matches_default_and_reloads_only_after_consumer_proof() {
    use crate::transformer::expert_reads::tests::options;
    use crate::transformer::standard::{ExpertSemantics, StateDictExpertProvider};
    use crate::transformer::{
        ExpertCachePolicy, ExpertReadSchedule, ExpertTransferControl, ExpertTransferPolicy,
        HostRows, Rows, RowsDType, RowsShape, StandardDecoderOperators, StateDictMaterializer,
    };
    use ferrule_backend::cuda::operators::linear::CudaOperators;
    use std::rc::Rc;
    let f = Fixture::new(2);
    let parameters = f
        .metadata
        .iter()
        .flat_map(|m| m.parameters().iter().cloned())
        .collect::<Vec<_>>();
    let semantics = f
        .metadata
        .iter()
        .map(ExpertSemantics::read_test_fixture)
        .collect::<Vec<_>>();
    let mut outputs = Vec::new();
    for (schedule, policy) in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead]
        .into_iter()
        .flat_map(|schedule| {
            [
                ExpertTransferPolicy::Synchronous,
                ExpertTransferPolicy::Retained {
                    policy_headroom_bytes: 4096,
                    readiness_timeout: Duration::from_secs(1),
                },
                ExpertTransferPolicy::GroupedSynchronous,
            ]
            .map(|policy| (schedule, policy))
        })
    {
        let mut owner = CudaStandardDecoderOperators::new_numeric_fp8(
            Rc::new(CudaOperators::new_on_device(0).unwrap()),
            &parameters,
            ExpertCachePolicy::Bounded(ExpertCacheLimits {
                max_experts: 1,
                max_bytes: 8192,
            }),
            1024,
        )
        .unwrap();
        assert!(!owner.retained_numeric_fp8_upload_enabled());
        owner.set_expert_transfer_policy(policy).unwrap();
        assert_eq!(owner.expert_transfer_policy(), policy);
        let mut materializer = StateDictMaterializer::new(128).unwrap();
        let mut limits = options(schedule);
        limits.max_host_bytes += 8192;
        materializer.configure_expert_read_pipeline(limits).unwrap();
        let mut provider = StateDictExpertProvider {
            layer: 0,
            bindings: &semantics,
            materializer: &materializer,
            reads: None,
        };
        provider
            .preflight_expert_reads(
                owner.image_generation(),
                &f.metadata.iter().collect::<Vec<_>>(),
            )
            .unwrap();
        let mut values = Vec::new();
        for id in [0, 1, 0] {
            let input = owner
                .bind_rows(Rows::Host(
                    HostRows::new(
                        RowsShape::new(1, 2).unwrap(),
                        RowsDType::F32,
                        None,
                        vec![1.0, 2.0],
                    )
                    .unwrap(),
                ))
                .unwrap();
            let result = owner
                .synchronous(|this| {
                    this.metadata_swiglu_rows(&f.metadata[id], &mut provider, &input, None)
                })
                .unwrap();
            values.push(owner.download_rows(result).unwrap().values().to_vec());
            assert!(!owner.needs_quarantine());
            assert!(owner.bindings.has_metadata_expert(&f.metadata[id]).unwrap());
            let observed = materializer.expert_read_pipeline_observation();
            assert_eq!(observed.current_charge.pinned_upload_bytes, 0);
            assert_eq!(observed.current_charge.packed_bytes, 0);
        }
        assert_eq!(owner.expert_cache_stats().unwrap().uploads, 3);
        assert_eq!(owner.expert_cache_stats().unwrap().evictions, 2);
        outputs.push(values);
    }
    for values in &outputs[1..] {
        assert_eq!(&outputs[0], values);
    }
}

struct MetadataLessProvider {
    reads: usize,
}
impl crate::transformer::ExpertProvider for MetadataLessProvider {
    fn expert(&mut self, _: usize, _: usize) -> Result<ExpertAvailability> {
        self.reads += 1;
        panic!("metadata-less retained route must reject before payload materialization")
    }
}

fn routing_owner(policy: crate::transformer::ExpertTransferPolicy) -> CudaStandardDecoderOperators {
    use crate::transformer::ExpertCachePolicy;
    let f = Fixture::new(1);
    let mut owner = CudaStandardDecoderOperators::new_numeric_fp8(
        std::rc::Rc::new(
            ferrule_backend::cuda::operators::linear::CudaOperators::new_on_device(0).unwrap(),
        ),
        f.bound.parameters(),
        ExpertCachePolicy::Bounded(ExpertCacheLimits {
            max_experts: 1,
            max_bytes: 8192,
        }),
        1024,
    )
    .unwrap();
    crate::transformer::ExpertTransferControl::set_expert_transfer_policy(&mut owner, policy)
        .unwrap();
    owner
}

#[test]
#[ignore = "GPU-owner authorization only; actual root routing preflight, zero payload reads"]
fn real_root_route_rejects_metadata_less_provider_before_payload_read() {
    use crate::transformer::{HostRows, RouterRoutes, Rows, RowsDType, RowsShape};
    for policy in [
        crate::transformer::ExpertTransferPolicy::Retained {
            policy_headroom_bytes: 16 << 20,
            readiness_timeout: Duration::from_secs(30),
        },
        crate::transformer::ExpertTransferPolicy::GroupedSynchronous,
    ] {
        let mut owner = routing_owner(policy);
        let mut provider = MetadataLessProvider { reads: 0 };
        let input = Rows::Host(
            HostRows::new(
                RowsShape::new(1, 2).unwrap(),
                RowsDType::F32,
                None,
                vec![1.0, 2.0],
            )
            .unwrap(),
        );
        let routes = RouterRoutes::new(1, 1, vec![0], vec![1.0]).unwrap();
        let error = owner
            .routed_swiglu_batched(0, &input, &routes, &mut provider, None)
            .unwrap_err();
        assert!(error.to_string().contains("metadata before payload reads"));
        assert_eq!(provider.reads, 0);
        assert_eq!(owner.expert_cache_stats().unwrap().uploads, 0);
    }
}

#[test]
#[ignore = "GPU-owner authorization only; actual worker routing preflight, zero payload reads"]
fn real_worker_route_rejects_metadata_less_provider_before_payload_read() {
    use crate::transformer::CudaExpertWorker;
    use crate::transformer::expert_parallel::{ExpertPlacement, ExpertToken, ExpertWorker};
    use ferrule_common::{ParallelRankId, execution::ExecutionTransactionId};
    for policy in [
        crate::transformer::ExpertTransferPolicy::Retained {
            policy_headroom_bytes: 16 << 20,
            readiness_timeout: Duration::from_secs(30),
        },
        crate::transformer::ExpertTransferPolicy::GroupedSynchronous,
    ] {
        let mut owner = routing_owner(policy);
        let mut provider = MetadataLessProvider { reads: 0 };
        let rank = ParallelRankId::new(0);
        let placement = ExpertPlacement::new([(0, 0, rank)]).unwrap();
        let token = ExpertToken {
            transaction: ExecutionTransactionId::new(1).unwrap(),
            sequence: 1,
            source_rank: rank,
            source_row: 0,
            route_slot: 0,
            expert: crate::moe::ExpertId::new(0, 0),
            weight: 1.0,
            payload: vec![1.0, 2.0],
        };
        let error = CudaExpertWorker::new(rank, &placement, &mut provider, &mut owner)
            .compute(&[token])
            .unwrap_err();
        assert!(error.to_string().contains("metadata before payload reads"));
        assert_eq!(provider.reads, 0);
        assert_eq!(owner.expert_cache_stats().unwrap().uploads, 0);
    }
}
