//! Actual grouped-versus-synchronous owner lifecycle and routing contracts.
use super::*;
use crate::transformer::ExpertCacheLimits;
use crate::transformer::expert_reads::tests::Fixture;

#[test]
#[ignore = "requires GPU-owner authorization; actual production miss/consume/evict lifecycle"]
fn real_owner_grouped_matches_synchronous_and_reloads_only_after_consumer_proof() {
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
        assert_eq!(
            owner.expert_transfer_policy(),
            ExpertTransferPolicy::Synchronous
        );
        owner.set_expert_transfer_policy(policy).unwrap();
        assert_eq!(owner.expert_transfer_policy(), policy);
        let mut materializer = StateDictMaterializer::new(128).unwrap();
        materializer
            .configure_expert_read_pipeline(options(schedule))
            .unwrap();
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
        panic!("metadata-less grouped route must reject before payload materialization")
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
    let mut owner = routing_owner(crate::transformer::ExpertTransferPolicy::GroupedSynchronous);
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

#[test]
#[ignore = "GPU-owner authorization only; actual worker routing preflight, zero payload reads"]
fn real_worker_route_rejects_metadata_less_provider_before_payload_read() {
    use crate::transformer::CudaExpertWorker;
    use crate::transformer::expert_parallel::{ExpertPlacement, ExpertToken, ExpertWorker};
    use ferrule_common::{ParallelRankId, execution::ExecutionTransactionId};
    let mut owner = routing_owner(crate::transformer::ExpertTransferPolicy::GroupedSynchronous);
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
