//! Real rank/PP/module exclusivity on the same device factory. No GPU execution
//! is run by host validation; these ignored tests compile the complete API.
use super::*;
use ferrule_common::Result;
use ferrule_model::transformer::parallel::TensorParallelCollective;

struct NoWire {
    members: Vec<ParallelRankId>,
    owner: ParallelRankId,
}
impl StandardTensorCollective for NoWire {
    fn owner(&self) -> ParallelRankId {
        self.owner
    }
    fn members(&self) -> &[ParallelRankId] {
        &self.members
    }
    fn exchange(
        &mut self,
        _: ExecutionTransactionId,
        _: u64,
        _: TensorParallelCollective,
        _: Vec<f32>,
    ) -> Result<Vec<f32>> {
        panic!("prepare performs no collective math")
    }
    fn abort(&mut self) {}
}

#[test]
#[ignore = "requires CUDA GPU; real TP1/2/4 prepare claims and cross-owner rejection"]
fn rank_claim_transfers_real_image_pool_state_and_excludes_pp_and_module() {
    let fixture = base::Fixture::for_degree(4, true);
    let resources = Arc::new(fixture.resources());
    let o = options(&resources);
    for degree in [1, 2, 4] {
        let members = [7, 19, 31, 63][..degree]
            .iter()
            .copied()
            .map(ParallelRankId::new)
            .collect::<Vec<_>>();
        let p = HybridParallelPlan::new(
            resources.spec(),
            0..4,
            HybridExpertScope {
                topology: ParallelTopologyId::new(907),
                replica: 0,
                stage: 0,
            },
            members.clone(),
            None,
        )
        .unwrap();
        let layout = Arc::new(p.rank_layout(0).unwrap());
        let m =
            CudaHybridRankMemory::for_rank(&resources, &segment(), &p, &layout, &o, 16).unwrap();
        let estimate = m.admission_estimate().unwrap();
        let device = HybridCudaDevice::new_on_device(
            0,
            HybridCudaMemoryBudget {
                kv_pages: 16,
                state_bytes: 3 * m.per_sequence_state_grants,
                weight_bytes: m.weight_grants,
                workspace_bytes: m.numeric_workspace_grants + m.operation_workspace_grants,
            },
        )
        .unwrap();
        let failed_claim = device.admit(estimate, 1).unwrap();
        let failure: Result<()> = (|| {
            let owner = CudaHybridRankSegment::prepare_rank_with_claim(
                &resources,
                segment(),
                p.clone(),
                layout.clone(),
                &o,
                device.clone(),
                Box::new(NoWire {
                    members: members.clone(),
                    owner: members[0],
                }),
                None,
                &failed_claim,
            )?;
            let mut lifecycle = owner.new_lifecycle()?;
            let _state = lifecycle.create()?;
            let _pool = TypedCudaPagedKvPool::<CudaHybridRankState>::from_strategy(
                device.operators().clone(),
                &owner.layout().kv_planes(2, 32)?,
                16,
            )?;
            Err(ferrule_common::Error::Model {
                message: "injected post-rank pool/state factory failure".into(),
            })
        })();
        assert!(failed_claim.finish(failure).is_err());
        assert_eq!(device.live_state_bytes(), 0);
        assert!(!device.needs_quarantine());
        let claim = device.admit(estimate, 1).unwrap();
        assert!(device.clone().admit(estimate, 1).is_err());
        let result = (|| {
            let owner = CudaHybridRankSegment::prepare_rank_with_claim(
                &resources,
                segment(),
                p.clone(),
                layout.clone(),
                &o,
                device.clone(),
                Box::new(NoWire {
                    members: members.clone(),
                    owner: members[0],
                }),
                None,
                &claim,
            )?;
            // Even the same genuine token is one image, not a reentrant permit.
            assert!(
                CudaHybridRankSegment::prepare_rank_with_claim(
                    &resources,
                    segment(),
                    p.clone(),
                    layout.clone(),
                    &o,
                    device.clone(),
                    Box::new(NoWire {
                        members: members.clone(),
                        owner: members[0]
                    }),
                    None,
                    &claim,
                )
                .is_err()
            );
            let mut lifecycle = owner.new_lifecycle()?;
            let state = lifecycle.create()?;
            let pool = TypedCudaPagedKvPool::<CudaHybridRankState>::from_strategy(
                device.operators().clone(),
                &owner.layout().kv_planes(2, 32)?,
                16,
            )?;
            Ok((owner, lifecycle, state, pool))
        })();
        let (mut owner, mut lifecycle, state, pool) = claim.finish(result).unwrap();
        assert!(device.admit(estimate, 1).is_err());
        assert!(
            CudaHybridDecoderSegment::prepare(
                &resources,
                segment(),
                ExecutionPrecisionPolicy::f32(),
                32,
                1 << 20,
                device.clone()
            )
            .is_err()
        );
        assert!(CudaHybridModule::prepare(resources.clone(), device.clone(), &o).is_err());
        assert!(!device.needs_quarantine());
        lifecycle.try_release(state).unwrap();
        drop(pool);
        owner.shutdown().unwrap();
        drop(owner);
        assert!(
            device.admit(estimate, 1).is_err(),
            "published factory never reopens"
        );
    }
}
