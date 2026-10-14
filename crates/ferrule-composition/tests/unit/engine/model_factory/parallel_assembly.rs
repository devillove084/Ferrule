//! Cross-cutting assembly regressions; no GPU initialization or alternate executor.
use super::assembly::PipelineAssembly;
use super::plan::Geometry;
use super::*;
use crate::model_factory::qwen35::{Qwen35ResolvedAdapter, ep::tests::Fixture};
use crate::model_factory::*;
use std::{cell::RefCell, rc::Rc};

fn request(f: &Fixture) -> ResolvedModelRequest {
    let mut request = f.request();
    request.max_layers = 4;
    if let Qwen35ResolvedAdapter::Moe {
        expert_placement, ..
    } = request.family_options.qwen35_mut().unwrap()
    {
        *expert_placement = None;
    }
    request.pipeline = Some(PipelineBuildOptions {
        parallelism: ferrule_common::ParallelismPlan {
            pipeline_parallel: 2,
            expert_parallel: 2,
            ..Default::default()
        },
        devices: Some(vec![0, 1, 0, 1, 0, 1]),
        ..Default::default()
    });
    request
}

#[test]
fn geometry_keeps_colocated_roots_and_experts_distinct_for_every_stage() {
    for (pp, tp, ep) in [(2, 1, 1), (2, 1, 4), (1, 2, 2), (2, 2, 2), (2, 4, 2)] {
        let count = pp * (tp + if ep == 1 { 0 } else { ep });
        let options = PipelineBuildOptions {
            parallelism: ferrule_common::ParallelismPlan {
                pipeline_parallel: pp,
                tensor_parallel: tp,
                expert_parallel: ep,
                ..Default::default()
            },
            devices: Some((0..count).map(|owner| owner % 2).collect()),
            ..Default::default()
        };
        let geometry = Geometry::new(4, &options).unwrap();
        assert_eq!(geometry.roots, pp * tp);
        assert_eq!(geometry.devices.len(), count);
        let mut owners = std::collections::BTreeSet::new();
        for stage in 0..pp {
            assert_eq!(
                geometry.root_members(stage).unwrap()[0].get() as usize,
                stage * tp
            );
            for rank in geometry
                .root_members(stage)
                .unwrap()
                .into_iter()
                .chain(geometry.expert_members(stage).unwrap())
            {
                assert!(owners.insert(rank));
            }
        }
        assert_eq!(owners.len(), count);
        assert!(geometry.root_members(pp).is_err());
        assert!(geometry.expert_members(pp).is_err());
    }
}

#[test]
fn geometry_rejects_invalid_partition_and_owner_abi_before_allocation() {
    let mut options = PipelineBuildOptions::default();
    assert!(Geometry::new(0, &options).is_err());
    options.parallelism.pipeline_parallel = 5;
    assert!(Geometry::new(4, &options).is_err());
    options.parallelism.pipeline_parallel = 1;
    options.parallelism.tensor_parallel = usize::MAX;
    assert!(Geometry::new(4, &options).is_err());
}

struct AssemblyProbe {
    fail: &'static str,
    events: Rc<RefCell<Vec<&'static str>>>,
    kv: ResidentKvPagePlan,
}
impl PipelineAssembly for AssemblyProbe {
    fn kv(&self) -> ResidentKvPagePlan {
        self.kv
    }
    fn admit(&self, _: &ResolvedModelRequest, _: &BoundDecoderResources) -> Result<()> {
        self.events.borrow_mut().push("admit");
        if self.fail == "admit" {
            return Err(invalid("admission refused"));
        }
        Ok(())
    }
    fn prepare_host(&self, resources: BoundDecoderResources) -> Result<BoundDecoderResources> {
        self.events.borrow_mut().push("host");
        if self.fail == "host" {
            return Err(invalid("host preparation refused"));
        }
        Ok(resources)
    }
    fn start(
        self,
        _: &ResolvedModelRequest,
        _: BoundDecoderResources,
    ) -> Result<PipelineParallelExecutor> {
        self.events.borrow_mut().push("start");
        Err(invalid("owner startup refused"))
    }
}

#[test]
fn shared_assembly_orders_admission_host_and_startup_and_stops_on_failure() {
    let f = Fixture::with_layers(4);
    let r = request(&f);
    let plan = crate::model_factory::qwen35::binding::pipeline_plan(&r, &f.resources).unwrap();
    for (fail, expected) in [
        ("admit", vec!["admit"]),
        ("host", vec!["admit", "host"]),
        ("start", vec!["admit", "host", "start"]),
    ] {
        let events = Rc::new(RefCell::new(Vec::new()));
        let result = assembly::build_bound(
            r.clone(),
            f.resources.clone(),
            TokenizerHandle::load(&f.dir).unwrap(),
            AssemblyProbe {
                fail,
                events: Rc::clone(&events),
                kv: plan.kv,
            },
        );
        assert!(result.is_err());
        assert_eq!(*events.borrow(), expected);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn shared_ready_rejects_wrong_owner_payload_uploads_and_unknown_custody_without_gpu() {
    use ferrule_model::{moe::ExpertId, transformer::ExpertCacheStats};
    use ferrule_runtime::parallel::expert::{ExpertOwnerStats, ExpertRank};
    let f = Fixture::with_layers(4);
    let plan =
        crate::model_factory::qwen35::binding::pipeline_plan(&request(&f), &f.resources).unwrap();
    let owner = &plan.owners[0];
    let stats = || {
        owner
            .group
            .members
            .iter()
            .enumerate()
            .map(|(slot, member)| ExpertOwnerStats {
                rank: ExpertRank {
                    slot: ParallelRankId::new(slot as u32),
                    owner: *member,
                },
                thread: std::thread::current().id(),
                calls: 0,
                tokens: 0,
                last_context: None,
                owned_experts: owner
                    .group
                    .layers
                    .clone()
                    .flat_map(|layer| {
                        (0..4).filter_map(move |expert| {
                            (owner.group.placement.owner(layer, expert) == Some(*member))
                                .then_some(ExpertId::new(layer, expert))
                        })
                    })
                    .collect(),
                cuda_cache: Some(ExpertCacheStats {
                    resident_experts: owner.counts[slot],
                    resident_bytes: owner.weights[slot],
                    uploads: owner.counts[slot] as u64,
                    ..Default::default()
                }),
            })
            .collect::<Vec<_>>()
    };
    experts::check_ready(owner, &stats()).unwrap();
    let mutations: [fn(&mut ExpertOwnerStats); 7] = [
        |s| s.rank.owner = ParallelRankId::new(99),
        |s| s.rank.slot = ParallelRankId::new(99),
        |s| s.owned_experts[0] = ExpertId::new(99, 99),
        |s| s.cuda_cache.as_mut().unwrap().resident_bytes += 1,
        |s| s.cuda_cache.as_mut().unwrap().uploads += 1,
        |s| s.cuda_cache.as_mut().unwrap().pending_upload_bytes = 1,
        |s| s.cuda_cache.as_mut().unwrap().quarantined = true,
    ];
    for mutate in mutations {
        let mut wrong = stats();
        mutate(&mut wrong[0]);
        assert!(experts::check_ready(owner, &wrong).is_err());
    }
}
