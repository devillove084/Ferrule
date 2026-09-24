//! Production dense standard CUDA PP×TP construction. Only host metadata is
//! captured; each physical owner loads its checkpoint shards and creates CUDA.

use std::collections::BTreeSet;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ferrule_backend::cuda::operators::linear::CudaOperators;
use ferrule_common::{ParallelGroupId, ParallelRankId, Result, ValidatedParallelTopology};
use ferrule_model::transformer::{
    BoundDecoderResources, DecoderModelSpec, LayerSegmentPlan, StandardTensorPlacement,
    StandardTensorPlan,
};

use super::tensor_scope::{TensorCollectiveSite, TensorScopedProgram};
use super::{PipelineConfig, PipelineParallelExecutor, PipelineRank, PipelineStage, error};
use crate::parallel::collective::HostCollectiveLimits;
use crate::parallel::tensor::decoder_collective::DecoderTensorCollective;

/// Explicit resource limits for the bounded host-staged TP collective. Device
/// order is global mesh order: stage * TP + tensor. Devices cannot be shared.
#[derive(Debug, Clone)]
pub struct StandardCudaTensorConfig {
    pub devices: Vec<usize>,
    pub collective_limits: HostCollectiveLimits,
    pub collective_timeout: Duration,
}

impl PipelineParallelExecutor {
    /// Serving factories may call this directly and retain the returned pipeline
    /// as their sole KV owner/decision authority. No external orchestration is
    /// needed. `load` runs on every persistent physical owner, before CUDA init;
    /// capture paths/options, not prepared device objects or full weights.
    ///
    /// Dense F32 only, TP1/2/4, DP1, EP1. PP plans are complete-layer segments.
    /// Process TP and expert injection are deliberately not admitted here.
    #[allow(clippy::too_many_arguments)]
    pub fn new_standard_cuda_tensor<F>(
        topology: ValidatedParallelTopology,
        plans: Vec<LayerSegmentPlan>,
        config: PipelineConfig,
        spec: &DecoderModelSpec,
        tensor_config: StandardCudaTensorConfig,
        load: F,
    ) -> Result<Self>
    where
        F: FnOnce(PipelineRank) -> Result<BoundDecoderResources> + Clone + Send + 'static,
    {
        super::validate_pipeline_layout(&topology, &plans, config, true)?;
        let tp = topology.plan().tensor_parallel;
        if tensor_config.devices.len() != topology.world_size() as usize
            || tensor_config.devices.iter().collect::<BTreeSet<_>>().len()
                != tensor_config.devices.len()
        {
            return Err(error(
                "standard tensor pipeline requires one distinct device per physical owner",
            ));
        }
        let scopes = topology
            .execution_scopes(0)
            .map_err(|e| error(format!("tensor pipeline scope: {e:?}")))?;
        let mut tensor_plans = Vec::new();
        let mut controls = Vec::new();
        let mut endpoints = Vec::new();
        for (stage, plan) in plans.iter().enumerate() {
            let members = scopes
                .tensor_collective_participants(stage as u32, 0)
                .map_err(|e| error(format!("tensor stage scope: {e:?}")))?;
            let tensor = StandardTensorPlan::new(
                spec,
                members
                    .iter()
                    .map(|owner| StandardTensorPlacement {
                        owner,
                        device: tensor_config.devices[owner.get() as usize],
                    })
                    .collect(),
            )?;
            tensor.validate_segment(spec, plan)?;
            let group = DecoderTensorCollective::new_group(
                &tensor,
                topology.topology_id(),
                ParallelGroupId::new(stage as u32 + 1),
                tensor_config.collective_limits,
                tensor_config.collective_timeout,
            )?;
            controls.push(group[0].control());
            endpoints.extend(group.into_iter().map(Some));
            tensor_plans.push(tensor);
        }
        let endpoints = Arc::new(Mutex::new(endpoints));
        let tensor_plans = Arc::new(tensor_plans);
        let expected = spec.clone();
        let mut pipeline = Self::new_thread_tensor_with_program_inner(
            topology,
            plans,
            config,
            move |owner, coordinate, plan| {
                let resources = load(owner)?;
                if resources.spec() != &expected {
                    return Err(error(
                        "tensor owner checkpoint specification changed after preflight",
                    ));
                }
                let tensor = tensor_plans[coordinate.stage() as usize].clone();
                let local = ParallelRankId::new(coordinate.tensor());
                let placement = tensor.placement(local)?;
                if placement.owner != owner.global
                    || owner.local.get() as usize / tp != coordinate.stage() as usize
                {
                    return Err(error("tensor owner mesh/placement mismatch"));
                }
                let endpoint = endpoints
                    .lock()
                    .map_err(|_| error("tensor endpoint startup poisoned"))?
                    [owner.local.get() as usize]
                    .take()
                    .ok_or_else(|| error("tensor endpoint already assigned"))?;
                let control = endpoint.control();
                control.register_sites(
                    owner.global,
                    TensorCollectiveSite::standard(&resources, &plan, tp)?,
                )?;
                let ops = Rc::new(CudaOperators::new_on_device(placement.device)?);
                PipelineStage::prepare_cuda_tensor(
                    &resources,
                    plan,
                    tensor,
                    local,
                    config,
                    ops,
                    Box::new(endpoint),
                )
                .map(|stage| {
                    stage.map_program(|program| {
                        TensorScopedProgram::new(program, control, owner.global)
                    })
                })
            },
        )?;
        pipeline.tensor_controls = controls;
        Ok(pipeline)
    }
}
