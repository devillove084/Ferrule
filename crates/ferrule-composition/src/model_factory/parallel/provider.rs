//! Owner-local CUDA provider preparation; no admission, engine or transaction loop.
use super::*;
use super::{budget::Admission, experts};
use ferrule_model::decoder::{HybridCudaDevice, HybridCudaMemoryBudget, HybridCudaRoutedExecutor};
use ferrule_runtime::parallel::{expert::ExpertGroup, pipeline::PipelineRank};
use std::sync::Arc;

pub(super) struct PreparedOwner {
    pub(super) device: HybridCudaDevice,
    pub(super) routed: Option<(ExpertGroup, Box<dyn HybridCudaRoutedExecutor>)>,
}

pub(super) fn prepare_owner(
    rank: PipelineRank,
    stage: usize,
    resources: &Arc<BoundDecoderResources>,
    admission: &Admission,
    timeout: std::time::Duration,
) -> ferrule_common::Result<PreparedOwner> {
    if !resources.state_dict().validate_source_identities() {
        return Err(ferrule_common::Error::Execution {
            message: "parallel checkpoint seal is stale".into(),
        });
    }
    admission
        .plan
        .validate_execution()
        .map_err(|error| ferrule_common::Error::Execution {
            message: error.to_string(),
        })?;
    let root = &admission.budgets[rank.local.get() as usize];
    let plan = &admission.plan;
    let device = prepare_device(resources, plan.config(), root)?;
    let routed = if let Some(owner) = plan.external_owner(stage)
        && rank.global == owner.group.source_rank
    {
        let devices = owner
            .group
            .members
            .iter()
            .map(|r| plan.devices[r.get() as usize])
            .collect();
        Some((
            owner.group.clone(),
            experts::prepare_attachment(
                owner,
                Arc::clone(resources),
                devices,
                plan.config(),
                plan.profile(),
                timeout,
                &device,
            )?,
        ))
    } else {
        None
    };
    Ok(PreparedOwner { device, routed })
}

/// Shared persistent-owner device leaf. Neither profile selection nor environment
/// mutation belongs here; callers have already admitted the complete ledger.
pub(super) fn prepare_device(
    resources: &BoundDecoderResources,
    config: PipelineConfig,
    root: &HybridStageBudget,
) -> ferrule_common::Result<HybridCudaDevice> {
    if !resources.state_dict().validate_source_identities() {
        return Err(ferrule_common::Error::Execution {
            message: "parallel checkpoint seal is stale".into(),
        });
    }
    let size = |n| {
        usize::try_from(n).map_err(|_| ferrule_common::Error::Execution {
            message: "owner budget exceeds address space".into(),
        })
    };
    HybridCudaDevice::new_on_device(
        root.device,
        HybridCudaMemoryBudget {
            kv_pages: config
                .physical_pages(PhysicalKvLayout::hybrid(ModelExecutionBackend::Cuda))?,
            state_bytes: size(root.state)?,
            weight_bytes: size(root.weights)?,
            workspace_bytes: size(root.workspace)?,
        },
    )
}
