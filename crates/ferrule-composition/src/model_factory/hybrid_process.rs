//! Dense hybrid process admission; execution remains the existing pipeline dispatcher.
use super::*;
use ferrule_runtime::parallel::process::decoder::{HybridBootFrame, HybridCudaBudgetFrame};
use std::sync::atomic::{AtomicU64, Ordering};

pub(super) struct Plan {
    pub(super) launch: ProcessLaunch,
    pub(super) boots: Vec<(ProcessIdentity, DecoderBoot)>,
    pub(super) options: ProcessOwnerConfig,
    pub(super) host_bytes: u64,
}
impl Plan {
    pub(super) fn new(
        request: &ResolvedModelRequest,
        resources: &BoundDecoderResources,
        plans: &[LayerSegmentPlan],
        config: PipelineConfig,
        stages: &[HybridStageBudget],
    ) -> Result<Self> {
        let placement = request
            .pipeline
            .as_ref()
            .ok_or_else(|| invalid("missing hybrid options"))?;
        placement.hybrid_precision(&request.family, request.backend)?;
        if placement.rank_backend != PipelineRankBackend::Process
            || plans.len() != 2
            || stages.len() != 2
        {
            return Err(invalid("hybrid process requires dense PP2 owners"));
        }
        let launch = placement
            .process_launch
            .clone()
            .ok_or_else(|| invalid("missing hybrid process launch"))?;
        let mut options = placement.process_owner_config();
        static NEXT_EPOCH: AtomicU64 = AtomicU64::new(1);
        let epoch = NEXT_EPOCH
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| invalid("hybrid process epoch exhausted"))?;
        let epoch = ProcessGroupEpoch::new(epoch).map_err(|e| invalid(e.to_string()))?;
        let size = |n: u64| {
            usize::try_from(n).map_err(|_| invalid("hybrid process budget exceeds address space"))
        };
        let mut boots = Vec::with_capacity(2);
        for (index, (plan, budget)) in plans.iter().zip(stages).enumerate() {
            let (device, cuda_budget) = match request.backend {
                ModelExecutionBackend::Cpu => (DecoderDevice::Cpu, None),
                ModelExecutionBackend::Cuda => (
                    DecoderDevice::Cuda {
                        ordinal: placement.device(index),
                    },
                    Some(HybridCudaBudgetFrame {
                        state_bytes: size(budget.state)?,
                        weight_bytes: size(budget.weights)?,
                        workspace_bytes: size(budget.workspace)?,
                        margin_bytes: size(budget.margin)?,
                    }),
                ),
            };
            let boot = DecoderBoot {
                version: DECODER_WIRE_VERSION,
                rank: index as u32,
                checkpoint: request.model_path.clone(),
                recipe: DecoderRecipeKind::Qwen35DenseHybrid,
                segment: SegmentFrame::encode(plan),
                precision: DecoderPrecision::F32,
                device,
                kv: KvConfigFrame::encode_for_hybrid(config, device)?,
                experts: None,
                hybrid: Some(HybridBootFrame::for_resources(
                    resources,
                    plan,
                    cuda_budget,
                )?),
            };
            boot.stage_boot()?;
            options.frame_limits = boot.fit_frame_limits(options.frame_limits)?;
            let bytes = serde_json::to_vec(&boot)
                .map_err(|_| invalid("hybrid boot serialization failed"))?
                .len();
            if bytes > options.frame_limits.max_config_bytes
                || bytes.checked_add(1024).is_none_or(|n| {
                    n > options.frame_limits.max_frame_bytes
                        || n > options.frame_limits.max_output_bytes
                })
            {
                return Err(invalid(
                    "hybrid boot/description exceeds negotiated frame limits",
                ));
            }
            let identity = ProcessIdentity::new(
                epoch,
                ParallelRankId::new(index as u32),
                ProcessOwnerInstanceId::new(index as u64 + 1)
                    .map_err(|e| invalid(e.to_string()))?,
            );
            boots.push((identity, boot));
        }
        for (_, boot) in &boots {
            boot.validate_frame_limits(options.frame_limits)?;
        }
        options.validate().map_err(|e| invalid(e.to_string()))?;
        let host_bytes = process_host_bytes(
            HybridSegmentMemoryEstimate::binding_metadata_bytes(resources)? as u64,
            options.frame_limits.max_frame_bytes as u64,
            boots.len() as u64,
        )?;
        Ok(Self {
            launch,
            boots,
            options,
            host_bytes,
        })
    }
}

fn process_host_bytes(binding_metadata: u64, frame_bytes: u64, owners: u64) -> Result<u64> {
    // Additional to the shared thread ledger: independent strict inventories,
    // catalogs, allocator/stack overhead, JSON DOMs, decode/encode frame copies.
    // Charge all child ceilings simultaneously, plus parent IPC staging. A small
    // live frame or shared checkpoint pathname does not discount these ceilings.
    let children = crate::model_factory::parallel::budget::process_child_overhead(
        binding_metadata,
        frame_bytes,
        owners,
    )?;
    add(children, mul(frame_bytes, 16)?)
}

#[cfg(test)]
#[path = "../../tests/unit/engine/model_factory/hybrid_process.rs"]
mod tests;
