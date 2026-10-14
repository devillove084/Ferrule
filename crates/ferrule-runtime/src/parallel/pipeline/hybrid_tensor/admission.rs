//! Payload-free checkpoint seal and actual CPU storage admission.
use super::*;
use crate::parallel::pipeline::PhysicalKvLayout;

pub(super) fn admit_owner(
    resources: &BoundDecoderResources,
    segment: &LayerSegmentPlan,
    parallel: &HybridParallelPlan,
    lane: usize,
    config: PipelineConfig,
    options: &HybridCpuTensorConfig,
) -> Result<()> {
    let layout = parallel.rank_layout(lane)?;
    let weights = layout.weight_budget(resources, segment)?;
    let required = layout
        .state_bytes()?
        .checked_mul(
            config
                .session_capacity
                .checked_add(1)
                .ok_or_else(|| error("state capacity overflow"))?,
        )
        .and_then(|n| n.checked_add(weights.resident_bytes))
        .and_then(|n| {
            layout
                .kv_planes(config.page_size, config.max_positions)
                .ok()?
                .checked_page_bytes()?
                .checked_mul(
                    config
                        .physical_pages(PhysicalKvLayout::hybrid(
                            ferrule_model::ModelExecutionBackend::Cpu,
                        ))
                        .ok()?,
                )?
                .checked_add(n)
        })
        .ok_or_else(|| error("hybrid owner residency overflow"))?;
    if required > options.max_owner_resident_bytes
        || weights.max_parameter_bytes as u64 > config.max_parameter_bytes
    {
        return Err(error("hybrid owner local residency exceeds budget"));
    }
    Ok(())
}

use ferrule_model::TensorRole;
use ferrule_model::checkpoint::CheckpointDType;
use ferrule_model::nn::{ParameterResidency, StorageEncoding};
use ferrule_model::transformer::{BoundParameter, ExpertMetadata};

/// Bound parameters contain only catalog metadata and immutable file snapshots,
/// never payloads. Keep parameter-to-source mappings, not merely a set of files.
/// Spec equality includes IDs, paths, residency, dtype/shape and alias metadata;
/// part equality includes slice/range, transforms and weight/scale snapshots.
#[derive(Clone)]
pub(super) struct HybridCheckpointBinding(Vec<BoundParameter>);
impl HybridCheckpointBinding {
    fn capture(resources: &BoundDecoderResources) -> Self {
        let mut parameters = resources.state_dict().parameters().to_vec();
        parameters.sort_by_key(|p| p.id());
        Self(parameters)
    }
    fn matches(&self, other: &Self) -> bool {
        self.0.len() == other.0.len()
            && self.0.iter().zip(&other.0).all(|(a, b)| {
                a.spec() == b.spec()
                    && a.role() == b.role()
                    && a.canonical_id() == b.canonical_id()
                    && a.weight() == b.weight()
                    && a.scale() == b.scale()
            })
    }
}

/// The first successful owner load seals the startup cohort. Every later load
/// must match that same seal before preparing any local image or spawning EP.
/// Self freshness checks supplement, never replace, this cross-owner comparison.
pub(super) fn seal_binding(
    seal: &Mutex<Option<HybridCheckpointBinding>>,
    resources: &BoundDecoderResources,
) -> Result<()> {
    if !resources.state_dict().validate_source_identities() {
        return Err(error("hybrid checkpoint source snapshot is stale"));
    }
    let binding = HybridCheckpointBinding::capture(resources);
    let mut seal = seal
        .lock()
        .map_err(|_| error("hybrid checkpoint seal poisoned"))?;
    match seal.as_ref() {
        Some(expected) if !expected.matches(&binding) => {
            Err(error("hybrid checkpoint binding differs from startup seal"))
        }
        Some(_) => Ok(()),
        None => {
            *seal = Some(binding);
            Ok(())
        }
    }
}

pub(super) fn reject_cpu_numeric_fp8(
    resources: &BoundDecoderResources,
    layers: std::ops::Range<usize>,
) -> Result<()> {
    for parameter in resources.state_dict().parameters() {
        if matches!(parameter.residency(), ParameterResidency::Expert { layer, .. } if layers.contains(layer))
            && (parameter.numeric_fp8_encoding().is_some()
                || parameter.weight().slice().dtype == CheckpointDType::F8E4M3)
        {
            return Err(error(
                "hybrid CPU EP numeric FP8 is unsupported before materialization",
            ));
        }
    }
    Ok(())
}

/// PreparedParameter owns the original dense bytes and PreparedLinear::Native
/// deep-clones those bytes. BF16 stays BF16 in both storages, not F32. CpuLocal
/// TP/shared matrices instead share an Arc and are charged by rank_layout above.
fn native_linear_resident_bytes(parameter: &BoundParameter) -> Result<usize> {
    if parameter.scale().is_some()
        || !matches!(
            parameter.weight().slice().dtype,
            CheckpointDType::F32 | CheckpointDType::Bf16
        )
        || parameter.weight().encoding() != StorageEncoding::Dense
    {
        return Err(error(
            "hybrid CPU expert requires unscaled dense F32/BF16 storage",
        ));
    }
    usize::try_from(parameter.weight().slice().bytes)
        .ok()
        .and_then(|bytes| bytes.checked_mul(2))
        .ok_or_else(|| error("expert residency overflow"))
}

pub(super) fn expert_resident_bytes(
    resources: &BoundDecoderResources,
    group: &ExpertGroup,
    owner: ParallelRankId,
) -> Result<usize> {
    reject_cpu_numeric_fp8(resources, group.layers.clone())?;
    let mut bytes = 0usize;
    for layer in group.layers.clone() {
        if let FeedForward::Moe(moe) = resources.spec().layers()[layer].feed_forward() {
            for expert in 0..moe.router_spec().num_experts() {
                if group.placement.owner(layer, expert) != Some(owner) {
                    continue;
                }
                let bindings = [
                    (
                        TensorRole::RoutedExpertGate,
                        moe.expert().gate().weight_shape(),
                    ),
                    (TensorRole::RoutedExpertUp, moe.expert().up().weight_shape()),
                    (
                        TensorRole::RoutedExpertDown,
                        moe.expert().down().weight_shape(),
                    ),
                ]
                .map(|(role, shape)| {
                    resources
                        .experts()
                        .require_shape(layer, expert, role, &shape)
                        .cloned()
                });
                let [gate, up, down] = bindings;
                let parameters = [gate?, up?, down?];
                // Use the shared metadata-only support contract. Its device byte
                // estimate is NOT the retained Native CPU storage ledger.
                ExpertMetadata::new(
                    layer,
                    expert,
                    parameters.clone(),
                    moe.expert().activation_limit(),
                )?;
                for parameter in &parameters {
                    bytes = bytes
                        .checked_add(native_linear_resident_bytes(parameter)?)
                        .ok_or_else(|| error("expert residency overflow"))?;
                }
            }
        }
    }
    Ok(bytes)
}

pub(super) fn admit_expert(
    resources: &BoundDecoderResources,
    group: &ExpertGroup,
    owner: ParallelRankId,
    limit: usize,
) -> Result<()> {
    if expert_resident_bytes(resources, group, owner)? > limit {
        return Err(error("hybrid expert weights exceed residency budget"));
    }
    Ok(())
}
