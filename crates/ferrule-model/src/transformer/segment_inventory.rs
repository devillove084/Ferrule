//! Semantic segment selection and metadata-only memory accounting. No provider layouts.

use super::{
    Attention, BoundDecoderResources, BoundParameter, DecoderLayer, FeedForward, LayerSegmentPlan,
};
use crate::checkpoint::CheckpointDType;
use crate::decoder::{GenericDecoderOptions, HybridStateSchema};
use crate::execution::ExecutionPrecisionPolicy;
use crate::nn::ParameterResidency;
use crate::support::TensorRole;
use ferrule_common::execution::KvLayoutSchema;
use ferrule_common::{Error, Result};
use std::collections::BTreeSet;

/// Logical views are kept intact; storage deduplication belongs to each ledger,
/// not selection. In particular an output-only owner retains a tied head view.
pub(super) struct HybridSegmentInventory<'a> {
    resources: &'a BoundDecoderResources,
    plan: &'a LayerSegmentPlan,
    schema: HybridStateSchema,
}
impl<'a> HybridSegmentInventory<'a> {
    pub(super) fn new(
        resources: &'a BoundDecoderResources,
        plan: &'a LayerSegmentPlan,
    ) -> Result<Self> {
        if plan.total_layers() != resources.spec().layers().len() {
            return Err(error("hybrid segment inventory layer count mismatch"));
        }
        Ok(Self {
            resources,
            plan,
            schema: HybridStateSchema::for_segment(resources.spec(), plan.layers())?,
        })
    }
    pub(super) fn schema(&self) -> &HybridStateSchema {
        &self.schema
    }
    pub(super) fn layers(&self) -> &[DecoderLayer] {
        &self.resources.spec().layers()[self.plan.layers()]
    }
    pub(super) fn parameters(&self) -> impl Iterator<Item = &'a BoundParameter> + '_ {
        self.resources
            .state_dict()
            .parameters()
            .iter()
            .filter(move |p| match p.residency() {
                ParameterResidency::Layer { layer } | ParameterResidency::Expert { layer, .. } => {
                    self.plan.layers().contains(layer)
                }
                ParameterResidency::Static => match p.role() {
                    TensorRole::TokenEmbedding => self.plan.owns_embedding(),
                    TensorRole::OutputNorm | TensorRole::OutputHead => self.plan.owns_output(),
                    _ => false,
                },
                ParameterResidency::Attachment { .. } => false,
            })
    }
}

/// Independent resident domains. CUDA owners retain `host` in addition to
/// `device`; these are never alternative representations of a single charge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HybridSegmentResidentMemory {
    /// Canonical source/decoded storage upper bound in this owner.
    pub parameter_bytes: usize,
    /// Host prepared views survive beside their sources, including tied views.
    /// Device vector/linear bindings have their own conservative allowance.
    pub binding_bytes: usize,
    pub rotary_bytes: usize,
    pub workspace_bytes: usize,
}
impl HybridSegmentResidentMemory {
    pub fn weight_bytes(&self) -> Result<usize> {
        add(
            add(self.parameter_bytes, self.binding_bytes)?,
            self.rotary_bytes,
        )
    }
}

/// Dense F32 TP1/EP1 segment units, before runtime capacity/placement policy.
/// State is per sequence; KV is per physical page (no implicit L/2L choice).
/// All fields are byte upper bounds; no payload reads or device allocations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridSegmentMemoryEstimate {
    schema: HybridStateSchema,
    execution_limits: [usize; 3],
    pub host: HybridSegmentResidentMemory,
    pub device: HybridSegmentResidentMemory,
    pub host_startup_temporary_bytes: usize,
    pub per_sequence_state_bytes: usize,
    pub per_page_kv_bytes: usize,
    /// One packed host boundary buffer; runtime chooses simultaneous copies.
    pub boundary_rows_bytes: usize,
}
impl HybridSegmentMemoryEstimate {
    pub fn for_segment(
        resources: &BoundDecoderResources,
        options: &GenericDecoderOptions,
        plan: &LayerSegmentPlan,
    ) -> Result<Self> {
        options.validate(resources)?;
        if options.precision() != ExecutionPrecisionPolicy::f32()
            || options.hybrid_cuda_numeric_fp8().is_some()
            || options
                .active_layers()
                .is_some_and(|n| n != plan.total_layers())
        {
            return Err(error(
                "hybrid segment estimate requires dense F32 full-depth options",
            ));
        }
        let inventory = HybridSegmentInventory::new(resources, plan)?;
        let mut canonical = BTreeSet::new();
        let mut parameter_bytes = 0;
        let mut host_bindings = 0;
        let mut largest_source = 0;
        let mut largest_conversion = 0;
        for p in inventory.parameters() {
            if p.numeric_fp8_encoding().is_some()
                || p.scale().is_some()
                || !matches!(
                    p.weight().slice().dtype,
                    CheckpointDType::F32 | CheckpointDType::Bf16
                )
            {
                return Err(error("dense hybrid storage must be unscaled F32/BF16"));
            }
            let elements = p
                .weight()
                .slice()
                .shape
                .iter()
                .try_fold(1, |n, &d| mul(n, d))?;
            let source = mul(
                elements,
                if p.weight().slice().dtype == CheckpointDType::F32 {
                    4
                } else {
                    2
                },
            )?;
            if source as u64 > options.max_parameter_bytes() {
                return Err(error("hybrid selected tensor exceeds max_parameter_bytes"));
            }
            let decoded = mul(elements, 4)?;
            // PreparedLinear clones native storage, while norms/GDN vectors keep
            // decoded values. Use F32 size for either without erasing alias views.
            host_bindings = add(host_bindings, decoded)?;
            if canonical.insert(p.canonical_id()) {
                parameter_bytes = add(parameter_bytes, decoded)?;
            }
            largest_source = largest_source.max(source);
            // CUDA dense upload can hold both an F32 Vec and serialized bytes.
            largest_conversion = largest_conversion.max(mul(decoded, 2)?);
        }
        let spec = resources.spec();
        let mut width = if plan.owns_output() {
            spec.hidden_size().max(spec.vocab_size())
        } else {
            spec.hidden_size()
        };
        let boundary_rows_bytes = mul(mul(width, options.capabilities().max_batch_tokens)?, 4)?;
        let mut scores = 0;
        let mut rotary_bytes = 0;
        for layer in inventory.layers() {
            match layer.attention() {
                Attention::Gqa(g) => {
                    let rotary = mul(
                        mul(options.max_positions(), g.rotary().region().dimensions())?,
                        4,
                    )?;
                    rotary_bytes = add(rotary_bytes, rotary)?;
                    largest_source = largest_source.max(rotary);
                    width = width.max(g.query().out_features());
                    scores = scores.max(mul(g.num_heads(), options.max_positions())?);
                }
                Attention::GatedDeltaNet(g) => {
                    width = width.max(g.conv_dim()).max(g.z().out_features());
                }
                _ => return Err(error("hybrid supports GQA/GatedDeltaNet only")),
            }
            let FeedForward::SwiGlu(ff) = layer.feed_forward() else {
                return Err(error("hybrid segment estimate supports dense TP1/EP1 only"));
            };
            width = width.max(ff.gate().out_features());
        }
        let workspace_bytes = mul(
            mul(
                add(mul(width, 32)?, scores)?,
                options.capabilities().max_batch_tokens,
            )?,
            4,
        )?;
        let host = HybridSegmentResidentMemory {
            parameter_bytes,
            binding_bytes: host_bindings,
            rotary_bytes,
            workspace_bytes,
        };
        // Keep the established conservative device vector + linear bound. Host
        // prepared view multiplicity must not be inferred from this deduped sum.
        let device = HybridSegmentResidentMemory {
            parameter_bytes,
            binding_bytes: parameter_bytes,
            workspace_bytes: add(
                workspace_bytes,
                Self::recurrent_workspace_bytes(
                    inventory.layers(),
                    options.capabilities().max_batch_tokens,
                )?,
            )?,
            ..host
        };
        host.weight_bytes()?;
        device.weight_bytes()?;
        Ok(Self {
            per_sequence_state_bytes: inventory.schema().per_sequence_state_bytes()?,
            per_page_kv_bytes: inventory
                .schema()
                .kv_planes(options.page_size(), options.max_positions())?
                .checked_page_bytes()
                .ok_or_else(|| error("hybrid KV page bytes overflow"))?,
            schema: inventory.schema,
            execution_limits: [
                options.page_size(),
                options.max_positions(),
                options.capabilities().max_batch_tokens,
            ],
            host,
            device,
            host_startup_temporary_bytes: mul(largest_source, 2)?.max(largest_conversion),
            boundary_rows_bytes,
        })
    }
    /// One owner-local recurrent allocation set, not one per layer/sequence.
    /// The transient row envelope above still covers projections and packed
    /// output; this separate reservation covers retained scratch and growth.
    pub(crate) fn recurrent_workspace_bytes(layers: &[DecoderLayer], rows: usize) -> Result<usize> {
        let mut capacity = RecurrentScratchPlan::default();
        for layer in layers {
            if let Attention::GatedDeltaNet(d) = layer.attention() {
                capacity.include(RecurrentScratchPlan::new(
                    rows,
                    d.conv_dim(),
                    d.num_value_heads(),
                    d.z().out_features(),
                )?);
            }
        }
        capacity.growth_peak_bytes()
    }

    pub fn schema(&self) -> &HybridStateSchema {
        &self.schema
    }

    /// Reject composing units with different page/row/position geometry. Runtime
    /// may change capacities and copy counts, but not these model sizing inputs.
    pub fn validate_execution_limits(
        &self,
        page_size: usize,
        max_positions: usize,
        max_batch_tokens: usize,
    ) -> Result<()> {
        if self.execution_limits != [page_size, max_positions, max_batch_tokens] {
            return Err(error("hybrid segment estimate execution limits mismatch"));
        }
        Ok(())
    }

    /// One complete bound catalog and future binding/map-node allowance. Runtime
    /// decides how many parent/owner catalogs coexist; this never reads payloads.
    pub fn binding_metadata_bytes(resources: &BoundDecoderResources) -> Result<usize> {
        resources
            .state_dict()
            .parameters()
            .iter()
            .try_fold(0, |sum, p| {
                let variable = add(
                    add(
                        p.path().as_str().len(),
                        p.weight().slice().path.as_os_str().len(),
                    )?,
                    p.weight().slice().name.len(),
                )?;
                add(sum, add(4096, mul(variable, 16)?)?)
            })
    }
}

/// Same eight independent F32 buffers for admission and CUDA execution:
/// qkv, a, b, z, convolved, recurrent output, normalized, gated.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(in crate::transformer) struct RecurrentScratchPlan {
    pub elements: [usize; 8],
}
impl RecurrentScratchPlan {
    pub fn new(rows: usize, channels: usize, heads: usize, width: usize) -> Result<Self> {
        if channels == 0 || heads == 0 || width == 0 {
            return Err(error("invalid recurrent scratch geometry"));
        }
        let mut elements = [0; 8];
        for (len, columns) in elements
            .iter_mut()
            .zip([channels, heads, heads, width, channels, width, width, width])
        {
            *len = mul(rows, columns)?;
            if columns > i32::MAX as usize || *len > i32::MAX as usize {
                return Err(error("recurrent scratch exceeds i32 ABI"));
            }
        }
        let plan = Self { elements };
        plan.growth_peak_bytes()?;
        Ok(plan)
    }

    // Different layers can maximize different buffers: max(total) is not safe.
    fn include(&mut self, other: Self) {
        for (capacity, required) in self.elements.iter_mut().zip(other.elements) {
            *capacity = (*capacity).max(required);
        }
    }

    fn growth_peak_bytes(self) -> Result<usize> {
        // At most one replacement per buffer per call; old buffers stay in
        // custody until the completion fence. Charge grants, not logical bytes:
        // each independently allocated buffer has its own allocator size class.
        let grants = self.elements.into_iter().try_fold(0, |sum, elements| {
            let granted = ferrule_backend::cuda_allocation_grant_bytes(mul(elements, 4)?)
                .ok_or_else(|| error("recurrent scratch allocation grant overflow"))?;
            add(sum, granted)
        })?;
        mul(grants, 2)
    }
}

fn error(message: &str) -> Error {
    Error::Model {
        message: message.into(),
    }
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| error("hybrid segment memory overflow"))
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| error("hybrid segment memory overflow"))
}

#[cfg(test)]
#[path = "../../tests/unit/segment_inventory/mod.rs"]
mod tests;
