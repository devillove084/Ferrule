//! Hybrid boot metadata over the existing segment and working-state journal.
use ferrule_model::decoder::{HybridLayerSchema, HybridStateSchema};
use ferrule_model::transformer::{BoundDecoderResources, LayerSegmentPlan};
use serde::{Deserialize, Serialize};

use super::ProcessFrameLimits;
use super::decoder::{DecoderBoot, DecoderDevice, DecoderPrecision, Result, error};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum HybridLayerFrame {
    FullAttention {
        kv_heads: usize,
        head_dim: usize,
    },
    GatedDeltaNet {
        key_heads: usize,
        value_heads: usize,
        key_dim: usize,
        value_dim: usize,
        kernel: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HybridCudaBudgetFrame {
    pub state_bytes: usize,
    pub weight_bytes: usize,
    pub workspace_bytes: usize,
    pub margin_bytes: usize,
}

/// Ordered geometry only, never sequence state or physical page authority.
/// Children reconstruct the schema from the strict checkpoint before preparing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HybridBootFrame {
    pub hidden: usize,
    pub vocabulary: usize,
    pub layers: Vec<HybridLayerFrame>,
    pub cuda_budget: Option<HybridCudaBudgetFrame>,
}
impl HybridBootFrame {
    pub fn for_resources(
        resources: &BoundDecoderResources,
        plan: &LayerSegmentPlan,
        cuda_budget: Option<HybridCudaBudgetFrame>,
    ) -> Result<Self> {
        if plan.total_layers() != resources.spec().layers().len() {
            return Err(error("hybrid checkpoint/plan depth mismatch"));
        }
        let schema = HybridStateSchema::for_segment(resources.spec(), plan.layers())?;
        Ok(Self {
            hidden: resources.spec().hidden_size(),
            vocabulary: resources.spec().vocab_size(),
            layers: schema
                .layers()
                .iter()
                .map(|layer| match layer {
                    HybridLayerSchema::FullAttention {
                        kv_heads, head_dim, ..
                    } => HybridLayerFrame::FullAttention {
                        kv_heads: *kv_heads,
                        head_dim: *head_dim,
                    },
                    HybridLayerSchema::GatedDeltaNet(s) => HybridLayerFrame::GatedDeltaNet {
                        key_heads: s.key_heads,
                        value_heads: s.value_heads,
                        key_dim: s.key_dim,
                        value_dim: s.value_dim,
                        kernel: s.kernel,
                    },
                })
                .collect(),
            cuda_budget,
        })
    }

    pub(super) fn geometry(&self) -> Result<(usize, usize, usize, usize)> {
        let mut geometry = None;
        if self.hidden == 0 || self.vocabulary == 0 || self.vocabulary > u32::MAX as usize {
            return Err(error("invalid hybrid decoder geometry"));
        }
        for layer in &self.layers {
            match layer {
                HybridLayerFrame::FullAttention { kv_heads, head_dim } => {
                    if *kv_heads == 0
                        || *head_dim == 0
                        || geometry.is_some_and(|g| g != (*kv_heads, *head_dim))
                    {
                        return Err(error("invalid hybrid full-attention geometry"));
                    }
                    geometry = Some((*kv_heads, *head_dim));
                }
                HybridLayerFrame::GatedDeltaNet {
                    key_heads,
                    value_heads,
                    key_dim,
                    value_dim,
                    kernel,
                } => {
                    if [*key_heads, *value_heads, *key_dim, *value_dim, *kernel].contains(&0)
                        || !value_heads.is_multiple_of(*key_heads)
                    {
                        return Err(error("invalid hybrid recurrent geometry"));
                    }
                    key_heads
                        .checked_mul(*key_dim)
                        .and_then(|k| k.checked_mul(2))
                        .and_then(|k| {
                            value_heads
                                .checked_mul(*value_dim)
                                .and_then(|v| k.checked_add(v))
                        })
                        .and_then(|qkv| qkv.checked_mul(*kernel))
                        .and_then(|conv| {
                            value_heads
                                .checked_mul(*value_dim)
                                .and_then(|v| v.checked_mul(*key_dim))
                                .and_then(|r| conv.checked_add(r))
                        })
                        .and_then(|n| n.checked_mul(4))
                        .ok_or_else(|| error("hybrid recurrent geometry overflow"))?;
                }
            }
        }
        let (heads, dim) = geometry.ok_or_else(|| error("hybrid stage requires full attention"))?;
        Ok((self.hidden, self.vocabulary, heads, dim))
    }

    pub(super) fn validate(&self, boot: &DecoderBoot) -> Result<()> {
        let plan = boot.segment.decode()?;
        if self.layers.len() != plan.layer_count()
            || boot.experts.is_some()
            || boot.precision != DecoderPrecision::F32
        {
            return Err(error(
                "hybrid process requires exact dense F32 segment schema",
            ));
        }
        self.geometry()?;
        match (boot.device, &self.cuda_budget) {
            (DecoderDevice::Cpu, None) => {}
            (DecoderDevice::Cuda { .. }, Some(b))
                if b.state_bytes > 0 && b.weight_bytes > 0 && b.workspace_bytes > 0 => {}
            _ => return Err(error("hybrid device/budget mismatch")),
        }
        Ok(())
    }

    pub fn validate_resources(
        &self,
        boot: &DecoderBoot,
        resources: &BoundDecoderResources,
    ) -> Result<()> {
        self.validate(boot)?;
        let expected =
            Self::for_resources(resources, &boot.segment.decode()?, self.cuda_budget.clone())?;
        if *self != expected
            || resources.spec().layers().iter().any(|layer| {
                !matches!(
                    layer.feed_forward(),
                    ferrule_model::transformer::FeedForward::SwiGlu(_)
                )
            })
        {
            return Err(error(
                "hybrid rebound schema differs from Boot or is not dense",
            ));
        }
        Ok(())
    }
}

impl DecoderBoot {
    /// Checked arithmetic only, without representative tensor/JSON allocation.
    /// Reserve 32 bytes per finite value, including endpoint F32-to-JSON-number
    /// promotion and separators.
    /// Reject impossible legal batches before spawning or model work.
    pub fn validate_frame_limits(&self, limits: ProcessFrameLimits) -> Result<()> {
        limits.validate().map_err(error)?;
        let Some((command, output)) = self.frame_payload_bounds()? else {
            return Ok(());
        };
        if command > limits.max_command_bytes
            || output > limits.max_output_bytes
            || command
                .max(output)
                .checked_add(1024)
                .is_none_or(|n| n > limits.max_frame_bytes)
        {
            return Err(error(
                "hybrid rows times vocabulary/hidden exceeds process frame limits",
            ));
        }
        Ok(())
    }

    /// Grow only the required negotiated limits, never beyond the process hard
    /// ceiling. The executable must accept these trusted child ceilings too.
    pub fn fit_frame_limits(&self, mut limits: ProcessFrameLimits) -> Result<ProcessFrameLimits> {
        limits.validate().map_err(error)?;
        if let Some((command, output)) = self.frame_payload_bounds()? {
            limits.max_command_bytes = limits.max_command_bytes.max(command);
            limits.max_output_bytes = limits.max_output_bytes.max(output);
            limits.max_frame_bytes = limits.max_frame_bytes.max(
                command
                    .max(output)
                    .checked_add(1024)
                    .ok_or_else(|| error("hybrid wire envelope overflow"))?,
            );
        }
        limits.validate().map_err(|_| error("hybrid full-vocabulary batch exceeds 128 MiB process ceiling; reduce max_batch_tokens/prefill_chunk_size"))?;
        self.validate_frame_limits(limits)?;
        Ok(limits)
    }

    fn frame_payload_bounds(&self) -> Result<Option<(usize, usize)>> {
        let Some(hybrid) = &self.hybrid else {
            return Ok(None);
        };
        self.stage_boot()?;
        let bound = |width: usize| -> Result<usize> {
            width
                .checked_mul(self.kv.max_batch_tokens)
                .and_then(|n| n.checked_mul(32))
                .and_then(|n| n.checked_add(4096))
                .ok_or_else(|| error("hybrid wire payload size overflow"))
        };
        let command = bound(hybrid.hidden)?;
        let output = bound(if self.segment.output {
            hybrid.vocabulary
        } else {
            hybrid.hidden
        })?;
        let metadata = self
            .kv
            .max_pages
            .checked_mul(128)
            .and_then(|n| {
                self.kv
                    .max_batch_tokens
                    .checked_mul(128)
                    .and_then(|m| n.checked_add(m))
            })
            .and_then(|n| n.checked_add(4096))
            .ok_or_else(|| error("hybrid wire metadata size overflow"))?;
        let command = command.max(metadata);
        let output = output.max(metadata);
        Ok(Some((command, output)))
    }
}
