//! Host buffer ledger for standard preparation followed by serial CUDA upload.
//! This is CPU metadata planning, not the CUDA device-memory estimator. Keep the
//! lifetimes aligned with PreparedLinear/PreparedNorm and CudaHybridModule:
//! all prepared host layers survive while one dense projection is converted.
use super::{HostMemoryBudget, invalid};
use crate::checkpoint::CheckpointDType;
use crate::nn::{ParameterId, ParameterResidency};
use crate::support::TensorRole;
use crate::transformer::{Attention, BoundDecoderResources};
use ferrule_common::Result;
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StandardHostStartupMemory {
    /// One original dense payload per canonical parameter, plus the native
    /// LinearWeight clone for each prepared linear view (including aliases).
    pub dense_source_bytes: u64,
    pub native_linear_clone_bytes: u64,
    /// NumericFp8Artifact/PreparedLinear share one compressed payload, no decode.
    pub compressed_bytes: u64,
    /// Norm/GDN conv/A_log/dt_bias keep their decoded Arc<[f32]> beside raw bytes.
    pub decoded_auxiliary_bytes: u64,
    pub rotary_bytes: u64,
    /// Future prepared bindings, expert metadata directories and Rust map nodes.
    /// Excludes the already bound catalog, whose RSS is in the OS snapshot.
    pub metadata_bytes: u64,
    pub reader_temporary_bytes: u64,
    /// One F32 values Vec AND one byte-serialization Vec during dense upload.
    pub conversion_temporary_bytes: u64,
}
fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| invalid("standard host startup bytes overflow"))
}
fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("standard host startup bytes overflow"))
}

impl StandardHostStartupMemory {
    pub fn for_resources(
        resources: &BoundDecoderResources,
        max_positions: usize,
        max_parameter_bytes: u64,
    ) -> Result<Self> {
        let mut ledger = Self::default();
        let mut seen = BTreeSet::<ParameterId>::new();
        for parameter in resources.state_dict().parameters() {
            // GPU bindings prepare metadata for routed experts too, but never a
            // second host expert payload. Charge only its Rust structures here.
            let variable = parameter.path().as_str().len() as u64
                + parameter.weight().slice().path.as_os_str().len() as u64
                + parameter.weight().slice().name.len() as u64;
            ledger.metadata_bytes = add(ledger.metadata_bytes, add(4096, mul(variable, 16)?)?)?;
            match parameter.residency() {
                ParameterResidency::Expert { .. } | ParameterResidency::Attachment { .. } => {
                    continue;
                }
                ParameterResidency::Static | ParameterResidency::Layer { .. } => {}
            }
            let weight = parameter.weight().slice();
            let bytes = add(
                weight.bytes,
                parameter.scale().map_or(0, |p| p.slice().bytes),
            )?;
            if bytes > max_parameter_bytes {
                return Err(invalid("base parameter exceeds startup reader limit"));
            }
            let first = seen.insert(parameter.canonical_id());
            ledger.reader_temporary_bytes = ledger.reader_temporary_bytes.max(mul(bytes, 2)?);
            if parameter.numeric_fp8_encoding().is_some() {
                if first {
                    ledger.compressed_bytes = add(ledger.compressed_bytes, bytes)?;
                }
                continue;
            }
            if parameter.scale().is_some()
                || !matches!(weight.dtype, CheckpointDType::Bf16 | CheckpointDType::F32)
            {
                return Err(invalid(
                    "standard host startup ledger requires dense BF16/F32 or paired numeric FP8",
                ));
            }
            let f32_bytes = mul(weight.element_count()? as u64, 4)?;
            if first {
                ledger.dense_source_bytes = add(ledger.dense_source_bytes, bytes)?;
            }
            let auxiliary = matches!(
                parameter.role(),
                TensorRole::OutputNorm
                    | TensorRole::AttentionNorm
                    | TensorRole::FeedForwardNorm
                    | TensorRole::AttentionQueryNorm
                    | TensorRole::AttentionKeyNorm
                    | TensorRole::LinearAttentionNorm
                    | TensorRole::LinearAttentionConv
                    | TensorRole::LinearAttentionALog
                    | TensorRole::LinearAttentionDtBias
            );
            if auxiliary {
                ledger.decoded_auxiliary_bytes = add(ledger.decoded_auxiliary_bytes, f32_bytes)?;
                // Vec -> Arc conversion or temporary vector upload. Not retained
                // for every tensor simultaneously; the maximum is reusable.
                ledger.conversion_temporary_bytes =
                    ledger.conversion_temporary_bytes.max(f32_bytes);
            } else {
                if weight.shape.len() != 2 {
                    return Err(invalid("unclassified standard host parameter layout"));
                }
                ledger.native_linear_clone_bytes = add(ledger.native_linear_clone_bytes, bytes)?;
                ledger.conversion_temporary_bytes =
                    ledger.conversion_temporary_bytes.max(mul(f32_bytes, 2)?);
            }
        }
        for layer in resources.spec().layers() {
            if let Attention::Gqa(gqa) = layer.attention() {
                // cos + sin: each positions * rotary_dimensions/2 F32 values.
                let bytes = mul(
                    mul(
                        max_positions as u64,
                        gqa.rotary().region().dimensions() as u64,
                    )?,
                    4,
                )?;
                ledger.rotary_bytes = add(ledger.rotary_bytes, bytes)?;
                ledger.reader_temporary_bytes = ledger.reader_temporary_bytes.max(bytes);
            }
        }
        Ok(ledger)
    }
    pub fn resident_bytes(self) -> Result<u64> {
        [
            self.dense_source_bytes,
            self.native_linear_clone_bytes,
            self.compressed_bytes,
            self.decoded_auxiliary_bytes,
            self.rotary_bytes,
            self.metadata_bytes,
        ]
        .into_iter()
        .try_fold(0, add)
    }
    pub fn temporary_bytes(self) -> u64 {
        self.reader_temporary_bytes
            .max(self.conversion_temporary_bytes)
    }
    /// Standard factory builds base after warm and has no base payload loaded at
    /// admission. Other callers must explicitly identify any already owned base.
    pub fn memory_budget(self) -> Result<HostMemoryBudget> {
        Ok(HostMemoryBudget {
            base_resident_bytes: self.resident_bytes()?,
            base_temporary_bytes: self.temporary_bytes(),
            ..Default::default()
        })
    }
}
