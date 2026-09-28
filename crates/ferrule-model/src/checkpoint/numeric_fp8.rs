//! Raw E4M3FN weights with **numeric**, row-major 128x128 block scales.
//!
//! This is deliberately separate from `LinearWeight`'s native E8M0/FP4 formats:
//! BF16/F32 scale bytes must never be submitted to an E8M0 MMA primitive. Names
//! such as `weight_scale_inv` do not change the contract: W = FP8 * scale.
//! Planning validates the full physical metadata, while bounded reads validate
//! only the selected payload (including every intersecting scale block).
//!
//! # Model/backend binding
//!
//! * Build a physical `ParameterSpec` with F8E4M3 and a required BF16/F32 scale
//!   spec of shape `[M.div_ceil(128), K.div_ceil(128)]` (with the same expert
//!   prefix for rank three). Keep physical storage `Dense`, not the native
//!   `StorageEncoding::Fp8Block128`, which continues to mean E8M0 only.
//! * Select `NumericFp8Encoding` explicitly and call
//!   `BoundParameter::numeric_fp8_source`, or construct `NumericFp8Source` with
//!   **catalog-time** snapshots of both files. Do not recapture snapshots at read
//!   time to make stale bindings appear current. Transforms are not implicit.
//! * Use `plan_tile`/`read`, `StateDictMaterializer::numeric_fp8_tile`, or
//!   `TensorParallelLinearPlan::read_numeric_fp8_shard`. All reads retain the
//!   original pair plus global row/column and expert coordinates. The storage
//!   budget counts the selected weight bytes AND the intersecting scale grid.
//! * Backend consumers use `weight_bytes`, `scale_bytes`, `encoding`, and
//!   `provenance`. For local coordinate `(r,c)`, the scale index is
//!   `((weight_row_origin+r)/128-scale_row_origin,
//!   (weight_column_origin+c)/128-scale_column_origin)`. Ragged views are NOT
//!   automatically block-aligned; a kernel must support these origins or reject
//!   them. Provider I/O can lower the paired `read_plan` and return separately
//!   packed weight/scale buffers through `NumericFp8Read::materialize`.
//! * `decode_f32` is an opt-in CPU oracle with a separate output-byte budget.
//!   No conversion to `LinearWeight` or native E8M0 MMA storage is provided.
//!
//! The multiplication and partial-edge semantics agree with the fixed local
//! vLLM reference's `fp8_utils.py` block dequantization (expand numeric scales,
//! crop to the weight, multiply). This implementation has no vLLM dependency.

use std::ops::Range;
use std::sync::Arc;

use ferrule_common::numeric_fp8::{
    ImmutableValidatedNumericFp8Payload, NumericFp8Layout, NumericFp8ScaleType,
};
use ferrule_common::{Error, Result};

use super::tensor::{CheckpointMatrixRead, CheckpointTensorSlice};
use super::{
    CheckpointDType, CheckpointReadPlan, CheckpointSourceFileIdentity, CheckpointTensorReader,
    decode_fp8_e4m3fn_byte,
};

/// Numeric scale storage, not exponent-only E8M0. Both variants use [128, 128].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericFp8Encoding {
    E4M3FnBlock128Bf16,
    E4M3FnBlock128F32,
}

impl NumericFp8Encoding {
    pub const fn block_shape(self) -> [usize; 2] {
        [128, 128]
    }

    pub const fn scale_dtype(self) -> CheckpointDType {
        match self {
            Self::E4M3FnBlock128Bf16 => CheckpointDType::Bf16,
            Self::E4M3FnBlock128F32 => CheckpointDType::F32,
        }
    }

    pub const fn scale_element_bytes(self) -> usize {
        match self {
            Self::E4M3FnBlock128Bf16 => 2,
            Self::E4M3FnBlock128F32 => 4,
        }
    }

    fn scale(self, bytes: &[u8]) -> f32 {
        match self {
            Self::E4M3FnBlock128Bf16 => {
                half::bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32()
            }
            Self::E4M3FnBlock128F32 => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        }
    }
}

/// Immutable, paired catalog provenance. Supports [M,K] and [experts,M,K].
/// Scales must be [ceil(M/128),ceil(K/128)] with the same expert prefix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumericFp8Source {
    weight: CheckpointTensorSlice,
    scale: CheckpointTensorSlice,
    encoding: NumericFp8Encoding,
    weight_source: CheckpointSourceFileIdentity,
    scale_source: CheckpointSourceFileIdentity,
}

impl NumericFp8Source {
    pub fn new(
        weight: CheckpointTensorSlice,
        scale: CheckpointTensorSlice,
        encoding: NumericFp8Encoding,
        weight_source: CheckpointSourceFileIdentity,
        scale_source: CheckpointSourceFileIdentity,
    ) -> Result<Self> {
        let source = Self::from_metadata(weight, scale, encoding, weight_source, scale_source)?;
        source.validate_source_identity()?;
        Ok(source)
    }

    /// Internal planning within a verified multi-source read boundary.
    pub(crate) fn from_metadata(
        weight: CheckpointTensorSlice,
        scale: CheckpointTensorSlice,
        encoding: NumericFp8Encoding,
        weight_source: CheckpointSourceFileIdentity,
        scale_source: CheckpointSourceFileIdentity,
    ) -> Result<Self> {
        if weight.dtype != CheckpointDType::F8E4M3 || scale.dtype != encoding.scale_dtype() {
            return Err(invalid(
                "expected E4M3FN weight and the explicitly selected BF16/F32 scale dtype",
            ));
        }
        if !(2..=3).contains(&weight.shape.len()) {
            return Err(invalid("weight must have shape [M,K] or [experts,M,K]"));
        }
        let mut expected = weight.shape.clone();
        let rank = expected.len();
        expected[rank - 2] = expected[rank - 2].div_ceil(128);
        expected[rank - 1] = expected[rank - 1].div_ceil(128);
        if scale.shape != expected {
            return Err(invalid(format!(
                "scale shape {:?} must be row-major {expected:?} for 128x128 blocks",
                scale.shape
            )));
        }
        validate_tensor(&weight, 1, &weight_source)?;
        validate_tensor(&scale, encoding.scale_element_bytes(), &scale_source)?;
        if weight.path == scale.path && weight_source != scale_source {
            return Err(invalid(
                "weight and scale have conflicting source snapshots",
            ));
        }
        if weight_source.canonical_path() == scale_source.canonical_path()
            && weight.offset < scale.end_offset()
            && scale.offset < weight.end_offset()
        {
            return Err(invalid("weight and scale physical ranges overlap"));
        }
        let source = Self {
            weight,
            scale,
            encoding,
            weight_source,
            scale_source,
        };
        Ok(source)
    }

    pub fn weight(&self) -> &CheckpointTensorSlice {
        &self.weight
    }
    pub fn scale(&self) -> &CheckpointTensorSlice {
        &self.scale
    }
    pub const fn encoding(&self) -> NumericFp8Encoding {
        self.encoding
    }
    pub fn matrix_shape(&self) -> [usize; 2] {
        let n = self.weight.shape.len();
        [self.weight.shape[n - 2], self.weight.shape[n - 1]]
    }
    pub fn expert_count(&self) -> Option<usize> {
        (self.weight.shape.len() == 3).then(|| self.weight.shape[0])
    }
    pub fn validate_source_identity(&self) -> Result<()> {
        if !self.weight_source.is_current()
            || (self.weight_source != self.scale_source && !self.scale_source.is_current())
        {
            return Err(invalid("stale paired weight/scale source identity"));
        }
        Ok(())
    }

    /// Plan a nonempty rectangle, including the minimal intersecting scale grid.
    /// `expert` is required only for rank-three storage. No block alignment is
    /// required: global origins are retained for ragged TP partitions and edges.
    /// The reader limit includes BOTH weight and scale bytes, before any I/O.
    pub fn plan_tile(
        &self,
        reader: &CheckpointTensorReader,
        expert: Option<usize>,
        rows: Range<usize>,
        columns: Range<usize>,
    ) -> Result<NumericFp8Read> {
        self.validate_source_identity()?;
        self.plan_tile_metadata(reader, expert, rows, columns)
    }

    pub(crate) fn plan_tile_metadata(
        &self,
        reader: &CheckpointTensorReader,
        expert: Option<usize>,
        rows: Range<usize>,
        columns: Range<usize>,
    ) -> Result<NumericFp8Read> {
        let weight = matrix_slice(&self.weight, expert, 1)?;
        let scale = matrix_slice(&self.scale, expert, self.encoding.scale_element_bytes())?;
        let weight_read = reader.plan_2d_storage_range(
            &weight,
            rows.clone(),
            columns.clone(),
            &self.weight_source,
        )?;
        let scale_read = reader.plan_2d_storage_range(
            &scale,
            rows.start / 128..rows.end.div_ceil(128),
            columns.start / 128..columns.end.div_ceil(128),
            &self.scale_source,
        )?;
        let mut snapshots = vec![self.weight_source.clone()];
        if self.weight.path != self.scale.path {
            snapshots.push(self.scale_source.clone());
        }
        let plan = CheckpointReadPlan::new(
            weight_read
                .read_plan()
                .extents()
                .iter()
                .chain(scale_read.read_plan().extents())
                .cloned(),
            Arc::from(snapshots),
        )?;
        check_limit(
            plan.storage_bytes(),
            reader.max_tensor_bytes(),
            "paired storage",
        )?;
        Ok(NumericFp8Read {
            source: self.clone(),
            expert,
            weight_read,
            scale_read,
            plan,
        })
    }
}

/// A provider-neutral paired read plan. Extents are weight first, scale second;
/// each part is packed in row-major order, not necessarily contiguous on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumericFp8Read {
    source: NumericFp8Source,
    expert: Option<usize>,
    weight_read: CheckpointMatrixRead,
    scale_read: CheckpointMatrixRead,
    plan: CheckpointReadPlan,
}

impl NumericFp8Read {
    pub fn source(&self) -> &NumericFp8Source {
        &self.source
    }
    pub const fn expert(&self) -> Option<usize> {
        self.expert
    }
    pub fn weight_read(&self) -> &CheckpointMatrixRead {
        &self.weight_read
    }
    pub fn scale_read(&self) -> &CheckpointMatrixRead {
        &self.scale_read
    }
    pub fn read_plan(&self) -> &CheckpointReadPlan {
        &self.plan
    }
    pub fn local_shape(&self) -> [usize; 2] {
        self.weight_read.local_shape()
    }

    pub fn read(&self, reader: &CheckpointTensorReader) -> Result<NumericFp8Artifact> {
        check_limit(
            self.plan.storage_bytes(),
            reader.max_tensor_bytes(),
            "paired storage",
        )?;
        reader
            .verified_read_session(&self.plan)?
            .read_validated(|mut packed| {
                let scale = packed.split_off(self.weight_read.read_plan().storage_bytes() as usize);
                self.materialize_bytes(packed, scale)
            })
    }

    /// Only the enclosing incremental session may publish these artifacts,
    /// after its final per-source identity check succeeds.
    pub(crate) fn read_incremental(
        &self,
        session: &super::VerifiedReadSession,
    ) -> Result<NumericFp8Artifact> {
        let mut packed = session.read_subset(&self.plan)?;
        let scale = packed.split_off(self.weight_read.read_plan().storage_bytes() as usize);
        self.materialize_bytes(packed, scale)
    }

    /// Accept packed bytes from a provider lowering `read_plan()`. The source
    /// must still match both catalog snapshots; lengths and values are checked.
    pub fn materialize(&self, weight: Vec<u8>, scale: Vec<u8>) -> Result<NumericFp8Artifact> {
        self.source.validate_source_identity()?;
        let artifact = self.materialize_bytes(weight, scale)?;
        self.source.validate_source_identity()?;
        Ok(artifact)
    }

    // Shared byte validation; the local reader encloses this in its verified
    // session, while external providers retain the legacy publication checks.
    fn materialize_bytes(&self, weight: Vec<u8>, scale: Vec<u8>) -> Result<NumericFp8Artifact> {
        if weight.len() as u64 != self.weight_read.read_plan().storage_bytes()
            || scale.len() as u64 != self.scale_read.read_plan().storage_bytes()
        {
            return Err(invalid("packed weight/scale byte length mismatch"));
        }
        let [n, k] = self.local_shape();
        let layout = NumericFp8Layout {
            n,
            k,
            row_origin: self.weight_read.rows().start,
            column_origin: self.weight_read.columns().start,
            scale_type: match self.source.encoding {
                NumericFp8Encoding::E4M3FnBlock128Bf16 => NumericFp8ScaleType::Bf16,
                NumericFp8Encoding::E4M3FnBlock128F32 => NumericFp8ScaleType::F32,
            },
        };
        if self.scale_read.rows().start != layout.row_origin / 128
            || self.scale_read.columns().start != layout.column_origin / 128
            || self.scale_read.local_shape() != layout.scale_shape()?
        {
            return Err(invalid(
                "packed scale grid does not match global weight origins",
            ));
        }
        let payload =
            ImmutableValidatedNumericFp8Payload::new(layout, weight, scale).map_err(|error| {
                match error {
                    Error::Internal { message } => Error::Model { message },
                    other => other,
                }
            })?;
        #[cfg(test)]
        proof_tests::validated(&payload);
        // This result is still private: read_validated's final source boundary
        // (or materialize's post-check) must succeed before it is published.
        Ok(NumericFp8Artifact {
            provenance: self.clone(),
            payload,
        })
    }
}

/// Validated compressed storage and independent catalog provenance. Clones
/// share the immutable payload; no second raw-byte copy is retained. Backends
/// reuse `validated_payload()` after separately checking source freshness.
#[derive(Debug, Clone)]
pub struct NumericFp8Artifact {
    provenance: NumericFp8Read,
    payload: ImmutableValidatedNumericFp8Payload,
}

impl PartialEq for NumericFp8Artifact {
    fn eq(&self, other: &Self) -> bool {
        self.provenance == other.provenance
            && self.payload.layout() == other.payload.layout()
            && self.weight_bytes() == other.weight_bytes()
            && self.scale_bytes() == other.scale_bytes()
    }
}
impl Eq for NumericFp8Artifact {}

impl NumericFp8Artifact {
    pub fn provenance(&self) -> &NumericFp8Read {
        &self.provenance
    }
    /// Content proof only, not a source-freshness or device-owner capability.
    pub fn validated_payload(&self) -> &ImmutableValidatedNumericFp8Payload {
        &self.payload
    }
    pub fn weight_bytes(&self) -> &[u8] {
        self.payload.weight_bytes()
    }
    pub fn scale_bytes(&self) -> &[u8] {
        self.payload.scale_bytes()
    }
    pub fn encoding(&self) -> NumericFp8Encoding {
        self.provenance.source.encoding
    }
    pub fn local_shape(&self) -> [usize; 2] {
        self.provenance.local_shape()
    }
    pub fn storage_bytes(&self) -> u64 {
        self.provenance.plan.storage_bytes()
    }

    /// CPU oracle for the selected tile only. Output has an independent byte
    /// budget (FP8 input size alone is not an F32 allocation bound). E4M3FN NaNs
    /// and invalid scales/products are rejected before artifact publication.
    pub fn decode_f32(&self, max_output_bytes: u64) -> Result<Vec<f32>> {
        self.provenance.source.validate_source_identity()?;
        let bytes = (self.weight_bytes().len() as u64)
            .checked_mul(4)
            .ok_or_else(|| invalid("F32 output size overflow"))?;
        check_limit(bytes, max_output_bytes, "F32 output")?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(self.weight_bytes().len())
            .map_err(|e| invalid(format!("allocate F32 tile: {e}")))?;
        let read = &self.provenance;
        let [_, columns] = read.local_shape();
        let weight_rows = read.weight_read.rows();
        let weight_columns = read.weight_read.columns();
        let scale_rows = read.scale_read.rows();
        let scale_columns = read.scale_read.columns();
        let encoding = self.encoding();
        let element_bytes = encoding.scale_element_bytes();
        for (index, byte) in self.weight_bytes().iter().copied().enumerate() {
            let row = (weight_rows.start + index / columns) / 128 - scale_rows.start;
            let col = (weight_columns.start + index % columns) / 128 - scale_columns.start;
            let offset = (row * scale_columns.len() + col) * element_bytes;
            let value =
                decode_fp8_e4m3fn_byte(byte) * encoding.scale(&self.scale_bytes()[offset..]);
            if !value.is_finite() {
                return Err(invalid("numeric FP8 product overflows F32"));
            }
            output.push(value);
        }
        Ok(output)
    }
}

fn validate_tensor(
    tensor: &CheckpointTensorSlice,
    element_bytes: usize,
    source: &CheckpointSourceFileIdentity,
) -> Result<()> {
    let bytes = tensor
        .element_count()?
        .checked_mul(element_bytes)
        .filter(|n| *n <= isize::MAX as usize)
        .ok_or_else(|| invalid("tensor size overflow"))?;
    if tensor.shape.contains(&0) || tensor.bytes != bytes as u64 {
        return Err(invalid("tensor shape/dtype/byte size mismatch"));
    }
    let end = tensor
        .offset
        .checked_add(tensor.bytes)
        .ok_or_else(|| invalid("tensor file range overflow"))?;
    if source.catalog_path() != tensor.path || end > source.length() {
        return Err(invalid("tensor range does not match source snapshot"));
    }
    Ok(())
}

fn matrix_slice(
    tensor: &CheckpointTensorSlice,
    expert: Option<usize>,
    element_bytes: usize,
) -> Result<CheckpointTensorSlice> {
    let mut matrix = tensor.clone();
    match (tensor.shape.as_slice(), expert) {
        ([_, _], None) => {}
        ([experts, rows, columns], Some(expert)) if expert < *experts => {
            // Full shape, byte count and file range were checked at construction.
            matrix.bytes = (rows * columns * element_bytes) as u64;
            matrix.offset += expert as u64 * matrix.bytes;
            matrix.shape = vec![*rows, *columns];
        }
        _ => {
            return Err(invalid(
                "expert index is missing, unexpected, or out of range",
            ));
        }
    }
    Ok(matrix)
}

fn check_limit(bytes: u64, limit: u64, label: &str) -> Result<()> {
    if bytes > limit {
        return Err(invalid(format!(
            "{label} exceeds bounded read size: {bytes} > {limit}"
        )));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Model {
        message: format!("numeric FP8: {}", message.into()),
    }
}

#[cfg(test)]
#[path = "numeric_fp8_tests.rs"]
pub(crate) mod proof_tests;
