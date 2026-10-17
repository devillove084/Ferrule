//! Prepared numeric rank images. Discontiguous rows remain separate FP8/scale
//! pairs. Only explicit native Mixed QKV preparation may concatenate complete
//! scale blocks after proving the semantic row order and packed grid geometry.
use std::{ops::Range, sync::Arc};

use ferrule_common::Result;

use super::{StateDictMaterializer, model_error};
use crate::checkpoint::{CheckpointReadPlan, NumericFp8Artifact, NumericFp8Read};
use crate::transformer::{
    BoundParameter,
    hybrid_parallel::{HybridParameterPartition, HybridRankLayout},
};

/// Encoded payload and the actual allocator size classes of separate weight and
/// scale uploads. Counts repeated intersecting scale blocks once per segment.
/// Does not include host staging, device segment slack, KV, or execution scratch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NumericFp8RankWeightBudget {
    pub weight_bytes: usize,
    pub scale_bytes: usize,
    pub payload_bytes: usize,
    pub allocation_grant_bytes: usize,
}

/// Metadata-only preparation recipe, bound to the model's exact rank geometry
/// and original catalog snapshots. Construction never opens a checkpoint file.
#[derive(Debug, Clone)]
pub struct NumericFp8RankPlan {
    binding: BoundParameter,
    layout: Arc<HybridRankLayout>,
    partition: HybridParameterPartition,
    reads: Arc<[NumericFp8Read]>,
    budget: NumericFp8RankWeightBudget,
}

impl NumericFp8RankPlan {
    pub fn new(binding: &BoundParameter, layout: Arc<HybridRankLayout>) -> Result<Self> {
        let encoding = binding.numeric_fp8_encoding().ok_or_else(|| {
            model_error("rank numeric preparation requires E4M3FN with BF16/F32 scales")
        })?;
        layout.validate_numeric_storage(encoding)?;
        let source = binding.numeric_fp8_source_metadata(encoding)?;
        if source.expert_count().is_some() {
            return Err(model_error(
                "rank numeric image requires a 2D non-expert projection",
            ));
        }
        let partition = layout.parameter_partition(binding)?;
        Self::from_partition(binding, layout, partition)
    }

    fn from_partition(
        binding: &BoundParameter,
        layout: Arc<HybridRankLayout>,
        partition: HybridParameterPartition,
    ) -> Result<Self> {
        let source = binding.numeric_fp8_source_metadata(
            binding
                .numeric_fp8_encoding()
                .ok_or_else(|| model_error("missing numeric encoding"))?,
        )?;
        let local = partition.local_shape()?;
        let reader = crate::checkpoint::CheckpointTensorReader::new(isize::MAX as u64);
        let mut reads = Vec::with_capacity(partition.rows.len());
        let mut budget = NumericFp8RankWeightBudget::default();
        let mut end = 0;
        for rows in &partition.rows {
            if rows.start < end {
                return Err(model_error(
                    "numeric rank rows must be ordered and disjoint",
                ));
            }
            end = rows.end;
            let read = source.plan_tile_metadata(
                &reader,
                None,
                rows.clone(),
                partition.columns.clone(),
            )?;
            let weight = usize::try_from(read.weight_read().read_plan().storage_bytes())
                .map_err(|_| model_error("numeric weight exceeds address space"))?;
            let scale = usize::try_from(read.scale_read().read_plan().storage_bytes())
                .map_err(|_| model_error("numeric scale exceeds address space"))?;
            budget.weight_bytes = add(budget.weight_bytes, weight)?;
            budget.scale_bytes = add(budget.scale_bytes, scale)?;
            budget.allocation_grant_bytes = add(
                budget.allocation_grant_bytes,
                add(grant(weight)?, grant(scale)?)?,
            )?;
            reads.push(read);
        }
        budget.payload_bytes = add(budget.weight_bytes, budget.scale_bytes)?;
        if local[0].checked_mul(local[1]) != Some(budget.weight_bytes) {
            return Err(model_error("numeric rank packed extent mismatch"));
        }
        Ok(Self {
            binding: binding.clone(),
            layout,
            partition,
            reads: reads.into(),
            budget,
        })
    }

    /// Native Mixed GDN uses the original combined QKV GEMM geometry, not three
    /// independently rounded GEMMs. This is metadata-only and never applies to
    /// F32/W8 execution. The source read plan and its provenance stay segmented.
    pub fn mixed_gemmex_qkv_layout(
        &self,
    ) -> Result<Option<ferrule_common::numeric_fp8::NumericFp8Layout>> {
        use crate::transformer::hybrid_parallel::HybridMixedProjectionArithmetic;
        use ferrule_common::numeric_fp8::{NumericFp8Layout, NumericFp8ScaleType};
        if !self.layout.precision().is_mixed()
            || self.binding.role() != &crate::TensorRole::LinearAttentionQkv
        {
            return Ok(None);
        }
        for read in self.reads() {
            let [n, k] = read.local_shape();
            HybridMixedProjectionArithmetic::packed_gemmex_layout(NumericFp8Layout {
                n,
                k,
                row_origin: read.weight_read().rows().start,
                column_origin: read.weight_read().columns().start,
                scale_type: match read.source().encoding().scale_dtype() {
                    crate::checkpoint::CheckpointDType::Bf16 => NumericFp8ScaleType::Bf16,
                    _ => return Err(model_error("Mixed QKV requires original BF16 scales")),
                },
            })?;
        }
        let [n, k] = self.local_shape();
        let packed = NumericFp8Layout {
            n,
            k,
            row_origin: 0,
            column_origin: 0,
            scale_type: NumericFp8ScaleType::Bf16,
        };
        let (weight, scales) = packed.storage_lengths()?;
        if (weight, scales) != (self.budget.weight_bytes, self.budget.scale_bytes)
            || add(grant(weight)?, grant(scales)?)? > self.budget.allocation_grant_bytes
        {
            return Err(model_error(
                "Mixed QKV packed storage exceeds source ledger",
            ));
        }
        Ok(Some(packed))
    }

    pub fn binding(&self) -> &BoundParameter {
        &self.binding
    }
    pub fn layout(&self) -> &HybridRankLayout {
        &self.layout
    }
    pub fn partition(&self) -> &HybridParameterPartition {
        &self.partition
    }
    pub fn reads(&self) -> &[NumericFp8Read] {
        &self.reads
    }
    pub fn budget(&self) -> NumericFp8RankWeightBudget {
        self.budget
    }
    pub fn global_shape(&self) -> [usize; 2] {
        self.reads[0].source().matrix_shape()
    }
    pub fn local_shape(&self) -> [usize; 2] {
        [
            self.budget.weight_bytes / self.partition.columns.len(),
            self.partition.columns.len(),
        ]
    }

    /// Metadata and snapshot equality only, deliberately no filesystem access.
    /// Admission must compare the entire source pair, not just tensor names.
    pub fn validate_binding(
        &self,
        binding: &BoundParameter,
        layout: &HybridRankLayout,
    ) -> Result<()> {
        if !self.binding.shares_storage_with(binding)
            || self.binding.spec() != binding.spec()
            || self.binding.role() != binding.role()
            || self.layout.as_ref() != layout
            || layout.parameter_partition(binding)? != self.partition
            || binding.numeric_fp8_source_metadata(self.reads[0].source().encoding())?
                != *self.reads[0].source()
        {
            return Err(model_error(
                "numeric rank binding/image/layout/paired provenance mismatch",
            ));
        }
        Ok(())
    }
}

/// One local output-feature segment with its original global origins intact.
#[derive(Debug, Clone)]
pub struct PreparedNumericFp8RankSegment {
    output_columns: Range<usize>,
    artifact: NumericFp8Artifact,
}
impl PreparedNumericFp8RankSegment {
    /// Columns in the logical packed rank-local output (Q, then K, then V).
    pub fn output_columns(&self) -> Range<usize> {
        self.output_columns.clone()
    }
    pub fn artifact(&self) -> &NumericFp8Artifact {
        &self.artifact
    }
}

/// Immutable raw rank-local storage, with no dense/CPU conversion entry point.
/// The plan owns metadata only; cloning shares each payload's validation proof.
#[derive(Debug, Clone)]
pub struct PreparedNumericFp8RankLinear {
    plan: NumericFp8RankPlan,
    segments: Arc<[PreparedNumericFp8RankSegment]>,
}
impl PreparedNumericFp8RankLinear {
    #[cfg(all(test, feature = "cuda"))]
    pub(crate) fn payload_liveness(&self) -> std::sync::Weak<[PreparedNumericFp8RankSegment]> {
        Arc::downgrade(&self.segments)
    }

    pub fn plan(&self) -> &NumericFp8RankPlan {
        &self.plan
    }
    pub fn segments(&self) -> &[PreparedNumericFp8RankSegment] {
        &self.segments
    }

    /// Preparation-only byte packing. No decode, changed scale value, or device
    /// allocation; each original paired source and output-column range is checked.
    pub fn packed_mixed_gemmex_qkv(
        &self,
    ) -> Result<ferrule_common::numeric_fp8::ImmutableValidatedNumericFp8Payload> {
        let layout = self
            .plan
            .mixed_gemmex_qkv_layout()?
            .ok_or_else(|| model_error("not a native Mixed GDN QKV projection"))?;
        self.plan
            .validate_binding(self.plan.binding(), self.plan.layout())?;
        let mut weight = Vec::with_capacity(self.plan.budget.weight_bytes);
        let mut scales = Vec::with_capacity(self.plan.budget.scale_bytes);
        let mut end = 0;
        for segment in self.segments() {
            let artifact = segment.artifact();
            artifact.provenance().source().validate_source_identity()?;
            let local = artifact.validated_payload().layout();
            if segment.output_columns() != (end..add(end, local.n)?) || local.k != layout.k {
                return Err(model_error("Mixed QKV semantic output columns differ"));
            }
            end = add(end, local.n)?;
            weight.extend_from_slice(artifact.weight_bytes());
            scales.extend_from_slice(artifact.scale_bytes());
        }
        if end != layout.n {
            return Err(model_error("Mixed QKV semantic width differs"));
        }
        let payload = ferrule_common::numeric_fp8::ImmutableValidatedNumericFp8Payload::new(
            layout, weight, scales,
        )?;
        for segment in self.segments() {
            segment
                .artifact()
                .provenance()
                .source()
                .validate_source_identity()?;
        }
        Ok(payload)
    }
}

impl StateDictMaterializer {
    /// Prepare only model-selected rectangles. Both the aggregate encoded read
    /// limit and physical upload grants are checked BEFORE opening files. Every
    /// segment is published under one paired-source pre/post read boundary.
    /// No full-parameter read or CPU dequantization is used, even for row TP.
    pub fn prepare_numeric_fp8_rank_linear(
        &self,
        plan: &NumericFp8RankPlan,
        max_allocation_grant_bytes: usize,
    ) -> Result<PreparedNumericFp8RankLinear> {
        if self.tensor.is_some() {
            return Err(model_error(
                "cannot mix standard TP and hybrid numeric preparation",
            ));
        }
        #[cfg(feature = "cpu-reference")]
        if self.hybrid.is_some() {
            return Err(model_error("cannot mix CPU hybrid and numeric preparation"));
        }
        plan.validate_binding(plan.binding(), plan.layout())?;
        if plan.budget.payload_bytes as u64 > self.max_parameter_bytes
            || plan.budget.allocation_grant_bytes > max_allocation_grant_bytes
        {
            return Err(model_error(
                "numeric rank aggregate payload/allocation grant exceeds budget",
            ));
        }
        let mut sources = Vec::new();
        for read in plan.reads.iter() {
            for source in read.read_plan().source_files() {
                if !sources.contains(source) {
                    sources.push(source.clone());
                }
            }
        }
        let paired = CheckpointReadPlan::new(
            plan.reads
                .iter()
                .flat_map(|read| read.read_plan().extents().iter().cloned()),
            Arc::<[_]>::from(sources),
        )?;
        let session = self.reader.verified_read_session(&paired)?;
        let segments = session.read_incrementally(|session| {
            let mut segments = Vec::with_capacity(plan.reads.len());
            let mut start = 0;
            for read in plan.reads.iter() {
                let artifact = read.read_incremental(session)?;
                let end = add(start, artifact.local_shape()[0])?;
                segments.push(PreparedNumericFp8RankSegment {
                    output_columns: start..end,
                    artifact,
                });
                start = end;
            }
            Ok(segments)
        })?;
        // Account both reads separately so scales cannot disappear from TP I/O evidence.
        let mut accounting = self
            .tensor_reads
            .lock()
            .map_err(|_| model_error("TP read accounting poisoned"))?;
        for read in plan.reads.iter() {
            for rectangle in [read.weight_read(), read.scale_read()] {
                accounting.push(super::TensorParallelPreparationRead {
                    parameter: plan.binding.id(),
                    canonical: plan.binding.canonical_id(),
                    role: plan.binding.role().clone(),
                    bytes: rectangle.read_plan().storage_bytes(),
                    rectangle: Some(rectangle.clone()),
                });
            }
        }
        Ok(PreparedNumericFp8RankLinear {
            plan: plan.clone(),
            segments: segments.into(),
        })
    }
}

fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .filter(|n| *n <= isize::MAX as usize)
        .ok_or_else(|| model_error("numeric rank budget/extent overflow"))
}
fn grant(bytes: usize) -> Result<usize> {
    ferrule_backend::cuda_allocation_grant_bytes(bytes)
        .filter(|n| *n <= isize::MAX as usize)
        .ok_or_else(|| model_error("numeric rank allocation grant overflow"))
}

/// Physical live allocation classes for one prepared CUDA rank linear and
/// independently allocated input/output segments. No allocator pool slack or KV.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NumericFp8RankExecutionBudget {
    pub workspace_bytes: usize,
    pub workspace_grant_bytes: usize,
    pub resident_grant_bytes: usize,
    pub input_grant_bytes: usize,
    pub output_grant_bytes: usize,
    pub total_grant_bytes: usize,
}

/// Image-generation-bound lookup token, not an owning GPU handle. The CUDA
/// operator owner retains all artifacts/workspace through completion/quarantine.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct CudaNumericFp8RankBinding {
    pub(crate) generation: super::PreparedDecoderGeneration,
    pub(crate) slot: usize,
    pub(crate) plan: NumericFp8RankPlan,
}
#[cfg(feature = "cuda")]
impl CudaNumericFp8RankBinding {
    pub fn plan(&self) -> &NumericFp8RankPlan {
        &self.plan
    }
}

#[cfg(test)]
#[path = "../../tests/unit/transformer/numeric_shard.rs"]
pub(crate) mod tests;
