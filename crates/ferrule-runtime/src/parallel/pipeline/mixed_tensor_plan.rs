//! Sealed, CPU-only protocol inputs for the experimental typed CUDA owner leaf.
//! This is not public profile/quality admission and never opens a device.
use super::*;
use crate::parallel::collective::HostCollectiveLimits;
use crate::parallel::hybrid_topology::HybridExpertTopology;
use crate::parallel::tensor::decoder_collective::MixedRankCollectivePlan;
use ferrule_backend::cuda_plan::bf16_attention::Bf16AttentionShape;
use ferrule_common::ParallelGroupId;
use ferrule_model::transformer::DecoderModelSpec;
use ferrule_model::transformer::hybrid_parallel::{
    HYBRID_AGREEMENT_VERSION, HybridMixedRankProgramPlan, HybridRankPrecision,
    HybridRoutedExpertExecution,
};

#[derive(Clone)]
pub struct MixedTensorExecutionPlan {
    config: PipelineConfig,
    programs: Vec<HybridMixedRankProgramPlan>,
    collective: Arc<MixedRankCollectivePlan>,
    attention_shapes: Vec<Bf16AttentionShape>,
}
impl MixedTensorExecutionPlan {
    pub fn new(
        spec: &DecoderModelSpec,
        topology: &ValidatedParallelTopology,
        segments: &[LayerSegmentPlan],
        layout: &HybridExpertTopology,
        config: PipelineConfig,
        limits: HostCollectiveLimits,
    ) -> Result<Self> {
        config.validate()?;
        let p = topology.plan();
        if p.pipeline_parallel != 1
            || p.tensor_parallel != 2
            || p.expert_parallel != 1
            || p.data_parallel != 1
            || p.sequence_parallel != 1
            || p.context_parallel != 1
            || config.precision != ExecutionPrecisionPolicy::bf16_compatibility()
            || segments.len() != 1
            || layout.stages().len() != 1
            || segments[0].layers() != (0..spec.layers().len())
            || !segments[0].owns_embedding()
            || !segments[0].owns_output()
        {
            return Err(error(
                "experimental Mixed owner requires complete PP1 TP2 EP1 BF16 segment",
            ));
        }
        layout.bind_tensor_scopes(topology, segments, &[])?;
        let stage = &layout.stages()[0];
        stage.validate_spec(spec)?;
        if stage.precision() != HybridRankPrecision::MixedBf16F32
            || stage.routed_execution() != &HybridRoutedExpertExecution::LocalTensorShards
        {
            return Err(error(
                "Mixed owner requires full resident local tensor expert shards",
            ));
        }
        let copies = config
            .session_capacity
            .checked_mul(2)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| error("Mixed state copies overflow"))?;
        let programs = (0..2)
            .map(|lane| {
                HybridMixedRankProgramPlan::for_segment(
                    Arc::new(stage.rank_layout(lane)?),
                    segments[0].clone(),
                    config.max_batch_tokens,
                    config.session_capacity,
                    config.max_positions,
                    config.page_size,
                    copies,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let source = &programs[0];
        let collective = Arc::new(
            MixedRankCollectivePlan::from_hybrid_sites(
                stage,
                ParallelGroupId::new(1),
                stage.tensor_members()[0],
                HYBRID_AGREEMENT_VERSION,
                &source.sites(),
                limits,
                config.session_capacity,
                config.max_batch_tokens,
            )?
            .with_route_sites(source.route_sites())?,
        );
        let attention_shapes = if source.gqa_layer().is_some() {
            (1..=config.max_positions)
                .flat_map(|tokens| {
                    (1..=config.max_batch_tokens.min(tokens)).map(move |rows| Bf16AttentionShape {
                        rows: rows as u32,
                        tokens: tokens as u32,
                    })
                })
                .collect()
        } else {
            Vec::new()
        };
        for program in &programs {
            if program.collective_binding() != source.collective_binding()
                || program.route_sites() != source.route_sites()
            {
                return Err(error(
                    "Mixed root programs disagree on exact data/control schedule",
                ));
            }
            if program.gqa_layer().is_some() {
                program.attention_plan(&attention_shapes, usize::MAX, usize::MAX)?;
            }
        }
        Ok(Self {
            config,
            programs,
            collective,
            attention_shapes,
        })
    }
    /// Explicit internal candidate; both owners retain the model-admitted program.
    /// Capacities are in lane order, not CUDA ordinal order. No device is opened.
    #[doc(hidden)]
    pub fn with_decoded_weight_cache(
        mut self,
        resources: &ferrule_model::transformer::BoundDecoderResources,
        requested_cache_grants: usize,
        device_capacities: [usize; 2],
        owner_margins: usize,
    ) -> Result<Self> {
        let pages = PhysicalKvLayout::hybrid(ferrule_model::ModelExecutionBackend::Cuda)
            .physical_pages(self.config.max_pages)?;
        self.programs = self
            .programs
            .into_iter()
            .zip(device_capacities)
            .map(|(program, capacity)| {
                program.with_decoded_weight_cache(
                    resources,
                    pages,
                    requested_cache_grants,
                    capacity,
                    owner_margins,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(self)
    }

    /// Preserve metadata-admitted model programs; only their cache admission may
    /// differ from the native schedule. No capacity/layout drift is accepted.
    #[doc(hidden)]
    pub fn with_programs(mut self, programs: Vec<HybridMixedRankProgramPlan>) -> Result<Self> {
        if programs.len() != self.programs.len()
            || programs.iter().zip(&self.programs).any(|(p, native)| {
                p.layout() != native.layout()
                    || p.segment() != native.segment()
                    || p.collective_binding() != native.collective_binding()
                    || p.route_sites() != native.route_sites()
                    || p.page_size() != native.page_size()
                    || p.max_positions() != native.max_positions()
                    || p.state_copies() != native.state_copies()
            })
        {
            return Err(error(
                "admitted Mixed programs differ from runtime layout/capacities",
            ));
        }
        self.programs = programs;
        Ok(self)
    }

    pub fn config(&self) -> PipelineConfig {
        self.config
    }
    pub fn program(&self, lane: usize) -> Result<&HybridMixedRankProgramPlan> {
        self.programs
            .get(lane)
            .ok_or_else(|| error("Mixed lane outside sealed roots"))
    }
    pub fn collective(&self) -> &Arc<MixedRankCollectivePlan> {
        &self.collective
    }
    pub fn attention_shapes(&self) -> &[Bf16AttentionShape] {
        &self.attention_shapes
    }
}
