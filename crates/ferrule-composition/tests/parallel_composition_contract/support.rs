use ferrule_common::{ParallelRankId, ParallelTopologyId, ParallelismPlan};
use ferrule_composition::model_factory::{
    BackendSelection, ModelFactoryOptions, NumericFp8Profile, PipelineBuildOptions,
    ResidentModelBuildPlan, ResidentModelPlanner, ServingEngineSelection, ServingModelOptions,
};
use ferrule_model::models::qwen35::{Qwen35Config, Qwen35Recipe};
use ferrule_model::transformer::hybrid_parallel::{
    HybridExpertAttachment, HybridExpertScope, HybridParallelPlan,
};
use ferrule_model::transformer::{DecoderModelSpec, ExpertPlacement};
use ferrule_model::{
    AttentionKind, AutoConfig, ModelDescriptor, ModelFamily, MoeSpec, RouterKind,
    TransformerSemantics, TransformerSpec, WeightSource,
};
use ferrule_runtime::parallel::hybrid_topology::HybridExpertTopology;

pub struct Case {
    pub pp: usize,
    pub tp: usize,
    pub ep: usize,
    pub roots: &'static [&'static [u32]],
    pub experts: &'static [&'static [u32]],
    pub devices: &'static [usize],
}

// Expected rosters are literal acceptance facts, not computed with the lowering algorithm.
pub const PP2_TP1_EP2: Case = Case {
    pp: 2,
    tp: 1,
    ep: 2,
    roots: &[&[0], &[1]],
    experts: &[&[2, 3], &[4, 5]],
    devices: &[0, 2, 0, 1, 2, 3],
};
pub const PP2_TP1_EP4: Case = Case {
    pp: 2,
    tp: 1,
    ep: 4,
    roots: &[&[0], &[1]],
    experts: &[&[2, 3, 4, 5], &[6, 7, 8, 9]],
    devices: &[0, 4, 0, 1, 2, 3, 4, 5, 6, 7],
};
pub const PP1_TP2_EP2: Case = Case {
    pp: 1,
    tp: 2,
    ep: 2,
    roots: &[&[0, 1]],
    experts: &[&[2, 3]],
    devices: &[0, 1, 2, 3],
};
pub const PP2_TP2_EP2: Case = Case {
    pp: 2,
    tp: 2,
    ep: 2,
    roots: &[&[0, 1], &[2, 3]],
    experts: &[&[4, 5], &[6, 7]],
    devices: &[0, 1, 2, 3, 4, 5, 6, 7],
};
pub const PP2_TP4_EP2: Case = Case {
    pp: 2,
    tp: 4,
    ep: 2,
    roots: &[&[0, 1, 2, 3], &[4, 5, 6, 7]],
    experts: &[&[8, 9], &[10, 11]],
    devices: &[0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3],
};
impl Case {
    pub fn parallelism(&self) -> ParallelismPlan {
        ParallelismPlan {
            pipeline_parallel: self.pp,
            tensor_parallel: self.tp,
            expert_parallel: self.ep,
            ..Default::default()
        }
    }
    pub fn options(&self) -> PipelineBuildOptions {
        PipelineBuildOptions {
            parallelism: self.parallelism(),
            devices: Some(self.devices.to_vec()),
            ..Default::default()
        }
    }
}

pub fn rank(n: usize) -> ParallelRankId {
    ParallelRankId::new(u32::try_from(n).unwrap())
}

pub fn spec() -> DecoderModelSpec {
    let value = serde_json::from_str(include_str!(
        "../../../ferrule-model/tests/qwen35_35b_fp8_config.json"
    ))
    .unwrap();
    Qwen35Recipe::spec(&Qwen35Config::from_value(&value).unwrap()).unwrap()
}

// No checkpoint files, payload allocations, tokenizer, CUDA context, or owners.
pub fn config(family: ModelFamily) -> AutoConfig {
    let moe = family == ModelFamily::Qwen35Moe;
    AutoConfig::from_descriptor(ModelDescriptor {
        path: std::path::PathBuf::from("metadata-only-no-checkpoint"),
        spec: TransformerSpec {
            family,
            architecture: Some(
                if moe {
                    "Qwen3_5MoeForConditionalGeneration"
                } else {
                    "Qwen3ForCausalLM"
                }
                .into(),
            ),
            weight_source: WeightSource::Safetensors,
            hidden_size: Some(2048),
            num_layers: Some(40),
            vocab_size: Some(248320),
            num_heads: Some(16),
            num_kv_heads: Some(2),
            head_dim: Some(256),
            attention: if moe {
                AttentionKind::Unknown("qwen35_hybrid_linear_full".into())
            } else {
                AttentionKind::GroupedQuery
            },
            moe: if moe {
                MoeSpec {
                    num_experts: Some(256),
                    num_experts_per_tok: Some(8),
                    has_shared_experts: true,
                    router: RouterKind::DenseTopK,
                }
            } else {
                MoeSpec::none()
            },
            semantics: TransformerSemantics::default(),
            tensor_count: None,
            quantization: Vec::new(),
            notes: Vec::new(),
        },
        tensor_classes: Vec::new(),
    })
}

pub fn options(numeric: Option<NumericFp8Profile>) -> ModelFactoryOptions {
    ModelFactoryOptions {
        max_layers: None,
        max_tensor_mebibytes: 1024,
        output_head_chunk_rows: 4096,
        expert_reader_max_tensor_mebibytes: 64,
        expert_cache: Default::default(),
        qwen35_moe_capacity: None,
        numeric_fp8_profile: numeric,
        qwen35_host_cache: None,
        moe_hotset_experts: 0,
        kv_cache_mebibytes: Some(64),
        scheduler_config: ferrule_runtime::ResidentSchedulerConfig {
            max_active_sequences: 2,
            max_batch_tokens: 8,
            prefill_chunk_size: 8,
            prefix_cache_capacity_pages: 0,
            ..Default::default()
        },
        driver_config: ferrule_runtime::ResidentTopKDriverConfig {
            ctx_size: 32,
            enable_native_proposals: false,
            ..Default::default()
        },
    }
}

pub fn prepare(
    parallel: PipelineBuildOptions,
    numeric: Option<NumericFp8Profile>,
) -> ferrule_composition::Result<ResidentModelBuildPlan> {
    ResidentModelPlanner.prepare_serving(
        &config(ModelFamily::Qwen35Moe),
        BackendSelection::Cuda,
        None,
        options(numeric),
        ServingModelOptions {
            engine: ServingEngineSelection::Pipeline,
            parallel: Some(parallel),
            max_tensor_mebibytes: Some(2048),
            ..Default::default()
        },
    )
}

pub fn topology(spec: &DecoderModelSpec, case: &Case) -> HybridExpertTopology {
    let stages = (0..case.pp)
        .map(|stage| {
            let layers = stage * 40 / case.pp..(stage + 1) * 40 / case.pp;
            let first_expert = case.pp * case.tp + stage * case.ep;
            HybridParallelPlan::new(
                spec,
                layers.clone(),
                HybridExpertScope {
                    topology: ParallelTopologyId::new(407),
                    replica: 0,
                    stage: stage as u32,
                },
                (0..case.tp)
                    .map(|lane| rank(stage * case.tp + lane))
                    .collect(),
                Some(HybridExpertAttachment {
                    source_rank: rank(stage * case.tp),
                    members: (0..case.ep).map(|slot| rank(first_expert + slot)).collect(),
                    placement: ExpertPlacement::new(layers.flat_map(|layer| {
                        (0..256).map(move |expert| {
                            (layer, expert, rank(first_expert + expert % case.ep))
                        })
                    }))
                    .unwrap(),
                }),
            )
            .unwrap()
        })
        .collect();
    HybridExpertTopology::new(spec, stages).unwrap()
}
