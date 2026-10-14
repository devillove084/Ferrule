//! Public metadata capability stays available with default features and no GPU.
use ferrule_common::{ParallelRankId, ParallelTopologyId, ParallelismPlan};
use ferrule_composition::model_factory::qwen35_composition::{
    HybridCapabilityError, HybridExecutionCapability, HybridExecutionProfile,
    Qwen35CompositionAdapter,
};
use ferrule_composition::model_factory::{NumericFp8Profile, PipelineRankBackend};
use ferrule_model::transformer::hybrid_parallel::*;
use ferrule_model::transformer::*;
use ferrule_model::{ModelExecutionBackend, models::qwen35::Qwen35Profile};
use ferrule_runtime::parallel::hybrid_topology::{HybridExpertTopology, HybridTopologyError};

fn rank(n: usize) -> ParallelRankId {
    ParallelRankId::new(n as u32)
}
fn spec(dense: bool) -> DecoderModelSpec {
    let norm = || RmsNorm::new(4, 1e-6).unwrap();
    let rope = RotaryEmbedding::new(
        2,
        10000.0,
        RotaryPairing::SplitHalf,
        RotaryRegion::Prefix { dimensions: 2 },
        RotaryScaling::None,
    )
    .unwrap();
    let moe = Moe::new(
        4,
        5,
        MoeRouterSpec::new(
            4,
            2,
            RouterScoreFunction::Softmax,
            RouterSelection::TopK,
            true,
            1.0,
        )
        .unwrap(),
        false,
    )
    .unwrap()
    .with_shared_expert(SwiGlu::new(4, 7, false).unwrap())
    .unwrap()
    .with_shared_expert_gate(Linear::new(4, 1, false).unwrap())
    .unwrap();
    let layers = (0..4)
        .map(|i| {
            DecoderLayer::new(
                i,
                norm(),
                if i % 2 == 0 {
                    Attention::GatedDeltaNet(
                        GatedDeltaNetAttention::new(4, 4, 8, 2, 2, 3, 1e-6, false).unwrap(),
                    )
                } else {
                    Attention::Gqa(GqaAttention::new(4, 8, 2, 2, false, rope.clone()).unwrap())
                },
                Residual::Add,
                norm(),
                if dense {
                    FeedForward::SwiGlu(SwiGlu::new(4, 5, false).unwrap())
                } else {
                    FeedForward::Moe(moe.clone())
                },
                Residual::Add,
            )
            .unwrap()
        })
        .collect();
    DecoderModelSpec::new(DecoderModelParts {
        architecture: "hybrid-tp-ep-reference".into(),
        hidden_size: 4,
        vocab_size: 8,
        max_sequence_length: Some(16),
        token_embedding: Embedding::new(8, 4, None).unwrap(),
        layers,
        final_norm: norm(),
        output: Linear::new(4, 8, false).unwrap(),
        tie_word_embeddings: false,
    })
    .unwrap()
}

fn layout(spec: &DecoderModelSpec, pp: usize, tp: usize, ep: usize) -> HybridExpertTopology {
    let stages =
        (0..pp)
            .map(|stage| {
                let layers = stage * (4 / pp)..(stage + 1) * (4 / pp);
                let attachment = matches!(spec.layers()[0].feed_forward(), FeedForward::Moe(_))
                    .then(|| HybridExpertAttachment {
                        source_rank: rank(stage * tp),
                        members: (0..ep)
                            .map(|slot| rank(pp * tp + stage * ep + slot))
                            .collect(),
                        placement: ExpertPlacement::new(layers.clone().flat_map(|layer| {
                            (0..4).map(move |expert| {
                                (layer, expert, rank(pp * tp + stage * ep + expert % ep))
                            })
                        }))
                        .unwrap(),
                    });
                HybridParallelPlan::new(
                    spec,
                    layers,
                    HybridExpertScope {
                        topology: ParallelTopologyId::new(1),
                        replica: 0,
                        stage: stage as u32,
                    },
                    (0..tp).map(|lane| rank(stage * tp + lane)).collect(),
                    attachment,
                )
                .unwrap()
            })
            .collect();
    HybridExpertTopology::new(spec, stages).unwrap()
}

fn profile(pp: usize, tp: usize) -> HybridExecutionProfile {
    HybridExecutionProfile {
        model: Qwen35Profile::Moe35BA3Bfp8,
        backend: ModelExecutionBackend::Cuda,
        rank_backend: PipelineRankBackend::Thread,
        parallelism: ParallelismPlan {
            pipeline_parallel: pp,
            tensor_parallel: tp,
            expert_parallel: 2,
            ..Default::default()
        },
        numeric: NumericFp8Profile::F32,
    }
}

fn rejection(error: ferrule_common::Error, expected: HybridCapabilityError) {
    assert_eq!(
        ferrule_common::error::ErrorChain::new(&error).downcast_ref::<HybridCapabilityError>(),
        Some(&expected)
    );
}

#[test]
fn metadata_queries_preserve_factory_support_and_quality_distinctions() {
    for pp in [1, 2] {
        for tp in [2, 4] {
            let topology = layout(&spec(false), pp, tp, 2);
            let adapter = Qwen35CompositionAdapter::new(&topology);
            let mut p = profile(pp, tp);
            assert_eq!(
                adapter.execution_capability(p).unwrap(),
                if tp == 2 {
                    HybridExecutionCapability::F32WithFullModelStrictEvidence
                } else {
                    HybridExecutionCapability::F32FactoryOnly
                }
            );
            adapter.validate_production_execution(p).unwrap();
            p.numeric = NumericFp8Profile::W8A16Bf16;
            assert_eq!(
                adapter.execution_capability(p).unwrap(),
                HybridExecutionCapability::W8A16QualityUnaccepted
            );
            rejection(
                adapter.validate_production_execution(p).unwrap_err(),
                HybridCapabilityError::W8A16QualityUnaccepted,
            );
        }
    }
}

#[test]
fn metadata_rejects_wrong_family_backend_rank_and_degrees_without_cuda() {
    let topology = layout(&spec(false), 2, 2, 2);
    let adapter = Qwen35CompositionAdapter::new(&topology);
    let mut p = profile(2, 2);
    p.model = Qwen35Profile::Dense08Bbf16;
    rejection(
        adapter.execution_capability(p).unwrap_err(),
        HybridCapabilityError::ModelProfile,
    );
    p = profile(2, 2);
    p.backend = ModelExecutionBackend::Cpu;
    rejection(
        adapter.execution_capability(p).unwrap_err(),
        HybridCapabilityError::Backend,
    );
    p = profile(2, 2);
    p.rank_backend = PipelineRankBackend::Process;
    rejection(
        adapter.execution_capability(p).unwrap_err(),
        HybridCapabilityError::RankBackend,
    );
    let mutations: [fn(&mut ParallelismPlan); 8] = [
        |p| p.pipeline_parallel = 0,
        |p| p.pipeline_parallel = 3,
        |p| p.tensor_parallel = 1,
        |p| p.tensor_parallel = usize::MAX,
        |p| p.expert_parallel = 4,
        |p| p.data_parallel = 2,
        |p| p.sequence_parallel = 2,
        |p| p.context_parallel = 2,
    ];
    for mutate in mutations {
        let mut p = profile(2, 2);
        mutate(&mut p.parallelism);
        rejection(
            adapter.execution_capability(p).unwrap_err(),
            HybridCapabilityError::Parallelism,
        );
    }
}

#[test]
fn dense_or_attached_specs_do_not_inherit_declared_moe_support() {
    let dense = layout(&spec(true), 2, 2, 2);
    rejection(
        Qwen35CompositionAdapter::new(&dense)
            .execution_capability(profile(2, 2))
            .unwrap_err(),
        HybridCapabilityError::ModelProfile,
    );
    let spec = spec(false)
        .with_attachment(DecoderAttachmentSpec::Proposal(
            ProposalAttachmentSpec::new(ProposalAttachmentParts {
                path: "proposal".into(),
                hidden_size: 4,
                block_size: 1,
                noise_token_id: None,
                target_layer_ids: vec![0],
                stages: vec![],
                main_projection: None,
                main_norm: None,
                heads: None,
            })
            .unwrap(),
        ))
        .unwrap();
    let topology = layout(&spec, 2, 2, 2);
    assert!(topology.has_attachments());
    rejection(
        Qwen35CompositionAdapter::new(&topology)
            .execution_capability(profile(2, 2))
            .unwrap_err(),
        HybridCapabilityError::ModelProfile,
    );
}

#[test]
fn neutral_roster_does_not_select_a_family_or_approve_execution() {
    let topology = layout(&spec(false), 2, 1, 4);
    let mut p = profile(2, 1);
    p.parallelism.expert_parallel = 4;
    topology
        .validate_external_expert_roster(p.parallelism)
        .unwrap();
    assert_eq!(topology.physical_members().len(), 10);
    rejection(
        Qwen35CompositionAdapter::new(&topology)
            .execution_capability(p)
            .unwrap_err(),
        HybridCapabilityError::Parallelism,
    );
    p.parallelism.tensor_parallel = usize::MAX;
    assert_eq!(
        topology.validate_external_expert_roster(p.parallelism),
        Err(HybridTopologyError::Parallelism)
    );
    p.parallelism.tensor_parallel = 1;
    p.parallelism.expert_parallel = usize::MAX;
    assert_eq!(
        topology.validate_external_expert_roster(p.parallelism),
        Err(HybridTopologyError::Parallelism)
    );
}
