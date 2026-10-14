use ferrule_common::{ParallelRankId, ParallelTopologyId};
use ferrule_composition::model_factory::qwen35_composition::{
    HybridExecutionProfile, Qwen35CompositionAdapter,
};
use ferrule_composition::model_factory::{NumericFp8Profile, PipelineRankBackend};
use ferrule_model::transformer::hybrid_parallel::*;
use ferrule_model::transformer::*;
use ferrule_runtime::parallel::hybrid_topology::HybridExpertTopology;

fn rank(n: usize) -> ParallelRankId {
    ParallelRankId::new(n as u32)
}
fn spec() -> DecoderModelSpec {
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
                FeedForward::Moe(moe.clone()),
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
fn plan(stage: usize, tp: usize) -> HybridParallelPlan {
    let members = (stage * 4..stage * 4 + tp).map(rank).collect::<Vec<_>>();
    let owners = (stage * 4..stage * 4 + 4).map(rank).collect::<Vec<_>>();
    let placement = ExpertPlacement::new(
        (stage * 2..stage * 2 + 2).flat_map(|l| (0..4).map(move |e| (l, e, rank(stage * 4 + e)))),
    )
    .unwrap();
    HybridParallelPlan::new(
        &spec(),
        stage * 2..stage * 2 + 2,
        HybridExpertScope {
            topology: ParallelTopologyId::new(17),
            replica: 2,
            stage: stage as u32,
        },
        members,
        Some(HybridExpertAttachment {
            source_rank: rank(stage * 4),
            members: owners,
            placement,
        }),
    )
    .unwrap()
}

#[test]
fn pp2_tp4_ep4_composition_remains_production_gated() {
    let topology = HybridExpertTopology::new(&spec(), vec![plan(0, 4), plan(1, 4)]).unwrap();
    assert!(
        Qwen35CompositionAdapter::new(&topology)
            .validate_production_execution(HybridExecutionProfile {
                model: ferrule_model::models::qwen35::Qwen35Profile::Moe35BA3Bfp8,
                backend: ferrule_model::ModelExecutionBackend::Cuda,
                rank_backend: PipelineRankBackend::Thread,
                parallelism: ferrule_common::ParallelismPlan {
                    pipeline_parallel: 2,
                    tensor_parallel: 4,
                    expert_parallel: 4,
                    ..Default::default()
                },
                numeric: NumericFp8Profile::F32,
            })
            .is_err()
    );
}
