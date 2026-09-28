//! Exercises preparation and the same FFN composition used by standard forward.
use super::*;
use crate::checkpoint::{CheckpointDType, CheckpointTensorSlice};
use crate::nn::{ModulePath, ParameterDType, ParameterId, ParameterResidency, ParameterSpec};
use crate::transformer::expert_parallel::{
    CpuExpertResultExecutor, CpuReferenceExpertWorker, ExpertDispatchLimits,
    ExpertParallelRoutedExecutor, ExpertPlacement,
};
use crate::transformer::{
    DecoderModelParts, Embedding, ExactNameMapper, GqaAttention, HostRows, MoeRouterSpec,
    NameMapping, RotaryPairing, RotaryRegion, RouterScoreFunction, RouterSelection, RowsDType,
    RowsShape, StateDictBinder, StateDictSchema,
};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

fn oracle() -> Value {
    serde_json::from_str(include_str!(
        "../../../tests/fixtures/shared_gate_oracle.json"
    ))
    .unwrap()
}
fn flat(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}
fn rows(v: &Value) -> Rows {
    let r = v.as_array().unwrap();
    Rows::Host(
        HostRows::new(
            RowsShape::new(r.len(), r[0].as_array().unwrap().len()).unwrap(),
            RowsDType::F32,
            None,
            r.iter().flat_map(flat).collect(),
        )
        .unwrap(),
    )
}
fn moe() -> Moe {
    Moe::new(
        4,
        3,
        MoeRouterSpec::new(
            3,
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
    .with_shared_expert(SwiGlu::new(4, 5, false).unwrap())
    .unwrap()
    .with_shared_expert_gate(Linear::new(4, 1, false).unwrap())
    .unwrap()
}
struct Fixture {
    file: PathBuf,
    resources: BoundDecoderResources,
    materializer: StateDictMaterializer,
    block: PreparedFeedForwardBlock,
}
impl Fixture {
    fn new(zero_gate: bool) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let file = std::env::temp_dir().join(format!(
            "ferrule-shared-gate-{}-{}.bin",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut schema = StateDictSchema::builder();
        let mut mapper = ExactNameMapper::new();
        let mut slices = Vec::new();
        let mut payload = Vec::new();
        let value = oracle();
        let mut weights = value["weights"].as_object().unwrap().clone();
        weights.insert(
            "norm".into(),
            serde_json::json!({"shape":[4], "values":[1.,1.,1.,1.]}),
        );
        for (name, tensor) in &weights {
            let (role, residency) = match name.as_str() {
                "norm" => (TensorRole::FeedForwardNorm, ParameterResidency::layer(0)),
                "router" => (TensorRole::RouterLogits, ParameterResidency::layer(0)),
                "shared_gate" => (
                    TensorRole::SharedExpertOutputGate,
                    ParameterResidency::layer(0),
                ),
                "shared.gate" => (TensorRole::SharedExpertGate, ParameterResidency::layer(0)),
                "shared.up" => (TensorRole::SharedExpertUp, ParameterResidency::layer(0)),
                "shared.down" => (TensorRole::SharedExpertDown, ParameterResidency::layer(0)),
                _ => {
                    let p = name.split('.').collect::<Vec<_>>();
                    let role = match p[2] {
                        "gate" => TensorRole::RoutedExpertGate,
                        "up" => TensorRole::RoutedExpertUp,
                        "down" => TensorRole::RoutedExpertDown,
                        _ => panic!("bad projection"),
                    };
                    (role, ParameterResidency::expert(0, p[1].parse().unwrap()))
                }
            };
            let shape = tensor["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect::<Vec<_>>();
            let mut values = flat(&tensor["values"]);
            if zero_gate && name == "shared_gate" {
                values.fill(0.);
            }
            let path = ModulePath::new(format!("layers.0.{name}.weight")).unwrap();
            schema
                .register_with_role(
                    ParameterSpec::new(
                        ParameterId::new(slices.len() as u64 + 1),
                        path.clone(),
                        ParameterDType::F32,
                        shape.clone(),
                        residency,
                    )
                    .unwrap(),
                    role.clone(),
                )
                .unwrap();
            mapper
                .insert(name.clone(), NameMapping::weight(path))
                .unwrap();
            let offset = payload.len() as u64;
            payload.extend(values.iter().flat_map(|v| v.to_le_bytes()));
            slices.push(CheckpointTensorSlice {
                name: name.clone(),
                role,
                path: file.clone(),
                offset,
                bytes: (values.len() * 4) as u64,
                dtype: CheckpointDType::F32,
                shape,
            });
        }
        std::fs::write(&file, payload).unwrap();
        let schema = schema.build().unwrap();
        let bound = StateDictBinder::new(&schema, &mapper)
            .bind_slices(slices)
            .unwrap();
        let norm = RmsNorm::new(4, 1e-6).unwrap();
        let rope = RotaryEmbedding::new(
            2,
            10000.,
            RotaryPairing::SplitHalf,
            RotaryRegion::Prefix { dimensions: 2 },
            RotaryScaling::None,
        )
        .unwrap();
        let layer = DecoderLayer::new(
            0,
            norm.clone(),
            Attention::Gqa(GqaAttention::new(4, 2, 1, 2, false, rope).unwrap()),
            Residual::Add,
            norm.clone(),
            FeedForward::Moe(moe()),
            Residual::Add,
        )
        .unwrap();
        let spec = DecoderModelSpec::new(DecoderModelParts {
            architecture: "generic-gated-shared-moe".into(),
            hidden_size: 4,
            vocab_size: 7,
            max_sequence_length: Some(16),
            token_embedding: Embedding::new(7, 4, None).unwrap(),
            layers: vec![layer.clone()],
            final_norm: norm,
            output: Linear::new(4, 7, false).unwrap(),
            tie_word_embeddings: false,
        })
        .unwrap();
        let resources = BoundDecoderResources::new(spec, Arc::new(bound)).unwrap();
        let materializer = StateDictMaterializer::new(4096).unwrap();
        let mut cache = MemoryLayerWeightCache::new();
        let norm = prepare_layer_norm(
            0,
            TensorRole::FeedForwardNorm,
            layer.post_attention_norm(),
            &resources,
            &materializer,
            &mut cache,
        )
        .unwrap();
        let block =
            prepare_feed_forward(&layer, &resources, &materializer, &mut cache, norm).unwrap();
        Self {
            file,
            resources,
            materializer,
            block,
        }
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.file);
    }
}
fn compare(actual: &Rows, expected: &Value) {
    let expected = rows(expected);
    assert_eq!(actual.shape(), expected.shape());
    for (a, e) in actual
        .host()
        .unwrap()
        .values()
        .iter()
        .zip(expected.host().unwrap().values())
    {
        assert!(
            a.is_finite() && (a - e).abs() <= 2e-6 + 2e-5 * e.abs(),
            "{a} != {e}"
        );
    }
}

#[test]
fn shared_gate_standard_ffn_matches_transformers_and_ungated_is_preserved() {
    let value = oracle();
    let input = rows(&value["input"]);
    let mut f = Fixture::new(false);
    let mut ops = CpuStandardDecoderOperators::new(ExecutionPrecisionPolicy::f32());
    let output =
        execute_feed_forward(&mut ops, &f.materializer, 0, &f.block.kind, &input, None).unwrap();
    compare(&output, &value["output"]);
    let PreparedFeedForwardKind::Routed { shared_gate, .. } = &mut f.block.kind else {
        unreachable!()
    };
    *shared_gate = None;
    let output =
        execute_feed_forward(&mut ops, &f.materializer, 0, &f.block.kind, &input, None).unwrap();
    compare(&output, &value["ungated"]);
    let PreparedFeedForwardKind::Routed { shared, .. } = &mut f.block.kind else {
        unreachable!()
    };
    *shared = None;
    let output =
        execute_feed_forward(&mut ops, &f.materializer, 0, &f.block.kind, &input, None).unwrap();
    compare(&output, &value["routed"]);
    let f = Fixture::new(true);
    let output =
        execute_feed_forward(&mut ops, &f.materializer, 0, &f.block.kind, &input, None).unwrap();
    compare(&output, &value["zero_gate"]);
}

#[test]
fn shared_gate_is_added_once_after_expert_parallel_weighted_combine() {
    let value = oracle();
    let input = rows(&value["input"]);
    let f = Fixture::new(false);
    let PreparedFeedForwardKind::Routed { experts, .. } = &f.block.kind else {
        unreachable!()
    };
    let placement = ExpertPlacement::new([
        (0, 0, ParallelRankId::new(0)),
        (0, 1, ParallelRankId::new(1)),
        (0, 2, ParallelRankId::new(0)),
    ])
    .unwrap();
    let mut p0 = StateDictExpertProvider {
        layer: 0,
        bindings: experts,
        activation_limit: None,
        materializer: &f.materializer,
    };
    let mut p1 = StateDictExpertProvider {
        layer: 0,
        bindings: experts,
        activation_limit: None,
        materializer: &f.materializer,
    };
    let mut w0 = CpuReferenceExpertWorker::new(ParallelRankId::new(0), &placement, &mut p0);
    let mut w1 = CpuReferenceExpertWorker::new(ParallelRankId::new(1), &placement, &mut p1);
    let mut results = CpuExpertResultExecutor::new(vec![&mut w0, &mut w1]).unwrap();
    let mut executor = ExpertParallelRoutedExecutor::new(
        vec![ParallelRankId::new(0), ParallelRankId::new(1)],
        &placement,
        ExpertDispatchLimits {
            max_tokens: 8,
            max_bytes: 128,
        },
        &mut results,
    );
    let transaction = ExecutionTransactionId::new(1).unwrap();
    let mut checks = 0;
    let mut check = |id| {
        assert_eq!(id, transaction);
        checks += 1;
        Ok(())
    };
    let mut execution = RoutedLayerExecution {
        transaction,
        source_rank: ParallelRankId::new(0),
        sequences: &[10, 10, 11, 11],
        executor: &mut executor,
        check_active: &mut check,
    };
    let mut ops = CpuStandardDecoderOperators::new(ExecutionPrecisionPolicy::f32());
    let output = execute_feed_forward(
        &mut ops,
        &f.materializer,
        0,
        &f.block.kind,
        &input,
        Some(&mut execution),
    )
    .unwrap();
    compare(&output, &value["output"]);
    assert!(checks > 0);
    assert!(
        f.resources
            .require_layer(0, TensorRole::SharedExpertOutputGate)
            .is_ok()
    );
}

#[test]
fn shared_gate_descriptor_and_operator_fail_closed() {
    let m = moe();
    assert_eq!(m.shared_expert_gate().unwrap().weight_shape(), [1, 4]);
    for gate in [
        Linear::new(4, 2, false).unwrap(),
        Linear::new(3, 1, false).unwrap(),
        Linear::new(4, 1, true).unwrap(),
    ] {
        assert!(m.clone().with_shared_expert_gate(gate).is_err());
    }
    let bare = Moe::new(4, 3, m.router_spec().clone(), false).unwrap();
    assert!(
        bare.with_shared_expert_gate(Linear::new(4, 1, false).unwrap())
            .is_err()
    );
    let mut ops = CpuStandardDecoderOperators::new(ExecutionPrecisionPolicy::f32());
    let input = serde_json::json!([[2., -4., 8.], [2., -4., 8.], [2., -4., 8.]]);
    let gate = serde_json::json!([[100.], [-100.], [0.]]);
    let output = ready(ops.shared_expert_gate(rows(&input), &rows(&gate)).unwrap()).unwrap();
    compare(
        &output,
        &serde_json::json!([[2., -4., 8.], [0., 0., 0.], [1., -2., 4.]]),
    );
    assert!(ops.shared_expert_gate(rows(&input), &rows(&input)).is_err());
    assert!(
        ops.shared_expert_gate(rows(&input), &rows(&serde_json::json!([[0.]])))
            .is_err()
    );
}
