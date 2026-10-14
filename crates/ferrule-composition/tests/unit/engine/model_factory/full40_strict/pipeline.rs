//! Shared full40 F32 acceptance/lifecycle body; adapters only construct factories.
use super::oracle::{ATOL, Oracle, RTOL, StrictReport};
use crate::model_factory::*;
use ferrule_common::execution::ForwardPhase;
use ferrule_model::models::qwen35::{Qwen35Adapter, Qwen35Profile};
use ferrule_model::transformer::BoundDecoderResources;
use ferrule_runtime::SessionId;
use ferrule_runtime::parallel::pipeline::PipelineParallelExecutor;
use serde_json::json;
use std::{io::Write, path::PathBuf, sync::atomic::AtomicBool};

const VOCABULARY: usize = 248_320;
const BASE_LOGITS: usize = 1_986_560;

#[derive(Clone, Copy, Debug)]
pub(in crate::model_factory) struct Topology {
    pub pp: usize,
    pub tp: usize,
    pub ep: usize,
}
impl Topology {
    pub fn owners(self) -> usize {
        self.pp * (self.tp + self.ep)
    }
    fn label(self) -> String {
        format!("pp{}-tp{}-ep{}-full40-f32", self.pp, self.tp, self.ep)
    }
    fn validate(self, devices: &[usize]) {
        assert!(matches!(
            (self.pp, self.tp, self.ep),
            (2, 1, 2 | 4) | (1 | 2, 2, 2)
        ));
        assert_eq!(
            devices.len(),
            self.owners(),
            "explicit roots-then-stage-EP placement required"
        );
        assert!(devices.iter().all(|&d| i32::try_from(d).is_ok()));
    }
}
fn required_path(name: &str) -> PathBuf {
    let value = std::env::var_os(name)
        .unwrap_or_else(|| panic!("set {name} explicitly; no fixture fallback"));
    assert!(!value.is_empty(), "empty {name}");
    PathBuf::from(value)
}

pub(in crate::model_factory) fn run(
    topology: Topology,
    build: impl FnOnce(&ResolvedModelRequest, BoundDecoderResources) -> PipelineParallelExecutor,
) {
    assert_eq!(
        std::env::var("FERRULE_QWEN35_STRICT_RUN").as_deref(),
        Ok("1"),
        "manual GPU owner opt-in requires FERRULE_QWEN35_STRICT_RUN=1 and --ignored"
    );
    let model = required_path("FERRULE_QWEN35_STRICT_MODEL");
    let reference = required_path("FERRULE_NUMERIC_FP8_F32_REFERENCE_DIR");
    let report_dir = required_path("FERRULE_QWEN35_STRICT_REPORT_DIR");
    let devices = std::env::var("FERRULE_QWEN35_STRICT_DEVICES")
        .expect("set FERRULE_QWEN35_STRICT_DEVICES: all PP-major TP roots then stage EP children")
        .split(',')
        .map(|n| n.parse::<usize>().expect("invalid CUDA ordinal"))
        .collect::<Vec<_>>();
    topology.validate(&devices);
    // Verify independent oracle provenance and exact coverage before starting GPU owners.
    let oracles = [
        Oracle::open(&reference, "hello"),
        Oracle::open(&reference, "capital"),
    ];
    let rows: usize = oracles
        .iter()
        .map(|oracle| {
            ["prefill", "decode.0"]
                .iter()
                .enumerate()
                .map(|(index, label)| {
                    let tokens = oracle.report["calls"][index]["input_token_ids"]
                        .as_array()
                        .unwrap();
                    assert!(!tokens.is_empty() && tokens.len() <= 8);
                    if index == 1 {
                        assert_eq!(tokens.len(), 1);
                    }
                    assert_eq!(
                        oracle.header[format!("{label}.logits")]["shape"],
                        json!([1, tokens.len(), VOCABULARY])
                    );
                    tokens.len()
                })
                .sum::<usize>()
        })
        .sum();
    assert_eq!(rows * VOCABULARY, BASE_LOGITS);
    std::fs::create_dir_all(&report_dir).unwrap();
    let report_path = report_dir.join(format!("{}.json", topology.label()));
    // A rerun must not overwrite historical acceptance evidence.
    let mut report_file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&report_path)
        .unwrap();
    ferrule_common::observability::init_tracing();
    let config = ferrule_model::AutoConfig::from_pretrained(&model).unwrap();
    let mut input = crate::model_factory::tests::options();
    input.max_tensor_mebibytes = 2048;
    input.numeric_fp8_profile = None; // Exercise the production F32 default, never W8.
    input.driver_config.enable_native_proposals = false;
    input.driver_config.ctx_size = 32;
    input.scheduler_config.max_active_sequences = 2;
    input.scheduler_config.max_batch_tokens = 8;
    input.scheduler_config.prefill_chunk_size = 8;
    let plan = ResidentModelPlanner
        .prepare_serving(
            &config,
            BackendSelection::Cuda,
            None,
            input,
            ServingModelOptions {
                // ModelFactoryOptions alone is overwritten by serving default resolution.
                max_tensor_mebibytes: Some(2048),
                engine: ServingEngineSelection::Pipeline,
                parallel: Some(PipelineBuildOptions {
                    parallelism: ferrule_common::ParallelismPlan {
                        pipeline_parallel: topology.pp,
                        tensor_parallel: topology.tp,
                        expert_parallel: topology.ep,
                        ..Default::default()
                    },
                    devices: Some(devices.clone()),
                    ..Default::default()
                }),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(
        plan.selected_implementation().kind,
        SelectedImplementationKind::Pipeline
    );
    let expected: fn(ModelBuildRequest) -> Result<BoxedSessionInferenceEngine> =
        build_hybrid_pipeline;
    assert!(std::ptr::fn_addr_eq(
        plan.request.execution.construction.builder(),
        expected
    ));
    assert_eq!(plan.numeric_fp8_profile(), Some(NumericFp8Profile::F32));
    assert_eq!(plan.observability().max_layers, 40);
    assert_eq!(plan.request.max_tensor_bytes, 2048 << 20);
    let request = plan.request;
    let (metadata, resources) = Qwen35Adapter::bind_hf_metadata(&model).unwrap();
    assert_eq!(metadata.config().profile(), Qwen35Profile::Moe35BA3Bfp8);
    assert_eq!(resources.spec().layers().len(), 40);
    let mut pp = build(&request, resources);
    assert_eq!(pp.stage_descriptions().len(), topology.pp * topology.tp);
    for (index, description) in pp.stage_descriptions().iter().enumerate() {
        let stage = index / topology.tp;
        assert_eq!(
            description.plan.layers(),
            stage * (40 / topology.pp)..(stage + 1) * (40 / topology.pp)
        );
        assert_eq!(description.expert_group.is_some(), index % topology.tp == 0);
        if let Some(group) = &description.expert_group {
            assert_eq!(group.source_rank.get() as usize, index);
            assert_eq!(
                group
                    .members
                    .iter()
                    .map(|r| r.get() as usize)
                    .collect::<Vec<_>>(),
                (0..topology.ep)
                    .map(|slot| topology.pp * topology.tp + stage * topology.ep + slot)
                    .collect::<Vec<_>>()
            );
        }
    }
    let mut result = StrictReport::default();
    for (case_index, (case, oracle)) in ["hello", "capital"].into_iter().zip(&oracles).enumerate() {
        let session = SessionId(10 + case_index as u64 * 10);
        let fork = SessionId(session.0 + 1);
        let mut position = 0;
        let mut prefill_tokens = Vec::new();
        let mut prefill_expected = Vec::new();
        for (index, label) in ["prefill", "decode.0"].iter().enumerate() {
            let tokens: Vec<u32> =
                serde_json::from_value(oracle.report["calls"][index]["input_token_ids"].clone())
                    .unwrap();
            assert_eq!(oracle.report["calls"][index]["position_start"], position);
            let expected = oracle.values(&format!("{label}.logits"));
            assert_eq!(expected.len(), tokens.len() * VOCABULARY);
            let phase = if index == 0 {
                ForwardPhase::Prefill
            } else {
                ForwardPhase::Decode
            };
            if index == 1 {
                pp.fork_session(session, fork).unwrap();
                let publications = pp.coordinator().publication_count();
                let pages = pp.page_manager().allocated_pages();
                assert!(
                    pp.forward_cancellable(session, &tokens, phase, &AtomicBool::new(true))
                        .is_err()
                );
                assert_eq!(pp.coordinator().publication_count(), publications);
                assert_eq!(pp.page_manager().allocated_pages(), pages);
                assert!(!pp.is_quarantined());
            }
            let actual = pp.forward(session, &tokens, phase).unwrap();
            for (row, expected) in expected.as_chunks::<VOCABULARY>().0.iter().enumerate() {
                result.values(
                    actual.logits.row(row).unwrap(),
                    expected,
                    &format!("{case}/{label}/row{row}"),
                );
                result.logits += VOCABULARY;
            }
            assert!(
                actual.logits.row(tokens.len()).is_none(),
                "unexpected extra logit rows"
            );
            position += tokens.len();
            assert_eq!(
                oracle.report["calls"][index]["sequence_length_after"],
                position
            );
            assert_eq!(
                pp.page_manager()
                    .block_table(pp.session_slot(session).unwrap())
                    .unwrap()
                    .committed_tokens(),
                position
            );
            if index == 0 {
                prefill_tokens = tokens;
                prefill_expected = expected;
            } else {
                let forked = pp.forward(fork, &tokens, phase).unwrap();
                for (row, expected) in expected.as_chunks::<VOCABULARY>().0.iter().enumerate() {
                    result.values(
                        forked.logits.row(row).unwrap(),
                        expected,
                        &format!("{case}/fork-decode/row{row}"),
                    );
                    assert_eq!(
                        actual.logits.row(row).unwrap(),
                        forked.logits.row(row).unwrap()
                    );
                }
                pp.release_session(fork).unwrap();
            }
        }
        pp.release_session(session).unwrap();
        assert_eq!(pp.page_manager().allocated_pages(), 0);
        assert_eq!(pp.outstanding(), 0);
        assert!(!pp.is_quarantined());
        let replay = pp
            .forward(session, &prefill_tokens, ForwardPhase::Prefill)
            .unwrap();
        for (row, expected) in prefill_expected
            .as_chunks::<VOCABULARY>()
            .0
            .iter()
            .enumerate()
        {
            result.values(
                replay.logits.row(row).unwrap(),
                expected,
                &format!("{case}/released-new-state/row{row}"),
            );
        }
        pp.release_session(session).unwrap();
        assert_eq!(pp.page_manager().allocated_pages(), 0);
        assert_eq!(pp.outstanding(), 0);
    }
    assert_eq!(result.logits, BASE_LOGITS);
    let all_elements: u64 = result
        .metrics
        .iter()
        .map(|m| m["elements"].as_u64().unwrap())
        .sum();
    assert_eq!(all_elements, 2 * BASE_LOGITS as u64);
    pp.shutdown().unwrap();
    let passed = result.failures.is_empty();
    let report = json!({"passed": passed, "atol": ATOL, "rtol": RTOL,
        "comparison": "all-vocabulary, all prefill/decode rows; F64 differences; both operands finite",
        "logits_compared": result.logits, "all_elements_including_fork_recreate": all_elements,
        "metrics": result.metrics, "failures": result.failures, "pp": topology.pp, "tp": topology.tp,
        "ep": topology.ep, "devices": devices, "model": model, "oracle_directory": reference,
        "layers": 40, "kv_layers": 10, "max_tensor_mebibytes": 2048, "precision": "f32",
        "state_conditions": "prefill/decode frontier, fork same logits, pre-admission cancellation preserves publication/pages, release/new-state recompute, zero pages/outstanding, shutdown",
        "route_oracle": "not exposed by PP owner API; NOT compared",
        "state_tensor_oracle": "not exposed by PP owner API; NOT compared directly", "shutdown": "complete"});
    report_file
        .write_all(&serde_json::to_vec_pretty(&report).unwrap())
        .unwrap();
    eprintln!(
        "{} passed={passed} report={}",
        topology.label(),
        report_path.display()
    );
    assert!(
        passed,
        "strict F32 pipeline oracle mismatch; inspect report"
    );
}

#[test]
fn strict_topologies_charge_stage_attachments_not_a_multiplicative_mesh() {
    for (pp, tp, ep, owners) in [(2, 1, 2, 6), (2, 1, 4, 10), (1, 2, 2, 4), (2, 2, 2, 8)] {
        let t = Topology { pp, tp, ep };
        assert_eq!(t.owners(), owners);
        t.validate(&vec![0; owners]);
        assert!(std::panic::catch_unwind(|| t.validate(&vec![0; owners - 1])).is_err());
    }
}
