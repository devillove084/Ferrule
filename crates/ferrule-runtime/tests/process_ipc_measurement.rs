//! PR26: CPU correctness and opt-in measurements over actual child pipes.
//! See scripts/bench_process_ipc.py for pinned inputs, environment and 100s bound.
#![cfg(unix)]

#[path = "support/pr26_process_fixture.rs"]
mod fixture;

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use ferrule_common::execution::ForwardPhase;
use ferrule_model::decoder::KvCommitBinding;
use ferrule_model::transformer::{HostRows, RowsDType, RowsShape, SegmentInput};
use ferrule_runtime::SessionId;
use ferrule_runtime::parallel::pipeline::{
    PipelineCommand, PipelineCommandKey, PipelineParallelExecutor, PipelineRank, PipelineReply,
    PipelineStageDescription, PipelineTransport,
};
use ferrule_runtime::parallel::process::decoder::{
    DecoderCommand, ProcessPipelineTransport, SharedProcessPipelineTransport,
};
use ferrule_runtime::parallel::process::*;
use fixture::{Fixture, config, identity, launch_mode, options, topology, tx};
use serde_json::{Value, json};

fn key() -> PipelineCommandKey {
    PipelineCommandKey::new(
        KvCommitBinding::new(
            tx(7),
            topology(2).topology_id(),
            topology(2).participants(),
            1,
        )
        .unwrap(),
        PipelineRank {
            local: ferrule_common::ParallelRankId::new(1),
            global: ferrule_common::ParallelRankId::new(1),
        },
        SessionId(19),
    )
    .unwrap()
}

fn hidden(rows: usize, width: usize, dtype: RowsDType, values: Vec<f32>) -> PipelineCommand {
    PipelineCommand::Execute {
        key: key(),
        input: SegmentInput::Hidden {
            next_layer: 1,
            rows: HostRows::new(RowsShape::new(rows, width).unwrap(), dtype, None, values).unwrap(),
        },
        cancellation: Arc::new(AtomicBool::new(false)),
    }
}

fn description() -> PipelineStageDescription {
    // Metadata only: adapter tests do not need a child or a checkpoint.
    PipelineStageDescription {
        plan: ferrule_model::transformer::LayerSegmentPlan::new(2, 1..2, false, true).unwrap(),
        physical_pages: config().max_pages,
        config: config(),
        hidden: 4,
        vocabulary: 8,
        kv_heads: 1,
        head_dim: 2,
        expert_group: None,
    }
}

fn assert_values(command: PipelineCommand, expected: &[f32], dtype: RowsDType) {
    let PipelineCommand::Execute {
        input: SegmentInput::Hidden { next_layer, rows },
        ..
    } = command
    else {
        panic!("hidden Execute expected")
    };
    assert_eq!(next_layer, 1);
    assert_eq!(rows.dtype(), dtype);
    assert_eq!(rows.shape().width(), 4);
    assert_eq!(rows.shape().rows() * 4, expected.len());
    assert_eq!(
        rows.values()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn activation_adapter_preserves_shape_dtype_and_exact_finite_values() {
    for dtype in [RowsDType::F32, RowsDType::Bf16] {
        for rows in [1, 2, 16] {
            let expected: Vec<f32> = (0..rows * 4).map(|i| (i as f32 - 19.0) / 8.0).collect();
            let command = hidden(rows, 4, dtype, expected.clone());
            let encoded = DecoderCommand::encode(&command).unwrap();
            let bytes = serde_json::to_vec(&encoded).unwrap();
            let wire: DecoderCommand = serde_json::from_slice(&bytes).unwrap();
            assert_values(wire.decode(&description()).unwrap(), &expected, dtype);
        }
    }
}

#[test]
fn activation_adapter_rejects_geometry_value_counts_and_nonfinite() {
    let wire = DecoderCommand::encode(&hidden(1, 4, RowsDType::F32, vec![1.0; 4])).unwrap();
    let valid = serde_json::to_value(&wire).unwrap();
    for (field, value) in [
        ("rows", json!(0)),
        ("rows", json!(17)),
        ("rows", json!(usize::MAX)),
        ("width", json!(0)),
        ("width", json!(5)),
        ("width", json!(usize::MAX)),
        ("values", json!([])),
        ("values", json!([1.0, 2.0, 3.0])),
        ("values", json!([1.0, 2.0, 3.0, 4.0, 5.0])),
    ] {
        let mut bad = valid.clone();
        bad["input"]["rows"][field] = value;
        // Serde alone is not validation: the domain adapter must reject it.
        let wire: DecoderCommand = serde_json::from_value(bad).unwrap();
        assert!(
            wire.decode(&description()).is_err(),
            "accepted malformed {field}"
        );
    }
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        // Exercise both domain->wire and wire->domain finite checks without
        // JSON's nonfinite->null conversion obscuring the failing boundary.
        let command = hidden(1, 4, RowsDType::F32, vec![bad, 1.0, 2.0, 3.0]);
        assert!(DecoderCommand::encode(&command).is_err());
        let mut wire = DecoderCommand::encode(&hidden(1, 4, RowsDType::F32, vec![1.0; 4])).unwrap();
        let DecoderCommand::Execute {
            input: ferrule_runtime::parallel::process::decoder_wire::InputFrame::Hidden { rows, .. },
            ..
        } = &mut wire
        else {
            unreachable!()
        };
        rows.values[0] = bad;
        assert!(wire.decode(&description()).is_err());
    }
    for value in [json!(null), json!("NaN")] {
        let mut bad = valid.clone();
        bad["input"]["rows"]["values"][0] = value;
        assert!(serde_json::from_value::<DecoderCommand>(bad).is_err());
    }
}

const WARMUP: usize = 4;
const ITERATIONS: usize = 32;

fn summary(samples: &[u128]) -> Value {
    assert_eq!(samples.len(), ITERATIONS);
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    json!({"count":samples.len(), "total_ns":samples.iter().sum::<u128>(),
        "min_ns":sorted[0], "median_ns":sorted[sorted.len()/2],
        "p95_ns":sorted[(sorted.len()*95).div_ceil(100)-1]})
}

fn artifact(name: &str, body: Value) {
    let path = std::path::PathBuf::from(
        std::env::var_os("FERRULE_PR26_ARTIFACT_DIR")
            .expect("use scripts/bench_process_ipc.py to select an ignored artifact directory"),
    );
    assert!(path.is_dir());
    let document = json!({"schema":"ferrule.pr26.cpu-baseline.v1", "name":name,
        "pid":std::process::id(), "warmup":WARMUP, "iterations":ITERATIONS,
        "instrumentation_compiled":matches!(option_env!("FERRULE_PROCESS_IPC_INSTRUMENT"), Some("1")),
        "debug_assertions":cfg!(debug_assertions), "cuda_feature":cfg!(feature="cuda"),
        "download":{"status":"not_applicable_cpu", "wall_ns":null}, "result":body});
    std::fs::write(
        path.join(format!("{name}.json")),
        serde_json::to_vec_pretty(&document).unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "opt-in real child pipe echo baseline; scripts/bench_process_ipc.py, 100s bound"]
fn cpu_echo_baseline() {
    type Owner = ProcessRankOwner<Value, Value, Value>;
    let mut owner = Owner::spawn(
        launch_mode("normal"),
        identity(0),
        &json!({"bias":10}),
        options(),
    )
    .unwrap();
    let pid = owner.pid().unwrap();
    assert_ne!(pid, std::process::id());
    let mut reports = Vec::new();
    for elements in [0, 32, 4096, 32768] {
        let blob: Vec<f32> = (0..elements)
            .map(|i| (i as i32 % 257 - 128) as f32 / 32.0)
            .collect();
        let input = json!({"value":32, "blob":blob});
        let mut wall = Vec::new();
        for i in 0..WARMUP + ITERATIONS {
            let started = Instant::now();
            let reply = owner.execute(tx(i as u64 + 1), 19, &input).unwrap();
            let elapsed = started.elapsed().as_nanos();
            assert_eq!(reply["pid"], pid);
            assert_eq!(reply["value"], 42);
            assert_eq!(reply["blob"], input["blob"]);
            if i >= WARMUP {
                wall.push(elapsed);
            }
        }
        reports.push(json!({"elements":elements, "host_f32_bytes":elements * 4,
            "input_json_bytes":serde_json::to_vec(&input).unwrap().len(), "owner_roundtrip":summary(&wall)}));
    }
    assert_eq!(owner.shutdown().unwrap().reap, ReapOutcome::Reaped);
    assert!(owner.exit_status().unwrap().success());
    artifact(
        "echo",
        json!({"child_pid":pid, "cases":reports,
        "scope":"actual ProcessRankOwner + fixture echo, not decoder throughput"}),
    );
}

struct MeasuredTransport {
    inner: SharedProcessPipelineTransport,
    executed: Rc<RefCell<Vec<(u32, u128)>>>,
}
impl PipelineTransport for MeasuredTransport {
    fn call_observed(
        &mut self,
        rank: PipelineRank,
        command: PipelineCommand,
        poll: &mut dyn FnMut(),
    ) -> ferrule_common::Result<PipelineReply> {
        let execute = matches!(command, PipelineCommand::Execute { .. });
        let start = Instant::now();
        let result = self.inner.call_observed(rank, command, poll);
        if execute {
            self.executed
                .borrow_mut()
                .push((rank.global.get(), start.elapsed().as_nanos()));
        }
        result
    }
    fn outstanding(&self) -> usize {
        self.inner.outstanding()
    }
    fn shutdown(&mut self) -> ferrule_common::Result<()> {
        self.inner.shutdown()
    }
    fn quarantine(&mut self) {
        self.inner.quarantine();
    }
}

#[test]
#[ignore = "opt-in actual CPU decoder PP2 checkpoint execution; scripts/bench_process_ipc.py"]
fn cpu_activation_baseline() {
    let fixture = Fixture::new();
    let boots = fixture.boots(2, false);
    let plans = boots
        .iter()
        .map(|(_, boot)| boot.segment.decode().unwrap())
        .collect();
    let transport = ProcessPipelineTransport::spawn(launch_mode("decoder"), boots, options())
        .unwrap()
        .shared();
    let pids: Vec<_> = (0..2)
        .map(|rank| transport.process_stats(rank).unwrap().pid)
        .collect();
    assert_ne!(pids[0], pids[1]);
    assert!(!pids.contains(&std::process::id()));
    let executed = Rc::new(RefCell::new(Vec::new()));
    let mut pipeline = PipelineParallelExecutor::new_with_transport(
        topology(2),
        plans,
        config(),
        MeasuredTransport {
            inner: transport.clone(),
            executed: executed.clone(),
        },
    )
    .unwrap();
    let (mut oracle, _) = fixture.pipeline(1, false);
    let mut reports = Vec::new();
    for rows in [1, 8] {
        let tokens = vec![1; rows];
        let expected = oracle
            .forward(SessionId(19), &tokens, ForwardPhase::Prefill)
            .unwrap()
            .logits;
        oracle.release_session(SessionId(19)).unwrap();
        assert!(expected.values().iter().any(|v| v.abs() > 1e-4));
        let mut wall = Vec::new();
        let mut owner_wait = [Vec::new(), Vec::new()];
        for i in 0..WARMUP + ITERATIONS {
            executed.borrow_mut().clear();
            let start = Instant::now();
            let result = pipeline
                .forward(SessionId(19), &tokens, ForwardPhase::Prefill)
                .unwrap();
            let elapsed = start.elapsed().as_nanos();
            assert_eq!(
                (result.logits.rows(), result.logits.width()),
                (expected.rows(), expected.width())
            );
            for (a, b) in result.logits.values().iter().zip(expected.values()) {
                assert!(a.is_finite() && (a - b).abs() < 1e-5);
            }
            assert_eq!(executed.borrow().len(), 2);
            if i >= WARMUP {
                wall.push(elapsed);
                for &(rank, ns) in executed.borrow().iter() {
                    owner_wait[rank as usize].push(ns);
                }
            }
            pipeline.release_session(SessionId(19)).unwrap();
            assert_eq!(pipeline.outstanding(), 0);
        }
        reports.push(json!({"rows":rows, "hidden":4, "activation_f32_bytes":rows*4*4,
            "forward_wall":summary(&wall), "execute_call_inclusive_wall_by_rank":[summary(&owner_wait[0]), summary(&owner_wait[1])]}));
    }
    for rank in 0..2 {
        let stats = transport.process_stats(rank).unwrap();
        assert_eq!(
            (
                stats.sessions,
                stats.active_transactions,
                stats.resident_pages
            ),
            (0, 0, 0)
        );
        assert_eq!(stats.free_pages, stats.physical_pages);
        assert_eq!(stats.executions, 2 * (WARMUP + ITERATIONS));
    }
    pipeline.shutdown().unwrap();
    oracle.shutdown().unwrap();
    artifact(
        "activation",
        json!({"child_pids":pids, "cases":reports,
        "scope":"two-layer synthetic CPU checkpoint; actual PP2 hidden activation pipes; PP1 numerical oracle; not a production-model baseline"}),
    );
}

#[test]
#[ignore = "opt-in isolated activation copy/adapter/JSON microbaseline, not transport"]
fn cpu_activation_codec_baseline() {
    let mut reports = Vec::new();
    for rows in [1, 8, 16] {
        let values: Vec<f32> = (0..rows * 4).map(|i| i as f32 / 8.0 - 3.0).collect();
        let command = hidden(rows, 4, RowsDType::F32, values.clone());
        let mut copy = Vec::new();
        let mut adapter = Vec::new();
        let mut encode = Vec::new();
        let mut decode = Vec::new();
        for i in 0..WARMUP + ITERATIONS {
            let start = Instant::now();
            let copied = std::hint::black_box(values.as_slice()).to_vec();
            let copy_ns = start.elapsed().as_nanos();
            std::hint::black_box(&copied);
            let start = Instant::now();
            let frame = DecoderCommand::encode(std::hint::black_box(&command)).unwrap();
            let adapter_ns = start.elapsed().as_nanos();
            let start = Instant::now();
            let bytes = serde_json::to_vec(&frame).unwrap();
            let encode_ns = start.elapsed().as_nanos();
            let start = Instant::now();
            let wire: DecoderCommand = serde_json::from_slice(&bytes).unwrap();
            let domain = wire.decode(&description()).unwrap();
            let decode_ns = start.elapsed().as_nanos();
            assert_values(domain, &values, RowsDType::F32);
            if i >= WARMUP {
                copy.push(copy_ns);
                adapter.push(adapter_ns);
                encode.push(encode_ns);
                decode.push(decode_ns);
            }
        }
        reports.push(json!({"rows":rows,"width":4,"f32_bytes":values.len()*4,
            "vec_copy":summary(&copy),"adapter_encode_including_finite_check_and_copy":summary(&adapter),
            "json_encode":summary(&encode),"json_decode_and_domain_validation":summary(&decode)}));
    }
    artifact(
        "codec",
        json!({"cases":reports,"scope":"separate microbaseline; never add these timings to measured transport wall"}),
    );
}
