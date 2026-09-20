#![cfg(feature = "cuda")]

use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use ferrule_backend::cuda::operators::norm::CudaOperators;
use ferrule_backend::cuda::providers::CudaContext;
use ferrule_common::{ParallelTopologyId, ParallelismPlan, ValidatedParallelTopology};
use ferrule_runtime::{
    Decision, DistributedTransaction, DistributedTransactionError, ExecutionTransactionId,
    FinalizeOutcome, ParallelRankId, TransactionState,
};

const WORLD_SIZE: usize = 8;
const WIDTH: usize = 4096;
const EPSILON: f32 = 1e-5;
const TIMEOUT: Duration = Duration::from_secs(180);
const INJECTED_PANIC: &str = "injected CUDA owner panic after synchronized success report";

type TestResult = Result<(), Box<dyn std::error::Error>>;
type WorkerResult = Result<(), String>;
type JoinedWorker = (ParallelRankId, thread::Result<WorkerResult>);

// Only host-owned values cross the owner thread boundary.
struct HostReport {
    rank: ParallelRankId,
    values: Result<Vec<f32>, String>,
    synchronized: Result<(), String>,
}

struct Owners {
    handles: Vec<(ParallelRankId, JoinHandle<WorkerResult>)>,
    starts: Vec<Sender<()>>,
    exit_gate: Option<Sender<bool>>,
}

impl Owners {
    fn join_all(&mut self, inject_failure: bool) -> Vec<JoinedWorker> {
        // Disconnect startup waiters on every error path before joining anyone.
        self.starts.clear();
        if let Some(gate) = self.exit_gate.take() {
            let _ = gate.send(inject_failure);
        }
        self.handles
            .drain(..)
            .map(|(rank, handle)| (rank, handle.join()))
            .collect()
    }
}

impl Drop for Owners {
    fn drop(&mut self) {
        // Assertions in the coordinator must not detach already-spawned owners.
        let _ = self.join_all(false);
    }
}

fn panic_message(payload: &(dyn Any + Send)) -> &str {
    payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .unwrap_or("non-string panic payload")
}

fn bf16_round(value: f32) -> f32 {
    let bits = value.to_bits();
    let bias = 0x7fff + ((bits >> 16) & 1);
    f32::from_bits(bits.wrapping_add(bias) & 0xffff_0000)
}

fn rms_case(ordinal: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let input: Vec<_> = (0..WIDTH)
        .map(|index| {
            ((index * (ordinal + 3) + ordinal * 17) % 251) as f32 / 64.0 - 1.75
                + ordinal as f32 / 32.0
        })
        .collect();
    let weight: Vec<_> = (0..WIDTH)
        .map(|index| 0.5 + ((index + ordinal * 13) % 67) as f32 / 128.0)
        .collect();
    // Match cuda_device_owners: affine RMS reduces F32, then rounds output to BF16.
    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / WIDTH as f32;
    let inverse_rms = (mean_square + EPSILON).sqrt().recip();
    let expected = input
        .iter()
        .zip(&weight)
        .map(|(&value, &weight)| bf16_round(value * inverse_rms * weight))
        .collect();
    (input, weight, expected)
}

fn bf16_ulp_distance(actual: f32, expected: f32) -> u32 {
    // Order BF16 encodings numerically, collapsing signed zero. Adjacent finite
    // values are one step apart, including at exponent boundaries and near zero.
    let ordered = |value: f32| {
        let bits = (value.to_bits() >> 16) as i32;
        if bits & 0x8000 != 0 {
            0x8000 - (bits & 0x7fff)
        } else {
            0x8000 + bits
        }
    };
    ordered(actual).abs_diff(ordered(expected))
}

fn verify_rms(rank: ParallelRankId, actual: &[f32]) -> Result<(f32, u32), String> {
    let (_, _, expected) = rms_case(rank.get() as usize);
    if actual.len() != expected.len() {
        return Err(format!("rank={rank:?} incorrect RMS output length"));
    }
    let mut max_error = 0.0f32;
    let mut max_ulps = 0;
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        let error = (actual - expected).abs();
        // F32 parallel reduction/rsqrt may cross one BF16 rounding midpoint,
        // but must never move the result more than one representable BF16 step.
        let ulps = bf16_ulp_distance(actual, expected);
        if !actual.is_finite()
            || !expected.is_finite()
            || actual.to_bits() & 0xffff != 0
            || expected.to_bits() & 0xffff != 0
            || ulps > 1
        {
            return Err(format!(
                "rank={rank:?} RMS[{index}] actual={actual} expected={expected} error={error} bf16_ulps={ulps} max_allowed_ulps=1"
            ));
        }
        max_error = max_error.max(error);
        max_ulps = max_ulps.max(ulps);
    }
    Ok((max_error, max_ulps))
}

fn run_owner(
    rank: ParallelRankId,
    ready: Sender<(ParallelRankId, WorkerResult)>,
    reports: Sender<HostReport>,
    start: Receiver<()>,
    exit_gate: Option<Receiver<bool>>,
) -> WorkerResult {
    let ordinal = rank.get() as usize;
    let operators = match CudaOperators::new_on_device(ordinal) {
        Ok(operators) => operators,
        Err(error) => {
            let error = format!("owner initialization: {error}");
            let _ = ready.send((rank, Err(error.clone())));
            return Err(error);
        }
    };

    // Keep the owner outside the unwind boundary so computation errors AND
    // panics still reach synchronization before the owner can be dropped.
    let values = catch_unwind(AssertUnwindSafe(|| {
        ready
            .send((rank, Ok(())))
            .map_err(|error| format!("ready delivery: {error}"))?;
        start
            .recv_timeout(TIMEOUT)
            .map_err(|error| format!("startup gate: {error}"))?;
        if operators.device_ordinal() != ordinal {
            return Err(format!("wrong device ordinal for rank={rank:?}"));
        }
        let (input, weight, _) = rms_case(ordinal);
        operators
            .rms_norm(&input, &weight, EPSILON)
            .map_err(|error| format!("RMS execution: {error}"))
    }))
    .unwrap_or_else(|payload| Err(format!("computation panic: {}", panic_message(&*payload))));

    // Evaluate both calls even when the first fails. Failed synchronization is
    // reported explicitly, never converted into a completion/quiescence claim.
    let compute_sync = operators
        .sync_stream()
        .map_err(|error| format!("compute stream sync: {error}"));
    let upload_sync = operators
        .sync_upload_stream()
        .map_err(|error| format!("upload stream sync: {error}"));
    let synchronized = compute_sync.and(upload_sync);
    drop(operators);
    let result = values
        .as_ref()
        .map(|_| ())
        .map_err(Clone::clone)
        .and(synchronized.clone());
    reports
        .send(HostReport {
            rank,
            values,
            synchronized,
        })
        .map_err(|error| format!("host report delivery: {error}"))?;

    if let Some(gate) = exit_gate {
        let inject_failure = gate
            .recv_timeout(TIMEOUT)
            .map_err(|error| format!("exit gate: {error}"))?;
        result?;
        if inject_failure {
            // This panic is intentionally outside the computation catch boundary,
            // after sync/drop and a successful host report, but before join.
            panic!("{INJECTED_PANIC}");
        }
    } else {
        result?;
    }
    Ok(())
}

fn run_transaction(inject_failure: bool) -> TestResult {
    let visible = CudaContext::device_count().expect("enumerate visible CUDA devices");
    assert!(
        visible >= WORLD_SIZE,
        "requires at least {WORLD_SIZE} visible CUDA GPUs, found {visible}; check CUDA_VISIBLE_DEVICES"
    );
    let topology = ValidatedParallelTopology::new(
        ParallelTopologyId::new(1),
        WORLD_SIZE as u32,
        ParallelRankId::new(0),
        ParallelismPlan {
            data_parallel: WORLD_SIZE,
            ..ParallelismPlan::default()
        },
    )
    .unwrap();
    let ranks: Vec<_> = topology.participants().iter().collect();
    let last_rank = *ranks.last().unwrap();
    let id = ExecutionTransactionId::new(1).unwrap();
    let mut transaction = DistributedTransaction::new(topology, WORLD_SIZE);
    transaction.begin(id).unwrap();
    for &rank in &ranks {
        transaction.prepare(id, rank).unwrap();
    }

    let (ready_tx, ready_rx) = mpsc::channel();
    let (report_tx, report_rx) = mpsc::channel();
    let (exit_tx, exit_rx) = mpsc::channel();
    let mut exit_rx = Some(exit_rx);
    let mut owners = Owners {
        handles: Vec::with_capacity(WORLD_SIZE),
        starts: Vec::with_capacity(WORLD_SIZE),
        exit_gate: Some(exit_tx),
    };
    let mut errors = Vec::new();
    for &rank in &ranks {
        let (start_tx, start_rx) = mpsc::channel();
        let ready = ready_tx.clone();
        let reports = report_tx.clone();
        let gate = if rank == last_rank {
            exit_rx.take()
        } else {
            None
        };
        match thread::Builder::new()
            .name(format!("runtime-cuda-owner-{}", rank.get()))
            .spawn(move || run_owner(rank, ready, reports, start_rx, gate))
        {
            Ok(handle) => {
                owners.handles.push((rank, handle));
                owners.starts.push(start_tx);
            }
            Err(error) => errors.push(format!("rank={rank:?} spawn: {error}")),
        }
    }
    drop(ready_tx);
    drop(report_tx);

    let mut operations = BTreeMap::new();
    let mut reports = BTreeMap::new();
    let observation = (|| -> WorkerResult {
        if !errors.is_empty() {
            return Err("not all eight owners could be spawned".into());
        }
        let mut ready_ranks = BTreeSet::new();
        for _ in &ranks {
            let (rank, initialized) = ready_rx
                .recv_timeout(TIMEOUT)
                .map_err(|error| format!("owner readiness: {error}"))?;
            initialized.map_err(|error| format!("rank={rank:?}: {error}"))?;
            if !ranks.contains(&rank) || !ready_ranks.insert(rank) {
                return Err(format!("invalid/duplicate ready rank={rank:?}"));
            }
        }

        // Admit work only after every thread owns its device. No CUDA operation
        // exists for a spawn/initialization failure, and startup is cancellable.
        for &rank in &ranks {
            let operation = transaction
                .communicate(id, rank)
                .map_err(|error| format!("rank={rank:?} admission: {error:?}"))?;
            operations.insert(rank, operation);
        }
        assert_eq!(transaction.in_use_credits(), WORLD_SIZE);
        assert_eq!(
            transaction.communicate(id, ranks[0]),
            Err(DistributedTransactionError::Backpressure)
        );
        assert_eq!(transaction.drain(), 0);
        assert_eq!(
            transaction.commit_decision(id),
            Err(DistributedTransactionError::InvalidState)
        );
        for start in &owners.starts {
            start
                .send(())
                .map_err(|error| format!("start delivery: {error}"))?;
        }
        for _ in &ranks {
            let report = report_rx
                .recv_timeout(TIMEOUT)
                .map_err(|error| format!("host result delivery: {error}"))?;
            let rank = report.rank;
            if !ranks.contains(&rank) || reports.insert(rank, report).is_some() {
                return Err(format!("invalid/duplicate host report rank={rank:?}"));
            }
        }

        // The last owner cannot exit until this coordinator opens its gate.
        // Receiving all eight results must not finalize even a single rank.
        let (_, last_owner) = owners
            .handles
            .iter()
            .find(|(rank, _)| *rank == last_rank)
            .unwrap();
        assert!(!last_owner.is_finished(), "last owner bypassed exit gate");
        assert_eq!(reports.len(), WORLD_SIZE);
        assert_eq!(transaction.pending_ranks(id).unwrap(), ranks);
        assert_eq!(transaction.state(id), Ok(TransactionState::Preparing));
        assert_eq!(transaction.publication_count(), 0);
        assert_eq!(
            transaction.publish(id),
            Err(DistributedTransactionError::InvalidState)
        );
        Ok(())
    })();

    // No verified finalization is possible before ALL joins, including the
    // deliberately late panic. Drop also joins if the observation assertions fail.
    let joined = owners.join_all(inject_failure && observation.is_ok());
    if let Err(error) = observation {
        errors.push(error);
    }
    // Timeout/error paths may still receive synchronized reports during join.
    for report in report_rx.try_iter() {
        let rank = report.rank;
        if !ranks.contains(&rank) || reports.insert(rank, report).is_some() {
            errors.push(format!("invalid/duplicate late report rank={rank:?}"));
        }
    }

    let mut joined_successfully = BTreeSet::new();
    let mut injected_panics = 0;
    for (rank, result) in joined {
        match result {
            Ok(Ok(())) => {
                joined_successfully.insert(rank);
            }
            Ok(Err(error)) => errors.push(format!("rank={rank:?} owner: {error}")),
            Err(payload)
                if inject_failure
                    && rank == last_rank
                    && panic_message(&*payload) == INJECTED_PANIC =>
            {
                injected_panics += 1;
            }
            Err(payload) => errors.push(format!(
                "rank={rank:?} unexpected join panic: {}",
                panic_message(&*payload)
            )),
        }
    }
    if injected_panics != usize::from(inject_failure) {
        errors.push(format!(
            "unexpected injected panic count: {injected_panics}"
        ));
    }

    let mut verified_outcomes = Vec::new();
    let mut drained = 0;
    for &rank in &ranks {
        let mut verified_rms = false;
        if let Some(report) = reports.get(&rank) {
            match &report.values {
                Ok(values) => match verify_rms(rank, values) {
                    Ok((max_error, max_ulps)) => {
                        verified_rms = true;
                        println!(
                            "rank={} ordinal={} RMS passed max_abs_error={max_error:.8} max_bf16_ulps={max_ulps}",
                            rank.get(),
                            rank.get()
                        );
                    }
                    Err(error) => errors.push(error),
                },
                Err(error) => errors.push(format!("rank={rank:?} computation: {error}")),
            }
            if let Err(error) = &report.synchronized {
                errors.push(format!("rank={rank:?} NOT quiescent: {error}"));
            }
        } else {
            errors.push(format!("rank={rank:?} missing synchronized host report"));
        }
        let synchronized = reports
            .get(&rank)
            .is_some_and(|report| report.synchronized.is_ok());
        let outcome = if verified_rms && synchronized && joined_successfully.contains(&rank) {
            FinalizeOutcome::Success
        } else {
            FinalizeOutcome::Failure
        };
        verified_outcomes.push(outcome);
        if let Some(&operation) = operations.get(&rank) {
            // Join alone is not proof of GPU quiescence. On an actual sync
            // failure retain this operation's credit and fail, rather than forge
            // a completion simply to make the accounting look clean.
            if synchronized {
                if let Err(error) = transaction.complete(operation, outcome) {
                    errors.push(format!("rank={rank:?} complete: {error:?}"));
                }
                drained += transaction.drain();
            }
        }
        if synchronized {
            if let Err(error) = transaction.prepare_vote(id, rank, outcome) {
                errors.push(format!("rank={rank:?} ready vote: {error:?}"));
            }
        }
    }
    if !transaction.pending_completions().is_empty() || transaction.in_use_credits() != 0 {
        errors.push(format!(
            "unproven/undrained GPU work: pending={:?}, credits={}",
            transaction.pending_completions(),
            transaction.in_use_credits()
        ));
    }
    let all_quiescent = ranks.iter().all(|rank| {
        reports
            .get(rank)
            .is_some_and(|report| report.synchronized.is_ok())
    });
    if !errors.is_empty() {
        if all_quiescent && transaction.pending_completions().is_empty() {
            transaction.abort_decision(id).unwrap();
            for &rank in &ranks {
                transaction
                    .finalize(id, rank, FinalizeOutcome::Success)
                    .unwrap();
            }
            transaction.cancel(id).unwrap();
            transaction.retire(id).unwrap();
        } else {
            // A join or error report is not proof of device quiescence.
            assert_eq!(transaction.state(id), Ok(TransactionState::Preparing));
            assert_eq!(transaction.pending_ranks(id).unwrap(), ranks);
        }
        assert_eq!(transaction.publication_count(), 0);
        return Err(errors.join("; ").into());
    }

    assert_eq!(operations.len(), WORLD_SIZE);
    assert_eq!(reports.len(), WORLD_SIZE);
    assert_eq!(drained, WORLD_SIZE);
    assert_eq!(transaction.drain(), 0);
    assert!(transaction.pending_completions().is_empty());
    assert_eq!(transaction.in_use_credits(), 0);
    assert_eq!(transaction.publication_count(), 0);
    assert_eq!(transaction.pending_ranks(id).unwrap(), ranks);
    let mut expected = vec![FinalizeOutcome::Success; WORLD_SIZE];
    if inject_failure {
        expected[WORLD_SIZE - 1] = FinalizeOutcome::Failure;
    }
    assert_eq!(verified_outcomes, expected);

    assert_eq!(
        transaction.commit_decision(id),
        Ok(if inject_failure {
            Decision::Abort
        } else {
            Decision::Commit
        })
    );
    assert_eq!(
        transaction.publish(id),
        Err(DistributedTransactionError::InvalidState)
    );
    // All streams are fenced and owners joined. Business failure does not imply
    // cleanup failure for this workload without mutable KV.
    for &rank in &ranks {
        transaction
            .finalize(id, rank, FinalizeOutcome::Success)
            .unwrap();
    }
    if inject_failure {
        transaction.cancel(id).unwrap();
        assert_eq!(transaction.state(id), Ok(TransactionState::Cancelled));
        assert_eq!(
            transaction.publish(id),
            Err(DistributedTransactionError::InvalidState)
        );
        assert_eq!(transaction.publication_count(), 0);
    } else {
        transaction.publish(id).unwrap();
        assert_eq!(transaction.state(id), Ok(TransactionState::Published));
        assert_eq!(transaction.publication_count(), 1);
        assert_eq!(
            transaction.publish(id),
            Err(DistributedTransactionError::InvalidState)
        );
        assert_eq!(transaction.publication_count(), 1);
    }
    assert_eq!(transaction.in_use_credits(), 0);
    transaction.retire(id).unwrap();
    assert_eq!(transaction.retained_transaction_count(), 0);
    Ok(())
}

#[test]
#[ignore = "requires at least eight real CUDA GPUs"]
fn runtime_cuda_eight_gpu_transaction_commits_after_all_workers_join() -> TestResult {
    run_transaction(false)
}

#[test]
#[ignore = "requires at least eight real CUDA GPUs"]
fn runtime_cuda_single_rank_failure_aborts_without_publication() -> TestResult {
    run_transaction(true)
}
