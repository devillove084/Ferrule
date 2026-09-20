//! Linear-workload execution interfaces, not checkpoint/model inference coverage.
//! Run ignored tests with an external process timeout and --test-threads=1.
//! Host staging only: no NCCL, device handles across threads, or replica barrier.
#![cfg(feature = "cuda")]

use std::fmt::Debug;
use std::mem::size_of;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;
use std::sync::mpsc::{self, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::{self, ThreadId};
use std::time::{Duration, Instant};

use ferrule_backend::cuda::operators::linear::CudaOperators;
use ferrule_backend::cuda::providers::CudaContext;
use ferrule_common::{
    ParallelGroupId, ParallelRankId, ParallelTopologyId, ParallelismPlan, ParticipantSet,
    ValidatedParallelTopology,
};
use ferrule_model::transformer::parallel::{
    CudaLinearShard, CudaShardError, TensorParallelLinearPartition as Partition,
    TensorParallelLinearPlan,
};
use ferrule_runtime::parallel::collective::HostCollectiveLimits;
use ferrule_runtime::parallel::data;
use ferrule_runtime::parallel::tensor::{
    TensorCommand, TensorParallelExecutor, TensorRank, TensorWork,
};
use ferrule_runtime::{
    Decision, DistributedTransaction, ExecutionTransactionId, FinalizeOutcome, SessionId,
    TransactionState,
};

const DEVICES: usize = 8;
const ROWS: usize = 3;
const INPUT: usize = 19; // Ragged for every tested row-parallel degree: 2, 4, 8.
const OUTPUT: usize = 17; // Ragged for every tested column-parallel degree: 2, 4, 8.
const EPSILON: f32 = 1e-5;
const WAIT: Duration = Duration::from_secs(180);
// F32 dot products have only 19 terms; allow small FMA/reduction-order differences.
const LINEAR_ATOL: f32 = 2e-6;
const LINEAR_RTOL: f32 = 2e-6;

type Checked<T = ()> = Result<T, String>;
// Host payloads only: actual ordinal, owner identity, call count, linear, RMS.
type Reply = (usize, ThreadId, usize, Vec<f32>, Vec<f32>);
type Shutdown = (usize, ThreadId, usize);
type Pool = data::DataParallelExecutor<Vec<f32>, Reply, String>;

fn text(error: impl Debug) -> String {
    format!("{error:?}")
}

fn require(condition: bool, message: impl Into<String>) -> Checked {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}

fn rank(index: usize) -> ParallelRankId {
    ParallelRankId::new(index as u32)
}

fn tx(value: u64) -> ExecutionTransactionId {
    ExecutionTransactionId::new(value).unwrap()
}

fn require_eight_devices() {
    let visible = CudaContext::device_count().expect("enumerate visible CUDA devices");
    assert!(
        visible >= DEVICES,
        "requires all eight visible CUDA GPUs, found {visible}; check CUDA_VISIBLE_DEVICES"
    );
}

fn topology(data_parallel: usize, tensor_parallel: usize) -> ValidatedParallelTopology {
    ValidatedParallelTopology::new(
        ParallelTopologyId::new(1),
        DEVICES as u32,
        rank(0),
        ParallelismPlan {
            data_parallel,
            tensor_parallel,
            ..ParallelismPlan::default()
        },
    )
    .unwrap()
}

fn weight() -> Vec<f32> {
    (0..OUTPUT * INPUT)
        .map(|i| {
            let sign = if (i / INPUT) % 3 == 1 { -1.0 } else { 1.0 };
            sign * (0.04 + ((i * 7 + i / INPUT) % 29) as f32 * 0.0031)
        })
        .collect()
}

fn input(salt: usize) -> Vec<f32> {
    // Positive, non-dyadic inputs and constant-sign weight rows avoid tiny,
    // cancellation-dominated references while retaining different rows/columns.
    (0..ROWS * INPUT)
        .map(|i| {
            0.25 + ((i * 11 + salt * 7) % 37) as f32 * 0.027
                + (i / INPUT) as f32 * 0.13
                + salt as f32 * 0.009
        })
        .collect()
}

fn norm_weight() -> Vec<f32> {
    (0..OUTPUT).map(|i| 0.75 + i as f32 * 0.013).collect()
}

fn bf16_round(value: f32) -> f32 {
    let bits = value.to_bits();
    f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000)
}

fn bf16_code(value: f32) -> i32 {
    let bits = (value.to_bits() >> 16) as i32;
    if bits & 0x8000 != 0 {
        0x8000 - (bits & 0x7fff)
    } else {
        0x8000 + bits
    }
}

fn oracle(input: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let weight = weight();
    let mut linear = vec![0.0f32; ROWS * OUTPUT];
    // Full, unsplit CPU linear; neither the TP plan nor collective is the oracle.
    for row in 0..ROWS {
        for output in 0..OUTPUT {
            for feature in 0..INPUT {
                linear[row * OUTPUT + output] +=
                    input[row * INPUT + feature] * weight[output * INPUT + feature];
            }
        }
    }
    let norm_weight = norm_weight();
    let rms = linear
        .chunks_exact(OUTPUT)
        .flat_map(|row| {
            let mean_square = row.iter().map(|value| value * value).sum::<f32>() / OUTPUT as f32;
            let inverse = (mean_square + EPSILON).sqrt().recip();
            row.iter()
                .zip(&norm_weight)
                .map(move |(&value, &weight)| bf16_round(value * inverse * weight))
        })
        .collect();
    (linear, rms)
}

fn verify(label: &str, linear: &[f32], rms: &[f32], input: &[f32]) -> Checked {
    let (expected_linear, expected_rms) = oracle(input);
    require(
        linear.len() == expected_linear.len(),
        format!("{label}: linear length"),
    )?;
    require(
        rms.len() == expected_rms.len(),
        format!("{label}: RMS length"),
    )?;
    let mut max_linear = 0.0f32;
    let mut max_rms = 0.0f32;
    let mut max_ulps = 0;
    for (i, (&actual, &expected)) in linear.iter().zip(&expected_linear).enumerate() {
        let error = (actual - expected).abs();
        require(
            actual.is_finite() && error <= LINEAR_ATOL + LINEAR_RTOL * expected.abs(),
            format!("{label}: linear[{i}] actual={actual} expected={expected} error={error}"),
        )?;
        max_linear = max_linear.max(error);
    }
    for (i, (&actual, &expected)) in rms.iter().zip(&expected_rms).enumerate() {
        let error = (actual - expected).abs();
        let ulps = bf16_code(actual).abs_diff(bf16_code(expected));
        require(
            actual.is_finite() && actual.to_bits() & 0xffff == 0 && ulps <= 1,
            format!(
                "{label}: RMS[{i}] actual={actual} expected={expected} error={error} BF16_ULP={ulps} (limit=1)"
            ),
        )?;
        max_rms = max_rms.max(error);
        max_ulps = max_ulps.max(ulps);
    }
    eprintln!(
        "{label}: max_linear_error={max_linear:.8e} max_rms_error={max_rms:.8e} max_bf16_ulp={max_ulps}"
    );
    Ok(())
}

fn caught<T>(work: impl FnOnce() -> Checked<T>) -> Checked<T> {
    catch_unwind(AssertUnwindSafe(work)).unwrap_or_else(|payload| {
        // A typed unfenced-device signal must reach the owner quarantine boundary,
        // never be converted into an ordinary (implicitly quiescent) worker Err.
        if payload.downcast_ref::<data::PanicQuiescence>() == Some(&data::PanicQuiescence::Unknown)
        {
            std::panic::resume_unwind(payload);
        }
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or("non-string panic");
        let error = format!("owner/coordinator panic: {message}");
        std::mem::forget(payload);
        Err(error)
    })
}

fn shard_error(error: ferrule_common::Error) -> String {
    if CudaShardError::from_error(&error).is_some_and(CudaShardError::needs_quarantine) {
        std::mem::forget(error);
        std::panic::panic_any(data::PanicQuiescence::Unknown);
    }
    text(error)
}

fn fence_attempt(fence: impl FnOnce() -> ferrule_common::Result<()>) -> bool {
    let result = catch_unwind(AssertUnwindSafe(fence));
    let complete = matches!(result, Ok(Ok(())));
    // Do not format/drop unknown native errors or arbitrary panic payloads.
    if !complete {
        std::mem::forget(result);
    }
    complete
}

fn sync_owner(ops: &CudaOperators) -> Checked {
    let compute = fence_attempt(|| ops.sync_stream());
    let upload = fence_attempt(|| ops.sync_upload_stream());
    require(compute && upload, "compute/upload fence failed")
}

fn sync_shard(ops: &CudaOperators, shard: &CudaLinearShard) -> Checked {
    let drained = fence_attempt(|| {
        shard.quiesce()?;
        if shard.is_quiescent() && !shard.needs_quarantine() {
            Ok(())
        } else {
            Err(ferrule_common::Error::Model {
                message: "shard requires quarantine".into(),
            })
        }
    });
    // A failed/panicking control drain must not skip either stream attempt.
    let streams = sync_owner(ops);
    require(
        drained && streams.is_ok(),
        "shard/compute/upload fence failed",
    )
}

fn on_owner<T>(
    ops: &Rc<CudaOperators>,
    shard: Option<&CudaLinearShard>,
    work: impl FnOnce() -> Checked<T>,
) -> Checked<T> {
    let result = catch_unwind(AssertUnwindSafe(work));
    let forced_unknown = result.as_ref().err().is_some_and(|payload| {
        payload.downcast_ref::<data::PanicQuiescence>() == Some(&data::PanicQuiescence::Unknown)
    });
    let synchronized = match shard {
        Some(shard) => sync_shard(ops, shard),
        None => sync_owner(ops),
    };
    if forced_unknown || synchronized.is_err() {
        // Factories have no worker hook yet; retain returned handles and owner.
        std::mem::forget(result);
        std::mem::forget(Rc::clone(ops));
        std::panic::panic_any(data::PanicQuiescence::Unknown);
    }
    match result {
        Ok(result) => result,
        Err(payload) => caught(|| std::panic::resume_unwind(payload)),
    }
}

fn gpu_rms(ops: &CudaOperators, values: &[f32]) -> Checked<Vec<f32>> {
    // Explicit H2D -> owner-local kernel -> fence -> D2H. A host collective
    // result alone is never evidence that the communication operation completed.
    let input = ops.upload_f32_buffer(values).map_err(text)?;
    let weight = ops.upload_norm_weight(&norm_weight()).map_err(text)?;
    let mut output = ops.zero_f32_buffer(values.len()).map_err(text)?;
    let launch = ops.rms_norm_rows_from_device_into(&input, ROWS, &weight, EPSILON, &mut output);
    let synchronized = sync_owner(ops); // Fence before these buffers drop, even on launch failure.
    if synchronized.is_err() {
        // Retain temporaries before unwinding; a later hook cannot undo this
        // already-unknown operation, even if its fences succeed.
        std::mem::forget((input, weight, output, launch));
        std::panic::panic_any(data::PanicQuiescence::Unknown);
    }
    match (launch, synchronized) {
        (Ok(()), Ok(())) => ops.download_f32_buffer(&output).map_err(text),
        (launch, synchronized) => Err(format!("RMS launch={launch:?}; sync={synchronized:?}")),
    }
}

// The workload implementation required by ReplicaWorker, not a test scheduler
// or duplicate transaction state machine. All CUDA state stays on its owner.
struct LinearReplica {
    ops: Rc<CudaOperators>,
    linear: CudaLinearShard,
    calls: usize,
    shutdown: SyncSender<Shutdown>,
}

impl data::ReplicaWorker<Vec<f32>> for LinearReplica {
    type Output = Reply;
    type Error = String;

    fn execute(&mut self, request: data::WorkRequest<Vec<f32>>) -> Checked<Reply> {
        let result = on_owner(&self.ops, Some(&self.linear), || {
            require(
                request.rank.get() as usize == self.ops.device_ordinal(),
                "DP owner mismatch",
            )?;
            require(
                !request.cancellation.is_requested(),
                "unexpected cancellation",
            )?;
            let linear = self
                .linear
                .execute(&request.input, ROWS)
                .map_err(shard_error)?;
            let rms = gpu_rms(&self.ops, &linear)?;
            Ok((linear, rms))
        });
        self.calls += 1;
        result.map(|(linear, rms)| {
            (
                self.ops.device_ordinal(),
                thread::current().id(),
                self.calls,
                linear,
                rms,
            )
        })
    }

    fn panic_quiescence(&mut self) -> data::PanicQuiescence {
        if sync_shard(&self.ops, &self.linear).is_ok() {
            data::PanicQuiescence::Quiescent
        } else {
            data::PanicQuiescence::Unknown
        }
    }

    fn shutdown(&mut self) -> Checked {
        let synchronized = sync_shard(&self.ops, &self.linear);
        if let Err(error) = &synchronized {
            eprintln!("CUDA shutdown quiescence unknown: {error}");
            std::panic::panic_any(data::PanicQuiescence::Unknown);
        }
        let delivered = self.shutdown.try_send((
            self.ops.device_ordinal(),
            thread::current().id(),
            self.calls,
        ));
        match (synchronized, delivered) {
            (Ok(()), Ok(())) => Ok(()),
            (sync, delivery) => Err(format!("shutdown sync={sync:?}; observation={delivery:?}")),
        }
    }
}

fn submit_dp(
    pool: &mut Pool,
    coordinator: &mut DistributedTransaction,
    session: SessionId,
    expected_rank: ParallelRankId,
    id: ExecutionTransactionId,
    input: Vec<f32>,
) -> Checked {
    let route = pool.route_session(session).map_err(text)?;
    require(route == expected_rank, "sticky route changed")?;
    let scope = ParticipantSet::new(coordinator.topology(), [route]).map_err(text)?;
    coordinator.begin_scoped(id, scope).map_err(text)?;
    coordinator.prepare(id, route).map_err(text)?;
    if let Err(error) = pool.try_submit(session, id, input) {
        coordinator
            .prepare_vote(id, route, FinalizeOutcome::Failure)
            .map_err(text)?;
        coordinator.commit_decision(id).map_err(text)?;
        coordinator
            .finalize(id, route, FinalizeOutcome::Success)
            .map_err(text)?;
        coordinator.cancel(id).map_err(text)?;
        coordinator.retire(id).map_err(text)?;
        return Err(format!("DP admission rejected: {error:?}"));
    }
    Ok(())
}

#[test]
#[ignore = "requires eight visible CUDA GPUs; use an external process timeout"]
fn cuda_gpu_data_parallel_eight_replicas() {
    require_eight_devices();
    let mut coordinator = DistributedTransaction::new(topology(DEVICES, 1), DEVICES);
    let host = thread::current().id();
    let (shutdown_tx, shutdown_rx) = mpsc::sync_channel(DEVICES);
    let mut pool = Pool::new(
        data::DataParallelConfig {
            replicas: DEVICES,
            max_outstanding_per_replica: 1,
            session_capacity: DEVICES,
        },
        move |rank| {
            require(thread::current().id() != host, "factory ran on coordinator")?;
            let ops = Rc::new(CudaOperators::new_on_device(rank.get() as usize).map_err(text)?);
            let linear = on_owner(&ops, None, || {
                require(
                    ops.device_ordinal() == rank.get() as usize,
                    "factory ordinal mismatch",
                )?;
                let plan = TensorParallelLinearPlan::new(OUTPUT, INPUT, 1, Partition::Column)
                    .map_err(text)?;
                CudaLinearShard::new(Rc::clone(&ops), plan, ParallelRankId::new(0), &weight())
                    .map_err(shard_error)
            })?;
            Ok(LinearReplica {
                ops,
                linear,
                calls: 0,
                shutdown: shutdown_tx.clone(),
            })
        },
    )
    .expect("initialize all eight resident CUDA owners");

    let mut owners = [None; DEVICES];
    let observation = caught(|| {
        let mut ids = [tx(1); DEVICES];
        let mut calls = [0; DEVICES];
        for (ordinal, id) in ids.iter_mut().enumerate() {
            *id = tx(ordinal as u64 + 1);
            submit_dp(
                &mut pool,
                &mut coordinator,
                SessionId(ordinal as u64 + 1),
                rank(ordinal),
                *id,
                input(ordinal),
            )?;
        }
        // Allocate globally at admission, not by rank or completion order.
        let mut next_id = DEVICES as u64 + 1;
        let deadline = Instant::now() + WAIT;
        for completed in 0..2 * DEVICES {
            let completion = loop {
                if let Some(completion) = pool.poll() {
                    break completion;
                }
                require(Instant::now() < deadline, "DP completion deadline")?;
                thread::yield_now();
            };
            let ordinal = completion.rank.get() as usize;
            require(ordinal < DEVICES, "invalid completion rank")?;
            require(
                completion.transaction == ids[ordinal],
                "wrong completion transaction",
            )?;
            require(
                completion.session == SessionId(ordinal as u64 + 1),
                "wrong session",
            )?;
            if matches!(
                &completion.outcome,
                data::CompletionOutcome::QuiescenceUnknown
            ) {
                return Err(format!(
                    "DP rank={ordinal}: quiescence unknown; retain transaction"
                ));
            }
            let cancelled = completion.cancellation_requested
                || matches!(&completion.outcome, data::CompletionOutcome::Cancelled);
            let verified = (|| -> Checked<usize> {
                require(!cancelled, "unexpected cancellation")?;
                let (actual, owner, invocation, linear, rms) = match completion.outcome {
                    data::CompletionOutcome::Success(reply) => reply,
                    other => return Err(format!("DP rank={ordinal}: {other:?}")),
                };
                require(
                    actual == ordinal && owner != host,
                    "incorrect actual ordinal/owner",
                )?;
                require(
                    invocation == calls[ordinal] + 1 && invocation <= 2,
                    "worker was recreated or replayed",
                )?;
                if let Some(first) = owners[ordinal] {
                    require(first == owner, "session moved to another owner thread")?;
                } else {
                    owners[ordinal] = Some(owner);
                }
                verify(
                    &format!("DP rank={ordinal} actual_ordinal={actual} invocation={invocation}"),
                    &linear,
                    &rms,
                    &input(ordinal + calls[ordinal] * DEVICES),
                )?;
                calls[ordinal] += 1;
                Ok(invocation)
            })();
            let outcome = if verified.is_ok() {
                FinalizeOutcome::Success
            } else {
                FinalizeOutcome::Failure
            };
            // Vote only after verified compute completion; no cross-replica barrier.
            coordinator
                .prepare_vote(completion.transaction, completion.rank, outcome)
                .map_err(text)?;
            require(
                coordinator
                    .commit_decision(completion.transaction)
                    .map_err(text)?
                    == if verified.is_ok() {
                        Decision::Commit
                    } else {
                        Decision::Abort
                    },
                "DP decision",
            )?;
            // This synchronized linear owner has no mutable KV cleanup remaining.
            coordinator
                .finalize(
                    completion.transaction,
                    completion.rank,
                    FinalizeOutcome::Success,
                )
                .map_err(text)?;
            if cancelled {
                coordinator.cancel(completion.transaction).map_err(text)?;
            } else {
                coordinator.publish(completion.transaction).map_err(text)?;
            }
            let terminal = coordinator.state(completion.transaction).map_err(text)?;
            coordinator.retire(completion.transaction).map_err(text)?;
            let invocation = verified?;
            require(terminal == TransactionState::Published, "DP publication")?;
            require(
                coordinator.publication_count() == completed + 1,
                "DP publication count",
            )?;
            require(
                pool.session_rank(completion.session) == Some(completion.rank),
                "lost sticky route",
            )?;
            require(
                coordinator.retained_transaction_count() == pool.outstanding(),
                "DP retirement retained a completed transaction or removed a peer",
            )?;
            require(
                coordinator.retained_operation_count() == 0,
                "DP retained operations",
            )?;
            if invocation == 1 {
                ids[ordinal] = tx(next_id);
                next_id += 1;
                // A second request can finish before some other rank's first.
                submit_dp(
                    &mut pool,
                    &mut coordinator,
                    completion.session,
                    completion.rank,
                    ids[ordinal],
                    input(ordinal + DEVICES),
                )?;
            }
        }
        require(
            calls == [2; DEVICES],
            "not every resident executed both requests",
        )?;
        require(pool.outstanding() == 0, "DP requests remain outstanding")?;
        require(coordinator.in_use_credits() == 0, "DP leaked credits")?;
        require(
            coordinator.retained_transaction_count() == 0,
            "DP retained transactions",
        )?;
        require(
            coordinator.retained_operation_count() == 0,
            "DP retained operations",
        )
    });
    // Always explicitly join all owners, even if validation/polling failed.
    let shutdown = pool.shutdown();
    assert!(
        shutdown.is_ok(),
        "DP shutdown={shutdown:?}; observation={observation:?}"
    );
    assert!(observation.is_ok(), "{observation:?}");
    let mut seen = [false; DEVICES];
    for _ in 0..DEVICES {
        let (ordinal, owner, calls) = shutdown_rx
            .recv_timeout(WAIT)
            .expect("shutdown observation");
        assert!(ordinal < DEVICES && !seen[ordinal]);
        seen[ordinal] = true;
        assert_eq!(owners[ordinal], Some(owner));
        assert_eq!(calls, 2);
        eprintln!("DP rank={ordinal} shutdown synchronized and joined, calls={calls}");
    }
    assert!(seen.into_iter().all(|seen| seen));
    assert!(pool.poll().is_none());
    assert_eq!(coordinator.publication_count(), 2 * DEVICES);
    eprintln!(
        "DP complete: replicas=8 requests=16 calls_per_owner=2 retained_txns=0 retained_ops=0 all_owners_joined=true"
    );
}

// Host-only observations for assertions, never used to schedule or release work.
#[derive(Default)]
struct TensorObservations {
    owners: Mutex<Vec<(TensorRank, ThreadId)>>,
    applied: Mutex<Vec<(TensorRank, Vec<f32>)>>,
    shutdown: Mutex<Vec<(TensorRank, ThreadId)>>,
}

struct TensorReplica {
    ops: Rc<CudaOperators>,
    linear: CudaLinearShard,
    rank: TensorRank,
    owner: ThreadId,
    observations: Arc<TensorObservations>,
}

impl data::ReplicaWorker<TensorWork> for TensorReplica {
    type Output = Vec<f32>;
    type Error = String;

    fn execute(&mut self, request: data::WorkRequest<TensorWork>) -> Checked<Vec<f32>> {
        on_owner(&self.ops, Some(&self.linear), || {
            require(thread::current().id() == self.owner, "TP owner changed")?;
            require(
                request.rank == self.rank.local,
                "TP pool-local rank mismatch",
            )?;
            require(
                request.input.rank == self.rank,
                "TP local/global rank mismatch",
            )?;
            require(
                self.ops.device_ordinal() == self.rank.global.get() as usize,
                "TP device ordinal mismatch",
            )?;
            require(
                !request.cancellation.is_requested(),
                "unexpected TP cancellation",
            )?;
            match request.input.command {
                TensorCommand::Compute { input, rows } => {
                    require(rows == ROWS, "TP Compute rows")?;
                    self.linear.execute(&input, rows).map_err(shard_error)
                }
                TensorCommand::Apply { values, rows } => {
                    require(
                        rows == ROWS && values.len() == rows * OUTPUT,
                        "TP Apply shape",
                    )?;
                    let rms = gpu_rms(&self.ops, &values)?;
                    self.observations
                        .applied
                        .lock()
                        .unwrap()
                        .push((self.rank, values));
                    // gpu_rms fences H2D/kernel/D2H; on_owner also fences both
                    // streams on success or failure before the completion returns.
                    Ok(rms)
                }
            }
        })
    }

    fn panic_quiescence(&mut self) -> data::PanicQuiescence {
        if sync_shard(&self.ops, &self.linear).is_ok() {
            data::PanicQuiescence::Quiescent
        } else {
            data::PanicQuiescence::Unknown
        }
    }

    fn shutdown(&mut self) -> Checked {
        let synchronized = sync_shard(&self.ops, &self.linear);
        if let Err(error) = &synchronized {
            eprintln!("CUDA shutdown quiescence unknown: {error}");
            std::panic::panic_any(data::PanicQuiescence::Unknown);
        }
        self.observations
            .shutdown
            .lock()
            .unwrap()
            .push((self.rank, thread::current().id()));
        synchronized
    }
}

#[test]
#[ignore = "runs the full TP=2,4,8 matrix; requires eight visible CUDA GPUs and an external timeout"]
fn cuda_gpu_tensor_parallel_linear_rank_matrix() {
    require_eight_devices();
    let mut next_id = 1;
    for degree in [2, 4, 8] {
        for partition in [Partition::Column, Partition::Row] {
            let id = tx(next_id);
            next_id += 1;
            let plan = TensorParallelLinearPlan::new(OUTPUT, INPUT, degree, partition).unwrap();
            let split_dimension = match partition {
                Partition::Column => OUTPUT,
                Partition::Row => INPUT,
            };
            assert_ne!(
                split_dimension % degree,
                0,
                "this must exercise ragged splits"
            );
            assert_ne!(
                plan.rank_range(rank(0)).unwrap().len(),
                plan.rank_range(rank(degree - 1)).unwrap().len()
            );
            let count = match partition {
                Partition::Column => plan.column_gather_count(ROWS).unwrap(),
                Partition::Row => ROWS * OUTPUT,
            };
            let output_count = if partition == Partition::Column {
                count * degree
            } else {
                count
            };

            // Nonzero replicas exercise local/global rank separation for TP=2/4.
            // This test explicitly maps global ranks to visible device ordinals.
            let replica: u32 = if degree == DEVICES { 0 } else { 1 };
            let observations = Arc::new(TensorObservations::default());
            let host = thread::current().id();
            let factory_plan = plan.clone();
            let factory_observations = Arc::clone(&observations);
            let mut executor = TensorParallelExecutor::new(
                topology(DEVICES / degree, degree),
                replica,
                plan,
                data::DataParallelConfig {
                    replicas: degree,
                    max_outstanding_per_replica: 1,
                    session_capacity: degree,
                },
                move |tensor_rank: TensorRank| {
                    let owner = thread::current().id();
                    require(owner != host, "TP factory ran on coordinator")?;
                    let ops = Rc::new(
                        CudaOperators::new_on_device(tensor_rank.global.get() as usize)
                            .map_err(text)?,
                    );
                    let linear = on_owner(&ops, None, || {
                        require(
                            ops.device_ordinal() == tensor_rank.global.get() as usize,
                            "TP factory ordinal mismatch",
                        )?;
                        CudaLinearShard::new(
                            Rc::clone(&ops),
                            factory_plan.clone(),
                            tensor_rank.local,
                            &weight(),
                        )
                        .map_err(shard_error)
                    })?;
                    factory_observations
                        .owners
                        .lock()
                        .unwrap()
                        .push((tensor_rank, owner));
                    Ok(TensorReplica {
                        ops,
                        linear,
                        rank: tensor_rank,
                        owner,
                        observations: Arc::clone(&factory_observations),
                    })
                },
                ParallelGroupId::new(id.get() as u32),
                HostCollectiveLimits {
                    max_ranks: degree,
                    max_elements_per_rank: count,
                    max_host_bytes: degree * (count + output_count) * size_of::<f32>(),
                },
            )
            .expect("initialize resident TP CUDA owners");

            let observation = caught(|| {
                let full_input = input(degree);
                let output = executor
                    .execute(
                        id,
                        SessionId(1000 + id.get() * 10),
                        full_input.clone().into(),
                        ROWS,
                    )
                    .map_err(text)?;
                require(output.transaction == id, "TP output transaction")?;
                require(output.ranks.len() == degree, "TP output rank count")?;
                let applied = observations.applied.lock().unwrap();
                require(applied.len() == degree, "TP Apply observation count")?;
                for (local, (tensor_rank, rms)) in output.ranks.iter().enumerate() {
                    require(
                        tensor_rank.local == rank(local)
                            && tensor_rank.global == rank(replica as usize * degree + local),
                        "TP returned local/global rank mismatch",
                    )?;
                    let (_, linear) = applied
                        .iter()
                        .find(|(rank, _)| rank == tensor_rank)
                        .ok_or_else(|| format!("missing TP Apply rank={tensor_rank:?}"))?;
                    verify(
                        &format!(
                            "TP={degree} {partition:?} local={} global={} actual_ordinal={}",
                            tensor_rank.local.get(),
                            tensor_rank.global.get(),
                            tensor_rank.global.get(),
                        ),
                        linear,
                        rms,
                        &full_input,
                    )?;
                }
                require(
                    executor.coordinator().publication_count() == 1,
                    "TP publication count",
                )
            });
            // Explicitly join every owner even if execution or oracle validation failed.
            let shutdown = executor.shutdown();
            assert!(
                shutdown.is_ok(),
                "TP shutdown={shutdown:?}; observation={observation:?}"
            );
            assert_eq!(executor.outstanding(), 0);
            assert_eq!(executor.coordinator().in_use_credits(), 0);
            assert_eq!(executor.coordinator().retained_transaction_count(), 0);
            assert_eq!(executor.coordinator().retained_operation_count(), 0);
            assert!(executor.coordinator().pending_completions().is_empty());
            assert!(executor.collective().active_descriptor().is_none());
            assert_eq!(executor.collective().owned_host_bytes(), 0);
            assert_eq!(executor.collective().reserved_host_bytes(), 0);
            assert!(observation.is_ok(), "{observation:?}");

            let owners = observations.owners.lock().unwrap();
            let stopped = observations.shutdown.lock().unwrap();
            assert_eq!(owners.len(), degree);
            assert_eq!(stopped.len(), degree);
            assert_eq!(
                owners
                    .iter()
                    .map(|(_, owner)| *owner)
                    .collect::<std::collections::HashSet<_>>()
                    .len(),
                degree,
            );
            for local in 0..degree {
                let expected = TensorRank {
                    local: rank(local),
                    global: rank(replica as usize * degree + local),
                };
                let owner = owners.iter().find(|(rank, _)| *rank == expected).unwrap();
                assert_ne!(owner.1, host);
                assert!(
                    stopped.contains(owner),
                    "shutdown moved rank={expected:?} to another owner"
                );
            }
            eprintln!(
                "TP={degree} {partition:?} complete: rows={ROWS} in={INPUT} out={OUTPUT} collective_count={count} retained_txns=0 retained_ops=0 all_owners_joined=true"
            );
        }
    }
}
