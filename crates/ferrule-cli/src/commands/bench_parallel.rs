//! CUDA-only F32 linear benchmark of the production parallel executors.
//! No CPU oracle, custom worker scheduler, or alternative collective protocol.

use crate::args::BenchParallelArgs;

pub(crate) fn cmd_bench_parallel(args: BenchParallelArgs) -> anyhow::Result<()> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = args;
        anyhow::bail!(
            "bench-parallel is CUDA-only; rebuild ferrule-cli with --features cuda (no CPU fallback)"
        );
    }
    #[cfg(feature = "cuda")]
    benchmark::gpu::run(args)
}

#[cfg(any(feature = "cuda", test))]
mod benchmark {
    #[cfg(test)]
    use std::mem::size_of;
    use std::path::PathBuf;

    use anyhow::{Context, Result, ensure};
    use ferrule_common::ParallelRankId;
    use ferrule_model::transformer::parallel::{
        TensorParallelLinearPartition as Partition, TensorParallelLinearPlan,
    };
    use ferrule_runtime::parallel::collective::HostCollectiveLimits;
    #[cfg(test)]
    use ferrule_runtime::parallel::data::PanicQuiescence;
    use ferrule_runtime::parallel::data::{CompletionOutcome, HostCompletion};
    use ferrule_runtime::{Decision, DistributedTransaction, FinalizeOutcome};
    use serde::{Deserialize, Serialize};

    use crate::args::{BenchParallelArgs, BenchParallelMode as Mode, RankBackend};
    #[cfg(test)]
    use crate::commands::parallel_worker::{RetainUntilQuiescent, owner_fenced, sync_both};
    use crate::commands::parallel_worker::{bytes, elements, runtime_error};

    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Fixture {
        rows: usize,
        in_features: usize,
        out_features: usize,
        weight: Vec<f32>,
        input: Vec<f32>,
    }

    #[derive(Debug, Clone, Copy, Serialize)]
    struct Shape {
        rows: usize,
        in_features: usize,
        out_features: usize,
    }

    struct Validated {
        shape: Shape,
        plan: TensorParallelLinearPlan,
        devices: Vec<usize>,
        collective_limits: HostCollectiveLimits,
        output_len: usize,
        #[cfg(unix)]
        process: Option<crate::commands::rank_worker::process::Metadata>,
    }

    fn validate(args: &BenchParallelArgs, fixture: &Fixture) -> Result<Validated> {
        crate::commands::rank_worker::validate_options(
            args.rank_backend,
            args.rank_timeout_ms,
            args.rank_restarts,
        )?;
        ensure!(
            args.ranks > 0 && u32::try_from(args.ranks).is_ok(),
            "ranks must be positive and fit u32"
        );
        ensure!(args.iterations > 0, "iterations must be greater than zero");
        bytes::<f64>(args.iterations)?;
        let rounds = args
            .warmup
            .checked_add(args.iterations)
            .context("iteration count overflow")?;
        // TP uses two private worker IDs per rank per round, starting at one.
        u64::try_from(rounds)?
            .checked_mul(u64::try_from(args.ranks)?)
            .and_then(|n| n.checked_mul(2))
            .and_then(|n| n.checked_add(1))
            .context("transaction identity count overflow")?;
        for (name, supplied, actual) in [
            ("rows", args.rows, fixture.rows),
            ("in-features", args.in_features, fixture.in_features),
            ("out-features", args.out_features, fixture.out_features),
        ] {
            ensure!(
                supplied.is_none_or(|value| value == actual),
                "--{name} must match fixture ({actual})"
            );
        }
        let input_len = elements(fixture.rows, fixture.in_features)?;
        let weight_len = elements(fixture.out_features, fixture.in_features)?;
        let output_len = elements(fixture.rows, fixture.out_features)?;
        ensure!(
            fixture.input.len() == input_len,
            "fixture input length: expected {input_len}, got {}",
            fixture.input.len()
        );
        ensure!(
            fixture.weight.len() == weight_len,
            "fixture weight length: expected {weight_len}, got {}",
            fixture.weight.len()
        );
        ensure!(
            fixture
                .input
                .iter()
                .chain(&fixture.weight)
                .all(|value| value.is_finite()),
            "fixture input and weight must contain only finite F32 values"
        );
        let partition = if args.mode == Mode::TpRow {
            Partition::Row
        } else {
            Partition::Column
        };
        let degree = if args.mode == Mode::Dp { 1 } else { args.ranks };
        let plan = TensorParallelLinearPlan::new(
            fixture.out_features,
            fixture.in_features,
            degree,
            partition,
        )?;
        // Rank zero has the largest shard for ragged partitions. Check the
        // existing CUDA F32 kernel's u32 indexing before constructing any owner.
        let (local_out, local_in) = plan.local_shape(ParallelRankId::new(0))?;
        for count in [
            elements(local_out, local_in)?,
            elements(fixture.rows, local_in)?,
            elements(fixture.rows, local_out)?,
        ] {
            ensure!(
                u32::try_from(count).is_ok(),
                "CUDA F32 element indices exceed u32"
            );
        }
        let count = if args.mode == Mode::TpColumn {
            plan.column_gather_count(fixture.rows)?
        } else {
            output_len
        };
        let gathered = if args.mode == Mode::TpColumn {
            elements(count, args.ranks)?
        } else {
            output_len
        };
        let peak_elements = count
            .checked_add(gathered)
            .and_then(|n| n.checked_mul(args.ranks))
            .context("collective host capacity overflow")?;
        let collective_limits = HostCollectiveLimits {
            max_ranks: args.ranks,
            max_elements_per_rank: count,
            max_host_bytes: bytes::<f32>(peak_elements)?,
        };
        bytes::<usize>(args.ranks)?;
        let devices = args
            .devices
            .clone()
            .unwrap_or_else(|| (0..args.ranks).collect());
        ensure!(
            devices.len() == args.ranks,
            "--devices must contain exactly ranks={} ordinals",
            args.ranks
        );
        let mut unique = devices.clone();
        unique.sort_unstable();
        ensure!(
            !unique.windows(2).any(|pair| pair[0] == pair[1]),
            "--devices must contain distinct CUDA ordinals"
        );
        ensure!(
            devices.iter().all(|&device| i32::try_from(device).is_ok()),
            "CUDA device ordinal exceeds i32"
        );
        Ok(Validated {
            shape: Shape {
                rows: fixture.rows,
                in_features: fixture.in_features,
                out_features: fixture.out_features,
            },
            plan,
            devices,
            collective_limits,
            output_len,
            #[cfg(unix)]
            process: if args.rank_backend == RankBackend::Process {
                Some(crate::commands::rank_worker::process::metadata(
                    crate::commands::rank_worker::WorkerSpec {
                        mode: args.mode,
                        ranks: args.ranks,
                        rank: 0,
                        device: 0,
                        rows: fixture.rows,
                        in_features: fixture.in_features,
                        out_features: fixture.out_features,
                    },
                    args.rank_timeout_ms,
                )?)
            } else {
                None
            },
        })
    }

    fn publish_dp_completion(
        coordinator: &mut DistributedTransaction,
        completion: HostCompletion<Vec<f32>, anyhow::Error>,
    ) -> Result<Vec<f32>> {
        if matches!(&completion.outcome, CompletionOutcome::QuiescenceUnknown) {
            anyhow::bail!(
                "fatal DP rank={} transaction={}: CUDA quiescence unknown; no finalize/publication/retirement; stop and join all owners, retain custody until process exit",
                completion.rank.get(),
                completion.transaction.get()
            );
        }
        let cancelled = completion.cancellation_requested
            || matches!(&completion.outcome, CompletionOutcome::Cancelled);
        let result = match completion.outcome {
            CompletionOutcome::Success(values) if !completion.cancellation_requested => Ok(values),
            CompletionOutcome::Success(_) | CompletionOutcome::Cancelled => {
                Err(anyhow::anyhow!("DP work cancelled"))
            }
            CompletionOutcome::Failed(error) => Err(error),
            CompletionOutcome::Panicked => {
                Err(anyhow::anyhow!("DP worker panicked but proved quiescence"))
            }
            CompletionOutcome::ReplicaUnavailable => Err(anyhow::anyhow!("DP worker unavailable")),
            CompletionOutcome::QuiescenceUnknown => unreachable!("handled above"),
        };
        let outcome = if result.is_ok() {
            FinalizeOutcome::Success
        } else {
            FinalizeOutcome::Failure
        };
        coordinator
            .prepare_vote(completion.transaction, completion.rank, outcome)
            .map_err(runtime_error)?;
        let decision = coordinator
            .commit_decision(completion.transaction)
            .map_err(runtime_error)?;
        ensure!(
            decision
                == if result.is_ok() {
                    Decision::Commit
                } else {
                    Decision::Abort
                },
            "unexpected DP decision"
        );
        // The owner proved quiescence; this linear workload has no mutable KV.
        // A business Failure is not a failed cleanup ACK.
        coordinator
            .finalize(
                completion.transaction,
                completion.rank,
                FinalizeOutcome::Success,
            )
            .map_err(runtime_error)?;
        if cancelled {
            coordinator
                .cancel(completion.transaction)
                .map_err(runtime_error)?;
        } else {
            coordinator
                .publish(completion.transaction)
                .map_err(runtime_error)?;
        }
        coordinator
            .retire(completion.transaction)
            .map_err(runtime_error)?;
        result
    }

    /// Only for requests rejected before any worker accepted device work.
    fn cancel_unsubmitted(
        coordinator: &mut DistributedTransaction,
        id: ferrule_runtime::ExecutionTransactionId,
        rank: ParallelRankId,
    ) -> Result<()> {
        coordinator.abort_decision(id).map_err(runtime_error)?;
        coordinator
            .finalize(id, rank, FinalizeOutcome::Success)
            .map_err(runtime_error)?;
        coordinator.cancel(id).map_err(runtime_error)?;
        coordinator.retire(id).map_err(runtime_error)
    }

    fn finish<T>(result: Result<T>, shutdown: Result<()>) -> Result<T> {
        match (result, shutdown) {
            (Ok(value), Ok(())) => Ok(value),
            (Err(error), Ok(())) | (Ok(_), Err(error)) => Err(error),
            (Err(error), Err(shutdown)) => {
                Err(error.context(format!("owner shutdown also failed: {shutdown:#}")))
            }
        }
    }

    #[derive(Debug, Serialize)]
    struct RankOutput {
        rank: usize,
        values: Vec<f32>,
    }

    // Outside the timer. This checks actual GPU evidence, not a CPU substitute.
    fn verify_outputs(
        outputs: &[RankOutput],
        ranks: usize,
        output_len: usize,
        mode: Mode,
    ) -> Result<()> {
        ensure!(
            outputs.len() == ranks,
            "expected {ranks} rank outputs, got {}",
            outputs.len()
        );
        for (rank, output) in outputs.iter().enumerate() {
            ensure!(
                output.rank == rank,
                "missing, duplicated or unordered rank output {rank}"
            );
            ensure!(
                output.values.len() == output_len,
                "rank {rank}: expected {output_len} output elements, got {}",
                output.values.len()
            );
            ensure!(
                output.values.iter().all(|value| value.is_finite()),
                "rank {rank}: nonfinite GPU output"
            );
            if mode != Mode::Dp {
                ensure!(
                    output.values == outputs[0].values,
                    "TP rank {rank} full output differs from rank 0"
                );
            }
        }
        Ok(())
    }

    #[derive(Debug, Serialize)]
    struct Timing {
        scope: &'static str,
        samples_seconds: Vec<f64>,
        mean_seconds: f64,
        p50_seconds: f64,
        p95_seconds: f64,
        total_seconds: f64,
    }

    fn summarize(samples: Vec<f64>) -> Result<Timing> {
        ensure!(
            !samples.is_empty() && samples.iter().all(|s| s.is_finite() && *s > 0.0),
            "timing samples must be nonempty, finite and positive"
        );
        let total_seconds: f64 = samples.iter().sum();
        ensure!(total_seconds.is_finite(), "timing total overflow");
        let mut sorted = samples.clone();
        sorted.sort_by(f64::total_cmp);
        // Matches linear interpolation at (n - 1) * p, including n = 1.
        let percentile = |p: f64| {
            let index = (sorted.len() - 1) as f64 * p;
            let lower = index.floor() as usize;
            let upper = index.ceil() as usize;
            sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower as f64)
        };
        Ok(Timing {
            scope: "host_to_host",
            mean_seconds: total_seconds / samples.len() as f64,
            p50_seconds: percentile(0.5),
            p95_seconds: percentile(0.95),
            total_seconds,
            samples_seconds: samples,
        })
    }

    #[derive(Debug, Serialize)]
    struct Throughput {
        scope: &'static str,
        batches_per_round: usize,
        rows_per_batch: usize,
        rows_per_round: usize,
        total_batches: usize,
        total_rows: usize,
        batches_per_second: f64,
        rows_per_second: f64,
    }

    #[derive(Serialize)]
    struct FixtureMetadata {
        path: PathBuf,
        dtype: &'static str,
        weight_layout: &'static str,
        input_layout: &'static str,
        bias: bool,
    }

    #[derive(Serialize)]
    struct Report {
        schema_version: u32,
        implementation: &'static str,
        transport: &'static str,
        rank_backend: RankBackend,
        rank_timeout_ms: u64,
        rank_timeout_scope: &'static str,
        rank_restarts: usize,
        execution_epoch: u64,
        restarts_attempted: usize,
        #[cfg(unix)]
        #[serde(skip_serializing_if = "Option::is_none")]
        process: Option<crate::commands::rank_worker::process::Metadata>,
        mode: Mode,
        ranks: usize,
        devices: Vec<usize>,
        shape: Shape,
        fixture: FixtureMetadata,
        warmup: usize,
        iterations: usize,
        timing: Timing,
        timing_description: &'static str,
        percentile_method: &'static str,
        throughput: Throughput,
        #[serde(skip_serializing_if = "Option::is_none")]
        outputs: Option<Vec<RankOutput>>,
    }

    fn report(
        args: &BenchParallelArgs,
        validated: &Validated,
        samples: Vec<f64>,
        outputs: Vec<RankOutput>,
    ) -> Result<Report> {
        ensure!(
            samples.len() == args.iterations,
            "sample count must match iterations"
        );
        verify_outputs(&outputs, args.ranks, validated.output_len, args.mode)?;
        let timing = summarize(samples)?;
        let batches = if args.mode == Mode::Dp { args.ranks } else { 1 };
        let total_batches = batches
            .checked_mul(args.iterations)
            .context("throughput batch count overflow")?;
        let total_rows = total_batches
            .checked_mul(validated.shape.rows)
            .context("throughput row count overflow")?;
        let batches_per_second = total_batches as f64 / timing.total_seconds;
        let rows_per_second = total_rows as f64 / timing.total_seconds;
        ensure!(
            batches_per_second.is_finite() && rows_per_second.is_finite(),
            "throughput overflow"
        );
        Ok(Report {
            schema_version: 1,
            implementation: "ferrule",
            transport: "host-staged",
            rank_backend: args.rank_backend,
            rank_timeout_ms: args.rank_timeout_ms,
            rank_timeout_scope: if args.rank_backend == RankBackend::Process {
                "per_owner_startup_command_shutdown_ipc"
            } else {
                "not_enforced_for_threads"
            },
            rank_restarts: args.rank_restarts,
            execution_epoch: crate::commands::rank_worker::EXECUTION_EPOCH,
            restarts_attempted: 0,
            #[cfg(unix)]
            process: validated.process.clone(),
            mode: args.mode,
            ranks: args.ranks,
            devices: validated.devices.clone(),
            shape: validated.shape,
            fixture: FixtureMetadata {
                path: args.fixture.clone(),
                dtype: "float32",
                weight_layout: "row-major [out_features, in_features]",
                input_layout: "row-major [rows, in_features]",
                bias: false,
            },
            warmup: args.warmup,
            iterations: args.iterations,
            timing,
            timing_description: "Instant wall time from host submission through all required synchronized host results and transaction publication; includes orchestration/polling, input sharding, H2D/D2H copies, per-call allocation, TP host-staged collective/Apply roundtrip and (process backend) bounded JSON IPC encoding/decoding/validation. Excludes fixture loading, owner/weight initialization, warmup, output verification, shutdown and report serialization. Not kernel-only or CUDA-event timing.",
            percentile_method: "linear interpolation at (n - 1) * p",
            throughput: Throughput {
                scope: "host_to_host",
                batches_per_round: batches,
                rows_per_batch: validated.shape.rows,
                rows_per_round: batches
                    .checked_mul(validated.shape.rows)
                    .context("throughput row count overflow")?,
                total_batches,
                total_rows,
                batches_per_second,
                rows_per_second,
            },
            outputs: args.emit_values.then_some(outputs),
        })
    }

    #[cfg(feature = "cuda")]
    pub(super) mod gpu {
        use std::fs::File;
        use std::io::{BufReader, Write};

        use std::sync::Arc;
        use std::time::{Duration, Instant};

        use ferrule_common::{
            ParallelGroupId, ParallelTopologyId, ParallelismPlan, ParticipantSet,
            ValidatedParallelTopology,
        };
        use ferrule_runtime::parallel::data::{DataParallelConfig, DataParallelExecutor};
        use ferrule_runtime::parallel::tensor::{TensorParallelExecutor, TensorRank};
        use ferrule_runtime::{ExecutionTransactionId, SessionId};

        use super::*;
        use crate::commands::rank_worker::{RankWorker, WorkerSpec};

        fn topology(mode: Mode, ranks: usize) -> Result<ValidatedParallelTopology> {
            ValidatedParallelTopology::new(
                ParallelTopologyId::new(1),
                u32::try_from(ranks)?,
                ParallelRankId::new(0),
                ParallelismPlan {
                    data_parallel: if mode == Mode::Dp { ranks } else { 1 },
                    tensor_parallel: if mode == Mode::Dp { 1 } else { ranks },
                    ..ParallelismPlan::default()
                },
            )
            .map_err(runtime_error)
        }

        fn pool_config(ranks: usize) -> DataParallelConfig {
            DataParallelConfig {
                replicas: ranks,
                max_outstanding_per_replica: 1,
                session_capacity: ranks,
            }
        }

        fn measure(
            args: &BenchParallelArgs,
            validated: &Validated,
            mut iteration: impl FnMut() -> Result<Vec<RankOutput>>,
        ) -> Result<(Vec<f64>, Vec<RankOutput>)> {
            for _ in 0..args.warmup {
                let outputs = iteration()?;
                verify_outputs(&outputs, args.ranks, validated.output_len, args.mode)?;
            }
            let mut samples = Vec::with_capacity(args.iterations);
            let mut last = Vec::new();
            for _ in 0..args.iterations {
                let started = Instant::now();
                let outputs = iteration()?;
                let seconds = started.elapsed().as_secs_f64();
                verify_outputs(&outputs, args.ranks, validated.output_len, args.mode)?;
                samples.push(seconds);
                last = outputs;
            }
            Ok((samples, last))
        }

        type DpPool = DataParallelExecutor<Arc<[f32]>, Vec<f32>, anyhow::Error>;

        fn dp_iteration(
            pool: &mut DpPool,
            coordinator: &mut DistributedTransaction,
            next_id: &mut u64,
            ranks: usize,
            input: &Arc<[f32]>,
        ) -> Result<Vec<RankOutput>> {
            let first_id = *next_id;
            let mut outputs: Vec<_> = (0..ranks)
                .map(|rank| RankOutput {
                    rank,
                    values: Vec::new(),
                })
                .collect();
            for index in 0..ranks {
                let rank = ParallelRankId::new(index as u32);
                let session = SessionId(index as u64);
                let id = ExecutionTransactionId::new(*next_id)?;
                *next_id += 1; // validate() reserved the entire run's ID space.
                pool.route_session_to(session, rank)
                    .map_err(runtime_error)?;
                let scope =
                    ParticipantSet::new(coordinator.topology(), [rank]).map_err(runtime_error)?;
                coordinator.begin_scoped(id, scope).map_err(runtime_error)?;
                coordinator.prepare(id, rank).map_err(runtime_error)?;
                if let Err(error) = coordinator.reserve(id, 1) {
                    cancel_unsubmitted(coordinator, id, rank)?;
                    return Err(runtime_error(error));
                }
                if let Err(error) = pool.try_submit_to(session, rank, id, Arc::clone(input)) {
                    cancel_unsubmitted(coordinator, id, rank)?;
                    return Err(runtime_error(error.kind));
                }
            }
            let mut failure = None;
            while pool.outstanding() != 0 {
                let Some(completion) = pool.poll() else {
                    std::thread::sleep(Duration::from_micros(50));
                    continue;
                };
                let index = completion.rank.get() as usize;
                ensure!(
                    index < ranks
                        && completion.transaction.get() == first_id + index as u64
                        && completion.session == SessionId(index as u64),
                    "unexpected DP completion identity"
                );
                let unknown = matches!(&completion.outcome, CompletionOutcome::QuiescenceUnknown);
                let result = publish_dp_completion(coordinator, completion);
                if unknown {
                    // A consumed queue notification is not a GPU fence. Stop
                    // before any more publication; run_dp always joins all owners.
                    if let Err(error) = &result {
                        eprintln!("bench-parallel: {error:#}");
                    }
                    return result.map(|_| outputs);
                }
                match result {
                    Ok(values) => outputs[index].values = values,
                    Err(error) => {
                        if failure.is_none() {
                            failure = Some(error);
                        }
                    }
                }
            }
            match failure {
                Some(error) => Err(error),
                None => Ok(outputs),
            }
        }

        fn run_dp(
            args: &BenchParallelArgs,
            validated: &Validated,
            weight: Arc<[f32]>,
            input: Arc<[f32]>,
        ) -> Result<(Vec<f64>, Vec<RankOutput>)> {
            let devices = validated.devices.clone();
            let spec_base = WorkerSpec {
                mode: args.mode,
                ranks: args.ranks,
                rank: 0,
                device: 0,
                rows: validated.shape.rows,
                in_features: validated.shape.in_features,
                out_features: validated.shape.out_features,
            };
            let backend = args.rank_backend;
            let timeout_ms = args.rank_timeout_ms;
            let mut pool = DpPool::new(pool_config(args.ranks), move |rank: ParallelRankId| {
                let mut spec = spec_base;
                spec.rank = rank.get();
                spec.device = devices[rank.get() as usize];
                RankWorker::new(backend, timeout_ms, spec, &weight)
            })
            .map_err(runtime_error)?;
            let mut coordinator =
                DistributedTransaction::new(topology(args.mode, args.ranks)?, args.ranks);
            let mut next_id = 1;
            let result = measure(args, validated, || {
                dp_iteration(
                    &mut pool,
                    &mut coordinator,
                    &mut next_id,
                    args.ranks,
                    &input,
                )
            });
            let shutdown = pool.shutdown().map_err(runtime_error);
            if coordinator.retained_transaction_count() != 0 {
                eprintln!(
                    "bench-parallel: fatal DP exit: retaining {} unresolved transactions and {} credits after joining owners; device custody is not released, process exit required",
                    coordinator.retained_transaction_count(),
                    coordinator.in_use_credits()
                );
                std::mem::forget(coordinator);
            }
            finish(result, shutdown)
        }

        fn run_tp(
            args: &BenchParallelArgs,
            validated: &Validated,
            weight: Arc<[f32]>,
            input: Arc<[f32]>,
        ) -> Result<(Vec<f64>, Vec<RankOutput>)> {
            let devices = validated.devices.clone();
            let rows = validated.shape.rows;
            let spec_base = WorkerSpec {
                mode: args.mode,
                ranks: args.ranks,
                rank: 0,
                device: 0,
                rows: validated.shape.rows,
                in_features: validated.shape.in_features,
                out_features: validated.shape.out_features,
            };
            let backend = args.rank_backend;
            let timeout_ms = args.rank_timeout_ms;
            let mut executor = TensorParallelExecutor::<RankWorker>::new(
                topology(args.mode, args.ranks)?,
                0,
                validated.plan.clone(),
                pool_config(args.ranks),
                move |rank: TensorRank| {
                    let mut spec = spec_base;
                    spec.rank = rank.local.get();
                    spec.device = devices[rank.local.get() as usize];
                    RankWorker::new(backend, timeout_ms, spec, &weight)
                },
                ParallelGroupId::new(1),
                validated.collective_limits,
            )
            .map_err(runtime_error)?;
            let mut next_id = 1;
            let result = measure(args, validated, || {
                let id = ExecutionTransactionId::new(next_id)?;
                next_id += 1;
                let output = executor
                    .execute(id, SessionId(0), Arc::clone(&input), rows)
                    .map_err(runtime_error)?;
                ensure!(output.transaction == id, "unexpected TP transaction");
                Ok(output
                    .ranks
                    .into_iter()
                    .map(|(rank, values)| RankOutput {
                        rank: rank.local.get() as usize,
                        values,
                    })
                    .collect())
            });
            let quarantined = executor.is_quarantined();
            let shutdown = executor.shutdown().map_err(runtime_error);
            if quarantined {
                eprintln!(
                    "bench-parallel: fatal TP quiescence unknown; retaining executor custody after joining owners until process exit"
                );
                std::mem::forget(executor);
                return finish(
                    result.context("fatal TP quiescence unknown; process exit required"),
                    shutdown,
                );
            }
            finish(result, shutdown)
        }

        pub(crate) fn run(args: BenchParallelArgs) -> Result<()> {
            let file = File::open(&args.fixture)
                .with_context(|| format!("open fixture {}", args.fixture.display()))?;
            let fixture: Fixture =
                serde_json::from_reader(BufReader::new(file)).context("parse fixture JSON")?;
            let validated = validate(&args, &fixture)?;

            eprintln!(
                "bench-parallel: CUDA {:?}, rank_backend={:?}, ranks={}, devices={:?}; host-to-host timing, initialization and warmup excluded; no restart/replay",
                args.mode, args.rank_backend, args.ranks, validated.devices
            );
            let weight: Arc<[f32]> = fixture.weight.into();
            let input: Arc<[f32]> = fixture.input.into();
            let (samples, outputs) = if args.mode == Mode::Dp {
                run_dp(&args, &validated, weight, input)?
            } else {
                run_tp(&args, &validated, weight, input)?
            };
            let report = report(&args, &validated, samples, outputs)?;
            let mut stdout = std::io::stdout().lock();
            if args.json {
                serde_json::to_writer(&mut stdout, &report)?;
                writeln!(stdout)?;
            } else {
                writeln!(
                    stdout,
                    "CUDA {:?}: ranks={}, shape=[{}, {}, {}], host-staged",
                    report.mode,
                    report.ranks,
                    report.shape.rows,
                    report.shape.in_features,
                    report.shape.out_features
                )?;
                writeln!(
                    stdout,
                    "host_to_host: mean={:.6}s p50={:.6}s p95={:.6}s total={:.6}s",
                    report.timing.mean_seconds,
                    report.timing.p50_seconds,
                    report.timing.p95_seconds,
                    report.timing.total_seconds
                )?;
                writeln!(
                    stdout,
                    "{:.3} batches_per_second; {:.3} rows_per_second ({} batches/round)",
                    report.throughput.batches_per_second,
                    report.throughput.rows_per_second,
                    report.throughput.batches_per_round
                )?;
                writeln!(stdout, "{}", report.timing_description)?;
                if let Some(outputs) = &report.outputs {
                    serde_json::to_writer(&mut stdout, outputs)?;
                    writeln!(stdout)?;
                }
            }
            Ok(())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::args::{Cli, Command};
        use clap::Parser;

        fn args(mode: &str) -> BenchParallelArgs {
            let cli = Cli::try_parse_from([
                "ferrule",
                "bench-parallel",
                "--mode",
                mode,
                "--ranks",
                "2",
                "--fixture",
                "fixture.json",
                "--iterations",
                "2",
            ])
            .unwrap();
            let Command::BenchParallel(args) = cli.command else {
                panic!("wrong command")
            };
            args
        }

        fn fixture() -> Fixture {
            serde_json::from_str(r#"{"rows":2,"in_features":3,"out_features":3,"weight":[1,2,3,4,5,6,7,8,9],"input":[1,2,3,4,5,6]}"#).unwrap()
        }

        fn outputs() -> Vec<RankOutput> {
            (0..2)
                .map(|rank| RankOutput {
                    rank,
                    values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                })
                .collect()
        }

        #[test]
        fn fixture_shape_defaults_and_ragged_plan() {
            for mode in ["dp", "tp-column", "tp-row"] {
                let args = args(mode);
                let validated = validate(&args, &fixture()).unwrap();
                assert_eq!(validated.devices, [0, 1]);
                assert_eq!(validated.shape.rows, 2);
                assert_eq!(validated.output_len, 6);
                assert_eq!(validated.plan.ranks(), if mode == "dp" { 1 } else { 2 });
                assert_eq!(
                    validated.collective_limits.max_elements_per_rank,
                    if mode == "tp-column" { 4 } else { 6 }
                );
                assert_eq!(validated.collective_limits.max_host_bytes, 96);
            }
            let mut args = args("tp-row");
            args.rows = Some(2);
            args.in_features = Some(3);
            args.out_features = Some(3);
            args.devices = Some(vec![3, 1]);
            assert_eq!(validate(&args, &fixture()).unwrap().devices, [3, 1]);
        }

        #[test]
        fn rejects_invalid_shapes_lengths_and_values() {
            let base = args("dp");
            for dimension in 0..3 {
                let mut f = fixture();
                match dimension {
                    0 => f.rows = 0,
                    1 => f.in_features = 0,
                    _ => f.out_features = 0,
                }
                assert!(validate(&base, &f).is_err());
                let mut a = base.clone();
                match dimension {
                    0 => a.rows = Some(9),
                    1 => a.in_features = Some(9),
                    _ => a.out_features = Some(9),
                }
                assert!(validate(&a, &fixture()).is_err());
            }
            for weight in [false, true] {
                let mut f = fixture();
                if weight {
                    f.weight.pop();
                } else {
                    f.input.pop();
                }
                assert!(validate(&base, &f).is_err());
                for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                    let mut f = fixture();
                    if weight {
                        f.weight[0] = value;
                    } else {
                        f.input[0] = value;
                    }
                    assert!(validate(&base, &f).is_err());
                }
            }
            for text in [
                "{}",
                r#"{"rows":1,"in_features":1,"out_features":1,"weight":[NaN],"input":[1]}"#,
                r#"{"rows":1,"in_features":1,"out_features":1,"weight":["placeholder"],"input":[1]}"#,
            ] {
                assert!(serde_json::from_str::<Fixture>(text).is_err());
            }
            let mut f = fixture();
            f.rows = usize::MAX;
            assert!(validate(&base, &f).is_err());
            assert!(elements(usize::MAX, 2).is_err());
            assert!(elements(isize::MAX as usize / size_of::<f32>() + 1, 1).is_err());
        }

        #[test]
        fn rejects_invalid_ranks_iterations_and_devices() {
            for mode in ["dp", "tp-column", "tp-row"] {
                let mut a = args(mode);
                a.ranks = 0;
                assert!(validate(&a, &fixture()).is_err());
                a = args(mode);
                a.iterations = 0;
                assert!(validate(&a, &fixture()).is_err());
                a = args(mode);
                a.warmup = usize::MAX;
                assert!(validate(&a, &fixture()).is_err());
                for devices in [vec![], vec![0], vec![0, 0], vec![0, usize::MAX]] {
                    a = args(mode);
                    a.devices = Some(devices);
                    assert!(validate(&a, &fixture()).is_err());
                }
            }
            for mode in ["tp-column", "tp-row"] {
                let mut a = args(mode);
                a.ranks = 4;
                assert!(validate(&a, &fixture()).is_err());
            }
        }

        #[test]
        fn summary_uses_interpolated_percentiles_and_original_sample_order() {
            let timing = summarize(vec![4.0, 1.0, 3.0, 2.0]).unwrap();
            assert_eq!(timing.samples_seconds, [4.0, 1.0, 3.0, 2.0]);
            assert_eq!(timing.total_seconds, 10.0);
            assert_eq!(timing.mean_seconds, 2.5);
            assert_eq!(timing.p50_seconds, 2.5);
            assert!((timing.p95_seconds - 3.85).abs() < 1e-12);
            let one = summarize(vec![0.125]).unwrap();
            assert_eq!(one.p50_seconds, 0.125);
            assert_eq!(one.p95_seconds, 0.125);
        }

        #[test]
        fn report_has_the_versioned_json_shape_and_values_are_opt_in() {
            for mode in ["dp", "tp-column", "tp-row"] {
                let mut options = args(mode);
                let validated = validate(&options, &fixture()).unwrap();
                let measured = report(&options, &validated, vec![1.0, 3.0], outputs()).unwrap();
                let json = serde_json::to_value(&measured).unwrap();
                assert_eq!(json["schema_version"], 1);
                assert_eq!(json["implementation"], "ferrule");
                assert_eq!(json["transport"], "host-staged");
                assert_eq!(json["rank_backend"], "thread");
                assert_eq!(json["rank_timeout_ms"], 30_000);
                assert_eq!(json["rank_timeout_scope"], "not_enforced_for_threads");
                assert_eq!(json["rank_restarts"], 0);
                assert_eq!(json["execution_epoch"], 1);
                assert!(json.get("process").is_none());
                assert_eq!(json["mode"], mode);
                assert_eq!(json["ranks"], 2);
                assert_eq!(json["devices"], serde_json::json!([0, 1]));
                assert_eq!(
                    json["shape"],
                    serde_json::json!({"rows": 2, "in_features": 3, "out_features": 3})
                );
                assert_eq!(json["fixture"]["path"], "fixture.json");
                assert_eq!(json["fixture"]["dtype"], "float32");
                assert_eq!(json["fixture"]["bias"], false);
                assert!(json["fixture"].get("weight").is_none());
                assert!(json["fixture"].get("input").is_none());
                assert_eq!(json["warmup"], 10);
                assert_eq!(json["iterations"], 2);
                assert_eq!(json["timing"]["scope"], "host_to_host");
                assert_eq!(
                    json["timing"]["samples_seconds"],
                    serde_json::json!([1.0, 3.0])
                );
                assert_eq!(json["timing"]["total_seconds"], 4.0);
                assert_eq!(json["timing"]["mean_seconds"], 2.0);
                assert_eq!(json["timing"]["p50_seconds"], 2.0);
                assert_eq!(json["timing"]["p95_seconds"], 2.9);
                assert_eq!(json["throughput"]["scope"], "host_to_host");
                assert_eq!(
                    json["throughput"]["batches_per_round"],
                    if mode == "dp" { 2 } else { 1 }
                );
                assert_eq!(json["throughput"]["rows_per_batch"], 2);
                assert_eq!(
                    json["throughput"]["rows_per_round"],
                    if mode == "dp" { 4 } else { 2 }
                );
                assert_eq!(
                    json["throughput"]["total_batches"],
                    if mode == "dp" { 4 } else { 2 }
                );
                assert_eq!(
                    json["throughput"]["total_rows"],
                    if mode == "dp" { 8 } else { 4 }
                );
                assert_eq!(
                    json["throughput"]["batches_per_second"],
                    if mode == "dp" { 1.0 } else { 0.5 }
                );
                assert_eq!(
                    json["throughput"]["rows_per_second"],
                    if mode == "dp" { 2.0 } else { 1.0 }
                );
                assert!(json.get("outputs").is_none());
                assert!(
                    !serde_json::to_string(&measured)
                        .unwrap()
                        .contains("requests_per")
                );

                options.emit_values = true;
                let measured = report(&options, &validated, vec![1.0, 3.0], outputs()).unwrap();
                let json = serde_json::to_value(measured).unwrap();
                let values = json["outputs"].as_array().unwrap();
                assert_eq!(values.len(), 2);
                for (rank, output) in values.iter().enumerate() {
                    assert_eq!(output["rank"], rank);
                    assert_eq!(
                        output["values"],
                        serde_json::json!([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                    );
                }
            }
        }

        #[test]
        fn process_backend_report_contains_bounded_owner_metadata_without_cuda_probe() {
            let mut options = args("dp");
            options.rank_backend = RankBackend::Process;
            options.rank_timeout_ms = 250;
            let validated = validate(&options, &fixture()).unwrap();
            let measured = report(&options, &validated, vec![1.0, 3.0], outputs()).unwrap();
            let json = serde_json::to_value(measured).unwrap();
            assert_eq!(json["rank_backend"], "process");
            assert_eq!(json["rank_timeout_ms"], 250);
            assert_eq!(
                json["rank_timeout_scope"],
                "per_owner_startup_command_shutdown_ipc"
            );
            assert_eq!(json["rank_restarts"], 0);
            assert_eq!(json["restarts_attempted"], 0);
            assert_eq!(json["process"]["protocol_version"], 1);
            assert_eq!(json["process"]["owners"].as_array().unwrap().len(), 2);
            assert!(
                json["process"]["frame_limits"]["max_frame_bytes"]
                    .as_u64()
                    .unwrap()
                    >= 1024
            );
            assert_eq!(json["process"]["child_io_timeout_ms"], 750);
        }

        #[test]
        fn tp_report_preserves_each_ranks_actual_buffer_and_values() {
            for mode in ["tp-column", "tp-row"] {
                let mut options = args(mode);
                options.emit_values = true;
                let validated = validate(&options, &fixture()).unwrap();
                let mut actual = outputs();
                // Signed zeros compare equal but have different bits. This
                // catches replacing a rank's actual result with rank 0's copy.
                actual[0].values[0] = 0.0;
                actual[1].values[0] = -0.0;
                let buffers: Vec<_> = actual.iter().map(|output| output.values.as_ptr()).collect();
                let measured = report(&options, &validated, vec![1.0, 3.0], actual).unwrap();
                let preserved = measured.outputs.as_ref().unwrap();
                assert_eq!(preserved.len(), 2);
                for (rank, output) in preserved.iter().enumerate() {
                    assert_eq!(output.rank, rank);
                    assert_eq!(output.values.as_ptr(), buffers[rank]);
                }
                assert_eq!(preserved[0].values[0].to_bits(), 0.0f32.to_bits());
                assert_eq!(preserved[1].values[0].to_bits(), (-0.0f32).to_bits());
                let json = serde_json::to_value(measured).unwrap();
                assert!(
                    json["outputs"][1]["values"][0]
                        .as_f64()
                        .unwrap()
                        .is_sign_negative()
                );
            }
        }

        #[test]
        fn fences_attempt_both_streams_even_after_error_or_panic() {
            use std::cell::RefCell;
            for fail_compute in [false, true] {
                for fail_upload in [false, true] {
                    let calls = RefCell::new(Vec::new());
                    let result = sync_both(
                        || {
                            calls.borrow_mut().push("compute");
                            ensure!(!fail_compute, "compute failed");
                            Ok(())
                        },
                        || {
                            calls.borrow_mut().push("upload");
                            ensure!(!fail_upload, "upload failed");
                            Ok(())
                        },
                    );
                    assert_eq!(*calls.borrow(), ["compute", "upload"]);
                    assert_eq!(result.is_ok(), !fail_compute && !fail_upload);
                }
            }
            let calls = RefCell::new(Vec::new());
            assert!(
                sync_both(
                    || {
                        calls.borrow_mut().push("compute");
                        panic!("compute fence panic")
                    },
                    || {
                        calls.borrow_mut().push("upload");
                        Ok(())
                    },
                )
                .is_err()
            );
            assert_eq!(*calls.borrow(), ["compute", "upload"]);
        }

        #[test]
        fn ordinary_errors_return_only_after_both_fences_and_typed_unknown_is_not_swallowed() {
            use std::cell::Cell;
            use std::panic::{AssertUnwindSafe, catch_unwind};
            let fences = Cell::new(0);
            let error = owner_fenced::<()>(
                || anyhow::bail!("ordinary execute error"),
                || {
                    fences.set(fences.get() + 1);
                    Ok(())
                },
                || {
                    fences.set(fences.get() + 1);
                    Ok(())
                },
            )
            .unwrap_err();
            assert_eq!(fences.get(), 2);
            assert!(error.to_string().contains("ordinary execute error"));
            let error = owner_fenced::<()>(
                || panic!("ordinary work panic"),
                || {
                    fences.set(fences.get() + 1);
                    Ok(())
                },
                || {
                    fences.set(fences.get() + 1);
                    Ok(())
                },
            )
            .unwrap_err();
            assert_eq!(fences.get(), 4);
            assert!(error.to_string().contains("ordinary work panic"));

            let panic = catch_unwind(AssertUnwindSafe(|| {
                owner_fenced::<()>(
                    || std::panic::panic_any(PanicQuiescence::Unknown),
                    || {
                        fences.set(fences.get() + 1);
                        Ok(())
                    },
                    || {
                        fences.set(fences.get() + 1);
                        Ok(())
                    },
                )
            }))
            .unwrap_err();
            assert_eq!(
                panic.downcast_ref::<PanicQuiescence>(),
                Some(&PanicQuiescence::Unknown)
            );
            assert_eq!(
                fences.get(),
                6,
                "typed Unknown must be preserved after both fences are attempted"
            );

            for fail_compute in [false, true] {
                let panic = catch_unwind(AssertUnwindSafe(|| {
                    owner_fenced::<()>(
                        || anyhow::bail!("ordinary execute error"),
                        || {
                            ensure!(!fail_compute, "compute failed");
                            Ok(())
                        },
                        || {
                            ensure!(fail_compute, "upload failed");
                            Ok(())
                        },
                    )
                }))
                .unwrap_err();
                assert_eq!(
                    panic.downcast_ref::<PanicQuiescence>(),
                    Some(&PanicQuiescence::Unknown)
                );
            }
        }

        #[test]
        fn failed_fences_and_unwinding_retain_resources_until_process_exit() {
            use std::cell::Cell;
            use std::panic::{AssertUnwindSafe, catch_unwind};
            use std::rc::Rc;
            #[derive(Debug)]
            struct DropProbe(Rc<Cell<usize>>);
            impl Drop for DropProbe {
                fn drop(&mut self) {
                    self.0.set(self.0.get() + 1);
                }
            }
            let drops = Rc::new(Cell::new(0));
            let value =
                owner_fenced(|| Ok(DropProbe(Rc::clone(&drops))), || Ok(()), || Ok(())).unwrap();
            assert_eq!(drops.get(), 0);
            drop(value);
            assert_eq!(drops.get(), 1);
            let panic = catch_unwind(AssertUnwindSafe(|| {
                owner_fenced(
                    || Ok(DropProbe(Rc::clone(&drops))),
                    || anyhow::bail!("compute failed"),
                    || Ok(()),
                )
            }))
            .unwrap_err();
            assert_eq!(
                panic.downcast_ref::<PanicQuiescence>(),
                Some(&PanicQuiescence::Unknown)
            );
            assert_eq!(drops.get(), 1, "unfenced return value must not be dropped");
            let panic = catch_unwind(AssertUnwindSafe(|| {
                let _factory_resource = RetainUntilQuiescent::new(DropProbe(Rc::clone(&drops)));
                std::panic::panic_any(PanicQuiescence::Unknown);
            }));
            assert!(panic.is_err());
            assert_eq!(
                drops.get(),
                1,
                "factory resources must survive stack unwinding"
            );
        }

        #[test]
        fn unknown_dp_completion_preserves_transaction_and_credit_custody() {
            use ferrule_common::{
                ParallelTopologyId, ParallelismPlan, ParticipantSet, ValidatedParallelTopology,
            };
            use ferrule_runtime::{ExecutionTransactionId, SessionId, TransactionState};
            for cancellation_requested in [false, true] {
                let topology = ValidatedParallelTopology::new(
                    ParallelTopologyId::new(1),
                    2,
                    ParallelRankId::new(0),
                    ParallelismPlan {
                        data_parallel: 2,
                        ..ParallelismPlan::default()
                    },
                )
                .unwrap();
                let mut coordinator = DistributedTransaction::new(topology, 2);
                for index in 0..2 {
                    let id = ExecutionTransactionId::new(index + 1).unwrap();
                    let rank = ParallelRankId::new(index as u32);
                    let scope = ParticipantSet::new(coordinator.topology(), [rank]).unwrap();
                    coordinator.begin_scoped(id, scope).unwrap();
                    coordinator.prepare(id, rank).unwrap();
                    coordinator.reserve(id, 1).unwrap();
                }
                let id = ExecutionTransactionId::new(1).unwrap();
                let error = publish_dp_completion(
                    &mut coordinator,
                    HostCompletion {
                        rank: ParallelRankId::new(0),
                        session: SessionId(0),
                        transaction: id,
                        cancellation_requested,
                        outcome: CompletionOutcome::QuiescenceUnknown,
                    },
                )
                .unwrap_err();
                assert!(error.to_string().contains("fatal DP rank=0 transaction=1"));
                assert_eq!(coordinator.state(id).unwrap(), TransactionState::Preparing);
                assert_eq!(coordinator.publication_count(), 0);
                assert_eq!(coordinator.retained_transaction_count(), 2);
                assert_eq!(coordinator.in_use_credits(), 2);

                // Pure protocol check: evidence from an independent, proven rank
                // cannot release the unknown rank's reservation or transaction.
                let values = publish_dp_completion(
                    &mut coordinator,
                    HostCompletion {
                        rank: ParallelRankId::new(1),
                        session: SessionId(1),
                        transaction: ExecutionTransactionId::new(2).unwrap(),
                        cancellation_requested: false,
                        outcome: CompletionOutcome::Success(vec![7.0]),
                    },
                )
                .unwrap();
                assert_eq!(values, [7.0]);
                assert_eq!(coordinator.state(id).unwrap(), TransactionState::Preparing);
                assert_eq!(coordinator.publication_count(), 1);
                assert_eq!(coordinator.retained_transaction_count(), 1);
                assert_eq!(coordinator.in_use_credits(), 1);
            }
        }

        #[test]
        fn dp_business_failures_ack_safe_cleanup_and_cancellation_never_publishes() {
            use ferrule_common::{ParallelTopologyId, ParallelismPlan, ValidatedParallelTopology};
            use ferrule_runtime::{ExecutionTransactionId, SessionId};
            for (outcome, cancellation_requested, succeeds, publications) in [
                (CompletionOutcome::Success(vec![7.0]), false, true, 1),
                (
                    CompletionOutcome::Failed(anyhow::anyhow!("compute failed")),
                    false,
                    false,
                    1,
                ),
                (CompletionOutcome::Panicked, false, false, 1),
                (CompletionOutcome::ReplicaUnavailable, false, false, 1),
                (CompletionOutcome::Cancelled, false, false, 0),
                (CompletionOutcome::Success(vec![7.0]), true, false, 0),
                (
                    CompletionOutcome::Failed(anyhow::anyhow!("cancelled compute")),
                    true,
                    false,
                    0,
                ),
            ] {
                let rank = ParallelRankId::new(0);
                let topology = ValidatedParallelTopology::new(
                    ParallelTopologyId::new(1),
                    1,
                    rank,
                    ParallelismPlan::default(),
                )
                .unwrap();
                let mut coordinator = DistributedTransaction::new(topology, 1);
                let id = ExecutionTransactionId::new(1).unwrap();
                coordinator.begin(id).unwrap();
                coordinator.prepare(id, rank).unwrap();
                coordinator.reserve(id, 1).unwrap();
                let result = publish_dp_completion(
                    &mut coordinator,
                    HostCompletion {
                        rank,
                        session: SessionId(0),
                        transaction: id,
                        cancellation_requested,
                        outcome,
                    },
                );
                assert_eq!(result.is_ok(), succeeds);
                assert_eq!(coordinator.publication_count(), publications);
                assert_eq!(coordinator.in_use_credits(), 0);
                assert_eq!(coordinator.retained_transaction_count(), 0);
            }
        }

        #[test]
        fn dp_rejected_reservation_or_submission_cleans_up_without_publication() {
            use ferrule_common::{ParallelTopologyId, ParallelismPlan, ValidatedParallelTopology};
            use ferrule_runtime::ExecutionTransactionId;
            for capacity in [0, 1] {
                let rank = ParallelRankId::new(0);
                let topology = ValidatedParallelTopology::new(
                    ParallelTopologyId::new(1),
                    1,
                    rank,
                    ParallelismPlan::default(),
                )
                .unwrap();
                let mut coordinator = DistributedTransaction::new(topology, capacity);
                let id = ExecutionTransactionId::new(1).unwrap();
                coordinator.begin(id).unwrap();
                coordinator.prepare(id, rank).unwrap();
                assert_eq!(coordinator.reserve(id, 1).is_ok(), capacity != 0);
                cancel_unsubmitted(&mut coordinator, id, rank).unwrap();
                assert_eq!(coordinator.publication_count(), 0);
                assert_eq!(coordinator.in_use_credits(), 0);
                assert_eq!(coordinator.retained_transaction_count(), 0);
            }
        }

        #[test]
        fn fatal_execution_and_join_failures_are_both_reported() {
            let error = finish::<()>(
                Err(anyhow::anyhow!("fatal DP quiescence unknown")),
                Err(anyhow::anyhow!("rank 1 join failed")),
            )
            .unwrap_err();
            let text = format!("{error:#}");
            assert!(text.contains("fatal DP quiescence unknown"));
            assert!(text.contains("rank 1 join failed"));
            assert!(finish(Ok(()), Err(anyhow::anyhow!("join failed"))).is_err());
            assert_eq!(finish(Ok(3), Ok(())).unwrap(), 3);
        }

        #[test]
        fn rejects_invalid_timing_samples_and_sample_count() {
            for samples in [
                vec![],
                vec![0.0],
                vec![-1.0],
                vec![f64::NAN],
                vec![f64::INFINITY],
                vec![f64::MAX, f64::MAX],
            ] {
                assert!(summarize(samples).is_err());
            }
            let options = args("dp");
            let validated = validate(&options, &fixture()).unwrap();
            assert!(report(&options, &validated, vec![1.0], outputs()).is_err());
        }

        #[cfg(not(feature = "cuda"))]
        #[test]
        fn default_build_rejects_all_modes_without_reading_fixture_or_cpu_fallback() {
            for mode in ["dp", "tp-column", "tp-row"] {
                let error = crate::commands::bench_parallel::cmd_bench_parallel(args(mode))
                    .unwrap_err()
                    .to_string();
                assert!(error.contains("CUDA-only"));
                assert!(error.contains("--features cuda"));
                assert!(error.contains("no CPU fallback"));
            }
        }

        #[test]
        fn output_verification_rejects_missing_nonfinite_or_different_tp_values() {
            assert!(verify_outputs(&outputs(), 2, 6, Mode::TpColumn).is_ok());
            assert!(verify_outputs(&outputs()[..1], 2, 6, Mode::TpColumn).is_err());
            for mode in [Mode::Dp, Mode::TpColumn, Mode::TpRow] {
                for rank in 0..2 {
                    for nonfinite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                        let mut bad = outputs();
                        bad[rank].values[0] = nonfinite;
                        assert!(verify_outputs(&bad, 2, 6, mode).is_err());
                    }
                    let mut short = outputs();
                    short[rank].values.pop();
                    assert!(verify_outputs(&short, 2, 6, mode).is_err());
                    let mut long = outputs();
                    long[rank].values.push(0.0);
                    assert!(verify_outputs(&long, 2, 6, mode).is_err());
                }
            }
            let mut different = outputs();
            different[1].values[0] += 1.0;
            assert!(verify_outputs(&different, 2, 6, Mode::TpRow).is_err());
            assert!(verify_outputs(&different, 2, 6, Mode::TpColumn).is_err());
            assert!(verify_outputs(&different, 2, 6, Mode::Dp).is_ok());
            different[1].rank = 0;
            assert!(verify_outputs(&different, 2, 6, Mode::Dp).is_err());
        }
    }
}
