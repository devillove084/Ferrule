//! Private process-rank endpoint and adapter, not a parallel executor.
//! The production DP/TP executors still own all routing and publication.

use crate::args::RankWorkerArgs;

pub(crate) fn serve(args: RankWorkerArgs) -> anyhow::Result<()> {
    #[cfg(unix)]
    return process::serve(args);
    #[cfg(not(unix))]
    {
        let _ = args;
        anyhow::bail!("__rank-worker requires Unix process pipes");
    }
}

#[cfg(any(feature = "cuda", test))]
use crate::args::{BenchParallelMode as Mode, RankBackend};
#[cfg(any(feature = "cuda", test))]
use crate::commands::parallel_worker::elements;
#[cfg(any(feature = "cuda", test))]
use anyhow::{Context, Result, ensure};
#[cfg(any(feature = "cuda", test))]
use ferrule_common::ParallelRankId;
#[cfg(any(feature = "cuda", test))]
use ferrule_model::transformer::parallel::{
    TensorParallelLinearPartition as Partition, TensorParallelLinearPlan,
};
#[cfg(any(feature = "cuda", test))]
use serde::{Deserialize, Serialize};

// One invocation is one execution epoch. There is deliberately no replay API.
#[cfg(any(feature = "cuda", test))]
pub(crate) const EXECUTION_EPOCH: u64 = 1;

#[cfg(any(feature = "cuda", test))]
pub(crate) fn validate_options(
    backend: RankBackend,
    timeout_ms: u64,
    restarts: usize,
) -> Result<()> {
    ensure!(
        restarts == 0,
        "--rank-restarts must be 0: safe whole-epoch retry is not integrated; no automatic replay of published DP siblings"
    );
    ensure!(timeout_ms > 0, "--rank-timeout-ms must be positive");
    std::time::Instant::now()
        .checked_add(std::time::Duration::from_millis(timeout_ms))
        .context("rank timeout overflows Instant")?;
    ensure!(
        backend != RankBackend::Process || cfg!(unix),
        "process rank backend requires Unix"
    );
    Ok(())
}

#[cfg(any(feature = "cuda", test))]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WorkerSpec {
    pub(crate) mode: Mode,
    pub(crate) ranks: usize,
    pub(crate) rank: u32,
    pub(crate) device: usize,
    pub(crate) rows: usize,
    pub(crate) in_features: usize,
    pub(crate) out_features: usize,
}

#[cfg(any(feature = "cuda", test))]
impl WorkerSpec {
    pub(crate) fn shard_rank(self) -> ParallelRankId {
        ParallelRankId::new(if self.mode == Mode::Dp { 0 } else { self.rank })
    }

    pub(crate) fn plan(self) -> Result<TensorParallelLinearPlan> {
        ensure!(
            self.ranks > 0
                && u32::try_from(self.ranks).is_ok()
                && (self.rank as usize) < self.ranks,
            "invalid worker rank geometry"
        );
        ensure!(
            i32::try_from(self.device).is_ok(),
            "CUDA device ordinal exceeds i32"
        );
        elements(self.rows, self.in_features)?;
        elements(self.rows, self.out_features)?;
        let plan = TensorParallelLinearPlan::new(
            self.out_features,
            self.in_features,
            if self.mode == Mode::Dp { 1 } else { self.ranks },
            if self.mode == Mode::TpRow {
                Partition::Row
            } else {
                Partition::Column
            },
        )?;
        let (out, input) = plan.local_shape(self.shard_rank())?;
        for count in [
            elements(out, input)?,
            elements(self.rows, out)?,
            elements(self.rows, input)?,
        ] {
            ensure!(
                u32::try_from(count).is_ok(),
                "CUDA F32 element indices exceed u32"
            );
        }
        Ok(plan)
    }
}

#[cfg(unix)]
pub(crate) mod process {
    use super::*;
    use ferrule_runtime::parallel::process::{
        ProcessChildConfig, ProcessChildHandler, ProcessFailureKind, ProcessFrameLimits,
        ProcessHandlerError, ProcessRequest, child_serve_with,
    };
    use std::fs::File;
    use std::os::fd::AsFd;
    use std::time::Duration;

    #[cfg(any(feature = "cuda", test))]
    use ferrule_runtime::parallel::process::{
        PROCESS_REAPER_CAPACITY, ProcessError, ProcessGroupEpoch, ProcessIdentity,
        ProcessOwnerInstanceId,
    };
    #[cfg(feature = "cuda")]
    use ferrule_runtime::parallel::process::{ProcessLaunch, ProcessOwnerConfig, ProcessRankOwner};

    #[cfg(any(feature = "cuda", test))]
    pub(crate) const TERMINATE_GRACE_MS: u64 = 100;
    #[cfg(any(feature = "cuda", test))]
    pub(crate) const KILL_GRACE_MS: u64 = 500;

    #[cfg(any(feature = "cuda", test))]
    #[derive(Debug, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct BootConfig {
        spec: WorkerSpec,
        weight: Vec<f32>,
    }

    #[cfg(any(feature = "cuda", test))]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    enum Phase {
        Compute,
        Apply,
    }

    #[cfg(any(feature = "cuda", test))]
    #[derive(Debug, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct WireCommand {
        // Outer transaction, not the TP executor's private DP admission ID.
        transaction: u64,
        rank: u32,
        phase: Phase,
        rows: usize,
        values: Vec<f32>,
    }

    #[cfg(any(feature = "cuda", test))]
    #[derive(Debug, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct WireOutput {
        transaction: u64,
        rank: u32,
        phase: Phase,
        rows: usize,
        values: Vec<f32>,
    }

    #[cfg(feature = "cuda")]
    type Owner = ProcessRankOwner<BootConfig, WireCommand, WireOutput>;

    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn frame_limits(spec: WorkerSpec) -> Result<ProcessFrameLimits> {
        spec.plan()?;
        ensure!(
            spec.ranks <= PROCESS_REAPER_CAPACITY,
            "process ranks exceed production reaper capacity {PROCESS_REAPER_CAPACITY}"
        );
        // Conservative finite-F32 JSON bound, including serde_json::Value's
        // f64 representation. Metadata and envelopes get separate headroom.
        let encoded = |n: usize| {
            n.checked_mul(32)
                .and_then(|n| n.checked_add(4096))
                .context("process JSON byte budget overflow")
        };
        let config = encoded(elements(spec.out_features, spec.in_features)?)?;
        let input = elements(spec.rows, spec.in_features)?;
        let output = elements(spec.rows, spec.out_features)?;
        let command = encoded(input.max(output))?;
        let output = encoded(output)?;
        let frame = config
            .max(command)
            .max(output)
            .checked_add(4096)
            .context("process frame budget overflow")?;
        let limits = ProcessFrameLimits {
            max_frame_bytes: frame,
            max_config_bytes: config,
            max_command_bytes: command,
            max_output_bytes: output,
            max_error_bytes: 4096,
        };
        limits
            .validate()
            .context("fixture exceeds supported bounded process frame size")?;
        Ok(limits)
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn idle_timeout_ms(timeout_ms: u64, ranks: usize) -> Result<u64> {
        // Keep the launch-time I/O allowance for serial initialization. This
        // bounds Boot and frame transfer, NOT idle READY children: child_serve
        // waits without a deadline until a new command frame starts.
        let rounds = u64::try_from(ranks)?
            .checked_add(1)
            .context("idle timeout overflow")?;
        let idle = timeout_ms
            .checked_mul(rounds)
            .context("idle timeout overflow")?;
        validate_options(RankBackend::Process, idle, 0)?;
        Ok(idle)
    }

    #[cfg(any(feature = "cuda", test))]
    #[derive(Debug, Clone, Serialize)]
    pub(crate) struct Metadata {
        protocol_version: u16,
        frame_limits: ProcessFrameLimits,
        startup_timeout_ms: u64,
        command_timeout_ms: u64,
        shutdown_timeout_ms: u64,
        child_io_timeout_ms: u64,
        terminate_grace_ms: u64,
        kill_grace_ms: u64,
        owners: Vec<ProcessIdentity>,
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn metadata(spec: WorkerSpec, timeout_ms: u64) -> Result<Metadata> {
        validate_options(RankBackend::Process, timeout_ms, 0)?;
        let limits = frame_limits(spec)?;
        Ok(Metadata {
            protocol_version: ferrule_runtime::parallel::process::PROCESS_PROTOCOL_VERSION,
            frame_limits: limits,
            startup_timeout_ms: timeout_ms,
            command_timeout_ms: timeout_ms,
            shutdown_timeout_ms: timeout_ms,
            child_io_timeout_ms: idle_timeout_ms(timeout_ms, spec.ranks)?,
            terminate_grace_ms: TERMINATE_GRACE_MS,
            kill_grace_ms: KILL_GRACE_MS,
            owners: (0..spec.ranks)
                .map(|rank| identity(rank as u32))
                .collect::<Result<_>>()?,
        })
    }

    #[cfg(any(feature = "cuda", test))]
    fn identity(rank: u32) -> Result<ProcessIdentity> {
        Ok(ProcessIdentity::new(
            ProcessGroupEpoch::new(EXECUTION_EPOCH)?,
            ParallelRankId::new(rank),
            ProcessOwnerInstanceId::new(u64::from(rank) + 1)?,
        ))
    }

    #[cfg(any(feature = "cuda", test))]
    fn check_command(spec: WorkerSpec, command: &WireCommand, transaction: u64) -> Result<()> {
        ensure!(
            command.transaction == transaction && transaction != 0 && command.rank == spec.rank,
            "rank/outer transaction mismatch"
        );
        ensure!(command.rows == spec.rows, "rank command rows mismatch");
        ensure!(
            spec.mode != Mode::Dp || command.phase == Phase::Compute,
            "DP does not accept Apply"
        );
        let width = if command.phase == Phase::Compute {
            spec.in_features
        } else {
            spec.out_features
        };
        ensure!(
            command.values.len() == elements(spec.rows, width)?,
            "rank command values length mismatch"
        );
        ensure!(
            command.values.iter().all(|x| x.is_finite()),
            "rank command contains nonfinite values"
        );
        Ok(())
    }

    #[cfg(any(feature = "cuda", test))]
    fn check_output(spec: WorkerSpec, command: &WireCommand, output: &WireOutput) -> Result<()> {
        ensure!(
            output.transaction == command.transaction
                && output.rank == spec.rank
                && output.phase == command.phase
                && output.rows == spec.rows,
            "rank output identity/phase/rows mismatch"
        );
        let width = if command.phase == Phase::Compute {
            spec.plan()?.local_shape(spec.shard_rank())?.0
        } else {
            spec.out_features
        };
        ensure!(
            output.values.len() == elements(spec.rows, width)?,
            "rank output values length mismatch"
        );
        ensure!(
            output.values.iter().all(|x| x.is_finite()),
            "rank output contains nonfinite values"
        );
        Ok(())
    }

    #[cfg(any(feature = "cuda", test))]
    fn command_rejection(error: &ProcessError) -> bool {
        error.is_command_rejection()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn spec(mode: Mode) -> WorkerSpec {
            WorkerSpec {
                mode,
                ranks: 2,
                rank: 1,
                device: 3,
                rows: 2,
                in_features: 3,
                out_features: 3,
            }
        }

        #[test]
        fn bounded_budgets_cover_actual_json_and_reject_oversize_geometry() {
            let spec = spec(Mode::TpColumn);
            let limits = frame_limits(spec).unwrap();
            let boot = BootConfig {
                spec,
                weight: vec![f32::MAX; 9],
            };
            // Match the runtime's intermediate Value encoding, not just direct F32 JSON.
            let encoded = serde_json::to_vec(&serde_json::to_value(&boot).unwrap()).unwrap();
            assert!(encoded.len() <= limits.max_config_bytes);
            let decoded: BootConfig = serde_json::from_slice(&encoded).unwrap();
            assert_eq!(decoded.weight, boot.weight);
            assert!(limits.max_config_bytes < limits.max_frame_bytes);
            let meta = metadata(spec, 500).unwrap();
            assert_eq!(meta.command_timeout_ms, 500);
            assert_eq!(meta.child_io_timeout_ms, 1500);
            assert_eq!(meta.owners.len(), 2);
            assert_eq!(meta.owners[1].rank.get(), 1);
            assert_eq!(meta.owners[1].epoch.get(), EXECUTION_EPOCH);
            assert_eq!(meta.owners[1].owner_instance.get(), 2);
            assert!(
                frame_limits(WorkerSpec {
                    rows: usize::MAX,
                    ..spec
                })
                .is_err()
            );
            assert!(
                frame_limits(WorkerSpec {
                    in_features: 4096,
                    out_features: 4096,
                    ..spec
                })
                .is_err()
            );
            assert!(
                frame_limits(WorkerSpec {
                    mode: Mode::Dp,
                    ranks: PROCESS_REAPER_CAPACITY + 1,
                    ..spec
                })
                .is_err()
            );
            assert!(idle_timeout_ms(u64::MAX, 2).is_err());
        }

        #[test]
        fn dto_roundtrip_preserves_outer_transaction_phase_and_f32_values() {
            for mode in [Mode::Dp, Mode::TpColumn, Mode::TpRow] {
                let spec = spec(mode);
                let command = WireCommand {
                    transaction: 17,
                    rank: 1,
                    phase: Phase::Compute,
                    rows: 2,
                    values: vec![0.0, -0.0, 0.12345679, -1.0, f32::MIN_POSITIVE, f32::MAX],
                };
                let encoded = serde_json::to_vec(&serde_json::to_value(&command).unwrap()).unwrap();
                assert!(encoded.len() <= frame_limits(spec).unwrap().max_command_bytes);
                let mut decoded: WireCommand = serde_json::from_slice(&encoded).unwrap();
                check_command(spec, &decoded, 17).unwrap();
                for (left, right) in command.values.iter().zip(&decoded.values) {
                    assert_eq!(left.to_bits(), right.to_bits());
                }
                assert!(check_command(spec, &decoded, 18).is_err());
                decoded.rank = 0;
                assert!(check_command(spec, &decoded, 17).is_err());
                decoded.rank = 1;
                decoded.rows = 1;
                assert!(check_command(spec, &decoded, 17).is_err());
                decoded.rows = 2;
                decoded.values[0] = f32::NAN;
                assert!(check_command(spec, &decoded, 17).is_err());
                decoded.values[0] = 0.0;
                decoded.values.pop();
                assert!(check_command(spec, &decoded, 17).is_err());

                let width = spec
                    .plan()
                    .unwrap()
                    .local_shape(spec.shard_rank())
                    .unwrap()
                    .0;
                let mut output = WireOutput {
                    transaction: 17,
                    rank: 1,
                    phase: Phase::Compute,
                    rows: 2,
                    values: vec![1.0; 2 * width],
                };
                check_output(spec, &command, &output).unwrap();
                output.transaction = 18;
                assert!(check_output(spec, &command, &output).is_err());
                output.transaction = 17;
                output.values[0] = f32::INFINITY;
                assert!(check_output(spec, &command, &output).is_err());
                output.values[0] = 1.0;
                output.values.push(1.0);
                assert!(check_output(spec, &command, &output).is_err());

                let apply = WireCommand {
                    phase: Phase::Apply,
                    values: vec![1.0; 6],
                    ..command
                };
                assert_eq!(check_command(spec, &apply, 17).is_ok(), mode != Mode::Dp);
                if mode != Mode::Dp {
                    check_output(
                        spec,
                        &apply,
                        &WireOutput {
                            transaction: 17,
                            rank: 1,
                            phase: Phase::Apply,
                            rows: 2,
                            values: vec![1.0; 6],
                        },
                    )
                    .unwrap();
                }
            }
        }

        #[test]
        fn os_reaping_and_process_loss_are_not_gpu_fence_evidence() {
            use ferrule_runtime::parallel::process::{
                ProcessTerminationReport, ReapOutcome, UnknownCause,
            };
            let failure = ProcessError::QuiescenceUnknown {
                cause: UnknownCause::Deadline,
                source: Box::new(ProcessError::Deadline),
                termination: ProcessTerminationReport {
                    reap: ReapOutcome::Reaped,
                    signal_errors: vec![],
                    reap_error: None,
                },
            };
            assert!(!command_rejection(&failure));
            assert!(!command_rejection(&ProcessError::OwnerUnavailable));
            assert!(!command_rejection(&ProcessError::PipeClosed));
            assert!(!command_rejection(&ProcessError::RemoteFailure {
                failure: ProcessHandlerError::unknown(ProcessFailureKind::Handler, "not fenced")
            }));
            assert!(command_rejection(&ProcessError::RemoteFailure {
                failure: ProcessHandlerError::fenced(
                    ProcessFailureKind::Handler,
                    "safe ordinary error"
                )
            }));
        }

        #[test]
        fn local_command_rejections_are_business_errors_not_lost_fences() {
            for error in [
                ProcessError::FrameTooLarge { limit: 1024 },
                ProcessError::Json {
                    operation: "serialize process value",
                    source: serde_json::from_str::<serde_json::Value>("invalid").unwrap_err(),
                },
                ProcessError::Deadline,
                ProcessError::IdentityExhausted,
                ProcessError::Cancelled,
            ] {
                assert!(command_rejection(&error), "{error:?}");
                let unknown = ProcessError::QuiescenceUnknown {
                    cause: ferrule_runtime::parallel::process::UnknownCause::ProtocolViolation,
                    source: Box::new(error),
                    termination: ferrule_runtime::parallel::process::ProcessTerminationReport {
                        reap: ferrule_runtime::parallel::process::ReapOutcome::Reaped,
                        signal_errors: vec![],
                        reap_error: None,
                    },
                };
                assert!(!command_rejection(&unknown));
            }
        }

        #[test]
        fn restart_and_timeout_contract_is_explicit() {
            for backend in [RankBackend::Thread, RankBackend::Process] {
                assert!(validate_options(backend, 30000, 0).is_ok());
                assert!(validate_options(backend, 0, 0).is_err());
                let error = validate_options(backend, 30000, 1).unwrap_err();
                assert!(error.to_string().contains("no automatic replay"));
            }
        }
    }

    #[cfg(feature = "cuda")]
    struct Handler {
        spec: WorkerSpec,
        linear: crate::commands::parallel_worker::cuda::LinearWorker,
    }

    #[cfg(feature = "cuda")]
    impl ProcessChildHandler for Handler {
        type Command = WireCommand;
        type Output = WireOutput;

        fn execute(
            &mut self,
            request: ProcessRequest<WireCommand>,
        ) -> std::result::Result<WireOutput, ProcessHandlerError> {
            use ferrule_runtime::parallel::tensor::TensorCommand;
            check_command(
                self.spec,
                &request.command,
                request.identity.transaction.get(),
            )
            .map_err(|error| {
                ProcessHandlerError::fenced(ProcessFailureKind::CommandDecode, format!("{error:#}"))
            })?;
            let command = request.command;
            let gpu = match command.phase {
                Phase::Compute => TensorCommand::Compute {
                    input: command.values.into(),
                    rows: command.rows,
                },
                Phase::Apply => TensorCommand::Apply {
                    values: command.values,
                    rows: command.rows,
                },
            };
            // The one shared worker supplies the actual CUDA computation/fences.
            // Typed Unknown panics propagate to child_serve's Unknown boundary.
            let values = self.linear.execute_command(gpu).map_err(|error| {
                ProcessHandlerError::fenced(ProcessFailureKind::Handler, format!("{error:#}"))
            })?;
            Ok(WireOutput {
                transaction: command.transaction,
                rank: self.spec.rank,
                phase: command.phase,
                rows: command.rows,
                values,
            })
        }

        fn shutdown(&mut self) -> std::result::Result<(), ProcessHandlerError> {
            self.linear.shutdown_worker().map_err(|error| {
                ProcessHandlerError::fenced(ProcessFailureKind::Shutdown, format!("{error:#}"))
            })
        }
    }

    #[cfg(not(feature = "cuda"))]
    fn no_cuda() -> ProcessHandlerError {
        ProcessHandlerError::fenced(
            ProcessFailureKind::Startup,
            "__rank-worker is CUDA-only; rebuild ferrule-cli with --features cuda (no CPU fallback)",
        )
    }

    enum Endpoint {
        #[cfg(feature = "cuda")]
        Linear(Handler),
        Decoder(ferrule_runtime::parallel::process::decoder::DecoderEndpoint),
    }
    impl ProcessChildHandler for Endpoint {
        type Command = serde_json::Value;
        type Output = serde_json::Value;
        fn execute(
            &mut self,
            request: ProcessRequest<Self::Command>,
        ) -> std::result::Result<Self::Output, ProcessHandlerError> {
            match self {
                Self::Decoder(child) => child.execute(request),
                #[cfg(feature = "cuda")]
                Self::Linear(child) => {
                    let command = serde_json::from_value(request.command).map_err(|e| {
                        ProcessHandlerError::fenced(
                            ProcessFailureKind::CommandDecode,
                            e.to_string(),
                        )
                    })?;
                    let reply = child.execute(ProcessRequest {
                        identity: request.identity,
                        session: request.session,
                        command,
                    })?;
                    serde_json::to_value(reply).map_err(|e| {
                        ProcessHandlerError::unknown(
                            ProcessFailureKind::OutputEncode,
                            e.to_string(),
                        )
                    })
                }
            }
        }
        fn shutdown(&mut self) -> std::result::Result<(), ProcessHandlerError> {
            match self {
                Self::Decoder(child) => child.shutdown(),
                #[cfg(feature = "cuda")]
                Self::Linear(child) => child.shutdown(),
            }
        }
    }

    pub(super) fn serve(args: RankWorkerArgs) -> anyhow::Result<()> {
        // Owned raw files: no stdout line buffer and no tracing on this branch.
        let reader = File::from(std::io::stdin().as_fd().try_clone_to_owned()?);
        let writer = File::from(std::io::stdout().as_fd().try_clone_to_owned()?);
        let ceiling = ProcessFrameLimits {
            max_frame_bytes: args.max_frame_bytes,
            max_config_bytes: args.max_frame_bytes,
            max_command_bytes: args.max_frame_bytes,
            max_output_bytes: args.max_frame_bytes,
            max_error_bytes: args.max_frame_bytes.min(4096),
        };
        child_serve_with::<_, _, Endpoint, _>(
            reader,
            writer,
            ProcessChildConfig {
                frame_limits: ceiling,
                io_timeout: Duration::from_millis(args.io_timeout_ms),
            },
            |boot| {
                use ferrule_runtime::parallel::process::decoder::DecoderEndpoint;
                if DecoderEndpoint::accepts(&boot.config) {
                    let executable = std::env::current_exe().map_err(|e| {
                        ProcessHandlerError::fenced(ProcessFailureKind::Startup, e.to_string())
                    })?;
                    let launch = ferrule_runtime::parallel::process::ProcessLaunch::new(executable)
                        .arg("__rank-worker")
                        .arg("--max-frame-bytes")
                        .arg(args.max_frame_bytes.to_string())
                        .arg("--io-timeout-ms")
                        .arg(args.io_timeout_ms.to_string());
                    return DecoderEndpoint::initialize(boot, launch).map(Endpoint::Decoder);
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = boot;
                    Err(no_cuda())
                }
                #[cfg(feature = "cuda")]
                {
                    let validated = (|| -> Result<BootConfig> {
                        let config: BootConfig = serde_json::from_value(boot.config)?;
                        ensure!(
                            boot.identity == identity(config.spec.rank)?,
                            "Boot identity mismatch"
                        );
                        ensure!(
                            boot.limits == frame_limits(config.spec)?,
                            "Boot limits do not match checked geometry"
                        );
                        ensure!(
                            config.weight.len()
                                == elements(config.spec.out_features, config.spec.in_features)?,
                            "Boot weight length mismatch"
                        );
                        ensure!(
                            config.weight.iter().all(|x| x.is_finite()),
                            "Boot weight contains nonfinite values"
                        );
                        Ok(config)
                    })()
                    .map_err(|error| {
                        ProcessHandlerError::fenced(
                            ProcessFailureKind::Startup,
                            format!("{error:#}"),
                        )
                    })?;
                    let spec = validated.spec;
                    let plan = spec.plan().map_err(|error| {
                        ProcessHandlerError::fenced(
                            ProcessFailureKind::Startup,
                            format!("{error:#}"),
                        )
                    })?;
                    let linear = crate::commands::parallel_worker::cuda::LinearWorker::new(
                        spec.device,
                        plan,
                        spec.shard_rank(),
                        &validated.weight,
                        spec.rows,
                    )
                    .map_err(|error| {
                        ProcessHandlerError::fenced(
                            ProcessFailureKind::Startup,
                            format!("{error:#}"),
                        )
                    })?;
                    Ok(Endpoint::Linear(Handler { spec, linear }))
                }
            },
        )?;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub(crate) struct Proxy {
        owner: Option<Owner>,
        pub(super) spec: WorkerSpec,
        quiescent: bool,
    }

    #[cfg(feature = "cuda")]
    impl Proxy {
        pub(super) fn spawn(spec: WorkerSpec, timeout_ms: u64, weight: &[f32]) -> Result<Self> {
            let limits = frame_limits(spec)?;
            let config = BootConfig {
                spec,
                weight: weight.to_vec(),
            };
            let launch = ProcessLaunch::new(std::env::current_exe()?)
                .arg("__rank-worker")
                .arg("--max-frame-bytes")
                .arg(limits.max_frame_bytes.to_string())
                .arg("--io-timeout-ms")
                .arg(idle_timeout_ms(timeout_ms, spec.ranks)?.to_string());
            let options = ProcessOwnerConfig {
                frame_limits: limits,
                startup_timeout: Duration::from_millis(timeout_ms),
                command_timeout: Duration::from_millis(timeout_ms),
                terminate_grace: Duration::from_millis(TERMINATE_GRACE_MS),
                kill_grace: Duration::from_millis(KILL_GRACE_MS),
            };
            let owner = match Owner::spawn(launch, identity(spec.rank)?, &config, options) {
                Ok(owner) => owner,
                Err(error) => {
                    let message = format!(
                        "fatal process rank={} startup: {error}; no retry/recovery claimed",
                        spec.rank
                    );
                    let unknown = error.source.is_quiescence_unknown();
                    // A retained failed-start owner must reach the bounded reaper
                    // before the DP factory's Unknown signal is raised.
                    drop(error);
                    if unknown {
                        eprintln!("{message}");
                        std::panic::panic_any(
                            ferrule_runtime::parallel::data::PanicQuiescence::Unknown,
                        );
                    }
                    anyhow::bail!(message);
                }
            };
            Ok(Self {
                owner: Some(owner),
                spec,
                quiescent: true,
            })
        }

        fn unknown(&mut self, message: impl std::fmt::Display) -> ! {
            self.quiescent = false;
            if let Some(mut owner) = self.owner.take() {
                // execute/shutdown normally already performed this escalation.
                // It is idempotent for a reaped owner; never a device fence.
                let report = owner.terminate();
                eprintln!(
                    "bench-parallel: fatal process rank={} epoch={}: {message}; OS termination={report:?}; CUDA quiescence remains unknown; no replay",
                    self.spec.rank, EXECUTION_EPOCH
                );
                drop(owner); // Unreaped child goes to runtime's bounded reaper.
            } else {
                eprintln!(
                    "bench-parallel: fatal process rank={}: {message}; CUDA quiescence unknown",
                    self.spec.rank
                );
            }
            std::panic::panic_any(ferrule_runtime::parallel::data::PanicQuiescence::Unknown)
        }

        pub(super) fn execute(
            &mut self,
            transaction: ferrule_runtime::ExecutionTransactionId,
            session: u64,
            command: ferrule_runtime::parallel::tensor::TensorCommand,
        ) -> Result<Vec<f32>> {
            use ferrule_runtime::parallel::tensor::TensorCommand;
            let (phase, rows, values) = match command {
                TensorCommand::Compute { input, rows } => (Phase::Compute, rows, input.to_vec()),
                TensorCommand::Apply { values, rows } => (Phase::Apply, rows, values),
            };
            let command = WireCommand {
                transaction: transaction.get(),
                rank: self.spec.rank,
                phase,
                rows,
                values,
            };
            check_command(self.spec, &command, transaction.get())?;
            self.quiescent = false;
            let result = self.owner.as_mut().expect("live process owner").execute(
                transaction,
                session,
                &command,
            );
            match result {
                Ok(output) => {
                    if let Err(error) = check_output(self.spec, &command, &output) {
                        self.unknown(error);
                    }
                    self.quiescent = true;
                    Ok(output.values)
                }
                Err(error) if command_rejection(&error) => {
                    self.quiescent = true;
                    Err(error.into())
                }
                Err(error) => self.unknown(error),
            }
        }

        pub(super) fn panic_quiescence(
            &mut self,
        ) -> ferrule_runtime::parallel::data::PanicQuiescence {
            use ferrule_runtime::parallel::data::PanicQuiescence;
            if self.quiescent {
                return PanicQuiescence::Quiescent;
            }
            if let Some(mut owner) = self.owner.take() {
                let report = owner.terminate();
                eprintln!(
                    "bench-parallel: process panic cleanup rank={}: {report:?}; not CUDA fence evidence",
                    self.spec.rank
                );
                drop(owner);
            }
            PanicQuiescence::Unknown
        }

        pub(super) fn shutdown(&mut self) -> Result<()> {
            self.quiescent = false;
            let result = self.owner.as_mut().expect("live process owner").shutdown();
            match result {
                Ok(_) => {
                    self.quiescent = true;
                    self.owner.take();
                    Ok(())
                }
                Err(error) => self.unknown(error),
            }
        }
    }
}

#[cfg(feature = "cuda")]
pub(crate) enum RankWorker {
    Thread(crate::commands::parallel_worker::cuda::LinearWorker),
    #[cfg(unix)]
    Process(process::Proxy),
}

#[cfg(feature = "cuda")]
impl RankWorker {
    pub(crate) fn new(
        backend: RankBackend,
        timeout_ms: u64,
        spec: WorkerSpec,
        weight: &[f32],
    ) -> Result<Self> {
        let plan = spec.plan()?;
        match backend {
            RankBackend::Thread => Ok(Self::Thread(
                crate::commands::parallel_worker::cuda::LinearWorker::new(
                    spec.device,
                    plan,
                    spec.shard_rank(),
                    weight,
                    spec.rows,
                )?,
            )),
            #[cfg(unix)]
            RankBackend::Process => Ok(Self::Process(process::Proxy::spawn(
                spec, timeout_ms, weight,
            )?)),
            #[cfg(not(unix))]
            RankBackend::Process => {
                let _ = timeout_ms;
                anyhow::bail!("process ranks require Unix")
            }
        }
    }
}

#[cfg(feature = "cuda")]
mod adapters {
    use super::*;
    use ferrule_runtime::parallel::data::{PanicQuiescence, ReplicaWorker, WorkRequest};
    use ferrule_runtime::parallel::tensor::{TensorCommand, TensorWork};
    use std::sync::Arc;

    impl ReplicaWorker<Arc<[f32]>> for RankWorker {
        type Output = Vec<f32>;
        type Error = anyhow::Error;
        fn execute(&mut self, request: WorkRequest<Arc<[f32]>>) -> Result<Vec<f32>> {
            match self {
                Self::Thread(worker) => worker.execute(request),
                #[cfg(unix)]
                Self::Process(proxy) => {
                    ensure!(
                        !request.cancellation.is_requested(),
                        "DP request cancelled before process dispatch"
                    );
                    ensure!(
                        request.rank.get() == proxy.spec.rank,
                        "DP process rank mismatch"
                    );
                    proxy.execute(
                        request.transaction,
                        request.session.0,
                        TensorCommand::Compute {
                            input: request.input,
                            rows: proxy.spec.rows,
                        },
                    )
                }
            }
        }
        fn panic_quiescence(&mut self) -> PanicQuiescence {
            match self {
                Self::Thread(worker) => <_ as ReplicaWorker<Arc<[f32]>>>::panic_quiescence(worker),
                #[cfg(unix)]
                Self::Process(proxy) => proxy.panic_quiescence(),
            }
        }
        fn shutdown(&mut self) -> Result<()> {
            match self {
                Self::Thread(worker) => worker.shutdown_worker(),
                #[cfg(unix)]
                Self::Process(proxy) => proxy.shutdown(),
            }
        }
    }

    impl ReplicaWorker<TensorWork> for RankWorker {
        type Output = Vec<f32>;
        type Error = anyhow::Error;
        fn execute(&mut self, request: WorkRequest<TensorWork>) -> Result<Vec<f32>> {
            match self {
                Self::Thread(worker) => worker.execute(request),
                #[cfg(unix)]
                Self::Process(proxy) => {
                    ensure!(
                        !request.cancellation.is_requested(),
                        "TP request cancelled before process dispatch"
                    );
                    ensure!(
                        request.rank.get() == proxy.spec.rank
                            && request.input.rank.local == request.rank
                            && request.input.rank.global == request.rank,
                        "TP process rank mismatch"
                    );
                    proxy.execute(
                        request.input.transaction,
                        request.session.0,
                        request.input.command,
                    )
                }
            }
        }
        fn panic_quiescence(&mut self) -> PanicQuiescence {
            match self {
                Self::Thread(worker) => <_ as ReplicaWorker<TensorWork>>::panic_quiescence(worker),
                #[cfg(unix)]
                Self::Process(proxy) => proxy.panic_quiescence(),
            }
        }
        fn shutdown(&mut self) -> Result<()> {
            match self {
                Self::Thread(worker) => worker.shutdown_worker(),
                #[cfg(unix)]
                Self::Process(proxy) => proxy.shutdown(),
            }
        }
    }
}
