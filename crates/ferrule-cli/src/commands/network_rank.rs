//! Linear-only CLI policy over runtime's mTLS transport. No scheduler or commit RPC.
use crate::args::RankServeArgs;

pub(crate) fn serve(args: RankServeArgs) -> anyhow::Result<()> {
    #[cfg(all(unix, feature = "cuda"))]
    return implementation::serve(args);
    #[cfg(not(all(unix, feature = "cuda")))]
    {
        let _ = args;
        anyhow::bail!(
            "rank-serve requires Unix + CUDA; rebuild with --features cuda (no CPU/plaintext fallback)"
        )
    }
}

#[cfg(all(unix, any(feature = "cuda", test)))]
#[path = "../../tests/unit"]
pub(crate) mod implementation {
    use std::collections::BTreeSet;
    use std::fs::File;
    use std::io::Read;
    use std::net::SocketAddr;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::Duration;

    use anyhow::{Context, Result, ensure};
    use ferrule_runtime::parallel::data::{PanicQuiescence, ReplicaWorker, WorkRequest};
    use ferrule_runtime::parallel::network::{
        ClientTls, LinearMode, NetworkBinding, NetworkEndpoint, NetworkError, NetworkOptions,
        NetworkRankOwner, ServerTls, TlsFiles,
    };
    use ferrule_runtime::parallel::process::{
        ProcessFrameLimits, ProcessLaunch, ProcessOwnerConfig,
    };
    use ferrule_runtime::parallel::tensor::{TensorCommand, TensorWork};
    use serde::{Deserialize, de::DeserializeOwned};

    use crate::args::{BenchParallelArgs, BenchParallelMode as Mode, RankBackend};
    use crate::commands::bench_parallel::benchmark::Fixture;
    use crate::commands::rank_worker::{
        WorkerSpec,
        process::{
            BootConfig, KILL_GRACE_MS, Phase, TERMINATE_GRACE_MS, WireCommand, WireOutput,
            check_command, check_output, frame_limits, identity,
        },
    };

    const MAX_CONFIG_BYTES: u64 = 1 << 20;
    const MAX_FIXTURE_BYTES: u64 = 64 << 20;

    pub(crate) fn read_json<T: DeserializeOwned>(path: &Path, limit: u64) -> Result<T> {
        let mut bytes = Vec::new();
        File::open(path)
            .context("open local network configuration")?
            .take(limit + 1)
            .read_to_end(&mut bytes)?;
        ensure!(
            bytes.len() as u64 <= limit,
            "network configuration byte limit exceeded"
        );
        serde_json::from_slice(&bytes).map_err(|error| {
            anyhow::anyhow!(
                "network JSON {:?} error at {}:{}",
                error.classify(),
                error.line(),
                error.column()
            )
        })
    }

    #[derive(Clone, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub(crate) struct RemoteRank {
        pub(crate) address: SocketAddr,
        pub(crate) server_name: String,
        pub(crate) tls: TlsFiles,
        pub(crate) binding: NetworkBinding,
        pub(crate) startup_timeout_ms: u64,
        pub(crate) command_timeout_ms: u64,
        pub(crate) limits: ProcessFrameLimits,
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    pub(crate) struct Manifest {
        pub(crate) ranks: Vec<RemoteRank>,
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    pub(crate) struct EndpointConfig {
        pub(crate) fixture: PathBuf,
        pub(crate) tls: TlsFiles,
        pub(crate) binding: NetworkBinding,
        pub(crate) startup_timeout_ms: u64,
        pub(crate) command_timeout_ms: u64,
        pub(crate) limits: ProcessFrameLimits,
    }

    fn options(startup: u64, command: u64, limits: ProcessFrameLimits) -> Result<NetworkOptions> {
        let options = NetworkOptions {
            limits,
            startup_timeout: Duration::from_millis(startup),
            command_timeout: Duration::from_millis(command),
        };
        options.validate()?;
        Ok(options)
    }

    pub(crate) fn spec(binding: &NetworkBinding) -> Result<WorkerSpec> {
        binding.validate()?;
        let t = &binding.target;
        let result = WorkerSpec {
            mode: match t.mode {
                LinearMode::Dp => Mode::Dp,
                LinearMode::TpColumn => Mode::TpColumn,
                LinearMode::TpRow => Mode::TpRow,
            },
            ranks: t.world as usize,
            rank: binding.identity.rank.get(),
            device: t.device as usize,
            rows: t.max_rows,
            in_features: t.in_features,
            out_features: t.out_features,
        };
        result.plan()?;
        Ok(result)
    }

    fn boot(spec: WorkerSpec, fixture: &Fixture) -> Result<BootConfig> {
        spec.plan()?;
        ensure!(
            spec.rows == fixture.rows
                && spec.in_features == fixture.in_features
                && spec.out_features == fixture.out_features,
            "network target differs from local fixture geometry"
        );
        ensure!(
            fixture.weight.len()
                == spec
                    .in_features
                    .checked_mul(spec.out_features)
                    .context("weight length overflow")?
                && fixture.input.len()
                    == spec
                        .rows
                        .checked_mul(spec.in_features)
                        .context("input length overflow")?
                && fixture
                    .weight
                    .iter()
                    .chain(&fixture.input)
                    .all(|v| v.is_finite()),
            "invalid local linear fixture"
        );
        Ok(BootConfig {
            spec,
            weight: fixture.weight.clone(),
        })
    }

    fn validate_boot_limits(boot: &BootConfig, limits: ProcessFrameLimits) -> Result<()> {
        limits.validate()?;
        // Match the transport's canonical f64 JSON representation of F32 payloads.
        let encoded = serde_json::to_vec(&serde_json::to_value(boot)?)?;
        ensure!(
            encoded.len() <= limits.max_config_bytes
                && encoded
                    .len()
                    .checked_add(4096)
                    .is_some_and(|n| n <= limits.max_frame_bytes),
            "linear Boot exceeds network frame/config budget"
        );
        Ok(())
    }

    impl Manifest {
        pub(crate) fn load(path: &Path) -> Result<Self> {
            read_json(path, MAX_CONFIG_BYTES)
        }
        pub(crate) fn validate(
            &mut self,
            args: &BenchParallelArgs,
            fixture: &Fixture,
        ) -> Result<()> {
            ensure!(
                args.rank_backend == RankBackend::Thread && args.devices.is_none(),
                "remote manifest cannot be combined with local placement"
            );
            ensure!(
                args.rank_restarts == 0,
                "network reconnect/replay is unsupported"
            );
            ensure!(
                !self.ranks.is_empty() && self.ranks.len() == args.ranks && args.ranks <= 128,
                "manifest must cover exactly 1..=128 ranks"
            );
            self.ranks.sort_by_key(|r| r.binding.identity.rank.get());
            let first = &self.ranks[0].binding;
            let mut lifetimes = BTreeSet::new();
            let mut placements = BTreeSet::new();
            let mut addresses = BTreeSet::new();
            for (index, rank) in self.ranks.iter().enumerate() {
                let s = spec(&rank.binding)?;
                ensure!(
                    s.rank as usize == index && s.ranks == args.ranks && s.mode == args.mode,
                    "network rank coverage/world/mode mismatch"
                );
                ensure!(
                    rank.binding.run_id == first.run_id
                        && rank.binding.controller_id == first.controller_id
                        && rank.binding.identity.epoch == first.identity.epoch,
                    "inconsistent network run/controller/epoch"
                );
                ensure!(
                    lifetimes.insert(rank.binding.identity.owner_instance.get()),
                    "duplicate network owner lifetime"
                );
                ensure!(
                    placements.insert((&rank.binding.node_id, s.device)),
                    "duplicate (node, device) placement"
                );
                ensure!(
                    addresses.insert(rank.address),
                    "duplicate rank endpoint address"
                );
                ensure!(
                    rank.address.port() != 0 && !rank.address.ip().is_unspecified(),
                    "invalid remote socket address"
                );
                ensure!(
                    rank.command_timeout_ms == args.rank_timeout_ms,
                    "manifest command timeout must match --rank-timeout-ms"
                );
                options(
                    rank.startup_timeout_ms,
                    rank.command_timeout_ms,
                    rank.limits,
                )?;
                let boot = boot(s, fixture)?;
                validate_boot_limits(&boot, rank.limits)?;
            }
            Ok(())
        }

        pub(crate) fn prepare(&self, fixture: &Fixture) -> Result<Vec<PreparedRank>> {
            self.ranks
                .iter()
                .map(|rank| {
                    Ok(PreparedRank {
                        address: rank.address,
                        binding: rank.binding.clone(),
                        tls: ClientTls::load(&rank.tls, &rank.server_name)?,
                        options: options(
                            rank.startup_timeout_ms,
                            rank.command_timeout_ms,
                            rank.limits,
                        )?,
                        boot: Arc::new(boot(spec(&rank.binding)?, fixture)?),
                    })
                })
                .collect()
        }
    }

    #[derive(Clone)]
    pub(crate) struct PreparedRank {
        address: SocketAddr,
        binding: NetworkBinding,
        tls: ClientTls,
        options: NetworkOptions,
        boot: Arc<BootConfig>,
    }
    impl PreparedRank {
        pub(crate) fn connect(&self) -> Result<Proxy> {
            let owner = match NetworkRankOwner::connect(
                self.address,
                &self.tls,
                self.binding.clone(),
                &*self.boot,
                self.options,
            ) {
                Ok(owner) => owner,
                Err(NetworkError::Unknown(_)) => std::panic::panic_any(PanicQuiescence::Unknown),
                Err(error) => return Err(error.into()),
            };
            Ok(Proxy {
                owner: Some(owner),
                spec: self.boot.spec,
                quiescent: true,
            })
        }
    }

    pub(crate) struct Proxy {
        owner: Option<NetworkRankOwner<BootConfig, WireCommand, WireOutput>>,
        spec: WorkerSpec,
        quiescent: bool,
    }
    impl Proxy {
        fn execute_command(
            &mut self,
            transaction: ferrule_runtime::ExecutionTransactionId,
            session: u64,
            command: TensorCommand,
        ) -> Result<Vec<f32>> {
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
            let Some(owner) = self.owner.as_mut() else {
                std::panic::panic_any(PanicQuiescence::Unknown);
            };
            let output = match owner.execute_or_quarantine(transaction, session, &command) {
                Ok(output) => output,
                Err(error) => {
                    // execute_or_quarantine only returns errors known to be fenced.
                    self.quiescent = true;
                    return Err(error.into());
                }
            };
            if check_output(self.spec, &command, &output).is_err() {
                // Close the transport before the executor retains this quarantined
                // worker. Socket/child cleanup is NOT device fence evidence.
                self.owner.take();
                std::panic::panic_any(PanicQuiescence::Unknown);
            }
            self.quiescent = true;
            Ok(output.values)
        }
        pub(crate) fn panic_quiescence(&mut self) -> PanicQuiescence {
            if self.quiescent {
                PanicQuiescence::Quiescent
            } else {
                PanicQuiescence::Unknown
            }
        }
        pub(crate) fn shutdown(&mut self) -> Result<()> {
            self.quiescent = false;
            let Some(owner) = self.owner.as_mut() else {
                std::panic::panic_any(PanicQuiescence::Unknown);
            };
            owner.shutdown_or_quarantine()?;
            self.owner.take();
            self.quiescent = true;
            Ok(())
        }
    }
    impl ReplicaWorker<Arc<[f32]>> for Proxy {
        type Output = Vec<f32>;
        type Error = anyhow::Error;
        fn execute(&mut self, request: WorkRequest<Arc<[f32]>>) -> Result<Vec<f32>> {
            ensure!(
                !request.cancellation.is_requested() && request.rank.get() == self.spec.rank,
                "DP network rank/cancellation mismatch"
            );
            self.execute_command(
                request.transaction,
                request.session.0,
                TensorCommand::Compute {
                    input: request.input,
                    rows: self.spec.rows,
                },
            )
        }
        fn panic_quiescence(&mut self) -> PanicQuiescence {
            self.panic_quiescence()
        }
        fn shutdown(&mut self) -> Result<()> {
            self.shutdown()
        }
    }
    impl ReplicaWorker<TensorWork> for Proxy {
        type Output = Vec<f32>;
        type Error = anyhow::Error;
        fn execute(&mut self, request: WorkRequest<TensorWork>) -> Result<Vec<f32>> {
            ensure!(
                !request.cancellation.is_requested()
                    && request.rank.get() == self.spec.rank
                    && request.input.rank.local == request.rank
                    && request.input.rank.global == request.rank,
                "TP network rank/cancellation mismatch"
            );
            self.execute_command(
                request.input.transaction,
                request.session.0,
                request.input.command,
            )
        }
        fn panic_quiescence(&mut self) -> PanicQuiescence {
            self.panic_quiescence()
        }
        fn shutdown(&mut self) -> Result<()> {
            self.shutdown()
        }
    }

    impl EndpointConfig {
        pub(crate) fn endpoint(
            &self,
            launch: ProcessLaunch,
        ) -> Result<NetworkEndpoint<BootConfig>> {
            let fixture: Fixture = read_json(&self.fixture, MAX_FIXTURE_BYTES)?;
            let spec = spec(&self.binding)?;
            let config = boot(spec, &fixture)?;
            validate_boot_limits(&config, self.limits)?;
            Ok(NetworkEndpoint {
                binding: self.binding.clone(),
                options: options(
                    self.startup_timeout_ms,
                    self.command_timeout_ms,
                    self.limits,
                )?,
                tls: ServerTls::load(&self.tls)?,
                launch,
                local_identity: identity(spec.rank)?,
                local_options: ProcessOwnerConfig {
                    frame_limits: frame_limits(spec)?,
                    startup_timeout: Duration::from_millis(self.startup_timeout_ms),
                    command_timeout: Duration::from_millis(self.command_timeout_ms),
                    terminate_grace: Duration::from_millis(TERMINATE_GRACE_MS),
                    kill_grace: Duration::from_millis(KILL_GRACE_MS),
                },
                config,
            })
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn serve(args: super::RankServeArgs) -> Result<()> {
        let cfg: EndpointConfig = read_json(&args.config, MAX_CONFIG_BYTES)?;
        let spec = spec(&cfg.binding)?;
        let launch = ProcessLaunch::new(std::env::current_exe()?)
            .arg("__rank-worker")
            .arg("--max-frame-bytes")
            .arg(frame_limits(spec)?.max_frame_bytes.to_string())
            .arg("--io-timeout-ms")
            .arg(
                crate::commands::rank_worker::process::idle_timeout_ms(
                    cfg.command_timeout_ms,
                    spec.ranks,
                )?
                .to_string(),
            );
        let endpoint = cfg.endpoint(launch)?;
        let listener = NetworkEndpoint::<BootConfig>::bind(args.bind, args.allow_remote_bind)?;
        eprintln!(
            "rank-serve bound {}; one-shot mTLS; do not probe with TCP",
            listener.local_addr()?
        );
        endpoint.serve_once(listener)?;
        Ok(())
    }

    #[cfg(test)]
    #[path = "network_rank_integration.rs"]
    mod tests;
}
