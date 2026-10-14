use super::*;
use crate::args::{Cli, Command};
use clap::Parser;
use ferrule_common::{
    ParallelGroupId, ParallelRankId, ParallelTopologyId, ParallelismPlan, ValidatedParallelTopology,
};
use ferrule_model::transformer::parallel::{
    TensorParallelLinearPartition, TensorParallelLinearPlan,
};
use ferrule_runtime::parallel::collective::HostCollectiveLimits;
use ferrule_runtime::parallel::data::{
    CompletionOutcome, DataParallelConfig, DataParallelExecutor,
};
use ferrule_runtime::parallel::tensor::TensorParallelExecutor;
use ferrule_runtime::{ExecutionTransactionId, SessionId};
use serde_json::{Value, json};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::time::Instant;

struct Credentials {
    dir: tempfile::TempDir,
    client: TlsFiles,
    server: TlsFiles,
}
impl Credentials {
    fn new() -> Self {
        use rcgen::{
            BasicConstraints, CertificateParams, CertifiedIssuer, ExtendedKeyUsagePurpose, IsCa,
            KeyPair, KeyUsagePurpose,
        };
        let dir = tempfile::tempdir().unwrap();
        let mut ca = CertificateParams::new(vec![]).unwrap();
        ca.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        ca.key_usages = vec![KeyUsagePurpose::KeyCertSign, KeyUsagePurpose::CrlSign];
        let ca = CertifiedIssuer::self_signed(ca, KeyPair::generate().unwrap()).unwrap();
        std::fs::write(dir.path().join("ca.pem"), ca.pem()).unwrap();
        for name in ["server", "client"] {
            let mut params = CertificateParams::new(vec!["localhost".into()]).unwrap();
            params.key_usages = vec![KeyUsagePurpose::DigitalSignature];
            params.extended_key_usages = vec![if name == "server" {
                ExtendedKeyUsagePurpose::ServerAuth
            } else {
                ExtendedKeyUsagePurpose::ClientAuth
            }];
            let key = KeyPair::generate().unwrap();
            let cert = params.signed_by(&key, &ca).unwrap();
            std::fs::write(dir.path().join(format!("{name}.pem")), cert.pem()).unwrap();
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(dir.path().join(format!("{name}.key")))
                .unwrap();
            file.write_all(key.serialize_pem().as_bytes()).unwrap();
        }
        let files = |own: &str, peer: &str| TlsFiles {
            ca: dir.path().join("ca.pem"),
            certificate_chain: dir.path().join(format!("{own}.pem")),
            private_key: dir.path().join(format!("{own}.key")),
            peer_leaf: dir.path().join(format!("{peer}.pem")),
        };
        Self {
            client: files("client", "server"),
            server: files("server", "client"),
            dir,
        }
    }
}
fn child() -> PathBuf {
    static CHILD: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    CHILD
        .get_or_init(|| {
            if let Some(path) = std::env::var_os("FERRULE_CLI_NETWORK_TEST_CHILD") {
                let path = PathBuf::from(path);
                assert!(path.is_absolute() && path.is_file());
                return path;
            }
            let log = tempfile::NamedTempFile::new().unwrap();
            let mut command = std::process::Command::new(env!("CARGO"));
            command
                .current_dir(env!("CARGO_MANIFEST_DIR"))
                .args([
                    "build",
                    "--locked",
                    "-p",
                    "ferrule-cli",
                    "--example",
                    "network_linear_test_child",
                    "--message-format=json",
                ])
                .stdout(std::process::Stdio::from(log.reopen().unwrap()))
                .stderr(std::process::Stdio::inherit());
            if cfg!(feature = "cpu-reference") {
                command.args(["--features", "cpu-reference"]);
            }
            let mut build = command.spawn().unwrap();
            let end = Instant::now() + Duration::from_secs(90);
            loop {
                if let Some(status) = build.try_wait().unwrap() {
                    assert!(status.success());
                    break;
                }
                if Instant::now() >= end {
                    let _ = build.kill();
                    let _ = build.wait();
                    panic!("prebuild network_linear_test_child: build exceeded 90s");
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            let text = std::fs::read_to_string(log.path()).unwrap();
            text.lines()
                .filter_map(|line| serde_json::from_str::<Value>(line).ok())
                .find_map(|v| {
                    (v["reason"] == "compiler-artifact"
                        && v["target"]["name"] == "network_linear_test_child")
                        .then(|| v["executable"].as_str().map(PathBuf::from))
                        .flatten()
                })
                .unwrap()
        })
        .clone()
}
fn fixture() -> Fixture {
    Fixture {
        rows: 2,
        in_features: 3,
        out_features: 5,
        weight: (1..=15).map(|n| n as f32 / 8.0).collect(),
        input: (1..=6).map(|n| n as f32).collect(),
    }
}
fn args(mode: Mode) -> BenchParallelArgs {
    let name = match mode {
        Mode::Dp => "dp",
        Mode::TpColumn => "tp-column",
        Mode::TpRow => "tp-row",
    };
    let cli = Cli::try_parse_from([
        "ferrule",
        "bench-parallel",
        "--mode",
        name,
        "--ranks",
        "2",
        "--fixture",
        "unused.json",
        "--remote-manifest",
        "manifest.json",
        "--rank-timeout-ms",
        "1500",
    ])
    .unwrap();
    let Command::BenchParallel(args) = cli.command else {
        unreachable!()
    };
    args
}
fn remote(creds: &Credentials, rank: u32, mode: Mode) -> RemoteRank {
    let linear = match mode {
        Mode::Dp => LinearMode::Dp,
        Mode::TpColumn => LinearMode::TpColumn,
        Mode::TpRow => LinearMode::TpRow,
    };
    let mut identity = identity(rank).unwrap();
    identity.epoch = ferrule_runtime::parallel::process::ProcessGroupEpoch::new(991).unwrap();
    let binding = NetworkBinding {
        run_id: "TEST-run".into(),
        controller_id: "TEST-controller".into(),
        node_id: format!("node-{rank}"),
        identity,
        target: ferrule_runtime::parallel::network::LinearTarget {
            mode: linear,
            world: 2,
            device: 0,
            in_features: 3,
            out_features: 5,
            max_rows: 2,
        },
    };
    RemoteRank {
        address: format!("127.0.0.1:{}", 19000 + rank).parse().unwrap(),
        server_name: "localhost".into(),
        tls: creds.client.clone(),
        limits: frame_limits(spec(&binding).unwrap()).unwrap(),
        binding,
        startup_timeout_ms: 3000,
        command_timeout_ms: 1500,
    }
}
struct Running {
    receiver: std::sync::mpsc::Receiver<Result<(), NetworkError>>,
    thread: std::thread::JoinHandle<()>,
}
impl Running {
    fn finish(self, success: bool) {
        let result = self
            .receiver
            .recv_timeout(Duration::from_secs(6))
            .expect("endpoint deadline exceeded");
        self.thread.join().unwrap();
        if success {
            result.unwrap();
        }
    }
}
fn start(creds: &Credentials, mode: Mode, fault: &str) -> (Vec<PreparedRank>, Vec<Running>) {
    let f = fixture();
    let file = creds.dir.path().join("fixture.json");
    std::fs::write(&file, serde_json::to_vec(&json!({"rows":f.rows,"in_features":f.in_features,"out_features":f.out_features,"input":f.input,"weight":f.weight})).unwrap()).unwrap();
    let mut manifest = Manifest {
        ranks: (0..2).map(|r| remote(creds, r, mode)).collect(),
    };
    let mut running = Vec::new();
    for rank in &mut manifest.ranks {
        let cfg = EndpointConfig {
            fixture: file.clone(),
            tls: creds.server.clone(),
            binding: rank.binding.clone(),
            startup_timeout_ms: rank.startup_timeout_ms,
            command_timeout_ms: rank.command_timeout_ms,
            limits: rank.limits,
        };
        let endpoint = cfg
            .endpoint(ProcessLaunch::new(child()).arg(fault))
            .unwrap();
        let listener =
            NetworkEndpoint::<BootConfig>::bind("127.0.0.1:0".parse().unwrap(), false).unwrap();
        rank.address = listener.local_addr().unwrap();
        let (send, receiver) = std::sync::mpsc::channel();
        let thread = std::thread::spawn(move || {
            let _ = send.send(endpoint.serve_once(listener));
        });
        running.push(Running { receiver, thread });
    }
    manifest.validate(&args(mode), &f).unwrap();
    (manifest.prepare(&f).unwrap(), running)
}
fn oracle() -> Vec<f32> {
    let f = fixture();
    (0..2)
        .flat_map(|row| {
            (0..5).map(move |col| {
                (0..3)
                    .map(|k| (row * 3 + k + 1) as f32 * (col * 3 + k + 1) as f32 / 8.)
                    .sum()
            })
        })
        .collect::<Vec<_>>()
        .into_iter()
        .take(f.rows * f.out_features)
        .collect()
}
fn pool_config() -> DataParallelConfig {
    DataParallelConfig {
        replicas: 2,
        max_outstanding_per_replica: 1,
        session_capacity: 4,
    }
}
#[test]
fn cli_network_dp_uses_real_tls_processes_and_outer_transactions() {
    let creds = Credentials::new();
    let (prepared, running) = start(&creds, Mode::Dp, "");
    let mut pool = DataParallelExecutor::new(pool_config(), move |rank: ParallelRankId| {
        prepared[rank.get() as usize].connect()
    })
    .unwrap();
    for rank in 0..2 {
        pool.try_submit_to(
            SessionId(rank as u64 + 17),
            ParallelRankId::new(rank),
            ExecutionTransactionId::new(rank as u64 + 900).unwrap(),
            Arc::<[f32]>::from(fixture().input),
        )
        .unwrap();
    }
    let end = Instant::now() + Duration::from_secs(5);
    let mut count = 0;
    while count < 2 {
        assert!(Instant::now() < end);
        if let Some(done) = pool.try_recv() {
            let CompletionOutcome::Success(values) = done.outcome else {
                panic!("DP failed")
            };
            assert_eq!(values, oracle());
            count += 1;
        } else {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    pool.shutdown().unwrap();
    for r in running {
        r.finish(true);
    }
}
fn tensor(mode: Mode, fault: &str) {
    let creds = Credentials::new();
    let (prepared, running) = start(&creds, mode, fault);
    let topology = ValidatedParallelTopology::new(
        ParallelTopologyId::new(1),
        2,
        ParallelRankId::new(0),
        ParallelismPlan {
            tensor_parallel: 2,
            ..Default::default()
        },
    )
    .unwrap();
    let plan = TensorParallelLinearPlan::new(
        5,
        3,
        2,
        if mode == Mode::TpRow {
            TensorParallelLinearPartition::Row
        } else {
            TensorParallelLinearPartition::Column
        },
    )
    .unwrap();
    let mut tp = TensorParallelExecutor::<Proxy>::new(
        topology,
        0,
        plan,
        pool_config(),
        move |rank| prepared[rank.local.get() as usize].connect(),
        ParallelGroupId::new(1),
        HostCollectiveLimits {
            max_ranks: 2,
            max_elements_per_rank: 64,
            max_host_bytes: 4096,
        },
    )
    .unwrap();
    let result = tp.execute(
        ExecutionTransactionId::new(991).unwrap(),
        SessionId(53),
        Arc::from(fixture().input),
        2,
    );
    if fault.is_empty() {
        let result = result.unwrap();
        for (_, values) in result.ranks {
            assert_eq!(values, oracle());
        }
        tp.shutdown().unwrap();
    } else {
        assert!(result.is_err());
        assert!(tp.is_quarantined());
        assert!(
            tp.execute(
                ExecutionTransactionId::new(992).unwrap(),
                SessionId(54),
                Arc::from(fixture().input),
                2
            )
            .is_err()
        );
        let _ = tp.shutdown();
    }
    for r in running {
        r.finish(fault.is_empty());
    }
}
#[test]
fn cli_network_tp_column_preserves_outer_transaction_through_apply() {
    tensor(Mode::TpColumn, "");
}
#[test]
fn cli_network_tp_row_preserves_outer_transaction_through_apply() {
    tensor(Mode::TpRow, "");
}
#[test]
fn cli_network_unknown_and_bad_success_quarantine_without_replay() {
    for fault in ["unknown", "bad_reply", "exit"] {
        tensor(Mode::TpRow, fault);
    }
}
#[test]
fn cli_network_manifest_strict_policy_and_node_local_devices() {
    let creds = Credentials::new();
    let good = vec![remote(&creds, 0, Mode::Dp), remote(&creds, 1, Mode::Dp)];
    Manifest {
        ranks: good.clone(),
    }
    .validate(&args(Mode::Dp), &fixture())
    .unwrap();
    for case in 0..8 {
        let mut m = Manifest {
            ranks: good.clone(),
        };
        match case {
            0 => {
                m.ranks.pop();
            }
            1 => m.ranks[1].binding.identity.rank = ParallelRankId::new(0),
            2 => m.ranks[1].binding.node_id = "node-0".into(),
            3 => m.ranks[1].binding.target.in_features = 4,
            4 => m.ranks[1].binding.run_id = "different".into(),
            5 => {
                m.ranks[1].binding.identity.owner_instance =
                    m.ranks[0].binding.identity.owner_instance
            }
            6 => m.ranks[1].command_timeout_ms = 0,
            _ => m.ranks[1].limits.max_config_bytes = 1,
        };
        assert!(
            m.validate(&args(Mode::Dp), &fixture()).is_err(),
            "case {case}"
        );
    }
    assert!(serde_json::from_str::<Manifest>(r#"{"ranks":[],"unknown":0}"#).is_err());
    assert!(
        serde_json::from_str::<EndpointConfig>(r#"{"fixture":"x","executable":"untrusted"}"#)
            .is_err()
    );
    assert!(NetworkEndpoint::<BootConfig>::bind("0.0.0.0:0".parse().unwrap(), false).is_err());
}
#[test]
fn cli_network_wrong_pin_and_startup_unknown_are_not_ordinary_factory_errors() {
    let creds = Credentials::new();
    let (mut prepared, running) = start(&creds, Mode::Dp, "");
    let mut bad_files = creds.client.clone();
    bad_files.peer_leaf = creds.client.certificate_chain.clone();
    prepared[0].tls = ClientTls::load(&bad_files, "localhost").unwrap();
    assert!(prepared[0].connect().is_err());
    let b = &prepared[1].boot;
    let mut weight = b.weight.clone();
    weight[0] += 1.;
    prepared[1].boot = Arc::new(BootConfig {
        spec: b.spec,
        weight,
    });
    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| prepared[1].connect()));
    let panic = failed
        .err()
        .expect("post-Hello failure must remain Unknown");
    assert_eq!(
        panic.downcast_ref::<PanicQuiescence>(),
        Some(&PanicQuiescence::Unknown)
    );
    for r in running {
        r.finish(false);
    }
}

#[test]
fn checked_in_network_json_templates_validate_with_ephemeral_tls_credentials() {
    let creds = Credentials::new();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/network");
    let fixture: Fixture = read_json(&root.join("linear.json"), MAX_FIXTURE_BYTES).unwrap();
    let mut manifest = Manifest::load(&root.join("manifest.json")).unwrap();
    let mut endpoint: EndpointConfig =
        read_json(&root.join("endpoint.json"), MAX_CONFIG_BYTES).unwrap();
    assert_eq!(endpoint.binding, manifest.ranks[0].binding);
    assert_eq!(endpoint.limits, manifest.ranks[0].limits);
    assert_eq!(
        endpoint.command_timeout_ms,
        manifest.ranks[0].command_timeout_ms
    );
    let mut options = args(Mode::TpColumn);
    options.rank_timeout_ms = 30000;
    manifest.validate(&options, &fixture).unwrap();
    for rank in &mut manifest.ranks {
        rank.tls = creds.client.clone();
    }
    assert_eq!(manifest.prepare(&fixture).unwrap().len(), 2);
    endpoint.fixture = root.join("linear.json");
    endpoint.tls = creds.server.clone();
    let ep = endpoint
        .endpoint(ProcessLaunch::new(std::env::current_exe().unwrap()))
        .unwrap();
    assert_eq!(ep.config.weight, fixture.weight);
    assert_eq!(ep.config.spec.rows, fixture.rows);
    assert_eq!(ep.config.spec.rank, 0);
}
