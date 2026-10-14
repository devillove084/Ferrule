//! TEST ONLY identities and artifacts. Private keys are ephemeral, owner-only,
//! and never checked in. Shared by network unit/integration tests.
#![allow(dead_code)]
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use super::runtime::parallel::network::*;
use super::runtime::parallel::process::*;
use ferrule_common::ParallelRankId;
use rcgen::{
    BasicConstraints, CertificateParams, CertifiedIssuer, DnType, ExtendedKeyUsagePurpose, IsCa,
    KeyPair, KeyUsagePurpose,
};
use serde_json::{Value, json};
use tempfile::TempDir;

pub struct Credentials {
    pub dir: TempDir,
    pub client: TlsFiles,
    pub server: TlsFiles,
    pub other_leaf: PathBuf,
}
impl Credentials {
    pub fn new() -> Self {
        let dir = tempfile::Builder::new()
            .prefix("ferrule-TEST-ONLY-network-")
            .tempdir()
            .unwrap();
        let mut ca = CertificateParams::new(vec![]).unwrap();
        ca.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        ca.key_usages = vec![KeyUsagePurpose::KeyCertSign, KeyUsagePurpose::CrlSign];
        ca.distinguished_name
            .push(DnType::CommonName, "Ferrule TEST ONLY ephemeral CA");
        let ca = CertifiedIssuer::self_signed(ca, KeyPair::generate().unwrap()).unwrap();
        fs::write(dir.path().join("ca.pem"), ca.pem()).unwrap();
        for name in ["server", "client", "other"] {
            let mut params = CertificateParams::new(vec!["localhost".into()]).unwrap();
            params
                .distinguished_name
                .push(DnType::CommonName, format!("Ferrule TEST ONLY {name}"));
            params.key_usages = vec![KeyUsagePurpose::DigitalSignature];
            params.extended_key_usages = vec![if name == "server" {
                ExtendedKeyUsagePurpose::ServerAuth
            } else {
                ExtendedKeyUsagePurpose::ClientAuth
            }];
            let key = KeyPair::generate().unwrap();
            let cert = params.signed_by(&key, &ca).unwrap();
            fs::write(dir.path().join(format!("{name}.pem")), cert.pem()).unwrap();
            let mut f = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(dir.path().join(format!("{name}.key")))
                .unwrap();
            f.write_all(key.serialize_pem().as_bytes()).unwrap();
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
            other_leaf: dir.path().join("other.pem"),
            dir,
        }
    }
    pub fn client_tls(&self) -> ClientTls {
        ClientTls::load(&self.client, "localhost").unwrap()
    }
    pub fn server_tls(&self) -> ServerTls {
        ServerTls::load(&self.server).unwrap()
    }
}

pub fn child() -> PathBuf {
    static CHILD: OnceLock<PathBuf> = OnceLock::new();
    CHILD
        .get_or_init(|| {
            // CI/nextest should explicitly provide its prebuilt current-checkout
            // artifact. No ancestry search or fallback build after a bad override.
            if let Some(path) = std::env::var_os("FERRULE_NETWORK_TEST_CHILD") {
                let path = PathBuf::from(path);
                assert!(
                    path.is_absolute() && path.is_file(),
                    "invalid explicit network child artifact"
                );
                verify_artifact(&path);
                return path;
            }
            let output = tempfile::NamedTempFile::new().unwrap();
            let mut command = Command::new(env!("CARGO"));
            command
                .current_dir(env!("CARGO_MANIFEST_DIR"))
                .args([
                    "build",
                    "--locked",
                    "--no-default-features",
                    "-p",
                    "ferrule-runtime",
                    "--example",
                    "network_rank_child",
                    "--message-format=json",
                ])
                .stdout(Stdio::from(output.reopen().unwrap()))
                .stderr(Stdio::inherit());
            if cfg!(feature = "cpu-reference") {
                command.args(["--features", "cpu-reference"]);
            }
            let mut build = command.spawn().unwrap();
            let end = Instant::now() + Duration::from_secs(90);
            loop {
                if let Some(status) = build.try_wait().unwrap() {
                    assert!(status.success(), "network child build failed");
                    break;
                }
                if Instant::now() >= end {
                    let _ = build.kill();
                    let _ = build.wait();
                    panic!("network child build exceeded 90s; prebuild for nextest");
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            let messages = fs::read_to_string(output.path()).unwrap();
            let artifacts: Vec<PathBuf> = messages
                .lines()
                .filter_map(|line| {
                    let v: Value = serde_json::from_str(line).ok()?;
                    if v["reason"] == "compiler-artifact"
                        && v["target"]["name"] == "network_rank_child"
                    {
                        v["executable"].as_str().map(PathBuf::from)
                    } else {
                        None
                    }
                })
                .collect();
            assert_eq!(artifacts.len(), 1);
            artifacts[0].clone()
        })
        .clone()
}
pub fn identity(rank: u32) -> ProcessIdentity {
    ProcessIdentity::new(
        ProcessGroupEpoch::new(91).unwrap(),
        ParallelRankId::new(rank),
        ProcessOwnerInstanceId::new(rank as u64 + 10).unwrap(),
    )
}
pub fn binding(rank: u32, world: u32, mode: LinearMode) -> NetworkBinding {
    NetworkBinding {
        run_id: "TEST-ONLY-run".into(),
        controller_id: "TEST-ONLY-controller".into(),
        node_id: format!("TEST-ONLY-node-{rank}"),
        identity: identity(rank),
        target: LinearTarget {
            mode,
            world,
            device: 0,
            in_features: 3,
            out_features: 5,
            max_rows: 4,
        },
    }
}
pub fn config(rank: u32, world: u32, mode: LinearMode) -> Value {
    json!({"rank":rank, "world":world, "mode":mode, "in_features":3, "out_features":5, "test_identity":"TEST ONLY CPU linear", "fraction":0.125})
}
pub fn options(ms: u64) -> NetworkOptions {
    NetworkOptions {
        startup_timeout: Duration::from_secs(3),
        command_timeout: Duration::from_millis(ms),
        ..NetworkOptions::default()
    }
}
pub fn endpoint(
    creds: &Credentials,
    binding: NetworkBinding,
    config: Value,
    options: NetworkOptions,
) -> NetworkEndpoint<Value> {
    NetworkEndpoint {
        local_identity: identity(binding.identity.rank.get()),
        binding,
        config,
        options,
        tls: creds.server_tls(),
        launch: ProcessLaunch::new(child()),
        local_options: ProcessOwnerConfig {
            command_timeout: Duration::from_secs(30),
            terminate_grace: Duration::from_millis(10),
            kill_grace: Duration::from_millis(200),
            ..ProcessOwnerConfig::default()
        },
    }
}
pub struct Running {
    pub address: std::net::SocketAddr,
    rx: std::sync::mpsc::Receiver<Result<(), NetworkError>>,
    thread: std::thread::JoinHandle<()>,
}
impl Running {
    pub fn start(endpoint: NetworkEndpoint<Value>) -> Self {
        let listener =
            NetworkEndpoint::<Value>::bind("127.0.0.1:0".parse().unwrap(), false).unwrap();
        let address = listener.local_addr().unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        let thread = std::thread::spawn(move || {
            let _ = tx.send(endpoint.serve_once(listener));
        });
        Self {
            address,
            rx,
            thread,
        }
    }
    pub fn finish(self) -> Result<(), NetworkError> {
        let result = self
            .rx
            .recv_timeout(Duration::from_secs(6))
            .expect("endpoint exceeded external test deadline");
        self.thread.join().unwrap();
        result
    }
}
pub type Owner = NetworkRankOwner<Value, Value, Value>;
pub fn input() -> Value {
    json!({"phase":"compute", "rows":2, "values":[1.0,2.0,3.0,4.0,5.0,6.0]})
}
pub fn tx(n: u64) -> ferrule_common::execution::ExecutionTransactionId {
    ferrule_common::execution::ExecutionTransactionId::new(n).unwrap()
}
pub fn oracle() -> Vec<f32> {
    (0..2)
        .flat_map(|row| {
            (0..5).map(move |col| {
                (0..3)
                    .map(|k| (row * 3 + k + 1) as f32 * (col * 3 + k + 1) as f32 / 8.0)
                    .sum()
            })
        })
        .collect()
}
pub fn absent(pid: u32) {
    let end = Instant::now() + Duration::from_secs(2);
    loop {
        #[allow(unsafe_code)]
        let result = unsafe { libc::kill(pid as i32, 0) };
        if result < 0 && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH) {
            break;
        }
        assert!(Instant::now() < end, "test-owned child still present");
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[path = "test_artifact.rs"]
mod test_artifact;
fn verify_artifact(path: &std::path::Path) {
    let stdout = tempfile::NamedTempFile::new().unwrap();
    let mut probe = Command::new(path)
        .arg("artifact")
        .stdout(Stdio::from(stdout.reopen().unwrap()))
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let end = Instant::now() + Duration::from_secs(2);
    loop {
        if let Some(status) = probe.try_wait().unwrap() {
            assert!(status.success(), "invalid network child artifact");
            break;
        }
        if Instant::now() >= end {
            let _ = probe.kill();
            let _ = probe.wait();
            panic!("network artifact identity probe timed out");
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(
        fs::read_to_string(stdout.path()).unwrap().trim(),
        test_artifact::fingerprint(),
        "stale or foreign explicit network child artifact; no Cargo fallback"
    );
}
