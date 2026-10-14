#![cfg(unix)]
//! CPU-only orchestration evidence: actual TLS sockets and OS child processes.
//! These tests never claim CUDA fencing or multi-host performance validation.
#[path = "support/network/test_support.rs"]
mod support;
use ferrule_common::{
    ParallelGroupId, ParallelRankId, ParallelTopologyId, ParallelismPlan, ValidatedParallelTopology,
};
use ferrule_model::transformer::parallel::{
    TensorParallelLinearPartition, TensorParallelLinearPlan,
};
use ferrule_runtime as runtime;
use ferrule_runtime::SessionId;
use ferrule_runtime::parallel::collective::HostCollectiveLimits;
use ferrule_runtime::parallel::data::{
    CompletionOutcome, DataParallelConfig, DataParallelExecutor, ReplicaWorker, WorkRequest,
};
use ferrule_runtime::parallel::network::*;
use ferrule_runtime::parallel::tensor::{
    TensorCommand, TensorParallelError, TensorParallelExecutor, TensorWork,
};
use serde_json::json;
use std::sync::Arc;
use std::time::{Duration, Instant};
use support::*;

#[test]
fn real_child_ready_compute_fenced_failure_and_shutdown() {
    let creds = Credentials::new();
    let cfg = config(0, 1, LinearMode::Dp);
    let b = binding(0, 1, LinearMode::Dp);
    let running = Running::start(endpoint(&creds, b.clone(), cfg.clone(), options(1500)));
    let mut owner =
        Owner::connect(running.address, &creds.client_tls(), b, &cfg, options(1500)).unwrap();
    let out = owner.execute(tx(11), 9, &input()).unwrap();
    assert_eq!(
        serde_json::from_value::<Vec<f32>>(out["values"].clone()).unwrap(),
        oracle()
    );
    let pid = out["pid"].as_u64().unwrap() as u32;
    assert_ne!(pid, std::process::id());
    assert_eq!(out["transaction"], 11);
    assert_eq!(out["session"], 9);
    assert!(matches!(
        owner.execute(tx(12), 9, &json!({"fault":"fenced"})),
        Err(NetworkError::Fenced(_))
    ));
    assert_eq!(owner.execute(tx(13), 9, &input()).unwrap()["command"], 3);
    owner.shutdown().unwrap();
    running.finish().unwrap();
    absent(pid);
}

#[test]
fn two_real_ranks_use_production_dp_executor() {
    let creds = Credentials::new();
    let mut running = Vec::new();
    let mut placements = Vec::new();
    for rank in 0..2 {
        let cfg = config(rank, 2, LinearMode::Dp);
        let b = binding(rank, 2, LinearMode::Dp);
        let r = Running::start(endpoint(&creds, b.clone(), cfg.clone(), options(1500)));
        placements.push((r.address, b, cfg));
        running.push(r);
    }
    let tls = creds.client_tls();
    let mut pool = DataParallelExecutor::new(pool_config(), move |rank: ParallelRankId| {
        let (address, b, cfg) = &placements[rank.get() as usize];
        Owner::connect(*address, &tls, b.clone(), cfg, options(1500))
    })
    .unwrap();
    for rank in 0..2 {
        pool.try_submit_to(
            SessionId(rank as u64),
            ParallelRankId::new(rank),
            tx(rank as u64 + 1),
            input(),
        )
        .unwrap();
    }
    let end = Instant::now() + Duration::from_secs(5);
    let mut pids = Vec::new();
    while pids.len() < 2 {
        assert!(Instant::now() < end);
        if let Some(done) = pool.try_recv() {
            let CompletionOutcome::Success(out) = done.outcome else {
                panic!("unexpected DP outcome");
            };
            assert_eq!(
                serde_json::from_value::<Vec<f32>>(out["values"].clone()).unwrap(),
                oracle()
            );
            pids.push(out["pid"].as_u64().unwrap() as u32);
        } else {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    assert_ne!(pids[0], pids[1]);
    pool.shutdown().unwrap();
    for r in running {
        r.finish().unwrap();
    }
    for pid in pids {
        absent(pid);
    }
}
fn pool_config() -> DataParallelConfig {
    DataParallelConfig {
        replicas: 2,
        max_outstanding_per_replica: 1,
        session_capacity: 4,
    }
}
struct TensorAdapter {
    owner: Owner,
    fault: bool,
}
impl ReplicaWorker<TensorWork> for TensorAdapter {
    type Output = Vec<f32>;
    type Error = NetworkError;
    fn execute(&mut self, request: WorkRequest<TensorWork>) -> Result<Vec<f32>, NetworkError> {
        let (phase, values, rows) = match request.input.command {
            TensorCommand::Compute { input, rows } => ("compute", input.to_vec(), rows),
            TensorCommand::Apply { values, rows } => ("apply", values, rows),
        };
        let mut command = json!({"phase":phase,"values":values,"rows":rows});
        if self.fault {
            command["fault"] = json!("unknown");
        }
        let out = self.owner.execute_or_quarantine(
            request.input.transaction,
            request.session.0,
            &command,
        )?;
        assert_eq!(out["transaction"], request.input.transaction.get());
        Ok(serde_json::from_value(out["values"].clone()).unwrap())
    }
    fn shutdown(&mut self) -> Result<(), NetworkError> {
        self.owner.shutdown_or_quarantine()
    }
}
fn tensor_run(mode: LinearMode, fault: bool) {
    let creds = Credentials::new();
    let mut running = Vec::new();
    let mut placements = Vec::new();
    for rank in 0..2 {
        let cfg = config(rank, 2, mode);
        let b = binding(rank, 2, mode);
        let r = Running::start(endpoint(&creds, b.clone(), cfg.clone(), options(1500)));
        placements.push((r.address, b, cfg));
        running.push(r);
    }
    let tls = creds.client_tls();
    let topology = ValidatedParallelTopology::new(
        ParallelTopologyId::new(1),
        2,
        ParallelRankId::new(0),
        ParallelismPlan {
            tensor_parallel: 2,
            ..ParallelismPlan::default()
        },
    )
    .unwrap();
    let partition = if mode == LinearMode::TpColumn {
        TensorParallelLinearPartition::Column
    } else {
        TensorParallelLinearPartition::Row
    };
    let plan = TensorParallelLinearPlan::new(5, 3, 2, partition).unwrap();
    let mut tp = TensorParallelExecutor::new(
        topology,
        0,
        plan,
        pool_config(),
        move |rank| {
            let (address, b, cfg) = &placements[rank.local.get() as usize];
            Ok(TensorAdapter {
                owner: Owner::connect(*address, &tls, b.clone(), cfg, options(1500))?,
                fault: fault && rank.local.get() == 0,
            })
        },
        ParallelGroupId::new(1),
        HostCollectiveLimits {
            max_ranks: 2,
            max_elements_per_rank: 64,
            max_host_bytes: 4096,
        },
    )
    .unwrap();
    let result = tp.execute(
        tx(991),
        SessionId(100),
        Arc::from([1., 2., 3., 4., 5., 6.]),
        2,
    );
    if fault {
        assert!(matches!(
            result,
            Err(TensorParallelError::QuiescenceUnknown(_))
        ));
        assert!(matches!(
            tp.execute(
                tx(992),
                SessionId(102),
                Arc::from([1., 2., 3., 4., 5., 6.]),
                2
            ),
            Err(TensorParallelError::Quarantined)
        ));
        let _ = tp.shutdown();
    } else {
        let result = result.unwrap();
        assert_eq!(result.ranks.len(), 2);
        for (_, values) in result.ranks {
            assert_eq!(values, oracle());
        }
        tp.shutdown().unwrap();
    }
    for r in running {
        let result = r.finish();
        if !fault {
            result.unwrap();
        }
    }
}
#[test]
fn tp_column_ragged_compute_collective_apply() {
    tensor_run(LinearMode::TpColumn, false);
}
#[test]
fn tp_row_ragged_compute_collective_apply() {
    tensor_run(LinearMode::TpRow, false);
}
#[test]
fn tp_unknown_quarantines_without_publication_or_replay() {
    tensor_run(LinearMode::TpRow, true);
}

#[test]
fn oversized_local_rejection_does_not_consume_command() {
    let creds = Credentials::new();
    let cfg = config(0, 1, LinearMode::Dp);
    let b = binding(0, 1, LinearMode::Dp);
    let mut opts = options(1000);
    opts.limits.max_command_bytes = 128;
    let r = Running::start(endpoint(&creds, b.clone(), cfg.clone(), opts));
    let mut owner = Owner::connect(r.address, &creds.client_tls(), b, &cfg, opts).unwrap();
    assert!(matches!(
        owner.execute(tx(1), 1, &json!({"blob":"x".repeat(1024)})),
        Err(NetworkError::FrameTooLarge)
    ));
    assert_eq!(owner.state(), NetworkOwnerState::Ready);
    assert_eq!(owner.execute(tx(2), 1, &input()).unwrap()["command"], 1);
    owner.shutdown().unwrap();
    r.finish().unwrap();
}

#[test]
fn hung_exited_and_unknown_child_are_terminal_no_replay() {
    for fault in ["hang", "exit", "unknown"] {
        let creds = Credentials::new();
        let mut cfg = config(0, 1, LinearMode::Dp);
        let b = binding(0, 1, LinearMode::Dp);
        let ledger = creds.dir.path().join("ledger");
        let started = creds.dir.path().join("started");
        cfg["ledger"] = json!(ledger);
        cfg["started"] = json!(started);
        let r = Running::start(endpoint(&creds, b.clone(), cfg.clone(), options(200)));
        let mut owner =
            Owner::connect(r.address, &creds.client_tls(), b, &cfg, options(200)).unwrap();
        let pid = std::fs::read_to_string(&started).unwrap().parse().unwrap();
        let begin = Instant::now();
        assert!(matches!(
            owner.execute(tx(1), 1, &json!({"fault":fault})),
            Err(NetworkError::Unknown(_))
        ));
        assert!(begin.elapsed() < Duration::from_secs(2));
        assert_eq!(owner.state(), NetworkOwnerState::Unknown);
        assert!(matches!(
            owner.execute(tx(2), 1, &input()),
            Err(NetworkError::Unavailable)
        ));
        assert!(r.finish().is_err());
        absent(pid);
        assert_eq!(std::fs::read_to_string(ledger).unwrap().lines().count(), 1);
    }
}

#[test]
fn shutdown_and_destructor_hang_use_remote_deadline_not_local_30_seconds() {
    for fault in ["shutdown_hang", "drop_hang"] {
        let creds = Credentials::new();
        let mut cfg = config(0, 1, LinearMode::Dp);
        let b = binding(0, 1, LinearMode::Dp);
        let started = creds.dir.path().join("started");
        cfg["started"] = json!(started);
        cfg[fault] = json!(true);
        let endpoint = endpoint(&creds, b.clone(), cfg.clone(), options(200));
        assert_eq!(
            endpoint.local_options.command_timeout,
            Duration::from_secs(30)
        );
        let r = Running::start(endpoint);
        let mut owner =
            Owner::connect(r.address, &creds.client_tls(), b, &cfg, options(200)).unwrap();
        let pid = std::fs::read_to_string(started).unwrap().parse().unwrap();
        let begin = Instant::now();
        assert!(matches!(owner.shutdown(), Err(NetworkError::Unknown(_))));
        assert!(r.finish().is_err());
        assert!(begin.elapsed() < Duration::from_secs(2));
        absent(pid);
    }
}

#[test]
fn identity_boot_and_limit_mismatches_never_spawn_child() {
    for field in [
        "run",
        "controller",
        "node",
        "epoch",
        "owner",
        "rank",
        "mode",
        "device",
        "geometry",
        "limits",
        "boot",
    ] {
        let creds = Credentials::new();
        let b = binding(0, 2, LinearMode::Dp);
        let mut cfg = config(0, 2, LinearMode::Dp);
        let started = creds.dir.path().join("started");
        cfg["started"] = json!(started);
        let r = Running::start(endpoint(&creds, b.clone(), cfg.clone(), options(500)));
        let mut wrong = b;
        let mut opts = options(500);
        match field {
            "run" => wrong.run_id.push('x'),
            "controller" => wrong.controller_id.push('x'),
            "node" => wrong.node_id.push('x'),
            "epoch" => {
                wrong.identity.epoch =
                    ferrule_runtime::parallel::process::ProcessGroupEpoch::new(92).unwrap()
            }
            "owner" => {
                wrong.identity.owner_instance =
                    ferrule_runtime::parallel::process::ProcessOwnerInstanceId::new(99).unwrap()
            }
            "rank" => wrong.identity.rank = ParallelRankId::new(1),
            "mode" => wrong.target.mode = LinearMode::TpRow,
            "device" => wrong.target.device = 1,
            "geometry" => wrong.target.max_rows += 1,
            "limits" => opts.limits.max_error_bytes = 100,
            "boot" => cfg["fraction"] = json!(0.25),
            _ => unreachable!(),
        }
        assert!(
            Owner::connect(r.address, &creds.client_tls(), wrong, &cfg, opts).is_err(),
            "{field}"
        );
        assert!(r.finish().is_err());
        assert!(!started.exists(), "{field}");
    }
}

#[test]
fn ca_name_and_both_leaf_pins_are_mandatory() {
    for case in [
        "ca",
        "name",
        "server_pin",
        "client_pin",
        "valid_unapproved_client",
    ] {
        let mut creds = Credentials::new();
        let foreign = Credentials::new();
        let b = binding(0, 1, LinearMode::Dp);
        let mut cfg = config(0, 1, LinearMode::Dp);
        let started = creds.dir.path().join("started");
        cfg["started"] = json!(started);
        let mut name = "localhost";
        match case {
            "ca" => creds.client.ca = foreign.client.ca.clone(),
            "name" => name = "not-localhost.invalid",
            "server_pin" => creds.client.peer_leaf = creds.other_leaf.clone(),
            "client_pin" => creds.server.peer_leaf = creds.other_leaf.clone(),
            "valid_unapproved_client" => {
                creds.client.certificate_chain = creds.other_leaf.clone();
                creds.client.private_key = creds.dir.path().join("other.key");
            }
            _ => unreachable!(),
        }
        let r = Running::start(endpoint(&creds, b.clone(), cfg.clone(), options(500)));
        assert!(
            Owner::connect(
                r.address,
                &ClientTls::load(&creds.client, name).unwrap(),
                b,
                &cfg,
                options(500)
            )
            .is_err(),
            "{case}"
        );
        assert!(r.finish().is_err());
        assert!(!started.exists());
    }
}

#[test]
fn drop_ready_connection_cleans_child_without_claiming_shutdown() {
    let creds = Credentials::new();
    let b = binding(0, 1, LinearMode::Dp);
    let mut cfg = config(0, 1, LinearMode::Dp);
    let started = creds.dir.path().join("started");
    cfg["started"] = json!(started);
    let r = Running::start(endpoint(&creds, b.clone(), cfg.clone(), options(500)));
    let owner = Owner::connect(r.address, &creds.client_tls(), b, &cfg, options(500)).unwrap();
    let pid = std::fs::read_to_string(started).unwrap().parse().unwrap();
    drop(owner);
    assert!(r.finish().is_err());
    absent(pid);
}

#[test]
fn unauthenticated_handshake_stall_is_bounded_and_does_not_spawn() {
    let creds = Credentials::new();
    let b = binding(0, 1, LinearMode::Dp);
    let mut cfg = config(0, 1, LinearMode::Dp);
    let started = creds.dir.path().join("started");
    cfg["started"] = json!(started);
    let mut opts = options(200);
    opts.startup_timeout = Duration::from_millis(200);
    let r = Running::start(endpoint(&creds, b, cfg, opts));
    let socket = std::net::TcpStream::connect(r.address).unwrap();
    let start = Instant::now();
    assert!(r.finish().is_err());
    assert!(start.elapsed() < Duration::from_secs(2));
    assert!(!started.exists());
    drop(socket);
}

#[test]
fn controller_two_endpoint_processes_and_two_rank_children() {
    use std::process::Command;
    struct EndpointProcess(std::process::Child);
    impl Drop for EndpointProcess {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    let creds = Credentials::new();
    let exe = child();
    let mut processes = Vec::new();
    let mut owners = Vec::new();
    let mut pids = Vec::new();
    for rank in 0..2 {
        let b = binding(rank, 2, LinearMode::Dp);
        let cfg = config(rank, 2, LinearMode::Dp);
        let ready = creds.dir.path().join(format!("endpoint-{rank}-ready"));
        let path = creds.dir.path().join(format!("endpoint-{rank}.json"));
        std::fs::write(
            &path,
            serde_json::to_vec(
                &json!({"binding":b,"tls":creds.server,"boot":cfg,"ready_file":ready}),
            )
            .unwrap(),
        )
        .unwrap();
        let process = EndpointProcess(
            Command::new(&exe)
                .arg("endpoint")
                .arg(path)
                .spawn()
                .unwrap(),
        );
        let end = Instant::now() + Duration::from_secs(4);
        let address = loop {
            if let Ok(text) = std::fs::read_to_string(&ready)
                && let Ok(address) = text.parse()
            {
                break address;
            }
            assert!(Instant::now() < end, "endpoint did not bind");
            std::thread::sleep(Duration::from_millis(5));
        };
        let mut owner =
            Owner::connect(address, &creds.client_tls(), b, &cfg, options(500)).unwrap();
        let output = owner.execute(tx(rank as u64 + 1), 10, &input()).unwrap();
        let pid = output["pid"].as_u64().unwrap() as u32;
        assert_ne!(pid, process.0.id());
        assert_ne!(pid, std::process::id());
        assert_eq!(
            serde_json::from_value::<Vec<f32>>(output["values"].clone()).unwrap(),
            oracle()
        );
        pids.push(pid);
        processes.push(process);
        owners.push(owner);
    }
    assert_ne!(pids[0], pids[1]);
    assert_ne!(processes[0].0.id(), processes[1].0.id());
    for owner in &mut owners {
        owner.shutdown().unwrap();
    }
    for process in &mut processes {
        let end = Instant::now() + Duration::from_secs(3);
        loop {
            if let Some(status) = process.0.try_wait().unwrap() {
                assert!(status.success());
                break;
            }
            assert!(Instant::now() < end);
            std::thread::sleep(Duration::from_millis(5));
        }
    }
    for pid in pids {
        absent(pid);
    }
}

#[test]
fn shutdown_until_propagates_shorter_budget() {
    let creds = Credentials::new();
    let b = binding(0, 1, LinearMode::Dp);
    let mut cfg = config(0, 1, LinearMode::Dp);
    cfg["shutdown_hang"] = json!(true);
    let r = Running::start(endpoint(&creds, b.clone(), cfg.clone(), options(5000)));
    let mut owner = Owner::connect(r.address, &creds.client_tls(), b, &cfg, options(5000)).unwrap();
    let start = Instant::now();
    assert!(matches!(
        owner.shutdown_until(start + Duration::from_millis(150)),
        Err(NetworkError::Unknown(_))
    ));
    assert!(r.finish().is_err());
    assert!(start.elapsed() < Duration::from_secs(2));
}

#[test]
fn dp_unknown_closes_admission_instead_of_returning_business_failure() {
    let creds = Credentials::new();
    let b = binding(0, 1, LinearMode::Dp);
    let cfg = config(0, 1, LinearMode::Dp);
    let r = Running::start(endpoint(&creds, b.clone(), cfg.clone(), options(500)));
    let address = r.address;
    let tls = creds.client_tls();
    let mut pool = DataParallelExecutor::new(
        DataParallelConfig {
            replicas: 1,
            max_outstanding_per_replica: 1,
            session_capacity: 1,
        },
        move |_| Owner::connect(address, &tls, b, &cfg, options(500)),
    )
    .unwrap();
    pool.try_submit(SessionId(1), tx(1), json!({"fault":"unknown"}))
        .unwrap();
    let end = Instant::now() + Duration::from_secs(3);
    loop {
        assert!(Instant::now() < end);
        if let Some(done) = pool.try_recv() {
            assert!(matches!(done.outcome, CompletionOutcome::QuiescenceUnknown));
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(pool.try_submit(SessionId(1), tx(2), input()).is_err());
    assert!(pool.shutdown().is_err());
    assert!(r.finish().is_err());
}

#[test]
fn shutdown_is_also_capped_by_shorter_local_process_budget() {
    let creds = Credentials::new();
    let b = binding(0, 1, LinearMode::Dp);
    let mut cfg = config(0, 1, LinearMode::Dp);
    cfg["shutdown_hang"] = json!(true);
    let mut e = endpoint(&creds, b.clone(), cfg.clone(), options(5000));
    e.local_options.command_timeout = Duration::from_millis(120);
    let r = Running::start(e);
    let mut owner = Owner::connect(r.address, &creds.client_tls(), b, &cfg, options(5000)).unwrap();
    let start = Instant::now();
    assert!(matches!(owner.shutdown(), Err(NetworkError::Unknown(_))));
    assert!(r.finish().is_err());
    assert!(start.elapsed() < Duration::from_secs(2));
}

#[test]
fn real_child_hung_boot_cannot_produce_network_ready() {
    let creds = Credentials::new();
    let b = binding(0, 1, LinearMode::Dp);
    let mut cfg = config(0, 1, LinearMode::Dp);
    cfg["startup_hang"] = json!(true);
    let started = creds.dir.path().join("started");
    cfg["started"] = json!(started);
    let mut opts = options(500);
    opts.startup_timeout = Duration::from_millis(200);
    let r = Running::start(endpoint(&creds, b.clone(), cfg.clone(), opts));
    let start = Instant::now();
    assert!(matches!(
        Owner::connect(r.address, &creds.client_tls(), b, &cfg, opts),
        Err(NetworkError::Unknown(_))
    ));
    assert!(r.finish().is_err());
    assert!(start.elapsed() < Duration::from_secs(2));
    absent(std::fs::read_to_string(started).unwrap().parse().unwrap());
}
