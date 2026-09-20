#![cfg(unix)]
//! Real OS children exercising the production owner/supervisor/child protocol.
//! Build the test-only fixture with:
//! cargo build --offline --locked -p ferrule-runtime --example process_rank_child

use std::os::unix::process::ExitStatusExt;

use std::time::{Duration, Instant};

use ferrule_common::ParallelRankId;
use ferrule_runtime::ExecutionTransactionId;
use ferrule_runtime::parallel::process::*;
use serde_json::{Value, json};

type Owner = ProcessRankOwner<Value, Value, Value>;
type Supervisor = ProcessRankSupervisor<Value, Value, Value>;

fn launch(mode: &str) -> ProcessLaunch {
    let executable = std::env::current_exe().unwrap();
    // Covers both deps/<test> and Cargo's build/<crate>/<hash>/out layout,
    // including a caller-selected target directory or target triple.
    let fixture = executable
        .ancestors()
        .map(|directory| directory.join("examples/process_rank_child"))
        .find(|candidate| candidate.is_file())
        .expect(
            "build test fixture first: cargo build -p ferrule-runtime --example process_rank_child",
        );
    ProcessLaunch::new(fixture).arg(mode)
}
fn identity(epoch: u64, owner: u64) -> ProcessIdentity {
    ProcessIdentity::new(
        ProcessGroupEpoch::new(epoch).unwrap(),
        ParallelRankId::new(0),
        ProcessOwnerInstanceId::new(owner).unwrap(),
    )
}
fn tx(value: u64) -> ExecutionTransactionId {
    ExecutionTransactionId::new(value).unwrap()
}
fn options() -> ProcessOwnerConfig {
    ProcessOwnerConfig {
        startup_timeout: Duration::from_secs(2),
        command_timeout: Duration::from_millis(150),
        terminate_grace: Duration::from_millis(30),
        kill_grace: Duration::from_secs(1),
        ..ProcessOwnerConfig::default()
    }
}
fn spawn(mode: &str) -> Owner {
    Owner::spawn(launch(mode), identity(1, 1), &json!({}), options()).unwrap()
}
fn lost(error: &ProcessError, cause: UnknownCause) {
    match error {
        ProcessError::QuiescenceUnknown {
            cause: actual,
            termination,
            ..
        } => {
            assert_eq!(*actual, cause, "{error:?}");
            assert_eq!(termination.reap, ReapOutcome::Reaped, "{error:?}");
            assert!(termination.signal_errors.is_empty());
        }
        _ => panic!("lost command was not Unknown: {error:?}"),
    }
}

#[test]
fn ready_follows_real_non_send_initialization_and_echo_is_correlated() {
    let start = Instant::now();
    let mut owner = Owner::spawn(
        launch("normal"),
        identity(7, 9),
        &json!({"bias":10,"delay_ms":60}),
        options(),
    )
    .unwrap();
    assert!(start.elapsed() >= Duration::from_millis(60));
    assert_eq!(owner.state(), ProcessOwnerState::Ready);
    let pid = owner.pid().unwrap();
    assert_ne!(pid, std::process::id());
    for command in 1..=3 {
        let result = owner
            .execute(tx(40 + command), 123, &json!({"value":32}))
            .unwrap();
        assert_eq!(result["value"], 42);
        assert_eq!(result["pid"], pid);
        assert_eq!(result["command"], command);
        assert_eq!(result["transaction"], 40 + command);
        assert_eq!(result["session"], 123);
        assert_eq!(result["epoch"], 7);
        assert_eq!(result["owner"], 9);
    }
    assert_eq!(owner.shutdown().unwrap().reap, ReapOutcome::Reaped);
    assert!(owner.exit_status().unwrap().success());
    assert!(owner.pid().is_none());
    assert_eq!(owner.shutdown().unwrap().reap, ReapOutcome::Reaped);
    assert_os_reaped(pid);
}

#[test]
fn fenced_handler_error_does_not_kill_owner_but_unknown_does() {
    let mut owner = spawn("normal");
    let error = owner.execute(tx(1), 0, &json!({"op":"fail"})).unwrap_err();
    assert!(matches!(
        error,
        ProcessError::RemoteFailure {
            failure: ProcessHandlerError {
                quiescence: ProcessQuiescence::Fenced,
                ..
            }
        }
    ));
    assert_eq!(owner.state(), ProcessOwnerState::Ready);
    assert_eq!(
        owner.execute(tx(2), 0, &json!({"value":9})).unwrap()["value"],
        9
    );
    lost(
        &owner
            .execute(tx(3), 0, &json!({"op":"unknown"}))
            .unwrap_err(),
        UnknownCause::Handler,
    );
    assert_eq!(owner.state(), ProcessOwnerState::Reaped);
    assert!(owner.shutdown().unwrap_err().is_quiescence_unknown());
}

#[test]
fn timeout_escalates_past_ignored_term_and_actually_reaps() {
    let mut owner = spawn("ignore_term");
    let pid = owner.pid().unwrap();
    let start = Instant::now();
    let error = owner.execute(tx(1), 0, &json!({"op":"hang"})).unwrap_err();
    lost(&error, UnknownCause::Deadline);
    assert!(start.elapsed() < Duration::from_secs(3));
    assert_eq!(owner.exit_status().unwrap().signal(), Some(libc::SIGKILL));
    assert_os_reaped(pid);
    assert!(matches!(
        owner.execute(tx(2), 0, &json!({})),
        Err(ProcessError::OwnerUnavailable)
    ));
}

#[test]
fn peer_exit_is_unknown_not_a_fence() {
    let mut owner = spawn("normal");
    lost(
        &owner.execute(tx(1), 0, &json!({"op":"exit"})).unwrap_err(),
        UnknownCause::PeerExited,
    );
    assert_eq!(owner.exit_status().unwrap().code(), Some(17));
}

#[test]
fn startup_failure_and_hung_initializer_are_bounded() {
    let error = Owner::spawn(
        launch("normal"),
        identity(1, 1),
        &json!({"startup_fail":true}),
        options(),
    )
    .unwrap_err();
    lost(&error.source, UnknownCause::Handler);
    let mut config = options();
    config.startup_timeout = Duration::from_millis(150);
    let start = Instant::now();
    let error = Owner::spawn(
        launch("normal"),
        identity(2, 2),
        &json!({"startup_hang":true}),
        config,
    )
    .unwrap_err();
    lost(&error.source, UnknownCause::Deadline);
    assert!(start.elapsed() < Duration::from_secs(3));
    assert!(error.owner.is_none());
}

#[test]
fn malformed_and_stale_startup_frames_cannot_produce_ready() {
    for mode in ["raw_stale_ready", "raw_oversized_ready"] {
        let error = Owner::spawn(launch(mode), identity(1, 1), &json!({}), options()).unwrap_err();
        lost(&error.source, UnknownCause::ProtocolViolation);
    }
}

#[test]
fn every_command_identity_component_is_validated() {
    for mode in [
        "raw_stale_epoch",
        "raw_stale_rank",
        "raw_stale_owner",
        "raw_stale_command",
        "raw_stale_txn",
    ] {
        let mut owner = spawn(mode);
        let error = owner.execute(tx(1), 3, &json!({})).unwrap_err();
        lost(&error, UnknownCause::ProtocolViolation);
    }
}

#[test]
fn oversized_truncated_and_partial_frames_fail_without_unbounded_wait() {
    for (mode, cause) in [
        ("raw_oversized", UnknownCause::ProtocolViolation),
        ("raw_truncated", UnknownCause::TruncatedFrame),
        ("raw_partial_prefix", UnknownCause::Deadline),
        ("raw_slow_body", UnknownCause::Deadline),
    ] {
        let mut owner = spawn(mode);
        let start = Instant::now();
        lost(&owner.execute(tx(1), 0, &json!({})).unwrap_err(), cause);
        assert!(start.elapsed() < Duration::from_secs(3));
    }
}

#[test]
fn partial_writes_share_the_command_deadline() {
    let mut owner = spawn("raw_no_read");
    let start = Instant::now();
    let error = owner
        .execute(tx(1), 0, &json!({"blob":"x".repeat(1024 * 1024)}))
        .unwrap_err();
    lost(&error, UnknownCause::Deadline);
    assert!(start.elapsed() < Duration::from_secs(3));
}

#[test]
fn limits_reject_locally_without_poisoning_a_ready_owner() {
    let mut limits = options();
    limits.frame_limits.max_command_bytes = 1024;
    let mut owner = Owner::spawn(launch("normal"), identity(1, 1), &json!({}), limits).unwrap();
    assert!(matches!(
        owner.execute(tx(1), 0, &json!({"blob":"x".repeat(2048)})),
        Err(ProcessError::FrameTooLarge { .. })
    ));
    let reply = owner.execute(tx(2), 0, &json!({"value":42})).unwrap();
    assert_eq!(
        reply["command"], 1,
        "rejected serialization must not consume identity"
    );
    owner.shutdown().unwrap();
    let mut limits = options();
    limits.frame_limits.max_config_bytes = 10;
    assert!(matches!(
        Owner::spawn(
            launch("normal"),
            identity(1, 1),
            &json!({"too_large":"x".repeat(1024)}),
            limits
        )
        .unwrap_err()
        .source,
        ProcessError::FrameTooLarge { .. }
    ));
}

#[test]
fn supervisor_requires_explicit_new_epoch_and_never_replays() {
    let rank = ParallelRankId::new(0);
    let config = ProcessSupervisorConfig {
        owner: options(),
        max_ranks: 1,
        max_quarantined: 1,
        max_restarts: 1,
    };
    let mut supervisor = Supervisor::new(ProcessGroupEpoch::new(10).unwrap(), config).unwrap();
    let old = supervisor
        .spawn_rank(rank, launch("normal"), &json!({}))
        .unwrap();
    lost(
        &supervisor
            .execute(rank, tx(1), 0, &json!({"op":"exit"}))
            .unwrap_err(),
        UnknownCause::PeerExited,
    );
    assert!(!supervisor.is_valid());
    assert!(matches!(
        supervisor.execute(rank, tx(2), 0, &json!({})),
        Err(ProcessError::GroupInvalidated)
    ));
    assert!(
        supervisor
            .restart_group(ProcessGroupEpoch::new(10).unwrap())
            .is_err()
    );
    supervisor
        .restart_group(ProcessGroupEpoch::new(11).unwrap())
        .unwrap();
    let new = supervisor
        .spawn_rank(rank, launch("normal"), &json!({}))
        .unwrap();
    assert!(new.epoch > old.epoch && new.owner_instance > old.owner_instance);
    let reply = supervisor
        .execute(rank, tx(3), 0, &json!({"value":5}))
        .unwrap();
    assert_eq!(reply["command"], 1);
    assert_eq!(reply["transaction"], 3);
    assert!(matches!(
        supervisor.restart_group(ProcessGroupEpoch::new(12).unwrap()),
        Err(ProcessError::RestartBudgetExhausted)
    ));
    assert!(
        supervisor
            .shutdown()
            .iter()
            .all(|(_, report)| report.reap == ReapOutcome::Reaped)
    );
}

#[test]
fn shutdown_and_destructor_hangs_do_not_block_the_parent() {
    for mode in ["shutdown_hang", "drop_hang"] {
        let mut owner = spawn(mode);
        let start = Instant::now();
        lost(&owner.shutdown().unwrap_err(), UnknownCause::Deadline);
        assert!(start.elapsed() < Duration::from_secs(3));
    }
}

#[test]
fn drop_hands_the_child_to_the_bounded_reaper_without_joining() {
    let owner = spawn("raw_no_read");
    let pid = owner.pid().unwrap();
    let start = Instant::now();
    drop(owner);
    assert!(start.elapsed() < Duration::from_millis(500));
    let end = Instant::now() + Duration::from_secs(3);
    while process_exists(pid) {
        assert!(Instant::now() < end, "dropped child was never reaped");
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_os_reaped(pid);
}

#[test]
fn tiny_reap_budget_retains_the_handle_until_later_reap() {
    let mut config = options();
    config.terminate_grace = Duration::from_nanos(1);
    config.kill_grace = Duration::from_nanos(1);
    let mut owner =
        Owner::spawn(launch("raw_no_read"), identity(1, 1), &json!({}), config).unwrap();
    let pid = owner.pid().unwrap();
    let report = owner.terminate();
    if report.reap != ReapOutcome::Reaped {
        assert_eq!(report.reap, ReapOutcome::NotReaped { pid });
        assert_eq!(owner.pid(), Some(pid));
        assert_eq!(owner.state(), ProcessOwnerState::Quarantined);
        assert!(matches!(
            owner.execute(tx(1), 0, &json!({})),
            Err(ProcessError::OwnerUnavailable)
        ));
    }
    let end = Instant::now() + Duration::from_secs(3);
    while owner.poll_reap().unwrap() != ReapOutcome::Reaped {
        assert!(Instant::now() < end);
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_os_reaped(pid);
    // This exercises real short-budget cleanup, not a claim to reproduce D-state.
}

#[test]
fn startup_failures_cannot_bypass_restart_budget() {
    let config = ProcessSupervisorConfig {
        owner: options(),
        max_ranks: 1,
        max_quarantined: 1,
        max_restarts: 1,
    };
    let mut supervisor = Supervisor::new(ProcessGroupEpoch::new(1).unwrap(), config).unwrap();
    let rank = ParallelRankId::new(0);
    for epoch in [1, 2] {
        assert!(
            supervisor
                .spawn_rank(rank, launch("normal"), &json!({"startup_fail":true}))
                .is_err()
        );
        assert!(matches!(
            supervisor.spawn_rank(rank, launch("normal"), &json!({})),
            Err(ProcessError::GroupInvalidated)
        ));
        let result = supervisor.restart_group(ProcessGroupEpoch::new(epoch + 1).unwrap());
        if epoch == 1 {
            result.unwrap();
        } else {
            assert!(matches!(result, Err(ProcessError::RestartBudgetExhausted)));
        }
    }
}

#[test]
fn fragmented_success_and_large_host_payload_use_the_real_protocol() {
    let mut owner = spawn("raw_fragmented");
    assert_eq!(owner.execute(tx(1), 0, &json!({})).unwrap()["value"], 42);
    assert_eq!(owner.terminate().reap, ReapOutcome::Reaped);
    let mut owner = spawn("normal");
    let payload = json!({"value":7,"blob":"x".repeat(128 * 1024)});
    assert_eq!(
        owner.execute(tx(1), 0, &payload).unwrap()["blob"],
        payload["blob"]
    );
    owner.shutdown().unwrap();
}

#[expect(
    unsafe_code,
    reason = "signal zero only probes the test child, without killing or reaping it"
)]
fn process_exists(pid: u32) -> bool {
    let result = unsafe { libc::kill(pid as libc::pid_t, 0) };
    result == 0 || std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
}

#[test]
fn zero_wire_ids_are_rejected_by_deserialization() {
    assert!(serde_json::from_str::<ProcessGroupEpoch>("0").is_err());
    assert!(serde_json::from_str::<ProcessCommandId>("0").is_err());
    assert!(serde_json::from_str::<ProcessTransactionId>("0").is_err());
}

#[expect(
    unsafe_code,
    reason = "test verifies production already reaped this exact child; WNOHANG never blocks"
)]
fn assert_os_reaped(pid: u32) {
    let result = unsafe { libc::waitpid(pid as libc::pid_t, std::ptr::null_mut(), libc::WNOHANG) };
    assert_eq!(result, -1);
    assert_eq!(
        std::io::Error::last_os_error().raw_os_error(),
        Some(libc::ECHILD)
    );
}

#[test]
fn idle_past_child_io_and_rank_timeout_keeps_ready_without_replay() {
    let mut owner = Owner::spawn(
        launch("normal").arg("150"),
        identity(1, 1),
        &json!({}),
        options(),
    )
    .unwrap();
    let pid = owner.pid().unwrap();
    for command in 1..=2 {
        std::thread::sleep(Duration::from_millis(450));
        let reply = owner.execute(tx(command), 0, &json!({"value":42})).unwrap();
        assert_eq!(owner.state(), ProcessOwnerState::Ready);
        assert_eq!(reply["pid"], pid);
        assert_eq!(reply["command"], command);
        assert_eq!(reply["value"], 42);
    }
    std::thread::sleep(Duration::from_millis(450));
    let start = Instant::now();
    assert_eq!(owner.shutdown().unwrap().reap, ReapOutcome::Reaped);
    assert!(start.elapsed() < Duration::from_secs(2));
    assert!(owner.exit_status().unwrap().success());
    assert_os_reaped(pid);
}

#[test]
fn deadline_before_first_write_preserves_ready_and_command_identity() {
    let mut owner = spawn("normal");
    let pid = owner.pid();
    let mut observations = 0;
    let error = owner
        .execute_observed(tx(1), 0, &json!({}), &mut || {
            observations += 1;
            if observations == 1 {
                std::thread::sleep(Duration::from_millis(200));
            }
        })
        .unwrap_err();
    assert!(matches!(error, ProcessError::Deadline), "{error:?}");
    assert_eq!(owner.state(), ProcessOwnerState::Ready);
    assert_eq!(owner.pid(), pid);
    let reply = owner.execute(tx(2), 0, &json!({"value":42})).unwrap();
    assert_eq!(reply["command"], 1);
    assert_eq!(reply["value"], 42);
    owner.shutdown().unwrap();
}

#[test]
fn envelope_oversize_preserves_ready_and_command_identity() {
    let mut config = options();
    config.frame_limits = ProcessFrameLimits {
        max_frame_bytes: 1024,
        max_command_bytes: 1024,
        max_config_bytes: 1024,
        max_output_bytes: 1024,
        max_error_bytes: 1024,
    };
    let mut owner = Owner::spawn(launch("normal"), identity(1, 1), &json!({}), config).unwrap();
    let input = json!({"blob":"x".repeat(900)});
    assert!(serde_json::to_vec(&input).unwrap().len() < 1024);
    assert!(matches!(
        owner.execute(tx(1), 0, &input),
        Err(ProcessError::FrameTooLarge { limit: 1024 })
    ));
    assert_eq!(owner.state(), ProcessOwnerState::Ready);
    let reply = owner.execute(tx(2), 0, &json!({"value":42})).unwrap();
    assert_eq!(reply["command"], 1);
    assert_eq!(reply["value"], 42);
    owner.shutdown().unwrap();
}

#[test]
fn local_encode_failure_and_encode_deadline_preserve_ready() {
    struct Input {
        fail: bool,
        delay: Duration,
    }
    impl serde::Serialize for Input {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            std::thread::sleep(self.delay);
            if self.fail {
                return Err(serde::ser::Error::custom("test input encode failure"));
            }
            serde::Serialize::serialize(&json!({"value":42}), serializer)
        }
    }
    let mut owner = ProcessRankOwner::<Value, Input, Value>::spawn(
        launch("normal"),
        identity(1, 1),
        &json!({}),
        options(),
    )
    .unwrap();
    let pid = owner.pid();
    let error = owner
        .execute(
            tx(1),
            0,
            &Input {
                fail: true,
                delay: Duration::ZERO,
            },
        )
        .unwrap_err();
    assert!(matches!(error, ProcessError::Json { .. }));
    let error = owner
        .execute(
            tx(2),
            0,
            &Input {
                fail: false,
                delay: Duration::from_millis(200),
            },
        )
        .unwrap_err();
    assert!(matches!(error, ProcessError::Deadline));
    assert_eq!(owner.state(), ProcessOwnerState::Ready);
    assert_eq!(owner.pid(), pid);
    let reply = owner
        .execute(
            tx(3),
            0,
            &Input {
                fail: false,
                delay: Duration::ZERO,
            },
        )
        .unwrap();
    assert_eq!(reply["command"], 1);
    assert_eq!(reply["value"], 42);
    owner.shutdown().unwrap();
}
