use super::super::{
    ProcessCommandId, ProcessGroupEpoch, ProcessLaunch, ProcessOwnerConfig, ProcessOwnerInstanceId,
    ProcessRankOwner, ProcessTransactionId, UnknownCause,
};
use super::*;
use serde::{Deserialize, Deserializer};
use serde_json::json;
use std::cell::Cell;
use std::os::unix::net::UnixStream;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

thread_local! { static EXPIRE_CONFIG: Cell<bool> = const { Cell::new(false) }; }
pub(super) fn after_config_decode() {
    if EXPIRE_CONFIG.with(|armed| armed.replace(false)) {
        ipc::test_clock::expire();
    }
}

#[derive(Default)]
struct Calls {
    initialized: AtomicUsize,
    executed: AtomicUsize,
    dropped: AtomicUsize,
}
struct Handler(Arc<Calls>);
impl Drop for Handler {
    fn drop(&mut self) {
        self.0.dropped.fetch_add(1, Ordering::Relaxed);
    }
}
struct Command;
impl<'de> Deserialize<'de> for Command {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(d)?;
        if value["expire"] == true {
            ipc::test_clock::expire();
        }
        if value["fail"] == true {
            return Err(serde::de::Error::custom("controlled decode failure"));
        }
        Ok(Self)
    }
}
impl ProcessChildHandler for Handler {
    type Command = Command;
    type Output = Value;
    fn execute(&mut self, _: ProcessRequest<Command>) -> Result<Value, ProcessHandlerError> {
        self.0.executed.fetch_add(1, Ordering::Relaxed);
        Ok(json!({"ok":true}))
    }
}
fn identity() -> ProcessIdentity {
    ProcessIdentity::new(
        ProcessGroupEpoch::new(1).unwrap(),
        ferrule_common::ParallelRankId::new(1),
        ProcessOwnerInstanceId::new(1).unwrap(),
    )
}
fn send(stream: &mut UnixStream, message: Message) {
    let bytes = ipc::encode(message, 1 << 20).unwrap();
    ipc::send(stream, &bytes, Instant::now() + Duration::from_secs(5)).unwrap();
}
fn receive(stream: &mut UnixStream) -> Result<Value, ProcessError> {
    let frame = ipc::receive(stream, 1 << 20, Instant::now() + Duration::from_secs(5))?;
    Ok(serde_json::to_value(frame.message()?).unwrap())
}
fn boot(stream: &mut UnixStream) {
    send(
        stream,
        Message::Boot {
            identity: identity(),
            limits: ProcessFrameLimits::default(),
            config: json!({"config":1}),
        },
    );
}
fn shutdown(stream: &mut UnixStream) {
    send(stream, Message::Shutdown { owner: identity() });
    assert_eq!(receive(stream).unwrap()["type"], "shutdown_ack");
}
fn execute(stream: &mut UnixStream, payload: Value) {
    send(
        stream,
        Message::Execute {
            identity: ProcessCommandIdentity {
                owner: identity(),
                command: ProcessCommandId::new(1).unwrap(),
                transaction: ProcessTransactionId::new(1).unwrap(),
            },
            session: 0,
            payload,
        },
    );
}

#[test]
fn boot_payload_check_and_decode_expiry_never_call_initializer() {
    for phase in ["payload_check", "config_decode", "normal"] {
        let calls = Arc::new(Calls::default());
        let observed = Arc::clone(&calls);
        let (mut peer, reader) = UnixStream::pair().unwrap();
        peer.set_nonblocking(true).unwrap();
        let writer = reader.try_clone().unwrap();
        let child = std::thread::spawn(move || {
            let _clock = ipc::test_clock::Reset;
            if phase == "payload_check" {
                ipc::test_clock::expire_after_payload_checks(1);
            }
            EXPIRE_CONFIG.with(|armed| armed.set(phase == "config_decode"));
            child_serve_with(
                reader,
                writer,
                ProcessChildConfig {
                    io_timeout: Duration::from_secs(5),
                    ..Default::default()
                },
                move |_| {
                    calls.initialized.fetch_add(1, Ordering::Relaxed);
                    Ok(Handler(calls))
                },
            )
        });
        boot(&mut peer);
        if phase == "normal" {
            assert_eq!(receive(&mut peer).unwrap()["type"], "ready");
            shutdown(&mut peer);
            child.join().unwrap().unwrap();
            assert_eq!(observed.initialized.load(Ordering::Relaxed), 1);
            assert_eq!(observed.dropped.load(Ordering::Relaxed), 1);
        } else {
            assert!(matches!(receive(&mut peer), Err(ProcessError::PipeClosed)));
            assert!(matches!(child.join().unwrap(), Err(ProcessError::Deadline)));
            assert_eq!(observed.initialized.load(Ordering::Relaxed), 0);
            assert_eq!(observed.dropped.load(Ordering::Relaxed), 0);
        }
        assert_eq!(observed.executed.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn command_payload_check_and_typed_decode_expiry_do_not_admit_or_release_handler() {
    for phase in [
        "payload_check",
        "typed_decode",
        "typed_decode_error",
        "normal",
        "decode_error",
    ] {
        let calls = Arc::new(Calls::default());
        let observed = Arc::clone(&calls);
        let (mut peer, reader) = UnixStream::pair().unwrap();
        peer.set_nonblocking(true).unwrap();
        let writer = reader.try_clone().unwrap();
        let child = std::thread::spawn(move || {
            let _clock = ipc::test_clock::Reset;
            // Boot check is first, Execute check is second. Expiry is injected
            // only AFTER the real payload check, not before frame receive.
            if phase == "payload_check" {
                ipc::test_clock::expire_after_payload_checks(2);
            }
            child_serve_with(
                reader,
                writer,
                ProcessChildConfig {
                    io_timeout: Duration::from_secs(5),
                    ..Default::default()
                },
                move |_| {
                    calls.initialized.fetch_add(1, Ordering::Relaxed);
                    Ok(Handler(calls))
                },
            )
        });
        boot(&mut peer);
        assert_eq!(receive(&mut peer).unwrap()["type"], "ready");
        execute(
            &mut peer,
            json!({"expire":phase.starts_with("typed_decode"),"fail":phase.ends_with("error")}),
        );
        if phase == "normal" || phase == "decode_error" {
            let reply = receive(&mut peer).unwrap();
            if phase == "normal" {
                assert_eq!(reply["type"], "complete");
                assert_eq!(reply["payload"], json!({"ok":true}));
            } else {
                assert_eq!(reply["type"], "failure");
                assert_eq!(reply["failure"]["quiescence"], "fenced");
            }
            shutdown(&mut peer);
            child.join().unwrap().unwrap();
            assert_eq!(observed.dropped.load(Ordering::Relaxed), 1);
        } else {
            assert!(matches!(receive(&mut peer), Err(ProcessError::PipeClosed)));
            assert!(matches!(child.join().unwrap(), Err(ProcessError::Deadline)));
            assert_eq!(
                observed.dropped.load(Ordering::Relaxed),
                0,
                "timeout must retain existing handler custody"
            );
        }
        assert_eq!(observed.initialized.load(Ordering::Relaxed), 1);
        assert_eq!(
            observed.executed.load(Ordering::Relaxed),
            usize::from(phase == "normal")
        );
    }
}

#[derive(Debug)]
struct SlowOutput;
impl<'de> Deserialize<'de> for SlowOutput {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let _ = Value::deserialize(d)?;
        ipc::test_clock::expire();
        Ok(Self)
    }
}

#[test]
fn parent_already_rejects_successful_decode_past_original_command_deadline() {
    let _clock = ipc::test_clock::Reset;
    let child =
        crate::build_process_child::build("ferrule-runtime", "example", "process_rank_child");
    let mut owner = ProcessRankOwner::<Value, Value, SlowOutput>::spawn(
        ProcessLaunch::new(child).arg("normal"),
        identity(),
        &json!({}),
        ProcessOwnerConfig {
            startup_timeout: Duration::from_secs(5),
            command_timeout: Duration::from_secs(5),
            terminate_grace: Duration::from_millis(1),
            kill_grace: Duration::from_secs(1),
            ..Default::default()
        },
    )
    .unwrap();
    let tx = ferrule_common::execution::ExecutionTransactionId::new(1).unwrap();
    let error = owner.execute(tx, 0, &json!({"value":1})).unwrap_err();
    assert!(
        matches!(error, ProcessError::QuiescenceUnknown {cause:UnknownCause::Deadline,source,..} if matches!(*source,ProcessError::Deadline))
    );
    assert!(matches!(
        owner.execute(tx, 0, &json!({})),
        Err(ProcessError::OwnerUnavailable)
    ));
}
