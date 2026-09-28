//! Private IPC invariants, included only by ipc.rs under cfg(test).
use super::*;
use std::os::unix::net::UnixStream;

fn identity() -> ProcessIdentity {
    ProcessIdentity::new(
        super::super::ProcessGroupEpoch::new(1).unwrap(),
        ferrule_common::ParallelRankId::new(0),
        super::super::ProcessOwnerInstanceId::new(1).unwrap(),
    )
}

fn receive_bytes(bytes: &[u8], limit: usize) -> Result<Message, ProcessError> {
    let (mut reader, mut writer) = UnixStream::pair().unwrap();
    reader.set_nonblocking(true).unwrap();
    writer.write_all(bytes).unwrap();
    drop(writer);
    receive(&mut reader, limit, Instant::now() + Duration::from_secs(1))
}

fn frame(body: &[u8]) -> Vec<u8> {
    let mut bytes = (body.len() as u32).to_be_bytes().to_vec();
    bytes.extend_from_slice(body);
    bytes
}

#[test]
fn bounded_serializer_keeps_exact_json_and_original_length_limit() {
    let value = serde_json::json!({"activation":[-1.5,0.0,2.25],"text":"\n雪"});
    let expected = serde_json::to_vec(&value).unwrap();
    assert_eq!(serialize(&value, expected.len()).unwrap(), expected);
    assert!(matches!(
        serialize(&value, expected.len() - 1),
        Err(ProcessError::FrameTooLarge { .. })
    ));
    assert_eq!(payload(&value, expected.len()).unwrap(), value);
    assert!(matches!(
        payload(&value, expected.len() - 1),
        Err(ProcessError::FrameTooLarge { .. })
    ));
    let message = Message::Shutdown { owner: identity() };
    let encoded = encode(message, 1024).unwrap();
    let envelope: Value = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(envelope["version"], PROCESS_PROTOCOL_VERSION);
    assert_eq!(envelope["message"]["type"], "shutdown");
    assert!(matches!(
        receive_bytes(&frame(&encoded), encoded.len()),
        Ok(Message::Shutdown { .. })
    ));
}

#[test]
fn malformed_version_length_json_and_truncation_remain_fail_closed() {
    let encoded = encode(Message::Shutdown { owner: identity() }, 1024).unwrap();
    let mut changed: Value = serde_json::from_slice(&encoded).unwrap();
    changed["version"] = serde_json::json!(PROCESS_PROTOCOL_VERSION + 1);
    let error = receive_bytes(&frame(&serde_json::to_vec(&changed).unwrap()), 1024).unwrap_err();
    assert!(error.to_string().contains("unsupported protocol version"));
    assert!(
        receive_bytes(&0u32.to_be_bytes(), 1024)
            .unwrap_err()
            .to_string()
            .contains("empty frame")
    );
    assert!(matches!(
        receive_bytes(&u32::MAX.to_be_bytes(), 1024),
        Err(ProcessError::FrameTooLarge { .. })
    ));
    assert!(matches!(
        receive_bytes(&frame(b"{"), 1024),
        Err(ProcessError::Json { .. })
    ));
    for bytes in [&[0, 0][..], &[0, 0, 0, 10, b'{'][..]] {
        assert!(matches!(
            receive_bytes(bytes, 1024),
            Err(ProcessError::TruncatedFrame)
        ));
    }
    assert!(matches!(
        receive_bytes(&[], 1024),
        Err(ProcessError::PipeClosed)
    ));
}

#[test]
fn expired_deadline_never_sends_prefix_and_observer_cannot_slide_deadline() {
    let (mut writer, mut reader) = UnixStream::pair().unwrap();
    writer.set_nonblocking(true).unwrap();
    reader.set_nonblocking(true).unwrap();
    let mut sent = false;
    let body = encode(Message::Shutdown { owner: identity() }, 1024).unwrap();
    assert!(matches!(
        send_observed(&mut writer, &body, Instant::now(), &mut || {}, &mut sent),
        Err(ProcessError::Deadline)
    ));
    assert!(!sent);
    assert_eq!(
        reader.read(&mut [0; 4]).unwrap_err().kind(),
        io::ErrorKind::WouldBlock
    );
    writer.write_all(&frame(&body)).unwrap();
    let deadline = Instant::now() + Duration::from_millis(10);
    assert!(matches!(
        receive_observed(&mut reader, 1024, deadline, &mut || {
            std::thread::sleep(Duration::from_millis(20));
        }),
        Err(ProcessError::Deadline)
    ));
}
