use super::*;

#[test]
fn bounded_json_refuses_before_unbounded_allocation() {
    assert!(matches!(
        bounded(&vec!["abcdef"; 1000], 1024),
        Err(NetworkError::FrameTooLarge)
    ));
    assert_eq!(
        bounded(&serde_json::json!({"f": 0.125}), 64).unwrap(),
        br#"{"f":0.125}"#
    );
}

#[test]
fn unknown_never_becomes_a_command_rejection() {
    assert!(NetworkError::Deadline.is_command_rejection());
    assert!(!NetworkError::Unknown(Box::new(NetworkError::Deadline)).is_command_rejection());
    assert!(!NetworkError::Unavailable.is_command_rejection());
}

use crate as runtime;
#[path = "fault_tests.rs"]
mod faults;
#[path = "../../../support/network/test_support.rs"]
mod support;

#[test]
fn bounded_json_keeps_wire_bytes_and_network_error_mapping() {
    let value = serde_json::json!({"bits": [0, u32::MAX], "text": "\n\"\\\u{96ea}", "f": -0.0});
    let expected = serde_json::to_vec(&value).unwrap();
    assert_eq!(bounded(&value, expected.len()).unwrap(), expected);
    assert_eq!(payload(&value, expected.len()).unwrap(), value);
    assert!(matches!(
        bounded(&value, expected.len() - 1),
        Err(NetworkError::FrameTooLarge)
    ));
    struct Fails;
    impl Serialize for Fails {
        fn serialize<S: serde::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
            Err(serde::ser::Error::custom("private serializer detail"))
        }
    }
    assert!(matches!(
        bounded(&Fails, 64),
        Err(NetworkError::Serialization)
    ));

    let owner = ProcessIdentity::new(
        super::super::process::ProcessGroupEpoch::new(1).unwrap(),
        ferrule_common::ParallelRankId::new(0),
        super::super::process::ProcessOwnerInstanceId::new(1).unwrap(),
    );
    let message = Message::ShutdownAck { owner };
    #[derive(Serialize)]
    struct Wire<'a> {
        version: u16,
        message: &'a Message,
    }
    let wire = serde_json::to_vec(&Wire {
        version: NETWORK_PROTOCOL_VERSION,
        message: &message,
    })
    .unwrap();
    assert_eq!(Envelope::encode(message, wire.len()).unwrap(), wire);
}
