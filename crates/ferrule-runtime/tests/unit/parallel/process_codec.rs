use super::*;
use std::os::unix::net::UnixStream;

#[test]
fn bounded_json_preserves_process_errors() {
    struct Fails;
    impl Serialize for Fails {
        fn serialize<S: serde::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
            Err(serde::ser::Error::custom("process serialization failure"))
        }
    }
    match serialize(&Fails, 64) {
        Err(ProcessError::Json { operation, source }) => {
            assert_eq!(operation, "serialize process value");
            assert!(source.to_string().contains("process serialization failure"));
        }
        other => panic!("wrong error: {other:?}"),
    }
    assert!(matches!(
        serialize(&12345, 4),
        Err(ProcessError::FrameTooLarge { limit: 4 })
    ));
}

#[test]
fn oversized_input_prefix_rejects_without_reading_or_allocating_body() {
    let (mut reader, mut writer) = UnixStream::pair().unwrap();
    reader.set_nonblocking(true).unwrap();
    writer.write_all(&u32::MAX.to_be_bytes()).unwrap();
    // Keep the peer open: attempting to read the absent body would time out.
    assert!(matches!(
        receive(&mut reader, 1024, Instant::now() + Duration::from_secs(1)),
        Err(ProcessError::FrameTooLarge { limit: 1024 })
    ));
}

fn raw_wire(kind: &str, payload: &str) -> Vec<u8> {
    format!(r#"{{"version":1,"message":{{"type":"{kind}","identity":{{"owner":{{"epoch":1,"rank":1,"owner_instance":1}},"command":1,"transaction":1}},"session":0,"payload":{payload}}}}}"#).into_bytes()
}

#[test]
fn raw_frames_borrow_payload_and_preserve_json_values_and_exact_limits() {
    use crate::parallel::expert_process::NumericCommand;
    let command = NumericCommand::Stats;
    let bytes = serialize(&command, 16).unwrap();
    let raw = raw_payload(&bytes).unwrap();
    let wire = raw_wire("execute", raw.get());
    let Message::Execute { payload, .. } = decode_frame(&wire).unwrap() else {
        panic!()
    };
    let range = wire.as_ptr_range();
    assert!(payload.get().as_ptr() >= range.start && payload.get().as_ptr() < range.end);
    check_payload(payload, bytes.len()).unwrap();
    assert!(matches!(
        check_payload(payload, bytes.len() - 1),
        Err(ProcessError::FrameTooLarge { .. })
    ));
    assert!(matches!(
        serde_json::from_str::<NumericCommand>(payload.get()).unwrap(),
        NumericCommand::Stats
    ));
    let old: Envelope = serde_json::from_slice(&wire).unwrap();
    assert_eq!(
        serde_json::to_value(old.message).unwrap(),
        serde_json::to_value(decode_frame(&wire).unwrap()).unwrap()
    );
    // Null remains a present payload, not a missing-field sentinel.
    assert!(decode_frame(&raw_wire("execute", "null")).is_ok());
    let padded = raw_wire("execute", "[ 1, 2 ]");
    let Message::Execute { payload, .. } = decode_frame(&padded).unwrap() else {
        panic!()
    };
    assert!(check_payload(payload, 7).is_err());
    check_payload(payload, 8).unwrap();
}

#[test]
fn raw_frames_keep_strict_envelope_variant_fields_and_order_independence() {
    let valid = String::from_utf8(raw_wire("execute", "{}")).unwrap();
    let mutations = [
        valid.replacen("\"version\":1", "\"version\":1,\"version\":1", 1),
        valid.replacen("\"message\":", "\"extra\":0,\"message\":", 1),
        valid.replacen(
            "\"type\":\"execute\"",
            "\"type\":\"execute\",\"type\":\"execute\"",
            1,
        ),
        valid.replacen("\"session\":0", "\"session\":0,\"session\":0", 1),
        valid.replacen("\"payload\":{}", "\"payload\":null,\"payload\":{}", 1),
        valid.replacen("\"payload\":{}", "\"config\":{},\"payload\":{}", 1),
        valid.replacen("\"epoch\":1", "\"epoch\":1,\"epoch\":1", 1),
        valid.replacen("\"command\":1", "\"command\":0", 1),
        valid.replacen("\"payload\":{}", "\"payload\":{},\"extra\":null", 1),
        valid.replacen("\"payload\":{}", "\"payload\":{},\"_borrow\":null", 1),
        valid.replacen("\"type\":\"execute\"", "\"type\":\"other\"", 1),
    ];
    for bad in mutations {
        assert!(
            serde_json::from_str::<Envelope>(&bad).is_err(),
            "legacy accepted {bad}"
        );
        assert!(decode_frame(bad.as_bytes()).is_err(), "raw accepted {bad}");
    }
    let tag_last = valid
        .replace("\"type\":\"execute\",", "")
        .replace("\"payload\":{}", "\"payload\":{},\"type\":\"execute\"");
    assert!(decode_frame(tag_last.as_bytes()).is_ok());
    for kind in ["ready", "shutdown", "shutdown_ack", "failure", "boot"] {
        assert!(decode_frame(&raw_wire(kind, "{}")).is_err());
    }
}

#[test]
fn raw_typed_payload_does_not_erase_duplicate_or_unknown_fields() {
    use crate::parallel::expert_process::{NumericCommand, NumericToken};
    for text in [
        r#"{"type":"compute","transaction":1,"transaction":2,"source":0,"layer":0,"tokens":[]}"#,
        r#"{"type":"compute","transaction":1,"source":0,"layer":0,"tokens":[],"extra":0}"#,
    ] {
        let wire = raw_wire("execute", text);
        let Message::Execute { payload, .. } = decode_frame(&wire).unwrap() else {
            panic!()
        };
        assert!(serde_json::from_str::<NumericCommand>(payload.get()).is_err());
    }
    let token = r#"{"transaction":1,"sequence":1,"source":0,"row":0,"route":0,"layer":0,"expert":0,"weight":0,"values":[],"values":[]}"#;
    assert!(serde_json::from_str::<NumericToken>(token).is_err());
}

#[test]
fn raw_frame_recursion_and_expired_decode_do_not_weaken_limits() {
    for depth in [1, 120, 124, 125, 126, 127, 128, 129, 1024] {
        let text = format!("{}0{}", "[".repeat(depth), "]".repeat(depth));
        let wire = raw_wire("execute", &text);
        let old = serde_json::from_slice::<Envelope>(&wire).is_ok();
        assert_eq!(decode_frame(&wire).is_ok(), old, "depth {depth}");
    }
    let wire = raw_wire("execute", r#"{"text":"[\\\"[[[{}"}"#);
    assert!(decode_frame(&wire).is_ok());
    let frame = Frame {
        body: raw_wire("execute", "{}"),
        deadline: Instant::now(),
    };
    assert!(matches!(frame.message(), Err(ProcessError::Deadline)));
}

#[test]
fn raw_limits_do_not_admit_numeric_spellings_with_larger_canonical_json() {
    for text in ["1e2", "-0", "1e-5", "1e999", "184467440737095516160"] {
        let raw: &RawValue = serde_json::from_str(text).unwrap();
        let old = serde_json::from_str::<Value>(text)
            .ok()
            .and_then(|v| serialize(&v, text.len()).ok());
        assert_eq!(
            check_payload(raw, text.len()).is_ok(),
            old.is_some(),
            "{text}"
        );
    }
    for text in [
        "0",
        "4294967295",
        "null",
        "true",
        r#"{"values":[0,4294967295],"text":"e-0\\\""}"#,
    ] {
        let raw: &RawValue = serde_json::from_str(text).unwrap();
        check_payload(raw, text.len()).unwrap();
    }
}

#[test]
fn canonical_integer_ids_and_u32_payloads_stay_raw_without_value_fallback() {
    for text in [
        "0",
        "-1",
        "999999999999999999",
        "1000000000000000000",
        "9223372036854775807",
        "9223372036854775808",
        "9999999999999999999",
        "10000000000000000000",
        "18446744073709551615",
        "-1000000000000000000",
        "-9223372036854775807",
        "-9223372036854775808",
    ] {
        assert!(
            !scan_json(text.as_bytes()).unwrap(),
            "canonical integer fallback: {text}"
        );
        let value: Value = serde_json::from_str(text).unwrap();
        assert_eq!(serde_json::to_string(&value).unwrap(), text);
        let before = VALUE_FALLBACKS.with(std::cell::Cell::get);
        let raw = raw_payload(text.as_bytes()).unwrap();
        check_payload(raw, text.len()).unwrap();
        assert_eq!(VALUE_FALLBACKS.with(std::cell::Cell::get), before);
    }
    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    #[serde(deny_unknown_fields)]
    struct Payload {
        id: u64,
        signed: i64,
        values: Vec<u32>,
    }
    for id in [
        1_000_000_000_000_000_000,
        9_999_999_999_999_999_999,
        10_000_000_000_000_000_000,
        u64::MAX,
    ] {
        let value = Payload {
            id,
            signed: i64::MIN,
            values: (0..2048).map(|i| u32::MAX - i).collect(),
        };
        let bytes = serialize(&value, 1 << 20).unwrap();
        let before = VALUE_FALLBACKS.with(std::cell::Cell::get);
        let raw = raw_payload(&bytes).unwrap();
        check_payload(raw, bytes.len()).unwrap();
        assert_eq!(
            raw.get().as_ptr(),
            bytes.as_ptr(),
            "payload must remain borrowed"
        );
        let wire = String::from_utf8(raw_wire("execute", raw.get()))
            .unwrap()
            .replace("\"epoch\":1", "\"epoch\":18446744073709551615")
            .replace(
                "\"owner_instance\":1",
                "\"owner_instance\":18446744073709551615",
            )
            .replace("\"command\":1", "\"command\":10000000000000000000")
            .replace("\"transaction\":1", "\"transaction\":18446744073709551615")
            .into_bytes();
        let Message::Execute { payload, .. } = decode_frame(&wire).unwrap() else {
            panic!()
        };
        check_payload(payload, bytes.len()).unwrap();
        assert_eq!(
            VALUE_FALLBACKS.with(std::cell::Cell::get),
            before,
            "integer IDs built a Value tree"
        );
        let decoded: Payload = serde_json::from_str(payload.get()).unwrap();
        let legacy: Payload =
            serde_json::from_value(serde_json::from_slice(&bytes).unwrap()).unwrap();
        assert_eq!(decoded, legacy);
        assert_eq!(decoded, value);
        assert!(matches!(
            check_payload(payload, bytes.len() - 1),
            Err(ProcessError::FrameTooLarge { .. })
        ));
    }
}

#[test]
fn normalization_overflow_and_rejection_precedence_remain_unchanged() {
    for text in [
        "18446744073709551616",
        "-9223372036854775809",
        "9999999999999999999999999999999",
        "-0",
        "1e2",
        "1.0",
        "1e999",
    ] {
        assert!(
            scan_json(text.as_bytes()).unwrap(),
            "missing fallback: {text}"
        );
        let raw: &RawValue = serde_json::from_str(text).unwrap();
        let before = VALUE_FALLBACKS.with(std::cell::Cell::get);
        // Raw length is checked before normalization, even for invalid numbers.
        assert!(matches!(
            check_payload(raw, text.len() - 1),
            Err(ProcessError::FrameTooLarge { .. })
        ));
        assert_eq!(VALUE_FALLBACKS.with(std::cell::Cell::get), before);
        let old = serde_json::from_str::<Value>(text)
            .ok()
            .and_then(|v| serialize(&v, text.len()).ok());
        assert_eq!(
            check_payload(raw, text.len()).is_ok(),
            old.is_some(),
            "{text}"
        );
        assert_eq!(VALUE_FALLBACKS.with(std::cell::Cell::get), before + 1);
    }
    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Id {
        #[serde(rename = "id")]
        _id: u64,
    }
    for text in [
        r#"{"id":18446744073709551615,"id":10000000000000000000}"#,
        r#"{"id":18446744073709551615,"unexpected":0}"#,
    ] {
        let raw = raw_payload(text.as_bytes()).unwrap();
        check_payload(raw, text.len()).unwrap();
        assert!(serde_json::from_str::<Id>(raw.get()).is_err(), "{text}");
    }
    let deep = format!("{}18446744073709551616{}", "[".repeat(128), "]".repeat(128));
    let before = VALUE_FALLBACKS.with(std::cell::Cell::get);
    assert!(
        matches!(raw_payload(deep.as_bytes()),Err(ProcessError::Json {source,..}) if source.to_string().contains("recursion limit"))
    );
    assert_eq!(VALUE_FALLBACKS.with(std::cell::Cell::get), before);
}

#[test]
fn raw_message_sequence_and_null_duplicate_fields_remain_rejected() {
    for message in [
        r#"["execute",null,null,null,0,{},null,null,null]"#,
        r#"{"type":"shutdown","owner":null,"owner":{"epoch":1,"rank":1,"owner_instance":1}}"#,
        r#"{"type":"failure","owner":{"epoch":1,"rank":1,"owner_instance":1},"command":null,"command":null,"failure":null}"#,
    ] {
        let wire = format!(r#"{{"version":1,"message":{message}}}"#);
        assert!(serde_json::from_str::<Envelope>(&wire).is_err());
        assert!(decode_frame(wire.as_bytes()).is_err());
    }
}

#[test]
fn raw_producer_rejects_out_of_range_numbers_before_send() {
    for text in ["1e999", "[1e999]", "-1e999"] {
        assert!(serde_json::from_str::<Value>(text).is_err());
        assert!(matches!(
            raw_payload(text.as_bytes()),
            Err(ProcessError::Json { .. })
        ));
    }
}

#[test]
fn numeric_raw_and_value_paths_preserve_all_finite_bit_patterns() {
    use crate::parallel::expert_process::{NumericCommand, NumericReply, NumericToken};
    let mut seed = 1234567u32;
    for count in [1, 2, 8] {
        let tokens = (0..count)
            .map(|route| NumericToken {
                transaction: 1,
                sequence: 99,
                source: 0,
                row: 0,
                route,
                layer: 0,
                expert: route,
                weight: (-0.0f32).to_bits(),
                values: (0..2048)
                    .map(|i| {
                        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                        if i < 8 {
                            [
                                0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000, 0x7f7fffff,
                                0xff7fffff,
                            ][i]
                        } else if seed & 0x7f800000 == 0x7f800000 {
                            seed ^ 0x00800000
                        } else {
                            seed
                        }
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        let command = NumericCommand::Compute {
            transaction: 1,
            source: 0,
            layer: 0,
            tokens: tokens.clone(),
        };
        let bytes = serialize(&command, 1 << 20).unwrap();
        let raw = raw_payload(&bytes).unwrap();
        let wire = raw_wire("execute", raw.get());
        let Message::Execute { payload, .. } = decode_frame(&wire).unwrap() else {
            panic!()
        };
        check_payload(payload, bytes.len()).unwrap();
        let decoded: NumericCommand = serde_json::from_str(payload.get()).unwrap();
        assert_eq!(
            serde_json::to_value(&decoded).unwrap(),
            serde_json::to_value(&command).unwrap()
        );
        let reply = NumericReply::Results { owner: 1, tokens };
        let bytes = serialize(&reply, 1 << 20).unwrap();
        let raw = raw_payload(&bytes).unwrap();
        let decoded: NumericReply = serde_json::from_str(raw.get()).unwrap();
        assert_eq!(
            serde_json::to_value(decoded).unwrap(),
            serde_json::to_value(reply).unwrap()
        );
    }
}

thread_local! { static VALUE_FALLBACKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }
pub(super) fn record_value_fallback() {
    VALUE_FALLBACKS.with(|calls| calls.set(calls.get() + 1));
}
