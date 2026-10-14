use super::*;
use serde::ser::SerializeSeq;

fn copy() -> Counter {
    COUNTERS.with(|c| c.borrow().0[Phase::Copy as usize])
}

struct InspectsFrame {
    calls_before: u64,
    fail: bool,
}
impl serde::Serialize for InspectsFrame {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(Some(32))?;
        for _ in 0..32 {
            seq.serialize_element(&u32::MAX)?;
            // Earlier accepted chunks must still be local to the writer.
            assert_eq!(copy().calls, self.calls_before);
        }
        if self.fail {
            return Err(serde::ser::Error::custom("after accepted chunks"));
        }
        seq.end()
    }
}

#[test]
fn aggregate_and_detailed_publish_only_after_the_frame() {
    for detailed in [false, true] {
        let before = copy();
        let value = InspectsFrame {
            calls_before: before.calls,
            fail: false,
        };
        let bytes = if detailed {
            encode_counted::<true, _>(&value, 4096)
        } else {
            encode_counted::<false, _>(&value, 4096)
        }
        .unwrap();
        assert_eq!(bytes, serde_json::to_vec(&vec![u32::MAX; 32]).unwrap());
        let after = copy();
        assert!(after.calls - before.calls > 32);
        assert_eq!(
            after.calls - before.calls,
            after.completed - before.completed
        );
        assert_eq!(
            after.completed_bytes - before.completed_bytes,
            bytes.len() as u64
        );
        assert_eq!(after.wall_ns.is_some(), detailed);
        assert_eq!(
            serde_json::to_value(after).unwrap()["wall_ns"].is_null(),
            !detailed
        );
    }
}

#[test]
fn failed_frames_keep_accepted_counts_without_fabricating_time() {
    let before = copy();
    assert!(matches!(
        encode_counted::<false, _>(
            &InspectsFrame {
                calls_before: before.calls,
                fail: true
            },
            4096
        ),
        Err(BoundedJsonError::Serialize(_))
    ));
    let partial = copy();
    assert!(partial.calls > before.calls);
    assert!(partial.completed_bytes > before.completed_bytes);
    assert!(partial.wall_ns.is_none());
    assert!(matches!(
        encode_counted::<false, _>(&[12345], 2),
        Err(BoundedJsonError::Limit)
    ));
    assert_eq!(
        copy().completed_bytes,
        partial.completed_bytes + 1,
        "only opening bracket accepted"
    );
    let partial = copy();
    assert!(matches!(
        encode_counted::<false, _>(&12345, 4),
        Err(BoundedJsonError::Limit)
    ));
    assert_eq!(
        copy().calls,
        partial.calls,
        "rejected first chunk never counted"
    );
    assert_eq!(copy().completed_bytes, partial.completed_bytes);
}

#[test]
fn configured_gate_records_only_when_enabled() {
    let before = copy();
    let value = serde_json::json!({"bits": [0, 2147483648u32, u32::MAX], "signed": -0.0});
    let expected = serde_json::to_vec(&value).unwrap();
    assert_eq!(
        super::super::serialize(&value, expected.len()).unwrap(),
        expected
    );
    let after = copy();
    if enabled() {
        assert!(after.calls > before.calls);
        assert_eq!(
            after.completed_bytes - before.completed_bytes,
            expected.len() as u64
        );
        assert_eq!(after.wall_ns.is_some(), DETAILED_COPY);
    } else {
        assert_eq!(after.calls, before.calls);
        assert_eq!(after.completed_bytes, before.completed_bytes);
        assert_eq!(after.wall_ns, before.wall_ns);
    }
}
