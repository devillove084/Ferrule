use super::*;
use std::cell::Cell;

#[test]
fn exact_json_bytes_and_inclusive_limits() {
    let value = serde_json::json!({
        "bits": [0, u32::MAX], "signed": -0.0,
        "text": "\n\"\\雪", "nested": [null, true, {"a": 1.25}]
    });
    let expected = serde_json::to_vec(&value).unwrap();
    assert_eq!(bounded_json(&value, expected.len()).unwrap(), expected);
    assert!(matches!(
        bounded_json(&value, expected.len() - 1),
        Err(BoundedJsonError::Limit)
    ));
    assert!(matches!(bounded_json(&0, 0), Err(BoundedJsonError::Limit)));
    assert_eq!(bounded_json("", 2).unwrap(), b"\"\"");
    assert_eq!(bounded_json(&0, usize::MAX).unwrap(), b"0");
}

#[test]
fn rejects_chunks_before_allocation_and_caps_capacity_growth() {
    let copies = Cell::new(0);
    let mut writer = BoundedJsonWriter::<_, true>::new(13, |_| copies.set(copies.get() + 1));
    assert_eq!(writer.bytes.capacity(), 0);
    assert!(writer.write(&[0; 14]).is_err());
    assert_eq!(writer.bytes.capacity(), 0);
    assert_eq!(copies.get(), 0);

    let mut writer = BoundedJsonWriter::<_, true>::new(13, |_| ());
    for _ in 0..13 {
        writer.write_all(b"x").unwrap();
        assert!(writer.bytes.capacity() <= 13);
    }
    let capacity = writer.bytes.capacity();
    assert!(writer.write(b"x").is_err());
    assert_eq!(writer.bytes.len(), 13);
    assert_eq!(writer.bytes.capacity(), capacity);
    writer.flush().unwrap();

    let writer = BoundedJsonWriter::<_, true>::new(usize::MAX, |_: usize| ());
    assert_eq!(writer.bytes.capacity(), 0, "limit is not a reservation");
}

#[test]
fn serializer_stops_at_limit_without_materializing_declared_sequence() {
    struct LongSequence<'a>(&'a Cell<usize>);
    impl Serialize for LongSequence<'_> {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            use serde::ser::SerializeSeq;
            let mut sequence = serializer.serialize_seq(Some(usize::MAX))?;
            for _ in 0..usize::MAX {
                self.0.set(self.0.get() + 1);
                sequence.serialize_element(&u32::MAX)?;
            }
            sequence.end()
        }
    }
    let visits = Cell::new(0);
    assert!(matches!(
        bounded_json(&LongSequence(&visits), 12),
        Err(BoundedJsonError::Limit)
    ));
    assert!(visits.get() > 0);
    assert!(
        visits.get() <= 2,
        "bounded writer kept iterating: {}",
        visits.get()
    );
}

#[test]
fn serde_failures_remain_distinct_from_limits() {
    struct Fails;
    impl Serialize for Fails {
        fn serialize<S: serde::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
            Err(serde::ser::Error::custom("deliberate serializer failure"))
        }
    }
    match bounded_json(&Fails, 64) {
        Err(BoundedJsonError::Serialize(source)) => {
            assert!(source.to_string().contains("deliberate serializer failure"));
        }
        other => panic!("wrong error: {other:?}"),
    }
}

#[test]
fn observation_guard_spans_only_accepted_copy_chunks() {
    struct Copy<'a>(&'a Cell<usize>, usize);
    impl Drop for Copy<'_> {
        fn drop(&mut self) {
            self.0.set(self.0.get() + self.1);
        }
    }
    let copied = Cell::new(0);
    let bytes = bounded_json_counted(&[1, 2, 3], 7, |n| Copy(&copied, n))
        .0
        .unwrap();
    assert_eq!(copied.get(), bytes.len());
    copied.set(0);
    assert!(matches!(
        bounded_json_counted(&12345, 4, |n| Copy(&copied, n)).0,
        Err(BoundedJsonError::Limit)
    ));
    assert_eq!(copied.get(), 0);
}

#[test]
fn uncounted_writer_skips_metrics_and_counted_bytes_are_exact() {
    let value =
        serde_json::json!({"bits": [0, 2147483648u32, u32::MAX], "signed": -0.0, "text": "雪\n"});
    let expected = serde_json::to_vec(&value).unwrap();
    let (plain, uncounted) = encode::<false, _, _>(&value, expected.len(), |_| ());
    let (counted, counts) = bounded_json_counted(&value, expected.len(), |_| ());
    assert_eq!(plain.unwrap(), expected);
    assert_eq!(counted.unwrap(), expected);
    assert_eq!(uncounted.calls, 0);
    assert_eq!(counts.bytes, expected.len());
    assert!(counts.calls > 1);
    let (failed, counts) = bounded_json_counted(&[12345], 2, |_| ());
    assert!(matches!(failed, Err(BoundedJsonError::Limit)));
    assert_eq!(counts, CopyCounts { calls: 1, bytes: 1 });
}
