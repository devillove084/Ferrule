use super::*;

#[test]
fn file_identity_hash_encoding_includes_each_ctime_coordinate() {
    let original = CheckpointFileIdentity::for_test(8, 1, 2, Some(3), Some(4));
    let first = CheckpointFileIdentity {
        change_time: Some((7, 8)),
        ..original
    };
    let second = CheckpointFileIdentity {
        change_time: Some((7, 9)),
        ..original
    };
    let third = CheckpointFileIdentity {
        change_time: Some((8, 8)),
        ..original
    };
    let hash = |identity: CheckpointFileIdentity| {
        let mut hasher = Sha256::new();
        identity.update_hash(&mut hasher);
        hasher.finalize()
    };
    assert_ne!(first, second);
    assert_ne!(first, third);
    assert_ne!(hash(first), hash(second));
    assert_ne!(hash(first), hash(third));
    assert_ne!(hash(first), hash(original));
    assert_eq!(hash(first), hash(first));
    // v2 binary source metadata is versioned; SHA-256/semantic-content format
    // and the old immutable snapshot value itself are not rewritten.
    let mut expected = Sha256::new();
    expected.update(b"ferrule-checkpoint-file-metadata-v2");
    expected.update(&8u64.to_be_bytes());
    expected.update(&1i64.to_be_bytes());
    expected.update(&2u32.to_be_bytes());
    expected.update(&[1]);
    expected.update(&3u64.to_be_bytes());
    expected.update(&4u64.to_be_bytes());
    expected.update(&[1]);
    expected.update(&7i64.to_be_bytes());
    expected.update(&8u32.to_be_bytes());
    assert_eq!(hash(first), expected.finalize());
    let encoded_digest = hash(first)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    assert_eq!(
        encoded_digest,
        "da1c3db985cb7329e8a7db8c77eaffc4579f2697de66b7b35060c6fd8c811e72"
    );
    assert_eq!(original.change_time, None);
}
