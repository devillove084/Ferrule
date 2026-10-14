use super::*;

#[cfg(unix)]
use crate::checkpoint::metadata_test_support as metadata_support;

#[cfg(unix)]
#[test]
fn ctime_changes_source_generation_but_not_semantic_identity_or_old_receipts() {
    let dir = std::env::temp_dir().join(format!(
        "ferrule-ctime-source-{}-{}",
        std::process::id(),
        unique_temp_nonce()
    ));
    std::fs::create_dir(&dir).unwrap();
    let path = dir.join("weight");
    std::fs::write(&path, [0x38; 8]).unwrap();
    let tensor = CheckpointTensorSlice {
        name: "weight".into(),
        role: crate::TensorRole::AttentionQuery,
        path: path.clone(),
        offset: 0,
        bytes: 8,
        dtype: crate::checkpoint::CheckpointDType::F8E4M3,
        shape: vec![2, 4],
    };
    let old_catalog = CheckpointSourceCatalog::capture([&tensor]).unwrap();
    let old = old_catalog
        .bundle_source(
            b"ctime-test-v1",
            PayloadEncodingId::new(9),
            std::slice::from_ref(&tensor),
        )
        .unwrap();
    let old_receipt = old.source();
    metadata_support::overwrite_restoring_mtime(&path);
    assert!(!old.validate_source_identity());
    let current = CheckpointSourceCatalog::capture([&tensor])
        .unwrap()
        .bundle_source(
            b"ctime-test-v1",
            PayloadEncodingId::new(9),
            std::slice::from_ref(&tensor),
        )
        .unwrap();
    assert_eq!(old.source(), old_receipt);
    assert_eq!(old.source().content_hash(), current.source().content_hash());
    assert_ne!(old.source().identity(), current.source().identity());
    assert_ne!(old.source().generation(), current.source().generation());
    assert!(current.validate_source_identity());
    let old_again = old_catalog
        .bundle_source(b"ctime-test-v1", PayloadEncodingId::new(9), &[tensor])
        .unwrap();
    assert_eq!(old_again.source(), old_receipt);
    std::fs::remove_dir_all(dir).unwrap();
}

#[cfg(unix)]
fn unique_temp_nonce() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos()
}
