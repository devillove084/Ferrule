use std::collections::{BTreeSet, HashSet};
use std::fs::{self, File, FileTimes};
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use super::{
    CheckpointDType, CheckpointPositionedReader, CheckpointReadExtent, CheckpointReadPlan,
    CheckpointSourceFileIdentity, CheckpointTensorReader, CheckpointTensorSlice,
};
use crate::TensorRole;

static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);
const ROWS: usize = 2048;
const COLUMNS: usize = 4;
const HEADER: usize = 23;

struct Fixture {
    directory: PathBuf,
    tensor: CheckpointTensorSlice,
    source: CheckpointSourceFileIdentity,
    bytes: Vec<u8>,
}

impl Fixture {
    fn new() -> Self {
        let directory = std::env::temp_dir().join(format!(
            "ferrule-shared-path-{}-{}",
            std::process::id(),
            NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&directory).unwrap();
        let path = directory.join("shard.bin");
        let mut bytes = vec![0xa5; HEADER];
        bytes.extend((0..ROWS * COLUMNS * 4).map(|index| (index % 251) as u8));
        fs::write(&path, &bytes).unwrap();
        let source = CheckpointSourceFileIdentity::capture(&path).unwrap();
        let tensor = CheckpointTensorSlice {
            name: "expert.weight".into(),
            role: TensorRole::RoutedExpertDown,
            path,
            offset: HEADER as u64,
            bytes: (ROWS * COLUMNS * 4) as u64,
            dtype: CheckpointDType::F32,
            shape: vec![ROWS, COLUMNS],
        };
        Self {
            directory,
            tensor,
            source,
            bytes,
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.directory).unwrap();
    }
}

// Counts distinct retained Arc allocations, not allocator calls, RSS, or frees.
fn unique_path_handles(plan: &CheckpointReadPlan) -> usize {
    plan.extents()
        .iter()
        .map(|extent| Arc::as_ptr(extent.shared_path()) as *const () as usize)
        .collect::<BTreeSet<_>>()
        .len()
}

#[test]
fn single_producer_2048_strided_rows_share_one_path_and_preserve_bytes() {
    let fixture = Fixture::new();
    let reader = CheckpointTensorReader::new(1 << 20);
    let read = reader
        .plan_2d_storage_range(&fixture.tensor, 0..ROWS, 1..3, &fixture.source)
        .unwrap();
    let plan = read.read_plan();
    assert_eq!(plan.extents().len(), ROWS);
    assert_eq!(unique_path_handles(plan), 1);
    assert_eq!(plan.storage_bytes(), (ROWS * 8) as u64);
    let first = plan.extents()[0].shared_path();
    let mut expected = Vec::with_capacity(ROWS * 8);
    for (row, extent) in plan.extents().iter().enumerate() {
        assert!(Arc::ptr_eq(first, extent.shared_path()));
        assert_eq!(extent.path(), fixture.tensor.path);
        assert_eq!(extent.offset(), (HEADER + row * COLUMNS * 4 + 4) as u64);
        assert_eq!(extent.bytes(), 8);
        expected.extend_from_slice(&fixture.bytes[extent.offset() as usize..extent.end() as usize]);
    }
    assert_eq!(reader.read_matrix(&read).unwrap().bytes(), expected);
    let positioned = CheckpointPositionedReader::new(8).read(plan).unwrap();
    assert_eq!(positioned.len(), ROWS);
    assert_eq!(positioned.concat(), expected);
    assert!(plan.validate_source_identity().is_ok());
    eprintln!(
        "strided_rows={ROWS} retained_unique_path_handles={}",
        unique_path_handles(plan)
    );
}

#[test]
fn merged_cloned_extents_retain_producer_path_and_payload_order() {
    let fixture = Fixture::new();
    let reader = CheckpointTensorReader::new(1 << 20);
    let read = reader
        .plan_2d_storage_range(&fixture.tensor, 0..ROWS, 1..3, &fixture.source)
        .unwrap();
    let plan = read.read_plan();
    let merged = CheckpointReadPlan::new(
        plan.extents()
            .iter()
            .rev()
            .chain(plan.extents().iter())
            .cloned()
            .collect::<Vec<_>>(),
        plan.source_files().to_vec(),
    )
    .unwrap();
    assert_eq!(merged.extents().len(), ROWS * 2);
    assert_eq!(unique_path_handles(&merged), 1);
    assert_eq!(merged.source_files(), plan.source_files());
    for extent in merged.extents() {
        assert!(Arc::ptr_eq(
            plan.extents()[0].shared_path(),
            extent.shared_path()
        ));
    }
    let original = CheckpointPositionedReader::new(8).read(plan).unwrap();
    let expected: Vec<_> = original
        .iter()
        .rev()
        .chain(original.iter())
        .cloned()
        .collect();
    assert_eq!(
        CheckpointPositionedReader::new(8).read(&merged).unwrap(),
        expected
    );
}

#[test]
fn different_producer_paths_require_their_actual_source_snapshots() {
    let first = Fixture::new();
    let second = Fixture::new();
    let reader = CheckpointTensorReader::new(1 << 20);
    let left = reader
        .plan_2d_storage_range(&first.tensor, 0..ROWS, 1..3, &first.source)
        .unwrap();
    let right = reader
        .plan_2d_storage_range(&second.tensor, 0..ROWS, 1..3, &second.source)
        .unwrap();
    assert!(!Arc::ptr_eq(
        left.read_plan().extents()[0].shared_path(),
        right.read_plan().extents()[0].shared_path()
    ));
    assert!(
        CheckpointReadPlan::new(
            right.read_plan().extents().to_vec(),
            vec![first.source.clone()]
        )
        .is_err()
    );
    let merged = CheckpointReadPlan::new(
        left.read_plan()
            .extents()
            .iter()
            .chain(right.read_plan().extents().iter())
            .cloned()
            .collect::<Vec<_>>(),
        vec![first.source.clone(), second.source.clone()],
    )
    .unwrap();
    assert_eq!(unique_path_handles(&merged), 2);
    assert!(merged.validate_source_identity().is_ok());
}

#[test]
fn extent_equality_order_and_hash_use_path_value_not_arc_identity() {
    let path = PathBuf::from("checkpoint/shard.bin");
    let left = CheckpointReadExtent::new(path.clone(), 23, 8).unwrap();
    let right = CheckpointReadExtent::new(path.clone(), 23, 8).unwrap();
    assert!(!Arc::ptr_eq(left.shared_path(), right.shared_path()));
    assert_eq!(left, right);
    assert_eq!(left.cmp(&right), std::cmp::Ordering::Equal);
    assert_eq!(HashSet::from([left.clone(), right.clone()]).len(), 1);
    assert_eq!(BTreeSet::from([left.clone(), right]).len(), 1);
    assert_ne!(
        left,
        CheckpointReadExtent::new(path.clone(), 24, 8).unwrap()
    );
    assert_ne!(left, CheckpointReadExtent::new(path, 23, 9).unwrap());
}

#[test]
fn noncanonical_aliases_are_not_collapsed_into_source_identity() {
    let fixture = Fixture::new();
    fs::create_dir(fixture.directory.join("nested")).unwrap();
    let alias = fixture.directory.join("nested/../shard.bin");
    assert_eq!(
        fs::canonicalize(&alias).unwrap(),
        fs::canonicalize(&fixture.tensor.path).unwrap()
    );
    let canonical =
        CheckpointReadExtent::new(fixture.tensor.path.clone(), HEADER as u64, 8).unwrap();
    let aliased = CheckpointReadExtent::new(alias.clone(), HEADER as u64, 8).unwrap();
    assert_ne!(canonical, aliased);
    assert_eq!(aliased.path(), alias);
    assert!(CheckpointReadPlan::new(vec![aliased.clone()], vec![fixture.source.clone()]).is_err());
    let alias_source = CheckpointSourceFileIdentity::capture(&alias).unwrap();
    let plan = CheckpointReadPlan::new(vec![aliased], vec![alias_source]).unwrap();
    assert!(plan.validate_source_identity().is_ok());
    assert_eq!(
        CheckpointPositionedReader::new(8).read(&plan).unwrap()[0],
        fixture.bytes[HEADER..HEADER + 8]
    );
}

#[test]
fn same_arc_does_not_make_replaced_inode_a_current_source() {
    let fixture = Fixture::new();
    let reader = CheckpointTensorReader::new(1 << 20);
    let read = reader
        .plan_2d_storage_range(&fixture.tensor, 0..ROWS, 1..3, &fixture.source)
        .unwrap();
    let clone = read.read_plan().clone();
    let metadata = fs::metadata(&fixture.tensor.path).unwrap();
    fs::rename(&fixture.tensor.path, fixture.directory.join("original.bin")).unwrap();
    fs::write(&fixture.tensor.path, &fixture.bytes).unwrap();
    File::options()
        .write(true)
        .open(&fixture.tensor.path)
        .unwrap()
        .set_times(FileTimes::new().set_modified(metadata.modified().unwrap()))
        .unwrap();
    assert_eq!(
        fs::metadata(&fixture.tensor.path).unwrap().len(),
        metadata.len()
    );
    assert_eq!(
        fs::metadata(&fixture.tensor.path)
            .unwrap()
            .modified()
            .unwrap(),
        metadata.modified().unwrap()
    );
    assert!(Arc::ptr_eq(
        clone.extents()[0].shared_path(),
        read.read_plan().extents()[0].shared_path()
    ));
    assert!(read.read_plan().validate_source_identity().is_err());
    assert!(clone.validate_source_identity().is_err());
    assert!(CheckpointPositionedReader::new(8).read(&clone).is_err());
    assert!(reader.read_matrix(&read).is_err());
}

#[cfg(unix)]
#[test]
fn symlink_retarget_is_rejected_without_changing_extent_arc() {
    let fixture = Fixture::new();
    let link = fixture.directory.join("alias.bin");
    std::os::unix::fs::symlink(&fixture.tensor.path, &link).unwrap();
    let source = CheckpointSourceFileIdentity::capture(&link).unwrap();
    let extent = CheckpointReadExtent::new(link.clone(), HEADER as u64, 8).unwrap();
    let retained = extent.clone();
    let plan = CheckpointReadPlan::new(vec![extent], vec![source]).unwrap();
    let replacement = fixture.directory.join("replacement.bin");
    fs::write(&replacement, &fixture.bytes).unwrap();
    fs::remove_file(&link).unwrap();
    std::os::unix::fs::symlink(replacement, link).unwrap();
    assert!(Arc::ptr_eq(
        retained.shared_path(),
        plan.extents()[0].shared_path()
    ));
    assert!(plan.validate_source_identity().is_err());
    assert!(CheckpointPositionedReader::new(8).read(&plan).is_err());
}

#[test]
fn extent_layout_reports_actual_size_without_allocator_claims() {
    let extent_bytes = size_of::<CheckpointReadExtent>();
    let shared_path_bytes = size_of::<Arc<Path>>();
    let owned_path_bytes = size_of::<PathBuf>();
    eprintln!(
        "extent_size_bytes={extent_bytes} arc_path_size_bytes={shared_path_bytes} pathbuf_size_bytes={owned_path_bytes}"
    );
    assert_eq!(extent_bytes, shared_path_bytes + 2 * size_of::<u64>());
}

#[test]
fn extent_and_total_byte_overflow_checks_are_preserved() {
    let path = PathBuf::from("checkpoint/shard.bin");
    assert!(CheckpointReadExtent::new(path.clone(), 0, 0).is_err());
    assert!(CheckpointReadExtent::new(path.clone(), u64::MAX, 1).is_err());
    assert!(CheckpointReadExtent::new(path.clone(), u64::MAX - 1, 2).is_err());
    assert_eq!(
        CheckpointReadExtent::new(path.clone(), u64::MAX - 1, 1)
            .unwrap()
            .end(),
        u64::MAX
    );
    let source = CheckpointSourceFileIdentity::for_test(path.clone(), path.clone(), u64::MAX);
    let extent = CheckpointReadExtent::new(path, 0, u64::MAX).unwrap();
    assert!(CheckpointReadPlan::new(vec![extent.clone(), extent], vec![source]).is_err());
}
