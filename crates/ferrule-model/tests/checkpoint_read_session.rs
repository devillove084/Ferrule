//! Host-only regression tests and an opt-in NAS expert I/O microbenchmark.
use std::fs::{File, FileTimes};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use ferrule_model::TensorRole;
use ferrule_model::checkpoint::{
    CheckpointDType, CheckpointReadExtent, CheckpointReadPlan, CheckpointSourceFileIdentity,
    CheckpointTensorReader, CheckpointTensorSlice, NumericFp8Encoding, NumericFp8Source,
    VerifiedReadSession,
};

// Makes process-wide source syscall deltas deterministic within this executable.
static TEST_LOCK: Mutex<()> = Mutex::new(());
static NEXT: AtomicU64 = AtomicU64::new(0);

struct Fixture {
    directory: PathBuf,
    weight: CheckpointTensorSlice,
    scale: CheckpointTensorSlice,
}
impl Fixture {
    fn new() -> Self {
        let directory = std::env::temp_dir().join(format!(
            "ferrule-read-session-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).unwrap();
        let path = directory.join("shard.bin");
        let mut bytes = vec![0xa5; 23];
        let weight = CheckpointTensorSlice {
            name: "expert.weight".into(),
            role: TensorRole::RoutedExpertDown,
            path: path.clone(),
            offset: bytes.len() as u64,
            bytes: 4 * 17 * 19,
            dtype: CheckpointDType::F8E4M3,
            shape: vec![4, 17, 19],
        };
        bytes.extend((0..weight.bytes).map(|i| (i % 126) as u8));
        let scale = CheckpointTensorSlice {
            name: "expert.weight_scale_inv".into(),
            role: TensorRole::RoutedExpertDown,
            path,
            offset: bytes.len() as u64,
            bytes: 16,
            dtype: CheckpointDType::F32,
            shape: vec![4, 1, 1],
        };
        bytes.extend(
            [0.25f32, 0.5, 0.75, 1.0]
                .iter()
                .flat_map(|x| x.to_le_bytes()),
        );
        std::fs::write(&weight.path, bytes).unwrap();
        Self {
            directory,
            weight,
            scale,
        }
    }
    fn source(&self) -> NumericFp8Source {
        let snapshot = CheckpointSourceFileIdentity::capture(&self.weight.path).unwrap();
        NumericFp8Source::new(
            self.weight.clone(),
            self.scale.clone(),
            NumericFp8Encoding::E4M3FnBlock128F32,
            snapshot.clone(),
            snapshot,
        )
        .unwrap()
    }
    fn plan(&self) -> CheckpointReadPlan {
        plan(&self.weight.path, 23, 19)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

fn plan(path: &Path, offset: u64, bytes: u64) -> CheckpointReadPlan {
    CheckpointReadPlan::new(
        [CheckpointReadExtent::new(path.to_path_buf(), offset, bytes).unwrap()],
        Arc::<[CheckpointSourceFileIdentity]>::from([
            CheckpointSourceFileIdentity::capture(path).unwrap()
        ]),
    )
    .unwrap()
}

fn old_payload(plan: &CheckpointReadPlan) -> Vec<u8> {
    let mut packed = Vec::new();
    for extent in plan.extents() {
        // Independent old open/seek/read path: compare exact compressed bytes,
        // not only a dequantized result which could hide a wrong scale origin.
        let mut file = File::open(extent.path()).unwrap();
        file.seek(SeekFrom::Start(extent.offset())).unwrap();
        let mut bytes = vec![0; extent.bytes() as usize];
        file.read_exact(&mut bytes).unwrap();
        packed.extend(bytes);
    }
    packed
}

#[test]
fn failed_open_does_not_poison_restored_original_snapshot() {
    let _guard = TEST_LOCK.lock().unwrap();
    let f = Fixture::new();
    let original = f.plan();
    let expected = old_payload(&original);
    let reader = CheckpointTensorReader::with_max_shards(1024, 1).unwrap();
    let backup = f.directory.join("backup");
    std::fs::rename(&f.weight.path, &backup).unwrap();
    let mut replacement = std::fs::read(&backup).unwrap();
    replacement[23] ^= 1;
    std::fs::write(&f.weight.path, replacement).unwrap();

    // A cold miss opens B, but must not publish that FD under A's snapshot.
    assert!(reader.verified_read_session(&original).is_err());
    let failed = reader.read_counters();
    std::fs::remove_file(&f.weight.path).unwrap();
    std::fs::rename(&backup, &f.weight.path).unwrap();
    assert!(original.source_files()[0].is_current());
    for _ in 0..3 {
        assert_eq!(
            reader
                .verified_read_session(&original)
                .unwrap()
                .read()
                .unwrap(),
            expected
        );
    }
    assert_eq!(failed.live_handles, 0);
    assert_eq!(failed.read_calls, 0);
    assert_eq!(reader.read_counters().open_calls, 2);
    assert_eq!(reader.read_counters().cache_hits, 2);
    assert_eq!(reader.read_counters().evictions, 0);
    assert_eq!(reader.read_counters().peak_handles, 1);
}

#[test]
fn failed_generation_does_not_remove_another_threads_valid_lease_or_cache_entry() {
    let _guard = TEST_LOCK.lock().unwrap();
    let f = Fixture::new();
    let original = f.plan();
    let backup = f.directory.join("backup");
    std::fs::rename(&f.weight.path, &backup).unwrap();
    let mut replacement = std::fs::read(&backup).unwrap();
    replacement[23] ^= 1;
    std::fs::write(&f.weight.path, replacement).unwrap();
    let current = f.plan();
    let expected = old_payload(&current);
    let reader = CheckpointTensorReader::with_max_shards(1024, 2).unwrap();
    let worker_reader = reader.clone();
    let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(0);
    let (resume_tx, resume_rx) = std::sync::mpsc::sync_channel(0);
    let worker = std::thread::spawn(move || {
        let lease = worker_reader.verified_read_session(&current).unwrap();
        ready_tx.send(()).unwrap();
        resume_rx.recv().unwrap();
        assert_eq!(lease.read().unwrap(), expected);
        assert_eq!(
            worker_reader
                .verified_read_session(&current)
                .unwrap()
                .read()
                .unwrap(),
            expected
        );
    });
    ready_rx.recv().unwrap();
    let failed = reader.verified_read_session(&original);
    let counters = reader.read_counters();
    resume_tx.send(()).unwrap();
    worker.join().unwrap();
    assert!(failed.is_err());
    assert_eq!(counters.live_handles, 1); // only B's valid cached/leased FD
    assert_eq!(reader.read_counters().open_calls, 2); // B plus the rejected A miss
    assert_eq!(reader.read_counters().cache_hits, 1); // B was not purged by path
    assert_eq!(reader.read_counters().evictions, 0);
    assert_eq!(reader.read_counters().peak_handles, 2);

    std::fs::remove_file(&f.weight.path).unwrap();
    std::fs::rename(backup, &f.weight.path).unwrap();
    assert_eq!(
        reader
            .verified_read_session(&original)
            .unwrap()
            .read()
            .unwrap(),
        old_payload(&original)
    );
}

#[test]
fn failed_cached_boundary_preserves_lease_when_original_path_is_restored() {
    let _guard = TEST_LOCK.lock().unwrap();
    let f = Fixture::new();
    let original = f.plan();
    let expected = old_payload(&original);
    let reader = CheckpointTensorReader::with_max_shards(1024, 1).unwrap();
    let lease = reader.verified_read_session(&original).unwrap();
    let backup = f.directory.join("backup");
    std::fs::rename(&f.weight.path, &backup).unwrap();
    std::fs::copy(&backup, &f.weight.path).unwrap();
    assert!(reader.verified_read_session(&original).is_err());
    std::fs::remove_file(&f.weight.path).unwrap();
    std::fs::rename(backup, &f.weight.path).unwrap();
    assert_eq!(lease.read().unwrap(), expected);
    assert_eq!(
        reader
            .verified_read_session(&original)
            .unwrap()
            .read()
            .unwrap(),
        expected
    );
    assert_eq!(reader.read_counters().open_calls, 1);
    assert_eq!(reader.read_counters().live_handles, 1);
    assert_eq!(reader.read_counters().peak_handles, 1);
    assert_eq!(reader.read_counters().evictions, 0);
}

#[test]
fn repeated_numeric_tiles_share_one_fd_and_only_two_batch_boundaries() {
    let _guard = TEST_LOCK.lock().unwrap();
    let f = Fixture::new();
    let reader = CheckpointTensorReader::new(1024);
    let source = f.source();
    let plans: Vec<_> = (0..4)
        .map(|expert| {
            source
                .plan_tile(&reader, Some(expert), 2..15, 3..18)
                .unwrap()
        })
        .collect();
    let expected: Vec<_> = plans
        .iter()
        .map(|read| old_payload(read.read_plan()))
        .collect();
    let before = CheckpointSourceFileIdentity::counters();
    for iteration in 0..128 {
        let read = &plans[iteration % 4];
        let artifact = read.read(&reader).unwrap();
        let mut actual = artifact.weight_bytes().to_vec();
        actual.extend(artifact.scale_bytes());
        assert_eq!(actual, expected[iteration % 4]);
    }
    let counters = reader.read_counters();
    assert_eq!(counters.open_calls, 1);
    assert_eq!(counters.cache_hits, 127);
    assert_eq!(counters.source_checks, 256);
    assert_eq!(counters.path_metadata_calls, 256); // one lstat for a regular shard per boundary
    assert_eq!(counters.handle_metadata_calls, 256);
    assert_eq!(counters.read_calls, 128 * 14); // 13 weight rows + scale
    assert_eq!(counters.bytes_read, 128 * (13 * 15 + 4));
    assert_eq!(counters.live_handles, 1);
    let after = CheckpointSourceFileIdentity::counters();
    assert_eq!(after.capture_calls, before.capture_calls);
    assert_eq!(after.canonicalize_calls, before.canonicalize_calls);
    assert_eq!(after.path_metadata_calls - before.path_metadata_calls, 256);
    // A new reader is intentionally not backed by a process-wide FD cache.
    let other = CheckpointTensorReader::new(1024);
    plans[0].read(&other).unwrap();
    assert_eq!(other.read_counters().open_calls, 1);
}

#[test]
fn paired_distinct_shards_are_checked_once_each_per_boundary() {
    let _guard = TEST_LOCK.lock().unwrap();
    let f = Fixture::new();
    let mut scale = f.scale.clone();
    scale.path = f.directory.join("scales");
    std::fs::copy(&f.scale.path, &scale.path).unwrap();
    let source = NumericFp8Source::new(
        f.weight.clone(),
        scale.clone(),
        NumericFp8Encoding::E4M3FnBlock128F32,
        CheckpointSourceFileIdentity::capture(&f.weight.path).unwrap(),
        CheckpointSourceFileIdentity::capture(&scale.path).unwrap(),
    )
    .unwrap();
    let reader = CheckpointTensorReader::with_max_shards(1024, 2).unwrap();
    let read = source.plan_tile(&reader, Some(0), 0..17, 0..19).unwrap();
    let expected = old_payload(read.read_plan());
    for _ in 0..8 {
        let artifact = read.read(&reader).unwrap();
        assert_eq!(
            [artifact.weight_bytes(), artifact.scale_bytes()].concat(),
            expected
        );
    }
    let counters = reader.read_counters();
    assert_eq!(counters.open_calls, 2);
    assert_eq!(counters.source_checks, 8 * 2 * 2);
    assert_eq!(counters.path_metadata_calls, 8 * 2 * 2);
    assert_eq!(counters.handle_metadata_calls, 8 * 2 * 2);
    assert_eq!(counters.read_calls, 8 * 2);
}

#[test]
fn limits_conflicting_snapshots_and_budget_fail_before_open_or_read() {
    let _guard = TEST_LOCK.lock().unwrap();
    assert!(CheckpointTensorReader::with_max_shards(100, 0).is_err());
    let f = Fixture::new();
    let reader = CheckpointTensorReader::new(1024);
    let read = f
        .source()
        .plan_tile(&reader, Some(0), 0..17, 0..19)
        .unwrap();
    let small = CheckpointTensorReader::new(read.read_plan().storage_bytes() - 1);
    assert!(read.read(&small).is_err());
    assert_eq!(small.read_counters(), Default::default());
    let second = f.directory.join("second");
    std::fs::copy(&f.weight.path, &second).unwrap();
    let sources = [
        CheckpointSourceFileIdentity::capture(&f.weight.path).unwrap(),
        CheckpointSourceFileIdentity::capture(&second).unwrap(),
    ];
    let two = CheckpointReadPlan::new(
        [
            CheckpointReadExtent::new(f.weight.path.clone(), 23, 1).unwrap(),
            CheckpointReadExtent::new(second, 23, 1).unwrap(),
        ],
        Arc::<[CheckpointSourceFileIdentity]>::from(sources),
    )
    .unwrap();
    let one = CheckpointTensorReader::with_max_shards(1024, 1).unwrap();
    assert!(one.verified_read_session(&two).is_err());
    assert_eq!(one.read_counters(), Default::default());
    let old = CheckpointSourceFileIdentity::capture(&f.weight.path).unwrap();
    File::options()
        .write(true)
        .open(&f.weight.path)
        .unwrap()
        .set_times(FileTimes::new().set_modified(SystemTime::UNIX_EPOCH + Duration::from_secs(42)))
        .unwrap();
    let new = CheckpointSourceFileIdentity::capture(&f.weight.path).unwrap();
    let conflict = CheckpointReadPlan::new(
        [CheckpointReadExtent::new(f.weight.path.clone(), 23, 1).unwrap()],
        Arc::<[CheckpointSourceFileIdentity]>::from([old, new]),
    )
    .unwrap();
    assert!(one.verified_read_session(&conflict).is_err());
    assert_eq!(one.read_counters(), Default::default());
}

#[test]
fn lru_eviction_and_active_session_pins_bound_actual_open_handles() {
    let _guard = TEST_LOCK.lock().unwrap();
    let f = Fixture::new();
    let first = f.plan();
    let second_path = f.directory.join("second");
    let third_path = f.directory.join("third");
    std::fs::copy(&f.weight.path, &second_path).unwrap();
    std::fs::copy(&f.weight.path, &third_path).unwrap();
    let second = plan(&second_path, 23, 19);
    let third = plan(&third_path, 23, 19);
    let reader = CheckpointTensorReader::with_max_shards(1024, 2).unwrap();
    let pinned1 = reader.verified_read_session(&first).unwrap();
    let pinned2 = reader.verified_read_session(&second).unwrap();
    assert!(reader.verified_read_session(&third).is_err());
    assert_eq!(reader.read_counters().open_calls, 2);
    drop(pinned2); // cancellation releases the pin, not an unbounded orphan FD
    reader
        .verified_read_session(&third)
        .unwrap()
        .read()
        .unwrap();
    assert_eq!(reader.read_counters().evictions, 1);
    assert_eq!(reader.read_counters().peak_handles, 2);
    assert_eq!(pinned1.read().unwrap(), old_payload(&first));
    reader
        .verified_read_session(&second)
        .unwrap()
        .read()
        .unwrap();
    assert_eq!(reader.read_counters().peak_handles, 2);
    assert_eq!(reader.read_counters().live_handles, 2);
}

#[test]
fn default_cache_holds_the_actual_fourteen_shard_snapshot() {
    let _guard = TEST_LOCK.lock().unwrap();
    let f = Fixture::new();
    let reader = CheckpointTensorReader::new(1024);
    assert_eq!(reader.max_shards(), 16);
    let plans: Vec<_> = (0..14)
        .map(|i| {
            let path = f.directory.join(format!("shard-{i}"));
            std::fs::copy(&f.weight.path, &path).unwrap();
            plan(&path, 23, 19)
        })
        .collect();
    for _ in 0..4 {
        for plan in &plans {
            assert_eq!(
                reader.verified_read_session(plan).unwrap().read().unwrap(),
                old_payload(plan)
            );
        }
    }
    assert_eq!(reader.read_counters().open_calls, 14);
    assert_eq!(reader.read_counters().evictions, 0);
    assert_eq!(reader.read_counters().peak_handles, 14);
}

#[test]
fn concurrent_clones_use_positioned_io_and_sessions_can_outlive_reader() {
    let _guard = TEST_LOCK.lock().unwrap();
    fn send_sync<T: Send + Sync>() {}
    send_sync::<CheckpointTensorReader>();
    send_sync::<VerifiedReadSession>();
    let f = Fixture::new();
    let reader = CheckpointTensorReader::with_max_shards(1024, 1).unwrap();
    let source = f.source();
    let workers: Vec<_> = (0..8)
        .map(|i| {
            let reader = reader.clone();
            let plan = source
                .plan_tile(&reader, Some(i % 4), 1..17, 2..19)
                .unwrap();
            let expected = old_payload(plan.read_plan());
            std::thread::spawn(move || {
                for _ in 0..32 {
                    let artifact = plan.read(&reader).unwrap();
                    let mut bytes = artifact.weight_bytes().to_vec();
                    bytes.extend(artifact.scale_bytes());
                    assert_eq!(bytes, expected);
                }
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }
    assert_eq!(reader.read_counters().open_calls, 1);
    assert_eq!(reader.read_counters().source_checks, 8 * 32 * 2);
    let plan = f.plan();
    let session = reader.verified_read_session(&plan).unwrap();
    drop(reader);
    assert_eq!(
        std::thread::spawn(move || session.read().unwrap())
            .join()
            .unwrap(),
        old_payload(&plan)
    );
}

#[test]
fn mutations_fail_on_cached_pre_boundary_and_after_open_post_boundary() {
    let _guard = TEST_LOCK.lock().unwrap();
    for after_open in [false, true] {
        for mutation in [
            "same_size",
            "mtime_only",
            "truncate",
            "replace_same_mtime",
            "rename",
            "unlink",
        ] {
            let f = Fixture::new();
            let plan = f.plan();
            let reader = CheckpointTensorReader::with_max_shards(1024, 1).unwrap();
            reader.verified_read_session(&plan).unwrap().read().unwrap();
            let session = after_open.then(|| reader.verified_read_session(&plan).unwrap());
            let modified = std::fs::metadata(&f.weight.path)
                .unwrap()
                .modified()
                .unwrap();
            match mutation {
                "same_size" | "mtime_only" => {
                    let mut file = File::options().write(true).open(&f.weight.path).unwrap();
                    if mutation == "same_size" {
                        file.seek(SeekFrom::Start(23)).unwrap();
                        file.write_all(&[0x38]).unwrap();
                    }
                    file.set_times(
                        FileTimes::new().set_modified(modified + Duration::from_secs(2)),
                    )
                    .unwrap();
                }
                "truncate" => File::options()
                    .write(true)
                    .open(&f.weight.path)
                    .unwrap()
                    .set_len(24)
                    .unwrap(),
                "replace_same_mtime" => {
                    let replacement = f.directory.join("replacement");
                    std::fs::copy(&f.weight.path, &replacement).unwrap();
                    File::options()
                        .write(true)
                        .open(&replacement)
                        .unwrap()
                        .set_times(FileTimes::new().set_modified(modified))
                        .unwrap();
                    std::fs::rename(replacement, &f.weight.path).unwrap();
                }
                "rename" => {
                    std::fs::rename(&f.weight.path, f.directory.join("old")).unwrap();
                    std::fs::copy(f.directory.join("old"), &f.weight.path).unwrap();
                }
                "unlink" => std::fs::remove_file(&f.weight.path).unwrap(),
                _ => unreachable!(),
            }
            assert!(!plan.source_files()[0].is_current(), "{mutation}");
            if let Some(session) = session {
                assert!(session.read().is_err(), "{mutation}");
            }
            assert!(reader.verified_read_session(&plan).is_err(), "{mutation}");
        }
    }
}

#[test]
fn new_generation_never_reuses_old_fd_even_with_same_length_and_mtime() {
    let _guard = TEST_LOCK.lock().unwrap();
    let f = Fixture::new();
    let old_plan = f.plan();
    let reader = CheckpointTensorReader::with_max_shards(1024, 1).unwrap();
    let old_bytes = reader
        .verified_read_session(&old_plan)
        .unwrap()
        .read()
        .unwrap();
    let replacement = f.directory.join("new");
    let modified = std::fs::metadata(&f.weight.path)
        .unwrap()
        .modified()
        .unwrap();
    let mut bytes = std::fs::read(&f.weight.path).unwrap();
    bytes[23] ^= 1;
    std::fs::write(&replacement, bytes).unwrap();
    File::options()
        .write(true)
        .open(&replacement)
        .unwrap()
        .set_times(FileTimes::new().set_modified(modified))
        .unwrap();
    std::fs::rename(replacement, &f.weight.path).unwrap();
    assert!(reader.verified_read_session(&old_plan).is_err());
    let new_plan = f.plan();
    assert_ne!(old_plan.source_files(), new_plan.source_files());
    let actual = reader
        .verified_read_session(&new_plan)
        .unwrap()
        .read()
        .unwrap();
    assert_eq!(actual, old_payload(&new_plan));
    assert_ne!(actual, old_bytes);
    assert_eq!(reader.read_counters().open_calls, 2);
    assert_eq!(reader.read_counters().peak_handles, 1);
}

#[cfg(unix)]
#[test]
fn symlink_retarget_and_same_target_link_replacement_are_rejected() {
    use std::os::unix::fs::symlink;
    let _guard = TEST_LOCK.lock().unwrap();
    for same_inode in [false, true] {
        let f = Fixture::new();
        let alias = f.directory.join("alias");
        let target = f.directory.join("target");
        if same_inode {
            std::fs::hard_link(&f.weight.path, &target).unwrap();
        } else {
            std::fs::copy(&f.weight.path, &target).unwrap();
        }
        symlink(&f.weight.path, &alias).unwrap();
        let plan = plan(&alias, 23, 19);
        let reader = CheckpointTensorReader::new(1024);
        let session = reader.verified_read_session(&plan).unwrap();
        let link = f.directory.join("new-link");
        symlink(&target, &link).unwrap();
        std::fs::rename(link, &alias).unwrap();
        assert!(!plan.source_files()[0].is_current());
        assert!(session.read().is_err());
        assert!(reader.verified_read_session(&plan).is_err());
    }
}
