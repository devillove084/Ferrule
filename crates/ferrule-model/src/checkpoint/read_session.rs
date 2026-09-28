//! Bounded, reader-owned shard handles and one-shot verified read batches.
//!
//! No mmap: truncation is an ordinary read/identity error, never SIGBUS. Cache
//! entries are keyed by the full catalog snapshot, not merely by pathname.
//! Sessions pin their handles; pinned entries cannot be evicted, so concurrency
//! cannot silently exceed the FD limit. Exhaustion returns an error, never waits
//! for another session (which could deadlock an async executor).

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use ferrule_common::{Error, Result};

use super::{CheckpointReadPlan, CheckpointSourceFileIdentity};

/// Covers the 14-shard 35B snapshot with two spare entries. Not a global cache.
pub const DEFAULT_MAX_OPEN_SHARDS: usize = 16;

/// Cumulative counters shared by clones of a tensor reader. Metadata counters
/// count actual calls, including failed calls; read_calls counts read syscalls,
/// not tensor/extent requests. Snapshots during concurrent I/O are approximate.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CheckpointReadCounters {
    pub open_calls: u64,
    pub cache_hits: u64,
    pub evictions: u64,
    pub source_checks: u64,
    pub path_metadata_calls: u64,
    pub handle_metadata_calls: u64,
    pub read_calls: u64,
    pub bytes_read: u64,
    pub live_handles: u64,
    pub peak_handles: u64,
}

#[derive(Debug, Default)]
struct Counters {
    open_calls: AtomicU64,
    cache_hits: AtomicU64,
    evictions: AtomicU64,
    source_checks: AtomicU64,
    path_metadata_calls: AtomicU64,
    handle_metadata_calls: AtomicU64,
    read_calls: AtomicU64,
    bytes_read: AtomicU64,
    live_handles: AtomicU64,
    peak_handles: AtomicU64,
}

impl Counters {
    fn snapshot(&self) -> CheckpointReadCounters {
        CheckpointReadCounters {
            open_calls: self.open_calls.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            source_checks: self.source_checks.load(Ordering::Relaxed),
            path_metadata_calls: self.path_metadata_calls.load(Ordering::Relaxed),
            handle_metadata_calls: self.handle_metadata_calls.load(Ordering::Relaxed),
            read_calls: self.read_calls.load(Ordering::Relaxed),
            bytes_read: self.bytes_read.load(Ordering::Relaxed),
            live_handles: self.live_handles.load(Ordering::Relaxed),
            peak_handles: self.peak_handles.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug)]
struct OpenShard {
    #[cfg(unix)]
    file: File,
    // Seek/read is safe only under the SAME lock, never on try_clone() handles
    // whose seek position can be shared with the original descriptor.
    #[cfg(not(unix))]
    file: Mutex<File>,
    counters: Arc<Counters>,
}

impl OpenShard {
    fn metadata(&self) -> io::Result<std::fs::Metadata> {
        self.counters
            .handle_metadata_calls
            .fetch_add(1, Ordering::Relaxed);
        #[cfg(unix)]
        {
            self.file.metadata()
        }
        #[cfg(not(unix))]
        {
            self.file
                .lock()
                .map_err(|_| io::Error::other("checkpoint file lock poisoned"))?
                .metadata()
        }
    }

    fn read_exact_at(&self, mut bytes: &mut [u8], mut offset: u64) -> io::Result<()> {
        #[cfg(not(unix))]
        use std::io::{Read, Seek, SeekFrom};
        #[cfg(unix)]
        use std::os::unix::fs::FileExt;
        #[cfg(not(unix))]
        let mut file = self
            .file
            .lock()
            .map_err(|_| io::Error::other("checkpoint file lock poisoned"))?;
        #[cfg(not(unix))]
        file.seek(SeekFrom::Start(offset))?;
        while !bytes.is_empty() {
            self.counters.read_calls.fetch_add(1, Ordering::Relaxed);
            #[cfg(unix)]
            let result = self.file.read_at(bytes, offset);
            #[cfg(not(unix))]
            let result = file.read(bytes);
            match result {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "checkpoint source truncated",
                    ));
                }
                Ok(count) => {
                    self.counters
                        .bytes_read
                        .fetch_add(count as u64, Ordering::Relaxed);
                    offset += count as u64;
                    bytes = &mut bytes[count..];
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }
}

impl Drop for OpenShard {
    fn drop(&mut self) {
        self.counters.live_handles.fetch_sub(1, Ordering::Relaxed);
    }
}

type CachedShards = Vec<(CheckpointSourceFileIdentity, Arc<OpenShard>)>;

#[derive(Debug, Clone)]
pub(super) struct ShardHandlePool {
    max_shards: usize,
    // Oldest used first; the bounded vector is tiny (normally <=16 entries).
    entries: Arc<Mutex<CachedShards>>,
    counters: Arc<Counters>,
}

impl ShardHandlePool {
    pub(super) fn new(max_shards: usize) -> Result<Self> {
        if max_shards == 0 {
            return Err(invalid("max_shards must be nonzero"));
        }
        Ok(Self {
            max_shards,
            entries: Arc::new(Mutex::new(Vec::new())),
            counters: Arc::new(Counters::default()),
        })
    }

    pub(super) fn counters(&self) -> CheckpointReadCounters {
        self.counters.snapshot()
    }
    pub(super) fn max_shards(&self) -> usize {
        self.max_shards
    }

    pub(super) fn begin(
        &self,
        plan: &CheckpointReadPlan,
        max_bytes: u64,
    ) -> Result<VerifiedReadSession> {
        if plan.storage_bytes() > max_bytes || plan.storage_bytes() > isize::MAX as u64 {
            return Err(invalid(
                "batch exceeds bounded read size or host address space",
            ));
        }
        let mut sources = BTreeMap::new();
        for source in plan.source_files() {
            if let Some(previous) = sources.insert(source.catalog_path(), source)
                && previous != source
            {
                return Err(invalid("conflicting source snapshots in batch"));
            }
        }
        if sources.len() > self.max_shards {
            return Err(invalid("batch exceeds max_shards"));
        }
        let mut handles = BTreeMap::new();
        let mut entries = self
            .entries
            .lock()
            .map_err(|_| invalid("shard cache lock poisoned"))?;
        // Pin all hits first so a miss cannot evict a later source in this batch.
        for source in sources.values() {
            if let Some(index) = entries.iter().position(|(cached, _)| cached == *source) {
                // A temporary path replacement must neither refresh the LRU nor
                // remove a previously verified FD that another session may own.
                self.validate_open_handle(source, &entries[index].1)?;
                let entry = entries.remove(index);
                handles.insert(source.catalog_path().to_path_buf(), Arc::clone(&entry.1));
                entries.push(entry);
                self.counters.cache_hits.fetch_add(1, Ordering::Relaxed);
            }
        }
        for source in sources.values() {
            if handles.contains_key(source.catalog_path()) {
                continue;
            }
            if entries.len() == self.max_shards {
                let index = entries
                    .iter()
                    .position(|(_, handle)| Arc::strong_count(handle) == 1)
                    .ok_or_else(|| invalid("max_shards exhausted by active read sessions"))?;
                // Drop before opening, so even transient live FDs stay bounded.
                entries.remove(index);
                self.counters.evictions.fetch_add(1, Ordering::Relaxed);
            }
            self.counters.open_calls.fetch_add(1, Ordering::Relaxed);
            let file = File::open(source.catalog_path()).map_err(|error| {
                invalid(format!(
                    "open '{}': {error}",
                    source.catalog_path().display()
                ))
            })?;
            let live = self.counters.live_handles.fetch_add(1, Ordering::Relaxed) + 1;
            self.counters
                .peak_handles
                .fetch_max(live, Ordering::Relaxed);
            let handle = Arc::new(OpenShard {
                #[cfg(unix)]
                file,
                #[cfg(not(unix))]
                file: Mutex::new(file),
                counters: Arc::clone(&self.counters),
            });
            // The file is not cache-visible until both the original path and
            // the FD metadata match this source generation.
            self.validate_open_handle(source, &handle)?;
            handles.insert(source.catalog_path().to_path_buf(), Arc::clone(&handle));
            entries.push(((*source).clone(), handle));
        }
        drop(entries);
        Ok(VerifiedReadSession {
            extents: plan
                .extents()
                .iter()
                .map(|e| (e.path().to_path_buf(), e.offset(), e.bytes()))
                .collect(),
            plan: plan.clone(),
            handles,
            // Retain the pool while a session is alive, even if the reader drops.
            _pool: self.clone(),
        })
    }

    fn validate_open_handle(
        &self,
        source: &CheckpointSourceFileIdentity,
        handle: &OpenShard,
    ) -> Result<()> {
        self.counters.source_checks.fetch_add(1, Ordering::Relaxed);
        if !source.is_current_counted(Some(&self.counters.path_metadata_calls))
            || !handle
                .metadata()
                .is_ok_and(|metadata| source.matches_open_metadata(&metadata))
        {
            return Err(invalid(format!(
                "stale checkpoint source identity '{}'",
                source.catalog_path().display()
            )));
        }
        Ok(())
    }
}

/// One-shot batch bound to catalog snapshots AND currently opened shard handles.
/// Creation checks each distinct source once; consuming the session checks again
/// before returning bytes. No session can be reused as a long-lived trust token.
/// It owns its handles (no mutex guard across awaits), is Send + Sync, and can be
/// moved to a blocking I/O worker. Dropping/cancelling a session releases pins.
#[derive(Debug)]
pub struct VerifiedReadSession {
    plan: CheckpointReadPlan,
    extents: BTreeSet<(std::path::PathBuf, u64, u64)>,
    handles: BTreeMap<std::path::PathBuf, Arc<OpenShard>>,
    _pool: ShardHandlePool,
}

impl VerifiedReadSession {
    pub fn read_plan(&self) -> &CheckpointReadPlan {
        &self.plan
    }

    pub fn read(self) -> Result<Vec<u8>> {
        self.read_validated(Ok)
    }

    // The callback is crate-private: only a checked consumer may construct its
    // artifact inside the batch; the post boundary still precedes publication.
    pub(super) fn read_validated<T>(
        self,
        validate: impl FnOnce(Vec<u8>) -> Result<T>,
    ) -> Result<T> {
        let result = validate(self.read_subset(&self.plan)?)?;
        self.validate_boundary()?;
        Ok(result)
    }

    /// The callback must join all workers before returning. Nothing escapes the
    /// transaction until the unique-source post boundary succeeds.
    pub(crate) fn read_incrementally<T>(self, read: impl FnOnce(&Self) -> Result<T>) -> Result<T> {
        let result = read(&self)?;
        self.validate_boundary()?;
        Ok(result)
    }

    pub(crate) fn read_subset(&self, plan: &CheckpointReadPlan) -> Result<Vec<u8>> {
        if plan
            .source_files()
            .iter()
            .any(|s| !self.plan.source_files().contains(s))
            || plan.extents().iter().any(|e| {
                !self
                    .extents
                    .contains(&(e.path().to_path_buf(), e.offset(), e.bytes()))
            })
        {
            return Err(invalid("incremental read is outside the verified session"));
        }
        let length = plan.storage_bytes() as usize;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|error| invalid(format!("allocate batch: {error}")))?;
        bytes.resize(length, 0);
        let mut cursor = 0;
        for extent in plan.extents() {
            let end = cursor + extent.bytes() as usize;
            self.handles[extent.path()]
                .read_exact_at(&mut bytes[cursor..end], extent.offset())
                .map_err(|error| {
                    invalid(format!(
                        "read '{}' extent {}..{}: {error}",
                        extent.path().display(),
                        extent.offset(),
                        extent.end()
                    ))
                })?;
            cursor = end;
        }
        Ok(bytes)
    }

    fn validate_boundary(&self) -> Result<()> {
        for (path, handle) in &self.handles {
            let source = self
                .plan
                .source_files()
                .iter()
                .find(|source| source.catalog_path() == path)
                .expect("session handles originate in its plan");
            self._pool.validate_open_handle(source, handle)?;
        }
        Ok(())
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Model {
        message: format!("checkpoint read session: {}", message.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::{CheckpointReadExtent, CheckpointTensorReader};

    #[test]
    fn incremental_post_failure_reclaims_every_unpublished_value() {
        let path =
            std::env::temp_dir().join(format!("ferrule-incremental-post-{}", std::process::id()));
        std::fs::write(&path, [1, 2, 3, 4]).unwrap();
        let source = CheckpointSourceFileIdentity::capture(&path).unwrap();
        let plan = CheckpointReadPlan::new(
            [CheckpointReadExtent::new(path.clone(), 0, 4).unwrap()],
            Arc::<[CheckpointSourceFileIdentity]>::from([source]),
        )
        .unwrap();
        let reader = CheckpointTensorReader::new(4);
        let live = Arc::new(AtomicU64::new(0));
        #[derive(Debug)]
        struct Tracked(Arc<AtomicU64>);
        impl Drop for Tracked {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::Relaxed);
            }
        }
        let result = reader
            .verified_read_session(&plan)
            .unwrap()
            .read_incrementally(|session| {
                let values = std::thread::scope(|scope| {
                    let workers = (0..4)
                        .map(|_| {
                            scope.spawn(|| {
                                assert_eq!(session.read_subset(&plan).unwrap(), [1, 2, 3, 4]);
                                live.fetch_add(1, Ordering::Relaxed);
                                Tracked(live.clone())
                            })
                        })
                        .collect::<Vec<_>>();
                    workers
                        .into_iter()
                        .map(|w| w.join().unwrap())
                        .collect::<Vec<_>>()
                });
                assert_eq!(live.load(Ordering::Relaxed), 4);
                File::options()
                    .write(true)
                    .open(&path)
                    .unwrap()
                    .set_len(0)
                    .unwrap();
                Ok(values)
            });
        assert!(result.is_err());
        assert_eq!(live.load(Ordering::Relaxed), 0);
        assert_eq!(reader.read_counters().source_checks, 2);
        assert_eq!(reader.read_counters().read_calls, 4);
        drop(reader);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn source_change_during_byte_validation_cannot_publish() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("ferrule-post-read-{}-{nonce}", std::process::id()));
        std::fs::write(&path, [1, 2, 3, 4]).unwrap();
        let source = CheckpointSourceFileIdentity::capture(&path).unwrap();
        let plan = CheckpointReadPlan::new(
            [CheckpointReadExtent::new(path.clone(), 0, 4).unwrap()],
            Arc::<[CheckpointSourceFileIdentity]>::from([source]),
        )
        .unwrap();
        let reader = CheckpointTensorReader::new(4);
        let session = reader.verified_read_session(&plan).unwrap();
        let result = session.read_validated(|bytes| {
            assert_eq!(bytes, [1, 2, 3, 4]);
            // All positional reads have completed, but publication has not.
            File::options()
                .write(true)
                .open(&path)
                .unwrap()
                .set_len(0)
                .unwrap();
            Ok(bytes)
        });
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("stale checkpoint source")
        );
        assert_eq!(reader.read_counters().source_checks, 2);
        assert_eq!(reader.read_counters().read_calls, 1);
        drop(reader);
        std::fs::remove_file(path).unwrap();
    }
}
