//! Provider-neutral ordered read plans for checkpoint-backed materialization.
//!
//! A plan owns the exact physical extents and the catalog-time file snapshots that
//! participated in source identity. It performs no I/O and contains no tensor,
//! model-family, device, or installer semantics. Readers preserve extent order;
//! installers interpret returned payloads using their separately retained semantic
//! descriptor.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ferrule_common::{Error, Result, StaleReason};

use super::CheckpointSourceFileIdentity;

/// One physical checkpoint byte range in payload order.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CheckpointReadExtent {
    path: PathBuf,
    offset: u64,
    bytes: u64,
}

impl CheckpointReadExtent {
    pub fn new(path: PathBuf, offset: u64, bytes: u64) -> Result<Self> {
        if bytes == 0 {
            return Err(Error::Model(
                "checkpoint read extent must contain at least one byte".into(),
            ));
        }
        offset
            .checked_add(bytes)
            .ok_or_else(|| Error::Model("checkpoint read extent overflows u64".into()))?;
        Ok(Self {
            path,
            offset,
            bytes,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub const fn offset(&self) -> u64 {
        self.offset
    }

    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    pub const fn end(&self) -> u64 {
        self.offset + self.bytes
    }
}

/// Ordered physical checkpoint source with immutable snapshot custody.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointReadPlan {
    extents: Arc<[CheckpointReadExtent]>,
    source_files: Arc<[CheckpointSourceFileIdentity]>,
    storage_bytes: u64,
}

impl CheckpointReadPlan {
    pub fn new(
        extents: impl IntoIterator<Item = CheckpointReadExtent>,
        source_files: impl Into<Arc<[CheckpointSourceFileIdentity]>>,
    ) -> Result<Self> {
        let extents = extents.into_iter().collect::<Vec<_>>();
        if extents.is_empty() {
            return Err(Error::Model(
                "checkpoint read plan requires at least one extent".into(),
            ));
        }
        let source_files = source_files.into();
        if source_files.is_empty() {
            return Err(Error::Model(
                "checkpoint read plan requires catalog-time source snapshots".into(),
            ));
        }
        let snapshots = source_files
            .iter()
            .map(|source_file| (source_file.catalog_path(), source_file))
            .collect::<BTreeMap<_, _>>();
        let mut storage_bytes = 0u64;
        for extent in &extents {
            let source_file = snapshots.get(extent.path()).ok_or_else(|| {
                Error::Model(format!(
                    "checkpoint read extent '{}' has no catalog-time source snapshot",
                    extent.path().display()
                ))
            })?;
            if extent.end() > source_file.length() {
                return Err(Error::Model(format!(
                    "checkpoint read extent {}..{} exceeds source length {} for '{}'",
                    extent.offset(),
                    extent.end(),
                    source_file.length(),
                    extent.path().display()
                )));
            }
            storage_bytes = storage_bytes.checked_add(extent.bytes()).ok_or_else(|| {
                Error::Model("checkpoint read plan byte size overflows u64".into())
            })?;
        }
        Ok(Self {
            extents: Arc::from(extents),
            source_files,
            storage_bytes,
        })
    }

    pub fn extents(&self) -> &[CheckpointReadExtent] {
        &self.extents
    }

    pub fn source_files(&self) -> &[CheckpointSourceFileIdentity] {
        &self.source_files
    }

    pub const fn storage_bytes(&self) -> u64 {
        self.storage_bytes
    }

    pub fn validate_source_identity(&self) -> std::result::Result<(), StaleReason> {
        if self
            .source_files
            .iter()
            .all(CheckpointSourceFileIdentity::is_current)
        {
            Ok(())
        } else {
            Err(StaleReason::SourceIdentityChanged)
        }
    }
}

/// Bounded provider-neutral positioned reader for checkpoint read plans.
///
/// This is the CPU/reference implementation and a deterministic mockable source
/// reader. Production providers may lower the same plan to io_uring, GDS, Metal,
/// ROCm, or another transport without changing source identity or extent order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointPositionedReader {
    max_extent_bytes: u64,
}

impl CheckpointPositionedReader {
    pub const fn new(max_extent_bytes: u64) -> Self {
        Self { max_extent_bytes }
    }

    pub const fn max_extent_bytes(self) -> u64 {
        self.max_extent_bytes
    }

    pub fn read(&self, plan: &CheckpointReadPlan) -> Result<Vec<Vec<u8>>> {
        use std::io::{Read, Seek, SeekFrom};

        plan.validate_source_identity()
            .map_err(stale_source_identity_error)?;
        let mut payloads = Vec::with_capacity(plan.extents().len());
        for extent in plan.extents() {
            if extent.bytes() > self.max_extent_bytes {
                return Err(Error::Model(format!(
                    "checkpoint read extent exceeds bounded read size: {} > {} bytes",
                    extent.bytes(),
                    self.max_extent_bytes
                )));
            }
            let length = usize::try_from(extent.bytes()).map_err(|_| {
                Error::Model("checkpoint read extent does not fit host address space".into())
            })?;
            let file = std::fs::File::open(extent.path()).map_err(|error| {
                Error::Model(format!(
                    "open checkpoint read source '{}': {error}",
                    extent.path().display()
                ))
            })?;
            let mut file = file;
            file.seek(SeekFrom::Start(extent.offset()))
                .map_err(|error| {
                    Error::Model(format!(
                        "seek checkpoint extent {} in '{}': {error}",
                        extent.offset(),
                        extent.path().display()
                    ))
                })?;
            let mut bytes = vec![0u8; length];
            file.read_exact(&mut bytes).map_err(|error| {
                Error::Model(format!(
                    "read checkpoint extent {}..{} from '{}': {error}",
                    extent.offset(),
                    extent.end(),
                    extent.path().display()
                ))
            })?;
            payloads.push(bytes);
        }
        plan.validate_source_identity()
            .map_err(stale_source_identity_error)?;
        Ok(payloads)
    }
}

fn stale_source_identity_error(reason: StaleReason) -> Error {
    Error::Model(format!("stale checkpoint source identity: {reason:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_preserves_extent_order_and_detects_source_change() {
        let dir = unique_temp_dir("ferrule-checkpoint-read-plan");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        std::fs::write(&path, [0u8; 32]).unwrap();
        let snapshot = CheckpointSourceFileIdentity::capture(&path).unwrap();
        let plan = CheckpointReadPlan::new(
            [
                CheckpointReadExtent::new(path.clone(), 16, 8).unwrap(),
                CheckpointReadExtent::new(path.clone(), 0, 4).unwrap(),
            ],
            Arc::from([snapshot]),
        )
        .unwrap();

        assert_eq!(plan.storage_bytes(), 12);
        assert_eq!(plan.extents()[0].offset(), 16);
        assert_eq!(plan.extents()[1].offset(), 0);
        assert_eq!(plan.validate_source_identity(), Ok(()));

        std::fs::write(&path, [0u8; 64]).unwrap();
        assert_eq!(
            plan.validate_source_identity(),
            Err(StaleReason::SourceIdentityChanged)
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn positioned_reader_preserves_plan_order() {
        let dir = unique_temp_dir("ferrule-checkpoint-positioned-reader");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        std::fs::write(&path, (0u8..32).collect::<Vec<_>>()).unwrap();
        let snapshot = CheckpointSourceFileIdentity::capture(&path).unwrap();
        let plan = CheckpointReadPlan::new(
            [
                CheckpointReadExtent::new(path.clone(), 12, 4).unwrap(),
                CheckpointReadExtent::new(path.clone(), 2, 3).unwrap(),
            ],
            Arc::from([snapshot]),
        )
        .unwrap();

        let payloads = CheckpointPositionedReader::new(8).read(&plan).unwrap();
        assert_eq!(payloads, vec![vec![12, 13, 14, 15], vec![2, 3, 4]]);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn positioned_reader_rejects_stale_source_before_payload_read() {
        let dir = unique_temp_dir("ferrule-checkpoint-positioned-reader-stale");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        std::fs::write(&path, [0u8; 16]).unwrap();
        let snapshot = CheckpointSourceFileIdentity::capture(&path).unwrap();
        let plan = CheckpointReadPlan::new(
            [CheckpointReadExtent::new(path.clone(), 0, 8).unwrap()],
            Arc::from([snapshot]),
        )
        .unwrap();
        std::fs::write(&path, [0u8; 32]).unwrap();

        let error = CheckpointPositionedReader::new(8).read(&plan).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("stale checkpoint source identity")
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn plan_rejects_extent_without_matching_snapshot() {
        let snapshot = CheckpointSourceFileIdentity::for_test(
            PathBuf::from("first.safetensors"),
            PathBuf::from("/test/first.safetensors"),
            32,
        );
        let error = CheckpointReadPlan::new(
            [CheckpointReadExtent::new(PathBuf::from("second.safetensors"), 0, 8).unwrap()],
            Arc::from([snapshot]),
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("has no catalog-time source snapshot")
        );
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{}-{nonce}", std::process::id()))
    }
}
