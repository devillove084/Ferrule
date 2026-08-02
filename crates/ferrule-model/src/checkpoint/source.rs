//! Stable source identity for checkpoint-backed materialization bundles.
//!
//! Building a catalog must not read multi-gigabyte tensor payloads. Identity is
//! therefore split into a semantic content descriptor and a catalog-time snapshot
//! of every referenced source file. Physical readers revalidate those snapshots
//! before publication; a changed snapshot produces a new source generation.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ferrule_common::{
    ContentHash, Error, PayloadEncodingId, Result, SourceGeneration, SourceIdentityHash,
};

use crate::checkpoint::CheckpointTensorSlice;
use crate::materialization::ResourceSource;

use super::hash::{CheckpointFileIdentity, Sha256};
use super::{CheckpointReadExtent, CheckpointReadPlan};

/// Canonical path and filesystem identity captured during checkpoint discovery.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CheckpointSourceFileIdentity {
    catalog_path: PathBuf,
    canonical_path: PathBuf,
    file_identity: CheckpointFileIdentity,
}

impl CheckpointSourceFileIdentity {
    pub fn capture(catalog_path: &Path) -> Result<Self> {
        let canonical_path = std::fs::canonicalize(catalog_path).map_err(|error| Error::Model {
            message: format!(
                "canonicalize checkpoint source '{}': {error}",
                catalog_path.display()
            ),
        })?;
        let metadata = std::fs::metadata(&canonical_path).map_err(|error| Error::Model {
            message: format!(
                "read checkpoint source metadata '{}': {error}",
                canonical_path.display()
            ),
        })?;
        let file_identity =
            CheckpointFileIdentity::from_metadata(&metadata).map_err(|error| Error::Model {
                message: format!(
                    "capture checkpoint source identity '{}': {error}",
                    canonical_path.display()
                ),
            })?;
        Ok(Self {
            catalog_path: catalog_path.to_path_buf(),
            canonical_path,
            file_identity,
        })
    }

    pub fn is_current(&self) -> bool {
        Self::capture(&self.catalog_path).is_ok_and(|current| current == *self)
    }

    pub fn catalog_path(&self) -> &Path {
        &self.catalog_path
    }

    pub fn canonical_path(&self) -> &Path {
        &self.canonical_path
    }

    pub const fn length(&self) -> u64 {
        self.file_identity.length()
    }

    fn update_hash(&self, hasher: &mut Sha256) {
        update_hash_path(hasher, &self.canonical_path);
        self.file_identity.update_hash(hasher);
    }

    #[cfg(test)]
    pub(crate) fn for_test(catalog_path: PathBuf, canonical_path: PathBuf, length: u64) -> Self {
        Self {
            catalog_path,
            canonical_path,
            file_identity: CheckpointFileIdentity::for_test(length, 1, 2, Some(3), Some(4)),
        }
    }
}

/// Immutable source identity and filesystem snapshots for one ordered tensor bundle.
///
/// Keeping these values together prevents a physical reader from validating a bundle
/// against snapshots that did not participate in its source identity. Readers must
/// revalidate the snapshots before reading and again before publication.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointBundleSource {
    source: ResourceSource,
    source_files: Arc<[CheckpointSourceFileIdentity]>,
}

impl CheckpointBundleSource {
    pub const fn source(&self) -> ResourceSource {
        self.source
    }

    pub fn source_files(&self) -> &[CheckpointSourceFileIdentity] {
        &self.source_files
    }

    pub fn validate_source_identity(&self) -> bool {
        self.source_files
            .iter()
            .all(CheckpointSourceFileIdentity::is_current)
    }

    pub fn read_plan(&self, tensors: &[CheckpointTensorSlice]) -> Result<CheckpointReadPlan> {
        CheckpointReadPlan::new(
            tensors
                .iter()
                .map(|tensor| {
                    CheckpointReadExtent::new(tensor.path.clone(), tensor.offset, tensor.bytes)
                })
                .collect::<Result<Vec<_>>>()?,
            Arc::clone(&self.source_files),
        )
    }

    pub(crate) fn source_file(&self, path: &Path) -> Option<&CheckpointSourceFileIdentity> {
        self.source_files
            .iter()
            .find(|source_file| source_file.catalog_path() == path)
    }

    #[cfg(test)]
    pub(crate) fn for_test(
        source: ResourceSource,
        source_files: impl IntoIterator<Item = CheckpointSourceFileIdentity>,
    ) -> Self {
        Self {
            source,
            source_files: Arc::from(source_files.into_iter().collect::<Vec<_>>()),
        }
    }
}

/// Metadata snapshots shared by all bundles from one checkpoint inventory.
///
/// Capturing once avoids one metadata syscall per layer/bundle while still doing no
/// payload I/O.
#[derive(Debug, Clone)]
pub struct CheckpointSourceCatalog {
    files: BTreeMap<PathBuf, CheckpointSourceFileIdentity>,
}

impl CheckpointSourceCatalog {
    pub fn capture<'a>(
        tensors: impl IntoIterator<Item = &'a CheckpointTensorSlice>,
    ) -> Result<Self> {
        let mut files = BTreeMap::new();
        for tensor in tensors {
            if !files.contains_key(&tensor.path) {
                files.insert(
                    tensor.path.clone(),
                    CheckpointSourceFileIdentity::capture(&tensor.path)?,
                );
            }
        }
        Ok(Self { files })
    }

    pub fn bundle_source(
        &self,
        domain: &[u8],
        encoding: PayloadEncodingId,
        tensors: &[CheckpointTensorSlice],
    ) -> Result<CheckpointBundleSource> {
        let semantics = tensors
            .iter()
            .map(|tensor| {
                let mut semantic = tensor.role.as_str().as_bytes().to_vec();
                semantic.push(0);
                semantic.extend_from_slice(tensor.name.as_bytes());
                semantic
            })
            .collect::<Vec<_>>();
        let descriptors = tensors
            .iter()
            .zip(&semantics)
            .map(|(tensor, semantic)| {
                let source_file = self.files.get(&tensor.path).ok_or_else(|| Error::Model {
                    message: format!(
                        "checkpoint source catalog has no snapshot for '{}'",
                        tensor.path.display()
                    ),
                })?;
                Ok(CheckpointSourceTensor {
                    semantic,
                    path: &tensor.path,
                    offset: tensor.offset,
                    bytes: tensor.bytes,
                    dtype: tensor.dtype.as_str(),
                    shape: &tensor.shape,
                    source_file,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let source = checkpoint_resource_source(domain, encoding, &descriptors)?;
        let source_files = tensors
            .iter()
            .map(|tensor| {
                self.files
                    .get(&tensor.path)
                    .cloned()
                    .ok_or_else(|| Error::Model {
                        message: format!(
                            "checkpoint source catalog has no snapshot for '{}'",
                            tensor.path.display()
                        ),
                    })
            })
            .collect::<Result<BTreeSet<_>>>()?
            .into_iter()
            .collect::<Vec<_>>();
        Ok(CheckpointBundleSource {
            source,
            source_files: Arc::from(source_files),
        })
    }
}

/// One tensor descriptor consumed by checkpoint bundle identity construction.
///
/// `semantic` is model-neutral binary data supplied by the family adapter (for
/// example a tensor role plus payload part). Paths and offsets are excluded from
/// semantic content and included only in source identity.
#[derive(Clone, Copy)]
pub(crate) struct CheckpointSourceTensor<'a> {
    pub(crate) semantic: &'a [u8],
    pub(crate) path: &'a Path,
    pub(crate) offset: u64,
    pub(crate) bytes: u64,
    pub(crate) dtype: &'a str,
    pub(crate) shape: &'a [usize],
    pub(crate) source_file: &'a CheckpointSourceFileIdentity,
}

/// Build exact source coordinates for an ordered checkpoint tensor bundle.
///
/// The domain must be versioned by the caller. Changing bundle semantics, tensor
/// ordering, or encoding requires a new domain and/or payload encoding ID.
pub(crate) fn checkpoint_resource_source(
    domain: &[u8],
    encoding: PayloadEncodingId,
    tensors: &[CheckpointSourceTensor<'_>],
) -> Result<ResourceSource> {
    if domain.is_empty() {
        return Err(Error::Model {
            message: "checkpoint resource identity domain must be non-empty".into(),
        });
    }
    if tensors.is_empty() {
        return Err(Error::Model {
            message: "checkpoint resource identity requires at least one tensor".into(),
        });
    }

    let mut content = Sha256::new();
    content.update(b"ferrule-checkpoint-semantic-content-v1");
    update_hash_bytes(&mut content, domain);
    update_hash_u64(&mut content, tensors.len() as u64);
    for (order, tensor) in tensors.iter().enumerate() {
        update_hash_u64(&mut content, order as u64);
        update_hash_bytes(&mut content, tensor.semantic);
        update_hash_bytes(&mut content, tensor.dtype.as_bytes());
        update_hash_u64(&mut content, tensor.shape.len() as u64);
        for &dimension in tensor.shape {
            update_hash_u64(&mut content, dimension as u64);
        }
        update_hash_u64(&mut content, tensor.bytes);
    }
    let content_hash_bytes = content.finalize();

    let mut source = Sha256::new();
    source.update(b"ferrule-checkpoint-physical-source-v1");
    update_hash_bytes(&mut source, domain);
    source.update(&content_hash_bytes);
    update_hash_u64(&mut source, tensors.len() as u64);
    for (order, tensor) in tensors.iter().enumerate() {
        update_hash_u64(&mut source, order as u64);
        update_hash_path(&mut source, tensor.path);
        update_hash_u64(&mut source, tensor.offset);
        tensor.source_file.update_hash(&mut source);
    }
    let source_hash_bytes = source.finalize();

    let mut generation_hash = Sha256::new();
    generation_hash.update(b"ferrule-checkpoint-source-generation-v1");
    generation_hash.update(&source_hash_bytes);
    let generation_hash = generation_hash.finalize();
    let mut generation_bytes = [0u8; 8];
    generation_bytes.copy_from_slice(&generation_hash[..8]);
    let generation = u64::from_be_bytes(generation_bytes);
    let generation = SourceGeneration::new(if generation == 0 { 1 } else { generation });

    ResourceSource::new(
        SourceIdentityHash::new(source_hash_bytes),
        ContentHash::new(content_hash_bytes),
        encoding,
        generation,
    )
}

fn update_hash_path(hasher: &mut Sha256, path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        update_hash_bytes(hasher, path.as_os_str().as_bytes());
    }
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        let words = path.as_os_str().encode_wide().collect::<Vec<_>>();
        update_hash_u64(hasher, words.len() as u64);
        for word in words {
            hasher.update(&word.to_be_bytes());
        }
    }
    #[cfg(not(any(unix, windows)))]
    update_hash_bytes(hasher, path.to_string_lossy().as_bytes());
}

fn update_hash_bytes(hasher: &mut Sha256, bytes: &[u8]) {
    update_hash_u64(hasher, bytes.len() as u64);
    hasher.update(bytes);
}

fn update_hash_u64(hasher: &mut Sha256, value: u64) {
    hasher.update(&value.to_be_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundle_source_detects_catalog_file_replacement() {
        let dir = unique_temp_dir("ferrule-checkpoint-bundle-source");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        std::fs::write(&path, [0u8; 16]).unwrap();
        let tensor = CheckpointTensorSlice {
            name: "weight".into(),
            role: crate::TensorRole::OutputHead,
            path: path.clone(),
            offset: 0,
            bytes: 16,
            dtype: crate::checkpoint::CheckpointDType::Bf16,
            shape: vec![2, 4],
        };
        let catalog = CheckpointSourceCatalog::capture([&tensor]).unwrap();
        let bundle = catalog
            .bundle_source(b"test-bundle-v1", PayloadEncodingId::new(7), &[tensor])
            .unwrap();
        assert!(bundle.validate_source_identity());

        std::fs::write(&path, [0u8; 32]).unwrap();
        assert!(!bundle.validate_source_identity());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn source_generation_changes_independently_of_semantic_content() {
        let first_file = CheckpointSourceFileIdentity::for_test(
            PathBuf::from("model.safetensors"),
            PathBuf::from("/first/model.safetensors"),
            128,
        );
        let second_file = CheckpointSourceFileIdentity::for_test(
            PathBuf::from("model.safetensors"),
            PathBuf::from("/second/model.safetensors"),
            128,
        );
        let first = CheckpointSourceTensor {
            semantic: b"output_head:weight",
            path: Path::new("model.safetensors"),
            offset: 16,
            bytes: 32,
            dtype: "BF16",
            shape: &[4, 4],
            source_file: &first_file,
        };
        let second = CheckpointSourceTensor {
            source_file: &second_file,
            ..first
        };

        let first =
            checkpoint_resource_source(b"test-bundle-v1", PayloadEncodingId::new(7), &[first])
                .unwrap();
        let second =
            checkpoint_resource_source(b"test-bundle-v1", PayloadEncodingId::new(7), &[second])
                .unwrap();

        assert_eq!(first.content_hash(), second.content_hash());
        assert_ne!(first.identity(), second.identity());
        assert_ne!(first.generation(), second.generation());
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{}-{nonce}", std::process::id()))
    }
}
