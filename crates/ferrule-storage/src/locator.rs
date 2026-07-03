//! Object locator — where bytes can be fetched from if no replica is resident.
//!
//! Locators are catalog entries, not execution handles. Never dereference a
//! locator on the hot path — use it to drive a transfer.

use std::path::PathBuf;

/// How to fetch an object's bytes when no suitable replica is resident.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ObjectLocator {
    /// Read from a local file at (offset, bytes).
    LocalFile {
        path: PathBuf,
        offset: u64,
        bytes: u64,
    },
    /// Memory-map a local file at (offset, bytes).
    LocalMmap {
        path: PathBuf,
        offset: u64,
        bytes: u64,
    },
    /// Read from a WeightPack chunk.
    WeightPack {
        path: PathBuf,
        chunk: String,
        offset: u64,
        bytes: u64,
    },
    /// Fetch from a remote object store (HTTP range / S3 GET).
    RemoteObject {
        uri: String,
        offset: u64,
        bytes: u64,
    },
    /// Fetch from a remote cache (key-based, Mooncake-like).
    RemoteCache {
        key: String,
        offset: u64,
        bytes: u64,
    },
}

impl ObjectLocator {
    /// How many bytes this locator can provide.
    pub fn bytes(&self) -> u64 {
        match self {
            Self::LocalFile { bytes, .. }
            | Self::LocalMmap { bytes, .. }
            | Self::WeightPack { bytes, .. }
            | Self::RemoteObject { bytes, .. }
            | Self::RemoteCache { bytes, .. } => *bytes,
        }
    }

    /// True if this locator reads from local storage (file, mmap, WeightPack).
    pub fn is_local(&self) -> bool {
        matches!(
            self,
            Self::LocalFile { .. } | Self::LocalMmap { .. } | Self::WeightPack { .. }
        )
    }

    /// True if this locator fetches from a remote source.
    pub fn is_remote(&self) -> bool {
        matches!(self, Self::RemoteObject { .. } | Self::RemoteCache { .. })
    }

    /// True if this locator supports zero-copy mmap (no allocation needed).
    pub fn supports_mmap(&self) -> bool {
        matches!(self, Self::LocalMmap { .. } | Self::WeightPack { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_file_bytes() {
        let loc = ObjectLocator::LocalFile {
            path: PathBuf::from("/data/model.safetensors"),
            offset: 1024,
            bytes: 4096,
        };
        assert_eq!(loc.bytes(), 4096);
        assert!(loc.is_local());
        assert!(!loc.is_remote());
        assert!(!loc.supports_mmap());
    }

    #[test]
    fn local_mmap_supports_mmap() {
        let loc = ObjectLocator::LocalMmap {
            path: PathBuf::from("/data/model.safetensors"),
            offset: 0,
            bytes: 8192,
        };
        assert!(loc.supports_mmap());
        assert!(loc.is_local());
    }

    #[test]
    fn weightpack_supports_mmap() {
        let loc = ObjectLocator::WeightPack {
            path: PathBuf::from("/data/model.qcache"),
            chunk: "layer0_gate".into(),
            offset: 256,
            bytes: 2048,
        };
        assert!(loc.supports_mmap());
        assert!(loc.is_local());
    }

    #[test]
    fn remote_object_is_remote() {
        let loc = ObjectLocator::RemoteObject {
            uri: "s3://bucket/model.safetensors".into(),
            offset: 0,
            bytes: 1_000_000,
        };
        assert!(loc.is_remote());
        assert!(!loc.is_local());
        assert!(!loc.supports_mmap());
    }

    #[test]
    fn remote_cache_with_offset() {
        let loc = ObjectLocator::RemoteCache {
            key: "expert:layer17:expert42".into(),
            offset: 512,
            bytes: 4096,
        };
        assert!(loc.is_remote());
        assert_eq!(loc.bytes(), 4096);
    }
}
