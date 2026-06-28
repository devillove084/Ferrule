//! Multi-tier tensor storage for the compute graph.
//!
//! Tiers (fastest to slowest):
//!   0. CPU RAM      — mmap zero-copy (default for safetensors)
//!   1. NVMe/disk    — direct I/O on miss
//!   2. HTTP remote   — byte-range requests (Phase 5)
#![allow(unsafe_code)]

use ferrule_core::{Error, Result};
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub enum StorageLocation {
    Ram {
        ptr: *const u8,
        len: usize,
    },
    Disk {
        path: PathBuf,
        offset: u64,
        len: u64,
    },
    Remote {
        url: String,
        offset: u64,
        len: u64,
    },
    Owned(Vec<u8>),
}

unsafe impl Send for StorageLocation {}
unsafe impl Sync for StorageLocation {}

impl StorageLocation {
    pub fn resolve(&self) -> Result<Vec<u8>> {
        match self {
            Self::Owned(v) => Ok(v.clone()),
            Self::Ram { ptr, len } => {
                let slice = unsafe { std::slice::from_raw_parts(*ptr, *len) };
                Ok(slice.to_vec())
            }
            Self::Disk { path, offset, len } => {
                let mut f = std::fs::File::open(path)?;
                f.seek(SeekFrom::Start(*offset))?;
                let mut buf = vec![0u8; *len as usize];
                f.read_exact(&mut buf)?;
                Ok(buf)
            }
            Self::Remote { url, .. } => {
                Err(Error::Internal(format!("Remote tier: not yet: {url}")))
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct WeightCatalogEntry {
    pub name: String,
    pub shape: [usize; 4],
    pub dtype: ferrule_core::QuantType,
    pub location: StorageLocation,
    pub n_bytes: usize,
}

pub struct WeightCatalog {
    pub entries: Vec<WeightCatalogEntry>,
}

impl WeightCatalog {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn add(&mut self, entry: WeightCatalogEntry) {
        self.entries.push(entry);
    }

    pub fn find(&self, name: &str) -> Option<&WeightCatalogEntry> {
        self.entries.iter().find(|e| e.name == name)
    }

    pub fn total_bytes(&self) -> u64 {
        self.entries.iter().map(|e| e.n_bytes as u64).sum()
    }
}

impl Default for WeightCatalog {
    fn default() -> Self {
        Self::new()
    }
}
