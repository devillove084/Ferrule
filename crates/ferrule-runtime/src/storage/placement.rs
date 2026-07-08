//! Placement — where a replica currently resides.
//!
//! Placement describes where a replica **is**, not where it came from. The
//! caller of `ResidencyRequest` provides a precise `Placement` so the manager
//! does not have to guess `device_id` or `memory` kind.

/// Where a replica currently lives.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Placement {
    Local(LocalPlacement),
    Remote(RemotePlacement),
}

impl Placement {
    /// True if this placement is on a local device (GPU VRAM etc.).
    pub fn is_device(&self) -> bool {
        matches!(self, Self::Local(LocalPlacement::Device { .. }))
    }

    /// True if this placement is in host memory.
    pub fn is_host(&self) -> bool {
        matches!(self, Self::Local(LocalPlacement::Host { .. }))
    }

    /// True if this placement is on local disk.
    pub fn is_disk(&self) -> bool {
        matches!(self, Self::Local(LocalPlacement::Disk { .. }))
    }

    /// True if this placement is remote (Phase 5).
    pub fn is_remote(&self) -> bool {
        matches!(self, Self::Remote(_))
    }

    /// Tier rank for eviction priority: 0 = hottest (device), 3 = coldest
    /// (remote). Higher = cheaper to evict.
    pub fn tier_rank(&self) -> u8 {
        match self {
            Self::Local(LocalPlacement::Device { .. }) => 0,
            Self::Local(LocalPlacement::Host { .. }) => 1,
            Self::Local(LocalPlacement::Disk { .. }) => 2,
            Self::Remote(_) => 3,
        }
    }
}

/// Local placement variants.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LocalPlacement {
    /// Accelerator device memory (GPU VRAM, unified memory, etc.).
    Device {
        device_id: u32,
        memory: DeviceMemoryKind,
    },
    /// Host memory (DRAM), optionally pinned for DMA.
    Host { pinned: bool },
    /// Local disk (filesystem, mmap, NVMe).
    Disk { volume: Option<String> },
}

/// Kind of device memory.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DeviceMemoryKind {
    Vram,
    Unified,
    Other(String),
}

/// Remote placement (Phase 5). Sketched now so `Placement` is complete.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RemotePlacement {
    pub endpoint: RemoteEndpoint,
    pub region: Option<String>,
}

/// Network endpoint for a remote placement.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RemoteEndpoint {
    pub scheme: RemoteScheme,
    pub host: String,
    pub port: Option<u16>,
}

/// Transport scheme for remote access.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RemoteScheme {
    /// RDMA-based distributed cache (Mooncake-like).
    Rdma,
    /// HTTP / S3-compatible object store.
    Http,
    /// gRPC LAN cache service.
    Grpc,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_placement_tier_rank() {
        let p = Placement::Local(LocalPlacement::Device {
            device_id: 0,
            memory: DeviceMemoryKind::Vram,
        });
        assert_eq!(p.tier_rank(), 0);
        assert!(p.is_device());
        assert!(!p.is_host());
    }

    #[test]
    fn host_placement_tier_rank() {
        let p = Placement::Local(LocalPlacement::Host { pinned: true });
        assert_eq!(p.tier_rank(), 1);
        assert!(p.is_host());
    }

    #[test]
    fn disk_placement_tier_rank() {
        let p = Placement::Local(LocalPlacement::Disk {
            volume: Some("nvme0".into()),
        });
        assert_eq!(p.tier_rank(), 2);
        assert!(p.is_disk());
    }

    #[test]
    fn remote_placement_tier_rank() {
        let p = Placement::Remote(RemotePlacement {
            endpoint: RemoteEndpoint {
                scheme: RemoteScheme::Rdma,
                host: "10.0.0.1".into(),
                port: Some(8888),
            },
            region: Some("us-east".into()),
        });
        assert_eq!(p.tier_rank(), 3);
        assert!(p.is_remote());
    }

    #[test]
    fn pinned_vs_unpinned_host() {
        let pinned = Placement::Local(LocalPlacement::Host { pinned: true });
        let unpinned = Placement::Local(LocalPlacement::Host { pinned: false });
        assert_ne!(pinned, unpinned);
    }
}
