//! Incremental host admission, not a reservation of the configured cache cap.
use super::{HostExpertCacheError, error, invalid};
use ferrule_common::Result;

/// Additional model allocations relative to the snapshot taken by warm().
/// Defaults to cache-only admission: generic caches cannot infer future runner
/// buffers. Factories must supply their remaining startup ledger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostMemoryBudget {
    pub base_resident_bytes: u64,
    /// Explicit subset of base_resident_bytes already in RSS/cgroup current.
    /// Never infer this credit from total RSS (catalogs/other models also live there).
    pub base_already_resident_bytes: u64,
    /// Largest simultaneously live serial preparation/upload scratch, NOT its sum
    /// over all tensors. Additional to base_resident_bytes.
    pub base_temporary_bytes: u64,
    /// Remaining allocations that truly overlap warm workers. Post-warm base
    /// construction and already allocated base do not belong here.
    pub concurrent_warm_bytes: u64,
    /// OS/runtime/allocator headroom beyond the explicit buffer ledger.
    pub headroom_bytes: u64,
}
impl Default for HostMemoryBudget {
    fn default() -> Self {
        Self {
            base_resident_bytes: 0,
            base_already_resident_bytes: 0,
            base_temporary_bytes: 0,
            concurrent_warm_bytes: 0,
            headroom_bytes: 1 << 30,
        }
    }
}

/// Cache-owned allocations still to be made after the memory snapshot.
/// Existing bound resources and read plans are already charged in RSS/current.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostCacheMemoryPlan {
    pub payload_bytes: u64,
    pub retained_metadata_bytes: u64,
    pub session_metadata_bytes: u64,
    pub worker_staging_bytes: u64,
    pub worker_stack_bytes: u64,
}
fn sum(parts: &[u64]) -> Result<u64> {
    parts.iter().try_fold(0u64, |total, &bytes| {
        total
            .checked_add(bytes)
            .ok_or_else(|| invalid("host memory ledger overflow"))
    })
}
impl HostCacheMemoryPlan {
    pub fn retained_bytes(self) -> Result<u64> {
        sum(&[self.payload_bytes, self.retained_metadata_bytes])
    }
    pub fn peak_bytes(self) -> Result<u64> {
        sum(&[
            self.retained_bytes()?,
            self.session_metadata_bytes,
            self.worker_staging_bytes,
            self.worker_stack_bytes,
        ])
    }
}

/// Values sampled after metadata planning, before the first payload read.
/// Cgroup fields describe the tightest visible limit/high/hierarchical limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostMemorySnapshot {
    pub system_available_bytes: u64,
    pub process_rss_bytes: u64,
    pub cgroup_limit_bytes: Option<u64>,
    pub cgroup_current_bytes: Option<u64>,
}
impl HostMemorySnapshot {
    pub fn available_bytes(self) -> u64 {
        self.cgroup_limit_bytes
            .zip(self.cgroup_current_bytes)
            .map_or(self.system_available_bytes, |(limit, used)| {
                self.system_available_bytes.min(limit.saturating_sub(used))
            })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostMemoryAdmission {
    pub cache: HostCacheMemoryPlan,
    pub base: HostMemoryBudget,
    pub snapshot: HostMemorySnapshot,
    pub warm_incremental_peak_bytes: u64,
    pub model_incremental_peak_bytes: u64,
    pub required_available_bytes: u64,
}
impl HostMemoryBudget {
    pub fn validate(self) -> Result<()> {
        if self.base_already_resident_bytes > self.base_resident_bytes {
            return Err(invalid(
                "already-resident base credit exceeds planned base residency",
            ));
        }
        Ok(())
    }
    /// OS available already excludes RSS/current. Only explicit base ownership
    /// credit removes already allocated planned bytes; never subtract RSS again.
    pub fn admit(
        self,
        cache: HostCacheMemoryPlan,
        snapshot: HostMemorySnapshot,
    ) -> Result<HostMemoryAdmission> {
        self.validate()?;
        let warm = sum(&[cache.peak_bytes()?, self.concurrent_warm_bytes])?;
        let model = sum(&[
            cache.retained_bytes()?,
            self.base_resident_bytes - self.base_already_resident_bytes,
            self.base_temporary_bytes,
        ])?;
        let required = sum(&[warm.max(model), self.headroom_bytes])?;
        let admission = HostMemoryAdmission {
            cache,
            base: self,
            snapshot,
            warm_incremental_peak_bytes: warm,
            model_incremental_peak_bytes: model,
            required_available_bytes: required,
        };
        tracing::info!(
            ?admission,
            available_bytes = snapshot.available_bytes(),
            "host incremental memory admission"
        );
        if required > snapshot.available_bytes() {
            return Err(error(HostExpertCacheError::AvailableMemory {
                required_bytes: required,
                available_bytes: snapshot.available_bytes(),
            }));
        }
        Ok(admission)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const G: u64 = 1 << 30;
    fn cache() -> HostCacheMemoryPlan {
        HostCacheMemoryPlan {
            payload_bytes: 30 * G,
            retained_metadata_bytes: G / 4,
            session_metadata_bytes: G / 8,
            worker_staging_bytes: G / 16,
            worker_stack_bytes: 4 << 20,
        }
    }
    fn snapshot(limit: u64, used: u64) -> HostMemorySnapshot {
        HostMemorySnapshot {
            system_available_bytes: 100 * G,
            process_rss_bytes: used,
            cgroup_limit_bytes: Some(limit),
            cgroup_current_bytes: Some(used),
        }
    }
    #[test]
    fn real_peak_not_capacity_and_base_cannot_hide_behind_a_small_cache_cap() {
        // Cache-only use fits 40 GiB despite a 40 GiB configured cache cap. The
        // old cap+headroom test falsely rejected this available 39 GiB.
        let only = HostMemoryBudget::default();
        assert!(only.admit(cache(), snapshot(40 * G, G)).is_ok());
        assert!(only.admit(cache(), snapshot(32 * G, G)).is_err());
        let factory = HostMemoryBudget {
            base_resident_bytes: 6 * G,
            base_temporary_bytes: 4 * G,
            ..only
        };
        assert!(factory.admit(cache(), snapshot(32 * G, G)).is_err());
        assert!(factory.admit(cache(), snapshot(40 * G, G)).is_err());
        assert!(factory.admit(cache(), snapshot(48 * G, G)).is_ok());
        let mut host_limited = snapshot(48 * G, G);
        host_limited.system_available_bytes = 40 * G;
        assert!(factory.admit(cache(), host_limited).is_err());
    }
    #[test]
    fn sequential_phases_reuse_staging_and_existing_base_is_not_charged_twice() {
        let base = HostMemoryBudget {
            base_resident_bytes: 6 * G,
            base_temporary_bytes: 4 * G,
            ..Default::default()
        };
        let before = base.admit(cache(), snapshot(48 * G, G)).unwrap();
        let after = HostMemoryBudget {
            base_already_resident_bytes: 6 * G,
            ..base
        }
        .admit(cache(), snapshot(48 * G, 7 * G))
        .unwrap();
        assert_eq!(
            before.required_available_bytes - after.required_available_bytes,
            6 * G
        );
        assert_eq!(
            before.snapshot.available_bytes() - after.snapshot.available_bytes(),
            6 * G
        );
        assert_eq!(
            before.required_available_bytes,
            cache().retained_bytes().unwrap() + 11 * G
        );
        // A truly concurrent load is charged in the warm phase, not hidden by max.
        let concurrent = HostMemoryBudget {
            concurrent_warm_bytes: 12 * G,
            ..base
        }
        .admit(cache(), snapshot(64 * G, G))
        .unwrap();
        assert_eq!(
            concurrent.required_available_bytes,
            cache().peak_bytes().unwrap() + 13 * G
        );
        // RSS is observability, NOT arbitrary model-memory credit.
        let mut high_rss = snapshot(48 * G, G);
        high_rss.process_rss_bytes = 20 * G;
        assert_eq!(
            base.admit(cache(), high_rss)
                .unwrap()
                .required_available_bytes,
            before.required_available_bytes
        );
        assert!(
            HostMemoryBudget {
                base_already_resident_bytes: 7 * G,
                ..base
            }
            .admit(cache(), high_rss)
            .is_err()
        );
    }
    #[test]
    fn overflow_is_not_an_unbounded_or_wrapped_admission() {
        assert!(
            HostMemoryBudget {
                base_temporary_bytes: u64::MAX,
                ..Default::default()
            }
            .admit(cache(), snapshot(u64::MAX, 0))
            .is_err()
        );
    }
}
