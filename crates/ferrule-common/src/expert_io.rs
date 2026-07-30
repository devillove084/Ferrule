//! Legacy expert-I/O prediction and broker-permit adapters shared by model and runtime.
//!
//! New physical lifecycle, completion, dependency, slab, residency, and retirement
//! code must use [`crate::io_protocol`]. These adapters remain temporarily so the
//! current CPU-default call sites can migrate without duplicating physical work.

use std::fmt;

use crate::{Error, Result};

/// Priority class for one physical expert-I/O operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExpertIoResourceClass {
    Prefetch,
    Prefill,
    Verification,
    Decode,
}

/// Exact resources retained by one physical expert materialization operation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct ExpertIoResourceDemand {
    pub read_slots: u64,
    pub storage_read_bytes: u64,
    pub pinned_host_bytes: u64,
    pub upload_slots: u64,
    pub h2d_bytes: u64,
    pub install_slots: u64,
    pub device_install_bytes: u64,
}

impl ExpertIoResourceDemand {
    pub const fn is_empty(self) -> bool {
        self.read_slots == 0
            && self.storage_read_bytes == 0
            && self.pinned_host_bytes == 0
            && self.upload_slots == 0
            && self.h2d_bytes == 0
            && self.install_slots == 0
            && self.device_install_bytes == 0
    }

    pub fn validate_within(self, capacity: Self, context: &str) -> Result<()> {
        for (name, requested, available) in [
            ("read slots", self.read_slots, capacity.read_slots),
            (
                "storage read bytes",
                self.storage_read_bytes,
                capacity.storage_read_bytes,
            ),
            (
                "pinned host bytes",
                self.pinned_host_bytes,
                capacity.pinned_host_bytes,
            ),
            ("upload slots", self.upload_slots, capacity.upload_slots),
            ("H2D bytes", self.h2d_bytes, capacity.h2d_bytes),
            ("install slots", self.install_slots, capacity.install_slots),
            (
                "device install bytes",
                self.device_install_bytes,
                capacity.device_install_bytes,
            ),
        ] {
            if requested > available {
                return Err(Error::Execution(format!(
                    "{context} {name} demand {requested} exceeds capacity {available}"
                )));
            }
        }
        Ok(())
    }
}

/// Physical capacity registered by one model/hardware adapter.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExpertIoResourceLimits {
    pub capacity: ExpertIoResourceDemand,
    /// Capacity unavailable to prefetch, guaranteeing demand progress.
    pub demand_reserve: ExpertIoResourceDemand,
}

impl ExpertIoResourceLimits {
    pub fn validate(self) -> Result<Self> {
        self.demand_reserve
            .validate_within(self.capacity, "expert-I/O demand reserve")?;
        Ok(self)
    }
}

/// Result of hard physical admission. Temporary unavailability is a normal wait,
/// while invalid or permanently oversized requests are returned as errors.
pub enum ExpertIoResourceAdmission {
    Granted(ExpertIoResourceGrant),
    TemporarilyUnavailable,
}

impl fmt::Debug for ExpertIoResourceAdmission {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Granted(grant) => formatter.debug_tuple("Granted").field(grant).finish(),
            Self::TemporarilyUnavailable => formatter.write_str("TemporarilyUnavailable"),
        }
    }
}

/// Independently releasable physical stage of one expert materialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpertIoResourceStage {
    Read,
    PinnedHost,
    Upload,
    Install,
}

/// Runtime implementation hidden behind an owning, non-cloneable model permit.
pub trait ExpertIoResourcePermit: Send {
    fn promote(&mut self, class: ExpertIoResourceClass) -> Result<()>;
    fn release_stage(&mut self, stage: ExpertIoResourceStage) -> Result<()>;
    fn release(self: Box<Self>) -> Result<()>;
}

/// Owning permit transferred with the physical read/upload continuation.
pub struct ExpertIoResourceGrant {
    operation_id: u64,
    demand: ExpertIoResourceDemand,
    permit: Option<Box<dyn ExpertIoResourcePermit>>,
}

impl fmt::Debug for ExpertIoResourceGrant {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExpertIoResourceGrant")
            .field("operation_id", &self.operation_id)
            .field("demand", &self.demand)
            .field("active", &self.permit.is_some())
            .finish()
    }
}

impl ExpertIoResourceGrant {
    pub fn new(
        operation_id: u64,
        demand: ExpertIoResourceDemand,
        permit: Box<dyn ExpertIoResourcePermit>,
    ) -> Result<Self> {
        if operation_id == 0 {
            return Err(Error::Execution(
                "expert-I/O resource operation ID must be non-zero".into(),
            ));
        }
        Ok(Self {
            operation_id,
            demand,
            permit: Some(permit),
        })
    }

    pub const fn operation_id(&self) -> u64 {
        self.operation_id
    }

    pub const fn demand(&self) -> ExpertIoResourceDemand {
        self.demand
    }

    pub fn promote(&mut self, class: ExpertIoResourceClass) -> Result<()> {
        self.permit
            .as_mut()
            .ok_or_else(|| {
                Error::Execution("expert-I/O resource grant is already released".into())
            })?
            .promote(class)
    }

    pub fn release_stage(&mut self, stage: ExpertIoResourceStage) -> Result<()> {
        self.permit
            .as_mut()
            .ok_or_else(|| {
                Error::Execution("expert-I/O resource grant is already released".into())
            })?
            .release_stage(stage)
    }

    pub fn release(mut self) -> Result<()> {
        self.permit
            .take()
            .ok_or_else(|| {
                Error::Execution("expert-I/O resource grant is already released".into())
            })?
            .release()
    }
}

impl Drop for ExpertIoResourceGrant {
    fn drop(&mut self) {
        if let Some(permit) = self.permit.take() {
            let _ = permit.release();
        }
    }
}

/// Hard-admission service implemented by runtime and injected into model code.
pub trait ExpertIoResourceControl: Send {
    fn try_acquire(
        &mut self,
        owner: u64,
        operation_id: u64,
        class: ExpertIoResourceClass,
        demand: ExpertIoResourceDemand,
    ) -> Result<ExpertIoResourceAdmission>;
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    struct RecordingPermit {
        releases: Arc<Mutex<usize>>,
        promotions: Arc<Mutex<Vec<ExpertIoResourceClass>>>,
    }

    impl ExpertIoResourcePermit for RecordingPermit {
        fn promote(&mut self, class: ExpertIoResourceClass) -> Result<()> {
            self.promotions.lock().unwrap().push(class);
            Ok(())
        }

        fn release_stage(&mut self, _stage: ExpertIoResourceStage) -> Result<()> {
            Ok(())
        }

        fn release(self: Box<Self>) -> Result<()> {
            *self.releases.lock().unwrap() += 1;
            Ok(())
        }
    }

    fn grant(
        releases: Arc<Mutex<usize>>,
        promotions: Arc<Mutex<Vec<ExpertIoResourceClass>>>,
    ) -> ExpertIoResourceGrant {
        ExpertIoResourceGrant::new(
            7,
            ExpertIoResourceDemand {
                read_slots: 1,
                pinned_host_bytes: 4096,
                ..ExpertIoResourceDemand::default()
            },
            Box::new(RecordingPermit {
                releases,
                promotions,
            }),
        )
        .unwrap()
    }

    #[test]
    fn owning_grant_promotes_and_releases_exactly_once() {
        let releases = Arc::new(Mutex::new(0));
        let promotions = Arc::new(Mutex::new(Vec::new()));
        let mut grant = grant(releases.clone(), promotions.clone());

        grant.promote(ExpertIoResourceClass::Decode).unwrap();
        grant.release().unwrap();

        assert_eq!(*releases.lock().unwrap(), 1);
        assert_eq!(
            *promotions.lock().unwrap(),
            vec![ExpertIoResourceClass::Decode]
        );
    }

    #[test]
    fn dropping_grant_releases_its_permit_once() {
        let releases = Arc::new(Mutex::new(0));
        let promotions = Arc::new(Mutex::new(Vec::new()));
        drop(grant(releases.clone(), promotions));
        assert_eq!(*releases.lock().unwrap(), 1);
    }

    #[test]
    fn resource_limits_reject_reserve_larger_than_capacity() {
        let error = ExpertIoResourceLimits {
            capacity: ExpertIoResourceDemand {
                upload_slots: 1,
                ..ExpertIoResourceDemand::default()
            },
            demand_reserve: ExpertIoResourceDemand {
                upload_slots: 2,
                ..ExpertIoResourceDemand::default()
            },
        }
        .validate()
        .unwrap_err();
        assert!(error.to_string().contains("exceeds capacity"));
    }
}
