//! Physical materialization capacity vocabulary shared by model and runtime.
//!
//! This module names the exact resources retained by one physical
//! materialization operation and the capacity limits a model/hardware adapter
//! reports. Admission and custody accounting live in the runtime-owned
//! resource broker; no parallel permit service exists here.

use snafu::Snafu;

pub type MaterializationResourceResult<T> = std::result::Result<T, MaterializationResourceError>;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Snafu)]
pub enum MaterializationResourceError {
    #[snafu(display("{context} {resource} requirement {requested} exceeds capacity {capacity}"))]
    ExceedsCapacity {
        context: &'static str,
        resource: &'static str,
        requested: u64,
        capacity: u64,
    },
    #[snafu(display("physical materialization plan requires non-zero {resource}"))]
    ZeroRequirement { resource: &'static str },
}

/// Exact resources retained by one physical materialization operation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct MaterializationResourceRequirements {
    pub read_slots: u64,
    pub storage_read_bytes: u64,
    pub pinned_host_bytes: u64,
    pub upload_slots: u64,
    pub h2d_bytes: u64,
    pub install_slots: u64,
    pub device_install_bytes: u64,
}

impl MaterializationResourceRequirements {
    pub const fn is_empty(self) -> bool {
        self.read_slots == 0
            && self.storage_read_bytes == 0
            && self.pinned_host_bytes == 0
            && self.upload_slots == 0
            && self.h2d_bytes == 0
            && self.install_slots == 0
            && self.device_install_bytes == 0
    }

    pub fn validate_within(
        self,
        capacity: Self,
        context: &'static str,
    ) -> MaterializationResourceResult<()> {
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
                return Err(MaterializationResourceError::ExceedsCapacity {
                    context,
                    resource: name,
                    requested,
                    capacity: available,
                });
            }
        }
        Ok(())
    }
}

/// Exact transient stage requirements and persistent residency size for one resource.
///
/// Providers freeze this plan before runtime hard admission. Stage requirements may
/// differ across storage, pinned host, transfer, and installed device layouts;
/// `resident_bytes` is retained after publication until eviction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterializationResourcePlan {
    pub requirements: MaterializationResourceRequirements,
    pub resident_bytes: u64,
}

impl MaterializationResourcePlan {
    pub fn uniform_payload(bytes: u64) -> MaterializationResourceResult<Self> {
        Self::new(
            MaterializationResourceRequirements {
                read_slots: 1,
                storage_read_bytes: bytes,
                pinned_host_bytes: bytes,
                upload_slots: 1,
                h2d_bytes: bytes,
                install_slots: 1,
                device_install_bytes: bytes,
            },
            bytes,
        )
    }

    pub fn new(
        requirements: MaterializationResourceRequirements,
        resident_bytes: u64,
    ) -> MaterializationResourceResult<Self> {
        Self {
            requirements,
            resident_bytes,
        }
        .validate()
    }

    pub fn validate(self) -> MaterializationResourceResult<Self> {
        let requirements = [
            ("read slots", self.requirements.read_slots),
            ("storage read bytes", self.requirements.storage_read_bytes),
            ("pinned host bytes", self.requirements.pinned_host_bytes),
            ("upload slots", self.requirements.upload_slots),
            ("H2D bytes", self.requirements.h2d_bytes),
            ("install slots", self.requirements.install_slots),
            (
                "device install bytes",
                self.requirements.device_install_bytes,
            ),
            ("resident bytes", self.resident_bytes),
        ];
        if let Some((resource, _)) = requirements
            .into_iter()
            .find(|(_, requirement)| *requirement == 0)
        {
            return Err(MaterializationResourceError::ZeroRequirement { resource });
        }
        Ok(self)
    }

    pub const fn completion_bytes(self, stage: crate::LoadStage) -> Option<u64> {
        match stage {
            crate::LoadStage::ReadSubmitted => Some(self.requirements.storage_read_bytes),
            crate::LoadStage::UploadSubmitted => Some(self.requirements.h2d_bytes),
            crate::LoadStage::Installing => Some(self.requirements.device_install_bytes),
            _ => None,
        }
    }

    pub const fn transition_cost(self) -> u64 {
        let maximum = if self.requirements.storage_read_bytes > self.requirements.pinned_host_bytes
        {
            self.requirements.storage_read_bytes
        } else {
            self.requirements.pinned_host_bytes
        };
        let maximum = if maximum > self.requirements.h2d_bytes {
            maximum
        } else {
            self.requirements.h2d_bytes
        };
        if maximum > self.requirements.device_install_bytes {
            maximum
        } else {
            self.requirements.device_install_bytes
        }
    }
}

/// Physical capacity registered by one model/hardware adapter.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MaterializationResourceLimits {
    pub capacity: MaterializationResourceRequirements,
    /// Capacity unavailable to speculative prefetch, guaranteeing admitted execution progress.
    pub execution_reserve: MaterializationResourceRequirements,
}

impl MaterializationResourceLimits {
    pub fn validate(self) -> MaterializationResourceResult<Self> {
        self.execution_reserve
            .validate_within(self.capacity, "materialization execution reserve")?;
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resource_plan_preserves_distinct_stage_and_resident_bytes() {
        let plan = MaterializationResourcePlan::new(
            MaterializationResourceRequirements {
                read_slots: 1,
                storage_read_bytes: 8192,
                pinned_host_bytes: 16384,
                upload_slots: 1,
                h2d_bytes: 4096,
                install_slots: 1,
                device_install_bytes: 6144,
            },
            2048,
        )
        .unwrap();

        assert_eq!(
            plan.completion_bytes(crate::LoadStage::ReadSubmitted),
            Some(8192)
        );
        assert_eq!(
            plan.completion_bytes(crate::LoadStage::UploadSubmitted),
            Some(4096)
        );
        assert_eq!(
            plan.completion_bytes(crate::LoadStage::Installing),
            Some(6144)
        );
        assert_eq!(plan.resident_bytes, 2048);
        assert_eq!(plan.transition_cost(), 16384);
    }

    #[test]
    fn resource_limits_reject_reserve_larger_than_capacity() {
        let error = MaterializationResourceLimits {
            capacity: MaterializationResourceRequirements {
                upload_slots: 1,
                ..MaterializationResourceRequirements::default()
            },
            execution_reserve: MaterializationResourceRequirements {
                upload_slots: 2,
                ..MaterializationResourceRequirements::default()
            },
        }
        .validate()
        .unwrap_err();
        assert!(error.to_string().contains("exceeds capacity"));
    }
}
