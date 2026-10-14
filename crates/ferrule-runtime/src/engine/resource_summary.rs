//! Structured shutdown-attempt reporting. No synchronization, ownership transfer,
//! retirement polling, or resource release occurs while building this report.

use ferrule_model::resource_observability::{
    EvidenceCoverage, PhysicalResourceObservation, ResourceEvidence, ResourceObservationSnapshot,
    ResourceSummaryError, resource_summary_json,
};
use serde::Serialize;

use super::{ResidentResourceObservability, ResidentShutdownProgress};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceShutdownOutcome {
    Pending,
    Complete,
    Error,
}

impl ResourceShutdownOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Complete => "complete",
            Self::Error => "error",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BrokerDomainObservation {
    /// Existing ResourceKind name; stable catalog labels, not provider names.
    pub kind: String,
    pub unit: &'static str,
    pub capacity: u64,
    pub execution_reserve: u64,
    pub in_use: u64,
    pub high_water: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResourceSummaryCoverage {
    pub runtime: EvidenceCoverage,
    pub physical: EvidenceCoverage,
    pub observed_broker_domains: usize,
}

/// Versioned, owner-local report at the end of one explicit shutdown attempt.
/// Complete means the lifecycle close returned success, NOT post-Drop release.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResidentResourceShutdownSummary {
    pub schema_version: u32,
    pub attempt: u64,
    pub outcome: ResourceShutdownOutcome,
    pub error: Option<String>,
    pub runtime: ResourceObservationSnapshot,
    pub broker_domains: Vec<ResourceEvidence<BrokerDomainObservation>>,
    pub physical: PhysicalResourceObservation,
    pub coverage: ResourceSummaryCoverage,
}

impl ResidentResourceShutdownSummary {
    pub(crate) fn new(
        attempt: u64,
        result: &crate::Result<ResidentShutdownProgress>,
        snapshot: ResidentResourceObservability,
    ) -> Self {
        let outcome = match result {
            Ok(ResidentShutdownProgress::Pending) => ResourceShutdownOutcome::Pending,
            Ok(ResidentShutdownProgress::Complete(_)) => ResourceShutdownOutcome::Complete,
            Err(_) => ResourceShutdownOutcome::Error,
        };
        let mut runtime_coverage = EvidenceCoverage::default();
        let runtime = &snapshot.runtime;
        macro_rules! fields {
            ($($field:ident),* $(,)?) => { $(runtime_coverage.record(stringify!($field), &runtime.$field);)* };
        }
        fields!(
            resident_bytes,
            pending_operations,
            scratch_bytes,
            quarantine,
            active_credit_grants,
            pending_retirements,
            retired_operations,
            fence,
            closure
        );
        let coverage = ResourceSummaryCoverage {
            runtime: runtime_coverage,
            physical: snapshot.model.coverage(),
            observed_broker_domains: snapshot.credits.len(),
        };
        let broker_domains = snapshot
            .credits
            .into_iter()
            .map(|credit| {
                ResourceEvidence::Observed(BrokerDomainObservation {
                    kind: format!("{:?}", credit.kind),
                    unit: match credit.unit {
                        crate::scheduling::ResourceUnit::Bytes => "bytes",
                        crate::scheduling::ResourceUnit::Operations => "operations",
                        crate::scheduling::ResourceUnit::Slots => "slots",
                    },
                    capacity: credit.capacity,
                    execution_reserve: credit.execution_reserve,
                    in_use: credit.in_use,
                    high_water: credit.high_water,
                })
            })
            .collect();
        Self {
            schema_version: 1,
            attempt,
            outcome,
            error: result.as_ref().err().map(ToString::to_string),
            runtime: snapshot.runtime,
            broker_domains,
            physical: snapshot.model,
            coverage,
        }
    }

    pub fn to_json(&self) -> Result<String, ResourceSummaryError> {
        resource_summary_json(self)
    }

    pub(crate) fn log(&self) {
        match self.to_json() {
            Ok(summary) => tracing::info!(
                target: "ferrule_resource_observability",
                event = "resource_shutdown_summary",
                schema_version = self.schema_version,
                attempt = self.attempt,
                outcome = self.outcome.as_str(),
                summary_json = %summary,
                "Owner-local resource shutdown attempt; not final-owner release"
            ),
            Err(error) => tracing::warn!(
                target: "ferrule_resource_observability",
                event = "resource_shutdown_summary_serialization_error",
                attempt = self.attempt,
                error = %error,
                "Resource summary unavailable; lifecycle result is unchanged"
            ),
        }
    }
}
