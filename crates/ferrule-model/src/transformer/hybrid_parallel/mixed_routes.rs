//! Exact routed control metadata, never floating-point collective payloads.
use super::*;
use crate::transformer::RouterRoutes;

/// Sealed program site: the data ordinal is the complete routed FFN output,
/// including the locally gated shared branch, never an independently reduced expert.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridMixedRouteSite {
    pub layer: usize,
    pub output_site: u64,
    pub experts: usize,
    pub top_k: usize,
    pub max_rows: usize,
    pub accumulation: HybridRoutedAccumulation,
    pub combine: HybridMoeCombineOrder,
}

/// Ordered row/slot identities and exact F32 weights. The runtime authenticates
/// the transaction and site against its own sealed plan and compares all owners.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridMixedRouteAgreement {
    site: HybridMixedRouteSite,
    rows: usize,
    expert_ids: Vec<usize>,
    weight_bits: Vec<u32>,
}
impl HybridMixedRouteAgreement {
    pub fn new(site: HybridMixedRouteSite, routes: &RouterRoutes) -> Result<Self> {
        if site.experts == 0
            || site.top_k == 0
            || site.top_k > site.experts
            || routes.rows() == 0
            || routes.rows() > site.max_rows
            || routes.top_k() != site.top_k
            || site.combine != HybridMoeCombineOrder::LocalBf16AddThenReduce
        {
            return Err(invalid("Mixed route control geometry/contract mismatch"));
        }
        for row in 0..routes.rows() {
            let (ids, weights) = routes.row(row)?;
            let mut unique = BTreeSet::new();
            if ids
                .iter()
                .any(|id| *id >= site.experts || !unique.insert(*id))
                || weights.iter().any(|w| !w.is_finite() || *w < 0.0)
            {
                return Err(invalid(
                    "Mixed route control invalid expert identity/weight",
                ));
            }
        }
        Ok(Self {
            site,
            rows: routes.rows(),
            expert_ids: routes.expert_ids().to_vec(),
            weight_bits: routes.weights().iter().map(|w| w.to_bits()).collect(),
        })
    }
    pub fn site(&self) -> &HybridMixedRouteSite {
        &self.site
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn expert_ids(&self) -> &[usize] {
        &self.expert_ids
    }
    pub fn weight_bits(&self) -> &[u32] {
        &self.weight_bits
    }
}
