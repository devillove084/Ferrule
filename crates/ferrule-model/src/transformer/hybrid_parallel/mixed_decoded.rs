#![cfg_attr(not(feature = "cuda"), allow(dead_code))]
//! Internal decoded-weight admission; source evidence is metadata-only.
use super::*;
use crate::TensorRole;
use crate::checkpoint::{CheckpointReadPlan, NumericFp8Read};
use crate::nn::{ParameterId, ParameterResidency};
use crate::transformer::{BoundParameter, NumericFp8RankPlan};
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub struct HybridMixedDecodedCacheBudget {
    /// Hard cache ceiling: backing grants AND cache-induced segment reserve.
    pub capacity_grants: usize,
    pub pinned_grants: usize,
    pub routed_working_set_grants: usize,
    pub routed_entries: usize,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DecodedCacheAdmission {
    pub budget: HybridMixedDecodedCacheBudget,
    pub arena: Arc<super::mixed_decoded_arena::DecodedArenaPlan>,
    pub physical_pages: usize,
    pub device_capacity: usize,
    pub owner_margins: usize,
}
impl DecodedCacheAdmission {
    pub fn validate(&self, pages: usize, total_grants: usize) -> Result<()> {
        if pages != self.physical_pages
            || checked_add(total_grants, self.owner_margins)? > self.device_capacity
        {
            return Err(invalid(
                "decoded cache owner ledger/configuration changed after admission",
            ));
        }
        Ok(())
    }
}
impl HybridMixedDecodedCacheBudget {
    pub(crate) fn new(capacity: usize, descriptors: &[DecodedDescriptor]) -> Result<Self> {
        let mut pinned = 0usize;
        let mut layers = std::collections::BTreeMap::<usize, usize>::new();
        let mut entries = 0;
        for d in descriptors {
            if let Some(layer) = d.routed_layer {
                let bytes = layers.entry(layer).or_default();
                *bytes = checked_add(*bytes, d.grants)?;
                entries += 1;
            } else {
                pinned = checked_add(pinned, d.grants)?;
            }
        }
        // A routed-local synchronous call can borrow every expert in one layer.
        let routed = layers.values().copied().max().unwrap_or(0);
        if capacity == 0 || checked_add(pinned, routed)? > capacity {
            return Err(invalid(
                "decoded cache cannot hold pins plus one routed layer",
            ));
        }
        Ok(Self {
            capacity_grants: capacity,
            pinned_grants: pinned,
            routed_working_set_grants: routed,
            routed_entries: entries,
        })
    }
    pub fn require_capacity(
        self,
        baseline: usize,
        margins: usize,
        capacity: usize,
    ) -> Result<usize> {
        let total = checked_add(checked_add(baseline, margins)?, self.capacity_grants)?;
        if total > capacity {
            return Err(invalid("decoded cache exceeds complete owner ledger"));
        }
        Ok(total)
    }
}
fn checked_add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| invalid("decoded cache grant overflow"))
}
// Clones share immutable Arc extents; comparisons never stat or open a file.
#[derive(Debug, Clone)]
pub(crate) struct DecodedSource(CheckpointReadPlan);
impl PartialEq for DecodedSource {
    fn eq(&self, b: &Self) -> bool {
        self.cmp(b) == Ordering::Equal
    }
}
impl Eq for DecodedSource {}
impl PartialOrd for DecodedSource {
    fn partial_cmp(&self, b: &Self) -> Option<Ordering> {
        Some(self.cmp(b))
    }
}
impl Ord for DecodedSource {
    fn cmp(&self, b: &Self) -> Ordering {
        (self.0.source_files(), self.0.extents()).cmp(&(b.0.source_files(), b.0.extents()))
    }
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct DecodedPart {
    pub slot: usize,
    pub segment: usize,
    parameter: ParameterId,
    role: TensorRole,
    weight: DecodedSource,
    scale: DecodedSource,
    rows: (usize, usize),
    columns: (usize, usize),
}
impl DecodedPart {
    fn new(slot: usize, segment: usize, b: &BoundParameter, r: &NumericFp8Read) -> Self {
        let rows = r.weight_read().rows();
        let cols = r.weight_read().columns();
        Self {
            slot,
            segment,
            parameter: b.id(),
            role: b.role().clone(),
            weight: DecodedSource(r.weight_read().read_plan().clone()),
            scale: DecodedSource(r.scale_read().read_plan().clone()),
            rows: (rows.start, rows.end),
            columns: (cols.start, cols.end),
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct DecodedIdentity {
    pub owner: u32,
    pub geometry: [usize; 2],
    pub parts: Vec<DecodedPart>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DecodedKey {
    pub generation: u64,
    pub identity: Arc<DecodedIdentity>,
}
impl PartialOrd for DecodedKey {
    fn partial_cmp(&self, b: &Self) -> Option<Ordering> {
        Some(self.cmp(b))
    }
}
impl Ord for DecodedKey {
    fn cmp(&self, b: &Self) -> Ordering {
        self.generation.cmp(&b.generation).then_with(|| {
            if Arc::ptr_eq(&self.identity, &b.identity) {
                Ordering::Equal
            } else {
                self.identity.cmp(&b.identity)
            }
        })
    }
}
#[derive(Debug, Clone)]
pub(crate) struct DecodedDescriptor {
    pub identity: Arc<DecodedIdentity>,
    pub grants: usize,
    pub routed_layer: Option<usize>,
    // Combined QKV is one existing FP8 artifact; gate/up is an ordered pair.
    pub operands: Vec<(usize, usize)>,
}
impl DecodedDescriptor {
    pub fn elements(&self) -> usize {
        self.identity.geometry[0] * self.identity.geometry[1]
    }
    pub fn key(&self, generation: u64) -> DecodedKey {
        DecodedKey {
            generation,
            identity: self.identity.clone(),
        }
    }
}
pub(crate) fn decoded_descriptors(
    layout: &Arc<HybridRankLayout>,
    bindings: &[BoundParameter],
) -> Result<Vec<DecodedDescriptor>> {
    let mut result = Vec::new();
    let up_slots = bindings
        .iter()
        .enumerate()
        .filter_map(|(slot, b)| {
            if b.role() != &TensorRole::RoutedExpertUp {
                return None;
            }
            match b.residency() {
                ParameterResidency::Expert { layer, expert } => Some(((*layer, *expert), slot)),
                _ => None,
            }
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    for (slot, b) in bindings.iter().enumerate() {
        if b.numeric_fp8_encoding().is_none()
            || !matches!(b.weight().logical_shape(), [_, _])
            || !super::mixed_estimate::mixed_projection_role(b.role())
            || b.role() == &TensorRole::RoutedExpertUp
        {
            continue;
        }
        let numeric = NumericFp8RankPlan::new(b, layout.clone())?;
        let routed_layer = match b.residency() {
            ParameterResidency::Expert { layer, .. } => Some(*layer),
            _ => None,
        };
        let make = |geometry: [usize; 2], parts, operands| -> Result<DecodedDescriptor> {
            let bytes = geometry[0]
                .checked_mul(geometry[1])
                .and_then(|v| v.checked_mul(2))
                .ok_or_else(|| invalid("decoded geometry overflow"))?;
            let grants = ferrule_backend::cuda_allocation_grant_bytes(bytes)
                .ok_or_else(|| invalid("decoded grant overflow"))?;
            Ok(DecodedDescriptor {
                identity: Arc::new(DecodedIdentity {
                    owner: layout.owner().get(),
                    geometry,
                    parts,
                }),
                grants,
                routed_layer,
                operands,
            })
        };
        if let Some(qkv) = numeric.mixed_gemmex_qkv_layout()? {
            let parts = numeric
                .reads()
                .iter()
                .enumerate()
                .map(|(i, r)| DecodedPart::new(slot, i, b, r))
                .collect();
            result.push(make([qkv.n, qkv.k], parts, vec![(slot, 0)])?);
        } else if b.role() == &TensorRole::RoutedExpertGate {
            let ParameterResidency::Expert { layer, expert } = b.residency() else {
                return Err(invalid("decoded routed gate missing expert identity"));
            };
            let up_slot = *up_slots
                .get(&(*layer, *expert))
                .ok_or_else(|| invalid("decoded gate missing ordered up projection"))?;
            let up = &bindings[up_slot];
            let up_plan = NumericFp8RankPlan::new(up, layout.clone())?;
            if numeric.reads().len() != 1
                || up_plan.reads().len() != 1
                || numeric.reads()[0].local_shape() != up_plan.reads()[0].local_shape()
            {
                return Err(invalid(
                    "decoded routed pair requires matching single segments",
                ));
            }
            let [n, k] = numeric.reads()[0].local_shape();
            result.push(make(
                [
                    n.checked_mul(2)
                        .ok_or_else(|| invalid("decoded pair overflow"))?,
                    k,
                ],
                vec![
                    DecodedPart::new(slot, 0, b, &numeric.reads()[0]),
                    DecodedPart::new(up_slot, 0, up, &up_plan.reads()[0]),
                ],
                vec![(slot, 0), (up_slot, 0)],
            )?);
        } else {
            for (i, r) in numeric.reads().iter().enumerate() {
                result.push(make(
                    r.local_shape(),
                    vec![DecodedPart::new(slot, i, b, r)],
                    vec![(slot, i)],
                )?);
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
#[path = "../../../tests/unit/transformer/mixed_decoded_plan.rs"]
mod tests;
