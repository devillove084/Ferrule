//! The single decoded-arena layout authority for admission and custody.
//! Static quotas guarantee residency for working sets fitting EVERY exact class,
//! not arbitrary sets whose aggregate bytes fit. No CUDA or payload I/O.
#![cfg_attr(not(feature = "cuda"), allow(dead_code))]
use super::invalid;
use super::mixed_decoded::{DecodedDescriptor, HybridMixedDecodedCacheBudget};
use ferrule_common::Result;
use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Deref,
    sync::Arc,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FixedSlot {
    pub descriptor: usize,
    pub offset_words: usize,
    pub elements: usize,
    pub slot_bytes: usize,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SlotClass {
    pub offset_words: usize,
    pub elements: usize,
    pub slot_bytes: usize,
    pub count: usize,
    pub layer_minimum: usize,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DecodedArenaPlan {
    pub arena_grants: usize,
    pub fixed: Vec<FixedSlot>,
    pub classes: Vec<SlotClass>,
    pub routed_slot_bytes: usize,
    pub unused_bytes: usize,
    pub slot_padding_bytes: usize,
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| invalid("decoded arena extent overflow"))
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("decoded arena extent overflow"))
}
impl DecodedArenaPlan {
    pub fn new(budget: HybridMixedDecodedCacheBudget, ds: &[DecodedDescriptor]) -> Result<Self> {
        if HybridMixedDecodedCacheBudget::new(budget.capacity_grants, ds)? != budget {
            return Err(invalid("decoded arena descriptor budget mismatch"));
        }
        let bytes = budget.capacity_grants;
        // >=64 MiB and an exact allocator grant: one huge allocation, not one
        // small standard-segment allocation per view. Rounded root bytes count.
        if bytes < 64 << 20 || ferrule_backend::cuda_allocation_grant_bytes(bytes) != Some(bytes) {
            return Err(invalid(
                "decoded arena requires a grant-aligned capacity of at least 64 MiB",
            ));
        }
        let mut identities = BTreeSet::new();
        let mut fixed = Vec::new();
        let mut offset = 0;
        let mut padding = 0;
        let mut inventory = BTreeMap::<(usize, usize), usize>::new();
        let mut layers = BTreeMap::<(usize, (usize, usize)), usize>::new();
        for (index, d) in ds.iter().enumerate() {
            if !identities.insert(d.identity.clone()) {
                return Err(invalid("duplicate decoded arena identity"));
            }
            let elements = mul(d.identity.geometry[0], d.identity.geometry[1])?;
            let payload = mul(elements, 2)?;
            if elements == 0
                || ferrule_backend::cuda_allocation_grant_bytes(payload) != Some(d.grants)
            {
                return Err(invalid(
                    "decoded arena slot differs from original descriptor extent",
                ));
            }
            if let Some(layer) = d.routed_layer {
                let class = (elements, d.grants);
                let count = inventory.entry(class).or_default();
                *count = add(*count, 1)?;
                let count = layers.entry((layer, class)).or_default();
                *count = add(*count, 1)?;
            } else {
                fixed.push(FixedSlot {
                    descriptor: index,
                    offset_words: offset / 2,
                    elements,
                    slot_bytes: d.grants,
                });
                offset = add(offset, d.grants)?;
                padding = add(padding, d.grants - payload)?;
            }
        }
        if offset != budget.pinned_grants || offset > bytes {
            return Err(invalid("decoded arena fixed slots exceed admission"));
        }
        let mut minima = BTreeMap::<(usize, usize), usize>::new();
        for ((_, class), count) in layers {
            let n = minima.entry(class).or_default();
            *n = (*n).max(count);
        }
        let required = minima
            .iter()
            .try_fold(0, |sum, ((_, slot), &n)| add(sum, mul(*slot, n)?))?;
        let available = bytes - offset;
        if required > available {
            return Err(invalid(
                "decoded arena cannot cover a whole layer for every exact size class",
            ));
        }
        let mut classes = Vec::new();
        let mut assigned = 0;
        for ((elements, slot_bytes), minimum) in minima {
            // Preserve each class's actual per-layer working-set ratio, rather
            // than greedily filling pair slots before down slots. Arithmetic is
            // widened so quota computation cannot overflow before division.
            let quota = ((available as u128) * (minimum as u128) / (required as u128)) as usize;
            let count = quota.min(inventory[&(elements, slot_bytes)]);
            if count < minimum {
                return Err(invalid("decoded arena class quota below layer minimum"));
            }
            assigned = add(assigned, mul(count, slot_bytes)?)?;
            classes.push(SlotClass {
                offset_words: 0,
                elements,
                slot_bytes,
                count,
                layer_minimum: minimum,
            });
        }
        let mut remaining = available - assigned;
        // Redistribute inventory-capped quotas by the same layer ratio. Each
        // pass consumes bytes or stops; this runs only during CPU planning.
        loop {
            let weight = classes
                .iter()
                .filter(|c| {
                    c.count < inventory[&(c.elements, c.slot_bytes)] && c.slot_bytes <= remaining
                })
                .try_fold(0, |sum, c| add(sum, mul(c.layer_minimum, c.slot_bytes)?))?;
            if weight == 0 {
                break;
            }
            let before = remaining;
            for class in &mut classes {
                let room = inventory[&(class.elements, class.slot_bytes)] - class.count;
                if room == 0 || class.slot_bytes > remaining {
                    continue;
                }
                let quota =
                    ((before as u128) * (class.layer_minimum as u128) / (weight as u128)) as usize;
                let extra = quota.max(1).min(room).min(remaining / class.slot_bytes);
                class.count += extra;
                let consumed = mul(extra, class.slot_bytes)?;
                assigned = add(assigned, consumed)?;
                remaining -= consumed;
            }
            if before == remaining {
                break;
            }
        }
        for class in &mut classes {
            class.offset_words = offset / 2;
            offset = add(offset, mul(class.count, class.slot_bytes)?)?;
            padding = add(
                padding,
                mul(class.count, class.slot_bytes - mul(class.elements, 2)?)?,
            )?;
        }
        if offset > bytes {
            return Err(invalid("decoded arena disjoint slots exceed root"));
        }
        Ok(Self {
            arena_grants: bytes,
            fixed,
            classes,
            routed_slot_bytes: assigned,
            unused_bytes: bytes - offset,
            slot_padding_bytes: padding,
        })
    }
}

/// Descriptor metadata paired with the exact layout checked at preparation.
/// Slice access preserves execution's descriptor lookup without exposing a
/// second planner or letting custody silently substitute a different layout.
#[derive(Debug, Clone)]
pub(crate) struct DecodedArenaDescriptors {
    descriptors: Vec<DecodedDescriptor>,
    pub layout: Arc<DecodedArenaPlan>,
}
impl DecodedArenaDescriptors {
    pub fn new(
        budget: HybridMixedDecodedCacheBudget,
        descriptors: Vec<DecodedDescriptor>,
    ) -> Result<Self> {
        let layout = Arc::new(DecodedArenaPlan::new(budget, &descriptors)?);
        Ok(Self {
            descriptors,
            layout,
        })
    }
    pub fn validate(&self, budget: HybridMixedDecodedCacheBudget) -> Result<()> {
        if *self.layout != DecodedArenaPlan::new(budget, &self.descriptors)? {
            return Err(invalid(
                "decoded arena layout differs from prepared descriptors",
            ));
        }
        Ok(())
    }
}
impl Deref for DecodedArenaDescriptors {
    type Target = [DecodedDescriptor];
    fn deref(&self) -> &Self::Target {
        &self.descriptors
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/transformer/mixed_decoded_arena.rs"]
mod tests;
