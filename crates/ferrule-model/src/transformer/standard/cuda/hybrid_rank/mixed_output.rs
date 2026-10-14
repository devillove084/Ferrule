//! Validated binding slots and stable route geometry for BF16 expert views.
use ferrule_common::{Error, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

#[derive(Debug, Clone, Copy)]
pub(super) struct RoutedOutputLayout {
    routes: usize,
    width: usize,
    elements: usize,
}

fn invalid() -> Error {
    Error::Model {
        message: "Mixed routed output extent overflow or out of bounds".into(),
    }
}

impl RoutedOutputLayout {
    pub(super) fn new(routes: usize, width: usize) -> Result<Self> {
        if routes == 0 || width == 0 {
            return Err(invalid());
        }
        let elements = routes.checked_mul(width).ok_or_else(invalid)?;
        elements
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(invalid)?;
        Ok(Self {
            routes,
            width,
            elements,
        })
    }

    pub(super) fn elements(self) -> usize {
        self.elements
    }

    pub(super) fn bucket(self, start: usize, rows: usize) -> Result<Range<usize>> {
        let end = start.checked_add(rows).ok_or_else(invalid)?;
        if rows == 0 || end > self.routes {
            return Err(invalid());
        }
        Ok(start * self.width..end * self.width)
    }
}

/// Capability is the existing routed native full-2N pair, not a new profile.
/// Either observer API needs the real split gate/up tensors; filtered observers
/// still choose that path deterministically, before any GPU submission.
pub(super) fn fuse_pair_swiglu(routed_native_pair: bool, observer_enabled: bool) -> bool {
    routed_native_pair && !observer_enabled
}

fn invalid_binding() -> Error {
    Error::Model {
        message: "Mixed routed binding slot/role/generation mismatch or incomplete resident layer"
            .into(),
    }
}

/// Slots are original projection Vec indices, never a reordered cache directory.
pub(super) struct RoutedBindingSlots {
    generation: u64,
    layers: Vec<RoutedLayerSlots>,
}
struct RoutedLayerSlots {
    layer: usize,
    experts: Box<[[usize; 3]]>,
}
#[derive(Clone, Copy)]
pub(super) struct RoutedBindingSlot {
    pub slot: usize,
    pub layer: usize,
    pub expert: usize,
    pub role: usize,
}
impl RoutedBindingSlots {
    pub(super) fn prepare(
        generation: u64,
        projections: usize,
        counts: &BTreeMap<usize, usize>,
        bindings: impl IntoIterator<Item = Result<Option<RoutedBindingSlot>>>,
    ) -> Result<Self> {
        counts
            .len()
            .checked_mul(std::mem::size_of::<RoutedLayerSlots>())
            .ok_or_else(invalid_binding)?;
        let mut layers = Vec::with_capacity(counts.len());
        for (&layer, &count) in counts {
            if count == 0 {
                return Err(invalid_binding());
            }
            count
                .checked_mul(std::mem::size_of::<[usize; 3]>())
                .ok_or_else(invalid_binding)?;
            layers.push(RoutedLayerSlots {
                layer,
                experts: vec![[usize::MAX; 3]; count].into_boxed_slice(),
            });
        }
        let mut seen = BTreeSet::new();
        for binding in bindings {
            let Some(binding) = binding? else { continue };
            if binding.slot >= projections || !seen.insert(binding.slot) {
                return Err(invalid_binding());
            }
            let index = layers
                .binary_search_by_key(&binding.layer, |l| l.layer)
                .map_err(|_| invalid_binding())?;
            let slot = layers[index]
                .experts
                .get_mut(binding.expert)
                .and_then(|slots| slots.get_mut(binding.role))
                .ok_or_else(invalid_binding)?;
            if *slot != usize::MAX {
                return Err(invalid_binding());
            }
            *slot = binding.slot;
        }
        if layers
            .iter()
            .any(|l| l.experts.iter().flatten().any(|&slot| slot == usize::MAX))
        {
            return Err(invalid_binding());
        }
        Ok(Self { generation, layers })
    }

    pub(super) fn get(&self, generation: u64, layer: usize, expert: usize) -> Result<[usize; 3]> {
        if generation != self.generation {
            return Err(invalid_binding());
        }
        let index = self
            .layers
            .binary_search_by_key(&layer, |l| l.layer)
            .map_err(|_| invalid_binding())?;
        self.layers[index]
            .experts
            .get(expert)
            .copied()
            .ok_or_else(invalid_binding)
    }

    pub(super) fn generation(&self) -> u64 {
        self.generation
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RoutedGroup {
    pub expert: usize,
    pub begin: usize,
    pub count: usize,
}
/// One stable route ordering for gather, expert views, down packing and combine.
pub(super) struct RoutedBatchPlan {
    pub layout: RoutedOutputLayout,
    pub groups: Vec<RoutedGroup>,
    pub row_ids: Vec<i32>,
    pub weights: Vec<f32>,
}
impl RoutedBatchPlan {
    pub(super) fn new(
        rows: usize,
        width: usize,
        experts: usize,
        ids: &[usize],
        weights: &[f32],
    ) -> Result<Self> {
        if rows == 0
            || rows > i32::MAX as usize
            || experts == 0
            || ids.len() != weights.len()
            || ids.is_empty()
            || !ids.len().is_multiple_of(rows)
        {
            return Err(invalid());
        }
        let layout = RoutedOutputLayout::new(ids.len(), width)?;
        let per_row = ids.len() / rows;
        let mut buckets = BTreeMap::<usize, (usize, usize, usize)>::new();
        for (index, (&expert, &weight)) in ids.iter().zip(weights).enumerate() {
            let row_begin = index / per_row * per_row;
            if expert >= experts
                || ids[row_begin..index].contains(&expert)
                || !weight.is_finite()
                || weight < 0.
            {
                return Err(invalid());
            }
            buckets.entry(expert).or_default().1 += 1;
        }
        let mut begin = 0;
        for (start, count, next) in buckets.values_mut() {
            *start = begin;
            *next = begin;
            begin += *count;
        }
        let mut row_ids = vec![0; ids.len()];
        let mut ordered_weights = vec![0.; ids.len()];
        for (index, (&expert, &weight)) in ids.iter().zip(weights).enumerate() {
            let (_, _, next) = buckets.get_mut(&expert).ok_or_else(invalid)?;
            row_ids[*next] = (index / per_row) as i32;
            ordered_weights[*next] = weight;
            *next += 1;
        }
        let mut groups = Vec::with_capacity(buckets.len());
        for (expert, (begin, count, _)) in buckets {
            groups.push(RoutedGroup {
                expert,
                begin,
                count,
            });
        }
        Ok(Self {
            layout,
            groups,
            row_ids,
            weights: ordered_weights,
        })
    }
}

/// Bounds of the existing opaque backend API, independent of observer state.
pub(super) fn batch_metadata_supported(rows: usize, routes: usize) -> bool {
    (1..=8).contains(&rows) && routes != 0 && routes <= rows * 8
}
