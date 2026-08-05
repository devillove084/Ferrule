//! Device-budget planning for routed experts.
//!
//! The planner is deliberately independent of CUDA and materialization. It turns
//! an exact post-static-image memory snapshot plus immutable expert frame sizes
//! into one capacity per execution layer. Execution uses the same materialization
//! protocol regardless of whether every expert fits.

use snafu::Snafu;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ExpertLayerInventory {
    pub execution_layer: usize,
    pub expert_count: usize,
    pub frame_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DeviceResidencyPlan {
    pub free_device_bytes: u64,
    pub reserved_dynamic_bytes: u64,
    pub expert_budget_bytes: u64,
    pub planned_expert_bytes: u64,
    pub layer_slot_capacities: Vec<(usize, usize)>,
    pub fully_resident: bool,
}

#[derive(Debug, Snafu, PartialEq, Eq)]
pub(crate) enum DeviceResidencyPlanError {
    #[snafu(display("routed-expert residency inventory is empty"))]
    EmptyInventory,
    #[snafu(display("duplicate routed-expert execution layer {execution_layer}"))]
    DuplicateLayer { execution_layer: usize },
    #[snafu(display(
        "routed-expert execution layer {execution_layer} has invalid inventory: experts={expert_count}, frame_bytes={frame_bytes}"
    ))]
    InvalidLayer {
        execution_layer: usize,
        expert_count: usize,
        frame_bytes: u64,
    },
    #[snafu(display(
        "minimum routed-expert slots {minimum_slots} exceed layer {execution_layer} expert count {expert_count}"
    ))]
    MinimumExceedsLayer {
        execution_layer: usize,
        minimum_slots: usize,
        expert_count: usize,
    },
    #[snafu(display(
        "device memory after static image ({free_device_bytes} bytes) is below the explicit dynamic reserve ({reserved_dynamic_bytes} bytes)"
    ))]
    ReserveExceedsFree {
        free_device_bytes: u64,
        reserved_dynamic_bytes: u64,
    },
    #[snafu(display("routed-expert residency byte calculation overflowed"))]
    ByteOverflow,
    #[snafu(display("routed-expert warmup slot count overflowed"))]
    WarmupSlotCountOverflow,
    #[snafu(display("routed-expert warmup layer {layer} exceeds usize"))]
    WarmupLayerOutOfRange { layer: u32 },
    #[snafu(display(
        "routed-expert warmup source references unplanned execution layer {execution_layer}"
    ))]
    WarmupUnplannedLayer { execution_layer: usize },
    #[snafu(display(
        "routed-expert warmup cannot fill {missing_slots} planned slots for execution layer {execution_layer}"
    ))]
    WarmupCapacityUnfilled {
        execution_layer: usize,
        missing_slots: usize,
    },
    #[snafu(display(
        "routed-expert budget {expert_budget_bytes} bytes cannot provide the minimum {minimum_slots} slots per layer; {minimum_required_bytes} bytes are required"
    ))]
    MinimumDoesNotFit {
        expert_budget_bytes: u64,
        minimum_required_bytes: u64,
        minimum_slots: usize,
    },
}

pub(crate) fn plan_device_residency(
    free_device_bytes: u64,
    reserved_dynamic_bytes: u64,
    minimum_slots: usize,
    manual_slot_cap: Option<usize>,
    mut layers: Vec<ExpertLayerInventory>,
) -> Result<DeviceResidencyPlan, DeviceResidencyPlanError> {
    if layers.is_empty() {
        return Err(DeviceResidencyPlanError::EmptyInventory);
    }
    layers.sort_unstable_by_key(|layer| layer.execution_layer);
    for window in layers.windows(2) {
        if window[0].execution_layer == window[1].execution_layer {
            return Err(DeviceResidencyPlanError::DuplicateLayer {
                execution_layer: window[0].execution_layer,
            });
        }
    }
    for layer in &layers {
        if layer.expert_count == 0 || layer.frame_bytes == 0 {
            return Err(DeviceResidencyPlanError::InvalidLayer {
                execution_layer: layer.execution_layer,
                expert_count: layer.expert_count,
                frame_bytes: layer.frame_bytes,
            });
        }
        if minimum_slots > layer.expert_count {
            return Err(DeviceResidencyPlanError::MinimumExceedsLayer {
                execution_layer: layer.execution_layer,
                minimum_slots,
                expert_count: layer.expert_count,
            });
        }
    }

    let expert_budget_bytes = free_device_bytes
        .checked_sub(reserved_dynamic_bytes)
        .ok_or(DeviceResidencyPlanError::ReserveExceedsFree {
            free_device_bytes,
            reserved_dynamic_bytes,
        })?;
    let minimum_required_bytes = layers.iter().try_fold(0u64, |total, layer| {
        let slots =
            u64::try_from(minimum_slots).map_err(|_| DeviceResidencyPlanError::ByteOverflow)?;
        let bytes = layer
            .frame_bytes
            .checked_mul(slots)
            .ok_or(DeviceResidencyPlanError::ByteOverflow)?;
        total
            .checked_add(bytes)
            .ok_or(DeviceResidencyPlanError::ByteOverflow)
    })?;
    if minimum_required_bytes > expert_budget_bytes {
        return Err(DeviceResidencyPlanError::MinimumDoesNotFit {
            expert_budget_bytes,
            minimum_required_bytes,
            minimum_slots,
        });
    }

    let targets = layers
        .iter()
        .map(|layer| {
            manual_slot_cap
                .unwrap_or(layer.expert_count)
                .max(minimum_slots)
                .min(layer.expert_count)
        })
        .collect::<Vec<_>>();
    let mut capacities = vec![minimum_slots; layers.len()];
    let mut planned_expert_bytes = minimum_required_bytes;

    // Add one slot per layer per pass. This keeps partial-residency plans fair even
    // when execution layers have different frame sizes and avoids privileging low
    // layer numbers with an arbitrary startup hotset.
    loop {
        let mut progressed = false;
        for (index, layer) in layers.iter().enumerate() {
            if capacities[index] >= targets[index] {
                continue;
            }
            let Some(next_bytes) = planned_expert_bytes.checked_add(layer.frame_bytes) else {
                return Err(DeviceResidencyPlanError::ByteOverflow);
            };
            if next_bytes <= expert_budget_bytes {
                capacities[index] += 1;
                planned_expert_bytes = next_bytes;
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
    }

    let fully_resident = capacities
        .iter()
        .zip(&layers)
        .all(|(capacity, layer)| *capacity == layer.expert_count);
    Ok(DeviceResidencyPlan {
        free_device_bytes,
        reserved_dynamic_bytes,
        expert_budget_bytes,
        planned_expert_bytes,
        layer_slot_capacities: layers
            .into_iter()
            .zip(capacities)
            .map(|(layer, capacity)| (layer.execution_layer, capacity))
            .collect(),
        fully_resident,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layers() -> Vec<ExpertLayerInventory> {
        vec![
            ExpertLayerInventory {
                execution_layer: 0,
                expert_count: 4,
                frame_bytes: 10,
            },
            ExpertLayerInventory {
                execution_layer: 1,
                expert_count: 4,
                frame_bytes: 10,
            },
        ]
    }

    #[test]
    fn full_fit_uses_every_slot_without_consuming_reserve() {
        let plan = plan_device_residency(100, 20, 1, None, layers()).unwrap();
        assert_eq!(plan.expert_budget_bytes, 80);
        assert_eq!(plan.planned_expert_bytes, 80);
        assert_eq!(plan.layer_slot_capacities, vec![(0, 4), (1, 4)]);
        assert!(plan.fully_resident);
        assert_eq!(plan.free_device_bytes - plan.planned_expert_bytes, 20);
    }

    #[test]
    fn partial_fit_distributes_slots_fairly() {
        let plan = plan_device_residency(70, 20, 1, None, layers()).unwrap();
        assert_eq!(plan.layer_slot_capacities, vec![(0, 3), (1, 2)]);
        assert_eq!(plan.planned_expert_bytes, 50);
        assert!(!plan.fully_resident);
    }

    #[test]
    fn manual_cap_bounds_each_layer() {
        let plan = plan_device_residency(100, 20, 1, Some(2), layers()).unwrap();
        assert_eq!(plan.layer_slot_capacities, vec![(0, 2), (1, 2)]);
        assert_eq!(plan.planned_expert_bytes, 40);
        assert!(!plan.fully_resident);
    }

    #[test]
    fn below_minimum_is_an_explicit_error() {
        let error = plan_device_residency(39, 20, 1, None, layers()).unwrap_err();
        assert_eq!(
            error,
            DeviceResidencyPlanError::MinimumDoesNotFit {
                expert_budget_bytes: 19,
                minimum_required_bytes: 20,
                minimum_slots: 1,
            }
        );
    }

    #[test]
    fn reserve_larger_than_free_is_an_explicit_error() {
        let error = plan_device_residency(19, 20, 1, None, layers()).unwrap_err();
        assert_eq!(
            error,
            DeviceResidencyPlanError::ReserveExceedsFree {
                free_device_bytes: 19,
                reserved_dynamic_bytes: 20,
            }
        );
    }

    #[test]
    fn byte_overflow_is_rejected() {
        let error = plan_device_residency(
            u64::MAX,
            0,
            2,
            None,
            vec![ExpertLayerInventory {
                execution_layer: 0,
                expert_count: 2,
                frame_bytes: u64::MAX,
            }],
        )
        .unwrap_err();
        assert_eq!(error, DeviceResidencyPlanError::ByteOverflow);
    }
}
