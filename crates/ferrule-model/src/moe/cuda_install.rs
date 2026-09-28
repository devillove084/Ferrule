//! Device slot-table submission and completion only. Identity checks, frame
//! custody, poisoning, controller publication and eviction stay in the provider.

use crate::moe::streaming::ExpertId;
use crate::runner::completion_notify_callback;
use ferrule_backend::cuda::operators::moe::{
    CudaComputeStreamAuthority, CudaExpertSlotBinding, CudaExpertSlotInstallTarget,
    CudaExpertSlotInstallTicket, CudaExpertSlotPointers, CudaExpertSlotTable, CudaOperators,
};
use ferrule_common::{CompletionHub, ExpertSlotBinding, Result};

/// A single backend install ticket, with no independent operation identity.
pub(super) struct SlotInstallTicket(CudaExpertSlotInstallTicket);

impl SlotInstallTicket {
    pub(super) fn is_complete(&self) -> Result<bool> {
        self.0.is_complete()
    }

    pub(super) fn complete(self, table: &mut CudaExpertSlotTable) -> Result<CudaExpertSlotBinding> {
        self.0.complete(table)
    }
}

pub(super) fn target(
    consumer: &CudaComputeStreamAuthority,
    previous: Option<(ExpertId, ExpertSlotBinding)>,
) -> Result<CudaExpertSlotInstallTarget> {
    Ok(if let Some((evicted, old)) = previous {
        let consumer_quiescence = consumer.record_event()?;
        CudaExpertSlotInstallTarget::Replacement {
            previous_expert: evicted.expert,
            previous_binding: CudaExpertSlotBinding {
                slot: i32::try_from(old.slot.get()).unwrap_or(-1),
                generation: i32::try_from(old.generation.get()).unwrap_or(-1),
            },
            consumer_quiescence,
        }
    } else {
        CudaExpertSlotInstallTarget::Empty
    })
}

pub(super) fn submit(
    ops: &CudaOperators,
    table: &mut CudaExpertSlotTable,
    target: CudaExpertSlotInstallTarget,
    expert: usize,
    binding: ExpertSlotBinding,
    pointers: CudaExpertSlotPointers,
) -> Result<SlotInstallTicket> {
    ops.submit_expert_slot_install(
        table,
        target,
        expert,
        binding.slot.get(),
        binding.generation.get(),
        pointers,
    )
    .map(SlotInstallTicket)
}

// Called after the provider releases its table lock, as before the split.
pub(super) fn notify(ops: &CudaOperators, completion_hub: &CompletionHub) {
    if ops
        .notify_upload_stream(completion_notify_callback(completion_hub.clone()))
        .is_err()
    {
        completion_hub.notify();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrule_common::{ExpertKey, ExpertSlotGeneration, ExpertSlotId};

    #[test]
    #[ignore = "requires real CUDA; slot metadata rejection and exact-event publication"]
    fn gpu_install_rejection_does_not_publish_or_resubmit_pending_mutation() {
        let ops = CudaOperators::new_on_device(0).expect("real CUDA install required");
        let consumer = ops.compute_stream_authority();
        let mut table = ops.expert_slot_table(2, 1).unwrap();
        let pointers = CudaExpertSlotPointers {
            gate_weight: 16,
            gate_scale: 32,
            up_weight: 48,
            up_scale: 64,
            down_weight: 80,
            down_scale: 96,
        };
        let binding = ExpertSlotBinding {
            key: ExpertKey::new(1, 0, 0),
            slot: ExpertSlotId::new(0),
            generation: ExpertSlotGeneration::new(1),
        };
        let empty = target(&consumer, None).unwrap();
        ops.reset_counters();
        ops.enable_capture_safe();
        let rejected = submit(&ops, &mut table, empty, 0, binding, pointers);
        ops.disable_capture_safe();
        assert!(
            rejected.is_err(),
            "pre-submit rejection must not create a ticket"
        );
        assert!(table.host().binding(0).is_none());
        assert!(!table.is_poisoned());
        assert_eq!(ops.counters().upload_kernel_launches, 0);
        let ticket = submit(
            &ops,
            &mut table,
            target(&consumer, None).unwrap(),
            0,
            binding,
            pointers,
        )
        .unwrap();
        assert_eq!(ops.counters().upload_kernel_launches, 1);
        // A different expert cannot treat an unpublished-but-submitted slot as empty.
        let conflicting = ExpertSlotBinding {
            key: ExpertKey::new(1, 0, 1),
            ..binding
        };
        assert!(
            submit(
                &ops,
                &mut table,
                target(&consumer, None).unwrap(),
                1,
                conflicting,
                pointers
            )
            .is_err()
        );
        assert_eq!(ops.counters().upload_kernel_launches, 1);
        assert!(table.host().binding(0).is_none());
        assert!(table.host().binding(1).is_none());
        assert!(!table.is_poisoned());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !ticket.is_complete().unwrap() {
            assert!(
                std::time::Instant::now() < deadline,
                "install event timeout"
            );
            std::thread::yield_now();
        }
        assert!(table.host().binding(0).is_none());
        let installed = ticket.complete(&mut table).unwrap();
        assert_eq!(table.host().binding(0), Some(installed));
        assert!(table.host().binding(1).is_none());
        assert_eq!(ops.counters().upload_kernel_launches, 1);
        assert_eq!(ops.counters().stream_wide_syncs, 0);
        assert!(!table.is_poisoned());
    }

    #[test]
    #[ignore = "requires a real CUDA device; run this exact slot metadata test explicitly"]
    fn gpu_slot_install_keeps_event_publication_and_replacement_boundaries() {
        let ops = CudaOperators::new_on_device(0).expect("slot install test requires CUDA");
        let consumer_ops = CudaOperators::new_on_device(0).expect("separate consumer owner");
        let consumer = consumer_ops.compute_stream_authority();
        let hub = CompletionHub::new();
        let mut table = ops.expert_slot_table(2, 1).unwrap();
        // Slot metadata only: these addresses are never dereferenced by an expert kernel.
        let pointers = CudaExpertSlotPointers {
            gate_weight: 16,
            gate_scale: 32,
            up_weight: 48,
            up_scale: 64,
            down_weight: 80,
            down_scale: 96,
        };
        let binding = ExpertSlotBinding {
            key: ExpertKey::new(1, 0, 0),
            slot: ExpertSlotId::new(0),
            generation: ExpertSlotGeneration::new(1),
        };
        let ticket = submit(
            &ops,
            &mut table,
            target(&consumer, None).unwrap(),
            0,
            binding,
            pointers,
        )
        .unwrap();
        notify(&ops, &hub);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !ticket.is_complete().unwrap() {
            assert!(
                std::time::Instant::now() < deadline,
                "install completion timeout"
            );
            std::thread::yield_now();
        }
        assert!(
            table.host().binding(0).is_none(),
            "event completion must not publish host binding"
        );
        let first = ticket.complete(&mut table).unwrap();
        assert_eq!(table.host().binding(0), Some(first));
        let replacement = target(&consumer, Some((ExpertId::new(0, 0), binding))).unwrap();
        let next = ExpertSlotBinding {
            key: ExpertKey::new(1, 0, 1),
            generation: ExpertSlotGeneration::new(2),
            ..binding
        };
        let ticket = submit(&ops, &mut table, replacement, 1, next, pointers).unwrap();
        assert_eq!(table.host().binding(0), Some(first));
        assert!(table.host().binding(1).is_none());
        let second = ticket.complete(&mut table).unwrap();
        assert!(table.host().binding(0).is_none());
        assert_eq!(table.host().binding(1), Some(second));
        assert_eq!((second.slot, second.generation), (0, 2));
        assert!(!table.is_poisoned());
    }
}
