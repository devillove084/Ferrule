//! Exact logical host-charge stages, shared by constructor and fault gates.
use super::model::transformer::{
    ExpertReadHostCharge, ExpertReadPipelineObservation, ExpertReadPipelineOptions,
};

pub fn assert_idle_retained(
    observed: &ExpertReadPipelineObservation,
    read: ExpertReadPipelineOptions,
) {
    assert!(observed.enabled && observed.worker_alive);
    assert!(!observed.worker_unknown && !observed.worker_join_pending);
    assert_eq!(observed.status, "idle_retained");
    assert_eq!(observed.limits, Some(read));
    assert_eq!(observed.requested_mode, Some(read.schedule));
    assert_eq!(observed.effective_mode, Some(read.schedule));
    assert!(observed.current_read.is_none() && observed.pending_read.is_none());
    let stats = observed.stats;
    assert_eq!(stats.workers_created, 1);
    assert_eq!(stats.workers_joined, 0);
    assert_eq!(stats.sequences_started, stats.sequences_finished);
    assert_eq!(
        stats.worker_reuses.checked_add(1),
        Some(stats.sequences_started)
    );
    assert_eq!(stats.submitted_reads, stats.completed_reads);
    assert_eq!(
        stats.completed_reads,
        stats.verified_reads + stats.cancelled_reads + stats.failed_reads
    );
    assert!(stats.worker_started_reads <= stats.submitted_reads);
    assert!(stats.io_started_reads <= stats.worker_started_reads);
    assert!(stats.verified_reads <= stats.io_started_reads);
    assert_eq!(
        observed.worker_read_observed,
        stats.io_started_reads != 0 && stats.proved_read_bytes != 0
    );
    assert_eq!(stats.unknown_reads, 0);
    assert_eq!(stats.unknown_bytes, 0);
    assert_eq!(stats.unobserved_terminal_reads, 0);
    assert_eq!(stats.unobserved_reader_intervals, 0);
    assert_eq!(stats.join_pending_sequences, 0);
    assert_eq!(stats.join_failed_sequences, 0);
    assert_eq!(
        observed.current_charge,
        ExpertReadHostCharge {
            metadata_bytes: read.max_metadata_bytes,
            requested_stack_bytes: read.worker_stack_bytes as u64,
            host_charged_bytes: read.max_metadata_bytes + read.worker_stack_bytes as u64,
            ..Default::default()
        }
    );
    assert!(stats.peak_host_charged_bytes >= observed.current_charge.host_charged_bytes);
    assert!(stats.peak_host_charged_bytes <= read.max_host_bytes);
    let payload = read.max_host_bytes - read.max_metadata_bytes - read.worker_stack_bytes as u64;
    assert!(stats.peak_packed_bytes <= payload / 2);
    assert!(stats.peak_artifact_bytes <= payload / 2);
    assert!(stats.peak_payload_bytes <= payload);
    assert_eq!(stats.peak_metadata_bytes, read.max_metadata_bytes);
    assert_eq!(
        stats.peak_requested_stack_bytes,
        read.worker_stack_bytes as u64
    );
}

// Call only after a real last-owner Drop and a passive completed-join witness.
#[allow(dead_code)] // Public constructor snapshots cannot observe after owner Drop.
pub fn assert_final_joined(
    observed: &ExpertReadPipelineObservation,
    idle: &ExpertReadPipelineObservation,
) {
    assert_idle_retained(idle, idle.limits.unwrap());
    assert_eq!(observed.status, "closed");
    assert!(!observed.worker_alive && !observed.worker_join_pending && !observed.worker_unknown);
    assert_eq!(observed.current_charge, ExpertReadHostCharge::default());
    assert!(observed.current_read.is_none() && observed.pending_read.is_none());
    let mut expected_stats = idle.stats;
    expected_stats.workers_joined = 1;
    assert_eq!(observed.stats, expected_stats);
}
