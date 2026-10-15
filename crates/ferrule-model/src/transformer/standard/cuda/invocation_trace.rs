//! One numeric decoder invocation; no request-level allocation of shared batch cost.
use super::super::super::performance::{CounterSnapshot, ProofTiming, micros};
use super::*;
use std::time::Instant;

pub(super) struct InvocationTrace {
    started: Instant,
    transaction: ExecutionTransactionId,
    rows: usize,
    sequences: usize,
    prefill_rows: usize,
    decode_rows: usize,
    before: CounterSnapshot,
    expert_reads_before: crate::transformer::ExpertReadPipelineStats,
}

impl CudaHybridModule {
    fn performance_snapshot(&self) -> Option<(CounterSnapshot, ExpertCacheStats)> {
        let cuda = self.device.operators().counters();
        let cache = self.operators.expert_cache_stats()?;
        let io = self.materializer.checkpoint_read_counters();
        Some((
            CounterSnapshot {
                kernel_launches: cuda.kernel_launches,
                h2d_copies: cuda.host_to_device_copies,
                h2d_bytes: cuda.host_to_device_bytes,
                d2h_copies: cuda.device_to_host_copies,
                d2h_bytes: cuda.device_to_host_bytes,
                artifact_uploads: cuda.artifact_uploads,
                artifact_upload_bytes: cuda.artifact_upload_bytes,
                allocation_attempts: cuda.device_allocation_attempts,
                allocation_bytes: cuda.device_allocation_bytes,
                stream_syncs: cuda.stream_wide_syncs,
                stream_sync_failures: cuda.stream_wide_sync_failures,
                expert_hits: cache.hits,
                expert_uploads: cache.uploads,
                expert_evictions: cache.evictions,
                expert_misses: cache.misses,
                expert_entry_limit_evictions: cache.entry_limit_evictions,
                expert_byte_limit_evictions: cache.byte_limit_evictions,
                expert_scratch_limit_evictions: cache.scratch_limit_evictions,
                expert_admission_stalls: cache.admission_stalls,
                counters_overflowed: cache.counters_overflowed,
                read_calls: io.read_calls,
                read_bytes: io.bytes_read,
                source_checks: io.source_checks,
                path_metadata_calls: io.path_metadata_calls,
                handle_metadata_calls: io.handle_metadata_calls,
            },
            cache,
        ))
    }

    pub(super) fn begin_performance(
        &mut self,
        transaction: ExecutionTransactionId,
        batch: &PackedDecoderBatch,
    ) {
        if (self.numeric_fp8_precision()
            != Some(crate::transformer::NumericFp8Precision::Bf16ModelMixedF32Accumulate)
            && self.expert_read_pipeline_options().is_none())
            || !tracing::enabled!(target: "ferrule_performance", tracing::Level::DEBUG)
        {
            return;
        }
        // Observation never repairs/replaces a live execution or its custody.
        self.end_performance("superseded");
        let started = Instant::now();
        let Some((before, _)) = self.performance_snapshot() else {
            return;
        };
        let mut prefill_rows = 0;
        let mut decode_rows = 0;
        for sequence in batch.sequences() {
            match sequence.phase() {
                ferrule_common::execution::ForwardPhase::Prefill => {
                    prefill_rows += sequence.query_len()
                }
                ferrule_common::execution::ForwardPhase::Decode => {
                    decode_rows += sequence.query_len()
                }
            }
        }
        self.performance = Some(InvocationTrace {
            started,
            transaction,
            rows: batch.len(),
            sequences: batch.sequences().len(),
            prefill_rows,
            decode_rows,
            before,
            expert_reads_before: self.expert_read_pipeline_stats(),
        });
        self.operators.performance_proofs = Some(ProofTiming::default());
    }

    pub(super) fn end_performance(&mut self, status: &'static str) {
        let Some(invocation) = self.performance.take() else {
            return;
        };
        let host_invocation_us = micros(invocation.started.elapsed());
        let proofs = self.operators.performance_proofs.take().unwrap_or_default();
        let quarantined = self.device.needs_quarantine() || self.operators.needs_quarantine();
        let event = if self.numeric_fp8_precision()
            == Some(crate::transformer::NumericFp8Precision::Bf16ModelMixedF32Accumulate)
        {
            "mixed_decoder_invocation"
        } else {
            "expert_read_decoder_invocation"
        };
        let expert_reads = self.expert_read_pipeline_observation();
        let expert_read_delta = expert_reads
            .stats
            .checked_delta(invocation.expert_reads_before);
        // One bounded serialized diagnostic per invocation, never per expert.
        // The gauges/peaks remain owner-lifetime; only counters are differenced.
        let expert_read_json = serde_json::to_string(&expert_reads).ok();
        let expert_read_delta_json =
            expert_read_delta.and_then(|delta| serde_json::to_string(&delta).ok());
        let Some((after, cache)) = self.performance_snapshot() else {
            tracing::debug!(target: "ferrule_performance", event,
                transaction = invocation.transaction.get(), status, quarantined,
                host_invocation_us, counters_available = false,
                expert_read_pipeline = expert_read_json.as_deref(),
                expert_read_pipeline_delta = expert_read_delta_json.as_deref(),
                expert_read_counter_reset_detected = expert_read_delta.is_none(),
                "Numeric decoder invocation lost its bounded cache observation");
            return;
        };
        let delta = after.delta(invocation.before);
        // These are facade counters, not native kernel counts. The high-water
        // cache gauge is owner-lifetime, not a per-invocation peak or a delta.
        tracing::debug!(target: "ferrule_performance",
            event,
            transaction = invocation.transaction.get(),
            device = self.device.operators().device_ordinal(),
            status, quarantined, host_invocation_us,
            timing_scope = "decoder_validate_through_finish_or_failure_not_device_kernel_or_publication",
            rows = invocation.rows, sequences = invocation.sequences,
            prefill_rows = invocation.prefill_rows, decode_rows = invocation.decode_rows,
            counters_available = true, counter_reset_detected = !delta.valid(),
            expert_read_pipeline = expert_read_json.as_deref(),
            expert_read_pipeline_delta = expert_read_delta_json.as_deref(),
            expert_read_counter_reset_detected = expert_read_delta.is_none(),
            expert_read_peak_scope = "owner_lifetime_not_invocation_delta",
            facade_kernel_launches = delta.kernel_launches,
            h2d_copies = delta.h2d_copies, h2d_bytes = delta.h2d_bytes,
            d2h_copies = delta.d2h_copies, d2h_bytes = delta.d2h_bytes,
            artifact_uploads = delta.artifact_uploads, artifact_upload_bytes = delta.artifact_upload_bytes,
            allocation_attempts = delta.allocation_attempts, allocation_bytes = delta.allocation_bytes,
            facade_stream_syncs = delta.stream_syncs, facade_stream_sync_failures = delta.stream_sync_failures,
            expert_hits = delta.expert_hits, expert_uploads = delta.expert_uploads, expert_evictions = delta.expert_evictions,
            expert_cache_diagnostics_enabled = cache.diagnostics_enabled,
            expert_cache_counters_overflowed = cache.counters_overflowed,
            expert_misses = cache.diagnostics_enabled.then_some(delta.expert_misses).flatten(),
            expert_entry_limit_evictions = cache.diagnostics_enabled.then_some(delta.expert_entry_limit_evictions).flatten(),
            expert_byte_limit_evictions = cache.diagnostics_enabled.then_some(delta.expert_byte_limit_evictions).flatten(),
            expert_scratch_limit_evictions = cache.diagnostics_enabled.then_some(delta.expert_scratch_limit_evictions).flatten(),
            expert_admission_stalls = cache.diagnostics_enabled.then_some(delta.expert_admission_stalls).flatten(),
            expert_miss_history = "unobserved_no_history",
            expert_eviction_selection_scope = "unobserved_no_selected_or_future_hint_at_cache_layer",
            expert_resident_entries = cache.resident_experts, expert_resident_bytes = cache.resident_bytes,
            expert_pending_upload_bytes = cache.pending_upload_bytes, expert_scratch_bytes = cache.scratch_bytes,
            expert_owner_peak_bytes = cache.peak_bytes, expert_unknown_quarantine_bytes = cache.unknown_quarantine_bytes,
            checkpoint_read_calls = delta.read_calls, checkpoint_read_bytes = delta.read_bytes,
            checkpoint_source_checks = delta.source_checks, checkpoint_path_metadata_calls = delta.path_metadata_calls,
            checkpoint_handle_metadata_calls = delta.handle_metadata_calls,
            consumer_event_record_sync_calls = proofs.consumer_event.calls,
            consumer_event_record_sync_failures = proofs.consumer_event.failures,
            consumer_event_record_sync_host_us = micros(proofs.consumer_event.elapsed),
            semantic_compute_sync_calls = proofs.compute_sync.calls,
            semantic_compute_sync_failures = proofs.compute_sync.failures,
            semantic_compute_sync_host_us = micros(proofs.compute_sync.elapsed),
            semantic_upload_sync_calls = proofs.upload_sync.calls,
            semantic_upload_sync_failures = proofs.upload_sync.failures,
            semantic_upload_sync_host_us = micros(proofs.upload_sync.elapsed),
            "Numeric owner-local decoder invocation summary"
        );
    }
}
