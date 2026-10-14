//! Opt-in rank summaries at the existing invocation/layer boundaries.
use super::performance::{CounterSnapshot, ProofTiming, micros};
use super::rank_timing::{PHASES, Phase, PhaseGuard, Sample, Timings};
use super::*;

#[derive(Clone, Copy)]
struct Snapshot {
    elapsed: std::time::Duration,
    counters: CounterSnapshot,
    phases: [Sample; PHASES],
    proofs: [Sample; 4],
    skips: u64,
    route_calls: u64,
    route_bytes: u64,
    retired_bytes: u64,
    retirements: u64,
    decoded_cache: [u64; 4],
    decoded_grants: (u64, u64),
    decoded_backing: [u64; 6],
}
struct Layer {
    index: usize,
    before: Snapshot,
}
pub(super) struct RankPerformance {
    pub timings: Rc<Timings>,
    before: Snapshot,
    layer: Option<Layer>,
    owner: u32,
    transaction: ExecutionTransactionId,
    rows: usize,
    sequences: usize,
    prefill_rows: usize,
    decode_rows: usize,
}
impl CudaStandardDecoderOperators {
    /// Default off. Enable before invocation AND enable ferrule_performance=debug
    /// on the tracing subscriber. Not a numerical observer or completion policy.
    pub fn set_mixed_rank_performance(&mut self, enabled: bool) -> Result<()> {
        if self.rank_performance.is_some() {
            return Err(cuda_error(
                "cannot change profiling during a Mixed invocation",
            ));
        }
        self.rank_performance_enabled = enabled;
        Ok(())
    }
    pub(super) fn begin_rank_performance(
        &mut self,
        transaction: ExecutionTransactionId,
        batch: &crate::decoder::PackedDecoderBatch,
    ) -> bool {
        if !self.rank_performance_enabled
            || self.rank_performance.is_some()
            || self.performance_proofs.is_some()
            || !tracing::enabled!(target: "ferrule_performance", tracing::Level::DEBUG)
        {
            return false;
        }
        let Some(block) = &self.mixed_gqa_block else {
            return false;
        };
        let owner = block.diagnostic_layout().owner().get();
        let timings = Rc::new(Timings::new());
        self.performance_proofs = Some(ProofTiming::default());
        let before = self.rank_snapshot(&timings);
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
        self.rank_performance = Some(RankPerformance {
            timings,
            before,
            layer: None,
            owner,
            transaction,
            rows: batch.len(),
            sequences: batch.sequences().len(),
            prefill_rows,
            decode_rows,
        });
        true
    }
    fn rank_snapshot(&self, timings: &Timings) -> Snapshot {
        let c = self.ops.counters(); // Host facade counters only; no driver query.
        let mut ledger = timings.ledger.borrow_mut();
        let elapsed = timings.started.elapsed();
        let phases = ledger.snapshot(elapsed);
        let proofs = self
            .performance_proofs
            .as_ref()
            .map_or([Sample::default(); 3], |p| {
                [
                    Sample::of(&p.consumer_event),
                    Sample::of(&p.compute_sync),
                    Sample::of(&p.upload_sync),
                ]
            });
        Snapshot {
            elapsed,
            counters: CounterSnapshot {
                kernel_launches: c.kernel_launches,
                h2d_copies: c.host_to_device_copies,
                h2d_bytes: c.host_to_device_bytes,
                d2h_copies: c.device_to_host_copies,
                d2h_bytes: c.device_to_host_bytes,
                artifact_uploads: c.artifact_uploads,
                artifact_upload_bytes: c.artifact_upload_bytes,
                allocation_attempts: c.device_allocation_attempts,
                allocation_bytes: c.device_allocation_bytes,
                stream_syncs: c.stream_wide_syncs,
                stream_sync_failures: c.stream_wide_sync_failures,
                ..Default::default()
            },
            phases,
            proofs: [proofs[0], proofs[1], proofs[2], Sample::of(&ledger.drain)],
            skips: ledger.upload_wait_skips,
            route_calls: ledger.routed_h2d_calls,
            route_bytes: ledger.routed_h2d_bytes,
            retired_bytes: ledger.retired_source_logical_bytes,
            retirements: ledger.retirement_batches,
            decoded_cache: ledger.decoded_cache,
            decoded_grants: self.mixed_decoded_grants(),
            decoded_backing: self.mixed_decoded_backing_counts(),
        }
    }
    pub(super) fn rank_phase(&self, phase: Phase) -> Option<PhaseGuard> {
        PhaseGuard::start(self.rank_performance.as_ref().map(|p| &p.timings), phase)
    }
    pub(super) fn measure_rank_phase<T>(
        &mut self,
        phase: Phase,
        run: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        let guard = self.rank_phase(phase);
        let result = run(self);
        if let Some(guard) = guard {
            guard.finish(result.is_ok());
        }
        result
    }
    pub(super) fn begin_rank_layer(&mut self, index: usize) {
        let Some(p) = &self.rank_performance else {
            return;
        };
        let before = self.rank_snapshot(&p.timings);
        self.rank_performance.as_mut().unwrap().layer = Some(Layer { index, before });
    }
    pub(super) fn end_rank_layer(&mut self, status: &'static str) {
        let Some(layer) = self.rank_performance.as_mut().and_then(|p| p.layer.take()) else {
            return;
        };
        let p = self.rank_performance.as_ref().unwrap();
        self.emit_rank_summary(
            p,
            Some(layer.index),
            status,
            layer.before,
            self.rank_snapshot(&p.timings),
        );
    }
    pub(super) fn end_rank_performance(&mut self, status: &'static str) {
        self.end_rank_layer(status);
        let Some(p) = self.rank_performance.take() else {
            return;
        };
        self.record_mixed_decoded_profile(p.timings.ledger.borrow().decoded_cache);
        self.emit_rank_summary(&p, None, status, p.before, self.rank_snapshot(&p.timings));
        self.performance_proofs = None;
    }
    pub(super) fn observe_route_upload(&self, bytes: usize) {
        if let Some(p) = &self.rank_performance {
            let mut l = p.timings.ledger.borrow_mut();
            l.routed_h2d_calls += 1;
            l.routed_h2d_bytes = l.routed_h2d_bytes.saturating_add(bytes as u64);
        }
    }
    pub(super) fn observe_source_retirement(&self, bytes: u64) {
        if let Some(p) = &self.rank_performance {
            let mut l = p.timings.ledger.borrow_mut();
            l.retired_source_logical_bytes = l.retired_source_logical_bytes.saturating_add(bytes);
            l.retirement_batches += 1;
        }
    }
    pub(super) fn observe_decoded_cache(&self, hit: bool, bytes: usize, before: u64, after: u64) {
        if let Some(p) = &self.rank_performance {
            let mut l = p.timings.ledger.borrow_mut();
            let i = if hit { 0 } else { 1 };
            l.decoded_cache[i] = l.decoded_cache[i].saturating_add(1);
            l.decoded_cache[2] = l.decoded_cache[2].saturating_add(bytes as u64);
            l.decoded_cache[3] = l.decoded_cache[3].saturating_add(after.saturating_sub(before));
        }
    }

    fn emit_rank_summary(
        &self,
        p: &RankPerformance,
        layer: Option<usize>,
        status: &'static str,
        before: Snapshot,
        after: Snapshot,
    ) {
        let d = after.counters.delta(before.counters);
        let phases: [Sample; PHASES] =
            std::array::from_fn(|i| after.phases[i].since(before.phases[i]));
        let proofs: [Sample; 4] = std::array::from_fn(|i| after.proofs[i].since(before.proofs[i]));
        let at = |phase: Phase| phases[phase as usize];
        tracing::debug!(target: "ferrule_performance",
            event = "mixed_rank_performance", schema_version = 1u64,
            scope = if layer.is_some() { "layer" } else { "invocation" },
            owner = p.owner, device = self.ops.device_ordinal(), transaction = p.transaction.get(),
            layer = layer.map(|v| v as u64), status, quarantined = self.needs_quarantine() || self.mixed_gqa_block.as_ref().is_some_and(|b| b.performance_quarantined()),
            rows = p.rows, sequences = p.sequences, prefill_rows = p.prefill_rows, decode_rows = p.decode_rows,
            host_wall_us = micros(after.elapsed.saturating_sub(before.elapsed)),
            diagnostic_observer_enabled = self.diagnostic.is_enabled(),
            timing_scope = "exclusive_host_wall_including_prior_gpu_wait_not_kernel_time_or_publication",
            phase_other_us = micros(at(Phase::Other).elapsed),
            attention_projection_calls = at(Phase::AttentionProjection).calls, attention_projection_host_us = micros(at(Phase::AttentionProjection).elapsed),
            gqa_calls = at(Phase::Gqa).calls, gqa_host_us = micros(at(Phase::Gqa).elapsed),
            gdn_calls = at(Phase::Gdn).calls, gdn_host_us = micros(at(Phase::Gdn).elapsed),
            router_calls = at(Phase::Router).calls, router_host_us = micros(at(Phase::Router).elapsed),
            route_host_read_calls = at(Phase::RouteHostRead).calls, route_host_read_us = micros(at(Phase::RouteHostRead).elapsed),
            route_agreement_calls = at(Phase::RouteAgreement).calls, route_agreement_host_us = micros(at(Phase::RouteAgreement).elapsed),
            shared_ffn_calls = at(Phase::SharedFfn).calls, shared_ffn_host_us = micros(at(Phase::SharedFfn).elapsed),
            dense_ffn_calls = at(Phase::DenseFfn).calls, dense_ffn_host_us = micros(at(Phase::DenseFfn).elapsed),
            routed_ffn_calls = at(Phase::RoutedFfn).calls, routed_ffn_host_us = micros(at(Phase::RoutedFfn).elapsed),
            collective_calls = at(Phase::Collective).calls, collective_host_us = micros(at(Phase::Collective).elapsed),
            head_calls = at(Phase::Head).calls, head_host_us = micros(at(Phase::Head).elapsed),
            retirement_host_us = micros(at(Phase::SourceRetirement).elapsed),
            phase_failures = phases.iter().map(|s| s.failures).sum::<u64>(),
            counter_reset_detected = !d.valid(),
            facade_kernel_launches = d.kernel_launches,
            allocation_attempts = d.allocation_attempts, allocation_requested_bytes = d.allocation_bytes,
            h2d_copies = d.h2d_copies, h2d_bytes = d.h2d_bytes, d2h_copies = d.d2h_copies, d2h_bytes = d.d2h_bytes,
            artifact_uploads = d.artifact_uploads, artifact_upload_bytes = d.artifact_upload_bytes,
            facade_stream_syncs = d.stream_syncs, facade_stream_sync_failures = d.stream_sync_failures,
            consumer_event_calls = proofs[0].calls, consumer_event_failures = proofs[0].failures, consumer_event_host_us = micros(proofs[0].elapsed),
            compute_sync_calls = proofs[1].calls, compute_sync_failures = proofs[1].failures, compute_sync_host_us = micros(proofs[1].elapsed),
            upload_sync_calls = proofs[2].calls, upload_sync_failures = proofs[2].failures, upload_sync_host_us = micros(proofs[2].elapsed),
            boundary_drain_calls = proofs[3].calls, boundary_drain_failures = proofs[3].failures, boundary_drain_host_us = micros(proofs[3].elapsed),
            upload_wait_actual_skips = after.skips.saturating_sub(before.skips),
            route_metadata_upload_calls = after.route_calls.saturating_sub(before.route_calls),
            route_metadata_upload_bytes = after.route_bytes.saturating_sub(before.route_bytes),
            source_retirement_batches = after.retirements.saturating_sub(before.retirements),
            source_retired_logical_bytes = after.retired_bytes.saturating_sub(before.retired_bytes),
            decoded_cache_hits = after.decoded_cache[0].saturating_sub(before.decoded_cache[0]),
            decoded_cache_misses = after.decoded_cache[1].saturating_sub(before.decoded_cache[1]),
            decoded_weight_bytes = after.decoded_cache[2].saturating_sub(before.decoded_cache[2]),
            decoded_cache_evictions = after.decoded_cache[3].saturating_sub(before.decoded_cache[3]),
            decoded_cache_charged_grants = after.decoded_grants.0,
            decoded_cache_peak_grants = after.decoded_grants.1,
            decoded_cache_active_grants = after.decoded_backing[0],
            decoded_cache_pooled_grants = after.decoded_backing[1],
            decoded_cache_allocator_segment_growth = after.decoded_backing[2],
            decoded_cache_segment_slack_reserve = after.decoded_backing[5],
            decoded_cache_backing_reuses = after.decoded_backing[3].saturating_sub(before.decoded_backing[3]),
            decoded_cache_native_fallbacks = after.decoded_backing[4].saturating_sub(before.decoded_backing[4]),
            source_bytes_scope = "numeric_layer_head_payloads_once_per_binding_excludes_external_aliases_not_os_free",
            "Mixed rank host-wall aggregate; proof durations overlap phase durations"
        );
    }
}
