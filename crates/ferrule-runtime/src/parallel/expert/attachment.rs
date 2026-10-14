//! Explicit immutable-owned attachment policy, not a default LRU policy.
//!
//! Ready snapshots are observations of the existing owners, never a second
//! owner-state authority. Dispatch, cancellation, quiescence and custody stay
//! with the existing executor; this attachment only enforces its declared
//! source/residency contract and bounds the model's single shutdown poll.
//!
//! # Composition API
//!
//! These public types are also reexported from `crate::parallel::expert`.
//! `ImmutableOwnedAttachment` and `ImmutableOwnedAttachmentConfig` require
//! the `cuda` feature; `ImmutableOwnedBaseline` is also available on Unix CPU
//! builds. The attachment is owner-local; composition must not require it to
//! implement `Send`.
//!
//! On Unix, `ImmutableOwnedBaseline::processes` and `check_processes` are
//! available even without CUDA for metadata-only residency checks. They do not
//! create a CPU numeric executor. Thread baselines and attachment execution
//! require CUDA; dependency crates do not inherit this crate's test cfg.
//!
//! Configuration and baseline constructors return `ferrule_common::Result`.
//! Capture them under the caller's existing Ready/admission cleanup boundary
//! before consuming a `HybridCudaExpertAdapter`. Wrapping with
//! `ImmutableOwnedAttachment::new` is infallible. Routed execution uses
//! `ferrule_model::transformer::expert_parallel::RoutedSwiGluExecutor`; drain,
//! shutdown and cancellation use `ferrule_model::decoder::HybridCudaRoutedExecutor`.

#[cfg(any(feature = "cuda", test))]
use std::time::{Duration, Instant};

#[cfg(any(feature = "cuda", test))]
use ferrule_common::ParallelRankId;
#[cfg(any(feature = "cuda", test))]
use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::{Error, Result};
#[cfg(any(feature = "cuda", test))]
use ferrule_model::decoder::WorkProgress;
#[cfg(any(feature = "cuda", test))]
use ferrule_model::transformer::expert_parallel::{RoutedSwiGluExecutor, RoutedSwiGluRequest};
#[cfg(any(feature = "cuda", test))]
use ferrule_model::transformer::{OperatorProgress, Rows};

#[cfg(feature = "cuda")]
use super::{ArcHostExpertCache, ExpertOwnerStats, HybridCudaExpertAdapter};
#[cfg(unix)]
use crate::parallel::expert_process::NumericProcessStats;

fn error(message: impl std::fmt::Display) -> Error {
    Error::Execution {
        message: message.to_string(),
    }
}

/// The attached source and the global layer at which a completed forward is
/// checked. The timeout bounds shutdown polling, including failed preparation.
#[cfg(any(feature = "cuda", test))]
#[derive(Debug, Clone, Copy)]
pub struct ImmutableOwnedAttachmentConfig {
    source: ParallelRankId,
    last_layer: usize,
    timeout: Duration,
}

#[cfg(any(feature = "cuda", test))]
impl ImmutableOwnedAttachmentConfig {
    pub fn new(source: ParallelRankId, last_layer: usize, timeout: Duration) -> Result<Self> {
        if timeout.is_zero() || Instant::now().checked_add(timeout).is_none() {
            return Err(error("invalid immutable expert attachment timeout"));
        }
        Ok(Self {
            source,
            last_layer,
            timeout,
        })
    }
}

/// An immutable Ready observation in membership order. Constructors do not
/// establish physical admission or replace the factory's exact Ready checks.
pub struct ImmutableOwnedBaseline {
    kind: Baseline,
}

enum Baseline {
    #[cfg(feature = "cuda")]
    Threads {
        stats: Vec<ExpertOwnerStats>,
        devices: Vec<usize>,
        host: Option<ArcHostExpertCache>,
        ready_host_hits: u64,
    },
    #[cfg(unix)]
    Processes(Vec<NumericProcessStats>),
}

impl ImmutableOwnedBaseline {
    /// `devices` is diagnostic slot order, not another device/placement plan.
    /// The retained Arc preserves the same shared host-image identity.
    #[cfg(feature = "cuda")]
    pub fn threads(
        stats: Vec<ExpertOwnerStats>,
        devices: Vec<usize>,
        host: Option<ArcHostExpertCache>,
        ready_host_hits: u64,
    ) -> Result<Self> {
        if stats.is_empty() || stats.len() != devices.len() {
            return Err(error("EP Ready owner/device count mismatch"));
        }
        check_owner_stats(&stats, &stats)?;
        if let Some(host) = &host {
            check_host_hits(ready_host_hits, host.hits())?;
        }
        Ok(Self {
            kind: Baseline::Threads {
                stats,
                devices,
                host,
                ready_host_hits,
            },
        })
    }

    /// Capture drained numeric process stats in the executor's membership order.
    /// This metadata-only constructor is available on Unix without CUDA.
    #[cfg(unix)]
    pub fn processes(stats: Vec<NumericProcessStats>) -> Result<Self> {
        if stats.is_empty() {
            return Err(error("empty numeric process Ready baseline"));
        }
        check_process_stats(&stats, &stats)?;
        Ok(Self {
            kind: Baseline::Processes(stats),
        })
    }

    /// Check a later drained observation without changing the Ready snapshot.
    #[cfg(unix)]
    pub fn check_processes(&self, after: &[NumericProcessStats]) -> Result<()> {
        match &self.kind {
            Baseline::Processes(before) => check_process_stats(before, after),
            #[cfg(feature = "cuda")]
            Baseline::Threads { .. } => Err(error("not a numeric process baseline")),
        }
    }
}

#[cfg(feature = "cuda")]
fn check_owner_stats(before: &[ExpertOwnerStats], after: &[ExpertOwnerStats]) -> Result<()> {
    if before.len() != after.len() {
        return Err(error("EP owner count changed after forward"));
    }
    for (before, after) in before.iter().zip(after) {
        let cache = after
            .cuda_cache
            .ok_or_else(|| error("EP CUDA stats missing"))?;
        let initial = before
            .cuda_cache
            .ok_or_else(|| error("EP Ready CUDA stats missing"))?;
        if before.rank != after.rank
            || before.owned_experts != after.owned_experts
            || cache.uploads != initial.uploads
            || cache.evictions != 0
            || cache.resident_bytes != initial.resident_bytes
            || cache.resident_experts != initial.resident_experts
            || cache.pending_upload_bytes != 0
            || cache.quarantined
        {
            return Err(error("EP immutable owned residency changed during forward"));
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn check_host_hits(before: u64, after: u64) -> Result<()> {
    if before != after {
        return Err(error(
            "EP unexpectedly rematerialized host experts after Ready",
        ));
    }
    Ok(())
}

#[cfg(unix)]
fn check_process_stats(
    before: &[NumericProcessStats],
    after: &[NumericProcessStats],
) -> Result<()> {
    if before.len() != after.len()
        || before.iter().zip(after).any(|(before, after)| {
            before.pid != after.pid
                || before.owner != after.owner
                || before.slot != after.slot
                || before.device != after.device
                || before.owned_experts != after.owned_experts
                || before.host_payload_bytes != after.host_payload_bytes
                || before.host_read_bytes != after.host_read_bytes
                || before.host_hits != after.host_hits
                || before.resident_experts != after.resident_experts
                || before.resident_bytes != after.resident_bytes
                || before.uploads != after.uploads
                || after.evictions != 0
                || after.pending_upload_bytes != 0
                || after.quarantined
        })
    {
        return Err(error("numeric process owned residency changed"));
    }
    Ok(())
}

// Private substitution seam for host-only policy tests. Applications can only
// attach the existing CUDA adapter, not supply another owner lifecycle.
#[cfg(any(feature = "cuda", test))]
trait AttachmentBackend: RoutedSwiGluExecutor {
    fn drain(&mut self) -> Result<WorkProgress>;
    fn shutdown(&mut self) -> Result<WorkProgress>;
    fn on_error(&mut self, transaction: ExecutionTransactionId);
    fn retain_unknown_shutdown(&mut self);
    #[cfg(feature = "cuda")]
    fn owner_stats(&mut self) -> Result<Vec<ExpertOwnerStats>>;
    #[cfg(unix)]
    fn process_stats(&mut self) -> Result<Vec<NumericProcessStats>>;
}

#[cfg(any(feature = "cuda", test))]
struct Attachment<B> {
    backend: B,
    config: ImmutableOwnedAttachmentConfig,
    baseline: ImmutableOwnedBaseline,
}

#[cfg(any(feature = "cuda", test))]
impl<B: AttachmentBackend> Attachment<B> {
    fn check_baseline(
        &mut self,
        _context: ferrule_model::transformer::expert_parallel::ExpertDispatchContext,
    ) -> Result<()> {
        match &self.baseline.kind {
            #[cfg(feature = "cuda")]
            Baseline::Threads {
                stats: before,
                devices,
                host,
                ready_host_hits,
            } => {
                let after = self.backend.owner_stats()?;
                check_owner_stats(before, &after)?;
                for (slot, (before, after)) in before.iter().zip(&after).enumerate() {
                    let initial = before.cuda_cache.expect("checked Ready stats");
                    let cache = after.cuda_cache.expect("checked live stats");
                    tracing::info!(
                        transaction = _context.transaction.get(),
                        source = self.config.source.get(),
                        layer = _context.layer,
                        owner = after.rank.owner.get(),
                        ordinal = devices[slot],
                        calls = after.calls,
                        tokens = after.tokens,
                        uploads_before = initial.uploads,
                        uploads_after = cache.uploads,
                        evictions = cache.evictions,
                        resident_bytes = cache.resident_bytes,
                        resident_experts = cache.resident_experts,
                        "EP actual drained owner stats"
                    );
                }
                if let Some(host) = host {
                    check_host_hits(*ready_host_hits, host.hits())?;
                    tracing::info!(generation = host.generation(), identity = ?std::sync::Arc::as_ptr(host),
                        hits = host.hits(), "EP unchanged shared host image after forward");
                }
            }
            #[cfg(unix)]
            Baseline::Processes(before) => {
                let after = self.backend.process_stats()?;
                check_process_stats(before, &after)?;
                tracing::info!(?after, "EP actual drained process stats");
            }
        }
        Ok(())
    }

    fn drain(&mut self) -> Result<WorkProgress> {
        self.backend.drain()
    }

    fn on_error(&mut self, transaction: ExecutionTransactionId) {
        self.backend.on_error(transaction);
    }

    fn shutdown(&mut self) -> Result<WorkProgress> {
        let deadline = Instant::now()
            .checked_add(self.config.timeout)
            .ok_or_else(|| error("immutable expert shutdown deadline overflow"))?;
        loop {
            match self.backend.shutdown()? {
                WorkProgress::Pending if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_micros(50));
                }
                WorkProgress::Pending => {
                    // Make expiry permanent in the existing owner authority,
                    // not in a second attachment-local lifecycle flag.
                    self.backend.retain_unknown_shutdown();
                    return Ok(WorkProgress::Unknown);
                }
                progress => return Ok(progress),
            }
        }
    }
}

#[cfg(any(feature = "cuda", test))]
impl<B: AttachmentBackend> RoutedSwiGluExecutor for Attachment<B> {
    fn routed_swiglu(
        &mut self,
        request: RoutedSwiGluRequest<'_>,
        active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> Result<OperatorProgress<Rows>> {
        let context = request.context;
        if context.source_rank != self.config.source {
            return Err(error("EP source identity changed"));
        }
        let result = self.backend.routed_swiglu(request, active);
        if result.is_err() && self.backend.drain()? == WorkProgress::Unknown {
            // The existing adapter retains custody. Unknown cannot become an
            // ordinary recoverable attachment/model error.
            std::panic::panic_any(crate::parallel::data::PanicQuiescence::Unknown);
        }
        let result = result?;
        if context.layer == self.config.last_layer && matches!(result, OperatorProgress::Ready(_)) {
            self.check_baseline(context)?;
        }
        Ok(result)
    }
}

#[cfg(feature = "cuda")]
impl AttachmentBackend for HybridCudaExpertAdapter {
    fn drain(&mut self) -> Result<WorkProgress> {
        ferrule_model::decoder::HybridCudaRoutedExecutor::drain(self)
    }
    fn shutdown(&mut self) -> Result<WorkProgress> {
        ferrule_model::decoder::HybridCudaRoutedExecutor::shutdown(self)
    }
    fn on_error(&mut self, transaction: ExecutionTransactionId) {
        ferrule_model::decoder::HybridCudaRoutedExecutor::on_error(self, transaction);
    }
    fn retain_unknown_shutdown(&mut self) {
        HybridCudaExpertAdapter::retain_unknown_shutdown(self);
    }
    fn owner_stats(&mut self) -> Result<Vec<ExpertOwnerStats>> {
        HybridCudaExpertAdapter::owner_stats(self)
    }
    #[cfg(unix)]
    fn process_stats(&mut self) -> Result<Vec<NumericProcessStats>> {
        HybridCudaExpertAdapter::process_stats(self)
    }
}

/// Opt-in immutable all-owned policy around the existing GPU combine adapter.
/// It adds no pool, transaction registry, reload policy or owner state flags.
#[cfg(feature = "cuda")]
pub struct ImmutableOwnedAttachment {
    inner: Attachment<HybridCudaExpertAdapter>,
}

#[cfg(feature = "cuda")]
impl ImmutableOwnedAttachment {
    /// Configure/capture the Ready baseline before consuming the adapter. This
    /// wrapping step is infallible and cannot hide cleanup behind a new Err path.
    /// Pass the existing executor's `into_hybrid_cuda` result, not the executor
    /// itself. This returns `Self`, not `Result<Self>`, so do not append `?`.
    pub fn new(
        adapter: HybridCudaExpertAdapter,
        config: ImmutableOwnedAttachmentConfig,
        baseline: ImmutableOwnedBaseline,
    ) -> Self {
        Self {
            inner: Attachment {
                backend: adapter,
                config,
                baseline,
            },
        }
    }
}

#[cfg(feature = "cuda")]
impl RoutedSwiGluExecutor for ImmutableOwnedAttachment {
    fn routed_swiglu(
        &mut self,
        request: RoutedSwiGluRequest<'_>,
        active: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> Result<OperatorProgress<Rows>> {
        self.inner.routed_swiglu(request, active)
    }
}

#[cfg(feature = "cuda")]
impl ferrule_model::decoder::HybridCudaRoutedExecutor for ImmutableOwnedAttachment {
    fn drain(&mut self) -> Result<WorkProgress> {
        self.inner.drain()
    }
    fn shutdown(&mut self) -> Result<WorkProgress> {
        self.inner.shutdown()
    }
    fn on_error(&mut self, transaction: ExecutionTransactionId) {
        self.inner.on_error(transaction);
    }
}

#[cfg(all(test, unix))]
#[path = "../../../tests/unit/parallel/expert_attachment.rs"]
mod tests;
