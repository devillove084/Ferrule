//! Generation-owned pageable compressed experts, not another streaming/LRU tier.
//!
//! A full prewarm is an atomic publication of incrementally read FP8 + scale
//! proofs. No F32 expansion, pinned allocation, or shell/script dependency. The
//! legacy streaming cache is not used by this standard state-dict path.

mod budget;
mod memory;
mod startup;
#[cfg(test)]
mod tests;
pub use budget::{HostCacheMemoryPlan, HostMemoryAdmission, HostMemoryBudget, HostMemorySnapshot};
pub use startup::StandardHostStartupMemory;

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::{BoundParameter, BoundStateDict, PreparedDecoderGeneration, PreparedParameter};
use crate::checkpoint::{
    CheckpointReadCounters, CheckpointReadPlan, CheckpointSourceFileIdentity,
    CheckpointTensorReader, NumericFp8Read,
};
use crate::nn::{ParameterId, ParameterResidency};
use crate::support::TensorRole;
use ferrule_common::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertPrewarmMode {
    /// All selected routed experts must fit and finish before readiness.
    Full,
    /// Explicit opt-out: no host retention, existing bounded demand reads.
    Lazy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostExpertCacheOptions {
    pub mode: ExpertPrewarmMode,
    /// Whole experts across all layers, not individual projections.
    pub max_experts: usize,
    /// Hard admission cap: compressed payload + worker staging + metadata reserve.
    pub max_bytes: u64,
    /// Bounded blocking pread/validation workers (1..=8).
    pub workers: usize,
    /// Defaults to cache-only incremental admission. A model factory must add
    /// the remaining base preparation peak; max_bytes is only a cache cap.
    pub memory_budget: HostMemoryBudget,
}

impl Default for HostExpertCacheOptions {
    fn default() -> Self {
        Self {
            mode: ExpertPrewarmMode::Full,
            max_experts: 40 * 256,
            max_bytes: 40u64 << 30,
            workers: 4,
            memory_budget: HostMemoryBudget::default(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum HostExpertCacheError {
    #[error(
        "host expert prewarm requires {required_bytes} bytes and {required_experts} experts; budget is {budget_bytes} bytes / {budget_experts} experts; increase --expert-host-cache-mb/entries or explicitly use --expert-prewarm lazy"
    )]
    Budget {
        required_bytes: u64,
        required_experts: usize,
        budget_bytes: u64,
        budget_experts: usize,
    },
    #[error(
        "host expert prewarm requires {required_bytes} available bytes including safety reserve, only {available_bytes} available after cgroup limits; use --expert-prewarm lazy to opt out"
    )]
    AvailableMemory {
        required_bytes: u64,
        available_bytes: u64,
    },
    #[error(
        "cannot establish available host/cgroup memory: {reason}; use explicit lazy mode on unsupported hosts"
    )]
    MemoryProbe { reason: String },
    #[error("host expert cache: {reason}")]
    Invalid { reason: String },
    #[error("host expert source changed: {path}")]
    SourceChanged { path: PathBuf },
    #[error(
        "host expert prewarm worker panicked; all workers joined and unpublished payloads reclaimed"
    )]
    WorkerPanicked,
    #[error("cannot spawn host expert prewarm worker: {source}")]
    WorkerSpawn { source: std::io::Error },
}

fn error(source: HostExpertCacheError) -> Error {
    Error::ModelSource {
        source: Box::new(source),
    }
}
fn invalid(reason: impl Into<String>) -> Error {
    error(HostExpertCacheError::Invalid {
        reason: reason.into(),
    })
}

impl HostExpertCacheOptions {
    pub fn validate(self) -> Result<()> {
        if !(1..=8).contains(&self.workers) {
            return Err(invalid("prewarm workers must be in 1..=8"));
        }
        if self.mode == ExpertPrewarmMode::Full && (self.max_bytes == 0 || self.max_experts == 0) {
            return Err(invalid(
                "full prewarm requires nonzero host byte and entry caps",
            ));
        }
        self.memory_budget.validate()
    }
}

#[derive(Debug)]
struct ParameterRead {
    binding: BoundParameter,
    read: NumericFp8Read,
}

/// Metadata-only admission. No source stat, payload read, CUDA, or large allocation.
#[derive(Debug)]
pub struct HostExpertWarmPlan {
    options: HostExpertCacheOptions,
    parameters: Vec<ParameterRead>,
    sources: Arc<[CheckpointSourceFileIdentity]>,
    experts: usize,
    payload_bytes: u64,
    required_bytes: u64,
    memory: HostCacheMemoryPlan,
}

impl HostExpertWarmPlan {
    pub fn new(
        state_dict: &BoundStateDict,
        options: HostExpertCacheOptions,
        max_parameter_bytes: u64,
    ) -> Result<Self> {
        options.validate()?;
        if options.mode != ExpertPrewarmMode::Full {
            return Err(invalid("a warm plan requires full mode"));
        }
        let reader = CheckpointTensorReader::new(max_parameter_bytes);
        let mut parameters = Vec::new();
        let mut sources = BTreeMap::new();
        let mut groups = BTreeMap::<(usize, usize), BTreeSet<usize>>::new();
        let mut payload_bytes = 0u64;
        let mut largest = 0u64;
        let mut ids = BTreeSet::new();
        for binding in state_dict.parameters() {
            let ParameterResidency::Expert { layer, expert } = *binding.residency() else {
                continue;
            };
            let role = match binding.role() {
                TensorRole::RoutedExpertGate => 0,
                TensorRole::RoutedExpertUp => 1,
                TensorRole::RoutedExpertDown => 2,
                _ => {
                    return Err(invalid(
                        "only routed gate/up/down parameters may enter host residency",
                    ));
                }
            };
            if binding.is_alias()
                || !ids.insert(binding.id())
                || !groups.entry((layer, expert)).or_default().insert(role)
            {
                return Err(invalid("duplicate/aliased routed expert parameter"));
            }
            let encoding = binding.numeric_fp8_encoding().ok_or_else(|| {
                invalid("host prewarm requires compressed numeric FP8 + BF16/F32 scales")
            })?;
            let source = binding.numeric_fp8_source_metadata(encoding)?;
            if source.expert_count().is_some() {
                return Err(invalid(
                    "host prewarm requires individual 2D expert bindings",
                ));
            }
            let [rows, columns] = source.matrix_shape();
            let read = source.plan_tile_metadata(&reader, None, 0..rows, 0..columns)?;
            let bytes = read.read_plan().storage_bytes();
            payload_bytes = payload_bytes
                .checked_add(bytes)
                .ok_or_else(|| invalid("payload bytes overflow"))?;
            largest = largest.max(bytes);
            for snapshot in read.read_plan().source_files() {
                if let Some(previous) =
                    sources.insert(snapshot.catalog_path().to_path_buf(), snapshot.clone())
                    && previous != *snapshot
                {
                    return Err(invalid("conflicting source snapshots"));
                }
            }
            parameters.push(ParameterRead {
                binding: binding.clone(),
                read,
            });
        }
        if groups.is_empty() || groups.values().any(|roles| roles.len() != 3) {
            return Err(invalid("full prewarm requires complete routed experts"));
        }
        if sources.len() > crate::checkpoint::DEFAULT_MAX_OPEN_SHARDS {
            return Err(invalid(
                "full prewarm exceeds the verified reader's 16-shard FD cap",
            ));
        }
        // Covers packed-read splitting / immutable allocation conversion plus a
        // conservative per-projection metadata allowance, all inside the cap.
        let staging = largest
            .checked_mul(2 * options.workers as u64)
            .ok_or_else(|| invalid("staging bytes overflow"))?;
        // Read plans and the source catalog already exist before the OS snapshot.
        // Charge only future clones/proofs/maps, including variable path/name data.
        let mut retained_metadata_bytes = 1u64 << 20;
        let mut session_metadata_bytes = 0u64;
        for parameter in &parameters {
            let slices = [
                parameter.binding.weight().slice(),
                parameter.binding.scale().expect("numeric pair").slice(),
            ];
            let variable = slices.iter().try_fold(0u64, |n, s| {
                n.checked_add((s.name.len() + s.path.as_os_str().len()) as u64)
                    .ok_or_else(|| invalid("metadata bytes overflow"))
            })?;
            let per_parameter = variable
                .checked_mul(16)
                .and_then(|n| n.checked_add(4096))
                .ok_or_else(|| invalid("metadata bytes overflow"))?;
            retained_metadata_bytes = retained_metadata_bytes
                .checked_add(per_parameter)
                .ok_or_else(|| invalid("metadata bytes overflow"))?;
            for extent in parameter.read.read_plan().extents() {
                // Union plan, session plan clone and BTree extent membership index.
                let bytes = (extent.path().as_os_str().len() as u64)
                    .checked_add(256)
                    .and_then(|n| n.checked_mul(3))
                    .ok_or_else(|| invalid("session metadata overflow"))?;
                session_metadata_bytes = session_metadata_bytes
                    .checked_add(bytes)
                    .ok_or_else(|| invalid("session metadata overflow"))?;
            }
        }
        let memory = HostCacheMemoryPlan {
            payload_bytes,
            retained_metadata_bytes,
            session_metadata_bytes,
            worker_staging_bytes: staging,
            worker_stack_bytes: (options.workers as u64) << 20,
        };
        let required_bytes = memory.peak_bytes()?;
        if required_bytes > options.max_bytes || groups.len() > options.max_experts {
            return Err(error(HostExpertCacheError::Budget {
                required_bytes,
                required_experts: groups.len(),
                budget_bytes: options.max_bytes,
                budget_experts: options.max_experts,
            }));
        }
        // Deterministic file-local traversal improves NAS locality without one
        // worker per shard/tensor (or a global Rayon pool sized to 128 CPUs).
        parameters.sort_by(|a, b| {
            (
                a.binding.weight().slice().path.as_path(),
                a.binding.weight().slice().offset,
            )
                .cmp(&(
                    b.binding.weight().slice().path.as_path(),
                    b.binding.weight().slice().offset,
                ))
        });
        Ok(Self {
            options,
            parameters,
            sources: sources.into_values().collect::<Vec<_>>().into(),
            experts: groups.len(),
            payload_bytes,
            required_bytes,
            memory,
        })
    }

    pub fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }
    pub fn required_bytes(&self) -> u64 {
        self.required_bytes
    }
    pub fn experts(&self) -> usize {
        self.experts
    }
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    pub fn memory_plan(&self) -> HostCacheMemoryPlan {
        self.memory
    }

    /// Legacy cache-only callers may pass an already-net available byte count.
    /// Factories configure options.memory_budget for remaining model allocations.
    pub fn check_available_bytes(&self, available: u64) -> Result<()> {
        self.check_memory_snapshot(HostMemorySnapshot {
            system_available_bytes: available,
            process_rss_bytes: 0,
            cgroup_limit_bytes: None,
            cgroup_current_bytes: None,
        })
        .map(|_| ())
    }

    pub fn check_memory_snapshot(
        &self,
        snapshot: HostMemorySnapshot,
    ) -> Result<HostMemoryAdmission> {
        self.options.memory_budget.admit(self.memory, snapshot)
    }

    /// Cache-only by default; does not pretend to budget a future model runner.
    /// The complete configured ledger is checked before any payload I/O/allocation.
    pub fn warm(self, mut progress: impl FnMut(HostExpertWarmProgress)) -> Result<HostExpertCache> {
        let admission = self.check_memory_snapshot(memory::snapshot()?)?;
        tracing::info!(
            cache_cap_bytes = self.options.max_bytes,
            ?admission,
            "host prewarm planned peak (cap is not a reservation)"
        );
        self.warm_admitted(admission, &mut progress)
    }

    fn warm_admitted(
        self,
        admission: HostMemoryAdmission,
        progress: &mut impl FnMut(HostExpertWarmProgress),
    ) -> Result<HostExpertCache> {
        let started = Instant::now();
        let reader = CheckpointTensorReader::new(self.payload_bytes);
        let plan = CheckpointReadPlan::new(
            self.parameters
                .iter()
                .flat_map(|p| p.read.read_plan().extents().iter().cloned()),
            self.sources.clone(),
        )?;
        // Checks every unique source (path AND FD) before any payload allocation.
        let session = reader.verified_read_session(&plan)?;
        let completed = AtomicUsize::new(0);
        let read_bytes = AtomicU64::new(0);
        let next = AtomicUsize::new(0);
        let cancel = AtomicBool::new(false);
        let entries = Arc::new(Mutex::new(BTreeMap::new()));
        let failure = Arc::new(Mutex::new(None));
        progress(HostExpertWarmProgress {
            completed_parameters: 0,
            total_parameters: self.parameters.len(),
            read_bytes: 0,
            total_bytes: self.payload_bytes,
            elapsed: started.elapsed(),
        });
        let parameters = session.read_incrementally(|session| {
            std::thread::scope(|scope| {
                let mut workers = Vec::new();
                for _ in 0..self.options.workers {
                    let entries = Arc::clone(&entries);
                    let worker_failure = Arc::clone(&failure);
                    let (cancel, next, read_bytes, completed, plans) =
                        (&cancel, &next, &read_bytes, &completed, &self.parameters);
                    let worker = std::thread::Builder::new()
                        .name(format!("expert-prewarm-{}", workers.len()))
                        .stack_size(1 << 20)
                        .spawn_scoped(scope, move || {
                            let result =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    while !cancel.load(Ordering::Acquire) {
                                        let index = next.fetch_add(1, Ordering::Relaxed);
                                        let Some(parameter) = plans.get(index) else {
                                            break;
                                        };
                                        match parameter.read.read_incremental(session) {
                                            Ok(artifact) => {
                                                let bytes = artifact.storage_bytes();
                                                let prepared =
                                                    Arc::new(PreparedParameter::from_host_numeric(
                                                        parameter.binding.clone(),
                                                        artifact,
                                                    ));
                                                entries.lock().unwrap().insert(
                                                    (
                                                        parameter.binding.residency().clone(),
                                                        parameter.binding.id(),
                                                    ),
                                                    prepared,
                                                );
                                                read_bytes.fetch_add(bytes, Ordering::Relaxed);
                                                completed.fetch_add(1, Ordering::Release);
                                            }
                                            Err(err) => {
                                                cancel.store(true, Ordering::Release);
                                                let mut failure = worker_failure.lock().unwrap();
                                                if failure.is_none() {
                                                    *failure = Some(err);
                                                }
                                                break;
                                            }
                                        }
                                    }
                                }));
                            if result.is_err() {
                                cancel.store(true, Ordering::Release);
                            }
                            result.is_err()
                        });
                    match worker {
                        Ok(worker) => workers.push(worker),
                        Err(source) => {
                            cancel.store(true, Ordering::Release);
                            *failure
                                .lock()
                                .map_err(|_| invalid("warm failure lock poisoned"))? =
                                Some(error(HostExpertCacheError::WorkerSpawn { source }));
                            break;
                        }
                    }
                }
                let mut last_report = Instant::now();
                while workers.iter().any(|worker| !worker.is_finished()) {
                    std::thread::sleep(Duration::from_millis(100));
                    let count = completed.load(Ordering::Acquire);
                    // Report at bounded cadence; no per-expert logging or queue.
                    if last_report.elapsed() >= Duration::from_secs(1) {
                        last_report = Instant::now();
                        progress(HostExpertWarmProgress {
                            completed_parameters: count,
                            total_parameters: self.parameters.len(),
                            read_bytes: read_bytes.load(Ordering::Relaxed),
                            total_bytes: self.payload_bytes,
                            elapsed: started.elapsed(),
                        });
                    }
                }
                let mut panicked = false;
                for worker in workers {
                    if worker.join().unwrap_or(true) {
                        cancel.store(true, Ordering::Release);
                        panicked = true;
                    }
                }
                if panicked {
                    return Err(error(HostExpertCacheError::WorkerPanicked));
                }
                if let Some(err) = failure
                    .lock()
                    .map_err(|_| invalid("warm failure lock poisoned"))?
                    .take()
                {
                    return Err(err);
                }
                let entries = Arc::try_unwrap(entries)
                    .map_err(|_| invalid("warm entries still referenced"))?;
                entries
                    .into_inner()
                    .map_err(|_| invalid("warm entries lock poisoned"))
            })
        })?;
        if parameters.len() != self.parameters.len() {
            return Err(invalid("incomplete host expert warm"));
        }
        let io = reader.read_counters();
        drop(reader); // Host hits retain proofs, not shard handles.
        progress(HostExpertWarmProgress {
            completed_parameters: parameters.len(),
            total_parameters: parameters.len(),
            read_bytes: io.bytes_read,
            total_bytes: self.payload_bytes,
            elapsed: started.elapsed(),
        });
        Ok(HostExpertCache {
            generation: PreparedDecoderGeneration::take()?,
            parameters,
            sources: self.sources,
            stats: HostExpertCacheStats {
                experts: self.experts,
                payload_bytes: self.payload_bytes,
                reserved_bytes: self.required_bytes,
                budget_bytes: self.options.max_bytes,
                available_bytes: admission.snapshot.available_bytes(),
                memory_admission: admission,
                workers: self.options.workers,
                elapsed: started.elapsed(),
                io,
            },
            hits: AtomicU64::new(0),
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct HostExpertWarmProgress {
    pub completed_parameters: usize,
    pub total_parameters: usize,
    pub read_bytes: u64,
    pub total_bytes: u64,
    pub elapsed: Duration,
}

#[derive(Debug, Clone, Copy)]
pub struct HostExpertCacheStats {
    pub experts: usize,
    pub payload_bytes: u64,
    pub reserved_bytes: u64,
    pub budget_bytes: u64,
    pub available_bytes: u64,
    pub memory_admission: HostMemoryAdmission,
    pub workers: usize,
    pub elapsed: Duration,
    /// End-of-read snapshot. The reader/FD pool is dropped before publication;
    /// live_handles here describes the read phase, not retained cache handles.
    pub io: CheckpointReadCounters,
}

/// One immutable image; bounds cover every expert, so readiness never depends on
/// eviction luck. Clones of resources/materializers share this same authority.
#[derive(Debug)]
pub struct HostExpertCache {
    generation: PreparedDecoderGeneration,
    parameters: BTreeMap<(ParameterResidency, ParameterId), Arc<PreparedParameter>>,
    sources: Arc<[CheckpointSourceFileIdentity]>,
    stats: HostExpertCacheStats,
    hits: AtomicU64,
}

impl HostExpertCache {
    pub fn generation(&self) -> u64 {
        self.generation.get()
    }
    pub fn stats(&self) -> HostExpertCacheStats {
        self.stats
    }
    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }
    /// A freshness boundary, separate from immutable content proof. CUDA already
    /// performs its own image-wide preflight before a transaction.
    pub fn preflight(&self) -> Result<()> {
        for source in self.sources.iter() {
            if !source.is_current() {
                return Err(error(HostExpertCacheError::SourceChanged {
                    path: source.catalog_path().into(),
                }));
            }
        }
        Ok(())
    }
    pub(crate) fn parameter(&self, binding: &BoundParameter) -> Result<Arc<PreparedParameter>> {
        let parameter = self
            .parameters
            .get(&(binding.residency().clone(), binding.id()))
            .ok_or_else(|| invalid("expert is outside this prewarmed generation"))?;
        if !binding.shares_storage_with(parameter.binding()) || binding.role() != parameter.role() {
            return Err(invalid("expert belongs to another bound image"));
        }
        self.hits.fetch_add(1, Ordering::Relaxed);
        Ok(Arc::clone(parameter))
    }
}
