use super::*;
use std::collections::{BTreeMap, HashMap, hash_map::RandomState};
use std::hash::BuildHasher;

#[path = "ingest.rs"]
mod ingest;
#[path = "report.rs"]
mod report;

fn add(a: usize, b: usize) -> KvSetResult<usize> {
    a.checked_add(b).ok_or(KvSetError::ArithmeticOverflow)
}
fn mul(a: usize, b: usize) -> KvSetResult<usize> {
    a.checked_mul(b).ok_or(KvSetError::ArithmeticOverflow)
}
fn count(n: usize) -> KvSetResult<u64> {
    u64::try_from(n).map_err(|_| KvSetError::ArithmeticOverflow)
}
fn bump(value: &mut u64, n: u64) -> KvSetResult<()> {
    *value = value.checked_add(n).ok_or(KvSetError::ArithmeticOverflow)?;
    Ok(())
}
fn limit(value: usize, maximum: usize, name: &'static str) -> KvSetResult<()> {
    if value > maximum {
        Err(KvSetError::ResourceLimit(name))
    } else {
        Ok(())
    }
}
fn invalid(reason: &'static str) -> KvSetError {
    KvSetError::InvalidTrace(reason)
}
fn capacity_vec<T>(capacity: usize) -> KvSetResult<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| KvSetError::AllocationFailed)?;
    Ok(values)
}
fn ratio(numerator: u64, denominator: u64) -> Option<Ratio> {
    (denominator != 0).then_some(Ratio {
        numerator,
        denominator,
    })
}

/// A newly appended node must include its old range, especially at powers of two.
#[derive(Default)]
struct Fenwick {
    tree: Vec<usize>,
}
impl Fenwick {
    fn sum(&self, mut end: usize) -> usize {
        let mut sum = 0;
        while end > 0 {
            sum += self.tree[end - 1];
            end &= end - 1;
        }
        sum
    }
    fn remove(&mut self, mut index: usize) {
        while index <= self.tree.len() {
            self.tree[index - 1] -= 1;
            let Some(next) = index.checked_add(index.isolate_lowest_one()) else {
                break;
            };
            index = next;
        }
    }
    fn append(&mut self) -> KvSetResult<usize> {
        let index = add(self.tree.len(), 1)?;
        let lowbit = index.isolate_lowest_one();
        let value = add(1, self.sum(index - 1) - self.sum(index - lowbit))?;
        self.tree
            .try_reserve(1)
            .map_err(|_| KvSetError::AllocationFailed)?;
        self.tree.push(value);
        Ok(index)
    }
}
struct PageKey {
    namespace: usize,
    parent: Option<usize>,
    tokens: Vec<u32>,
}

/// Hashes index collision buckets only; full equality always resolves collisions.
/// Neither internal IDs nor hash values are persisted as content identities.
#[derive(Default)]
struct History {
    buckets: HashMap<u64, Vec<usize>>,
    hasher: RandomState,
    keys: Vec<PageKey>,
    last: Vec<Option<usize>>,
    fenwick: Fenwick,
    token_bytes: usize,
    #[cfg(test)]
    force_collision: bool,
}
impl History {
    fn intern(
        &mut self,
        namespace: usize,
        parent: Option<usize>,
        tokens: &[u32],
        limits: &ResourceLimits,
    ) -> KvSetResult<usize> {
        let hash = self.hasher.hash_one((namespace, parent, tokens));
        #[cfg(test)]
        let hash = if self.force_collision { 0 } else { hash };
        if let Some(bucket) = self.buckets.get(&hash) {
            for &id in bucket {
                let key = &self.keys[id];
                if key.namespace == namespace && key.parent == parent && key.tokens == tokens {
                    return Ok(id);
                }
            }
        }
        let id = self.keys.len();
        limit(add(id, 1)?, limits.max_unique_keys, "unique keys")?;
        let bytes = add(self.token_bytes, mul(tokens.len(), size_of::<u32>())?)?;
        limit(
            bytes,
            limits.max_identity_token_bytes,
            "identity token bytes",
        )?;
        self.keys
            .try_reserve(1)
            .map_err(|_| KvSetError::AllocationFailed)?;
        self.last
            .try_reserve(1)
            .map_err(|_| KvSetError::AllocationFailed)?;
        self.buckets
            .try_reserve(1)
            .map_err(|_| KvSetError::AllocationFailed)?;
        let bucket = self.buckets.entry(hash).or_default();
        bucket
            .try_reserve(1)
            .map_err(|_| KvSetError::AllocationFailed)?;
        let mut owned = capacity_vec(tokens.len())?;
        owned.extend_from_slice(tokens);
        self.keys.push(PageKey {
            namespace,
            parent,
            tokens: owned,
        });
        self.last.push(None);
        bucket.push(id);
        self.token_bytes = bytes;
        Ok(id)
    }
    fn distance(&self, id: usize) -> StackDistance {
        match self.last[id] {
            None => StackDistance::Cold,
            Some(last) => StackDistance::Warm(
                self.fenwick.sum(self.fenwick.tree.len()) - self.fenwick.sum(last),
            ),
        }
    }
    fn publish(&mut self, id: usize) -> KvSetResult<()> {
        if let Some(last) = self.last[id] {
            self.fenwick.remove(last);
        }
        self.last[id] = Some(self.fenwick.append()?);
        Ok(())
    }
}

type Histogram = BTreeMap<usize, u64>;
fn histogram_add(
    map: &mut Histogram,
    key: usize,
    amount: u64,
    entries: &mut usize,
    limits: &ResourceLimits,
) -> KvSetResult<()> {
    if let Some(value) = map.get_mut(&key) {
        return bump(value, amount);
    }
    let next = add(*entries, 1)?;
    limit(next, limits.max_histogram_entries, "histogram entries")?;
    map.insert(key, amount);
    *entries = next;
    Ok(())
}
fn histogram_at(map: &Histogram, capacity: usize) -> KvSetResult<u64> {
    map.range(..=capacity).try_fold(0u64, |sum, (_, value)| {
        sum.checked_add(*value)
            .ok_or(KvSetError::ArithmeticOverflow)
    })
}
fn histogram_requirement(map: &Histogram, need: u64) -> KvSetResult<Option<usize>> {
    let mut total = 0;
    for (&capacity, &amount) in map {
        bump(&mut total, amount)?;
        if total >= need {
            return Ok(Some(capacity));
        }
    }
    Ok(None)
}
struct Pending {
    generation: u64,
    namespace: usize,
    /// Full prompt followed by any actually committed decoded suffix; one copy only.
    known: Vec<u32>,
    prompt_tokens: usize,
    committed: usize,
    eligible: usize,
    measured: bool,
}
struct FiniteCache {
    capacity: usize,
    /// LRU first. All probes happen before the first touch of a Demand batch.
    resident: Vec<usize>,
    page_hits: u64,
    prefix_tokens: u64,
    achieved: Vec<u64>,
}
impl FiniteCache {
    fn touch(&mut self, id: usize) {
        if self.capacity == 0 {
            return;
        }
        if let Some(index) = self.resident.iter().position(|&key| key == id) {
            self.resident.remove(index);
        } else if self.resident.len() == self.capacity {
            self.resident.remove(0);
        }
        self.resident.push(id);
    }
}
struct Probe {
    id: usize,
    tokens: usize,
    distance: StackDistance,
    prefix_required: Option<usize>,
}
struct Engine {
    header: TraceHeader,
    config: AnalyzerConfig,
    namespaces: HashMap<String, usize>,
    history: History,
    pending: HashMap<String, Pending>,
    generations: HashMap<String, u64>,
    last_seq: Option<u64>,
    poisoned: bool,
    usage: AnalyzerUsage,
    workload: WorkloadCounts,
    outcomes: OutcomeCounts,
    actual: ActualTotals,
    page_histogram: Histogram,
    token_histogram: Histogram,
    targets: Vec<Histogram>,
    unreachable: Vec<u64>,
    reachable: Vec<u64>,
    caches: Vec<FiniteCache>,
}

fn validate_header(header: &TraceHeader) -> KvSetResult<()> {
    if header.schema != TRACE_SCHEMA
        || header.schema_version != TRACE_SCHEMA_VERSION
        || header.semantics_version != SEMANTICS_VERSION
    {
        return Err(invalid("unsupported schema or semantics version"));
    }
    if header.page_size == 0 {
        return Err(invalid("page_size must be positive"));
    }
    if header.window_mode != WindowMode::WarmHistory {
        return Err(invalid("reset_per_window is unsupported"));
    }
    limit(
        header.namespaces.len(),
        header.limits.max_namespaces,
        "namespaces",
    )?;
    let mut bytes = 0;
    for definition in &header.namespaces {
        let ns = &definition.namespace;
        for field in [
            &definition.id,
            &ns.model,
            &ns.adapter,
            &ns.position,
            &ns.layout,
            &ns.backend,
            &ns.plan,
            &ns.run_scope,
        ] {
            if field.trim().is_empty() {
                return Err(invalid("namespace fields must be nonempty"));
            }
            bytes = add(bytes, field.len())?;
            limit(bytes, header.limits.max_namespace_bytes, "namespace bytes")?;
        }
    }
    Ok(())
}
fn normalize_ratios(values: &mut Vec<Ratio>) -> KvSetResult<()> {
    for value in values.iter_mut() {
        value.validate()?;
        let (mut a, mut b) = (value.numerator, value.denominator);
        while b != 0 {
            (a, b) = (b, a % b);
        }
        value.numerator /= a;
        value.denominator /= a;
    }
    values.sort_unstable_by(|a, b| {
        (u128::from(a.numerator) * u128::from(b.denominator))
            .cmp(&(u128::from(b.numerator) * u128::from(a.denominator)))
    });
    values.dedup();
    Ok(())
}
fn report_size(points: usize, config: &AnalyzerConfig, limits: &ResourceLimits) -> KvSetResult<()> {
    limit(points, limits.max_capacity_points, "capacity points")?;
    let rows = add(
        mul(points, add(1, config.alphas.len())?)?,
        mul(config.alphas.len(), config.p_values.len())?,
    )?;
    limit(rows, limits.max_report_points, "report points")
}
fn validate_config(header: &TraceHeader, config: &mut AnalyzerConfig) -> KvSetResult<()> {
    let limits = &header.limits;
    limit(
        add(config.alphas.len(), config.p_values.len())?,
        limits.max_target_parameters,
        "target parameters",
    )?;
    normalize_ratios(&mut config.alphas)?;
    normalize_ratios(&mut config.p_values)?;
    if let Some(window) = config.measurement_window
        && window.start_seq > window.end_seq
    {
        return Err(KvSetError::InvalidConfig("inverted measurement window"));
    }
    if let Some(capacities) = &mut config.capacities {
        limit(
            capacities.len(),
            limits.max_capacity_points,
            "capacity points",
        )?;
        if capacities.is_empty() {
            return Err(KvSetError::InvalidConfig("capacity list is empty"));
        }
        capacities.sort_unstable();
        capacities.dedup();
    }
    if header.recency_model == RecencyModel::DeferredAdmissionPrefixLruV1 {
        let capacities = config.capacities.as_ref().ok_or(KvSetError::InvalidConfig(
            "deferred requires explicit capacities",
        ))?;
        let total = capacities
            .iter()
            .try_fold(0, |sum, &capacity| add(sum, capacity))?;
        limit(
            total,
            limits.max_total_capacity_entries,
            "finite cache entries",
        )?;
        if let Some(budget) = config.capacity_budget
            && capacities.binary_search(&budget).is_err()
        {
            return Err(KvSetError::InvalidConfig("deferred budget must be sampled"));
        }
    }
    report_size(
        config.capacities.as_ref().map_or(1, Vec::len),
        config,
        limits,
    )
}
impl Engine {
    fn new(
        header: TraceHeader,
        mut config: AnalyzerConfig,
        model: RecencyModel,
    ) -> KvSetResult<Self> {
        validate_header(&header)?;
        if header.recency_model != model {
            return Err(KvSetError::InvalidConfig("analyzer/header model mismatch"));
        }
        validate_config(&header, &mut config)?;
        let mut namespaces = HashMap::new();
        let mut normalized = HashMap::new();
        for definition in &header.namespaces {
            let next = normalized.len();
            let index = *normalized.entry(&definition.namespace).or_insert(next);
            if namespaces.insert(definition.id.clone(), index).is_some() {
                return Err(invalid("duplicate namespace id"));
            }
        }
        let mut caches = Vec::new();
        if model == RecencyModel::DeferredAdmissionPrefixLruV1 {
            for &capacity in config.capacities.as_ref().expect("validated capacities") {
                caches.push(FiniteCache {
                    capacity,
                    resident: capacity_vec(capacity)?,
                    page_hits: 0,
                    prefix_tokens: 0,
                    achieved: vec![0; config.alphas.len()],
                });
            }
        }
        let alphas = config.alphas.len();
        Ok(Self {
            header,
            config,
            namespaces,
            history: History::default(),
            pending: HashMap::new(),
            generations: HashMap::new(),
            last_seq: None,
            poisoned: false,
            usage: AnalyzerUsage::default(),
            workload: WorkloadCounts::default(),
            outcomes: OutcomeCounts::default(),
            actual: ActualTotals::default(),
            page_histogram: Histogram::new(),
            token_histogram: Histogram::new(),
            targets: vec![Histogram::new(); alphas],
            unreachable: vec![0; alphas],
            reachable: vec![0; alphas],
            caches,
        })
    }
    fn namespace(&self, id: &str) -> KvSetResult<usize> {
        self.namespaces
            .get(id)
            .copied()
            .ok_or(invalid("unknown namespace"))
    }
    fn measured(&self, seq: u64) -> bool {
        self.config
            .measurement_window
            .is_none_or(|window| seq >= window.start_seq && seq < window.end_seq)
    }
}

pub fn analyze(input: &TraceInput, config: AnalyzerConfig) -> KvSetResult<AnalysisReport> {
    // Reject unbounded metadata before cloning the borrowed header.
    validate_header(&input.header)?;
    let mut engine = Engine::new(input.header.clone(), config, input.header.recency_model)?;
    for event in &input.events {
        engine.ingest(event)?;
    }
    engine.finish()
}

/// Snapshot-demand / publish-recency analyzer. report is nonterminal; finish consumes it.
pub struct Analyzer {
    engine: Engine,
}
impl Analyzer {
    pub fn new(header: TraceHeader, config: AnalyzerConfig) -> KvSetResult<Self> {
        Ok(Self {
            engine: Engine::new(header, config, RecencyModel::PublishRecencySnapshotV1)?,
        })
    }
    pub fn ingest(&mut self, event: &TraceEvent) -> KvSetResult<Option<DemandSummary>> {
        self.engine.ingest(event)
    }
    /// A failed report also poisons the analyzer. This method does not close requests.
    pub fn report(&mut self) -> KvSetResult<AnalysisReport> {
        self.engine.report(false)
    }
    pub fn finish(self) -> KvSetResult<AnalysisReport> {
        self.engine.finish()
    }
}
/// Capacity-specific hit-prefix-touch simulation; explicit sampled capacities required.
pub struct DeferredAnalyzer {
    engine: Engine,
}
impl DeferredAnalyzer {
    pub fn new(header: TraceHeader, config: AnalyzerConfig) -> KvSetResult<Self> {
        Ok(Self {
            engine: Engine::new(header, config, RecencyModel::DeferredAdmissionPrefixLruV1)?,
        })
    }
    pub fn ingest(&mut self, event: &TraceEvent) -> KvSetResult<Option<DemandSummary>> {
        self.engine.ingest(event)
    }
    pub fn report(&mut self) -> KvSetResult<AnalysisReport> {
        self.engine.report(false)
    }
    pub fn finish(self) -> KvSetResult<AnalysisReport> {
        self.engine.finish()
    }
}

#[cfg(test)]
#[path = "../../tests/unit/kvset/engine_tests.rs"]
mod internal_tests;
