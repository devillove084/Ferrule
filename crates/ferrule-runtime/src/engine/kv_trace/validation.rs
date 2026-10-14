//! Shared batch integrity, frontier, and entry-prefix semantics; no KVSET DTOs.
use super::*;
use std::collections::{HashMap, HashSet};

/// Only the shared validator can produce a complete replay report.
#[derive(Debug)]
pub(super) struct ValidatedTrace<'a> {
    pub(super) header: &'a RuntimeKvTraceHeader,
    pub(super) report: RuntimePrefixReplay,
}

/// Actual replay consumes the same validated stream without allocating DTOs or pages.
pub(super) fn no_expand(_: &RuntimeKvTraceHeader, _: u64, _: Projected<'_>) -> TraceResult<()> {
    Ok(())
}

pub(super) fn validate_semantics(
    batches: &[RuntimeKvTraceSnapshot],
    limits: RuntimeKvProjectionLimits,
    emit: impl FnMut(&RuntimeKvTraceHeader, u64, Projected<'_>) -> TraceResult<()>,
) -> TraceResult<ValidatedTrace<'_>> {
    // Batch integrity precedes all event semantics; entry checks precede frontier
    // checks. Keep this ordering shared by both sinks, including error priority.
    let result = validate_inner(batches, limits, emit);
    match &result {
        Ok(validated) => tracing::debug!(
            component = "runtime_kv_trace_validation",
            state = "final_summary",
            batches = batches.len(),
            resident_entries = validated.report.resident_entries,
            charged_pages = validated.report.charged_pages,
            "runtime KV trace validation complete"
        ),
        Err(error) => tracing::debug!(
            component = "runtime_kv_trace_validation",
            state = "rejected",
            reason = error.0,
            "runtime KV trace validation rejected"
        ),
    }
    result
}

fn tokens(batch: &RuntimeKvTraceSnapshot, span: TokenSpan) -> TraceResult<&[u32]> {
    let end = span
        .start
        .checked_add(span.len)
        .ok_or(RuntimeKvTraceError("token range overflow"))?;
    batch
        .tokens
        .get(span.start..end)
        .ok_or(RuntimeKvTraceError("token range outside arena"))
}
fn validate_batches(
    batches: &[RuntimeKvTraceSnapshot],
    limits: RuntimeKvProjectionLimits,
) -> TraceResult<&RuntimeKvTraceHeader> {
    let header = &batches
        .first()
        .ok_or(RuntimeKvTraceError("missing trace batches"))?
        .header;
    if header.schema != RUNTIME_KV_TRACE_SCHEMA
        || header.schema_version != 2
        || header.page_size == 0
        || header.config.run_scope.trim().is_empty()
        || header.config.run_scope.len() > 4096
    {
        return Err(RuntimeKvTraceError("invalid trace header"));
    }
    if let Some(n) = &header.config.stable_content_namespace {
        for s in [
            &n.model,
            &n.adapter,
            &n.position,
            &n.layout,
            &n.backend,
            &n.plan,
            &n.run_scope,
        ] {
            if s.trim().is_empty() || s.len() > 4096 {
                return Err(RuntimeKvTraceError("invalid content namespace"));
            }
        }
    }
    let mut next = 0u64;
    let mut token_count = 0usize;
    let mut usage = RuntimeKvTraceUsage::default();
    for (index, b) in batches.iter().enumerate() {
        token_count = token_count
            .checked_add(b.tokens.len())
            .ok_or(RuntimeKvTraceError("input token overflow"))?;
        if token_count > limits.max_tokens
            || b.lifecycle_events.len() > header.config.max_events
            || b.tokens.len() > header.config.max_token_bytes / 4
        {
            return Err(RuntimeKvTraceError("input resource limit"));
        }
        let batch_events = u64::try_from(b.lifecycle_events.len())
            .map_err(|_| RuntimeKvTraceError("event count overflow"))?;
        let batch_token_bytes = u64::try_from(b.tokens.len())
            .ok()
            .and_then(|n| n.checked_mul(size_of::<u32>() as u64))
            .ok_or(RuntimeKvTraceError("token byte count overflow"))?;
        usage.events = usage
            .events
            .checked_add(batch_events)
            .ok_or(RuntimeKvTraceError("event usage overflow"))?;
        usage.token_bytes = usage
            .token_bytes
            .checked_add(batch_token_bytes)
            .ok_or(RuntimeKvTraceError("token usage overflow"))?;
        if b.header != *header
            || !b.status.is_complete()
            || b.start_seq != next
            || b.next_seq.checked_sub(b.start_seq) != Some(batch_events)
            || b.finalized != (index + 1 == batches.len())
            || (b.complete && !b.finalized)
        {
            return Err(RuntimeKvTraceError("incomplete or discontinuous trace"));
        }
        for e in &b.lifecycle_events {
            if matches!(e.kind, RuntimeKvTraceKind::Demand { .. }) {
                usage.observations = usage
                    .observations
                    .checked_add(1)
                    .ok_or(RuntimeKvTraceError("observation usage overflow"))?;
            }
            if e.seq != next {
                return Err(RuntimeKvTraceError("event sequence mismatch"));
            }
            next = next
                .checked_add(1)
                .ok_or(RuntimeKvTraceError("sequence overflow"))?;
        }
        // Usage is cumulative even for empty drains/finalization. Cross-check
        // all three domains instead of trusting a serialized complete flag.
        if b.usage != usage || usage.events != b.next_seq {
            return Err(RuntimeKvTraceError("trace usage mismatch"));
        }
        if next > limits.max_events as u64 {
            return Err(RuntimeKvTraceError("event resource limit"));
        }
    }
    if !batches.last().unwrap().complete {
        return Err(RuntimeKvTraceError("trace is not complete"));
    }
    Ok(header)
}

struct Pending<'a> {
    request: u64,
    session: u64,
    prompt_len: usize,
    known: Vec<u32>,
    frontier: usize,
    input: Option<(u64, usize, &'a [u32])>,
    actual: Option<(usize, usize)>,
}
pub(super) enum Projected<'a> {
    Demand {
        observation: u64,
        request: u64,
        tokens: &'a [u32],
    },
    Publish {
        observation: u64,
        request: u64,
        tokens: &'a [u32],
    },
    Outcome {
        observation: u64,
        request: u64,
        status: TerminalStatus,
        actual: Option<(usize, usize)>,
    },
}
#[derive(Default)]
struct Budget {
    events: usize,
    tokens: usize,
    pages: usize,
}
impl Budget {
    fn charge(
        &mut self,
        tokens: usize,
        pages: usize,
        limits: RuntimeKvProjectionLimits,
    ) -> TraceResult<()> {
        self.events = self
            .events
            .checked_add(1)
            .ok_or(RuntimeKvTraceError("projection event overflow"))?;
        self.tokens = self
            .tokens
            .checked_add(tokens)
            .ok_or(RuntimeKvTraceError("projection token overflow"))?;
        self.pages = self
            .pages
            .checked_add(pages)
            .ok_or(RuntimeKvTraceError("projection page overflow"))?;
        if self.events > limits.max_events
            || self.tokens > limits.max_tokens
            || self.pages > limits.max_publish_pages
        {
            return Err(RuntimeKvTraceError("projection resource limit"));
        }
        Ok(())
    }
}
fn validate_inner(
    batches: &[RuntimeKvTraceSnapshot],
    limits: RuntimeKvProjectionLimits,
    mut emit: impl FnMut(&RuntimeKvTraceHeader, u64, Projected<'_>) -> TraceResult<()>,
) -> TraceResult<ValidatedTrace<'_>> {
    let header = validate_batches(batches, limits)?;
    let mut prefix = PrefixReplayState::default();
    let mut pending: HashMap<u64, Pending<'_>> = HashMap::new();
    let mut seen = HashSet::new();
    let mut used_executions = HashSet::new();
    let mut snapshots: HashMap<u64, usize> = HashMap::new();
    let mut budget = Budget::default();
    let mut known_tokens = 0usize;
    for batch in batches {
        for event in &batch.lifecycle_events {
            prefix.observe(header, batch, event)?;
            match &event.kind {
                RuntimeKvTraceKind::Demand {
                    observation,
                    request,
                    session,
                    tokens: span,
                } => {
                    if *observation != seen.len() as u64
                        || !seen.insert(*observation)
                        || pending
                            .values()
                            .any(|p| p.request == *request || p.session == *session)
                    {
                        return Err(RuntimeKvTraceError("duplicate demand identity"));
                    }
                    if pending.len() >= header.config.max_observations {
                        return Err(RuntimeKvTraceError("observation resource limit"));
                    }
                    let input = tokens(batch, *span)?;
                    budget.charge(input.len(), 0, limits)?;
                    known_tokens = known_tokens
                        .checked_add(input.len())
                        .ok_or(RuntimeKvTraceError("known token overflow"))?;
                    if known_tokens > limits.max_tokens {
                        return Err(RuntimeKvTraceError("known token resource limit"));
                    }
                    emit(
                        header,
                        event.seq,
                        Projected::Demand {
                            observation: *observation,
                            request: *request,
                            tokens: input,
                        },
                    )?;
                    pending.insert(
                        *observation,
                        Pending {
                            request: *request,
                            session: *session,
                            prompt_len: input.len(),
                            known: input.to_vec(),
                            frontier: 0,
                            input: None,
                            actual: None,
                        },
                    );
                }
                RuntimeKvTraceKind::Lookup {
                    observation,
                    outcome,
                    entry,
                    matched_tokens,
                    ..
                } => {
                    if let Some(id) = observation {
                        let p = pending
                            .get_mut(id)
                            .ok_or(RuntimeKvTraceError("lookup without demand"))?;
                        if p.actual.is_some() {
                            return Err(RuntimeKvTraceError("lookup after admission"));
                        }
                    }
                    if matches!(outcome, LookupOutcome::Hit | LookupOutcome::Miss)
                        && observation.is_none()
                    {
                        return Err(RuntimeKvTraceError("hit/miss requires observation"));
                    }
                    if (*outcome == LookupOutcome::Hit) != (entry.is_some() && *matched_tokens > 0)
                        || (*outcome != LookupOutcome::Hit
                            && (entry.is_some() || *matched_tokens != 0))
                    {
                        return Err(RuntimeKvTraceError("inconsistent lookup match"));
                    }
                }
                RuntimeKvTraceKind::AdmissionPublished {
                    observation,
                    actual_tokens,
                    actual_pages,
                } => {
                    let p = pending
                        .get_mut(observation)
                        .ok_or(RuntimeKvTraceError("admission without demand"))?;
                    if p.actual.is_some()
                        || p.input.is_some()
                        || *actual_tokens > p.prompt_len.saturating_sub(1)
                        || *actual_pages != actual_tokens.div_ceil(header.page_size)
                    {
                        return Err(RuntimeKvTraceError("invalid admission frontier"));
                    }
                    p.frontier = *actual_tokens;
                    p.actual = Some((*actual_tokens, *actual_pages));
                }
                RuntimeKvTraceKind::ExecutionInput {
                    observation,
                    reservation,
                    start,
                    tokens: span,
                } => {
                    let p = pending
                        .get_mut(observation)
                        .ok_or(RuntimeKvTraceError("execution without demand"))?;
                    let input = tokens(batch, *span)?;
                    if p.actual.is_none()
                        || p.input.is_some()
                        || *start != p.frontier
                        || input.is_empty()
                        || !used_executions.insert((*observation, *reservation))
                    {
                        return Err(RuntimeKvTraceError(
                            "execution does not continue proven frontier",
                        ));
                    }
                    start
                        .checked_add(input.len())
                        .ok_or(RuntimeKvTraceError("execution frontier overflow"))?;
                    let overlap = p.known.len().saturating_sub(*start).min(input.len());
                    if p.known[*start..*start + overlap] != input[..overlap] {
                        return Err(RuntimeKvTraceError("execution differs from exact prefix"));
                    }
                    p.input = Some((*reservation, *start, input));
                }
                RuntimeKvTraceKind::LogicalKvPublished {
                    observation,
                    reservation,
                    committed_tokens,
                } => {
                    let p = pending
                        .get_mut(observation)
                        .ok_or(RuntimeKvTraceError("publish without demand"))?;
                    let (id, start, input) = p
                        .input
                        .take()
                        .ok_or(RuntimeKvTraceError("publish without execution"))?;
                    if id != *reservation
                        || start != p.frontier
                        || *committed_tokens < start
                        || *committed_tokens > start + input.len()
                    {
                        return Err(RuntimeKvTraceError("publish crosses unproven frontier"));
                    }
                    // Cumulative limits are checked BEFORE extending known content,
                    // allocating pages, or invoking the projection callback.
                    budget.charge(
                        *committed_tokens,
                        committed_tokens.div_ceil(header.page_size),
                        limits,
                    )?;
                    let extra = committed_tokens.saturating_sub(p.known.len());
                    known_tokens = known_tokens
                        .checked_add(extra)
                        .ok_or(RuntimeKvTraceError("known token overflow"))?;
                    if known_tokens > limits.max_tokens {
                        return Err(RuntimeKvTraceError("known token resource limit"));
                    }
                    if extra > 0 {
                        p.known.extend_from_slice(
                            &input[p.known.len() - start..*committed_tokens - start],
                        );
                    }
                    p.frontier = *committed_tokens;
                    if p.frontier > 0 {
                        emit(
                            header,
                            event.seq,
                            Projected::Publish {
                                observation: *observation,
                                request: p.request,
                                tokens: &p.known[..p.frontier],
                            },
                        )?;
                    }
                }
                RuntimeKvTraceKind::SnapshotCaptured {
                    snapshot,
                    committed_tokens,
                    pages,
                } => {
                    if *committed_tokens == 0
                        || *pages != committed_tokens.div_ceil(header.page_size)
                        || snapshots.insert(*snapshot, *pages).is_some()
                    {
                        return Err(RuntimeKvTraceError("invalid snapshot capture"));
                    }
                }
                RuntimeKvTraceKind::SnapshotReleased {
                    snapshot,
                    retiring_pages,
                } => {
                    let pages = snapshots
                        .remove(snapshot)
                        .ok_or(RuntimeKvTraceError("release absent snapshot"))?;
                    if *retiring_pages > pages {
                        return Err(RuntimeKvTraceError("release exceeds snapshot pages"));
                    }
                }
                RuntimeKvTraceKind::SnapshotForkPublished {
                    snapshot,
                    observation,
                } => {
                    if !snapshots.contains_key(snapshot) || !pending.contains_key(observation) {
                        return Err(RuntimeKvTraceError("fork absent snapshot or demand"));
                    }
                }
                RuntimeKvTraceKind::ReservationAborted {
                    observation,
                    reservation,
                } => {
                    let p = pending
                        .get_mut(observation)
                        .ok_or(RuntimeKvTraceError("abort without demand"))?;
                    if p.input.take().map(|i| i.0) != Some(*reservation) {
                        return Err(RuntimeKvTraceError("abort reservation mismatch"));
                    }
                }
                RuntimeKvTraceKind::Cancellation { observation } => {
                    if !pending.contains_key(observation) {
                        return Err(RuntimeKvTraceError("event without demand"));
                    }
                }
                RuntimeKvTraceKind::ReserveSucceeded { observation, .. } => {
                    if observation.is_none_or(|id| !pending.contains_key(&id)) {
                        return Err(RuntimeKvTraceError("reserve without demand"));
                    }
                }
                RuntimeKvTraceKind::Terminal {
                    observation,
                    status,
                } => {
                    let p = pending
                        .remove(observation)
                        .ok_or(RuntimeKvTraceError("terminal without demand"))?;
                    if p.input.is_some() {
                        return Err(RuntimeKvTraceError("terminal with open execution"));
                    }
                    known_tokens -= p.known.len();
                    budget.charge(0, 0, limits)?;
                    emit(
                        header,
                        event.seq,
                        Projected::Outcome {
                            observation: *observation,
                            request: p.request,
                            status: *status,
                            actual: p.actual,
                        },
                    )?;
                }
                RuntimeKvTraceKind::PrefixEntryPublished { tokens: span, .. }
                | RuntimeKvTraceKind::PrefixEntryRemoved { tokens: span, .. } => {
                    tokens(batch, *span)?;
                }
                _ => {}
            }
        }
    }
    if !pending.is_empty() {
        return Err(RuntimeKvTraceError("unclosed demand"));
    }
    Ok(ValidatedTrace {
        header,
        report: prefix.finish()?,
    })
}

struct Entry<'a> {
    tokens: &'a [u32],
    charge: usize,
    generation: u64,
    pins: u32,
}
fn add(a: usize, b: usize) -> TraceResult<usize> {
    a.checked_add(b)
        .ok_or(RuntimeKvTraceError("replay arithmetic overflow"))
}

/// Shared entry-lifecycle validation, advanced by the semantic validator before
/// admission can promote a lookup result to a proven execution frontier.
/// Borrows recorded keys; never expands committed prefixes into KVSET pages.
#[derive(Default)]
struct PrefixReplayState<'a> {
    report: RuntimePrefixReplay,
    entries: HashMap<u64, Entry<'a>>,
    demands: HashMap<u64, &'a [u32]>,
    hits: HashMap<u64, usize>,
    used_ids: HashSet<u64>,
    generation: u64,
    capacity_removals: Option<(usize, usize)>,
}

fn increment(value: u64) -> TraceResult<u64> {
    value
        .checked_add(1)
        .ok_or(RuntimeKvTraceError("replay count overflow"))
}

impl<'a> PrefixReplayState<'a> {
    fn observe(
        &mut self,
        header: &RuntimeKvTraceHeader,
        batch: &'a RuntimeKvTraceSnapshot,
        event: &RuntimeKvTraceEvent,
    ) -> TraceResult<()> {
        if self.capacity_removals.is_some()
            && !matches!(
                event.kind,
                RuntimeKvTraceKind::PrefixEntryRemoved {
                    reason: RemovalReason::Capacity,
                    ..
                } | RuntimeKvTraceKind::PrefixEntryPublished { .. }
            )
        {
            return Err(RuntimeKvTraceError(
                "capacity eviction without atomic insertion",
            ));
        }
        match &event.kind {
            RuntimeKvTraceKind::Demand {
                observation,
                tokens: span,
                ..
            } => {
                self.demands.insert(*observation, tokens(batch, *span)?);
            }
            RuntimeKvTraceKind::Lookup {
                observation,
                outcome,
                entry,
                matched_tokens,
                generation: observed,
            } => {
                if matches!(outcome, LookupOutcome::Hit | LookupOutcome::Miss) {
                    let id =
                        observation.ok_or(RuntimeKvTraceError("hit/miss requires observation"))?;
                    let prompt = self
                        .demands
                        .get(&id)
                        .ok_or(RuntimeKvTraceError("lookup without demand"))?;
                    let best = self
                        .entries
                        .iter()
                        .filter(|(_, e)| {
                            e.tokens.len() < prompt.len() && prompt.starts_with(e.tokens)
                        })
                        .max_by_key(|(_, e)| e.tokens.len())
                        .map(|(id, e)| (*id, e.tokens.len()));
                    if best != entry.map(|id| (id, *matched_tokens)) {
                        return Err(RuntimeKvTraceError(
                            "lookup differs from exact longest prefix",
                        ));
                    }
                    match best {
                        Some((entry, matched)) => {
                            if *outcome != LookupOutcome::Hit || *observed <= self.generation {
                                return Err(RuntimeKvTraceError("invalid hit recency"));
                            }
                            self.generation = *observed;
                            let e = self.entries.get_mut(&entry).unwrap();
                            e.generation = self.generation;
                            e.pins = e
                                .pins
                                .checked_add(1)
                                .ok_or(RuntimeKvTraceError("pin overflow"))?;
                            self.hits.insert(id, matched);
                            self.report.lookup_hits = increment(self.report.lookup_hits)?;
                        }
                        None => {
                            self.hits.remove(&id);
                            self.report.lookup_misses = increment(self.report.lookup_misses)?;
                        }
                    }
                } else if let Some(id) = observation {
                    self.hits.remove(id);
                }
            }
            RuntimeKvTraceKind::UnpinAck { entry } => {
                let e = self
                    .entries
                    .get_mut(entry)
                    .ok_or(RuntimeKvTraceError("unpin absent entry"))?;
                e.pins = e
                    .pins
                    .checked_sub(1)
                    .ok_or(RuntimeKvTraceError("unpin underflow"))?;
            }
            RuntimeKvTraceKind::AdmissionPublished {
                observation,
                actual_tokens,
                ..
            } => {
                if self.hits.get(observation).copied().unwrap_or(0) != *actual_tokens {
                    return Err(RuntimeKvTraceError("admission differs from lookup"));
                }
                self.report.published_admissions = increment(self.report.published_admissions)?;
                self.report.actual_reused_tokens = self
                    .report
                    .actual_reused_tokens
                    .checked_add(*actual_tokens as u64)
                    .ok_or(RuntimeKvTraceError("actual total overflow"))?;
            }
            RuntimeKvTraceKind::PrefixEntryPublished {
                entry,
                tokens: span,
                charge_pages,
                generation: observed,
                replaced,
            } => {
                let key = tokens(batch, *span)?;
                if key.is_empty()
                    || *charge_pages != key.len().div_ceil(header.page_size)
                    || *observed <= self.generation
                {
                    return Err(RuntimeKvTraceError("invalid entry publication"));
                }
                self.generation = *observed;
                if *replaced {
                    let old = self
                        .entries
                        .remove(entry)
                        .ok_or(RuntimeKvTraceError("replacement absent entry"))?;
                    if old.tokens != key || old.pins != 0 {
                        return Err(RuntimeKvTraceError(
                            "replacement changes identity or pinned payload",
                        ));
                    }
                    self.report.charged_pages -= old.charge;
                    self.report.removed_charge_pages =
                        add(self.report.removed_charge_pages, old.charge)?;
                } else if !self.used_ids.insert(*entry)
                    || self.entries.values().any(|e| e.tokens == key)
                {
                    return Err(RuntimeKvTraceError("duplicate entry identity"));
                }
                if let Some((removed, last_charge)) = self.capacity_removals.take() {
                    let prospective = add(add(self.report.charged_pages, removed)?, *charge_pages)?;
                    let required = prospective.saturating_sub(header.prefix_capacity_pages);
                    if required == 0 || removed < required || removed - last_charge >= required {
                        return Err(RuntimeKvTraceError(
                            "capacity eviction differs from insertion preflight",
                        ));
                    }
                }
                self.report.charged_pages = add(self.report.charged_pages, *charge_pages)?;
                if self.report.charged_pages > header.prefix_capacity_pages {
                    return Err(RuntimeKvTraceError("entry capacity exceeded"));
                }
                self.entries.insert(
                    *entry,
                    Entry {
                        tokens: key,
                        charge: *charge_pages,
                        generation: self.generation,
                        pins: 0,
                    },
                );
            }
            RuntimeKvTraceKind::PrefixEntryRemoved {
                tokens: span,
                charge_pages,
                generation: observed,
                reason,
            } => {
                let key = tokens(batch, *span)?;
                let id = self
                    .entries
                    .iter()
                    .find(|(_, e)| e.tokens == key)
                    .map(|(id, _)| *id)
                    .ok_or(RuntimeKvTraceError("removed entry absent"))?;
                let oldest = self
                    .entries
                    .iter()
                    .filter(|(_, e)| e.pins == 0)
                    .min_by_key(|(id, e)| (e.generation, **id))
                    .map(|(id, _)| *id);
                if oldest != Some(id) {
                    return Err(RuntimeKvTraceError("removal violates unpinned entry LRU"));
                }
                let e = self.entries.remove(&id).unwrap();
                if e.charge != *charge_pages || e.generation != *observed {
                    return Err(RuntimeKvTraceError("removed metadata mismatch"));
                }
                if *reason == RemovalReason::Capacity {
                    self.capacity_removals = Some((
                        add(self.capacity_removals.map_or(0, |v| v.0), e.charge)?,
                        e.charge,
                    ));
                }
                self.report.charged_pages -= e.charge;
                self.report.removed_charge_pages = add(self.report.removed_charge_pages, e.charge)?;
            }
            RuntimeKvTraceKind::BackendReleaseAck { pages } => {
                if *pages == 0 {
                    return Err(RuntimeKvTraceError("empty backend release ACK"));
                }
                self.report.backend_released_pages =
                    add(self.report.backend_released_pages, *pages)?;
            }
            RuntimeKvTraceKind::RetirementConfirmed { pages } => {
                self.report.confirmed_pages = add(self.report.confirmed_pages, *pages)?;
                if self.report.confirmed_pages > self.report.backend_released_pages {
                    return Err(RuntimeKvTraceError("confirmation before backend ACK"));
                }
            }
            RuntimeKvTraceKind::Terminal { observation, .. } => {
                self.demands.remove(observation);
                self.hits.remove(observation);
            }
            _ => {}
        }
        Ok(())
    }

    fn finish(mut self) -> TraceResult<RuntimePrefixReplay> {
        if self.capacity_removals.is_some() {
            return Err(RuntimeKvTraceError("unfinished capacity insertion"));
        }
        if self.entries.values().any(|e| e.pins != 0) {
            return Err(RuntimeKvTraceError("unclosed lookup pin"));
        }
        self.report.resident_entries = self.entries.len();
        Ok(self.report)
    }
}
