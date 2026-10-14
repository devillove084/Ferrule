use super::*;

impl Engine {
    pub(super) fn ingest(&mut self, event: &TraceEvent) -> KvSetResult<Option<DemandSummary>> {
        if self.poisoned {
            return Err(KvSetError::Poisoned);
        }
        let result = self.ingest_inner(event);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    fn ingest_inner(&mut self, event: &TraceEvent) -> KvSetResult<Option<DemandSummary>> {
        let next_events = add(self.usage.events, 1)?;
        limit(next_events, self.header.limits.max_events, "events")?;
        if self.last_seq.is_some_and(|last| event.seq() <= last) {
            return Err(invalid("seq must strictly increase"));
        }
        if event.request().id.is_empty() {
            return Err(invalid("empty request id"));
        }
        limit(
            event.request().id.len(),
            self.header.limits.max_request_id_bytes,
            "request id bytes",
        )?;
        let summary = match event {
            TraceEvent::Demand(value) => Some(self.demand(value)?),
            TraceEvent::Publish(value) => {
                self.publish(value)?;
                None
            }
            TraceEvent::Outcome(value) => {
                self.outcome(value)?;
                None
            }
        };
        self.last_seq = Some(event.seq());
        self.usage.events = next_events;
        Ok(summary)
    }

    fn demand(&mut self, event: &DemandEvent) -> KvSetResult<DemandSummary> {
        let namespace = self.namespace(&event.namespace_id)?;
        let limits = &self.header.limits;
        if self.pending.contains_key(&event.request.id) {
            return Err(invalid("duplicate pending request"));
        }
        if self
            .generations
            .get(&event.request.id)
            .is_some_and(|&g| event.request.generation <= g)
        {
            return Err(invalid("request generation is not increasing"));
        }
        limit(
            event.prompt_tokens.len(),
            limits.max_tokens_per_event,
            "tokens per event",
        )?;
        limit(
            add(self.pending.len(), 1)?,
            limits.max_pending_requests,
            "pending requests",
        )?;
        let pending_tokens = add(self.usage.pending_tokens, event.prompt_tokens.len())?;
        limit(pending_tokens, limits.max_pending_tokens, "pending tokens")?;
        let history_bytes = add(
            self.usage.request_history_bytes,
            if self.generations.contains_key(&event.request.id) {
                0
            } else {
                event.request.id.len()
            },
        )?;
        limit(
            history_bytes,
            limits.max_request_history_bytes,
            "request history bytes",
        )?;
        let eligible = event.reuse_limit.eligible_tokens(event.prompt_tokens.len());
        let page_count = eligible.div_ceil(self.header.page_size);
        let summary_bound = add(
            mul(page_count, add(self.caches.len(), 2)?)?,
            self.caches.len(),
        )?;
        limit(
            summary_bound,
            limits.max_demand_summary_entries,
            "demand summary entries",
        )?;
        let mut probes = capacity_vec(page_count)?;
        let mut parent = None;
        let mut theoretical = 0;
        let mut first_cold = None;
        let mut cold_pages = 0;
        let mut prefix_required = Some(0usize);
        for tokens in event.prompt_tokens[..eligible].chunks(self.header.page_size) {
            let id = self.history.intern(namespace, parent, tokens, limits)?;
            let distance = self.history.distance(id);
            match distance {
                StackDistance::Cold => {
                    cold_pages += 1;
                    first_cold.get_or_insert(probes.len());
                    prefix_required = None;
                }
                StackDistance::Warm(distance) => {
                    if let Some(current) = prefix_required {
                        prefix_required = Some(current.max(add(distance, 1)?));
                        theoretical = add(theoretical, tokens.len())?;
                    }
                }
            }
            probes.push(Probe {
                id,
                tokens: tokens.len(),
                distance,
                prefix_required,
            });
            parent = Some(id);
        }
        let measured = self.measured(event.seq);
        let basis = match self.config.target_basis {
            TargetBasis::TheoreticalReusable => theoretical,
            TargetBasis::EligibleTokens => eligible,
            TargetBasis::FullPrompt => event.prompt_tokens.len(),
        };
        let targets: Vec<u64> = self
            .config
            .alphas
            .iter()
            .map(|alpha| alpha.ceil_mul(count(basis)?))
            .collect::<KvSetResult<_>>()?;
        if measured {
            self.record_demand(
                &probes,
                event.prompt_tokens.len(),
                eligible,
                theoretical,
                cold_pages,
                &targets,
            )?;
        }
        let mut capacity_results = capacity_vec(self.caches.len())?;
        for cache in &mut self.caches {
            let mut page_hits = capacity_vec(probes.len())?;
            let mut hit_count = 0;
            let mut contiguous = 0;
            let mut prefix_pages = 0;
            let mut open = true;
            for probe in &probes {
                let hit = cache.resident.contains(&probe.id);
                page_hits.push(hit);
                if hit {
                    hit_count += 1;
                }
                if open && hit {
                    contiguous += probe.tokens;
                    prefix_pages += 1;
                } else {
                    open = false;
                }
            }
            // Snapshot all probes first. Never touch hits following a prefix hole.
            for probe in &probes[..prefix_pages] {
                cache.touch(probe.id);
            }
            if measured {
                bump(&mut cache.page_hits, count(hit_count)?)?;
                bump(&mut cache.prefix_tokens, count(contiguous)?)?;
                for (index, &target) in targets.iter().enumerate() {
                    if theoretical > 0 && count(contiguous)? >= target {
                        bump(&mut cache.achieved[index], 1)?;
                    }
                }
            }
            capacity_results.push(DemandCapacity {
                capacity_pages: cache.capacity,
                page_hits,
                contiguous_prefix_tokens: contiguous,
            });
        }
        let summary = DemandSummary {
            seq: event.seq,
            request: event.request.clone(),
            measured,
            prompt_tokens: event.prompt_tokens.len(),
            eligible_tokens: eligible,
            theoretical_reusable_tokens: theoretical,
            first_cold_page: first_cold,
            page_distances: (self.header.recency_model == RecencyModel::PublishRecencySnapshotV1)
                .then(|| probes.iter().map(|p| p.distance).collect()),
            capacity_results,
        };
        let mut known = capacity_vec(event.prompt_tokens.len())?;
        known.extend_from_slice(&event.prompt_tokens);
        self.pending
            .try_reserve(1)
            .map_err(|_| KvSetError::AllocationFailed)?;
        self.generations
            .try_reserve(1)
            .map_err(|_| KvSetError::AllocationFailed)?;
        self.pending.insert(
            event.request.id.clone(),
            Pending {
                generation: event.request.generation,
                namespace,
                known,
                prompt_tokens: event.prompt_tokens.len(),
                committed: 0,
                eligible,
                measured,
            },
        );
        self.generations
            .insert(event.request.id.clone(), event.request.generation);
        self.usage.pending_tokens = pending_tokens;
        self.usage.request_history_bytes = history_bytes;
        Ok(summary)
    }

    fn record_demand(
        &mut self,
        probes: &[Probe],
        prompt: usize,
        eligible: usize,
        theoretical: usize,
        cold: usize,
        targets: &[u64],
    ) -> KvSetResult<()> {
        bump(&mut self.workload.demands, 1)?;
        bump(&mut self.workload.prompt_tokens, count(prompt)?)?;
        bump(&mut self.workload.eligible_tokens, count(eligible)?)?;
        bump(
            &mut self.workload.theoretical_reusable_tokens,
            count(theoretical)?,
        )?;
        bump(&mut self.workload.page_references, count(probes.len())?)?;
        bump(&mut self.workload.cold_page_references, count(cold)?)?;
        if cold > 0 {
            bump(&mut self.workload.requests_with_cold_pages, 1)?;
        }
        if eligible == 0 {
            bump(&mut self.workload.no_eligible_tokens, 1)?;
        } else if theoretical == 0 {
            bump(&mut self.workload.cold_prefix_requests, 1)?;
        } else {
            bump(&mut self.workload.opportunity_requests, 1)?;
        }
        if theoretical == 0 {
            bump(&mut self.workload.no_opportunity_requests, 1)?;
        }
        let snapshot_model = self.header.recency_model == RecencyModel::PublishRecencySnapshotV1;
        if snapshot_model {
            for probe in probes {
                if let StackDistance::Warm(distance) = probe.distance {
                    histogram_add(
                        &mut self.page_histogram,
                        add(distance, 1)?,
                        1,
                        &mut self.usage.histogram_entries,
                        &self.header.limits,
                    )?;
                }
                if let Some(required) = probe.prefix_required {
                    histogram_add(
                        &mut self.token_histogram,
                        required,
                        count(probe.tokens)?,
                        &mut self.usage.histogram_entries,
                        &self.header.limits,
                    )?;
                }
            }
        }
        for (index, &target) in targets.iter().enumerate() {
            if target > count(theoretical)? {
                bump(&mut self.unreachable[index], 1)?;
                continue;
            }
            // A zero target cannot manufacture reuse opportunities from cold requests.
            if theoretical == 0 {
                continue;
            }
            bump(&mut self.reachable[index], 1)?;
            if snapshot_model {
                let mut required = 0;
                let mut tokens = 0;
                if target > 0 {
                    for probe in probes {
                        required = probe.prefix_required.expect("target is within warm prefix");
                        tokens += probe.tokens;
                        if count(tokens)? >= target {
                            break;
                        }
                    }
                }
                histogram_add(
                    &mut self.targets[index],
                    required,
                    1,
                    &mut self.usage.histogram_entries,
                    &self.header.limits,
                )?;
            }
        }
        Ok(())
    }

    fn publish(&mut self, event: &PublishEvent) -> KvSetResult<()> {
        let namespace = self.namespace(&event.namespace_id)?;
        let pending = self
            .pending
            .get(&event.request.id)
            .ok_or(invalid("publish without demand"))?;
        if pending.generation != event.request.generation || pending.namespace != namespace {
            return Err(invalid("publish request or namespace mismatch"));
        }
        let limits = &self.header.limits;
        let positions = add(self.usage.publish_positions, event.pages.len())?;
        limit(positions, limits.max_publish_positions, "publish positions")?;
        // Preflight the entire input without constructing slices, flattened copies or keys.
        let mut total = 0;
        let mut supplied = 0;
        for (index, page) in event.pages.iter().enumerate() {
            if page.valid_count == 0
                || page.valid_count > page.tokens.len()
                || page.tokens.len() > self.header.page_size
                || (index + 1 != event.pages.len() && page.valid_count != self.header.page_size)
            {
                return Err(invalid("invalid page shape"));
            }
            supplied = add(supplied, page.tokens.len())?;
            total = add(total, page.valid_count)?;
            limit(supplied, limits.max_tokens_per_event, "tokens per event")?;
            limit(
                add(
                    self.usage.pending_tokens,
                    total.saturating_sub(pending.known.len()),
                )?,
                limits.max_pending_tokens,
                "pending tokens",
            )?;
        }
        if total < pending.committed {
            return Err(invalid("committed prefix moved backwards"));
        }
        let mut offset = 0;
        for page in &event.pages {
            for &token in &page.tokens[..page.valid_count] {
                if pending
                    .known
                    .get(offset)
                    .is_some_and(|&known| token != known)
                {
                    return Err(invalid("publish differs from known prefix"));
                }
                offset += 1;
            }
        }
        let extra = total.saturating_sub(pending.known.len());
        let pending_tokens = add(self.usage.pending_tokens, extra)?;
        let pending = self
            .pending
            .get_mut(&event.request.id)
            .expect("validated pending");
        pending
            .known
            .try_reserve_exact(extra)
            .map_err(|_| KvSetError::AllocationFailed)?;
        let old_len = pending.known.len();
        offset = 0;
        let mut parent = None;
        // Canonical parent chains ensure that two pages in one prefix cannot have
        // equal identities. No O(page_count) staging vector is needed for Publish.
        for page in &event.pages {
            let tokens = &page.tokens[..page.valid_count];
            let id = self.history.intern(namespace, parent, tokens, limits)?;
            self.history.publish(id)?;
            for cache in &mut self.caches {
                cache.touch(id);
            }
            for &token in tokens {
                if offset >= old_len {
                    pending.known.push(token);
                }
                offset += 1;
            }
            parent = Some(id);
        }
        pending.committed = total;
        self.usage.pending_tokens = pending_tokens;
        self.usage.publish_positions = positions;
        Ok(())
    }

    fn outcome(&mut self, event: &OutcomeEvent) -> KvSetResult<()> {
        let pending = self
            .pending
            .get(&event.request.id)
            .ok_or(invalid("outcome without demand"))?;
        if pending.generation != event.request.generation {
            return Err(invalid("outcome generation mismatch"));
        }
        if let Some(actual) = &event.actual {
            if actual.reused_prefix_tokens > pending.eligible
                || actual.reused_prefix_tokens > pending.prompt_tokens
            {
                return Err(invalid("actual reuse exceeds demand limit"));
            }
            if actual.reused_pages != actual.reused_prefix_tokens.div_ceil(self.header.page_size) {
                return Err(invalid("actual page/token counts disagree"));
            }
            if actual.reused_pages > actual.cache_capacity_pages {
                return Err(invalid("actual reuse exceeds observed capacity"));
            }
        }
        let pending = self
            .pending
            .remove(&event.request.id)
            .expect("validated pending");
        self.usage.pending_tokens = self
            .usage
            .pending_tokens
            .checked_sub(pending.known.len())
            .ok_or(KvSetError::ArithmeticOverflow)?;
        if pending.measured {
            match event.status {
                RequestStatus::Success => bump(&mut self.outcomes.success, 1)?,
                RequestStatus::Failed => bump(&mut self.outcomes.failed, 1)?,
                RequestStatus::Cancelled => bump(&mut self.outcomes.cancelled, 1)?,
                RequestStatus::Abandoned => bump(&mut self.outcomes.abandoned, 1)?,
            }
            if let Some(actual) = &event.actual {
                bump(&mut self.actual.observed_requests, 1)?;
                bump(
                    &mut self.actual.reused_prefix_tokens,
                    count(actual.reused_prefix_tokens)?,
                )?;
                bump(&mut self.actual.reused_pages, count(actual.reused_pages)?)?;
            }
        }
        Ok(())
    }
}
