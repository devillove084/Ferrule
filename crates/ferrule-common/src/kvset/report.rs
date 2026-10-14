use super::*;
use std::collections::BTreeSet;

fn capacities(engine: &Engine) -> KvSetResult<Vec<usize>> {
    if let Some(values) = &engine.config.capacities {
        report_size(values.len(), &engine.config, &engine.header.limits)?;
        return Ok(values.clone());
    }
    let mut values = BTreeSet::new();
    let mut insert = |capacity| -> KvSetResult<()> {
        if !values.contains(&capacity) {
            // Check before allocating each new breakpoint, not after collecting all.
            report_size(add(values.len(), 1)?, &engine.config, &engine.header.limits)?;
            values.insert(capacity);
        }
        Ok(())
    };
    insert(0)?;
    for &capacity in engine
        .page_histogram
        .keys()
        .chain(engine.token_histogram.keys())
        .chain(engine.targets.iter().flat_map(|hist| hist.keys()))
    {
        insert(capacity)?;
    }
    let mut result = capacity_vec(values.len())?;
    result.extend(values);
    Ok(result)
}

/// Sorted capacities and thresholds allow a single linear merge, not C histogram scans.
fn sample(histogram: &Histogram, capacities: &[usize]) -> KvSetResult<Vec<u64>> {
    let mut points = capacity_vec(capacities.len())?;
    let mut cursor = histogram.iter().peekable();
    let mut total = 0;
    for &capacity in capacities {
        while let Some(&(&threshold, &weight)) = cursor.peek() {
            if threshold > capacity {
                break;
            }
            bump(&mut total, weight)?;
            cursor.next();
        }
        points.push(total);
    }
    Ok(points)
}

fn achieved(engine: &Engine, alpha: usize, capacity: usize) -> KvSetResult<u64> {
    if engine.header.recency_model == RecencyModel::DeferredAdmissionPrefixLruV1 {
        let index = engine
            .caches
            .binary_search_by_key(&capacity, |cache| cache.capacity)
            .map_err(|_| invalid("capacity is not sampled"))?;
        Ok(engine.caches[index].achieved[alpha])
    } else {
        histogram_at(&engine.targets[alpha], capacity)
    }
}

fn requirement(engine: &Engine, alpha: usize, p: Ratio) -> KvSetResult<CapacityRequirement> {
    let denominator = match engine.config.coverage_denominator {
        CoverageDenominator::Opportunity => engine.workload.opportunity_requests,
        CoverageDenominator::AllDemands => engine.workload.demands,
    };
    if denominator == 0 {
        return Ok(CapacityRequirement::Undefined);
    }
    let need = p.ceil_mul(denominator)?;
    // A zero coverage target is exact, even if zero was never sampled.
    if need == 0 {
        return Ok(CapacityRequirement::Exact { capacity: 0 });
    }
    if engine.reachable[alpha] < need {
        return Ok(CapacityRequirement::Unreachable);
    }
    if engine.header.recency_model == RecencyModel::PublishRecencySnapshotV1 {
        // Reporting only a subset of capacities does not weaken the exact histogram.
        let capacity = histogram_requirement(&engine.targets[alpha], need)?
            .ok_or(invalid("coverage histogram is inconsistent"))?;
        if let Some(budget) = engine.config.capacity_budget
            && capacity > budget
        {
            return Ok(CapacityRequirement::BeyondBudget { budget });
        }
        return Ok(CapacityRequirement::Exact { capacity });
    }
    let first = engine
        .caches
        .iter()
        .find(|cache| cache.achieved[alpha] >= need);
    match (first, engine.config.capacity_budget) {
        (Some(cache), Some(budget)) if cache.capacity > budget => {
            Ok(CapacityRequirement::BeyondBudget { budget })
        }
        (Some(cache), _) => Ok(CapacityRequirement::AtSampledCapacity {
            capacity: cache.capacity,
        }),
        (None, Some(budget)) => Ok(CapacityRequirement::BeyondBudget { budget }),
        (None, None) => Ok(CapacityRequirement::NotReachedAtSampledCapacities),
    }
}

impl Engine {
    pub(super) fn finish(mut self) -> KvSetResult<AnalysisReport> {
        if self.poisoned {
            return Err(KvSetError::Poisoned);
        }
        for request in self.pending.values() {
            if request.measured {
                bump(&mut self.outcomes.abandoned_at_trace_end, 1)?;
            }
        }
        self.pending.clear();
        self.usage.pending_tokens = 0;
        self.report(true)
    }

    pub(super) fn report(&mut self, finalized: bool) -> KvSetResult<AnalysisReport> {
        if self.poisoned {
            return Err(KvSetError::Poisoned);
        }
        let result = self.report_inner(finalized);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    fn report_inner(&self, finalized: bool) -> KvSetResult<AnalysisReport> {
        let capacities = capacities(self)?;
        let snapshot = self.header.recency_model == RecencyModel::PublishRecencySnapshotV1;
        let page_hits = if snapshot {
            sample(&self.page_histogram, &capacities)?
        } else {
            self.caches.iter().map(|c| c.page_hits).collect()
        };
        let token_hits = if snapshot {
            sample(&self.token_histogram, &capacities)?
        } else {
            self.caches.iter().map(|c| c.prefix_tokens).collect()
        };
        let mut capacity_points = capacity_vec(capacities.len())?;
        for (index, &capacity) in capacities.iter().enumerate() {
            let (pages, tokens) = (page_hits[index], token_hits[index]);
            capacity_points.push(CapacityPoint {
                capacity_pages: capacity,
                page_hits: pages,
                page_reference_hit_rate: ratio(pages, self.workload.page_references),
                contiguous_prefix_tokens: tokens,
                contiguous_prefix_token_hit_rate_theoretical: ratio(
                    tokens,
                    self.workload.theoretical_reusable_tokens,
                ),
                contiguous_prefix_token_hit_rate_eligible: ratio(
                    tokens,
                    self.workload.eligible_tokens,
                ),
                contiguous_prefix_token_hit_rate_prompt: ratio(tokens, self.workload.prompt_tokens),
            });
        }
        let mut coverage = capacity_vec(self.config.alphas.len())?;
        let mut targets = capacity_vec(mul(self.config.alphas.len(), self.config.p_values.len())?)?;
        for (index, &alpha) in self.config.alphas.iter().enumerate() {
            let samples = if snapshot {
                sample(&self.targets[index], &capacities)?
            } else {
                self.caches.iter().map(|c| c.achieved[index]).collect()
            };
            let mut points = capacity_vec(capacities.len())?;
            for (&capacity, achieved) in capacities.iter().zip(samples) {
                points.push(CoveragePoint {
                    capacity_pages: capacity,
                    achieved_requests: achieved,
                    coverage_opportunity: ratio(achieved, self.workload.opportunity_requests),
                    coverage_all_demands: ratio(achieved, self.workload.demands),
                });
            }
            let beyond_budget_requests = self
                .config
                .capacity_budget
                .map(|budget| {
                    self.reachable[index]
                        .checked_sub(achieved(self, index, budget)?)
                        .ok_or(KvSetError::ArithmeticOverflow)
                })
                .transpose()?;
            coverage.push(RequestTargetCoverage {
                alpha,
                unreachable_target_requests: self.unreachable[index],
                beyond_budget_requests,
                points,
            });
            for &p in &self.config.p_values {
                targets.push(TargetCapacity {
                    alpha,
                    p,
                    denominator: self.config.coverage_denominator,
                    requirement: requirement(self, index, p)?,
                });
            }
        }
        let mut usage = self.usage.clone();
        usage.unique_keys = self.history.keys.len();
        usage.identity_token_bytes = self.history.token_bytes;
        usage.pending_requests = self.pending.len();
        usage.finite_cache_entries = self
            .caches
            .iter()
            .try_fold(0, |total, cache| add(total, cache.resident.len()))?;
        usage.retained_token_payload_bytes = add(
            usage.identity_token_bytes,
            mul(usage.pending_tokens, size_of::<u32>())?,
        )?;
        let mut outcomes = self.outcomes.clone();
        outcomes.pending = count(
            self.pending
                .values()
                .filter(|pending| pending.measured)
                .count(),
        )?;
        Ok(AnalysisReport {
            schema_version: TRACE_SCHEMA_VERSION,
            semantics_version: SEMANTICS_VERSION,
            recency_model: self.header.recency_model,
            page_size: self.header.page_size,
            finalized,
            config: self.config.clone(),
            workload: self.workload.clone(),
            outcomes,
            actual: self.actual.clone(),
            theoretical_opportunity_fraction_of_prompt: ratio(
                self.workload.theoretical_reusable_tokens,
                self.workload.prompt_tokens,
            ),
            capacity_points,
            request_target_coverage: coverage,
            target_capacities: targets,
            usage,
        })
    }
}
