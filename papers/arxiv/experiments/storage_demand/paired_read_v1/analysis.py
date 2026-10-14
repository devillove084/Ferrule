"""Prespecified CPU-only paired summaries; missing pairs never disappear."""
from __future__ import annotations

import itertools
import math
import statistics

from runnerhelpers.evidence_v2 import finite, require

METRICS = ('request_e2e_seconds', 'ttft_seconds', 'qualified_goodput_tokens_per_second')


def paired_interval(log_ratios):
    require(len(log_ratios) == 5 and all(finite(x) for x in log_ratios), 'exactly five finite paired log ratios required')
    draws = sorted(statistics.median(draw) for draw in itertools.product(log_ratios, repeat=5))
    # Nearest-rank quantiles, with explicit 1-based ranks (79 and 3047).
    lower = draws[math.ceil(0.025 * len(draws)) - 1]
    upper = draws[math.ceil(0.975 * len(draws)) - 1]
    center = statistics.median(log_ratios)
    return dict(ratio=math.exp(center), fractional_latency_reduction=1 - math.exp(center),
        interval_ratio=[math.exp(lower), math.exp(upper)],
        method='exact enumerated paired-block percentile bootstrap; nearest-rank 2.5/97.5%',
        independent_resampling_unit='paired block, NOT request', n=5, draws=len(draws),
        ranks=[math.ceil(0.025 * len(draws)), math.ceil(0.975 * len(draws))],
        interpretation='exploratory descriptive interval; no coverage/significance/power claim')


def arm_metrics(row):
    if not row.get('assessment', {}).get('can_continue') or not row.get('read_ledger', {}).get('can_continue'):
        return None
    measured = [r for r in row.get('lifecycle', {}).get('requests', {}).get('observations', []) if r.get('measured') is True]
    if len(measured) != 5 or any(r.get('status') != 'success' or r.get('metric_eligible') is not True for r in measured):
        return None
    times, ttfts, tokens = [], [], []
    for r in measured:
        actual = r.get('actual', {})
        e2e, ttft = actual.get('request_e2e_seconds'), actual.get('ttft_seconds')
        count = (actual.get('usage') or {}).get('completion_tokens')
        if not finite(e2e) or not finite(ttft) or not 0 < ttft <= e2e or type(count) is not int or count != 8:
            return None
        times.append(e2e)
        ttfts.append(ttft)
        tokens.append(count)
    return dict(request_e2e_seconds=statistics.median(times), ttft_seconds=statistics.median(ttfts),
                qualified_goodput_tokens_per_second=sum(tokens) / sum(times),
                measured_slots=5, quality_qualified_tokens=sum(tokens), client_e2e_denominator_seconds=sum(times))


def summarize(report):
    rows = report.get('lifecycles', [])
    by_trial = {row['trial']: row for row in rows}
    require(len(by_trial) == len(rows), 'duplicate lifecycle cannot repair a failed pair')
    order = report.get('order', [])
    require([row.get('block') for row in order] == [f'P{i}' for i in range(1, 6)], 'all five planned paired blocks required')
    expected = {block['block'] + '-' + mode for block in order for mode in ('blocking', 'lookahead')}
    require(set(by_trial) == expected, 'ten planned lifecycles including unissued placeholders required')
    blocks = []
    for block in order:
        arms = {mode: arm_metrics(by_trial[block['block'] + '-' + mode]) for mode in ('blocking', 'lookahead')}
        qualified = all(arms.values())
        ratios = {metric: math.log(arms['lookahead'][metric] / arms['blocking'][metric]) for metric in METRICS} if qualified else None
        blocks.append(dict(block=block['block'], execution_order=block['modes'], arms=arms,
                           qualified=qualified, log_ratios=ratios, failure_pair_retained=not qualified))
    complete = all(row['qualified'] for row in blocks)
    aggregate = {metric: paired_interval([row['log_ratios'][metric] for row in blocks]) for metric in METRICS} if complete else None
    if aggregate:
        aggregate['qualified_goodput_tokens_per_second'].pop('fractional_latency_reduction')
        aggregate['qualified_goodput_tokens_per_second']['fractional_goodput_increase'] = aggregate['qualified_goodput_tokens_per_second']['ratio'] - 1
    denominators = {key: sum(row['summary'][key] for row in rows) for key in
                    ('scheduled', 'issued', 'terminal', 'successful', 'failed', 'censored', 'unissued')}
    measured = {key: sum(row['summary']['measured_qa'][key] for row in rows) for key in ('scheduled', 'success', 'eligible_success')}
    require(denominators['scheduled'] == 120 and measured['scheduled'] == 50
            and denominators['issued'] + denominators['unissued'] == 120
            and denominators['successful'] + denominators['failed'] + denominators['censored'] == denominators['issued'],
            'failure/censoring denominators do not conserve preregistered slots')
    return dict(scheduled_blocks=5, qualified_blocks=sum(row['qualified'] for row in blocks),
        failed_or_incomplete_blocks=sum(not row['qualified'] for row in blocks), blocks=blocks,
        aggregate=aggregate, aggregate_policy='all five pairs or null; no complete-case replacement/selection',
        denominators=denominators, measured_qa=measured, physical_nas_backend_bytes=None,
        primary_metric='request_e2e_seconds', secondary_metrics=list(METRICS[1:]),
        profiling_included=False, scientific_acceptance=False)
