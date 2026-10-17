"""Prespecified CPU-only paired summaries; missing pairs never disappear."""
from __future__ import annotations

import itertools
import math
import statistics

from runnerhelpers.evidence_v2 import finite, require
from runnerhelpers.http_expectations import (
    BOUND_HTTP_REFERENCE, HISTORICAL_LENGTH, http_output_matches, normalize_http_expectation,
)

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


def summarize(report, *, contract=None, reference_context=None, producer_contract=None):
    if report.get('study_layout', {}).get('schema') == TRANSFER_LAYOUT:
        return summarize_transfer(report, contract=contract, reference_context=reference_context,
            producer_contract=producer_contract)
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


TRANSFER_LAYOUT = 'ferrule.storage-demand.http-transfer-layout.v1'
POLICIES = ('synchronous', 'grouped-synchronous')


def transfer_order():
    return [dict(block=f'{schedule[0].upper()}{i}', read_schedule=schedule,
                 modes=[schedule + '-' + policy for policy in policies])
        for schedule in ('blocking', 'lookahead')
        for i, policies in enumerate((POLICIES, POLICIES[::-1]), 1)]


def transfer_trials(order=None):
    return [block['block'] + '-' + mode for block in (transfer_order() if order is None else order)
            for mode in block['modes']]


def request_metrics(planned, observed=None, *, contract=HISTORICAL_LENGTH, reference_context=None):
    """Scheduled-slot metrics, including failures; token counts are terminal usage, not chunks."""
    from runnerhelpers.evidence_v2 import integer
    observed = observed or {}
    actual, trace = observed.get('actual', {}), observed.get('trace', {})
    usage = actual.get('usage') if isinstance(actual.get('usage'), dict) else {}
    dispatch, sent, ended = (actual.get(k) for k in ('client_dispatch_at', 'sent_at', 'ended_at'))
    content = actual.get('content_event_times', [])
    first = content[0] if content and finite(content[0]) else None
    admission = observed.get('admission', {}).get('monotonic')
    def elapsed(end, start):
        return end - start if finite(end) and finite(start) and end >= start else None
    native_e2e, native_ttft = elapsed(ended, sent), elapsed(first, sent)
    timing = all(finite(x) for x in (admission, dispatch, sent, first, ended)) and admission <= dispatch <= sent <= first <= ended
    timing = timing and native_ttft > 0 and native_e2e > 0 and all(
        finite(actual.get(k)) and math.isclose(actual[k], v, rel_tol=1e-9, abs_tol=1e-9)
        for k, v in (('ttft_seconds', native_ttft), ('request_e2e_seconds', native_e2e)))
    success = observed.get('status') == actual.get('status') == 'success' and finite(ended)
    quality = success and observed.get('issued') is True and observed.get('metric_eligible') is True and timing
    quality = quality and trace.get('terminal_reconciled') is True and http_output_matches(
        planned, actual, trace, contract=contract, reference_context=reference_context)
    quality = quality and integer(observed.get('server_http_admission_us'))
    count = usage.get('completion_tokens')
    count = count if integer(count) else None
    return dict(name=planned['name'], population=planned['population'], measured=planned['measured'],
        status=observed.get('status', 'unissued'), issued=observed.get('issued') is True,
        success=success, reference_qualified=bool(quality), completion_tokens=count,
        unknown_partial_tokens=count is None and observed.get('issued') is True,
        client_dispatch_at=dispatch, sent_at=sent, ended_at=ended,
        native_ttft_seconds=native_ttft, native_e2e_seconds=native_e2e,
        dispatch_ttft_seconds=elapsed(first, dispatch), dispatch_e2e_seconds=elapsed(ended, dispatch),
        admission_ttft_seconds=elapsed(first, admission), admission_e2e_seconds=elapsed(ended, admission),
        send_dispatch_delay_seconds=elapsed(sent, dispatch), server_http_admission_us=observed.get('server_http_admission_us'),
        actual_generated_ids=trace.get('actual_generated_ids'), text=actual.get('text'),
        finish_reason=actual.get('finish_reason'), usage=actual.get('usage'),
        route_ids=None, logits=None)


def transfer_arm_metrics(row, schedule, *, contract=HISTORICAL_LENGTH, reference_context=None, producer_contract=None):
    from runnerhelpers.trace_v2 import transfer_read_continuity
    lifecycle = row.get('lifecycle', {})
    requests = lifecycle.get('requests', {})
    observations = requests.get('observations', [])
    by_name = {r['name']: r for r in observations}
    require(len(by_name) == len(observations) and [r['name'] for r in observations]
            == [r['name'] for r in schedule[:len(observations)]],
            'duplicate/unknown request cannot repair a scheduled slot')
    slots = [request_metrics(planned, by_name.get(planned['name']),
        contract=contract, reference_context=reference_context) for planned in schedule]
    measured = [r for r in slots if r['measured']]
    require(len(measured) == 5, 'frozen five measured QA slots required')
    issued = [r for r in measured if r['issued']]
    starts = [r['client_dispatch_at'] for r in issued if finite(r['client_dispatch_at'])]
    ends = [r['ended_at'] for r in issued if finite(r['ended_at'])]
    known_span = max(ends) - min(starts) if starts and ends and max(ends) > min(starts) else None
    wall_known = bool(issued) and len(starts) == len(ends) == len(issued)
    wall_known = wall_known and all(r['ended_at'] >= r['client_dispatch_at'] for r in issued)
    wall = known_span if wall_known else None
    successful = [r for r in measured if r['success']]
    qualified = [r for r in measured if r['reference_qualified']]
    counts = {k: sum(r['status'] == k for r in measured) for k in ('success', 'failed', 'censored', 'unissued')}
    require(sum(counts.values()) == 5, 'unknown scheduled outcome')
    def token_sum(rows):
        return sum(r['completion_tokens'] for r in rows) if all(r['completion_tokens'] is not None for r in rows) else None
    def rate(n):
        return n / wall if n is not None and wall is not None else None
    transfer = requests.get('expert_transfers', {})
    continuity = transfer_read_continuity([r.get('logical_counters', {}) for r in observations],
        transfer.get('requested_case', {}).get('read_options'), initial_read=requests.get('initial_read_endpoint'),
        producer_contract=producer_contract)
    accepted = row.get('assessment', {}).get('can_continue') is True and row.get('read_ledger', {}).get('can_continue') is True
    accepted = accepted and lifecycle.get('status') == 'complete'
    accepted = accepted and lifecycle.get('closure', {}).get('final', {}).get('can_continue') is True
    accepted = accepted and continuity['can_continue'] and len(qualified) == 5 and all(r['reference_qualified'] for r in slots)
    total_e2e = sum(r['native_e2e_seconds'] for r in qualified) if len(qualified) == 5 else None
    return dict(requests=slots, scheduled=5, counts=counts, successful=len(successful), reference_qualified=len(qualified),
        success_fraction=len(successful) / 5, qualified_fraction=len(qualified) / 5 if accepted else 0,
        request_only_reference_qualified_fraction=len(qualified) / 5,
        qualified_arm=bool(accepted), read_continuity={k: continuity[k] for k in
            ('can_continue', 'read_continuity', 'stop_reason')},
        phase_wall_seconds=wall, observed_span_lower_bound_seconds=known_span,
        phase_wall_definition='first MEASURED client dispatch to last MEASURED client-observed terminal; includes gaps/failures',
        phase_requests_per_second=rate(len(successful)), phase_tokens_per_second=rate(token_sum(successful)),
        phase_request_only_reference_qualified_requests_per_second=rate(len(qualified)),
        phase_request_only_reference_qualified_tokens_per_second=rate(token_sum(qualified)),
        phase_qualified_requests_per_second=rate(len(qualified)) if accepted else None,
        phase_qualified_tokens_per_second=rate(token_sum(qualified)) if accepted else None,
        serialized_service_normalized_qa8_tokens_per_second=40 / total_e2e if total_e2e else None,
        serialized_service_reference_tokens_per_second=token_sum(qualified) / total_e2e if total_e2e else None,
        serialized_service_scope='request-only reference-qualified; QA8 normalization is not actual EOS token throughput',
        serialized_service_denominator_seconds=total_e2e,
        median_request_e2e_seconds=statistics.median(r['native_e2e_seconds'] for r in qualified) if len(qualified) == 5 else None,
        median_ttft_seconds=statistics.median(r['native_ttft_seconds'] for r in qualified) if len(qualified) == 5 else None,
        unknown_partial_token_slots=sum(r['unknown_partial_tokens'] for r in measured))


def summarize_transfer(report, *, contract=None, reference_context=None, producer_contract=None):
    layout, rows = report['study_layout'], report.get('lifecycles', [])
    require(layout.get('schema') == TRANSFER_LAYOUT and report.get('order') == transfer_order(), 'unknown transfer layout/order')
    by_trial = {r['trial']: r for r in rows}
    require(len(by_trial) == len(rows) and set(by_trial) == set(transfer_trials()),
            'all eight planned lifecycles, including unissued placeholders, required')
    schedule = layout['schedule']
    contract = layout.get('http_expectation_contract', HISTORICAL_LENGTH) if contract is None else contract
    reference_context = layout.get('http_reference_context') if reference_context is None else reference_context
    producer_contract = layout.get('upload_producer_contract') if producer_contract is None else producer_contract
    blocks = [transfer_pair_metrics(block, by_trial, schedule,
        contract=contract, reference_context=reference_context, producer_contract=producer_contract)
        for block in transfer_order()]
    summaries = {}
    for schedule_name in ('blocking', 'lookahead'):
        selected = [b for b in blocks if b['read_schedule'] == schedule_name]
        complete = all(b['qualified'] for b in selected)
        summaries[schedule_name] = dict(scheduled_pairs=2, qualified_pairs=sum(b['qualified'] for b in selected),
            median_paired_ratios={k: statistics.median(b['grouped_over_synchronous'][k] for b in selected)
                for k in selected[0]['grouped_over_synchronous']} if complete else None,
            interval=None, significance_claim=False, independent_unit='PAIR, not request')
    denominators = {key: sum(r['summary'][key] for r in rows) for key in
        ('scheduled', 'issued', 'terminal', 'successful', 'failed', 'censored', 'unissued')}
    require(denominators['scheduled'] == 96 and denominators['issued'] + denominators['unissued'] == 96
        and denominators['successful'] + denominators['failed'] + denominators['censored'] == denominators['issued'],
        'scheduled failure/censoring denominators do not conserve 96 slots')
    all_arms = [arm for block in blocks for arm in block['arms'].values()]
    measured_qa = {key: sum(arm['counts'][key] for arm in all_arms)
        for key in ('success', 'failed', 'censored', 'unissued')}
    measured_qa.update(scheduled=40, reference_qualified=sum(arm['reference_qualified'] for arm in all_arms),
        lifecycle_qualified=sum(arm['reference_qualified'] for arm in all_arms if arm['qualified_arm']),
        unknown_partial_token_slots=sum(arm['unknown_partial_token_slots'] for arm in all_arms))
    require(sum(measured_qa[k] for k in ('success', 'failed', 'censored', 'unissued')) == 40,
            'measured scheduled denominator changed')
    expected_measured_tokens = 320
    if contract == BOUND_HTTP_REFERENCE:
        expected_measured_tokens = len(by_trial) * sum(normalize_http_expectation(
            planned, contract=contract, reference_context=reference_context)['actual_output_tokens']
            for planned in schedule if planned['measured'])
    return dict(schema='ferrule.storage-demand.http-transfer-analysis.v1', measured_qa=measured_qa, blocks=blocks, by_read_schedule=summaries,
        denominators=denominators, scheduled_measured_qa=40, expected_measured_tokens=expected_measured_tokens,
        scheduled_measured_token_ceiling=320,
        aggregate_policy='both pairs per schedule or null; no pooling schedules, dropping failures or replacements',
        pilot_only=True, scientific_acceptance=False, vllm_comparison=False,
        routes_observed=False, logits_observed=False, grouped_backend_call_count=None)


def transfer_pair_metrics(block, by_trial, schedule, *, contract=HISTORICAL_LENGTH, reference_context=None, producer_contract=None):
    from runnerhelpers.trace_v2 import TRANSFER_WORK
    trials = [block['block'] + '-' + block['read_schedule'] + '-' + p for p in POLICIES]
    arms = {policy: transfer_arm_metrics(by_trial[trial], schedule,
        contract=contract, reference_context=reference_context, producer_contract=producer_contract)
        for policy, trial in zip(POLICIES, trials)}
    signatures, work = [], []
    for trial in trials:
        observations = by_trial[trial].get('lifecycle', {}).get('requests', {}).get('observations', [])
        signatures.append([(r['name'], r.get('trace', {}).get('actual_generated_ids'),
            r.get('actual', {}).get('text'), r.get('actual', {}).get('finish_reason'),
            r.get('actual', {}).get('usage')) for r in observations])
        work.append([[(tuple(inv['fields'].get(k) for k in TRANSFER_WORK)) for inv in
            r.get('logical_counters', {}).get('rows', [])] for r in observations])
    equal_output = len(signatures[0]) == 12 and signatures[0] == signatures[1]
    equal_work = bool(work[0]) and all(work[0]) and work[0] == work[1]
    qualified = all(a['qualified_arm'] for a in arms.values()) and equal_output and equal_work
    metrics = ('median_request_e2e_seconds', 'median_ttft_seconds',
        'serialized_service_normalized_qa8_tokens_per_second', 'serialized_service_reference_tokens_per_second',
        'phase_qualified_tokens_per_second')
    ratios = {k: arms[POLICIES[1]][k] / arms[POLICIES[0]][k] for k in metrics} if qualified else None
    return dict(**block, arms=arms, qualified=qualified, crosspolicy_exact_output=equal_output,
        crosspolicy_equal_work=equal_work, grouped_over_synchronous=ratios, failure_pair_retained=not qualified)
