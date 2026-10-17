"""CPU-only regressions of shared HTTP semantics, metric accounting and raw read continuity."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from paired_read_v1 import analysis, quality
from runnerhelpers import http_expectations as expected, trace_v2 as trace
from runnerhelpers.evidence_v2 import ct, COUNTERS, censoring_summary
import test_quality_v1 as workers


def schedule():
    return ct.load(HERE / 'requests.json')['schedule']


def reference_context(rows, *, short_qa=False):
    references = {}
    for row in rows:
        ids = row['reference']['reference_ids'] if row['reference'] else [1, 2]
        text = row['reference']['reference_text'] if row['reference'] else 'Hello fixture'
        short = short_qa and row['reference_id'] == 'qa'
        if short:
            ids, text = ids[:3], 'CPU fixture EOS output'
        references[row['name']] = dict(
            **{k: copy.deepcopy(row[k]) for k in ('prompt', 'input_tokens', 'max_tokens', 'payload')},
            reference_ids=copy.deepcopy(ids), reference_text=text, finish_reason='stop' if short else 'length',
            usage=dict(prompt_tokens=row['input_tokens'], completion_tokens=len(ids),
                       total_tokens=row['input_tokens'] + len(ids)))
    context = dict(source='/cpu-fixture/not-a-qualification/http-references.json',
                   document=dict(schema=expected.REFERENCE_SCHEMA, references=references))
    return rebind_fixture(context)


def rebind_fixture(context):
    """Fixture reviewer only: byte hash deliberately differs from parsed-document digest."""
    context['source_sha256'] = hashlib.sha256(json.dumps(context['document'], indent=2).encode()).hexdigest()
    context['reviewed_binding'] = dict(source=context['source'], source_sha256=context['source_sha256'],
                                      document_sha256=ct.digest(context['document']))
    return context


def invocation(index, mode):
    options = workers.options(mode)
    read = workers.observation(options, reads=6 * (index + 1), reuses=index)
    before = workers.observation(options, configured=index == 0, reads=6 * index, reuses=max(0, index - 1))
    delta = {k: read['stats'][k] - before['stats'][k] for k in quality.V3_COUNTERS}
    delta.update({k: read['stats'][k] for k in quality.V3_PEAKS})
    return dict(event='mixed_decoder_invocation', status='completed', device=0,
        counters_available=True, counter_reset_detected=False, quarantined=False,
        **dict.fromkeys(COUNTERS, 1), rows=1, sequences=1, prefill_rows=1, decode_rows=0,
        expert_cache_counters_overflowed=False, expert_read_counter_reset_detected=False,
        expert_pending_upload_bytes=0, expert_unknown_quarantine_bytes=0, facade_stream_sync_failures=0,
        expert_resident_entries=1, expert_resident_bytes=1024, expert_scratch_bytes=8 << 20,
        expert_owner_peak_bytes=(8 << 20) + 1024,
        expert_read_pipeline=json.dumps(read), expert_read_pipeline_delta=json.dumps(delta))


def window(fields, mode, **boundary):
    return trace.transfer_counter_window([json.dumps(dict(fields=fields))], workers.options(mode), **boundary)


def zero_delta(fields):
    delta = json.loads(fields['expert_read_pipeline_delta'])
    delta.update(dict.fromkeys(quality.V3_COUNTERS, 0))
    fields['expert_read_pipeline_delta'] = json.dumps(delta)
    return fields


def arm(mode='blocking', policy='synchronous', *, context=None):
    rows, observations = schedule(), []
    initial = workers.observation(workers.options(mode), configured=True)
    previous = initial
    for i, planned in enumerate(rows):
        contract = expected.BOUND_HTTP_REFERENCE if context is not None else expected.HISTORICAL_LENGTH
        ref = expected.normalize_http_expectation(planned, contract=contract, reference_context=context)
        dispatch = 100. + i * 10.
        counters = window(invocation(i, mode), mode, previous_read=previous)
        previous = counters['final_read_endpoint']
        observations.append(dict(planned, issued=True, status='success', metric_eligible=True,
            admission=dict(monotonic=dispatch - .5), server_http_admission_us=12,
            trace=dict(terminal_reconciled=True, actual_generated_ids=ref['reference_ids'] or [1, 2]),
            logical_counters=counters,
            actual=dict(status='success', adapter_http_report_complete=True,
                client_dispatch_at=dispatch, sent_at=dispatch + .25, ended_at=dispatch + 2.25,
                client_observed_at=dispatch + 2.5, done_at=dispatch + 2.,
                content_event_times=[dispatch + .75], ttft_seconds=.5, request_e2e_seconds=2.,
                actual_output_tokens=ref['actual_output_tokens'], text=ref['reference_text'] or 'Hello fixture',
                finish_reason=ref['finish_reason'], usage=copy.deepcopy(ref['usage']))))
    return dict(mode=mode + '-' + policy, assessment=dict(can_continue=True), read_ledger=dict(can_continue=True),
        lifecycle=dict(status='complete', requests=dict(observations=observations, initial_read_endpoint=initial,
            expert_transfers=dict(requested_case=dict(read_schedule=mode,
                transfer_policy=policy, read_options=workers.options(mode)))),
            closure=dict(final=dict(can_continue=True))), summary=censoring_summary(observations, rows))


class CPUOnly(unittest.TestCase):
    def setUp(self):
        for name in ('subprocess.Popen', 'subprocess.run', 'os.kill'):
            self.enterContext(mock.patch(name, side_effect=AssertionError('CPU semantics only; no process/GPU/signal')))
        self.enterContext(mock.patch.object(ct, 'exclusive_owner_lock', side_effect=AssertionError('no GPU locks')))


class HTTPExpectationTests(CPUOnly):
    def test_bound_short_stop_qualifies_without_payload_change(self):
        rows = schedule()
        frozen_payloads = copy.deepcopy([row['payload'] for row in rows])
        context = reference_context(rows, short_qa=True)
        for mode in ('blocking', 'lookahead'):
            for policy in analysis.POLICIES:
                row = arm(mode, policy, context=context)['lifecycle']['requests']['observations'][-1]
                result = analysis.request_metrics(rows[-1], row,
                    contract=expected.BOUND_HTTP_REFERENCE, reference_context=context)
                self.assertTrue(result['reference_qualified'])
                self.assertEqual(result['completion_tokens'], 3)
                self.assertEqual(result['finish_reason'], 'stop')
        self.assertEqual([row['payload'] for row in rows], frozen_payloads)
        self.assertTrue(all('ignore_eos' not in payload for payload in frozen_payloads))

    def test_explicit_contract_keeps_historical_length_semantics(self):
        planned = schedule()[-1]
        short_context = reference_context(schedule(), short_qa=True)
        short = arm(context=short_context)['lifecycle']['requests']['observations'][-1]
        self.assertFalse(analysis.request_metrics(planned, short)['reference_qualified'])
        self.assertFalse(expected.http_output_matches(planned, short['actual'], short['trace'],
            contract=expected.HISTORICAL_LENGTH, reference_context=short_context))
        full_context = reference_context(schedule())
        self.assertFalse(expected.http_output_matches(planned, short['actual'], short['trace'],
            contract=expected.BOUND_HTTP_REFERENCE, reference_context=full_context))
        full = arm()['lifecycle']['requests']['observations'][-1]
        self.assertTrue(analysis.request_metrics(planned, full)['reference_qualified'])

    def test_wrong_ids_text_finish_usage_and_count_reject(self):
        planned, context = schedule()[-1], reference_context(schedule(), short_qa=True)
        original = arm(context=context)['lifecycle']['requests']['observations'][-1]
        mutations = [lambda r: r['trace'].update(actual_generated_ids=[1, 2, 3]),
            lambda r: r['actual'].update(text='wrong'), lambda r: r['actual'].update(finish_reason='length'),
            lambda r: r['actual'].update(actual_output_tokens=8),
            lambda r: r['actual']['usage'].update(completion_tokens=8),
            lambda r: r['actual']['usage'].update(total_tokens=9),
            lambda r: r['actual']['usage'].update(prompt_tokens=True),
            lambda r: r['actual'].update(actual_output_tokens=True),
            lambda r: r['actual'].update(usage=['not terminal usage']),
            lambda r: r['trace'].update(terminal_reconciled=False)]
        for mutate in mutations:
            changed = copy.deepcopy(original)
            mutate(changed)
            self.assertFalse(analysis.request_metrics(planned, changed,
                contract=expected.BOUND_HTTP_REFERENCE, reference_context=context)['reference_qualified'])

    def test_execution_gate_uses_same_bound_eos_semantics(self):
        from runnerhelpers import execution_v2 as execution
        planned, context = schedule()[-1], reference_context(schedule(), short_qa=True)
        row = arm(context=context)['lifecycle']['requests']['observations'][-1]
        row['trace'].update(metric_eligible=True, worker_terminal_enqueue=[dict(
            usage_available=True, delivery_boundary='worker_channel_enqueue', **row['actual']['usage'])])
        arguments = (planned, row['actual'], row['trace'], row['logical_counters'], dict(status='shared_client_delta'))
        self.assertTrue(execution.request_gate(*arguments, expectation_contract=expected.BOUND_HTTP_REFERENCE,
            reference_context=context))
        self.assertFalse(execution.request_gate(*arguments))
        row['actual']['finish_reason'] = 'length'
        self.assertFalse(execution.request_gate(*arguments, expectation_contract=expected.BOUND_HTTP_REFERENCE,
            reference_context=context))

    def test_reference_context_requires_exact_reviewed_source_and_document_hash(self):
        planned, context = schedule()[-1], reference_context(schedule(), short_qa=True)
        mutations = [lambda c: c.update(source='/different/reference.json'),
            lambda c: c.update(source_sha256='0' * 64),
            lambda c: c['reviewed_binding'].update(document_sha256='0' * 64),
            lambda c: c['document']['references'][planned['name']].update(reference_text='mutated'),
            lambda c: c.pop('reviewed_binding')]
        for mutate in mutations:
            changed = copy.deepcopy(context)
            mutate(changed)
            with self.assertRaises(ValueError):
                expected.normalize_http_expectation(planned, contract=expected.BOUND_HTTP_REFERENCE,
                    reference_context=changed)
        with self.assertRaises(ValueError):
            expected.normalize_http_expectation(planned, contract=expected.BOUND_HTTP_REFERENCE)
        with self.assertRaises(ValueError):
            expected.normalize_http_expectation(planned, contract='unrecognized-contract')

    def test_bound_reference_semantics_and_request_crossbinding_fail_closed(self):
        planned, context = schedule()[-1], reference_context(schedule(), short_qa=True)
        mutations = [lambda r: r.update(finish_reason='length'),
            lambda r: r.update(reference_ids=[1] * 9), lambda r: r.update(reference_ids=[False, 1, 2]),
            lambda r: r.update(reference_text=None), lambda r: r['usage'].update(completion_tokens=4),
            lambda r: r.update(prompt='foreign prompt'), lambda r: r['payload'].update(ignore_eos=True),
            lambda r: r.update(input_tokens=True)]
        for mutate in mutations:
            changed = copy.deepcopy(context)
            mutate(changed['document']['references'][planned['name']])
            rebind_fixture(changed)
            with self.assertRaises(ValueError):
                expected.normalize_http_expectation(planned, contract=expected.BOUND_HTTP_REFERENCE,
                    reference_context=changed)


class ReadContinuityTests(CPUOnly):
    def test_explicit_baseline_and_previous_endpoint_reconcile_both_schedules_policies(self):
        for mode in ('blocking', 'lookahead'):
            for policy in analysis.POLICIES:
                with self.subTest(mode=mode, policy=policy):
                    initial = workers.observation(workers.options(mode), configured=True)
                    first = window(invocation(0, mode), mode, initial_read=initial)
                    second = window(invocation(1, mode), mode, previous_read=first['final_read_endpoint'])
                    self.assertTrue(first['read_continuity_valid'])
                    self.assertTrue(second['read_continuity_valid'])
                    result = trace.transfer_read_continuity([first, second], workers.options(mode), initial_read=initial)
                    self.assertTrue(result['can_continue'], result['stop_reason'])
                    self.assertEqual(result['final_read_endpoint'], second['final_read_endpoint'])
                    self.assertEqual(second['rows'][0]['read_delta']['submitted_reads'], 6)

    def test_missing_baseline_never_infers_zero_or_claims_reconciled(self):
        local = window(invocation(0, 'blocking'), 'blocking')
        self.assertTrue(local['valid'])
        self.assertFalse(local['read_continuity_valid'])
        self.assertEqual(local['read_continuity'], 'unobserved_initial_baseline')
        result = trace.transfer_read_continuity([local], workers.options(), initial_read=None)
        self.assertFalse(result['can_continue'])
        self.assertIn('unobserved', result['stop_reason'])

    def test_zero_first_invocation_delta_rejected_at_initial_and_each_request_boundary(self):
        for mode in ('blocking', 'lookahead'):
            for policy in analysis.POLICIES:
                for index in range(3):
                    with self.subTest(mode=mode, policy=policy, window=index):
                        initial = workers.observation(workers.options(mode), configured=True)
                        windows = [window(zero_delta(invocation(i, mode)) if i == index else invocation(i, mode), mode)
                                   for i in range(3)]
                        self.assertTrue(windows[index]['valid'])  # Local validation is not continuity.
                        result = trace.transfer_read_continuity(windows, workers.options(mode), initial_read=initial)
                        self.assertFalse(result['can_continue'])
                        self.assertIn('delta mismatch', result['stop_reason'])

    def test_multiple_invocations_and_unaccounted_interrequest_work(self):
        for mode in ('blocking', 'lookahead'):
            initial = workers.observation(workers.options(mode), configured=True)
            first = trace.transfer_counter_window([json.dumps(dict(fields=invocation(i, mode))) for i in (0, 1)],
                workers.options(mode), initial_read=initial)
            self.assertTrue(first['read_continuity_valid'])
            second = window(invocation(2, mode), mode)
            self.assertTrue(trace.transfer_read_continuity([first, second], workers.options(mode),
                initial_read=initial)['can_continue'])
            skipped = window(invocation(3, mode), mode)
            result = trace.transfer_read_continuity([first, skipped], workers.options(mode), initial_read=initial)
            self.assertFalse(result['can_continue'])
            self.assertIn('delta mismatch', result['stop_reason'])

    def test_optional_upload_zeros_require_explicit_reviewed_producer(self):
        for mode in ('blocking', 'lookahead'):
            initial = workers.observation(workers.options(mode), configured=True)
            fields = invocation(0, mode)
            read = json.loads(fields['expert_read_pipeline'])
            read['current_charge']['unknown_upload_bytes'] = 0
            fields['expert_read_pipeline'] = json.dumps(read)
            self.assertFalse(window(fields, mode, initial_read=initial)['valid'])
            bound = window(fields, mode, initial_read=initial, producer_contract=quality.TRANSFER_UPLOAD_PRODUCER)
            self.assertTrue(bound['valid'])
            self.assertTrue(trace.transfer_read_continuity([bound], workers.options(mode), initial_read=initial,
                producer_contract=quality.TRANSFER_UPLOAD_PRODUCER)['can_continue'])
            read['current_charge']['unknown_upload_bytes'] = 1
            fields['expert_read_pipeline'] = json.dumps(read)
            self.assertFalse(window(fields, mode, initial_read=initial,
                producer_contract=quality.TRANSFER_UPLOAD_PRODUCER)['valid'])

    def test_reset_unknown_pending_and_prior_invalid_cannot_heal(self):
        for mode in ('blocking', 'lookahead'):
            mutations = [lambda f: f.update(counter_reset_detected=True),
                lambda f: f.update(expert_read_counter_reset_detected=True),
                lambda f: f.update(expert_cache_counters_overflowed=True),
                lambda f: f.update(expert_pending_upload_bytes=1)]
            for mutate in mutations:
                bad = invocation(1, mode)
                mutate(bad)
                windows = [window(invocation(0, mode), mode), window(bad, mode), window(invocation(2, mode), mode)]
                initial = workers.observation(workers.options(mode), configured=True)
                self.assertFalse(trace.transfer_read_continuity(windows, workers.options(mode), initial_read=initial)['can_continue'])
                # Even forged aggregate PASS flags cannot hide the invalid raw invocation.
                windows[1].update(valid=True, transfer_health_valid=True)
                self.assertFalse(trace.transfer_read_continuity(windows, workers.options(mode), initial_read=initial)['can_continue'])
            initial['worker_unknown'] = True
            self.assertFalse(trace.transfer_read_continuity([window(invocation(0, mode), mode)],
                workers.options(mode), initial_read=initial)['can_continue'])
            reset = [window(invocation(1, mode), mode), window(invocation(0, mode), mode)]
            self.assertFalse(trace.transfer_read_continuity(reset, workers.options(mode),
                initial_read=workers.observation(workers.options(mode), configured=True))['can_continue'])


class TransferMetricTests(CPUOnly):
    def test_phase_wall_native_dispatch_admission_and_serialized_denominators(self):
        result = analysis.transfer_arm_metrics(arm(), schedule())
        self.assertTrue(result['qualified_arm'], result['read_continuity'])
        self.assertEqual(result['phase_wall_seconds'], 42.25)
        self.assertEqual(result['serialized_service_denominator_seconds'], 10.)
        self.assertEqual(result['serialized_service_normalized_qa8_tokens_per_second'], 4.)
        self.assertAlmostEqual(result['phase_qualified_tokens_per_second'], 40 / 42.25)
        last = result['requests'][-1]
        for key, value in (('native_ttft_seconds', .5), ('native_e2e_seconds', 2.),
                           ('dispatch_e2e_seconds', 2.25), ('admission_e2e_seconds', 2.75)):
            self.assertEqual(last[key], value)

    def test_eos_throughput_uses_observed_reference_tokens_not_qa8_normalization(self):
        context = reference_context(schedule(), short_qa=True)
        for mode in ('blocking', 'lookahead'):
            for policy in analysis.POLICIES:
                result = analysis.transfer_arm_metrics(arm(mode, policy, context=context), schedule(),
                    contract=expected.BOUND_HTTP_REFERENCE, reference_context=context)
                self.assertTrue(result['qualified_arm'])
                self.assertEqual(result['reference_qualified'], 5)
                self.assertAlmostEqual(result['phase_tokens_per_second'], 15 / 42.25)
                self.assertAlmostEqual(result['phase_qualified_tokens_per_second'], 15 / 42.25)
                self.assertEqual(result['serialized_service_reference_tokens_per_second'], 1.5)
                self.assertEqual(result['serialized_service_normalized_qa8_tokens_per_second'], 4.)

    def test_failed_ledger_assessment_closure_or_baseline_keep_request_only_metrics(self):
        mutations = [lambda r: r['read_ledger'].update(can_continue=False),
            lambda r: r['assessment'].update(can_continue=False),
            lambda r: r['lifecycle'].update(status='cleanup_unverified'),
            lambda r: r['lifecycle']['closure']['final'].update(can_continue=False),
            lambda r: r['lifecycle'].pop('closure'),
            lambda r: r['lifecycle']['requests'].pop('initial_read_endpoint')]
        for mode in ('blocking', 'lookahead'):
            for policy in analysis.POLICIES:
                for mutate in mutations:
                    row = arm(mode, policy)
                    raw = copy.deepcopy(row['lifecycle']['requests']['observations'])
                    mutate(row)
                    result = analysis.transfer_arm_metrics(row, schedule())
                    self.assertFalse(result['qualified_arm'])
                    self.assertEqual(result['reference_qualified'], 5)
                    self.assertEqual(result['request_only_reference_qualified_fraction'], 1.)
                    self.assertIsNone(result['phase_qualified_requests_per_second'])
                    self.assertIsNone(result['phase_qualified_tokens_per_second'])
                    self.assertAlmostEqual(result['phase_request_only_reference_qualified_tokens_per_second'], 40 / 42.25)
                    self.assertEqual(row['lifecycle']['requests']['observations'], raw)

    def test_analysis_rechecks_raw_boundary_even_with_forged_read_ledger_pass(self):
        for mode in ('blocking', 'lookahead'):
            for policy in analysis.POLICIES:
                row = arm(mode, policy)
                counters = row['lifecycle']['requests']['observations'][7]['logical_counters']
                zero_delta(counters['rows'][0]['fields'])
                result = analysis.transfer_arm_metrics(row, schedule())
                self.assertFalse(result['qualified_arm'])
                self.assertFalse(result['read_continuity']['can_continue'])
                self.assertEqual(result['reference_qualified'], 5)
                self.assertIsNone(result['phase_qualified_tokens_per_second'])

    def test_failures_unknown_partial_tokens_and_unissued_slots_remain(self):
        row = arm()
        observations = row['lifecycle']['requests']['observations']
        observations[-2].update(status='censored', metric_eligible=False)
        observations[-2]['actual'].update(status='timeout', usage=None, actual_output_tokens=None)
        observations[-1].update(status='failed', metric_eligible=False)
        observations[-1]['actual'].update(status='http_error', usage=None, actual_output_tokens=None)
        result = analysis.transfer_arm_metrics(row, schedule())
        self.assertEqual(result['counts'], dict(success=3, failed=1, censored=1, unissued=0))
        self.assertEqual(result['unknown_partial_token_slots'], 2)
        self.assertEqual(result['phase_wall_seconds'], 42.25)
        self.assertAlmostEqual(result['phase_requests_per_second'], 3 / 42.25)
        self.assertIsNone(result['phase_qualified_tokens_per_second'])
        observations[-1]['actual'].pop('ended_at')
        self.assertIsNone(analysis.transfer_arm_metrics(row, schedule())['phase_wall_seconds'])
        self.assertEqual(analysis.transfer_arm_metrics({}, schedule())['counts']['unissued'], 5)

    def test_pair_statistics_remain_independent_by_schedule_and_retain_bad_pair(self):
        context, rows = reference_context(schedule(), short_qa=True), []
        for block in analysis.transfer_order():
            for mode in block['modes']:
                policy = mode[len(block['read_schedule']) + 1:]
                row = arm(block['read_schedule'], policy, context=context)
                row['trial'] = block['block'] + '-' + mode
                rows.append(row)
        report = dict(order=analysis.transfer_order(), lifecycles=rows,
            study_layout=dict(schema=analysis.TRANSFER_LAYOUT, schedule=schedule(),
                http_expectation_contract=expected.BOUND_HTTP_REFERENCE))
        result = analysis.summarize(report, reference_context=context)
        self.assertEqual(result['denominators']['scheduled'], 96)
        self.assertEqual(result['expected_measured_tokens'], 120)
        self.assertEqual(result['scheduled_measured_token_ceiling'], 320)
        self.assertEqual(result['measured_qa']['lifecycle_qualified'], 40)
        for summary in result['by_read_schedule'].values():
            self.assertEqual(summary['qualified_pairs'], 2)
            self.assertEqual(set(summary['median_paired_ratios'].values()), {1.})
            self.assertIsNone(summary['interval'])
            self.assertFalse(summary['significance_claim'])
        rows[0]['read_ledger']['can_continue'] = False
        result = analysis.summarize(report, reference_context=context)
        self.assertIsNone(result['by_read_schedule']['blocking']['median_paired_ratios'])
        self.assertEqual(result['by_read_schedule']['lookahead']['qualified_pairs'], 2)
        self.assertEqual(result['measured_qa']['reference_qualified'], 40)
        self.assertEqual(result['measured_qa']['lifecycle_qualified'], 35)
        self.assertEqual(len(result['blocks']), 4)
        self.assertEqual(result['expected_measured_tokens'], 120)
        self.assertEqual(result['scheduled_measured_token_ceiling'], 320)
        failed = rows[1]['lifecycle']['requests']['observations'][-1]
        failed.update(status='failed', metric_eligible=False)
        failed['actual'].update(status='http_error', usage=None, actual_output_tokens=None)
        rows[1]['summary'] = censoring_summary(rows[1]['lifecycle']['requests']['observations'], schedule())
        rows[2] = dict(trial=rows[2]['trial'], summary=censoring_summary([], schedule()))
        result = analysis.summarize(report, reference_context=context)
        self.assertEqual(result['measured_qa']['failed'], 1)
        self.assertEqual(result['measured_qa']['unissued'], 5)
        self.assertEqual(result['expected_measured_tokens'], 120)
        self.assertEqual(result['scheduled_measured_token_ceiling'], 320)

    def test_expected_tokens_include_unissued_arms_and_preserve_historical_contract(self):
        planned = schedule()
        context = reference_context(planned, short_qa=True)
        first_measured = next(row for row in planned if row['measured'])
        reference = context['document']['references'][first_measured['name']]
        reference['reference_ids'] = reference['reference_ids'][:2]
        reference['usage'].update(completion_tokens=2, total_tokens=first_measured['input_tokens'] + 2)
        rebind_fixture(context)
        for contract, reference_context_value, token_count in (
            (expected.BOUND_HTTP_REFERENCE, context, 112), (expected.HISTORICAL_LENGTH, None, 320)):
            with self.subTest(contract=contract):
                report = dict(order=analysis.transfer_order(), lifecycles=[
                    dict(trial=trial, summary=censoring_summary([], planned)) for trial in analysis.transfer_trials()],
                    study_layout=dict(schema=analysis.TRANSFER_LAYOUT, schedule=planned,
                        http_expectation_contract=contract))
                result = analysis.summarize(report, reference_context=reference_context_value)
                self.assertEqual(result['denominators']['issued'], 0)
                self.assertEqual(result['measured_qa']['unissued'], 40)
                self.assertEqual(result['expected_measured_tokens'], token_count)
                self.assertEqual(result['scheduled_measured_token_ceiling'], 320)


if __name__ == '__main__':
    unittest.main()
