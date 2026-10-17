"""CPU fixtures for maintained HTTP/metric functions, not a second runner."""
from contextlib import nullcontext
import copy
import json
from pathlib import Path
import sys
import time
import unittest
from unittest import mock

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / 'tests'))
sys.path.insert(0, str(HERE))
from paired_read_v1 import run_v3 as run, analysis, quality
from runnerhelpers import execution_v2 as ex, trace_v2 as trace
from runnerhelpers.evidence_v2 import ct, COUNTERS, censoring_summary
import test_quality_v1 as workers
import test_paired_read_v1 as shared
import test_execution_v2 as http_fixtures
from runnerhelpers import http_expectations as expectations


def fixture_layout():
    # Historical pure-metric fixtures never satisfy prospective launch admission.
    return dict(run.transfer_plan(), http_expectation_contract=expectations.HISTORICAL_LENGTH)


def schedule():
    return ct.load(HERE / 'requests.json')['schedule']


def invocation(index=0, read_schedule='blocking'):
    options = workers.options(read_schedule)
    read = workers.observation(options, reads=6 * (index + 1), reuses=index)
    before = workers.observation(options, configured=index == 0,
        reads=6 * index, reuses=max(0, index - 1))
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


def window(value, read_schedule='blocking'):
    return trace.transfer_counter_window([json.dumps(dict(fields=value))], workers.options(read_schedule),
        producer_contract=quality.TRANSFER_UPLOAD_PRODUCER)


def startup(policy):
    debug = 'Synchronous' if policy == 'synchronous' else 'GroupedSynchronous'
    return dict(message='configured resident expert transfers before engine composition',
        requested_expert_transfer_policy='Some(' + debug + ')', expert_transfer_policy=debug)


def arm(trial, read_schedule='blocking', policy='synchronous'):
    rows = []
    for i, planned in enumerate(schedule()):
        dispatch = 10.0 * i + 100.0
        ref = planned['reference']
        rows.append(dict(planned, issued=True, status='success', metric_eligible=True,
            admission=dict(monotonic=dispatch - .5), server_http_admission_us=12,
            trace=dict(terminal_reconciled=True, actual_generated_ids=ref['reference_ids'] if ref else [1, 2]),
            logical_counters=window(invocation(i, read_schedule), read_schedule),
            actual=dict(status='success', adapter_http_report_complete=True,
                client_dispatch_at=dispatch, sent_at=dispatch + .25, ended_at=dispatch + 2.25,
                content_event_times=[dispatch + .75], ttft_seconds=.5, request_e2e_seconds=2.,
                actual_output_tokens=planned['max_tokens'], text=ref['reference_text'] if ref else 'fixture',
                finish_reason='length', usage=dict(prompt_tokens=planned['input_tokens'],
                    completion_tokens=planned['max_tokens'], total_tokens=planned['input_tokens'] + planned['max_tokens']))))
    initial = workers.observation(workers.options(read_schedule), configured=True)
    previous = initial
    for row in rows:
        row['logical_counters'] = trace.transfer_counter_window(
            [json.dumps(dict(fields=r['fields'])) for r in row['logical_counters']['rows']],
            workers.options(read_schedule), previous_read=previous,
            producer_contract=quality.TRANSFER_UPLOAD_PRODUCER)
        previous = row['logical_counters']['final_read_endpoint']
    final = rows[-1]['logical_counters']['rows'][-1]['read_endpoint']
    case = dict(read_schedule=read_schedule, transfer_policy=policy, read_options=workers.options(read_schedule))
    observed = dict(status='complete', requests=dict(initial_read_endpoint=workers.observation(workers.options(read_schedule), configured=True),
        observations=rows, expert_transfers=dict(requested_case=case,
        startup=trace.transfer_startup([json.dumps(dict(fields=startup(policy)))], policy, 'a' * 64))),
        closure=dict(final=dict(can_continue=True), attempts=[dict(summary=dict(physical=dict(
            expert_read_pipeline=dict(state='observed', value=final))))]))
    return dict(trial=trial, lifecycle=observed, assessment=dict(can_continue=True),
        read_ledger=run.transfer_ledger(observed, read_schedule + '-' + policy),
        summary=censoring_summary(rows, schedule()))


class LayoutTests(shared.CPUOnly):
    def test_fixed_eight_arms_and_independent_read_policy_cases(self):
        value = run.transfer_plan()
        self.assertEqual(value['trial_order'], [
            'B1-blocking-synchronous', 'B1-blocking-grouped-synchronous',
            'B2-blocking-grouped-synchronous', 'B2-blocking-synchronous',
            'L1-lookahead-synchronous', 'L1-lookahead-grouped-synchronous',
            'L2-lookahead-grouped-synchronous', 'L2-lookahead-synchronous'])
        self.assertEqual((value['scheduled_starts'], value['scheduled_requests'], value['scheduled_measured_qa']), (8, 96, 40))
        self.assertEqual(value['schedule'], schedule())
        self.assertTrue(all('ignore_eos' not in r['payload'] for r in value['schedule']))
        self.assertEqual(value['limits']['pilot_seconds'], 7200)
        self.assertEqual(value['limits']['outer_lifecycle_seconds'], 650)
        candidate = dict(candidate=dict(binary=str(self.directory / 'fixture-cli'), binary_sha256='a' * 64))
        for read_schedule in ('blocking', 'lookahead'):
            argv = []
            for policy in analysis.POLICIES:
                case = value['cases'][read_schedule + '-' + policy]
                with mock.patch.object(run.historical, 'case_for_mode', return_value=case):
                    spec = run.planned.spec_for(candidate, http_fixtures.UUIDS[0], read_schedule + '-' + policy)
                argv.append(spec['argv'])
                self.assertEqual(spec['expert_transfer_case']['transfer_policy'], policy)
                self.assertEqual(spec['expert_transfer_case']['read_schedule'], read_schedule)
                self.assertEqual(spec['frozen_cli_sha256'], 'a' * 64)
                flags = dict(zip(spec['argv'][3::2], spec['argv'][4::2]))
                self.assertEqual(flags['--model-profile'], 'qwen35-mixed-bf16')
                self.assertEqual(flags['--expert-read-host-mb'], '64')
                self.assertEqual(flags['--decode-cohort-target'], '1')
                self.assertEqual(spec['environment']['CUBLAS_WORKSPACE_CONFIG'], ':0:0')
            self.assertEqual([(a, b) for a, b in zip(*argv) if a != b], [('synchronous', 'grouped-synchronous')])

    def test_unbound_layout_and_old_pin_fail_before_output_or_gpu(self):
        value = run.transfer_plan()
        path = self.directory / 'layout.json'
        ct.write(path, value)
        run.validate_transfer_layout(path, candidate_required=False)
        with self.assertRaisesRegex(ValueError, 'candidate pending'):
            run.validate_transfer_layout(path)
        for key in ('production_commit', 'cases_sha256', 'schedule_sha256', 'production_source_pins'):
            changed = copy.deepcopy(value)
            changed[key] = 'old-source-cut' if key != 'production_source_pins' else {'old-source': 'a' * 64}
            changed_path = self.directory / (key + '.json')
            ct.write(changed_path, changed)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'old source'):
                run.validate_transfer_layout(changed_path)
        with self.assertRaisesRegex(ValueError, 'renewed'):
            run.transfer_start(self.directory / 'artifacts-transfer-pilot-fixture', path, False)
        self.assertFalse((self.directory / 'artifacts-transfer-pilot-fixture').exists())

    def test_real_lexical_source_dispatch_and_mutations(self):
        texts = {p: (ct.ROOT / p).read_text() for p in run.PROOF_PATHS}
        self.assertFalse(run.review_transfer_source(texts)['direct_backend_call_counter_available'])
        for index, token in ((0, 'Option<ServeExpertTransferPolicy>'), (0, 'ServeExpertTransferPolicy::Synchronous => Self::Synchronous'),
            (1, 'args.expert_transfer_policy.map'), (4, 'serving.model_precision = Some(ModelPrecision::Bf16)'),
            (6, 'self.forward_executor.set_expert_transfer_policy'),
            (3, 'serving.expert_read_pipeline.is_none'), (5, 'runner.set_expert_transfer_policy'),
            (8, 'this.grouped_metadata_upload'), (9, 'self.grouped_upload = Some'),
            (10, 'upload_validated_numeric_fp8_group_sync'), (11, 'stream.synchronize')):
            changed = dict(texts)
            changed[run.PROOF_PATHS[index]] = changed[run.PROOF_PATHS[index]].replace(token, 'MUTATED')
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, 'dispatch binding'):
                run.review_transfer_source(changed)

    def test_adapter_restores_after_nested_cancellation(self):
        value = run.transfer_plan()
        path = self.directory / 'layout.json'
        ct.write(path, value)
        before = (run.historical.PROTOCOL_PATH, run.historical.trial_order, run.maintained.binding, ct.write)
        with mock.patch.object(run, 'validate_transfer_layout', return_value=value):
            with self.assertRaises(KeyboardInterrupt):
                with run.transfer_adapter(path), run.transfer_adapter(path):
                    self.assertEqual(run.historical.trial_order(), value['trial_order'])
                    self.assertEqual(run.historical.PROTOCOL_PATH, path)
                    self.assertEqual(run.planned.ENTRY, run.ENTRY)
                    raise KeyboardInterrupt()
        self.assertEqual(before, (run.historical.PROTOCOL_PATH, run.historical.trial_order, run.maintained.binding, ct.write))

    def test_authenticated_worker_launch_forwards_layout_to_existing_entry(self):
        payload = dict(directory=str(self.directory), contract_sha256='fixture', adapter_pins={},
            execution_path=str(self.directory / 'layout.json'), spec=dict(expert_transfer_case={}))
        with mock.patch.object(run.maintained.pilot.supervisor, 'run_process', return_value=http_fixtures.drained()) as supervise:
            result = run.maintained.pilot.supervise(payload, 530)
        self.assertTrue(ct.drained(result))
        self.assertEqual(supervise.call_args.args[0][-2:], ['--layout', payload['execution_path']])


class HealthTests(shared.CPUOnly):
    def test_both_policies_both_schedules_startup_and_positive_dispatch(self):
        for read_schedule in ('blocking', 'lookahead'):
            for policy in analysis.POLICIES:
                value = arm('fixture', read_schedule, policy)
                self.assertTrue(value['read_ledger']['can_continue'], value['read_ledger'])
                self.assertEqual(value['read_ledger']['host_idle_charged_bytes'], 10 << 20)
                self.assertGreater(value['read_ledger']['cache_uploads'], 0)
                self.assertIsNone(value['read_ledger']['grouped_backend_call_count'])
                self.assertFalse(value['read_ledger']['quiescence_certificate'])
                for counter in ('expert_uploads', 'h2d_bytes'):
                    changed = copy.deepcopy(value['lifecycle'])
                    changed['requests']['observations'][0]['logical_counters']['deltas'][counter] = 0
                    self.assertFalse(run.transfer_ledger(changed, read_schedule + '-' + policy)['can_continue'])
                wrong = startup(policy)
                wrong['expert_transfer_policy'] = 'Retained'
                with self.assertRaisesRegex(ValueError, 'policy mismatch'):
                    trace.transfer_startup([json.dumps(dict(fields=wrong))], policy, 'a' * 64)
                with self.assertRaisesRegex(ValueError, 'startup'):
                    trace.transfer_startup([], policy, 'a' * 64)

    def test_missing_unknown_overflow_pending_and_baseline_health_fail_closed(self):
        for read_schedule in ('blocking', 'lookahead'):
            for policy in analysis.POLICIES:
                for field, bad in (('expert_cache_counters_overflowed', True), ('quarantined', True),
                    ('counter_reset_detected', True), ('expert_read_counter_reset_detected', True),
                    ('expert_pending_upload_bytes', 1), ('expert_unknown_quarantine_bytes', 1),
                    ('facade_stream_sync_failures', 1), ('expert_owner_peak_bytes', 321 << 20)):
                    changed = invocation(read_schedule=read_schedule)
                    changed[field] = bad
                    with self.subTest(read_schedule=read_schedule, policy=policy, field=field):
                        self.assertFalse(window(changed, read_schedule)['valid'])
                for field in COUNTERS + ('expert_cache_counters_overflowed', 'expert_pending_upload_bytes',
                    'expert_unknown_quarantine_bytes', 'expert_read_pipeline_delta', 'expert_resident_entries'):
                    changed = invocation(read_schedule=read_schedule)
                    del changed[field]
                    with self.subTest(policy=policy, missing=field):
                        self.assertFalse(window(changed, read_schedule)['valid'])
                self.assertFalse(trace.transfer_counter_window([], workers.options(read_schedule))['valid'])

    def test_optional_zero_upload_fields_but_nonzero_unknown_rejected(self):
        for name in ('upload_reserved_packed_bytes', 'pinned_upload_bytes', 'upload_ticket_metadata_bytes',
            'upload_policy_headroom_bytes', 'unknown_upload_bytes'):
            value = invocation()
            read = json.loads(value['expert_read_pipeline'])
            read['current_charge'][name] = 0
            value['expert_read_pipeline'] = json.dumps(read)
            self.assertTrue(window(value)['valid'])
            for bad in (1, None, False):
                read['current_charge'][name] = bad
                value['expert_read_pipeline'] = json.dumps(read)
                self.assertFalse(window(value)['valid'])
        read = workers.observation(workers.options())
        read['current_charge']['host_charged_bytes'] = 0
        with self.assertRaises(ValueError):
            quality.read_observation_v3(read, workers.options())

    def test_counter_reset_and_shutdown_unknown_rejected(self):
        fields = invocation()
        lines = [json.dumps(dict(fields=fields)), json.dumps(dict(fields=fields))]
        self.assertFalse(trace.transfer_counter_window(lines, workers.options())['valid'])
        for policy in analysis.POLICIES:
            value = arm('fixture', policy=policy)
            final = value['lifecycle']['closure']['attempts'][0]['summary']['physical']['expert_read_pipeline']['value']
            final['worker_unknown'] = True
            self.assertFalse(run.transfer_ledger(value['lifecycle'], 'blocking-' + policy)['can_continue'])


class MetricTests(shared.CPUOnly):
    def test_phase_wall_includes_gaps_not_sum_e2e_or_done(self):
        value = analysis.transfer_arm_metrics(arm('fixture'), schedule())
        self.assertTrue(value['qualified_arm'])
        self.assertEqual(value['phase_wall_seconds'], 42.25)
        self.assertEqual(value['serialized_service_denominator_seconds'], 10.)
        self.assertEqual(value['serialized_service_normalized_qa8_tokens_per_second'], 4.)
        self.assertAlmostEqual(value['phase_qualified_tokens_per_second'], 40 / 42.25)
        self.assertAlmostEqual(value['phase_requests_per_second'], 5 / 42.25)
        request = value['requests'][-1]
        self.assertEqual(request['native_ttft_seconds'], .5)
        self.assertEqual(request['native_e2e_seconds'], 2.)
        self.assertEqual(request['dispatch_ttft_seconds'], .75)
        self.assertEqual(request['dispatch_e2e_seconds'], 2.25)
        self.assertEqual(request['admission_e2e_seconds'], 2.75)
        self.assertEqual(request['send_dispatch_delay_seconds'], .25)
        self.assertIsNone(request['route_ids'])

    def test_output_usage_finish_ids_and_timestamps_not_sse_chunk_count(self):
        planned = schedule()[-1]
        original = arm('fixture')['lifecycle']['requests']['observations'][-1]
        original['actual']['content_events'] = 1
        self.assertTrue(analysis.request_metrics(planned, original)['reference_qualified'])
        for key, bad in (('finish_reason', 'stop'), ('text', 'wrong'), ('actual_output_tokens', 1),
            ('usage', dict(completion_tokens=7)), ('client_dispatch_at', None), ('ended_at', None),
            ('request_e2e_seconds', 1)):
            value = copy.deepcopy(original)
            value['actual'][key] = bad
            with self.subTest(key=key):
                self.assertFalse(analysis.request_metrics(planned, value)['reference_qualified'])
        original['trace']['actual_generated_ids'] = [1] * 8
        self.assertFalse(analysis.request_metrics(planned, original)['reference_qualified'])

    def test_failure_censoring_unissued_and_unknown_partial_tokens_stay_visible(self):
        value = arm('fixture')
        observations = value['lifecycle']['requests']['observations']
        observations[-2].update(status='censored', metric_eligible=False)
        observations[-2]['actual'].update(status='timeout', usage=None, actual_output_tokens=None)
        observations[-1].update(status='failed', metric_eligible=False)
        observations[-1]['actual'].update(status='http_error', usage=None, actual_output_tokens=None)
        result = analysis.transfer_arm_metrics(value, schedule())
        self.assertEqual(result['counts'], dict(success=3, failed=1, censored=1, unissued=0))
        self.assertEqual(result['success_fraction'], 3 / 5)
        self.assertEqual(result['unknown_partial_token_slots'], 2)
        self.assertEqual(result['phase_wall_seconds'], 42.25)
        self.assertAlmostEqual(result['phase_requests_per_second'], 3 / 42.25)
        self.assertIsNone(result['serialized_service_normalized_qa8_tokens_per_second'])
        observations[-1]['actual'].pop('ended_at')
        result = analysis.transfer_arm_metrics(value, schedule())
        self.assertIsNone(result['phase_wall_seconds'])
        self.assertIsNotNone(result['observed_span_lower_bound_seconds'])
        result = analysis.transfer_arm_metrics({}, schedule())
        self.assertEqual(result['counts']['unissued'], 5)
        self.assertEqual(result['qualified_fraction'], 0)
        self.assertIsNone(result['phase_wall_seconds'])

    def report(self):
        value = fixture_layout()
        rows = []
        for block in value['order']:
            for mode in block['modes']:
                policy = mode[len(block['read_schedule']) + 1:]
                rows.append(arm(block['block'] + '-' + mode, block['read_schedule'], policy))
        return dict(study_layout=value, order=value['order'], lifecycles=rows)

    def test_pair_unit_no_request_bootstrap_or_pooled_schedules(self):
        result = analysis.summarize(self.report())
        self.assertEqual(result['denominators']['scheduled'], 96)
        self.assertEqual(result['scheduled_measured_qa'], 40)
        self.assertEqual(len(result['blocks']), 4)
        for name in ('blocking', 'lookahead'):
            summary = result['by_read_schedule'][name]
            self.assertEqual(summary['qualified_pairs'], 2)
            self.assertEqual(set(summary['median_paired_ratios'].values()), {1.})
            self.assertIsNone(summary['interval'])
            self.assertFalse(summary['significance_claim'])

    def test_incomplete_pair_retained_and_crosspolicy_work_mismatch_null(self):
        value = self.report()
        value['lifecycles'][0]['assessment']['can_continue'] = False
        result = analysis.summarize(value)
        self.assertIsNone(result['by_read_schedule']['blocking']['median_paired_ratios'])
        self.assertEqual(len(result['blocks']), 4)
        self.assertTrue(result['blocks'][0]['failure_pair_retained'])
        value = self.report()
        value['lifecycles'][0]['lifecycle']['requests']['observations'][0]['logical_counters']['rows'][0]['fields']['h2d_bytes'] += 1
        result = analysis.summarize(value)
        self.assertFalse(result['blocks'][0]['crosspolicy_equal_work'])
        self.assertIsNone(result['blocks'][0]['grouped_over_synchronous'])
        value['lifecycles'].pop()
        with self.assertRaisesRegex(ValueError, 'all eight'):
            analysis.summarize(value)

    def test_maintained_finalizer_retains_all_unissued_slots(self):
        directory = self.directory / 'artifacts-transfer-pilot-fixture'
        directory.mkdir()
        value = fixture_layout()
        path = self.directory / 'layout.json'
        ct.write(path, value)
        registered = dict(study_layout=value, schedule=value['schedule'], order=value['order'],
            trial_order=value['trial_order'], scheduled_blocks=4, started=dict(fixture=True),
            pilot_deadline=10000, binding_sha256='a' * 64, execution_sha256='b' * 64,
            binding_path=str(path), selected_uuid=http_fixtures.UUIDS[0], eight_uuids=http_fixtures.UUIDS)
        run.maintained.stop(directory, 'fixture STOP; no retries')
        with mock.patch.object(run, 'validate_transfer_layout', return_value=value), \
                mock.patch.object(run, 'transfer_registration', return_value=registered):
            with run.transfer_adapter(path):
                result = run.maintained.finalize(directory)
        final = ct.load(directory / 'paired-final.json')
        self.assertEqual(final['schema'], 'ferrule.storage-demand.http-transfer-actual.v2')
        self.assertEqual(final['scheduled_starts'], 8)
        self.assertEqual(final['scheduled_measured_qa'], 40)
        self.assertEqual(result['analysis']['denominators']['unissued'], 96)
        self.assertEqual(len(result['analysis']['blocks']), 4)


class MaintainedRequestTests(http_fixtures.CPUOnly):
    def test_real_serial_requests_strict_policy_metrics_and_startup(self):
        for policy in analysis.POLICIES:
            directory = self.directory / policy
            directory.mkdir()
            from test_http_transfer_controls import initial_event, initial_producer_binding
            server_identity = dict(pid=12345, birth_ticks=8, ppid=12344)
            (directory / 'server.log').write_text(json.dumps(dict(fields=startup(policy))) + '\n' +
                json.dumps(initial_event(workers.options(), policy, server_identity['pid'])) + '\n' +
                json.dumps(dict(fields=dict(message='Ferrule OpenAI server listening'))) + '\n')
            ready_log_end = (directory / 'server.log').stat().st_size
            order = []
            delegate = http_fixtures.MockClient(directory, order)
            def request(config, identity, cutoff=None):
                index, dispatch = len(order), time.monotonic()
                result = delegate.request(config, identity, cutoff)
                result['client_dispatch_at'] = dispatch
                # MockClient stamps sent_at before dispatch to simplify old fixtures.
                result['sent_at'] = dispatch
                result['ttft_seconds'] = result['content_event_times'][0] - dispatch
                result['request_e2e_seconds'] = result['ended_at'] - dispatch
                with (directory / 'server.log').open('a') as log:
                    log.write(json.dumps(dict(fields=invocation(index))) + '\n')
                    log.write(json.dumps(dict(fields=dict(event='request_http_admitted',
                        request_id=100 + index, http_admission_us=1))) + '\n')
                # Replace the old fixture invocation rather than relaxing strict health.
                lines = (directory / 'server.log').read_text().splitlines()
                lines = [line for line in lines if not (json.loads(line)['fields'].get('event')
                    == 'mixed_decoder_invocation' and 'expert_cache_counters_overflowed' not in json.loads(line)['fields'])]
                (directory / 'server.log').write_text('\n'.join(lines) + '\n')
                return result
            client = mock.Mock(request=request, close=lambda: None)
            deadline = time.monotonic() + 2000
            import test_http_transfer_metrics as metrics_fixtures
            context = metrics_fixtures.reference_context(schedule())
            spec = dict(url='http://127.0.0.1:18369/v1/completions', model='qwen35-storage-demand-paper',
                frozen_cli_sha256='a' * 64, upload_producer_contract=quality.TRANSFER_UPLOAD_PRODUCER,
                initial_read_producer_binding=initial_producer_binding(),
                http_expectation_contract=expectations.BOUND_HTTP_REFERENCE, http_reference_context=context,
                expert_transfer_case=dict(read_schedule='blocking',
                    transfer_policy=policy, read_options=workers.options()))
            # The fixture client appends corrected logs without truncating prior windows.
            with mock.patch.object(ex.http, 'Client', return_value=client):
                result = ex.serial_requests(directory, schedule(), spec, run.transfer_plan()['limits'],
                    deadline, deadline, deadline, lambda: None, self.sampling(directory), lambda: {},
                    server_identity=server_identity, ready_log_end=ready_log_end)
            self.assertTrue(result['can_continue'], result['stop_reason'])
            self.assertEqual(len(order), 12)
            self.assertEqual(result['expert_transfers']['startup']['requested'], policy)
            self.assertTrue(all(r['metric_eligible'] for r in result['observations']))
            self.assertTrue(all(analysis.request_metrics(r, r)['reference_qualified'] for r in result['observations']))

    def test_lost_worker_terminal_keeps_dispatch_and_unknown_tokens(self):
        planned = schedule()[7]
        folder = self.directory / planned['name']
        folder.mkdir()
        ct.write(folder / 'issued.json', dict(planned, issued=True, admission=dict(monotonic=1)))
        ct.write(folder / 'http-terminal.json', dict(status='timeout', client_dispatch_at=2,
            ended_at=4, usage=None, actual_output_tokens=None))
        rows = ex.recover_observations(self.directory, schedule())
        self.assertEqual(rows[0]['status'], 'censored')
        metric = analysis.request_metrics(planned, rows[0])
        self.assertTrue(metric['unknown_partial_tokens'])
        self.assertIsNone(metric['completion_tokens'])
        self.assertEqual(metric['dispatch_e2e_seconds'], 2)
        counts = ex.recover_requests(self.directory, schedule())
        self.assertEqual(counts['issued'], 1)
        self.assertEqual(counts['censored'], 1)
        self.assertEqual(counts['unissued'], 11)


class CandidateTests(shared.CPUOnly):
    def test_legacy_boolean_or_dummy_candidate_rejected(self):
        layout = run.transfer_plan()
        layout['candidate'] = dict(binary=str(self.directory / 'fixture-never-executable'),
            binary_sha256='a' * 64, same_cut_cli_quality_accepted=True)
        with self.assertRaisesRegex(ValueError, 'candidate pending'):
            run.validate_transfer_candidate(layout)



class CoordinatorTests(shared.CPUOnly):
    def test_existing_next_arm_uses_eight_locks_and_pair_mismatch_stops_admission(self):
        directory = self.directory / 'artifacts-transfer-pilot-fixture'
        directory.mkdir()
        layout = fixture_layout()
        path = self.directory / 'layout.json'
        candidate = dict(candidate=dict(binary=str(self.directory / 'fixture-cli'), binary_sha256='a' * 64,
            runtime_pin_manifest=str(self.directory / 'fixture-runtime.json')), uuid_inventory=http_fixtures.UUIDS)
        ct.write(path, layout)
        registered = dict(study_layout=layout, trial_order=layout['trial_order'], order=layout['order'],
            limits=layout['limits'], schedule=layout['schedule'], pilot_deadline=10000,
            selected_uuid=http_fixtures.UUIDS[0], eight_uuids=http_fixtures.UUIDS,
            binding_path=str(path), execution_path=str(path), binding_sha256=ct.sha(path), execution_sha256=ct.sha(path))
        ct.write(directory / 'registration.json', dict(fixture=True))
        prior = arm(layout['trial_order'][0])
        prior_folder = directory / prior['trial']
        prior_folder.mkdir()
        ct.write(prior_folder / 'attempt.json', dict(fixture=True))
        ct.write(prior_folder / 'assessment.json', prior)
        ct.write(prior_folder / 'locks-held.json', dict(fixture=True))
        terminal = run.maintained.coordinator_terminal_binding(prior_folder)
        ct.write(prior_folder / 'locks-released.json', dict(terminal, phase='release', context_exited=True, all_eight_local=True))
        ct.write(prior_folder / 'locks-reprobed.json', dict(terminal, phase='reprobe', all_eight_nonblocking_reacquired=True, context_exited=True, local_only=True))
        current = arm(layout['trial_order'][1], policy='grouped-synchronous')
        current['lifecycle']['requests']['observations'][0]['actual']['text'] = 'cross-policy mismatch'
        payloads = []
        def supervise(payload, seconds):
            payloads.append(copy.deepcopy(payload))
            self.assertEqual(seconds, 530)
            return http_fixtures.drained()
        pilot = run.maintained.pilot
        with mock.patch.object(run, 'validate_transfer_layout', return_value=layout), \
                mock.patch.object(run, 'validate_transfer_candidate', return_value=candidate), \
                mock.patch.object(run, 'transfer_registration', return_value=registered), \
                mock.patch.object(run, 'transfer_ledger', return_value=dict(can_continue=True)), \
                mock.patch.object(ct, 'exclusive_owner_lock', side_effect=lambda uuid: nullcontext(dict(fixture_uuid=uuid))), \
                mock.patch.object(ct, 'validate_owner_lock'), \
                mock.patch.object(pilot.telemetry, 'authorize'), \
                mock.patch.object(pilot, 'idle', return_value=dict(status='verified', all_eight_idle=True)), \
                mock.patch.object(pilot, 'baseline', return_value={}), \
                mock.patch.object(pilot, 'remaining'), \
                mock.patch.object(pilot.idle_window, 'no_recorded_owner', return_value=True), \
                mock.patch.object(run.planned, 'supervise', side_effect=supervise), \
                mock.patch.object(ex, 'lifecycle_assessment', return_value=dict(can_continue=True, stop_reasons=[])), \
                mock.patch.object(ex, 'recover_observations', return_value=current['lifecycle']['requests']['observations']), \
                mock.patch.object(ex, 'recover_requests', return_value=current['summary']), \
                mock.patch.object(run.time, 'monotonic', return_value=100):
            with run.transfer_adapter(path):
                row = run.maintained.execute_next(directory)
        self.assertFalse(row['assessment']['can_continue'])
        self.assertFalse(row['paired_quality']['crosspolicy_exact_output'])
        self.assertTrue((directory / 'study-stop.json').is_file())
        self.assertEqual(len(payloads), 1)
        payload = payloads[0]
        self.assertEqual(len(payload['all_uuid_locks']), 8)
        self.assertEqual(payload['spec']['expert_transfer_case']['transfer_policy'], 'grouped-synchronous')
        self.assertEqual(payload['spec']['argv'][-2:], ['--expert-transfer-policy', 'grouped-synchronous'])
        self.assertEqual(payload['schedule'], schedule())
        self.assertEqual(payload['contract_sha256'], ct.digest({k: payload[k] for k in run.planned.CONTRACT_KEYS}))
        with self.assertRaisesRegex(ValueError, 'permanently stopped'):
            run.maintained.next_trial(directory, registered)


if __name__ == '__main__':
    unittest.main()
