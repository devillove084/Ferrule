"""CPU fixtures exercise maintained HTTP/serial/token paths, never GPU evidence."""
from contextlib import contextmanager
import copy
import io
import json
import os
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest import mock

import subset_adapter as a

sys.path.insert(0, str(a.p.HERE.parent / 'tests'))
import test_execution_v2 as fixtures


def source_lines(point):
    case = a.case_for_point(point)
    f = a.design()['pressure']['fixed']
    fields = dict(max_experts=case['requested_entries'], max_bytes=case['requested_cache_bytes'],
                  scratch_bytes=f['scratch_bytes'], workspace_bytes=f['workspace_bytes'], kv_bytes=f['kv_bytes'],
                  device_bytes=f['device_bytes'], allocator_margin_bytes=f['allocator_margin_bytes'],
                  max_positions=128, max_sequences=2, max_batch_tokens=8)
    def rust(name, values):
        return name + ' { ' + ', '.join(k + ': ' + str(v).lower() for k, v in values.items()) + ' }'
    capacity = rust('Qwen35MoeCapacityLimits', fields)
    scheduler = rust('ResidentSchedulerConfig', case['expected_scheduler'])
    report = ('profile=cuda-hybrid-numeric-fp8-model-bf16-mixed-f32-ssm; '
              'storage: NumericFp8E4m3fnBf16Scales; context=128; '
              'max_layers: descriptor default -> 40 (fixture); '
              + capacity + ' -> ' + capacity + '; scheduler requested=' + scheduler + ', effective=' + scheduler)
    return [json.dumps(dict(fields=dict(message='resident model build entering owner-local admission', report=report))),
            'expert read host staging: schedule=Blocking host_cap_bytes=67108864 metadata_bytes=8388608 worker_stack_bytes=2097152; fixture']


def read_ledger():
    options = dict(a.case_for_point('middle')['read_options'], schedule='blocking')
    charge = dict(packed_bytes=0, artifact_bytes=0, payload_bytes=0, metadata_bytes=0,
                  requested_stack_bytes=0, host_charged_bytes=0, reserved_read_slots=0,
                  unknown_packed_bytes=0, unknown_stack_bytes=0, unknown_read_slots=0)
    stats = dict(unknown_reads=0, unknown_bytes=0, unobserved_terminal_reads=0,
                 unobserved_reader_intervals=0, join_failed_sequences=0,
                 submitted_reads=1, completed_reads=1, verified_reads=1, io_started_reads=1, proved_read_bytes=1,
                 peak_packed_bytes=1, peak_artifact_bytes=1, peak_payload_bytes=2,
                 peak_metadata_bytes=8 << 20, peak_requested_stack_bytes=2 << 20, peak_host_charged_bytes=(10 << 20) + 2)
    return dict(enabled=True, requested_mode='blocking', effective_mode='blocking', limits=options,
                status='closed', worker_read_observed=True, stats=stats, current_charge=charge)


class FourClient(fixtures.MockClient):
    def request(self, config, identity, cutoff=None):
        index = len(self.order)
        request_id = 100 if self.duplicate else 100 + index
        log = self.directory / 'server.log'
        for event in ('request_admitted', 'request_http_admitted'):
            with log.open('a') as stream:
                stream.write(json.dumps(dict(fields=dict(event=event, request_id=request_id, session_id=request_id))) + '\n')
        result = super().request(config, identity, cutoff)
        with log.open('a') as stream:
            stream.write(json.dumps(dict(fields=dict(event='request_worker_terminal', request_id=request_id,
                                                     session_id=request_id, status='finished'))) + '\n')
        return result


class SubsetTests(fixtures.CPUOnly):
    def execute(self, failure=None, duplicate=False, check=lambda: None):
        (self.directory / 'server.log').touch()
        order = []
        self.stack.enter_context(mock.patch.object(a.ex.http, 'Client',
            side_effect=lambda: FourClient(self.directory, order, failure, duplicate)))
        deadline = time.monotonic() + 60
        result = a.serial_requests(self.directory, a.schedule(),
            dict(url='http://127.0.0.1:18369/v1/completions', model='qwen35-storage-demand-paper'),
            a.p.LIMITS, deadline, deadline, deadline, check, self.sampling(self.directory), lambda: {})
        return result, order

    def join(self, requests, lines=None):
        lines = a.ex.log_lines(self.directory / 'server.log') if lines is None else lines
        return a.whole_log_join(a.trace.parse_token_trace(lines), requests, lines)

    def test_real_maintained_http_serial_all_four_only_and_26_tokens(self):
        result, order = self.execute()
        self.assertEqual(order, list(a.NAMES))
        self.assertTrue(result['can_continue'], result['stop_reason'])
        self.assertEqual(result['summary']['successful'], 4)
        self.assertEqual(result['summary']['measured_qa']['scheduled'], 0)
        joined = self.join(result['observations'])
        self.assertEqual(sum(len(r['actual_generated_ids']) for r in joined), 26)
        self.assertTrue(all(r['exact_token_pass'] is True for r in joined[1:]))
        self.assertIsNone(joined[0]['exact_token_pass'])
        self.assertEqual([r['actual_worker_request_id'] for r in joined], [100, 101, 102, 103])
        self.assertEqual(a.ct.load(self.directory / 'requests-final.json'), result)

    def test_shared_twelve_validator_and_function_globals_unchanged(self):
        before = a.ex.serial_requests.__globals__['validate_schedule']
        result, _ = self.execute()
        self.assertTrue(result['can_continue'])
        self.assertIs(a.ex.serial_requests.__globals__['validate_schedule'], before)
        with self.assertRaisesRegex(ValueError, '12'):
            before(a.schedule())
        self.assertEqual(before(fixtures.schedule())['count'], 12)
        self.assertIn('len(observations) == 12', a.inspect.getsource(a.ex.serial_requests))
        self.assertIn("len(result['whole_log_token_join']) == 12", a.inspect.getsource(a.pilot.worker))

    def test_partial_empty_extra_reordered_schedules_rejected_before_dispatch(self):
        rows = a.schedule()
        bad = [[], rows[:3], rows + [rows[-1]], list(reversed(rows)), fixtures.schedule()]
        changed = copy.deepcopy(rows)
        changed[1]['reference']['reference_ids'][-1] += 1
        bad.append(changed)
        for value in bad:
            with self.subTest(count=len(value)), self.assertRaises(ValueError):
                a.validate_schedule(value)
        for ceiling in (12, 3, True, 4.0):
            with self.assertRaises(ValueError):
                a.validate_schedule(rows, ceiling)

    def test_whole_log_requires_exact_four_and_rejects_extra_unissued_admission(self):
        result, _ = self.execute()
        rows = result['observations']
        for observed in ([], rows[:3], rows + [rows[-1]], list(reversed(rows))):
            with self.assertRaises(ValueError):
                self.join(observed)
        lines = a.ex.log_lines(self.directory / 'server.log')
        extra = json.dumps(dict(fields=dict(event='request_http_admitted', request_id=104, session_id=104)))
        with self.assertRaisesRegex(ValueError, 'whole-log event'):
            self.join(rows, lines + [extra])

    def test_whole_log_rejects_extra_tokens_wrong_ids_and_terminal_usage(self):
        result, _ = self.execute()
        rows = result['observations']
        lines = a.ex.log_lines(self.directory / 'server.log')
        extra = json.dumps(dict(fields=dict(event='request_worker_token_enqueued', request_id=104, session_id=104,
            emission_ordinal=1, actual_token_id=9, delivery_boundary='worker_channel_enqueue')))
        with self.assertRaisesRegex(ValueError, 'whole-log tokens'):
            self.join(rows, lines + [extra])
        for key, replacement in (('actual_token_id', 987654321), ('usage_available', False)):
            changed = []
            for line in lines:
                event = json.loads(line)
                fields = event['fields']
                if key in fields and fields.get('request_id') == 101:
                    fields[key] = replacement
                changed.append(json.dumps(event))
            with self.assertRaises(ValueError):
                self.join(rows, changed)

    def test_failure_denominators_no_retry_and_duplicate_id_fail_closed(self):
        result, order = self.execute(failure='http')
        self.assertEqual(order, ['hello2', 'canary-qa'])
        self.assertEqual(result['summary']['counts'], dict(success=1, failed=1, censored=0, unissued=2))
        self.assertFalse(result['can_continue'])
        with self.assertRaises(ValueError):
            self.join(result['observations'])

    def test_duplicate_id_stops_at_second_slot(self):
        result, order = self.execute(duplicate=True)
        self.assertEqual(order, ['hello2', 'canary-qa'])
        self.assertFalse(result['can_continue'])
        self.assertTrue(result['observations'][-1]['duplicate_lifecycle_request_id'])

    def test_custody_loss_before_dispatch_keeps_all_four_unissued(self):
        def lost():
            raise ValueError('unknown custody')
        result, order = self.execute(check=lost)
        self.assertEqual(order, [])
        self.assertEqual(result['summary']['unissued'], 4)
        self.assertFalse(result['can_continue'])

    def test_issued_without_terminal_remains_censored(self):
        with mock.patch.object(a.ex, 'one_request', side_effect=TimeoutError('lost terminal')):
            result, order = self.execute()
        self.assertEqual(order, [])
        self.assertEqual(result['summary']['counts'], dict(success=0, failed=0, censored=1, unissued=3))
        self.assertTrue((self.directory / 'hello2/issued.json').exists())
        self.assertFalse((self.directory / 'canary-qa/issued.json').exists())

    def test_absolute_aggregate_deadline_does_not_slide(self):
        from runnerhelpers.evidence_v2 import request_budget_seconds
        self.assertEqual(request_budget_seconds(a.p.LIMITS, 100, 100, 300, 400, 500), 15)
        self.assertEqual(request_budget_seconds(a.p.LIMITS, 159, 100, 300, 400, 500), 1)
        self.assertEqual(request_budget_seconds(a.p.LIMITS, 160, 100, 300, 400, 500), 0)
        self.assertFalse(a.p.can_admit(419.999))
        self.assertTrue(a.p.can_admit(420))

    def test_three_real_source_option_shapes_and_drift_rejections(self):
        for point in a.POINT_IDS:
            lines = source_lines(point)
            result = a.source_options(point, lines)
            self.assertEqual(result['effective_metadata_options']['max_experts'], a.case_for_point(point)['requested_entries'])
            self.assertIsNone(result['physical_admission'])
            with self.assertRaises(ValueError):
                a.source_options(point, [])
            for before, after in (('mixed-f32-ssm', 'fp4'), ('context=128', 'context=256'),
                                  ('decode_cohort_target: 1', 'decode_cohort_target: 4')):
                with self.assertRaises(ValueError):
                    a.source_options(point, [line.replace(before, after) for line in lines])
        with self.assertRaises(ValueError):
            a.source_options('small', source_lines('middle'))

    def test_private_idle_keeps_three_samples_and_absolute_ten_second_cap(self):
        calls = []
        def sample(deadline, snapshot):
            calls.append(deadline)
            snapshot.update(gpu='\n'.join(f'{index}, {uuid}, 0 MiB, 0 %' for index, uuid in enumerate(fixtures.UUIDS)),
                            apps='')
        now = time.monotonic()
        deadline = now + 10
        with mock.patch.object(a.pilot.idle_window, 'sample', side_effect=sample), \
                mock.patch.object(a.pilot.time, 'sleep'):
            result = a.idle(self.directory, 'fixture-idle', fixtures.UUIDS, deadline)
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls, [deadline] * 3)
        self.assertEqual(result['required_consecutive'], 3)
        self.assertLessEqual(result['deadline_seconds'], 10)
        self.assertEqual(a.pilot.idle_window.SECONDS, 15)
        self.assertEqual(result['schema'], a.pilot.idle_window.SCHEMA)
        with self.assertRaises(ValueError):
            a.idle(self.directory, 'expired', fixtures.UUIDS, now - 1)

    def test_private_idle_rejects_foreign_memory_and_fake_custody(self):
        def sample(deadline, snapshot):
            snapshot.update(gpu='\n'.join(f'{index}, {uuid}, 1 MiB, 0 %' for index, uuid in enumerate(fixtures.UUIDS)),
                            apps='')
        with mock.patch.object(a.pilot.idle_window, 'sample', side_effect=sample) as sampled:
            with self.assertRaises(ValueError):
                a.idle(self.directory, 'foreign', fixtures.UUIDS, time.monotonic() + 10)
            self.assertEqual(sampled.call_count, 1)
            with self.assertRaises(ValueError):
                a.idle(self.directory, 'fake-custody', fixtures.UUIDS, time.monotonic() + 10,
                       dict(returncode=0, custody=dict(echild=True)))
            self.assertEqual(sampled.call_count, 1)

    def test_static_header_arithmetic_is_never_actual_all_caps_lowering(self):
        for case in a.p.cases(a.design()):
            value = case['static_header_preflight']
            self.assertEqual(value['metadata_margin_bytes'], 684032)
            self.assertEqual(value['double_buffer_payload_cap_bytes'], 56623104)
            self.assertEqual(value['double_buffer_required_bytes'], 6292224)
            self.assertFalse(value['actual_cli_lowering_executed_for_point'])
            self.assertFalse(value['actual_header_reopened'])
        value = a.design()
        value['pressure']['fixed']['metadata_bytes'] = 7 << 20
        with self.assertRaises(ValueError):
            a.p.cases(value)

    def complete_fixture(self, result):
        ledger = read_ledger()
        rows = copy.deepcopy(result['observations'])
        for row in rows:
            for invocation in row['logical_counters']['rows']:
                invocation['fields'].update(expert_read_pipeline=json.dumps(ledger), expert_read_counter_reset_detected=False)
        cache = dict(resident_entries=8, charged_bytes=(8 << 20) + 2, peak_bytes=(8 << 20) + 3,
                     scratch_bytes=8 << 20, pending_upload_bytes=0, unknown_quarantine_bytes=0, quarantined=False)
        return dict(status='complete', requests=dict(observations=rows), closure=dict(final=dict(can_continue=True),
            attempts=[dict(summary=dict(physical=dict(expert_read_pipeline=dict(state='observed', value=ledger),
                                                     expert_cache=dict(state='observed', value=cache))))]))

    def test_each_point_needs_nonempty_source_reads_and_visible_join_not_numeric_pass(self):
        result, _ = self.execute()
        observed = self.complete_fixture(result)
        tokens = a.ex.log_lines(self.directory / 'server.log')
        for point in a.POINT_IDS:
            accepted = a.point_assessment(point, observed, source_lines(point) + tokens)
            self.assertTrue(accepted['can_continue'], accepted.get('stop_reason'))
            self.assertEqual(accepted['full_numerical_per_point'], 'unverified')
            self.assertIsNone(accepted['invocation_counter_totals']['expert_cache_misses'])
            self.assertIsNone(accepted['invocation_counter_totals']['working_set_unique_expert_ids'])
            self.assertFalse(a.point_assessment(point, {}, [])['can_continue'])
            self.assertFalse(a.point_assessment(point, observed, tokens)['can_continue'])
        zero = copy.deepcopy(observed)
        for row in zero['requests']['observations']:
            for invocation in row['logical_counters']['rows']:
                invocation['fields']['checkpoint_source_checks'] = 0
        self.assertFalse(a.point_assessment('small', zero, source_lines('small') + tokens)['can_continue'])
        reset = copy.deepcopy(observed)
        reset['requests']['observations'][0]['logical_counters']['rows'][0]['fields']['expert_read_counter_reset_detected'] = True
        self.assertFalse(a.point_assessment('small', reset, source_lines('small') + tokens)['can_continue'])

    def test_private_worker_reuses_authentication_and_restores_maps_on_exception(self):
        maps_before, capture_before = a.ex.library_maps, a.ex.capture_root
        payload = dict(trial='small-blocking')
        with self.assertRaisesRegex(RuntimeError, 'fixture stop'):
            with a.private_worker(payload) as worker:
                self.assertIs(worker.__globals__['launch_guard'].authenticate, a.pilot.launch_guard.authenticate)
                self.assertIs(worker.__globals__['worker_contract'], a.worker_contract)
                self.assertIs(worker.__globals__['ex'].join_token_trace_http, a.whole_log_join)
                self.assertIs(worker.__globals__['ex'].one_request, a.ex.one_request)
                self.assertEqual(worker.__globals__['HERE'], a.p.HERE)
                raise RuntimeError('fixture stop')
        self.assertIs(a.ex.library_maps, maps_before)
        self.assertIs(a.ex.capture_root, capture_before)

    def test_private_worker_authentication_failure_precedes_candidate_or_launch(self):
        payload = dict(trial='small-blocking')
        with a.private_worker(payload) as worker, mock.patch('sys.stdin', io.StringIO(json.dumps(payload))), \
                mock.patch.object(a.ct, 'require_cpu_python'), \
                mock.patch.object(a.pilot.launch_guard, 'authenticate', side_effect=ValueError('unauthenticated')):
            with self.assertRaisesRegex(ValueError, 'unauthenticated'):
                worker('invalid-launch-hash')

    def test_exact_current_cases_scoped_to_two_knobs_and_middle_numerics_only(self):
        candidate = dict(candidate=a.p.load(a.BINDING_PATH)['candidate'], uuid_inventory=fixtures.UUIDS)
        baseline = a.spec_for(candidate, fixtures.UUIDS[0], 'middle')
        allowed = {baseline['argv'].index(flag) + 1 for flag in a.design()['pressure']['vary_only']}
        for point in a.POINT_IDS:
            spec = a.spec_for(candidate, fixtures.UUIDS[0], point)
            self.assertEqual(spec['environment'], baseline['environment'])
            self.assertTrue({i for i, pair in enumerate(zip(spec['argv'], baseline['argv'])) if pair[0] != pair[1]} <= allowed)
            self.assertEqual(a.case_for_point(point)['full_numerical_per_point'], 'unverified')
        with self.assertRaises(ValueError):
            a.spec_for(candidate, fixtures.UUIDS[1], 'small')

    def test_next_point_no_retry_partial_replay_or_fourth_lifecycle(self):
        registered = dict(trial_order=list(a.TRIALS), limits=a.p.LIMITS)
        self.assertEqual(a.next_point(self.directory, registered), 'small-blocking')
        for trial in a.TRIALS:
            folder = self.directory / trial
            folder.mkdir()
            with self.assertRaises(ValueError):
                a.next_point(self.directory, registered)
            a.ct.write(folder / 'attempt.json', dict(fixture_only=True))
            a.ct.write(folder / 'assessment.json', dict(trial=trial, assessment=dict(can_continue=True)))
        with self.assertRaises(ValueError):
            a.next_point(self.directory, registered)
        a.ct.write(self.directory / 'study-stop.json', dict(fixture_only=True))
        with self.assertRaises(ValueError):
            a.next_point(self.directory, registered)

    def test_false_auth_schema_wrong_ceiling_and_empty_candidate_cannot_pass(self):
        with self.assertRaises(ValueError):
            a.worker_contract({})
        payload = {key: None for key in a.CONTRACT_KEYS}
        payload.update(schema=a.SCHEMA, trial='small-blocking', trial_order=list(a.TRIALS), mode='blocking',
                       request_ceiling=12, schedule=a.schedule())
        payload['contract_sha256'] = a.ct.digest({key: payload[key] for key in a.CONTRACT_KEYS})
        with self.assertRaisesRegex(ValueError, 'ceiling'):
            a.worker_contract(payload)
        self.assertFalse(a.ct.drained(dict(returncode=0, custody=dict(echild=True))))

    def test_seal_rejects_changed_scoped_case_or_missing_candidate(self):
        value = a.p.load(a.BINDING_PATH)
        value['budget_variant']['request_ceiling'] = 12
        with mock.patch.object(a.p, 'check_bindings', return_value=value), self.assertRaisesRegex(ValueError, 'seal stale'):
            a.validate_execution()
        with self.assertRaises(ValueError):
            a.validate_candidate(self.directory / 'unbound.json')

    def test_specialization_rejects_unpinned_shared_source(self):
        value = a.p.load(a.BINDING_PATH)
        value['files']['papers/arxiv/experiments/storage_demand/runnerhelpers/execution_v2.py'] = '0' * 64
        with mock.patch.object(a.p, 'load', return_value=value), self.assertRaisesRegex(ValueError, 'source changed'):
            a.serial_requests()

    def test_full_worker_contract_keeps_all_eight_locks_and_exact_scope(self):
        candidate = dict(candidate=a.p.load(a.BINDING_PATH)['candidate'], uuid_inventory=fixtures.UUIDS)
        locks = [dict(fixture_uuid=uuid) for uuid in fixtures.UUIDS]
        pins = {str(a.BINDING_PATH): a.ct.sha(a.BINDING_PATH)}
        payload = {key: None for key in a.CONTRACT_KEYS}
        payload.update(schema=a.SCHEMA, trial='small-blocking', trial_order=list(a.TRIALS), mode='blocking',
            request_ceiling=4, schedule=a.schedule(), limits=a.p.LIMITS, granted=True, owner_note=a.NOTE,
            uuid=fixtures.UUIDS[0], eight_uuids=fixtures.UUIDS, all_uuid_locks=locks, owner_lock=locks[0],
            coordinator=dict(pid=12345), adapter_pins=pins, spec=a.spec_for(candidate, fixtures.UUIDS[0], 'small'))
        for key in ('candidate', 'execution'):
            payload[key + '_path'] = str(a.BINDING_PATH)
            payload[key + '_sha256'] = a.ct.sha(a.BINDING_PATH)
        def sealed(value):
            value = copy.deepcopy(value)
            value['contract_sha256'] = a.ct.digest({key: value[key] for key in a.CONTRACT_KEYS})
            return value
        with mock.patch.object(a, 'validate_execution', return_value=dict(artifact_pins=pins)), \
                mock.patch.object(a, 'validate_candidate', return_value=candidate), \
                mock.patch.object(a.pilot.telemetry, 'authorize') as authorize, \
                mock.patch.object(a.ct, 'validate_owner_lock') as check:
            self.assertEqual(a.worker_contract(sealed(payload)), candidate)
            self.assertEqual(check.call_count, 8)
            for uuid, lock in zip(fixtures.UUIDS, locks):
                check.assert_any_call(lock, uuid, 12345)
            authorize.assert_called_once_with(True, fixtures.UUIDS[0], a.NOTE, 'device_exclusive_local_custody')
            for key, changed in (('granted', False), ('all_uuid_locks', locks[:7]),
                                 ('owner_lock', locks[1]), ('request_ceiling', 12),
                                 ('limits', dict(a.p.LIMITS, outer_lifecycle_seconds=241))):
                invalid = dict(payload, **{key: changed})
                with self.subTest(key=key), self.assertRaises(ValueError):
                    a.worker_contract(sealed(invalid))
            invalid = copy.deepcopy(payload)
            invalid['spec']['environment']['FERRULE_PROFILE_NVTX'] = '1'
            with self.assertRaises(ValueError):
                a.worker_contract(sealed(invalid))

    def test_coordinator_assessment_does_not_replace_authenticated_custody_with_point_pass(self):
        result, _ = self.execute()
        observed = self.complete_fixture(result)
        lines = source_lines('small') + a.ex.log_lines(self.directory / 'server.log')
        idle = dict(status='verified', all_eight_idle=True)
        self.assertTrue(a.lifecycle_assessment('small', observed, lines, fixtures.drained(), idle)['can_continue'])
        for receipt, post in ((dict(returncode=0, custody=dict(echild=True)), idle),
                              (fixtures.drained(), dict(status='blocked', all_eight_idle=False))):
            assessed = a.lifecycle_assessment('small', observed, lines, receipt, post)
            self.assertFalse(assessed['can_continue'])
            self.assertTrue(assessed['pressure']['can_continue'])

    def test_cli_rejects_general_loop_and_incomplete_authorized_commands(self):
        with mock.patch.object(a.ct, 'require_cpu_python'), mock.patch.object(a.signal, 'signal'), \
                mock.patch.object(a.os, 'umask'):
            for mode in ('run', 'execute-next'):
                with self.subTest(mode=mode), mock.patch('sys.stderr', io.StringIO()), self.assertRaises(SystemExit):
                    a.main([mode])
            for mode in ('--_worker', 'start', 'next', 'finalize'):
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    a.main([mode])


class WorkerIntegrationTests(fixtures.CPUOnly):
    def prepare_worker(self):
        summary = fixtures.known_summary()
        summary['physical']['expert_read_pipeline'] = dict(state='observed', value=read_ledger())
        summary['physical']['expert_cache'] = dict(state='observed', value=dict(
            resident_entries=8, resident_bytes=2, pending_upload_bytes=0, scratch_bytes=8 << 20,
            charged_bytes=(8 << 20) + 2, peak_bytes=(8 << 20) + 3,
            unknown_quarantine_bytes=0, quarantined=False))
        args, _, _, _ = fixtures.MockPilotEndToEndTests.fixture(self, summary=summary)
        def idle_sample(deadline, snapshot):
            snapshot.update(gpu='\n'.join(f'{index}, {uuid}, 0 MiB, 0 %' for index, uuid in enumerate(fixtures.UUIDS)),
                            apps='')
        self.stack.enter_context(mock.patch.object(a.pilot.idle_window, 'sample', side_effect=idle_sample))
        self.stack.enter_context(mock.patch.object(a.pilot.time, 'sleep'))
        candidate = dict(candidate=dict(binary='/fixture/immutable/ferrule', binary_sha256='0' * 64,
            runtime_pin_manifest=str(self.directory / 'runtime.json'), artifact_pins={}))
        args.candidate.write_text(json.dumps(candidate))
        self.stack.enter_context(mock.patch.object(a, 'worker_contract', return_value=candidate))
        read_text = Path.read_text
        def read(path, *values, **keywords):
            if str(path) == f'/proc/{fixtures.ROOT["pid"]}/environ':
                return ('CUDA_VISIBLE_DEVICES=' + fixtures.UUIDS[0] + '\0NVIDIA_VISIBLE_DEVICES='
                        + fixtures.UUIDS[0] + '\0FERRULE_PROFILE_NVTX=0\0')
            return read_text(path, *values, **keywords)
        self.stack.enter_context(mock.patch.object(Path, 'read_text', read))
        real_private = a.private_worker
        @contextmanager
        def local_scope(payload):
            with real_private(payload) as private:
                # The fixture owns only its temporary directory; no real owned output.
                private.__globals__['HERE'] = self.directory
                yield private
        self.stack.enter_context(mock.patch.object(a, 'private_worker', local_scope))
        mappings = self.stack.enter_context(mock.patch.object(a.mapped.maps, 'library_maps',
            return_value=dict(libraries={'fixture-required-role': 'fixture-hash'})))
        journal = self.directory / 'artifacts-fixture'
        journal.mkdir()
        return args, candidate, journal, mappings

    def invoke_point(self, point, args, candidate, journal, source_point=None):
        folder = journal / (point + '-blocking')
        folder.mkdir()
        order = []
        payload = dict(trial=point + '-blocking', directory=str(folder), limits=a.p.LIMITS,
            schedule=a.schedule(), spec=dict(argv=[candidate['candidate']['binary']], environment={},
                url='http://127.0.0.1:18369/v1/completions', model='qwen35-storage-demand-paper'),
            eight_uuids=fixtures.UUIDS, uuid=fixtures.UUIDS[0], baseline={}, owner_note='CPU fixture only',
            owner_lock={}, coordinator=dict(pid=os.getpid()), adapter_pins={},
            outer_deadline=time.monotonic() + 240, pilot_deadline=time.monotonic() + 720)
        for key, path in (('candidate', args.candidate), ('execution', args.execution_prepared)):
            payload[key + '_path'] = str(path)
            payload[key + '_sha256'] = a.ct.sha(path)
        original_launch = a.ex.launch_capture
        def launch(*values, **keywords):
            capture = original_launch(*values, **keywords)
            with (folder / 'server.log').open('a') as stream:
                stream.write('\n'.join(source_lines(source_point or point)) + '\n')
            return capture
        class PipelineClient(FourClient):
            def request(self, *values, **keywords):
                result = super().request(*values, **keywords)
                path = self.directory / 'server.log'
                updated = []
                for line in path.read_text().splitlines():
                    try:
                        event = json.loads(line)
                    except ValueError:
                        updated.append(line)
                        continue
                    fields = event.get('fields', {})
                    if fields.get('event') == 'mixed_decoder_invocation':
                        fields.update(expert_read_pipeline=json.dumps(read_ledger()), expert_read_counter_reset_detected=False)
                    updated.append(json.dumps(event))
                path.write_text('\n'.join(updated) + '\n')
                return result
        with mock.patch.object(a.ex, 'launch_capture', side_effect=launch), \
                mock.patch.object(a.ex.http, 'Client', side_effect=lambda: PipelineClient(folder, order)), \
                mock.patch('sys.stdin', io.StringIO(json.dumps(payload))):
            code = a.worker('fixture-launch-hash')
        return code, order, folder

    def test_actual_maintained_worker_three_cpu_fixtures_twelve_requests_and_positive_maps(self):
        args, candidate, journal, mappings = self.prepare_worker()
        orders = []
        for point in a.POINT_IDS:
            code, order, folder = self.invoke_point(point, args, candidate, journal)
            observed = a.ct.load(folder / 'lifecycle.json')
            pressure = a.ct.load(folder / 'point-assessment.json')
            self.assertEqual(code, 0, (observed.get('error'), observed.get('cleanup_error'), pressure))
            self.assertEqual(order, list(a.NAMES))
            self.assertEqual(len(observed['whole_log_token_join']), 4)
            self.assertTrue(pressure['can_continue'])
            self.assertEqual(pressure['full_numerical_per_point'], 'unverified')
            self.assertTrue((folder / 'source-options.json').is_file())
            self.assertEqual(observed['requests']['summary']['successful'], 4)
            orders.extend(order)
        self.assertEqual(len(orders), 12)
        self.assertEqual(mappings.call_count, 6)
        self.assertEqual([call.kwargs['directory'].name for call in mappings.call_args_list],
                         ['ready-library-evidence', 'after-requests-library-evidence'] * 3)
        self.assertTrue(all(call.kwargs['role'] == 'cli' for call in mappings.call_args_list))
        self.assertTrue(all(call.kwargs['parent'] == fixtures.TARGET for call in mappings.call_args_list))

    def test_source_option_rejection_blocks_requests_and_is_not_repaired_to_pass(self):
        args, candidate, journal, _ = self.prepare_worker()
        code, order, folder = self.invoke_point('small', args, candidate, journal, source_point='middle')
        self.assertEqual(code, 1)
        self.assertEqual(order, [])
        self.assertFalse(a.ct.load(folder / 'point-assessment.json')['can_continue'])
        self.assertFalse((folder / 'hello2/issued.json').exists())
        self.assertEqual(a.ex.recover_requests(folder, a.schedule())['unissued'], 4)


class CommandTests(fixtures.CPUOnly):
    def test_go_requires_explicit_grant_and_profiler_handoff_before_review(self):
        with mock.patch.object(a, 'validate_candidate') as review:
            for grant, paused in ((False, False), (True, False), (False, True)):
                with self.assertRaisesRegex(ValueError, 'GO'):
                    a.start(self.directory, grant, paused)
            review.assert_not_called()

    def test_entry_specializes_maintained_coordinator_without_shared_mutation(self):
        original = a.coordinator.execute_next.__globals__['registration']
        private = a.maintained_next()
        self.assertIs(a.coordinator.execute_next.__globals__['registration'], original)
        self.assertIs(private.__globals__['registration'], a.registration)
        self.assertIs(private.__globals__['next_trial'], a.next_point)
        self.assertIs(private.__globals__['pilot'].idle, a.idle)
        self.assertIs(private.__globals__['planned'].supervise, a.supervise)
        self.assertIs(private.__globals__['lifecycle_assessment'], a.lifecycle_assessment)
        self.assertIs(private.__globals__['actual_adapter'], a.nullcontext)
        self.assertIn('request_ceiling=12', a.inspect.getsource(a.coordinator.execute_next))

    def test_precise_expression_drift_fails_before_dispatch(self):
        with self.assertRaisesRegex(ValueError, 'expression changed'):
            a._specializations(a.coordinator.execute_next, (('invented expression', '4'),))

    def test_command_admission_keeps_entire_point_and_cleanup_reserve(self):
        with mock.patch.object(a.pilot.time, 'monotonic', return_value=100):
            self.assertTrue(a.can_admit(520))
            self.assertFalse(a.can_admit(519.999))

    def test_seals_are_exclusive_readonly_and_reject_rewrite(self):
        path = self.directory / 'GO.json'
        a.sealed_write(path, dict(schema='fixture', owner=True))
        self.assertTrue(a.unseal(path, 'fixture')['owner'])
        self.assertEqual(path.stat().st_mode & 0o222, 0)
        with self.assertRaises(FileExistsError):
            a.sealed_write(path, dict(schema='fixture', owner=False))
        with self.assertRaises(ValueError):
            a.unseal(path, 'wrong')

    def test_supervisor_keeps_authentication_and_only_changes_worker_entry(self):
        payload = dict(directory=str(self.directory))
        sealed = dict(payload, worker_token='fixture-secret')
        with mock.patch.object(a.pilot.launch_guard, 'seal', return_value=(sealed, 'fixture-hash')), \
                mock.patch.object(a.ct, 'isolated_argv', side_effect=lambda argv, env: argv), \
                mock.patch.object(a.pilot.supervisor, 'run_process', return_value=dict(fixture=True)) as run:
            self.assertEqual(a.supervise(payload, 190), dict(fixture=True))
        argv, seconds, data = run.call_args.args
        self.assertEqual(argv[-3:], [str(a.ENTRY), '--_worker', 'fixture-hash'])
        self.assertEqual(seconds, 190)
        self.assertEqual(data, sealed)
        self.assertNotEqual(a.pilot.ENTRY, a.ENTRY)

    def test_maintained_coordinator_issues_only_small_four_slot_payload(self):
        candidate = dict(candidate=a.p.load(a.BINDING_PATH)['candidate'], uuid_inventory=fixtures.UUIDS)
        registered = dict(trial_order=list(a.TRIALS), limits=a.p.LIMITS, schedule=a.schedule(),
            pilot_deadline=time.monotonic() + 900, eight_uuids=fixtures.UUIDS, selected_uuid=fixtures.UUIDS[0],
            binding_path=str(a.BINDING_PATH), binding_sha256=a.ct.sha(a.BINDING_PATH),
            execution_path=str(a.BINDING_PATH), execution_sha256=a.ct.sha(a.BINDING_PATH))
        (self.directory / 'registration.json').write_text('{}')
        @contextmanager
        def lock(uuid):
            yield dict(uuid=uuid)
        private = a.maintained_next()
        g = private.__globals__
        g['registration'] = lambda directory: registered
        g['binding'] = a.SimpleNamespace(validate_binding=lambda path: candidate,
            validate_execution=lambda path: dict(artifact_pins={str(a.BINDING_PATH): a.ct.sha(a.BINDING_PATH)}))
        captured = []
        def supervise(payload, seconds):
            captured.append((payload, seconds))
            return fixtures.drained()
        g['planned'].supervise = supervise
        g['pilot'].idle = lambda *args: dict(status='verified', all_eight_idle=True)
        g['pilot'].baseline = lambda *args: dict(fixture_only=True)
        with mock.patch.object(a.ct, 'exclusive_owner_lock', side_effect=lock), \
                mock.patch.object(a.ct, 'validate_owner_lock'), \
                mock.patch.object(a.pilot.telemetry, 'authorize'), \
                mock.patch.object(a.pilot.idle_window, 'no_recorded_owner', return_value=True):
            row = private(self.directory)
        self.assertEqual(len(captured), 1)
        payload, seconds = captured[0]
        self.assertEqual(seconds, 190)
        self.assertEqual(payload['schema'], a.SCHEMA)
        self.assertEqual(payload['trial'], 'small-blocking')
        self.assertEqual(payload['request_ceiling'], 4)
        self.assertEqual(payload['schedule'], a.schedule())
        self.assertEqual(payload['spec'], a.spec_for(candidate, fixtures.UUIDS[0], 'small'))
        self.assertEqual(len(payload['all_uuid_locks']), 8)
        self.assertLessEqual(payload['outer_deadline'] - time.monotonic(), 240)
        self.assertEqual(payload['pilot_deadline'], registered['pilot_deadline'] - 180)
        self.assertFalse(row['assessment']['can_continue'])
        self.assertEqual(row['summary']['unissued'], 4)
        self.assertTrue((self.directory / 'study-stop.json').is_file())
        with self.assertRaises(ValueError):
            a.next_point(self.directory, registered)


if __name__ == '__main__':
    unittest.main()
