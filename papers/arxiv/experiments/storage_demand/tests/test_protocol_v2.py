"""Dedicated CPU fixtures, never numerical/run/physical-IO observations."""
from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.dont_write_bytecode = True
os.environ.update(CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none')
STORAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STORAGE))
with mock.patch('subprocess.Popen', side_effect=AssertionError('CPU test: no launch')), \
        mock.patch('subprocess.run', side_effect=AssertionError('CPU test: no command/GPU query')):
    import runnerhelpers as helpers
    from runnerhelpers import closure_v2, evidence_v2, mountstats_v2, trace_v2
    from newversionedprotocol import protocol


class NoGPU(unittest.TestCase):
    def setUp(self):
        stack = self.enterContext(ExitStack())
        stack.enter_context(mock.patch('subprocess.Popen', side_effect=AssertionError('No launch in CPU tests')))
        stack.enter_context(mock.patch('subprocess.run', side_effect=AssertionError('No command/GPU query in CPU tests')))
        stack.enter_context(mock.patch.object(evidence_v2.ct, 'exclusive_owner_lock', side_effect=AssertionError('No GPU lock')))


def refs():
    return evidence_v2.ct.load(STORAGE / 'canaries.json')['rows']


def schedule():
    return helpers.build_schedule(refs())


def observation(plan, status, **extra):
    return dict(plan, status=status, **extra)


def known_summary():
    def observed(value):
        return dict(state='observed', value=value)
    return dict(
        schema_version=1, attempt=1, outcome='complete', error=None,
        runtime=dict(scope=dict(owner=101, generation=1), resident_bytes=observed(4096),
                     pending_operations=observed(0), fence=dict(state='unobserved'),
                     closure=observed('closed'), quarantine=dict(state='unobserved')),
        physical=dict(scope=observed(dict(owner=202, generation=1)),
                      quiescence=observed('quiescent'), close=observed('closed'),
                      residency=observed('retained'), release=dict(state='unobserved'),
                      quarantine=observed(False),
                      expert_cache=observed(dict(pending_upload_bytes=0,
                          unknown_quarantine_bytes=0, quarantined=False, resident_bytes=4096)),
                      allocator=observed(dict(live_requested_bytes=4096, shutdown_retained=True))),
        broker_domains=[observed(dict(kind='Arena', unit='slots', in_use=1))],
        coverage=dict(runtime=dict(unobserved=['fence', 'quarantine']), physical=dict(unobserved=['release'])),
    )


def token_line(request_id, ordinal, token_id, **extra):
    return json.dumps(dict(fields=dict(event='request_worker_token_enqueued', request_id=request_id,
        session_id=request_id, emission_ordinal=ordinal, actual_token_id=token_id,
        resident_token_index=ordinal - 1, delivery_boundary='worker_channel_enqueue', **extra)))


def http_row(identifier='cmpl-ferrule-9', count=2):
    data = [dict(id=identifier, choices=[dict(index=0, text='two tokens in one chunk', finish_reason=None)]),
            dict(id=identifier, choices=[dict(index=0, text='', finish_reason='length')]),
            dict(id=identifier, choices=[], usage=dict(prompt_tokens=5, completion_tokens=count, total_tokens=5 + count))]
    return dict(name='qa-fixture', reference=dict(reference_ids=[100, 200]), actual=dict(
        status='success', actual_output_tokens=count, usage=data[-1]['usage'], done_at=4.0,
        raw_sse_events=[dict(data=json.dumps(value), at=float(index + 1)) for index, value in enumerate(data)]
            + [dict(data='[DONE]', at=4.0)]))


def mount_snapshot(read=100, age=50, clock=1.0):
    raw = ('device nas:/models mounted on /mnt/nas1 with fstype nfs4 statvers=1.1\n'
           f' age: {age}\n bytes: 1 2 3 4 {read} 6 7 8\n'
           f' READ: 10 10 0 20 {read} 0 1 2\n')
    snapshot = helpers.parse_mountstats(raw, int(clock * 1000000000), clock, '/proc/fixture/mountstats')
    snapshot.update(boot_id='fixture-boot', mount_namespace='mnt:[fixture]',
                    acquisition_begin=dict(unix_ns=int(clock * 1000000000), monotonic=clock),
                    acquisition_end=dict(unix_ns=int(clock * 1000000000), monotonic=clock))
    snapshot['selected_mount'] = helpers.select_nfs_mount(snapshot, ['/mnt/nas1/hf/model/shard01'])
    return snapshot


class ScheduleTests(NoGPU):
    def test_exact_twelve_ceiling_and_phases(self):
        rows = schedule()
        self.assertEqual(helpers.request_phase_counts(rows), dict(initial=4, qa_excluded_first=1, qa_warmup=2, qa_measured=5))
        self.assertEqual(helpers.validate_schedule(rows)['ceiling'], 12)
        self.assertEqual([row['name'] for row in rows[:4]], ['hello2', 'canary-qa', 'canary-arithmetic', 'canary-chinese'])
        self.assertEqual(sum(row['measured'] for row in rows), 5)
        self.assertEqual(sum(row['max_tokens'] for row in rows), 90)
        self.assertEqual(rows[4]['population'], 'excluded_first')
        self.assertEqual(rows[4]['prompt'], rows[1]['prompt'])
        for row in rows:
            self.assertEqual(row['payload']['temperature'], 0)
            self.assertEqual(row['payload']['seed'], 1)
            self.assertEqual(row['payload']['n'], 1)

    def test_old_ten_ceiling_and_truncation_rejected(self):
        with self.assertRaises(ValueError):
            helpers.validate_schedule(schedule(), ceiling=10)
        with self.assertRaises(ValueError):
            helpers.validate_schedule(schedule()[:10])

    def test_order_drift_rejected(self):
        rows = schedule()
        rows[4], rows[5] = rows[5], rows[4]
        with self.assertRaises(ValueError):
            helpers.validate_schedule(rows)

    def test_unknown_phase_rejected(self):
        rows = schedule()
        rows[0]['phase'] = 'first_after_ready'
        with self.assertRaises(ValueError):
            helpers.validate_schedule(rows)

    def test_literal_finite_envelope(self):
        value = protocol.load_protocol()
        self.assertEqual(protocol.validate_protocol(value)['pilot_seconds'], 7200)
        limits = value['limits']
        self.assertEqual(limits['startup_seconds'] + limits['all_requests_seconds'] + limits['worker_bookkeeping_seconds'], 1240)
        self.assertLess(limits['all_requests_seconds'], 12 * limits['request_seconds'])
        self.assertLessEqual(limits['target_lifetime_seconds'], 1560)
        for field, bad in [('startup_seconds', 800), ('request_seconds', float('inf')),
                           ('supervisor_seconds', 1240), ('pilot_seconds', 3600), ('shutdown_seconds', 46)]:
            changed = copy.deepcopy(value)
            changed['limits'][field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                protocol.validate_protocol(changed)

    def test_request_cutoff_is_absolute_not_sliding(self):
        limits = protocol.load_protocol()['limits']
        self.assertEqual(evidence_v2.request_budget_seconds(limits, 400, 0, 1240, 1440, 7200), 80)
        self.assertEqual(evidence_v2.request_budget_seconds(limits, 480, 0, 1240, 1440, 7200), 0)
        self.assertEqual(evidence_v2.request_budget_seconds(limits, 200, 0, 205, 1440, 7200), 5)

    def test_phase_labels_measured_flags_and_seed_drift_rejected(self):
        for key, value in (('phase', 'qa_measured'), ('measured', 1)):
            rows = schedule()
            rows[0][key] = value
            with self.assertRaises(ValueError):
                helpers.validate_schedule(rows)
        rows = schedule()
        rows[0]['payload']['seed'] = True
        with self.assertRaises(ValueError):
            helpers.validate_schedule(rows)

    def test_fixed_five_not_retry_or_quality_claim(self):
        value = protocol.load_protocol()
        self.assertEqual(value['trial_order'], ['L1', 'L2', 'L3', 'L4', 'L5'])
        self.assertEqual(value['retry_count'], 0)
        self.assertIsNone(value['candidate_binary'])
        self.assertEqual(value['scheduled_requests'], 60)
        self.assertEqual(value['scheduled_measured_qa'], 25)
        self.assertTrue(value['custody']['no_drop_cache'])


class DenominatorTests(NoGPU):
    def test_failed_censored_and_unissued_conserve_all_twelve(self):
        plan = schedule()
        observed = [observation(plan[0], 'success'), observation(plan[1], 'failed'),
                    observation(plan[2], 'timeout', right_censored=True, censor_reason='deadline')]
        result = helpers.censoring_summary(observed, plan)
        self.assertEqual((result['scheduled'], result['issued'], result['terminal']), (12, 3, 2))
        self.assertEqual(result['counts'], dict(success=1, failed=1, censored=1, unissued=9))
        self.assertEqual(result['measured_qa'], dict(scheduled=5, success=0, eligible_success=0))
        self.assertTrue(result['conservation'])

    def test_missing_terminal_after_journaled_send_is_censored(self):
        plan = schedule()
        plan[0]['issued'] = True
        result = helpers.censoring_summary([], plan)
        self.assertEqual(result['censored'], 1)
        self.assertEqual(result['unissued'], 11)
        self.assertEqual(result['terminal'], 0)

    def test_unknown_issued_status_not_success_or_unissued(self):
        plan = schedule()
        result = helpers.censoring_summary([observation(plan[0], 'helper_lost')], plan)
        self.assertEqual(result['censored'], 1)
        self.assertEqual(result['successful'], 0)

    def test_success_not_metric_eligible_without_quality_and_trace(self):
        plan = schedule()
        rows = [observation(row, 'success', metric_eligible=row['index'] == 7) for row in plan]
        result = helpers.censoring_summary(rows, plan)
        self.assertEqual(result['measured_qa']['success'], 5)
        self.assertEqual(result['measured_qa']['eligible_success'], 1)

    def test_duplicate_or_unregistered_request_cannot_disappear(self):
        plan = schedule()
        for rows in ([plan[0], plan[0]], [dict(plan[0], name='extra-request')]):
            with self.assertRaises(ValueError):
                helpers.censoring_summary(rows, plan)


class TraceTests(NoGPU):
    def test_actual_id_ordinal_join_independent_of_client_chunks(self):
        trace = helpers.parse_token_trace([token_line(9, 1, 100), token_line(9, 2, 200)])
        result = helpers.join_token_trace_http(trace, [http_row()])[0]
        self.assertEqual(result['actual_worker_request_id'], 9)
        self.assertEqual(result['actual_generated_ids'], [100, 200])
        self.assertEqual(result['enqueue_ordinals'], [1, 2])
        self.assertTrue(result['trace_valid'])
        self.assertTrue(result['exact_token_pass'])
        self.assertEqual(len(result['client_consumption']['content_events']), 1)
        self.assertFalse(result['client_consumption']['content_events_are_token_boundaries'])

    def test_opaque_ids_retained_not_digit_guessed(self):
        trace = helpers.parse_token_trace([token_line(9, 1, 100), token_line(9, 2, 200)])
        for identifier in ('opaque-9', 'cmpl-other-9', 'cmpl-ferrule-09', '9', 'chatcmpl-ferrule-9'):
            result = helpers.join_token_trace_http(trace, [http_row(identifier)])[0]
            self.assertEqual(result['actual_http_completion_ids'], [identifier])
            self.assertIsNone(result['actual_worker_request_id'])
            self.assertFalse(result['trace_valid'])
            self.assertIsNone(result['exact_token_pass'])

    def test_duplicate_gap_reordering_and_wrong_session_block(self):
        cases = [
            [token_line(9, 1, 100), token_line(9, 1, 200)],
            [token_line(9, 1, 100), token_line(9, 3, 200)],
            [token_line(9, 2, 200), token_line(9, 1, 100)],
        ]
        for lines in cases:
            result = helpers.join_token_trace_http(helpers.parse_token_trace(lines), [http_row()])[0]
            self.assertFalse(result['trace_valid'])
        trace = helpers.parse_token_trace([token_line(9, 1, 100), token_line(9, 2, 200)])
        trace[0]['session_id'] = 7
        self.assertFalse(helpers.join_token_trace_http(trace, [http_row()])[0]['trace_valid'])

    def test_missing_trace_and_usage_mismatch_stay_unverified(self):
        self.assertFalse(helpers.join_token_trace_http([], [http_row()])[0]['trace_valid'])
        trace = helpers.parse_token_trace([token_line(9, 1, 100), token_line(9, 2, 200)])
        self.assertFalse(helpers.join_token_trace_http(trace, [http_row(count=3)])[0]['trace_valid'])
        result = helpers.join_token_trace_http(trace, [http_row(), http_row()])
        self.assertFalse(result[1]['trace_valid'])
        self.assertTrue(result[1]['duplicate_http_binding'])

    def test_terminal_enqueue_and_http_response_join_are_separate(self):
        trace = helpers.parse_token_trace([token_line(9, 1, 100), token_line(9, 2, 200)])
        terminal = [json.dumps(dict(fields=dict(event='request_worker_terminal_enqueue', request_id=9, status='finished', terminal_event_enqueued=True, last_emission_ordinal=2))), json.dumps(dict(fields=dict(event='request_http_terminal', request_id=9, status='finished', token_events=2)))]
        result = helpers.join_token_trace_http(trace, [http_row()], terminal)[0]
        self.assertTrue(result['terminal_reconciled'])
        self.assertTrue(result['metric_eligible'])
        self.assertFalse(helpers.join_token_trace_http(trace, [http_row()])[0]['metric_eligible'])

    def test_bad_token_trace_types_and_boundary_rejected(self):
        for field, value in [('actual_token_id', False), ('emission_ordinal', 0),
                             ('request_id', '9'), ('delivery_boundary', 'http_consumed')]:
            data = json.loads(token_line(9, 1, 100))
            data['fields'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                helpers.parse_token_trace([json.dumps(data)])

    def test_logical_missing_counters_not_zero(self):
        result = helpers.logical_counter_window(['not JSON'])
        self.assertFalse(result['valid'])
        self.assertIsNone(result['deltas']['checkpoint_read_bytes'])
        self.assertIsNone(result['physical_nas_bytes'])


class MountstatsTests(NoGPU):
    def test_raw_timestamped_shared_delta_not_physical_nas(self):
        before, after = mount_snapshot(100, clock=1), mount_snapshot(130, age=51, clock=2)
        result = helpers.mountstats_delta(before, after)
        self.assertEqual(result['status'], 'shared_client_delta')
        self.assertEqual(result['deltas']['serverreadbytes'], 30)
        self.assertEqual(result['deltas']['read_rpc_4'], 30)
        self.assertTrue(result['shared_client_traffic'])
        self.assertIsNone(result['physical_nas_backend_bytes'])
        self.assertIsNone(result['owned_server_physical_bytes'])
        self.assertIn('device nas:', before['raw_text'])

    def test_counter_decrease_or_age_reset_invalidates_all_deltas(self):
        before = mount_snapshot(100, age=50, clock=1)
        for after in (mount_snapshot(90, age=51, clock=2), mount_snapshot(130, age=1, clock=2)):
            result = helpers.mountstats_delta(before, after)
            self.assertEqual(result['status'], 'counter_reset')
            self.assertIsNone(result['deltas'])

    def test_missing_counter_or_snapshot_not_zero(self):
        before, after = mount_snapshot(100, clock=1), mount_snapshot(130, clock=2)
        del after['selected_mount']['counters']['serverreadbytes']
        for value in (None, {}, after):
            result = helpers.mountstats_delta(before, value)
            self.assertEqual(result['status'], 'missing_data')
            self.assertIsNone(result['deltas'])

    def test_boot_namespace_mount_device_or_time_drift(self):
        before = mount_snapshot(clock=1)
        changes = [dict(boot_id='other'), dict(mount_namespace='other'), dict(captured_monotonic=0)]
        for change in changes:
            after = dict(mount_snapshot(130, clock=2), **change)
            self.assertIsNone(helpers.mountstats_delta(before, after)['deltas'])
        after = mount_snapshot(130, clock=2)
        after['selected_mount']['device'] = 'other:/export'
        self.assertIsNone(helpers.mountstats_delta(before, after)['deltas'])

    def test_selects_longest_mount_for_all_paths_and_refuses_missing(self):
        snapshot = mount_snapshot()
        snapshot['mounts'].append(dict(snapshot['mounts'][0], mountpoint='/mnt/nas1/hf'))
        self.assertEqual(helpers.select_nfs_mount(snapshot, ['/mnt/nas1/hf/a', '/mnt/nas1/hf/b'])['mountpoint'], '/mnt/nas1/hf')
        with self.assertRaises(ValueError):
            helpers.select_nfs_mount(snapshot, ['/other/a'])
        snapshot['mounts'].append(dict(snapshot['mounts'][-1]))
        with self.assertRaises(ValueError):
            helpers.select_nfs_mount(snapshot, ['/mnt/nas1/hf/a'])

    def test_offline_parser_does_not_fabricate_capture_timestamps(self):
        parsed = helpers.parse_mountstats(mount_snapshot()['raw_text'])
        self.assertIsNone(parsed['captured_unix_ns'])
        self.assertIsNone(parsed['captured_monotonic'])


class ClosureTests(NoGPU):
    def test_retained_mapped_closed_owner_can_continue_but_not_final_verified(self):
        result = helpers.closure_from_resource_summary(known_summary(), mapped_owner=True)
        self.assertTrue(result['owner_retained'])
        self.assertTrue(result['can_continue'])
        self.assertFalse(helpers.stop_on_unknown_custody(result))
        self.assertEqual(result['resource_contract_complete'], 'unverified')
        self.assertFalse(result['universal_final_release_verified'])
        self.assertEqual(result['fences']['physical'], dict(state='observed', value='quiescent'))
        self.assertEqual(result['release']['state'], 'unobserved')

    def test_missing_data_blocks_without_claiming_unknown_physical_custody(self):
        for summary in ({}, dict(known_summary(), physical={})):
            result = helpers.closure_from_resource_summary(summary, mapped_owner=True)
            self.assertFalse(result['can_continue'])
            self.assertFalse(helpers.stop_on_unknown_custody(result))
            self.assertEqual(result['custody_state'], 'not_established')
        self.assertFalse(helpers.closure_from_resource_summary(known_summary())['can_continue'])

    def test_explicit_unknown_qualifiers_always_stop(self):
        for partition, field in [('runtime', 'fence'), ('runtime', 'closure'), ('physical', 'scope'),
                                 ('physical', 'quiescence'), ('physical', 'release')]:
            value = known_summary()
            value[partition][field] = dict(state='unknown')
            result = helpers.closure_from_resource_summary(value, mapped_owner=True)
            self.assertTrue(helpers.stop_on_unknown_custody(result), field)
            self.assertFalse(result['can_continue'])

    def test_pending_or_quarantine_not_closed(self):
        value = known_summary()
        value['physical']['expert_cache']['value']['pending_upload_bytes'] = 1
        result = helpers.closure_from_resource_summary(value, mapped_owner=True)
        self.assertTrue(result['pending'])
        self.assertFalse(result['can_continue'])
        value = known_summary()
        value['physical']['quarantine']['value'] = True
        self.assertTrue(helpers.stop_on_unknown_custody(helpers.closure_from_resource_summary(value, mapped_owner=True)))

    def test_runtime_pending_fence_and_allocator_work_block(self):
        value = known_summary()
        value['runtime']['fence'] = dict(state='observed', value='pending')
        self.assertFalse(helpers.closure_from_resource_summary(value, mapped_owner=True)['can_continue'])
        value = known_summary()
        value['physical']['allocator']['value']['pending_unfenced_bytes'] = 1
        self.assertFalse(helpers.closure_from_resource_summary(value, mapped_owner=True)['can_continue'])

    def test_no_fabricated_global_verified_or_retention(self):
        value = known_summary()
        value['resource_contract_complete'] = 'verified'
        for partition, key in [('physical', 'quiescence'), ('physical', 'close'), ('runtime', 'closure')]:
            value[partition][key]['state'] = 'verified'
        result = helpers.closure_from_resource_summary(value, mapped_owner=True)
        self.assertEqual(result['resource_contract_complete'], 'unverified')
        with self.assertRaises(ValueError):
            helpers.closure_from_resource_summary(value, owner_retained=True, mapped_owner=True)

    def test_raw_summary_json_and_unknown_attempt_sticky(self):
        first, second = known_summary(), known_summary()
        first['physical']['quiescence'] = dict(state='unknown')
        second['attempt'] = 2
        lines = [json.dumps(dict(fields=dict(event='resource_shutdown_summary', attempt=value['attempt'],
                       outcome=value['outcome'], summary_json=json.dumps(value)))) for value in (first, second)]
        result = helpers.parse_resource_shutdown_summaries(lines, mapped_owner=True)
        self.assertEqual(len(result['attempts']), 2)
        self.assertTrue(result['sticky_unknown'])
        self.assertFalse(result['final']['can_continue'])
        self.assertEqual(result['final']['custody_state'], 'unknown')
        self.assertEqual(helpers.parse_resource_shutdown_summaries([])['final']['resource_contract_complete'], 'unverified')


class QualificationAndCustodyTests(NoGPU):
    def test_unknown_candidate_qualification_never_seals(self):
        for state in (None, 'pending', 'unknown', 'pass', 'historical_pass'):
            with self.assertRaises(ValueError):
                protocol.prepare_candidate(dict(schema='ferrule.storage-demand.candidate-binding.v2', status=state), case={})

    def test_quality_requires_exact_raw_schema_config_and_qualifier(self):
        case = dict(quality_test_config=dict(name='storage-middle64'))
        from test_build_receipts_v1 import quality_reports
        reports = quality_reports(case['quality_test_config'])
        self.assertEqual(protocol.validate_quality_reports(reports, case)['status'], 'pass_same_StorageMiddle64')
        for label, field, value in [('full40', 'passed', None), ('fresh', 'complete', 'true'),
                                    ('cold', 'test_config', None), ('fresh', 'quarantine', None),
                                    ('full40', 'diagnostic_capture_only', True)]:
            changed = copy.deepcopy(reports)
            changed[label][field] = value
            with self.subTest(label=label, field=field), self.assertRaises(ValueError):
                protocol.validate_quality_reports(changed, case)

    def test_pid_birth_uuid_local_lock_contract(self):
        uuid = 'GPU-01234567-0123-0123-0123-0123456789ab'
        identity = dict(pid=123, birth_ticks=456, ppid=100)
        lock = dict(mode='local_nonblocking_flock', global_reservation=False)
        result = helpers.validate_custody_identity(identity, lock, uuid)
        self.assertFalse(result['global_reservation'])
        for bad in ({'pid': 123, 'ppid': 100}, dict(identity, birth_ticks=False)):
            with self.assertRaises(ValueError):
                helpers.validate_custody_identity(bad, lock, uuid)
        with self.assertRaises(ValueError):
            helpers.validate_custody_identity(identity, dict(lock, global_reservation=True), uuid)

    def test_cli_has_no_run_mode(self):
        with mock.patch('sys.stderr'), self.assertRaises(SystemExit):
            protocol.main(['run'])


if __name__ == '__main__':
    unittest.main()
