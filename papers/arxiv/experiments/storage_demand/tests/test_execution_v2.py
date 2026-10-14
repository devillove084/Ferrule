"""No-GPU execution regressions. All server Popen, GPU queries and locks are mocked."""
from contextlib import ExitStack, nullcontext, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

os.environ.update(CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none')
STORAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STORAGE))
from newversionedprotocol import run_pilot as pilot
from runnerhelpers import execution_v2 as ex
from runnerhelpers.evidence_v2 import COUNTERS
from test_protocol_v2 import schedule, known_summary, mount_snapshot
import _profile_capture as capture_api

UUIDS = [f'GPU-00000000-0000-0000-0000-{index:012d}' for index in range(8)]
ROOT = dict(pid=91003, birth_ticks=30, ppid=91002, state='S')
TARGET = dict(pid=91002, birth_ticks=20, ppid=91001, pgrp=91002, session=91002)
AUTH = dict(worker=dict(pid=91001, birth_ticks=10, ppid=91000, pgrp=91001, session=91001),
            helper=dict(pid=91000, birth_ticks=9, ppid=1), coordinator={}, launch_id='fixture')


def drained():
    helper = dict(pid=91000, birth_ticks=9)
    return dict(returncode=0, timed_out=False, cleanup_errors=[], descendant_custody='proven_drained',
                handshake=dict(containment_verified=True, launch_id='fixture', helper_identity=helper),
                custody=dict(containment_verified=True, echild=True, status='proven_drained',
                             launch_id='fixture', helper_identity=helper))


def summary_line(value):
    return json.dumps(dict(fields=dict(event='resource_shutdown_summary', attempt=value['attempt'],
                                      outcome=value['outcome'], summary_json=json.dumps(value)))) + '\n'


class MockClient:
    def __init__(self, directory, order, failure=None, duplicate=False):
        self.directory, self.order, self.failure, self.duplicate = directory, order, failure, duplicate

    def close(self):
        pass

    def request(self, config, identity, cutoff=None):
        index = len(self.order)
        planned = schedule()[index]
        self.order.append(planned['name'])
        if config['payload'] != planned['payload']:
            raise AssertionError('maintained HTTP payload drift')
        request_id = 100 if self.duplicate else 100 + index
        count = planned['max_tokens']
        usage = dict(prompt_tokens=planned['input_tokens'], completion_tokens=count,
                     total_tokens=planned['input_tokens'] + count)
        text = planned['reference']['reference_text'] if planned['reference'] else 'Hello fixture'
        token_ids = planned['reference']['reference_ids'] if planned['reference'] else [1, 2]
        lines = []
        for ordinal, token in enumerate(token_ids, 1):
            lines.append(json.dumps(dict(fields=dict(event='request_worker_token_enqueued', request_id=request_id,
                session_id=request_id, emission_ordinal=ordinal, resident_token_index=ordinal - 1,
                actual_token_id=token, delivery_boundary='worker_channel_enqueue'))))
        lines.append(json.dumps(dict(fields=dict(event='mixed_decoder_invocation', status='completed', device=0,
            counters_available=True, counter_reset_detected=False, quarantined=False,
            **{key: 1 for key in COUNTERS}))))
        lines.append(json.dumps(dict(fields=dict(event='request_worker_terminal_enqueue', request_id=request_id,
            session_id=request_id, last_emission_ordinal=count, terminal_event_enqueued=self.failure != 'backpressure',
            delivery_boundary='worker_channel_enqueue', status='finished', usage_available=self.failure != 'terminal-usage', **usage))))
        lines.append(json.dumps(dict(fields=dict(event='request_http_terminal', request_id=request_id,
            session_id=request_id, token_events=count, status='finished'))))
        with (self.directory / 'server.log').open('a') as log:
            log.write('\n'.join(lines) + '\n')
        now = time.monotonic()
        stream = ex.http.Stream()
        for value in (
            dict(id=f'cmpl-ferrule-{request_id}', choices=[dict(index=0, text=text, finish_reason=None)]),
            dict(id=f'cmpl-ferrule-{request_id}', choices=[dict(index=0, text='', finish_reason='length')]),
            dict(id=f'cmpl-ferrule-{request_id}', choices=[], usage=usage),
        ):
            stream.event(json.dumps(value), '', now)
        stream.event('[DONE]', '', now)
        stream.eof()
        status = 'sse_error' if self.failure == 'http' and index == 1 else 'success'
        result = dict(identity, status=status, sent_at=now - .001, ended_at=now,
                      deadline_at=cutoff, error=None, length_matched_success=status == 'success')
        result.update(stream.metrics(now - .001, now, status))
        return result


class CPUOnly(unittest.TestCase):
    def setUp(self):
        self.stack = self.enterContext(ExitStack())
        self.stack.enter_context(mock.patch('subprocess.Popen', side_effect=AssertionError('unexpected launch')))
        self.stack.enter_context(mock.patch('subprocess.run', side_effect=AssertionError('unexpected GPU query')))
        self.stack.enter_context(mock.patch.object(ex.ct, 'exclusive_owner_lock', side_effect=AssertionError('unexpected GPU lock')))
        self.stack.enter_context(mock.patch('os.kill', side_effect=AssertionError('numeric/foreign signal forbidden')))
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(mock.patch.object(ex.http.platform, 'platform', return_value='CPU fixture platform'))
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory(prefix='storage-pilot-cpu-')))

    def sampling(self, directory):
        counter = [0]
        def sample(label):
            counter[0] += 1
            value = mount_snapshot(read=100 + counter[0], age=100, clock=float(counter[0]))
            value.update(label=label, reader_identity=AUTH['worker'], shared_client_traffic=True)
            ex.ct.write(directory / (label + '-mountstats.json'), value)
            with (directory / (label + '-mountstats.txt')).open('x') as raw:
                raw.write(value['raw_text'])
            return value
        return sample

    def execute_requests(self, failure=None, duplicate=False, check=lambda: None, limits=None, target=None):
        (self.directory / 'server.log').touch()
        order = []
        self.stack.enter_context(mock.patch.object(ex.http, 'Client',
            side_effect=lambda: MockClient(self.directory, order, failure, duplicate)))
        limits = limits or pilot.protocol.load_protocol()['limits']
        deadline = time.monotonic() + 2000
        result = ex.serial_requests(self.directory, schedule(), dict(url='http://127.0.0.1:18369/v1/completions',
            model='qwen35-storage-demand-paper'), limits, target or deadline, deadline, deadline,
            check, self.sampling(self.directory), lambda: dict(root_identity=ROOT, counters=dict(read_bytes=3), physical_nfs_bytes=None))
        return result, order


class AdmissionTests(CPUOnly):
    def test_unbound_candidate_rejected_before_lock_query_launch_or_output(self):
        candidate = self.directory / 'unbound.json'
        ex.ct.write(candidate, dict(status='cpu_prepared_candidate_unbound_not_authorized', sealed_final_candidate=False))
        args = SimpleNamespace(execution_prepared=self.directory / 'execution.json', candidate=candidate,
            output=self.directory / 'artifacts-rejected', gpu_uuid=UUIDS[0], eight_uuids=UUIDS,
            sole_gpu_owner_granted=True, owner_note='fixture only')
        with mock.patch.object(pilot, 'validate_execution', return_value={}):
            with self.assertRaisesRegex(ValueError, 'unbound'):
                pilot.run(args)
        self.assertFalse(args.output.exists())

    def test_candidate_rechecks_actual_build_quality_not_a_boolean_seal(self):
        path = self.directory / 'candidate.json'
        value = dict(sealed_final_candidate=True, status='quality_passed_cpu_prepared_not_authorized', candidate={})
        value['digest'] = ex.ct.digest(value)
        ex.ct.write(path, value)
        with mock.patch.object(pilot.protocol, 'prepare_candidate', side_effect=ValueError('build/full40/fresh/cold missing')) as gate:
            with self.assertRaisesRegex(ValueError, 'full40'):
                pilot.validate_candidate(path)
        gate.assert_called_once_with({})

    def test_exact_eight_inventory_and_selected_uuid(self):
        pilot.validate_inventory(UUIDS, UUIDS[4])
        for values, chosen in ((UUIDS[:7], UUIDS[0]), (UUIDS[:7] + [UUIDS[0]], UUIDS[0]),
                               (UUIDS, '0'), (UUIDS, 'GPU-99999999-9999-9999-9999-999999999999')):
            with self.subTest(values=values, chosen=chosen), self.assertRaises(ValueError):
                pilot.validate_inventory(values, chosen)

    def test_spec_binds_uuid_not_literal_zero_and_exact_case(self):
        value = pilot.spec_for(dict(candidate=dict(binary='/fixture/immutable/ferrule')), UUIDS[4])
        self.assertEqual(value['environment']['CUDA_VISIBLE_DEVICES'], UUIDS[4])
        self.assertEqual(value['environment']['NVIDIA_VISIBLE_DEVICES'], UUIDS[4])
        self.assertEqual(value['argv'][1:], ex.ct.load(pilot.protocol.CASE_PATH)['argv_after_binary'])
        self.assertEqual(value['environment']['FERRULE_LOG_FORMAT'], 'json')

    def test_sealed_worker_reuses_launch_guard_and_maintained_supervisor(self):
        payload = dict(directory=str(self.directory), contract_sha256='fixture', adapter_pins={})
        with mock.patch.object(pilot.supervisor, 'run_process', return_value=drained()) as supervise:
            receipt = pilot.supervise(payload, 1340)
        self.assertTrue(ex.ct.drained(receipt))
        argv, seconds, sealed = supervise.call_args.args
        self.assertEqual(seconds, 1340)
        self.assertIn(str(pilot.ENTRY), argv)
        self.assertIn('--_worker', argv)
        self.assertEqual(len(sealed['worker_token']), 64)
        self.assertEqual(ex.ct.load(self.directory / 'worker-launch.json'), sealed)
        self.assertEqual((self.directory / 'worker-launch.json').stat().st_mode & 0o222, 0)

    def test_post_idle_passes_real_custody_receipt_to_versioned_helper(self):
        receipt = drained()
        with mock.patch.object(pilot.idle_window, 'observe', return_value=dict(status='verified', all_eight_idle=True)) as observe:
            pilot.idle(self.directory, 'post-idle', UUIDS, time.monotonic() + 20, receipt)
        observe.assert_called_once_with(UUIDS, custody=receipt)


class RequestExecutionTests(CPUOnly):
    def test_twelve_named_requests_real_http_main_trace_and_measured_numerator(self):
        result, order = self.execute_requests()
        self.assertEqual(order, [row['name'] for row in schedule()])
        self.assertTrue(result['can_continue'], result['stop_reason'])
        self.assertEqual(result['summary']['successful'], 12)
        self.assertEqual(result['summary']['measured_qa'], dict(scheduled=5, success=5, eligible_success=5))
        for row in result['observations']:
            self.assertTrue(row['trace']['terminal_reconciled'])
            self.assertEqual(row['trace']['actual_http_completion_ids'], [f"cmpl-ferrule-{100 + row['index']}"])
            self.assertEqual(row['trace']['enqueue_ordinals'], list(range(1, row['max_tokens'] + 1)))
            self.assertIsNone(row['nfs']['physical_nas_backend_bytes'])
            self.assertTrue((self.directory / row['name'] / 'issued.json').exists())
            self.assertTrue((self.directory / row['name'] / 'http-terminal.json').exists())

    def test_failed_http_retains_failure_denominator_and_no_retry(self):
        result, order = self.execute_requests(failure='http')
        self.assertEqual(order, ['hello2', 'canary-qa'])
        self.assertEqual(result['summary']['counts'], dict(success=1, failed=1, censored=0, unissued=10))
        self.assertEqual(result['summary']['measured_qa']['scheduled'], 5)
        self.assertFalse(result['can_continue'])

    def test_backpressure_terminal_not_enqueued_stops_quality_not_http_denominator(self):
        result, order = self.execute_requests(failure='backpressure')
        self.assertEqual(order, ['hello2'])
        self.assertEqual(result['summary']['successful'], 1)
        self.assertEqual(result['summary']['unissued'], 11)
        self.assertFalse(result['observations'][0]['metric_eligible'])
        self.assertFalse(result['observations'][0]['trace']['terminal_reconciled'])

    def test_terminal_usage_missing_suppresses_metric_even_with_successful_http(self):
        result, order = self.execute_requests(failure='terminal-usage')
        self.assertEqual(order, ['hello2'])
        self.assertEqual(result['summary']['successful'], 1)
        self.assertFalse(result['observations'][0]['metric_eligible'])

    def test_http_cleanup_failure_does_not_qualify_successful_response(self):
        (self.directory / 'server.log').touch()
        order = []
        client = MockClient(self.directory, order)
        client.close = mock.Mock(side_effect=ValueError('fixture client close failed'))
        with mock.patch.object(ex.http, 'Client', return_value=client):
            result = ex.serial_requests(self.directory, schedule(), dict(url='http://127.0.0.1:18369/v1/completions',
                model='qwen35-storage-demand-paper'), pilot.protocol.load_protocol()['limits'],
                time.monotonic() + 2000, time.monotonic() + 2000, time.monotonic() + 2000,
                lambda: None, self.sampling(self.directory), lambda: {})
        self.assertEqual(order, ['hello2'])
        self.assertEqual(result['summary']['successful'], 1)
        self.assertFalse(result['observations'][0]['actual']['adapter_http_report_complete'])
        self.assertFalse(result['observations'][0]['metric_eligible'])

    def test_duplicate_actual_http_id_cannot_bind_to_second_request(self):
        result, order = self.execute_requests(duplicate=True)
        self.assertEqual(order, ['hello2', 'canary-qa'])
        self.assertTrue(result['observations'][-1]['duplicate_lifecycle_request_id'])
        self.assertFalse(result['observations'][-1]['metric_eligible'])

    def test_unknown_custody_stops_before_next_dispatch(self):
        calls = [0]
        def check():
            calls[0] += 1
            if calls[0] == 4:
                raise ex.observer.CustodyLost('birth/parent/UUID unknown')
        result, order = self.execute_requests(check=check)
        self.assertEqual(order, ['hello2'])
        self.assertEqual(result['summary']['issued'], 1)
        self.assertEqual(result['summary']['unissued'], 11)
        self.assertIn('CustodyLost', result['stop_reason'])

    def test_post_send_unknown_retains_censored_admission_journal(self):
        calls = [0]
        def check():
            calls[0] += 1
            if calls[0] == 3:
                raise ex.observer.CustodyLost('unknown after send')
        result, order = self.execute_requests(check=check)
        self.assertEqual(order, ['hello2'])
        self.assertEqual(result['summary']['censored'], 1)
        recovered = ex.recover_requests(self.directory, schedule())
        self.assertEqual(recovered['censored'], 1)
        self.assertEqual(recovered['unissued'], 11)

    def test_aggregate_deadline_consumed_in_preparation_leaves_all_unissued(self):
        limits = copy.deepcopy(pilot.protocol.load_protocol()['limits'])
        limits['all_requests_seconds'] = .001
        def check():
            time.sleep(.003)
        result, order = self.execute_requests(check=check, limits=limits)
        self.assertEqual(order, [])
        self.assertEqual(result['summary']['unissued'], 12)

    def test_append_only_no_resume_or_overwrite(self):
        result, order = self.execute_requests()
        first = ex.ct.sha(self.directory / 'requests-final.json')
        with self.assertRaises(FileExistsError):
            ex.serial_requests(self.directory, schedule(), {}, pilot.protocol.load_protocol()['limits'],
                time.monotonic() + 2000, time.monotonic() + 2000, time.monotonic() + 2000,
                lambda: None, lambda label: None, lambda: None)
        self.assertEqual(ex.ct.sha(self.directory / 'requests-final.json'), first)


class CaptureAndClosureTests(CPUOnly):
    def test_actual_run_target_popen_devnull_and_normal_direct_child_close(self):
        child = mock.Mock(pid=ROOT['pid'], returncode=None)
        child.poll.side_effect = lambda: child.returncode
        child.wait.side_effect = lambda timeout: setattr(child, 'returncode', 0)
        ex.stop_capture(self.directory)
        with mock.patch.object(capture_api.subprocess, 'Popen', return_value=child) as popen:
            code = capture_api.run_target(self.directory, ['/fixture/ferrule'], 1240, 45)
        self.assertEqual(code, 0)
        popen.assert_called_once_with(['/fixture/ferrule'], stdin=subprocess.DEVNULL)
        child.terminate.assert_called_once_with()
        child.wait.assert_called_once_with(timeout=45)
        self.assertEqual(ex.ct.load(self.directory / 'profile-server.json')['pid'], ROOT['pid'])

    def test_launch_adapter_keeps_popen_module_constants_and_no_fd_exec_path(self):
        spec = pilot.spec_for(dict(candidate=dict(binary='/fixture/immutable/ferrule')), UUIDS[0])
        with mock.patch.object(ex.ct, 'pinned_executable', return_value=nullcontext(42)), \
                mock.patch.object(ex.runner, 'require_free_port'), mock.patch.object(ex.subprocess, 'Popen') as popen:
            ex.launch_capture(self.directory, spec, '0' * 64, 1240, 45)
        argv = popen.call_args.args[0]
        self.assertIn(str(ex.ct.ROOT / 'scripts/_profile_capture.py'), argv)
        self.assertIn('/fixture/immutable/ferrule', argv)
        self.assertNotIn('/proc/self/fd/42', argv)
        self.assertEqual(popen.call_args.kwargs['stdin'], subprocess.DEVNULL)
        self.assertEqual(popen.call_args.kwargs['stderr'], subprocess.STDOUT)

    def test_capture_timeout_leaves_descendants_to_maintained_subreaper(self):
        child = mock.Mock(pid=ROOT['pid'], returncode=None)
        child.poll.return_value = None
        child.wait.side_effect = subprocess.TimeoutExpired('fixture', 45)
        ex.stop_capture(self.directory)
        with mock.patch.object(capture_api.subprocess, 'Popen', return_value=child):
            code = capture_api.run_target(self.directory, ['fixture'], 1240, 45)
        self.assertEqual(code, 1)
        self.assertTrue(ex.ct.load(self.directory / 'profile-server.json')['shutdown_timeout'])
        child.kill.assert_not_called()

    def test_startup_absolute_identity_deadline_does_not_slide(self):
        process = mock.Mock(pid=TARGET['pid'])
        with mock.patch.object(ex.observer, 'proc_identity', return_value=TARGET):
            with self.assertRaisesRegex(TimeoutError, 'absolute startup'):
                ex.capture_root(process, self.directory, time.monotonic() - 1, AUTH)
        process.poll.assert_not_called()

    def test_absolute_alarm_expires_and_restores_handler(self):
        previous = signal.getsignal(signal.SIGALRM)
        with self.assertRaises(TimeoutError):
            with ex.absolute_alarm(time.monotonic() + .01):
                time.sleep(.05)
        self.assertEqual(signal.getsignal(signal.SIGALRM), previous)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))

    def test_foreign_nonselected_app_blocked_without_foreign_signal(self):
        monitor = pilot.PilotMonitor.__new__(pilot.PilotMonitor)
        monitor.uuids = [UUIDS[0]]
        with mock.patch.object(ex.observer.Monitor, 'gpu_delta_check', return_value={}):
            with self.assertRaisesRegex(ValueError, 'foreign nonselected'):
                monitor.gpu_delta_check(dict(apps=[dict(uuid=UUIDS[1], pid=999)]))

    def test_retained_actual_closed_owner_distinct_from_final_release(self):
        result = ex.owner_summary([summary_line(known_summary())], ROOT, UUIDS[0])
        self.assertTrue(result['final']['can_continue'])
        self.assertTrue(result['final']['owner_retained'])
        self.assertEqual(result['source_binding']['original_scopes']['physical'], dict(owner=202, generation=1))
        self.assertEqual(result['final']['resource_contract_complete'], 'unverified')
        self.assertFalse(result['final']['universal_final_release_verified'])
        assessed = ex.lifecycle_assessment(dict(status='complete', closure=result), drained(),
                                         dict(status='verified', all_eight_idle=True))
        self.assertTrue(assessed['can_continue'])

    def test_owner_scope_change_unknown_sticky_and_missing_not_zero(self):
        first = known_summary()
        second = copy.deepcopy(first)
        second['attempt'] = 2
        second['physical']['scope']['value']['generation'] = 2
        result = ex.owner_summary([summary_line(first), summary_line(second)], ROOT, UUIDS[0])
        self.assertEqual(result['final']['custody_state'], 'unknown')
        self.assertFalse(result['final']['can_continue'])
        self.assertFalse(ex.owner_summary([], ROOT, UUIDS[0])['final']['can_continue'])
        first['physical']['quiescence'] = dict(state='unknown')
        second = known_summary()
        second['attempt'] = 2
        result = ex.owner_summary([summary_line(first), summary_line(second)], ROOT, UUIDS[0])
        self.assertTrue(result['sticky_unknown'])
        self.assertFalse(result['final']['can_continue'])

    def test_process_io_is_birth_bound_and_not_nfs_physical(self):
        root = ex.observer.proc_identity(os.getpid())
        result = ex.process_io(root)
        self.assertEqual(result['root_identity']['birth_ticks'], root['birth_ticks'])
        self.assertIn('read_bytes', result['counters'])
        self.assertIsNone(result['physical_nfs_bytes'])
        changed = dict(root, birth_ticks=root['birth_ticks'] + 1)
        with self.assertRaises(ex.observer.CustodyLost):
            ex.process_io(changed)

    def test_observer_stop_only_writes_capture_marker_never_signals_server(self):
        server = ex.CapturedServer(mock.Mock(), self.directory, ROOT)
        with mock.patch.object(ex.observer, 'verify_root', return_value=ROOT):
            server.terminate()
        self.assertTrue((self.directory / 'profile-stop').exists())


class MockPilotEndToEndTests(CPUOnly):
    def fixture(self, unknown=False, summary=None, shutdown_sample_fail=False):
        schedule_path = self.directory / 'requests-v2.json'
        ex.ct.write(schedule_path, dict(schedule=schedule()))
        candidate_path = self.directory / 'candidate.json'
        execution_path = self.directory / 'execution.json'
        runtime_path = self.directory / 'runtime.json'
        ex.ct.write(candidate_path, dict(fixture_only=True))
        ex.ct.write(execution_path, dict(fixture_only=True))
        ex.ct.write(runtime_path, dict(pins={}))
        candidate = dict(candidate=dict(binary='/fixture/immutable/ferrule', binary_sha256='0' * 64,
                                       runtime_pin_manifest=str(runtime_path), artifact_pins={}))
        execution = dict(limits=pilot.protocol.load_protocol()['limits'], trial_order=['L1', 'L2', 'L3', 'L4', 'L5'], artifact_pins={})
        calls, orders, sample_labels = [], [], []
        self.stack.enter_context(mock.patch.object(pilot, 'HERE', self.directory))
        self.stack.enter_context(mock.patch.object(pilot, 'validate_execution', return_value=execution))
        self.stack.enter_context(mock.patch.object(pilot, 'validate_candidate', return_value=candidate))
        self.stack.enter_context(mock.patch.object(pilot, 'worker_contract', return_value=candidate))
        self.stack.enter_context(mock.patch.object(ex.ct, 'check_pins'))
        self.stack.enter_context(mock.patch.object(ex.ct, 'exclusive_owner_lock', return_value=nullcontext(dict(fixture_only=True))))
        self.stack.enter_context(mock.patch.object(ex.ct, 'validate_owner_lock'))
        self.stack.enter_context(mock.patch.object(pilot.idle_window, 'observe', return_value=dict(status='verified', all_eight_idle=True)))
        self.stack.enter_context(mock.patch.object(pilot, 'baseline', return_value={}))
        self.stack.enter_context(mock.patch.object(pilot.launch_guard, 'authenticate', return_value=AUTH))
        self.stack.enter_context(mock.patch.object(pilot.launch_guard, 'identity', return_value=TARGET))
        self.stack.enter_context(mock.patch.object(pilot.os, 'getpgrp', side_effect=os.getpid))
        self.stack.enter_context(mock.patch.object(ex.observer, 'verify_root', return_value=ROOT))
        self.stack.enter_context(mock.patch.object(ex, 'capture_root', return_value=(TARGET, ROOT, {})))
        self.stack.enter_context(mock.patch.object(ex.http, 'wait_ready', side_effect=lambda *args: time.monotonic()))
        self.stack.enter_context(mock.patch.object(ex, 'library_maps', return_value=dict(libraries={})))
        self.stack.enter_context(mock.patch.object(ex, 'process_io', return_value=dict(physical_nfs_bytes=None)))
        self.stack.enter_context(mock.patch.object(ex.ct, 'pinned_executable', return_value=nullcontext(42)))
        self.stack.enter_context(mock.patch.object(ex.runner, 'require_free_port'))
        real_read = Path.read_text
        def read(path, *args, **kwargs):
            if str(path) == f'/proc/{ROOT["pid"]}/environ':
                return 'CUDA_VISIBLE_DEVICES=' + UUIDS[0] + '\0NVIDIA_VISIBLE_DEVICES=' + UUIDS[0] + '\0'
            return real_read(path, *args, **kwargs)
        self.stack.enter_context(mock.patch.object(Path, 'read_text', read))
        real_sha = ex.ct.sha
        self.stack.enter_context(mock.patch.object(ex.ct, 'sha', side_effect=lambda path:
            '0' * 64 if str(path) == f'/proc/{ROOT["pid"]}/exe' else real_sha(path)))
        class Monitor:
            def __init__(self, *args, **kwargs):
                self.failed = threading.Event()
                self.stop = threading.Event()
                self.thread = mock.Mock()
                self.thread.is_alive.return_value = False
                self.errors = []
            def start(self):
                pass
            def mark_ready(self, at):
                pass
            def finish(self):
                return dict(status='observed_within_targets', fixture_only=True)
        self.stack.enter_context(mock.patch.object(pilot, 'PilotMonitor', Monitor))
        current = {}
        def sampler(directory, reader, root, paths):
            if directory not in current:
                current[directory] = self.sampling(directory)
            def sample(label):
                sample_labels.append(label)
                if (shutdown_sample_fail is True and label == 'shutdown-pre') or shutdown_sample_fail == label:
                    raise ValueError('fixture shutdown sample unavailable')
                return current[directory](label)
            return sample
        self.stack.enter_context(mock.patch.object(ex, 'nfs_sampler', side_effect=sampler))
        shutdown_summary = summary or known_summary()
        class Capture:
            pid = TARGET['pid']
            returncode = None
            def __init__(self, argv):
                self.argv = argv
                self.directory = Path(argv[argv.index('--directory') + 1])
                self.command = argv[argv.index('--') + 1:]
            def poll(self):
                return self.returncode
            def wait(self, timeout):
                server = mock.Mock(pid=ROOT['pid'], returncode=None)
                server.poll.side_effect = lambda: server.returncode
                def terminate():
                    with (self.directory / 'server.log').open('a') as stream:
                        stream.write(summary_line(shutdown_summary))
                server.terminate.side_effect = terminate
                server.wait.side_effect = lambda timeout: setattr(server, 'returncode', 0)
                with mock.patch.object(capture_api.subprocess, 'Popen', return_value=server):
                    self.returncode = capture_api.run_target(self.directory, self.command, 1240, 45)
                return self.returncode
        def popen(argv, **kwargs):
            self.assertEqual(kwargs['stdin'], subprocess.DEVNULL)
            self.assertEqual(kwargs['stderr'], subprocess.STDOUT)
            self.assertIn('_profile_capture.py', ' '.join(argv))
            return Capture(argv)
        self.stack.enter_context(mock.patch.object(ex.subprocess, 'Popen', side_effect=popen))
        def supervised(payload, seconds):
            calls.append(payload['trial'])
            order = []
            orders.append(order)
            payload = dict(payload, coordinator=dict(pid=os.getpid()))
            with mock.patch('sys.stdin', io.StringIO(json.dumps(payload))), \
                    mock.patch.object(ex.http, 'Client', side_effect=lambda: MockClient(Path(payload['directory']), order)):
                code = pilot.worker('fixture-seal')
            receipt = drained()
            receipt['returncode'] = code
            if unknown:
                receipt['descendant_custody'] = 'unknown'
            return receipt
        self.stack.enter_context(mock.patch.object(pilot, 'supervise', side_effect=supervised))
        args = SimpleNamespace(execution_prepared=execution_path, candidate=candidate_path,
            output=self.directory / 'artifacts-mock', gpu_uuid=UUIDS[0], eight_uuids=UUIDS,
            sole_gpu_owner_granted=True, owner_note='mock CPU fixture only, not authorization')
        return args, calls, orders, sample_labels

    def test_actual_capture_adapter_five_lifecycles_sixty_requests_twentyfive_measured(self):
        args, calls, orders, labels = self.fixture()
        result = pilot.run(args)
        self.assertEqual(result['status'], 'complete', result.get('stop_reason'))
        self.assertEqual(calls, ['L1', 'L2', 'L3', 'L4', 'L5'])
        self.assertEqual([len(order) for order in orders], [12] * 5)
        self.assertEqual(result['denominators']['scheduled'], 60)
        self.assertEqual(result['denominators']['successful'], 60)
        self.assertEqual(result['measured_qa'], dict(scheduled=25, success=25, eligible_success=25))
        for label in ('prelaunch', 'ready', 'requests-pre', 'requests-post', 'shutdown-pre', 'shutdown-post'):
            self.assertEqual(labels.count(label), 5)
        self.assertTrue(all(row['lifecycle']['closure']['final']['owner_retained'] for row in result['lifecycles']))
        self.assertFalse(result['universal_final_release_verified'])
        self.assertIsNone(result['physical_nas_backend_bytes'])

    def test_unknown_supervisor_stops_all_following_trials_even_after_exit_and_idle(self):
        args, calls, orders, labels = self.fixture(unknown=True)
        result = pilot.run(args)
        self.assertEqual(calls, ['L1'])
        self.assertEqual(result['denominators']['scheduled'], 60)
        self.assertEqual(result['denominators']['unissued'], 48)
        self.assertEqual(result['measured_qa']['scheduled'], 25)
        self.assertEqual(result['measured_qa']['eligible_success'], 0)
        self.assertIn('unknown authenticated', result['stop_reason'])

    def test_unknown_resource_close_stops_without_manufactured_final_release(self):
        summary = known_summary()
        summary['physical']['quiescence'] = dict(state='unknown')
        args, calls, orders, labels = self.fixture(summary=summary)
        result = pilot.run(args)
        self.assertEqual(calls, ['L1'])
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(result['lifecycles'][0]['lifecycle']['closure']['final']['custody_state'], 'unknown')

    def test_startup_timeout_preserves_all_sixty_scheduled_and_no_replacement(self):
        args, calls, orders, labels = self.fixture()
        with mock.patch.object(ex.http, 'wait_ready', side_effect=TimeoutError('absolute startup deadline')):
            result = pilot.run(args)
        self.assertEqual(calls, ['L1'])
        self.assertEqual(orders, [[]])
        self.assertEqual(result['denominators']['unissued'], 60)
        self.assertEqual(result['measured_qa'], dict(scheduled=25, success=0, eligible_success=0))
        self.assertIn('startup deadline', result['lifecycles'][0]['lifecycle']['error'])

    def test_pending_final_owner_close_blocks_next_lifecycle(self):
        summary = known_summary()
        summary['outcome'] = 'pending'
        args, calls, orders, labels = self.fixture(summary=summary)
        result = pilot.run(args)
        self.assertEqual(calls, ['L1'])
        self.assertEqual(result['status'], 'incomplete')
        self.assertTrue(result['lifecycles'][0]['lifecycle']['closure']['final']['pending'])
        self.assertEqual(result['measured_qa']['eligible_success'], 0)

    def test_shutdown_post_sampling_failure_preserves_actual_resource_summary(self):
        args, calls, orders, labels = self.fixture(shutdown_sample_fail='shutdown-post')
        result = pilot.run(args)
        self.assertEqual(calls, ['L1'])
        lifecycle = result['lifecycles'][0]['lifecycle']
        self.assertTrue(lifecycle['closure']['final']['can_continue'])
        self.assertTrue(lifecycle['closure']['final']['owner_retained'])
        self.assertEqual(lifecycle['shutdown_nfs']['status'], 'missing_data')
        self.assertFalse(result['lifecycles'][0]['assessment']['can_continue'])
        self.assertEqual(result['measured_qa']['eligible_success'], 0)

    def test_shutdown_sampling_failure_still_closes_capture_and_retains_failure(self):
        args, calls, orders, labels = self.fixture(shutdown_sample_fail=True)
        result = pilot.run(args)
        self.assertEqual(calls, ['L1'])
        lifecycle = result['lifecycles'][0]['lifecycle']
        self.assertEqual(lifecycle['capture']['returncode'], 0)
        self.assertIn('shutdown_sample_error', lifecycle)
        self.assertEqual(lifecycle['shutdown_nfs']['status'], 'missing_data')
        self.assertEqual(result['status'], 'incomplete')


if __name__ == '__main__':
    unittest.main()
