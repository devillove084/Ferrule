"""CPU clocks/fixtures only; no GPU, network, real serving process or new data."""
import signal
import time
import unittest
from unittest import mock

import subset_adapter as a
from phase_timing import Timings, deadline_cap, PHASES
import test_subset_adapter as owned


class Clock:
    def __init__(self):
        self.now = 100.0
    def advance(self, seconds):
        self.now += seconds
    def __call__(self):
        return self.now


class DeadlineTests(owned.fixtures.CPUOnly):
    def test_all_phases_compose_from_one_point_origin(self):
        expected = dict(startup=60, requests=180, bookkeeping=195, shutdown=215,
                        worker_finalization=219, supervisor_cleanup=230, postidle=235, point=240)
        self.assertEqual(a.p.budget_proof(a.p.LIMITS), expected)
        payload = dict(limits=a.p.LIMITS, outer_deadline=340, pilot_deadline=1000)
        self.assertEqual(a.phase_deadlines(payload), {k: 100 + v for k, v in expected.items()})
        # Startup includes both idles, candidate review and authentication: not
        # a new 60s clock after launch, and cleanup isn't added beyond 240s.
        self.assertEqual(60 + 120 + 15 + 45, 240)
        self.assertEqual(3 * 240 + 180, 900)
        self.assertEqual(a.p.LIMITS['request_seconds'], a.p.load(a.p.HERE / 'plan.json')['pressure']['limits']['request_seconds'])
        for key in a.p.LIMITS:
            value = dict(a.p.LIMITS)
            value[key] += 1
            with self.assertRaises(ValueError):
                a.p.budget_proof(value)

    def test_new_study_rejects_historical_directory_and_identity(self):
        with self.assertRaisesRegex(ValueError, 'historical resume'):
            a.directory_check(a.p.HERE / 'artifacts-go-20261015-01')
        with self.assertRaisesRegex(ValueError, 'historical resume'):
            a.directory_check(a.p.HERE / 'artifacts-go-v2-20261015-01')
        self.assertEqual(a.directory_check(a.p.HERE / 'artifacts-go-v2-20261015-02').name,
                         'artifacts-go-v2-20261015-02')
        with self.assertRaises(ValueError):
            a.p.validate_design(a.p.load(a.p.HERE / 'plan.json'))
        self.assertNotEqual(a.BINDING_PATH, a.p.HERE / 'bindings.json')

    def test_nested_alarm_restores_remaining_not_original_duration(self):
        clock = Clock()
        state = [0.0, 0.0]
        def gettimer(which):
            return tuple(state)
        def settimer(which, delay, interval=0):
            state[:] = [delay, interval]
        with mock.patch('phase_timing.time.monotonic', clock), \
                mock.patch('phase_timing.signal.getitimer', gettimer), \
                mock.patch('phase_timing.signal.setitimer', settimer), \
                mock.patch('phase_timing.signal.signal'):
            with deadline_cap(110):
                self.assertEqual(state[0], 10)
                with deadline_cap(103):
                    self.assertEqual(state[0], 3)
                    clock.advance(2)
                self.assertEqual(state[0], 8)
            self.assertEqual(state, [0, 0])

    def test_finalization_stop_cannot_report_continuation_from_earlier_assessment(self):
        (self.directory / 'study-stop.json').write_text('{}')
        row = dict(trial='small-blocking', assessment=dict(can_continue=True), summary={})
        with mock.patch.object(a, 'execute_next', return_value=row), \
                mock.patch.object(a.ct, 'require_cpu_python'), mock.patch.object(a.signal, 'signal'):
            code = a.main(['next', '--directory', str(self.directory)])
        self.assertEqual(code, 1)

    def test_supervisor_deadline_preserves_owned_cleanup_receipt(self):
        clock = Clock()
        payload = dict(directory=str(self.directory), limits=a.p.LIMITS,
                       outer_deadline=340, pilot_deadline=1000)
        sealed = dict(payload, worker_token='fixture-secret')
        receipt = dict(owned_child_cleanup=True, returncode=1, timed_out=True)
        def timed_out(argv, seconds, data):
            clock.advance(220)
            error = TimeoutError('owned supervisor fixture')
            error.supervisor_result = receipt
            raise error
        with mock.patch.object(a.pilot.time, 'monotonic', clock), \
                mock.patch('phase_timing.signal.getitimer', return_value=(0, 0)), \
                mock.patch('phase_timing.signal.setitimer'), \
                mock.patch.object(a.pilot.launch_guard, 'seal', return_value=(sealed, 'fixture-hash')), \
                mock.patch.object(a.ct, 'isolated_argv', side_effect=lambda argv, env: argv), \
                mock.patch.object(a.pilot.supervisor, 'run_process', side_effect=timed_out) as run:
            value = a.supervise(payload, 219)
        self.assertTrue(value['owned_child_cleanup'])
        self.assertTrue(value['timed_out'])
        self.assertEqual(run.call_args.args[1], 219)
        self.assertEqual(run.call_args.args[2], sealed)

    def test_real_blocking_callback_is_interrupted_and_alarm_restored(self):
        original = signal.getsignal(signal.SIGALRM)
        with self.assertRaisesRegex(TimeoutError, 'pressure phase deadline'):
            with deadline_cap(time.monotonic() + .02):
                time.sleep(1)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0, 0))
        self.assertIs(signal.getsignal(signal.SIGALRM), original)

    def test_observer_thread_spans_and_unobserved_are_not_serial_gap_estimates(self):
        timer = Timings(self.directory)
        with timer.span('pre_observer'):
            timer.call('nfs', lambda: None)
        with self.assertRaises(RuntimeError):
            timer.call('candidate_pins', lambda: (_ for _ in ()).throw(RuntimeError('secret-not-logged')))
        timer.finish()
        value = a.ct.load(self.directory / 'phase-timings.json')
        self.assertEqual(set(value['inclusive_wall_seconds']), set(PHASES))
        self.assertIsNone(value['inclusive_wall_seconds']['observer_pss'])
        self.assertIsNone(value['historical_helper_durations_seconds'])
        self.assertFalse(value['additive_across_names_or_threads'])
        self.assertEqual(value['open_spans'], 0)
        self.assertNotIn('secret-not-logged', (self.directory / 'phase-timings.jsonl').read_text())


class SlowCallbacks(owned.fixtures.CPUOnly):
    def run_serial(self, *, aggregate=120, sample_delay=None, check_delay=3, target=400):
        clock, order = Clock(), []
        (self.directory / 'server.log').touch()
        sampler = self.sampling(self.directory)
        def sample(label):
            clock.advance(sample_delay(label) if sample_delay else .2)
            return sampler(label)
        def check():
            clock.advance(check_delay)
        def io_sample():
            clock.advance(.1)
            return {}
        class Client(owned.FourClient):
            def request(self, *args, **kwargs):
                clock.advance(8)
                return super().request(*args, **kwargs)
        limits = dict(a.p.LIMITS, all_requests_seconds=aggregate)
        with mock.patch.object(a.pilot.time, 'monotonic', clock), \
                mock.patch('phase_timing.signal.getitimer', return_value=(0, 0)), \
                mock.patch('phase_timing.signal.setitimer'), \
                mock.patch.object(a.ex.http, 'Client', side_effect=lambda: Client(self.directory, order)):
            result = a.serial_requests(self.directory, a.schedule(),
                dict(url='http://127.0.0.1:18369/v1/completions', model='qwen35-storage-demand-paper'),
                limits, target, 500, 600, check, sample, io_sample)
        return result, order, clock

    def test_twelve_slow_checks_and_eight_observers_charge_one_120s_phase(self):
        result, order, clock = self.run_serial()
        self.assertTrue(result['can_continue'], result['stop_reason'])
        self.assertEqual(order, list(a.NAMES))
        self.assertAlmostEqual(clock.now - 100, 4 * (8 + 3 * 3 + .6))
        self.assertEqual(result['absolute_aggregate_deadline'], 220)
        self.assertEqual(result['summary']['successful'], 4)
        phases = a.ct.load(self.directory / 'phase-timings.json')['inclusive_wall_seconds']
        for label in ('pre_admission_validation', 'pre_dispatch_validation', 'post_http_validation'):
            self.assertAlmostEqual(phases[label], 12)
        self.assertAlmostEqual(phases['http'], 32)
        self.assertAlmostEqual(phases['nfs'], 1.6)
        self.assertAlmostEqual(phases['pid_io'], .8)

    def test_same_cpu_fixture_60s_budget_censors_last_request_not_a_gpu_result(self):
        result, order, _ = self.run_serial(aggregate=60)
        self.assertEqual(order, list(a.NAMES))
        self.assertFalse(result['can_continue'])
        self.assertEqual(result['summary']['counts'], dict(success=3, failed=0, censored=1, unissued=0))
        self.assertEqual(result['absolute_aggregate_deadline'], 160)

    def test_preobserver_expiry_is_unissued_not_censored(self):
        result, order, _ = self.run_serial(sample_delay=lambda label: 121, check_delay=0)
        self.assertEqual(order, [])
        self.assertEqual(result['summary']['unissued'], 4)
        self.assertEqual(result['summary']['counts']['censored'], 0)
        self.assertFalse((self.directory / 'hello2/issued.json').exists())

    def test_postobserver_expiry_censors_issued_and_stops_next_admission(self):
        result, order, _ = self.run_serial(sample_delay=lambda label: 121 if label.endswith('-post') else 0,
                                         check_delay=0)
        self.assertEqual(order, ['hello2'])
        self.assertEqual(result['summary']['counts'], dict(success=0, failed=0, censored=1, unissued=3))
        self.assertTrue((self.directory / 'hello2/http-terminal.json').exists())
        self.assertFalse((self.directory / 'canary-qa/issued.json').exists())

    def test_earlier_target_deadline_wins_over_full_aggregate(self):
        result, order, _ = self.run_serial(target=105)
        self.assertEqual(order, [])
        self.assertFalse(result['can_continue'])
        self.assertEqual(result['summary']['unissued'], 4)


class CleanupTests(owned.fixtures.CPUOnly):
    prepare_worker = owned.WorkerIntegrationTests.prepare_worker
    invoke_point = owned.WorkerIntegrationTests.invoke_point

    def test_timeout_after_owned_launch_runs_maintained_stop_and_reap(self):
        args, candidate, journal, _ = self.prepare_worker()
        with mock.patch.object(a, 'http_request', side_effect=TimeoutError('CPU timeout fixture')), \
                mock.patch.object(a.ex, 'stop_capture', wraps=a.ex.stop_capture) as stop:
            code, order, folder = self.invoke_point('small', args, candidate, journal)
        self.assertEqual(code, 1)
        self.assertEqual(order, [])
        self.assertTrue((folder / 'profile-stop').is_file())
        self.assertTrue((folder / 'profile-server.json').is_file())
        self.assertTrue(all(call.args == (folder,) for call in stop.call_args_list))
        self.assertEqual(a.ex.recover_requests(folder, a.schedule())['counts'],
                         dict(success=0, failed=0, censored=1, unissued=3))
        phases = a.ct.load(folder / 'phase-timings.json')['inclusive_wall_seconds']
        self.assertIsNotNone(phases['shutdown'])
        self.assertIsNotNone(phases['library_maps'])
        self.assertIsNotNone(phases['candidate_pins'])

    def test_slow_real_worker_pin_callbacks_consume_phase_and_cleanup_unissued(self):
        args, candidate, journal, _ = self.prepare_worker()
        clock = Clock()
        with mock.patch.object(a.pilot.time, 'monotonic', clock), \
                mock.patch('phase_timing.signal.getitimer', return_value=(0, 0)), \
                mock.patch('phase_timing.signal.setitimer'), \
                mock.patch.object(a.ct, 'check_pins', side_effect=lambda pins: clock.advance(61)), \
                mock.patch.object(a.ex, 'stop_capture', wraps=a.ex.stop_capture) as stop:
            code, order, folder = self.invoke_point('small', args, candidate, journal)
        self.assertEqual(code, 1)
        self.assertEqual(order, [])
        self.assertEqual(a.ex.recover_requests(folder, a.schedule())['unissued'], 4)
        self.assertTrue((folder / 'profile-server.json').is_file())
        self.assertTrue(stop.called)
        self.assertTrue(all(call.args == (folder,) for call in stop.call_args_list))
        phases = a.ct.load(folder / 'phase-timings.json')['inclusive_wall_seconds']
        self.assertEqual(phases['workflow_pins'], 61)
        self.assertEqual(phases['candidate_pins'], 61)
        self.assertIsNotNone(phases['shutdown'])

    def test_callback_counts_and_all_named_worker_phases_without_hash_cache(self):
        args, candidate, journal, _ = self.prepare_worker()
        with mock.patch.object(a.ct, 'check_pins') as pins:
            code, order, folder = self.invoke_point('small', args, candidate, journal)
        self.assertEqual(code, 0)
        self.assertEqual(order, list(a.NAMES))
        # 3 per request + final retained check, each rechecks BOTH pin sets.
        self.assertEqual(pins.call_count, 2 * (3 * 4 + 1))
        phases = a.ct.load(folder / 'phase-timings.json')['inclusive_wall_seconds']
        for name in ('worker_contract', 'checkpoint_metadata', 'runtime_metadata', 'source_options',
                     'source_custody_checks', 'workflow_pins', 'candidate_pins', 'receipt_hashes',
                     'library_maps', 'nfs', 'pid_io', 'token_join', 'source_counters',
                     'pre_observer', 'post_observer', 'observer_finish', 'worker_finalization'):
            self.assertIsNotNone(phases[name], name)
        # CPU fixture never starts the observer thread. Unobserved != zero.
        self.assertIsNone(phases['observer_pss'])


if __name__ == '__main__':
    unittest.main()
