"""Finite CPU-only tests of the existing direct-child profiler wrapper."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location('profile_capture_under_test', Path(__file__).resolve().parents[1] / '_profile_capture.py')
capture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture)


class NaturalCompletionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def test_real_cpu_natural_child_uses_same_wrapper_and_one_spawn_observer(self):
        seen = []
        with mock.patch.dict(os.environ, CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none'):
            result = capture.run_target(self.directory,
                [sys.executable, '-S', '-B', '-c', 'import time; time.sleep(.05)'], 2, 1,
                natural_completion=True, on_spawn=lambda child: seen.append(child.pid))
        receipt = json.loads((self.directory / 'profile-server.json').read_text())
        self.assertEqual(result, 0)
        self.assertEqual(seen, [receipt['pid']])
        self.assertEqual(receipt['stop_reason'], 'natural_completion')
        self.assertEqual(receipt['returncode'], 0)
        self.assertNotIn('shutdown_timeout', receipt)

    def test_server_default_does_not_accept_natural_exit(self):
        child = mock.Mock(pid=42, returncode=0)
        child.poll.return_value = 0
        with mock.patch.object(capture.subprocess, 'Popen', return_value=child):
            self.assertEqual(capture.run_target(self.directory, ['cpu-fixture'], 2, 1), 1)
        child.terminate.assert_not_called()

    def test_natural_failure_and_stop_marker_are_not_success_or_retry(self):
        for code, marker in ((1, False), (0, True)):
            with self.subTest(code=code, marker=marker), tempfile.TemporaryDirectory() as folder:
                directory = Path(folder)
                if marker:
                    (directory / 'profile-stop').touch()
                child = mock.Mock(pid=42, returncode=code)
                child.poll.side_effect = [None, None] if marker else [code, code]
                with mock.patch.object(capture.subprocess, 'Popen', return_value=child) as spawn:
                    result = capture.run_target(directory, ['cpu-fixture'], 2, 1, natural_completion=True)
                self.assertEqual(result, 1)
                self.assertEqual(spawn.call_count, 1)
                self.assertEqual(child.terminate.call_count, int(marker))

    def test_observer_failure_does_not_add_descendant_or_group_cleanup(self):
        child = mock.Mock(pid=42, returncode=None)
        with mock.patch.object(capture.subprocess, 'Popen', return_value=child) as spawn:
            with self.assertRaisesRegex(ValueError, 'birth mismatch'):
                capture.run_target(self.directory, ['cpu-fixture'], 2, 1, natural_completion=True,
                    on_spawn=mock.Mock(side_effect=ValueError('birth mismatch')))
        self.assertEqual(spawn.call_count, 1)
        child.terminate.assert_not_called()
        child.wait.assert_not_called()  # Outer maintained subreaper owns failure drain.

    def test_natural_mode_requires_boolean_not_truthy_string(self):
        with mock.patch.object(capture.subprocess, 'Popen') as spawn, self.assertRaises(ValueError):
            capture.run_target(self.directory, ['cpu-fixture'], 2, 1, natural_completion='yes')
        spawn.assert_not_called()


@unittest.skipUnless(hasattr(os, 'WNOWAIT'), 'Linux direct-child terminal observation')
class PreReapTerminalTests(unittest.TestCase):
    setUp = NaturalCompletionTests.setUp
    def test_real_cpu_child_keeps_birth_anchor_until_ack_then_one_owner_wait(self):
        seen, waits = {}, []
        def identity(pid):
            fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
            return dict(pid=pid, birth_ticks=int(fields[19]), ppid=int(fields[1]), state=fields[0])
        def spawned(child):
            seen.update(child=child, identity=identity(child.pid))
            original = child.wait
            def wait(*args, **kwargs):
                self.assertTrue(seen['ack'])
                waits.append(child.pid)
                return original(*args, **kwargs)
            child.wait = wait
            child.poll = mock.Mock(side_effect=AssertionError('Observed path must not poll/reap'))
        def terminal(child, event, deadline, active):
            self.assertIs(child, seen['child'])
            self.assertIsNone(child.returncode)
            current = identity(child.pid)
            self.assertEqual(current['state'], 'Z')
            for key in ('pid', 'birth_ticks', 'ppid'):
                self.assertEqual(current[key], seen['identity'][key])
            self.assertEqual(event['pid'], child.pid)
            self.assertEqual(event['status'], 0)
            self.assertLess(event['observed_monotonic'], deadline)
            self.assertNotIn('last_running_monotonic', event)
            ack = threading.Event()
            def observer():
                self.assertEqual(identity(child.pid)['state'], 'Z')
                self.assertEqual(os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT).si_pid, child.pid)
                ack.set()
            thread = threading.Thread(target=observer)
            thread.start(); thread.join(timeout=1)
            self.assertFalse(thread.is_alive())
            self.assertTrue(ack.is_set())
            self.assertEqual(identity(child.pid)['state'], 'Z')
            self.assertEqual(waits, [])
            seen['ack'] = True
            active()
        with mock.patch.dict(os.environ, CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none'), \
                mock.patch.object(capture.os, 'kill', side_effect=AssertionError('No signals')), \
                mock.patch.object(capture.os, 'killpg', side_effect=AssertionError('No group signals')):
            self.assertEqual(capture.run_target(self.directory,
                [sys.executable, '-S', '-B', '-c', 'import time; time.sleep(.05)'], 2, 1,
                natural_completion=True, on_spawn=spawned, on_terminal=terminal), 0)
        self.assertEqual(waits, [seen['child'].pid])
        seen['child'].poll.assert_not_called()
        self.assertFalse(Path(f"/proc/{seen['child'].pid}").exists())

    def test_actual_zombie_with_mock_delayed_waitid_does_not_claim_liveness(self):
        # Real Linux child/proc state; delay waitid visibility deterministically.
        # This does not claim to recreate pthread_exit with a remaining thread.
        original_waitid = capture.os.waitid
        withheld, waits, seen = [], [], {}
        def state(pid):
            return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
        def delayed_waitid(*args):
            event = original_waitid(*args)
            if event is not None and len(withheld) < 2:
                self.assertEqual(state(event.si_pid), 'Z')
                withheld.append(capture.time.monotonic())
                return None
            return event
        def spawned(child):
            seen['child'] = child
            original = child.wait
            def wait(*args, **kwargs):
                waits.append(child.pid)
                return original(*args, **kwargs)
            child.wait = wait
            child.poll = mock.Mock(side_effect=AssertionError('No reaping poll'))
        def terminal(child, event, deadline, active):
            self.assertEqual(len(withheld), 2)
            self.assertEqual(state(child.pid), 'Z')
            self.assertLessEqual(withheld[-1], event['observed_monotonic'])
            self.assertNotIn('last_running_monotonic', event)
            self.assertEqual(set(event), {'pid', 'code', 'status', 'started_monotonic', 'observed_monotonic'})
            active()
        with mock.patch.dict(os.environ, CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none'), \
                mock.patch.object(capture.os, 'waitid', side_effect=delayed_waitid), \
                mock.patch.object(capture.os, 'kill', side_effect=AssertionError('No signals')), \
                mock.patch.object(capture.os, 'killpg', side_effect=AssertionError('No group signals')):
            result = capture.run_target(self.directory,
                [sys.executable, '-S', '-B', '-c', 'pass'], 2, 1,
                natural_completion=True, on_spawn=spawned, on_terminal=terminal)
        self.assertEqual(result, 0)
        self.assertEqual(waits, [seen['child'].pid])
        seen['child'].poll.assert_not_called()

    def observed(self, hook, *, spawn=None, status=0):
        child = mock.Mock(pid=42, returncode=None)
        terminal = SimpleNamespace(si_pid=42, si_code=os.CLD_EXITED, si_status=status)
        def wait(**kwargs):
            child.returncode = status
            return status
        child.wait.side_effect = wait
        with mock.patch.object(capture.subprocess, 'Popen', return_value=child), \
                mock.patch.object(capture.os, 'waitid', return_value=terminal) as waitid:
            try:
                result = capture.run_target(self.directory, ['cpu-fixture'], 2, 1,
                    natural_completion=True, on_spawn=spawn, on_terminal=hook)
                return result, child, waitid
            finally:
                self.child = child

    def test_ack_error_missing_cancel_and_late_hook_never_wait_or_succeed(self):
        for action in ('missing', 'cancel', 'signal', 'late'):
            with self.subTest(action=action), tempfile.TemporaryDirectory() as folder:
                self.directory = Path(folder)
                clock = [100.]
                def hook(child, event, deadline, active):
                    if action == 'missing': raise TimeoutError('Missing observer acknowledgement')
                    if action == 'cancel': (self.directory / 'profile-stop').touch()
                    if action == 'signal': capture.signal.getsignal(capture.signal.SIGTERM)(15, None)
                    if action == 'late': clock[0] = deadline + 1
                with mock.patch.object(capture.time, 'monotonic', side_effect=lambda: clock[0]), \
                        self.assertRaises((RuntimeError, TimeoutError)):
                    self.observed(hook)
                self.child.poll.assert_not_called()
                self.child.wait.assert_not_called()
                self.child.terminate.assert_not_called()
                receipt = json.loads((self.directory / 'profile-server.json').read_text())
                self.assertIsNone(receipt['returncode'])
                self.assertNotEqual(receipt['stop_reason'], 'natural_completion')

    def test_startup_is_inside_deadline_and_hook_cannot_reap_or_change_terminal(self):
        clock = [100.]
        def spawn(child): clock[0] = 103.
        with mock.patch.object(capture.time, 'monotonic', side_effect=lambda: clock[0]), \
                self.assertRaises(TimeoutError):
            self.observed(mock.Mock(), spawn=spawn)
        self.child.wait.assert_not_called()
        def bad_hook(child, *args): child.returncode = 0
        with self.assertRaisesRegex(RuntimeError, 'custody'):
            self.observed(bad_hook)
        self.child.wait.assert_not_called()

    def test_nonzero_terminal_is_not_success_and_hook_is_opt_in(self):
        result, child, waitid = self.observed(mock.Mock(), status=1)
        self.assertEqual(result, 1)
        child.wait.assert_called_once(); child.poll.assert_not_called()
        self.assertEqual(waitid.call_count, 2)
        with mock.patch.object(capture.subprocess, 'Popen') as spawn, self.assertRaises(ValueError):
            capture.run_target(self.directory, ['cpu-fixture'], 2, 1, on_terminal=mock.Mock())
        spawn.assert_not_called()


    def test_observed_lifetime_must_be_finite_and_bounded_before_spawn(self):
        for lifetime in (0, -1, float('nan'), float('inf'), 1561):
            with self.subTest(lifetime=lifetime), mock.patch.object(capture.subprocess, 'Popen') as spawn, \
                    self.assertRaises(ValueError):
                capture.run_target(self.directory, ['cpu-fixture'], lifetime, 1,
                    natural_completion=True, on_terminal=mock.Mock())
            spawn.assert_not_called()


@unittest.skipUnless(hasattr(os, "WNOWAIT") and Path("/proc/self/stat").exists(),
                     "Linux live direct-child planned stop")
class PlannedStopTests(unittest.TestCase):
    setUp = NaturalCompletionTests.setUp

    def ack(self, child, event, deadline, active):
        active()
        return dict(stop_event=event, observer_stopped=True)

    def planned(self, hook=None, *, status=-15, spawn=None, absolute_deadline=None):
        child = mock.Mock(pid=42, returncode=None)
        self.child = child
        root = dict(pid=42, birth_ticks=17, ppid=os.getpid())
        def spawned(child):
            if spawn is not None:
                spawn(child)
            (self.directory / "profile-stop").touch()
        def wait(**kwargs):
            child.returncode = status
            return status
        child.wait.side_effect = wait
        def live(child):
            if child.returncode is not None:
                raise RuntimeError("Lost unreaped direct child")
            return root
        with mock.patch.object(capture.subprocess, "Popen", return_value=child), \
                mock.patch.object(capture, "live_child_identity", side_effect=live):
            result = capture.run_target(self.directory, ["cpu-fixture"], 2, 1,
                on_spawn=spawned, on_stop=hook or self.ack, absolute_deadline=absolute_deadline)
        return result, json.loads((self.directory / "profile-server.json").read_text())

    def assert_unsafe_not_stopped(self, returncode=None):
        self.child.poll.assert_not_called()
        self.child.wait.assert_not_called()
        self.child.terminate.assert_not_called()
        receipt = json.loads((self.directory / "profile-server.json").read_text())
        self.assertFalse(receipt["planned_stop_validated"])
        self.assertEqual(receipt["returncode"], returncode)
        self.assertNotEqual(receipt["stop_reason"], "planned_stop")

    def test_real_sigterm_is_only_after_monitor_ack_and_preserves_actual_code(self):
        seen, calls = {}, []
        def spawned(child):
            seen["child"] = child
            original_poll, original_wait, original_terminate = child.poll, child.wait, child.terminate
            def poll():
                self.assertTrue(seen.get("ack"), "No pre-ACK poll/reap")
                return original_poll()
            def wait(*args, **kwargs):
                self.assertTrue(seen.get("ack"))
                calls.append("wait")
                return original_wait(*args, **kwargs)
            def terminate():
                self.assertTrue(seen.get("ack"))
                calls.append("SIGTERM")
                return original_terminate()
            child.poll, child.wait, child.terminate = poll, wait, terminate
            (self.directory / "profile-stop").touch()
        def stop(child, event, deadline, active):
            self.assertIs(child, seen["child"])
            self.assertIsNone(child.returncode)
            self.assertEqual(calls, [])
            self.assertEqual({k: event[k] for k in ("pid", "birth_ticks", "ppid")},
                             capture.live_child_identity(child))
            sampled, acknowledged = threading.Event(), threading.Event()
            def observer():
                sampled.set()
                acknowledged.wait(timeout=1)
            thread = threading.Thread(target=observer)
            thread.start()
            try:
                self.assertTrue(sampled.wait(timeout=1))
                active()
            finally:
                acknowledged.set()
                thread.join(timeout=1)
            self.assertFalse(thread.is_alive())
            self.assertEqual(calls, [])
            seen["ack"] = True
            return dict(stop_event=event, observer_stopped=True)
        with mock.patch.dict(os.environ, CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="none"), \
                mock.patch.object(capture.os, "killpg", side_effect=AssertionError("No group signals")):
            result = capture.run_target(self.directory,
                [sys.executable, "-S", "-B", "-c", "import time; time.sleep(10)"], 3, 2,
                on_spawn=spawned, on_stop=stop)
        receipt = json.loads((self.directory / "profile-server.json").read_text())
        self.assertEqual(result, 0)
        self.assertEqual(receipt["returncode"], -15)
        self.assertEqual(receipt["stop_reason"], "planned_stop")
        self.assertTrue(receipt["planned_stop_validated"])
        self.assertEqual(calls, ["SIGTERM", "wait"])
        self.assertLessEqual(receipt["observer_acknowledged_monotonic"], receipt["shutdown_started_monotonic"])
        self.assertEqual(receipt["stop_ack"]["stop_event"], receipt["stop_event"])
        self.assertFalse(Path(f"/proc/{seen["child"].pid}").exists())

    def test_real_pre_signal_exit_seen_by_original_terminate_is_not_planned_stop(self):
        seen = {}
        release = self.directory / "exit-before-signal"
        def spawned(child):
            seen["child"] = child
            original_terminate, original_poll, original_wait = child.terminate, child.poll, child.wait
            def cleanup():
                child.poll, child.wait = original_poll, original_wait
                if original_poll() is None:
                    original_terminate()
                original_wait(timeout=1)
            self.addCleanup(cleanup)
            def delayed_terminate():
                self.assertTrue(seen.get("ack"))
                self.assertIsNone(child.returncode)
                release.touch()
                limit = capture.time.monotonic() + 1
                while os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
                    self.assertLess(capture.time.monotonic(), limit)
                    capture.time.sleep(.005)
                # Exercise the original POSIX terminate: its poll consumes the
                # terminal status, and it must never reach os.kill in this case.
                with mock.patch.object(capture.os, "kill", side_effect=AssertionError("No signal after terminal poll")):
                    original_terminate()
                self.assertEqual(child.returncode, 0)
            child.terminate = delayed_terminate
            child.wait = mock.Mock(wraps=original_wait)
            (self.directory / "profile-stop").touch()
        def stop(child, event, deadline, active):
            active()
            seen["ack"] = True
            return dict(stop_event=event, observer_stopped=True)
        code = ("from pathlib import Path; import time; "
                "p=Path(" + repr(str(release)) + "); end=time.monotonic()+5\n"
                "while not p.exists() and time.monotonic()<end: time.sleep(.005)")
        with mock.patch.object(capture.os, "killpg", side_effect=AssertionError("No group signals")), \
                self.assertRaisesRegex(RuntimeError, "terminal before SIGTERM"):
            capture.run_target(self.directory, [sys.executable, "-S", "-B", "-c", code],
                4, 3, on_spawn=spawned, on_stop=stop)
        child = seen["child"]
        child.wait.assert_not_called()
        receipt = json.loads((self.directory / "profile-server.json").read_text())
        self.assertEqual(receipt["returncode"], 0)
        self.assertEqual(receipt["stop_reason"], "pre_signal_terminal")
        self.assertFalse(receipt["planned_stop_validated"])
        self.assertTrue(receipt["stop_ack"]["observer_stopped"])
        self.assertEqual(receipt["stop_ack"]["stop_event"], receipt["stop_event"])

    def test_real_sigterm_handler_zero_exit_is_accepted_after_authoritative_wait(self):
        ready = self.directory / "handler-ready"
        seen = {}
        def spawned(child):
            seen["child"] = child
            original_poll, original_wait, original_terminate = child.poll, child.wait, child.terminate
            def cleanup():
                if original_poll() is None:
                    original_terminate()
                original_wait(timeout=1)
            self.addCleanup(cleanup)
            limit = capture.time.monotonic() + 1
            while not ready.exists():
                self.assertLess(capture.time.monotonic(), limit)
                capture.time.sleep(.005)
            (self.directory / "profile-stop").touch()
        code = ("import signal,sys,time; from pathlib import Path; "
                "signal.signal(signal.SIGTERM, lambda *args: sys.exit(0)); "
                "Path(" + repr(str(ready)) + ").touch(); time.sleep(5)")
        with mock.patch.object(capture.os, "kill", wraps=capture.os.kill) as kill, \
                mock.patch.object(capture.os, "killpg", side_effect=AssertionError("No group signals")):
            result = capture.run_target(self.directory, [sys.executable, "-S", "-B", "-c", code],
                4, 3, on_spawn=spawned, on_stop=self.ack)
        kill.assert_called_once_with(seen["child"].pid, capture.signal.SIGTERM)
        receipt = json.loads((self.directory / "profile-server.json").read_text())
        self.assertEqual(result, 0)
        self.assertEqual(receipt["returncode"], 0)
        self.assertEqual(receipt["stop_reason"], "planned_stop")
        self.assertTrue(receipt["planned_stop_validated"])

    def test_absolute_deadline_rejects_invalid_values_before_spawn(self):
        for value in (False, True, "101", 0, -1, float("nan"), float("inf"), 10 ** 1000):
            with self.subTest(value_type=type(value).__name__), \
                    mock.patch.object(capture.subprocess, "Popen") as spawn, self.assertRaises(ValueError):
                capture.run_target(self.directory, ["cpu-fixture"], 2, 1,
                    on_stop=self.ack, absolute_deadline=value)
            spawn.assert_not_called()

    def test_real_clock_expired_absolute_deadline_rejects_entry_without_spawn(self):
        sealed_deadline = capture.time.monotonic() - 1
        with mock.patch.object(capture.subprocess, "Popen") as spawn, self.assertRaises(TimeoutError):
            capture.run_target(self.directory, ["cpu-fixture"], 2, 1,
                on_stop=self.ack, absolute_deadline=sealed_deadline)
        spawn.assert_not_called()
        receipt = json.loads((self.directory / "profile-server.json").read_text())
        self.assertEqual(receipt["absolute_deadline"], sealed_deadline)
        self.assertEqual(receipt["stop_reason"], "absolute_deadline_expired")
        self.assertNotIn("pid", receipt)

    def test_absolute_deadline_survives_slow_validation_and_is_rechecked_at_spawn(self):
        for delay in ("caller_recheck", "validation", "report_setup"):
            with self.subTest(delay=delay), tempfile.TemporaryDirectory() as folder:
                self.directory = Path(folder)
                clock = [100.]
                sealed_deadline = 100.5
                def validate_marker():
                    if delay == "validation":
                        clock[0] = 101.
                    return False
                def report_time():
                    if delay == "report_setup":
                        clock[0] = 101.
                    return 1
                # Model input/hash revalidation after reading the immutable cap,
                # before entering run_target: the new relative lifetime cannot renew it.
                if delay == "caller_recheck":
                    clock[0] = 101.
                with mock.patch.object(capture.time, "monotonic", side_effect=lambda: clock[0]), \
                        mock.patch.object(capture.time, "time_ns", side_effect=report_time), \
                        mock.patch.object(capture.Path, "exists", side_effect=validate_marker), \
                        mock.patch.object(capture.subprocess, "Popen") as spawn, self.assertRaises(TimeoutError):
                    capture.run_target(self.directory, ["cpu-fixture"], 2, 1,
                        on_stop=self.ack, absolute_deadline=sealed_deadline)
                spawn.assert_not_called()
                receipt = json.loads((self.directory / "profile-server.json").read_text())
                self.assertEqual(receipt["lifetime_deadline"], sealed_deadline)
                self.assertEqual(receipt["stop_reason"], "absolute_deadline_expired")

    def test_absolute_cap_never_extends_relative_lifetime_and_accepts_int(self):
        clock = [100.]
        def stop(child, event, deadline, active):
            self.assertEqual(deadline, 102.)
            return self.ack(child, event, deadline, active)
        with mock.patch.object(capture.time, "monotonic", side_effect=lambda: clock[0]):
            result, receipt = self.planned(stop, absolute_deadline=200,
                spawn=lambda child: clock.__setitem__(0, 101.5))
        self.assertEqual(result, 0)
        self.child.wait.assert_called_once_with(timeout=.5)
        self.assertEqual(receipt["lifetime_deadline"], 102.)

    def test_absolute_cap_nonobserved_mode_rejects_bad_relative_budgets_and_late_exit(self):
        for lifetime, shutdown in ((float("nan"), 1), (float("inf"), 1), (0, 1), (1561, 1),
                                   (2, float("nan")), (2, float("inf")), (2, 0), (2, 46)):
            with self.subTest(lifetime=lifetime, shutdown=shutdown), \
                    mock.patch.object(capture.subprocess, "Popen") as spawn, self.assertRaises(ValueError):
                capture.run_target(self.directory, ["cpu-fixture"], lifetime, shutdown,
                    absolute_deadline=100.5)
            spawn.assert_not_called()
        clock = [100.]
        child = mock.Mock(pid=42, returncode=0)
        def poll():
            clock[0] = 101.
            return 0
        child.poll.side_effect = poll
        with mock.patch.object(capture.time, "monotonic", side_effect=lambda: clock[0]), \
                mock.patch.object(capture.subprocess, "Popen", return_value=child):
            result = capture.run_target(self.directory, ["cpu-fixture"], 2, 1,
                natural_completion=True, absolute_deadline=100.5)
        self.assertEqual(result, 1)
        self.assertEqual(json.loads((self.directory / "profile-server.json").read_text())["stop_reason"],
                         "absolute_deadline_expired")
        child.terminate.assert_not_called()
        child.wait.assert_not_called()

    def test_absolute_cap_includes_stop_ack_and_authoritative_wait(self):
        clock = [100.]
        def stop(child, event, deadline, active):
            self.assertEqual(deadline, 100.5)
            self.assertEqual(event["absolute_deadline"], 100.5)
            clock[0] = 100.25
            return self.ack(child, event, deadline, active)
        with mock.patch.object(capture.time, "monotonic", side_effect=lambda: clock[0]):
            result, receipt = self.planned(stop, absolute_deadline=100.5)
        self.assertEqual(result, 0)
        self.child.wait.assert_called_once_with(timeout=.25)
        self.assertEqual(receipt["absolute_deadline"], 100.5)
        with tempfile.TemporaryDirectory() as folder:
            self.directory = Path(folder)
            clock[0] = 100.
            def late(child, event, deadline, active):
                clock[0] = deadline
                return dict(stop_event=event, observer_stopped=True)
            with mock.patch.object(capture.time, "monotonic", side_effect=lambda: clock[0]), \
                    self.assertRaises(TimeoutError):
                self.planned(late, absolute_deadline=100.5)
            self.assert_unsafe_not_stopped()

    def test_explicit_planned_stop_only_accepts_zero_or_sigterm_not_other_failures(self):
        for status in (0, -15, -9, 1):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as folder:
                self.directory = Path(folder)
                result, receipt = self.planned(status=status)
                self.assertEqual(result, int(status not in (0, -15)))
                self.assertEqual(receipt["returncode"], status)
                self.assertEqual(receipt["planned_stop_validated"], status in (0, -15))
                self.child.terminate.assert_called_once()
                self.child.wait.assert_called_once()

    def test_missing_stale_foreign_or_unhealthy_ack_never_signals_or_reaps(self):
        for action in ("none", "stale", "pid", "birth_ticks", "ppid", "false", "truthy", "sticky",
                       "role_foreign", "map_read_failure", "request_failure"):
            with self.subTest(action=action), tempfile.TemporaryDirectory() as folder:
                self.directory = Path(folder)
                def stop(child, event, deadline, active):
                    if action == "none":
                        return None
                    if action in ("role_foreign", "map_read_failure", "request_failure"):
                        raise RuntimeError(action)
                    ack = dict(stop_event=dict(event), observer_stopped=True)
                    if action == "stale":
                        ack["stop_event"]["stop_id"] = "old-stop"
                    if action in ("pid", "birth_ticks", "ppid"):
                        ack["stop_event"][action] += 1
                    if action in ("false", "truthy"):
                        ack["observer_stopped"] = False if action == "false" else 1
                    if action == "sticky":
                        ack["sticky_error"] = True
                    return ack
                with self.assertRaises(RuntimeError):
                    self.planned(stop)
                self.assert_unsafe_not_stopped()

    def test_cancel_late_ack_or_callback_reap_fails_without_unsafe_stop(self):
        for action in ("signal", "late", "reaped"):
            with self.subTest(action=action), tempfile.TemporaryDirectory() as folder:
                self.directory = Path(folder)
                clock = [100.]
                def stop(child, event, deadline, active):
                    if action == "signal":
                        capture.signal.getsignal(capture.signal.SIGTERM)(15, None)
                    if action == "late":
                        clock[0] = deadline
                    if action == "reaped":
                        child.returncode = 0
                    return self.ack(child, event, deadline, active)
                with mock.patch.object(capture.time, "monotonic", side_effect=lambda: clock[0]), \
                        self.assertRaises((RuntimeError, TimeoutError)):
                    self.planned(stop)
                self.assert_unsafe_not_stopped(returncode=0 if action == "reaped" else None)

    def test_startup_and_shutdown_share_original_absolute_budget(self):
        clock = [100.]
        with mock.patch.object(capture.time, "monotonic", side_effect=lambda: clock[0]), \
                self.assertRaises(TimeoutError):
            self.planned(spawn=lambda child: clock.__setitem__(0, 103.))
        self.assert_unsafe_not_stopped()
        with tempfile.TemporaryDirectory() as folder:
            self.directory = Path(folder)
            clock[0] = 100.
            def stop(child, event, deadline, active):
                self.assertEqual(deadline, 101.)
                clock[0] = 100.75
                return self.ack(child, event, deadline, active)
            with mock.patch.object(capture.time, "monotonic", side_effect=lambda: clock[0]):
                self.planned(stop)
            self.child.wait.assert_called_once_with(timeout=.25)
        with tempfile.TemporaryDirectory() as folder:
            self.directory = Path(folder)
            clock[0] = 100.
            def stop(child, event, deadline, active):
                self.assertEqual(deadline, 102.)
                clock[0] = 101.75
                return self.ack(child, event, deadline, active)
            with mock.patch.object(capture.time, "monotonic", side_effect=lambda: clock[0]):
                self.planned(stop, spawn=lambda child: clock.__setitem__(0, 101.5))
            self.child.wait.assert_called_once_with(timeout=.25)

    def test_identity_replacement_and_early_terminal_never_reach_hook_or_signal(self):
        root = dict(pid=42, birth_ticks=17, ppid=os.getpid())
        for bad in (dict(root, birth_ticks=18), dict(root, ppid=1),
                    FileNotFoundError("missing authenticated proc"), RuntimeError("terminal child")):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as folder:
                self.directory = Path(folder)
                child = mock.Mock(pid=42, returncode=None)
                hook = mock.Mock()
                with mock.patch.object(capture.subprocess, "Popen", return_value=child), \
                        mock.patch.object(capture, "live_child_identity", side_effect=[root, bad]), \
                        self.assertRaises((RuntimeError, FileNotFoundError)):
                    capture.run_target(self.directory, ["cpu-fixture"], 2, 1,
                        on_spawn=lambda child: (self.directory / "profile-stop").touch(), on_stop=hook)
                hook.assert_not_called()
                child.terminate.assert_not_called()
                child.wait.assert_not_called()
                child.poll.assert_not_called()

    def test_live_identity_rejects_terminal_zombie_wrong_parent_or_missing_proc(self):
        child = mock.Mock(pid=42, returncode=None)
        with mock.patch.object(capture.os, "waitid", return_value=object()), \
                mock.patch.object(capture.Path, "read_text") as read, self.assertRaises(RuntimeError):
            capture.live_child_identity(child)
        read.assert_not_called()
        for state, parent in (("Z", os.getpid()), ("X", os.getpid()), ("S", 1)):
            stat = "42 (fixture) " + " ".join([state, str(parent), *(["0"] * 17), "99"])
            with mock.patch.object(capture.os, "waitid", return_value=None), \
                    mock.patch.object(capture.Path, "read_text", return_value=stat), self.assertRaises(RuntimeError):
                capture.live_child_identity(child)
        with mock.patch.object(capture.os, "waitid", return_value=None), \
                mock.patch.object(capture.Path, "read_text", side_effect=FileNotFoundError), \
                self.assertRaises(FileNotFoundError):
            capture.live_child_identity(child)

    def test_identity_read_cannot_consume_deadline_and_then_signal(self):
        child = mock.Mock(pid=42, returncode=None)
        clock = [100.]
        root = dict(pid=42, birth_ticks=17, ppid=os.getpid())
        reads = []
        def live(child):
            reads.append(child.pid)
            if len(reads) > 1:
                clock[0] = 103.
            return root
        hook = mock.Mock()
        with mock.patch.object(capture.time, "monotonic", side_effect=lambda: clock[0]), \
                mock.patch.object(capture.subprocess, "Popen", return_value=child), \
                mock.patch.object(capture, "live_child_identity", side_effect=live), \
                self.assertRaises(TimeoutError):
            capture.run_target(self.directory, ["cpu-fixture"], 2, 1,
                on_spawn=lambda child: (self.directory / "profile-stop").touch(), on_stop=hook)
        hook.assert_not_called()
        child.terminate.assert_not_called()
        child.wait.assert_not_called()

    def test_shutdown_timeout_remains_failure_and_keeps_outer_owner_drain(self):
        def stop(child, event, deadline, active):
            child.wait.side_effect = capture.subprocess.TimeoutExpired("cpu-fixture", 1)
            return self.ack(child, event, deadline, active)
        with self.assertRaises(capture.subprocess.TimeoutExpired):
            self.planned(stop)
        self.child.terminate.assert_called_once()
        self.child.wait.assert_called_once()
        self.child.kill.assert_not_called()
        receipt = json.loads((self.directory / "profile-server.json").read_text())
        self.assertTrue(receipt["shutdown_timeout"])
        self.assertFalse(receipt["planned_stop_validated"])
        self.assertIsNone(receipt["returncode"])

    def test_default_marker_does_not_normalize_sigterm_without_observer_hook(self):
        child = mock.Mock(pid=42, returncode=None)
        child.poll.return_value = None
        def wait(**kwargs):
            child.returncode = -15
            return -15
        child.wait.side_effect = wait
        with mock.patch.object(capture.subprocess, "Popen", return_value=child):
            result = capture.run_target(self.directory, ["cpu-fixture"], 2, 1,
                on_spawn=lambda child: (self.directory / "profile-stop").touch())
        self.assertEqual(result, 1)
        receipt = json.loads((self.directory / "profile-server.json").read_text())
        self.assertEqual(receipt["returncode"], -15)
        self.assertEqual(receipt["stop_reason"], "matrix_worker_stop_marker")
        self.assertNotIn("planned_stop_validated", receipt)

    def test_real_planned_stop_or_failures_preserve_root_status_and_owner_drain(self):
        from test_vllm_validation import harness
        for action in ("ok", "missing", "stale", "cancel", "sticky", "map_failure", "early_exit"):
            with self.subTest(action=action), tempfile.TemporaryDirectory() as folder:
                directory = Path(folder)
                lines = ["import sys, signal", "from pathlib import Path",
                    "sys.path.insert(0, " + repr(str(Path(capture.__file__).parent)) + ")",
                    "import _profile_capture as c", "directory = Path(" + repr(folder) + ")",
                    "action = " + repr(action),
                    "def spawned(child):",
                    "    if action != \"early_exit\": (directory / \"profile-stop\").touch()",
                    "def stop(child, event, deadline, active):",
                    "    if action == \"cancel\": signal.getsignal(signal.SIGTERM)(15, None); active()",
                    "    if action == \"sticky\": raise RuntimeError(action)",
                    "    if action == \"map_failure\": (directory / \"missing-runtime-map\").read_bytes()",
                    "    if action == \"ok\": active(); return dict(stop_event=event, observer_stopped=True)",
                    "    if action == \"stale\": return dict(stop_event=dict(event, stop_id=\"old-stop\"), observer_stopped=True)",
                    "    return None",
                    "command = [sys.executable, \"-S\", \"-B\", \"-c\", \"import time; time.sleep(.05 if \" + repr(action == \"early_exit\") + \" else 30)\"]",
                    "sys.exit(c.run_target(directory, command, 2, 1, on_spawn=spawned, on_stop=stop))"]
                result = harness.run_process([sys.executable, "-S", "-B", "-c", "\n".join(lines)], 5)
                self.assertEqual(result["returncode"], 0 if action == "ok" else 1, result)
                self.assertFalse(result["timed_out"], result)
                self.assertEqual(result["descendant_custody"], "proven_drained", result)
                self.assertTrue(result["custody"]["echild"], result)
                self.assertTrue(all(row["reaped"] for row in result["custody"]["entries"]))
                receipt = json.loads((directory / "profile-server.json").read_text())
                self.assertEqual(receipt["planned_stop_validated"], action == "ok")
                self.assertEqual(receipt["returncode"], -15 if action == "ok" else None)
                if action != "ok":
                    self.assertNotIn("shutdown_started_monotonic", receipt)

    def test_opt_in_rejects_stale_marker_invalid_budget_and_conflicting_modes_before_spawn(self):
        variants = [dict(natural_completion=True), dict(on_terminal=mock.Mock()), dict(on_stop="yes")]
        variants += [dict(lifetime_seconds=v) for v in (0, float("nan"), float("inf"), 1561)]
        variants += [dict(shutdown_seconds=v) for v in (0, float("nan"), float("inf"), 46)]
        for extra in variants:
            args = dict(lifetime_seconds=2, shutdown_seconds=1, on_stop=self.ack)
            args.update(extra)
            with self.subTest(extra=extra), mock.patch.object(capture.subprocess, "Popen") as spawn, \
                    self.assertRaises(ValueError):
                capture.run_target(self.directory, ["cpu-fixture"], **args)
            spawn.assert_not_called()
        (self.directory / "profile-stop").touch()
        with mock.patch.object(capture.subprocess, "Popen") as spawn, self.assertRaises(ValueError):
            capture.run_target(self.directory, ["cpu-fixture"], 2, 1, on_stop=self.ack)
        spawn.assert_not_called()


if __name__ == '__main__':
    unittest.main()
