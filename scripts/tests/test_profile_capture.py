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


if __name__ == '__main__':
    unittest.main()
