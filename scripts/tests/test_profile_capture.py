"""Finite CPU-only tests of the existing direct-child profiler wrapper."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
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


if __name__ == '__main__':
    unittest.main()
