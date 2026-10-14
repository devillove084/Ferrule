"""No-GPU worker/profile-target integration; fixtures are not run evidence."""
from contextlib import ExitStack, nullcontext
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

os.environ.update(CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none', PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
with mock.patch('subprocess.Popen', side_effect=AssertionError('No real process launch allowed')), \
        mock.patch('subprocess.run', side_effect=AssertionError('No command or GPU query allowed')):
    import run_middle as middle


class ProfileTargetIntegration(unittest.TestCase):
    def test_worker_run_target_reaches_mocked_popen_with_devnull(self):
        with ExitStack() as stack:
            directory = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            (directory / 'baseline.json').write_text('{}\n')
            (directory / 'current-runtime-pins.json').write_text('{"pins": {}}\n')
            case = dict(argv=['mock-target', 'serve', 'mock-model'],
                binary='mock-target', binary_sha256='mock-sha',
                environment=dict(CUDA_VISIBLE_DEVICES='0', NVIDIA_VISIBLE_DEVICES='0'))
            payload = dict(directory=str(directory), uuid='mock-uuid',
                finalized_prepared='mock-prepared', finalized_prepared_sha256='mock-sha',
                baseline={}, baseline_sha256='mock-sha', runtime_pins_sha256='mock-sha',
                owner_lock={}, coordinator=dict(pid=1), config={}, owner_note='CPU fixture only')
            child = mock.Mock(pid=4242, returncode=None)
            child.poll.side_effect = lambda: child.returncode

            def wait(timeout):
                child.returncode = 0
                return 0

            child.wait.side_effect = wait
            monitor = mock.Mock(root=dict(pid=child.pid))
            monitor.finish.return_value = dict(status='observed_within_targets')

            def patch(obj, name, **kwargs):
                return stack.enter_context(mock.patch.object(obj, name, **kwargs))

            forbidden = AssertionError('No real process, GPU query, lock or HTTP allowed')
            patch(middle.subprocess, 'run', side_effect=forbidden)
            popen = patch(middle.subprocess, 'Popen', return_value=child)
            patch(middle.base.supervisor, 'run_process', side_effect=forbidden)
            patch(middle.ct, 'exclusive_owner_lock', side_effect=forbidden)
            patch(middle.observer, 'all_gpu_sample', side_effect=forbidden)
            patch(middle, 'one_request', side_effect=forbidden)
            patch(middle, 'map_snapshot', side_effect=forbidden)
            patch(middle.ct, 'require_cpu_python')
            patch(middle.guard, 'authenticate', return_value={})
            patch(middle, 'binding')
            patch(middle, 'validate_finalized')
            patch(middle, 'frozen', return_value=(case, []))
            patch(middle.ct, 'validate_owner_lock')
            patch(middle.ct, 'check_pins')
            patch(middle.os, 'getpgrp', return_value=os.getpid())
            patch(middle.ct, 'sha', return_value='mock-sha')
            patch(middle.ct, 'validate_staged_binary', return_value='mock-target')
            pinned = patch(middle.ct, 'pinned_executable', return_value=nullcontext(17))
            patch(middle.observer, 'idle', return_value={})
            patch(middle, 'strict_idle')
            patch(middle.base, 'require_free_port')
            patch(middle, 'schedule', return_value=[])
            patch(middle.observer, 'Monitor', return_value=monitor)
            ready = patch(middle.http, 'wait_ready',
                side_effect=ValueError('Mock startup stops before HTTP or GPU access'))
            patch(middle.sys, 'stdin', new=io.StringIO(json.dumps(payload)))

            result = middle.worker('mock-seal')
            report = json.loads((directory / 'case.json').read_text())
            self.assertNotIn('target_error', report)
            popen.assert_called_once()
            self.assertEqual(popen.call_args.kwargs['stdin'], middle.subprocess.DEVNULL)
            self.assertEqual(popen.call_args.kwargs['pass_fds'], (17,))
            pinned.assert_called_once_with('mock-target', 'mock-sha')
            ready.assert_called_once()
            self.assertEqual(report['server_pid'], child.pid)
            self.assertEqual(report['target_returncode'], 0)
            self.assertEqual(report['target']['pid'], child.pid)
            self.assertEqual(report['target']['stop_reason'], 'matrix_worker_stop_marker')
            self.assertEqual(report['target']['returncode'], 0)
            child.terminate.assert_called_once_with()
            child.wait.assert_called_once_with(timeout=45)
            monitor.start.assert_called_once_with()
            monitor.finish.assert_called_once_with()
            self.assertEqual(report['error'],
                'ValueError: Mock startup stops before HTTP or GPU access')
            self.assertEqual(report['requests'], [])
            self.assertEqual(result, 1)  # The deliberate readiness failure is not a successful run.


if __name__ == '__main__':
    unittest.main()
