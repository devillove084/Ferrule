"""CPU-only checks for the official micro adapter; no GPU queries or launches."""
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nsys_micro as micro


class OfficialMicro(unittest.TestCase):
    def test_exact_official_collector_and_bounded_application_trace(self):
        argv = micro.profile_argv(Path('/fixture/session'), 'a'*64)
        self.assertEqual(argv[0], str(micro.NSYS))
        for option in ('--trace=cuda,nvtx,osrt', '--sample=none', '--cpuctxsw=none',
                       '--gpu-metrics-devices=none', '--gpuctxsw=false', '--duration=150',
                       '--kill=none', '--wait=all', '--force-overwrite=false'):
            self.assertIn(option, argv)
        self.assertFalse(any('cuda-sw' in item or 'cuda-trace-scope' in item for item in argv))
        self.assertEqual(argv[-2:], ['--digest', 'a'*64])

    def test_singleton_samples_preserve_injection_and_only_narrow_visibility(self):
        selected = ','.join(micro.shared.UUIDS)
        env = dict(CUDA_VISIBLE_DEVICES=selected, NVIDIA_VISIBLE_DEVICES=selected,
                   LD_PRELOAD='/fixture/libToolsInjection.so', LD_LIBRARY_PATH='/fixture')
        with mock.patch.dict(os.environ, env, clear=True):
            for uuid in micro.shared.UUIDS:
                result = micro.selected_environment(env, uuid)
                self.assertEqual(result['CUDA_VISIBLE_DEVICES'], uuid)
                self.assertEqual(result['NVIDIA_VISIBLE_DEVICES'], uuid)
                self.assertEqual(result['LD_PRELOAD'], env['LD_PRELOAD'])
                self.assertEqual(result['LD_LIBRARY_PATH'], env['LD_LIBRARY_PATH'])
            with self.assertRaises(ValueError):
                micro.selected_environment(env, micro.INVENTORY[0])

    def test_widened_or_reordered_parent_is_rejected(self):
        for visibility in ('0,1', ','.join(reversed(micro.shared.UUIDS)), ','.join(micro.INVENTORY)):
            with mock.patch.dict(os.environ, dict(CUDA_VISIBLE_DEVICES=visibility,
                                                  NVIDIA_VISIBLE_DEVICES=visibility), clear=True):
                with self.assertRaises(ValueError):
                    micro.selected_environment(os.environ, micro.shared.UUIDS[0])

    def test_progress_uses_actual_monitor_failed_event(self):
        import threading
        from types import SimpleNamespace
        server = SimpleNamespace(poll=lambda: None)
        monitor = SimpleNamespace(failed=threading.Event())
        with mock.patch.object(micro.time, 'monotonic', return_value=1):
            micro.check_progress(server, monitor, 2)
            monitor.failed.set()
            with self.assertRaises(ValueError):
                micro.check_progress(server, monitor, 2)
            monitor.failed.clear()
            with self.assertRaises(ValueError):
                micro.check_progress(server, monitor, 1)
            server.poll = lambda: 0
            with self.assertRaises(ValueError):
                micro.check_progress(server, monitor, 2)

    def test_eight_lock_scope_keeps_original_tolerances(self):
        self.assertEqual(len(set(micro.INVENTORY)), 8)
        self.assertEqual(micro.INVENTORY, sorted(micro.INVENTORY))
        self.assertEqual(micro.live.scope.PROBE_CEILING_BYTES, 16 << 20)
        self.assertEqual(micro.live.scope.TOLERANCE_BYTES, 1 << 20)
        self.assertEqual(micro.live.scope.PROBE_SECONDS, 60)


if __name__ == '__main__':
    unittest.main()
