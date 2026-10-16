"""Parent/helper protocol failure mocks; primary failure cannot be hidden by cleanup."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import tempfile
import subprocess
import unittest
from unittest import mock

from test_vllm_validation import harness


class SupervisorFailureTests(unittest.TestCase):
    def fixture(self, status='proven_drained'):
        identity = dict(pid=123456, birth_ticks=10, ppid=1, pgrp=123456, session=123456)
        ready = dict(kind='ready', protocol=1, launch_id='fixture', helper_identity=identity,
                     containment_verified=True, method='linux-subreaper-waitid')
        data = dict(returncode=7, timed_out=False, cleanup_errors=[], custody=dict(status=status,
                    launch_id='fixture', helper_identity=identity, containment_verified=True, echild=True))
        process = mock.Mock(pid=123456, returncode=None)
        parent, child = mock.Mock(), mock.Mock()
        parent.recv.return_value = (json.dumps(ready)+'\n').encode()
        return ready, data, process, parent, child

    def invoke(self, data, process, parent, child):
        with mock.patch.object(harness.uuid, 'uuid4', return_value=mock.Mock(hex='fixture')), \
                mock.patch.object(harness.socket, 'socketpair', return_value=(parent, child)), \
                mock.patch.object(harness.subprocess, 'Popen', return_value=process), \
                mock.patch.object(harness.Path, 'read_text', return_value=json.dumps(data)), \
                mock.patch.object(harness.os, 'killpg', side_effect=AssertionError('no group authority')):
            return harness.run_process(['mock-root'], 1)

    def test_timeout_keeps_primary_when_helper_wait_and_evidence_fail(self):
        _, data, process, parent, child = self.fixture()
        process.wait.side_effect = [subprocess.TimeoutExpired('original', 9),
                                   subprocess.TimeoutExpired('cleanup', 8), -9]
        report = self.invoke(data, process, parent, child)
        self.assertTrue(report['timed_out'])
        self.assertIn('original', report['primary_error'])
        self.assertEqual(report['custody']['status'], 'unknown')
        self.assertGreaterEqual(len(report['cleanup_errors']), 2)
        process.kill.assert_called_once()

    def test_owner_interrupt_reraised_after_helper_drained_preserves_primary(self):
        _, data, process, parent, child = self.fixture()
        primary = KeyboardInterrupt('outer cutoff')
        calls = []
        def wait(timeout):
            calls.append(timeout)
            if len(calls) == 1:
                raise primary
            process.returncode = 0
            return 0
        process.wait.side_effect = wait
        data.update(primary_error='RuntimeError: Owner cancellation', error='Helper interrupted')
        with self.assertRaises(KeyboardInterrupt) as caught:
            self.invoke(data, process, parent, child)
        self.assertIs(caught.exception, primary)
        self.assertIn('outer cutoff', primary.supervisor_result['primary_error'])
        self.assertEqual(primary.supervisor_result['custody']['status'], 'proven_drained')
        self.assertEqual(primary.supervisor_result['returncode'], 7)
        self.assertIn(mock.call(b'cancel\n'), parent.sendall.call_args_list)
        process.kill.assert_not_called()

    def test_invalid_handshake_never_authorizes_root(self):
        ready, data, process, parent, child = self.fixture()
        ready['launch_id'] = 'foreign-launch'
        parent.recv.return_value = (json.dumps(ready)+'\n').encode()
        process.returncode = 0
        report = self.invoke(data, process, parent, child)
        parent.sendall.assert_not_called()
        process.kill.assert_not_called()
        self.assertEqual(report['custody']['status'], 'unknown')
        self.assertIn('handshake', report['primary_error'])

    def test_failed_helper_exit_or_no_echild_invalidates_even_written_drain(self):
        for code, echild in ((-9, True), (0, False)):
            _, data, process, parent, child = self.fixture()
            process.returncode = code
            data['custody']['echild'] = echild
            report = self.invoke(data, process, parent, child)
            self.assertEqual(report['custody']['status'], 'unknown')
            self.assertIn('error', report)
            process.kill.assert_not_called()

    def test_root_failure_and_cleanup_failure_both_retained(self):
        _, data, process, parent, child = self.fixture(status='unknown')
        process.returncode = 0
        data.update(primary_error='root failure', cleanup_errors=['PermissionError: cleanup denied'],
                    error='cleanup incomplete')
        report = self.invoke(data, process, parent, child)
        self.assertEqual(report['returncode'], 7)
        self.assertEqual(report['primary_error'], 'root failure')
        self.assertIn('cleanup denied', report['cleanup_errors'][0])
        self.assertEqual(report['custody']['status'], 'unknown')


_spec = importlib.util.spec_from_file_location('owned_subreaper_contract',
    Path(__file__).resolve().parents[1] / '_owned_subreaper.py')
helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helper)


class PostSetsidRootTests(unittest.TestCase):
    def setUp(self):
        self.pre = dict(pid=100, birth_ticks=10, ppid=99, pgrp=99, session=99)
        self.post = dict(self.pre, pgrp=100, session=100)
        self.phase = 'pre'
        self.snapshots = []
        with mock.patch.object(helper, 'identity', return_value=dict(self.pre, pid=99)):
            self.owner = helper.Custodian('fixture', Path('/synthetic/custody.json'))
        self.owner.pid = 99
        for name, value in (('fork', 100), ('close', None), ('waitid', None)):
            patch = mock.patch.object(helper.os, name, return_value=value)
            setattr(self, name, patch.start()); self.addCleanup(patch.stop)
        patch = mock.patch.object(helper.os, 'pipe', side_effect=[(10, 11), (12, 13)])
        patch.start(); self.addCleanup(patch.stop)
        patch = mock.patch.object(helper, 'identity',
            side_effect=lambda pid: copy.deepcopy(self.post if self.phase == 'post' else self.pre))
        self.identity = patch.start(); self.addCleanup(patch.stop)
        patch = mock.patch.object(helper.select, 'select', return_value=([12], [], []))
        self.select = patch.start(); self.addCleanup(patch.stop)
        def ready(fd, count):
            self.assertEqual((fd, count), (12, 1)); self.phase = 'post'; return b'R'
        patch = mock.patch.object(helper.os, 'read', side_effect=ready)
        self.read = patch.start(); self.addCleanup(patch.stop)
        def checkpoint(path, value): self.snapshots.append(copy.deepcopy(value))
        patch = mock.patch.object(helper, 'checkpoint', side_effect=checkpoint)
        self.checkpoint = patch.start(); self.addCleanup(patch.stop)
        patch = mock.patch.object(helper.os, 'write')
        self.write = patch.start(); self.addCleanup(patch.stop)
        patch = mock.patch.object(helper.os, 'kill')
        self.kill = patch.start(); self.addCleanup(patch.stop)
        patch = mock.patch.object(helper.os, 'killpg', side_effect=AssertionError('No group authority'))
        patch.start(); self.addCleanup(patch.stop)

    def test_pre_post_setsid_snapshot_records_exact_final_identity_before_exec_release(self):
        def released(fd, data):
            self.assertEqual((fd, data), (11, b'G'))
            self.assertTrue(self.snapshots[-1]['root_session_ready'])
            self.assertEqual(self.snapshots[-1]['root_identity'], self.post)
        self.write.side_effect = released
        self.owner.launch(['never-exec'], 7, .25)
        self.assertTrue(all(v['root_identity'] == self.post for v in self.snapshots))
        self.assertEqual(self.owner.entries[0]['identity'], self.post)
        self.waitid.assert_called_once_with(os.P_PID, 100, helper.WAIT)
        self.select.assert_called_once_with([12], [], [], .25)
        self.write.assert_called_once_with(11, b'G')
        self.kill.assert_not_called()

    def test_missing_expired_or_malformed_ready_never_releases_exec(self):
        for value in (None, b'', b'X'):
            self.select.return_value = ([], [], []) if value is None else ([12], [], [])
            self.read.side_effect = None; self.read.return_value = value
            with self.subTest(ready=value), self.assertRaisesRegex(RuntimeError, 'readiness'):
                self.owner.launch(['never-exec'], 7, .25)
            self.write.assert_not_called(); self.waitid.assert_not_called()
            self.owner.custody['root_session_ready'] = False
            self.owner.active.clear(); self.owner.entries.clear()
            self.owner.custody['root_identity'] = None
            self.owner.root_pid = None
            helper.os.pipe.side_effect = [(10, 11), (12, 13)]

    def test_ready_with_pre_setsid_or_foreign_root_identity_fails_closed(self):
        for key, value in (('pgrp', 99), ('session', 99), ('pid', 101), ('ppid', 98)):
            self.post = dict(self.pre, pgrp=100, session=100); self.post[key] = value
            with self.subTest(field=key), self.assertRaises(RuntimeError):
                self.owner.launch(['never-exec'], 7, .25)
            self.write.assert_not_called(); self.kill.assert_not_called()
            self.owner.active.clear(); self.owner.entries.clear()
            self.owner.custody.update(root_identity=None, root_session_ready=False)
            helper.os.pipe.side_effect = [(10, 11), (12, 13)]

    def test_unpersisted_post_setsid_identity_never_releases_exec(self):
        self.checkpoint.side_effect = OSError('fixture fsync failure')
        with self.assertRaisesRegex(RuntimeError, 'checkpoint failed before exec'):
            self.owner.launch(['never-exec'], 7)
        self.write.assert_not_called(); self.kill.assert_not_called()
        self.assertTrue(self.owner.custody['reporting_errors'])

    def test_pid_reuse_foreign_parent_and_nonchild_never_gain_cleanup_signal_authority(self):
        self.phase = 'post'; self.owner.root_pid = 100
        self.owner.register(100)
        for field, value in (('birth_ticks', 11), ('ppid', 98)):
            saved = self.post[field]; self.post[field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                self.owner.signal_child(100, signal.SIGTERM)
            self.post[field] = saved; self.kill.assert_not_called()
        self.waitid.side_effect = ChildProcessError('not an unreaped direct child')
        for pid in (100, -100, 999):
            with self.subTest(pid=pid), self.assertRaises(ChildProcessError):
                self.owner.signal_child(pid, signal.SIGKILL)
        self.kill.assert_not_called()

    def test_reused_active_root_birth_rejected_before_exec_gate(self):
        self.owner.root_pid = 100; self.owner.register(100)
        self.post['birth_ticks'] = 11
        with self.assertRaisesRegex(RuntimeError, 'birth changed'):
            self.owner.launch(['never-exec'], 7)
        self.write.assert_not_called(); self.kill.assert_not_called()


class PostSetsidLiveCPUTests(unittest.TestCase):
    def test_real_cpu_child_sees_exact_post_setsid_persisted_identity(self):
        with tempfile.TemporaryDirectory(prefix='ferrule-session-contract-') as folder:
            journal = str(Path(folder) / 'custody.json')
            code = ('import json,os,pathlib;'
                'fields=pathlib.Path("/proc/self/stat").read_text().rsplit(")",1)[1].split();'
                'live=dict(pid=os.getpid(),birth_ticks=int(fields[19]),ppid=int(fields[1]),pgrp=int(fields[2]),session=int(fields[3]));'
                'data=json.loads(pathlib.Path(' + repr(journal) + ').read_text());'
                'assert data["root_session_ready"] is True and data["root_identity"]==live;'
                'assert live["pgrp"]==live["session"]==live["pid"];print("POST_SETSID_EXACT_IDENTITY")')
            value = harness.run_process([harness.sys.executable, '-S', '-B', '-c', code], 5,
                dict(manifest={}, directory=folder))
        self.assertEqual(value['returncode'], 0, value)
        self.assertEqual(value['stdout'].strip(), 'POST_SETSID_EXACT_IDENTITY')
        self.assertEqual(value['custody']['status'], 'proven_drained')
        self.assertTrue(value['custody']['echild'])
        self.assertTrue(value['custody']['root_session_ready'])
        self.assertTrue(all(not e['signals'] for e in value['custody']['entries']))


if __name__ == '__main__':
    unittest.main()
