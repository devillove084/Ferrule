"""Parent/helper protocol failure mocks; primary failure cannot be hidden by cleanup."""
import json
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


if __name__ == '__main__':
    unittest.main()
