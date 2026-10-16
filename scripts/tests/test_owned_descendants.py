"""Real Linux subreaper children and kernel-authority mocks; no GPUs/packages."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import textwrap
import time
import unittest
from unittest import mock

from test_vllm_validation import harness

spec = importlib.util.spec_from_file_location('owned_subreaper', harness.SCRIPT.with_name('_owned_subreaper.py'))
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def ident(pid=123, birth=10, ppid=None):
    return dict(pid=pid, birth_ticks=birth, ppid=os.getpid() if ppid is None else ppid,
                pgrp=pid, session=pid)


class IdentityBoundaryTests(unittest.TestCase):
    def test_foreign_or_reaped_pid_is_not_signal_authority(self):
        owner = helper.Custodian('fixture')
        with mock.patch.object(helper.os, 'waitid', side_effect=ChildProcessError), \
                mock.patch.object(helper, 'identity') as stat, mock.patch.object(helper.os, 'kill') as kill:
            with self.assertRaises(ChildProcessError):
                owner.signal_child(123, signal.SIGKILL)
        stat.assert_not_called()
        kill.assert_not_called()

    def test_pid_birth_mismatch_and_nonchild_ppid_refuse_signal(self):
        for value in (ident(birth=11), ident(ppid=1)):
            owner = helper.Custodian('fixture')
            with mock.patch.object(helper.os, 'waitid', return_value=None), \
                    mock.patch.object(helper, 'identity', return_value=ident()):
                owner.register(123)
            with mock.patch.object(helper.os, 'waitid', return_value=None), \
                    mock.patch.object(helper, 'identity', return_value=value), \
                    mock.patch.object(helper.os, 'kill') as kill:
                with self.assertRaises(RuntimeError):
                    owner.signal_child(123, signal.SIGKILL)
            kill.assert_not_called()

    def test_new_kernel_child_reusing_old_root_pid_cannot_replace_root_credentials(self):
        owner = helper.Custodian('fixture')
        owner.root_pid = 123
        with mock.patch.object(helper.os, 'waitid', return_value=None), \
                mock.patch.object(helper, 'identity', return_value=ident()):
            original = owner.register(123)
        original.update(reaped=True, exit_code=7)
        del owner.active[123]
        owner.root_code = 7
        with mock.patch.object(helper.os, 'waitid', return_value=None), \
                mock.patch.object(helper, 'identity', return_value=ident(birth=11)):
            adopted = owner.register(123)
        self.assertEqual(adopted['role'], 'adopted')
        self.assertEqual(owner.custody['root_identity']['birth_ticks'], 10)
        self.assertEqual(owner.root_code, 7)

    def test_signal_preceded_by_wnowait_not_reap(self):
        owner = helper.Custodian('fixture')
        with mock.patch.object(helper.os, 'waitid', return_value=None) as wait, \
                mock.patch.object(helper, 'identity', return_value=ident()), \
                mock.patch.object(helper.os, 'kill') as kill:
            owner.signal_child(123, signal.SIGSTOP)
        self.assertEqual(wait.call_args.args, (os.P_PID, 123, helper.WAIT))
        self.assertTrue(wait.call_args.args[2] & os.WNOWAIT)
        kill.assert_called_once_with(123, signal.SIGSTOP)

    def test_terminal_at_registration_or_during_metadata_is_never_signalled_or_consumed(self):
        terminal = os.waitid_result((123, os.getuid(), signal.SIGCHLD, 0, os.CLD_EXITED))
        for at_registration in (True, False):
            owner = helper.Custodian("fixture")
            with self.subTest(at_registration=at_registration), \
                    mock.patch.object(helper.os, "waitid", side_effect=[terminal if at_registration else None, terminal]) as wait, \
                    mock.patch.object(helper, "identity", return_value=ident()), \
                    mock.patch.object(helper.os, "kill") as kill:
                owner.signal_child(123, signal.SIGSTOP)
            kill.assert_not_called()
            self.assertEqual(wait.call_args_list, [mock.call(os.P_PID, 123, helper.WAIT)] * 2)
            self.assertFalse(owner.active[123]["reaped"])
            self.assertEqual(owner.active[123]["signals"], [])

    def test_changed_group_and_session_do_not_remove_direct_child_signal_authority(self):
        owner = helper.Custodian("fixture")
        original = ident()
        changed = dict(original, pgrp=456, session=456)
        with mock.patch.object(helper.os, "waitid", return_value=None), \
                mock.patch.object(helper, "identity", return_value=original):
            owner.register(123)
        with mock.patch.object(helper.os, "waitid", return_value=None) as wait, \
                mock.patch.object(helper, "identity", return_value=changed), \
                mock.patch.object(helper.os, "kill") as kill:
            owner.signal_child(123, signal.SIGSTOP)
        kill.assert_called_once_with(123, signal.SIGSTOP)
        self.assertEqual(wait.call_args.args, (os.P_PID, 123, helper.WAIT))
        self.assertEqual(owner.active[123]["identity"], original)

    def test_empty_child_list_is_not_echild_and_timeout_is_unknown(self):
        owner = helper.Custodian('fixture')
        with mock.patch.object(helper.os, 'waitid', return_value=None), \
                mock.patch.object(owner, 'children', return_value=[]):
            owner.cleanup(seconds=0)
        self.assertEqual(owner.custody['status'], 'unknown')
        self.assertFalse(owner.custody['echild'])

    def test_prctl_denial_and_unverified_state_cannot_open_launch_gate(self):
        for returncode in (-1, 0):
            libc = mock.Mock()
            libc.prctl.return_value = returncode
            with mock.patch.object(helper.ctypes, 'CDLL', return_value=libc), \
                    mock.patch.object(helper.signal, 'signal') as handler, \
                    mock.patch.object(helper.os, 'fork') as fork:
                with self.assertRaises((OSError, RuntimeError)):
                    helper.enable_subreaper()
            fork.assert_not_called()
            self.assertEqual(handler.call_args.args[0], signal.SIGCHLD)
            self.assertNotEqual(handler.call_args.args[1], signal.SIG_IGN)

    def test_permission_error_not_drain(self):
        owner = helper.Custodian('fixture')
        with mock.patch.object(helper.os, 'waitid', return_value=None), \
                mock.patch.object(helper, 'identity', return_value=ident()), \
                mock.patch.object(helper.os, 'kill', side_effect=PermissionError):
            with self.assertRaises(PermissionError):
                owner.signal_child(123, signal.SIGKILL)
        self.assertNotEqual(owner.custody['status'], 'proven_drained')

    def test_echild_with_unreconciled_records_is_not_proof(self):
        owner = helper.Custodian('fixture')
        owner.active[123] = {'identity': ident()}
        with mock.patch.object(helper.os, 'waitid', side_effect=ChildProcessError):
            with self.assertRaisesRegex(RuntimeError, 'unreconciled'):
                owner.reap_available()


    def test_cleanup_reaps_echild_before_signalling_or_scanning_children(self):
        owner = helper.Custodian('fixture')
        with mock.patch.object(owner, 'reap_available', return_value=True), \
                mock.patch.object(owner, 'children') as children, \
                mock.patch.object(helper.os, 'kill') as kill:
            owner.cleanup()
        children.assert_not_called(); kill.assert_not_called()
        self.assertEqual(owner.custody['status'], 'proven_drained')

    def test_metadata_never_authorizes_foreign_pid_or_unknown_kernel_identity(self):
        owner = helper.Custodian('fixture')
        with mock.patch.object(helper.os, 'waitid', side_effect=ChildProcessError), \
                mock.patch.object(helper.os, 'readlink') as image, \
                mock.patch.object(helper.os, 'kill') as kill:
            with self.assertRaises(ChildProcessError):owner.signal_child(123, signal.SIGKILL)
        image.assert_not_called(); kill.assert_not_called()
        for changed in (ident(birth=11), ident(ppid=1), FileNotFoundError('identity unobserved')):
            owner = helper.Custodian('fixture')
            with mock.patch.object(helper.os, 'waitid', return_value=None), \
                    mock.patch.object(helper, 'identity', side_effect=[ident(), changed]), \
                    mock.patch.object(helper.os, 'kill') as kill:
                with self.assertRaises((RuntimeError, FileNotFoundError)):
                    owner.signal_child(123, signal.SIGKILL)
            kill.assert_not_called()

    def test_inaccessible_exec_metadata_is_unobserved_not_signal_authority(self):
        owner = helper.Custodian('fixture')
        with mock.patch.object(helper.os, 'waitid', return_value=None), \
                mock.patch.object(helper, 'identity', return_value=ident()), \
                mock.patch.object(helper.os, 'readlink', side_effect=FileNotFoundError), \
                mock.patch.object(helper.Path, 'open', side_effect=FileNotFoundError), \
                mock.patch.object(helper.os, 'kill') as kill:
            owner.signal_child(123, signal.SIGSTOP)
        kill.assert_called_once_with(123, signal.SIGSTOP)
        observation=owner.entries[0]['process_observations']['before_signal_19']
        self.assertEqual(observation['executable']['state'], 'unobserved')
        self.assertEqual(observation['command']['state'], 'unobserved')
        self.assertEqual(observation['signals_already_recorded'], 0)
        self.assertEqual(observation['before'], observation['after'])


@unittest.skipUnless(Path('/proc/self/stat').exists(), 'Linux procfs required')
class RealDescendantTests(unittest.TestCase):
    def run_code(self, code, timeout=2, payload=None):
        return harness.run_process([harness.sys.executable, '-B', '-c', textwrap.dedent(code)], timeout, payload)

    def drained(self, report, minimum=1):
        self.assertNotIn('error', report, report)
        custody = report['custody']
        self.assertEqual(custody['status'], 'proven_drained', report)
        self.assertTrue(custody['echild'])
        self.assertTrue(report['handshake']['containment_verified'])
        self.assertGreaterEqual(len(custody['entries']), minimum)
        for row in custody['entries']:
            self.assertTrue(row['reaped'], row)
            self.assertEqual(row['identity']['ppid'], custody['helper_identity']['pid'])
            try:
                current = helper.identity(row['identity']['pid'])
            except FileNotFoundError:
                continue
            self.assertNotEqual(current['birth_ticks'], row['identity']['birth_ticks'], row)
        self.assertEqual(report['gpu_quiescence'], 'unverified')

    def test_helper_dies_before_handshake_never_launches_application(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'_owned_subreaper.py').write_text('import os; os._exit(23)')
            marker = root/'application-must-not-run'
            with mock.patch.object(harness, 'SCRIPT', root/'test_vllm_qwen35.py'):
                report = harness.run_process([harness.sys.executable, '-B', '-c',
                    'import pathlib,sys; pathlib.Path(sys.argv[1]).touch()', str(marker)], 1)
            self.assertFalse(marker.exists())
            self.assertEqual(report['custody']['status'], 'unknown')
            self.assertIn('error', report)
            self.assertNotIn('handshake', report)

    def test_kernel_probe_needs_no_pidfd_or_cgroup(self):
        capability = harness.custody_capabilities()
        self.assertTrue(capability['containment'], capability)
        self.drained(capability['probe'], minimum=0)

    def test_natural_exit_exec_error_and_nonzero_status_preserved(self):
        for code, expected in (('print("hello")', 0), ('raise SystemExit(7)', 7)):
            report = self.run_code(code)
            self.drained(report)
            self.assertEqual(report['returncode'], expected)
        report = harness.run_process(['/no-such-owned-executable'], 2)
        self.drained(report)
        self.assertEqual(report['returncode'], 127)
        self.assertIn('Root exec', report['stderr'])

    def test_successful_root_allows_adopted_child_to_finish_before_any_signal(self):
        with tempfile.TemporaryDirectory() as folder:
            report = self.run_code('''
                import json,os,pathlib,sys,time
                config=json.load(sys.stdin)
                journal=pathlib.Path(config["directory"])/"custody.json"
                if os.fork() != 0:
                    os._exit(0)
                deadline=time.monotonic()+1.5
                while time.monotonic()<deadline:
                    data=json.loads(journal.read_text())
                    if data.get("natural_drain",{}).get("outcome")=="waiting":
                        print("ADOPTED_NATURAL_DRAIN_ACK",flush=True)
                        os._exit(0)
                    time.sleep(.005)
                os._exit(11)
            ''', timeout=2, payload=dict(manifest={}, directory=folder))
        self.drained(report, minimum=2)
        self.assertEqual(report['returncode'], 0)
        self.assertFalse(report['timed_out'])
        self.assertIn('ADOPTED_NATURAL_DRAIN_ACK', report['stdout'])
        self.assertTrue(all(not row['signals'] and row['exit_code']==0 for row in report['custody']['entries']))
        natural=report['custody']['natural_drain']
        self.assertEqual(natural['outcome'], 'echild_without_signals')
        self.assertLessEqual(natural['deadline_monotonic'],natural['original_job_deadline_monotonic'])
        self.assertLessEqual(natural['deadline_monotonic']-natural['started_monotonic'],helper.CLEANUP_SECONDS)
        adopted=[r for r in report['custody']['entries'] if r['role']=='adopted']
        evidence=adopted[0]['process_observations']['registration']
        self.assertEqual(evidence['role'], 'adopted')
        self.assertEqual(evidence['before']['birth_ticks'],adopted[0]['identity']['birth_ticks'])
        self.assertEqual(evidence['executable']['state'], 'observed')
        self.assertEqual(evidence['command']['state'], 'observed')

    def test_stuck_descendant_uses_only_remaining_job_deadline_then_owned_cleanup(self):
        report = self.run_code('''
            import os,signal,time
            read,write=os.pipe()
            if os.fork()==0:
                os.close(read)
                signal.signal(signal.SIGTERM,signal.SIG_IGN)
                os.write(write,b"R");os.close(write)
                time.sleep(10)
                os._exit(0)
            os.close(write)
            assert os.read(read,1)==b"R"
            os.close(read);os._exit(0)
        ''', timeout=.3)
        self.drained(report, minimum=2)
        self.assertEqual(report['returncode'],0)
        self.assertTrue(report['timed_out'])
        self.assertLess(report['wall_seconds'],3)
        natural=report['custody']['natural_drain']
        self.assertEqual(natural['outcome'],'original_job_deadline')
        self.assertEqual(natural['deadline_monotonic'],natural['original_job_deadline_monotonic'])
        self.assertLess(natural['deadline_monotonic']-natural['started_monotonic'],.3)
        signaled=[r for r in report['custody']['entries'] if r['signals']]
        self.assertEqual(len(signaled),1)
        self.assertEqual(signaled[0]['role'],'adopted')
        self.assertIn(signal.SIGKILL,signaled[0]['signals'])
        before=signaled[0]['process_observations']['before_signal_19']
        self.assertEqual(before['signals_already_recorded'],0)
        self.assertEqual(before['executable']['state'],'observed')
        self.assertEqual(before['command']['state'],'observed')
        self.assertEqual(before['before']['ppid'],report['custody']['helper_identity']['pid'])
        self.assertEqual(before['before']['birth_ticks'],signaled[0]['identity']['birth_ticks'])
        self.assertTrue(all(not r['signals'] for r in report['custody']['entries'] if r['role']=='root'))

    def test_natural_grace_cap_does_not_consume_the_entire_long_job_budget(self):
        report = self.run_code('''
            import os,signal,time
            read,write=os.pipe()
            if os.fork()==0:
                os.close(read);signal.signal(signal.SIGTERM,signal.SIG_IGN)
                os.write(write,b"R");os.close(write);time.sleep(12);os._exit(0)
            os.close(write);assert os.read(read,1)==b"R";os.close(read);os._exit(0)
        ''', timeout=9)
        self.drained(report, minimum=2)
        self.assertFalse(report['timed_out'])
        natural=report['custody']['natural_drain']
        self.assertEqual(natural['outcome'],'grace_exhausted')
        self.assertEqual(natural['deadline_monotonic'],natural['started_monotonic']+helper.CLEANUP_SECONDS)
        self.assertLess(natural['deadline_monotonic'],natural['original_job_deadline_monotonic'])
        self.assertTrue(any(r['signals'] for r in report['custody']['entries']))
        self.assertLess(report['wall_seconds'],8)

    def test_timeout_stubborn_root_escalates_without_group_signals(self):
        report = self.run_code('''
            import signal,time
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            time.sleep(4)
        ''', timeout=.2)
        self.drained(report)
        self.assertTrue(report['timed_out'])
        self.assertEqual(report['returncode'], -signal.SIGKILL)
        self.assertIn('TimeoutExpired', report['primary_error'])
        self.assertLess(report['wall_seconds'], 3)

    def test_unobserved_setsid_doublefork_adopted_even_after_root_exit(self):
        report = self.run_code('''
            import os,signal,time
            child=os.fork()
            if child == 0:
                os.setsid()
                if os.fork() != 0:
                    os._exit(0)
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                time.sleep(4)
                os._exit(0)
            time.sleep(.1)
            os._exit(0)
        ''')
        self.drained(report, minimum=3)
        self.assertEqual(report['returncode'], 0)
        self.assertTrue(any(row['role'] == 'adopted' for row in report['custody']['entries']))
        self.assertTrue(any(signal.SIGKILL in row['signals'] for row in report['custody']['entries']))

    def test_late_fork_from_term_handler_and_nested_subreaper_are_drained(self):
        report = self.run_code('''
            import ctypes,os,signal,time
            ctypes.CDLL(None).prctl(36,1,0,0,0)
            def late(signum, frame):
                if os.fork() == 0:
                    os.setsid()
                    signal.signal(signal.SIGTERM, signal.SIG_IGN)
                    time.sleep(4)
                    os._exit(0)
                os._exit(0)
            signal.signal(signal.SIGTERM, late)
            time.sleep(4)
        ''', timeout=.2)
        self.drained(report, minimum=2)
        self.assertTrue(report['timed_out'])
        self.assertEqual(report['returncode'], 0)

    def test_foreign_same_executable_and_caller_children_are_untouched(self):
        foreign = subprocess.Popen([harness.sys.executable, '-B', '-c', 'import time; time.sleep(3)'])
        before = helper.identity(foreign.pid)
        try:
            report = self.run_code('import time; time.sleep(4)', timeout=.05)
            self.drained(report)
            self.assertIsNone(foreign.poll())
            self.assertEqual(before['birth_ticks'], helper.identity(foreign.pid)['birth_ticks'])
            self.assertNotIn(foreign.pid, [row['identity']['pid'] for row in report['custody']['entries']])
        finally:
            foreign.terminate()
            foreign.wait(timeout=2)

    def test_durable_checkpoint_exists_while_root_runs_and_is_finalized(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self.run_code('''
                import json,sys,pathlib
                config=json.load(sys.stdin)
                data=json.loads((pathlib.Path(config['directory'])/'custody.json').read_text())
                assert data['containment_verified'] and data['root_identity']['pid'] > 0
                print(data['launch_id'])
            ''', payload=dict(manifest={}, directory=directory))
            self.drained(report)
            saved = json.loads((Path(directory)/'custody.json').read_text())
            self.assertEqual(saved, report['custody'])
            self.assertEqual(report['stdout'].strip(), saved['launch_id'])

    def test_parent_control_eof_still_drains_and_records_original_cause(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryFile() as inp:
            root = Path(directory)
            result_path = root/'result.json'
            parent, child = socket.socketpair()
            process = subprocess.Popen([harness.sys.executable, '-B', str(harness.SCRIPT.with_name('_owned_subreaper.py')),
                        str(child.fileno()), str(inp.fileno()), str(result_path), 'disconnect-fixture'],
                        pass_fds=(child.fileno(), inp.fileno()), stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL, start_new_session=True)
            child.close()
            try:
                parent.settimeout(2)
                data = bytearray()
                while not data.endswith(b'\n'):
                    data.extend(parent.recv(4096))
                ready = json.loads(data)
                self.assertTrue(ready['containment_verified'])
                parent.sendall((json.dumps(dict(launch_id='disconnect-fixture',
                    argv=[harness.sys.executable, '-B', '-c', 'import time; time.sleep(4)'],
                    timeout=3, journal=str(root/'custody.json'))) + '\n').encode())
                deadline = time.monotonic()+2
                while time.monotonic() < deadline:
                    if (root/'custody.json').exists():
                        receipt = json.loads((root/'custody.json').read_text())
                        if receipt['root_identity'] is not None:
                            break
                    time.sleep(.01)
                else:
                    self.fail('Root did not reach live durable checkpoint')
                parent.close()
                process.wait(timeout=7)
                report = json.loads(result_path.read_text())
                self.assertIn('Owner connection lost', report['primary_error'])
                self.assertEqual(report['custody']['status'], 'proven_drained')
                self.assertTrue(report['custody']['echild'])
                self.assertTrue(all(row['reaped'] for row in report['custody']['entries']))
            finally:
                parent.close()
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=7)

    def test_real_owner_interrupt_waits_for_cleanup_then_reraises_same_primary(self):
        primary = KeyboardInterrupt('real owner cutoff')
        previous = signal.getsignal(signal.SIGALRM)
        def interrupt(signum, frame):
            raise primary
        try:
            signal.signal(signal.SIGALRM, interrupt)
            signal.setitimer(signal.ITIMER_REAL, .2)
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.run_code('import time; print("root-live",flush=True); time.sleep(4)', timeout=3)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
        self.assertIs(caught.exception, primary)
        report = primary.supervisor_result
        self.assertIn('root-live', report['stdout'])
        self.assertIn('real owner cutoff', report['primary_error'])
        self.assertEqual(report['custody']['status'], 'proven_drained')
        self.assertTrue(report['custody']['echild'])
        self.assertIn('Owner cancellation', report['helper_primary_error'])

    def test_threaded_callers_and_thread_spawned_descendants_have_isolated_reapers(self):
        from concurrent.futures import ThreadPoolExecutor
        code = """
            import os,subprocess,sys,threading,time
            def spawn():
                subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(4)'],start_new_session=True)
            thread=threading.Thread(target=spawn); thread.start(); thread.join()
            time.sleep(.1)
            os._exit(0)
        """
        with ThreadPoolExecutor(max_workers=2) as pool:
            reports = list(pool.map(lambda _: self.run_code(code), range(2)))
        for report in reports:
            self.drained(report, minimum=2)
        self.assertNotEqual(reports[0]['custody']['launch_id'], reports[1]['custody']['launch_id'])
        self.assertFalse(set(row['identity']['pid'] for row in reports[0]['custody']['entries']) &
                         set(row['identity']['pid'] for row in reports[1]['custody']['entries']))

    def test_checkpoint_io_failure_does_not_prevent_kernel_cleanup(self):
        owner = helper.Custodian('fixture', journal=Path('/fixture/custody.json'))
        with mock.patch.object(helper, 'checkpoint', side_effect=OSError('disk error')):
            owner.persist()
            with mock.patch.object(owner, 'children', return_value=[]), \
                    mock.patch.object(helper.os, 'waitid', side_effect=ChildProcessError):
                owner.cleanup()
        self.assertTrue(owner.custody['echild'])
        self.assertEqual(owner.custody['status'], 'proven_drained')
        self.assertIn('disk error', owner.custody['reporting_errors'][0])

    def test_inherited_sigchld_ignore_cannot_autoreap_helper_children(self):
        # Isolate SIGCHLD changes too; do not alter the test runner's reaper state.
        code = '''
            import importlib.util,signal,sys,json
            signal.signal(signal.SIGCHLD,signal.SIG_IGN)
            spec=importlib.util.spec_from_file_location('h',sys.argv[1])
            h=importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
            r=h.run_process([sys.executable,'-B','-c','import sys; sys.exit(9)'],2)
            print(json.dumps(r))
        '''
        outer = subprocess.run([harness.sys.executable, '-B', '-c', textwrap.dedent(code), str(harness.SCRIPT)],
                               capture_output=True, text=True, timeout=5)
        self.assertEqual(outer.returncode, 0, outer.stderr)
        report = json.loads(outer.stdout)
        self.drained(report)
        self.assertEqual(report['returncode'], 9)


if __name__ == '__main__':
    unittest.main()
