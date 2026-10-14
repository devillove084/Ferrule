"""CPU mocks plus a Python-only custodian round trip; no GPU telemetry/server."""
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from test_vllm_validation import harness

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('pinned_jobs', ROOT / 'scripts/run_pinned_jobs.py')
runner = importlib.util.module_from_spec(spec)
with mock.patch.dict(sys.modules, {'test_vllm_qwen35': harness}):
    spec.loader.exec_module(runner)


def receipt():
    identity = dict(pid=123, birth_ticks=456)
    return dict(returncode=0, timed_out=False, cleanup_errors=[], descendant_custody='proven_drained',
                handshake=dict(containment_verified=True, launch_id='owned', helper_identity=identity),
                custody=dict(containment_verified=True, launch_id='owned', helper_identity=identity,
                             echild=True, status='proven_drained'))


def idle():
    return dict(gpu='\n'.join(f'{i}, GPU-{i}, 0 MiB, 0 %' for i in range(8)), apps='')


class PinnedJobs(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / 'source'
        self.input.write_text('frozen')
        self.library = self.root / 'library'
        self.library.write_text('library bytes, not loaded')
        self.libraries = self.root / 'libraries.json'
        self.libraries.write_text(json.dumps(dict(resolved_libraries=[dict(
            realpath=str(self.library), sha256=runner.sha(self.library))])))
        self.jobs = [dict(name='cpu', argv=[sys.executable, '-B', '-c', 'print("CPU only")'],
                         seconds=5, pins={str(self.input): runner.sha(self.input)},
                         env=dict(CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='void'))]
        runner.pin_jobs(self.jobs, self.libraries)
        self.output = self.root / 'receipts'

    def execute(self, result=None, sample=None):
        with mock.patch.object(runner.supervisor, 'run_process', return_value=result or receipt()) as run, \
                mock.patch.object(runner, 'sample', return_value=sample or idle()), \
                mock.patch.object(runner.time, 'sleep'):
            code = runner.execute(self.jobs, self.output, self.libraries, granted=True)
        return code, run

    def test_check_and_missing_grant_never_sample_or_launch(self):
        jobs = self.root / 'jobs.json'
        jobs.write_text(json.dumps(self.jobs))
        with mock.patch.object(runner, 'sample') as sample, \
                mock.patch.object(runner.supervisor, 'run_process') as run, \
                mock.patch('sys.stdout', new_callable=io.StringIO):
            self.assertEqual(runner.main(['--jobs', str(jobs), '--libraries', str(self.libraries)]), 0)
            with self.assertRaisesRegex(ValueError, 'grant required'):
                runner.execute(self.jobs, self.output, self.libraries)
            sample.assert_not_called()
            run.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_changed_source_library_binary_or_workflow_fails_before_telemetry(self):
        for path in (self.input, self.library):
            before = path.read_bytes()
            path.write_text('changed')
            with mock.patch.object(runner, 'sample') as sample, mock.patch.object(runner.supervisor, 'run_process') as run:
                with self.assertRaises(ValueError):
                    runner.execute(self.jobs, self.output, self.libraries, granted=True)
                sample.assert_not_called()
                run.assert_not_called()
            path.write_bytes(before)
        for path in (Path(sys.executable).resolve(), runner.SCRIPT):
            jobs = copy.deepcopy(self.jobs)
            jobs[0]['pins'][str(path)] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'Pinned input changed'):
                runner.validate(jobs, self.libraries)

    def test_unpinned_old_job_needs_reprepare_not_implicit_upgrade(self):
        del self.jobs[0]['environment']
        with self.assertRaisesRegex(ValueError, 'reprepare'):
            runner.validate(self.jobs, self.libraries)

    def test_independent_environments_do_not_leak_cpu_visibility_or_arithmetic(self):
        gpu = dict(name='gpu-mock', argv=self.jobs[0]['argv'], seconds=2, pins={},
                   env={'CUBLAS_WORKSPACE_CONFIG': ':0:0'})
        self.jobs.append(gpu)
        runner.pin_jobs(self.jobs, self.libraries)
        before = dict(os.environ)
        code, run = self.execute()
        self.assertEqual(code, 0)
        self.assertEqual(dict(os.environ), before)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertIn('CUDA_VISIBLE_DEVICES=', commands[0])
        self.assertIn('CUDA_VISIBLE_DEVICES=0', commands[1])
        self.assertIn('CUBLAS_WORKSPACE_CONFIG=:0:0', commands[1])
        self.assertEqual(commands[0][:2], ['/usr/bin/env', '-i'])
        self.assertEqual([call.args[1] for call in run.call_args_list], [5, 2])
        self.assertTrue((self.output / 'cpu-supervisor.json').exists())
        release = json.loads((self.output / 'release.json').read_text())
        self.assertEqual(release['release'], 'released')
        self.assertTrue(release['drained'])

    def test_timeout_nonzero_forged_or_unknown_custody_stops_sequence(self):
        self.jobs.append(dict(self.jobs[0], name='must-not-run'))
        runner.pin_jobs(self.jobs, self.libraries)
        for mutation in ('timeout', 'exit', 'identity', 'launch', 'echild', 'cleanup', 'unknown'):
            with self.subTest(mutation=mutation):
                result = receipt()
                if mutation == 'timeout': result['timed_out'] = True
                if mutation == 'exit': result['returncode'] = 7
                if mutation == 'identity': result['custody']['helper_identity'] = {'pid': 987}
                if mutation == 'launch': result['custody']['launch_id'] = 'forged'
                if mutation == 'echild': result['custody']['echild'] = False
                if mutation == 'cleanup': result['cleanup_errors'] = ['unreconciled']
                if mutation == 'unknown': result['custody']['status'] = 'unknown'
                self.output = self.root / mutation
                with mock.patch.object(runner.supervisor, 'run_process', return_value=result) as run, \
                        mock.patch.object(runner, 'sample', return_value=idle()), mock.patch.object(runner.time, 'sleep'):
                    with self.assertRaisesRegex(ValueError, 'Job failed'):
                        runner.execute(self.jobs, self.output, self.libraries, granted=True)
                    run.assert_called_once()
                self.assertFalse((self.output / 'must-not-run-supervisor.json').exists())
                release = json.loads((self.output / 'release.json').read_text())
                self.assertEqual(release['release'], 'released' if mutation in ('timeout', 'exit') else 'unverified')

    def test_busy_gpu_before_launch_and_after_drain_never_claim_release(self):
        busy = dict(idle(), apps='foreign PID')
        with self.assertRaisesRegex(ValueError, 'quiescence'):
            self.execute(sample=busy)
        self.assertEqual(json.loads((self.output / 'release.json').read_text())['release'], 'unverified')
        self.output = self.root / 'after-busy'
        with mock.patch.object(runner.supervisor, 'run_process', return_value=receipt()), \
                mock.patch.object(runner, 'sample', side_effect=[idle(), busy, busy, busy]), \
                mock.patch.object(runner.time, 'sleep'):
            with self.assertRaisesRegex(ValueError, 'release is unverified'):
                runner.execute(self.jobs, self.output, self.libraries, granted=True)

    def test_interrupt_preserves_supervisor_receipt_and_release(self):
        error = KeyboardInterrupt()
        error.supervisor_result = receipt()
        with mock.patch.object(runner.supervisor, 'run_process', side_effect=error), \
                mock.patch.object(runner, 'sample', return_value=idle()), mock.patch.object(runner.time, 'sleep'):
            with self.assertRaises(KeyboardInterrupt):
                runner.execute(self.jobs, self.output, self.libraries, granted=True)
        self.assertEqual(json.loads((self.output / 'cpu-supervisor.json').read_text()), error.supervisor_result)
        self.assertTrue((self.output / 'release.json').exists())

    def test_refuses_existing_receipts_without_launch(self):
        self.output.mkdir()
        sentinel = self.output / 'release.json'
        sentinel.write_text('historical')
        with mock.patch.object(runner.supervisor, 'run_process') as run:
            with self.assertRaises(FileExistsError):
                runner.execute(self.jobs, self.output, self.libraries, granted=True)
            run.assert_not_called()
        self.assertEqual(sentinel.read_text(), 'historical')

    def test_frozen_launch_drift_blocks_before_telemetry(self):
        for mutate in (lambda j: j['environment'].update(LD_LIBRARY_PATH='/changed'),
                       lambda j: j['argv'].append('changed'), lambda j: j.update(seconds=6)):
            jobs = copy.deepcopy(self.jobs)
            mutate(jobs[0])
            with mock.patch.object(runner, 'sample') as sample:
                with self.assertRaisesRegex(ValueError, 'Frozen argv/environment/timeout changed'):
                    runner.execute(jobs, self.output, self.libraries, granted=True)
                sample.assert_not_called()

    def test_bad_timeout_name_and_preload_rejected(self):
        for field, value in [('seconds', float('nan')), ('seconds', 0), ('seconds', 3601), ('name', '../escape')]:
            jobs = copy.deepcopy(self.jobs)
            jobs[0][field] = value
            with self.assertRaises(ValueError): runner.validate(jobs, self.libraries)
        with mock.patch.dict(os.environ, {'LD_PRELOAD': 'not-loaded'}):
            with self.assertRaisesRegex(ValueError, 'LD_PRELOAD'):
                runner.validate(self.jobs, self.libraries)

    @unittest.skipUnless(Path('/proc/self/stat').exists(), 'Linux custodian required')
    def test_real_python_only_child_has_authenticated_drain_and_hidden_devices(self):
        self.jobs[0]['argv'][-1] = 'import os; assert os.environ["CUDA_VISIBLE_DEVICES"] == ""; print("CPU only")'
        runner.pin_jobs(self.jobs, self.libraries)
        with mock.patch.object(runner, 'sample', return_value=idle()) as sample, mock.patch.object(runner.time, 'sleep'):
            self.assertEqual(runner.execute(self.jobs, self.output, self.libraries, granted=True), 0)
        result = json.loads((self.output / 'cpu-supervisor.json').read_text())
        self.assertTrue(runner.drained(result), result)
        self.assertIn('CPU only', result['stdout'])
        self.assertEqual(sample.call_count, 4)

    def test_attention_wrapper_and_all_preparers_use_maintained_runner(self):
        base = ROOT / 'crates/ferrule-backend/tests/attention_bf16'
        wrapper = (base / 'run-after-grant.sh').read_text()
        self.assertIn('scripts/run_pinned_jobs.py', wrapper)
        self.assertNotIn('owned.py', wrapper)
        self.assertIn('--sole-gpu-owner-granted', wrapper)
        for name in ('suite.py', 'prepared.py', 'matrix.py'):
            self.assertIn('write_jobs(out,jobs)', (base / name).read_text())
        result = subprocess.run(['sh', str(base / 'run-after-grant.sh')], capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 2)
        self.assertIn('Usage', result.stderr)


if __name__ == '__main__':
    unittest.main()
