"""Offline profile planning/parser contracts; no GPU, profiler or server launch."""

import importlib.util
import io
import json
from pathlib import Path
import tempfile
import os
import signal
import socket
import textwrap
import sqlite3
import subprocess
import sys
import time
import unittest
from unittest import mock

from test_bench_ferrule_vllm import bench
from test_bench_matrix import corpus, profiles

spec = importlib.util.spec_from_file_location('profile_ipc', Path(__file__).parents[1] / 'bench_process_ipc.py')
ipc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ipc)
LIMITS = dict(startup_seconds=300, request_seconds=180, measurement_seconds=180, teardown_seconds=120)


def profile_fixture(root):
    profile = profiles()[0]
    profile['metadata']['topology']['expert_policy']['rank_backend'] = 'process'
    profile['metadata']['precision']['profile'] = 'f32'
    profile['metadata']['cache']['prefix_mode'] = 'off'
    profile['argv'] = ['/usr/bin/env', '/original/ferrule', 'serve', '/fixture/model']
    binary, nsys = root / 'ferrule', root / 'nsys'
    for path in (binary, nsys):
        path.write_text('CPU FIXTURE NOT AN EXECUTABLE\n')
        path.chmod(0o700)
    config = bench.prepare_profiling([profile], binary, nsys)
    return profile, config


def counters():
    phases = {p: dict(calls=2, completed=2, completed_bytes=64, wall_ns=1_000_000_000) for p in ipc.PHASES}
    phases['copy']['wall_ns'] = 250_000_000
    phases['pipe_wait']['wall_ns'] = 500_000_000
    return dict(schema='ferrule.ipc-timing.v1', pid=101, thread='ThreadId(2)', phases=phases)


class ProfilePlanTests(unittest.TestCase):
    def test_bounded_plan_whole_lifecycle_and_changed_binary_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile, config = profile_fixture(root)
            plan = bench.build_manifest(corpus(), [profile], repetitions=1, requests=1, warmup=1,
                                        limits=LIMITS, profiling=config)
            self.assertEqual([c['id'] for c in plan['cells']], ['p128-o1-c1', 'p512-o1-c1', 'p2048-o1-c1'])
            self.assertEqual(len(plan['schedule']), 3)
            self.assertEqual(bench.supervisor_budget(plan, plan['cells'][2]), 825)
            self.assertEqual(profile['metadata']['quality_gate']['status'], 'unverified')
            self.assertEqual(profile['metadata']['model']['source_manifest_sha256'], 'unknown')
            commands = [bench.profiling_argv(config, profile, c, root, profile['argv']) for c in plan['cells']]
            self.assertEqual(sum('--duration=780' in argv for argv in commands), 3)
            self.assertIn('--kill=none', commands[2])
            self.assertIn('--sample=none', commands[2])
            self.assertIn('--delay=0', commands[2])
            Path(config['binary']).write_text('changed')
            with self.assertRaisesRegex(bench.ProtocolError, 'artifact changed'):
                bench.profiling_argv(config, profile, plan['cells'][0], root, profile['argv'])

    def test_reject_large_grid_repeat_and_wrong_topology(self):
        with tempfile.TemporaryDirectory() as directory:
            profile, config = profile_fixture(Path(directory))
            for kwargs in (dict(repetitions=2), dict(requests=2), dict(warmup=0),
                           dict(cell_ids=['p8192-o1-c1']), dict(limits=dict(LIMITS, request_seconds=181))):
                args = dict(repetitions=1, requests=1, warmup=1, limits=LIMITS, profiling=config)
                args.update(kwargs)
                with self.subTest(kwargs=kwargs), self.assertRaises(bench.ProtocolError):
                    bench.build_manifest(corpus(), [profile], **args)
            profile['metadata']['topology']['tp'] = 4
            with self.assertRaises(bench.ProtocolError):
                bench.validate_profiling(config, profile)

    def test_worker_omits_first_request_and_reports_missing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile, config = profile_fixture(root)
            plan = bench.build_manifest(corpus(), [profile], repetitions=1, requests=1, warmup=1,
                                        limits=LIMITS, profiling=config)
            with mock.patch.object(bench.os, 'getpgrp', return_value=bench.os.getpid()), \
                    mock.patch.object(bench, 'own_group_members', return_value=[]), \
                    mock.patch.object(bench.socket, 'socket'), \
                    mock.patch.object(bench.subprocess, 'Popen') as popen, \
                    mock.patch.object(bench, 'wait_ready', return_value=1), \
                    mock.patch.object(bench, 'main', return_value=0) as main, \
                    mock.patch.object(bench, 'shutdown_profile_capture', return_value=dict(timed_out=False, returncode=0)), \
                    mock.patch.object(bench, 'supervisor_module', return_value=mock.Mock(scrub=lambda text: text)):
                self.assertEqual(bench.matrix_worker(dict(manifest=plan, profile=profile,
                    cell=plan['cells'][2], directory=str(root))), 1)
            argv = main.call_args.args[0]
            self.assertNotIn('--first-after-ready', argv)
            for flag in ('--max-tokens', '--warmup', '--requests', '--concurrency'):
                self.assertEqual(argv[argv.index(flag) + 1], '1')
            self.assertEqual(argv[argv.index('--timeout') + 1], '180')
            self.assertIn('--duration=780', popen.call_args.args[0])
            lifecycle = json.loads((root / 'lifecycle.json').read_text())
            self.assertEqual(lifecycle['work_status'], 'complete')
            self.assertEqual(lifecycle['status'], 'profile_incomplete')

    def test_request_timeout_stops_schedule_even_with_clean_custody(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile, config = profile_fixture(root)
            plan = bench.build_manifest(corpus(), [profile], repetitions=1, requests=1, warmup=1,
                                        limits=LIMITS, profiling=config)
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps(plan))
            supervisor = mock.Mock()
            supervisor.custody_capabilities.return_value = dict(containment=True)
            def failed(argv, budget, child):
                target = Path(child['directory'])
                (target / 'lifecycle.json').write_text(json.dumps(dict(status='request_failure',
                    phases=dict(shutdown=dict(timed_out=False)))))
                return dict(returncode=1, timed_out=False, custody=dict(status='proven_drained'))
            supervisor.run_process.side_effect = failed
            with mock.patch.object(bench, 'supervisor_module', return_value=supervisor):
                self.assertEqual(bench.main(['matrix-run', '--manifest', str(manifest), '--output',
                    str(root / 'run'), '--execute', '--gpu-owner-ready']), 1)
            self.assertEqual(supervisor.run_process.call_count, 1)
            result = json.loads((root / 'run/matrix.json').read_text())
            self.assertEqual(result['runs'][1]['status'], 'not_run')

    def test_cli_plan_and_dry_run_never_invoke_supervisor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile, config = profile_fixture(root)
            (root / 'corpus.json').write_text(json.dumps(corpus()))
            (root / 'profiles.json').write_text(json.dumps([profile]))
            with mock.patch.object(bench, 'supervisor_module', side_effect=AssertionError('offline')), \
                    mock.patch.object(bench.subprocess, 'Popen', side_effect=AssertionError('no profiler')), \
                    mock.patch('sys.stdout', new=io.StringIO()):
                self.assertEqual(bench.main(['matrix-plan', '--corpus', str(root / 'corpus.json'),
                    '--profiles', str(root / 'profiles.json'), '--profiling-binary', config['binary'],
                    '--nsys', config['nsys'], '--cell', 'p2048-o1-c1', '--process-runs', '1',
                    '--requests', '1', '--warmup', '1', '--output', str(root / 'plan.json')]), 0)
                self.assertEqual(bench.main(['matrix-run', '--manifest', str(root / 'plan.json'),
                    '--output', str(root / 'dry')]), 0)
            result = json.loads((root / 'dry/matrix.json').read_text())
            self.assertEqual(len(result['runs']), 1)
            self.assertFalse(result['executed'])
            self.assertEqual(result['runs'][0]['cell_id'], 'p2048-o1-c1')

    def test_gpu_handoff_required_before_supervisor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile, config = profile_fixture(root)
            plan = bench.build_manifest(corpus(), [profile], repetitions=1, requests=1, warmup=1,
                                        limits=LIMITS, profiling=config)
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps(plan))
            with mock.patch.object(bench, 'supervisor_module', side_effect=AssertionError('no launch')), \
                    mock.patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
                bench.main(['matrix-run', '--manifest', str(manifest), '--output', str(root / 'run'), '--execute'])


class ProfileParserTests(unittest.TestCase):
    def test_units_overlap_failed_calls_and_missing_not_zero(self):
        record = counters()
        record['phases']['encode']['completed'] = 1
        parsed = ipc.summarize_ipc([record])
        self.assertEqual(parsed['phases']['encode']['inclusive_thread_wall_seconds'], 1)
        self.assertEqual(parsed['phases']['copy']['inclusive_thread_wall_seconds'], .25)
        self.assertEqual(parsed['phases']['encode']['incomplete_calls'], 1)
        self.assertEqual(parsed['phases']['encode']['completed_bytes'], 64)
        self.assertIsNone(parsed['kernel_fraction'])
        self.assertEqual(ipc.summarize_ipc([])['status'], 'missing_not_zero')

    def test_reject_duplicate_missing_negative_fractional_and_bool_counters(self):
        with self.assertRaises(ValueError):
            ipc.summarize_ipc([counters(), counters()])
        for value in (-1, 1.5, True):
            record = counters()
            record['phases']['encode']['wall_ns'] = value
            with self.assertRaises(ValueError):
                ipc.summarize_ipc([record])
        record = counters()
        del record['phases']['copy']
        with self.assertRaises(ValueError):
            ipc.summarize_ipc([record])

    def test_nsys_duration_sums_never_percent_or_wall_share(self):
        rows = ipc.parse_nsys_summary('"Time (%)","Total Time (ns)","Instances","Name"\n'
                                      '95.0,2000000000,3,"gated_delta_kernel<float, 8>"\n')
        self.assertEqual(rows[0]['duration_sum_seconds'], 2)
        self.assertEqual(rows[0]['calls'], 3)
        self.assertIsNone(rows[0]['wall_fraction'])
        self.assertNotIn('Time (%)', rows[0])
        with self.assertRaises(ValueError):
            ipc.parse_nsys_summary('Total Time (us),Name\n20,kernel\n')
        with self.assertRaises(ValueError):
            ipc.parse_nsys_summary('Total Time (ns),Name\n-1,kernel\n')

    def test_artifacts_bound_to_matrix_pins_and_missing_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = root / 'matrix.json'
            matrix.write_text(json.dumps(dict(schema='ferrule.http_matrix.v1', manifest_sha256='a'*64,
                manifest=dict(profiling=dict(pins={'/fixture/ferrule': 'b'*64})),
                runs=[dict(index=0, status='request_failure')])))
            run = root / 'run-00000'
            run.mkdir()
            (run / 'ipc-timing.jsonl').write_text(json.dumps(counters()) + '\n')
            output = root / 'summary.json'
            self.assertEqual(ipc.summarize_main(['--matrix', str(matrix), '--output', str(output)]), 0)
            result = json.loads(output.read_text())
            self.assertEqual(result['pins']['/fixture/ferrule'], 'b'*64)
            self.assertEqual(result['custody']['status'], 'unknown')
            self.assertEqual(result['matrix_sha256'], ipc.sha256(matrix))
            self.assertTrue(result['metrics_missing'])


class CaptureExecutionTests(unittest.TestCase):
    def test_explicit_600s_budget_fits_single_1800s_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile, config = profile_fixture(root)
            config['request_seconds'] = 600
            limits = dict(LIMITS, request_seconds=600, measurement_seconds=600)
            plan = bench.build_manifest(corpus(), [profile], repetitions=1, requests=1, warmup=1,
                limits=limits, profiling=config, cell_ids=['p2048-o1-c1'])
            self.assertEqual(bench.supervisor_budget(plan, plan['cells'][0]), 1665)
            argv = bench.profiling_argv(config, profile, plan['cells'][0], root, profile['argv'])
            self.assertIn('--duration=1620', argv)
            self.assertIn('--delay=0', argv)
            self.assertEqual(argv.count('/usr/bin/env'), 1)
            self.assertEqual(argv[-3:], [config['binary'], 'serve', '/fixture/model'])

    def test_environment_outside_nsys_preserves_injection_and_rejects_env_i(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile, config = profile_fixture(root)
            profile['argv'].insert(1, 'LD_LIBRARY_PATH=/fixture/cuda')
            argv = bench.profiling_argv(config, profile, dict(prefill_tokens=128), root, profile['argv'])
            self.assertLess(argv.index('LD_LIBRARY_PATH=/fixture/cuda'), argv.index(config['nsys']))
            profile['argv'].insert(1, '-i')
            with self.assertRaisesRegex(bench.ProtocolError, 'explicit environment'):
                bench.profiling_argv(config, profile, dict(prefill_tokens=128), root, profile['argv'])

    def test_stop_marker_waits_for_export_never_signals_profiler(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = mock.Mock()
            process.wait.return_value = 0
            result = bench.shutdown_profile_capture(process, root, 120)
            self.assertTrue((root / 'profile-stop').exists())
            self.assertFalse(result['timed_out'])
            process.terminate.assert_not_called()
            process.kill.assert_not_called()
            process.wait.assert_called_once_with(timeout=120)
        with tempfile.TemporaryDirectory() as directory:
            process.wait.side_effect = subprocess.TimeoutExpired('fixture', 120)
            result = bench.shutdown_profile_capture(process, Path(directory), 120)
            self.assertTrue(result['timed_out'])
            process.terminate.assert_not_called()
            process.kill.assert_not_called()

    def test_worker_preserves_primary_timeout_and_persistent_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile, config = profile_fixture(root)
            plan = bench.build_manifest(corpus(), [profile], repetitions=1, requests=1, warmup=1,
                                        limits=LIMITS, profiling=config)
            def spawn(argv, **kwargs):
                os.write(kwargs['stdout'].fileno(), b'partial profiler data\n')
                return mock.Mock()
            with mock.patch.object(bench.os, 'getpgrp', return_value=bench.os.getpid()), \
                    mock.patch.object(bench, 'own_group_members', return_value=[]), \
                    mock.patch.object(bench.socket, 'socket'), \
                    mock.patch.object(bench.subprocess, 'Popen', side_effect=spawn), \
                    mock.patch.object(bench, 'wait_ready', return_value=1), \
                    mock.patch.object(bench, 'main', side_effect=TimeoutError('original warmup timeout')), \
                    mock.patch.object(bench, 'shutdown_profile_capture', side_effect=OSError('export error')), \
                    mock.patch.object(bench, 'supervisor_module', return_value=mock.Mock(scrub=lambda text: text)):
                self.assertEqual(bench.matrix_worker(dict(manifest=plan, profile=profile,
                    cell=plan['cells'][0], directory=str(root))), 1)
            life = json.loads((root / 'lifecycle.json').read_text())
            self.assertEqual(life['work_status'], 'timeout')
            self.assertIn('original warmup timeout', life['error'])
            self.assertEqual(life['phases']['shutdown']['error'], 'export error')
            self.assertIn(b'partial profiler data', (root / 'server.log').read_bytes())

    def test_real_cpu_child_flushes_on_cooperative_stop(self):
        # CPU-only executable, not nsys or a model. Its child emits only on TERM,
        # exercising the same exit-time sink constraint as Rust TLS counters.
        wrapper = Path(__file__).parents[1] / '_profile_capture.py'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / 'cpu_server.py'
            fixture.write_text("import os, signal, sys, time\nfrom pathlib import Path\n"
                "def stop(sig, frame):\n"
                "    Path(os.environ['FERRULE_PROCESS_IPC_TIMING']).write_text('flushed')\n"
                "    sys.exit(0)\n"
                "signal.signal(signal.SIGTERM, stop)\n"
                "Path(sys.argv[1]).write_text(os.environ['PROFILE_ENV_FIXTURE'])\n"
                "time.sleep(10)\n")
            ready = root / 'ready'
            env = dict(os.environ, FERRULE_PROCESS_IPC_TIMING=str(root / 'ipc-timing.jsonl'),
                       PROFILE_ENV_FIXTURE='inherited', CUDA_VISIBLE_DEVICES='')
            process = subprocess.Popen([sys.executable, '-B', str(wrapper), '--directory', str(root),
                '--lifetime-seconds', '5', '--shutdown-seconds', '2', '--', sys.executable,
                '-B', str(fixture), str(ready)], env=env)
            try:
                deadline = time.monotonic() + 4
                while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertTrue(ready.exists())
                self.assertEqual(ready.read_text(), 'inherited')
                (root / 'profile-stop').touch()
                self.assertEqual(process.wait(timeout=5), 0)
                self.assertEqual((root / 'ipc-timing.jsonl').read_text(), 'flushed')
                report = json.loads((root / 'profile-server.json').read_text())
                self.assertEqual(report['stop_reason'], 'matrix_worker_stop_marker')
                self.assertEqual(report['returncode'], 0)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)


    @unittest.skipUnless(sys.platform == 'linux', 'verified Linux subreaper required')
    def test_cpu_fake_profiler_http_counter_export_and_real_custody(self):
        if not bench.supervisor_module().custody_capabilities().get('containment'):
            self.skipTest('subreaper capability unavailable')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with socket.socket() as reservation:
                reservation.bind(('127.0.0.1', 0))
                port = reservation.getsockname()[1]
            server = root / 'cpu-server'
            server.write_text('#!' + sys.executable + '\n' + textwrap.dedent("""
                import http.server, json, os, signal, sys
                def stop(sig, frame):
                    record = RECORD
                    record['pid'] = os.getpid()
                    with open(os.environ['FERRULE_PROCESS_IPC_TIMING'], 'w') as sink:
                        sink.write(json.dumps(record) + '\\n')
                    sys.exit(0)
                signal.signal(signal.SIGTERM, stop)
                class Handler(http.server.BaseHTTPRequestHandler):
                    def log_message(self, *args): pass
                    def do_GET(self):
                        body = json.dumps({'data':[{'id':'mock-model'}]}).encode()
                        self.send_response(200); self.send_header('Content-Length', str(len(body)))
                        self.end_headers(); self.wfile.write(body)
                    def do_POST(self):
                        self.rfile.read(int(self.headers['Content-Length']))
                        chunks = [{'choices':[{'index':0,'text':'CPU','finish_reason':None}]},
                                  {'choices':[{'index':0,'text':'','finish_reason':'length'}]},
                                  {'choices':[], 'usage':{'prompt_tokens':128,'completion_tokens':1,'total_tokens':129}}]
                        body = (''.join('data: '+json.dumps(c)+'\\n\\n' for c in chunks)+'data: [DONE]\\n\\n').encode()
                        self.send_response(200); self.send_header('Content-Type', 'text/event-stream')
                        self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
                http.server.HTTPServer(('127.0.0.1', int(sys.argv[-1])), Handler).serve_forever()
            """).replace('RECORD', repr(counters())))
            profiler = root / 'cpu-profiler'
            profiler.write_text('#!' + sys.executable + '\n' + textwrap.dedent("""
                import pathlib, subprocess, sys
                target = sys.argv.index('-B') - 1
                code = subprocess.call(sys.argv[target:])
                output = next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('--output='))
                pathlib.Path(output + '.nsys-rep').write_text('CPU FIXTURE ONLY, NOT A GPU TRACE')
                sys.exit(code)
            """))
            for path in (server, profiler):
                path.chmod(0o700)
            profile = profiles()[0]
            profile['metadata']['topology']['expert_policy']['rank_backend'] = 'process'
            profile['metadata']['precision']['profile'] = 'f32'
            profile['metadata']['cache']['prefix_mode'] = 'off'
            profile['argv'] = ['/usr/bin/env', 'CUDA_VISIBLE_DEVICES=', str(server), 'serve', str(port)]
            profile['url'] = f'http://127.0.0.1:{port}/v1/completions'
            config = bench.prepare_profiling([profile], server, profiler)
            plan = bench.build_manifest(corpus(), [profile], repetitions=1, requests=1, warmup=1,
                limits=LIMITS, profiling=config, cell_ids=['p128-o1-c1'])
            manifest = root / 'plan.json'
            manifest.write_text(json.dumps(plan))
            # A separate outer timeout bounds a broken CPU fixture without
            # depending on the much larger production profile budgets.
            result = subprocess.run([sys.executable, '-B', str(bench.SCRIPT), 'matrix-run',
                '--manifest', str(manifest), '--output', str(root / 'run'),
                '--execute', '--gpu-owner-ready'], timeout=30, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            matrix = json.loads((root / 'run/matrix.json').read_text())
            slot = matrix['runs'][0]
            self.assertEqual(slot['supervisor']['custody']['status'], 'proven_drained')
            self.assertTrue(slot['supervisor']['custody']['echild'])
            run = root / 'run/run-00000'
            self.assertTrue((run / 'nsys.nsys-rep').is_file())
            self.assertTrue(ipc.read_profile_ipc(run / 'ipc-timing.jsonl')['nonzero_transport_observed'])
            self.assertEqual(json.loads((run / 'http.json').read_text())['status'], 'complete')


class CoverageTests(unittest.TestCase):
    def test_partial_sink_preserves_valid_lines_without_fake_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ipc.jsonl'
            path.write_text(json.dumps(counters()) + '\n{"schema":')
            result = ipc.read_profile_ipc(path)
            self.assertEqual(result['status'], 'partial_invalid_not_zero')
            self.assertEqual(result['pids'], [101])
            self.assertEqual(result['parse_errors'][0]['line'], 2)
            self.assertTrue(result['nonzero_transport_observed'])
            path.write_text(json.dumps(counters()) + '\n' + json.dumps(counters()) + '\n')
            result = ipc.read_profile_ipc(path)
            self.assertEqual(result['threads'], 1)
            self.assertIn('Duplicate', result['parse_errors'][0]['error'])

    def test_sqlite_coverage_is_events_not_wall_fraction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.sqlite'
            with sqlite3.connect(path) as db:
                db.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INT, end INT, deviceId INT, globalPid INT)')
                db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?)',
                               [(0, 20, 0, 123), (10, 30, 0, 123), (20, 40, 1, 456)])
            result = ipc.sqlite_gpu_coverage(path)
            self.assertEqual(result['status'], 'gpu_events_observed_not_request_scoped')
            self.assertEqual(result['kernel_devices'], [0, 1])
            row = result['tables']['CUPTI_ACTIVITY_KIND_KERNEL'][0]
            self.assertEqual(row['duration_sum_ns'], 40)
            self.assertEqual(row['last_end_ns'], 30)
            self.assertIsNone(row['wall_fraction'])
            with sqlite3.connect(path) as db:
                db.execute('DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL')
            self.assertEqual(ipc.sqlite_gpu_coverage(path)['status'], 'missing_not_zero')

    def test_require_artifacts_saves_failed_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = root / 'matrix.json'
            matrix.write_text(json.dumps(dict(schema='ferrule.http_matrix.v1', manifest_sha256='a'*64,
                manifest=dict(profiling=dict(pins={})), runs=[dict(index=0, status='request_failure')])))
            result_path = root / 'summary.json'
            self.assertEqual(ipc.summarize_main(['--matrix', str(matrix), '--output', str(result_path),
                                               '--require-artifacts']), 1)
            self.assertFalse(json.loads(result_path.read_text())['artifact_smoke_pass'])



if __name__ == '__main__':
    unittest.main()
