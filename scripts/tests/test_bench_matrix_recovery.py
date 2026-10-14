"""Durable interrupted-matrix contracts. Mock processes only; no services or GPUs."""
import io
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest import mock

from test_bench_ferrule_vllm import bench, record
from test_bench_matrix import corpus, profiles, http_report


class MatrixRecoveryTests(unittest.TestCase):
    def fixture(self):
        return bench.build_manifest(corpus(), profiles()[:1], repetitions=2, requests=2, warmup=2,
                                    cell_ids=['p8192-o32-c1'])

    def execute(self, root, manifest, supervisor):
        (root/'manifest.json').write_text(json.dumps(manifest))
        with mock.patch.object(bench, 'supervisor_module', return_value=supervisor), \
                mock.patch.object(bench.subprocess, 'Popen', side_effect=AssertionError('no processes')):
            code = bench.main(['matrix-run', '--manifest', str(root/'manifest.json'),
                               '--output', str(root/'run'), '--execute'])
        return code, json.loads((root/'run/matrix.json').read_text())

    def test_second_run_outer_signal_checkpoint_precedes_spawn_and_keeps_first(self):
        manifest = self.fixture()
        supervisor = mock.Mock()
        supervisor.custody_capabilities.return_value = dict(containment=True)  # mock proof, never host authority
        prior_handler = signal.getsignal(signal.SIGTERM)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []
            def run(argv, budget, config):
                snapshot = json.loads((root/'run/matrix.json').read_text())
                index = len(calls)
                calls.append(snapshot)
                self.assertEqual(snapshot['status'], 'in_progress')
                self.assertFalse(snapshot['finalized'])
                self.assertEqual(snapshot['runs'][index]['status'], 'in_progress')
                self.assertIn('summaries', snapshot)
                child = Path(config['directory'])
                if index == 0:
                    http = http_report()
                    http.update(measurements=dict(work_completed=True))
                    http['aggregate'].update(success=2, offered=2, not_issued=0)
                    (child/'http.json').write_text(json.dumps(http))
                    (child/'lifecycle.json').write_text(json.dumps(dict(status='complete', phases=dict(
                        shutdown=dict(status='drained_process_group')))))
                    return dict(returncode=0, timed_out=False, custody=dict(status='proven_drained'))
                (child/'http.json').write_text('')
                (child/'lifecycle.json').write_text(json.dumps(dict(status='warmup_measure', phases={})))
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            supervisor.run_process.side_effect = run
            code, report = self.execute(root, manifest, supervisor)
            self.assertEqual((root/'run/run-00001/http.json').read_text(), '')
        self.assertEqual(code, 1)
        self.assertEqual(signal.getsignal(signal.SIGTERM), prior_handler)
        self.assertEqual(len(calls), 2)
        self.assertEqual(report['status'], 'incomplete')
        self.assertTrue(report['finalized'])
        self.assertEqual(report['runs'][0]['status'], 'complete')
        self.assertEqual(report['runs'][1]['status'], 'interrupted')
        self.assertIn('SIGTERM', report['runs'][1]['error'])
        self.assertTrue(report['runs'][1]['artifact_errors'])
        summary = report['summaries'][0]
        self.assertEqual(summary['scheduled_requests'], 4)
        self.assertEqual(summary['successful'], 2)
        self.assertEqual(summary['unobserved_request_budget'], 2)
        self.assertEqual(summary['not_issued_known'], 0)
        self.assertIsNone(summary['throughput_ci']['estimate'])

    def test_no_containment_capability_blocks_before_any_server_spawn(self):
        supervisor = mock.Mock()
        supervisor.custody_capabilities.return_value = dict(containment=False, pidfd=False)
        with tempfile.TemporaryDirectory() as directory:
            code, report = self.execute(Path(directory), self.fixture(), supervisor)
        self.assertEqual(code, 1)
        supervisor.run_process.assert_not_called()
        self.assertTrue(all(run['status'] == 'blocked' for run in report['runs']))
        self.assertTrue(all('containment unavailable' in ';'.join(run['reasons']) for run in report['runs']))

    def test_group_empty_or_missing_custody_never_allows_next_restart(self):
        for custody in (None, dict(status='unknown', registered_drained=True)):
            with self.subTest(custody=custody), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                supervisor = mock.Mock()
                supervisor.custody_capabilities.return_value = dict(containment=True)  # mock proof, never host authority
                def run(argv, budget, config):
                    child = Path(config['directory'])
                    http = http_report()
                    http['measurements'] = dict(work_completed=True)
                    (child/'http.json').write_text(json.dumps(http))
                    (child/'lifecycle.json').write_text(json.dumps(dict(status='complete', phases=dict(
                        shutdown=dict(status='drained_process_group')))))
                    return dict(returncode=0, timed_out=False, **({'custody': custody} if custody else {}))
                supervisor.run_process.side_effect = run
                code, report = self.execute(root, self.fixture(), supervisor)
                self.assertEqual(code, 1)
                supervisor.run_process.assert_called_once()
                self.assertEqual(report['runs'][0]['status'], 'cleanup_unknown')
                self.assertEqual(report['runs'][0]['work_status'], 'complete')
                self.assertEqual(report['runs'][1]['status'], 'not_run')
                self.assertTrue(report['halted_for_cleanup'])
                self.assertIsNone(report['summaries'][0]['throughput_ci']['estimate'])

    def test_timeout_survives_truncated_missing_keys_and_secondary_summary_error(self):
        for raw in ('', '{', '{}', '[]', '{"schema":"ferrule.http_bench.v1"}'):
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                supervisor = mock.Mock()
                supervisor.custody_capabilities.return_value = dict(containment=True)  # mock proof, never host authority
                def run(argv, budget, config):
                    (Path(config['directory'])/'http.json').write_text(raw)
                    return dict(returncode=-9, timed_out=True, primary_error='TimeoutExpired: original')
                supervisor.run_process.side_effect = run
                with mock.patch.object(bench, 'matrix_comparisons', side_effect=KeyError('secondary')):
                    code, report = self.execute(root, self.fixture(), supervisor)
                self.assertEqual(code, 1)
                self.assertEqual(report['runs'][0]['status'], 'supervisor_timeout')
                self.assertIn('original', report['runs'][0]['supervisor']['primary_error'])
                self.assertTrue(report['runs'][0]['artifact_errors'])
                self.assertEqual(report['runs'][1]['status'], 'not_run')
                self.assertTrue(report['reporting_errors'])
                self.assertIn('summaries', report)
                self.assertEqual(report['comparisons'], [])
                self.assertEqual(report['summaries'][0]['not_issued_known'], 2)
                self.assertEqual(report['summaries'][0]['unobserved_request_budget'], 2)

    def test_comparison_missing_or_malformed_slot_is_ineligible_not_keyerror(self):
        manifest = self.fixture()
        runs = [dict(profile_id='ferrule', cell_id='p8192-o32-c1', repetition=0, status='complete',
                     http=http_report())]
        other = dict(manifest['profiles'][0], id='other')
        manifest['profiles'].append(other)
        result = bench.matrix_comparisons(manifest, runs)
        self.assertFalse(result[0]['comparability'])
        self.assertTrue(any('missing' in reason for reason in result[0]['reasons']))

    def test_summary_exception_is_recorded_not_raised_over_interruption(self):
        supervisor = mock.Mock()
        supervisor.custody_capabilities.return_value = dict(containment=True)  # mock proof, never host authority
        supervisor.run_process.side_effect = KeyboardInterrupt('owner cutoff')
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(bench, 'summarize_matrix', side_effect=KeyError('summaries')):
            code, report = self.execute(Path(directory), self.fixture(), supervisor)
        self.assertEqual(code, 1)
        self.assertIn('owner cutoff', report['runs'][0]['error'])
        self.assertEqual(report['summaries'], [])
        self.assertTrue(report['reporting_errors'])

    def test_failed_atomic_replace_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'matrix.json'
            bench.dump_json(path, dict(status='in_progress', summaries=[]))
            before = path.read_bytes()
            with mock.patch.object(bench.Path, 'replace', side_effect=OSError('disk error')):
                with self.assertRaisesRegex(OSError, 'disk error'):
                    bench.dump_json(path, dict(status='complete'))
            self.assertEqual(path.read_bytes(), before)


class PartialHttpTests(unittest.TestCase):
    def test_warmup_and_inflight_measure_are_durable_and_unknown_is_not_notissued(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'http.json'
            client = mock.Mock()
            warmup_snapshots = []
            def request(config, identity, cutoff=None):
                warmup_snapshots.append(json.loads(path.read_text()))
                return dict(record(identity['index']), population=identity['population'])
            client.request.side_effect = request
            def interrupted_measure(clients, args, corpus, repetition, progress):
                before = json.loads(path.read_text())
                self.assertEqual(before['aggregate']['not_issued'], 0)
                self.assertEqual(before['aggregate']['unobserved_request_budget'], 2)
                progress([record()])
                after = json.loads(path.read_text())
                self.assertEqual(after['aggregate']['success'], 1)
                self.assertEqual(after['aggregate']['unobserved_request_budget'], 1)
                self.assertIsNone(after['aggregate']['successful_output_tokens_per_second'])
                raise KeyboardInterrupt('cutoff with one unobserved request')
            with mock.patch.object(bench, 'Client', return_value=client), \
                    mock.patch.object(bench, 'measure', side_effect=interrupted_measure), \
                    mock.patch('sys.stdout', new=io.StringIO()):
                code = bench.main(['--engine', 'ferrule', '--url', 'http://example.invalid/v1/completions',
                    '--model', 'mock', '--output', str(path), '--window-kind', 'fixed-work',
                    '--warmup', '2', '--repeat', '1', '--requests', '2', '--first-after-ready'])
            report = json.loads(path.read_text())
        self.assertEqual(code, 1)
        self.assertTrue(report['finalized'])
        self.assertEqual(len(warmup_snapshots[-1]['warmups']), 1)
        self.assertIn('first_after_ready', warmup_snapshots[1])
        self.assertEqual(report['aggregate']['success'], 1)
        self.assertEqual(report['aggregate']['not_issued'], 0)
        self.assertEqual(report['aggregate']['unobserved_request_budget'], 1)
        self.assertIn('cutoff', report['errors'][0])
        self.assertEqual(report['status'], 'incomplete')


class CustodyGateTests(unittest.TestCase):
    def test_process_group_attestation_is_not_a_gate_or_a_custody_proof(self):
        manifest = bench.build_manifest(corpus(), profiles(), cell_ids=['p128-o32-c1'])
        for profile in manifest['profiles']:
            profile['metadata']['topology']['expert_policy']['rank_backend'] = 'process'
            profile['argv'].append('--rank-backend=process')
            profile['ownership']['all_workers_in_process_group'] = False
            self.assertEqual(bench.cell_blockers(manifest, profile, manifest['cells'][0]), [])
        manifest = bench.build_manifest(corpus(), manifest['profiles'][:1], cell_ids=['p128-o32-c1'])
        manifest['profiles'][0]['ownership']['containment'] = True
        supervisor = mock.Mock()
        supervisor.custody_capabilities.return_value = dict(containment=False)
        with tempfile.TemporaryDirectory() as directory:
            code, report = MatrixRecoveryTests().execute(Path(directory), manifest, supervisor)
        self.assertEqual(code, 1)
        supervisor.run_process.assert_not_called()
        self.assertTrue(all(row['status'] == 'blocked' for row in report['runs']))

    def test_reparented_same_group_detected_but_foreign_and_separate_groups_not_targeted(self):
        # Parent identity changes to init; membership is still the owned group.
        owned = mock.Mock()
        owned.parent.name = '333'
        owned.read_text.return_value = '333 (worker) S 1 222 222 0'
        escaped = mock.Mock()
        escaped.parent.name = '444'
        escaped.read_text.return_value = '444 (same name) S 1 444 444 0'
        foreign = mock.Mock()
        foreign.parent.name = '555'
        foreign.read_text.return_value = '555 (worker) S 1 555 555 0'
        with mock.patch.object(bench, 'Path') as path, \
                mock.patch.object(bench.os, 'getpid', return_value=222), \
                mock.patch.object(bench.os, 'getpgrp', return_value=222), \
                mock.patch.object(bench.os, 'kill') as kill, mock.patch.object(bench.os, 'killpg') as killpg:
            path.return_value.is_file.return_value = True
            path.return_value.glob.return_value = [owned, escaped, foreign]
            self.assertEqual(bench.own_group_members(), [333])
            kill.assert_not_called()
            killpg.assert_not_called()

    def test_worker_keeps_primary_timeout_when_cleanup_and_log_reporting_fail(self):
        manifest = bench.build_manifest(corpus(), profiles()[:1], cell_ids=['p128-o32-c1'])
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(bench.os, 'getpid', return_value=222), \
                mock.patch.object(bench.os, 'getpgrp', return_value=222), \
                mock.patch.object(bench, 'own_group_members', return_value=[]), \
                mock.patch.object(bench.socket, 'socket'), \
                mock.patch.object(bench.subprocess, 'Popen'), \
                mock.patch.object(bench, 'wait_ready', side_effect=TimeoutError('original startup deadline')), \
                mock.patch.object(bench, 'shutdown_owned_group', side_effect=PermissionError('cleanup denied')), \
                mock.patch.object(bench, 'supervisor_module', side_effect=RuntimeError('log extraction')):
            code = bench.matrix_worker(dict(manifest=manifest, profile=manifest['profiles'][0],
                                            cell=manifest['cells'][0], directory=directory))
            report = json.loads((Path(directory)/'lifecycle.json').read_text())
        self.assertEqual(code, 1)
        self.assertEqual(report['work_status'], 'timeout')
        self.assertIn('original startup deadline', report['error'])
        self.assertIn('cleanup denied', report['phases']['shutdown']['error'])
        self.assertTrue(report['reporting_errors'])


class RealCpuProcessMatrixTests(unittest.TestCase):
    def test_existing_foreign_listener_is_never_replaced_or_signalled(self):
        import socket
        import sys
        with tempfile.TemporaryDirectory() as directory, socket.socket() as foreign:
            root = Path(directory)
            marker = root/'must-not-spawn'
            foreign.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            foreign.bind(('127.0.0.1', 0))
            foreign.listen(4)
            port = foreign.getsockname()[1]
            profile = profiles()[0]
            profile['url'] = f'http://127.0.0.1:{port}/v1/completions'
            profile['argv'] = [sys.executable, '-B', '-c',
                              'import pathlib,sys; pathlib.Path(sys.argv[1]).touch()', str(marker)]
            manifest = bench.build_manifest(corpus(), [profile], repetitions=2, requests=1,
                                            cell_ids=['p128-o32-c1'])
            (root/'manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(bench.main(['matrix-run', '--manifest', str(root/'manifest.json'),
                             '--output', str(root/'run'), '--execute']), 1)
            self.assertFalse(marker.exists())
            report = json.loads((root/'run/matrix.json').read_text())
            self.assertTrue(all(r['lifecycle']['phases']['shutdown']['status'] == 'not_spawned'
                                for r in report['runs']))
            with socket.create_connection(('127.0.0.1', port), timeout=1):
                pass

    def test_two_process_profile_restarts_require_real_subreaper_handshake_and_drain(self):
        import socket
        import sys
        import textwrap
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1', 0))
            port = reservation.getsockname()[1]
        # HTTP protocol fixture only: no inference, model or GPU imports. The
        # server intentionally leaves a separate-session descendant to adopt.
        server = textwrap.dedent("""
            import http.server,json,subprocess,sys,time
            subprocess.Popen([sys.executable,'-B','-c',
                'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(4)'],
                start_new_session=True)
            class Handler(http.server.BaseHTTPRequestHandler):
                def log_message(self,*args): pass
                def do_GET(self):
                    body=json.dumps({'data':[{'id':'mock-model'}]}).encode()
                    self.send_response(200); self.send_header('Content-Length',str(len(body)))
                    self.end_headers(); self.wfile.write(body)
                def do_POST(self):
                    self.rfile.read(int(self.headers['Content-Length']))
                    chunks=[{'choices':[{'index':0,'text':'CPU fixture','finish_reason':None}]},
                            {'choices':[{'index':0,'text':'','finish_reason':'length'}]},
                            {'choices':[], 'usage':{'prompt_tokens':128,'completion_tokens':32,'total_tokens':160}}]
                    body=(''.join('data: '+json.dumps(c)+'\\n\\n' for c in chunks)+'data: [DONE]\\n\\n').encode()
                    self.send_response(200); self.send_header('Content-Type','text/event-stream')
                    self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
            http.server.HTTPServer(('127.0.0.1',int(sys.argv[1])),Handler).serve_forever()
        """)
        profile = profiles()[0]
        profile['argv'] = [sys.executable, '-B', '-c', server, str(port)]
        profile['url'] = f'http://127.0.0.1:{port}/v1/completions'
        profile['metadata']['topology']['expert_policy']['rank_backend'] = 'process'
        profile['ownership']['all_workers_in_process_group'] = False
        manifest = bench.build_manifest(corpus(), [profile], repetitions=2, requests=2, warmup=2,
                    cell_ids=['p128-o32-c1'], limits=dict(startup_seconds=5, request_seconds=2,
                    measurement_seconds=5, teardown_seconds=2))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'manifest.json').write_text(json.dumps(manifest))
            code = bench.main(['matrix-run', '--manifest', str(root/'manifest.json'),
                               '--output', str(root/'run'), '--execute'])
            report = json.loads((root/'run/matrix.json').read_text())
            self.assertEqual(code, 0, [(r['status'], r.get('error'), r.get('lifecycle', {}).get('error'),
                                       r.get('supervisor', {}).get('error')) for r in report['runs']])
            self.assertEqual(report['status'], 'complete')
            self.assertTrue(report['custody_capabilities']['containment'])
            launches = set()
            for run in report['runs']:
                self.assertEqual(run['status'], 'complete')
                receipt = run['supervisor']['custody']
                self.assertEqual(receipt['status'], 'proven_drained')
                self.assertTrue(receipt['echild'])
                self.assertTrue(all(row['reaped'] for row in receipt['entries']))
                self.assertTrue(any(row['role'] == 'adopted' for row in receipt['entries']))
                launches.add(receipt['launch_id'])
                self.assertEqual(run['custody_checkpoint']['status'], 'proven_drained')
            self.assertEqual(len(launches), 2)
            with socket.socket() as released:
                released.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                released.bind(('127.0.0.1', port))
                released.listen(1)


if __name__ == '__main__':
    unittest.main()
