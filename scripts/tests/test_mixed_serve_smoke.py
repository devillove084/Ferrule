"""CPU mocks only: no HTTP listener, CUDA calls or model/server subprocesses."""
import copy
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import mixed_serve_smoke as smoke


def plan():
    return dict(schema='ferrule.mixed-public-serve-smoke.v1', executable='/not-run/ferrule',
                executable_sha256='pin', python='/not-run/python', python_sha256='pin',
                runtime_libraries={'/not-run/libcudart.so.12': 'pin'},
                argv=smoke.serve_argv('/not-run/ferrule', smoke.MODEL), model=str(smoke.MODEL),
                policy=copy.deepcopy(smoke.POLICY), limits=copy.deepcopy(smoke.LIMITS),
                source_sha256={'source': 'pin'}, workflow_sha256={'script': 'pin'},
                model_metadata_sha256={'config.json': 'pin'}, rounds=2, prompt='Hello',
                input_tokens=1, normal_tokens=2, cancel_tokens=4,
                environment=dict(CUDA_VISIBLE_DEVICES='0', CUBLAS_WORKSPACE_CONFIG=':0:0',
                                 FERRULE_CUDA_ARCH='sm_86'))


def receipt():
    identity = dict(pid=123, birth_ticks=999)
    return dict(returncode=0, timed_out=False, cleanup_errors=[], stdout='STARTUP / NORMAL SSE / TEARDOWN', stderr='',
                handshake=dict(launch_id='owned', helper_identity=identity, containment_verified=True),
                custody=dict(status='proven_drained', launch_id='owned', helper_identity=identity,
                             containment_verified=True, echild=True))


def lifecycle():
    return dict(status='complete', server_returncode=0, port_released=True)


class Contracts(unittest.TestCase):
    def test_gpu_quiescence_needs_two_idle_observations(self):
        idle = dict(devices=[dict(index=0, used_mib=0)], compute_apps='')
        busy = dict(devices=[dict(index=0, used_mib=12)], compute_apps='123, GPU-owned, 12')
        with mock.patch.object(smoke, 'gpu_snapshot', side_effect=[busy, idle, idle]) as query, \
                mock.patch.object(smoke.time, 'sleep'), mock.patch('sys.stdout', new_callable=io.StringIO):
            self.assertEqual(smoke.wait_gpu_quiescent('mock')['status'], 'verified')
            self.assertEqual(query.call_count, 3)
        with mock.patch.object(smoke, 'gpu_snapshot') as query:
            with self.assertRaises(RuntimeError): smoke.wait_gpu_quiescent('mock', seconds=0)
            query.assert_not_called()

    def test_scoped_freeze_ignores_test_fixtures_not_production(self):
        snapshot = {'crates/x/src/main.rs':'a', 'crates/x/native/kernel.cu':'b',
                    'crates/x/tests/attention.rs':'c', 'Cargo.lock':'d'}
        self.assertEqual(smoke.production_inputs(snapshot), {k:v for k,v in snapshot.items() if '/tests/' not in k})

    def test_metadata_precision_topology_and_public_cli(self):
        value = plan()
        with mock.patch.object(smoke, 'sha', return_value='pin'), \
                mock.patch.object(smoke, 'source_snapshot', return_value={'source': 'pin'}):
            smoke.validate(value)
            for key, bad in [('ssm', 'bf16'), ('tp', 2), ('owners', 2), ('device_budget_bytes', 6 << 30)]:
                changed = copy.deepcopy(value)
                changed['policy'][key] = bad
                with self.assertRaises(smoke.http.ProtocolError): smoke.validate(changed)
            for flags in ('--model-profile', '--ctx-size', '--max-active-sequences', '--max-batch-tokens', '--cuda-device-budget-mb'):
                changed = copy.deepcopy(value)
                changed['argv'][changed['argv'].index(flags) + 1] = 'wrong'
                with self.assertRaises(smoke.http.ProtocolError): smoke.validate(changed)
            value['environment']['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            with self.assertRaises(smoke.http.ProtocolError): smoke.validate(value)
        self.assertEqual(smoke.FLAGS['--expert-prewarm'], 'lazy')
        self.assertNotIn('--numeric-fp8-precision', smoke.FLAGS)
        self.assertEqual(smoke.FLAGS['--kv-execution-policy'], 'resident')

    def test_source_or_binary_drift_blocks_before_spawn(self):
        with mock.patch.object(smoke, 'sha', return_value='pin'), \
                mock.patch.object(smoke, 'source_snapshot', return_value={'source': 'changed'}), \
                mock.patch.object(smoke.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(smoke.http.ProtocolError, 'MUST rebuild'): smoke.validate(plan())
            spawn.assert_not_called()
        with mock.patch.object(smoke, 'sha', return_value='changed'):
            with self.assertRaises(smoke.http.ProtocolError): smoke.validate(plan())

    def test_unknown_timeout_exit_or_forged_custody_never_restarts(self):
        for change in ('timeout', 'exit', 'unknown', 'echild', 'identity', 'launch', 'cleanup'):
            r = receipt()
            if change == 'timeout': r['timed_out'] = True
            elif change == 'exit': r['returncode'] = 1
            elif change == 'unknown': r['custody']['status'] = 'unknown'
            elif change == 'echild': r['custody']['echild'] = False
            elif change == 'identity': r['custody']['helper_identity'] = dict(pid=234)
            elif change == 'launch': r['custody']['launch_id'] = 'foreign'
            else: r['cleanup_errors'] = ['unreconciled']
            self.assertFalse(smoke.restart_allowed(r, lifecycle()))
            with tempfile.TemporaryDirectory() as d, mock.patch.object(smoke, 'validate'), \
                    mock.patch.object(smoke, 'wait_gpu_quiescent', return_value={'status':'verified'}), \
                    mock.patch.object(smoke.supervisor, 'run_process', return_value=r) as run, \
                    mock.patch('sys.stdout', new_callable=io.StringIO):
                self.assertEqual(smoke.execute(plan(), Path(d)/'run'), 1)
                run.assert_called_once()
        self.assertTrue(smoke.restart_allowed(receipt(), lifecycle()))

    def test_one_cold_restart_uses_two_separate_authenticated_supervisors(self):
        def run(argv, timeout, config):
            self.assertEqual(timeout, 720)
            self.assertEqual(argv[-1], '--_worker')
            smoke.write(Path(config['directory'])/'lifecycle.json', lifecycle())
            return receipt()
        with tempfile.TemporaryDirectory() as d, mock.patch.object(smoke, 'validate'), \
                    mock.patch.object(smoke, 'wait_gpu_quiescent', return_value={'status':'verified'}), \
                mock.patch.object(smoke.supervisor, 'run_process', side_effect=run) as runner, \
                mock.patch('sys.stdout', new_callable=io.StringIO) as printed:
            self.assertEqual(smoke.execute(plan(), Path(d)/'run'), 0)
            self.assertEqual(runner.call_count, 2)
            self.assertIn('cold process reload', printed.getvalue())
            saved = json.loads((Path(d)/'run/smoke.json').read_text())
            self.assertEqual(saved['cold_restart'], 'pass')
            self.assertEqual(saved['gpu_quiescence'], 'verified')

    def test_worker_early_exit_or_readiness_timeout_always_tears_down(self):
        for error in (RuntimeError('Server exited before Ready'), TimeoutError('Startup deadline exceeded')):
            with tempfile.TemporaryDirectory() as d, mock.patch.object(smoke, 'validate'), \
                    mock.patch.object(smoke, 'wait_gpu_quiescent', return_value={'status':'verified'}), \
                    mock.patch.object(smoke, 'port_released'), \
                    mock.patch.object(smoke.os, 'getpgrp', return_value=smoke.os.getpid()), \
                    mock.patch.object(smoke.subprocess, 'Popen') as spawn, \
                    mock.patch.object(smoke.http, 'wait_ready', side_effect=error), \
                    mock.patch.object(smoke.http, 'shutdown_owned_group', return_value={'timed_out':False}) as close, \
                    mock.patch.object(smoke, 'complete') as request, mock.patch('sys.stdout', new_callable=io.StringIO):
                spawn.return_value.returncode = 0
                self.assertEqual(smoke.worker(dict(manifest=plan(), directory=d)), 1)
                close.assert_called_once()
                request.assert_not_called()
                saved = json.loads((Path(d)/'lifecycle.json').read_text())
                self.assertIn(str(error), saved['error'])
                self.assertEqual(spawn.call_args.kwargs['env']['CUBLAS_WORKSPACE_CONFIG'], ':0:0')
                self.assertEqual(spawn.call_args.args[0], plan()['argv'])

    def test_shutdown_timeout_keeps_root_unknown_without_foreign_kill(self):
        server = mock.Mock()
        server.poll.return_value = None
        server.wait.side_effect = subprocess.TimeoutExpired('owned server', 45)
        with mock.patch.object(smoke.http.os, 'getpgrp', return_value=smoke.http.os.getpid()):
            result = smoke.http.shutdown_owned_group(server, 45)
        self.assertTrue(result['timed_out'])
        server.terminate.assert_called_once()
        server.kill.assert_not_called()

    def test_sse_completion_uses_existing_parser_and_actual_usage(self):
        records = [
            {'choices':[{'index':0,'text':', I','finish_reason':None}]},
            {'choices':[{'index':0,'text':'','finish_reason':'length'}]},
            {'choices':[], 'usage':{'prompt_tokens':1,'completion_tokens':2,'total_tokens':3}},
        ]
        def request(config, emit):
            self.assertEqual(config['payload']['prompt'], 'Hello')
            self.assertEqual(config['payload']['max_tokens'], 2)
            self.assertEqual(config['payload']['temperature'], 0)
            for item in records: emit(dict(kind='event', data=json.dumps(item), event='', at=1))
            emit(dict(kind='event', data='[DONE]', event='', at=2))
            emit(dict(kind='end', status='success'))
        with mock.patch.object(smoke.http, 'http_request', side_effect=request), \
                mock.patch('sys.stdout', new_callable=io.StringIO) as out:
            result = smoke.complete(plan())
            self.assertEqual(result['usage']['completion_tokens'], 2)
            self.assertIn('NORMAL SSE', out.getvalue())
        records[-1]['usage']['completion_tokens'] = 1
        records[-1]['usage']['total_tokens'] = 2
        with mock.patch.object(smoke.http, 'http_request', side_effect=request):
            with self.assertRaises(smoke.http.ProtocolError): smoke.complete(plan())

    def test_cancel_closes_owned_response_after_first_content(self):
        event = dict(id='cmpl-ferrule-7', choices=[dict(index=0, text=',', finish_reason=None)])
        response = mock.MagicMock()
        response.status = 200
        response.headers.get_content_type.return_value = 'text/event-stream'
        response.__enter__.return_value = response
        response.__iter__.return_value = iter([('data: '+json.dumps(event)+'\n').encode()])
        opener = mock.Mock()
        opener.open.return_value = response
        with mock.patch.object(smoke.urllib.request, 'build_opener', return_value=opener), \
                mock.patch('sys.stdout', new_callable=io.StringIO):
            self.assertEqual(smoke.cancel(plan()), 7)
        response.__exit__.assert_called_once()
        request = json.loads(opener.open.call_args.args[0].data)
        self.assertEqual((request['prompt'], request['max_tokens']), ('Hello', 4))

    def test_cancel_completed_response_cannot_pass_as_cancelled(self):
        event = dict(id='cmpl-ferrule-7', choices=[dict(index=0, text=',', finish_reason='length')])
        response = mock.MagicMock()
        response.status = 200
        response.headers.get_content_type.return_value = 'text/event-stream'
        response.__enter__.return_value = response
        response.__iter__.return_value = iter([('data: '+json.dumps(event)+'\n').encode()])
        opener = mock.Mock()
        opener.open.return_value = response
        with mock.patch.object(smoke.urllib.request, 'build_opener', return_value=opener):
            with self.assertRaisesRegex(smoke.http.ProtocolError, 'raced normal completion'):
                smoke.cancel(plan())
        response.__exit__.assert_called_once()

    def test_cancel_requires_matching_disconnect_and_released_runtime_leases(self):
        snapshot = dict(requests={'held':0}, prompt_bytes={'held':0}, runtime=dict(
            status='available', **{k:{'held':0} for k in ('waiting_requests','request_identities','session_identities')}))
        with tempfile.TemporaryDirectory() as d:
            log = Path(d)/'server.log'
            log.write_text(json.dumps({'fields':{'event':'request_http_terminal','request_id':7,'finish_reason':'client_disconnect'}})+'\n')
            with mock.patch.object(smoke, 'get_json', return_value=snapshot) as query, \
                    mock.patch('sys.stdout', new_callable=io.StringIO):
                self.assertEqual(smoke.wait_released(log, 7), snapshot)
                query.assert_called_once_with('/admission', timeout=30)
            with mock.patch.object(smoke, 'get_json', return_value=snapshot), \
                    mock.patch.object(smoke.time, 'sleep', side_effect=TimeoutError('no matching cancellation')):
                with self.assertRaises(TimeoutError): smoke.wait_released(log, 8)


if __name__ == '__main__':
    unittest.main()
