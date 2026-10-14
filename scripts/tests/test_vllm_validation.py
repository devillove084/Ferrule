#!/opt/conda310/bin/python
"""CPU-only stdlib contract tests for scripts/test_vllm_qwen35.py."""
import importlib.util
import inspect
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / 'scripts/test_vllm_qwen35.py'
spec = importlib.util.spec_from_file_location('vllm_qwen35_harness', TARGET)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


class FakeMetadata:
    def __init__(self, version, requires=(), requires_python=None):
        self.version = version
        self.requires = list(requires)
        self.metadata = {'Requires-Python': requires_python}


class FakeRequirement:
    """Tiny injected parser; production uses packaging, not this test double."""
    def __init__(self, text):
        self.name, self.spec = text.split('>=')
        self.specifier = self
        self.marker = None
        self.url = None
        self.extras = ()

    def contains(self, version, prereleases=True):
        return version >= self.spec


class FakeSpecifier:
    def __init__(self, value):
        self.value = value

    def contains(self, version, prereleases=True):
        return True


class HarnessContracts(unittest.TestCase):
    def test_mock_metadata_reports_actual_missing_and_mismatch(self):
        values = {
            'vllm': FakeMetadata('0.28.0', ['good>=1', 'bad>=2']),
            'good': FakeMetadata('1.5'),
            'bad': FakeMetadata('1.0'),
        }

        def get_dist(name):
            if name not in values:
                raise harness.metadata.PackageNotFoundError(name)
            return values[name]

        result = harness.dependencies('vllm', get_dist=get_dist,
                                      packaging=(FakeRequirement, FakeSpecifier))
        self.assertEqual(result['status'], 'fail')
        self.assertEqual(result['versions']['vllm'], '0.28.0')
        self.assertTrue(any(item['name'] == 'bad' and item['reason'] == 'version mismatch'
                            for item in result['missingdeps']))

    def test_missing_packaging_does_not_guess_dependency_commands(self):
        values = {'vllm': FakeMetadata('0.28.0', ['foo>=2; python_version >= "3.0"'])}

        def get_dist(name):
            if name not in values:
                raise harness.metadata.PackageNotFoundError(name)
            return values[name]

        original = harness.importlib.import_module
        harness.importlib.import_module = lambda name: (_ for _ in ()).throw(ImportError('mock missing'))
        try:
            result = harness.dependencies('vllm', get_dist=get_dist)
        finally:
            harness.importlib.import_module = original
        self.assertEqual(result['status'], 'fail')
        self.assertEqual(result['missingdeps'], [{'name': 'packaging', 'reason': 'requirement evaluator unavailable'}])
        self.assertEqual(result['unevaluated'], ['foo>=2; python_version >= "3.0"'])
        self.assertIn('cannot be evaluated', result['errors'][0])
        self.assertNotIn('pip install', result['errors'][0])

    def test_mock_model_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'config.json').write_text(json.dumps({'model_type': 'qwen3_5_moe',
                                                           'architectures': ['Qwen3_5MoeForCausalLM']}))
            (path / 'tokenizer.json').write_text('{}')
            (path / 'tokenizer_config.json').write_text('{}')
            value = harness.model_check(path)
            self.assertEqual(value['model_type'], 'qwen3_5_moe')
            self.assertFalse(value['full_checkpoint_verified'])
            self.assertIsNone(value['checkpoint_revision'])

    def test_sse_counts_content_events_not_tokens_and_uses_actual_usage(self):
        lines = [
            'data: {"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}' + '\n', '\n',
            'data: {"choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}' + '\n', '\n',
            'data: {"choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}' + '\n', '\n',
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}' + '\n', '\n',
            'data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":4,"total_tokens":11}}' + '\n', '\n',
            'data: [DONE]\n', '\n',
        ]
        ticks = iter([1.0, 1.2, 1.3, 1.4, 1.5, 1.6])
        result = harness.parse_sse(lines, 0.9, clock=lambda: next(ticks))
        self.assertEqual(result['nonempty_content_events'], 2)
        self.assertEqual(result['actual_output_tokens'], 4)
        self.assertEqual(result['text'], 'hello world')
        self.assertIsNone(result['tpot_seconds'])
        self.assertIn('not exact first token', result['ttft_definition'])
        self.assertAlmostEqual(result['ttft_seconds'], 0.3)

    def test_sse_requires_terminal_usage_and_done(self):
        lines = [
            'data: {"choices":[{"index":0,"delta":{"content":"x"},"finish_reason":"stop"}]}\n', '\n',
        ]
        with self.assertRaises(RuntimeError):
            harness.parse_sse(lines, 0.0, clock=lambda: 1.0)

    def test_timeout_is_bounded_and_only_own_process_is_cleaned(self):
        result = harness.run_process([
            harness.sys.executable, '-c', 'import time; time.sleep(30)'], 0.05)
        self.assertTrue(result['timed_out'])
        self.assertIsNotNone(result['returncode'])

    def test_help_is_stdlib_and_no_mode_defaults_to_check(self):
        parsed = harness.parser().parse_args([])
        self.assertEqual(parsed.mode, 'check')
        help_result = subprocess.run([harness.sys.executable, '-B', str(TARGET), '--help'],
                                     text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     timeout=5, check=False)
        self.assertEqual(help_result.returncode, 0)
        self.assertIn('No packages are installed or modified', help_result.stdout)


class SafetyAndProtocolTests(unittest.TestCase):
    def test_default_mock_preflight_never_enters_gpu_or_model(self):
        deps = dict(status='pass', versions={'vllm': 'fixture'}, missingdeps=[], errors=[])
        proc = dict(returncode=0, timed_out=False, stdout='fixture', stderr='')
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(harness, 'dependencies', return_value=deps), \
                mock.patch.object(harness, 'model_check', return_value={'fixture': True}), \
                mock.patch.object(harness, 'run_process', return_value=proc) as run, \
                mock.patch.object(harness, 'isolated_worker', side_effect=AssertionError('GPU/HTTP forbidden')), \
                mock.patch.dict(harness.sys.modules, {'torch': None, 'vllm': None, 'packaging': None}), \
                mock.patch('sys.stdout', new_callable=io.StringIO) as stdout:
            output = Path(temp) / 'mock-preflight'
            self.assertEqual(harness.main(['--output', str(output)]), 0)
            printed = stdout.getvalue()
            self.assertIn('vLLM Qwen3.5 validation summary', printed)
            self.assertIn('Status: PASS', printed)
            self.assertIn('Measurements:', printed)
            self.assertIn('N/A (not run)', printed)
            self.assertIn(str(output / 'report.json'), printed)
            report = json.loads((output / 'report.json').read_text())
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(len(commands), 4)
        self.assertEqual(commands[0][0], 'nvidia-smi')
        self.assertEqual(commands[1][-2:], ['pip', 'check'])
        self.assertEqual(report['config']['mode'], 'check')
        for mode in ('gpu', 'generate', 'http', 'oracle'):
            self.assertIsNone(report['measurements'][mode])
        self.assertEqual(report['phases']['generate']['status'], 'notrun')

    def test_generate_missing_dependencies_blocks_all_workers(self):
        with tempfile.TemporaryDirectory() as temp, \
                mock.patch.object(harness, 'dependencies', return_value=dict(status='fail', versions={},
                    missingdeps=[{'name': 'torch'}], errors=['missing'])), \
                mock.patch.object(harness, 'isolated_worker', side_effect=AssertionError('must not run')), \
                mock.patch.object(harness, 'model_check', side_effect=AssertionError('must fail first')), \
                mock.patch('sys.stdout', new_callable=io.StringIO) as stdout:
            output = Path(temp) / 'blocked'
            self.assertEqual(harness.main(['generate', '--output', str(output)]), 1)
            printed = stdout.getvalue()
            self.assertIn('Status: FAIL', printed)
            self.assertIn('Missing dependencies/import errors:', printed)
            self.assertIn('torch', printed)
            self.assertIn('Next steps (manual only', printed)
            report = json.loads((output / 'report.json').read_text())
            self.assertEqual(report['missingdeps'], [{'name': 'torch'}])
            self.assertIsNone(report['measurements']['generate'])

    def test_quiet_keeps_legacy_one_line_output(self):
        deps = dict(status='pass', versions={'vllm': 'fixture'}, missingdeps=[], errors=[])
        proc = dict(returncode=0, timed_out=False, stdout='fixture', stderr='')
        with tempfile.TemporaryDirectory() as temp, \
                mock.patch.object(harness, 'dependencies', return_value=deps), \
                mock.patch.object(harness, 'model_check', return_value={'fixture': True}), \
                mock.patch.object(harness, 'run_process', return_value=proc), \
                mock.patch('sys.stdout', new_callable=io.StringIO) as stdout:
            output = Path(temp) / 'quiet'
            self.assertEqual(harness.main(['--quiet', '--output', str(output)]), 0)
            self.assertEqual(stdout.getvalue().splitlines(), [f'pass: {output / "report.json"}'])

    def test_existing_output_refused_before_work(self):
        with tempfile.TemporaryDirectory() as temp, \
                mock.patch.object(harness, 'preflight', side_effect=AssertionError('must not run')), \
                mock.patch('sys.stderr', new_callable=io.StringIO):
            self.assertEqual(harness.main(['--output', temp]), 2)

    def test_import_missing_module_and_abi_error_are_reported(self):
        for message in ("ModuleNotFoundError: No module named 'sample_dep'", 'ImportError: undefined symbol: ABI'):
            args = harness.parser().parse_args([])
            args.timeout = 1
            report = dict(phases={}, versions={}, missingdeps=[], errors=[], sourceidentity={})
            failed = dict(returncode=1, timed_out=False, stdout='', stderr=message)
            with mock.patch.object(harness, 'dep_phase', return_value=True), \
                    mock.patch.object(harness, 'model_check', return_value={}), \
                    mock.patch.object(harness, 'run_process', return_value=failed):
                harness.preflight(report, args)
            self.assertEqual(report['phases']['import_vllm']['status'], 'fail')
            self.assertIn(message, str(report['errors']))
            if 'sample_dep' in message:
                self.assertEqual(report['missingdeps'][0]['module'], 'sample_dep')

    def test_supervisor_interface_launches_helper_not_application_directly(self):
        with mock.patch.object(harness.subprocess, 'Popen', wraps=subprocess.Popen) as popen, \
                mock.patch.object(harness.os, 'killpg') as kill:
            value = harness.run_process([harness.sys.executable, '-B', '-c', 'pass'], 2)
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        self.assertIn('_owned_subreaper.py', popen.call_args.args[0][2])
        self.assertEqual(value['returncode'], 0)
        self.assertEqual(value['custody']['status'], 'proven_drained')
        kill.assert_not_called()

    def test_marker_and_extras_delegate_to_evaluator(self):
        base = mock.Mock()
        base.name, base.url, base.extras = 'extra-dep', None, {'image'}
        base.marker = None
        base.specifier.contains.return_value = True
        extra = mock.Mock()
        extra.name, extra.url, extra.extras = 'leaf', None, ()
        extra.marker.evaluate.side_effect = lambda env: env['extra'] == 'image'
        skipped = mock.Mock()
        skipped.marker.evaluate.return_value = False
        packages = {'vllm': FakeMetadata('1', ['base', 'skip']),
                    'extra-dep': FakeMetadata('1', ['extra'])}
        def get_dist(name):
            if name not in packages:
                raise harness.metadata.PackageNotFoundError(name)
            return packages[name]
        req = {'base': base, 'extra': extra, 'skip': skipped}.__getitem__
        result = harness.dependencies('vllm', get_dist, (req, FakeSpecifier))
        self.assertEqual(result['missingdeps'][0]['name'], 'leaf')
        self.assertEqual(len(result['missingdeps']), 1)
        extra.marker.evaluate.assert_any_call({'extra': 'image'})

    def test_requires_python_and_unknown_url_fail_closed(self):
        req = mock.Mock(url='https://invalid/package')
        req.marker = None
        specs = mock.Mock()
        specs.return_value.contains.return_value = False
        result = harness.dependencies('vllm', lambda name: FakeMetadata('1', ['direct'], '>=999'),
                                      (lambda text: req, specs))
        self.assertEqual(result['status'], 'fail')
        self.assertIn('Requires-Python', result['errors'][0])
        self.assertIn('identity not verified', result['unevaluated'][0]['reason'])

    def test_engine_args_forwarding_supported_without_guessing_kwargs(self):
        config = vars(harness.parser().parse_args(['generate', '--tp', '8', '--ep']))
        def llm(model, tokenizer, tensor_parallel_size, **kwargs):
            pass
        engine_names = ('enable_expert_parallel max_model_len max_num_seqs gpu_memory_utilization '
                        'seed enable_prefix_caching trust_remote_code cpu_offload_gb generation_config '
                        'pipeline_parallel_size data_parallel_size speculative_config').split()
        def engine():
            pass
        engine.__signature__ = inspect.Signature([
            inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY) for name in engine_names])
        def sampling(temperature, max_tokens, min_tokens, ignore_eos, seed):
            pass
        options, params = harness.generation_options(config, llm, engine, sampling)
        self.assertEqual(options['tensor_parallel_size'], 8)
        self.assertTrue(options['enable_expert_parallel'])
        self.assertEqual(params['min_tokens'], 32)
        self.assertTrue(params['ignore_eos'])
        with self.assertRaisesRegex(RuntimeError, 'API lacks'):
            harness.generation_options(config, lambda model: None, engine, sampling)

    def test_sse_invalid_terminal_cases(self):
        def event(value):
            return 'data: ' + (value if isinstance(value, str) else json.dumps(value)) + '\n\n'
        content = event({'choices': [{'index': 0, 'delta': {'content': 'a'}, 'finish_reason': None}]})
        terminal = event({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})
        usage = event({'choices': [], 'usage': {'prompt_tokens': 2, 'completion_tokens': 1, 'total_tokens': 3}})
        done = event('[DONE]')
        valid = content + terminal + usage + done
        bad = [content + usage + done, content + terminal + done, content + terminal + usage,
               valid + done, valid + content, event({'error': {'message': 'OOM'}}) + valid,
               terminal + usage + done, content + terminal + usage + usage + done,
               valid.replace('"total_tokens": 3', '"total_tokens": -1'),
               valid.replace('"completion_tokens": 1', '"completion_tokens": true'),
               valid[:-1], 'event: error\n\n' + valid]
        for stream in bad:
            with self.subTest(stream=stream), self.assertRaises((RuntimeError, ValueError)):
                harness.parse_sse(stream.splitlines(True), 0, clock=lambda: 1)
        stream = ': keepalive\n\n' + event({'choices': [{'index': 0, 'delta': {'reasoning': 'hidden'},
                                                       'finish_reason': None}]}) + valid
        result = harness.parse_sse(stream.splitlines(True), 0, clock=lambda: 1)
        self.assertEqual(result['nonempty_content_events'], 1)
        self.assertIsNone(result['token_itl_seconds'])

    def test_http_mock_response_uses_usage_and_does_not_start_server(self):
        events = [{'choices': [{'index': 0, 'delta': {'content': 'abc'}, 'finish_reason': 'length'}]},
                  {'choices': [], 'usage': {'prompt_tokens': 4, 'completion_tokens': 2, 'total_tokens': 6}},
                  '[DONE]']
        raw = ''.join('data: ' + (e if isinstance(e, str) else json.dumps(e)) + '\n\n' for e in events)
        response = mock.MagicMock(status=200)
        response.__enter__.return_value = response
        response.headers.get_content_type.return_value = 'text/event-stream'
        response.readline.side_effect = io.BytesIO(raw.encode()).readline
        opener = mock.Mock()
        opener.open.return_value = response
        config = vars(harness.parser().parse_args(['http', '--served-model', 'fixture']))
        config['timeout'] = 1
        with mock.patch.object(harness.urllib.request, 'build_opener', return_value=opener):
            result = harness.http_worker(config)
        self.assertEqual(result['actual_output_tokens'], 2)
        self.assertEqual(result['requested_max_tokens'], 32)
        payload = json.loads(opener.open.call_args.args[0].data)
        self.assertTrue(payload['stream_options']['include_usage'])
        self.assertNotIn('ignore_eos', payload)
        self.assertEqual(opener.open.call_args.kwargs['timeout'], 1)

    def test_http_warmup_failure_stops_without_retry(self):
        args = harness.parser().parse_args(['http', '--served-model', 'fixture', '--repeat', '3'])
        args.timeout = 1
        report = dict(measurements={}, phases={}, errors=[])
        child = dict(result=None, process=dict(returncode=1, timed_out=True))
        with mock.patch.object(harness, 'isolated_worker', return_value=child) as worker:
            harness.http_command(report, args)
        worker.assert_called_once()
        self.assertEqual(report['phases']['http']['not_issued'], 3)
        self.assertIsNone(report['measurements']['http'][0]['measurements'])

    def test_redaction_and_invalid_cli(self):
        with mock.patch.dict(harness.os.environ, {'HF_TOKEN': 'fixture-secret'}):
            value = harness.scrub('fixture-secret Bearer secret-password ' +
                                  '-----BEGIN PRIVATE KEY-----\nprivate\n-----END PRIVATE KEY-----')
        for secret in ('fixture-secret', 'secret-password', '\nprivate\n'):
            self.assertNotIn(secret, value)
        with mock.patch('sys.stderr', new_callable=io.StringIO):
            for argv in (['--timeout', 'nan'], ['--memory', '2'], ['http'],
                         ['http', '--served-model', 'x', '--url', 'http://u:p@host/v1']):
                with self.subTest(argv=argv), self.assertRaises(SystemExit):
                    harness.main(argv)


class ReportSummaryTests(unittest.TestCase):
    def invoke(self, mode, action, quiet=False):
        with tempfile.TemporaryDirectory() as temp, mock.patch('sys.stdout', new_callable=io.StringIO) as stdout, \
                mock.patch.object(harness, 'preflight', side_effect=action), \
                mock.patch.object(harness, 'explicit_worker', side_effect=action), \
                mock.patch.object(harness, 'http_command', side_effect=action), \
                mock.patch.object(harness.subprocess, 'Popen', side_effect=AssertionError('no subprocess')), \
                mock.patch.dict(harness.sys.modules, {'torch': None, 'vllm': None, 'packaging': None}):
            argv = [mode, '--output', str(Path(temp) / 'summary')]
            if mode == 'http':
                argv += ['--served-model', 'remote-fixture']
            if quiet:
                argv.append('--quiet')
            code = harness.main(argv)
            report = json.loads((Path(temp) / 'summary/report.json').read_text())
            return code, stdout.getvalue(), report

    def test_every_mode_prints_summary_after_complete_json(self):
        def success(report, args):
            report['versions'].update(vllm='fixture-vllm', torch='fixture-torch')
            report['phases']['model'] = dict(status='pass', details=dict(model_type='fixture',
                files={'config.json': {}, 'tokenizer.json': {}}))
            report['phases']['driver'] = dict(status='pass', details=dict(stdout='RTX 3090, UUID, fixture-driver, 24576'))
            if args.mode in ('gpu', 'generate'):
                report['phases'][args.mode] = dict(status='pass')
                report['measurements'][args.mode] = dict(cuda_runtime='fixture-cuda', actual_output_tokens=7,
                    total_seconds=1.5, tpot_seconds=None)
            elif args.mode == 'http':
                report['phases']['http'] = dict(status='pass')
                report['measurements']['http'] = [dict(status='pass', population='measured', measurements=dict(
                    actual_output_tokens=7, wall_seconds=2.0, ttft_seconds=0.3, tpot_seconds=None))]
        for mode in ('check', 'gpu', 'generate', 'http'):
            with self.subTest(mode=mode):
                code, text, report = self.invoke(mode, success)
                self.assertEqual(code, 0)
                self.assertEqual(report['schema'], 'ferrule.vllm-qwen35-check.v1')
                for expected in ('Status: PASS', f'Mode: {mode}', 'Python:', 'fixture-vllm', 'fixture-torch',
                                 'fixture-driver', 'passed=', 'failed=', 'skipped=', 'Model checks: pass',
                                 'config.json', 'Measurements:', 'oracle: N/A', 'Next steps:'):
                    self.assertIn(expected, text)
                if mode in ('gpu', 'generate'):
                    self.assertIn('CUDA runtime=fixture-cuda', text)
                if mode != 'check':
                    self.assertIn('actual_output_tokens=7', text)
                    self.assertIn('tpot_seconds=N/A', text)
                else:
                    self.assertIn('CUDA runtime=N/A', text)
                if mode == 'http':
                    self.assertIn('Model: remote-fixture', text)
                    self.assertIn('ttft_seconds=0.3', text)

    def test_check_failure_prints_metadata_import_and_model_causes_first(self):
        def failure(report, args):
            report['errors'] = [dict(phase='dependencies', message='dependency gate failed'),
                                dict(phase='import_vllm', message='Traceback\nframe\nImportError: undefined symbol: fixture_abi')]
            report['missingdeps'] = [dict(name='fixture-package', installed='1', requirement='fixture-package>=2',
                                          reason='version mismatch')]
            report['phases']['dependencies'] = dict(status='fail', details=dict(
                errors=['packaging unavailable; markers cannot be evaluated'],
                unevaluated=['unknown>=1; extra == "optional"']))
            report['phases']['import_vllm'] = dict(status='fail', details=dict(stderr=report['errors'][1]['message']))
            report['phases']['model'] = dict(status='fail', errors=['Missing tokenizer.json/tokenizer.model'])
            report['phases']['driver'] = dict(status='timeout', details=dict(stderr='driver timeout'))
        code, text, report = self.invoke('check', failure)
        self.assertEqual(code, 1)
        for expected in ('Status: FAIL', 'Key failures:', 'fixture-package>=2', 'version mismatch', 'fixture_abi',
                         'packaging unavailable', 'Unevaluated requirement', 'Missing tokenizer',
                         '-B -m pip check', '-B -m pip show vllm torch packaging', 'nvidia-smi'):
            self.assertIn(expected, text)
        self.assertLess(text.index('Key failures:'), text.index('Mode:'))
        self.assertLess(text.index('Next steps'), text.index('Mode:'))
        self.assertNotIn('pip install', text)
        self.assertIsNone(report['measurements']['generate'])

    def test_skipped_imports_are_not_reported_as_errors(self):
        code, text, report = self.invoke('check', lambda report, args: None)
        self.assertEqual(code, 0)
        self.assertNotIn('import_vllm: failed', text)
        self.assertIn('skipped=9', text)
        self.assertIn('none reported (see skipped phases', text)

    def test_summary_is_bounded_and_json_retains_all_sanitized_evidence(self):
        def failure(report, args):
            report['missingdeps'] = [dict(name=f'package-{n}', reason='missing') for n in range(100)]
            report['errors'] = [dict(phase='import_vllm', message='frame\n' * 1000 +
                'ImportError: fixture-credential Bearer another-secret ' +
                '-----BEGIN PRIVATE KEY-----\nprivate-material\n-----END PRIVATE KEY-----')]
        with mock.patch.dict(harness.os.environ, {'HF_TOKEN': 'fixture-credential', 'UNRELATED': 'do-not-dump-env'}):
            code, text, report = self.invoke('check', failure)
        self.assertEqual(code, 1)
        self.assertLess(len(text), 12000)
        self.assertIn('more dependency issues in JSON', text)
        self.assertEqual(len(report['missingdeps']), 100)
        self.assertIn('package-99', json.dumps(report))
        for secret in ('fixture-credential', 'another-secret', 'private-material', 'do-not-dump-env'):
            self.assertNotIn(secret, text)
            self.assertNotIn(secret, json.dumps(report))
        self.assertIn('<redacted>', json.dumps(report))

    def test_http_failure_shows_worker_error_and_unavailable_measurements(self):
        def failure(report, args):
            report['errors'] = [dict(phase='http', message='request failed')]
            report['phases']['http'] = dict(status='fail')
            report['measurements']['http'] = [dict(status='fail', population='warmup', measurements=None,
                process=dict(stderr='RuntimeError: Missing terminal SSE [DONE]'))]
        code, text, report = self.invoke('http', failure)
        self.assertEqual(code, 1)
        self.assertIn('Missing terminal SSE [DONE]', text)
        self.assertIn('warmup fail: N/A', text)
        self.assertIn('already-running server', text)
        self.assertIsNone(report['measurements']['http'][0]['measurements'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
