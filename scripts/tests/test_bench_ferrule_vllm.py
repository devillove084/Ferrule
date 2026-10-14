#!/usr/bin/env python3
"""CPU-only mocks: no inference imports, GPU, server, packages or network."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import signal
import tempfile
import threading
import time
import unittest
from unittest import mock
import urllib.error

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('bench_ferrule_vllm', ROOT / 'scripts/bench_ferrule_vllm.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def frame(value, event=''):
    return (('event: ' + event + '\n') if event else '') + 'data: ' + (
        value if isinstance(value, str) else json.dumps(value)) + '\n\n'


def content(text, reason=None):
    return {'choices': [{'index': 0, 'text': text, 'finish_reason': reason}]}


def usage(n=3):
    return {'choices': [], 'usage': {'prompt_tokens': 5, 'completion_tokens': n, 'total_tokens': 5+n}}


def valid_stream(n=3):
    return (': keepalive\n\n' + frame(content('')) + frame(content('ab')) + frame(content('c'))
            + frame(content('', 'length')) + frame(usage(n)) + frame('[DONE]'))


def parse(text, clock=lambda: 1):
    stream = bench.Stream()
    bench.parse_sse(text.splitlines(True), stream, lambda _: None, clock)
    return stream


def record(index=0, n=3, status='success'):
    stream = parse(valid_stream(n))
    result = stream.metrics(0, 2, status)
    result.update(status=status, index=index, repetition=0, prompt_sha256='same',
                  length_matched_success=status == 'success', sent_at=0, ended_at=2)
    return result


class SseContracts(unittest.TestCase):
    def test_multitoken_chunks_are_not_tokens_and_empty_keepalive_not_ttft(self):
        ticks = iter([1, 2, 3, 4, 5, 6])
        result = parse(valid_stream(), lambda: next(ticks)).metrics(0, 7, 'success')
        self.assertEqual(result['text'], 'abc')
        self.assertEqual(result['content_events'], 2)
        self.assertEqual(result['actual_output_tokens'], 3)
        self.assertEqual(result['ttft_seconds'], 2)
        self.assertEqual(result['content_event_itl_seconds'], [1])
        self.assertEqual(result['mean_tpot_usage_proxy_seconds'], .5)
        self.assertIsNone(result['token_itl_seconds'])
        self.assertIsNone(result['exact_token_tpot_seconds'])
        self.assertEqual(result['done_at'], 6)
        self.assertEqual(result['request_e2e_seconds'], 7)

    def test_chat_role_empty_reasoning_do_not_count_as_visible_content(self):
        text = ''.join(frame({'choices': [{'index': 0, 'delta': delta}]}) for delta in
                       [{'role': 'assistant'}, {'content': ''}, {'reasoning_content': 'think'},
                        {'content': 'answer'}, {'content': 'more'}])
        text += frame({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})
        text += frame(usage()) + frame('[DONE]')
        ticks = iter(range(1, 9))
        result = parse(text, lambda: next(ticks)).metrics(0, 9, 'success')
        self.assertEqual(result['ttft_seconds'], 4)
        self.assertEqual(result['content_events'], 2)
        self.assertTrue(result['reasoning_observed'])
        self.assertIsNone(result['mean_tpot_usage_proxy_seconds'])

    def test_one_event_many_tokens_does_not_invent_zero_tpot(self):
        text = frame(content('many tokens', 'length')) + frame(usage(8)) + frame('[DONE]')
        result = parse(text).metrics(0, 2, 'success')
        self.assertEqual(result['actual_output_tokens'], 8)
        self.assertIsNone(result['mean_tpot_usage_proxy_seconds'])

    def test_zero_token_eos_is_valid_but_has_no_ttft_or_tpot(self):
        result = parse(frame(content('', 'stop')) + frame(usage(0)) + frame('[DONE]')).metrics(0, 2, 'success')
        self.assertEqual(result['actual_output_tokens'], 0)
        self.assertIsNone(result['ttft_seconds'])
        self.assertIsNone(result['mean_tpot_usage_proxy_seconds'])

    def test_strict_terminals_usage_done_and_errors(self):
        good = valid_stream()
        variants = [
            good.replace(frame('[DONE]'), ''), good + frame('[DONE]'),
            good + frame(content('late')), good.replace(frame(usage()), ''),
            good.replace(frame(usage()), frame(usage())*2),
            good.replace(frame(content('', 'length')), ''),
            good.replace(frame(content('', 'length')), frame(content('', 'length'))*2),
            good.replace(frame(content('', 'length')), frame(content('', 'error'))),
            frame(usage()) + good,
            good.replace(frame(usage()), frame({'choices': [], 'usage': {
                'prompt_tokens': 5, 'completion_tokens': True, 'total_tokens': 6}})),
            good.replace(frame(usage()), frame({'choices': [], 'usage': {
                'prompt_tokens': 5, 'completion_tokens': 3, 'total_tokens': 9}})),
            frame({'error': {'message': 'bad'}}), frame('bad', 'error'), 'event: error\n\n',
            frame('{bad json'), frame('NaN'),
            frame({'choices': [{'index': 1, 'text': 'bad'}]}),
            frame({'choices': [{'index': 0, 'text': 42}]}),
            good.rstrip('\n'),
        ]
        for value in variants:
            with self.subTest(value=value), self.assertRaises((bench.ProtocolError, ValueError)):
                parse(value)

    def test_multiline_data_crlf_and_sse_metadata(self):
        text = 'id: 1\r\nretry: 100\r\ndata: {"choices":\r\ndata: [{"index":0,"text":"ok","finish_reason":"stop"}]}\r\n\r\n'
        text += frame(usage(1)) + frame('[DONE]')
        self.assertEqual(parse(text).content, ['ok'])

    def test_partial_timeout_preserves_text_events_without_inventing_token_count(self):
        stream = bench.Stream()

        def lines():
            yield from frame(content('several words')).splitlines(True)
            raise TimeoutError('mock deadline')
        with self.assertRaises(TimeoutError):
            bench.parse_sse(lines(), stream, lambda _: None, clock=lambda: 1)
        partial = stream.metrics(0, 2, 'timeout')
        self.assertEqual(partial['text'], 'several words')
        self.assertEqual(partial['ttft_seconds'], 1)
        self.assertEqual(len(partial['raw_sse_events']), 1)
        self.assertIsNone(partial['actual_output_tokens'])
        self.assertIsNone(partial['mean_tpot_usage_proxy_seconds'])


class HttpContracts(unittest.TestCase):
    def config(self):
        return dict(url='http://example.invalid/v1/completions', timeout=1,
                    payload=bench.make_payload('model', 'Hello', 3, 0))

    def invoke(self, response=None, error=None):
        messages = []
        opener = mock.Mock()
        opener.open.side_effect = error
        if response is not None:
            opener.open.return_value = response
        with mock.patch.object(bench.urllib.request, 'build_opener', return_value=opener), \
                mock.patch.object(bench.signal, 'setitimer'), mock.patch.object(bench.signal, 'signal'):
            bench.http_request(self.config(), messages.append)
        return messages, opener

    def response(self, text, status=200, mime='text/event-stream'):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.status = status
        response.headers.get_content_type.return_value = mime
        response.readline.side_effect = io.BytesIO(text.encode()).readline
        return response

    def test_shared_payload_and_success_without_gpu_or_ignore_eos(self):
        messages, opener = self.invoke(self.response(valid_stream()))
        self.assertEqual(messages[-1]['status'], 'success')
        payload = json.loads(opener.open.call_args.args[0].data)
        self.assertEqual(payload, dict(model='model', prompt='Hello', max_tokens=3, temperature=0,
                                       seed=0, n=1, stream=True, stream_options={'include_usage': True}))
        self.assertNotIn('ignore_eos', payload)
        self.assertNotIn('messages', payload)
        self.assertEqual(len([m for m in messages if m['kind'] == 'event']), 6)

    def test_http_error_cannot_be_success(self):
        error = urllib.error.HTTPError('http://example.invalid', 429, 'capacity', {}, io.BytesIO(b'capacity'))
        messages, _ = self.invoke(error=error)
        self.assertEqual(messages[-1]['status'], 'http_error')
        self.assertEqual(messages[-2]['status'], 429)

    def test_http_status_and_mime_rejected(self):
        for status, mime in [(201, 'text/event-stream'), (200, 'application/json')]:
            with self.subTest(status=status, mime=mime):
                messages, _ = self.invoke(self.response(valid_stream(), status, mime))
                self.assertEqual(messages[-1]['status'], 'protocol_error')

    def test_sse_error_and_timeout_partial_are_not_discarded(self):
        messages, _ = self.invoke(self.response(frame(content('partial')) + frame({'error': {'message': 'OOM'}})))
        self.assertEqual(messages[-1]['status'], 'sse_error')
        self.assertEqual(sum(m['kind'] == 'event' for m in messages), 2)
        response = self.response('')
        response.readline.side_effect = [line.encode() for line in frame(content('partial')).splitlines(True)] + [TimeoutError('drip')]
        messages, _ = self.invoke(response)
        self.assertEqual(messages[-1]['status'], 'timeout')
        self.assertEqual(sum(m['kind'] == 'event' for m in messages), 1)

    def test_redirect_is_refused(self):
        with self.assertRaises(bench.ProtocolError):
            bench.NoRedirect().redirect_request(None, None, 302, '', {}, 'https://elsewhere.invalid')

    def test_real_cpu_client_worker_startup_and_cleanup_without_traffic(self):
        client = bench.Client()
        self.assertIsNone(client.process.poll())
        client.close()
        self.assertIsNotNone(client.process.returncode)

    def test_parent_success_records_actual_input_output_and_eof_time(self):
        client = bench.Client.__new__(bench.Client)
        client.process = mock.Mock()
        client.process.stdin = io.StringIO()
        messages = [dict(kind='start', at=50, unix=100),
                    dict(kind='headers', at=50.01, status=200, content_type='text/event-stream')]
        stream = parse(valid_stream(2), clock=lambda: 50.1)
        messages.extend(dict(kind='event', **event) for event in stream.events)
        messages.append(dict(kind='end', at=50.9, status='success', error=None))
        client.process.stdout = io.StringIO(''.join(json.dumps(m)+'\n' for m in messages))
        with mock.patch.object(bench.threading, 'Timer'), \
                mock.patch.object(bench.time, 'monotonic', return_value=50):
            result = client.request(self.config(), {'expected_input_tokens': 5})
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['actual_output_tokens'], 2)
        self.assertTrue(result['input_length_matches'])
        self.assertFalse(result['output_length_matches'])
        self.assertFalse(result['length_matched_success'])
        self.assertEqual(result['done_at'], 50.1)
        self.assertAlmostEqual(result['request_e2e_seconds'], .9)
        self.assertEqual(json.loads(client.process.stdin.getvalue())['payload']['prompt'], 'Hello')

    def test_parent_deadline_retains_events_from_killed_client(self):
        client = bench.Client.__new__(bench.Client)
        client.process = mock.Mock()
        client.process.poll.return_value = None
        client.process.stdin = io.StringIO()
        timer = mock.Mock()
        callback = []
        messages = iter([json.dumps(dict(kind='start', at=50, unix=100)) + '\n',
                         json.dumps(dict(kind='event', at=51, event='', data=json.dumps(content('partial')))) + '\n'])

        def readline():
            try:
                return next(messages)
            except StopIteration:
                callback[0]()
                return ''
        client.process.stdout.readline.side_effect = readline

        def make_timer(delay, action):
            callback.append(action)
            return timer
        with mock.patch.object(bench.threading, 'Timer', side_effect=make_timer), \
                mock.patch.object(bench.time, 'monotonic', return_value=50):
            result = client.request(self.config(), {'expected_input_tokens': 5})
        self.assertEqual(result['status'], 'timeout')
        self.assertEqual(result['text'], 'partial')
        self.assertEqual(result['ttft_seconds'], 1)
        self.assertIsNone(result['actual_output_tokens'])
        self.assertTrue(result['right_censored'])
        client.process.kill.assert_called_once()

    @unittest.skipUnless(hasattr(signal, 'setitimer'), 'POSIX deadline required')
    def test_real_whole_request_timer_interrupts_blocked_mock_transport(self):
        opener = mock.Mock()
        opener.open.side_effect = lambda *a, **kw: time.sleep(1)
        config = self.config()
        config['timeout'] = .02
        messages = []
        previous = signal.getsignal(signal.SIGALRM)
        started = time.monotonic()
        try:
            with mock.patch.object(bench.urllib.request, 'build_opener', return_value=opener):
                bench.http_request(config, messages.append)
        finally:
            signal.signal(signal.SIGALRM, previous)
        self.assertEqual(messages[-1]['status'], 'timeout')
        self.assertLess(time.monotonic()-started, .8)


class AccountingContracts(unittest.TestCase):
    def test_failure_partial_usage_and_unissued_stay_in_denominators(self):
        partial = record(1, 2, 'timeout')
        unknown = record(2, 2, 'timeout')
        unknown['actual_output_tokens'] = None
        summary = bench.aggregate([record(), partial, unknown], 4, 5)
        self.assertEqual(summary['scheduled'], 5)
        self.assertEqual(summary['not_issued'], 2)
        self.assertEqual(summary['offered'], 3)
        self.assertEqual(summary['success_fraction'], 1/3)
        self.assertEqual(summary['successful_output_tokens'], 3)
        self.assertEqual(summary['known_usage_output_tokens'], 5)
        self.assertEqual(summary['requests_without_usage'], 1)
        self.assertEqual(summary['successful_requests_per_second'], .25)
        self.assertEqual(summary['successful_output_tokens_per_second'], .75)
        self.assertIsNone(summary['committed_visible_token_rate'])
        self.assertEqual(summary['distributions']['failure_ttft_seconds']['count'], 2)

    def test_all_failed_window_is_zero_not_missing_and_startup_failure_is_na(self):
        self.assertEqual(bench.aggregate([record(status='sse_error')], 10, 1)['successful_output_tokens_per_second'], 0)
        result = bench.aggregate([], 0, 4)
        self.assertIsNone(result['successful_output_tokens_per_second'])
        self.assertEqual(result['not_issued'], 4)

    def test_slo_goodput_requires_success_length_and_thresholds(self):
        matched, short, failed = record(), record(1), record(2, status='timeout')
        short['length_matched_success'] = False
        result = bench.aggregate([matched, short, failed], 4, 3, ttft_slo=2, e2e_slo=3)
        self.assertEqual(result['slo_goodput_requests_per_second'], .25)
        self.assertIsNone(bench.aggregate([matched], 4, 1)['slo_goodput_requests_per_second'])

    def test_different_actual_output_or_input_lengths_block_comparison(self):
        left = dict(status='complete', corpus=['same'], workload={'max_tokens': 8},
                    runs=[dict(summary={'not_issued': 0}, requests=[record()])])
        for field in ('actual_output_tokens', 'actual_input_tokens'):
            right = copy.deepcopy(left)
            right['runs'][0]['requests'][0][field] += 1
            result = bench.comparison_eligibility(left, right)
            self.assertFalse(result['length_workload_eligible'])
            self.assertIsNone(result['speedup'])
        result = bench.comparison_eligibility(left, left)
        self.assertTrue(result['length_workload_eligible'])
        self.assertIsNone(result['speedup'])  # Identical lengths still aren't a hardware/quality audit.

    def test_percentile_convention(self):
        self.assertEqual(bench.distribution([1, 2, 3, None]), dict(count=3, mean=2, p50=2, p95=2.9))
        self.assertIsNone(bench.percentile([], 95))

    def test_fixed_window_does_not_shrink_to_successful_requests(self):
        args = bench.parser().parse_args(['--engine', 'ferrule', '--url', 'http://example.invalid/v1/completions',
                                         '--model', 'x', '--output', 'unused.json', '--measurement-seconds', '.02',
                                         '--requests', '3', '--concurrency', '2'])
        clients = [mock.Mock(), mock.Mock()]
        for client in clients:
            client.request.side_effect = lambda config, identity, cutoff: dict(record(identity['index']), **{
                'repetition': identity['repetition']})
        start = time.monotonic()
        run = bench.measure(clients, args, bench.corpus_for(args), 0)
        self.assertGreaterEqual(time.monotonic()-start, .02)
        self.assertEqual(run['summary']['offered'], 3)
        self.assertEqual(run['summary']['wall_seconds'], .02)
        self.assertEqual(run['summary']['successful_output_tokens_per_second'], 9/.02)
        self.assertEqual(run['measurements']['window_kind'], 'fixed-time')
        self.assertGreater(run['measurements']['intentional_idle_seconds'], 0)
        self.assertEqual(run['measurements']['denominator_seconds'], .02)

    def fixed_work_args(self, requests=4, concurrency=1):
        return bench.parser().parse_args([
            '--engine', 'ferrule', '--url', 'http://example.invalid/v1/completions',
            '--model', 'x', '--output', 'unused.json', '--window-kind', 'fixed-work',
            '--measurement-seconds', '60', '--requests', str(requests), '--concurrency', str(concurrency),
            '--prompt', 'Hello', '--input-tokens', '1', '--max-tokens', '2'])

    def test_fixed_work_no_padding_and_real_concurrency_until_work_finishes(self):
        args = self.fixed_work_args(requests=6, concurrency=2)
        clients = [mock.Mock(), mock.Mock()]
        barrier = threading.Barrier(2)
        lock = threading.Lock()
        activity = dict(live=0, peak=0)
        indices = []

        def request(config, identity, cutoff):
            with lock:
                activity['live'] += 1
                activity['peak'] = max(activity['peak'], activity['live'])
                indices.append(identity['index'])
            barrier.wait(timeout=2)
            with lock:
                activity['live'] -= 1
            return record(identity['index'])
        for client in clients:
            client.request.side_effect = request
        with mock.patch.object(bench.time, 'sleep', side_effect=AssertionError('fixed work must not pad')):
            run = bench.measure(clients, args, bench.corpus_for(args), 0)
        measurements = run['measurements']
        self.assertEqual(sorted(indices), list(range(6)))
        self.assertEqual(activity['peak'], 2)
        self.assertEqual(activity['live'], 0)
        self.assertTrue(measurements['work_completed'])
        self.assertEqual(measurements['intentional_idle_seconds'], 0)
        self.assertEqual(measurements['denominator_seconds'], measurements['terminal_observed_at']-measurements['dispatch_start_at'])
        self.assertEqual(measurements['time_to_complete_seconds'], run['summary']['wall_seconds'])
        self.assertLess(run['summary']['wall_seconds'], 60)
        self.assertEqual(run['summary']['successful_output_tokens_per_second'], 18/run['summary']['wall_seconds'])
        self.assertEqual(run['summary']['window_kind'], 'fixed-work')

    def test_fixed_work_timeout_time_remains_in_denominator_and_incomplete_work_is_explicit(self):
        args = self.fixed_work_args()
        clock = [100.0]
        client = mock.Mock()

        def request(config, identity, cutoff):
            clock[0] += 2 if identity['index'] == 0 else 5
            return record(identity['index'], status='success' if identity['index'] == 0 else 'timeout')
        client.request.side_effect = request
        with mock.patch.object(bench.time, 'monotonic', side_effect=lambda: clock[0]), \
                mock.patch.object(bench.time, 'sleep', side_effect=AssertionError('no padded idle')):
            run = bench.measure([client], args, bench.corpus_for(args), 0)
        self.assertEqual(run['summary']['wall_seconds'], 7)
        self.assertEqual(run['summary']['offered'], 2)
        self.assertEqual(run['summary']['failed'], 1)
        self.assertEqual(run['summary']['not_issued'], 2)
        self.assertEqual(run['summary']['successful_output_tokens_per_second'], 3/7)
        self.assertIsNone(run['measurements']['time_to_complete_seconds'])
        self.assertFalse(run['measurements']['work_completed'])
        self.assertFalse(run['measurements']['all_requests_successful'])
        self.assertEqual(run['not_issued_indices'], [2, 3])

    def test_fixed_work_uses_actual_first_send_and_keeps_final_timeout(self):
        args = self.fixed_work_args(requests=2)
        clock = [100.0]
        client = mock.Mock()

        def request(config, identity, cutoff):
            index = identity['index']
            clock[0] = 104 if index == 0 else 107
            result = record(index, status='success' if index == 0 else 'timeout')
            result['sent_at'] = 102 if index == 0 else 104
            return result
        client.request.side_effect = request
        with mock.patch.object(bench.time, 'monotonic', side_effect=lambda: clock[0]):
            run = bench.measure([client], args, bench.corpus_for(args), 0)
        self.assertEqual(run['measurements']['dispatch_start_at'], 102)
        self.assertEqual(run['summary']['wall_seconds'], 5)
        self.assertEqual(run['measurements']['time_to_complete_seconds'], 5)
        self.assertTrue(run['measurements']['work_completed'])
        self.assertFalse(run['measurements']['all_requests_successful'])
        self.assertEqual(run['summary']['failed'], 1)
        self.assertEqual(run['summary']['successful_output_tokens_per_second'], 3/5)

    def test_fixed_work_http_failure_is_in_elapsed_time_and_remaining_work_is_issued(self):
        args = self.fixed_work_args(requests=3)
        clock = [100.0]
        client = mock.Mock()

        def request(config, identity, cutoff):
            clock[0] += 2
            return record(identity['index'], status='http_error' if identity['index'] == 0 else 'success')
        client.request.side_effect = request
        with mock.patch.object(bench.time, 'monotonic', side_effect=lambda: clock[0]), \
                mock.patch.object(bench.time, 'sleep', side_effect=AssertionError('no padded idle')):
            run = bench.measure([client], args, bench.corpus_for(args), 0)
        self.assertEqual(run['summary']['wall_seconds'], 6)
        self.assertEqual(run['summary']['success_fraction'], 2/3)
        self.assertEqual(run['summary']['successful_output_tokens_per_second'], 1)
        self.assertTrue(run['measurements']['all_scheduled_requests_terminal'])
        self.assertTrue(run['measurements']['work_completed'])
        self.assertFalse(run['measurements']['all_requests_successful'])
        self.assertEqual(run['measurements']['time_to_complete_seconds'], 6)

    def test_fixed_work_cutoff_without_timeout_does_not_masquerade_as_complete_batch(self):
        args = self.fixed_work_args(requests=2)
        clock = [100.0]
        client = mock.Mock()

        def request(config, identity, cutoff):
            clock[0] = cutoff
            return record(identity['index'])
        client.request.side_effect = request
        with mock.patch.object(bench.time, 'monotonic', side_effect=lambda: clock[0]):
            run = bench.measure([client], args, bench.corpus_for(args), 0)
        self.assertEqual(run['summary']['not_issued'], 1)
        self.assertEqual(run['summary']['wall_seconds'], 60)
        self.assertIsNotNone(run['stop_reason'])
        self.assertIsNone(run['measurements']['time_to_complete_seconds'])

    def test_modes_and_declared_length_mismatches_block_comparison(self):
        left = dict(status='complete', corpus=['same'], workload={'window_kind': 'fixed-work'},
                    runs=[dict(summary={'not_issued': 0}, requests=[record()])])
        right = copy.deepcopy(left)
        right['workload']['window_kind'] = 'fixed-time'
        self.assertFalse(bench.comparison_eligibility(left, right)['length_workload_eligible'])
        left['runs'][0]['requests'][0]['input_length_matches'] = False
        self.assertFalse(bench.comparison_eligibility(left, left)['length_workload_eligible'])

    def test_fixed_work_report_repeats_use_sum_of_completion_windows_not_time_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'result.json'
            client = mock.Mock()
            clock = [100.0]

            def request(config, identity, cutoff=None):
                clock[0] += 20 if identity['population'] == 'warmup' else 2
                return record(identity['index'])
            client.request.side_effect = request
            with mock.patch.object(bench, 'Client', return_value=client), \
                    mock.patch.object(bench.time, 'monotonic', side_effect=lambda: clock[0]), \
                    mock.patch.object(bench.time, 'sleep', side_effect=AssertionError('no padded idle')), \
                    mock.patch('sys.stdout', new=io.StringIO()):
                code = bench.main([
                    '--engine', 'ferrule', '--url', 'http://example.invalid/v1/completions',
                    '--model', 'x', '--output', str(output), '--window-kind', 'fixed-work',
                    '--measurement-seconds', '60', '--requests', '2', '--repeat', '2', '--warmup', '1'])
            result = json.loads(output.read_text())
            self.assertEqual(code, 0)
            self.assertEqual(result['schema'], 'ferrule.http_bench.v1')
            self.assertEqual(result['aggregate']['wall_seconds'], 8)
            self.assertEqual(result['measurements']['denominator_seconds'], 8)
            self.assertEqual(result['measurements']['time_to_complete_seconds'], 8)
            self.assertEqual(result['measurements']['intentional_idle_seconds'], 0)
            self.assertEqual(len(result['warmups']), 2)
            self.assertEqual(clock[0], 148)
            self.assertIn('dependencies', result['metadata']['unavailable_categories'])
            self.assertIn('precision', result['metadata']['unavailable_categories'])

    def test_window_timeout_stops_submissions_without_hiding_unissued(self):
        args = bench.parser().parse_args(['--engine', 'vllm', '--url', 'http://example.invalid/v1/completions',
                                         '--model', 'x', '--output', 'unused.json', '--measurement-seconds', '.01',
                                         '--requests', '4'])
        client = mock.Mock()
        client.request.return_value = record(status='timeout')
        run = bench.measure([client], args, bench.corpus_for(args), 0)
        self.assertEqual(run['summary']['offered'], 1)
        self.assertEqual(run['summary']['not_issued'], 3)
        self.assertIsNotNone(run['stop_reason'])

    def test_complete_report_excludes_warmup_and_audits_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'result.json'
            reference = Path(directory) / 'reference.json'
            metadata = Path(directory) / 'metadata.json'
            reference.write_text(json.dumps(dict(schema='ferrule.http_bench.v1', status='complete', runs=[])))
            metadata.write_text(json.dumps(dict(source={'commit': 'fixture'}, binary={'sha256': 'fixture'},
                                                precision={'storage': 'fixture'}, launch_argv=['owner-supplied'])))
            client = mock.Mock()
            client.request.side_effect = lambda config, identity, cutoff=None: dict(record(identity['index']),
                                                                                  population=identity['population'])
            with mock.patch.object(bench, 'Client', return_value=client), mock.patch('sys.stdout', new=io.StringIO()):
                code = bench.main(['--engine', 'vllm', '--url', 'http://example.invalid/v1/completions',
                                   '--model', 'x', '--output', str(output), '--repeat', '1', '--requests', '2',
                                   '--warmup', '1', '--measurement-seconds', '.01',
                                   '--metadata', str(metadata), '--compare', str(reference)])
            result = json.loads(output.read_text())
            self.assertEqual(code, 0)
            self.assertEqual(result['aggregate']['offered'], 2)
            self.assertEqual(result['aggregate']['wall_seconds'], .01)
            self.assertEqual(len(result['warmups']), 1)
            self.assertEqual(result['metadata']['source']['commit'], 'fixture')
            self.assertEqual(result['metadata']['operator_extra']['launch_argv'], ['owner-supplied'])
            self.assertIsNone(result['comparison']['speedup'])
            self.assertFalse(result['comparison']['length_workload_eligible'])
            self.assertIn('mean_tpot_usage_proxy', result['definitions'])

    def test_warmup_failure_creates_report_and_never_measures(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'result.json'
            client = mock.Mock()
            client.request.return_value = record(status='http_error')
            with mock.patch.object(bench, 'Client', return_value=client), \
                    mock.patch.object(bench, 'measure', side_effect=AssertionError('must not measure')):
                code = bench.main(['--engine', 'ferrule', '--url', 'http://example.invalid/v1/completions',
                                   '--model', 'x', '--output', str(output), '--repeat', '2', '--requests', '4'])
            result = json.loads(output.read_text())
            self.assertEqual(code, 1)
            self.assertEqual(result['aggregate']['not_issued'], 8)
            self.assertEqual(result['aggregate']['unstarted_windows'], 2)
            self.assertIsNone(result['aggregate']['successful_requests_per_second'])
            self.assertEqual(len(result['warmups']), 1)
            self.assertEqual(result['metadata']['precision']['status'], 'unavailable')
            client.close.assert_called_once()
            with mock.patch.object(bench, 'Client', side_effect=AssertionError('must not launch client')), \
                    self.assertRaises(FileExistsError):
                bench.main(['--engine', 'ferrule', '--url', 'http://example.invalid/v1/completions',
                            '--model', 'x', '--output', str(output)])


if __name__ == '__main__':
    unittest.main()
