"""CPU-only local HTTP/socket and evidence tests. No CUDA/model process is launched."""
import copy
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import client
import run

IDS = [25358, 13, 198, 760, 6511, 314, 6124, 369]


def sse(text='', identity='cmpl-ferrule-1', finish=None, usage=None):
    return dict(id=identity, choices=[dict(text=text, finish_reason=finish)] if usage is None else [], usage=usage)


def actual_cancel():
    return dict(status='controlled_client_disconnect', completion_id='cmpl-ferrule-1', text=' Tokyo',
                closed_after_content=True, tcp_shutdown=True, content_event_count=1, usage=None,
                finish_reason=None, done_at=None, raw_sse_events=[dict(data=json.dumps(sse(' Tokyo')), at=1)])


def events(request_id=1, cancelled=True, ids=None):
    ids = IDS[:2] if ids is None and cancelled else IDS if ids is None else ids
    common = dict(request_id=request_id, session_id=request_id)
    status = 'cancelled' if cancelled else 'finished'
    usage = dict(prompt_tokens=5, completion_tokens=8, total_tokens=13)
    rows = [dict(common, event=name) for name in ('request_admitted', 'request_http_admitted')]
    rows += [dict(common, event='request_worker_token_enqueued', actual_token_id=token, emission_ordinal=i + 1,
                  delivery_boundary='worker_channel_enqueue') for i, token in enumerate(ids)]
    rows += [dict(common, event='request_http_terminal', status=status,
                  finish_reason='client_disconnect' if cancelled else 'length', token_events=len(ids)),
             dict(common, event='request_worker_terminal', status=status, last_emission_ordinal=len(ids),
                  cancellation_requested=cancelled, cancellation_submitted=cancelled, cancellation_failed=False,
                  requires_restore_or_shutdown=False),
             dict(common, event='request_worker_terminal_enqueue', status=status, last_emission_ordinal=len(ids),
                  terminal_event_enqueued=not cancelled, usage_available=not cancelled,
                  delivery_boundary='worker_channel_enqueue', **({} if cancelled else usage))]
    return rows


def lines(rows):
    return [json.dumps(dict(fields=row)) for row in rows]


class LocalHTTPTests(unittest.TestCase):
    def exercise(self, frames, cancel):
        received = []
        closed = threading.Event()
        errors = []
        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'
            def log_message(self, *args):
                pass
            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                try:
                    for frame in frames:
                        self.wfile.write(frame)
                        self.wfile.flush()
                    self.connection.settimeout(2)
                    if self.connection.recv(1) == b'':
                        closed.set()
                except ConnectionResetError:
                    closed.set()
                except OSError as error:
                    errors.append(str(error))
                self.close_connection = True
        server = HTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        try:
            value = client.stream(f'http://127.0.0.1:{server.server_port}/v1/completions',
                                  run.schedule()[0]['payload'], cancel=cancel, deadline=time.monotonic() + 3)
            self.assertTrue(closed.wait(2), errors)
            self.assertEqual(len(received), 1)
            self.assertEqual(received[0], run.schedule()[0]['payload'])
            return value
        finally:
            thread.join(3)
            server.server_close()

    def test_close_only_after_actual_content_not_header_keepalive_or_empty(self):
        frames = [b': keep-alive\n\n', b'\n', self.frame(sse('')), self.frame(sse(' Tokyo'))]
        value = self.exercise(frames, True)
        self.assertEqual(value['content_event_count'], 1)
        self.assertEqual(len(value['raw_sse_events']), 2)
        self.assertEqual(value['text'], ' Tokyo')
        self.assertTrue(value['tcp_shutdown'])
        self.assertIsNone(value['usage'])
        self.assertIsNone(value['actual_output_tokens'])

    @staticmethod
    def frame(value):
        return ('data: ' + json.dumps(value) + '\n\n').encode()

    def test_recovery_reads_usage_and_done(self):
        value = self.exercise([self.frame(sse(' Tokyo.\nThe capital of Japan is')),
                               self.frame(sse(finish='length')),
                               self.frame(sse(usage=dict(prompt_tokens=5, completion_tokens=8, total_tokens=13))),
                               b'data: [DONE]\n\n'], False)
        self.assertEqual(value['status'], 'success')
        self.assertEqual(value['actual_output_tokens'], 8)
        self.assertTrue(value['adapter_http_report_complete'])

    def test_identity_change_rejected(self):
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            self.exercise([self.frame(sse('')), self.frame(sse(' Tokyo', identity='cmpl-ferrule-2'))], True)

    def test_recovery_missing_usage_terminal_rejected(self):
        with self.assertRaisesRegex(ValueError, 'missing exact terminal/usage'):
            self.exercise([self.frame(sse(' Tokyo')), b'data: [DONE]\n\n'], False)


class CancellationEvidenceTests(unittest.TestCase):
    def test_prefix_not_eight_id_quality_pass_and_closed_channel_expected(self):
        result = client.cancellation_evidence(lines(events()), actual_cancel(), IDS)
        self.assertTrue(result['cancellation_accepted'])
        self.assertEqual(result['enqueued_token_count'], 2)
        self.assertEqual(result['client_content_events'], 1)
        self.assertIsNone(result['exact_token_pass'])
        self.assertFalse(result['metric_eligible'])

    def test_missing_runtime_terminal_not_socket_success(self):
        rows = [r for r in events() if r['event'] != 'request_worker_terminal']
        result = client.cancellation_evidence(lines(rows), actual_cancel(), IDS)
        self.assertFalse(result['cancellation_accepted'])

    def test_identity_and_cancel_submitted_required(self):
        for mutate in ('identity', 'submitted'):
            rows = events()
            terminal = next(r for r in rows if r['event'] == 'request_worker_terminal')
            if mutate == 'identity':
                terminal['session_id'] = 2
            else:
                terminal['cancellation_submitted'] = False
            self.assertFalse(client.cancellation_evidence(lines(rows), actual_cancel(), IDS)['cancellation_accepted'])

    def test_late_disconnect_after_natural_finish_is_not_cancelled(self):
        rows = events(cancelled=False)
        terminal = next(r for r in rows if r['event'] == 'request_worker_terminal')
        terminal['cancellation_requested'] = True
        result = client.cancellation_evidence(lines(rows), actual_cancel(), IDS)
        self.assertTrue(result['natural_finish_race'])
        self.assertFalse(result['cancellation_accepted'])

    def test_token_order_or_wrong_prefix_rejected(self):
        for ids in ([13, 25358], [1], IDS + [1]):
            self.assertFalse(client.cancellation_evidence(lines(events(ids=ids)), actual_cancel(), IDS)['cancellation_accepted'])

    def test_passive_wait_missing_terminal_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'server.log'
            log.write_text('\n'.join(lines(events()[:2])) + '\n')
            before = time.monotonic()
            value = client.wait_cancel(log, actual_cancel(), IDS, deadline=before + .03)
            self.assertFalse(value['cancellation_accepted'])
            self.assertEqual(value['wait_status'], 'deadline')
            self.assertLess(time.monotonic() - before, .5)

    def test_partial_json_not_counted_as_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'server.log'
            log.write_text('{"fields": {"event": "request_worker_terminal"')
            self.assertEqual(client.complete_lines(log), [])


class AdapterTests(unittest.TestCase):
    def test_actual_canonical_spec_overrides_staged_visibility(self):
        role = run.ct.load(run.maintained.roles.ROLE_PATH)
        candidate = dict(candidate=dict(binary=str(run.maintained.qualified.QUALIFICATION / 'staged-final/cli')),
                         uuid_inventory=['GPU-b980f22b-58a3-fc88-21c1-ae86053d8535'])
        spec = run.spec_for(candidate)
        self.assertEqual(spec['environment']['CUDA_VISIBLE_DEVICES'], candidate['uuid_inventory'][0])
        self.assertEqual(spec['environment']['NVIDIA_VISIBLE_DEVICES'], candidate['uuid_inventory'][0])
        self.assertEqual(spec['environment']['FERRULE_PROFILE_NVTX'], '0')
        self.assertEqual(spec['argv'][spec['argv'].index('--expert-read-schedule') + 1], 'lookahead')
        self.assertEqual(role['candidate_identity']['cli_binary_sha256'], run.BINARY_SHA)

    def test_exact_two_frozen_requests_no_smoke(self):
        rows = run.schedule()
        run.validate_schedule(rows)
        self.assertEqual([r['name'] for r in rows], ['cancel-qa8', 'recovery-qa8'])
        self.assertEqual(rows[0]['payload'], rows[1]['payload'])
        with self.assertRaises(ValueError):
            run.validate_schedule(rows + [copy.deepcopy(rows[0])])

    def test_private_worker_specialization_compiles_without_launch(self):
        original = run.maintained._specializations
        built = []
        def compile_only(function, replacements, **bindings):
            compiled = original(function, replacements, **bindings)
            built.append(compiled)
            return lambda launch_hash: 0
        import io
        payload = dict(outer_deadline=time.monotonic() + 270)
        with patch.object(run.sys, 'stdin', io.StringIO(json.dumps(payload))), patch.object(run.maintained, '_specializations', compile_only):
            self.assertEqual(run.worker('cpu-test-not-authentication'), 0)
        self.assertEqual(len(built), 1)

    def test_serial_expected_cancel_then_exact_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'server.log'
            log.write_text('')
            calls = []
            def fake_http(folder, planned, spec, deadline):
                calls.append(planned['name'])
                cancel = planned['controlled_cancel']
                rows = events(request_id=1 if cancel else 2, cancelled=cancel)
                with log.open('a') as output:
                    output.write('\n'.join(lines(rows)) + '\n')
                if cancel:
                    value = actual_cancel()
                    value['cancellation_evidence'] = client.cancellation_evidence(lines(rows), value, IDS)
                    return value
                usage = dict(prompt_tokens=5, completion_tokens=8, total_tokens=13)
                return dict(status='success', text=planned['reference']['reference_text'], usage=usage,
                            finish_reason='length', done_at=time.monotonic(), actual_output_tokens=8,
                            adapter_http_report_complete=True, raw_sse_events=[
                                dict(data=json.dumps(sse(planned['reference']['reference_text'], identity='cmpl-ferrule-2'))),
                                dict(data=json.dumps(sse(identity='cmpl-ferrule-2', finish='length'))),
                                dict(data=json.dumps(sse(identity='cmpl-ferrule-2', usage=usage))), dict(data='[DONE]')])
            with patch.object(run, 'one_request', fake_http), patch.object(run.ex, 'logical_counter_window', return_value=dict(valid=True)), patch.object(run.ex, 'mountstats_delta', return_value=dict(status='shared_client_delta')):
                result = run.serial_requests(directory, run.schedule(), {}, run.LIMITS,
                    time.monotonic() + 10, time.monotonic() + 10, time.monotonic() + 10,
                    lambda: None, lambda label: {}, lambda: {})
            self.assertTrue(result['can_continue'], result['stop_reason'])
            self.assertEqual(calls, ['cancel-qa8', 'recovery-qa8'])
            self.assertEqual(result['summary']['controlled_censors'], 1)
            self.assertEqual(result['summary']['recovery_exact_pass'], 1)
            self.assertIsNone(result['observations'][0]['quality_passed'])
            self.assertFalse(result['observations'][0]['metric_eligible'])
            bound = run.join(run.trace.parse_token_trace(log.read_text().splitlines()), result['observations'], log.read_text().splitlines())
            self.assertTrue(all(row['protocol_accepted'] for row in bound))

    def test_cancel_failure_stops_without_second_request(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'server.log'
            log.write_text('')
            calls = []
            def failed_cancel(folder, planned, spec, deadline):
                calls.append(planned['name'])
                log.write_text('\n'.join(lines(events(cancelled=False))) + '\n')
                value = actual_cancel()
                value['cancellation_evidence'] = dict(cancellation_accepted=False)
                return value
            with patch.object(run, 'one_request', failed_cancel), patch.object(run.ex, 'logical_counter_window', return_value=dict(valid=True)), patch.object(run.ex, 'mountstats_delta', return_value=dict(status='shared_client_delta')):
                result = run.serial_requests(directory, run.schedule(), {}, run.LIMITS,
                    time.monotonic() + 10, time.monotonic() + 10, time.monotonic() + 10,
                    lambda: None, lambda label: {}, lambda: {})
            self.assertEqual(calls, ['cancel-qa8'])
            self.assertFalse(result['can_continue'])
            self.assertEqual(result['summary']['unissued'], 1)
            self.assertEqual(result['observations'][0]['status'], 'controlled_censor')
            self.assertIsNone(result['observations'][0]['quality_passed'])
            self.assertFalse(result['observations'][0]['metric_eligible'])


if __name__ == '__main__':
    unittest.main()
