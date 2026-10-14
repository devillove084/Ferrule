#!/usr/bin/env python3
"""Shared stdlib HTTP/SSE benchmark for existing Ferrule and vLLM servers.

Legacy mode is client-only. Matrix execution launches servers only with --execute.
No GPU probes, inference imports or package changes.
Client-only worker processes enforce whole-request deadlines (including DNS,
headers and keepalive/drip streams) without losing partial SSE observations.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import math
import os
import random
import re
import tempfile
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

sys.dont_write_bytecode = True
SCRIPT = Path(__file__).resolve()
MAX_LINE = 1024 * 1024
MAX_BODY = 16 * 1024 * 1024


class ProtocolError(RuntimeError):
    pass


class ServerError(ProtocolError):
    pass


class ContractBlocked(ProtocolError):
    pass


def require(condition, message):
    if not condition:
        raise ProtocolError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def strict_json(text):
    def invalid(value):
        raise ValueError('Non-finite JSON number: ' + value)
    return json.loads(text, parse_constant=invalid)


class Stream:
    """Single-choice OpenAI stream. Content events are NOT token boundaries."""

    def __init__(self):
        self.events = []
        self.content = []
        self.arrivals = []
        self.finish_reason = None
        self.finish_at = None
        self.usage = None
        self.usage_at = None
        self.done_at = None
        self.reasoning_observed = False

    def event(self, data, event_type, at):
        self.events.append(dict(at=at, event=event_type, data=data))
        require(self.done_at is None, 'Data/event after [DONE]')
        if event_type == 'error':
            raise ServerError('SSE error event: ' + data[:2048])
        require(event_type in ('', 'message'), 'Unsupported SSE event type')
        if data == '[DONE]':
            require(self.finish_reason is not None and self.usage is not None,
                    '[DONE] before finish_reason/usage')
            self.done_at = at
            return
        value = strict_json(data)
        require(isinstance(value, dict), 'SSE JSON must be an object')
        if 'error' in value:
            raise ServerError('SSE error object: ' + data[:2048])
        choices = value.get('choices')
        require(isinstance(choices, list), 'Missing choices list')
        usage = value.get('usage')
        if usage is not None:
            require(self.finish_reason is not None and choices == [] and self.usage is None,
                    'Usage must occur once, after finish, with empty choices')
            require(isinstance(usage, dict), 'Invalid usage object')
            require(all(type(usage.get(k)) is int and usage[k] >= 0
                        for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')),
                    'Invalid usage token counts')
            require(usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens'],
                    'Inconsistent total_tokens')
            self.usage, self.usage_at = usage, at
            return
        require(self.finish_reason is None and len(choices) == 1,
                'Choices after finish or wrong choice count')
        choice = choices[0]
        require(isinstance(choice, dict) and type(choice.get('index')) is int and choice['index'] == 0,
                'Invalid choice index')
        if 'delta' in choice:
            delta = choice['delta']
            require(isinstance(delta, dict), 'Invalid delta')
            require(not delta.get('tool_calls') and not delta.get('refusal'),
                    'Tools/refusals unsupported')
            self.reasoning_observed |= bool(delta.get('reasoning') or delta.get('reasoning_content'))
            content = delta.get('content')
        else:
            require('text' in choice, 'Missing completion text')
            content = choice['text']
        require(content is None or isinstance(content, str), 'Non-text content')
        if content:
            self.content.append(content)
            self.arrivals.append(at)
        reason = choice.get('finish_reason')
        if reason is not None:
            require(reason in ('length', 'stop'), 'Unsupported/error finish_reason')
            self.finish_reason, self.finish_at = reason, at

    def eof(self):
        require(self.done_at is not None, 'Missing [DONE]')
        require(not self.arrivals or self.usage['completion_tokens'] > 0,
                'Content with zero completion_tokens')

    def metrics(self, sent, ended, status):
        n = self.usage['completion_tokens'] if self.usage is not None else None
        times = self.arrivals
        # This is only a usage-normalized span of visible content, NOT exact TPOT.
        # Multiple tokens in the first/last chunk and hidden tokens bias this span.
        tpot = ((times[-1] - times[0]) / (n - 1)
                if status == 'success' and n is not None and n > 1 and len(times) > 1
                and not self.reasoning_observed else None)
        return dict(text=''.join(self.content), raw_sse_events=self.events,
                    content_event_times=times, content_event_itl_seconds=[b-a for a, b in zip(times, times[1:])],
                    content_events=len(times), usage=self.usage, actual_output_tokens=n,
                    actual_input_tokens=self.usage['prompt_tokens'] if self.usage is not None else None,
                    finish_reason=self.finish_reason, finish_at=self.finish_at,
                    usage_at=self.usage_at, done_at=self.done_at,
                    ttft_seconds=times[0]-sent if times and sent is not None else None,
                    request_e2e_seconds=ended-sent if sent is not None and ended >= sent else None,
                    mean_tpot_usage_proxy_seconds=tpot, token_itl_seconds=None,
                    exact_token_tpot_seconds=None, reasoning_observed=self.reasoning_observed,
                    output_token_ids=None,
                    partial_token_count=None if n is None else n)


def parse_sse(lines, stream, emit, clock=time.monotonic):
    data, event_type = [], ''
    for raw in lines:
        line = raw.decode('utf-8') if isinstance(raw, bytes) else raw
        line = line.rstrip('\r\n')
        if line:
            if line.startswith(':'):
                continue
            field, _, value = line.partition(':')
            value = value[1:] if value.startswith(' ') else value
            if field == 'data':
                data.append(value)
            elif field == 'event':
                event_type = value
            elif field not in ('id', 'retry'):
                raise ProtocolError('Unsupported SSE field: ' + field)
            continue
        if data or event_type:
            payload, at = '\n'.join(data), clock()
            emit(dict(kind='event', data=payload, event=event_type, at=at))
            stream.event(payload, event_type, at)
        data, event_type = [], ''
    require(not data and not event_type, 'Unterminated SSE frame')
    stream.eof()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ProtocolError('HTTP redirect refused; use the exact endpoint URL')


def deadline_signal(signum, frame):
    raise TimeoutError('Whole-request deadline exceeded')


def http_request(config, emit):
    """Runs only in a client worker's main thread, where SIGALRM bounds all I/O."""
    stream = Stream()
    started = time.monotonic()
    emit(dict(kind='start', at=started, unix=time.time()))
    status, error = 'success', None
    signal.signal(signal.SIGALRM, deadline_signal)
    signal.setitimer(signal.ITIMER_REAL, config['timeout'])
    try:
        request = urllib.request.Request(config['url'], data=json.dumps(config['payload']).encode(),
                                         headers={'Content-Type': 'application/json',
                                                  'Accept': 'text/event-stream'})
        # No inherited proxy, redirect, credential or connection-pool policy.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        with opener.open(request, timeout=config['timeout']) as response:
            emit(dict(kind='headers', at=time.monotonic(), status=response.status,
                      content_type=response.headers.get_content_type()))
            require(response.status == 200, 'Expected HTTP 200')
            require(response.headers.get_content_type() == 'text/event-stream', 'Expected text/event-stream')

            def lines():
                total = 0
                while True:
                    line = response.readline(MAX_LINE + 1)
                    if not line:
                        return
                    total += len(line)
                    require(len(line) <= MAX_LINE and total <= MAX_BODY, 'SSE size limit exceeded')
                    yield line
            parse_sse(lines(), stream, emit)
    except urllib.error.HTTPError as exc:
        # Do not let an unbounded error body delay/error-mask the HTTP status.
        emit(dict(kind='headers', at=time.monotonic(), status=exc.code, content_type=None))
        status, error = 'http_error', f'HTTP {exc.code}: {exc.reason}'
        exc.close()
    except (TimeoutError, socket.timeout) as exc:
        status, error = 'timeout', str(exc)
    except urllib.error.URLError as exc:
        status = 'timeout' if isinstance(exc.reason, TimeoutError) else 'transport_error'
        error = str(exc)
    except ServerError as exc:
        status, error = 'sse_error', str(exc)
    except (ProtocolError, ValueError, UnicodeError) as exc:
        status, error = 'protocol_error', str(exc)
    except Exception as exc:
        status, error = 'transport_error', f'{type(exc).__name__}: {exc}'
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    emit(dict(kind='end', at=time.monotonic(), status=status, error=error))


def worker_main():
    def emit(message):
        print(json.dumps(message, allow_nan=False), flush=True)
    emit(dict(kind='ready'))
    for line in sys.stdin:
        http_request(strict_json(line), emit)


class Client:
    """One reusable CPU client process per concurrency slot, never a model runner."""

    def __init__(self):
        self.process = subprocess.Popen([sys.executable, '-B', str(SCRIPT), '--_worker'],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL, text=True, bufsize=1)
        timer = threading.Timer(10, self.process.kill)
        timer.start()
        try:
            require(strict_json(self.process.stdout.readline()) == {'kind': 'ready'}, 'Client startup failed')
        except Exception:
            self.close()
            raise
        finally:
            timer.cancel()
            timer.join()

    def close(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=5)
        self.process.stdin.close()
        self.process.stdout.close()

    def request(self, config, identity, cutoff=None):
        stream = Stream()
        record = dict(identity, status='client_error', error=None, sent_at=None,
                      client_dispatch_at=time.monotonic(), http_status=None, headers_at=None,
                      request_payload=config['payload'], after_deadline_events=[])
        deadline = min(record['client_dispatch_at'] + config['timeout'], cutoff or float('inf'))
        timed_out = threading.Event()

        def expire():
            timed_out.set()
            if self.process.poll() is None:
                self.process.kill()
        # Parent deadline also bounds a wedged worker or blocked emit. No subprocess is
        # launched per request; startup of these CPU clients is outside measured W.
        timer = threading.Timer(max(0, deadline-time.monotonic()), expire)
        timer.start()
        try:
            if deadline <= time.monotonic():
                raise TimeoutError('Measurement window ended before client dispatch')
            wire = dict(config, timeout=max(0.001, deadline-time.monotonic()))
            self.process.stdin.write(json.dumps(wire) + '\n')
            self.process.stdin.flush()
            while True:
                line = self.process.stdout.readline()
                if not line:
                    raise RuntimeError('Client worker exited before terminal report')
                message = strict_json(line)
                kind, at = message['kind'], message.get('at')
                if kind == 'start':
                    record.update(sent_at=at, sent_unix=message['unix'])
                elif kind == 'headers':
                    record.update(http_status=message['status'], headers_at=at,
                                  content_type=message['content_type'])
                elif kind == 'event':
                    if at > deadline:
                        record['after_deadline_events'].append(message)
                        continue
                    try:
                        stream.event(message['data'], message['event'], at)
                    except (ProtocolError, ValueError) as exc:
                        record['error'] = str(exc)
                elif kind == 'end':
                    record.update(status=message['status'], ended_at=at, error=message['error'])
                    break
                else:
                    raise ProtocolError('Unknown client message')
        except Exception as exc:
            record.update(status='timeout' if timed_out.is_set() or isinstance(exc, TimeoutError)
                          else 'client_error', error=f'{type(exc).__name__}: {exc}',
                          ended_at=min(time.monotonic(), deadline) if timed_out.is_set() else time.monotonic())
        finally:
            timer.cancel()
            timer.join()
        if timed_out.is_set() or record['ended_at'] > deadline:
            record.update(status='timeout', error='Request/measurement deadline; server drain unverified',
                          ended_at=deadline)
        if record['status'] == 'success':
            try:
                stream.eof()
            except ProtocolError as exc:
                record.update(status='protocol_error', error=str(exc))
        if (record['status'] == 'success' and stream.usage is not None
                and stream.usage['completion_tokens'] > config['payload']['max_tokens']):
            record.update(status='protocol_error', error='usage.completion_tokens exceeds max_tokens')
        record['deadline_at'] = deadline
        record.update(stream.metrics(record['sent_at'], record['ended_at'], record['status']))
        expected_input = identity.get('expected_input_tokens')
        expected_output = config['payload']['max_tokens']
        record['input_length_matches'] = (record['actual_input_tokens'] == expected_input
                                          if expected_input is not None else None)
        record['output_length_matches'] = (record['actual_output_tokens'] == expected_output
                                           if record['actual_output_tokens'] is not None else None)
        record['length_matched_success'] = (record['status'] == 'success'
                                            and record['output_length_matches'] is True
                                            and record['input_length_matches'] is not False)
        record['client_observed_at'] = time.monotonic()
        record['right_censored'] = record['status'] == 'timeout'
        return record


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    position = (len(values)-1) * p / 100
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high]-values[low]) * (position-low)


def distribution(values):
    values = [v for v in values if v is not None]
    return dict(count=len(values), mean=sum(values)/len(values) if values else None,
                p50=percentile(values, 50), p95=percentile(values, 95))


def aggregate(records, wall_seconds, scheduled, ttft_slo=None, e2e_slo=None):
    good = [r for r in records if r['status'] == 'success']
    failed = [r for r in records if r['status'] != 'success']
    matched = [r for r in good if r['length_matched_success']]
    counts = Counter(r['status'] for r in records)
    successful_tokens = sum(r['actual_output_tokens'] for r in good if r['actual_output_tokens'] is not None)
    known_tokens = sum(r['actual_output_tokens'] for r in records if r['actual_output_tokens'] is not None)
    unknown = sum(r['actual_output_tokens'] is None for r in records)
    slo = [r for r in matched if (ttft_slo is None or r['ttft_seconds'] is not None
                                  and r['ttft_seconds'] <= ttft_slo)
           and (e2e_slo is None or r['request_e2e_seconds'] is not None
                and r['request_e2e_seconds'] <= e2e_slo)]
    result = dict(scheduled=scheduled, offered=len(records), not_issued=scheduled-len(records),
                  statuses=dict(counts), success=len(good), failed=len(failed),
                  success_fraction=len(good)/len(records) if records else None,
                  scheduled_success_fraction=len(good)/scheduled if scheduled else None,
                  wall_seconds=wall_seconds, successful_output_tokens=successful_tokens,
                  known_usage_output_tokens=known_tokens, requests_without_usage=unknown,
                  committed_visible_token_rate=None,
                  committed_visible_token_rate_reason='SSE text is not token IDs; usage may include hidden/EOS tokens; partial counts unknown',
                  actual_output_counts=dict(Counter(str(r['actual_output_tokens']) for r in records)),
                  actual_input_counts=dict(Counter(str(r['actual_input_tokens']) for r in records)),
                  length_matched_success=len(matched),
                  slo_qualified_success=len(slo) if ttft_slo is not None or e2e_slo is not None else None)
    for key, value in (('successful_requests_per_second', len(good)),
                       ('successful_output_tokens_per_second', successful_tokens),
                       ('length_matched_requests_per_second', len(matched)),
                       ('length_matched_output_tokens_per_second', sum(r['actual_output_tokens'] for r in matched))):
        result[key] = value/wall_seconds if wall_seconds and records else None
    result['slo_goodput_requests_per_second'] = (len(slo)/wall_seconds if wall_seconds and records
                                                and (ttft_slo is not None or e2e_slo is not None) else None)
    result['distributions'] = {}
    for label, population in (('success', good), ('failure', failed)):
        for metric in ('ttft_seconds', 'request_e2e_seconds', 'mean_tpot_usage_proxy_seconds'):
            result['distributions'][label + '_' + metric] = distribution(r[metric] for r in population)
        result['distributions'][label + '_content_event_itl_seconds'] = distribution(
            t for r in population for t in r['content_event_itl_seconds'])
    return result


def make_payload(model, prompt, max_tokens, seed):
    return dict(model=model, prompt=prompt, temperature=0, seed=seed, n=1,
                max_tokens=max_tokens, stream=True, stream_options=dict(include_usage=True))


def measure(clients, args, corpus, repetition, progress=None):
    records, next_index = [], 0
    started = time.monotonic()
    cutoff = started + args.measurement_seconds
    stopped = None

    def submit(pool, client, index):
        item = corpus[index % len(corpus)]
        payload = make_payload(args.model, item['prompt'], args.max_tokens, args.seed)
        identity = dict(index=index, repetition=repetition, population='measured',
                        prompt_id=item['id'], prompt_sha256=item['sha256'],
                        expected_input_tokens=item['input_tokens'])
        return pool.submit(client.request, dict(url=args.url, payload=payload, timeout=args.timeout),
                           identity, cutoff)

    with ThreadPoolExecutor(max_workers=len(clients)) as pool:
        active = {}
        while time.monotonic() < cutoff and next_index < args.requests and len(active) < len(clients):
            client = clients[len(active)]
            active[submit(pool, client, next_index)] = client
            next_index += 1
        while active:
            complete, _ = wait(active, return_when=FIRST_COMPLETED)
            available = []
            for future in complete:
                client = active.pop(future)
                result = future.result()
                records.append(result)
                if progress is not None:
                    progress(records)
                available.append(client)
                if result['status'] in ('timeout', 'client_error'):
                    stopped = 'Client timeout/failure: stop new submissions; server drain unknown'
            for client in available:
                if not stopped and time.monotonic() < cutoff and next_index < args.requests:
                    active[submit(pool, client, next_index)] = client
                    next_index += 1
    terminal_observed_at = time.monotonic()
    sent_times = [r['sent_at'] for r in records
                  if r.get('sent_at') is not None and r['sent_at'] >= started]
    dispatch_start = min(sent_times) if sent_times else started
    terminal_elapsed = terminal_observed_at - dispatch_start
    all_terminal = len(records) == args.requests
    work_completed = all_terminal
    all_requests_successful = all_terminal and all(r['status'] == 'success' for r in records)
    idle_seconds = 0.0
    if args.window_kind == 'fixed-time':
        # Offered-load delivery rate retains the preregistered idle budget. This
        # is not the completion throughput of a fixed amount of work.
        idle_seconds = max(0, cutoff-terminal_observed_at)
        time.sleep(idle_seconds)
        wall = args.measurement_seconds
        denominator = 'preregistered fixed-time window, including intentional idle and failures'
    else:
        # Include every offered request through client-observed EOF/error/deadline,
        # plus dispatch/collection overhead. No post-work idle enters this rate.
        wall = terminal_elapsed
        denominator = 'first measured HTTP send (dispatch fallback if none) to last client-observed terminal/deadline, including failures and client overhead'
        if not all_terminal and stopped is None:
            stopped = 'Fixed-work cutoff reached before all scheduled requests were issued'
    measurements = dict(window_kind=args.window_kind, denominator=denominator,
                        denominator_seconds=wall, dispatch_start_at=dispatch_start,
                        terminal_observed_at=terminal_observed_at,
                        elapsed_to_last_terminal_seconds=terminal_elapsed,
                        time_to_complete_seconds=terminal_elapsed if all_terminal else None,
                        all_scheduled_requests_terminal=all_terminal, work_completed=work_completed,
                        all_requests_successful=all_requests_successful,
                        intentional_idle_seconds=idle_seconds,
                        deadline_limit_seconds=args.measurement_seconds)
    summary = aggregate(records, wall, args.requests, args.ttft_slo, args.e2e_slo)
    summary.update(window_kind=args.window_kind, denominator=denominator, denominator_seconds=wall,
                   time_to_complete_seconds=measurements['time_to_complete_seconds'])
    records.sort(key=lambda r: r['index'])
    return dict(repetition=repetition, started_at=started, cutoff_at=cutoff,
                observed_end_at=time.monotonic(), stop_reason=stopped, requests=records,
                not_issued_indices=list(range(next_index, args.requests)),
                measurements=measurements, summary=summary)


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('must be finite and > 0')
    return value


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('must be > 0')
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError('must be >= 0')
    return value


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--engine', required=True, choices=('ferrule', 'vllm'))
    cli.add_argument('--url', required=True, help='Existing raw /v1/completions endpoint; no redirects/proxies/auth')
    cli.add_argument('--model', required=True, help='Exact served model ID; not a local tokenizer path')
    prompts = cli.add_mutually_exclusive_group()
    prompts.add_argument('--prompt', default='Hello', help='Exact raw prompt; no chat template or thinking parser')
    prompts.add_argument('--prompts-json', type=Path, help='JSON list of {id,prompt,input_tokens}; bytes replayed unchanged')
    cli.add_argument('--input-tokens', type=positive_int, help='Expected server usage.prompt_tokens; NOT a tokenizer estimate')
    cli.add_argument('--max-tokens', type=positive_int, default=32)
    cli.add_argument('--seed', type=int, default=0)
    cli.add_argument('--warmup', type=nonnegative_int, default=2, help='Sequential requests per repeat, outside W')
    cli.add_argument('--requests', type=positive_int, default=100, help='Scheduled request cap per repeat; leftovers are not_issued')
    cli.add_argument('--repeat', type=positive_int, default=3, help='Same-server windows, NOT independent server replicas')
    cli.add_argument('--concurrency', type=positive_int, default=1)
    cli.add_argument('--timeout', type=positive, default=300)
    cli.add_argument('--window-kind', choices=('fixed-time', 'fixed-work'), default='fixed-time',
                     help='fixed-time: offered-load delivery over full window (legacy default); fixed-work: actual batch completion throughput, no padding')
    cli.add_argument('--measurement-seconds', type=positive, default=60,
                     help='fixed-time: denominator/window; fixed-work: safety cutoff only, NOT the throughput denominator')
    cli.add_argument('--ttft-slo', type=positive)
    cli.add_argument('--e2e-slo', type=positive)
    cli.add_argument('--metadata', type=Path, help='Operator-supplied remote server identity/resource/cache JSON; never auto-probed')
    cli.add_argument('--compare', type=Path, help='Existing other-engine report; audit lengths/workload, never auto-claim speedup')
    cli.add_argument('--first-after-ready', action='store_true', help=argparse.SUPPRESS)
    cli.add_argument('--output', required=True, type=Path, help='New JSON file; must not already exist')
    return cli


def corpus_for(args):
    rows = strict_json(args.prompts_json.read_text()) if args.prompts_json else [dict(id='prompt-0', prompt=args.prompt)]
    require(isinstance(rows, list) and rows, 'Prompt corpus must be a nonempty JSON list')
    result, ids = [], set()
    for index, row in enumerate(rows):
        require(isinstance(row, dict) and isinstance(row.get('prompt'), str) and row['prompt'], 'Invalid prompt')
        identity = row.get('id', str(index))
        require(isinstance(identity, str) and identity not in ids, 'Invalid/duplicate prompt ID')
        ids.add(identity)
        n = row.get('input_tokens', args.input_tokens)
        require(n is None or type(n) is int and n > 0, 'Invalid expected input token count')
        result.append(dict(id=identity, prompt=row['prompt'], input_tokens=n,
                           sha256=sha256(row['prompt'].encode())))
    return result


def metadata_for(args):
    data = strict_json(args.metadata.read_text()) if args.metadata else {}
    require(isinstance(data, dict), 'Metadata must be an object')
    result = {key: data.get(key, dict(status='unavailable', reason='Not supplied by server owner'))
              for key in ('source', 'model', 'binary', 'dependencies', 'precision', 'topology',
                          'resources', 'cache', 'startup', 'quality_gate', 'tokenizer', 'engine_capability',
                          'runtime_contract', 'comparison_intent')}
    result['unavailable_categories'] = [key for key, value in result.items()
                                        if not value or isinstance(value, dict)
                                        and value.get('status') == 'unavailable']
    result['operator_extra'] = {key: value for key, value in data.items() if key not in result}
    result['verification'] = 'Operator declarations only; HTTP client cannot verify remote runtime or physical resources'
    result['metadata_file_sha256'] = sha256(args.metadata.read_bytes()) if args.metadata else None
    return result


MATRIX_PREFILLS = (128, 512, 2048, 4096, 8192)
MATRIX_OUTPUTS = (32, 128, 256)
MATRIX_CONCURRENCIES = (1, 2, 4, 8)
PRECISION_FIELDS = ('weights_dtype', 'activation_dtype', 'accumulation_dtype', 'rounding',
                    'scale_profile', 'admitted_resources', 'kv_dtype', 'output_dtype',
                    'numeric_operands_dtype', 'state_boundary_dtype', 'gdn_recurrent_dtype', 'gdn_convolution_dtype')
TOPOLOGY_FIELDS = ('dp', 'tp', 'ep', 'pp', 'pcp', 'rank_layout',
                   'model_worker_count', 'expert_policy', 'physical_placement', 'canonical')
CAPABILITY_FIELDS = ('max_context_tokens', 'max_concurrency', 'streaming', 'raw_completions',
                     'eos_policy', 'prefix_cache')


def known(value):
    if value is None or value == '' or isinstance(value, str) and value.lower() in (
            'unknown', 'unavailable', 'n/a', 'pending', 'unverified'):
        return False
    if isinstance(value, dict):
        return bool(value) and all(known(v) for v in value.values())
    if isinstance(value, list):
        return bool(value) and all(known(v) for v in value)
    return True


def declared(metadata, group, fields):
    value = metadata.get(group) if isinstance(metadata, dict) else None
    if (not isinstance(value, dict) or 'status' in value and not known(value['status'])
            or not all(key in value and known(value[key]) for key in fields)):
        return None
    return {key: value[key] for key in fields}


def logical_groups(topology):
    """Canonical group IDs do not contain process/owner IDs."""
    return ([dict(kind='tensor', pp=p, dp=d, pcp=c, members=list(range(topology['tp'])))
             for p in range(topology['pp']) for d in range(topology['dp']) for c in range(topology['pcp'])]
            + [dict(kind='expert', pp=p, dp=0, pcp=0, members=list(range(topology['ep'])))
               for p in range(topology['pp'])])


def canonical_topology(metadata):
    """Normalize declarations, not runtime proof; unknowns never certify equality.

    v1 describes partitioned tensors and per-layer routed experts. Physical
    equality is exact per canonical slot; GPU renaming/permutation is unsupported.
    """
    result = dict(logical=None, physical=None, owners=None, reasons=[])
    topology = metadata.get('topology') or {}
    try:
        require(isinstance(topology, dict), 'object required')
        degrees = {k: topology[k] for k in ('dp', 'tp', 'ep', 'pp', 'pcp')}
        require(all(type(v) is int and v > 0 for v in degrees.values()), 'invalid degrees')
        require('status' not in topology or known(topology['status']), 'topology unverified')
        groups = logical_groups(degrees)
        plan = topology['canonical']
        require(isinstance(plan, dict) and plan.get('schema') == 'ferrule.logical_topology.v1', 'schema missing/unsupported')
        require('status' not in plan or known(plan['status']), 'canonical plan unverified')
        require(plan.get('device_permutation_policy', 'exact_canonical_slot') == 'exact_canonical_slot',
                'device permutation policy unsupported')
        require(type(plan['num_layers']) is int and plan['num_layers'] > 0, 'num_layers unknown/invalid')
        group_key = lambda r: (r['kind'], r['pp'], r['dp'], r['pcp'])
        declared_groups = sorted([dict(r, members=sorted(r['members'])) for r in plan['logical_groups']],
                                 key=group_key)
        require(known(declared_groups) and json.dumps(declared_groups, sort_keys=True) ==
                json.dumps(sorted(groups, key=group_key), sort_keys=True), 'logical groups missing/inconsistent')
        ranges = sorted(plan['pp_layer_ranges'], key=lambda r: r['pp'])
        require(len(ranges) == degrees['pp'], 'PP ranges incomplete')
        end = 0
        for stage, row in enumerate(ranges):
            require(set(row) == {'pp', 'start', 'end'} and all(type(v) is int for v in row.values())
                    and row['pp'] == stage and row['start'] == end and row['end'] > end, 'PP ranges invalid')
            end = row['end']
        require(end == plan['num_layers'], 'PP ranges do not cover model')
        tensors, experts = [], []
        partitions, expert_partitions = {}, {}
        tensor_layers, expert_layers = set(), set()
        for row in plan['tensor_shard_map']:
            keys = ('pp', 'dp', 'pcp', 'lane', 'layer', 'axis', 'start', 'end', 'size')
            require(isinstance(row, dict) and set(row) == set(keys) | {'tensor'} and known(row)
                    and all(type(row[k]) is int and row[k] >= 0 for k in keys)
                    and isinstance(row['tensor'], str), 'tensor shard invalid/unknown')
            p, d, c, lane, layer = (row[k] for k in keys[:5])
            require(p < degrees['pp'] and d < degrees['dp'] and c < degrees['pcp'] and lane < degrees['tp']
                    and ranges[p]['start'] <= layer < ranges[p]['end']
                    and row['start'] < row['end'] <= row['size'], 'tensor shard outside group/layer/range')
            partitions.setdefault((p, d, c, layer, row['tensor']), []).append(row)
            tensor_layers.add((p, d, c, layer))
            tensors.append(dict(row))
        expected_layers = {(p, d, c, layer) for p in range(degrees['pp']) for d in range(degrees['dp'])
                           for c in range(degrees['pcp']) for layer in range(ranges[p]['start'], ranges[p]['end'])}
        require(tensor_layers == expected_layers, 'tensor map layer/group coverage incomplete')
        for rows in partitions.values():
            require(sorted(r['lane'] for r in rows) == list(range(degrees['tp']))
                    and len({(r['axis'], r['size']) for r in rows}) == 1, 'tensor shard lanes/shape inconsistent')
            end = 0
            for row in sorted(rows, key=lambda r: r['start']):
                require(row['start'] == end, 'tensor shard gap/overlap')
                end = row['end']
            require(end == rows[0]['size'], 'tensor shards incomplete')
        moe_layers = plan['moe_layers']
        require(isinstance(moe_layers, list) and moe_layers and
                all(type(layer) is int and 0 <= layer < plan['num_layers'] for layer in moe_layers)
                and len(set(moe_layers)) == len(moe_layers), 'MoE layers unknown/invalid')
        for row in plan['expert_shard_map']:
            keys = ('pp', 'lane', 'layer', 'total_experts')
            require(isinstance(row, dict) and set(row) == set(keys) | {'experts'}
                    and all(type(row[k]) is int and row[k] >= 0 for k in keys), 'expert shard invalid')
            p, lane, layer = row['pp'], row['lane'], row['layer']
            ids = row['experts']
            require(p < degrees['pp'] and lane < degrees['ep'] and layer in moe_layers
                    and ranges[p]['start'] <= layer < ranges[p]['end'] and row['total_experts'] > 0
                    and isinstance(ids, list) and all(type(i) is int and 0 <= i < row['total_experts'] for i in ids),
                    'expert shard outside group/layer/range')
            expert_partitions.setdefault((p, layer), []).append(row)
            expert_layers.add(layer)
            experts.append(dict(row, experts=sorted(ids)))
        require(expert_layers == set(moe_layers), 'expert map layer coverage incomplete')
        for rows in expert_partitions.values():
            require(sorted(r['lane'] for r in rows) == list(range(degrees['ep']))
                    and len({r['total_experts'] for r in rows}) == 1
                    and sorted(i for r in rows for i in r['experts']) == list(range(rows[0]['total_experts'])),
                    'expert shard lanes/coverage inconsistent')
        result['logical'] = dict(schema=plan['schema'], degrees=degrees, logical_groups=groups,
            num_layers=plan['num_layers'], moe_layers=sorted(moe_layers), pp_layer_ranges=ranges,
            tensor_shard_map=sorted(tensors, key=lambda r: (r['pp'], r['dp'], r['pcp'], r['layer'], r['tensor'], r['lane'])),
            expert_shard_map=sorted(experts, key=lambda r: (r['pp'], r['layer'], r['lane'])))
    except (KeyError, TypeError, ValueError, IndexError, ProtocolError) as exc:
        result['reasons'].append('topology.canonical unknown/invalid: ' + str(exc))

    try:
        require(isinstance(topology, dict), 'object required')
        require(all(type(topology[k]) is int and topology[k] > 0
                    for k in ('pp', 'tp', 'ep', 'dp', 'pcp', 'model_worker_count')), 'invalid degrees/worker count')
        pp, tp, ep, dp, pcp = (topology[k] for k in ('pp', 'tp', 'ep', 'dp', 'pcp'))
        layout = topology['rank_layout']
        if layout == 'separate-tp-ep':
            require(dp == pcp == 1, 'separate owners require DP1 PCP1')
            expected = [(p, role, lane) for role, width in (('tensor', tp), ('expert', ep if ep > 1 else 0))
                        for p in range(pp) for lane in range(width)]
        else:
            require(layout == 'shared-tp-dp-pcp' and ep in (1, tp*dp*pcp), 'unsupported shared EP/layout')
            expected = [(p, 'shared', lane) for p in range(pp) for lane in range(tp*dp*pcp)]
        require(topology['model_worker_count'] == len(expected), 'owner count inconsistent')
        policy = declared(topology, 'expert_policy', ('residency', 'partition', 'placement', 'rank_backend', 'transport'))
        require(policy is not None, 'expert policy unknown/incomplete')
        rows = topology['physical_placement']
        require(isinstance(rows, list) and len(rows) == len(expected), 'owner placement incomplete')
        aliases = {'root': 'tensor', 'tp-root': 'tensor', 'ep-expert': 'expert', 'shared-tp-ep': 'shared'}
        owners, slots = [], []
        for i, (row, (p, role, lane)) in enumerate(zip(rows, expected)):
            require(isinstance(row, dict) and all(type(row.get(k)) is int and row[k] >= 0
                    for k in ('owner', 'pp', 'lane', 'cuda_ordinal', 'physical_ordinal'))
                    and row['owner'] == i and (row['pp'], aliases.get(row.get('role')), row['lane']) == (p, role, lane),
                    'owner placement invalid/inconsistent')
            owners.append({k: row[k] for k in ('owner', 'pp', 'role', 'lane')})
            device = dict(physical_ordinal=row['physical_ordinal'], gpu_uuid=row.get('gpu_uuid'))
            if role in ('tensor', 'shared'):
                slots.append(dict(kind='tensor', pp=p, dp=lane//(tp*pcp), pcp=lane//tp % pcp,
                                  lane=lane % tp, **device))
            if role in ('expert', 'shared') or ep == 1:
                # Unsharded experts can be replicated on multiple tensor workers.
                slots.append(dict(kind='expert', pp=p, dp=0, pcp=0, lane=lane if ep > 1 else 0, **device))
        result['owners'] = dict(rank_layout=layout, model_worker_count=len(expected), owners=owners,
                                rank_backend=policy['rank_backend'], transport=policy['transport'])
        resources = declared(metadata.get('precision', {}), 'admitted_resources',
                             ('gpu_uuids', 'per_gpu_limit_bytes', 'host_limit_bytes', 'pinned_limit_bytes'))
        require(resources is not None, 'GPU/budget unknown')
        uuids, limits = resources['gpu_uuids'], resources['per_gpu_limit_bytes']
        require(isinstance(uuids, list) and all(isinstance(u, str) and known(u) for u in uuids)
                and len(set(uuids)) == len(uuids) and isinstance(limits, list) and len(uuids) == len(limits)
                and all(type(v) is int and v > 0 for v in limits)
                and all(type(resources[k]) is int and resources[k] >= 0 for k in ('host_limit_bytes', 'pinned_limit_bytes')),
                'GPU/budget invalid')
        devices, ordinals, cuda_devices = {}, {}, {}
        for row in rows:
            uuid, ordinal, cuda = row.get('gpu_uuid'), row['physical_ordinal'], row['cuda_ordinal']
            require(isinstance(uuid, str) and known(uuid), 'GPU UUID unknown')
            require(devices.setdefault(uuid, ordinal) == ordinal and ordinals.setdefault(ordinal, uuid) == uuid
                    and cuda_devices.setdefault(cuda, (ordinal, uuid)) == (ordinal, uuid), 'GPU identity inconsistent')
        require(set(devices) == set(uuids), 'unique GPU set differs from admitted budget')
        result['physical'] = dict(device_permutation_policy='exact_canonical_slot',
            shard_to_device=sorted(slots, key=lambda r: (r['kind'], r['pp'], r['dp'], r['pcp'], r['lane'], r['gpu_uuid'])),
            unique_gpu_count=len(devices), gpus=sorted([dict(gpu_uuid=u, physical_ordinal=devices[u], limit_bytes=b)
                                                      for u, b in zip(uuids, limits)], key=lambda r: r['gpu_uuid']),
            host_limit_bytes=resources['host_limit_bytes'], pinned_limit_bytes=resources['pinned_limit_bytes'],
            expert_residency=policy['residency'])
    except (KeyError, TypeError, ValueError, IndexError, ProtocolError) as exc:
        result['reasons'].append('owner/device resources unknown/invalid: ' + str(exc))
    return result


def comparison_contract(left, right):
    """Fail closed, including nested unknowns and stateful oracle coverage."""
    left = left if isinstance(left, dict) else {}
    right = right if isinstance(right, dict) else {}
    labels, reasons = {}, []
    contracts = dict(precision=tuple(k for k in PRECISION_FIELDS if k != 'admitted_resources'),
                     engine_capability=CAPABILITY_FIELDS,
                     model=('checkpoint_sha256',), tokenizer=('revision', 'files_sha256', 'encoding'),
                     cache=('prefix_mode', 'os_cache', 'state_reset'))
    for group, fields in contracts.items():
        a, b = declared(left, group, fields), declared(right, group, fields)
        same = a is not None and b is not None and a == b
        labels[group] = dict(same=same, left=a, right=b)
        if not same:
            reasons.append(group + (' unknown/incomplete' if a is None or b is None else ' differs'))
    normalized = [canonical_topology(metadata) for metadata in (left, right)]
    for group, key in (('topology', 'logical'), ('physical_resources', 'physical'), ('owner_organization', 'owners')):
        a, b = (item[key] for item in normalized)
        same = a is not None and b is not None and a == b
        labels[group] = dict(same=same, left=a, right=b)
        if not same and group != 'owner_organization':
            reasons.append(group + (' unknown/incomplete' if a is None or b is None else ' differs'))
    for item in normalized:
        reasons.extend(item['reasons'])
    for metadata in (left, right):
        capability = declared(metadata, 'engine_capability', CAPABILITY_FIELDS)
        if capability is None or any(capability[k] is not True for k in ('streaming', 'raw_completions')) or (
                capability['eos_policy'] != 'permitted' or any(type(capability[k]) is not int or capability[k] <= 0
                for k in ('max_context_tokens', 'max_concurrency'))):
            reasons.append('engine_capability invalid/unsupported')
        cache = metadata.get('cache') or {}
        if cache.get('os_cache') == 'uncontrolled' or cache.get('state_reset') != 'per_request':
            reasons.append('cache/state isolation unknown or uncontrolled')
        precision = declared(metadata, 'precision', PRECISION_FIELDS)
        if precision is None or not all(isinstance(precision[k], str) for k in
                ('weights_dtype', 'activation_dtype', 'accumulation_dtype', 'rounding', 'kv_dtype', 'output_dtype',
                 'numeric_operands_dtype', 'state_boundary_dtype', 'gdn_recurrent_dtype', 'gdn_convolution_dtype')):
            reasons.append('precision.dtype/rounding invalid')
        resources = declared(metadata.get('precision', {}), 'admitted_resources',
                             ('gpu_uuids', 'per_gpu_limit_bytes', 'host_limit_bytes', 'pinned_limit_bytes'))
        experts = declared(metadata.get('topology', {}), 'expert_policy',
                           ('residency', 'partition', 'placement', 'rank_backend', 'transport'))
        if resources is None:
            reasons.append('precision.admitted_resources unknown/incomplete')
        scales = declared(metadata.get('precision'), 'scale_profile', ('dtype', 'interpretation', 'block'))
        if scales is None:
            reasons.append('precision.scale_profile unknown/incomplete')
        if experts is None:
            reasons.append('topology.expert_policy unknown/incomplete')
        gate = declared(metadata, 'quality_gate', ('status', 'reference_sha256', 'coverage_sha256',
                                                  'logits', 'routes', 'recurrent_state', 'convolution_state'))
        if gate is None or gate['status'] != 'pass' or any(gate[key] is not True for key in
                ('logits', 'routes', 'recurrent_state', 'convolution_state')):
            reasons.append('stateful quality gate missing/failed')
    if (left.get('quality_gate') or {}).get('coverage_sha256') != (right.get('quality_gate') or {}).get('coverage_sha256'):
        reasons.append('stateful quality coverage differs')
    return dict(contract_match=not reasons, reasons=sorted(set(reasons)), labels=labels,
                sameprecision=labels['precision']['same'] and not any('precision.' in r for r in reasons),
                sametopology=labels['topology']['same'],
                same_logical_topology=labels['topology']['same'],
                owner_organization_match=labels['owner_organization']['same'],
                physical_resources_match=labels['physical_resources']['same'],
                engine_capability_match=labels['engine_capability']['same']
                and not any('engine_capability' in r for r in reasons))


def bootstrap_ci(values, seed=0, resamples=10000):
    """No removal/imputation of missing process runs; CI is unavailable instead."""
    require(resamples > 0, 'bootstrap resamples must be positive')
    observed = [v for v in values if type(v) in (int, float) and math.isfinite(v)]
    result = dict(scheduled=len(values), observed=len(observed), estimate=None, low=None, high=None,
                  seed=seed, resamples=resamples, method='mean; process-run percentile bootstrap, linear quantiles')
    if not values or len(observed) != len(values):
        return result
    result['estimate'] = sum(values) / len(values)
    if len(values) >= 2 and min(values) == max(values):
        result.update(low=values[0], high=values[0])
    elif len(values) >= 2:
        rng = random.Random(seed)
        samples = [sum(rng.choice(values) for _ in values) / len(values) for _ in range(resamples)]
        result.update(low=percentile(samples, 2.5), high=percentile(samples, 97.5))
    return result


def validate_corpus(data):
    require(isinstance(data, dict), 'Corpus must be an object')
    tok = declared(data, 'tokenizer', ('revision', 'files_sha256', 'encoding'))
    require(tok is not None and isinstance(tok['files_sha256'], dict), 'Pin tokenizer assets/revision/encoding')
    require(all(isinstance(v, str) and len(v) == 64 and all(c in '0123456789abcdef' for c in v)
                for v in tok['files_sha256'].values()), 'Tokenizer asset SHA256 must be lowercase hex')
    rows = data.get('prompts')
    require(isinstance(rows, list) and rows, 'Corpus prompts must be nonempty')
    ids = set()
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get('id'), str) and row['id'] not in ids,
                'Unique prompt IDs required')
        require(isinstance(row.get('prompt'), str) and row['prompt'], 'Nonempty literal prompt required')
        n, tokens = row.get('input_tokens'), row.get('input_ids')
        require(type(n) is int and n > 0 and isinstance(tokens, list) and len(tokens) == n
                and all(type(t) is int and t >= 0 for t in tokens), 'Exact exported input_ids/count required')
        ids.add(row['id'])
    require(set(MATRIX_PREFILLS) <= {row['input_tokens'] for row in rows}, 'Missing required prefill lengths')


def validate_profiles(profiles):
    require(isinstance(profiles, list) and profiles, 'Profiles must be a nonempty list')
    ids = set()
    for profile in profiles:
        identity = profile.get('id')
        require(isinstance(identity, str) and identity and identity not in ids
                and all(c.isalnum() or c in '-_' for c in identity), 'Unique filesystem-safe profile ID required')
        ids.add(identity)
        url = urllib.parse.urlsplit(profile['url'])
        require(url.scheme == 'http' and url.hostname == '127.0.0.1' and url.port
                and url.path == '/v1/completions' and not url.query and not url.fragment
                and not url.username and not url.password, 'Supervised mode requires reserved IPv4 loopback port')
        require(profile.get('engine') in ('ferrule', 'vllm') and profile.get('model'), 'Invalid engine/model')
        argv = profile.get('argv')
        require(isinstance(argv, list) and argv and all(isinstance(x, str) and x for x in argv)
                and Path(argv[0]).is_absolute(), 'argv must start with absolute owner executable; no shell expansion')
        require(isinstance(profile.get('metadata'), dict), 'Profile metadata required (unknown blocks comparison)')
        require(profile.get('comparison_intent', 'system-level') in ('system-level', 'exact'),
                'Invalid comparison intent')


def export_corpus(directory, revision, output):
    """CPU-only optional tokenizers export; execution/replay remains stdlib-only."""
    require(not output.exists(), 'Refusing existing corpus output')
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise ProtocolError('matrix-corpus needs an existing tokenizers installation; no packages installed') from exc
    tokenizer = Tokenizer.from_file(str(directory / 'tokenizer.json'))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    assets = [p for p in directory.iterdir() if p.is_file() and
              ('token' in p.name or p.name in ('vocab.json', 'merges.txt', 'chat_template.jinja'))]
    identity = dict(revision=revision, files_sha256={p.name: sha256(p.read_bytes()) for p in sorted(assets)},
                    encoding=dict(add_special_tokens=False, template='none', thinking=False))
    rows = []
    for length in MATRIX_PREFILLS:
        for index, phrase in enumerate(('Describe the route through a quiet forest. ',
                                        'Compare two deterministic matrix algorithms. ')):
            tokens = tokenizer.encode(phrase * 8192, add_special_tokens=False).ids[:length]
            text = tokenizer.decode(tokens, skip_special_tokens=False)
            require(tokenizer.encode(text, add_special_tokens=False).ids == tokens and len(tokens) == length,
                    'Tokenizer roundtrip not exact; supply an owner-exported exact corpus instead')
            rows.append(dict(id=f'p{length}-{index}', prompt=text, input_tokens=length, input_ids=tokens))
    data = dict(tokenizer=identity, prompts=rows)
    validate_corpus(data)
    with output.open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_profiling(config, profile):
    require(set(config) == {'binary', 'nsys', 'pins', 'capture', 'request_seconds',
                            'instrumentation'}, 'Unknown profiling fields')
    require(config['capture'] == 'whole_lifecycle' and type(config['request_seconds']) is int
            and config['request_seconds'] in (180, 600),
            'Only bounded 180s/600s lifecycle capture is supported')
    require(config['instrumentation'] == 'compile_time_unverified_until_sink', 'Invalid instrumentation contract')
    paths = [config['binary'], config['nsys'], str(SCRIPT),
             str(SCRIPT.with_name('test_vllm_qwen35.py')), str(SCRIPT.with_name('_owned_subreaper.py')),
             str(SCRIPT.with_name('_profile_capture.py')), str(SCRIPT.with_name('bench_process_ipc.py')),
             str(Path(sys.executable).resolve())]
    require(all(isinstance(p, str) and Path(p).is_absolute() for p in paths)
            and set(config['pins']) == set(paths), 'Incomplete profiling binary/script pins')
    require(all(isinstance(v, str) and len(v) == 64 and all(c in '0123456789abcdef' for c in v)
                for v in config['pins'].values()), 'Invalid profiling SHA256')
    topology = profile['metadata'].get('topology', {})
    require(all(topology.get(k) == v for k, v in dict(tp=1, pp=1, dp=1, ep=8).items())
            and topology.get('expert_policy', {}).get('rank_backend') == 'process'
            and profile['metadata'].get('precision', {}).get('profile') == 'f32'
            and profile['metadata'].get('cache', {}).get('prefix_mode') == 'off',
            'Profiling is scoped to F32 process EP8 TP1/PP1/DP1 with prefix cache off')
    require(profile['argv'].count(config['binary']) == 1
            and profile['argv'][profile['argv'].index(config['binary']) + 1:][:1] == ['serve'],
            'Pinned binary must be the serving executable')


def prepare_profiling(profiles, binary, nsys, request_seconds=180):
    require(len(profiles) == 1 and profiles[0]['engine'] == 'ferrule' and nsys,
            'Select one Ferrule profile and an existing --nsys')
    binary, nsys = binary.resolve(), nsys.resolve()
    require(all(p.is_file() and os.access(p, os.X_OK) for p in (binary, nsys)), 'Missing executable')
    profile = profiles[0]
    require(profile['argv'].count('serve') == 1, 'Expected one existing serve command')
    index = profile['argv'].index('serve') - 1
    require(index >= 0 and Path(profile['argv'][index]).is_absolute(), 'Cannot identify original CLI')
    profile['argv'][index] = str(binary)
    pins = {str(p): file_digest(p) for p in (binary, nsys, SCRIPT,
            SCRIPT.with_name('test_vllm_qwen35.py'), SCRIPT.with_name('_owned_subreaper.py'),
            SCRIPT.with_name('_profile_capture.py'), SCRIPT.with_name('bench_process_ipc.py'),
            Path(sys.executable).resolve())}
    config = dict(binary=str(binary), nsys=str(nsys), pins=pins, capture='whole_lifecycle',
                  request_seconds=request_seconds,
                  instrumentation='compile_time_unverified_until_sink')
    # A new instrumented executable does not inherit the old binary's strict oracle or source identity.
    profile['metadata']['model'].update(binary_sha256=pins[str(binary)], source_manifest_sha256='unknown',
                                        source_snapshot='unknown')
    profile['metadata']['quality_gate'] = dict(status='unverified', reason='New profiling executable; no inherited oracle')
    profile['metadata']['profiling'] = config
    validate_profiling(config, profile)
    return config


def profiling_argv(config, profile, cell, directory, argv):
    validate_profiling(config, profile)
    for path, expected in config['pins'].items():
        require(file_digest(path) == expected, f'Profiling artifact changed: {path}; regenerate plan')
    # Counter lifetime includes startup/warmup/shutdown. Never append to an earlier run's sink.
    sink = (directory / 'ipc-timing.jsonl').resolve()
    require(not sink.exists(), 'Refusing an existing IPC sink')
    index = argv.index(config['binary'])
    require(argv[0] == '/usr/bin/env' and all('=' in arg and not arg.startswith('-')
            for arg in argv[1:index]),
            'Profiling requires the existing explicit environment contract')
    require(not any(arg.split('=', 1)[0] in ('FERRULE_PROCESS_IPC_TIMING', 'FERRULE_PROCESS_IPC_INSTRUMENT', 'LD_PRELOAD', 'TMPDIR')
                    for arg in argv[1:index]), 'Conflicting profiling environment')
    stop = (directory / 'profile-stop').resolve()
    require(not stop.exists(), 'Refusing an existing capture stop marker')
    duration = 300 + 2 * config['request_seconds'] + 120
    python = str(Path(sys.executable).resolve())
    wrapper = str(SCRIPT.with_name('_profile_capture.py').resolve())
    target = [python, '-B', wrapper, '--directory', str(directory.resolve()),
              '--lifetime-seconds', str(duration - 60), '--shutdown-seconds', '45', '--'] + argv[index:]
    return (argv[:index] + ['TMPDIR=' + str((directory / 'nsys-tmp').resolve()),
            'FERRULE_PROCESS_IPC_INSTRUMENT=1',
            'FERRULE_PROCESS_IPC_TIMING=' + str(sink),
            config['nsys'], 'profile', '--trace=cuda,nvtx,osrt', '--sample=none',
            '--cpuctxsw=none', '--trace-fork-before-exec=false', '--kill=none', '--wait=primary',
            '--stop-on-exit=true', '--delay=0', '--duration=' + str(duration),
            '--cuda-flush-interval=1000', '--force-overwrite=false',
            '--output=' + str((directory / 'nsys').resolve())] + target)


def shutdown_profile_capture(server, directory, timeout):
    """Drain the application first, leaving nsys alive to export on failure too.

    The wrapper signals only its direct server child; the verified outer
    subreaper remains the sole descendant cleanup authority.
    """
    started = time.monotonic()
    (directory / 'profile-stop').touch(exist_ok=False)
    try:
        code = server.wait(timeout=timeout)
        return dict(status='profiler_exited_descendants_unknown', returncode=code,
                    seconds=time.monotonic()-started, timed_out=False,
                    gpu_quiescence='unverified; no GPU probe')
    except subprocess.TimeoutExpired:
        return dict(status='profile_export_timeout', seconds=time.monotonic()-started,
                    timed_out=True, gpu_quiescence='unverified; outer subreaper must drain')


def profiling_artifacts(directory):
    import importlib.util
    spec = importlib.util.spec_from_file_location('profile_ipc', SCRIPT.with_name('bench_process_ipc.py'))
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    artifacts = {}
    for name in ('ipc-timing.jsonl', 'nsys.nsys-rep', 'profile-server.json'):
        path = directory / name
        artifacts[name] = dict(bytes=path.stat().st_size if path.is_file() else 0)
    ipc = parser.read_profile_ipc(directory / 'ipc-timing.jsonl')
    wrapper_path = directory / 'profile-server.json'
    wrapper = strict_json(parser.bounded_text(wrapper_path)) if wrapper_path.is_file() else {}
    present = (all(a['bytes'] for a in artifacts.values())
               and ipc['status'] == 'observed_lifetime_counters' and ipc['nonzero_transport_observed']
               and wrapper.get('returncode') == 0
               and wrapper.get('stop_reason') == 'matrix_worker_stop_marker')
    return dict(status='present_content_unverified' if present else 'missing_or_invalid_not_zero',
                files=artifacts, ipc=ipc, server_returncode=wrapper.get('returncode'),
                coverage='Require offline CUDA PID/device/interval inspection; report bytes alone are not GPU coverage')


def build_manifest(corpus, profiles, seed=0, repetitions=5, requests=100, warmup=8,
                   cell_ids=None, limits=None, profiling=None):
    validate_corpus(corpus)
    validate_profiles(profiles)
    require(repetitions >= (1 if profiling else 2) and requests > 0 and warmup >= 0 and seed >= 0,
            'Invalid matrix counts/seed')
    if profiling:
        require((repetitions, requests, warmup) == (1, 1, 1) and len(profiles) == 1
                and profiles[0]['engine'] == 'ferrule', 'Profiling requires one Ferrule profile, warm1/measure1/run1')
        require(limits == dict(startup_seconds=300, request_seconds=profiling['request_seconds'],
                               measurement_seconds=profiling['request_seconds'], teardown_seconds=120),
                'Profiling limits must match the explicit 180s/600s budget')
        validate_profiling(profiling, profiles[0])
    cells = [dict(id=f'p{p}-o{o}-c{c}', prefill_tokens=p, output_tokens=o, context_tokens=p+o,
                  concurrency=c) for p in ((128, 512, 2048) if profiling else MATRIX_PREFILLS)
             for o in ((1,) if profiling else MATRIX_OUTPUTS)
             for c in ((1,) if profiling else MATRIX_CONCURRENCIES)]
    if cell_ids is not None:
        require(isinstance(cell_ids, list) and cell_ids and len(set(cell_ids)) == len(cell_ids)
                and set(cell_ids) <= {c['id'] for c in cells}, 'Unknown/duplicate/empty cell selection')
        cells = [c for c in cells if c['id'] in cell_ids]
    default_limits = dict(startup_seconds=900, request_seconds=300, measurement_seconds=3600, teardown_seconds=120)
    if profiling:
        default_limits['request_seconds'] = 600
    if limits is not None:
        require(isinstance(limits, dict) and set(limits) == set(default_limits)
                and all(type(v) in (int, float) and math.isfinite(v) and 0 < v <= default_limits[k]
                        for k, v in limits.items()), 'Limits must be positive and no larger than protocol defaults')
    rng, schedule = random.Random(seed), []
    for repetition in range(repetitions):
        order = list(cells)
        if not profiling:
            rng.shuffle(order)
        for cell in order:
            engines = list(profiles)
            rng.shuffle(engines)
            for profile in engines:
                schedule.append(dict(index=len(schedule), cell_id=cell['id'], profile_id=profile['id'],
                                     repetition=repetition))
    return dict(schema='ferrule.http_matrix_manifest.v1', seed=seed, repetitions=repetitions,
                requests=requests, warmup=warmup, corpus=corpus, profiles=profiles, cells=cells, schedule=schedule,
                **(dict(selected_cell_ids=[c['id'] for c in cells]) if cell_ids is not None else {}),
                **(dict(profiling=profiling) if profiling else {}),
                limits=limits if limits is not None else default_limits,
                policy=dict(context='prefill + output', eos='permitted; short output retained, not length-matched',
                            window_kind='fixed-work', prefix='off or first-cold-then-warm; never global dropcache',
                            statistics='independent fresh server per schedule entry; no retries'))


def supervisor_budget(manifest, cell):
    limit = manifest['limits']
    return (limit['startup_seconds'] + (manifest['warmup']+(0 if manifest.get('profiling') else 1))*limit['request_seconds']
            + limit['measurement_seconds'] + limit['teardown_seconds'] + 15*cell['concurrency'] + 30)


def matrix_budget(manifest):
    """Offline worst-case schedule budget, retaining blocked/unsupported cells."""
    per_cell = {cell['id']: supervisor_budget(manifest, cell) for cell in manifest['cells']}
    total = sum(per_cell[slot['cell_id']] for slot in manifest['schedule'])
    cleanup = 30 * len(manifest['schedule'])  # handshake, control, helper drain/reap failure margins
    probe = 30  # one isolated capability probe; no application launched
    return dict(scheduled_server_starts=len(manifest['schedule']),
                scheduled_measured_requests=len(manifest['schedule'])*manifest['requests'],
                supervisor_seconds_per_cell=per_cell, supervisor_seconds_total=total,
                supervisor_cleanup_reserve_seconds=cleanup,
                capability_probe_reserve_seconds=probe,
                serial_seconds_with_cleanup_reserve=total+cleanup+probe,
                serial_hours_with_cleanup_reserve=(total+cleanup+probe)/3600,
                excluded='report fsync/CPU work, owner GPU/port checks, model preparation; not a hard wall bound',
                required_cells_per_profile=60, selected_cells_per_profile=len(manifest['cells']),
                all_required_cells_selected=len(manifest['cells']) == 60)



def launch_contract(profile, argv=None, cell=None):
    """Bind candidate metadata to shell-free argv; legacy system profiles stay separate."""
    if 'launch_contract' not in profile and profile.get('comparison_intent') != 'exact':
        return []
    reasons = []
    try:
        argv = list(profile['argv'] if argv is None else argv)
        binding = profile['launch_contract']
        require(binding['schema'] == 'ferrule.launch_contract.v1', 'launch schema')
        require(argv[0] == '/usr/bin/env', 'explicit environment required')
        env, index = {}, 1
        while index < len(argv) and '=' in argv[index] and not argv[index].startswith('--'):
            key, value = argv[index].split('=', 1)
            require(key not in env, 'duplicate environment key')
            env[key] = value
            index += 1
        require(argv[index:index+3] == [binding['executable'], 'serve', binding['model_path']],
                'executable/model argv differs')
        require(Path(binding['executable']).is_absolute(), 'absolute executable required')
        flags, index = {}, index + 3
        while index < len(argv):
            key = argv[index]
            require(key.startswith('--') and '=' not in key and key not in flags,
                    'duplicate/ambiguous flag: ' + key)
            index += 1
            value = True
            if index < len(argv) and not argv[index].startswith('--'):
                value = argv[index]
                index += 1
            flags[key] = value
        expected_flags = binding['flags']
        if cell is not None:
            expected_flags = {k: v.replace('{context_tokens}', str(cell['context_tokens']))
                .replace('{concurrency}', str(cell['concurrency'])) if isinstance(v, str) else v
                for k, v in expected_flags.items()}
        require(flags == expected_flags, 'actual flags differ from frozen launch contract')
        require(env == binding['environment'], 'actual environment differs from frozen launch contract')
        meta = profile['metadata']
        t, precision = meta['topology'], meta['precision']
        pp = t['pp']
        require(type(pp) is int and pp in (1, 2) and
                all(type(t[k]) is int and t[k] == v for k, v in dict(tp=2, ep=2, dp=1, pcp=1).items()),
                'candidate requires PP1/2 TP2 EP2 DP1 PCP1')
        ferrule = profile['engine'] == 'ferrule'
        expected = {'--pipeline-parallel' if ferrule else '--pipeline-parallel-size': str(pp),
                    '--tensor-parallel' if ferrule else '--tensor-parallel-size': '2'}
        if ferrule:
            expected.update({'--expert-parallel': '2', '--rank-backend': 'thread',
                             '--rank-timeout-ms': '30000', '--rank-restarts': '0'})
            numeric = flags.get('--numeric-fp8-precision')
            require(numeric in ('f32', 'w8a16-bf16'),
                    'BF16 boundaries are unsupported by this PP1/2 TP2 EP2 benchmark adapter; '
                    'single-owner Mixed serving does not authorize this topology')
            require(precision['profile'] == numeric, 'numeric profile differs from argv')
            require(precision['numeric_operands_dtype'] == ('f32' if numeric == 'f32' else 'bf16'),
                    'numeric operands differ')
            rounding = ('fused-M1 F32 FMA; multirow TF32x3' if numeric == 'f32' else
                        'BF16-RNE numeric operands; decode*scale separately rounded F32 before BF16')
            require(precision['rounding'] == rounding, 'numeric rounding contract differs')
            require(meta['source']['binary'] == binding['executable'], 'Ferrule binary identity differs')
            require(all(precision[k] == 'f32' for k in ('activation_dtype', 'accumulation_dtype',
                    'output_dtype', 'kv_dtype', 'state_boundary_dtype', 'gdn_recurrent_dtype',
                    'gdn_convolution_dtype')), 'Ferrule numeric mode does not change F32 boundaries')
            layout, count = 'separate-tp-ep', pp * 4
            devices = [int(x) for x in flags['--devices'].split(',')]
            expected_owners = [(stage, role, lane) for role in ('tp-root', 'ep-expert')
                               for stage in range(pp) for lane in range(2)]
        else:
            expected.update({'--data-parallel-size': '1', '--prefill-context-parallel-size': '1',
                '--enable-expert-parallel': True, '--distributed-executor-backend': 'mp',
                '--dtype': 'bfloat16', '--kv-cache-dtype': 'auto', '--mamba-cache-dtype': 'bfloat16',
                '--mamba-ssm-cache-dtype': 'float32', '--moe-backend': 'marlin', '--linear-backend': 'marlin'})
            require(precision['profile'] == 'marlin' and precision['rounding'] ==
                    'Marlin; not Ferrule numeric arithmetic', 'vLLM numeric backend contract differs')
            require(all(precision[k] == 'bf16' for k in ('activation_dtype', 'output_dtype',
                    'numeric_operands_dtype', 'state_boundary_dtype', 'kv_dtype', 'gdn_convolution_dtype'))
                    and precision['gdn_recurrent_dtype'] == 'f32', 'vLLM dtype/cache flags differ')
            layout, count = 'shared-tp-dp-pcp', pp * 2
            devices = list(range(count))
            expected_owners = [(stage, 'shared-tp-ep', lane) for stage in range(pp) for lane in range(2)]
        require(all(flags.get(k) == v for k, v in expected.items()), 'topology/runtime flags differ')
        require(t['rank_layout'] == layout and t['model_worker_count'] == count,
                'owner layout/worker count differs')
        require(precision['weights_dtype'] == 'fp8_e4m3fn' and
                meta['model']['locator'] == binding['model_path'], 'checkpoint/weight storage differs')
        policy = t['expert_policy']
        require(policy['rank_backend'] == ('thread' if ferrule else 'process')
                and policy['partition'] == 'round_robin' and policy['residency'] == 'all_owned',
                'expert owner policy differs')
        if not ferrule:
            require(flags.get('--expert-placement-strategy') == policy['partition'] and
                    flags.get('--all2all-backend') == policy['transport'], 'expert transport/placement flags differ')
        visible = env['CUDA_VISIBLE_DEVICES'].split(',')
        require(env['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID' and len(set(visible)) == len(visible)
                and all(x.isdigit() for x in visible), 'explicit unique PCI_BUS_ID visible ordinals required')
        rows = t['physical_placement']
        require(len(rows) == count and len(devices) == count, 'owner placement cardinality differs')
        for rank, (row, device, owner) in enumerate(zip(rows, devices, expected_owners)):
            require(0 <= device < len(visible) and row['owner'] == rank and
                    (row['pp'], row['role'], row['lane']) == owner and row['cuda_ordinal'] == device
                    and row['physical_ordinal'] == int(visible[device]), 'owner-to-physical placement differs')
    except (KeyError, ValueError, TypeError, IndexError, ProtocolError) as exc:
        reasons.append('launch contract: ' + str(exc))
    return reasons


def runtime_contract(profile, argv, log_bytes, cell=None):
    """Audit native logs, never synthesize observations from requested flags.

    Current logs expose plans/ranks, not all effective boundaries and physical
    UUIDs. Those missing observations must block fair measurement, even when the
    observed subset agrees. A future adapter must consume real runtime evidence.
    """
    reasons = launch_contract(profile, argv, cell)
    text = log_bytes.decode('utf-8', 'replace')
    observed = {}
    t = profile['metadata']['topology']
    if profile['engine'] == 'vllm':
        pattern = (r'rank (\d+) in world size (\d+) is assigned as DP rank (\d+), '
                   r'PP rank (\d+), PCP rank (\d+), TP rank (\d+), EP rank (\d+),')
        ranks = sorted(set(tuple(map(int, row)) for row in re.findall(pattern, text)))
        observed['rank_assignments'] = ranks
        expected = [(rank, t['model_worker_count'], 0, rank // 2, 0, rank % 2, rank % 2)
                    for rank in range(t['model_worker_count'])]
        if ranks != expected:
            reasons.append('runtime rank assignments missing or differ from topology')
    else:
        plans = re.findall(r'pipeline PP=(\d+) TP=(\d+) EP=(\d+) cuda Thread owners,', text)
        observed['plans'] = plans
        if set(plans) != {(str(t['pp']), str(t['tp']), str(t['ep']))}:
            reasons.append('runtime Ferrule topology plan missing or differs')
        placements = re.findall(r'CUDA ordinals \(PP-major TP roots, then stage EP children\): (\[[0-9, ]+\])', text)
        expected = [r['cuda_ordinal'] for r in t['physical_placement']]
        if not placements or any(json.loads(row) != expected for row in placements):
            reasons.append('runtime CUDA placement plan missing or differs')
        numeric = profile['metadata']['precision']['profile']
        marker = 'F32Tf32x3; F32 operands;' if numeric == 'f32' else 'W8A16Bf16F32Accumulate; BF16-RNE operands;'
        if marker not in text or 'F32 accumulation/output/model/KV/state boundaries' not in text:
            reasons.append('runtime numeric/boundary description missing or differs')
    reasons.extend(['runtime per-owner physical GPU UUID/device observations unavailable',
                    'runtime effective numeric/KV/GDN boundary observations incomplete'])
    return dict(schema='ferrule.runtime_contract.v1', status='blocked', reasons=sorted(set(reasons)),
                argv_sha256=sha256(json.dumps(argv).encode()), server_log_sha256=sha256(log_bytes),
                observed=observed, note='Native plans are not evidence of successful owner construction')


def cell_blockers(manifest, profile, cell):
    metadata, reasons = profile['metadata'], launch_contract(profile)
    if profile.get('comparison_intent') == 'exact':
        peers = [p for p in manifest['profiles'] if p['engine'] != profile['engine']]
        if not peers:
            reasons.append('fair execution requires an opposite-engine peer')
        for peer in peers:
            if peer.get('comparison_intent') != 'exact':
                reasons.append('fair execution requires an exact-intent peer')
            reasons.extend(launch_contract(peer))
            for field in ('activation_dtype', 'state_boundary_dtype', 'kv_dtype', 'output_dtype',
                          'gdn_recurrent_dtype', 'gdn_convolution_dtype'):
                if metadata.get('precision', {}).get(field) != peer['metadata'].get('precision', {}).get(field):
                    reasons.append('fair execution blocked: precision.' + field + ' differs')
            reasons.extend('fair execution blocked: ' + reason for reason in
                           comparison_contract(metadata, peer['metadata'])['reasons'])
        reasons.extend('fair execution blocked: ' + reason for reason in
                       comparison_contract(metadata, metadata)['reasons'])
    cap = declared(metadata, 'engine_capability', CAPABILITY_FIELDS)
    if cap is None:
        reasons.append('unknown engine capability')
    else:
        for field, required in (('max_context_tokens', cell['context_tokens']),
                                ('max_concurrency', cell['concurrency'])):
            if type(cap[field]) is not int or cap[field] < required:
                reasons.append('unsupported ' + field)
        if cap['streaming'] is not True or cap['raw_completions'] is not True or cap['eos_policy'] != 'permitted':
            reasons.append('unsupported HTTP/EOS contract')
    if metadata.get('tokenizer') != manifest['corpus']['tokenizer']:
        reasons.append('tokenizer identity/encoding differs or unknown')
    cache = declared(metadata, 'cache', ('prefix_mode', 'os_cache', 'state_reset'))
    if cache is None or cache['prefix_mode'] not in ('off', 'first-cold-then-warm') or cache['state_reset'] != 'per_request':
        reasons.append('unknown/unsupported cache or request state reset')
    elif cache['prefix_mode'] == 'first-cold-then-warm':
        rows = [r for r in manifest['corpus']['prompts'] if r['input_tokens'] == cell['prefill_tokens']]
        if manifest['warmup'] < len(rows) or cap is None or cap['prefix_cache'] is not True:
            reasons.append('prefix warm requires supported cache and priming every prompt')
        if metadata['cache'].get('fresh_process_empty_prefix') is not True:
            reasons.append('prefix cold requires fresh-process empty-cache declaration')
    ownership = profile.get('ownership') or {}
    if not all(ownership.get(k) is True for k in ('exclusive_resources', 'no_daemonize')):
        reasons.append('owner must attest exclusive resources and a foreground server root')
    # Group membership/rank backend is not custody authority. Separate-session
    # descendants are handled by the verified subreaper handshake at run time.
    return reasons


def dump_json(path, value):
    """Atomic checkpoint inside an exclusively reserved experiment directory."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def supervisor_module():
    # The existing test script's bounded session/log/kill supervisor; no check/GPU mode is invoked.
    import importlib.util
    spec = importlib.util.spec_from_file_location('ferrule_vllm_supervisor', SCRIPT.with_name('test_vllm_qwen35.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def own_group_members():
    """Linux local PID namespace only. Never map NVML/foreign namespace PIDs."""
    require(Path('/proc/self/stat').is_file(), 'Own-group drain inspection requires Linux /proc')
    members = []
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == os.getpgrp() and int(path.parent.name) != os.getpid():
                members.append(int(path.parent.name))
        except FileNotFoundError:
            pass
    return members


def shutdown_owned_group(server, timeout):
    """Compatibility interface: graceful direct-root shutdown, not a group-drain proof.

    The outer isolated subreaper owns all adopted descendants and wait/reap authority. Do
    not race it with killpg or numeric PID signals after the root is reaped.
    """
    require(os.getpgrp() == os.getpid(), 'Worker must own its launch session')
    started = time.monotonic()
    timed_out = False
    try:
        if server.poll() is None:
            server.terminate()  # unreaped direct child; no PID-reuse window
        server.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
    return dict(status='shutdown_timeout' if timed_out else 'root_exited_descendants_unknown',
                seconds=time.monotonic()-started, timed_out=timed_out,
                gpu_quiescence='unverified; no GPU probe')


def wait_ready(profile, server, timeout):
    """Whole-phase alarm also bounds DNS/header/body drips; HTTP Ready is not load/prewarm telemetry."""
    started = time.monotonic()
    url = profile['url'].replace('/v1/completions', '/v1/models')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    previous = signal.signal(signal.SIGALRM, deadline_signal)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        while time.monotonic()-started < timeout:
            require(server.poll() is None, 'Server exited before Ready')
            try:
                with opener.open(url, timeout=min(1, timeout)) as response:
                    body = response.read(MAX_BODY + 1)
                    require(len(body) <= MAX_BODY, 'Ready body exceeds limit')
                    data = strict_json(body.decode())
                    if response.status == 200 and any(row.get('id') == profile['model']
                                                      for row in data.get('data', [])):
                        return time.monotonic()
            except (OSError, ValueError):
                pass
            time.sleep(.1)
        raise TimeoutError('Startup deadline exceeded')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def matrix_worker(config):
    """Only called in the new session owned by the existing bounded supervisor."""
    manifest, profile, cell = config['manifest'], config['profile'], config['cell']
    directory = Path(config['directory'])
    lifecycle = dict(status='starting', phases={}, error=None)
    started, server = time.monotonic(), None
    require(os.getpgrp() == os.getpid(), 'Worker must be supervised in a fresh process group')
    limits = manifest['limits']
    log = ((directory / 'server.log').open('x+b') if manifest.get('profiling')
           else tempfile.TemporaryFile())
    try:
        require(not cell_blockers(manifest, profile, cell), 'Profile blocked by capability/custody gate')
        require(not own_group_members(), 'Unexpected prior members in fresh process group')
        url = urllib.parse.urlsplit(profile['url'])
        with socket.socket() as reservation:
            # Permit a previous run's TIME_WAIT, but never SO_REUSEPORT sharing.
            # listen also rejects a concurrently bound live listener.
            reservation.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            reservation.bind(('127.0.0.1', url.port))
            reservation.listen(1)
        argv = [arg.replace('{context_tokens}', str(cell['context_tokens']))
                .replace('{concurrency}', str(cell['concurrency'])) for arg in profile['argv']]
        require(not launch_contract(profile, argv, cell), 'Resolved launch contract mismatch')
        if manifest.get('profiling'):
            argv = profiling_argv(manifest['profiling'], profile, cell, directory, argv)
            lifecycle['profiling'] = manifest['profiling']
            (directory / 'nsys-tmp').mkdir(mode=0o700)
        lifecycle.update(argv=argv, status='startup',
            prefix=dict(first_after_ready=('cold_declared' if profile['metadata']['cache']['prefix_mode']
                        == 'first-cold-then-warm' else 'off'),
                        measured=('warm_primed_not_hit_verified' if profile['metadata']['cache']['prefix_mode']
                        == 'first-cold-then-warm' else 'off'), os_cache=profile['metadata']['cache']['os_cache']))
        dump_json(directory / 'lifecycle.json', lifecycle)
        # The outer supervisor tracks descendants during execution; a group is not containment.
        server = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        ready = wait_ready(profile, server, limits['startup_seconds'])
        lifecycle['phases']['startup'] = dict(started_at=started, ready_at=ready, seconds=ready-started,
                                               load_seconds=None, prewarm_seconds=None)
        if 'launch_contract' in profile:
            log_bytes = os.pread(log.fileno(), min(os.fstat(log.fileno()).st_size, MAX_BODY), 0)
            (directory / 'runtime-startup.log').write_bytes(log_bytes)
            audit = runtime_contract(profile, argv, log_bytes, cell)
            audit['log_truncated'] = os.fstat(log.fileno()).st_size > len(log_bytes)
            lifecycle['runtime_contract'] = audit
            profile['metadata']['runtime_contract'] = audit
            dump_json(directory / 'runtime-contract.json', audit)
            if profile.get('comparison_intent') == 'exact' and audit['status'] != 'verified':
                raise ContractBlocked('Runtime contract blocked: ' + '; '.join(audit['reasons']))
        lifecycle['status'] = 'warmup_measure'
        dump_json(directory / 'lifecycle.json', lifecycle)
        rows = [row for row in manifest['corpus']['prompts'] if row['input_tokens'] == cell['prefill_tokens']]
        dump_json(directory / 'prompts.json', rows)
        dump_json(directory / 'metadata.json', profile['metadata'])
        code = main(['--engine', profile['engine'], '--url', profile['url'], '--model', profile['model'],
                     '--prompts-json', str(directory / 'prompts.json'), '--input-tokens', str(cell['prefill_tokens']),
                     '--max-tokens', str(cell['output_tokens']), '--seed', str(manifest['seed']),
                     '--warmup', str(manifest['warmup']), '--repeat', '1', '--requests', str(manifest['requests']),
                     '--concurrency', str(cell['concurrency']), '--timeout', str(limits['request_seconds']),
                     '--measurement-seconds', str(limits['measurement_seconds']), '--window-kind', 'fixed-work',
                     *([] if manifest.get('profiling') else ['--first-after-ready']),
                     '--metadata', str(directory / 'metadata.json'),
                     '--output', str(directory / 'http.json')])
        lifecycle['status'] = 'complete' if code == 0 else 'request_failure'
        http_path = directory / 'http.json'
        if http_path.exists():
            http = strict_json(http_path.read_text())
            lifecycle['phases']['first_after_ready'] = http.get('first_after_ready')
            lifecycle['phases']['warmup'] = http.get('phases', [])
            lifecycle['phases']['measure'] = [r['measurements'] for r in http.get('runs', [])]
    except (Exception, KeyboardInterrupt) as exc:
        lifecycle.update(status='blocked' if isinstance(exc, ContractBlocked) else
                         'timeout' if isinstance(exc, TimeoutError) else 'failed',
                         error=f'{type(exc).__name__}: {exc}')
    finally:
        lifecycle['phases'].setdefault('startup', dict(started_at=started, seconds=time.monotonic()-started,
                                                       ready_at=None, load_seconds=None, prewarm_seconds=None))
        if server is not None:
            try:
                lifecycle['phases']['shutdown'] = (
                    shutdown_profile_capture(server, directory, limits['teardown_seconds'])
                    if manifest.get('profiling') else shutdown_owned_group(server, limits['teardown_seconds']))
            except Exception as exc:
                lifecycle['phases']['shutdown'] = dict(status='cleanup_unknown', error=str(exc), timed_out=True)
        else:
            lifecycle['phases']['shutdown'] = dict(status='not_spawned', timed_out=False)
        lifecycle['work_status'] = lifecycle['status']
        if manifest.get('profiling'):
            try:
                lifecycle['profile_artifacts'] = profiling_artifacts(directory)
                if lifecycle['status'] == 'complete' and (
                        lifecycle['profile_artifacts']['status'] != 'present_content_unverified'
                        or lifecycle['phases']['shutdown'].get('returncode') != 0):
                    lifecycle['status'] = 'profile_incomplete'
            except Exception as exc:
                lifecycle.setdefault('reporting_errors', []).append(f'Profile artifacts: {exc}')
                if lifecycle['status'] == 'complete':
                    lifecycle['status'] = 'profile_incomplete'
        if lifecycle['phases']['shutdown']['timed_out']:
            lifecycle['status'] = 'shutdown_timeout'
        # Preserve work failure before ancillary log extraction and checkpointing.
        dump_json(directory / 'lifecycle.json', lifecycle)
        try:
            # Retain complete evidence as well as the scrubbed report tail.
            log.seek(0)
            if not manifest.get('profiling'):
                with (directory / 'server.log').open('wb') as saved:
                    while chunk := log.read(1024 * 1024):
                        saved.write(chunk)
            size = os.fstat(log.fileno()).st_size
            log.seek(max(0, size - 65536))
            lifecycle['server_log_tail'] = supervisor_module().scrub(log.read(65536).decode('utf-8', 'replace'))
            lifecycle['server_log_truncated'] = size > 65536
            lifecycle['oom_observed_in_log'] = any(term in lifecycle['server_log_tail'].lower()
                                                   for term in ('out of memory', 'outofmemoryerror', 'cuda oom'))
        except Exception as exc:
            lifecycle.setdefault('reporting_errors', []).append(f'Log: {type(exc).__name__}: {exc}')
        finally:
            log.close()
        dump_json(directory / 'lifecycle.json', lifecycle)
    return 0 if lifecycle['status'] == 'complete' else 1


def summarize_matrix(manifest, runs):
    summaries = []
    for profile in manifest['profiles']:
        for cell in manifest['cells']:
            group = [run for run in runs if run['profile_id'] == profile['id'] and run['cell_id'] == cell['id']]
            rates, fractions = [], []
            for run in group:
                summary = run.get('http', {}).get('aggregate', {})
                fractions.append(summary.get('success', 0) / manifest['requests'])
                rates.append(summary.get('successful_output_tokens_per_second')
                             if run['status'] == 'complete' else None)
            summaries.append(dict(profile_id=profile['id'], cell_id=cell['id'],
                scheduled_process_runs=len(group), statuses=dict(Counter(run['status'] for run in group)),
                scheduled_requests=len(group)*manifest['requests'],
                offered_known=sum(run.get('http', {}).get('aggregate', {}).get('offered', 0) for run in group),
                unobserved_request_budget=sum(run.get('http', {}).get('aggregate', {}).get('unobserved_request_budget',
                    manifest['requests'] if 'http' not in run and run['status'] not in ('not_run', 'blocked') else 0)
                    for run in group),
                not_issued_known=sum(run.get('http', {}).get('aggregate', {}).get('not_issued',
                    manifest['requests'] if run['status'] in ('not_run', 'blocked') else 0)
                    for run in group),
                successful=sum(run.get('http', {}).get('aggregate', {}).get('success', 0) for run in group),
                failed_known=sum(run.get('http', {}).get('aggregate', {}).get('failed', 0) for run in group),
                counts_definition='Observed terminal requests only; missing/in-flight outcomes remain unknown',
                success_fraction_ci=bootstrap_ci(fractions, manifest['seed']),
                success_fraction_definition='observed successes / fixed scheduled requests; lower bound if report missing',
                throughput_ci=bootstrap_ci(rates, manifest['seed'])))
    return summaries


def matrix_comparisons(manifest, runs):
    comparisons = []
    profiles = manifest['profiles']
    for i, left in enumerate(profiles):
        for right in profiles[i+1:]:
            for cell in manifest['cells']:
                pairs, reasons = [], []
                for repetition in range(manifest['repetitions']):
                    pair = []
                    for profile in (left, right):
                        matches = [r for r in runs if r.get('profile_id') == profile['id']
                                   and r.get('cell_id') == cell['id']
                                   and r.get('repetition') == repetition]
                        if len(matches) != 1:
                            pair = []
                            reasons.append('missing/duplicate scheduled process run')
                            break
                        pair.append(matches[0])
                    if len(pair) != 2 or any(r.get('status') != 'complete' for r in pair):
                        reasons.append('missing/failed/blocked process run')
                        continue
                    try:
                        reports = [strict_json(Path(r['http_report']).read_text()) if r.get('http_report')
                                   else r['http'] for r in pair]
                        # Re-read this run's evidence, not an operator-declared PASS.
                        for profile, run in zip((left, right), pair):
                            if profile.get('comparison_intent') == 'exact':
                                directory = Path(run['directory'])
                                life = strict_json((directory / 'lifecycle.json').read_text())
                                runtime = runtime_contract(profile, life['argv'],
                                    (directory / 'runtime-startup.log').read_bytes(), cell)
                                require(runtime['status'] == 'verified', 'Current-run runtime evidence blocked')
                        audit = comparison_eligibility(*reports)
                        if not audit['comparability']:
                            reasons.extend(audit['reasons'] + audit['labels']['reasons'])
                            continue
                        rates = tuple(r['http']['aggregate']['successful_output_tokens_per_second'] for r in pair)
                        if any(type(rate) not in (int, float) or not math.isfinite(rate) for rate in rates):
                            reasons.append('missing/invalid complete-run throughput')
                            continue
                        pairs.append(rates)
                    except (OSError, ValueError, KeyError, TypeError, ProtocolError) as exc:
                        reasons.append(f'child report unavailable: {type(exc).__name__}')
                complete = len(pairs) == manifest['repetitions'] and not reasons
                comparisons.append(dict(left=left['id'], right=right['id'], cell_id=cell['id'],
                    scheduled_pairs=manifest['repetitions'], eligible_pairs=len(pairs), comparability=complete,
                    fairmatch=complete, reasons=sorted(set(reasons)),
                    labels=comparison_contract(left['metadata'], right['metadata']), speedup=None,
                    paired_rate_difference_ci=(bootstrap_ci([b-a for a, b in pairs], manifest['seed'])
                                               if complete else None)))
    return comparisons


class MatrixInterrupted(KeyboardInterrupt):
    """Catchable owner termination; SIGKILL cannot be finalized in-process."""


def read_matrix_artifacts(run, directory):
    """Best-effort child evidence ingestion, independent of supervisor outcome."""
    for name, filename in (('lifecycle', 'lifecycle.json'), ('http', 'http.json'), ('custody_checkpoint', 'custody.json')):
        path = directory / filename
        if not path.exists():
            continue
        try:
            data = strict_json(path.read_text())
            require(isinstance(data, dict), 'Child artifact must be a JSON object')
            if name == 'http':
                require(data.get('schema') == 'ferrule.http_bench.v1'
                        and isinstance(data.get('aggregate'), dict)
                        and isinstance(data.get('measurements'), dict)
                        and isinstance(data.get('status'), str), 'Incomplete HTTP report schema')
                summary = data['aggregate']
                require(all(type(summary.get(k)) is int and summary[k] >= 0
                            for k in ('success', 'offered', 'not_issued')), 'Incomplete HTTP aggregate counts')
                run['http_report'] = str(path)
                run[name] = {key: data[key] for key in ('schema', 'status', 'aggregate', 'measurements')}
            elif name == 'custody_checkpoint':
                require(isinstance(data.get('root_identity'), dict) and isinstance(data.get('entries'), list)
                        and isinstance(data.get('launch_id'), str), 'Incomplete custody checkpoint schema')
                run[name] = data  # evidence only; never a cleanup proof or PID-kill authority
            else:
                require(isinstance(data.get('status'), str) and isinstance(data.get('phases'), dict),
                        'Incomplete lifecycle report schema')
                require(isinstance(data['phases'].get('shutdown', {}), dict), 'Invalid shutdown phase')
                run[name] = data
        except (OSError, ValueError, KeyError, TypeError, ProtocolError) as exc:
            error = dict(artifact=str(path), error=f'{type(exc).__name__}: {exc}')
            if error not in run.setdefault('artifact_errors', []):
                run['artifact_errors'].append(error)


def checkpoint_matrix(path, report):
    """Every checkpoint has summary keys; secondary failures cannot hide execution."""
    report['checkpoint_unix'] = time.time()
    for key, operation in (('summaries', summarize_matrix), ('comparisons', matrix_comparisons)):
        try:
            report[key] = operation(report['manifest'], report['runs'])
        except Exception as exc:
            report[key] = []
            error = dict(phase=key, error=f'{type(exc).__name__}: {exc}')
            if error not in report['reporting_errors']:
                report['reporting_errors'].append(error)
    dump_json(path, report)


def execute_matrix(args, manifest):
    report = dict(schema='ferrule.http_matrix.v1', manifest_sha256=sha256(args.manifest.read_bytes()),
                  manifest=manifest, budget=matrix_budget(manifest), executed=args.execute, status='in_progress' if args.execute else 'planned',
                  finalized=False, summaries=[], comparisons=[], reporting_errors=[], errors=[], halted_for_cleanup=False,
                  client_sha256=sha256(SCRIPT.read_bytes()),
                  supervisor_sha256=sha256(SCRIPT.with_name('test_vllm_qwen35.py').read_bytes()),
                  supervisor_helper_sha256=sha256(SCRIPT.with_name('_owned_subreaper.py').read_bytes()),
                  **(dict(profiling_owner_attestation=bool(args.execute and args.gpu_owner_ready))
                     if manifest.get('profiling') else {}),
                  runs=[dict(slot, status='not_run', scheduled=manifest['requests']) for slot in manifest['schedule']],
                  notes=['Isolated subreaper signals only its kernel-verified unreaped direct children',
                         'An unfinalized checkpoint is last-known state, never proof of cleanup or non-execution',
                         'Complete drain requires verified helper exit and kernel ECHILD; GPU quiescence remains unverified'])
    path = args.output / 'matrix.json'
    halted, active, directory = False, None, None
    previous = {}

    def interrupt(signum, frame):
        # Let bounded supervisor cleanup/final checkpoint finish after the first signal.
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        raise MatrixInterrupted(f'Owner signal {signal.Signals(signum).name}')

    try:
        if args.execute:
            for sig in (signal.SIGTERM, signal.SIGINT):
                previous[sig] = signal.signal(sig, interrupt)
        checkpoint_matrix(path, report)
        supervisor = supervisor_module() if args.execute else None
        if args.execute:
            report['custody_capabilities'] = supervisor.custody_capabilities()
        for run in report['runs']:
            profile = next(p for p in manifest['profiles'] if p['id'] == run['profile_id'])
            cell = next(c for c in manifest['cells'] if c['id'] == run['cell_id'])
            reasons = cell_blockers(manifest, profile, cell)
            if args.execute and report['custody_capabilities'].get('containment') is not True:
                reasons.append('supervisor containment unavailable: no complete descendant custody proof; launch refused')
            if reasons:
                run.update(status='blocked', reasons=reasons)
            elif args.execute and not halted:
                active = run
                directory = (args.output / f'run-{run["index"]:05d}').resolve()
                directory.mkdir()
                config = dict(manifest=manifest, profile=profile, cell=cell, directory=str(directory))
                budget = supervisor_budget(manifest, cell)
                run.update(status='in_progress', started_unix=time.time(), directory=str(directory),
                           supervisor_budget_seconds=budget, request_outcomes='unknown until child checkpoint')
                # Persist dispatch BEFORE launching. A hard kill leaves in_progress, never not_run.
                checkpoint_matrix(path, report)
                try:
                    process = supervisor.run_process([sys.executable, '-B', str(SCRIPT), '--_matrix_worker'], budget, config)
                    run['supervisor'] = process
                    read_matrix_artifacts(run, directory)
                    lifecycle = run.get('lifecycle', {})
                    run['status'] = lifecycle.get('status', 'supervisor_failure')
                    if process['timed_out'] or process.get('error'):
                        run['status'] = 'supervisor_timeout' if process['timed_out'] else 'supervisor_failure'
                    if run['status'] == 'complete' and (process['returncode'] != 0
                            or run.get('http', {}).get('status') != 'complete' or run.get('artifact_errors')):
                        run['status'] = 'supervisor_failure'
                    custody_proven = process.get('custody', {}).get('status') == 'proven_drained'
                    halted = (process['timed_out'] or bool(process.get('error')) or not custody_proven
                              or (bool(manifest.get('profiling')) and run['status'] != 'complete'))
                    if not custody_proven and run['status'] == 'complete':
                        run.update(work_status='complete', status='cleanup_unknown')
                except (Exception, KeyboardInterrupt) as exc:
                    run.update(status='interrupted', error=f'{type(exc).__name__}: {exc}')
                    if hasattr(exc, 'supervisor_result'):
                        run['supervisor'] = exc.supervisor_result
                    read_matrix_artifacts(run, directory)
                    halted = True
                run['ended_unix'] = time.time()
                report['halted_for_cleanup'] = halted
                checkpoint_matrix(path, report)
                active = None
            checkpoint_matrix(path, report)
    except (Exception, KeyboardInterrupt) as exc:
        report['errors'].append(f'{type(exc).__name__}: {exc}')
        halted = True
        if active is not None:
            active.setdefault('error', f'{type(exc).__name__}: {exc}')
            active['status'] = 'interrupted'
            if directory is not None:
                read_matrix_artifacts(active, directory)
    finally:
        report.update(finalized=True, halted_for_cleanup=halted,
                      status=('complete' if all(r['status'] == 'complete' for r in report['runs'])
                              and not report['errors'] and not report['reporting_errors'] else 'incomplete')
                             if args.execute else 'planned')
        try:
            checkpoint_matrix(path, report)
            if args.execute and report['reporting_errors'] and report['status'] == 'complete':
                report['status'] = 'incomplete'
                dump_json(path, report)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    return 0 if not args.execute or report['status'] == 'complete' else 1


def matrix_main(argv):
    cli = argparse.ArgumentParser(description='Offline matrix plan or explicit bounded supervised execution')
    cli.add_argument('mode', choices=('matrix-corpus', 'matrix-plan', 'matrix-run'))
    cli.add_argument('--output', required=True, type=Path, help='New file (corpus/plan) or new directory (run)')
    cli.add_argument('--tokenizer-dir', type=Path, help='matrix-corpus only: local tokenizer assets, never downloaded')
    cli.add_argument('--revision', help='matrix-corpus only: owner-pinned tokenizer revision')
    cli.add_argument('--manifest', type=Path)
    cli.add_argument('--corpus', type=Path)
    cli.add_argument('--profiles', type=Path)
    cli.add_argument('--cell', action='append', help='Plan only: explicit cell ID; repeat for a small subset')
    cli.add_argument('--profile-id', help='Plan only: select exactly one profile from the input list')
    cli.add_argument('--limits-json', type=Path, help='Plan only: bounded limits, no larger than protocol defaults')
    cli.add_argument('--max-server-runs', type=positive_int, default=10,
                     help='Execution safety cap; larger schedules require an explicit override')
    cli.add_argument('--profiling-binary', type=Path, help='Plan only: opt-in warm1/measure1 out1 profiling; pin this instrumented CLI')
    cli.add_argument('--nsys', type=Path, help='Profiling plan only: existing nsys, bounded whole-lifecycle capture')
    cli.add_argument('--profile-request-seconds', type=int, choices=(180, 600),
                     help='Profiling plan only: per-request and measured-window budget (default 180)')
    cli.add_argument('--gpu-owner-ready', action='store_true',
                     help='Execution only: attest current GPU handoff, exclusive devices and prelaunch quiescence')
    cli.add_argument('--execute', action='store_true', help='Only this opt-in starts servers and sends inference traffic')
    cli.add_argument('--seed', type=nonnegative_int, default=0)
    cli.add_argument('--process-runs', type=positive_int, default=5)
    cli.add_argument('--requests', type=positive_int, default=100)
    cli.add_argument('--warmup', type=nonnegative_int, default=8)
    args = cli.parse_args(argv)
    try:
        if args.mode == 'matrix-corpus':
            require(args.tokenizer_dir and args.revision and not args.execute,
                    'Corpus export requires --tokenizer-dir and --revision; never --execute')
            export_corpus(args.tokenizer_dir, args.revision, args.output)
            return 0
        if args.mode == 'matrix-plan':
            require(args.corpus and args.profiles and not args.execute, 'Plan requires --corpus and --profiles, no --execute')
            profiles = strict_json(args.profiles.read_text())
            validate_profiles(profiles)
            if args.profile_id:
                profiles = [p for p in profiles if p['id'] == args.profile_id]
                require(profiles, 'Unknown profile ID')
            profiling = prepare_profiling(profiles, args.profiling_binary, args.nsys,
                                          args.profile_request_seconds or 180) if args.profiling_binary else None
            require(not (args.nsys or args.profile_request_seconds) or profiling,
                    '--nsys/--profile-request-seconds require --profiling-binary')
            limits = strict_json(args.limits_json.read_text()) if args.limits_json else None
            if profiling:
                require(limits is None, 'Profiling uses fixed limits; no --limits-json')
                limits = dict(startup_seconds=300, request_seconds=profiling['request_seconds'],
                              measurement_seconds=profiling['request_seconds'], teardown_seconds=120)
            manifest = build_manifest(strict_json(args.corpus.read_text()), profiles,
                                      args.seed, args.process_runs, args.requests, args.warmup, args.cell,
                                      limits, profiling)
            with args.output.open('x') as stream:
                json.dump(manifest, stream, indent=2, allow_nan=False)
                stream.write('\n')
            print(f'Planned {len(manifest["cells"])} cells/profile; {len(manifest["schedule"])} fresh server runs. No traffic.')
            return 0
        require(args.manifest, 'Run requires --manifest')
        require(not args.cell and not args.profile_id and not args.limits_json
                and not args.profiling_binary and not args.nsys and not args.profile_request_seconds,
                'Run cannot override frozen plan cells/profiles/limits')
        manifest = strict_json(args.manifest.read_text())
        require(manifest.get('schema') == 'ferrule.http_matrix_manifest.v1', 'Manifest schema mismatch')
        expected = build_manifest(manifest['corpus'], manifest['profiles'], manifest['seed'],
                                  manifest['repetitions'], manifest['requests'], manifest['warmup'],
                                  manifest.get('selected_cell_ids'), manifest['limits'], manifest.get('profiling'))
        require(manifest == expected, 'Manifest does not match canonical protocol/schedule; regenerate it')
        if args.execute:
            if manifest.get('profiling'):
                require(args.gpu_owner_ready, 'Profiling requires a fresh --gpu-owner-ready handoff attestation')
                for path, expected_hash in manifest['profiling']['pins'].items():
                    require(file_digest(path) == expected_hash, f'Profiling artifact changed: {path}')
            require(len(manifest['schedule']) <= args.max_server_runs,
                    'Schedule exceeds --max-server-runs; select a small stage or explicitly raise the cap')
            require(Path('/proc/self/stat').is_file(), 'Supervised execution requires Linux /proc')
        args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    except (OSError, ValueError, KeyError, TypeError, ProtocolError) as exc:
        cli.error(str(exc))
    return execute_matrix(args, manifest)


def comparison_eligibility(left, right):
    """Never return a blind speedup; caller must audit resources/quality too."""
    reasons = []
    for key in ('corpus', 'workload'):
        if left.get(key) != right.get(key):
            reasons.append(key + ' differs')
    for report in (left, right):
        if report.get('status') != 'complete':
            reasons.append('incomplete/error run')
    lengths = []
    for report in (left, right):
        signature = []
        for run in report.get('runs', []):
            if run['summary']['not_issued']:
                reasons.append('unissued requests')
            for request in run['requests']:
                if request['status'] != 'success' or request['actual_input_tokens'] is None or request['actual_output_tokens'] is None:
                    reasons.append('failed/unknown-count request')
                if request.get('input_length_matches') is False or request.get('output_length_matches') is False:
                    reasons.append('declared input/output length mismatch')
                signature.append((request['repetition'], request['index'], request['prompt_sha256'],
                                  request['actual_input_tokens'], request['actual_output_tokens']))
        lengths.append(signature)
    if not lengths[0] or lengths[0] != lengths[1]:
        reasons.append('actual per-request input/output lengths differ or are missing')
    audit = comparison_contract(left.get('metadata', {}), right.get('metadata', {}))
    for report in (left, right):
        runtime = report.get('metadata', {}).get('runtime_contract', {})
        runtime = runtime if isinstance(runtime, dict) else {}
        if (runtime.get('schema') != 'ferrule.runtime_contract.v1' or runtime.get('status') != 'verified'
                or runtime.get('reasons') != [] or any(not isinstance(runtime.get(key), str)
                or re.fullmatch(r'[0-9a-f]{64}', runtime[key]) is None
                for key in ('argv_sha256', 'server_log_sha256'))):
            audit['contract_match'] = False
            audit['reasons'].append('runtime argv/topology/precision/placement evidence missing or blocked')
    if any(r.get('input_length_matches') is not True or r.get('output_length_matches') is not True
           for report in (left, right) for run in report.get('runs', []) for r in run['requests']):
        audit['contract_match'] = False
        audit['reasons'].append('input/output length verification missing or failed')
    return dict(length_workload_eligible=not reasons, reasons=sorted(set(reasons)), speedup=None,
                comparability=not reasons and audit['contract_match'],
                fairmatch=not reasons and audit['contract_match'], labels=audit,
                note='Operator-declared audit, not independent verification; speedup remains null')


def checkpoint_http(report, args):
    """Persist observed terminals; never call unobserved in-flight work not-issued."""
    records = [r for run in report['runs'] for r in run['requests']] + report.get('active_requests', [])
    summary = aggregate(records, None, args.requests*args.repeat, args.ttft_slo, args.e2e_slo)
    known_unissued = sum(run['summary']['not_issued'] for run in report['runs'])
    if not report.get('measurement_in_progress'):
        known_unissued += (args.repeat-len(report['runs']))*args.requests
    summary.update(offered_definition='observed terminal requests; lower bound while measurement is in progress',
                   not_issued=known_unissued,
                   unobserved_request_budget=args.requests*args.repeat-len(records)-known_unissued)
    report.update(aggregate=summary, finalized=False, checkpoint_unix=time.time(),
                  measurements=dict(work_completed=False, denominator_seconds=None,
                                    accounting='partial terminal observations; no interrupted-window throughput'))
    dump_json(args.output, report)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ('matrix-corpus', 'matrix-plan', 'matrix-run'):
        return matrix_main(argv)
    cli = parser()
    args = cli.parse_args(argv)
    url = urllib.parse.urlsplit(args.url)
    if (url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password
            or url.query or url.fragment or not url.path.endswith('/v1/completions')):
        cli.error('--url must be a raw /v1/completions URL without credentials/query/fragment')
    if args.seed < 0:
        cli.error('--seed must be >= 0')
    try:
        corpus, metadata = corpus_for(args), metadata_for(args)
        reference = strict_json(args.compare.read_text()) if args.compare else None
        if reference is not None:
            require(isinstance(reference, dict) and reference.get('schema') == 'ferrule.http_bench.v1',
                    'Comparison report schema mismatch')
    except (ValueError, OSError, ProtocolError) as exc:
        cli.error(str(exc))
    workload = dict(max_tokens=args.max_tokens, seed=args.seed, temperature=0, n=1,
                    endpoint='raw completions', template='none; literal UTF-8 prompt',
                    input_tokens_expected=args.input_tokens, warmup=args.warmup, requests=args.requests,
                    repeat=args.repeat, concurrency=args.concurrency, request_timeout_seconds=args.timeout,
                    measurement_seconds=args.measurement_seconds, window_kind=args.window_kind,
                    arrival='finite closed-loop; no retries',
                    ttft_slo=args.ttft_slo, e2e_slo=args.e2e_slo)
    report = dict(schema='ferrule.http_bench.v1', engine=args.engine, url=args.url, served_model=args.model,
                  created_unix=time.time(), status='incomplete', metadata=metadata, workload=workload,
                  corpus=corpus, client=dict(script_sha256=sha256(SCRIPT.read_bytes()),
                                            python=sys.version, executable=sys.executable, platform=platform.platform(),
                                            transport='urllib HTTP/1.1, new TCP/TLS per request; no proxy; persistent CPU client workers',
                                            checkpointing=('atomic fsync at phases and observed terminals; client checkpoint cost included in measured wall time'
                                                           if args.first_after_ready else 'atomic fsync at phases; no per-terminal measured checkpoint')),
                  definitions=dict(send='client HTTP operation start, includes serialization/connect/TLS; not packet-level send',
                                   ttft='send to first nonempty content SSE event; not exact token time',
                                   request_e2e='send to full response EOF/error/deadline; DONE time is separately recorded',
                                   content_event_itl='gaps between nonempty content events, possibly multi-token; NOT token ITL',
                                   mean_tpot_usage_proxy='(last content event - first content event)/(usage.completion_tokens - 1); success, N>1, >=2 events, no observed reasoning; batching/hidden tokens can bias it',
                                   goodput='successful usage tokens or length-matched requests divided by the selected common wall denominator; no quality claim; optional SLO thresholds explicit',
                                   percentiles='linear interpolation at (count-1)*p; separate success/failure populations',
                                   repeats='same-server windows, not independent cold replicas',
                                   wall='fixed-time: measurement_seconds including padded idle; fixed-work: first measured HTTP send (dispatch fallback if none) to last client-observed terminal/deadline, no padded idle; failures stay in both denominators'),
                  warmups=[], runs=[], errors=[], comparison_speedup=None)
    if args.first_after_ready:
        report['workload']['first_after_ready'] = 'separate excluded request before warmup'
    # Reserve the artifact before any traffic. Do not overwrite another experiment.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x'):
        checkpoint_http(report, args)
        clients = []
        try:
            for _ in range(args.concurrency):
                clients.append(Client())
            if args.first_after_ready:
                item = corpus[0]
                first = clients[0].request(
                    dict(url=args.url, payload=make_payload(args.model, item['prompt'], args.max_tokens, args.seed),
                         timeout=args.timeout),
                    dict(index=0, repetition=0, population='first_after_ready', prompt_id=item['id'],
                         prompt_sha256=item['sha256'], expected_input_tokens=item['input_tokens']))
                report['first_after_ready'] = first
                checkpoint_http(report, args)
                require(first['status'] == 'success', 'First-after-Ready failed; no warmup/measure')
            for repetition in range(args.repeat):
                warmup_started = time.monotonic()
                warmup_failed = False
                for index in range(args.warmup):
                    item = corpus[index % len(corpus)]
                    result = clients[index % len(clients)].request(
                        dict(url=args.url, payload=make_payload(args.model, item['prompt'], args.max_tokens, args.seed),
                             timeout=args.timeout),
                        dict(index=index, repetition=repetition, population='warmup', prompt_id=item['id'],
                             prompt_sha256=item['sha256'], expected_input_tokens=item['input_tokens']))
                    report['warmups'].append(result)
                    checkpoint_http(report, args)
                    if result['status'] != 'success':
                        report['errors'].append('Warmup failed; measured window not started')
                        warmup_failed = True
                        break
                if args.first_after_ready:
                    report.setdefault('phases', []).append(dict(phase='warmup', repetition=repetition,
                        started_at=warmup_started, ended_at=time.monotonic()))
                if warmup_failed:
                    break
                report['measurement_in_progress'] = True
                report['active_requests'] = []
                checkpoint_http(report, args)
                def progress(records):
                    report['active_requests'] = list(records)
                    checkpoint_http(report, args)
                run = measure(clients, args, corpus, repetition,
                              progress=progress if args.first_after_ready else None)
                report['runs'].append(run)
                report['measurement_in_progress'] = False
                report['active_requests'] = []
                checkpoint_http(report, args)
                print(json.dumps(dict(repetition=repetition, summary=run['summary']), allow_nan=False), flush=True)
                if run['stop_reason']:
                    report['errors'].append(run['stop_reason'])
                    break
            if len(report['runs']) == args.repeat and not report['errors']:
                report['status'] = 'complete'
        except (Exception, KeyboardInterrupt) as exc:
            report['errors'].append(f'{type(exc).__name__}: {exc}')
        finally:
            for client in clients:
                try:
                    client.close()
                except Exception as exc:
                    report['errors'].append(f'Client cleanup failed: {exc}')
                    report['status'] = 'incomplete'
            all_records = [request for run in report['runs'] for request in run['requests']]
            walls = [run['summary']['wall_seconds'] for run in report['runs']
                     if run['summary'].get('wall_seconds') is not None]
            report['aggregate'] = aggregate(all_records, sum(walls) if walls else None,
                                             args.requests*args.repeat, args.ttft_slo, args.e2e_slo)
            report['aggregate']['unstarted_windows'] = args.repeat-len(report['runs'])
            completed = (len(report['runs']) == args.repeat
                         and all(run['measurements']['work_completed'] for run in report['runs']))
            report['measurements'] = dict(
                window_kind=args.window_kind, denominator='sum of started-run wall_seconds; warmup/startup excluded',
                denominator_seconds=report['aggregate']['wall_seconds'], work_completed=completed,
                all_requests_successful=(completed and report['aggregate']['failed'] == 0),
                time_to_complete_seconds=(sum(run['measurements']['time_to_complete_seconds']
                                              for run in report['runs']) if completed else None),
                intentional_idle_seconds=sum(run['measurements']['intentional_idle_seconds']
                                             for run in report['runs']))
            report['aggregate'].update(window_kind=args.window_kind,
                                       denominator=report['measurements']['denominator'],
                                       denominator_seconds=report['aggregate']['wall_seconds'])
            if reference is not None:
                report['comparison'] = comparison_eligibility(report, reference)
            report['server_drain'] = 'unverified; client completion/timeout/close does not prove GPU quiescence'
            if report.get('measurement_in_progress'):
                checkpoint_http(report, args)
            else:
                report['aggregate']['unobserved_request_budget'] = 0
            report['finalized'] = True
            dump_json(args.output, report)
    return 0 if report['status'] == 'complete' and report['aggregate']['failed'] == 0 else 1


if __name__ == '__main__':
    if sys.argv[1:] == ['--_worker']:
        worker_main()
    elif sys.argv[1:] == ['--_matrix_worker']:
        raise SystemExit(matrix_worker(strict_json(sys.stdin.read())))
    else:
        raise SystemExit(main())
