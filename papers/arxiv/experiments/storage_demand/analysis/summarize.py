#!/usr/bin/env python3
"""Offline stdlib-only summarizer for the fixed storage-demand -01/-02 receipts."""
import argparse
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
BASE = 'papers/arxiv/experiments/storage_demand/runs/'
RUNS = ('storage-demand-tp1-middle-20261013-01', 'storage-demand-tp1-middle-20261013-02')
NAMES = ('hello2', 'canary-qa', 'canary-arithmetic', 'canary-chinese')
DELTAS = (
    'checkpoint_read_calls', 'checkpoint_read_bytes', 'checkpoint_source_checks',
    'checkpoint_path_metadata_calls', 'checkpoint_handle_metadata_calls',
    'h2d_copies', 'h2d_bytes', 'artifact_uploads', 'artifact_upload_bytes',
    'expert_hits', 'expert_uploads', 'expert_evictions',
)
GAUGES = (
    'expert_resident_entries', 'expert_resident_bytes', 'expert_pending_upload_bytes',
    'expert_scratch_bytes', 'expert_owner_peak_bytes', 'expert_unknown_quarantine_bytes',
)
COMMON = ('report.json', 'case.json', 'registration.json', 'supervisor.json',
          'custody.json', 'baseline.json', 'prelaunch.json', 'post-idle.json', 'server.log')
HOST_COMPONENTS = ('anonymous_bytes', 'pss_anon_bytes', 'pss_file_bytes', 'private_dirty_bytes')
PHASES = ('startup', 'ready_before_first', 'first_request', 'after_first_request')
CAVEATS = (
    'One successful instrumented run (n=1), not independent repetitions; no statistics or confidence intervals.',
    'ferrule_performance DEBUG overhead is unmeasured; these latencies are not a fast baseline or comparable vLLM throughput.',
    'Read calls/bytes are logical checkpoint-reader/syscall counters; physical NAS/NFS IO and OS page-cache state are unknown.',
    'H2D/artifact counters are facade activity, not physical bus traffic or device-kernel timing. Startup IO is not covered by invocation sums.',
    'Expert uploads are not cache misses. Misses and IO credits are unobserved, not inferred from hits/uploads.',
    'SSE text/usage matches are not generated-token-ID or logits evidence; frozen reference IDs are never observed generated IDs.',
    'TTFT is send-to-first-nonempty-SSE content; E2E is send-to-EOF; TPOT is the recorded content-event/usage proxy, not exact token time.',
    'Cache occupancy/scratch/owner-peak fields are gauges: last invocation and observed maxima, never additive deltas or a final ledger.',
    'Memory values are independent sampled component maxima, not an additive decomposition, exact peak, hard cap, or continuous coverage.',
    'Anonymous is resident anonymous memory, not PssAnon. PssAnon/PssFile and physical host cap are unobserved.',
    'GPU memory is selected-device aggregate telemetry; owned-process attribution and absence of foreign interference remain unproven.',
    'After-first-request samples include later canaries, not steady state. Shutdown resource_contract_complete is unknown despite exit/custody/idle evidence.',
)


class SchemaError(ValueError):
    pass


def paths():
    result = [BASE + run + '/' + name for run in RUNS for name in COMMON]
    prefix = BASE + RUNS[1] + '/'
    result += [prefix + name for name in ('ready.json', 'profile-server.json', 'samples.jsonl')]
    result += [prefix + name + '/' + leaf for name in NAMES
               for leaf in ('observed.json', 'invocations.json', 'http.json',
                            'generated-controls.json', 'prompts.json')]
    return sorted(result)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                       allow_nan=False) + '\n').encode('utf-8')


def load_sources():
    return {path: (ROOT / path).read_bytes() for path in paths()}


def integer(value):
    return type(value) is int and value >= 0


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def equal(label, actual, expected, issues):
    if actual != expected:
        issues.append(dict(check=label, actual=actual, expected=expected))


def close(label, actual, expected, issues):
    if not (number(actual) and number(expected)
            and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9)):
        equal(label, actual, expected, issues)


def request_quality(observed):
    """Reference presence, not a shared receipt PASS/label, gates reference checks."""
    reference = observed['reference']
    if reference is None:
        return dict(quality_scope='HTTP completion and usage checks only; no reference comparison',
                    reference_comparison_performed=False, reference_text_match=None,
                    reference_output_usage_match=None, reference_quality_passed=None)
    actual = observed['actual']
    text_match = actual['text'] == reference['reference_text']
    output_usage_match = actual['usage']['completion_tokens'] == len(reference['reference_ids'])
    return dict(
        quality_scope='Visible text and eight-output-token usage checks against frozen HF generation8 only; no exact generated IDs or numeric acceptance',
        reference_comparison_performed=True, reference_text_match=text_match,
        reference_output_usage_match=output_usage_match,
        reference_quality_passed=actual['status'] == 'success' and observed['status'] == 'pass'
        and text_match and output_usage_match)


def aggregate(rows):
    """Counter flags gate all deltas; absent/malformed fields never become zero."""
    fields = [row['fields'] for row in rows]
    flags_ok = bool(fields) and all(
        row.get('counters_available') is True and row.get('counter_reset_detected') is False
        for row in fields)
    deltas = {key: sum(row[key] for row in fields)
              if flags_ok and all(integer(row.get(key)) for row in fields) else None
              for key in DELTAS}
    last = {key: fields[-1].get(key) if fields and integer(fields[-1].get(key)) else None
            for key in GAUGES}
    maxima = {key: max(row[key] for row in fields)
              if fields and all(integer(row.get(key)) for row in fields) else None
              for key in GAUGES}
    return dict(invocations=len(fields), valid=flags_ok and all(v is not None for v in deltas.values()),
                deltas=deltas, last_gauges=last, observed_gauge_maxima=maxima)


def compare_counters(label, computed, recorded, issues):
    if not isinstance(recorded, dict):
        raise SchemaError(label + ': missing counter receipt')
    for key in ('invocations', 'valid', 'deltas', 'last_gauges'):
        equal(label + '/' + key, computed[key], recorded.get(key), issues)
    for key in ('expert_cache_miss_counter', 'io_credits', 'physical_nas_bytes',
                'proc_read_bytes', 'nfs_shared_mount_bytes'):
        equal(label + '/unobserved_' + key, recorded.get(key), None, issues)


def invocations(data, start=0, end=None):
    end = len(data) if end is None else end
    if not (integer(start) and integer(end) and 0 <= start <= end <= len(data)):
        raise SchemaError('invalid server-log byte anchor')
    if (start and data[start - 1:start] != b'\n') or (end and data[end - 1:end] != b'\n'):
        raise SchemaError('server-log byte anchor splits a line')
    result = []
    for line, raw in enumerate(data[start:end].splitlines(), data[:start].count(b'\n') + 1):
        try:
            event = json.loads(raw)
        except (ValueError, UnicodeError):
            if b'mixed_decoder_invocation' in raw:
                raise SchemaError('malformed invocation JSON at server-log line ' + str(line))
            continue  # Fixed logs also contain non-JSON model-plan/listening lines.
        fields = event.get('fields', {}) if isinstance(event, dict) else {}
        if isinstance(fields, dict) and fields.get('event') == 'mixed_decoder_invocation':
            result.append(dict(invocation_id='server-log-line-' + str(line), line=line,
                               raw=event, fields=fields))
    return result


def memory(samples, uuid):
    hosts = [row['host'] for row in samples if (row.get('host') or {}).get('status') == 'complete']
    gpus = [row['gpu'] for row in samples if row.get('gpu') is not None]
    def peak(key):
        return max((host[key] for host in hosts if integer(host.get(key))), default=None)
    return dict(
        sample_count=len(samples), complete_host_samples=len(hosts), gpu_samples=len(gpus),
        gpu_peak_bytes={uuid: max((device['gpu_bytes'] for gpu in gpus for device in gpu['devices']
                                  if device['uuid'] == uuid and integer(device.get('gpu_bytes'))), default=None)},
        tree_pss_peak_bytes=peak('pss_bytes'), tree_rss_peak_bytes=peak('rss_bytes'),
        **{'tree_' + key.removesuffix('_bytes') + '_peak_bytes': peak(key) for key in HOST_COMPONENTS},
        component_sample_counts={key: sum(integer(host.get(key)) for host in hosts)
                                 for key in HOST_COMPONENTS})


def phase_memory(samples, uuid, ready, sent, ended):
    phases = {name: [] for name in PHASES}
    unassigned = 0
    for row in samples:
        start, end = row['monotonic'], row.get('sample_end_monotonic')
        if not number(start) or not number(end) or end < start:
            unassigned += 1
        elif end < ready:
            phases['startup'].append(row)
        elif start >= ready and end < sent:
            phases['ready_before_first'].append(row)
        elif start >= sent and end < ended:
            phases['first_request'].append(row)
        elif start >= ended:
            phases['after_first_request'].append(row)
        else:
            unassigned += 1
    return {name: memory(rows, uuid) for name, rows in phases.items()}, unassigned


def check_memory(label, computed, recorded, issues):
    if not isinstance(recorded, dict):
        raise SchemaError(label + ': missing sampled-memory receipt')
    for key, value in computed.items():
        equal(label + '/' + key, value, recorded.get(key), issues)


def analyze(sources, bindings):
    if set(sources) != set(paths()) or set(bindings) != set(paths()):
        raise SchemaError('input set must be exactly the fixed -01/-02 source paths')
    issues, manifest, attempts = [], [], []
    for path in paths():
        actual_hash = digest(sources[path])
        manifest.append(dict(path=path, sha256=actual_hash, bound_sha256=bindings[path]))
        equal('input_sha256/' + path, actual_hash, bindings[path], issues)
    def obj(path):
        return json.loads(sources[path])
    pin_count = 0
    for run in RUNS:
        prefix = BASE + run + '/'
        report, case, reg, supervisor, custody, baseline, pre, post = (
            obj(prefix + name) for name in COMMON[:-1])
        equal(run + '/case_report', case, report['case'], issues)
        equal(run + '/schedule', [row['name'] for row in reg['schedule']], list(NAMES), issues)
        equal(run + '/planned', report['request_count'], len(reg['schedule']), issues)
        equal(run + '/n', report['n'], 1, issues)
        equal(run + '/registered_n', reg['n'], 1, issues)
        equal(run + '/custody', supervisor['custody'], custody, issues)
        equal(run + '/resource_contract', report['resource_contract_complete'],
              case['shutdown_contract']['resource_contract_complete'], issues)
        pins = {BASE + key.split('/storage_demand/runs/', 1)[1]: value
                for key, value in report['artifact_pins'].items() if '/storage_demand/runs/' in key}
        for path, data in sources.items():
            if path.startswith(prefix) and path != prefix + 'report.json':
                equal('receipt_pin/' + path, digest(data), pins.get(path), issues)
                pin_count += 1
        log = sources[prefix + 'server.log']
        equal(run + '/log_sha256', digest(log), case['log_sha256'], issues)
        all_rows = invocations(log)
        totals = aggregate(all_rows)
        compare_counters(run + '/all_invocations', totals, case['all_invocation_counters'], issues)
        for row in all_rows:
            for key, expected in (('device', 0), ('status', 'completed'), ('quarantined', False),
                                  ('counters_available', True), ('counter_reset_detected', False)):
                equal(run + '/log_line_' + str(row['line']) + '/' + key,
                      row['fields'].get(key), expected, issues)
        if all_rows:
            equal(run + '/complete_delta_coverage', totals['valid'], True, issues)
            equal(run + '/complete_gauge_coverage',
                  all(v is not None for v in totals['observed_gauge_maxima'].values()), True, issues)
        requests, mapped_lines, request_deltas = [], [], []
        http_windows, sent_times, terminal_times, e2e_times = [], [], [], []
        for index, observed in enumerate(case['requests']):
            scheduled = reg['schedule'][index]
            name = observed['name']
            if name not in NAMES:
                raise SchemaError('unexpected request ' + name)
            request_prefix = prefix + name + '/'
            equal(name + '/case_observed', observed, obj(request_prefix + 'observed.json'), issues)
            anchor = observed['anchor']
            raw_rows = invocations(log, anchor['log_start_byte'], anchor['log_end_byte'])
            equal(name + '/raw_log_invocations', raw_rows, obj(request_prefix + 'invocations.json'), issues)
            computed = aggregate(raw_rows)
            compare_counters(name, computed, observed['counters'], issues)
            if raw_rows:
                equal(name + '/complete_delta_coverage', computed['valid'], True, issues)
            mapped_lines += [row['line'] for row in raw_rows]
            request_deltas.append(computed['deltas'])
            http = obj(request_prefix + 'http.json')
            actuals = [row for window in http['runs'] for row in window['requests']]
            if len(actuals) != 1:
                raise SchemaError(name + ': expected exactly one HTTP request')
            actual = actuals[0]
            equal(name + '/http_status', actual['status'], 'success', issues)
            equal(name + '/receipt_status', observed['status'], 'pass', issues)
            equal(name + '/http_observed', actual, observed['actual'], issues)
            equal(name + '/controls', obj(request_prefix + 'generated-controls.json'), scheduled['payload'], issues)
            equal(name + '/observed_controls', observed['generated_controls'], scheduled['payload'], issues)
            equal(name + '/sent_controls', actual['request_payload'], scheduled['payload'], issues)
            equal(name + '/prompts', obj(request_prefix + 'prompts.json'),
                  [dict(id=name, prompt=scheduled['prompt'], input_tokens=scheduled['input_tokens'])], issues)
            equal(name + '/reference', observed['reference'], scheduled['reference'], issues)
            equal(name + '/input_usage', actual['usage']['prompt_tokens'], actual['actual_input_tokens'], issues)
            equal(name + '/output_usage', actual['usage']['completion_tokens'], actual['actual_output_tokens'], issues)
            equal(name + '/total_usage', actual['usage']['total_tokens'],
                  actual['actual_input_tokens'] + actual['actual_output_tokens'], issues)
            content, usage = [], None
            for event in actual['raw_sse_events']:
                if event['data'] == '[DONE]':
                    continue
                payload = json.loads(event['data'])
                for choice in payload.get('choices', []):
                    if choice.get('text'):
                        content.append((event['at'], choice['text']))
                if 'usage' in payload:
                    usage = payload['usage']
            equal(name + '/sse_usage', usage, actual['usage'], issues)
            equal(name + '/sse_text', ''.join(text for _, text in content), actual['text'], issues)
            equal(name + '/sse_times', [at for at, _ in content], actual['content_event_times'], issues)
            close(name + '/ttft', actual['ttft_seconds'], content[0][0] - actual['sent_at'], issues)
            close(name + '/e2e', actual['request_e2e_seconds'], actual['ended_at'] - actual['sent_at'], issues)
            close(name + '/tpot_proxy', actual['mean_tpot_usage_proxy_seconds'],
                  (content[-1][0] - content[0][0]) / (actual['actual_output_tokens'] - 1), issues)
            equal(name + '/input_length', actual['actual_input_tokens'], scheduled['input_tokens'], issues)
            equal(name + '/output_length', actual['actual_output_tokens'], scheduled['max_tokens'], issues)
            equal(name + '/generated_ids_absent', actual['output_token_ids'], None, issues)
            equal(name + '/observed_ids_absent', observed['observed_token_ids'], None, issues)
            for key in ('scheduled', 'offered', 'success'):
                equal(name + '/http_denominator_' + key, http['aggregate'][key], 1, issues)
            equal(name + '/http_output_tokens', http['aggregate']['successful_output_tokens'],
                  actual['actual_output_tokens'], issues)
            http_windows.append(http['aggregate']['denominator_seconds'])
            sent_times.append(actual['sent_at'])
            terminal_times.append(actual['client_observed_at'])
            e2e_times.append(actual['request_e2e_seconds'])
            requests.append(dict(
                name=name, status=actual['status'], visible_text_usage_receipt_status=observed['status'],
                input_tokens=actual['actual_input_tokens'], output_tokens=actual['actual_output_tokens'],
                ttft_seconds=actual['ttft_seconds'], request_e2e_seconds=actual['request_e2e_seconds'],
                mean_tpot_usage_proxy_seconds=actual['mean_tpot_usage_proxy_seconds'],
                observed_generated_token_ids=actual['output_token_ids'], exact_token_pass=observed['exact_token_pass'],
                reference_ids_are_observed_generated_ids=False,
                **request_quality(observed),
                log_byte_interval=[anchor['log_start_byte'], anchor['log_end_byte']],
                source_log_lines=[row['line'] for row in raw_rows],
                invocations=computed['invocations'], additive_deltas=computed['deltas'],
                last_invocation_gauges=computed['last_gauges'],
                observed_gauge_maxima=computed['observed_gauge_maxima']))
        equal(run + '/request_mapping', sorted(mapped_lines), [row['line'] for row in all_rows], issues)
        summed_requests = {key: sum(row[key] for row in request_deltas)
                           if request_deltas and all(integer(row.get(key)) for row in request_deltas) else None
                           for key in DELTAS}
        equal(run + '/request_sums_vs_log', summed_requests, totals['deltas'], issues)
        dispatched = sum(number(row['actual'].get('sent_at')) for row in case['requests'])
        success = sum(row['status'] == 'success' for row in requests)
        uuid = next(key for key, value in baseline.items() if value['index'] == 0)
        sampled, phases, ready_gpu = memory([], uuid), None, None
        startup = None
        if run == RUNS[1]:
            equal(run + '/successful_schedule', [row['name'] for row in requests], list(NAMES), issues)
            equal(run + '/completion_status', report['status'], 'complete', issues)
            ready = obj(prefix + 'ready.json')
            target = obj(prefix + 'profile-server.json')
            equal(run + '/target', case['target'], target, issues)
            equal(run + '/ready', ready['monotonic'], case['ready_monotonic'], issues)
            equal(run + '/argv', ready['declared_argv'], reg['spec']['argv'], issues)
            startup = case['ready_monotonic'] - case['launch_monotonic']
            close(run + '/startup', case['startup_seconds'], startup, issues)
            ready_gpu = ready['observed']['gpu_bytes']
            sample_events = [json.loads(line) for line in sources[prefix + 'samples.jsonl'].splitlines()]
            samples = [row for row in sample_events if row.get('kind') == 'sample']
            sampled = memory(samples, uuid)
            check_memory(run + '/sampled_memory', sampled, case['resources']['memory_components'], issues)
            phases, unassigned = phase_memory(samples, uuid, ready['monotonic'],
                                              case['requests'][0]['actual']['sent_at'],
                                              case['requests'][0]['actual']['ended_at'])
            recorded_phases = case['resources']['phase_memory']
            for phase in PHASES:
                check_memory(run + '/phase_' + phase, phases[phase], recorded_phases['phases'][phase], issues)
            equal(run + '/unassigned_samples', unassigned,
                  recorded_phases['unassigned_or_boundary_samples'], issues)
        else:
            equal(run + '/failed_status', report['status'], 'failed', issues)
            equal(run + '/failed_requests', case['requests'], [], issues)
            equal(run + '/failed_log_empty', len(log), 0, issues)
            equal(run + '/failed_no_server_pid', case.get('server_pid'), None, issues)
            equal(run + '/failed_no_launch', case.get('launch_monotonic'), None, issues)
        argv = reg['spec']['argv']
        declared = {flag.removeprefix('--').replace('-', '_'): argv[argv.index(flag) + 1]
                    for flag in ('--backend', '--engine', '--model-profile', '--expert-prewarm',
                                 '--ctx-size', '--cuda-device-budget-mb', '--cuda-workspace-mb',
                                 '--cuda-expert-device-entries', '--cuda-expert-device-bytes',
                                 '--cuda-expert-scratch-mb')}
        attempts.append(dict(
            run=run, source_directory=prefix.rstrip('/'), status=report['status'],
            failure_stage='before_server_launch' if run == RUNS[0] else None,
            failure_error=case.get('target_error'), n=report['n'], declared_config=declared,
            quality=dict(exact_token_pass=report['exact_token_pass'],
                         full_current_math_pass=report['full_current_math_pass'],
                         token_ids_exposed=case['token_ids_exposed'],
                         reference_comparison_requests=sum(row['reference_comparison_performed'] for row in requests),
                         reference_quality_passed_requests=sum(row['reference_quality_passed'] is True for row in requests),
                         http_only_success_requests=sum(row['status'] == 'success' and not row['reference_comparison_performed']
                                                        for row in requests)),
            debug_logging=reg['spec']['environment'].get('FERRULE_LOG'), observer_effect=reg['observer_effect'],
            denominator=dict(planned=len(reg['schedule']), dispatched=dispatched,
                             not_run=len(reg['schedule']) - dispatched, success=success,
                             failed_dispatched=dispatched - success,
                             successful_output_tokens=sum(row['output_tokens'] for row in requests if row['status'] == 'success'),
                             invocations=totals['invocations']),
            timing=dict(startup_seconds=startup, request_e2e_sum_seconds=sum(e2e_times) if e2e_times else None,
                        summed_http_fixed_work_windows_seconds=sum(http_windows) if http_windows else None,
                        first_send_to_last_client_terminal_seconds=max(terminal_times) - min(sent_times) if sent_times else None,
                        supervisor_wall_seconds=supervisor['wall_seconds'], case_wall_seconds=case['wall_seconds']),
            requests=requests, additive_deltas=totals['deltas'], per_request_delta_sum=summed_requests,
            last_invocation_gauges=totals['last_gauges'], observed_gauge_maxima=totals['observed_gauge_maxima'],
            sampled_memory=sampled, phase_sampled_memory=phases, ready_device_bytes=ready_gpu,
            unobserved=dict(expert_cache_misses=None, io_credits=None,
                            physical_nas_bytes=report['physical_nas_bytes'], proc_read_bytes=None,
                            nfs_shared_mount_bytes=None, page_cache_state=None,
                            PssAnon=report['PssAnon'], physical_host_cap_bytes=report['physical_host_cap_bytes'],
                            observed_generated_token_ids=None, startup_io_counters=None,
                            owned_process_gpu_peak_bytes=None),
            shutdown=dict(resource_contract_complete=report['resource_contract_complete'],
                          evidence=case['shutdown_contract'], supervisor_returncode=supervisor['returncode'],
                          custody_status=custody['status'], prelaunch_idle_status=pre['status'],
                          post_idle_status=post['status'], authenticated_drained=report['authenticated_drained'],
                          all_eight_post_idle=report['all_eight_post_idle'])))
    return dict(schema='storage_demand_fixed_receipts_analysis.v1', inputs=manifest,
                counter_semantics=dict(additive=list(DELTAS), gauges=list(GAUGES),
                                       gauge_policy='last invocation and observed maxima; never sum',
                                       mapping='serialized request/log-byte interval; transaction is not request ID'),
                attempts=attempts, caveats=list(CAVEATS),
                consistency=dict(status='pass' if not issues else 'fail',
                                 input_hashes_checked=len(manifest), receipt_pins_checked=pin_count,
                                 checks=['bound input hashes', 'receipt artifact/log hashes', 'case/report/observed/HTTP equality',
                                         'invocations.json vs raw anchored log', 'invocation vs request vs run delta sums',
                                         'last gauges (not sums)', 'SSE text/usage and latency', 'raw sampled memory and phases'],
                                 divergences=issues))


def fmt(value, places=None):
    if value is None:
        return '—'
    return f'{value:.{places}f}' if places is not None else f'{value:,}'


def tables(summary):
    lines = ['# Storage-demand TP1 middle: offline receipt tables', '',
             'Single instrumented successful run, n=1; descriptive observations only. No throughput comparison, statistics, or confidence intervals.', '',
             '## Attempt accounting', '',
             '| Attempt | Status | Planned | Dispatched | Success | Not run | Usage output tokens | Invocations |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for attempt in summary['attempts']:
        d = attempt['denominator']
        lines.append('| ' + ' | '.join([attempt['run'][-3:], attempt['status']] +
                     [fmt(d[key]) for key in ('planned', 'dispatched', 'success', 'not_run', 'successful_output_tokens', 'invocations')]) + ' |')
    lines += ['', '-01 failed before server launch; its unobserved counters remain null, not zero.', '',
              '## Per-request latency (-02)', '',
              '| Request | Input / output usage tokens | Invocations | TTFT (s) | E2E (s) | Content-event / usage TPOT proxy (s) |',
              '|---|---:|---:|---:|---:|---:|']
    successful = summary['attempts'][1]
    for row in successful['requests']:
        lines.append(f"| {row['name']} | {row['input_tokens']} / {row['output_tokens']} | {row['invocations']} | " +
                     ' | '.join(fmt(row[key], 3) for key in ('ttft_seconds', 'request_e2e_seconds', 'mean_tpot_usage_proxy_seconds')) + ' |')
    lines += ['', f"Launch-to-ready: {fmt(successful['timing']['startup_seconds'], 3)} s. "
              'Latency follows the HTTP receipt, not exact token or device-kernel time. No generated IDs were observed.', '',
              'Hello2: HTTP completion and usage checks only; no reference comparison. '
              f"Only {successful['quality']['reference_comparison_requests']} canaries have visible-text/eight-output-usage reference checks; "
              'HTTP success is not a frozen-reference PASS. No exact generated IDs or numeric acceptance.', '',
              '## Logical additive invocation counters (-02)', '',
              '| Request | Read calls | Reader bytes | H2D copies | H2D bytes | Artifact uploads | Expert hits / uploads / evictions |',
              '|---|---:|---:|---:|---:|---:|---:|']
    counter_rows = [(row['name'], row['additive_deltas']) for row in successful['requests']]
    counter_rows.append((f"Total ({successful['denominator']['invocations']} invocations)", successful['additive_deltas']))
    for name, d in counter_rows:
        lines.append('| ' + name + ' | ' + ' | '.join(fmt(d[key]) for key in
                     ('checkpoint_read_calls', 'checkpoint_read_bytes', 'h2d_copies', 'h2d_bytes', 'artifact_uploads')) +
                     ' | ' + ' / '.join(fmt(d[key]) for key in ('expert_hits', 'expert_uploads', 'expert_evictions')) + ' |')
    lines += ['', 'Reader bytes are logical checkpoint demand, not physical NAS/NFS IO. Facade H2D is not physical bus traffic. Uploads are not cache misses.', '',
              '## Cache gauges (-02; never summed)', '',
              '| Gauge | Last observed invocation | Observed maximum |', '|---|---:|---:|']
    for key in GAUGES:
        lines.append(f"| `{key}` | {fmt(successful['last_invocation_gauges'][key])} | {fmt(successful['observed_gauge_maxima'][key])} |")
    lines += ['', 'These are invocation observations, not a final shutdown ledger. Byte gauges are bytes; resident entries are a count.', '',
              '## Sampled memory (-02; bytes)', '',
              '| Phase | Samples | Selected-device peak | Tree PSS peak | Tree RSS peak | Tree Anonymous peak |',
              '|---|---:|---:|---:|---:|---:|']
    for name, m in [('All samples', successful['sampled_memory'])] + list(successful['phase_sampled_memory'].items()):
        lines.append('| ' + name + ' | ' + str(m['sample_count']) + ' | ' + fmt(next(iter(m['gpu_peak_bytes'].values()))) +
                     ' | ' + ' | '.join(fmt(m[key]) for key in ('tree_pss_peak_bytes', 'tree_rss_peak_bytes', 'tree_anonymous_peak_bytes')) + ' |')
    lines += ['', 'Independent sampled maxima, not additive components or exact peaks. Ready-device snapshot: ' +
              fmt(successful['ready_device_bytes']) + ' bytes. PssAnon/PssFile and physical host cap are unknown. After-first-request is not steady state.', '',
              'resource_contract_complete = unknown for both attempts. Full caveats/provenance: `REPORT.md` and `summary.json`.', '',
              '**Source consistency: ' + summary['consistency']['status'].upper() + '**.', '']
    return '\n'.join(lines).encode('utf-8')


def report_text(summary):
    check = summary['consistency']
    successful, failed = summary['attempts'][1], summary['attempts'][0]
    d = successful['additive_deltas']
    def accounting(attempt):
        values = attempt['denominator']
        return ', '.join(label + ' ' + str(values[key]) for key, label in
                         (('planned', 'planned'), ('dispatched', 'dispatched'),
                          ('success', 'successful'), ('not_run', 'not run'),
                          ('successful_output_tokens', 'usage output tokens'),
                          ('invocations', 'invocations')))
    lines = ['# Storage-demand offline receipt analysis', '',
             'Scope: the fixed `storage-demand-tp1-middle-20261013-01` and `-02` receipts only. '
             'No GPU queries, launches, production imports, or raw-receipt edits.', '',
             '## Result', '',
             f"- -01: {accounting(failed)}; failed before server launch: `{failed['failure_error']}`.",
             f'- -02: {accounting(successful)}.',
             f"- Source consistency: **{check['status'].upper()}**; {check['input_hashes_checked']} bound input hashes and "
             f"{check['receipt_pins_checked']} consumed receipt pins checked; {len(check['divergences'])} divergences.",
             '- Raw-log invocation deltas match per-request and whole-run receipts when consistency is PASS. '
             'Gauge checks use last values, never sums. Sampled memory and phase maxima are recomputed from raw samples.',
             '- -02 totals: ' + ', '.join(key + '=' + str(d[key]) for key in
                 ('checkpoint_read_calls', 'checkpoint_read_bytes', 'h2d_copies', 'h2d_bytes',
                  'expert_uploads', 'expert_evictions', 'artifact_uploads', 'expert_hits')) + '.', '',
             '## Quality scope correction', '',
             '- Hello2 derived quality_scope: **HTTP completion and usage checks only; no reference comparison**. '
             'Its reference is null and reference_quality_passed is null (not a frozen-reference PASS).',
             '- The frozen Hello2 receipt inherited a shared quality_scope label mentioning frozen HF generation8. '
             'The generator normalizes derived metadata from reference presence, not that label; all frozen receipts remain unchanged.',
             f"- Only {successful['quality']['reference_comparison_requests']} canaries (canary-qa, canary-arithmetic, canary-chinese) "
             f"have visible-text/eight-output-usage reference checks; {successful['quality']['reference_quality_passed_requests']} pass those limited checks. "
             f"Hello2 accounts for {successful['quality']['http_only_success_requests']} HTTP-only successful request. "
             'No exact generated IDs were observed, and none of these checks establishes numeric acceptance.', '',
             '## Caveats', ''] + ['- ' + caveat for caveat in summary['caveats']]
    lines += ['', '## Reproducibility', '',
              '`TABLES.md` provides descriptive manuscript-ready tables; exact values are in `summary.json`.',
              'The JSON binds relative input paths to SHA-256 (actual and sealed values). `../inputs.json` seals '
              'all consumed sources, including report receipts that cannot self-pin. Existing report artifact pins '
              'are independently checked for consumed leaves. Absolute receipt pin names are normalized only at '
              'the fixed storage-demand run path; no binary, checkpoint payload, or production-source rehash is implied.',
              'The generator and input-manifest hashes are also recorded. No new timestamp, mtime, host metadata, '
              'or hardware observation enters the output. `--check` recomputes all artifacts and rejects byte-level '
              'summary/table/report divergence without overwriting them.', '',
              'From the repository root:', '',
              '```sh',
              '/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B -m unittest discover -s papers/arxiv/experiments/storage_demand/analysis/tests -v',
              '/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/analysis/summarize.py',
              '/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/analysis/summarize.py --check',
              '```', '']
    return '\n'.join(lines).encode('utf-8')


def render(summary):
    return {'summary.json': encode(summary), 'TABLES.md': tables(summary), 'REPORT.md': report_text(summary)}


def check_outputs(directory, expected):
    return [name for name, data in expected.items()
            if not (directory / name).is_file() or (directory / name).read_bytes() != data]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check existing output bytes without writing')
    args = parser.parse_args()
    try:
        bindings_data = (HERE / 'inputs.json').read_bytes()
        bindings = json.loads(bindings_data)
        if bindings['algorithm'] != 'sha256':
            raise SchemaError('input manifest must use sha256')
        summary = analyze(load_sources(), bindings['files'])
        summary['provenance'] = dict(
            generator_path='papers/arxiv/experiments/storage_demand/analysis/summarize.py',
            generator_sha256=digest(Path(__file__).read_bytes()),
            input_manifest_path='papers/arxiv/experiments/storage_demand/analysis/inputs.json',
            input_manifest_sha256=digest(bindings_data),
            time_policy='no analysis timestamp; timings come only from bound receipts')
        if summary['consistency']['status'] != 'pass':
            print(encode(summary['consistency']).decode('utf-8'), end='')
            return 1
        output = render(summary)
        directory = HERE / 'output'
        if args.check:
            stale = check_outputs(directory, output)
            if stale:
                print('STALE OUTPUT: ' + ', '.join(stale))
                return 1
        if not args.check:
            if not directory.is_dir() or directory.resolve().parent != HERE:
                raise SchemaError('output directory must be analysis/output')
            for name, data in output.items():
                (directory / name).write_bytes(data)
        print('PASS: ' + str(summary['consistency']['input_hashes_checked']) + ' input hashes; '
              + str(summary['consistency']['receipt_pins_checked']) + ' receipt pins; 0 divergences')
        print(('Checked' if args.check else 'Generated') + ': ' + str(directory.relative_to(ROOT)))
        return 0
    except (OSError, ValueError, KeyError, TypeError, IndexError, ZeroDivisionError) as error:
        print('ERROR: ' + type(error).__name__ + ': ' + str(error))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
