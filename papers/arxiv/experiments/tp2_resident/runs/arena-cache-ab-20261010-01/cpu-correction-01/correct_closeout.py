"""CPU-only correction of historical closeout bookkeeping; never a launcher."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINAL_SHA = '918ced0f852b36c01b6d3616b2b9fbe30e4421305df9d8cbb7352bf7f5d4d8bb'
REPORT_SHA = 'd571acaed8b9a003d7fbf40b6376d05ce7b5dccf96521a52db516ecdeb1e87b4'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def load(path):
    return json.loads(Path(path).read_text())


def link(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def gpu_sample_counts(rows):
    samples = present = missing = 0
    excluded = Counter()
    for row in rows:
        if row.get('kind') != 'sample':
            excluded[row.get('kind') or 'missing_kind'] += 1
            continue
        samples += 1
        if row.get('gpu') is None:
            missing += 1
        else:
            present += 1
    return dict(sample_rows=samples, raw_gpu_samples=present, missing_gpu_samples=missing,
                excluded_non_sample_rows=dict(excluded))


def run_window_annotation(report):
    require(len(report['runs']) == 1, 'Expected one measured run')
    run = report['runs'][0]
    measured = run['measurements']; summary = run['summary']
    seconds = measured['denominator_seconds']
    require(measured['window_kind'] == 'fixed-work' and seconds > 0, 'Wrong run window')
    require(math.isclose(seconds, measured['terminal_observed_at'] - measured['dispatch_start_at'], abs_tol=1e-9),
            'Recorded run window differs')
    require(summary['denominator_seconds'] == summary['wall_seconds'] == seconds
            and math.isclose(summary['successful_output_tokens_per_second'],
                             summary['successful_output_tokens'] / seconds, rel_tol=1e-12),
            'Historical throughput formula differs')
    last_client = max(r['client_observed_at'] for r in run['requests'])
    last_end = max(r['ended_at'] for r in run['requests'])
    require(measured['terminal_observed_at'] >= last_client >= last_end, 'Invalid terminal ordering')
    return dict(label='actual runner-observed fixed-work run window',
        recorded_run_window_seconds=seconds,
        formula='successful_output_tokens / recorded_run_window_seconds (unchanged)',
        dispatch_start_at=measured['dispatch_start_at'], runner_terminal_observed_at=measured['terminal_observed_at'],
        last_request_client_observed_at=last_client, last_request_ended_at=last_end,
        tail_after_last_client_observed_ms=1000 * (measured['terminal_observed_at'] - last_client),
        tail_after_last_request_ended_ms=1000 * (measured['terminal_observed_at'] - last_end),
        scope='Runner timestamp is taken after final progress/terminal checkpoint and executor exit; '
              'not just first-send to last-request EOF. Tail components are not separately timed. '
              'No adjusted throughput is substituted.')


def build_correction(original_path):
    original_path = Path(original_path).resolve()
    require(sha(original_path) == ORIGINAL_SHA, 'Historical owner closeout changed')
    old = load(original_path); result = deepcopy(old)
    require(sha(original_path.parent/'RESULTS.md') == REPORT_SHA, 'Historical report changed')
    correction = dict(cpu_only=True, gpu_queries_performed=False, current_source_or_quality_revalidated=False,
        historical_owner_closeout=link(original_path), historical_report=link(original_path.parent/'RESULTS.md'),
        count_policy='Only kind=sample enters sample/missing-GPU denominators; monitor_terminal is control metadata',
        historical_rates_and_metrics_preserved=True, samples=[], run_windows=[])
    for row in result['lifecycles']:
        directory = original_path.parent if row['kind'] == 'ab' else Path(old['smoke_receipt']['path']).parent
        samples = directory/'runs'/row['slot']/'session/samples.jsonl'
        require(sha(samples) == old['artifact_pins'][str(samples)], 'Historical raw samples changed')
        with samples.open() as stream:
            counts = gpu_sample_counts(json.loads(line) for line in stream)
        correction['samples'].append(dict(kind=row['kind'], slot=row['slot'], input=link(samples),
            previous_raw_gpu_samples=row['raw_gpu_samples'], previous_missing_gpu_samples=row['missing_gpu_samples'], **counts))
        row.update(raw_gpu_samples=counts['raw_gpu_samples'], missing_gpu_samples=counts['missing_gpu_samples'])
    summary_path = Path(old['ab_summary']['path'])
    require(sha(summary_path) == old['ab_summary']['sha256'], 'Historical A/B summary changed')
    summary = load(summary_path)
    for row in summary['rows']:
        path = original_path.parent/'runs'/row['slot']/'session/qa8-V1/http.json'
        require(sha(path) == summary['artifact_pins'][str(path)], 'Historical HTTP evidence changed')
        report = load(path)
        require(report['runs'][0]['summary'] == row['summary'], 'Historical HTTP/A-B summaries differ')
        correction['run_windows'].append(dict(slot=row['slot'], input=link(path), **run_window_annotation(report)))
    require(result['metrics'] == old['metrics'], 'Correction must not change historical rates')
    result['schema'] = 'ferrule.paper.arena-smoke-cache-ab-owner-closeout.v2'
    result['cpu_correction'] = correction
    result['artifact_pins'][str(original_path)] = ORIGINAL_SHA
    result['artifact_pins'][str(original_path.parent/'RESULTS.md')] = REPORT_SHA
    return result


def main():
    result = build_correction(HERE.parent/'owner-closeout.json')
    output = HERE/'owner-closeout-corrected.json'; report = HERE/'RESULTS-corrected.md'
    require(not output.exists() and not report.exists(), 'No overwrite of correction receipts')
    correction = result['cpu_correction']
    text = '# CPU closeout correction (historical GPU run, no new GPU work)\n\n'
    text += 'Supersedes only sample bookkeeping and run-window wording. Original owner-closeout and RESULTS remain byte-identical.\n'
    text += f"Original owner-closeout SHA256: `{ORIGINAL_SHA}`. Original RESULTS SHA256: `{REPORT_SHA}`.\n\n"
    text += '| Lifecycle | kind=sample | GPU present | GPU missing | excluded monitor_terminal |\n|---|---:|---:|---:|---:|\n'
    for row in correction['samples']:
        text += f"| {row['slot']} | {row['sample_rows']} | {row['raw_gpu_samples']} | {row['missing_gpu_samples']} | {row['excluded_non_sample_rows'].get('monitor_terminal', 0)} |\n"
    text += '\n## Throughput denominator clarification (rates unchanged)\n\n'
    text += 'The recorded `wall_seconds` is the **actual runner-observed fixed-work run window**: first measured HTTP send to the runner timestamp after terminal/progress checkpoint and executor exit. It is not simply the last request EOF/client terminal span. Startup and warmups are excluded.\n\n'
    for window, metric in zip(correction['run_windows'], result['metrics']):
        require(window['slot'] == metric['slot'], 'Metric order differs')
        text += f"- {window['slot']}: {window['recorded_run_window_seconds']:.12f} s; tail after last request client_observed_at = {window['tail_after_last_client_observed_ms']:.6f} ms (after ended_at = {window['tail_after_last_request_ended_ms']:.6f} ms). Original output rate remains {metric['output_tokens_per_second']:.12f} tok/s.\n"
    text += '\nTail components are not individually timed; no adjusted rate replaces the old result. All original performance metrics, source pins and limitations remain historical and unchanged.\n'
    text += '\nNo current production/source/quality validation or GPU/proc/lock query was performed. Final idle/release fields refer to the historical receipts, not a new observation. NVML host/local PID attribution remains unavailable; no proof of absence of foreign interference. Null activity does not prove zero decode; 54,621 evictions per ON owner remain whole-lifecycle, not measured/hot-phase counters. Same flags with different sources remains non-isolated and non-statistical.\n'
    with output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    with report.open('x') as stream:
        stream.write(text)
    print(output)
    print(report)


if __name__ == '__main__':
    main()
