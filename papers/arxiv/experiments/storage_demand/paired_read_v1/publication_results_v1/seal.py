#!/usr/bin/env python3
"""Assemble derived public receipts, then hash the new bundle; never reads originals."""
import itertools
import math
import statistics

from common import HERE, encode, seal, sha, write
from check import rebuild
from prespecified import METRICS


def main():
    result = rebuild(HERE)
    write(HERE, 'recomputed.json.gz', result)
    draws = []
    for indices in itertools.product(range(5), repeat=5):
        logs = {metric: statistics.median(result['analysis']['blocks'][i]['log_ratios'][metric] for i in indices)
                for metric in METRICS}
        draws.append(dict(blocks=['P' + str(i + 1) for i in indices], log_ratios=logs,
                          ratios={metric: math.exp(value) for metric, value in logs.items()}))
    write(HERE, 'bootstrap-enumeration.json.gz', dict(
        resampling_unit='paired block, not request', draws=draws, count=len(draws),
        log_ratio_rows_sha256=sha(encode([d['log_ratios'] for d in draws])),
        nearest_rank_endpoints=[79, 3047], interpretation='Descriptive n=5; no coverage guarantee'))
    lines = ['# Public results: completed paired HTTP read-schedule study', '',
             'Source generation: `mapfixed-20261015-01`; immutable study date: **2026-10-15**.', '',
             'Generated from all archived observations by `seal.py` / `check.py`; not hand-entered outcomes.', '',
             '## Prespecified paired results', '',
             'Ratios are Lookahead / Blocking. Each block uses its arm median of five measured QA',
             'requests for E2E and TTFT; goodput is 40 qualified generated tokens divided by',
             'the sum of those five client E2E durations. The aggregate is the exponentiated',
             'median of the five block log ratios.', '',
             '| Metric | Ratio | Relative change | Descriptive interval |',
             '|---|---:|---:|---|']
    names = dict(request_e2e_seconds='Client E2E', ttft_seconds='Client TTFT',
                 qualified_goodput_tokens_per_second='Qualified serial-window goodput')
    for metric, value in result['analysis']['aggregate'].items():
        lo, hi = value['interval_ratio']
        lines.append(f"| {names[metric]} | {value['ratio']:.10f} | {(value['ratio'] - 1) * 100:+.6f}% | [{lo:.10f}, {hi:.10f}] |")
    lines += ['', 'All **3,125** ordered five-block bootstrap resamples are archived in',
              '`bootstrap-enumeration.json.gz`. Endpoints use nearest ranks 79 and 3047.',
              'This is exploratory/descriptive n=5, **not a guaranteed-coverage CI**, significance',
              'test, power calculation, or general serving-performance claim.', '',
              '| Block | Order | Blocking E2E median (s) | Lookahead E2E median (s) | E2E ratio |',
              '|---|---|---:|---:|---:|']
    for b in result['analysis']['blocks']:
        lines.append(f"| {b['block']} | {' then '.join(b['execution_order'])} | {b['arms']['blocking']['request_e2e_seconds']:.9f} | {b['arms']['lookahead']['request_e2e_seconds']:.9f} | {math.exp(b['log_ratios']['request_e2e_seconds']):.10f} |")
    lines += ['', '## Denominators and evidence', '',
              '- 5 complete pairs, 10 completed lifecycles; 120/120 issued, terminal and successful requests.',
              '- 50/50 measured QA; 110 exact eight-token oracle matches and 10 two-token length-only smoke requests.',
              '- 900 actual worker-generated tokens joined to HTTP completion IDs, terminal usage and SSE/DONE.',
              '- 1,020 decoder invocation records; 150 NFS-client deltas (120 request + 30 phase).',
              '- 240 PID/birth-bound request I/O snapshots; 20 library snapshots with 12/12 required pins each.',
              '- No retries, replacements, pooled historical attempts, extra pilot lifecycles, or profiling timings.',
              f"- Recorded study elapsed time: {result['elapsed_study_seconds']:.9f} s.", '',
              '| Arm | Owner read bytes (logical) | Demand wait (s) | Speculative reads | Wasted read bytes | Peak logical host charge (bytes) |',
              '|---|---:|---:|---:|---:|---:|']
    for a in result['arms']:
        s = a['pipeline_stats']
        lines.append(f"| {a['trial']} | {s['proved_read_bytes']} | {s['demand_wait_ns'] / 1e9:.9f} | {s['speculative_reads']} | {s['wasted_read_bytes']} | {s['peak_host_charged_bytes']} |")
    lines += ['', 'Read totals, waits and charge are owner-local logical/host observations, not NAS backend traffic or GPU overlap.',
              'The final logical read charge is zero in every arm; owner-local resources are known closed',
              'and retained. That is **not** a post-drop physical-release certificate.', '',
              '## Separate historical failures', '',
              'Each earlier study (`artifacts-current-20261015-01` and `artifacts-v3-final-20261015-01`)',
              'had 1 attempted lifecycle, 0 completed lifecycles, 120 unissued requests, and 50 unissued',
              'measured-QA slots. Their five-pair aggregates remain null. Neither is pooled with this study.', '',
              '## Outstanding claim limits', '']
    lines += ['- ' + limitation for limitation in result['limitations'] + result['collection_deviations']]
    write(HERE, 'RESULTS.md', ('\n'.join(lines) + '\n').encode(), raw=True)
    seal(HERE)
    print('Derived results and all 3125 bootstrap draws sealed; run check.py and test_publication.py next')


if __name__ == '__main__':
    main()
