#!/usr/bin/env python3
"""Stdlib-only, read-only derivation of the sealed mapfixed paired study.

No harness imports, GPU queries, network access, private raw copies, or input resealing.
"""
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
BASE = 'papers/arxiv/experiments/storage_demand/paired_read_v1/'
RAW = BASE + 'artifacts-v3-mapfixed-20261015-01/'
PLAN = BASE + 'plan-mapfixed-20261015-01/'
ANALYSIS = BASE + 'analysis_v1/'
FIG = 'papers/arxiv/figures/paired-read-'
MIB, GIB = 2**20, 2**30
MODES = ('blocking', 'lookahead')
ORDER = [dict(block=f'P{i}', modes=list(modes)) for i, modes in enumerate(
    (MODES, MODES, MODES[::-1], MODES[::-1], MODES), 1)]
TRIALS = [r['block'] + '-' + mode for r in ORDER for mode in r['modes']]
NAMES = ['hello2', 'canary-qa', 'canary-arithmetic', 'canary-chinese',
         'qa-first-excluded', 'qa-warm-1', 'qa-warm-2'] + [f'qa-measured-{i}' for i in range(1, 6)]
METRICS = ('request_e2e_seconds', 'ttft_seconds', 'qualified_goodput_tokens_per_second')
COUNTERS = ('checkpoint_read_calls', 'checkpoint_read_bytes', 'checkpoint_source_checks',
            'checkpoint_path_metadata_calls', 'checkpoint_handle_metadata_calls', 'h2d_copies',
            'h2d_bytes', 'artifact_uploads', 'artifact_upload_bytes', 'expert_hits',
            'expert_uploads', 'expert_evictions')
PHASE_SNAPSHOTS = dict(startup_nfs=('prelaunch', 'ready'),
                       requests_nfs=('requests-pre', 'requests-post'),
                       shutdown_nfs=('shutdown-pre', 'shutdown-post'))


class EvidenceError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def digest(content):
    return hashlib.sha256(content).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + '\n').encode()


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def integer(value):
    return type(value) is int and value >= 0


def units(value, divisor):
    require(number(value) and divisor in (1, 2**10, MIB, GIB, 10**9), 'invalid or unknown quantity/unit')
    return value / divisor


def gauge(values):
    """A missing observation makes a maximum unknown; gauges are never accumulated."""
    return dict(last=values[-1] if values and integer(values[-1]) else None,
                maximum=max(values) if values and all(integer(v) for v in values) else None)


def paired_estimate(log_ratios):
    require(len(log_ratios) == 5 and all(type(x) in (int, float) and math.isfinite(x) for x in log_ratios),
            'exactly five finite paired-block log ratios required')
    draws = sorted(statistics.median(draw) for draw in itertools.product(log_ratios, repeat=5))
    ranks = [math.ceil(.025 * len(draws)), math.ceil(.975 * len(draws))]
    return dict(ratio=math.exp(statistics.median(log_ratios)),
                descriptive_range=[math.exp(draws[r - 1]) for r in ranks],
                draws=len(draws), nearest_ranks_1based=ranks, n=5,
                estimator='exp(median of five log(lookahead/blocking) ratios)',
                resampling_unit='paired block, not individual request',
                interpretation='2.5/97.5 percentile descriptive range; no coverage or significance claim')


def paired_summary(arms):
    """Keep every planned pair; a failed/missing arm suppresses every aggregate."""
    require(set(arms) == set(TRIALS), 'all ten planned arms, including null failed arms, required')
    blocks = []
    for row in ORDER:
        values = {m: arms[row['block'] + '-' + m] for m in MODES}
        qualified = all(values.values())
        ratios = {k: math.log(values['lookahead'][k] / values['blocking'][k]) for k in METRICS} if qualified else None
        blocks.append(dict(block=row['block'], execution_order=row['modes'], arms=values,
                           qualified=qualified, failure_pair_retained=not qualified, log_ratios=ratios))
    aggregate = {k: paired_estimate([r['log_ratios'][k] for r in blocks]) for k in METRICS} if all(
        r['qualified'] for r in blocks) else None
    return dict(blocks=blocks, aggregate=aggregate, scheduled_blocks=5,
                qualified_blocks=sum(r['qualified'] for r in blocks),
                failed_or_incomplete_blocks=sum(not r['qualified'] for r in blocks))


def input_paths():
    paths = [RAW + 'paired-final.json']
    paths += [PLAN + n for n in ('poststudy-audit.json', 'preregistration.json',
                                'actual-binding-v3.json', 'actual-build-review-v3.json')]
    paths += [BASE + n for n in ('protocol-v3.json', 'cases-v3.json', 'order.json', 'oracles.json', 'requests.json')]
    for trial in TRIALS:
        prefix = RAW + trial + '/'
        paths += [prefix + n for n in ('requests-final.json', 'lifecycle.json', 'read-ledger-assessment.json',
                                       'server.log', 'samples.jsonl')]
        paths += [prefix + name + '-mountstats.json' for pair in PHASE_SNAPSHOTS.values() for name in pair]
        for name in NAMES:
            paths += [prefix + name + '/' + n for n in ('issued.json', 'observed.json', 'http.json', 'http-terminal.json')]
            paths += [prefix + name + '-' + side + '-mountstats.json' for side in ('pre', 'post')]
    return sorted(paths)


def load_sources(root=ROOT):
    manifest_bytes = (root / ANALYSIS / 'inputs.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    require(manifest['algorithm'] == 'sha256' and set(manifest['files']) == set(input_paths()),
            'input set differs from this study; no old-source count or historical pooling')
    sources = {}
    for path in input_paths():
        content = (root / path).read_bytes()
        require(digest(content) == manifest['files'][path], 'input hash mismatch: ' + path)
        sources[path] = content
    return sources, manifest, digest(manifest_bytes)


def log_rows(content):
    rows = []
    for line, raw in enumerate(content.splitlines(), 1):
        if raw.startswith(b'{'):
            record = json.loads(raw)
            rows.append(dict(line=line, raw=record, fields=record.get('fields', {})))
    return rows

