#!/usr/bin/env python3
"""CPU-only current-evidence plots; no benchmark execution or legacy-renderer imports.

Run with the Python 3.12.4 executable documented in analysis_v1/PLOTS.md.
Inputs are immutable curated JSON plus locally retained, hash-bound HTTP receipts.
--check regenerates in memory and compares every output, including the manifest.
For label-only changes, --verified-data-manifest plus its explicit SHA-256 reuses
prior verified plot data without reopening raw receipts; validation is inherited.
"""
import argparse
from collections import Counter
import hashlib
from importlib import metadata
import io
import json
import math
from pathlib import Path
from statistics import median
import sys
import zlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / 'papers/arxiv/experiments/storage_demand/paired_read_v1'
PINS = {
    'http-transfer-v1/pilot-attempt-03.json': '15fe86b9963118e65fe1433e0b01e04ce6418428f381a88f1e0cb258095b49fe',
    'http-transfer-v1/pilot-attempt-03-requests.json': '7eec5d3e52f96f76bf0a3a9c8c44fadd98705808628d67021f9a731ac2009034',
    'http-transfer-v1/acceptance-results.json': '853270d42d79e00e7afe4ddd3f01c6c9026cce77b4763857a5e4b37b1baf147d',
    'engineering-20261016-v1/completed-stages.json': '8492747e3db129ba63faaf947d23e12c22d94a1e9ffdf1da4ab0bbdcc1bf81ec',
    'engineering-20261016-v1/grouped-quality-results.json': '6f66971ac2659112b3d1c0a751011aee65d7405d4789b3868f3e61dd4ac35552',
    'engineering-20261016-v1/grouped-profile-results.json': '033515cc938a3bf7c266a29918f33d122406fb6da5b32b44cc4304c97a5b3141',
    'engineering-20261016-v1/profile-results.json': 'df917069df9780ee12e798109490b7965c390cc1a1b3f7b60d5882f0bb217271',
    'engineering-20261016-v1/performance-results.json': '5b395cfc4d4cc3840621a300b13ce63f2181622ee182bda108ef48a91f1e9f89',
    'engineering-20261016-v1/grouped-performance-results.json': '8c5ee3cda54a53da2a6b7273f5279d7f29807122a3cac593ddc33592d4f1231b',
    'analysis_v1/renderer-pins.json': '5cec3527f097cac8f91eb7820a85d32bfe6a04b75a44256a6a0c389c996f5d2a',
}
COLORS = dict(blocking='#0072B2', lookahead='#008577', bad='#D55E00',
              ink='#202832', gray='#68727D', grid='#D9DDE3')
METRICS = ('median_ttft_seconds', 'median_request_e2e_seconds',
           'phase_qualified_tokens_per_second')
INPUTS = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode()


def load(path, expected):
    content = path.read_bytes()
    require(sha(content) == expected, f'Input hash mismatch: {path}')
    INPUTS[str(path.relative_to(ROOT))] = expected
    return json.loads(content)


def close(actual, expected, name):
    require(math.isclose(actual, expected, rel_tol=1e-13, abs_tol=1e-12), name)


def environment(pins):
    import matplotlib as mpl
    mpl.use('Agg', force=True)
    from matplotlib import font_manager, ft2font
    require(sys.version.split()[0] == pins['python'], 'Use pinned Python 3.12.4')
    for name, version in pins['dependencies'].items():
        require(metadata.version(name) == version, f'Renderer dependency: {name}')
    require(ft2font.__freetype_version__ == pins['freetype'], 'FreeType mismatch')
    require(zlib.ZLIB_VERSION == pins['zlib_build'], 'zlib build mismatch')
    require(zlib.ZLIB_RUNTIME_VERSION == pins['zlib_runtime'], 'zlib runtime mismatch')
    for weight, pin in pins['fonts'].items():
        font = Path(mpl.get_data_path()) / 'fonts/ttf' / pin['name']
        require(sha(font.read_bytes()) == pin['sha256'], f'Font mismatch: {weight}')
        resolved = Path(font_manager.findfont(font_manager.FontProperties(
            family=pins['font_family'], weight=weight)))
        require(sha(resolved.read_bytes()) == pin['sha256'], 'Font substitution')
    mpl.rcParams.update({'font.family': pins['font_family'], 'font.size': 8,
        'axes.labelsize': 8, 'axes.titlesize': 8, 'axes.titleweight': 'bold',
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': COLORS['gray'], 'text.color': COLORS['ink'],
        'axes.labelcolor': COLORS['ink'], 'xtick.color': COLORS['ink'],
        'ytick.color': COLORS['ink'], 'pdf.fonttype': 42,
        'svg.hashsalt': 'ferrule-current-evidence-v1',
        'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    return pins


def http_data(pilot, requests, acceptance):
    require(pilot['schema'] == 'ferrule.http-transfer.pilot-attempt03.v1', 'Pilot identity')
    require(pilot['attempt'] == 3 and pilot['actual_completed_lifecycles'] == 8, 'Pilot units')
    require(pilot['request_records']['sha256'] == PINS['http-transfer-v1/pilot-attempt-03-requests.json'], 'Request binding')
    require(pilot['denominators'] == requests['denominators'] == dict(
        scheduled=96, issued=96, successful=96, terminal=96, failed=0, censored=0, unissued=0), 'Pilot denominators')
    require(len(requests['rows']) == requests['row_count'] == 96, 'Request count')
    require(pilot['actual_measured_tokens'] == requests['actual_measured_tokens'] == 320, 'Token denominator')
    require(acceptance['actual_server_starts'] == len(acceptance['cells']) == 4, 'Acceptance cells')
    require(acceptance['scheduled_requests'] == acceptance['successful_requests'] == 48, 'Acceptance denominator')
    expected_schedule = {x['index']: x for x in requests['fixed_schedule']}
    require(len(expected_schedule) == 12, 'Fixed request schedule')
    raw_root = ROOT / pilot['raw_evidence_root']
    arms = {a['trial']: a for a in pilot['arms']}
    computed = {}
    for trial, arm in arms.items():
        rows = [r for r in requests['rows'] if r['trial'] == trial]
        require(len(rows) == 12 and sorted(r['index'] for r in rows) == list(range(12)), 'Trial request identities')
        for row in rows:
            schedule = expected_schedule[row['index']]
            ref = schedule['expected']
            require(row['name'] == schedule['name'] and row['measured'] == schedule['measured'], 'Schedule identity')
            require(row['reference_qualified'] and row['lifecycle_qualified'] and row['status'] == 'success', 'HTTP qualification')
            require(row['actual_generated_ids'] == ref['reference_ids'], 'Actual generated IDs')
            require(row['text'] == ref['reference_text'] and row['usage'] == ref['usage'], 'Text and usage')
            require(row['finish_reason'] == ref['finish_reason'], 'EOS/finish semantics')
            observed = load(raw_root / row['observed']['path_relative_to_raw_root'], row['observed']['sha256'])
            terminal = load(raw_root / row['http_terminal']['path_relative_to_raw_root'], row['http_terminal']['sha256'])
            require(observed['actual'] == terminal, 'Raw HTTP terminal binding')
            require(observed['trace']['actual_generated_ids'] == row['actual_generated_ids'], 'Raw worker IDs')
            require(observed['trace']['actual_worker_request_id'] == row['actual_worker_request_id'], 'Worker request binding')
            require(observed['trace']['actual_http_completion_ids'] == row['actual_http_completion_ids'], 'Completion ID binding')
            require(observed['trace']['terminal_reconciled'], 'Terminal reconciliation')
            for key in ('text', 'usage', 'finish_reason', 'sent_at', 'client_dispatch_at', 'ended_at'):
                require(terminal[key] == row[key], f'Raw field {key}')
            events = [e for e in terminal['raw_sse_events'] if e['data'] != '[DONE]']
            contents = [(e['at'], c.get('text', c.get('delta', {}).get('content', '')))
                        for e in events for c in json.loads(e['data']).get('choices', [])]
            nonempty = [(t, s) for t, s in contents if s]
            require(''.join(s for _, s in nonempty) == row['text'], 'Raw SSE text')
            payloads = [json.loads(e['data']) for e in events]
            require([x['usage'] for x in payloads if 'usage' in x] == [ref['usage']], 'Raw SSE usage')
            finishes = [c['finish_reason'] for x in payloads for c in x.get('choices', []) if c.get('finish_reason')]
            require(finishes == [ref['finish_reason']], 'Raw SSE finish reason')
            require(any(e['data'] == '[DONE]' for e in terminal['raw_sse_events']), 'Raw SSE terminal')
            require(terminal['request_payload'] == schedule['payload'], 'Actual EOS/payload contract')
            close(nonempty[0][0], row['first_nonempty_content_at'], 'First content endpoint')
            close(row['first_nonempty_content_at'] - row['sent_at'], row['native_ttft_seconds'], 'Client TTFT')
            close(row['ended_at'] - row['sent_at'], row['native_e2e_seconds'], 'Client terminal E2E')
        measured = [r for r in rows if r['measured']]
        require(len(measured) == 5 and all(r['completion_tokens'] == 8 for r in measured), 'Measured QA units')
        wall = max(r['ended_at'] for r in measured) - min(r['client_dispatch_at'] for r in measured)
        computed[trial] = dict(median_ttft_seconds=median(r['native_ttft_seconds'] for r in measured),
            median_request_e2e_seconds=median(r['native_e2e_seconds'] for r in measured),
            phase_qualified_tokens_per_second=sum(r['completion_tokens'] for r in measured) / wall)
        close(wall, arm['metrics']['phase_wall_seconds'], 'Phase W includes gaps')
        for metric in METRICS:
            close(computed[trial][metric], arm['metrics'][metric], metric)
    require(sum(r['measured'] for r in requests['rows']) == 40, 'Measured request count')
    require(sum(r['completion_tokens'] for r in requests['rows']) == pilot['all_output_tokens'] == 720, 'All actual tokens')
    pairs = {}
    for pair in pilot['paired_blocks']:
        schedule, block = pair['read_schedule'], pair['block']
        require(pair['qualified'] and pair['crosspolicy_equal_work'] and pair['crosspolicy_exact_output'], 'Pair qualification')
        sync = computed[f'{block}-{schedule}-synchronous']
        grouped = computed[f'{block}-{schedule}-grouped-synchronous']
        values = {m: grouped[m] / sync[m] for m in METRICS}
        for metric, value in values.items():
            close(value, pair['grouped_over_synchronous'][metric], 'Pair ratio')
        pairs[block] = values
    require(set(pairs) == {'B1', 'B2', 'L1', 'L2'}, 'Exact pair identities')
    for schedule, blocks in [('blocking', ('B1', 'B2')), ('lookahead', ('L1', 'L2'))]:
        summary = pilot['by_read_schedule'][schedule]
        require(summary['qualified_pairs'] == 2 and summary['interval'] is None and not summary['significance_claim'], 'n=2 scope')
        for metric in METRICS:
            close(median(pairs[b][metric] for b in blocks), summary['median_paired_ratios'][metric], 'Arithmetic median of two ratios')
    # The four acceptance stats snapshots are checked independently of pilot timings.
    for cell in acceptance['cells']:
        ref = cell['server_log']
        path = ROOT / acceptance['raw_evidence_root'] / ref['path_relative_to_raw_root']
        content = path.read_bytes()
        require(sha(content) == ref['sha256'], 'Acceptance server-log hash')
        INPUTS[str(path.relative_to(ROOT))] = ref['sha256']
        endpoints = [json.loads(x['fields']['expert_read_pipeline'])
                     for line in content.splitlines() if line.startswith(b'{')
                     for x in [json.loads(line)] if 'expert_read_pipeline' in x.get('fields', {})]
        endpoint = endpoints[-1]
        require(sha(canonical(endpoint['stats'])) == cell['final']['full_stats_sha256'], 'Full acceptance stats hash')
        require(sha(canonical(endpoint)) == cell['final']['full_endpoint_sha256'], 'Full acceptance endpoint hash')
        require(endpoint['stats']['peak_host_charged_bytes'] == 16777984, 'Host peak')
        require(endpoint['current_charge']['host_charged_bytes'] == 10485760, 'Live idle charge')
    return pairs


def mechanism_data(group, negative, performance, worker, quality, grouped_performance):
    require(group['status'] == 'failed' and not group['all4_passed'], 'Preserve failed profile stage')
    require(set(group['qualified_schedule_pair_index']) == {'blocking', 'lookahead'}, 'Qualified schedule index')
    require(not group['full_trace_accepted'] and group['dropped_event_count'] is None, 'Partial trace scope')
    require(group['multi_stage_interpretation']['same_frozen_application_binary'], 'Same application')
    observations = list(group['qualified_process_observations'].values())
    observations += list(group['additional_profile_stages'][0]['qualified_process_observations'].values())
    require(len(observations) >= 4 and all(o['observed_kernel_rows'] == 409411 for o in observations), 'Observed kernel rows')
    require(quality['actual_gpu_tests_issued'] == len(quality['rows']) == 14, 'Grouped quality gates')
    require(Counter(r['role'] for r in quality['rows']) == dict(backend=1, model=5, composition=8), 'Quality gate families')
    require(all(r['status'] == 'passed' for r in quality['rows']), 'Quality results')
    gates = worker['stages'][0]['summary']['gates']
    require(len(gates) == 9 and all(g['passed'] for g in gates), 'Worker nine gates')
    for gate in gates[1:7]:
        require(gate['last_worker_counts']['workers_created'] == 1, 'One worker')
        require(gate['last_worker_counts']['worker_reuses'] == (159 if 'full40' in gate['id'] else 2559), 'Worker reuse')
        require(gate['last_host_charge'] == 10485760, 'Worker live idle')
    require(not performance['all24_passed'] and not grouped_performance['all24_passed'], 'Do not promote failed planned stages')
    require(len(performance['paired_whole_test_summary']['pairs']) == 8, 'Negative whole-test pairs')
    require(performance['minimum_retained_logical_reservation_bytes'] == 36701896, 'Rejected host peak')
    rows = {}
    for schedule, comparison in [('blocking', group['comparisons']['blocking']),
            ('lookahead', group['additional_profile_stages'][0]['lookahead_comparison'])]:
        require(group['qualified_schedule_pair_index'][schedule]['qualified_partial_pair'], 'Qualified partial pair')
        require(comparison['normalization_denominator'] == 25116, 'Logical upload denominator')
        require(comparison['ordered_work_checks_and_fingerprints_equal'] and comparison['kernel_name_counts_equal'], 'Same observed work')
        sync = next(a for a in comparison['all_observed_api_deltas'] if a['name'] == 'cuStreamSynchronize')
        require((sync['synchronous_count'], sync['grouped_count'], sync['count_delta']) == (382743, 257163, -125580), 'Native sync counts')
        require(sync['count_delta_per_logical_upload'] == -5, 'Sync normalization')
        copy = comparison['copy_counts_and_bytes']['CUDA_MEMCPY_KIND_HTOD']
        require(copy['count_equal'] and copy['bytes_equal'] and copy['synchronous_count'] == 195985
                and copy['synchronous_bytes'] == 82512273944, 'Observed H2D equality')
        neg = next(c for c in negative['comparisons'] if c['schedule'] == schedule)
        require(neg['logical_upload_denominator'] == 25116, 'Separate negative denominator')
        pins = [a for a in neg['api_deltas'] if a['synchronous']['family'] == 'host_pin_unpin']
        require(len(pins) == 2 and all((a['synchronous']['count'], a['retained']['count']) == (2, 25118) for a in pins), 'Pin/free churn')
        denominator = comparison['normalization_denominator']
        rows[schedule] = dict(sync=[sync['synchronous_count'] / denominator, sync['grouped_count'] / denominator],
            sync_count_delta=sync['count_delta'], sync_delta_per_upload=sync['count_delta_per_logical_upload'],
            uploads=denominator, pin_counts=[pins[0][policy]['count'] for policy in ('synchronous', 'retained')],
            pin_free_extra_seconds=neg['pinned_host_allocate_plus_free']['host_duration_delta_ns'] / 1e9)
    return rows


def http_figure(pairs):
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 3, figsize=(6.35, 3.45), sharey=True)
    fig.subplots_adjust(left=.13, right=.98, bottom=.20, top=.77, wspace=.20)
    titles = ['(a) Client TTFT\nLower is better', '(b) Client E2E\nLower is better',
              '(c) Phase-qualified goodput\nHigher is better']
    for ax, metric, title in zip(axes, METRICS, titles):
        for schedule, blocks, y in [('blocking', ('B1', 'B2'), 1.25), ('lookahead', ('L1', 'L2'), 0)]:
            color = COLORS['bad'] if schedule == 'blocking' and metric == METRICS[0] else COLORS[schedule]
            values = [pairs[b][metric] for b in blocks]
            for value, offset, marker in zip(values, (.22, .04), ('o', '^')):
                ax.plot(value, y + offset, marker=marker, ms=4.5, color=color, linestyle='None')
            value = median(values)
            ax.plot(value, y - .16, marker='D', color=COLORS['ink'], ms=4, linestyle='None')
            ax.text(.5, y - .47, f'{value:.4f} ({(value - 1) * 100:+.2f}%)',
                    transform=ax.get_yaxis_transform(), ha='center', color=color, weight='bold',
                    bbox=dict(facecolor='white', edgecolor='none', pad=1))
        ax.axvline(1, color=COLORS['gray'], ls='--', lw=.8, zorder=0)
        ax.set(xlim=(.85, 1.115), ylim=(-.70, 1.85), xticks=[.9, 1, 1.1], title=title)
        ax.grid(axis='x', color=COLORS['grid'], lw=.5)
        ax.tick_params(axis='y', length=0)
    axes[0].set_yticks([1.29, .04], ['Blocking', 'Lookahead'])
    fig.legend(handles=[Line2D([], [], marker=m, color=COLORS['ink'], ls='None', ms=4.5, label=l)
        for m, l in [('o', 'Pair 1 (B1 / L1)'), ('^', 'Pair 2 (B2 / L2)'), ('D', 'Median of two ratios')]],
        loc='upper center', bbox_to_anchor=(.5, .97), ncol=3, frameon=False)
    fig.text(.5, .09, 'Grouped / synchronous within each read schedule; dashed line = no change', ha='center')
    n = len(pairs) // 2
    fig.text(.5, .025, f'Analysis unit: pair (n = {n} per schedule). No intervals or significance test.', ha='center')
    return fig


def mechanism_figure(data):
    from matplotlib import pyplot as plt
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.35, 3.50))
    fig.subplots_adjust(left=.15, right=.97, bottom=.31, top=.76, wspace=.70)
    left.set_title('(a) Grouped transfer\nQualified partial process captures', loc='left', pad=12)
    for schedule, y, hatch in [('blocking', 3, ''), ('lookahead', 1, '///')]:
        for value, offset, face in zip(data[schedule]['sync'], (0, -1), ('white', COLORS[schedule])):
            left.barh(y + offset, value, height=.55, facecolor=face, edgecolor=COLORS['ink'], hatch=hatch, lw=.7)
            left.text(value + .3, y + offset, f'{value:.2f}', va='center')
    left.set(yticks=[3, 2, 1, 0], yticklabels=['B: sync', 'B: grouped', 'L: sync', 'L: grouped'],
             xlim=(0, 19), xticks=[0, 5, 10, 15], ylim=(-.65, 3.7), xlabel='Observed stream-sync calls / upload')
    right.set_title('(b) Rejected retained alternative\nSeparate older paired captures', loc='left', pad=12)
    for schedule, y, hatch in [('blocking', 1, ''), ('lookahead', 0, '///')]:
        value = data[schedule]['pin_free_extra_seconds']
        right.barh(y, value, height=.35, color=COLORS['bad'], edgecolor=COLORS['ink'], hatch=hatch, lw=.7)
        right.text(.8, y + .24, f'+{value:.3f} s', va='bottom', color=COLORS['bad'], weight='bold')
    right.set(yticks=[1, 0], yticklabels=['Blocking', 'Lookahead'], xlim=(0, 34), xticks=[0, 10, 20, 30],
              ylim=(-.5, 1.65), xlabel='Extra pin + free host API sum (s)')
    for ax in (left, right):
        ax.set_axisbelow(True)
        ax.grid(axis='x', color=COLORS['grid'], lw=.5)
        ax.tick_params(axis='y', length=0)
    fig.text(.5, .93, 'Mechanism observations are not a causal elapsed-time decomposition', ha='center', weight='bold')
    values = data['blocking']
    fig.text(.5, .13, f'Both experiments: {values["uploads"]:,} logical uploads per test; separate source cuts and controls.', ha='center')
    counts = values['pin_counts']
    fig.text(.5, .07, f'(a) {values["sync_count_delta"]:,} calls = {values["sync_delta_per_upload"]:g} / upload.  '
             f'(b) Pin allocations and frees: {counts[0]:,} → {counts[1]:,} each.', ha='center')
    fig.text(.5, .01, 'Process-wide selected-model observations only; full-trace completeness and drop counts unknown.', ha='center')
    return fig


def validate_layout(fig):
    """Reject clipped or mutually overlapping visible text at publication size."""
    from matplotlib.text import Text
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    seen, texts = set(), []
    for text in fig.findobj(match=Text):
        if id(text) in seen or not text.get_visible() or not text.get_text():
            continue
        seen.add(id(text))
        box = text.get_window_extent(renderer)
        if box.width <= 0 or box.height <= 0:
            continue
        require(text.get_fontsize() >= 8, 'Publication font below eight points')
        require(fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1),
                f'Clipped text: {text.get_text()}')
        texts.append((text, box))
    for i, (text, box) in enumerate(texts):
        for other, bounds in texts[i + 1:]:
            require(not box.overlaps(bounds),
                    f'Overlapping text: {text.get_text()} / {other.get_text()}')


def verified_plot_data(path, expected):
    require(expected is not None, 'Verified data requires its explicit SHA-256')
    content = path.read_bytes()
    require(sha(content) == expected, 'Verified manifest snapshot hash mismatch')
    previous = json.loads(content)
    require(previous['schema'] == 'ferrule.current-evidence.figures.v1', 'Verified manifest schema')
    keys = ('schema', 'inputs', 'generator_sha256', 'renderer', 'validation',
            'http_pair_ratios', 'mechanism')
    require(sha(encode({key: previous[key] for key in keys})) == previous['evidence_sha256'],
            'Verified manifest evidence identity')
    for name, digest in PINS.items():
        require(previous['inputs'][str((BASE / name).relative_to(ROOT))] == digest,
                f'Verified manifest input pin: {name}')
    toolchain = environment(load(BASE / 'analysis_v1/renderer-pins.json',
                                PINS['analysis_v1/renderer-pins.json']))
    require(toolchain == previous['renderer'], 'Verified manifest renderer identity')
    INPUTS.update(previous['inputs'])
    return (toolchain, previous['http_pair_ratios'], previous['mechanism'],
            previous['validation'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--verified-data-manifest', type=Path,
                        help='Reuse a prior verified manifest for presentation-only changes')
    parser.add_argument('--verified-data-sha256',
                        help='Exact SHA-256 of the preserved verified manifest')
    args = parser.parse_args()
    require((args.verified_data_manifest is None) == (args.verified_data_sha256 is None),
            'Supply both verified-data arguments or neither')
    if args.verified_data_manifest is not None:
        toolchain, pairs, mechanism, validation = verified_plot_data(
            args.verified_data_manifest, args.verified_data_sha256)
    else:
        sources = {name: load(BASE / name, digest) for name, digest in PINS.items()}
        toolchain = environment(sources['analysis_v1/renderer-pins.json'])
        pairs = http_data(*(sources['http-transfer-v1/' + name] for name in
            ('pilot-attempt-03.json', 'pilot-attempt-03-requests.json', 'acceptance-results.json')))
        mechanism = mechanism_data(*(sources['engineering-20261016-v1/' + name] for name in
            ('grouped-profile-results.json', 'profile-results.json', 'performance-results.json',
             'completed-stages.json', 'grouped-quality-results.json', 'grouped-performance-results.json')))
        validation = dict(pilot_raw_requests=96, acceptance_full_stats_snapshots=4,
            pilot_measured_requests=40, pilot_measured_tokens=320, pairs_per_read_schedule=2)
    provenance = dict(schema='ferrule.current-evidence.figures.v1', inputs=INPUTS,
        generator_sha256=sha(Path(__file__).read_bytes()), renderer=toolchain,
        validation=validation, http_pair_ratios=pairs, mechanism=mechanism)
    identity = sha(encode(provenance))
    outputs = {}
    figures = {}
    from matplotlib import pyplot as plt
    for name, fig in [('current-http-outcomes', http_figure(pairs)),
                      ('current-transfer-mechanism', mechanism_figure(mechanism))]:
        validate_layout(fig)
        figures[name] = dict(size_inches=list(fig.get_size_inches()),
                            label='fig:' + name, minimum_font_points=8)
        for extension in ('pdf', 'png'):
            buffer = io.BytesIO()
            meta = dict(Title=name, Author='Ferrule', Subject=f'current-evidence-sha256:{identity}',
                        Creator='render_current_evidence.py', CreationDate=None, ModDate=None)
            if extension == 'png':
                meta = dict(Title=name, Software='render_current_evidence.py', Description=f'current-evidence-sha256:{identity}')
            fig.savefig(buffer, format=extension, dpi=180, metadata=meta)
            outputs[name + '.' + extension] = buffer.getvalue()
        plt.close(fig)
    provenance.update(evidence_sha256=identity, figures=figures,
        outputs={name: dict(sha256=sha(content), bytes=len(content)) for name, content in outputs.items()})
    outputs['current-evidence-manifest.json'] = encode(provenance)
    for name, content in outputs.items():
        path = HERE / name
        if args.check:
            require(path.read_bytes() == content, f'Stale generated output: {name}')
        else:
            path.write_bytes(content)
    scope = ('prior verified data reused; raw receipts not reopened' if args.verified_data_manifest
             else '96 raw HTTP requests, 4 full stats snapshots verified')
    print(f'{"Checked" if args.check else "Generated"} {len(outputs)} artifacts; {scope}; pinned renderer and source hashes verified.')


if __name__ == '__main__':
    main()
