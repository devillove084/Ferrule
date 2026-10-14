#!/usr/bin/env python3
"""Offline, hash-bound plots of the fixed -01/-02 receipts; no harness imports."""
import argparse
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('storage_demand_summary', HERE / 'summarize.py')
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)
ROOT = S.ROOT
OUT = 'papers/arxiv/experiments/storage_demand/analysis/output/'
FIG = 'papers/arxiv/figures/storage-demand-'
GENERATOR = 'papers/arxiv/experiments/storage_demand/analysis/plot_storage_demand.py'
MIB, GIB = 2**20, 2**30
LABELS = ('Hello2', 'QA canary', 'Arithmetic canary', 'Chinese canary')
COLORS = dict(read='#0072B2', h2d='#D55E00', resident='#009E73', peak='#CC79A7',
              limit='#555555', grid='#D9DDE3', ink='#202832')
CAPTIONS = {
    'logical-transfer':
        'Paired logical checkpoint-reader and facade H2D byte deltas for four serialized '
        'requests in storage-demand TP1 middle -02. GiB means 2^30 bytes; the right panel '
        'resolves the small H2D-minus-reader differences in KiB (2^10 bytes). These are '
        'application counters, not physical storage I/O or physical bus traffic; startup '
        'I/O is outside invocation accounting.',
    'cache-reconciliation':
        'Expert entry reconciliation and separate cache gauges in storage-demand TP1 '
        'middle -02. The 11,519 uploads minus 11,455 evictions equal 64 entries, matching '
        'the last observed resident-entry gauge; the implied initial occupancy is zero, '
        'not independently measured. Subsequent observed boundaries satisfy the entry '
        'balance. The recorded cache.peak_bytes value, 209.574 MiB, is the cache '
        'charged-byte high-water (resident + pending + scratch), bounded by '
        'ExpertCacheLimits.max_bytes = 320 MiB, a cache accounting ceiling, not owner '
        'physical memory. No independently recorded owner physical peak or owner physical '
        'ceiling is available; no contradiction is implied. Invocation gauges and lifetime '
        'peaks are not accumulated across invocations or a final owner ledger.',
    'timing':
        'Launch-to-ready startup and per-request HTTP TTFT/E2E in the single successful '
        'DEBUG-instrumented storage-demand TP1 middle -02 run (n=1). TTFT is send to first '
        'nonempty SSE content; E2E is send to EOF. Startup has a different origin and is '
        'not part of the request bars. Instrumentation overhead is unmeasured; there '
        'are no statistics, confidence intervals, kernel timings, or comparative speedups.',
    'invocation-timeline':
        'Per-invocation expert upload/eviction deltas and resident-entry observations '
        'for the 30 completed invocations in storage-demand TP1 middle -02. Serialized '
        'request/log-byte anchors assign the 3/9/9/9 invocations to requests. The x-axis '
        'is log-order invocation ordinal, not elapsed time, output-token index, or '
        'transaction-as-request ID. Entry occupancy is observed only at invocation '
        'completion, not continuously; uploads are not inferred cache misses.',
}
QUALITY_NOTE = ('Hello2: HTTP/usage only (reference null). Three canaries: text + eight-output-usage '
                'checks only. Generated IDs, exact-token acceptance and numeric acceptance: unknown.')


def checked_summary():
    """Recompute all original checks; a stale summary cannot silently feed the figures."""
    bindings_data = (HERE / 'inputs.json').read_bytes()
    bindings = json.loads(bindings_data)
    if bindings['algorithm'] != 'sha256':
        raise S.SchemaError('input manifest must use sha256')
    sources = S.load_sources()
    summary = S.analyze(sources, bindings['files'])
    summary['provenance'] = dict(
        generator_path='papers/arxiv/experiments/storage_demand/analysis/summarize.py',
        generator_sha256=S.digest((HERE / 'summarize.py').read_bytes()),
        input_manifest_path='papers/arxiv/experiments/storage_demand/analysis/inputs.json',
        input_manifest_sha256=S.digest(bindings_data),
        time_policy='no analysis timestamp; timings come only from bound receipts')
    if summary['consistency']['status'] != 'pass':
        raise S.SchemaError('receipt consistency failed: ' + str(summary['consistency']['divergences']))
    stale = S.check_outputs(HERE / 'output', S.render(summary))
    if stale:
        raise S.SchemaError('stale summary artifacts; run summarize.py first: ' + ', '.join(stale))
    return summary, sources


def units(value, divisor):
    if not S.number(value) or value < 0 or divisor not in (1, 2**10, MIB, GIB):
        raise S.SchemaError('unknown/invalid quantity or binary unit')
    return value / divisor


def gauge_observations(rows, key):
    """No summation and no substitution of zero for an absent gauge."""
    values = [row['fields'].get(key) for row in rows]
    return dict(last=values[-1] if values and S.integer(values[-1]) else None,
                maximum=max(values) if values and all(S.integer(v) for v in values) else None)


def reconcile(rows):
    """The first pre-invocation occupancy is implied, never an observed sample."""
    if not rows:
        raise S.SchemaError('no anchored invocation counters to reconcile')
    cumulative, previous, result = 0, None, []
    for row in rows:
        f = row['fields']
        u, e, n = (f.get(key) for key in ('expert_uploads', 'expert_evictions', 'expert_resident_entries'))
        if not all(S.integer(v) for v in (u, e, n)):
            raise S.SchemaError('unknown entry counter/gauge; no reconciliation plot')
        cumulative += u - e
        residual = None if previous is None else n - previous - (u - e)
        if residual not in (None, 0):
            raise S.SchemaError('entry reconciliation failed at log line ' + str(row['line']))
        result.append(dict(cumulative_net_entries=cumulative, observed_entries=n,
                           change_from_previous_observation=None if previous is None else n - previous,
                           balance_residual=residual))
        previous = n
    initial = result[0]['observed_entries'] - result[0]['cumulative_net_entries']
    if initial < 0:
        raise S.SchemaError('negative implied initial occupancy')
    return dict(implied_initial_entries=initial, initial_entries_observed=None,
                total_net_entries=cumulative, last_observed_entries=previous, points=result)


def build_data(summary, sources):
    if summary['consistency']['status'] != 'pass':
        raise S.SchemaError('plots require source consistency PASS')
    for item in summary['inputs']:
        if S.digest(sources[item['path']]) != item['sha256'] or item['sha256'] != item['bound_sha256']:
            raise S.SchemaError('plot input hash mismatch: ' + item['path'])
    failed, run = summary['attempts']
    if failed['requests'] or run['n'] != 1:
        raise S.SchemaError('expected unexecuted -01 and single instrumented -02')
    prefix = run['source_directory'] + '/'
    log = sources[prefix + 'server.log']
    case = json.loads(sources[prefix + 'case.json'])
    requests, timeline, raw_rows = [], [], []
    for index, request in enumerate(run['requests']):
        observed = json.loads(sources[prefix + request['name'] + '/observed.json'])
        anchor = observed['anchor']
        actual = observed['actual']
        start, end = request['log_byte_interval']
        rows = S.invocations(log, start, end)
        if S.aggregate(rows)['deltas'] != request['additive_deltas']:
            raise S.SchemaError('anchored plot deltas do not match summary')
        if [row['line'] for row in rows] != request['source_log_lines']:
            raise S.SchemaError('anchored plot lines do not match summary')
        if any(value is not None for value in (actual['output_token_ids'], observed['observed_token_ids'],
                                                  request['observed_generated_token_ids'], request['exact_token_pass'])):
            raise S.SchemaError('fixed receipts do not expose generated IDs or exact-token acceptance')
        quality = S.request_quality(observed)
        if any(request[k] != v for k, v in quality.items()):
            raise S.SchemaError('derived quality differs from reference-presence checks')
        delta = request['additive_deltas']
        read, h2d = delta['checkpoint_read_bytes'], delta['h2d_bytes']
        if not all(S.integer(v) for v in (read, h2d)):
            raise S.SchemaError('unobserved bytes cannot be plotted as zero')
        ttft = actual['content_event_times'][0] - actual['sent_at']
        e2e = actual['ended_at'] - actual['sent_at']
        if not (0 <= ttft <= e2e and ttft == request['ttft_seconds'] and e2e == request['request_e2e_seconds']):
            raise S.SchemaError('HTTP timing equation failed')
        requests.append(dict(
            name=request['name'], label=LABELS[index], local_request_id=anchor['local_request_id'],
            log_byte_interval=[start, end], source_log_lines=request['source_log_lines'],
            invocations=len(rows), logical_read_bytes=read, facade_h2d_bytes=h2d,
            logical_read_gib=units(read, GIB), facade_h2d_gib=units(h2d, GIB),
            h2d_minus_read_bytes=h2d - read, h2d_minus_read_kib=(h2d - read) / 2**10,
            uploads=delta['expert_uploads'], evictions=delta['expert_evictions'],
            net_entries=delta['expert_uploads'] - delta['expert_evictions'],
            last_resident_entries=request['last_invocation_gauges']['expert_resident_entries'],
            ttft_seconds=ttft, e2e_seconds=e2e, output_usage_tokens=request['output_tokens'],
            observed_generated_token_ids=None, exact_token_pass=request['exact_token_pass'],
            numeric_acceptance_pass=None, **quality))
        for row in rows:
            f = row['fields']
            if f.get('counters_available') is not True or f.get('counter_reset_detected') is not False:
                raise S.SchemaError('unavailable/reset timeline counters')
            components = ('expert_resident_bytes', 'expert_pending_upload_bytes', 'expert_scratch_bytes')
            if not all(S.integer(f.get(key)) for key in components + ('expert_owner_peak_bytes',)):
                raise S.SchemaError('unknown cache accounting quantity')
            charged_bytes = sum(f[key] for key in components)
            if f['expert_owner_peak_bytes'] < charged_bytes:
                raise S.SchemaError('cache peak is below current charged bytes')
            timeline.append(dict(ordinal=len(timeline) + 1, request=request['name'],
                                 local_request_id=anchor['local_request_id'], log_byte_interval=[start, end],
                                 source_log_line=row['line'], invocation_id=row['invocation_id'],
                                 log_timestamp=row['raw']['timestamp'], uploads=f['expert_uploads'],
                                 evictions=f['expert_evictions'], cache_charged_bytes=charged_bytes,
                                 cache_peak_bytes=f['expert_owner_peak_bytes'],
                                 gauges={k: f.get(k) for k in S.GAUGES}))
            raw_rows.append(row)
    if [r['line'] for r in raw_rows] != [r['line'] for r in S.invocations(log)]:
        raise S.SchemaError('timeline is not the complete request-period raw log')
    reconciliation = reconcile(raw_rows)
    for point, balance in zip(timeline, reconciliation['points']):
        point.update(balance)
    config = run['declared_config']
    limits = dict(entries=int(config['cuda_expert_device_entries']),
                  cache_accounting_bytes=int(config['cuda_expert_device_bytes']),
                  scratch_bytes=int(config['cuda_expert_scratch_mb']) * MIB,
                  owner_physical_ceiling_bytes=None)
    # The bound startup plan confirms the legacy --*-mb flag's binary meaning.
    if ('scratch_bytes: ' + str(limits['scratch_bytes'])).encode() not in log:
        raise S.SchemaError('scratch unit not confirmed by bound startup plan')
    gauges = []
    for key, label, divisor, limit_key, limit_label, source_key in (
        ('expert_resident_entries', 'Resident entries', 1, 'entries', 'Entry ceiling', 'expert_resident_entries'),
        ('expert_resident_bytes', 'Resident bytes', MIB, 'cache_accounting_bytes',
         'Cache accounting ceiling', 'expert_resident_bytes'),
        ('expert_scratch_bytes', 'Scratch', MIB, 'scratch_bytes', 'Scratch ceiling', 'expert_scratch_bytes'),
        ('cache.peak_bytes', 'Cache charged-byte peak', MIB, 'cache_accounting_bytes',
         'Cache accounting ceiling', 'expert_owner_peak_bytes')):
        observed = gauge_observations(raw_rows, source_key)
        limit = limits[limit_key]
        if observed['maximum'] is not None and limit is not None and observed['maximum'] > limit:
            raise S.SchemaError('observed gauge exceeds its recorded limit: ' + key)
        gauges.append(dict(key=key, source_key=source_key, label=label,
                           unit='entries' if divisor == 1 else 'MiB', divisor=divisor,
                           last=observed['last'], maximum=observed['maximum'],
                           configured_limit=limit, configured_limit_label=limit_label,
                           limit_known=limit is not None))
    cache_peak = gauge_observations(raw_rows, 'expert_owner_peak_bytes')
    cache_accounting = dict(
        field='cache.peak_bytes', source_field='expert_owner_peak_bytes',
        meaning='cache charged-byte high-water: resident + pending + scratch',
        peak_bytes=cache_peak['maximum'], peak_last_bytes=cache_peak['last'],
        accounting_ceiling_field='ExpertCacheLimits.max_bytes',
        accounting_ceiling_bytes=limits['cache_accounting_bytes'],
        accounting_ceiling_label='Cache accounting ceiling',
        owner_physical_peak_bytes=None, owner_physical_ceiling_bytes=None,
        owner_physical_status='not independently recorded; cache peak is not owner physical memory')
    startup = case['ready_monotonic'] - case['launch_monotonic']
    if startup != run['timing']['startup_seconds']:
        raise S.SchemaError('startup equation failed')
    return dict(
        schema='storage_demand_fixed_receipt_plots.v2',
        provenance=dict(inputs=summary['inputs'], summary_sha256=S.digest(S.encode(summary)),
                        summarizer=summary['provenance'], generator_path=GENERATOR,
                        generator_sha256=S.digest((ROOT / GENERATOR).read_bytes())),
        units=dict(GiB_bytes=GIB, MiB_bytes=MIB, KiB_bytes=2**10, time='seconds',
                   legacy_scratch_mb='binary MiB; confirmed by bound startup plan'),
        attempts=[dict(run=a['run'], n=a['n'], status=a['status'], denominator=a['denominator'],
                       startup_seconds=a['timing']['startup_seconds'],
                       logical_read_bytes=a['additive_deltas']['checkpoint_read_bytes'])
                  for a in summary['attempts']],
        measured_run=run['run'], independent_successful_runs=1, debug_logging=run['debug_logging'],
        requests=requests, timeline=timeline, reconciliation=reconciliation, gauges=gauges,
        cache_accounting=cache_accounting, configured_limits=limits, startup_seconds=startup,
        totals=dict(logical_read_bytes=sum(r['logical_read_bytes'] for r in requests),
                    facade_h2d_bytes=sum(r['facade_h2d_bytes'] for r in requests),
                    uploads=sum(r['uploads'] for r in requests), evictions=sum(r['evictions'] for r in requests)),
        other_gauges={k: gauge_observations(raw_rows, k) for k in
                      ('expert_pending_upload_bytes', 'expert_unknown_quarantine_bytes')},
        quality=dict(**run['quality'], numeric_acceptance_status='not_established',
                     numeric_acceptance_pass=None, observed_generated_token_ids=None),
        unobserved=dict(run['unobserved'], owner_physical_peak_bytes=None, owner_physical_ceiling_bytes=None),
        captions=CAPTIONS, caveats=summary['caveats'],
        equations=dict(bytes='R_q=sum_i checkpoint_read_bytes_i; H_q=sum_i h2d_bytes_i',
                       entries='N_i-N_(i-1)=uploads_i-evictions_i (observed boundaries i>=2)',
                       first_boundary='N_0 implied = N_1 - uploads_1 + evictions_1; N_0 not measured',
                       gauges='last and observed maximum only; never accumulate across invocations; cache peak is not owner physical memory',
                       cache_peak='cache.peak_bytes = high-water resident + pending + scratch; bounded by ExpertCacheLimits.max_bytes',
                       timing='startup=ready-launch; TTFT=first_nonempty_content-send; E2E=EOF-send'),
        unsupported=['physical I/O and page-cache state', 'cache misses and IO credits',
                     'full C=F+R+I+L+T credit partition: component ledgers unobserved',
                     'physical-capacity/storage-necessity acceptance', 'continuous occupancy and shutdown ledger',
                     'exact generated IDs and full numerical acceptance', 'confidence intervals and comparative speedup'])


def latex_caption(caption):
    """Escape code identifiers while preserving their exact terminology."""
    return (caption.replace('2^30', r'\(2^{30}\)').replace('2^10', r'\(2^{10}\)')
            .replace('cache.peak_bytes', r'\texttt{cache.peak\_bytes}')
            .replace('ExpertCacheLimits.max_bytes', r'\texttt{ExpertCacheLimits.max\_bytes}'))


def plot_guide(data):
    lines = ['# Storage-demand receipt plots', '',
             'Single successful TP1 DEBUG run (-02), n=1; -01 failed before server launch. '
             'No physical-I/O, cache-miss, numerical-acceptance, or speedup inference.', '',
             QUALITY_NOTE, '', '## Figures and exact paper includes', '',
             'Use the following snippets from `papers/arxiv` with the existing `graphicx` package. '
             'They are suggestions only; the generator does not edit the manuscript.', '']
    for name, caption in CAPTIONS.items():
        lines += [f'### {name}', '', f'PDF: `papers/arxiv/figures/storage-demand-{name}.pdf`',
                  f'Preview: `papers/arxiv/figures/storage-demand-{name}.png`', '', caption, '',
                  '```latex', r'\begin{figure*}[t]', r'  \centering',
                  rf'  \includegraphics[width=\textwidth]{{figures/storage-demand-{name}.pdf}}',
                  '  \\caption{' + latex_caption(caption) + '}',
                  rf'  \label{{fig:storage-demand-{name}}}', r'\end{figure*}', '```', '']
    lines += ['## Mathematics and limits', '',
              '- GiB = 2^30 bytes; MiB = 2^20 bytes; KiB = 2^10 bytes. Seconds come only from bound HTTP/launch receipts.',
              '- Per-request reader/H2D bytes sum only additive invocation deltas. The KiB inset prevents rounded GiB labels from implying exact equality.',
              '- Entry reconciliation is count arithmetic, not byte conservation. Evicted-byte deltas are not recorded, so no byte-flow balance is fabricated.',
              '- Initial occupancy is inferred as zero by the first balance equation, not an independently observed pre-request gauge. Every later observed boundary reconciles exactly.',
              '- Entries: 64; cache accounting ceiling: 335,544,320 bytes = 320 MiB; scratch cap: 8,388,608 bytes = 8 MiB. Resident bytes are one component of the cache charge, not a separate resident-only byte budget. The bound startup log confirms the binary meaning of the legacy scratch-MB flag.',
              '- `cache.peak_bytes` (raw receipt key `expert_owner_peak_bytes`) is cache charged-byte high-water (resident + pending + scratch), bounded by `ExpertCacheLimits.max_bytes` = 320 MiB, a cache accounting ceiling, not owner physical memory.',
              '- The 209.574 MiB value is a cache accounting peak, not an independently recorded owner-physical peak. Owner physical peak/ceiling remain unobserved; no contradiction with the 320 MiB cache accounting ceiling is implied. Pending-upload/quarantine observations are zero but do not prove shutdown closure.',
              '- Device budget (7168 MiB) and workspace (2048 MiB) have different scopes; they are not cache ceilings and are not summed with cache gauges.',
              '- These are configured accounting limits, not physical capacity. The manuscript admission and credit-conservation equations lack full ledger observations here; the figures do not purport to verify them.',
              '- The timeline is log-order only, mapped by actual byte anchors, with no interpolated samples or fabricated token/time alignment.', '',
              '## Reproduction and provenance', '',
              'Run from the repository root using the already installed Python 3.12 / Matplotlib toolchain:', '',
              '```sh',
              '/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B papers/arxiv/experiments/storage_demand/analysis/summarize.py --check',
              '/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B -m unittest discover -s papers/arxiv/experiments/storage_demand/analysis/tests -v',
              '/home/admin/wxw403883/anaconda3/bin/python3.12 -B papers/arxiv/experiments/storage_demand/analysis/plot_storage_demand.py',
              '/home/admin/wxw403883/anaconda3/bin/python3.12 -B papers/arxiv/experiments/storage_demand/analysis/plot_storage_demand.py --check',
              '```', '',
              'The data derivation and tests use only stdlib (`-S` supported); rendering lazily uses existing '
              'Matplotlib/Agg, NumPy and Pillow. No installs, network, GPU calls, harness imports or TeX engine are needed.', '',
              '`plot-data.json` binds all 41 sealed receipts, the checked summary, input manifest, summarizer '
              'and plot-generator hashes. `plot-manifest.json` seals data, guide, every PDF/PNG and the contact '
              'sheet, and records renderer/font versions. PDFs and PNGs embed the input/generator/data hashes. '
              'No current date, hostname, random identifier or mtime is included. `--check` re-renders in memory '
              'and compares every expected byte without overwriting artifacts or resealing inputs. '
              'Byte reproducibility is scoped to the recorded renderer/font toolchain, not promised across '
              'arbitrary Matplotlib/FreeType versions.', '',
              'Preview contact sheet: `papers/arxiv/figures/storage-demand-preview.png`.', '',
              'Unknown quantities remain null and are excluded from plotted series. -01 is documented as '
              'unexecuted, not inserted as a zero-valued observation. No error bars, fitted model, distribution '
              'summary, or statistical denominator is manufactured.', '']
    return '\n'.join(lines).encode('utf-8')


def render_figures(data):
    """Finite CPU-only rendering; metadata and font selection are deterministic."""
    with tempfile.TemporaryDirectory(prefix='.storage-plot-', dir=HERE / 'output') as cache:
        os.environ['MPLCONFIGDIR'] = cache
        import matplotlib as mpl
        mpl.use('Agg', force=True)
        from matplotlib import pyplot as plt, font_manager, ft2font
        import numpy as np
        from PIL import Image, __version__ as pillow_version
        font = Path(font_manager.findfont('DejaVu Sans'))
        renderer = dict(matplotlib=mpl.__version__, numpy=np.__version__, pillow=pillow_version,
                        freetype=ft2font.__freetype_version__, font='DejaVu Sans',
                        font_sha256=S.digest(font.read_bytes()), backend='Agg', preview_dpi=150)
        hashes = dict(generator_sha256=data['provenance']['generator_sha256'],
                      input_manifest_sha256=data['provenance']['summarizer']['input_manifest_sha256'],
                      plot_data_sha256=S.digest(S.encode(data)))
        metadata = json.dumps(hashes, sort_keys=True)
        style = {'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
                 'axes.labelsize': 9, 'axes.edgecolor': COLORS['limit'],
                 'axes.spines.top': False, 'axes.spines.right': False,
                 'text.color': COLORS['ink'], 'axes.labelcolor': COLORS['ink'],
                 'pdf.fonttype': 42, 'savefig.facecolor': 'white', 'figure.facecolor': 'white'}
        outputs = {}

        def finish(fig, name, notes):
            fig.suptitle('Storage-demand TP1 middle -02 | single DEBUG run, n=1', fontsize=12, fontweight='bold', y=.975)
            fig.text(.5, .015, notes, ha='center', va='bottom', fontsize=8, linespacing=1.6)
            for ax in fig.axes:
                ax.grid(axis='y', color=COLORS['grid'], linewidth=.6, zorder=0)
                ax.set_axisbelow(True)
            for suffix in ('pdf', 'png'):
                buffer = io.BytesIO()
                if suffix == 'pdf':
                    meta = dict(Title='Storage demand: ' + name, Subject=metadata,
                                Creator=GENERATOR, CreationDate=None, ModDate=None)
                else:
                    meta = dict(Software=GENERATOR, Description=metadata)
                fig.savefig(buffer, format=suffix, dpi=150, metadata=meta)
                outputs[FIG + name + '.' + suffix] = buffer.getvalue()
            plt.close(fig)

        with mpl.rc_context(style):
            requests, x = data['requests'], np.arange(len(data['requests']))
            fig, (ax, diff) = plt.subplots(1, 2, figsize=(10, 4.7), gridspec_kw={'width_ratios': [2.25, 1]})
            fig.subplots_adjust(left=.07, right=.97, top=.80, bottom=.25, wspace=.36)
            for offset, key, color, hatch, label in (
                (-.19, 'logical_read_gib', COLORS['read'], '', 'Logical checkpoint reader'),
                (.19, 'facade_h2d_gib', COLORS['h2d'], '//', 'Facade H2D')):
                bars = ax.bar(x + offset, [r[key] for r in requests], width=.35, color=color,
                              hatch=hatch, edgecolor='white', linewidth=.7, label=label, zorder=3)
                ax.bar_label(bars, fmt='%.3f', fontsize=8, padding=3)
            ax.set(xticks=x, xticklabels=['Hello2', 'QA\ncanary', 'Arithmetic\ncanary', 'Chinese\ncanary'],
                   ylabel='Additive bytes (GiB)', ylim=(0, max(r['facade_h2d_gib'] for r in requests) * 1.24))
            ax.set_title('Paired request-period demand', loc='left')
            ax.legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(0, 1.20), ncol=2)
            bars = diff.bar(x, [r['h2d_minus_read_kib'] for r in requests], color=COLORS['h2d'], width=.55, zorder=3)
            diff.bar_label(bars, fmt='%.3f', fontsize=8, padding=3)
            diff.set(xticks=x, xticklabels=['Hello2', 'QA', 'Arith.', 'Chinese'], ylabel='H2D minus reader (KiB)',
                     ylim=(0, max(r['h2d_minus_read_kib'] for r in requests) * 1.25))
            diff.set_title('Small differences, resolved', loc='left')
            finish(fig, 'logical-transfer', 'GiB = 2^30 bytes; KiB = 2^10 bytes. Left labels round; right panel resolves the difference.\n'
                   'Logical reads are NOT physical I/O; facade H2D is NOT physical bus traffic. Startup I/O excluded.')

            fig = plt.figure(figsize=(10, 6.7))
            gs = fig.add_gridspec(2, 4, left=.07, right=.97, top=.81, bottom=.20, hspace=.78, wspace=.65)
            flow, balance = fig.add_subplot(gs[0, :2]), fig.add_subplot(gs[0, 2:])
            for sign, key, color, hatch in ((1, 'uploads', COLORS['h2d'], ''), (-1, 'evictions', COLORS['read'], '//')):
                values = [sign * r[key] for r in requests]
                bars = flow.bar(x, values, width=.56, color=color, edgecolor='white', hatch=hatch,
                                label='Uploads (+)' if sign == 1 else 'Evictions (-)', zorder=3)
                flow.bar_label(bars, labels=[f'{abs(v):,}' for v in values], padding=3, fontsize=8)
            flow.axhline(0, color=COLORS['ink'], linewidth=.7)
            flow.set(xticks=x, xticklabels=['Hello2', 'QA', 'Arith.', 'Chinese'], ylabel='Additive expert entries', ylim=(-4700, 5300))
            flow.set_title('Entry churn, not inferred misses', loc='left')
            flow.legend(frameon=False, ncol=2, fontsize=8, loc='upper left', bbox_to_anchor=(0, 1.32))
            cumulative = np.cumsum([r['net_entries'] for r in requests])
            balance.plot(x, cumulative, '--s', color=COLORS['h2d'], label='Cumulative uploads - evictions', markersize=8)
            balance.plot(x, [r['last_resident_entries'] for r in requests], 'o', color=COLORS['resident'],
                         markerfacecolor='white', markeredgewidth=1.8, label='Observed end-request entries', markersize=5)
            balance.set(xticks=x, xticklabels=['Hello2', 'QA', 'Arith.', 'Chinese'], ylabel='Entries (count)', ylim=(0, 95))
            balance.set_title(f"{data['totals']['uploads']:,} - {data['totals']['evictions']:,} = {data['reconciliation']['total_net_entries']}", loc='left')
            balance.text(.5, .24, 'Implied initial entries = 0\n(not an observed gauge)', transform=balance.transAxes,
                         ha='center', fontsize=8)
            balance.legend(frameon=False, fontsize=8, loc='upper left')
            for col, gauge in enumerate(data['gauges']):
                ax = fig.add_subplot(gs[1, col])
                last, maximum, ceiling = (None if v is None else units(v, gauge['divisor'])
                                          for v in (gauge['last'], gauge['maximum'], gauge['configured_limit']))
                color = COLORS['peak'] if col == 3 else COLORS['resident']
                if last is not None:
                    bars = ax.bar([0], [last], color=color, width=.45, label='Last', zorder=3)
                    ax.bar_label(bars, labels=[f'{last:.3f}' if col in (1, 3) else f'{last:g}'], padding=4, fontsize=8)
                if maximum is not None:
                    ax.plot([0], [maximum], 'D', color=COLORS['ink'], markerfacecolor='white', label='Observed max', zorder=4)
                if ceiling is not None:
                    ax.axhline(ceiling, color=COLORS['limit'], linestyle='--', label=gauge['configured_limit_label'])
                    ceiling_label = gauge['configured_limit_label'].replace(' ceiling', '\nceiling')
                    ax.text(.98, ceiling, f'{ceiling_label} {ceiling:g}', ha='right', va='bottom',
                            fontsize=7, transform=ax.get_yaxis_transform())
                else:
                    ax.text(.5, .16, 'Own ceiling\nnot recorded', ha='center', transform=ax.transAxes, fontsize=8)
                if gauge['key'] == 'cache.peak_bytes':
                    ax.text(.5, .17, 'Not owner physical memory', ha='center',
                            transform=ax.transAxes, fontsize=7)
                if last is None and maximum is None:
                    ax.text(.5, .65, 'Gauge unobserved', ha='center', transform=ax.transAxes, fontsize=8)
                extent = max((v for v in (last, maximum, ceiling) if v is not None), default=1)
                ax.set(xticks=[], ylabel=gauge['unit'], ylim=(0, max(extent, 1) * 1.30))
                ax.set_title(gauge['label'].replace('charged-byte peak', 'charged-byte\npeak'), loc='left')
            fig.text(.5, .14, 'Bar = last invocation; diamond = observed maximum; dashed line = the labeled accounting ceiling', ha='center', fontsize=8)
            finish(fig, 'cache-reconciliation', 'Cache peak: high-water of resident + pending + scratch; 320 MiB = cache accounting ceiling. MiB = 2^20 bytes.\n'
                   '209.574 MiB is not owner physical memory; owner-physical peak/ceiling unobserved. Final owner ledger remains unknown.')

            fig, (startup, latency) = plt.subplots(1, 2, figsize=(10, 4.9), gridspec_kw={'width_ratios': [1, 2.6]})
            fig.subplots_adjust(left=.08, right=.98, top=.80, bottom=.27, wspace=.60)
            bars = startup.bar([0], [data['startup_seconds']], color=COLORS['resident'], width=.45, zorder=3)
            startup.bar_label(bars, fmt='%.3f s', padding=5)
            startup.set(xticks=[0], xticklabels=['Launch to ready'], ylabel='Seconds', ylim=(0, 24))
            startup.set_title('Startup (separate origin)', loc='left')
            for offset, key, color, hatch, label in ((-.19, 'ttft_seconds', COLORS['read'], '', 'HTTP TTFT'),
                                                    (.19, 'e2e_seconds', COLORS['h2d'], '//', 'HTTP E2E')):
                bars = latency.bar(x + offset, [r[key] for r in requests], width=.35, color=color,
                                   edgecolor='white', hatch=hatch, label=label, zorder=3)
                latency.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)
            latency.set(xticks=x, xticklabels=['Hello2\n2 usage tokens', 'QA\n8 usage tokens',
                                             'Arithmetic\n8 usage tokens', 'Chinese\n8 usage tokens'],
                        ylabel='Seconds from HTTP send', ylim=(0, 23))
            latency.set_title('Individual request observations, no statistics', loc='left')
            latency.legend(frameon=False, ncol=2, loc='upper left', bbox_to_anchor=(0, 1.19), fontsize=8)
            finish(fig, 'timing', 'TTFT: send to first nonempty SSE content. E2E: send to EOF. DEBUG overhead unmeasured; no speedup claim.\n'
                   'Hello2: HTTP/usage only. Three canaries: text + usage checks only. Exact IDs and numeric acceptance unknown.')

            fig, (churn, occupancy) = plt.subplots(2, 1, figsize=(10, 5.8), sharex=True,
                                                 gridspec_kw={'height_ratios': [2.0, 1]})
            fig.subplots_adjust(left=.09, right=.97, top=.79, bottom=.21, hspace=.40)
            t = data['timeline']
            ordinals = [r['ordinal'] for r in t]
            churn.plot(ordinals, [r['uploads'] for r in t], '-o', color=COLORS['h2d'], markersize=5,
                       label='Uploads per invocation', linewidth=1.2)
            churn.plot(ordinals, [r['evictions'] for r in t], '--x', color=COLORS['read'], markersize=5,
                       label='Evictions per invocation', linewidth=1.2)
            churn.set(ylabel='Additive expert entries', ylim=(0, max(r['uploads'] for r in t) * 1.3))
            churn.set_title('Observed churn at anchored invocation boundaries', loc='left')
            churn.legend(frameon=False, ncol=2, loc='upper left', bbox_to_anchor=(0, 1.31), fontsize=8)
            occupancy.plot(ordinals, [r['observed_entries'] for r in t], 'o', color=COLORS['resident'],
                           markersize=3.5, label='Resident entries (completion observations)')
            occupancy.axhline(data['configured_limits']['entries'], color=COLORS['limit'], linestyle='--',
                              label='Entry ceiling = 64', zorder=0)
            occupancy.set(ylabel='Resident entries\n(gauge)', ylim=(0, 88), xlim=(.5, len(t) + .5),
                          xlabel='Invocation ordinal in log order (NOT time or token index)',
                          xticks=[1, 3, 4, 8, 12, 13, 17, 21, 22, 26, 30])
            occupancy.legend(frameon=False, fontsize=8, loc='lower right')
            start = 1
            for index, request in enumerate(requests):
                end = start + request['invocations'] - 1
                for ax in (churn, occupancy):
                    if index % 2 == 0:
                        ax.axvspan(start - .5, end + .5, color=COLORS['grid'], alpha=.28, zorder=-1)
                    if index:
                        ax.axvline(start - .5, color=COLORS['limit'], linewidth=.7, linestyle=':')
                churn.text((start + end) / 2, .97, request['label'], ha='center', va='top', fontsize=8,
                           transform=churn.get_xaxis_transform())
                start = end + 1
            finish(fig, 'invocation-timeline', '30 actual invocations; request mapping uses serialized log-byte anchors, not transaction IDs.\n'
                   'Points are observed completion gauges, not continuous occupancy. Uploads are NOT inferred cache misses.')

        previews = [Image.open(io.BytesIO(outputs[FIG + name + '.png'])).convert('RGB') for name in CAPTIONS]
        width = 1000
        thumbnails = [im.resize((width, round(im.height * width / im.width)), Image.Resampling.LANCZOS) for im in previews]
        sheet = Image.new('RGB', (width * 2, max(im.height for im in thumbnails[:2]) + max(im.height for im in thumbnails[2:])), 'white')
        for index, image in enumerate(thumbnails):
            sheet.paste(image, ((index % 2) * width, 0 if index < 2 else max(im.height for im in thumbnails[:2])))
        from PIL.PngImagePlugin import PngInfo
        info = PngInfo()
        info.add_text('Description', metadata)
        info.add_text('Software', GENERATOR)
        buffer = io.BytesIO()
        sheet.save(buffer, format='PNG', pnginfo=info)
        outputs[FIG + 'preview.png'] = buffer.getvalue()
        return outputs, renderer


def render(data):
    outputs, renderer = render_figures(data)
    outputs[OUT + 'plot-data.json'] = S.encode(data)
    outputs[OUT + 'PLOTS.md'] = plot_guide(data)
    manifest = dict(schema='storage_demand_plot_artifacts.v1', provenance=data['provenance'], renderer=renderer,
                    policy='No current timestamp; --check recomputes bytes with the recorded rendering toolchain',
                    artifacts={path: S.digest(content) for path, content in sorted(outputs.items())})
    outputs[OUT + 'plot-manifest.json'] = S.encode(manifest)
    return outputs


def check_outputs(root, outputs):
    return [path for path, content in outputs.items()
            if not (root / path).is_file() or (root / path).read_bytes() != content]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Re-render and compare all bytes without writing artifacts')
    args = parser.parse_args()
    try:
        summary, sources = checked_summary()
        data = build_data(summary, sources)
        outputs = render(data)
        stale = check_outputs(ROOT, outputs) if args.check else []
        if stale:
            print('STALE PLOT OUTPUT: ' + ', '.join(stale))
            return 1
        if not args.check:
            for path, content in outputs.items():
                if not (path.startswith(FIG) or path.startswith(OUT)) or not (ROOT / path).parent.is_dir():
                    raise S.SchemaError('output escaped the allowed existing plot directories')
            for path, content in outputs.items():
                (ROOT / path).write_bytes(content)
        print('PASS: 41 input hashes; 39 receipt pins; 30 anchored invocations; entry balance residuals = 0')
        print(('Checked' if args.check else 'Generated') + ': 4 PDFs, 4 PNGs, contact sheet, plot data, guide and hash manifest')
        return 0
    except (OSError, ValueError, KeyError, TypeError, IndexError, ImportError) as error:
        print('ERROR: ' + type(error).__name__ + ': ' + str(error))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
