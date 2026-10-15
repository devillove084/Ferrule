"""CPU-only deterministic Matplotlib/Agg rendering, with text-bound layout checks."""
import io
import itertools
from importlib.metadata import version
import json
import math
import sys
import zlib
from pathlib import Path
from plot_common import ANALYSIS, FIG, MIB, GIB, MODES, digest, encode, require

COLORS = dict(blocking='#0072B2', lookahead='#008577', ink='#202832', gray='#68727D', grid='#D9DDE3')
MARKERS = dict(blocking='o', lookahead='s')
NAMES = ('latency', 'read-wait', 'accounting')


def font_paths():
    import matplotlib as mpl
    directory = Path(mpl.get_data_path()) / 'fonts' / 'ttf'
    return {'normal': directory / 'DejaVuSans.ttf', 'bold': directory / 'DejaVuSans-Bold.ttf'}


def toolchain():
    import matplotlib as mpl
    mpl.use('Agg', force=True)
    import numpy as np
    from matplotlib import ft2font
    from PIL import __version__ as pillow_version
    fonts = {}
    for weight in ('normal', 'bold'):
        path = font_paths()[weight]
        fonts[weight] = dict(name=path.name, sha256=digest(path.read_bytes()))
    return dict(python=sys.version.split()[0], zlib_build=zlib.ZLIB_VERSION, zlib_runtime=zlib.ZLIB_RUNTIME_VERSION,
                matplotlib=mpl.__version__, numpy=np.__version__, pillow=pillow_version,
                freetype=ft2font.__freetype_version__, font_family='DejaVu Sans', fonts=fonts,
                backend='Agg', png_dpi=180, pdf_fonttype=42,
                dependencies={name: version(name) for name in ('matplotlib', 'numpy', 'pillow', 'contourpy',
                              'cycler', 'fonttools', 'kiwisolver', 'packaging', 'pyparsing', 'python-dateutil', 'six')},
                renderer_source_sha256=digest((Path(__file__)).read_bytes()))


def render_figures(data, renderer, *, creator=ANALYSIS + 'plot_paired_read.py'):
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    from matplotlib.text import Text
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    hashes = dict(input_manifest_path=data['provenance']['input_manifest_path'],
                  input_manifest_sha256=data['provenance']['input_manifest_sha256'],
                  generator_sha256=digest(encode(data['provenance']['source_hashes'])),
                  renderer_source_sha256=renderer['renderer_source_sha256'],
                  renderer_toolchain_sha256=digest(encode(renderer)), plot_data_sha256=digest(encode(data)))
    metadata = json.dumps(hashes, sort_keys=True)
    style = {'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
             'axes.titleweight': 'bold', 'axes.labelsize': 9, 'axes.edgecolor': COLORS['gray'],
             'axes.spines.top': False, 'axes.spines.right': False, 'text.color': COLORS['ink'],
             'axes.labelcolor': COLORS['ink'], 'xtick.color': COLORS['ink'], 'ytick.color': COLORS['ink'],
             'pdf.fonttype': 42, 'savefig.facecolor': 'white', 'figure.facecolor': 'white',
             'axes.formatter.useoffset': False, 'axes.unicode_minus': True}
    outputs, layouts = {}, {}
    by_trial = {r['trial']: r for r in data['arms']}
    order = data['design']['order']
    pair_labels = [r['block'] + '  ' + ('B/L' if r['modes'][0] == 'blocking' else 'L/B') for r in order]
    handles = [Line2D([], [], color=COLORS[m], marker=MARKERS[m], linestyle='None',
                      markersize=6, label=m.capitalize() + (' (B)' if m == 'blocking' else ' (L)')) for m in MODES]

    def title(fig, text):
        fig.suptitle(text, x=.5, y=.975, fontsize=13, fontweight='bold')
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .94), ncol=2, frameon=False)

    def paired(ax, getter, xlabel, panel_title):
        for i, block in enumerate(order):
            y = 4 - i
            values = [getter(by_trial[block['block'] + '-' + m]) for m in MODES]
            ax.plot(values, [y, y], color=COLORS['gray'], lw=1.15, zorder=2)
            for mode, value in zip(MODES, values):
                ax.plot(value, y, marker=MARKERS[mode], color=COLORS[mode], markersize=6.5, linestyle='None', zorder=3)
        ax.set_yticks(range(5), pair_labels[::-1])
        ax.set_ylim(-.55, 4.55)
        ax.set_xlabel(xlabel)
        ax.set_title(panel_title, loc='left', pad=9)
        ax.grid(axis='x', color=COLORS['grid'], lw=.6)
        ax.set_axisbelow(True)

    def finish(fig, name, note):
        fig.text(.5, .025, note, ha='center', va='bottom', fontsize=7.4, linespacing=1.5)
        # Bind each text artist to the hashed bundled fonts, not a host font lookup.
        for artist in fig.findobj(match=Text):
            properties = artist.get_fontproperties().copy()
            weight = properties.get_weight()
            bold = weight in ('bold', 'heavy', 'black') or isinstance(weight, (int, float)) and weight >= 700
            properties.set_file(str(font_paths()['bold' if bold else 'normal']))
            artist.set_fontproperties(properties)
        fig.canvas.draw()
        canvas = fig.canvas.get_renderer()
        width, height = fig.bbox.width, fig.bbox.height
        texts, outside = [], []
        omitted_ticks = set()
        for ax in fig.axes:
            for axis in (ax.xaxis, ax.yaxis):
                low, high = sorted(axis.get_view_interval())
                for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                    if not low <= tick.get_loc() <= high:
                        omitted_ticks.update((tick.label1, tick.label2))
        for artist in fig.findobj(match=Text):
            if artist in omitted_ticks or not artist.get_visible() or not artist.get_text():
                continue
            box = artist.get_window_extent(canvas)
            record = dict(text=artist.get_text(), bbox_fraction=[round(box.x0 / width, 6),
                          round(box.y0 / height, 6), round(box.x1 / width, 6), round(box.y1 / height, 6)])
            texts.append(record)
            if box.width > 0 and box.height > 0 and (box.x0 < -1 or box.y0 < -1 or box.x1 > width + 1 or box.y1 > height + 1):
                outside.append(artist.get_text())
        require(not outside, 'figure text extends beyond canvas: ' + name + ': ' + ', '.join(outside))
        overlaps = []
        for a, b in itertools.combinations(texts, 2):
            x, y, right, top = a['bbox_fraction']
            u, v, other_right, other_top = b['bbox_fraction']
            if min(right, other_right) - max(x, u) > .0005 and min(top, other_top) - max(y, v) > .0005:
                overlaps.append([a['text'], b['text']])
        require(not overlaps, 'overlapping text boxes in ' + name + ': ' + str(overlaps))
        layouts[name] = dict(size_inches=list(fig.get_size_inches()), outside_canvas_text=[],
                             text_bbox_overlaps=[], text_artists=texts,
                             validation='Visible text inside canvas with no pairwise text-box overlaps; marker/line intersections require visual review')
        for suffix in ('pdf', 'png'):
            buffer = io.BytesIO()
            meta = (dict(Title='Paired read: ' + name, Subject=metadata,
                         Creator=creator, CreationDate=None, ModDate=None)
                    if suffix == 'pdf' else dict(Software=creator, Description=metadata))
            fig.savefig(buffer, format=suffix, dpi=renderer['png_dpi'], metadata=meta)
            outputs[FIG + name + '.' + suffix] = buffer.getvalue()
        plt.close(fig)

    with mpl.rc_context({**mpl.rcParamsDefault, **style, 'backend': 'Agg'}):
        fig = plt.figure(figsize=(7.5, 5.8))
        grid = fig.add_gridspec(2, 2, left=.105, right=.97, top=.81, bottom=.23,
                                wspace=.36, hspace=.62, width_ratios=[1.04, 1.0])
        e2e, ttft, ratio = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[:, 1])
        title(fig, 'Five paired blocks: client latency and qualified goodput')
        paired(e2e, lambda a: a['metrics']['request_e2e_seconds'], 'Median client E2E (s)', '(a) E2E: median of five measured QA')
        paired(ttft, lambda a: a['metrics']['ttft_seconds'], 'Median client TTFT (s)', '(b) TTFT: median of five measured QA')
        e2e.set_xlim(6.7, 8.15)
        ttft.set_xlim(1.88, 2.4)
        ratio.set_title('(c) Paired L/B ratios', loc='left', pad=9)
        ratio.set_xscale('log')
        ticks = [.85, .9, 1., 1.1, 1.2]
        ratio.xaxis.set_major_locator(FixedLocator(ticks))
        ratio.xaxis.set_major_formatter(FixedFormatter([f'{v:g}' for v in ticks]))
        ratio.xaxis.set_minor_locator(NullLocator())
        ratio.set_xlim(.82, 1.225)
        ratio.set_ylim(-.57, 2.85)
        ratio.axvline(1, color=COLORS['gray'], lw=1, ls='--')
        ratio.grid(axis='x', color=COLORS['grid'], lw=.6)
        ratio.set_yticks([2.35, 1.15, -.05], ['E2E\nlower better', 'TTFT\nlower better', 'Goodput\nhigher better'])
        ratio.tick_params(axis='y', length=0, labelsize=8)
        metrics = ('request_e2e_seconds', 'ttft_seconds', 'qualified_goodput_tokens_per_second')
        for y, metric in zip((2.45, 1.25, .05), metrics):
            for i, block in enumerate(data['paired']['blocks']):
                value = math.exp(block['log_ratios'][metric])
                py = y + (.24 - i * .115)
                marker = 'o' if block['execution_order'][0] == 'blocking' else '^'
                ratio.plot(value, py, marker=marker, color=COLORS['lookahead'], ms=5.5, linestyle='None', zorder=3)
                ratio.annotate(block['block'], (value, py), xytext=(5, 0), textcoords='offset points', va='center', fontsize=7.4)
            estimate = data['paired']['aggregate'][metric]
            lo, hi = estimate['descriptive_range']
            ratio.fill_betweenx([y - .4, y - .32], lo, hi, color=COLORS['gray'], alpha=.28, linewidth=0)
            ratio.plot(estimate['ratio'], y - .36, marker='D', color=COLORS['ink'], ms=5, linestyle='None')
            ratio.text(.015 if metric == 'qualified_goodput_tokens_per_second' else .985, y - .36, f"{estimate['ratio']:.5f}  [{lo:.5f}, {hi:.5f}]",
                       transform=ratio.get_yaxis_transform(), ha='left' if metric == 'qualified_goodput_tokens_per_second' else 'right', va='center', fontsize=7.1,
                       bbox=dict(facecolor='white', alpha=.8, edgecolor='none', pad=1))
        ratio.set_xlabel('Lookahead / blocking (log scale)\nDashed line = unity')
        finish(fig, 'latency', 'n = 5 paired blocks, not 50 independent requests. Rows show block and frozen execution order (B/L or L/B).\n'
               'Ratio points: P1–P5 (circle B/L; triangle L/B). Diamond: exp(median log ratio). Gray strip: descriptive 2.5/97.5 percentiles\n'
               'from all 3,125 paired-block resamples; no confidence-coverage or significance claim. Startup probe time is separate.')

        fig, (wait, change) = plt.subplots(1, 2, figsize=(7.5, 4.6), gridspec_kw={'width_ratios': [1, 1.15]})
        fig.subplots_adjust(left=.105, right=.97, top=.76, bottom=.28, wspace=.40)
        title(fig, 'Lower read-demand wait accompanies lower measured E2E')
        paired(wait, lambda a: a['median_request_demand_wait_seconds'], 'Median request read-demand wait (s)',
               '(a) Median request read-demand wait')
        wait.set_xlim(2.35, 3.4)
        change.set_title('(b) Paired change: L − B', loc='left', pad=9)
        change.axvline(0, color=COLORS['gray'], lw=1, ls='--')
        change.axhline(0, color=COLORS['gray'], lw=1, ls='--')
        for r in data['wait_latency_changes']:
            reverse = r['execution_order'][0] == 'lookahead'
            change.plot(r['wait_change_seconds'], r['e2e_change_seconds'], '^' if reverse else 'o',
                        color=COLORS['lookahead'], ms=7)
            offset = {'P3': (-8, 8), 'P5': (7, -8)}.get(r['block'], (7, 8))
            change.annotate(r['block'] + (' L/B' if reverse else ' B/L'),
                            (r['wait_change_seconds'], r['e2e_change_seconds']), xytext=offset,
                            textcoords='offset points', ha='right' if r['block'] == 'P3' else 'left', fontsize=8)
        change.set_xlim(-.81, .06)
        change.set_ylim(-.96, .10)
        change.set_xlabel('Change in median request\nread-demand wait (s)')
        change.set_ylabel('Change in median client E2E (s)')
        change.grid(color=COLORS['grid'], lw=.6)
        change.set_axisbelow(True)
        finish(fig, 'read-wait', 'Each point is one paired block; each arm is the median of five measured-QA requests. No fit or causal coefficient.\n'
               'Host demand-wait intervals are not CUDA overlap timelines and are not additive to client latency.\n'
               'Association is consistent with less waiting, but does not establish critical-path causality; shared-cache conditions remain.')

        fig = plt.figure(figsize=(7.5, 5.8))
        grid = fig.add_gridspec(2, 3, left=.075, right=.975, top=.80, bottom=.25,
                                height_ratios=[.85, 1.3], hspace=.75, wspace=.48)
        table = fig.add_subplot(grid[0, :])
        host, device, nfs = (fig.add_subplot(grid[1, i]) for i in range(3))
        title(fig, 'Equal logical work; separate host and device scopes')
        table.axis('off')
        table.set_title('(a) Request-period totals per lifecycle: identical in all five blocks', loc='left', pad=9)
        table.text(.02, .91, 'Application / read-pipeline counter', weight='bold', transform=table.transAxes)
        table.text(.51, .91, 'Blocking', color=COLORS['blocking'], weight='bold', transform=table.transAxes)
        table.text(.77, .91, 'Lookahead', color=COLORS['lookahead'], weight='bold', transform=table.transAxes)
        sample = data['arms'][0]
        rows = [('Logical checkpoint-reader bytes', f"{sample['logical_counters']['checkpoint_read_bytes']:,}"),
                ('Requested host-to-device\nbytes', f"{sample['logical_counters']['h2d_bytes']:,}"),
                ('Verified reads / recorded wasted bytes', f"{sample['pipeline_stats']['verified_reads']:,} / {sample['pipeline_stats']['wasted_read_bytes']}"),
                ('Unknown reads / unknown bytes', f"{sample['pipeline_stats']['unknown_reads']} / {sample['pipeline_stats']['unknown_bytes']}")]
        for i, (label, value) in enumerate(rows):
            y = (.74, .46, .18, -.06)[i]
            table.text(.02, y, label, transform=table.transAxes, fontsize=9, linespacing=.95, va='center')
            table.text(.51, y, value, transform=table.transAxes, fontsize=9, va='center')
            table.text(.77, y, value, transform=table.transAxes, fontsize=9, va='center')
        for ax, value, limit, heading, limit_label, scope in (
            (host, sample['host_logical_peak_bytes'] / MIB, sample['host_logical_limit_bytes'] / MIB,
             '(b) Logical host\nreservation', '64 MiB limit', 'Not RSS or physical host usage'),
            (device, sample['sampled_device_max_bytes'] / MIB, sample['configured_device_target_bytes'] / MIB,
             '(c) Device-wide\nsampled maximum', '7 GiB target', 'Not exact process-owned GPU bytes')):
            for y, mode in enumerate(MODES[::-1]):
                ax.barh(y, value, height=.35, color=COLORS[mode], edgecolor=COLORS['ink'],
                        linewidth=.5, hatch='//' if mode == 'lookahead' else None, zorder=3)
                ax.text(value + limit * .035, y, f'{value:.6f}' if ax is host else f'{value:,.0f}', va='center', fontsize=8.3)
            ax.axvline(limit, color=COLORS['gray'], lw=1, ls='--')
            ax.text(limit, 1.53, limit_label, ha='right', fontsize=8)
            ax.set_yticks([0, 1], ['L', 'B'])
            ax.set_ylim(-.7, 1.8)
            ax.set_xlim(0, limit * 1.08)
            ax.set_xlabel('MiB (2^20 bytes)')
            ax.set_title(heading, loc='left', pad=9, fontsize=9.5)
            ax.grid(axis='x', color=COLORS['grid'], lw=.6)
            ax.text(.5, .02, scope, ha='center', va='bottom', transform=ax.transAxes, fontsize=7.1)
        host.set_xticks([0, 16, 32, 48, 64])
        device.set_xticks([0, 4096, 7168], ['0', '4,096', '7,168'])
        device.tick_params(axis='x', labelsize=7.4)
        nfs.set_title('(d) Shared NFS client\nserver-read bytes', loc='left', pad=9, fontsize=9.3)
        for i, arm in enumerate(data['arms']):
            value = arm['nfs_phase_client_serverreadbytes']['requests_nfs'] / MIB
            nfs.plot(value, 9 - i, marker=MARKERS[arm['mode']], color=COLORS[arm['mode']], ms=4.5, linestyle='None')
        nfs.set_yticks(range(10), [r['trial'].replace('blocking', 'B').replace('lookahead', 'L') for r in data['arms']][::-1], fontsize=7.3)
        nfs.set_ylim(-.7, 10)
        nfs.set_xlim(-18, 345)
        nfs.set_xticks([0, 100, 200, 300])
        nfs.grid(axis='x', color=COLORS['grid'], lw=.6)
        nfs.set_xlabel('Request-period client delta (MiB)')
        nfs.text(.94, .79, 'P1-B: 314,825,320 B\nOther nine arms: 0 B', ha='right', va='top', transform=nfs.transAxes, fontsize=7.5)
        finish(fig, 'accounting', 'Bars: the same maximum in every arm, not a sum of lifetime peaks. Host logical peak = 16,777,984 B; device maximum = 4,125 MiB.\n'
               'Reader and H2D totals are equal across modes, not to one another. Recorded zero waste is not unknown physical backend traffic.\n'
               'Do not add host PSS to device or logical gauges. Shared NFS/page cache may be warm; startup/shutdown PID-I/O deltas remain unknown.')

    previews = [Image.open(io.BytesIO(outputs[FIG + name + '.png'])).convert('RGB') for name in NAMES]
    width = 1400
    thumbs = [im.resize((width, round(im.height * width / im.width)), Image.Resampling.LANCZOS) for im in previews]
    sheet = Image.new('RGB', (width, sum(im.height for im in thumbs)), 'white')
    offset = 0
    for image in thumbs:
        sheet.paste(image, (0, offset))
        offset += image.height
    info = PngInfo()
    info.add_text('Description', metadata)
    info.add_text('Software', creator)
    buffer = io.BytesIO()
    sheet.save(buffer, format='PNG', pnginfo=info)
    outputs[FIG + 'preview.png'] = buffer.getvalue()
    return outputs, layouts
