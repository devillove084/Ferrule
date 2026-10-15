#!/usr/bin/env python3
"""One combined, deterministic CPU/Agg figure from the portable SQL projection."""
import argparse
import hashlib
import importlib.util
import io
import itertools
import json
from pathlib import Path
import sys
import zlib

HERE = Path(__file__).resolve().parent
DEFAULT_ARCHIVE = HERE.parent / 'experiments/storage_demand/profile_overlap_v1/publication_v1'
COLORS = {'read': '#0072B2', 'pread': '#E69F00', 'kernel': '#009E73',
          'blocking': '#0072B2', 'lookahead': '#CC79A7', 'ink': '#202832', 'grid': '#D9DDE3'}
WINDOW = 20000000


def checker(root):
    spec = importlib.util.spec_from_file_location('overlap_public_check', root / 'check.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def track_data(data):
    return {'read': [r for r in data['nvtx'] if r['kind'] == 'read_subset'],
            'pread': data['pread'], 'kernel': data['kernel']}


def zoom_bounds(data):
    reads = track_data(data)['read']
    # Empty input is a rendering test only, not an accepted actual-05 capture.
    start = min((r['start_ns'] for r in reads), default=0)
    return start, start + WINDOW


def clip(rows, start, end):
    return [(max(r['start_ns'], start), min(r['end_ns'], end)) for r in rows
            if r['start_ns'] < end and r['end_ns'] > start]


def toolchain(c):
    import matplotlib as mpl
    import numpy
    from matplotlib import font_manager, ft2font
    from PIL import __version__ as pillow
    fonts = {}
    for weight in ('normal', 'bold'):
        p = Path(font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans', weight=weight)))
        fonts[weight] = dict(name=p.name, sha256=c.digest(p.read_bytes()))
    return dict(python=sys.version.split()[0], matplotlib=mpl.__version__, numpy=numpy.__version__,
                pillow=pillow, freetype=ft2font.__freetype_version__, zlib=zlib.ZLIB_RUNTIME_VERSION,
                backend='Agg', font_family='DejaVu Sans', fonts=fonts, png_dpi=180, pdf_fonttype=42)


def build_figure(data, results):
    import matplotlib as mpl
    mpl.use('Agg', force=True)
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator

    style = {'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
             'axes.titleweight': 'bold', 'axes.spines.top': False, 'axes.spines.right': False,
             'axes.edgecolor': '#68727D', 'text.color': COLORS['ink'], 'axes.labelcolor': COLORS['ink'],
             'xtick.color': COLORS['ink'], 'ytick.color': COLORS['ink'], 'pdf.fonttype': 42,
             'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'axes.formatter.useoffset': False}
    with mpl.rc_context(style):
        fig = plt.figure(figsize=(9.0, 8.9))
        grid = fig.add_gridspec(3, 2, left=.185, right=.965, top=.82, bottom=.205,
                               height_ratios=[1, 1, .96], hspace=.91, wspace=.42)
        fig.suptitle('Read–compute overlap: scoped evidence from two fresh captures',
                     x=.5, y=.975, fontsize=12.2, fontweight='bold')
        read_count = len(track_data(data['blocking'])['read'])
        fig.text(.5, .943, f'One profiled QA per arm · same frozen CLI · {read_count:,} source-verified read operations per arm',
                 ha='center', fontsize=8.8)
        handles = [Line2D([], [], color=COLORS['read'], lw=5, marker='o', ms=4, label='NVTX read envelope'),
                   Line2D([], [], color=COLORS['pread'], lw=5, marker='s', ms=4, label='OSRT pread wall'),
                   Line2D([], [], color=COLORS['kernel'], lw=3, marker='|', ms=8, label='Actual expert GPU kernels')]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .921), ncol=3, frameon=False, fontsize=8)
        fig.text(.5, .875, 'Exact interval widths; symbols mark starts (not minimum durations). Zoom selection does not use overlap.',
                 ha='center', fontsize=8)
        counts = {}
        for i, arm in enumerate(('blocking', 'lookahead')):
            ax = fig.add_subplot(grid[i, :])
            start, end = zoom_bounds(data[arm])
            tracks = track_data(data[arm])
            counts[arm] = dict(zoom_start_ns=start, zoom_end_ns=end, tracks={})
            for kind, y, marker in [('read', 2, 'o'), ('pread', 1, 's'), ('kernel', 0, '|')]:
                selected = clip(tracks[kind], start, end)
                counts[arm]['tracks'][kind] = len(selected)
                spans = [((a - start) / 1e6, (b - a) / 1e6) for a, b in selected]
                ax.broken_barh(spans, (y - .16, .32), facecolors=COLORS[kind], linewidth=0, zorder=3)
                # Starts are explicitly glyphs: horizontal extents remain unmodified.
                starts = [(r['start_ns'] - start) / 1e6 for r in tracks[kind] if start <= r['start_ns'] < end]
                ax.plot(starts, [y] * len(starts), linestyle='None', marker=marker,
                        ms=2.7 if kind != 'kernel' else 7.0, markeredgewidth=.5,
                        color=COLORS[kind], zorder=4)
            for r in tracks['read']:
                if start <= r['start_ns'] < r['end_ns'] <= end:
                    ax.text(((r['start_ns'] + r['end_ns']) / 2 - start) / 1e6, 2.30,
                            str(r['operation']), ha='center', va='bottom', fontsize=7)
            ax.set_xlim(0, 20)
            ax.set_ylim(-.4, 2.87)
            ax.set_yticks([2, 1, 0], ['Worker read\nenvelope', 'Worker pread\nwall interval', 'Expert math\nGPU activity'])
            ax.tick_params(axis='y', length=0, labelsize=8)
            ax.xaxis.set_major_locator(FixedLocator([0, 5, 10, 15, 20]))
            ax.grid(axis='x', color=COLORS['grid'], lw=.6, zorder=0)
            ax.set_xlabel(f'Time after first read start (ms); request-relative offset = {start / 1e6:.6f} ms', fontsize=8)
            ax.set_title(f'({"ab"[i]}) {arm.capitalize()} · first 20 ms from operation 1 · numbers = source operation IDs',
                         loc='left', pad=8, fontsize=9.4)
        for i, (numerator, denominator, title) in enumerate([
            ('read_envelope_kernel_intersection_ns', 'worker_read_envelope_union_ns',
             '(c) Read-envelope ∩ GPU math'),
            ('pread_kernel_intersection_ns', 'worker_pread_union_ns', '(d) pread-wall ∩ GPU math')]):
            ax = fig.add_subplot(grid[2, i])
            xmax = 220 if i == 0 else 1.0
            for arm, y, marker in [('blocking', 1, 'o'), ('lookahead', 0, 's')]:
                value = results[arm][numerator] / 1e6
                den = results[arm][denominator]
                percent = 100 * results[arm][numerator] / den if den else None
                label = f'{value:.6f} ms ({percent:.5f}%)' if percent is not None else f'{value:.6f} ms (denominator zero)'
                if value == 0 and percent is not None:
                    label = '0 ms (0%)'
                ax.barh(y, value, height=.22, color=COLORS[arm], alpha=.85, zorder=3)
                ax.plot(value, y, marker=marker, color=COLORS[arm], ms=5, clip_on=False, zorder=4)
                ax.text(0, y + .23, label, va='bottom', fontsize=7.7)
            ax.set_xlim(0, xmax)
            ax.set_ylim(-.35, 1.65)
            ax.set_yticks([1, 0], ['Blocking', 'Lookahead'])
            ax.tick_params(axis='y', length=0, labelsize=8)
            ax.set_title(title, loc='left', fontsize=9.2, pad=9)
            ax.set_xlabel('Whole-request intersection (ms)', fontsize=8)
            ax.set_xticks([0, 50, 100, 150, 200] if i == 0 else [0, .25, .5, .75, 1])
            ax.grid(axis='x', color=COLORS['grid'], lw=.6, zorder=0)
        def seconds(key):
            return f"B {results['blocking'][key] / 1e9:.6f} s / L {results['lookahead'][key] / 1e9:.6f} s"

        fig.text(.185, .138, 'Percent denominators: union of the corresponding worker intervals, not request length or summed spans.\n'
                 f"Read envelope: {seconds('worker_read_envelope_union_ns')}.  pread wall: {seconds('worker_pread_union_ns')}.\n"
                 f"GPU union: {seconds('expert_kernel_union_ns')}; gaps, leading delay, upload and CPU submission are excluded.",
                 fontsize=8, linespacing=1.5, va='top')
        fig.text(.5, .045, 'SCOPED ACCEPTED  ·  FULL TRACE NOT ACCEPTED  ·  DROPPED-EVENT COUNT UNKNOWN',
                 ha='center', fontsize=8.3, fontweight='bold')
        fig.text(.5, .019, 'Three wrapper warnings per arm remain. No active-I/O, physical NAS-service, or ordinary-speed confidence claim.',
                 ha='center', fontsize=7.7)
        fig.canvas.draw()
    return fig, counts


def layout_check(fig, c):
    from matplotlib.text import Text
    renderer = fig.canvas.get_renderer()
    width, height = fig.bbox.width, fig.bbox.height
    records = []
    for a in fig.findobj(match=Text):
        if not a.get_visible() or not a.get_text():
            continue
        box = a.get_window_extent(renderer)
        c.require(box.x0 >= -1 and box.y0 >= -1 and box.x1 <= width + 1 and box.y1 <= height + 1,
                  'text outside canvas: ' + a.get_text())
        records.append(dict(text=a.get_text(), bounds=[round(box.x0 / width, 6), round(box.y0 / height, 6),
                                                      round(box.x1 / width, 6), round(box.y1 / height, 6)]))
    conflicts = []
    for a, b in itertools.combinations(records, 2):
        x, y, r, t = a['bounds']; u, v, rr, tt = b['bounds']
        if min(r, rr) - max(x, u) > .0005 and min(t, tt) - max(y, v) > .0005:
            conflicts.append([a['text'], b['text']])
    c.require(not conflicts, 'overlapping labels: ' + repr(conflicts))
    return dict(size_inches=list(fig.get_size_inches()), outside_canvas_text=[], overlapping_text=[], text=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument('--output-directory', type=Path, default=HERE)
    parser.add_argument('--check', action='store_true', help='Compare exact output bytes; pinned renderer required')
    args = parser.parse_args()
    c = checker(args.archive)
    manifest, _, data, results = c.load(args.archive)
    fig, counts = build_figure(data, results)
    renderer = toolchain(c)
    layout = layout_check(fig, c)
    hashes = dict(manifest_sha256=c.digest(c.encode(manifest)), renderer_source_sha256=c.digest(Path(__file__).read_bytes()),
                  toolchain_sha256=c.digest(c.encode(renderer)))
    outputs = {}
    for suffix in ('pdf', 'png'):
        buffer = io.BytesIO()
        metadata = (dict(Title='Scoped read compute overlap', Subject=json.dumps(hashes, sort_keys=True),
                         Creator='read-compute-overlap.py', CreationDate=None, ModDate=None)
                    if suffix == 'pdf' else dict(Software='read-compute-overlap.py', Description=json.dumps(hashes, sort_keys=True)))
        import matplotlib as mpl
        with mpl.rc_context({'pdf.fonttype': 42, 'savefig.facecolor': 'white'}):
            fig.savefig(buffer, format=suffix, dpi=180, metadata=metadata)
        outputs[args.output_directory / ('read-compute-overlap.' + suffix)] = buffer.getvalue()
    from matplotlib import pyplot as plt
    plt.close(fig)
    record = dict(schema='ferrule.overlap.figure.v1', hashes=hashes, toolchain=renderer, layout=layout, zooms=counts,
                  results=results, figures={p.name: dict(sha256=c.digest(b), bytes=len(b)) for p, b in outputs.items()})
    outputs[args.archive / 'figure.json'] = c.encode(record)
    for path, contents in outputs.items():
        if args.check:
            c.require(path.read_bytes() == contents, 'rendered output differs: ' + path.name)
        else:
            path.write_bytes(contents)
    print(json.dumps(dict(status='figure_bytes_verified' if args.check else 'rendered',
                          figures=record['figures'], layout='no clipped or overlapping text', zooms=counts), indent=2))


if __name__ == '__main__':
    main()
