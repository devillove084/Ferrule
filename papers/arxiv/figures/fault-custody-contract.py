#!/usr/bin/env python3
"""CPU/Agg companion figure; data verification uses only the Python stdlib.

Use the existing Matplotlib environment. No installation, GPU query or model run.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import sys
import zlib

HERE = Path(__file__).resolve().parent
DEFAULT_ARCHIVE = HERE.parent/'experiments/storage_demand/pressure_fault_v1/publication_fault_v1'
COLORS = dict(control='#246494', clean='#087F77', unknown='#AE493B', ink='#252B33', free='#ECEFF2', grid='#D7DCE1')
STEM = 'fault-custody-contract'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def renderer_metadata():
    import matplotlib as mpl
    import numpy
    from matplotlib import font_manager, ft2font
    from PIL import __version__ as pillow
    fonts = {}
    for weight in ('normal', 'bold'):
        p = Path(font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans', weight=weight)))
        fonts[weight] = dict(name=p.name, sha256=digest(p.read_bytes()))
    return dict(python=sys.version.split()[0], matplotlib=mpl.__version__, numpy=numpy.__version__,
                pillow=pillow, freetype=ft2font.__freetype_version__, zlib=zlib.ZLIB_RUNTIME_VERSION,
                backend='Agg', font_family='DejaVu Sans', fonts=fonts, png_dpi=180, pdf_fonttype=42)


def data_view(model):
    return {'schema': 'ferrule.fault-custody-figure-data.v1',
            'scope': model['scope'], 'denominator': model['denominator'],
            'cells': [{k: c[k] for k in ('case', 'schedule', 'snapshots', 'boundary', 'assertion_backed',
                                        'committed_content', 'host_read_statistics', 'local_final_release')}
                      for c in model['cells']]}


def build_figure(data):
    import matplotlib as mpl
    mpl.use('Agg', force=True)
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'text.color': COLORS['ink'],
                         'axes.labelcolor': COLORS['ink'], 'xtick.color': COLORS['ink'],
                         'ytick.color': COLORS['ink'], 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.edgecolor': '#68727D',
                         'pdf.fonttype': 42, 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    fig = plt.figure(figsize=(11.4, 9.6), dpi=180)
    fig.text(.055, .974, 'Completion evidence governs custody after a real routed-down fault',
             fontsize=15, fontweight='bold', va='top')
    fig.text(.055, .946, 'Mixed CUDA component · 4/4 cells PASS (2 selectors × 2 schedules) · width 16, GDN/GQA, 4 experts/top-3 · BF16; F32 SSM',
             fontsize=10.3, va='top')

    flow = fig.add_axes([.055, .642, .91, .255])
    flow.set(xlim=(0, 1), ylim=(0, 1))
    flow.axis('off')
    flow.text(0, 1.04, 'a  Transaction path  (logical stages, not elapsed time)', fontsize=12, fontweight='bold')

    def box(x, y, text, color, size=10.2):
        flow.text(x, y, text, ha='center', va='center', fontsize=size, linespacing=1.45,
                  bbox=dict(boxstyle='round,pad=0.55', fc='white', ec=color, lw=1.8))

    box(.095, .49, 'Tx1 published\nFrontier 1 · slot 0\nCommitted metadata', COLORS['control'])
    box(.377, .49, 'Tx2 continuation [2, 3]\nReal routed-down submit\nActual consumer wait succeeds', COLORS['control'], 10)
    box(.797, .82, 'CLEAN: owner proof retained\nAbort frees 512 B + 2 transient KV slots\nTx3 succeeds; then aborted, not published', COLORS['clean'], 10)
    box(.797, .18, 'UNKNOWN: owner proof discarded\n2 compute + 2 upload waits succeed*\nStill Unknown; publish / reset / shutdown rejected*', COLORS['unknown'], 9.6)
    for a, b, color, style in (((.197,.49),(.248,.49),COLORS['control'],'-'),
                              ((.517,.54),(.601,.82),COLORS['clean'],'-'),
                              ((.517,.44),(.591,.18),COLORS['unknown'],'--')):
        flow.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='-|>', color=color, lw=2.1, linestyle=style))
    flow.text(0, -.015, '* Source + action log + PASS back these outcomes; no individual fence/error receipts were serialized.',
              fontsize=9.4)

    state = fig.add_axes([.08, .30, .36, .255])
    state.set_title('b  Retained sequence-state bytes', loc='left', fontsize=12, fontweight='bold', pad=33)
    stages = ('before', 'prepared', 'after_failure', 'after_cleanup')
    clean, unknown = data['cells'][0], data['cells'][2]
    # Both schedules have identical state/custody trajectories, checked before rendering.
    for left, right in ((0,1), (2,3)):
        for stage in stages:
            assert data['cells'][left]['snapshots'][stage]['live_state_bytes'] == data['cells'][right]['snapshots'][stage]['live_state_bytes']
    xs = list(range(4))
    baseline = clean['snapshots']['before']['live_state_bytes']
    state.plot(xs, [baseline]*4, ':', color=COLORS['control'], lw=2.5, zorder=2)
    state.plot(xs, [unknown['snapshots'][s]['live_state_bytes'] for s in stages], '--^', color=COLORS['unknown'], lw=3.5, ms=8, zorder=3)
    state.plot(xs, [clean['snapshots'][s]['live_state_bytes'] for s in stages], '-o', color=COLORS['clean'], lw=1.9, ms=4, zorder=4)
    state.set(xlim=(-.13, 3.24), ylim=(0, 1820), ylabel='Bytes (default + committed + working)')
    state.set_xticks(xs, ['Frontier 1', 'Prepare tx2', 'Fault', 'Cleanup'])
    state.tick_params(axis='x', labelsize=9)
    state.set_yticks([0, 512, 1024, 1536])
    state.grid(axis='y', color=COLORS['grid'], lw=.6)
    state.text(3.08, 1590, '1,536', ha='right', fontsize=10, color=COLORS['unknown'])
    state.text(3.08, 835, '1,024', ha='right', fontsize=10, color=COLORS['clean'])
    state.legend(handles=[Line2D([],[],color=COLORS['control'],ls=':',lw=2,label='Baseline control'),
                          Line2D([],[],color=COLORS['clean'],ls='-',marker='o',lw=2,label='Clean'),
                          Line2D([],[],color=COLORS['unknown'],ls='--',marker='^',lw=2,label='Unknown')],
                 loc='lower left', bbox_to_anchor=(-.02,1.02), ncol=3, frameon=False,
                 fontsize=8.7, handlelength=1.5, columnspacing=.85, borderaxespad=0)
    fig.text(.08, .246, 'Clean: no active transaction; 31 free KV slots.\nClean KV / conv / SSM bitwise unchanged*.\nUnknown: 1 active transaction; 29 free slots.\nBoth schedules agree; committed metadata stays at frontier 1.',
             fontsize=9.1, va='top', linespacing=1.5)

    cache = fig.add_axes([.58, .30, .37, .255])
    cache.set_title('c  Cache budget after cleanup', loc='left', fontsize=12, fontweight='bold', pad=33)
    for y, cell in zip((3,2,1,0), data['cells']):
        v = cell['snapshots']['after_cleanup']['cache']
        is_unknown = cell['case'] == 'mixed_unknown'
        color = COLORS['unknown' if is_unknown else 'clean']
        cache.barh(y, v['charged'], height=.32, color=color, edgecolor=COLORS['ink'], lw=.6,
                   hatch='////' if is_unknown else None, zorder=3)
        cache.barh(y, v['free'], left=v['charged'], height=.32, color=COLORS['free'], edgecolor='#AAB3BC', lw=.6)
        cache.text(65536, y+.23, f"{v['free']:,} free + {v['charged']:,} charged", ha='right', va='bottom', fontsize=9.1)
    cache.set(xlim=(0,65536), ylim=(-.46,3.65), xlabel='Decoded-cache budget (KiB); 65,536 B per row')
    cache.xaxis.label.set_size(9.2)
    cache.set_xticks([0,16384,32768,49152,65536], ['0','16','32','48','64'])
    cache.set_yticks([3,2,1,0], ['Clean / B','Clean / L','Unknown / B','Unknown / L'])
    cache.tick_params(axis='y', labelsize=9)
    cache.grid(axis='x', color=COLORS['grid'], lw=.6, zorder=0)
    cache.legend(handles=[Patch(facecolor=COLORS['free'],edgecolor='#AAB3BC',label='Free'),
                          Patch(facecolor=COLORS['clean'],label='Charged'),
                          Patch(facecolor='white',edgecolor=COLORS['ink'],hatch='////',label='Unknown subset')],
                 loc='lower left', bbox_to_anchor=(-.02,1.02), ncol=3, frameon=False, fontsize=8.7,
                 handlelength=1.5, columnspacing=.85, borderaxespad=0)
    fig.text(.58, .246, 'Hatch: 6,918 B Unknown is a subset, never added again.\nUnknown = 774 pending + 6,144 scratch; clean = 4,096 scratch.\nB = Blocking; L = Lookahead. Free + charged = 65,536 B.',
             fontsize=9.1, va='top', linespacing=1.5)

    fig.text(.055, .165, 'Host reads close independently: charge = 0, host Unknown = 0 in all four cells.\nLookahead wastes 1 read (774 B) per cell; Blocking wastes 0. Host cleanup does not release GPU custody.',
             fontsize=9.8, linespacing=1.5, va='top', color=COLORS['control'])
    fig.text(.055, .111, 'Post-Unknown KV / SSM CONTENTS UNOBSERVED — only committed metadata is unchanged.',
             fontsize=10.5, fontweight='bold', va='top', color=COLORS['unknown'])
    fig.text(.055, .081, 'The 318,720 B recurrent workspace is a separate, overlapping owner view: no additive physical-memory total.\nClean-only explicit owner drop + allocator trim returns fixture requested bytes 0 → 0; no device-wide release certificate.\nComponent contract only: not integrated broker / CLI E5, a 35B model fault test, or physical driver proof loss.',
             fontsize=9.2, linespacing=1.5, va='top')
    return fig


def layout_audit(fig):
    from matplotlib.text import Text
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    labels = []
    for item in fig.findobj(match=Text):
        if not item.get_visible() or not item.get_text().strip():
            continue
        b = item.get_window_extent(renderer)
        if b.width == 0 or b.height == 0:
            continue
        assert b.x0 >= 0 and b.y0 >= 0 and b.x1 <= width and b.y1 <= height, 'text outside canvas'
        labels.append((item.get_text(), b))
    collisions = []
    for (ta,a), (tb,b) in itertools.combinations(labels, 2):
        if min(a.x1,b.x1)-max(a.x0,b.x0) > 1 and min(a.y1,b.y1)-max(a.y0,b.y0) > 1:
            collisions.append((ta,tb))
    if collisions:
        raise ValueError('Overlapping text: '+repr(collisions))
    return dict(text_count=len(labels), overlapping_text_pairs=0, all_text_inside_canvas=True,
                figure_inches=list(fig.get_size_inches()), png_pixels=[width,height],
                labels=[dict(text=t, bounds=[round(b.x0/width,6),round(b.y0/height,6),
                                             round(b.x1/width,6),round(b.y1/height,6)]) for t,b in labels])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument('--output-dir', type=Path, default=HERE)
    parser.add_argument('--check', action='store_true', help='re-render in memory and verify pinned bytes; no output writes')
    args = parser.parse_args()
    root = args.archive.resolve()
    sys.path.insert(0, str(root))
    import check
    import evidence
    check.check(root, include_figure=False)
    data = data_view(evidence.build_model(root))
    os.environ.setdefault('MPLCONFIGDIR', str(root/'.cache'))
    fig = build_figure(data)
    layout = layout_audit(fig)
    artifacts = {}
    for ext in ('pdf','png'):
        buffer = io.BytesIO()
        metadata = ({'Creator': 'Ferrule fault-custody-contract', 'CreationDate': None, 'ModDate': None}
                    if ext == 'pdf' else {'Software': 'Ferrule fault-custody-contract'})
        fig.savefig(buffer, format=ext, dpi=180, metadata=metadata)
        artifacts[STEM+'.'+ext] = buffer.getvalue()
    record = dict(schema='ferrule.fault-custody-figure.v1', renderer=renderer_metadata(), colors=COLORS,
                  source_sha256=digest(Path(__file__).read_bytes()),
                  publication_manifest_sha256=evidence.sha256(root/'manifest.json'),
                  data_sha256=digest(evidence.encode(data)), data=data, layout=layout,
                  outputs={name:dict(sha256=digest(raw),bytes=len(raw)) for name,raw in artifacts.items()})
    if args.check:
        evidence.equal(evidence.load(root/'figure.json'), record, 'figure metadata/data drift')
        for name, raw in artifacts.items():
            evidence.equal((args.output_dir/name).read_bytes().hex(), raw.hex(), 'figure image drift')
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, raw in artifacts.items():
            (args.output_dir/name).write_bytes(raw)
        (root/'figure.json').write_bytes(evidence.encode(record))
    print(json.dumps({'status':'PASS', 'mode':'check' if args.check else 'render',
                      'text_overlap_pairs':0, 'outputs':record['outputs'], 'renderer':record['renderer']}, indent=2))


if __name__ == '__main__':
    main()
