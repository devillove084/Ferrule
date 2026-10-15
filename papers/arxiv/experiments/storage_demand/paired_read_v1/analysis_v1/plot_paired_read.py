#!/usr/bin/env python3
"""Public rendering by default; deprecated private derivation requires --private-inputs."""
import argparse
import os
import tempfile
import sys
from pathlib import Path


def captions(data):
    import plot_common as D
    a = data['paired']['aggregate']
    e, t, g = (a[k] for k in D.METRICS)
    return {
        'latency': (
            'Client latency and quality-qualified goodput in the sealed five-block paired-read study. '
            'Each connected arm value is the median of five measured-QA requests; the independent '
            'resampling unit is the paired block (n=5), not the 50 requests. B (blue circles) is '
            'blocking and L (teal squares) is lookahead. Rows retain the frozen execution orders '
            'P1 B/L, P2 B/L, P3 L/B, P4 L/B, P5 B/L. The log-scale ratio panel shows each block '
            '(circles B/L, triangles L/B), unity, and the aggregate exp(median log(L/B)) as a diamond. '
            f"E2E, TTFT, and goodput ratios are {e['ratio']:.5f}, {t['ratio']:.5f}, and {g['ratio']:.5f}. "
            'Gray strips and bracketed numbers are descriptive nearest-rank 2.5/97.5 percentile ranges '
            'from all 3,125 ordered paired-block bootstrap resamples (ranks 79 and 3047), not '
            'coverage-guaranteed confidence intervals or significance tests. Goodput is 40 actual '
            'quality-qualified completion tokens divided by the sum of five client E2E durations per arm. '
            'Startup launch-to-ready probe times are separate and excluded from request latency. '
            'This single instrumented study supports no repetition-based confidence, power, or SOTA claim.'),
        'read-wait': (
            'Read-demand wait and client-latency changes in the same five paired blocks. Each request '
            'wait is the sum of its anchored per-invocation host demand_wait_ns deltas, converted to '
            'seconds; each arm is the median of five measured-QA request waits. Connected points show '
            'blocking (blue circles) and lookahead (teal squares), with frozen execution order beside '
            'each block. The right panel plots lookahead-minus-blocking differences of the arm medians '
            'for request wait and client E2E; labels and circle/triangle markers retain order. '
            'All five blocks have lower observed wait and lower E2E with lookahead. This is an '
            'association consistent with less waiting, not proof that wait reduction caused the '
            'critical-path latency change. Host wait intervals are not CUDA timelines, are not additive '
            'to client latency, and do not establish GPU overlap. No fitted trend, causal coefficient, '
            'or extra independent replicates are inferred; shared-client/cache conditions remain a limitation.'),
        'accounting': (
            'Equal logical work and distinct memory/accounting scopes for all ten lifecycles. '
            'Each lifecycle records 128,811,263,616 logical checkpoint-reader bytes and 128,811,927,312 '
            'requested host-to-device bytes over its 12 requests: equal across modes, not equal to each other '
            '(H2D minus reader is 663,696 bytes). Every arm records 40,943 verified reads, zero wasted '
            'read bytes, and zero unknown read counts/bytes in the read-pipeline ledger. These '
            'application counters are not physical backend I/O or bus traffic. The owner-lifetime '
            'logical host-reservation maximum is 16,777,984 bytes (16.000732 MiB) against a 64 MiB '
            'logical limit; it is not RSS. Separately, the device-wide sampled maximum is 4,125 MiB '
            'against a configured 7 GiB target in every arm; it is neither an exact continuous peak '
            'nor exact process-owned GPU bytes. Bars are common observed maxima, never sums of '
            'lifetime peaks; do not add PSS, logical reservations, and device memory. Request-period '
            'shared NFS-client serverreadbytes are 314,825,320 for P1 blocking and recorded zero '
            'for the other nine arms. P1 blocking individual request windows sum to 312,732,264 bytes; the remaining 2,093,056 bytes lie between those windows within the enclosing shared-client phase. Fresh servers do not establish cold NFS/page caches, and '
            'shared-client counters do not identify physical NAS backend traffic. Startup/shutdown '
            'PID-I/O deltas remain unobserved (null), not zero. Zero final logical charges do not '
            'prove universal physical release. MiB and GiB denote 2^20 and 2^30 bytes.')}


def latex(text):
    return text.replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('%', r'\%').replace('&', r'\&').replace('^', r'\textasciicircum{}')


def guide(data):
    import plot_common as D
    cap = captions(data)
    aggregate = data['paired']['aggregate']
    labels = {'latency': 'f:paired-latency', 'read-wait': 'f:paired-read-wait', 'accounting': 'f:paired-accounting'}
    lines = ['# Paired-read figures: sealed mapfixed study', '',
             'Deprecated custodian reconstruction only. Historical figures, manuscript, production code, raw receipts, and '
             '`publication_results_v1` are untouched. No launch, GPU query, network access, harness '
             'import, or new dependency is used.', '', '## Exact captions and suggested LaTeX labels', '']
    for name, caption in cap.items():
        lines += [f'### {name}', '', f'PDF: `{D.FIG}{name}.pdf`', f'PNG: `{D.FIG}{name}.png`', '',
                  caption, '', '```latex', r'\begin{figure*}[t]', r'  \centering',
                  rf'  \includegraphics[width=\textwidth]{{figures/paired-read-{name}.pdf}}',
                  r'  \caption{' + latex(caption) + '}', r'  \label{' + labels[name] + '}',
                  r'\end{figure*}', '```', '']
    lines += ['## Estimator and denominators', '',
              '- All five paired blocks qualify; no failed/censored/unissued slot is dropped. '
              '120/120 requests completed across 10 fresh server lifecycles; 50 measured QA requests '
              'are nested within arms, not 50 independent replicates.',
              '- E2E and TTFT: median of five measured requests per arm. Per-block ratio is L/B; '
              'aggregate is `exp(median(log(L/B)))`. Goodput: 40 actual qualified tokens divided by '
              'the sum of five client E2E durations, followed by the same block estimator.',
              '- Exactly `5^5 = 3125` ordered resamples of five paired-block log-ratios; take each '
              'sample median, sort, take 1-based ranks 79 and 3047, exponentiate. These descriptive '
              '2.5/97.5 percentile ranges are NOT a coverage-guaranteed 95% confidence interval. '
              'No significance, power, SOTA, or 50-request bootstrap.', '',
              '| Outcome | Aggregate L/B | Descriptive range |', '|---|---:|---:|']
    for key, label in zip(D.METRICS, ('E2E', 'TTFT', 'Qualified goodput')):
        r = aggregate[key]
        lines.append(f"| {label} | {r['ratio']:.8f} | [{r['descriptive_range'][0]:.8f}, {r['descriptive_range'][1]:.8f}] |")
    lines += ['', '## Reading the plots without overclaiming', '',
              '- Palette: Blocking blue `#0072B2`, Lookahead teal `#008577`; circle/square endpoints, '
              'direct mode/order labels, and hatching retain meaning in grayscale. Ratio and change '
              'points encode order by circle (B/L) versus triangle (L/B), not mode. Neutral diamonds '
              'denote aggregate estimates, and neutral gray strips denote descriptive ranges.',
              '- Read-wait is the per-request sum of host pipeline deltas, then a within-arm median. '
              'The scatter compares differences of medians, not the median of requestwise differences '
              'and not a causal decomposition. E2E is send-to-EOF; TTFT is send-to-first nonempty SSE '
              'content. Neither is reconstructed from worker token counts.',
              '- Equal work is checked for each arm against the 102 actual invocation records and '
              '12 request windows. No route-ID signature or cache-miss equivalence is inferred. '
              'Source bytes and H2D differ by 663,696 bytes per lifecycle; neither counts physical storage traffic.',
              '- Host logical reservation and device samples use separate units/axes and scope labels. '
              'Lifetime peak fields are maxima, not additive deltas. PSS is deliberately not plotted or added. '
              'Final zero logical charges are not physical-release proof.',
              '- Shared NFS request-phase `serverreadbytes`: P1 blocking 314,825,320 bytes; others 0. '
              'The P1 blocking individual request windows sum to 312,732,264 bytes; 2,093,056 bytes lie between those windows inside the enclosing phase and are not assigned to requests. Startup and shutdown NFS phase windows are separately checked. Missing startup/shutdown '
              'PID-I/O and startup logical-reader accounting stay null. Physical backend bytes, owned '
              'GPU attribution, continuous device peak, and causal critical-path savings stay null.',
              '- Actual quality coverage: 900 worker-enqueued token IDs, joined through worker/HTTP IDs '
              'and ordinals; 110 exact eight-token oracle requests plus 10 two-token smoke requests '
              'validated for usage/length only. The 1,020 invocation rows are not tokens or replicates. '
              'SSE content events are not token boundaries.',
              '- The two earlier stopped studies remain excluded, with their original denominators '
              'retained in plot data. This analysis never imports their raw receipts or historical plot '
              'summaries. Actual source archive count is 1,484, cross-checked against the actual binding '
              'and build-review receipts, not copied from an older analysis.', '',
              '## Separate startup probe observations', '',
              'These are recorded launch-to-ready probe durations, **not** a subtraction from the '
              'coordinator/lifecycle start timestamp, not request latency, and not repeat-study statistics.', '',
              '| Arm | Startup probe (s) | Device samples |', '|---|---:|---:|']
    for arm in data['arms']:
        lines.append(f"| {arm['trial']} | {arm['startup_probe_seconds']:.6f} | {arm['device_sample_count']} |")
    lines += ['', '## Deprecated custodian-only reproduction',
              'Public users must use `render_public.py --output-dir DIR`; the following private-input reconstruction is not needed for public figures.', '',
              f"`inputs.json` binds {data['provenance']['input_count']} exact inputs with SHA-256; "
              f"{data['provenance']['checked_original_pin_count']} selected inputs also match original study/audit/preregistration pins. "
              'The validator recomputes from all ten raw server timelines, request/HTTP copies, '
              'resource samples and parsed mount-counter snapshots, checking against the final report '
              'and poststudy audit. This is not a rerun of every build/map/custody audit.',
              '`plot-data.json` contains only allowlisted derived fields, relative source paths, and '
              'hashes. No raw environment, credentials, worker launch specification, process map, '
              'private NAS export path, or raw receipt copy is published.',
              '`plot-manifest.json` binds every generated artifact, the input manifest, all generator '
              'and renderer source hashes, and the rendering toolchain/font hashes. PDF/PNG metadata '
              'binds input manifest, generator set, renderer source, renderer toolchain, and plot-data '
              'hashes. There is no current timestamp, hostname, mtime, or random identifier. PDF dates '
              'are omitted; DejaVu Sans normal/bold are pinned by hash. Byte reproducibility is scoped '
              'to that renderer/FreeType/font environment, not promised across arbitrary installations.', '',
              'From the repository root:', '', '```sh',
              'python3 -S -B -m unittest discover -s ' + D.ANALYSIS + 'tests -p test_paired_plots.py -v',
              '/home/admin/wxw403883/anaconda3/bin/python3.12 -B ' + D.ANALYSIS + 'plot_paired_read.py --private-inputs --output-dir /tmp/paired-read-private',
              '/home/admin/wxw403883/anaconda3/bin/python3.12 -B ' + D.ANALYSIS + 'plot_paired_read.py --private-inputs --output-dir /tmp/paired-read-private --check',
              '```', '',
              '`--check` rereads and validates sealed inputs, derives data, re-renders in memory, '
              'and compares every byte, including the manifest and guide. It never rewrites outputs '
              'or reseals inputs. A changed source or renderer intentionally makes committed outputs stale.',
              '`layout.json` records every visible text string and canvas-normalized bounding box, '
              'and fails if text extends outside the canvas or text bounding boxes overlap. It is not a mathematical proof of no marker/line '
              'overlap. `paired-read-preview.png` is a deterministic vertical contact sheet. Full-size '
              'PDFs/PNGs are intended for a two-column-width figure; do not shrink these to a narrow single column. Figure canvases are 7.5 inches wide; use approximately full text width.',
              '', '## Exact plot text', '',
              'The authoritative complete, generated text inventory is in `layout.json` '
              '(titles, panel labels, tick labels, values, legends, notes). Captions above are exact '
              'and generated from the same script, not inserted into the manuscript.', '']
    return '\n'.join(lines).encode()


def check_outputs(root, outputs):
    return sorted(path for path, content in outputs.items()
                  if not (root / path).is_file() or (root / path).read_bytes() != content)


def generate(sources, manifest, manifest_hash):
    import derive as D
    import study as S
    import render as R
    data = S.build_data(sources, manifest, manifest_hash)
    renderer = R.toolchain()
    data['provenance']['generator_sha256'] = D.digest(D.encode(data['provenance']['source_hashes']))
    data['provenance']['renderer_source_sha256'] = renderer['renderer_source_sha256']
    data['provenance']['renderer'] = renderer
    data['provenance']['renderer_toolchain_sha256'] = D.digest(D.encode(renderer))
    data['captions'] = captions(data)
    outputs, layout = R.render_figures(data, renderer)
    outputs[D.ANALYSIS + 'plot-data.json'] = D.encode(data)
    outputs[D.ANALYSIS + 'PLOTS.md'] = guide(data)
    outputs[D.ANALYSIS + 'layout.json'] = D.encode(layout)
    artifact_manifest = dict(schema='ferrule.paired-read.plot-manifest.v1',
        provenance=data['provenance'], renderer=renderer,
        artifacts={Path(path).name: D.digest(content) for path, content in sorted(outputs.items())},
        policy='No timestamp; check rederives and compares exact bytes; manifest excludes its own recursive hash')
    outputs[D.ANALYSIS + 'plot-manifest.json'] = D.encode(artifact_manifest)
    return outputs, data


def main():
    if '--private-inputs' not in sys.argv[1:]:
        from render_public import main as public_main
        return public_main()
    import derive as D
    print('DEPRECATED CUSTODIAN MODE: reads 840 private inputs; public users should run render_public.py', file=sys.stderr)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-inputs', action='store_true', help='Deprecated custodian-only reconstruction; never required for public figures')
    parser.add_argument('--output-dir', type=Path, required=True, help='Explicit private-render output directory, separate from public outputs')
    parser.add_argument('--check', action='store_true', help='Compare every regenerated byte; no writes or input resealing')
    args = parser.parse_args()
    try:
        sources, manifest, manifest_hash = D.load_sources()
        with tempfile.TemporaryDirectory(prefix='.paired-read-mpl-', dir=D.HERE) as cache:
            os.environ['MPLCONFIGDIR'] = cache
            outputs, data = generate(sources, manifest, manifest_hash)
        outputs = {Path(path).name: content for path, content in outputs.items()}
        if args.check:
            stale = check_outputs(args.output_dir, outputs)
            if stale:
                print('STALE: ' + ', '.join(stale))
                return 1
        else:
            D.require(args.output_dir.resolve() != D.HERE, 'private output must not replace public plot data')
            D.require(args.output_dir.resolve() != D.ROOT / 'papers/arxiv/figures', 'private output must not replace paper figures')
            public = D.HERE.parent / 'publication_results_v1'
            D.require(args.output_dir.resolve() != public and public not in args.output_dir.resolve().parents,
                      'private output must not modify frozen public archive')
            args.output_dir.mkdir(parents=True, exist_ok=True)
            for path, content in outputs.items():
                D.require(not (args.output_dir / path).is_symlink(), 'refusing symlink output')
                (args.output_dir / path).write_bytes(content)
        print(('CHECKED' if args.check else 'GENERATED') + ': 3 PDFs, 3 PNGs, contact sheet, data, guide, layout, manifest')
        print(f"PASS: {len(sources)} input hashes; 5 blocks; 120 requests; 900 actual tokens; 1020 anchored invocations")
        print('L/B ratios: ' + ', '.join(f"{k}={data['paired']['aggregate'][k]['ratio']:.8f}" for k in D.METRICS))
        return 0
    except D.EvidenceError as error:
        print('EVIDENCE ERROR: ' + str(error))
        return 1
    except (OSError, ValueError, KeyError, TypeError, IndexError, ImportError) as error:
        # Private raw data must never be interpolated into terminal diagnostics.
        print('ERROR: ' + type(error).__name__ + '; check required inputs and the pinned renderer environment')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
