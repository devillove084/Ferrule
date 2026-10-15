"""Public-only CPU regressions. Detached exports contain no private tree or old plot data."""
from __future__ import annotations

import copy
import gzip
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import render_public as P


def load(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def projection(root, name):
    return load(root / name)['data']


def export_public(destination):
    shutil.copytree(P.PUBLIC, destination / 'publication_results_v1')
    analysis = destination / 'analysis_v1'
    analysis.mkdir()
    for name in P.RENDER_SOURCES + ('plot_paired_read.py',):
        shutil.copyfile(HERE / name, analysis / name)
    return analysis


class PublicDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='paired-data-with_private_tree_absent-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.export = Path(cls.temporary.name)
        export_public(cls.export)
        cls.root = cls.export / 'publication_results_v1'
        cls.data = P.build_data(cls.root)
        cls.original_manifest = (cls.root / 'manifest.json').read_bytes()

    def test_public_inventory_and_denominators(self):
        self.assertFalse((self.export / 'artifacts-v3-mapfixed-20261015-01').exists())
        self.assertFalse((self.export / 'analysis_v1' / 'derive.py').exists())
        self.assertFalse((self.export / 'analysis_v1' / 'plot-data.json').exists())
        inventory = self.data['provenance']['verified_public_inventory']
        manifest = load(self.root / 'manifest.json')
        self.assertEqual(inventory['files'], len(manifest['files']) + 1)
        self.assertEqual(inventory['files'], 623)
        self.assertEqual(inventory['uncompressed_member_bytes'], sum(v['uncompressed_bytes'] for v in manifest['files'].values()))
        self.assertEqual(inventory['original_raw_files'], 1461)
        self.assertEqual(self.data['denominators'], dict(scheduled=120, issued=120, terminal=120,
                                                       successful=120, failed=0, censored=0, unissued=0))
        self.assertEqual(self.data['coverage']['actual_worker_generated_tokens'], 900)
        self.assertEqual(self.data['coverage']['pipeline_invocation_jsons'], 1020)
        self.assertEqual(self.data['coverage']['exact_eight_token_oracles'], 110)
        self.assertEqual(self.data['coverage']['length_only_two_token_smoke'], 10)
        self.assertEqual(self.data['coverage']['measured_qa'], 50)
        self.assertTrue(all(not h['pooled'] and h['unissued_requests'] == 120 for h in self.data['excluded_historical_attempts']))
        self.assertEqual((self.root / 'manifest.json').read_bytes(), self.original_manifest)

    def test_ten_arm_values_and_wait_medians_from_public_request_events(self):
        e2e = [7.760480474680662, 7.228216227144003, 7.732933735474944, 6.905777765437961,
               7.210535394027829, 7.594348918646574, 7.163044951856136, 7.965932961553335,
               7.709717178717256, 7.0793295204639435]
        waits = [3.033446574, 2.607421743, 3.005097676, 2.494986122, 2.592377345,
                 2.990115408, 2.543655661, 3.246461408, 3.08228542, 2.582914561]
        for arm, expected_e2e, expected_wait in zip(self.data['arms'], e2e, waits):
            with self.subTest(trial=arm['trial']):
                prefix = 'arms/' + arm['trial'] + '/'
                events = [json.loads(line) for line in gzip.decompress((self.root / (prefix + 'server-events.jsonl.gz')).read_bytes()).splitlines()]
                durations, ttfts, request_waits, tokens = [], [], [], []
                for i in range(1, 6):
                    request = projection(self.root, prefix + f'qa-measured-{i}/observed.json.gz')
                    actual = request['actual']
                    durations.append(actual['ended_at'] - actual['sent_at'])
                    ttfts.append(actual['content_event_times'][0] - actual['sent_at'])
                    tokens.append(actual['usage']['completion_tokens'])
                    window = [e['record']['fields'] for e in events
                              if request['log_start_byte'] <= e['source_byte_start']
                              and e['source_byte_end'] <= request['log_end_byte']
                              and e['record']['fields']['event'] == 'mixed_decoder_invocation']
                    self.assertEqual(len(window), 9)
                    request_waits.append(sum(json.loads(f['expert_read_pipeline_delta'])['demand_wait_ns'] for f in window) / 1e9)
                self.assertEqual(arm['metrics']['request_e2e_seconds'], statistics.median(durations))
                self.assertEqual(arm['metrics']['request_e2e_seconds'], expected_e2e)
                self.assertEqual(arm['metrics']['ttft_seconds'], statistics.median(ttfts))
                self.assertEqual(sum(tokens), 40)
                self.assertEqual(arm['metrics']['qualified_goodput_tokens_per_second'], 40 / sum(durations))
                self.assertEqual(arm['median_request_demand_wait_seconds'], statistics.median(request_waits))
                self.assertEqual(arm['median_request_demand_wait_seconds'], expected_wait)

    def test_all_five_block_values_and_3125_resamples_independently(self):
        arms = {a['trial']: a for a in self.data['arms']}
        for metric, expected in zip(P.METRICS, (.9182346584653581, .8875530391241149, 1.1090822446721382)):
            logs = []
            for block in self.data['paired']['blocks']:
                trial = block['block']
                value = math.log(arms[trial + '-lookahead']['metrics'][metric] / arms[trial + '-blocking']['metrics'][metric])
                self.assertEqual(block['log_ratios'][metric], value)
                logs.append(value)
            draws = sorted(statistics.median(draw) for draw in itertools.product(logs, repeat=5))
            aggregate = self.data['paired']['aggregate'][metric]
            self.assertEqual(aggregate['ratio'], math.exp(statistics.median(logs)))
            self.assertEqual(aggregate['ratio'], expected)
            self.assertEqual(aggregate['descriptive_range'], [math.exp(draws[78]), math.exp(draws[3046])])
            self.assertEqual(aggregate['draws'], len(draws))
            self.assertEqual(aggregate['nearest_ranks_1based'], [79, 3047])
        for block, change in zip(self.data['design']['order'], self.data['wait_latency_changes']):
            b, l = (arms[block['block'] + '-' + mode] for mode in P.MODES)
            self.assertEqual(change['wait_change_seconds'], l['median_request_demand_wait_seconds'] - b['median_request_demand_wait_seconds'])
            self.assertEqual(change['e2e_change_seconds'], l['metrics']['request_e2e_seconds'] - b['metrics']['request_e2e_seconds'])

    def test_accounting_host_gauges_and_device_summary_samples(self):
        counts = [54, 50, 53, 49, 50, 53, 50, 53, 53, 50]
        for arm, count in zip(self.data['arms'], counts):
            prefix = 'arms/' + arm['trial'] + '/'
            fields = [json.loads(line)['record']['fields'] for line in gzip.decompress((self.root / (prefix + 'server-events.jsonl.gz')).read_bytes()).splitlines()]
            invocations = [f for f in fields if f['event'] == 'mixed_decoder_invocation']
            self.assertEqual(len(invocations), 102)
            for key, total in (('checkpoint_read_bytes', 128811263616), ('h2d_bytes', 128811927312)):
                self.assertEqual(sum(f[key] for f in invocations), total)
                self.assertEqual(arm['logical_counters'][key], total)
            host_peak = max(json.loads(f['expert_read_pipeline'])['stats']['peak_host_charged_bytes'] for f in invocations)
            self.assertEqual(arm['host_logical_peak_bytes'], host_peak)
            self.assertEqual(host_peak, 16777984)
            self.assertEqual(host_peak / P.MIB, 16.000732421875)
            self.assertEqual(arm['host_logical_limit_bytes'], 64 * P.MIB)
            self.assertEqual(arm['sampled_device_max_bytes'], 4125 * P.MIB)
            self.assertEqual(arm['configured_device_target_bytes'], 7 * P.GIB)
            resource = projection(self.root, prefix + 'lifecycle.json.gz')['resources']
            device = projection(self.root, prefix + 'configuration.json')['selected_device_alias']
            self.assertEqual(arm['sampled_device_max_bytes'], resource['gpu_peak_bytes'][device])
            self.assertEqual(arm['device_sample_count'], count)
            self.assertEqual(count, resource['gpu_samples'])
            self.assertEqual(count, resource['memory_components']['sample_count'])
            self.assertFalse(resource['continuous_coverage'])
            self.assertFalse(resource['exact_peaks'])
            self.assertEqual(arm['pipeline_stats']['verified_reads'], 40943)
            self.assertEqual(arm['pipeline_stats']['wasted_read_bytes'], 0)
            self.assertEqual(arm['pipeline_stats']['unknown_bytes'], 0)
        self.assertTrue(all(v is None for v in self.data['unknowns'].values()))
        self.assertIn('not measured physical bus traffic', self.data['interpretation']['accounting'])
        self.assertIn('raw device sample streams are not', self.data['provenance']['device_sample_scope'])

    def test_nfs_phase_and_between_window_gap_from_public_snapshots(self):
        for arm in self.data['arms']:
            snapshots = projection(self.root, 'arms/' + arm['trial'] + '/nfs-snapshots.json.gz')
            def counter(name):
                return snapshots[name]['selected_mount']['counters']['serverreadbytes']
            phase = counter('requests-post') - counter('requests-pre')
            windows = sum(counter(r['name'] + '-post') - counter(r['name'] + '-pre') for r in arm['requests'])
            self.assertEqual(arm['nfs_phase_client_serverreadbytes']['requests_nfs'], phase)
            self.assertEqual(arm['nfs_request_window_sum_bytes'], windows)
            self.assertEqual(arm['nfs_between_request_windows_bytes'], phase - windows)
            self.assertEqual((phase, windows, phase - windows), (314825320, 312732264, 2093056) if arm['trial'] == 'P1-blocking' else (0, 0, 0))

    def test_public_tampering_or_missing_member_cannot_feed_renderer(self):
        name = self.root / 'arms/P1-blocking/qa-measured-1/observed.json.gz'
        original = name.read_bytes()
        try:
            obj = load(name)
            obj['data']['actual']['request_e2e_seconds'] += 1
            name.write_bytes(gzip.compress(json.dumps(obj).encode(), mtime=0))
            with self.assertRaisesRegex(P.C.Invalid, 'stale copy hash'):
                P.build_data(self.root)
            name.unlink()
            with self.assertRaisesRegex(P.C.Invalid, 'missing or unlisted'):
                P.build_data(self.root)
        finally:
            name.write_bytes(original)
        self.assertEqual((self.root / 'manifest.json').read_bytes(), self.original_manifest)


class PublicRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='paired-render-with_private_tree_absent-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.analysis = export_public(cls.root)
        cls.local = cls.root / 'local-output'
        cls.detached = cls.root / 'detached-output'
        cls.run_cli(HERE / 'render_public.py', cls.local)
        cls.run_cli(cls.analysis / 'render_public.py', cls.detached)

    @classmethod
    def run_cli(cls, script, output, *args, success=True):
        env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'MPLCONFIGDIR')}
        env['PYTHONNOUSERSITE'] = '1'
        result = subprocess.run([sys.executable, '-B', str(script), '--output-dir', str(output), *args],
                                cwd=cls.root, env=env, capture_output=True, text=True, timeout=60)
        if (result.returncode == 0) != success:
            raise AssertionError(result.stdout + result.stderr)
        return result

    def test_detached_public_only_outputs_are_byte_identical(self):
        self.assertEqual({p.name for p in self.detached.iterdir()}, set(P.OUTPUTS))
        self.assertFalse((self.analysis / 'derive.py').exists())
        self.assertFalse((self.analysis / 'study.py').exists())
        self.assertFalse((self.analysis / 'receipts.py').exists())
        self.assertFalse((self.analysis / 'inputs.json').exists())
        self.assertFalse((self.analysis / 'plot-data.json').exists())
        for name in P.OUTPUTS:
            self.assertEqual((self.local / name).read_bytes(), (self.detached / name).read_bytes(), name)
        self.assertFalse(list((self.root / 'publication_results_v1').rglob('__pycache__')))

    def test_output_manifest_actual_hashes_and_public_input_binding(self):
        manifest = load(self.detached / 'plot-manifest.json')
        data = load(self.detached / 'plot-data.json')
        provenance = data['provenance']
        self.assertEqual(manifest['provenance'], provenance)
        self.assertEqual(provenance['input_manifest_path'], 'publication_results_v1/manifest.json')
        self.assertEqual(provenance['input_manifest_sha256'], hashlib.sha256((self.root / 'publication_results_v1/manifest.json').read_bytes()).hexdigest())
        self.assertEqual(provenance['generator_sha256'], P.digest(P.encode(provenance['source_hashes'])))
        self.assertEqual(set(manifest['artifacts']), set(P.OUTPUTS) - {'plot-manifest.json'})
        for name, digest in manifest['artifacts'].items():
            self.assertEqual(hashlib.sha256((self.detached / name).read_bytes()).hexdigest(), digest)
        for name, digest in provenance['source_hashes'].items():
            self.assertEqual(hashlib.sha256((self.root / name).read_bytes()).hexdigest(), digest)

    def test_png_pdf_metadata_fonts_dimensions_and_no_dates(self):
        from PIL import Image
        data = load(self.detached / 'plot-data.json')
        provenance = data['provenance']
        for name in P.FIGURES:
            raw = (self.detached / name).read_bytes()
            if name.endswith('.png'):
                image = Image.open(io.BytesIO(raw))
                self.assertGreater(image.width, 1000)
                self.assertGreater(image.height, 700)
                metadata = json.loads(image.info['Description'])
                for key in ('input_manifest_path', 'input_manifest_sha256', 'generator_sha256', 'renderer_source_sha256', 'renderer_toolchain_sha256'):
                    self.assertEqual(metadata[key], provenance[key])
                self.assertEqual(metadata['plot_data_sha256'], P.digest((self.detached / 'plot-data.json').read_bytes()))
                self.assertIsNone(image.info.get('Creation Time'))
            else:
                self.assertTrue(raw.startswith(b'%PDF-'))
                self.assertNotIn(b'/CreationDate', raw)
                self.assertNotIn(b'/ModDate', raw)
                self.assertIn(b'/FontFile2', raw)
                self.assertIn(provenance['input_manifest_sha256'].encode(), raw)
                self.assertIn(provenance['generator_sha256'].encode(), raw)
                self.assertIn(b'analysis_v1/render_public.py', raw)
        pins = load(self.analysis / 'renderer-pins.json')
        self.assertEqual(provenance['renderer']['fonts'], pins['fonts'])
        self.assertEqual(provenance['renderer']['backend'], 'Agg')

    def test_academic_label_and_bounded_nonoverlapping_layout(self):
        layout = load(self.detached / 'layout.json')
        accounting = layout['accounting']['text_artists']
        texts = [r['text'] for r in accounting]
        self.assertIn('Requested host-to-device\nbytes', texts)
        self.assertNotIn('Facade H2D bytes', texts)
        self.assertEqual(texts.count('128,811,927,312'), 2)
        for figure in layout.values():
            self.assertEqual(figure['outside_canvas_text'], [])
            self.assertEqual(figure['text_bbox_overlaps'], [])
            for text in figure['text_artists']:
                left, bottom, right, top = text['bbox_fraction']
                self.assertTrue(0 <= left < right <= 1 and 0 <= bottom < top <= 1, text['text'])

    def test_public_check_default_and_tamper_fail_without_rewriting(self):
        before = {p.name: p.read_bytes() for p in self.detached.iterdir()}
        self.run_cli(self.analysis / 'plot_paired_read.py', self.detached, '--check')
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.detached.iterdir()})
        path = self.detached / 'paired-read-accounting.png'
        try:
            path.write_bytes(before[path.name] + b'tampered')
            result = self.run_cli(self.analysis / 'render_public.py', self.detached, '--check', success=False)
            self.assertIn('output hash differs', result.stdout)
            self.assertEqual(path.read_bytes(), before[path.name] + b'tampered')
        finally:
            path.write_bytes(before[path.name])

    def test_reject_output_into_frozen_bundle_and_unpinned_fonts(self):
        self.run_cli(self.analysis / 'render_public.py', self.root / 'publication_results_v1', success=False)
        import render
        altered = render.toolchain()
        altered['fonts'] = copy.deepcopy(altered['fonts'])
        altered['fonts']['normal']['sha256'] = '0' * 64
        with mock.patch.object(render, 'toolchain', return_value=altered), mock.patch.object(render, 'render_figures') as draw:
            with self.assertRaisesRegex(P.EvidenceError, 'environment/font pins differ'):
                P.generate(self.root / 'publication_results_v1')
            draw.assert_not_called()


if __name__ == '__main__':
    unittest.main()
