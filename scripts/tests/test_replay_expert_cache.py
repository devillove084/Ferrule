"""Offline parser/replay contracts. Synthetic fixtures never stand in for GPU evidence."""
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import random
import shutil
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location('replay_expert_cache', Path(__file__).parents[1] / 'replay_expert_cache.py')
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)
ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / 'papers/arxiv/experiments/storage_demand/profile_overlap_v1/publication_v1'


def key(expert, layer=0, image=1, precision=replay.PRECISION, source='CPU_FIXTURE_ONLY', roles=replay.ROLES):
    return source, image, precision, layer, expert, roles


def fixture():
    rows, pid = [], 7 << 24
    selections = [(0, 2), (1, 3), (0, 2), (1, 4)]
    for operation, (layer, expert) in enumerate(selections, 1):
        for part, kind in enumerate(('read_subset', 'layer.expert.upload', 'layer.expert')):
            row = dict.fromkeys(replay.NVTX_FIELDS, 'null')
            row.update(row_id=str(len(rows) + 1), range_id=str(len(rows) + 1), kind=kind,
                       start_ns=str(operation * 10 + part * 2), end_ns=str(operation * 10 + part * 2 + 1),
                       global_pid=str(pid), global_tid=str(pid + (2 if part == 0 else 1)), event_type='60', domain_id='1')
            if part == 0:
                row.update(operation=str(operation), source_generation='1', requested_bytes=str(replay.PAYLOAD_BYTES))
            else:
                row.update(image_generation='1', layer=str(layer), expert=str(expert), rows='1')
            rows.append(row)
    source = dict(candidate_identity=dict(source_archive_sha256='a' * 64))
    stats = dict.fromkeys(('verified_reads', 'submitted_reads', 'completed_reads', 'demanded_reads',
                          'sequences_started', 'sequences_finished'), 4)
    stats.update(dict.fromkeys(('failed_reads', 'unknown_reads', 'unknown_bytes', 'resident_skips',
                               'materialization_failures'), 0))
    stats['proved_read_bytes'] = 4 * replay.PAYLOAD_BYTES
    proof = dict(source_binding=dict(image_generation=1, source_generation=1), request_duration_ns=50,
                 metrics=dict(nvtx_range_counts=dict.fromkeys(('read_subset', 'layer.expert.upload', 'layer.expert'), 4)),
                 ledger=dict(stats=stats, invocation_count=2))
    return rows, source, proof, pid


def source_fixture():
    relative = dict(replay.BINDING_PINS, **{'Cargo.toml': 'b' * 64})
    original = {'/CPU_FIXTURE/' + name: digest for name, digest in relative.items()}
    digest = replay.sha256(json.dumps(original, sort_keys=True).encode())
    source = dict(candidate_identity=dict(commit=replay.SOURCE_COMMIT, source_archive_sha256='a' * 64,
                                         source_digest=digest, source_members=len(relative)),
                  source_pins=dict(replay.BINDING_PINS))
    manifest = dict(commit=replay.SOURCE_COMMIT, archive_sha256='a' * 64, original_source_pins=original,
                    pins_relative_to_frozen_source=relative, source_pins_digest=digest,
                    source_pins_digest_relative=replay.sha256(json.dumps(relative, sort_keys=True).encode()),
                    file_count=len(relative))
    return source, manifest


class ParserTests(unittest.TestCase):
    def test_complete_ranges_only_preserve_order_and_lower_qualified_identity(self):
        rows, source, proof, pid = fixture()
        keys, batches = replay.parse_demands(list(reversed(rows)), source, proof, pid, layer_count=2)
        self.assertEqual([(k[3], k[4]) for k in keys], [(0, 2), (1, 3), (0, 2), (1, 4)])
        self.assertEqual(len(batches), 4)
        self.assertEqual(keys[0], keys[2])
        self.assertEqual(keys[0][2], replay.PRECISION)
        self.assertEqual(keys[0][5], replay.ROLES)
        self.assertNotIn('parameter_id', repr(keys))

    def test_reject_missing_invalid_aliased_or_ambiguous_evidence(self):
        mutations = [
            lambda rows, proof: rows.pop(),
            lambda rows, proof: rows[2].update(image_generation='2'),
            lambda rows, proof: rows[2].update(expert='null'),
            lambda rows, proof: rows[2].update(rows='0'),
            lambda rows, proof: rows[2].update(layer='2'),
            lambda rows, proof: rows[2].update(expert='256'),
            lambda rows, proof: rows[2].update(row_id=rows[1]['row_id']),
            lambda rows, proof: rows[2].update(range_id=rows[1]['range_id']),
            lambda rows, proof: rows[2].update(global_tid=str(8 << 24)),
            lambda rows, proof: rows[2].update(end_global_tid=rows[2]['global_tid']),
            lambda rows, proof: rows[2].update(start_ns='NaN'),
            lambda rows, proof: rows[2].update(end_ns='100'),
            lambda rows, proof: rows[2].update(start_ns=rows[1]['start_ns']),
            lambda rows, proof: rows[0].update(source_generation='2'),
            lambda rows, proof: rows[0].update(requested_bytes='3146111'),
            lambda rows, proof: rows[0].update(operation='2'),
            lambda rows, proof: rows[1].update(expert='100'),
            lambda rows, proof: proof['ledger']['stats'].update(unknown_reads=1),
            lambda rows, proof: proof['ledger']['stats'].update(resident_skips=1),
            lambda rows, proof: proof['ledger']['stats'].update(sequences_started=5),
        ]
        for mutate in mutations:
            rows, source, proof, pid = fixture()
            mutate(rows, proof)
            with self.subTest(mutate=mutate), self.assertRaises((ValueError, TypeError)):
                replay.parse_demands(rows, source, proof, pid, layer_count=2)

    def test_source_manifest_links_digest_members_and_reviewed_binding(self):
        source, manifest = source_fixture()
        result = replay.verify_source_manifest(source, manifest)
        self.assertIn('not_exported_canonical_ids', result['scope'])
        for location in ('original_source_pins', 'pins_relative_to_frozen_source'):
            wrong = copy.deepcopy(manifest)
            wrong[location].pop(next(iter(wrong[location])))
            with self.subTest(location=location), self.assertRaises(ValueError):
                replay.verify_source_manifest(source, wrong)
        wrong = copy.deepcopy(source)
        wrong['source_pins'][next(iter(replay.BINDING_PINS))] = 'c' * 64
        with self.assertRaisesRegex(ValueError, 'binding'):
            replay.verify_source_manifest(wrong, manifest)
        wrong = copy.deepcopy(manifest)
        wrong['archive_sha256'] = 'd' * 64
        with self.assertRaises(ValueError):
            replay.verify_source_manifest(source, wrong)

    def test_json_and_expansion_bounds_do_not_silently_truncate(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}'):
            with self.assertRaises(ValueError):
                replay.json_bytes(raw)
        with mock.patch.object(replay, 'MAX_INPUT_BYTES', 16):
            with self.assertRaisesRegex(ValueError, 'byte bound'):
                replay.decode(gzip.compress(b'x' * 17), compressed=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'large'
            path.write_bytes(b'x' * 17)
            with self.assertRaises(ValueError):
                replay.bounded_read(path, 16)


class ReplayTests(unittest.TestCase):
    def run_policy(self, batches, policy, slots=3, **kwargs):
        return replay.replay(batches, policy, slots=slots, max_bytes=100, scratch_bytes=1,
                             payload_bytes=10, layer_count=kwargs.pop('layer_count', 1), **kwargs)

    def test_lru_hits_require_actual_retention_and_preserve_source_identity(self):
        variants = [key(1), key(1, image=2), key(1, precision='another'), key(1, source='another'),
                    key(1, roles=tuple(reversed(replay.ROLES)))]
        result = self.run_policy([[k] for k in variants] + [[variants[-1]]], 'global_lru')
        self.assertEqual((result['hits'], result['misses']), (1, 5))
        self.assertEqual(result['estimated_uploaded_bytes'], 50)
        self.assertEqual(result['estimated_requested_read_bytes'], 50)
        result = self.run_policy([[key(i)] for i in [1, 2, 3, 1, 4, 2]], 'global_lru')
        self.assertEqual((result['hits'], result['misses']), (1, 5))
        self.assertEqual(result['evictions'], 2)

    def test_current_selected_retention_does_not_reorder_and_bypass_is_charged(self):
        batches = [[key(1), key(2)], [key(3), key(4), key(1), key(2)]]
        baseline = self.run_policy(batches, 'global_lru')
        protected = self.run_policy(batches, 'layer_protected_bypass')
        self.assertEqual(baseline['hits'], 0)
        self.assertEqual(baseline['evicted_selected'], 3)
        self.assertEqual(baseline['evicted_future_hint'], 2)
        self.assertEqual(protected['hits'], 2)
        self.assertEqual(protected['bypasses'], 2)
        self.assertEqual(protected['persistent_slots'], 2)
        self.assertEqual(protected['transient_slots'], 1)
        self.assertEqual(protected['peak_charged_payload_and_scratch_bytes'], 31)
        self.assertEqual(protected['estimated_uploaded_bytes'], 40)
        self.assertEqual(protected['evicted_selected'], 0)
        self.assertEqual([k[4] for k in batches[1]], [3, 4, 1, 2])

    def test_static_layer_quotas_do_not_use_future_or_borrow_capacity(self):
        batches = [[key(1, 0), key(2, 0)], [key(1, 1)], [key(1, 0), key(2, 0)]]
        result = self.run_policy(batches, 'layer_protected_bypass', slots=3, layer_count=2)
        self.assertEqual(result['hits'], 1)
        self.assertEqual(result['bypasses'], 2)
        self.assertEqual(result['layer_results'][0]['misses'], 3)
        self.assertEqual(result['layer_results'][1]['misses'], 1)

    def test_history_compares_before_increment_and_ties_bypass(self):
        batches = [[key(i)] for i in [1, 2, 3, 4, 3, 3]]
        result = self.run_policy(batches, 'history_frequency_bypass')
        # First scan of 3/4 bypasses; second 3 ties old 1/2 and still bypasses.
        # Third 3 strictly beats prior count 1, then is admitted, not a hit yet.
        self.assertEqual((result['hits'], result['misses']), (0, 6))
        self.assertEqual(result['bypasses'], 3)
        result = self.run_policy(batches + [[key(3)]], 'history_frequency_bypass')
        self.assertEqual(result['hits'], 1)
        self.assertEqual(result['history_sketch_storage_bytes'], 8192)
        self.assertGreaterEqual(result['max_history_bytes'], 8192)
        self.assertLess(result['max_history_bytes'], 9216)
        self.assertEqual(result['frequency']['compare'], 'before_increment')

    def test_sketch_is_fixed_seed_bounded_saturating_and_aged(self):
        a, b = replay.FrequencySketch(), replay.FrequencySketch()
        for _ in range(300):
            a.increment(key(8))
            b.increment(key(8))
        self.assertEqual(a.counts, b.counts)
        self.assertEqual(a.prior(key(8)), 255)
        for _ in range(a.PERIOD - 300):
            a.increment(key(8))
        self.assertEqual(a.age, 0)
        self.assertEqual(a.prior(key(8)), 127)
        self.assertEqual(len(a.counts), 8192)

    def test_budget_includes_scratch_transient_and_both_limits(self):
        batches = [[key(i)] for i in range(150)]
        for slots in (64, 128, 320):
            for policy in replay.POLICIES:
                result = replay.replay(batches, policy, slots=slots)
                self.assertEqual(result['effective_slots'], min(slots, 103))
                self.assertEqual(result['full_requested_capacity_fits'], slots == 64)
                self.assertLessEqual(result['peak_charged_payload_and_scratch_bytes'], 320 * replay.MIB)
                self.assertEqual(result['persistent_slots'] + result['transient_slots'], min(slots, 103))
        with self.assertRaises(ValueError):
            replay.replay([[key(1)]], 'layer_protected_bypass', slots=1)
        with self.assertRaises(ValueError):
            replay.replay([[key(1)]], 'global_lru', max_bytes=8 * replay.MIB)

    def test_exact_stack_distance_matches_naive_and_is_not_access_distance(self):
        rng = random.Random(33)
        keys = [key(rng.randrange(12)) for _ in range(300)]
        last, distances = {}, []
        for i, k in enumerate(keys):
            if k in last:
                distances.append(len(set(keys[last[k] + 1:i])))
            last[k] = i
        result = replay.trace_diagnostics(keys)
        self.assertEqual(result['reuse_distance_min'], min(distances))
        self.assertEqual(result['reuse_distance_max'], max(distances))
        self.assertEqual(result['repeated_demands'], len(distances))
        self.assertIn('not_cold', result['history_scope'])

    def test_future_trace_suffix_does_not_change_decisions(self):
        prefix = [[key(i, i % 2)] for i in [0, 1, 2, 3, 0, 1]]
        for policy in replay.POLICIES:
            short = self.run_policy(prefix, policy, layer_count=2)
            long = self.run_policy(prefix + [[key(100, 0)]], policy, layer_count=2)
            self.assertEqual(short['cumulative_misses_after_layer'],
                             long['cumulative_misses_after_layer'][:len(prefix)])


class PublicArchiveTests(unittest.TestCase):
    def test_existing_archives_are_finite_complete_observed_windows_not_cold(self):
        traces = []
        for arm in ('blocking', 'lookahead'):
            rows, source, proof, pid, provenance = replay.load_public_archive(ARCHIVE, arm)
            keys, batches = replay.parse_demands(rows, source, proof, pid)
            d = replay.trace_diagnostics(keys)
            self.assertEqual((d['demands'], d['distinct_keys'], d['repeated_demands']), (3678, 1833, 1845))
            self.assertEqual((d['reuse_distance_min'], d['reuse_distance_max']), (314, 1784))
            self.assertEqual(len(batches), 360)
            self.assertFalse(provenance['full_trace_accepted'])
            baseline = replay.replay(batches, 'global_lru')
            self.assertEqual(baseline['hits'], 0)
            self.assertEqual(baseline['selected_at_start_hits'], 0)
            self.assertEqual(baseline['estimated_uploaded_bytes'], 11_571_399_936)
            json.dumps(baseline, allow_nan=False)
            traces.append(keys)
        self.assertEqual(*traces)

    def test_existing_archive_admission_replay_is_deterministic_and_same_budget(self):
        rows, source, proof, pid, _ = replay.load_public_archive(ARCHIVE, 'lookahead')
        _, batches = replay.parse_demands(rows, source, proof, pid)
        results = {}
        for policy in replay.POLICIES[1:]:
            a, b = replay.replay(batches, policy), replay.replay(batches, policy)
            self.assertEqual(a, b)
            self.assertEqual(a['effective_slots'], 64)
            self.assertEqual(a['persistent_slots'], 63)
            self.assertEqual(a['estimated_expert_reads'] + a['hits'], 3678)
            self.assertGreater(a['hits'], 0)
            self.assertEqual(a['estimated_uploaded_bytes'], a['misses'] * replay.PAYLOAD_BYTES)
            self.assertEqual(a['evicted_selected'], 0)
            self.assertLess(a['policy_metadata_python_peak_bytes'], 128 * 1024)
            results[policy] = a
        layer = results['layer_protected_bypass']
        history = results['history_frequency_bypass']
        self.assertEqual((layer['misses'], history['misses']), (3513, 3410))
        # Better total does not mean every decode invocation or layer improves.
        third = lambda r: r['cumulative_misses_after_layer'][119] - r['cumulative_misses_after_layer'][79]
        self.assertEqual((third(layer), third(history)), (306, 316))
        self.assertEqual(sum(b['misses'] > a['misses'] for a, b in
                             zip(layer['layer_results'], history['layer_results'])), 10)

    def test_archive_pin_failure_and_summary_csv_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('manifest.json', 'source.json', 'lookahead-proof.json', 'lookahead-nvtx.csv.gz'):
                shutil.copyfile(ARCHIVE / name, root / name)
            nvtx = root / 'lookahead-nvtx.csv.gz'
            nvtx.write_bytes(gzip.compress(b'kind,count\nlayer.expert,3678\n'))
            with self.assertRaisesRegex(ValueError, 'digest/length'):
                replay.load_public_archive(root, 'lookahead')
            manifest = json.loads((root / 'manifest.json').read_text())
            manifest['files'][nvtx.name] = dict(bytes=nvtx.stat().st_size, sha256=replay.sha256(nvtx.read_bytes()))
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'full NVTX CSV'):
                replay.load_public_archive(root, 'lookahead')


if __name__ == '__main__':
    unittest.main()
