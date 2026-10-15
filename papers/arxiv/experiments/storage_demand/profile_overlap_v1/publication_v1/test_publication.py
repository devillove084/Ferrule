"""Regression tests exercise semantic rejection, not only file checksum failures."""
from contextlib import contextmanager
import copy
import gzip
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import check as c


class Intervals(unittest.TestCase):
    def test_union_deduplicates_cross_dataset_events(self):
        self.assertEqual(c.length([(10, 20), (10, 20), (12, 14), (20, 23)]), 13)
        self.assertEqual(c.length([(10, 20)] * 61000), 10)

    def test_gaps_and_leading_delay_are_not_kernel_activity(self):
        self.assertEqual(c.union([(1000, 1002), (1008, 1010)]), [(1000, 1002), (1008, 1010)])
        self.assertEqual(c.length([(1000, 1002), (1008, 1010)]), 4)
        self.assertEqual(c.intersection([(0, 1000)], [(1000, 1002)]), [])

    def test_intersection_uses_unions_not_pair_sums(self):
        self.assertEqual(c.intersection([(0, 8), (4, 10)], [(3, 6), (5, 9)]), [(3, 9)])
        self.assertEqual(c.intersection([], [(1, 2)]), [])

    def test_reversed_and_empty_intervals(self):
        for pair in [(2, 1), (1, 1), (1.0, 2), (True, 2)]:
            with self.assertRaisesRegex(ValueError, 'interval'):
                c.union([pair])


class Archive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.proofs, cls.data, cls.results = c.load()

    @contextmanager
    def edit(self, obj, key, value):
        old = obj[key]
        obj[key] = value
        try:
            yield
        finally:
            obj[key] = old

    def reject(self, pattern, arm='lookahead'):
        with self.assertRaisesRegex(ValueError, pattern):
            c.validate_arm(self.proofs[arm], self.data[arm])

    def test_actual_pair_exact_results(self):
        for arm in c.ARMS:
            self.assertEqual(tuple(self.results[arm][k] for k in c.METRICS), c.EXPECTED[arm])
            self.assertEqual(self.results[arm]['source_verified_read_bytes'], 11571399936)
            self.assertIsNone(self.results[arm]['osrt_read_bytes'])
            self.assertEqual(self.results[arm]['expert_kernels'], 33403)

    def test_bad_interval(self):
        r = self.data['lookahead']['pread'][0]
        with self.edit(r, 'end_ns', r['start_ns']):
            self.reject('invalid request interval')

    def test_valid_but_corrupted_duration_changes_union(self):
        r = self.data['lookahead']['kernel'][0]
        with self.edit(r, 'end_ns', r['end_ns'] + 1):
            self.reject('interval unions differ')

    def test_noncanonical_order(self):
        rows = self.data['lookahead']['kernel']
        rows[0], rows[1] = rows[1], rows[0]
        try:
            self.reject('noncanonical')
        finally:
            rows[0], rows[1] = rows[1], rows[0]

    def test_source_generation(self):
        r = next(r for r in self.data['lookahead']['nvtx'] if r['kind'] == 'read_subset')
        with self.edit(r, 'source_generation', 2):
            self.reject('source/byte/identity')

    def test_duplicate_source_operation(self):
        reads = [r for r in self.data['lookahead']['nvtx'] if r['kind'] == 'read_subset']
        with self.edit(reads[1], 'operation', reads[0]['operation']):
            self.reject('operations not contiguous/unique')

    def test_osrt_null_cannot_be_zero_bytes(self):
        with self.edit(self.data['lookahead']['pread'][0], 'osrt_bytes', 0):
            self.reject('not ssize_t')

    def test_osrt_return_value_cannot_be_requested_bytes(self):
        with self.edit(self.data['lookahead']['pread'][0], 'return_value', 3146112):
            self.reject('not ssize_t')

    def test_duplicate_correlation(self):
        rows = self.data['lookahead']['kernel']
        with self.edit(rows[1], 'correlation_id', rows[0]['correlation_id']):
            self.reject('duplicate kernel/API correlation')

    def test_duplicate_launch_row(self):
        rows = self.data['lookahead']['kernel']
        with self.edit(rows[1], 'api_row_id', rows[0]['api_row_id']):
            self.reject('duplicate launch row')

    def test_kernel_before_api(self):
        r = self.data['lookahead']['kernel'][0]
        with self.edit(r, 'api_start_ns', r['end_ns'] + 1):
            self.reject('kernel/API order')

    def test_read_range_wrong_join(self):
        with self.edit(self.data['lookahead']['pread'][0], 'read_range_id', 999999):
            self.reject('misassigned pread')

    def test_expert_range_wrong_join(self):
        with self.edit(self.data['lookahead']['kernel'][0], 'expert_range_id', 999999):
            self.reject('misassigned kernel')

    def test_unknown_warning_rejected_even_with_old_classification(self):
        r = next(r for r in self.proofs['lookahead']['diagnostics'] if r['severity'] == 2)
        with self.edit(r, 'text', 'Unknown collector warning'), self.edit(r, 'original_text_sha256', c.digest(b'Unknown collector warning')):
            self.reject('unknown/target/global warning')

    def test_known_warning_on_model_is_not_waived(self):
        proof = self.proofs['lookahead']
        r = next(r for r in proof['diagnostics'] if r['severity'] == 2)
        model = next(r['globalPid'] for r in proof['process_roles'] if r['role'] == 'model')
        with self.edit(r, 'globalPid', model):
            self.reject('unknown/target/global warning')

    def test_warning_affects_full_trace(self):
        r = next(r for r in self.proofs['lookahead']['diagnostics'] if r['severity'] == 2)
        with self.edit(r, 'affects_full_trace', False):
            self.reject('unknown/target/global warning')

    def test_unknown_drops_cannot_be_zero(self):
        with self.edit(self.proofs['lookahead']['final_scope'], 'dropped_event_count', 0):
            self.reject('unknown must remain unknown')

    def test_global_acceptance_cannot_be_upgraded(self):
        with self.edit(self.proofs['lookahead']['final_scope'], 'full_trace_accepted', True):
            self.reject('scoped-only')

    def test_cut_raw_header(self):
        p = self.proofs['lookahead']
        header = dict(p['capture_header']); del header['OS_RUNTIME_OPTIONS:DURATION_THRESHOLD_NS']
        with self.edit(p, 'capture_header', header):
            self.reject('cut/filtered collector header')

    def test_cut_raw_field(self):
        proof = dict(self.proofs['lookahead']); del proof['osrt_schema_columns']
        with self.assertRaisesRegex(ValueError, 'cut/unknown raw evidence'):
            c.validate_proof(proof)

    def test_cut_diagnostic_field(self):
        proof = copy.deepcopy(self.proofs['lookahead'])
        del proof['diagnostics'][0]['original_row_sha256']
        with self.assertRaisesRegex(ValueError, 'cut diagnostic fields'):
            c.validate_proof(proof)

    def test_cut_raw_source_inventory(self):
        proof = copy.deepcopy(self.proofs['lookahead'])
        del proof['original_files']['collector_manifest']
        with self.assertRaisesRegex(ValueError, 'cut raw-source inventory'):
            c.validate_proof(proof)

    def test_ledger_bytes_not_relabelled_osrt(self):
        with self.edit(self.proofs['lookahead']['ledger']['stats'], 'proved_read_bytes', 11571399935):
            self.reject('typed bytes/calls mismatch')

    def test_incomplete_verification(self):
        with self.edit(self.proofs['lookahead']['ledger']['stats'], 'verified_reads', 3677):
            self.reject('unverified read ledger')

    def test_source_binding_corruption(self):
        with self.edit(self.proofs['lookahead']['source_binding'], 'source_commit', '0' * 40):
            self.reject('source binding changed')

    def test_cut_csv_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'bad.csv.gz'
            p.write_bytes(gzip.compress(b'row_id,start_ns\n1,2\n'))
            with self.assertRaisesRegex(ValueError, 'CSV columns'):
                c.read_csv(p, 'pread')

    def test_archive_hash_catches_corruption_before_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'manifest.json').write_bytes((c.HERE / 'manifest.json').read_bytes())
            for name in self.manifest['files']:
                (root / name).symlink_to(c.HERE / name)
            f = root / 'lookahead-pread.csv.gz'
            f.unlink()
            f.write_bytes((c.HERE / f.name).read_bytes() + b'corrupted')
            with self.assertRaisesRegex(ValueError, 'hash/size mismatch'):
                c.load(root)

    def test_zero_blocking_overlap_is_observed_not_missing(self):
        self.assertGreater(self.results['blocking']['worker_pread_union_ns'], 0)
        self.assertGreater(self.results['blocking']['expert_kernel_union_ns'], 0)
        self.assertEqual(self.results['blocking']['pread_kernel_intersection_ns'], 0)

    def test_numeric_intervals_contain_no_paths_or_environment(self):
        for arm in c.ARMS:
            for kind in c.FIELDS:
                raw = gzip.decompress((c.HERE / (arm + '-' + kind + '.csv.gz')).read_bytes())
                for forbidden in (b'/home/', b'ENVIRONMENT', b'Bearer ', b'nonce', b'worker_token'):
                    self.assertNotIn(forbidden, raw)


if __name__ == '__main__':
    unittest.main()
