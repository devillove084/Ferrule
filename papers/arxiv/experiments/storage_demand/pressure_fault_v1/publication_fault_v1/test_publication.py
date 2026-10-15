"""Generative corruption tests on copies only; no CUDA or private input required."""
from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import archive
import check
import evidence as e

REJECT = (e.EvidenceError, KeyError, TypeError, ValueError)
IGNORE = shutil.ignore_patterns('__pycache__', '.cache', '.plot-env', '.preview')


class PublicFaultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = {c: e.read_rows(e.HERE/'evidence'/(c+'.jsonl')) for c in e.CASES}
        cls.custody = e.load(e.HERE/'custody.json')
        cls.source = e.load(e.HERE/'source-manifest.json')

    def reject_rows(self, case, mutate):
        rows = deepcopy(self.rows[case])
        mutate(rows)
        with self.assertRaises(REJECT):
            e.validate_rows(rows, case)

    def reject_custody(self, mutate):
        d = deepcopy(self.custody)
        mutate(d)
        with self.assertRaises(REJECT):
            archive.validate_custody(d)

    def test_actual_four_cells(self):
        self.assertEqual(check.check()['passed_cells'], 4)

    def test_corrupted_metrics_all_cells(self):
        keys = ('real_compute_launch_delta', 'real_h2d_copy_delta', 'artifact_upload_delta',
                'artifact_upload_byte_delta', 'real_read_byte_delta', 'faults_observed',
                'committed_frontier_before', 'committed_frontier_after')
        for case in e.CASES:
            for line in (19, 38):
                for key in keys:
                    with self.subTest(case=case, line=line, metric=key):
                        self.reject_rows(case, lambda r: r[line]['evidence'].__setitem__(key, r[line]['evidence'][key]+1))

    def test_cache_equations(self):
        for case in e.CASES:
            for line in (5, 7, 10, 13 if case == 'mixed_clean' else 17):
                for key in e.LEDGER_KEYS:
                    with self.subTest(case=case, line=line, key=key):
                        self.reject_rows(case, lambda r: r[line]['value']['cache_ledger'].__setitem__(key, r[line]['value']['cache_ledger'][key]+1))

    def test_unknown_custody_not_refunded(self):
        for line in (10, 17, 29, 36):
            for key, value in (('free_slots', 31), ('active_transactions', 0), ('committed_page_slot', 1)):
                self.reject_rows('mixed_unknown', lambda r: r[line]['value']['kv'].__setitem__(key, value))
            self.reject_rows('mixed_unknown', lambda r: r[line]['value'].__setitem__('live_state_bytes_including_default_committed_working', 1024))
            self.reject_rows('mixed_unknown', lambda r: r[line]['value']['image'].__setitem__('bf16_holds', 0))
            self.reject_rows('mixed_unknown', lambda r: r[line]['value'].__setitem__('device_quarantined', False))
        self.reject_rows('mixed_unknown', lambda r: r[18].__setitem__('forced_allocator_zero', True))

    def test_consistent_false_cache_refund(self):
        def refund(rows):
            s = rows[17]['value']
            s['cache_ledger'] = dict(budget=65536, free=65536, resident=0, pending=0, scratch=0, charged=0, unknown_subset=0)
            for key in e.CACHE_MAP.values():
                s['image']['cache'][key] = 0
            s['physical_owner']['expert_cache']['value'] = deepcopy(s['image']['cache'])
            rows[19]['evidence']['after_cleanup'] = deepcopy(s)
        self.reject_rows('mixed_unknown', refund)

    def test_frontier_generation_and_source_identity(self):
        for case in e.CASES:
            for key in ('frontier', 'generation', 'topology', 'gdn_position', 'state_bytes'):
                self.reject_rows(case, lambda r: r[10]['value']['committed'].__setitem__(key, r[10]['value']['committed'][key]+1))
            self.reject_rows(case, lambda r: r[9]['submission']['reads']['last_receipt']['identity'].__setitem__('image_generation', 999))
            self.reject_rows(case, lambda r: r[9].__setitem__('transaction', 3))

    def test_duplicate_fault(self):
        for case in e.CASES:
            self.reject_rows(case, lambda r: r.insert(10, deepcopy(r[9])))
            self.reject_rows(case, lambda r: r.__setitem__(8, deepcopy(r[9])))
            self.reject_rows(case, lambda r: r[9].__setitem__('faults_observed', 2))

    def test_missing_seal_or_reordered_cells(self):
        for case in e.CASES:
            self.reject_rows(case, lambda r: r.pop())
            self.reject_rows(case, lambda r: r.__setitem__(slice(1,39), r[20:39]+r[1:20]))
            self.reject_rows(case, lambda r: r[20].__setitem__('schedule', 'blocking'))

    def test_false_unknown_content_proof(self):
        self.reject_rows('mixed_unknown', lambda r: r[19]['evidence'].__setitem__('committed_state_content_check', e.CLEAN_CONTENT))
        self.reject_rows('mixed_unknown', lambda r: r[17]['value'].__setitem__('state_contents', e.CLEAN_CONTENT))
        for cell in e.build_model()['cells'][2:]:
            self.assertIsNone(cell['committed_content']['equality'])
            self.assertIsNone(cell['snapshots']['after_cleanup']['state_content_bytes'])
            self.assertIsNone(cell['snapshots']['after_cleanup']['aggregate_physical_retained_bytes'])

    def test_false_clean_release(self):
        self.reject_rows('mixed_clean', lambda r: r[13]['value'].__setitem__('live_state_bytes_including_default_committed_working', 0))
        self.reject_rows('mixed_clean', lambda r: r[18].__setitem__('allocator_requested_after_drop_trim', 512))
        self.reject_rows('mixed_clean', lambda r: r[18].__setitem__('scope', 'component_only'))

    def test_wait_and_discard_observation(self):
        for case in e.CASES:
            for field in ('compute_event_completed', 'upload_sync_completed', 'deliberately_discarded'):
                self.reject_rows(case, lambda r: r[9]['physical_proof_before_test_discards_receipt'].__setitem__(field, not r[9]['physical_proof_before_test_discards_receipt'][field]))

    def test_host_cleanup_and_waste(self):
        for case in e.CASES:
            self.reject_rows(case, lambda r: r[11].__setitem__('device_release_authority', True))
            for key in ('wasted_reads', 'wasted_read_bytes', 'cancelled_reads'):
                self.reject_rows(case, lambda r: r[38]['evidence']['read_statistics_delta'].__setitem__(key, 0))

    def test_false_quiescence(self):
        for field in ('quiescence', 'release', 'allocator'):
            self.reject_rows('mixed_unknown', lambda r: r[17]['value']['physical_owner'].__setitem__(field, {'state': 'observed'}))

    def test_source_pins_not_relabelled(self):
        for field in ('component_commit', 'archive_commit', 'archive_sha256', 'source_digest', 'executable_sha256', 'final_handoff_sha256'):
            d = deepcopy(self.source)
            d['identity'][field] = '0'*len(d['identity'][field])
            with self.subTest(field=field), self.assertRaises(REJECT):
                check.validate_source(d)
        d = deepcopy(self.source)
        d['frozen_source_sha256'][next(iter(d['frozen_source_sha256']))] = '0'*64
        with self.assertRaises(REJECT):
            check.validate_source(d)

    def test_echild_nvml_not_release_proof(self):
        self.reject_custody(lambda d: d.__setitem__('physical_owner_release_certificate_observed', True))
        self.reject_custody(lambda d: d['selectors'][0]['supervisor'].__setitem__('gpu_quiescence', 'verified'))
        self.reject_custody(lambda d: d['selectors'][1]['custody'].__setitem__('echild', False))
        self.reject_custody(lambda d: d['locks'].__setitem__('global_reservation', True))
        self.reject_custody(lambda d: d['final'].__setitem__('physical_cuda_release', 'verified'))

    def test_idle_transients_not_censored(self):
        name = 'gpu-run/02-unknown/post-idle.json'
        self.reject_custody(lambda d: d['idle'][name]['samples'].pop(0))
        self.reject_custody(lambda d: d['idle'][name].__setitem__('accepted_sample_indices', [0,1,2]))
        self.reject_custody(lambda d: d['idle'][name]['samples'][1]['devices'][0].__setitem__('memory_mib', 1))
        self.reject_custody(lambda d: d['idle'][name]['samples'][1].__setitem__('compute_apps', 1))

    def test_private_fields_missing_and_bool_quantities(self):
        self.reject_rows('mixed_clean', lambda r: r[0].__setitem__('worker_token', 'not-a-real-secret'))
        self.reject_rows('mixed_clean', lambda r: r[0]['config'].__setitem__('hidden', True))
        self.reject_rows('mixed_clean', lambda r: r[13]['value']['cache_ledger'].pop('charged'))

    def test_json_duplicates_and_nonfinite(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}'):
            with self.assertRaises(REJECT):
                e.loads(text)

    def test_assertions_not_individual_receipts(self):
        for cell in e.build_model()['cells']:
            a = cell['assertion_backed']
            self.assertIsNone(a['individual_error_payloads'])
            self.assertIsNone(a['standalone_wait_receipts'])
            self.assertIn('not_individual_API_receipts', a['basis'])
            if cell['case'] == 'mixed_unknown':
                self.assertEqual(a['independent_compute_waits_succeeded'], 2)
                self.assertEqual(a['independent_upload_waits_succeeded'], 2)

    def test_stdout_selector_and_stderr_consistency(self):
        out = (e.HERE/'evidence/mixed_clean.stdout').read_text()
        err = (e.HERE/'evidence/mixed_clean.stderr').read_text()
        for badout, baderr in ((out.replace('1 passed','2 passed'),err), (out,err+'extra\n')):
            with self.assertRaises(REJECT):
                e.validate_streams(self.rows['mixed_clean'], badout, baderr, e.SELECTORS[0])

    def test_resealed_false_projection_rejected(self):
        mutations = (
            ('postfailure.json', lambda d: d['cells'][2]['committed_content'].__setitem__('equality', True)),
            ('postfailure.json', lambda d: d['cells'][2]['snapshots']['after_cleanup']['cache'].__setitem__('charged', 0)),
            ('postfailure.json', lambda d: d['cells'][0]['assertion_backed'].__setitem__('individual_error_payloads', [])),
            ('custody.json', lambda d: d.__setitem__('physical_owner_release_certificate_observed', True)),
            ('source-manifest.json', lambda d: d['identity'].__setitem__('executable_sha256', '0'*64)),
            ('history.json', lambda d: d['preparation'].__setitem__('gpu_execution', True)),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'publication'
            shutil.copytree(e.HERE, root, ignore=IGNORE)
            original_manifest = (root/'manifest.json').read_bytes()
            for name, mutate in mutations:
                with self.subTest(name=name):
                    original = (root/name).read_bytes()
                    data = e.load(root/name)
                    mutate(data)
                    (root/name).write_bytes(e.encode(data))
                    manifest = e.load(root/'manifest.json')
                    manifest['files'][name] = {'bytes': (root/name).stat().st_size, 'sha256': e.sha256(root/name)}
                    (root/'manifest.json').write_bytes(e.encode(manifest))
                    with self.assertRaises(REJECT):
                        check.check(root)
                    (root/name).write_bytes(original)
                    (root/'manifest.json').write_bytes(original_manifest)

    def test_isolated_stdlib_copy_without_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'publication'
            shutil.copytree(e.HERE, root, ignore=IGNORE)
            self.assertFalse((Path(tmp)/'target').exists())
            result = subprocess.run([sys.executable, '-S', '-B', str(root/'check.py')], cwd=tmp,
                                    env={'PATH': os.defpath, 'PYTHONDONTWRITEBYTECODE': '1'},
                                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertEqual(e.loads(result.stdout)['passed_cells'], 4)


if __name__ == '__main__':
    unittest.main()
