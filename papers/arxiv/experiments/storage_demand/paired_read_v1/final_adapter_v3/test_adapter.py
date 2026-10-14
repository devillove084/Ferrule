"""CPU-only admission regressions; synthetic cases never become evidence receipts."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE.parent))
from paired_read_v1.final_adapter_v3 import binding as b, evidence as e
from paired_read_v1 import current_binding_v3 as old


class PolicyTests(unittest.TestCase):
    def inventory(self):
        return dict(archive_and_all_pinned_source_bytes_verified=True, archive_members=1484,
                    missing=[], extra_files=[dict(relative_path=n, sha256=pin, path='/fixture/' + n,
                    source_pin_member=False, archive_member=False, evidence_manifest_pin=pin)
                    for n, pin in b.EXTRAS.items()])

    def manifest(self):
        return dict(pins_relative_to_directory={'frozen-source/' + n: pin for n, pin in b.EXTRAS.items()})

    def test_only_three_exact_noncompiled_manifest_pins_admit(self):
        with patch.object(b, 'immutable'), patch.object(b.ct, 'sha', side_effect=lambda p: b.EXTRAS[str(p)[9:]]):
            value = b.disposition(self.inventory(), self.manifest())
        self.assertEqual(value['compiled_archive_members_unchanged'], 1484)
        self.assertFalse(value['manifest_rewritten'])

    def test_extra_missing_wrong_hash_or_unpinned_reject(self):
        cases = []
        for key, value in (('archive_members', 1483), ('missing', ['crates/source.rs'])):
            row = self.inventory(); row[key] = value; cases.append((row, self.manifest()))
        row = self.inventory(); row['extra_files'].append(dict(relative_path='other.pyc', sha256='0'*64)); cases.append((row, self.manifest()))
        row = self.inventory(); row['extra_files'][0]['sha256'] = '0'*64; cases.append((row, self.manifest()))
        row = self.inventory(); row['extra_files'][0]['source_pin_member'] = True; cases.append((row, self.manifest()))
        row = self.inventory(); row['extra_files'][0]['evidence_manifest_pin'] = None; cases.append((row, self.manifest()))
        for row, manifest in cases:
            with self.subTest(row=row), self.assertRaises(ValueError):
                b.disposition(row, manifest)

    def test_manifest_pinned_bytecode_mutation_rejects(self):
        with patch.object(b, 'immutable'), patch.object(b.ct, 'sha', return_value='0'*64), self.assertRaises(ValueError):
            b.disposition(self.inventory(), self.manifest())

    def test_fingerprint_wrong_or_missing_rejects_even_when_passed(self):
        primary = dict(passed=True, prediction_fingerprints=dict(calls={'a': '1', 'b': '2'}))
        self.assertEqual(e.paired_fingerprints(primary, primary, 2), 2)
        for actual in (dict(passed=True), dict(passed=True, prediction_fingerprints=dict(calls={'a': '9', 'b': '2'}))):
            with self.assertRaises(ValueError):
                e.paired_fingerprints(actual, primary, 2)
        self.assertEqual(e.paired_fingerprints(dict(prediction_fingerprints=dict(calls={'b': '2'})), primary, 1, subset=True), 1)
        with self.assertRaises(ValueError):
            e.paired_fingerprints(dict(prediction_fingerprints=dict(calls={'z': '2'})), primary, 1, subset=True)

    def test_unsafe_source_paths_reject(self):
        for n in ('../escape', '/absolute', 'a//b', './a', ''):
            with self.subTest(n=n), self.assertRaises(ValueError):
                e.safe_member(n)

    def test_fake_custody_never_admits(self):
        self.assertFalse(b.ct.successful_process(dict(returncode=0, timed_out=False)))
        self.assertFalse(b.ct.successful_process(dict(returncode=0, timed_out=False,
            custody=dict(echild=True), descendant_custody='proven_drained')))

    def test_seal_writer_is_exclusive_and_exact_destination_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / 'binding.json'
            with patch.object(b, 'OUT', root), patch.object(b, 'BINDING_PATH', path):
                b.seal_to(path, dict(schema='cpu-test'))
                self.assertEqual(b.ct.load(path)['digest'], b.ct.digest(dict(schema='cpu-test')))
                with self.assertRaises(ValueError):
                    b.seal_to(path, dict(schema='replacement'))
                with self.assertRaises(ValueError):
                    b.seal_to(root / 'other.json', {})

    def test_inherited_default_paths_are_rebound_explicitly(self):
        with patch.object(b, '_validate_binding', return_value={}) as validate:
            b._base.validate_binding()
            validate.assert_called_once_with(b.BINDING_PATH)
        with patch.object(b, '_validate_execution', return_value={}) as validate:
            b._base.validate_execution()
            validate.assert_called_once_with(b.EXECUTION_PATH)

    def test_new_globals_do_not_relabel_historical_binding(self):
        self.assertNotEqual(b.ADAPTER, old.ADAPTER)
        self.assertNotEqual(b.BINDING_PATH, old.BINDING_PATH)
        self.assertEqual(b._base.ADAPTER, b.ADAPTER)
        self.assertIs(b._base.binding_value, b.binding_value)
        self.assertIs(b._base.execution_value, b.execution_value)


@unittest.skipUnless((e.ROOT / 'FINAL.json').is_file(), 'private qualified candidate unavailable')
class CandidateNegativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = e.load('source-freeze.json')
        cls.qualification = e.load('qualification-plan-v2.json')
        cls.runtime = e.load('runtime-library-pins-final.json')
        cls.artifacts = cls.qualification['artifacts']
        cls.identity = old.candidate_identity(e.ROOT)
        cls.module = old.reviewer(e.ROOT, cls.identity)
        cls.load = b.ct.load

    def changed(self, path, value):
        original = type(self).load
        return patch.object(b.ct, 'load', side_effect=lambda p: value if Path(p) == path else original(p))

    def test_changed_archived_source_hash_rejects(self):
        changed = copy.deepcopy(self.source)
        changed['pins_relative_to_frozen_source']['Cargo.toml'] = '0'*64
        with self.changed(e.ROOT / 'source-freeze.json', changed), self.assertRaises(ValueError):
            e.source_inventory(self.identity, e.load('evidence-manifest.json'))

    def test_old_layout_wrong_binary_hash_wrong_source_reject(self):
        for key, value in (('executable', str(e.ROOT / 'staged/cli')), ('sha256', '0'*64), ('source_digest', '0'*64)):
            artifacts = copy.deepcopy(self.artifacts); artifacts['cli'][key] = value
            with self.subTest(key=key), patch.object(b.ct, 'check_pins'), self.assertRaises((ValueError, FileNotFoundError)):
                e.build_review(self.module, self.source, artifacts, self.runtime)

    def test_runtime_pin_mismatch_rejects(self):
        runtime = copy.deepcopy(self.runtime)
        runtime['per_role']['cli'].pop(next(iter(runtime['per_role']['cli'])))
        with patch.object(b.ct, 'check_pins'), self.assertRaises(ValueError):
            e.build_review(self.module, self.source, self.artifacts, runtime)

    def test_gpu_fake_custody_rejects_even_when_result_passed(self):
        gate = self.qualification['gates'][0]
        path = e.ROOT / 'gates' / gate['id'] / 'supervisor.json'
        changed = copy.deepcopy(type(self).load(path)); changed['custody']['echild'] = False
        with self.changed(path, changed), self.assertRaises(ValueError):
            self.module.review_gate(e.ROOT, gate, self.artifacts, self.runtime, self.source)

    def test_gpu_runtime_mapping_not_covered_rejects(self):
        runtime = copy.deepcopy(self.runtime); runtime['pins'] = {}
        with self.assertRaises(ValueError):
            self.module.review_gate(e.ROOT, self.qualification['gates'][0], self.artifacts, runtime, self.source)

    def test_missing_or_failed_quality_rejects(self):
        gate = self.qualification['gates'][0]
        result_path = e.ROOT / 'gates' / gate['id'] / 'gate-result.json'
        changed = copy.deepcopy(type(self).load(result_path)); changed['report_sha256'] = '0'*64
        with self.changed(result_path, changed), self.assertRaises(ValueError):
            e.gate_review(self.module, self.qualification, self.source, self.runtime)
        path = e.ROOT / 'gates' / gate['id'] / 'quality.json'
        report = type(self).load(path); report['passed'] = False
        with self.changed(path, report), self.assertRaises(ValueError):
            e.gate_review(self.module, self.qualification, self.source, self.runtime)

    def test_actual_cli_source_admission_is_separate(self):
        value = b.source_admission(self.identity)
        self.assertEqual(value['metadata_required'], 7704576)
        self.assertEqual(value['metadata_limit'], 8388608)
        self.assertEqual(value['cpu_build_custody'], 'unobserved')
        path = b.ADMISSION / 'FINAL.json'
        changed = copy.deepcopy(type(self).load(path)); changed['source_admission_blocks_v3'] = True
        with self.changed(path, changed), self.assertRaises(ValueError):
            b.source_admission(self.identity)


if __name__ == '__main__':
    unittest.main()
