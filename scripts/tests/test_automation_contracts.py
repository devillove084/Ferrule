"""CPU-only selector/recipe contracts; compiled membership is checked separately."""

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('membership', ROOT / 'scripts/check_nextest_membership.py')
membership = importlib.util.module_from_spec(spec)
spec.loader.exec_module(membership)


class SelectionContracts(unittest.TestCase):
    def setUp(self):
        self.config = membership.load_config((ROOT / '.config/nextest.toml').read_text())
        self.owner = dict(package='ferrule-backend', binary='ferrule_backend', kind='lib',
                          test='cuda::operator_owner::pr01_tests::allocation_contract')
        self.ep = dict(package='ferrule-composition', binary='ferrule_composition', kind='lib',
                       test='model_factory::renamed_adapter::tests::tiny_ep2_full_hybrid_matches_local_and_lifecycle')

    def test_current_config_selects_owner_and_module_neutral_composition_suffix(self):
        for profile, row in [('gpu-smoke', self.owner), ('gpu-nightly', self.ep)]:
            self.assertEqual(membership.check(self.config, profile, [row]), [row['test']])

    def test_stale_owner_namespace_fails_even_with_other_smoke_tests(self):
        self.config['profile']['gpu-smoke']['default-filter'] = self.config['profile']['gpu-smoke']['default-filter'].replace('operator_owner', 'context')
        unrelated = dict(self.owner, binary='cuda_standard', test='unrelated')
        with self.assertRaisesRegex(ValueError, 'excluded'):
            membership.check(self.config, 'gpu-smoke', [self.owner, unrelated])

    def test_stale_package_fails_even_with_other_nightly_tests(self):
        self.config['profile']['gpu-nightly']['default-filter'] = self.config['profile']['gpu-nightly']['default-filter'].replace('ferrule-composition', 'ferrule-runtime')
        with self.assertRaisesRegex(ValueError, 'excluded'):
            membership.check(self.config, 'gpu-nightly', [self.ep])

    def test_empty_branch_wrong_package_or_wrong_kind_cannot_pass(self):
        for rows in ([], [dict(self.ep, package='ferrule-runtime')], [dict(self.ep, kind='test')]):
            with self.assertRaisesRegex(ValueError, 'zero compiled tests'):
                membership.check(self.config, 'gpu-nightly', rows)

    def test_nextest_native_filter_mismatch_also_fails(self):
        # Native nextest 0.9.146 JSON: binary fields are flattened into the suite.
        data = {'rust-suites': {'ferrule-composition': {
            'package-name': 'ferrule-composition', 'binary-id': 'ferrule-composition',
            'package-id': 'path+file:///repo/crates/ferrule-composition#0.2.0',
            'binary-name': 'ferrule_composition', 'kind': 'lib', 'status': 'listed',
            'testcases': {self.ep['test']: {'ignored': True, 'kind': 'test',
                'filter-match': {'status': 'mismatch', 'reason': 'ignored'}}}}}}
        with self.assertRaisesRegex(ValueError, 'excluded'):
            membership.check(self.config, 'gpu-nightly', membership.nextest_rows(data))
        data['rust-suites']['ferrule-composition']['testcases'][self.ep['test']]['filter-match'] = {'status': 'matches'}
        self.assertEqual(len(membership.check(self.config, 'gpu-nightly', membership.nextest_rows(data))), 1)

    def test_package_id_formats(self):
        for value in ('ferrule-composition 0.2.0 (path+file:///repo)',
                      'path+file:///repo/crates/ferrule-composition#0.2.0',
                      'path+file:///repo#ferrule-composition@0.2.0'):
            self.assertEqual(membership.package_name(value), 'ferrule-composition')

    def test_cargo_metadata_only_lists_cuda_library_with_hidden_devices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / 'tests'
            binary.touch()
            artifact = dict(reason='compiler-artifact', executable=str(binary),
                            package_id='path+file:///repo#ferrule-composition@0.2.0',
                            profile={'test': True}, target={'kind': ['lib'], 'name': 'ferrule_composition'},
                            features=['cuda'])
            path = root / 'build.jsonl'
            path.write_text(json.dumps(artifact) + '\n')
            with mock.patch.object(membership.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0,
                    stdout=self.ep['test'] + ': test\n', stderr='')) as run:
                rows = membership.cargo_rows(path, 'gpu-nightly')
            self.assertEqual(len(membership.check(self.config, 'gpu-nightly', rows)), 1)
            self.assertEqual(run.call_args.args[0], [str(binary), '--list', '--format', 'terse'])
            self.assertEqual(run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual(run.call_args.kwargs['env']['NVIDIA_VISIBLE_DEVICES'], 'void')
            artifact['features'] = []
            path.write_text(json.dumps(artifact))
            with mock.patch.object(membership.subprocess, 'run') as run:
                with self.assertRaisesRegex(ValueError, 'exactly one CUDA'):
                    membership.cargo_rows(path, 'gpu-nightly')
                run.assert_not_called()

    def test_missing_or_duplicate_profile_fails_closed(self):
        source = (ROOT / '.config/nextest.toml').read_text()
        for changed in (source.replace('[profile.gpu-smoke]', '[profile.retired-smoke]'),
                        source + "\n[profile.gpu-smoke]\ndefault-filter = '''all()'''\n"):
            with self.assertRaises(ValueError):
                membership.load_config(changed)

    def test_unknown_filter_syntax_fails_closed(self):
        self.config['profile']['gpu-smoke']['default-filter'] = 'all()'
        with self.assertRaises(ValueError):
            membership.check(self.config, 'gpu-smoke', [self.owner])


class RecipeContracts(unittest.TestCase):
    def test_native_lists_preserve_each_execution_lanes_ignored_policy(self):
        source = (ROOT / 'justfile').read_text()
        for profile, policy in (('gpu-smoke', 'all'), ('gpu-nightly', 'only')):
            lines = [line for line in source.splitlines()
                     if 'cargo nextest list ' in line and '--profile ' + profile in line]
            self.assertEqual(len(lines), 1)
            self.assertIn('--run-ignored ' + policy, lines[0])
            self.assertIn("CUDA_VISIBLE_DEVICES=''", lines[0])

    def test_retired_recipes_not_reintroduced_as_false_aliases(self):
        source = (ROOT / 'justfile').read_text()
        for name in ('expert-stream-smoke', 'deepseek-v4-generate', 'dsv4-cuda-generate',
                     'dsv4-cuda-generate-json', 'dsv4-cuda-moe-profile'):
            self.assertNotIn(name, source)
        self.assertIn('dsv4-runtime-driver-bench ', source)
        self.assertIn('just run-cuda bench-interactive ', source)
        self.assertNotIn(' -q ', source)
        self.assertIn('chat {{ model }} --backend cuda', source)
        self.assertIn('chat models/DeepSeek-V4-Flash-0731 --backend cuda', source)


if __name__ == '__main__':
    unittest.main()
