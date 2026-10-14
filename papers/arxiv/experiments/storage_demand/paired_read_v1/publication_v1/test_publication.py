"""Publication-only CPU tests; no runner/contract imports, private data or GPU calls."""
import copy
import json
import io
import sys
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import check
import stage


class PublicationTests(unittest.TestCase):
    def setUp(self):
        for name in ('subprocess.run', 'subprocess.Popen', 'os.kill'):
            self.enterContext(mock.patch(name, side_effect=AssertionError('offline only: no process/query/signal')))

    def test_workspace_and_publication_only_checks(self):
        result = check.verify()
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['original_files_verified_unchanged'], 76)
        self.assertFalse(result['original_authentication_reperformed'])
        published = check.verify(check_sources=False, check_private_originals=False)
        self.assertEqual(published['status'], 'passed')
        self.assertEqual(published['original_files_verified_unchanged'], 0)

    def test_checker_needs_no_private_raw_files_when_bundle_is_copied(self):
        with tempfile.TemporaryDirectory(prefix='paired-publication-offline-') as directory:
            scope = Path(directory)
            target = scope / 'publication_v1'
            shutil.copytree(check.HERE, target)
            with mock.patch.object(check, 'HERE', target), mock.patch.object(check, 'SCOPE', scope):
                result = check.verify(scope, check_sources=False, check_private_originals=False)
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(result['original_files_verified_unchanged'], 0)
            self.assertFalse((scope / 'artifacts-current-20261015-01').exists())

    def test_private_credential_fields_and_environment_dumps_rejected(self):
        for key in ('worker_token', 'launch_id', 'nonce', 'password', 'api_key', 'access_token', 'raw_maps', 'raw_text'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'private JSON'):
                check.no_credentials({key: 'test sentinel, not a credential'})
        with self.assertRaisesRegex(ValueError, 'environment dump'):
            check.no_credentials(dict(environment=dict(UNREVIEWED_ENV='test fixture')))
        check.no_credentials(dict(environment=dict(FERRULE_PROFILE_NVTX='0'), input_tokens=5, reference_ids=[1, 2]))

    def test_portable_regular_paths_reject_traversal_and_symlink(self):
        with tempfile.TemporaryDirectory(prefix='paired-publication-paths-') as directory:
            root = Path(directory)
            (root / 'safe.json').write_text('{}\n')
            self.assertEqual(check.regular(root, 'safe.json'), root / 'safe.json')
            for name in ('../safe.json', '/safe.json', './safe.json', ''):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    check.regular(root, name)
            (root / 'link.json').symlink_to(root / 'safe.json')
            with self.assertRaisesRegex(ValueError, 'symlink'):
                check.regular(root, 'link.json')

    def test_projection_cannot_drop_denominators_or_claim_zero_latency(self):
        original = check.projection('study.json')
        check.validate_denominators(original)
        altered = copy.deepcopy(original)
        altered['paired_blocks'].pop()
        with self.assertRaisesRegex(ValueError, 'paired blocks'):
            check.validate_denominators(altered)
        altered = copy.deepcopy(original)
        altered['client_e2e_seconds'] = 0
        with self.assertRaisesRegex(ValueError, 'stay null'):
            check.validate_denominators(altered)
        altered = copy.deepcopy(original)
        altered['denominators']['unissued'] = 108
        with self.assertRaisesRegex(ValueError, 'denominator'):
            check.validate_denominators(altered)

    def test_stage_generator_requires_cli_format_and_prints_only_literal_paths(self):
        with mock.patch.object(sys, 'stderr', new_callable=io.StringIO), self.assertRaises(SystemExit) as error:
            stage.main([])
        self.assertEqual(error.exception.code, 2)
        whitelist = check.load(check.HERE / 'STAGE_FILES.json')
        with mock.patch.object(sys, 'stdout', new_callable=io.StringIO) as output:
            self.assertEqual(stage.main(['--format', 'json']), 0)
        self.assertEqual(json.loads(output.getvalue()),
            [whitelist['project_prefix'] + '/' + name for name in whitelist['files']])
        with mock.patch.object(sys, 'stdout', new_callable=io.StringIO) as output:
            self.assertEqual(stage.main(['--format', 'tsv']), 0)
        self.assertEqual(output.getvalue(), (check.HERE / 'STAGE_FILES.tsv').read_text())

    def test_whitelist_excludes_private_raw_and_candidate_executable_artifacts(self):
        whitelist = check.load(check.HERE / 'STAGE_FILES.json')
        for name in whitelist['files']:
            self.assertNotIn('target/', name)
            self.assertFalse(name.startswith(('artifacts-', 'admission-')))
            self.assertNotIn(Path(name).suffix, {'.stderr', '.so', '.a', '.tar', '.pyc'})
        self.assertIn('publication_v1/test_publication.py', whitelist['files'])
        self.assertIn('run_v3.py', whitelist['files'])
        self.assertLess(whitelist['size_limit_bytes'], 20 << 20)

    def test_verbatim_startup_hash_and_projection_labels_are_distinct(self):
        manifest = check.load(check.HERE / 'manifest.json')
        startup = manifest['verbatim_startup']
        self.assertFalse(startup['redacted'])
        self.assertEqual(startup['original_sha256'], startup['publication_sha256'])
        self.assertEqual(startup['publication_sha256'], check.sha(check.HERE / 'startup.log'))
        projected = check.load(check.HERE / 'supervision.json')
        self.assertTrue(projected['redacted_projection'])
        self.assertFalse(projected['replaces_original_receipt'])
        self.assertNotEqual(projected['original_sources'][0]['original_sha256'], check.sha(check.HERE / 'supervision.json'))


if __name__ == '__main__':
    unittest.main()
