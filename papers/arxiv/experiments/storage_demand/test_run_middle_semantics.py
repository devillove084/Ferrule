"""Five fast CPU-only RQ4 record tests; fixtures are not runtime observations."""
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

os.environ.update(CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none', PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
with mock.patch('subprocess.Popen', side_effect=AssertionError('No process launch allowed')), \
        mock.patch('subprocess.run', side_effect=AssertionError('No command or GPU query allowed')):
    import run_middle as middle


def mock_owner_sample():
    """Explicit hypothetical final owner sample, never written to a run/receipt."""
    return dict(pending_requests_observed=0, pending_operations_observed=0,
        custody_unknown_observed=False, quarantine_observed=False,
        ledger_credits_outstanding_observed=0, fence_quiescent_observed=True,
        source_completion_observed=True)


class ShutdownSemantics(unittest.TestCase):
    def setUp(self):
        for obj, name in ((middle.subprocess, 'Popen'), (middle.subprocess, 'run'),
                          (middle.ct, 'exclusive_owner_lock'), (middle.ct, 'write'),
                          (middle, 'frozen'), (middle, 'cpu_contracts'),
                          (middle.observer, 'all_gpu_sample'), (middle.http, 'main')):
            patch = mock.patch.object(obj, name, side_effect=AssertionError('No run, validation retry or GPU access allowed'))
            patch.start()
            self.addCleanup(patch.stop)

    def test_exit_zero_with_null_ledgers_is_unknown(self):
        row = middle.shutdown_contract(0)
        self.assertTrue(row['process_exit_success'])
        self.assertTrue(row['normal_exit_observed'])
        self.assertEqual(row['resource_contract_complete'], 'unknown')
        self.assertNotIn('contract_complete', row)
        for field in mock_owner_sample():
            self.assertIsNone(row[field], field)
        self.assertEqual(set(row['resource_evidence_coverage']['missing_fields']), set(mock_owner_sample()))

    def test_failed_or_unobserved_exit_cannot_verify_complete_mock_evidence(self):
        for returncode in (1, -15, None):
            with self.subTest(returncode=returncode):
                row = middle.shutdown_contract(returncode, mock_owner_sample())
                self.assertEqual(row['resource_contract_complete'], 'unknown' if returncode is None else 'failed')
                self.assertIsNone(row['process_exit_success']) if returncode is None else self.assertFalse(row['process_exit_success'])

    def test_only_explicit_complete_mock_owner_sample_verifies(self):
        sample = mock.Mock(return_value=mock_owner_sample())
        row = middle.shutdown_contract(0, sample())
        sample.assert_called_once_with()
        self.assertEqual(row['resource_contract_complete'], 'verified')
        self.assertEqual(row['resource_evidence_coverage']['missing_fields'], [])
        self.assertEqual(row['resource_evidence_coverage']['invalid_fields'], [])
        self.assertEqual(row['resource_evidence_coverage']['failed_fields'], [])

    def test_each_missing_final_observation_remains_unknown_and_null(self):
        for field in mock_owner_sample():
            for present in (False, True):
                with self.subTest(field=field, explicit_null=present):
                    sample = mock_owner_sample()
                    if present:
                        sample[field] = None
                    else:
                        sample.pop(field)
                    row = middle.shutdown_contract(0, sample)
                    self.assertEqual(row['resource_contract_complete'], 'unknown')
                    self.assertIsNone(row[field])
                    self.assertEqual(row['resource_evidence_coverage']['missing_fields'], [field])

    def test_pending_unknown_quarantine_and_invalid_credits_fail_closed(self):
        changes = [('pending_requests_observed', 1), ('pending_operations_observed', 1),
            ('custody_unknown_observed', True), ('quarantine_observed', True),
            ('ledger_credits_outstanding_observed', 1), ('fence_quiescent_observed', False),
            ('source_completion_observed', False), ('ledger_credits_outstanding_observed', False),
            ('ledger_credits_outstanding_observed', -1), ('ledger_credits_outstanding_observed', float('nan')),
            ('pending_operations_observed', '0'), ('custody_unknown_observed', 0)]
        for field, value in changes:
            with self.subTest(field=field, value=value):
                sample = mock_owner_sample()
                sample[field] = value
                row = middle.shutdown_contract(0, sample)
                self.assertEqual(row['resource_contract_complete'], 'failed')
                coverage = row['resource_evidence_coverage']
                self.assertIn(field, coverage['invalid_fields'] + coverage['failed_fields'])


if __name__ == '__main__':
    unittest.main()
