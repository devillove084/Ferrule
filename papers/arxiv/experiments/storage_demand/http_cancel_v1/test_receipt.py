"""Offline publication checks; no launch, device query, or inference."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import curate


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.value = curate.ct.load(curate.OUTPUT)

    def test_cancel_is_controlled_censor_not_quality_pass(self):
        cancel, recovery = self.value['requests']
        self.assertTrue(cancel['cancellation_accepted'])
        self.assertTrue(cancel['controlled_censor'])
        self.assertIsNone(cancel['quality_passed'])
        self.assertFalse(cancel['metric_eligible'])
        self.assertIsNone(cancel['client_usage'])
        self.assertIsNone(cancel['exact_eight_id_quality_pass'])
        self.assertTrue(recovery['exact_eight_id_quality_pass'])
        self.assertTrue(recovery['new_request_and_session'])
        self.assertEqual([cancel['worker_request_id'], recovery['worker_request_id']], [1, 2])

    def test_missing_request_ledger_never_inferred_zero(self):
        for row in self.value['requests']:
            self.assertIsNone(row['per_request_pending_operations'])
            self.assertIsNone(row['per_request_KV_snapshot'])
            self.assertIsNone(row['per_request_epoch'])
        self.assertIsNone(self.value['requests'][0]['previous_completed_invocation'])
        self.assertIsNone(self.value['final_owner']['per_request_ledger'])
        self.assertIsNone(self.value['final_owner']['physical_final_release'])
        self.assertFalse(self.value['scope']['Unknown_proved'])
        self.assertFalse(self.value['scope']['integrated_transaction_ledger_proved'])

    def test_exact_denominators_and_cleanup(self):
        self.assertEqual(self.value['actual_lifecycles'], 1)
        self.assertEqual(self.value['actual_requests'], 2)
        self.assertEqual(self.value['raw_event_counts']['request_worker_token_enqueued'], 9)
        self.assertEqual(self.value['summary']['accepted_cancellations'], 1)
        self.assertEqual(self.value['summary']['recovery_exact_pass'], 1)
        for key in ('authenticated_ECHILD', 'recorded_owners_absent', 'all_eight_stable_idle', 'all_eight_locks_released_and_reprobed'):
            self.assertTrue(self.value['cleanup'][key])
        self.assertEqual(self.value['cleanup']['required_consecutive'], 3)

    def test_known_credential_under_innocent_key_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'innocent.json'
            path.write_text('{"note":"private-authentication-fixture-only"}')
            with patch.object(curate, 'credentials', return_value={'private-authentication-fixture-only'}):
                with self.assertRaisesRegex(ValueError, 'private credential'):
                    curate.check_public_files([path])

    def test_original_raw_hash_names_are_relative_not_credentials(self):
        self.assertEqual(len(self.value['raw_sha256']), 63)
        for name, digest in self.value['raw_sha256'].items():
            self.assertFalse(Path(name).is_absolute())
            self.assertNotIn('..', Path(name).parts)
            self.assertRegex(digest, '^[0-9a-f]{64}$')


if __name__ == '__main__':
    unittest.main()
