"""CPU-only actual-adapter and persistent no-replay regressions; no GPU/HTTP launch."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from paired_read_v1 import current_binding_v2 as binding, run_actual_v2 as actual, protocol as p
from runnerhelpers.evidence_v2 import ct
import test_paired_read_v1 as shared


class ActualAdapterTests(shared.CPUOnly):
    def test_actual_raw_lowercase_observation_separate_from_titlecase_factory_config(self):
        self.assertEqual(binding.raw_options('blocking')['schedule'], 'blocking')
        self.assertEqual(p.case_for_mode('blocking')['quality_test_config']['expert_read_pipeline']['schedule'], 'Blocking')
        with self.assertRaises(ValueError):
            binding.raw_options('disabled')
        observed = shared.LedgerTests.ledger(self)
        value = observed['closure']['attempts'][0]['summary']['physical']['expert_read_pipeline']['value']
        value.update(requested_mode='blocking', effective_mode='blocking', limits=binding.raw_options('blocking'))
        self.assertTrue(actual.ledger(observed, 'blocking')['can_continue'])
        self.assertFalse(actual.ledger(observed, 'lookahead')['can_continue'])
        value['status'] = 'join_pending_at_scope_end'
        self.assertFalse(actual.ledger(observed, 'blocking')['can_continue'])

    def test_omitted_summary_unknown_not_repaired_by_exit_or_idle(self):
        observed = dict(status='complete', capture=dict(returncode=0), closure=dict(attempts=[]))
        result = actual.ledger(observed, 'blocking')
        self.assertFalse(result['can_continue'])
        self.assertEqual(result['resource_contract'], 'unknown')
        self.assertIsNone(result['final_read_ledger'])

    def test_next_trial_fixed_order_successful_only_and_no_replay(self):
        registered = dict(trial_order=p.trial_order())
        self.assertEqual(actual.next_trial(self.directory, registered), 'P1-blocking')
        folder = self.directory / 'P1-blocking'
        folder.mkdir()
        ct.write(folder / 'attempt.json', dict(fixture_only=True))
        with self.assertRaisesRegex(ValueError, 'lost/incomplete'):
            actual.next_trial(self.directory, registered)
        ct.write(folder / 'assessment.json', dict(trial='P1-blocking', assessment=dict(can_continue=True)))
        self.assertEqual(actual.next_trial(self.directory, registered), 'P1-lookahead')
        actual.stop(self.directory, 'fixture failure')
        with self.assertRaisesRegex(ValueError, 'permanently stopped'):
            actual.next_trial(self.directory, registered)

    def test_failed_assessment_cannot_be_replaced_or_resumed(self):
        folder = self.directory / 'P1-blocking'
        folder.mkdir()
        ct.write(folder / 'attempt.json', dict(fixture_only=True))
        ct.write(folder / 'assessment.json', dict(trial='P1-blocking', assessment=dict(can_continue=False)))
        with self.assertRaisesRegex(ValueError, 'failed/unknown'):
            actual.next_trial(self.directory, dict(trial_order=p.trial_order()))

    def test_scoped_adapter_restores_original_baseline_after_cancel(self):
        before = p.validate_prepared
        entry = actual.planned.ENTRY
        with self.assertRaises(KeyboardInterrupt):
            with actual.actual_adapter():
                self.assertIs(p.validate_prepared, binding.validate_execution)
                self.assertEqual(actual.planned.ENTRY, actual.ENTRY)
                raise KeyboardInterrupt()
        self.assertIs(p.validate_prepared, before)
        self.assertEqual(actual.planned.ENTRY, entry)

    def test_current_actual_quality_receipts_read_only_mode_source_and_fingerprints(self):
        reports = {}
        for name in ('04-blocking-fresh', '06-blocking-ordinary'):
            reports[name] = ct.load(binding.QUALIFICATION / 'gates' / name / 'quality.json')
        prior = reports['04-blocking-fresh']
        path = binding.QUALIFICATION / 'gates/04-blocking-fresh/quality.json'
        binding.validate_reads(prior, 'blocking')
        binding.validate_ordinary(reports['06-blocking-ordinary'], prior, ct.sha(path), 'blocking')
        with self.assertRaises(ValueError):
            binding.validate_reads(prior, 'lookahead')
        wrong = copy.deepcopy(reports['06-blocking-ordinary'])
        wrong['diagnostic_primary_report_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'primary/config'):
            binding.validate_ordinary(wrong, prior, ct.sha(path), 'blocking')

    def test_bound_artifact_hash_is_not_old_adapter_declaration(self):
        self.assertNotEqual(binding.ADAPTER, 'ferrule.storage-demand.paired-nvtx-cli-build-adapter.v1')
        value = ct.load(binding.BINDING_PATH)
        self.assertFalse(value['old_adapter_acceptance_transferred'])
        self.assertFalse(value['cli_serving_accepted'])
        self.assertEqual(value['candidate']['binary_sha256'], binding.CLI_SHA)
        self.assertFalse(value['gpu_authorized_by_this_receipt'])


if __name__ == '__main__':
    unittest.main()
