"""CPU-only checks of the actual stopped run; never restarts it or queries a GPU."""
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from paired_read_v1 import current_binding_v2 as binding, run_actual_v2 as actual
from runnerhelpers.evidence_v2 import ct
import test_paired_read_v1 as shared

RUN = HERE / 'artifacts-current-20261015-01'


class StoppedRunTests(shared.CPUOnly):
    def test_failed_first_arm_keeps_all_denominators_and_no_replacement(self):
        final = ct.load(RUN / 'paired-final.json')
        self.assertEqual(final['status'], 'stopped')
        self.assertEqual(final['actual_issued_lifecycles'], 1)
        self.assertEqual(final['actual_completed_lifecycles'], 0)
        counts = final['analysis']['denominators']
        self.assertEqual((counts['scheduled'], counts['issued'], counts['successful'], counts['censored'], counts['unissued']),
                         (120, 0, 0, 0, 120))
        self.assertEqual(final['analysis']['measured_qa'], dict(scheduled=50, success=0, eligible_success=0))
        self.assertEqual(len(final['analysis']['blocks']), 5)
        self.assertIsNone(final['analysis']['aggregate'])
        with self.assertRaisesRegex(ValueError, 'permanently stopped'):
            actual.next_trial(RUN, ct.load(RUN / 'registration.json'))

    def test_missing_scientific_evidence_remains_null_not_fabricated_zero(self):
        closeout = ct.load(RUN / 'closeout.json')
        for key in ('client_e2e_seconds', 'client_ttft_seconds', 'quality_qualified_goodput_tokens_per_second',
                    'pipeline_positive_worker_read', 'pipeline_wasted_reads', 'pipeline_wasted_bytes',
                    'pipeline_host_credit', 'pipeline_wait_ns', 'pipeline_io_duration_ns'):
            self.assertIsNone(closeout[key], key)
        self.assertEqual(closeout['actual_http_token_id_join'], [])
        self.assertIn('unknown', closeout['final_owner_ledger'])
        self.assertFalse(closeout['scientific_acceptance'])
        self.assertFalse(closeout['cli_ready'])

    def test_authenticated_drain_and_final_idle_not_owner_release_certificate(self):
        receipt = ct.load(RUN / 'P1-blocking/supervisor.json')
        self.assertTrue(ct.drained(receipt))
        self.assertFalse(receipt['timed_out'])
        self.assertEqual(receipt['returncode'], 1)
        self.assertTrue(all(not entry.get('signals') for entry in receipt['custody']['entries']))
        idle = ct.load(RUN / 'final-all-eight-idle.json')
        self.assertEqual(idle['status'], 'verified')
        self.assertTrue(idle['authenticated_echild'])
        self.assertTrue(idle['recorded_owners_absent'])
        self.assertEqual(len(idle['accepted_sample_indices']), 3)
        release = ct.load(RUN / 'final-lock-release.json')
        self.assertTrue(release['all_eight_local_locks_released'])
        self.assertFalse(release['physical_release_certificate'])

    def test_actual_log_and_pinned_accessor_explain_pre_worker_failure(self):
        log = (RUN / 'P1-blocking/server.log').read_text()
        self.assertIn('prepared source profile changed', log)
        self.assertNotIn('cuInit', log)
        self.assertNotIn('OpenAI API listening', log)
        frozen = binding.QUALIFICATION / 'frozen-source/crates'
        cli = (frozen / 'ferrule-cli/src/commands/serve.rs').read_text()
        accessor = (frozen / 'ferrule-composition/src/model_factory/mixed_tensor_serving.rs').read_text()
        self.assertLess(cli.index('prepared.mixed_tp2_cache_requirements()?'), cli.index('spawn_model_worker_with(move'))
        self.assertIn('if self.prepared_source_plan.is_some()', accessor)
        self.assertIn('return Err(invalid("prepared source profile changed"))', accessor)

    def test_nfs_coverage_shared_and_no_http_windows(self):
        nfs = ct.load(RUN / 'closeout.json')['nfs']
        self.assertIsNone(nfs['physical_nas_backend_bytes'])
        self.assertIsNone(nfs['ready_startup_window'])
        self.assertEqual(nfs['http_request_windows'], [])
        self.assertFalse(nfs['negative_deltas_clamped'])
        self.assertEqual(len(nfs['raw_samples']), 3)
        self.assertEqual(nfs['shutdown_window']['status'], 'shared_client_delta')
        self.assertTrue(nfs['shutdown_window']['shared_client_traffic'])
        self.assertIsNone(nfs['shutdown_window']['owned_server_physical_bytes'])


if __name__ == '__main__':
    unittest.main()
