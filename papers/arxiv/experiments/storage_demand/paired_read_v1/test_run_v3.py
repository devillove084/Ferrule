"""CPU-only integration regressions; fixtures are never candidate PASS receipts."""
from contextlib import nullcontext
import copy
import io
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from paired_read_v1 import current_binding_v3 as binding, run_v3 as run, plan_v3 as plan
from paired_read_v1 import protocol as old, current_binding_v2 as prior
from runnerhelpers.evidence_v2 import ct
import test_paired_read_v1 as fixtures


class V3IntegrationTests(fixtures.CPUOnly):
    def candidate(self):
        return dict(candidate=dict(binary=str(self.directory / 'fixture-not-executable')),
            uuid_inventory=[f'GPU-00000000-0000-0000-0000-00000000000{i}' for i in range(8)])

    def test_actual_maintained_spec_uses_v3_cohort_argv_both_arms(self):
        before = old.case_for_mode
        with run.adapter():
            for mode in plan.MODES:
                spec = run.planned.spec_for(self.candidate(), self.candidate()['uuid_inventory'][0], mode)
                self.assertEqual(spec['argv'][1:], binding.case_for_mode(mode)['argv_after_binary'])
                self.assertEqual(spec['argv'][-4:], plan.COHORT_FLAGS)
                self.assertEqual(plan.cli_scheduler(spec['argv'][1:]), plan.SCHEDULER)
                self.assertEqual(spec['environment']['FERRULE_PROFILE_NVTX'], '0')
                self.assertEqual(spec['argv'][spec['argv'].index('--expert-read-schedule') + 1], mode)
            self.assertEqual(run.planned.ENTRY, run.ENTRY)
            self.assertIs(old.validate_candidate, binding.validate_binding)
        self.assertIs(old.case_for_mode, before)
        self.assertEqual(old.CASE_PATH.name, 'cases.json')

    def test_nested_adapter_restores_after_cancellation(self):
        before = (old.PROTOCOL_PATH, old.validate_prepared, run.maintained.binding, ct.write)
        with self.assertRaises(KeyboardInterrupt):
            with run.adapter():
                with run.adapter():
                    self.assertEqual(old.PROTOCOL_PATH, plan.PROTOCOL_PATH)
                    raise KeyboardInterrupt()
        self.assertEqual((old.PROTOCOL_PATH, old.validate_prepared, run.maintained.binding, ct.write), before)

    def test_real_freeze_identity_is_exact_not_candidate_pass(self):
        identity = binding.candidate_identity(binding.QUALIFICATION)
        self.assertEqual(identity, binding.EXPECTED_IDENTITY)
        self.assertNotEqual(identity['commit'], prior.COMMIT)
        self.assertNotEqual(identity['cli_binary_sha256'], prior.CLI_SHA)
        self.assertFalse(binding.BINDING_PATH.exists())
        self.assertFalse(binding.EXECUTION_PATH.exists())

    def test_unknown_source_archive_or_cli_candidate_fails_closed(self):
        original_load = ct.load
        for field in ('commit', 'source_pins_digest', 'archive_sha256', 'file_count', 'sha256'):
            def altered(path, field=field):
                value = copy.deepcopy(original_load(path))
                if (Path(path).name == 'source-freeze.json' and field != 'sha256') or (
                        Path(path).name == 'artifact-cli.json' and field == 'sha256'):
                    value[field] = 1 if field == 'file_count' else 'f' * (40 if field == 'commit' else 64)
                return value
            with self.subTest(field=field), mock.patch.object(ct, 'load', side_effect=altered), self.assertRaises(ValueError):
                binding.candidate_identity(binding.QUALIFICATION)
        with self.assertRaisesRegex(ValueError, 'unknown source candidate'):
            binding.candidate_identity(prior.QUALIFICATION)

    def test_pending_candidate_cannot_write_review_binding_or_execution(self):
        with mock.patch.object(binding, 'candidate_identity', return_value=binding.EXPECTED_IDENTITY), \
                mock.patch.object(plan, 'validate_prepared'), mock.patch.object(Path, 'is_file', return_value=False), \
                mock.patch.object(binding, 'reviewer') as reviewer, self.assertRaisesRegex(ValueError, 'candidate pending'):
            binding.review(binding.QUALIFICATION)
        reviewer.assert_not_called()
        self.assertFalse(binding.REVIEW_PATH.exists())
        self.assertFalse(binding.BINDING_PATH.exists())
        self.assertFalse(binding.EXECUTION_PATH.exists())

    def test_explicit_binding_cli_qualification_argument_is_required(self):
        with mock.patch.object(ct, 'require_cpu_python'), self.assertRaisesRegex(ValueError, '--qualification'):
            binding.main(['review-and-bind'])
        with mock.patch.object(ct, 'require_cpu_python'), mock.patch.object(binding, 'review', side_effect=ValueError('fixture reviewed argument')) as review:
            with self.assertRaisesRegex(ValueError, 'fixture reviewed argument'):
                binding.main(['review-and-bind', '--qualification', str(binding.QUALIFICATION)])
            review.assert_called_once_with(binding.QUALIFICATION)

    def test_reviewer_reuses_semantics_but_isolates_old_candidate_globals(self):
        module = binding.reviewer(binding.QUALIFICATION, binding.EXPECTED_IDENTITY)
        self.assertEqual(module.CLI_SHA, binding.EXPECTED_IDENTITY['cli_binary_sha256'])
        self.assertEqual(prior.CLI_SHA, 'de5159951fe724e97e35c0fc86367642d33735994328eee1a5e604cf9d2b037b')
        self.assertEqual(module.review_build.__code__.co_code, prior.review_build.__code__.co_code)
        self.assertIsNot(module.review_build.__globals__, prior.review_build.__globals__)
        self.assertEqual(module.prereg.case_for_mode('blocking')['argv_after_binary'][-4:], plan.COHORT_FLAGS)
        self.assertIs(module.ORIGINAL_VALIDATE_PREPARED, plan.validate_prepared)

    def test_v2_seals_and_historical_directory_cannot_resume_as_v3(self):
        for path, check in ((prior.BINDING_PATH, binding.validate_binding), (prior.EXECUTION_PATH, binding.validate_execution)):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'unknown/stale seal'):
                check(path)
        with self.assertRaisesRegex(ValueError, 'no historical V2 resume'):
            run.directory_check(HERE / 'artifacts-current-20261015-01')
        with self.assertRaisesRegex(ValueError, 'renewed'):
            run.start(HERE / 'artifacts-v3-fixture', False)

    def test_plan_cli_is_cpu_only_and_does_not_rewrite_the_preregistration(self):
        with mock.patch.object(ct, 'require_cpu_python'), mock.patch.object(sys, 'stdout', new_callable=io.StringIO) as output:
            self.assertEqual(run.main(['plan']), 0)
        value = json.loads(output.getvalue())
        self.assertEqual(value['protocol_id'], plan.ID)
        self.assertEqual(value['versioned_execution_entry'], str(run.ENTRY))
        self.assertTrue(value['integration_does_not_amend_preregistration'])
        self.assertEqual(value['actual_lifecycles'], 0)
        self.assertIsNone(value['candidate_binary'])
        self.assertEqual((value['scheduled_blocks'], value['scheduled_starts'], value['scheduled_requests'], value['scheduled_measured_qa']), (5, 10, 120, 50))
        self.assertEqual((value['limits']['pilot_seconds'], value['limits']['outer_lifecycle_seconds'], value['limits']['study_cleanup_reserve_seconds']), (7200, 650, 100))

    def test_authenticated_worker_entry_dispatches_maintained_worker(self):
        def worker(launch_hash):
            self.assertEqual(launch_hash, 'fixture-launch-hash')
            self.assertEqual(run.planned.ENTRY, run.ENTRY)
            self.assertEqual(old.case_for_mode('lookahead')['argv_after_binary'][-4:], plan.COHORT_FLAGS)
            return 0
        with mock.patch.object(ct, 'require_cpu_python'), mock.patch.object(run.planned, 'worker', side_effect=worker) as dispatched:
            self.assertEqual(run.main(['--_worker', 'fixture-launch-hash']), 0)
        dispatched.assert_called_once_with('fixture-launch-hash')

    def test_actual_finite_coordinator_payload_has_all12_and_v3_cohorts(self):
        candidate = self.candidate()
        directory = self.directory / 'fixture-journal'
        directory.mkdir()
        schedule = ct.load(HERE / 'requests.json')['schedule']
        prepared = plan.validate_prepared()
        registered = dict(trial_order=prepared['trial_order'], order=prepared['order'], limits=prepared['limits'],
            schedule=schedule, pilot_deadline=10000, selected_uuid=candidate['uuid_inventory'][0],
            eight_uuids=candidate['uuid_inventory'], binding_path=str(self.directory / 'fixture-binding'),
            execution_path=str(self.directory / 'fixture-execution'), binding_sha256='a' * 64,
            execution_sha256='b' * 64)
        ct.write(directory / 'registration.json', dict(fixture_only=True))
        execution = dict(artifact_pins={'fixture-not-a-receipt': 'c' * 64})
        payloads = []
        def supervise(payload, seconds):
            payloads.append(copy.deepcopy(payload))
            self.assertEqual(seconds, 530)
            return dict(fixture_only=True)
        summary = fixtures.synthetic_arm('P1-blocking')['summary']
        with mock.patch.object(run, 'registration', return_value=registered), \
                mock.patch.object(binding, 'validate_binding', return_value=candidate), \
                mock.patch.object(binding, 'validate_execution', return_value=execution), \
                mock.patch.object(ct, 'exclusive_owner_lock', side_effect=lambda uuid: nullcontext(dict(fixture_uuid=uuid))), \
                mock.patch.object(ct, 'validate_owner_lock'), mock.patch.object(ct, 'drained', return_value=True), \
                mock.patch.object(run.maintained.pilot.telemetry, 'authorize'), \
                mock.patch.object(run.maintained.pilot, 'idle', return_value=dict(status='verified')), \
                mock.patch.object(run.maintained.pilot, 'baseline', return_value={}), \
                mock.patch.object(run.maintained.pilot, 'remaining'), \
                mock.patch.object(run.maintained.pilot.idle_window, 'no_recorded_owner', return_value=True), \
                mock.patch.object(run.planned, 'supervise', side_effect=supervise), \
                mock.patch.object(run.maintained.ex, 'lifecycle_assessment', return_value=dict(can_continue=True, stop_reasons=[])), \
                mock.patch.object(run.maintained.ex, 'recover_requests', side_effect=lambda *args: copy.deepcopy(summary)), \
                mock.patch.object(run.maintained, 'ledger', return_value=dict(can_continue=True)), \
                mock.patch.object(run.time, 'monotonic', return_value=100):
            with run.adapter():
                row = run.maintained.execute_next(directory)
        self.assertTrue(row['assessment']['can_continue'])
        self.assertEqual(len(payloads), 1)
        payload = payloads[0]
        self.assertEqual(payload['trial'], 'P1-blocking')
        self.assertEqual(payload['spec']['argv'][-4:], plan.COHORT_FLAGS)
        self.assertEqual(payload['request_ceiling'], 12)
        self.assertEqual(payload['schedule'], schedule)
        self.assertEqual(len(payload['schedule']), 12)
        self.assertEqual(sum(r['measured'] for r in schedule), 5)
        self.assertEqual(payload['outer_deadline'], 750)
        self.assertEqual(payload['pilot_deadline'], 9900)
        self.assertEqual(payload['contract_sha256'], ct.digest({k: payload[k] for k in run.planned.CONTRACT_KEYS}))
        self.assertEqual(len(payload['all_uuid_locks']), 8)

    def test_exact_source_cli_lowering_and_all_nine_gate_selectors(self):
        lowering = binding.review_lowering(binding.QUALIFICATION)
        self.assertEqual(lowering['scheduler_fields'], sorted(plan.SCHEDULER))
        self.assertTrue(lowering['public_defaults_unchanged'])
        gates = ct.load(prior.QUALIFICATION / 'qualification-plan-v2.json')['gates']
        binding.validate_gate_plan(gates)
        for field, value in (('selector', 'unknown'), ('role', 'cli'), ('schedule', 'blocking'), ('seconds', 1)):
            altered = copy.deepcopy(gates)
            altered[0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'unknown nine-gate'):
                binding.validate_gate_plan(altered)

    def test_failed_finalization_retains_all_slots_and_produces_v3_not_v2(self):
        directory = self.directory / 'artifacts-v3-fixture'
        directory.mkdir()
        registered = dict(schedule=ct.load(HERE / 'requests.json')['schedule'],
            trial_order=old.trial_order(), order=old.expected_order(), started=dict(fixture_only=True),
            pilot_deadline=10000, binding_sha256='a' * 64, execution_sha256='b' * 64,
            binding_path=str(self.directory / 'fixture-binding'), selected_uuid=self.candidate()['uuid_inventory'][0],
            eight_uuids=self.candidate()['uuid_inventory'])
        run.maintained.stop(directory, 'fixture failed startup; no retry')
        with mock.patch.object(run, 'HERE', self.directory), mock.patch.object(run, 'registration', return_value=registered):
            with run.adapter():
                result = run.maintained.finalize(directory)
        final = ct.load(directory / 'paired-final.json')
        self.assertEqual(final['schema'], 'ferrule.storage-demand.paired-read-actual.v3')
        self.assertEqual(final['protocol_id'], plan.ID)
        self.assertFalse(final['historical_resume_permitted'])
        self.assertEqual(result['status'], 'stopped')
        self.assertEqual(len(final['lifecycles']), 10)
        self.assertEqual(final['analysis']['denominators']['unissued'], 120)
        self.assertEqual(final['analysis']['measured_qa']['scheduled'], 50)
        self.assertEqual(len(final['analysis']['blocks']), 5)
        self.assertIsNone(final['analysis']['aggregate'])
        with self.assertRaisesRegex(ValueError, 'permanently stopped'):
            run.maintained.next_trial(directory, registered)

    def test_source_review_rejects_extra_unarchived_members(self):
        source = ct.load(binding.QUALIFICATION / 'source-freeze.json')
        frozen = Path(source['frozen_source'])
        expected = {str(frozen / name) for name in source['pins_relative_to_frozen_source']}
        with mock.patch.object(binding, 'inventory_tree', return_value=expected | {str(frozen / 'unexpected.pyc')}), \
                self.assertRaisesRegex(ValueError, 'source coverage differs'):
            binding.review_source(binding.QUALIFICATION, binding.EXPECTED_IDENTITY)

    def test_preserved76_originals_remain_byte_identical(self):
        manifest = ct.load(HERE / 'publication_v1/manifest.json')
        self.assertEqual(manifest['preserved_original_file_count'], 76)
        for name, pin in manifest['preserved_originals'].items():
            self.assertEqual(ct.sha(HERE / name), pin['original_sha256'], name)


if __name__ == '__main__':
    unittest.main()
