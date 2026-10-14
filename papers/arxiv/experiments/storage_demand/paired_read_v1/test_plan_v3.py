"""Strictly CPU-only prospective V3 regressions; historical V2 bytes are not edited."""
import copy
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from paired_read_v1 import plan_v3 as v3, protocol as old, current_binding_v2 as v2
from runnerhelpers.evidence_v2 import ct
import test_paired_read_v1 as fixtures


class ProspectiveV3Tests(fixtures.CPUOnly):
    def cases(self):
        return ct.load(v3.CASES_PATH)

    def report(self, mode='blocking'):
        number = '04' if mode == 'blocking' else '05'
        return ct.load(v2.QUALIFICATION / 'gates' / (number + '-' + mode + '-fresh') / 'quality.json')

    def test_new_identity_unbound_source_no_launch_or_candidate_transfer(self):
        plan = v3.plan()
        self.assertEqual(plan['protocol_id'], v3.ID)
        self.assertNotEqual(plan['protocol_id'], ct.load(old.PROTOCOL_PATH)['id'])
        self.assertEqual(plan['actual_lifecycles'], 0)
        for key in ('candidate_binary', 'candidate_source_cut'):
            self.assertIsNone(plan[key])
        for key in ('sealed_final_candidate', 'gpu_authorized', 'launch_entry', 'bind_entry'):
            self.assertFalse(plan[key])
        self.assertEqual((plan['gpu_queries'], plan['gpu_locks'], plan['gpu_runs'], plan['server_runs']), (0, 0, 0, 0))
        for candidate in (ct.load(v2.BINDING_PATH), ct.load(v2.EXECUTION_PATH)):
            with self.assertRaisesRegex(ValueError, 'wrong protocol/case identity'):
                v3.require_v3_identity(candidate)
        namespace_only = dict(schema='ferrule.storage-demand.paired-read-candidate-input.v3', protocol_id=v3.ID,
            protocol_sha256=ct.sha(v3.PROTOCOL_PATH), cases_sha256=ct.sha(v3.CASES_PATH))
        self.assertFalse(v3.require_v3_identity(namespace_only)['candidate_qualified'])

    def test_render_explicit_cohort_flags_and_read_mode_only_arm_difference(self):
        cases = self.cases()
        v3.validate_cases(cases)
        for mode, case in cases.items():
            expected = ct.load(old.CASE_PATH)[mode]['argv_after_binary'] + [
                '--decode-cohort-target', '1', '--decode-cohort-max-deferrals', '0']
            self.assertEqual(case['argv_after_binary'], expected)
            self.assertEqual(v3.cli_scheduler(expected), v3.SCHEDULER)
            self.assertEqual(v3.flags(expected)['--expert-read-schedule'], mode)
        blocking, lookahead = (cases[mode] for mode in v3.MODES)
        differences = [(a, b) for a, b in zip(blocking['argv_after_binary'], lookahead['argv_after_binary']) if a != b]
        self.assertEqual(differences, [('blocking', 'lookahead')])
        self.assertEqual(blocking['environment'], lookahead['environment'])
        self.assertEqual(blocking['environment']['FERRULE_PROFILE_NVTX'], '0')

    def test_original_argv_public_defaults_and_configuration_rejection_are_not_rewritten(self):
        previous = ct.load(old.CASE_PATH)
        old.validate_cases(previous)
        for mode in v3.MODES:
            argv = previous[mode]['argv_after_binary']
            self.assertNotIn('--decode-cohort-target', argv)
            self.assertNotIn('--decode-cohort-max-deferrals', argv)
            scheduler = v3.cli_scheduler(argv)
            self.assertEqual((scheduler['decode_cohort_target'], scheduler['decode_cohort_max_deferrals']), (4, 3))
            with self.assertRaises(v3.ConfigurationMismatch) as caught:
                v3.compare_report_configuration(previous[mode], self.report(mode))
            self.assertEqual(set(caught.exception.differences), {'cli.decode_cohort_target', 'cli.decode_cohort_max_deferrals'})
        self.assertEqual(v3.CLI_COHORT_DEFAULTS, dict(decode_cohort_target=4, decode_cohort_max_deferrals=3))

    def test_all_nine_scheduler_fields_checked_at_cli_test_and_each_factory_endpoint(self):
        self.assertEqual(len(v3.SCHEDULER), 9)
        case, report = self.cases()['blocking'], self.report()
        result = v3.compare_report_configuration(case, report)
        self.assertEqual(result['scheduler_fields_compared'], sorted(v3.SCHEDULER))
        self.assertFalse(result['candidate_qualified'])
        self.assertFalse(result['v3_quality_pass'])
        for field, expected in v3.SCHEDULER.items():
            for location in ('test_config', 'requested', 'effective'):
                altered = copy.deepcopy(report)
                row = altered['test_config']['capacity_and_options']['scheduler'] if location == 'test_config' \
                      else altered['effective_config'][location]['scheduler']
                row[field] = not expected if type(expected) is bool else expected + 1
                with self.subTest(field=field, location=location), self.assertRaises(v3.ConfigurationMismatch):
                    v3.compare_report_configuration(case, altered)
            cli = copy.deepcopy(v3.SCHEDULER)
            cli[field] = not expected if type(expected) is bool else expected + 1
            with self.subTest(field=field, location='cli'), mock.patch.object(v3, 'cli_scheduler', return_value=cli), \
                    self.assertRaises(v3.ConfigurationMismatch):
                v3.compare_report_configuration(case, report)

    def test_missing_extra_and_wrong_scheduler_types_fail_closed(self):
        for kind in ('missing', 'extra', 'wrong_type'):
            scheduler = copy.deepcopy(v3.SCHEDULER)
            if kind == 'missing':
                scheduler.pop('prefill_reserve_tokens')
            elif kind == 'extra':
                scheduler['unreviewed'] = 0
            else:
                scheduler['decode_cohort_target'] = True
            with self.subTest(kind=kind), self.assertRaises(v3.ConfigurationMismatch):
                v3.compare_scheduler(scheduler)
        duplicate = self.cases()['blocking']['argv_after_binary'] + ['--decode-cohort-target', '1']
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            v3.cli_scheduler(duplicate)

    def test_explicit_bounds_and_literal_cohort_tokens_required(self):
        for flag in ('--decode-cohort-target', '--decode-cohort-max-deferrals', '--expert-read-host-mb',
                     '--cuda-expert-device-bytes', '--max-active-sequences', '--max-batch-tokens'):
            cases = self.cases()
            argv = cases['blocking']['argv_after_binary']
            argv[argv.index(flag) + 1] = '4' if flag == '--decode-cohort-target' else '3'
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                v3.validate_cases(cases)
        cases = self.cases()
        cases['blocking']['argv_after_binary'] = cases['blocking']['argv_after_binary'][:-4]
        with self.assertRaisesRegex(ValueError, 'explicit cohort'):
            v3.validate_cases(cases)
        for invalid in ('disabled', 'auto', None):
            with self.assertRaises(ValueError):
                v3.expected_case(invalid)

    def test_historical_budget_oracles_qa_text_ids_and_workload_tokens_unchanged(self):
        originals = ct.load(old.CASE_PATH)
        for mode, case in self.cases().items():
            for field in ('limits', 'model', 'topology', 'read_options', 'quality_test_config',
                          'environment', 'request_ceiling', 'cache_bytes_include_scratch', 'pagecache_policy'):
                self.assertEqual(case[field], originals[mode][field], field)
            opts = case['quality_test_config']['capacity_and_options']
            self.assertEqual((opts['max_experts'], opts['max_bytes'], opts['scratch_bytes'], opts['device_bytes']),
                             (64, 320 << 20, 8 << 20, 7 << 30))
            self.assertEqual((case['read_options']['max_host_bytes'], case['read_options']['max_metadata_bytes'],
                              case['read_options']['worker_stack_bytes']), (64 << 20, 8 << 20, 2 << 20))
        references = ct.load(v3.PROTOCOL_PATH)['reference_bindings']
        for name in ('requests.json', 'oracles.json', 'order.json'):
            self.assertEqual(references[name], ct.sha(HERE / name))
        qa = ct.load(HERE / 'oracles.json')['qa']
        self.assertEqual(qa['input_ids'], [760, 6511, 314, 6124, 369])
        self.assertEqual(qa['reference_ids'], [25358, 13, 198, 760, 6511, 314, 6124, 369])
        self.assertEqual(qa['reference_text'], ' Tokyo.\nThe capital of Japan is')

    def test_same_declared_seed_order_n5_and_no_transfer_of_failed_denominators(self):
        plan = v3.plan()
        self.assertEqual(plan['order'], old.expected_order())
        self.assertEqual(plan['trial_order'], old.trial_order())
        self.assertEqual((plan['scheduled_blocks'], plan['scheduled_starts'], plan['scheduled_requests'],
                          plan['scheduled_measured_qa']), (5, 10, 120, 50))
        self.assertEqual((plan['limits']['pilot_seconds'], plan['limits']['outer_lifecycle_seconds'],
                          plan['limits']['study_cleanup_reserve_seconds']), (7200, 650, 100))
        self.assertFalse(plan['historical_study']['replacement_or_resume'])
        self.assertFalse(plan['historical_study']['timing_outcomes_observed'])
        prior = ct.load(HERE / 'artifacts-current-20261015-01/paired-final.json')
        self.assertEqual(prior['status'], 'stopped')
        self.assertEqual(prior['analysis']['denominators']['unissued'], 120)
        self.assertEqual(prior['analysis']['measured_qa']['scheduled'], 50)
        self.assertEqual(prior['actual_issued_lifecycles'], 1)
        self.assertEqual(plan['actual_lifecycles'], 0)

    def test_v2_still_valid_and_append_only_expected_semantic_failure(self):
        self.assertEqual(v2.validate_binding()['status'], 'current_nine_gate_bound_cli_serving_pending')
        self.assertEqual(v2.validate_execution()['status'], 'preregistered_actual_binding_ready_owner_authorization_required')
        audit = v3.audit_v2()
        self.assertTrue(audit['v2_binding_digest_valid'])
        self.assertTrue(audit['v2_execution_digest_and_workflow_pins_valid'])
        self.assertFalse(audit['v3_candidate_qualified'])
        self.assertFalse(audit['old_files_changed'])
        for errors in audit['differences'].values():
            self.assertEqual(errors['cli.decode_cohort_target'], dict(actual=4, expected=1))
            self.assertEqual(errors['cli.decode_cohort_max_deferrals'], dict(actual=3, expected=0))
        path = self.directory / 'append-only-rejection.json'
        ct.write(path, audit)
        with self.assertRaises(FileExistsError):
            ct.write(path, audit)

    def test_v3_cpu_checker_rejects_previous_protocol_seals(self):
        for path in (old.PREPARED_PATH, v2.EXECUTION_PATH, v2.BINDING_PATH):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'unknown/stale seal'):
                v3.validate_prepared(path)

    def test_all_68_preexisting_files_byte_identical(self):
        preserved = ct.load(v3.PROTOCOL_PATH)['historical_preservation']
        self.assertEqual(preserved['files'], 68)
        self.assertEqual(ct.digest(preserved['artifact_pins']), preserved['inventory_digest'])
        ct.check_pins(preserved['artifact_pins'])


if __name__ == '__main__':
    unittest.main()
