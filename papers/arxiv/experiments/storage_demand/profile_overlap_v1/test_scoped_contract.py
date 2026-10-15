"""Synthetic scope tests, never substitutes for an actual profiled request."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from profile_overlap_v1 import analyze as a, protocol as p, scope_contract as s
from profile_overlap_v1.test_profile_overlap import fixture, GPID, PID, WORKER, COMPUTE

WRAPPER = GPID - (1 << 24)
GATES = dict.fromkeys(('one_frozen_qa_http_token_join', 'authenticated_model_pid_birth_source',
                     'ready_and_after_raw_maps', 'dynamic_owner_pair_closed', 'authenticated_echild',
                     'original_all_eight_stable_idle', 'collector_export_closed'), True)


class ScopedSQLTests(unittest.TestCase):
    def setUp(self):
        self.db, self.context = fixture()
        self.addCleanup(self.db.close)
        self.db.execute('INSERT INTO PROCESSES VALUES(?,?,?)', (WRAPPER, PID - 1, 'synthetic-controller'))
        self.context.update(scope_contract=s.policy('none'), model_source_generation=7,
            role_executable_sha256=dict(model='a' * 64, non_cuda_wrapper='a' * 64),
            role_source_binding_sha256=dict(model='b' * 64, non_cuda_wrapper='b' * 64),
            process_roles=[dict(role=role, globalPid=key, identity=dict(pid=pid, birth_ticks=birth, ppid=parent),
                                executable_sha256='a' * 64, source_binding_sha256='b' * 64)
                for role, key, pid, birth, parent in (('model', GPID, PID, 999, PID - 1),
                                                     ('non_cuda_wrapper', WRAPPER, PID - 1, 998, PID - 2))])

    def audit(self):
        return a.audit_sqlite(self.db, self.context)

    def warning(self, text, key=GPID, severity=2, source=3, timestamp_type=2):
        self.db.execute('INSERT INTO DIAGNOSTIC_EVENT VALUES(1,?,?,?,?,?)',
                        (timestamp_type, source, severity, text, key))

    def accept(self, report, gates=GATES):
        return s.acceptance(dict(report, log_issues=[]), gates)

    def test_positive_sql_is_not_lifecycle_acceptance(self):
        result = self.audit()
        self.assertTrue(result['scoped_sql_evidence_consistent'], result['scoped_issues'])
        self.assertFalse(result['scoped_overlap_accepted'])
        self.assertFalse(result['full_trace_accepted'])
        self.assertTrue(result['math_scope_criterion'])
        self.assertFalse(result['thread_state_analysis_required'])
        self.assertEqual(result['pread_kernel_intersection_ns'], 70)
        self.assertIsNone(result['dropped_event_count'])
        self.assertFalse(self.accept(result, {})['scoped_overlap_accepted'])
        self.assertTrue(self.accept(result)['scoped_overlap_accepted'])
        self.assertTrue(self.accept(result)['full_trace_accepted'])

    def test_exact_global_no_scheduling_is_outside_only_none_scope(self):
        self.warning(s.SCHEDULING, s.GLOBAL_DIAGNOSTIC_PID)
        result = self.audit()
        self.assertEqual(result['status'], 'rejected')
        self.assertTrue(result['scoped_sql_evidence_consistent'], result['scoped_issues'])
        self.assertEqual(len(result['warnings']), 1)
        self.assertTrue(self.accept(result)['scoped_overlap_accepted'])
        self.assertFalse(self.accept(result)['full_trace_accepted'])
        self.assertFalse(result['cpu_scheduling_claim'])
        self.assertFalse(result['active_io_claim'])
        self.context['scope_contract'] = s.policy('process-tree')
        self.assertFalse(self.audit()['scoped_sql_evidence_consistent'])

    def test_model_scoped_scheduling_warning_is_not_global(self):
        self.warning(s.SCHEDULING)
        self.assertFalse(self.audit()['scoped_sql_evidence_consistent'])

    def test_all_three_exact_authenticated_wrapper_warnings_retained(self):
        for message in s.NO_PAYLOAD:
            self.warning(message, WRAPPER)
        result = self.audit()
        self.assertTrue(result['scoped_sql_evidence_consistent'], result['scoped_issues'])
        self.assertEqual([r['text'] for r in result['warnings']], list(s.NO_PAYLOAD))
        self.assertTrue(self.accept(result)['scoped_overlap_accepted'])
        self.assertFalse(self.accept(result)['full_trace_accepted'])

    def test_every_no_payload_model_warning_rejects(self):
        for message in s.NO_PAYLOAD:
            with self.subTest(message=message):
                self.db.execute('SAVEPOINT model_warning')
                self.warning(message)
                self.assertFalse(self.audit()['scoped_sql_evidence_consistent'])
                self.db.execute('ROLLBACK TO model_warning')
                self.db.execute('RELEASE model_warning')

    def test_pid_list_alone_is_not_a_bound_wrapper_role(self):
        self.warning(s.NO_PAYLOAD[0], WRAPPER)
        self.context['process_roles'] = self.context['process_roles'][:1]
        self.assertFalse(self.audit()['scoped_sql_evidence_consistent'])

    def test_missing_birth_or_source_pin_or_wrong_vm_rejects(self):
        original = copy.deepcopy(self.context['process_roles'])
        for mutate in ('birth', 'source', 'vm', 'duplicate', 'model_role'):
            with self.subTest(mutate=mutate):
                self.context['process_roles'] = copy.deepcopy(original)
                role = self.context['process_roles'][1]
                if mutate == 'birth':
                    del role['identity']['birth_ticks']
                elif mutate == 'source':
                    role['source_binding_sha256'] = 'unknown'
                elif mutate == 'vm':
                    role['globalPid'] += 1 << 48
                elif mutate == 'duplicate':
                    self.context['process_roles'].append(copy.deepcopy(role))
                else:
                    self.context['process_roles'][0]['role'] = 'non_cuda_wrapper'
                self.assertFalse(self.audit()['scoped_sql_evidence_consistent'])

    def test_reused_model_birth_rejects(self):
        self.context['process_roles'][0]['identity']['birth_ticks'] += 1
        self.assertFalse(self.audit()['scoped_sql_evidence_consistent'])

    def test_unknown_warning_on_wrapper_is_not_an_exemption(self):
        self.warning('Different wrapper warning', WRAPPER)
        self.assertFalse(self.audit()['scoped_sql_evidence_consistent'])

    def test_warning_source_and_clock_kind_are_not_ignored(self):
        self.warning(s.NO_PAYLOAD[0], WRAPPER, source=99)
        self.assertFalse(self.audit()['scoped_sql_evidence_consistent'])
        self.db.execute('UPDATE DIAGNOSTIC_EVENT SET source=3,timestampType=99 WHERE timestamp=1')
        self.assertFalse(self.audit()['scoped_sql_evidence_consistent'])

    def test_loss_any_process_or_severity_rejects_matching_counts(self):
        for key in (GPID, WRAPPER, s.GLOBAL_DIAGNOSTIC_PID):
            for severity in (1, 2, 3):
                with self.subTest(key=key, severity=severity):
                    self.db.execute('SAVEPOINT loss')
                    self.warning('Dropped events: count unknown', key, severity)
                    report = self.audit()
                    self.assertEqual(report['loss_diagnostic_count'], 1)
                    self.assertFalse(report['scoped_sql_evidence_consistent'])
                    self.assertFalse(self.accept(report)['scoped_overlap_accepted'])
                    self.assertTrue(all(v == 0 for v in report['correlation_counts'].values()))
                    self.assertIsNone(report['dropped_event_count'])
                    self.db.execute('ROLLBACK TO loss')
                    self.db.execute('RELEASE loss')

    def test_unknown_diagnostic_severity_rejects(self):
        self.warning('unrecognized diagnostic', WRAPPER, severity=999)
        self.assertFalse(self.audit()['scoped_sql_evidence_consistent'])

    def test_wrapper_with_nvtx_payload_is_not_non_cuda_no_payload(self):
        self.warning(s.NO_PAYLOAD[0], WRAPPER)
        self.db.execute('INSERT INTO NVTX_EVENTS(start,eventType,globalTid,textId,domainId) VALUES(0,75,?,8,1)',
                        (WRAPPER | (PID - 1),))
        self.assertFalse(self.audit()['scoped_sql_evidence_consistent'])

    def test_role_model_equals_wrapper_pid_never_hides_model_warning(self):
        self.warning(s.NO_PAYLOAD[1])
        self.context['wrapper_pids'].append(PID)
        self.assertFalse(self.audit()['scoped_sql_evidence_consistent'])

    def test_both_directions_of_orphans_reject_scoped_acceptance(self):
        for table in ('CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_RUNTIME'):
            with self.subTest(table=table):
                self.db.execute('SAVEPOINT orphan')
                self.db.execute('DELETE FROM ' + table + ' WHERE correlationId=1')
                self.assertFalse(self.audit()['scoped_sql_evidence_consistent'])
                self.db.execute('ROLLBACK TO orphan')
                self.db.execute('RELEASE orphan')

    def test_owner_source_generation_must_match_all_range_generations(self):
        self.context['model_source_generation'] = 8
        self.assertFalse(self.audit()['scoped_sql_evidence_consistent'])

    def test_matching_activity_counts_cannot_replace_actual_read_ledger(self):
        self.context['read_stats']['io_started_reads'] = 2
        self.assertFalse(self.audit()['scoped_sql_evidence_consistent'])

    def test_every_independent_final_gate_is_required(self):
        result = self.audit()
        for gate in GATES:
            with self.subTest(gate=gate):
                accepted = self.accept(result, dict(GATES, **{gate: False}))
                self.assertFalse(accepted['scoped_overlap_accepted'])
                self.assertFalse(accepted['full_trace_accepted'])
                self.assertEqual(accepted['missing_gates'], [gate])

    def test_log_loss_cannot_be_overruled_by_sql_eligibility(self):
        report = dict(self.audit(), log_issues=['collector reports loss'])
        self.assertFalse(s.acceptance(report, GATES)['scoped_overlap_accepted'])

    def test_raw_summary_never_upgrades_historical_or_partial_trace(self):
        result = a.raw_summary(self.db, self.context)
        self.assertFalse(result['scoped_overlap_accepted'])
        self.assertFalse(result['scoped_sql_evidence_consistent'])
        self.assertFalse(result['full_trace_accepted'])
        self.assertIsNone(result['pread_kernel_intersection_ns'])

    def test_unknown_or_modified_scope_is_not_an_allowlist(self):
        self.context['scope_contract']['active_io_claim'] = True
        self.assertFalse(self.audit()['scoped_sql_evidence_consistent'])

    def test_kernel_shape_does_not_invent_missing_columns(self):
        report = self.audit()
        self.assertEqual(report['attributed_kernel_shapes'],
                         [dict(name='bf16_gemv', grid=[None] * 3, block=[None] * 3, count=2)])
        self.assertEqual(report['expert_range_counts_by_layer'], {3: 1})

    def test_unknown_pid_exact_wrapper_warning_is_not_exempt(self):
        self.warning(s.NO_PAYLOAD[1], WRAPPER - (1 << 24))
        self.assertFalse(self.audit()['scoped_sql_evidence_consistent'])

    def test_full_diagnostic_process_key_not_numeric_pid_or_tid(self):
        for key in (PID - 1, WRAPPER | (PID - 1), WRAPPER + (1 << 48)):
            with self.subTest(key=key):
                self.db.execute('SAVEPOINT key')
                self.warning(s.NO_PAYLOAD[0], key)
                self.assertFalse(self.audit()['scoped_sql_evidence_consistent'])
                self.db.execute('ROLLBACK TO key')
                self.db.execute('RELEASE key')

    def test_pinned_cli_sha_and_wrapper_source_not_arbitrary_hashes(self):
        self.context['process_roles'][0]['executable_sha256'] = 'c' * 64
        self.assertFalse(self.audit()['scoped_sql_evidence_consistent'])
        self.context['process_roles'][0]['executable_sha256'] = 'a' * 64
        self.context['process_roles'][1]['source_binding_sha256'] = 'c' * 64
        self.assertFalse(self.audit()['scoped_sql_evidence_consistent'])

    def test_model_parent_and_live_role_required(self):
        self.context['process_roles'][0]['identity']['ppid'] -= 1
        self.assertFalse(self.audit()['scoped_sql_evidence_consistent'])
        self.context['process_roles'][0]['identity']['ppid'] += 1
        self.context['process_roles'][1]['identity']['state'] = 'Z'
        self.assertFalse(self.audit()['scoped_sql_evidence_consistent'])

    def test_zero_dropped_is_not_invented_from_matching_rows(self):
        report = dict(self.audit(), log_issues=[], dropped_event_count=0)
        self.assertFalse(s.acceptance(report, GATES)['scoped_overlap_accepted'])

    def test_startup_kernel_closed_but_never_expert_compute(self):
        self.db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(10,20,0,?,9,1,0,NULL)', (COMPUTE,))
        self.db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(20,80,0,1,1,9,?,2,NULL)', (GPID,))
        report = self.audit()
        self.assertTrue(report['scoped_sql_evidence_consistent'], report['scoped_issues'])
        self.assertEqual(report['counts']['CUPTI_ACTIVITY_KIND_KERNEL'], 3)
        self.assertEqual(report['attributed_kernel_count'], 2)
        self.assertEqual(report['expert_kernel_union_ns'], 110)



class OwnerSafePlanTests(unittest.TestCase):
    def test_original_target_shape_and_fixed_process_tree_policy(self):
        case = Path('/synthetic/private')
        argv = p.profile_argv(case, 'a' * 64)
        self.assertEqual(argv[argv.index(p.PYTHON):], [p.PYTHON, '-S', '-B', str(p.HERE / 'run.py'),
            'target', '--directory', str(case), '--digest', 'a' * 64])
        self.assertIn('--sample=none', argv)
        self.assertIn('--cpuctxsw=process-tree', argv)
        self.assertIn('--kill=none', argv)
        self.assertFalse(hasattr(p, 'direct_profile_argv'))

    def test_new_plan_preserves_four_attempts_and_waits_for_fresh_go(self):
        with patch('subprocess.run', side_effect=AssertionError('no tool commands')):
            plan = p.plan()
        self.assertEqual(plan['generation'], 'artifacts-go-20261015-05')
        self.assertEqual(list(plan['predecessors'].values()), ['rejected', 'blocked', 'blocked', 'rejected'])
        self.assertFalse(plan['direct_cli_adapter_verified'])
        self.assertTrue(plan['owner_safe_wrapper_verified'])
        self.assertFalse(plan['execute_allowed'])
        self.assertEqual(plan['overlap_scope'], s.policy('process-tree'))
        self.assertFalse(hasattr(s, 'direct_shutdown_capability'))


if __name__ == '__main__':
    unittest.main()
