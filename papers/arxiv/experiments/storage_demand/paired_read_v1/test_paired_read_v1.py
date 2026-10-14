"""CPU-only dedicated regressions. Every process/GPU/HTTP side effect is mocked."""
from contextlib import ExitStack, nullcontext
import copy
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.dont_write_bytecode = True
STORAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STORAGE))
sys.path.insert(0, str(STORAGE / 'tests'))
from paired_read_v1 import protocol as p, run_paired as run, analysis, build_adapter, profile_audit, quality
from runnerhelpers.evidence_v2 import ct, censoring_summary
from runnerhelpers import execution_v2 as ex
from runnerhelpers import build_receipts_v1 as original_adapter
import test_execution_v2 as fixtures
from test_build_receipts_v1 import BuildFixture


class CPUOnly(unittest.TestCase):
    def setUp(self):
        self.stack = self.enterContext(ExitStack())
        for name in ('subprocess.Popen', 'subprocess.run', 'os.kill'):
            self.stack.enter_context(mock.patch(name, side_effect=AssertionError('CPU only: no launch/query/signal')))
        self.stack.enter_context(mock.patch.object(ct, 'exclusive_owner_lock', side_effect=AssertionError('CPU only: no lock')))
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory(prefix='paired-read-cpu-')))


def synthetic_arm(trial, qualify=True, latency=2.0):
    schedule = ct.load(p.HERE / 'requests.json')['schedule']
    observed = []
    for planned in schedule:
        observed.append(dict(planned, issued=True, status='success', metric_eligible=True,
            actual=dict(request_e2e_seconds=latency, ttft_seconds=latency / 2,
                        usage=dict(completion_tokens=planned['max_tokens']))))
    summary = censoring_summary(observed, schedule)
    if not qualify:
        summary['measured_qa']['eligible_success'] = 0
    return dict(trial=trial, summary=summary, assessment=dict(can_continue=qualify),
        read_ledger=dict(can_continue=qualify), lifecycle=dict(status='complete',
            requests=dict(observations=observed)))


class DesignTests(CPUOnly):
    def test_fixed_seed_order_and_complete_denominators(self):
        value = p.validate_design()
        self.assertEqual([r['modes'] for r in p.expected_order()], [
            ['blocking', 'lookahead'], ['blocking', 'lookahead'], ['lookahead', 'blocking'],
            ['lookahead', 'blocking'], ['blocking', 'lookahead']])
        self.assertEqual(len(p.trial_order()), 10)
        self.assertEqual(value['scheduled_measured_qa'], 50)
        with self.assertRaisesRegex(ValueError, 'seed'):
            p.expected_order(0)

    def test_mode_is_the_only_cli_difference_and_budgets_stay_fixed(self):
        b, l = (p.case_for_mode(mode) for mode in p.MODES)
        self.assertEqual([(x, y) for x, y in zip(b['argv_after_binary'], l['argv_after_binary']) if x != y],
                         [('blocking', 'lookahead')])
        self.assertEqual(b['environment'], l['environment'])
        self.assertEqual(b['environment']['FERRULE_PROFILE_NVTX'], '0')
        p.validate_cases(dict(blocking=b, lookahead=l))
        for bad in ('disabled', 'auto', 'unknown', None):
            with self.subTest(mode=bad), self.assertRaises(ValueError):
                p.case_for_mode(bad)
        for flag in ('--expert-read-host-mb', '--expert-read-metadata-mb', '--expert-read-worker-stack-mb',
                     '--cuda-expert-device-bytes', '--cuda-device-budget-mb'):
            altered = copy.deepcopy(dict(blocking=b, lookahead=l))
            argv = altered['lookahead']['argv_after_binary']
            argv[argv.index(flag) + 1] = '1'
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, 'budgets'):
                p.validate_cases(altered)

    def test_qa_token_text_and_eight_oracle(self):
        qa = ct.load(p.HERE / 'oracles.json')['qa']
        self.assertEqual(qa['input_ids'], [760, 6511, 314, 6124, 369])
        self.assertEqual(len(qa['reference_ids']), 8)
        self.assertEqual(qa['reference_text'], ' Tokyo.\nThe capital of Japan is')

    def test_cleanup_reserve_and_no_shortened_lifecycle(self):
        self.assertTrue(run.can_admit(750, now=0))
        self.assertFalse(run.can_admit(749.999, now=0))
        self.assertTrue(run.can_admit(1400, lifecycles=2, now=0))
        self.assertFalse(run.can_admit(1399, lifecycles=2, now=0))
        with self.assertRaises(ValueError):
            run.can_admit(10000, lifecycles=3, now=0)

    def test_source_hash_and_seal_are_fail_closed(self):
        pins = ct.load(p.HERE / 'source-cut.json')['production_source_pins']
        wrong = dict(pins)
        wrong[next(iter(wrong))] = '0' * 64
        with mock.patch.object(ct, 'source_pins', return_value=wrong), self.assertRaisesRegex(ValueError, 'inventory'):
            p.validate_design()
        path = self.directory / 'seal.json'
        ct.write(path, dict(schema='ferrule.storage-demand.paired-read-cpu-prepared.v1', digest='0' * 64))
        with self.assertRaisesRegex(ValueError, 'stale'):
            p.validate_prepared(path)

    def test_cpu_plan_has_no_candidate_or_side_effects(self):
        value = p.plan()
        self.assertEqual((value['gpu_queries'], value['gpu_locks'], value['gpu_runs']), (0, 0, 0))
        self.assertIsNone(value['candidate_binary'])
        self.assertFalse(value['sealed_final_candidate'])
        self.assertFalse(value['profile_launch_entry'])

    def test_unbound_candidate_blocks_before_output_lock_or_query(self):
        path = self.directory / 'pending.json'
        ct.write(path, dict(schema='pending', status='pending'))
        args = SimpleNamespace(prepared=self.directory / 'cpu.json', candidate=path,
                               output=self.directory / 'artifacts-not-created')
        with mock.patch.object(p, 'validate_prepared', return_value={}):
            with self.assertRaises(ValueError):
                run.run(args)
        self.assertFalse(args.output.exists())

    def test_scoped_adapter_restores_baseline_even_after_cancellation(self):
        before = {k: getattr(run.pilot, k) for k in ('HERE', 'ENTRY', 'worker_contract')}
        with self.assertRaises(KeyboardInterrupt):
            with run.maintained_worker_adapter():
                self.assertEqual(run.pilot.ENTRY, run.ENTRY)
                self.assertIs(run.pilot.worker_contract, run.worker_contract)
                raise KeyboardInterrupt()
        for k, v in before.items():
            self.assertIs(getattr(run.pilot, k), v)


class AnalysisTests(CPUOnly):
    def report(self, failed=None):
        return dict(order=p.expected_order(), lifecycles=[synthetic_arm(trial,
            qualify=trial != failed, latency=1.0 if trial.endswith('-lookahead') else 2.0)
            for trial in p.trial_order()])

    def test_paired_log_ratio_and_qualified_goodput_not_request_bootstrap(self):
        result = analysis.summarize(self.report())
        e2e = result['aggregate']['request_e2e_seconds']
        self.assertAlmostEqual(e2e['ratio'], 0.5)
        self.assertEqual(e2e['interval_ratio'], [0.5, 0.5])
        self.assertEqual((e2e['n'], e2e['draws'], e2e['ranks']), (5, 3125, [79, 3047]))
        self.assertEqual(result['aggregate']['qualified_goodput_tokens_per_second']['ratio'], 2.0)
        self.assertEqual(result['denominators']['scheduled'], 120)
        self.assertEqual(result['measured_qa']['scheduled'], 50)

    def test_known_nonconstant_interval_nearest_rank(self):
        value = analysis.paired_interval([math.log(x) for x in (1, 2, 3, 4, 5)])
        self.assertAlmostEqual(value['ratio'], 3)
        self.assertAlmostEqual(value['interval_ratio'][0], 1)
        self.assertAlmostEqual(value['interval_ratio'][1], 5)
        for values in ([1] * 4, [0, 0, 0, 0, float('nan')]):
            with self.assertRaises(ValueError):
                analysis.paired_interval(values)

    def test_failed_pair_retained_no_complete_case_or_replacement(self):
        result = analysis.summarize(self.report(failed='P3-blocking'))
        self.assertIsNone(result['aggregate'])
        self.assertEqual(len(result['blocks']), 5)
        self.assertEqual(result['qualified_blocks'], 4)
        self.assertTrue(result['blocks'][2]['failure_pair_retained'])
        duplicate = self.report()
        duplicate['lifecycles'].append(duplicate['lifecycles'][0])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            analysis.summarize(duplicate)

    def test_unissued_censored_failed_slots_conserve(self):
        schedule = ct.load(p.HERE / 'requests.json')['schedule']
        observations = [dict(schedule[0], status='success', issued=True),
                        dict(schedule[1], status='failed', issued=True),
                        dict(schedule[2], status='censored', issued=True)]
        summary = censoring_summary(observations, schedule)
        self.assertEqual((summary['successful'], summary['failed'], summary['censored'], summary['unissued']), (1, 1, 1, 9))
        self.assertEqual(summary['issued'] + summary['unissued'], 12)

    def test_quality_http_success_and_lifecycle_qualification_are_separate(self):
        row = synthetic_arm('P1-blocking')
        row['lifecycle']['requests']['observations'][-1]['metric_eligible'] = False
        self.assertIsNone(analysis.arm_metrics(row))
        row = synthetic_arm('P1-blocking')
        row['lifecycle']['requests']['observations'][-1]['actual']['usage']['completion_tokens'] = 9
        self.assertIsNone(analysis.arm_metrics(row))


class LedgerTests(CPUOnly):
    def ledger(self, mode='blocking'):
        options = p.case_for_mode(mode)['read_options']
        read = dict(enabled=True, requested_mode=mode.title(), effective_mode=mode.title(), limits=options,
            worker_read_observed=True, status='closed', stats={k: 0 for k in
                ('unknown_reads', 'unknown_bytes', 'unobserved_terminal_reads', 'unobserved_reader_intervals',
                 'join_failed_sequences', 'peak_host_charged_bytes', 'peak_metadata_bytes',
                 'peak_requested_stack_bytes', 'peak_payload_bytes', 'peak_packed_bytes', 'peak_artifact_bytes')},
            current_charge={k: 0 for k in ('unknown_packed_bytes', 'unknown_stack_bytes', 'unknown_read_slots', 'packed_bytes', 'artifact_bytes', 'payload_bytes', 'metadata_bytes', 'requested_stack_bytes', 'host_charged_bytes', 'reserved_read_slots')})
        read['stats'].update({k: 1 for k in ('submitted_reads', 'completed_reads', 'verified_reads', 'io_started_reads', 'proved_read_bytes')})
        return dict(closure=dict(attempts=[dict(summary=dict(physical=dict(expert_read_pipeline=dict(state='observed', value=read))))]))

    def test_final_actual_ledger_mode_unknown_pending_and_budget(self):
        observed = self.ledger()
        self.assertTrue(run.read_ledger_assessment(observed, 'blocking')['can_continue'])
        self.assertFalse(run.read_ledger_assessment(observed, 'lookahead')['can_continue'])
        for key, value in (('status', 'quarantined'), ('status', 'join_pending_at_scope_end'),
                           ('enabled', False), ('effective_mode', None), ('worker_read_observed', False)):
            bad = copy.deepcopy(observed)
            bad['closure']['attempts'][0]['summary']['physical']['expert_read_pipeline']['value'][key] = value
            self.assertFalse(run.read_ledger_assessment(bad, 'blocking')['can_continue'])
        bad = copy.deepcopy(observed)
        bad['closure']['attempts'][0]['summary']['physical']['expert_read_pipeline']['value']['stats']['unknown_bytes'] = 1
        self.assertFalse(run.read_ledger_assessment(bad, 'blocking')['can_continue'])
        bad = copy.deepcopy(observed)
        bad['closure']['attempts'][0]['summary']['physical']['expert_read_pipeline']['value']['stats']['peak_host_charged_bytes'] = (64 << 20) + 1
        self.assertFalse(run.read_ledger_assessment(bad, 'blocking')['can_continue'])

    def test_worker_calls_public_lifecycle_and_preserves_stdin_and_cleanup_result(self):
        folder = self.directory / 'P1-blocking'
        folder.mkdir()
        observed = self.ledger()
        ct.write(folder / 'lifecycle.json', observed)
        raw = json.dumps(dict(directory=str(folder), mode='blocking'))
        original = io.StringIO(raw)
        with mock.patch('sys.stdin', original), mock.patch.object(run.pilot, 'worker', return_value=1) as worker:
            self.assertEqual(run.worker('actual-sealed-hash'), 1)
            worker.assert_called_once_with('actual-sealed-hash')
            self.assertIs(sys.stdin, original)
        self.assertTrue(ct.load(folder / 'read-ledger-assessment.json')['can_continue'])


class PairedLifecycleTests(fixtures.CPUOnly):
    def fixture(self, unknown=False):
        args, calls, orders, labels = fixtures.MockPilotEndToEndTests.fixture(self, unknown=unknown)
        ct.write(self.directory / 'requests.json', dict(schedule=ct.load(p.HERE / 'requests.json')['schedule']))
        candidate = run.pilot.validate_candidate(args.candidate)
        execution = dict(limits=ct.load(p.PROTOCOL_PATH)['limits'], trial_order=p.trial_order(), artifact_pins={})
        self.stack.enter_context(mock.patch.object(run, 'HERE', self.directory))
        self.stack.enter_context(mock.patch.object(p, 'validate_prepared', return_value=execution))
        self.stack.enter_context(mock.patch.object(p, 'validate_candidate', return_value=candidate))
        self.stack.enter_context(mock.patch.object(run, 'read_ledger_assessment', return_value=dict(can_continue=True)))
        previous = Path.read_text
        def read(path, *args, **kwargs):
            text = previous(path, *args, **kwargs)
            if str(path).startswith('/proc/') and str(path).endswith('/environ'):
                text += '\0FERRULE_PROFILE_NVTX=0\0'
            return text
        self.stack.enter_context(mock.patch.object(Path, 'read_text', read))
        args.prepared = args.execution_prepared
        return args, calls, orders, labels

    def test_reuses_actual_maintained_worker_ten_fresh_lifecycles_120_requests(self):
        args, calls, orders, labels = self.fixture()
        result = run.run(args)
        self.assertEqual(result['status'], 'complete', result.get('stop_reason'))
        self.assertEqual(calls, p.trial_order())
        self.assertEqual([len(order) for order in orders], [12] * 10)
        self.assertEqual(result['analysis']['denominators']['successful'], 120)
        self.assertEqual(result['analysis']['measured_qa']['eligible_success'], 50)
        self.assertEqual(labels.count('shutdown-post'), 10)

    def test_unknown_custody_stops_without_drop_or_replacement(self):
        args, calls, orders, labels = self.fixture(unknown=True)
        result = run.run(args)
        self.assertEqual(calls, ['P1-blocking'])
        self.assertEqual(result['analysis']['denominators']['scheduled'], 120)
        self.assertEqual(result['analysis']['denominators']['unissued'], 108)
        self.assertEqual(result['analysis']['measured_qa']['eligible_success'], 0)
        self.assertEqual(len(result['analysis']['blocks']), 5)
        self.assertIsNone(result['analysis']['aggregate'])

    def test_cancellation_through_maintained_cleanup_keeps_all_slots(self):
        args, calls, orders, labels = self.fixture()
        self.stack.enter_context(mock.patch.object(fixtures.MockClient, 'request', side_effect=KeyboardInterrupt('fixture cancel')))
        result = run.run(args)
        self.assertEqual(calls, ['P1-blocking'])
        self.assertIn('shutdown-post', labels)
        self.assertTrue((args.output / 'P1-blocking/profile-stop').exists())
        summary = result['analysis']['denominators']
        self.assertEqual(summary['scheduled'], 120)
        self.assertEqual(summary['censored'], 1)
        self.assertEqual(summary['unissued'], 119)


class BuildPolicyTests(CPUOnly):
    def test_baseline_adapter_rejects_new_feature_policy_and_is_unchanged(self):
        f = BuildFixture(self.directory)
        original_adapter.review_command(f.build)
        value = copy.deepcopy(f.build)
        value['schema'] = build_adapter.BUILD_SCHEMA
        value['argv'][9] = 'cuda,nvtx-profile'
        value['outer_supervisor_argv'] = ['/usr/bin/timeout', '1800', *value['argv']]
        value['environment']['FERRULE_NVTX_INCLUDE_DIR'] = str(self.directory / 'official-headers')
        build_adapter.review_command(value)
        with self.assertRaisesRegex(ValueError, 'schema'):
            original_adapter.review_command(value)
        value['argv'][9] = 'cuda,nvtx-profile,cpu-reference'
        with self.assertRaisesRegex(ValueError, 'argv'):
            build_adapter.review_command(value)

    def test_build_review_rejects_unknown_adapter_or_cuda_only_binary(self):
        f = BuildFixture(self.directory)
        with self.assertRaisesRegex(ValueError, 'explicit'):
            build_adapter.review_cli_build(f.candidate, f.build, f.case)
        f.candidate['build_adapter'] = build_adapter.ADAPTER
        with self.assertRaisesRegex(ValueError, 'schema'):
            build_adapter.review_cli_build(f.candidate, f.build, f.case)


class ProfileTests(CPUOnly):
    def test_correlation_closure_is_not_collection_completeness(self):
        result = profile_audit.audit([], [], {}, 7)
        self.assertFalse(result['official_collection_completeness_verified'])
        self.assertFalse(result['full_trace_accepted'])

    def test_graph_launch_requires_actual_official_export_expansion(self):
        api = dict(globalTid=7 * 0x1000000, correlationId=10, name='cudaGraphLaunch', start=1, end=2, returnValue=0)
        windows = dict(qa=dict(start_ns=0, end_ns=10))
        result = profile_audit.audit([], [api], windows, 7)
        self.assertEqual(result['graph_launches'], 1)
        self.assertTrue(result['graph_issues'])
        self.assertFalse(result['full_trace_accepted'])


if __name__ == '__main__':
    unittest.main()
