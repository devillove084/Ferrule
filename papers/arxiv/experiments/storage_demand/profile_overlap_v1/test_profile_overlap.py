"""CPU-only contract/SQLite fixtures. No GPU queries, CUDA launches, or captures."""
from __future__ import annotations


from contextlib import ExitStack
import copy
import io
import importlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
import unittest
from unittest.mock import patch, Mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from profile_overlap_v1 import analyze as a, protocol as p, run, tp1_collector as tc

VM = 1 << 48
PID = 166208
GPID = VM | PID << 24
COMPUTE = GPID | PID
WORKER = GPID | (PID + 1)
EPOCH = 1_700_000_000_000_000_000


def fixture():
    db = sqlite3.connect(':memory:')
    db.executescript((HERE / 'fixture.sql').read_text())
    db.execute('INSERT INTO PROCESSES VALUES (?,?,?)', (GPID, PID, 'cli'))
    for i in range(8):
        uuid = tc.UUID.removeprefix('GPU-') if i == 0 else f'00000000-0000-0000-0000-{i:012d}'
        db.execute('INSERT INTO TARGET_INFO_GPU VALUES (?,?,?)', (VM, i, uuid))
    db.execute('INSERT INTO TARGET_INFO_CUDA_DEVICE VALUES (0,0,?)', (PID,))
    db.execute('INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES (?,NULL,NULL)', (EPOCH,))
    db.execute('INSERT INTO ANALYSIS_DETAILS VALUES (?,1000,200,1200)', (VM,))
    meta = dict(EXPORT_PRODUCT_VERSION='2025.3.1.90', EXPORT_SCHEMA_VERSION='3.20.2', EXPORT_PARAM_LAZY='false',
                EXPORT_PARAM_TIME_NORMALIZE='false', EXPORT_PARAM_TIME_SHIFT='0',
                EXPORT_PARAM_TABLES_TO_EXPORT='', EXPORT_PARAM_TIMES_TO_EXPORT='')
    db.executemany('INSERT INTO META_DATA_EXPORT VALUES (?,?)', meta.items())
    db.executemany('INSERT INTO META_DATA_CAPTURE VALUES (?,?)', [
        ('CAPTURE_EVENT_TYPE', name) for name in ('Cuda', 'NvtxEvents', 'OSRuntime')])
    db.execute("INSERT INTO META_DATA_CAPTURE VALUES ('OS_RUNTIME_OPTIONS:DURATION_THRESHOLD_NS','1000')")
    db.executemany('INSERT INTO StringIds VALUES (?,?)', [
        (1, 'cudaLaunchKernel_v7000'), (2, 'bf16_gemv'), (3, 'pread64'), (4, 'cudaMemcpyAsync_v3020'),
        (5, 'read_subset operation=1 source_generation=7 requested_bytes=16'),
        (6, 'layer.expert image_generation=7 layer=3 expert=2 rows=1'),
        (7, 'layer.expert.upload image_generation=7 layer=3 expert=2 rows=1'),
        (8, 'ferrule_profile'), (9, 'cuLaunchKernel'), (10, 'cudaStreamSynchronize_v3020')])
    db.executemany('INSERT INTO ENUM_NSYS_EVENT_CLASS VALUES (?,?,?)',
        [(27, 'TRACE_PROCESS_EVENT_OS_RUNTIME', 'OSRT runtime'),
         (31, 'TRACE_PROCESS_EVENT_OS_RUNTIME_START', 'OSRT start'),
         (32, 'TRACE_PROCESS_EVENT_OS_RUNTIME_FINISH', 'OSRT finish')])
    db.executemany('INSERT INTO ENUM_NSYS_EVENT_TYPE VALUES (?,?,?)',
                   [(75, 'NvtxDomainCreate', 'NvtxDomainCreate'), (60, 'NvtxStartEndRange', 'NvtxStartEndRange')])
    db.executemany('INSERT INTO ENUM_DIAGNOSTIC_SEVERITY_LEVEL VALUES (?,?,?)',
                   [(1, 'Info', 'Info'), (2, 'Warning', 'Warning'), (3, 'Error', 'Error')])
    db.execute('INSERT INTO NVTX_EVENTS(start,eventType,globalTid,textId,domainId) VALUES(0,75,?,8,1)', (COMPUTE,))
    db.executemany('INSERT INTO NVTX_EVENTS(start,end,eventType,rangeId,globalTid,endGlobalTid,textId,domainId) VALUES(?,?,60,?,?,?,?,1)',
        [(200, 400, 1, WORKER, WORKER, 5), (220, 250, 2, COMPUTE, COMPUTE, 6), (110, 150, 3, COMPUTE, COMPUTE, 7)])
    db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,0,?,?,?,?,NULL)', [
        (225, 230, COMPUTE, 1, 1, 0), (235, 240, COMPUTE, 2, 9, 0),
        (120, 125, COMPUTE, 3, 4, 0), (410, 420, COMPUTE, 4, 10, 0)])
    db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,0,1,1,?,?,2,NULL)',
                   [(260, 350, 1, GPID), (300, 370, 2, GPID)])
    db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(130,210,0,?,3,16)', (GPID,))
    db.executemany('INSERT INTO OSRT_API VALUES(?,?,27,?,3,8,0,0)', [(250, 280, WORKER), (320, 380, WORKER)])
    db.execute('INSERT INTO DIAGNOSTIC_EVENT VALUES(0,2,3,1,?,?)',
               ('Number of OS runtime libraries events collected: \t2.', GPID))
    cupti = str(p.TOOL / 'target-linux-x64/libcupti.so.12.9')
    db.execute('INSERT INTO DIAGNOSTIC_EVENT VALUES(0,1,1,1,?,?)', ('Loaded CUPTI library: ' + cupti, GPID))
    context = dict(model_pid=PID, model_birth_ticks=999, uuid=tc.UUID, wrapper_pids=[PID-1],
        request_begin_unix_ns=EPOCH + 100, request_end_unix_ns=EPOCH + 500,
        read_stats=dict(io_started_reads=1, proved_read_bytes=16), cupti_path=cupti,
        last_receipt=dict(identity=dict(operation=1, image_generation=7), requested_bytes=16, io_bytes=16))
    return db, context


class SQLiteTests(unittest.TestCase):
    def setUp(self):
        self.db, self.context = fixture()
        self.addCleanup(self.db.close)

    def audit(self):
        return a.audit_sqlite(self.db, self.context)

    def rejected(self):
        try:
            report = self.audit()
        except ValueError:
            return
        self.assertEqual(report['status'], 'rejected', report)

    def test_actual_kernel_union_not_host_or_copy_duration(self):
        report = self.audit()
        self.assertEqual(report['status'], 'sql_evidence_consistent', report['issues'])
        self.assertEqual(report['expert_kernel_union_ns'], 110)
        self.assertEqual(report['worker_read_envelope_union_ns'], 200)
        self.assertEqual(report['worker_pread_union_ns'], 90)
        self.assertEqual(report['read_envelope_kernel_intersection_ns'], 110)
        self.assertEqual(report['pread_kernel_intersection_ns'], 70)
        self.assertEqual(report['worker_pread_calls'], 2)
        self.assertEqual(report['read_envelope_overlap_percent'], 55.0)
        self.assertAlmostEqual(report['pread_overlap_percent'], 100 * 70 / 90)
        self.assertTrue(all(v == 0 for v in report['correlation_counts'].values()))
        self.assertIsNone(report['dropped_event_count'])
        self.assertFalse(report['full_trace_accepted'])
        self.assertFalse(report['ordinary_performance'])
        self.assertNotIn('kernel_events', report)

    def test_positive_ranges_with_zero_actual_overlap_are_valid(self):
        self.db.execute('UPDATE OSRT_API SET start=start-70,end=end-70 WHERE start=250')
        self.db.execute('UPDATE OSRT_API SET start=380,end=390 WHERE start=320')
        # First interval must remain contained in the worker envelope.
        self.db.execute('UPDATE OSRT_API SET start=200,end=220 WHERE start=180')
        report = self.audit()
        self.assertEqual(report['status'], 'sql_evidence_consistent')
        self.assertEqual(report['pread_kernel_intersection_ns'], 0)

    def test_serialized_pid_decoding_preserves_vm_key(self):
        self.assertEqual(a.pid(WORKER), PID)
        self.assertEqual(a.process_key(WORKER), GPID)
        self.assertNotEqual(a.process_key(WORKER), PID)
        self.db.execute('UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET globalTid=globalTid+? WHERE correlationId=1', (1 << 48,))
        self.rejected()

    def test_correlation_alone_does_not_join_foreign_pid(self):
        self.db.execute('UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET globalTid=globalTid+? WHERE correlationId=1', (1 << 24,))
        self.rejected()

    def test_duplicate_pid_identity_rejected(self):
        self.db.execute('INSERT INTO PROCESSES SELECT * FROM PROCESSES')
        self.rejected()

    def test_physical_mapping_not_device_ordinal(self):
        self.db.execute('UPDATE TARGET_INFO_CUDA_DEVICE SET gpuId=1')
        self.rejected()

    def test_duplicate_device_map_rejected(self):
        self.db.execute('INSERT INTO TARGET_INFO_CUDA_DEVICE SELECT * FROM TARGET_INFO_CUDA_DEVICE')
        self.rejected()

    def test_bidirectional_orphan_and_duplicate_fail(self):
        for sql in ('DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId=1',
                    'DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE correlationId=1',
                    'INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL LIMIT 1',
                    'INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId=1'):
            with self.subTest(sql=sql):
                self.db.execute('SAVEPOINT bad')
                self.db.execute(sql)
                self.rejected()
                self.db.execute('ROLLBACK TO bad')
                self.db.execute('RELEASE bad')

    def test_copy_enqueue_cannot_qualify_math(self):
        self.db.execute('UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET nameId=4 WHERE correlationId=1')
        self.rejected()

    def test_wrong_submission_thread_rejected(self):
        self.db.execute('UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET globalTid=? WHERE correlationId IN (1,2)', (WORKER,))
        self.rejected()

    def test_wrong_domain_and_unknown_string_rejected(self):
        for sql in ("UPDATE StringIds SET value='ferrule_performance' WHERE id=8",
                    'UPDATE NVTX_EVENTS SET textId=99 WHERE rangeId=1',
                    "UPDATE NVTX_EVENTS SET text='different' WHERE rangeId=1"):
            with self.subTest(sql=sql):
                self.db.execute('SAVEPOINT bad')
                self.db.execute(sql)
                self.rejected()
                self.db.execute('ROLLBACK TO bad')
                self.db.execute('RELEASE bad')

    def test_direct_nvtx_text_is_supported(self):
        self.db.execute('UPDATE NVTX_EVENTS SET text=(SELECT value FROM StringIds WHERE id=NVTX_EVENTS.textId),textId=NULL')
        self.assertEqual(self.audit()['status'], 'sql_evidence_consistent')

    def test_field_metadata_is_required(self):
        for value in ('read_subset', 'read_subset operation=1 source_generation=7 requested_bytes=0',
                      'read_subset operation=1 operation=2 source_generation=7 requested_bytes=16'):
            with self.subTest(value=value):
                self.db.execute('UPDATE StringIds SET value=? WHERE id=5', (value,))
                self.rejected()

    def test_source_generation_must_match_actual_expert_image(self):
        self.db.execute("UPDATE StringIds SET value='read_subset operation=1 source_generation=8 requested_bytes=16' WHERE id=5")
        self.rejected()

    def test_duplicate_read_operation_cannot_inflate_overlap(self):
        self.db.execute('INSERT INTO NVTX_EVENTS SELECT start,end,eventType,4,category,color,text,globalTid,endGlobalTid,textId,domainId FROM NVTX_EVENTS WHERE rangeId=1')
        self.rejected()

    def test_unknown_worker_pread_coverage_fails(self):
        for sql in ('DELETE FROM OSRT_API WHERE start=250', 'UPDATE OSRT_API SET returnValue=-1 WHERE start=250',
                    'UPDATE OSRT_API SET start=199 WHERE start=250',
                    'INSERT INTO OSRT_API SELECT * FROM OSRT_API LIMIT 1'):
            with self.subTest(sql=sql):
                self.db.execute('SAVEPOINT bad')
                self.db.execute(sql)
                self.rejected()
                self.db.execute('ROLLBACK TO bad')
                self.db.execute('RELEASE bad')

    def test_current_counter_mismatch_fails(self):
        self.context['read_stats']['io_started_reads'] = 2
        self.rejected()

    def test_last_receipt_identity_mismatch_fails(self):
        self.context['last_receipt']['identity']['operation'] = 2
        self.rejected()

    def test_graph_node_or_unknown_many_to_one_fails(self):
        self.db.execute('UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET graphNodeId=1')
        self.rejected()

    def test_graph_table_even_without_nodes_in_kernels_fails(self):
        self.db.execute('INSERT INTO CUDA_GRAPH_NODE_EVENTS VALUES(1)')
        self.rejected()

    def test_graph_launch_rejected(self):
        self.db.execute("UPDATE StringIds SET value='cudaGraphLaunch_v10000' WHERE id=1")
        self.rejected()

    def test_diagnostic_rejection_despite_complete_rows(self):
        for severity, text in ((3, 'collector error'), (1, 'Not all CUDA events were collected'),
                               (2, 'CUDA profiling might have not been started correctly.'), (2, 'new warning')):
            with self.subTest(text=text):
                self.db.execute('SAVEPOINT bad')
                self.db.execute('INSERT INTO DIAGNOSTIC_EVENT VALUES(1,1,1,?,?,?)', (severity, text, GPID))
                self.rejected()
                self.db.execute('ROLLBACK TO bad')
                self.db.execute('RELEASE bad')

    def test_non_cuda_wrapper_warning_preserved_not_transferred_to_model(self):
        message = 'No NVTX events collected. Does the process use NVTX?'
        self.db.execute('INSERT INTO DIAGNOSTIC_EVENT VALUES(1,1,1,2,?,?)', (message, GPID-(1 << 24)))
        result = self.audit()
        self.assertEqual(result['status'], 'rejected')
        self.assertFalse(result['full_trace_accepted'])
        self.assertIsNone(result['warnings'][0]['classification'])

    def test_known_sampling_warning_still_rejects(self):
        message = ('Scheduling information is absent. The thread activity is deduced based on OS runtime libraries traces. '
                   'This is inaccurate and does not take into account asynchronous interrupts and exception faults.')
        self.db.execute('INSERT INTO DIAGNOSTIC_EVENT VALUES(1,1,1,2,?,?)', (message, GPID))
        report = self.audit()
        self.assertEqual(report['status'], 'rejected')
        self.assertEqual(len(report['warnings']), 1)
        self.assertFalse(report['full_trace_accepted'])

    def test_unknown_lazy_filtered_or_missing_schema_rejected(self):
        for key, value in [('EXPORT_PARAM_LAZY', 'true'), ('EXPORT_SCHEMA_VERSION', 'unknown'),
                           ('EXPORT_PARAM_TABLES_TO_EXPORT', 'KERNEL'), ('EXPORT_PARAM_TIMES_TO_EXPORT', '1:2')]:
            with self.subTest(key=key):
                self.db.execute('SAVEPOINT bad')
                self.db.execute('UPDATE META_DATA_EXPORT SET value=? WHERE name=?', (value, key))
                self.rejected()
                self.db.execute('ROLLBACK TO bad')
                self.db.execute('RELEASE bad')
        self.db.execute('DROP TABLE NVTX_EVENTS')
        self.rejected()

    def test_truncated_request_coverage_rejected(self):
        self.context['request_end_unix_ns'] = EPOCH + 1001
        self.rejected()

    def test_unclosed_nvtx_range_rejected(self):
        self.db.execute('UPDATE NVTX_EVENTS SET end=NULL WHERE rangeId=1')
        self.rejected()

    def test_interval_union_intersection(self):
        self.assertEqual(a.union([(1, 5), (3, 9), (9, 10), (12, 15)]), [(1, 10), (12, 15)])
        self.assertEqual(a.intersection([(1, 10)], [(3, 5), (7, 12)]), [(3, 5), (7, 10)])


class ContractTests(unittest.TestCase):
    def test_plan_is_cpu_only_one_qa_no_extra_hello(self):
        with patch('subprocess.run', side_effect=AssertionError('no commands in plan')):
            plan = p.plan()
        self.assertEqual(plan['order'], ['blocking', 'lookahead'])
        self.assertEqual((plan['pair_ceiling'], plan['requests_per_arm'], plan['qa']['max_tokens']), (1, 1, 8))
        self.assertEqual((plan['hello_requests'], plan['warmups'], plan['retries']), (0, 0, 0))
        self.assertFalse(plan['execute_allowed'])

    def test_finite_official_flags_and_full_export(self):
        argv = p.profile_argv(Path('/private/session'), 'digest')
        self.assertIn('--duration=150', argv)
        self.assertIn('--kill=none', argv)
        self.assertIn('--wait=all', argv)
        self.assertIn('--osrt-file-access=true', argv)
        self.assertIn('--sample=none', argv)
        self.assertIn('--cpuctxsw=process-tree', argv)
        self.assertIn('--gpu-metrics-devices=none', argv)
        self.assertFalse(any('graph-trace' in s or 'cuda-trace-scope' in s for s in argv))
        self.assertIn('--lazy=false', p.export_argv(Path('/private/session')))
        self.assertEqual(p.LIMITS['shutdown_seconds'], 45)
        self.assertEqual(p.LIMITS['startup_seconds'], 120)
        self.assertEqual(p.LIMITS['request_seconds'], 60)
        self.assertEqual(p.LIMITS['supervisor_seconds'], 300)

    def test_profile_environment_changes_only_nvtx_not_ordinary_case(self):
        original = p.binding.case_for_mode('blocking')
        before = copy.deepcopy(original)
        candidate = {'candidate': {'binary': '/qualified/cli'}}
        profile = p.spec(candidate, 'blocking')
        self.assertEqual(original, before)
        self.assertEqual(original['environment']['FERRULE_PROFILE_NVTX'], '0')
        self.assertEqual(profile['environment']['FERRULE_PROFILE_NVTX'], '1')
        self.assertEqual(profile['argv'], ['/qualified/cli', *original['argv_after_binary']])
        self.assertEqual(profile['environment']['CUDA_VISIBLE_DEVICES'], tc.UUID)

    def test_private_tp1_modules_do_not_mutate_two_gpu_protocol(self):
        original = importlib.import_module('instrumentation_live')
        scope = importlib.import_module('instrumentation_scope')
        capture = importlib.import_module('instrumentation_capture')
        self.assertIsNot(original, tc.live)
        self.assertIsNot(scope, tc.scope)
        self.assertEqual(len(original.shared.UUIDS), 2)
        self.assertEqual(scope.PROBE_SECONDS, 60)
        self.assertEqual(tc.live.shared.UUIDS, (tc.UUID,))
        self.assertIs(tc.capture.live, tc.live)
        self.assertIs(capture.live, original)

    def test_tp1_grant_rejects_two_gpu_or_generic_scope(self):
        inventory = [tc.UUID] + [f'GPU-00000000-0000-0000-0000-{i:012d}' for i in range(7)]
        grant = dict(scope=tc.SCOPE, granted=True, note='parent GO', gpu_uuids=sorted(inventory), model_gpu_uuids=[tc.UUID])
        self.assertEqual(tc.tp1_grant_scope(inventory, (tc.UUID,), grant), sorted(inventory))
        with self.assertRaises(ValueError):
            tc.tp1_grant_scope(inventory, inventory[:2], grant)
        grant['scope'] = 'all_eight_instrumentation_probe'
        with self.assertRaises(ValueError):
            tc.tp1_grant_scope(inventory, (tc.UUID,), grant)

    def test_capture_wrapper_engine_reused_with_finite_budget(self):
        wrapper = Mock()
        direct = wrapper.run_target
        with patch.object(tc, '_original_load_wrapper', return_value=wrapper):
            bounded = tc.bounded_wrapper()
        bounded.run_target('case', ['cli'], 1450, 45)
        direct.assert_called_once_with('case', ['cli'], 150, 45)

    def test_run_without_grant_cannot_query_launch_or_lock(self):
        with (patch.object(p, 'prepared', side_effect=AssertionError('preflight should not start')),
              patch.object(tc.live.observer, 'idle', side_effect=AssertionError('GPU query'))):
            with self.assertRaisesRegex(ValueError, 'sole-owner'):
                run.run(HERE / 'artifacts-test', None, None, False)

    def test_qa_quality_rejects_no_tokens_before_sql(self):
        with self.assertRaisesRegex(ValueError, 'generation IDs'):
            run.qa_result(Path('/private'), dict(status='success', actual_output_tokens=8), [])

    def test_foreign_model_parent_rejected_without_any_signal(self):
        auth = dict(worker=dict(pid=10))
        server = Mock(pid=20)
        target = dict(pid=30, birth_ticks=3, ppid=20)
        root = dict(pid=40, birth_ticks=4, ppid=99)
        def identity(pid):
            return dict(pid=20, birth_ticks=2, ppid=10) if pid == 20 else root
        def load(path):
            if path.name == 'collector-target-ready.json':
                return dict(identity=target)
            return dict(pid=40)
        server.poll.return_value = None
        with patch.object(tc.live.observer, 'proc_identity', side_effect=identity), \
             patch.object(tc.ct, 'load', side_effect=load), \
             patch.object(tc.live, 'live_auth'), patch.object(tc.live, 'descendant'):
            with self.assertRaisesRegex(ValueError, 'direct child'):
                tc.model_identity(server, Path('/private'), float('inf'), auth)
        server.terminate.assert_not_called()


def qa_fixture():
    planned = p.qa()
    usage = dict(prompt_tokens=5, completion_tokens=8, total_tokens=13)
    actual = dict(status='success', actual_output_tokens=8, adapter_http_report_complete=True,
        text=planned['reference']['reference_text'], usage=usage, finish_reason='length', done_at=1.0,
        raw_sse_events=[dict(data=json.dumps(dict(id='cmpl-ferrule-1', choices=[dict(text=' Tokyo.')]))),
                        dict(data='[DONE]')])
    records = [dict(event='request_admitted', request_id=1)]
    records += [dict(event='request_worker_token_enqueued', request_id=1, session_id=1,
                     emission_ordinal=i+1, actual_token_id=token, delivery_boundary='worker_channel_enqueue')
                for i, token in enumerate(planned['reference']['reference_ids'])]
    records += [dict(event='request_worker_terminal_enqueue', request_id=1, terminal_event_enqueued=True,
                     status='finished', last_emission_ordinal=8, delivery_boundary='worker_channel_enqueue',
                     usage_available=True, **usage),
                dict(event='request_http_terminal', request_id=1, status='finished', token_events=8),
                dict(event='request_worker_terminal', request_id=1)]
    lines = [json.dumps(dict(timestamp='2026-10-15T00:00:00.000000Z' if i == 0 else '2026-10-15T00:00:01.000000Z',
                             level='DEBUG', fields=row)) for i, row in enumerate(records)]
    return actual, lines


class LifecycleHookTests(unittest.TestCase):
    def test_exact_qa_ids_and_generation_request_denominator(self):
        actual, lines = qa_fixture()
        result = run.qa_result(Path('/private'), actual, lines)
        self.assertEqual(result['trace']['actual_generated_ids'], p.qa()['reference']['reference_ids'])
        self.assertEqual(result['request_end_unix_ns'] - result['request_begin_unix_ns'], 1_000_000_000)
        with self.assertRaisesRegex(ValueError, 'extra/unbound'):
            run.qa_result(Path('/private'), actual, lines + [lines[0]])
        bad = list(lines)
        row = json.loads(bad[1])
        row['fields']['actual_token_id'] = 9
        bad[1] = json.dumps(row)
        with self.assertRaisesRegex(ValueError, 'generation IDs'):
            run.qa_result(Path('/private'), actual, bad)

    def test_nullable_error_field_is_not_a_collector_error(self):
        benign = json.dumps(dict(level='INFO', fields=dict(error=None, warning_count=0)))
        bad = json.dumps(dict(level='WARN', fields=dict(message='model warning')))
        rows = run.collector_log_issues('/private/log', [benign, bad, 'ERROR: collector failed'])
        self.assertEqual([row['line'] for row in rows], [2, 3])

    def test_unknown_go_rejected_before_queries(self):
        resolved = dict(collector_contract=dict(status='resolved', launch_allowed=True, blockers=[]))
        with (patch.object(p, 'prepared', return_value=resolved), patch.object(run.live, 'sealed', return_value={}),
              patch.object(run.live.observer, 'idle', side_effect=AssertionError('no GPU query'))):
            with self.assertRaisesRegex(ValueError, 'parent GO'):
                run.authorization(Path('/private/pair'), Path('/private/go'), 'sha')

    def test_cli_role_requirements_not_removed_for_profiler(self):
        runtime = dict(pins={'/original/libcuda.so': 'cuda'}, per_role={'cli': {'/original/libcuda.so': 'cuda'}})
        candidate = dict(candidate=dict(runtime_pin_manifest='/private/runtime', binary='/qualified/cli'))
        with patch.object(p.runtime_contract, 'inventory', return_value=(runtime, {}, {})) as inventory:
            extended, _, _ = p.profile_runtime(candidate)
        self.assertEqual(runtime['per_role']['cli'], {'/original/libcuda.so': 'cuda'})
        self.assertEqual(extended['per_role']['cli']['/original/libcuda.so'], 'cuda')
        inventory.assert_called_once_with(p.TOOL, p.VERSION, p.PRIOR, candidate)

    def mocked_worker(self, ready_failure=False, finish_failure=False, audit_failure=False):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            pair = Path(temporary)
            case = pair / 'runs/blocking/session'
            case.mkdir(parents=True)
            payload = dict(directory=str(case), mode='blocking', owner_locks={}, baseline={},
                           baseline_sha256='sha', prepared_sha256='sha', worker_token='test-only')
            prepared = dict(directory=str(pair), collector_contract=dict(status='resolved', launch_allowed=True, blockers=[]),
                            specs={'blocking': p.spec({'candidate': {'binary': '/qualified/cli'}}, 'blocking')})
            root = dict(pid=1234, birth_ticks=2, ppid=1233)
            target = dict(pid=1233, birth_ticks=1, ppid=1232)
            auth = dict(coordinator=dict(pid=1231))
            server = Mock(pid=1232)
            monitor = Mock(root=dict(pid=1232), manifest={})
            monitor.failed.is_set.return_value = False
            monitor.finish.return_value = dict(errors=[], memory_components=dict(gpu_samples=1))
            if finish_failure:
                monitor.finish.side_effect = RuntimeError('monitor finish failed')
            actual, lines = qa_fixture()
            def request(*args):
                (case / 'server.log').write_text('\n'.join(lines) + '\n')
                return actual
            mocks = [(sys, 'stdin', io.StringIO(json.dumps(payload))),
                (p, 'prepared', prepared), (run.ct, 'sha', 'sha'),
                (run.live.guard, 'authenticate', auth), (run.live, 'validate_locks', None),
                (run.live.observer, 'idle', dict(status='verified', baseline={})),
                (run.ex.runner, 'require_free_port', None), (run.live, 'target_manifest', 'digest'),
                (run.subprocess, 'Popen', server), (run.live, 'CollectorMonitor', monitor),
                (tc, 'model_identity', (target, root)), (run.ex.http, 'wait_ready', time.monotonic()),
                (run, 'check_environment', dict(FERRULE_PROFILE_NVTX='1')),
                (run, 'snapshot', dict(status='passed')), (run.live, 'live_auth', None),
                (run.role_context, 'capture', dict(path='synthetic', sha256='a' * 64)),
                (run, 'read_ledger', dict(read={})), (run, 'export', None),
                (run, 'audit_capture', dict(status='rejected', log_issues=[], warnings=['bound wrapper only'],
                    scoped_sql_evidence_consistent=True, scoped_issues=[], loss_diagnostic_count=0, dropped_event_count=None)),
                (run.live, 'shutdown', dict(returncode=0, timed_out=False))]
            handles = {}
            for obj, name, value in mocks:
                if obj is sys:
                    handles[name] = stack.enter_context(patch.object(obj, name, value))
                else:
                    handles[name] = stack.enter_context(patch.object(obj, name, return_value=value))
            handles['one_request'] = stack.enter_context(patch.object(run.ex, 'one_request', side_effect=request))
            if audit_failure:
                handles['audit_capture'].return_value = dict(status='rejected', log_issues=[], warnings=['official warning'])
            if ready_failure:
                handles['wait_ready'].side_effect = TimeoutError('Ready failed')
            code = run.worker('digest')
            self.assertEqual(handles['Popen'].call_count, 1)
            self.assertEqual(handles['shutdown'].call_count, 1)
            self.assertEqual(handles['one_request'].call_count, 0 if ready_failure else 1)
            handles['export'].assert_not_called() if ready_failure or finish_failure else handles['export'].assert_called_once()
            return code

    def test_single_qa_ready_hook_uses_maintained_cleanup_and_export(self):
        self.assertEqual(self.mocked_worker(), 0)

    def test_official_audit_failure_returns_failed_worker_after_cleanup(self):
        self.assertEqual(self.mocked_worker(audit_failure=True), 1)

    def test_ready_failure_never_issues_qa_or_retries(self):
        self.assertEqual(self.mocked_worker(ready_failure=True), 1)

    def test_monitor_finish_error_still_invokes_owned_shutdown(self):
        self.assertEqual(self.mocked_worker(finish_failure=True), 1)


class ExistingReadOnlySchemaTests(unittest.TestCase):
    def test_accepted_micro_process_mapping_and_observed_schema(self):
        path = p.ct.HERE / 'tp2_resident/runs/nsys-official-micro-20261009-002/runs/P1/session/micro.sqlite'
        if not path.exists():
            self.skipTest('private historical micro not installed')
        from nsys_micro_accept import accept
        with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
            result = accept(db)
            self.assertEqual(result['status'], 'pass_sqlite')
            self.assertEqual(result['kernel_rows'], 2)
            self.assertEqual(result['logical_kernel_device_ids'], [0])
            for table in ('PROCESSES', 'StringIds', 'CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_KERNEL', 'OSRT_API'):
                self.assertLessEqual(set(a.REQUIRED[table].split()), {r[1] for r in db.execute('PRAGMA table_info(' + table + ')')})
            # Lazy micro has no Ferrule ranges: never inherit its acceptance.
            with self.assertRaises(ValueError):
                a.audit_sqlite(db, {})

    def test_actual_nvtx_schema_without_reading_large_kernel_artifacts(self):
        path = p.ct.HERE / 'tp2_resident/runs/tp2-official-full-profile-20261010-001/runs/P1/session/nsight/nsys.sqlite'
        if not path.exists():
            self.skipTest('historical official export not installed')
        with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
            self.assertLessEqual(set(a.REQUIRED['NVTX_EVENTS'].split()), {r[1] for r in db.execute('PRAGMA table_info(NVTX_EVENTS)')})
            self.assertEqual(db.execute('SELECT name FROM ENUM_NSYS_EVENT_TYPE WHERE id=60').fetchone()[0], 'NvtxStartEndRange')
            self.assertEqual(db.execute('SELECT name FROM ENUM_NSYS_EVENT_TYPE WHERE id=75').fetchone()[0], 'NvtxDomainCreate')


if __name__ == '__main__':
    unittest.main()
