"""CPU-only private adapter and raw live-role bridge regressions."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from profile_overlap_v1 import protocol as p, role_context as r, run, scope_contract as s, tp1_collector as tc
from profile_overlap_v1.test_profile_overlap import GPID, PID


class RoleBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.pair = Path(self.temporary.name) / p.GENERATION
        self.case = self.pair / 'runs/blocking/session'
        self.case.mkdir(parents=True)
        self.command = ['/synthetic/frozen/cli', 'serve']
        self.prepared = dict(directory=str(self.pair), binding_sha256='b' * 64,
            pins={str(Path(p.PYTHON).resolve()): 'a' * 64}, specs={'blocking': dict(argv=self.command)})
        p.ct.write(self.pair / 'prepared.json', self.prepared)
        def ident(pid, parent, birth):
            return dict(pid=pid, ppid=parent, birth_ticks=birth, state='S')
        self.root = ident(PID, PID - 1, 999)
        self.target = ident(PID - 1, PID - 2, 998)
        self.collector = ident(PID - 3, PID - 4, 997)
        auth = dict(launch_id='test-only', worker=ident(PID - 4, PID - 5, 996))
        self.worker = dict(mode='blocking', server_identity=self.root, target_identity=self.target,
                           collector_identity=self.collector, authenticated_worker=auth, process_role_receipts={})
        launch = dict(directory=str(self.case), mode='blocking', prepared_sha256=p.ct.sha(self.pair / 'prepared.json'))
        launch_sha = tc.live.readonly(self.case / 'worker-launch.json', launch)
        manifest = dict(directory=str(self.case), authenticated=auth, command=self.command,
            command_sha256=p.ct.digest(self.command), binary_sha256=p.BINARY_SHA,
            worker_launch_sha256=launch_sha, nonce='fixture-nonce')
        digest = tc.live.readonly(self.case / 'collector-manifest.json', manifest)
        target = dict(identity=self.target, manifest_sha256=digest, nonce=manifest['nonce'])
        tc.live.readonly(self.case / 'collector-target-ready.json', target)
        tc.live.readonly(self.case / 'model-release.json', dict(target=target, manifest_sha256=digest,
            nonce=manifest['nonce'], launch_id=auth['launch_id'], collector=self.collector, model_gpu_uuids=[tc.UUID]))
        self.processes = [dict(pid=PID, globalPid=GPID), dict(pid=PID-1, globalPid=GPID-(1 << 24))]

    def capture(self, label, identities=None, sha_change=False, command_change=False):
        _, wanted = r.expected(self.case, self.worker, self.prepared)
        by_pid = {v['identity']['pid']: v for v in wanted.values()}
        original_sha, original_read = p.ct.sha, Path.read_bytes
        def sha(path):
            path = Path(path)
            if path.parent.parent == Path('/proc'):
                return 'c' * 64 if sha_change else by_pid[int(path.parent.name)]['executable_sha256']
            return original_sha(path)
        def read(path):
            if path.parent.parent == Path('/proc'):
                command = ['foreign'] if command_change else by_pid[int(path.parent.name)]['command']
                return ('\0'.join(command) + '\0').encode()
            return original_read(path)
        with (patch.object(tc.live, 'live_auth'), patch.object(tc.live, 'descendant'),
              patch.object(tc.live.observer, 'proc_identity', side_effect=identities or (lambda pid: by_pid[pid]['identity'])),
              patch.object(p.ct, 'sha', side_effect=sha), patch.object(Path, 'read_bytes', read)):
            result = r.capture(self.case, label, self.worker, self.prepared)
        self.worker['process_role_receipts'][label] = result
        return result

    def closed(self):
        for label in ('ready', 'after-requests'):
            self.capture(label)
        return r.bind(self.case, self.worker, self.prepared, self.processes)

    def tamper(self, label, mutate):
        path = self.case / (label + '-process-roles.json')
        value = p.ct.load(path)
        mutate(value)
        path.chmod(0o600)
        path.write_text(json.dumps(value))
        path.chmod(0o400)
        self.worker['process_role_receipts'][label]['sha256'] = p.ct.sha(path)

    def test_both_live_phases_bind_exact_cli_and_only_actual_wrapper(self):
        context = self.closed()
        self.assertEqual([v['role'] for v in context['process_roles']], ['model', 'non_cuda_wrapper'])
        self.assertEqual(context['role_executable_sha256']['model'], p.BINARY_SHA)
        self.assertNotIn(self.collector['pid'], [v['identity']['pid'] for v in context['process_roles']])
        self.assertEqual(context['role_source_binding_sha256']['model'], self.prepared['binding_sha256'])

    def test_pid_reuse_parent_change_and_zombie_during_capture_reject(self):
        for change in (dict(birth_ticks=1000), dict(ppid=2), dict(state='Z')):
            with self.subTest(change=change):
                identities = [self.root, dict(self.root, **change)]
                with self.assertRaisesRegex(ValueError, 'liveness changed'):
                    self.capture('ready', identities)
                self.assertFalse((self.case / 'ready-process-roles.json').exists())

    def test_executable_sha_or_actual_argv_not_role_name(self):
        for changed in ('sha_change', 'command_change'):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, 'executable/argv'):
                self.capture('ready', **{changed: True})

    def test_missing_second_live_observation_not_inferred_from_sql(self):
        self.capture('ready')
        with self.assertRaises(KeyError):
            r.bind(self.case, self.worker, self.prepared, self.processes)

    def test_child_generation_and_birth_not_supplied_by_warning_pid(self):
        self.closed()
        self.tamper('after-requests', lambda value: value['roles']['model']['after'].update(birth_ticks=1000))
        with self.assertRaisesRegex(ValueError, 'not live'):
            r.bind(self.case, self.worker, self.prepared, self.processes)

    def test_wrong_source_or_generation_even_resealed_rejects(self):
        self.closed()
        for label in ('ready', 'after-requests'):
            self.tamper(label, lambda value: value.update(generation='artifacts-go-20261015-03'))
        with self.assertRaisesRegex(ValueError, 'receipt binding'):
            r.bind(self.case, self.worker, self.prepared, self.processes)

    def test_duplicate_raw_model_pid_is_never_guessed(self):
        self.closed()
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            r.bind(self.case, self.worker, self.prepared, self.processes + [dict(pid=PID, globalPid=GPID+(1 << 48))])

    def test_collector_pid_cannot_be_substituted_for_traced_wrapper(self):
        self.worker['target_identity'] = self.collector
        with self.assertRaisesRegex(ValueError, 'target gate'):
            r.expected(self.case, self.worker, self.prepared)

    def test_foreign_prepared_binding_not_historical_role_upgrade(self):
        self.prepared['directory'] = str(self.pair.parent / 'artifacts-go-20261015-01')
        with self.assertRaisesRegex(ValueError, 'generation/arm'):
            r.expected(self.case, self.worker, self.prepared)

    def test_raw_adapter_to_sql_scope_preserves_wrapper_warnings_without_full_pass(self):
        import sqlite3
        from profile_overlap_v1.test_profile_overlap import fixture
        from profile_overlap_v1.test_scoped_contract import GATES
        self.prepared['overlap_scope'] = s.policy('process-tree')
        self.closed()
        db, context = fixture()
        db.execute('INSERT INTO PROCESSES VALUES(?,?,?)', (GPID-(1 << 24), PID-1, 'python-fixture'))
        for warning in s.NO_PAYLOAD:
            db.execute('INSERT INTO DIAGNOSTIC_EVENT VALUES(1,2,3,2,?,?)', (warning, GPID-(1 << 24)))
        db.commit()
        with sqlite3.connect(self.case / 'nsys.sqlite') as output:
            db.backup(output)
        db.close()
        for name in ('server.log', 'export.stdout', 'export.stderr'):
            (self.case / name).write_text('')
        (self.case / 'nsys.nsys-rep').write_bytes(b'SYNTHETIC TEST ONLY; NOT AN NSYS CAPTURE')
        qa = {key: context[key] for key in ('request_begin_unix_ns', 'request_end_unix_ns')}
        ledger = dict(model_source_generation=7, read=dict(stats=context['read_stats'], last_receipt=context['last_receipt']))
        self.prepared['candidate'] = {}
        source = dict(read_source_boundary=context['read_source_boundary'])
        p.ct.write(self.pair / 'nvtx-source-qualification.json', source)
        with (patch.object(p, 'prepared', return_value=self.prepared),
              patch.object(p, 'source_qualification', return_value=source)):
            report = run.audit_capture(self.case, self.worker, qa, ledger)
        self.assertTrue(s.sql_eligible(report), report['scoped_issues'])
        self.assertEqual(report['status'], 'rejected')
        self.assertEqual(len(report['warnings']), 3)
        self.assertFalse(report['scoped_overlap_accepted'])
        final = s.acceptance(report, GATES)
        self.assertTrue(final['scoped_overlap_accepted'])
        self.assertFalse(final['full_trace_accepted'])
        self.assertIsNone(final['dropped_event_count'])



class AdapterAndGateTests(unittest.TestCase):
    def test_private_context_aliases_and_function_globals(self):
        result = tc.adapter_binding()
        self.assertTrue(result['capture_aliases_bound'])
        self.assertFalse(result['new_execution_loop'])
        with patch.object(tc.capture, 'observer', object()):
            with self.assertRaisesRegex(ValueError, 'context binding'):
                tc.adapter_binding()

    def test_preflight_never_calls_tool_or_reprobes_pidfd(self):
        if not (p.PREDECESSORS[-1] / 'collector-contract.json').exists():
            self.skipTest('private CPU receipts not installed')
        with patch('subprocess.run', side_effect=AssertionError('no nsys or GPU execution')):
            result = p.collector_preflight()
        self.assertTrue(result['launch_allowed'])
        self.assertTrue(result['parent_GO_required_after_pressure_cleanup'])
        self.assertEqual(result['tool_commands_executed'], 0)

    def test_unknown_log_loss_even_info_blocks_scope(self):
        lines = ['INFO: Partial collection; event count unknown',
                 json.dumps(dict(level='INFO', fields=dict(message='Events were dropped')))]
        self.assertEqual(len(run.collector_log_issues('/synthetic/log', lines)), 2)

    def test_permission_failure_is_not_out_of_scope(self):
        from profile_overlap_v1.test_scoped_contract import ScopedSQLTests
        fixture = ScopedSQLTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.context['scope_contract'] = s.policy('process-tree')
        fixture.warning('Permission denied collecting scheduling information', s.GLOBAL_DIAGNOSTIC_PID)
        self.assertFalse(fixture.audit()['scoped_sql_evidence_consistent'])

    def test_old_go_cannot_be_reused_before_gpu(self):
        prepared = dict(collector_contract=dict(status='resolved', launch_allowed=True, blockers=[]))
        grant = dict(schema=p.ID + '.parent-go', decision='GO', prepared_sha256='sha', directory='/synthetic',
            pair_ceiling=1, order=p.ORDER, ordinary_timing_closed=True, sole_gpu_owner=True, instrumentation_all_eight=True)
        with (patch.object(p, 'prepared', return_value=prepared), patch.object(tc.live, 'sealed', return_value=grant),
              patch.object(p.ct, 'sha', return_value='sha'), patch.object(tc.scope, 'probe_locks', side_effect=AssertionError('no locks')),
              patch.object(tc.live.observer, 'idle', side_effect=AssertionError('no GPU query'))):
            with self.assertRaisesRegex(ValueError, 'parent GO'):
                run.run(Path('/synthetic'), Path('/synthetic/go'), 'sha', True)


class TypedReadCoverageTests(unittest.TestCase):
    def setUp(self):
        self.stats = dict.fromkeys(('submitted_reads', 'worker_started_reads', 'completed_reads',
                                   'verified_reads', 'io_started_reads'), 2)
        self.stats.update(proved_read_bytes=32, failed_reads=0, cancelled_reads=0, unknown_reads=0,
                          unobserved_terminal_reads=0, reader_interval_calls=4, reader_interval_bytes=32, unobserved_reader_intervals=0)
        self.read = dict(stats=self.stats, last_receipt=dict(identity=dict(image_generation=7), outcome='source_verified'))
        self.closure = dict(final=dict(can_continue=True), attempts=[dict(summary=dict(physical=dict(
            expert_read_pipeline=dict(state='observed', value=self.read),
            image_generation=dict(state='observed', value=7))))])
        self.fields = dict(event='expert_read_decoder_invocation', status='completed',
            expert_read_counter_reset_detected=False, expert_read_pipeline_delta=json.dumps(self.stats))

    def ledger(self):
        lines = [json.dumps(dict(level='DEBUG', fields=self.fields))]
        with patch.object(run.ex, 'owner_summary', return_value=self.closure), patch.object(run.quality, 'read_observation'):
            return run.read_ledger(lines, dict(pid=PID), 'blocking')

    def test_actual_typed_read_counts_and_owner_image_are_required(self):
        result = self.ledger()
        self.assertEqual(result['model_source_generation'], 7)
        self.assertEqual(result['read']['stats']['io_started_reads'], 2)

    def test_missing_reset_or_mismatched_delta_fails(self):
        for delta in ({}, dict(io_started_reads=1, proved_read_bytes=32), dict(io_started_reads=2, proved_read_bytes=16)):
            with self.subTest(delta=delta):
                self.fields['expert_read_pipeline_delta'] = json.dumps(delta)
                with self.assertRaises((KeyError, ValueError)):
                    self.ledger()

    def test_complete_counts_do_not_waive_unobserved_or_failed_reads(self):
        for key in ('failed_reads', 'cancelled_reads', 'unknown_reads', 'unobserved_terminal_reads'):
            with self.subTest(key=key):
                self.stats[key] = 1
                with self.assertRaises(ValueError):
                    self.ledger()
                self.stats[key] = 0

    def test_known_source_generation_not_guessed_from_sql_labels(self):
        self.read['last_receipt']['identity']['image_generation'] = 8
        with self.assertRaisesRegex(ValueError, 'owner/source'):
            self.ledger()

    def test_partial_submission_balance_cannot_pass(self):
        self.stats['submitted_reads'] = 3
        with self.assertRaisesRegex(ValueError, 'reconciliation failed|positively balanced'):
            self.ledger()


class FinalIdleReplayTests(unittest.TestCase):
    def test_raw_memory_cannot_hide_behind_saved_pass(self):
        import gpu_idle_window_v2 as idle
        uuids = [f'GPU-00000000-0000-0000-0000-{i:012d}' for i in range(8)]
        raw = '\n'.join(f'{i}, {uuid}, 0 MiB, 0 %' for i, uuid in enumerate(uuids))
        samples = [dict(gpu=raw, apps='', classification='idle', consecutive_idle=i+1,
                        monotonic_started=i, monotonic_finished=i+0.1) for i in range(3)]
        receipt = dict(schema=idle.SCHEMA, status='verified', all_eight_idle=True, authenticated_echild=True,
            recorded_owners_absent=True, mode='post_authenticated_echild', deadline_seconds=15,
            required_consecutive=3, elapsed_seconds=2.1, accepted_sample_indices=[0, 1, 2], consecutive_idle=3, samples=samples)
        with patch.object(idle.subprocess, 'run', side_effect=AssertionError('offline replay must not query')):
            run.verify_final_idle(receipt, uuids)
            samples[1]['gpu'] = raw.replace('0 MiB', '1 MiB', 1)
            with self.assertRaisesRegex(ValueError, 'memory/apps'):
                run.verify_final_idle(receipt, uuids)



if __name__ == '__main__':
    unittest.main()
