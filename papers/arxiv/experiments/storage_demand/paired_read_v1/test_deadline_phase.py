"""Small CPU tests of real admission/coordinator clocks; no production build,
archive, model, GPU or server. Only candidate semantics, telemetry, UUID locks
and supervisor results are substitutes; receipt/journal/adapter code is real.
"""
from contextlib import ExitStack, contextmanager
import copy
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paired_read_v1 import run_v3 as run, run_actual_v2 as actual
from runnerhelpers.evidence_v2 import ct
from runnerhelpers import execution_v2 as ex

REAL_CANDIDATE_VALIDATOR = run.validate_transfer_candidate

UUIDS = [f'GPU-00000000-0000-0000-0000-{i:012d}' for i in range(8)]


class DeadlinePhaseTests(unittest.TestCase):
    @contextmanager
    def fixture(self, *, static_delay=63.480, active_delay=0, mutation=None,
                cleanup_failure=False, supervisor_failure=False, real_io=False):
        with tempfile.TemporaryDirectory(prefix='deadline-phase-') as temp, ExitStack() as stack:
            self.directory = root = Path(temp)
            self.now, self.events, self.payloads = 1000., [], []
            compiled = root / 'compiled-inputs'
            compiled.mkdir()
            evidence, workflow, mutable = compiled / 'architecture_target.rs', root / 'workflow.py', root / 'reference'
            for path in (evidence, workflow, mutable):
                path.write_text('CPU component fixture, not production evidence\n')
                path.chmod(0o400)
            self.evidence, self.mutable = evidence, mutable
            self.mutable_pin = ct.sha(mutable)
            layout = dict(purpose='public_cli_acceptance', cases={'blocking-synchronous': {}},
                artifact_pins={str(workflow): ct.sha(workflow)}, input_contract={'reference': str(mutable)},
                http_reference_binding=None, http_expectation_contract='CPU component only', upload_producer_contract={},
                limits=dict(outer_lifecycle_seconds=650, study_cleanup_reserve_seconds=100),
                schedule=ct.load(HERE / 'requests.json')['schedule'], trial_order=['A1-blocking-synchronous'],
                order=[dict(block='A1', modes=['blocking-synchronous'])], candidate=dict(uuid_inventory=UUIDS))
            self.layout = layout
            path = root / 'layout.json'
            ct.write(path, layout)
            path.chmod(0o400)
            self.path, self.trial = path, layout['trial_order'][0]
            registered = dict(schema='ferrule.storage-demand.actual-http-transfer-registration.v1',
                study_layout=layout, binding_path=str(path), execution_path=str(path),
                binding_sha256=ct.sha(path), execution_sha256=ct.sha(path), boot_id=actual.boot_id(),
                trial_order=layout['trial_order'], order=layout['order'], limits=layout['limits'], schedule=layout['schedule'],
                owner_authorization=True, owner_note=run.transfer_note(layout), eight_uuids=UUIDS,
                selected_uuid=UUIDS[0], started=ex.stamp(),
                pilot_deadline=10000. if not real_io else time.monotonic()+7200)
            ct.write(root / 'registration.json', dict(registered, digest=ct.digest(registered)))
            bound = dict(candidate=dict(binary=str(evidence), binary_sha256=ct.sha(evidence),
                runtime_pin_manifest=str(evidence), artifact_pins={str(evidence): ct.sha(evidence)}),
                uuid_inventory=UUIDS, build_review=dict(initial_read_producer={}),
                approval_subject=dict(purpose=layout['purpose'], workflow=ct.digest(layout['artifact_pins'])))
            required_pins = dict(bound['candidate']['artifact_pins'], **layout['artifact_pins'],
                **{str(mutable): self.mutable_pin})
            # Independent trusted-authority component fixture: issued before any
            # preparation receipt, never rebuilt from a receipt's current stats.
            with ex.absolute_alarm(time.monotonic()+3):
                anchor = dict(verified_inputs={name: run.hash_verified_identity(name, pin)
                    for name, pin in required_pins.items()}, verified_directories={})
            def authority(selected, **kwargs):
                return dict(bound_candidate=copy.deepcopy(bound), required_pins=dict(required_pins),
                    required_trees={str(compiled): [str(evidence)]}, required_globs={}, acceptance_roots=[],
                    identity_anchor=copy.deepcopy(anchor), always_hash={str(workflow), str(mutable)})
            def review(selected, **kwargs):
                if kwargs.get('freshness') is not None:
                    return REAL_CANDIDATE_VALIDATOR(selected, **kwargs)
                self.events.append('full-static-review')
                from runnerhelpers.build_receipts_v1 import inventory_tree
                inventory_tree(evidence.parent)
                ct.check_pins(bound['candidate']['artifact_pins'])
                if real_io:
                    time.sleep(static_delay)
                else:
                    self.now += static_delay
                if mutation == 'during-static':
                    self.rewrite(evidence)
                return copy.deepcopy(bound)
            def inputs(contract):
                if ct.sha(mutable) != self.mutable_pin:
                    raise ValueError('mutable reference changed')
            @contextmanager
            def lock(uuid):
                self.events.append('lock-enter')
                attempt = ct.load(root / self.trial / 'attempt.json')
                receipt = ct.load(attempt['static_preparation']['path'])
                self.assertEqual(receipt['trial'], self.trial)
                self.assertEqual(ct.sha(attempt['static_preparation']['path']), attempt['static_preparation']['sha256'])
                self.assertLessEqual(receipt['ended']['monotonic'], attempt['admission']['monotonic'])
                self.assertEqual(attempt['outer_deadline'] - attempt['admission']['monotonic'], 650)
                try:
                    yield dict(uuid=uuid)
                finally:
                    self.events.append('lock-exit')
            def idle(folder, label, uuids, deadline, custody=None):
                self.events.append(label)
                if label == 'coordinator-pre-idle':
                    self.now += active_delay
                    if mutation == 'before-launch':
                        self.rewrite(mutable)
                else:
                    self.assertNotIn('lock-exit', self.events)
                    if not self.payloads:
                        self.assertIsNone(custody)
                    if cleanup_failure:
                        raise OSError('forced post-idle failure')
                return dict(status='verified', all_eight_idle=True)
            def supervise(payload, seconds):
                self.events.append('supervisor-enter')
                self.assertEqual(seconds, 530)
                self.assertGreaterEqual(payload['outer_deadline'] - ex.time.monotonic(), 610)
                self.assertEqual(payload['request_ceiling'], 12)
                self.payloads.append(copy.deepcopy(payload))
                if supervisor_failure:
                    raise RuntimeError('primary supervisor error')
                return {'cpu_custody_substitute': True}
            for module, name, value in (
                (run, 'validate_transfer_layout', lambda *a, **k: copy.deepcopy(layout)),
                (run, 'validate_transfer_candidate', review), (run, 'transfer_candidate_authority', authority),
                (run, 'validate_transfer_inputs', inputs),
                (run, 'validate_http_references', lambda layout: None), (run, 'http_reference_context', lambda layout: None),
                (run, 'directory_check', lambda directory, **kwargs: Path(directory)),
                (run.planned, 'spec_for', lambda bound, uuid, mode: dict(argv=[str(evidence)], environment={})),
                (run.planned, 'supervise', supervise), (run, 'transfer_ledger', lambda *a: dict(can_continue=True)),
                (ct, 'exclusive_owner_lock', lock), (ct, 'validate_owner_lock', lambda *a: None),
                (ct, 'drained', lambda receipt: receipt.get('cpu_custody_substitute') is True),
                (actual.pilot.telemetry, 'authorize', lambda *a: None),
                (actual.pilot, 'idle', idle), (actual.pilot, 'baseline', lambda *a: {}),
                (actual.pilot.idle_window, 'no_recorded_owner', lambda receipt: receipt.get('cpu_custody_substitute') is True),
                (ex, 'lifecycle_assessment', lambda *a, **k: dict(can_continue=True, stop_reasons=[])),
                (ex, 'recover_observations', lambda *a: []),
                (ex, 'recover_requests', lambda *a: ex.censoring_summary([], layout['schedule'])),
            ):
                stack.enter_context(mock.patch.object(module, name, value))
            if not real_io:
                stack.enter_context(mock.patch.object(time, 'monotonic', side_effect=lambda: self.now))
            self.popen = stack.enter_context(mock.patch.object(ex.subprocess, 'Popen', side_effect=AssertionError('no Popen permitted')))
            yield
            self.popen.assert_not_called()

    @staticmethod
    def rewrite(path):
        before = path.stat()
        path.chmod(0o600)
        path.write_text('X' * before.st_size)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        path.chmod(0o400)

    def execute(self):
        return actual.execute_next(self.directory,
            admission=lambda directory: run.prepare_transfer_admission(directory, self.path))

    def test_static_delay_does_not_spend_active_650_or_610_reserve(self):
        with self.fixture():
            self.assertTrue(self.execute()['assessment']['can_continue'])
            self.assertEqual(self.events.count('full-static-review'), 1)
            self.assertLess(self.events.index('full-static-review'), self.events.index('lock-enter'))
            self.assertEqual(self.events.count('lock-enter'), 16)
            attempt = ct.load(self.directory / self.trial / 'attempt.json')
            receipt = ct.load(attempt['static_preparation']['path'])
            self.assertAlmostEqual(receipt['duration_seconds'], 63.480)
            self.assertEqual(attempt['admission']['monotonic'], 1063.480)
            self.assertAlmostEqual(attempt['static_preparation_seconds'], 63.480)
            self.assertFalse(receipt['throughput_denominator'])
            self.assertTrue(actual.coordinator_completion(self.directory / self.trial))

    def test_real_clock_static_delay_is_separate_without_early_popen(self):
        with self.fixture(static_delay=.03, real_io=True):
            self.assertTrue(self.execute()['assessment']['can_continue'])
            attempt = ct.load(self.directory / self.trial / 'attempt.json')
            receipt = ct.load(attempt['static_preparation']['path'])
            self.assertGreaterEqual(receipt['duration_seconds'], .025)
            self.assertGreaterEqual(attempt['admission']['monotonic'], receipt['ended']['monotonic'])

    def test_delay_after_active_start_refuses_and_cleans_without_echild(self):
        with self.fixture(active_delay=41):
            row = self.execute()
            folder = self.directory / self.trial
            self.assertFalse(row['assessment']['can_continue'])
            self.assertIn('insufficient absolute reserve', row['coordinator_error'])
            self.assertFalse(self.payloads)
            refusal = ct.load(folder / 'worker-refusal.json')
            self.assertEqual(refusal['worker_state'], 'never_spawned')
            self.assertFalse(refusal['authenticated_echild'])
            self.assertEqual(refusal['physical_owner_release'], 'unverified')
            self.assertEqual(row['summary']['issued'], 0)
            self.assertEqual(self.events.count('lock-enter'), 16)
            self.assertTrue((folder / 'locks-reprobed.json').is_file())
            self.assertTrue(actual.never_spawned(folder))
            reference = ct.load(folder / 'attempt.json')['static_preparation']
            with run.transfer_adapter(self.path, prepared=reference):
                final = actual.finalize(self.directory)
            self.assertEqual(final['actual_issued_lifecycles'], 0)
            self.assertFalse(final['final_authenticated_echild'])
            self.assertTrue(final['final_all_eight_idle'])

    def test_candidate_staled_during_preparation_rejected_before_locks(self):
        with self.fixture(mutation='during-static'):
            with self.assertRaisesRegex(ValueError, 'stale after static review'):
                self.execute()
            self.assertNotIn('lock-enter', self.events)
            receipt = ct.load(self.directory / ('static-refusal-' + self.trial + '.json'))
            self.assertEqual(receipt['worker_state'], 'never_spawned')
            self.assertFalse(receipt['gpu_locks_acquired'])

    def test_mutable_reference_revalidation_still_blocks_launch(self):
        with self.fixture(mutation='before-launch'):
            row = self.execute()
            self.assertFalse(row['assessment']['can_continue'])
            self.assertIn('reference', row['coordinator_error'])
            self.assertFalse(self.payloads)
            self.assertTrue((self.directory / self.trial / 'locks-reprobed.json').is_file())

    def test_finite_static_deadline_does_not_start_arm(self):
        with self.fixture(static_delay=181):
            with self.assertRaisesRegex(ValueError, 'static preparation deadline'):
                self.execute()
            self.assertFalse((self.directory / self.trial).exists())
            self.assertNotIn('lock-enter', self.events)

    def test_cleanup_failure_preserves_primary_refusal_and_never_heals(self):
        with self.fixture(active_delay=41, cleanup_failure=True):
            row = self.execute()
            folder = self.directory / self.trial
            self.assertIn('insufficient absolute reserve', ct.load(folder / 'assessment.json')['coordinator_error'])
            self.assertFalse(row['assessment']['can_continue'])
            self.assertIn('forced post-idle failure', row['coordinator_failure']['reason'])
            self.assertEqual(row['coordinator_failure']['assessment_sha256'], ct.sha(folder / 'assessment.json'))
            self.assertTrue((folder / 'locks-reprobed.json').is_file())
            with self.assertRaisesRegex(ValueError, 'permanently stopped'):
                actual.next_trial(self.directory, dict(trial_order=[self.trial]))

    def test_unknown_supervisor_custody_cannot_invent_never_spawned_or_idle(self):
        with self.fixture(supervisor_failure=True):
            row = self.execute()
            folder = self.directory / self.trial
            self.assertIn('primary supervisor error', row['coordinator_error'])
            self.assertFalse((folder / 'worker-refusal.json').exists())
            self.assertNotIn('coordinator-post-idle', self.events)
            self.assertFalse(row['post_idle']['all_eight_idle'])
            self.assertTrue((folder / 'locks-reprobed.json').exists())

    def test_receipt_layout_workflow_candidate_substitutions_rejected(self):
        for mutation in ('receipt', 'layout', 'workflow', 'candidate'):
            with self.subTest(mutation=mutation), self.fixture():
                with run.prepare_transfer_admission(self.directory, self.path):
                    reference = actual.binding.static_preparation
                    target = {'receipt': Path(reference['path']), 'layout': self.path,
                        'workflow': Path(next(iter(self.layout['artifact_pins']))), 'candidate': self.evidence}[mutation]
                    self.rewrite(target)
                    with self.assertRaisesRegex(ValueError, 'changed|Stale|content hash differs'):
                        run.validate_static_preparation(reference, self.path)
                    self.assertNotIn('lock-enter', self.events)

    def test_candidate_mutation_during_revalidation_rejected_before_locks(self):
        with self.fixture():
            with run.prepare_transfer_admission(self.directory, self.path):
                reference = actual.binding.static_preparation
                with mock.patch.object(run, 'validate_transfer_inputs', side_effect=lambda *a: self.rewrite(self.evidence)):
                    with self.assertRaisesRegex(ValueError, 'changed during revalidation'):
                        run.validate_static_preparation(reference, self.path)
                self.assertNotIn('lock-enter', self.events)

    def test_real_static_alarm_bounds_a_blocking_review(self):
        with self.fixture(static_delay=.05, real_io=True), mock.patch.object(run, 'STATIC_PREPARATION_SECONDS', .01):
            with self.assertRaisesRegex(TimeoutError, 'absolute pilot phase deadline'):
                self.execute()
            self.assertNotIn('lock-enter', self.events)
            self.assertFalse((self.directory / self.trial).exists())
            self.assertTrue((self.directory / 'study-stop.json').exists())

    def test_reprobe_failure_after_primary_refusal_appends_without_rewriting(self):
        with self.fixture(active_delay=41):
            write = ct.write
            def fail_reprobe(path, value):
                if Path(path).name == 'locks-reprobed.json':
                    raise OSError('forced reprobe receipt failure')
                write(path, value)
            # The adapter uses the maintained exclusive writer, not a replaceable
            # receipt writer. Inject at that narrow boundary for this component.
            with mock.patch.object(run, 'ORIGINAL_WRITE', side_effect=fail_reprobe):
                row = self.execute()
            folder = self.directory / self.trial
            self.assertIn('insufficient absolute reserve', ct.load(folder / 'assessment.json')['coordinator_error'])
            self.assertIn('forced reprobe receipt failure', row['coordinator_failure']['reason'])
            self.assertFalse(row['assessment']['can_continue'])
            self.assertEqual(row['coordinator_failure']['assessment_sha256'], ct.sha(folder / 'assessment.json'))
            self.assertEqual(self.events.count('lock-enter'), 16)

    def test_capture_prepared_branch_preserves_original_target_cap_and_rejects_expiry(self):
        import _profile_capture as wrapper
        for expired in (False, True):
            with self.subTest(expired=expired), self.fixture():
                with run.prepare_transfer_admission(self.directory, self.path):
                    reference = actual.binding.static_preparation
                    prepared = run.validate_static_preparation(reference, self.path)
                    bound = prepared['bound_candidate']
                    folder = self.directory / self.trial
                    folder.mkdir()
                    spec = dict(argv=[str(self.evidence)], environment={}, static_preparation=reference,
                        input_contract=self.layout['input_contract'], frozen_cli_sha256=ct.sha(self.evidence))
                    payload = dict(directory=str(folder), spec=spec, adapter_pins=self.layout['artifact_pins'],
                        execution_path=str(self.path), execution_sha256=ct.sha(self.path), eight_uuids=UUIDS,
                        all_uuid_locks=[{} for uuid in UUIDS], coordinator=dict(pid=os.getpid()), limits=dict(shutdown_seconds=20))
                    ct.write(folder / 'worker-launch.json', payload)
                    target_deadline = self.now + 100
                    @contextmanager
                    def pinned(*args):
                        if expired:
                            self.now = target_deadline + 1
                        yield
                    with mock.patch.object(ex, 'authenticate_capture_parent', return_value={}), \
                            mock.patch.object(run, 'validate_http_capture_entry', return_value=bound), \
                            mock.patch.object(ct, 'pinned_executable', side_effect=pinned), \
                            mock.patch.object(wrapper, 'run_target', return_value=0) as target:
                        if expired:
                            with self.assertRaisesRegex(ValueError, 'absolute target deadline expired'):
                                ex.http_capture_main(folder, ct.sha(folder / 'worker-launch.json'), target_deadline)
                            target.assert_not_called()
                        else:
                            self.assertEqual(ex.http_capture_main(folder, ct.sha(folder / 'worker-launch.json'), target_deadline), 0)
                            self.assertEqual(target.call_args.kwargs['absolute_deadline'], target_deadline)
                            self.assertEqual(target.call_args.args[2], 100)

    def test_delayed_file_hash_io_is_static_and_not_repeated_in_active_setup(self):
        with self.fixture(static_delay=0, real_io=True):
            original_sha, reads = ct.sha, []
            def delayed_sha(path):
                value = original_sha(path)
                if Path(path) == self.evidence:
                    reads.append(path)
                    time.sleep(.03)  # Injected I/O latency, not production timing.
                return value
            with mock.patch.object(ct, 'sha', side_effect=delayed_sha):
                self.assertTrue(self.execute()['assessment']['can_continue'])
            attempt = ct.load(self.directory / self.trial / 'attempt.json')
            receipt = ct.load(attempt['static_preparation']['path'])
            self.assertGreaterEqual(receipt['duration_seconds'], .025)
            self.assertGreaterEqual(attempt['static_preparation_seconds'], receipt['duration_seconds'])
            self.assertEqual(len(reads), 1)

    def test_worker_dispatch_reuses_prepared_receipt_without_repeating_review(self):
        import io
        import json
        import signal
        with self.fixture():
            with run.prepare_transfer_admission(self.directory, self.path):
                reference = actual.binding.static_preparation
                incoming = io.StringIO(json.dumps(dict(spec=dict(static_preparation=reference), outer_deadline=self.now+650, pilot_deadline=10000)))
                previous = signal.getsignal(signal.SIGTERM)
                try:
                    with mock.patch.object(ct, 'require_cpu_python'), mock.patch.object(sys, 'stdin', incoming), \
                            mock.patch.object(run.planned, 'worker', return_value=0) as worker:
                        self.assertEqual(run.main(['--_worker', 'cpu-only-test-hash', '--layout', str(self.path)]), 0)
                        worker.assert_called_once_with('cpu-only-test-hash')
                        self.assertIs(sys.stdin, incoming)
                finally:
                    signal.signal(signal.SIGTERM, previous)
                self.assertEqual(self.events.count('full-static-review'), 1)
                self.assertNotIn('lock-enter', self.events)

    def test_compiled_inventory_addition_rejected_without_new_review(self):
        with self.fixture():
            with run.prepare_transfer_admission(self.directory, self.path):
                reference = actual.binding.static_preparation
                self.evidence.with_name('extra-input').write_text('unreviewed')
                with self.assertRaisesRegex(ValueError, 'inventory changed'):
                    run.validate_static_preparation(reference, self.path)
                self.assertEqual(self.events.count('full-static-review'), 1)
                self.assertNotIn('lock-enter', self.events)

    def test_delay_in_final_admission_setup_cannot_escape_static_deadline(self):
        with self.fixture():
            @contextmanager
            def admission(directory):
                with run.prepare_transfer_admission(directory, self.path):
                    original = actual.binding.validate_execution
                    def delayed(path):
                        value = original(path)
                        self.now += 181
                        return value
                    with mock.patch.object(actual.binding, 'validate_execution', side_effect=delayed):
                        yield
            with self.assertRaisesRegex(ValueError, 'static admission deadline exceeded'):
                actual.execute_next(self.directory, admission=admission)
            self.assertNotIn('lock-enter', self.events)
            refusal = ct.load(self.directory / ('static-refusal-' + self.trial + '.json'))
            self.assertEqual(refusal['worker_state'], 'never_spawned')
            self.assertFalse(refusal['gpu_locks_acquired'])

    def test_directory_permission_mutation_invalidates_preparation(self):
        with self.fixture():
            with run.prepare_transfer_admission(self.directory, self.path):
                reference = actual.binding.static_preparation
                self.evidence.parent.chmod(0o700)
                with self.assertRaisesRegex(ValueError, 'directory changed'):
                    run.validate_static_preparation(reference, self.path)

    def test_failed_assessment_and_later_reprobe_error_are_both_append_only(self):
        with self.fixture():
            write = ct.write
            def fail_reprobe(path, value):
                if Path(path).name == 'locks-reprobed.json':
                    raise OSError('later reprobe failure')
                write(path, value)
            with mock.patch.object(ex, 'lifecycle_assessment', return_value=dict(
                    can_continue=False, stop_reasons=['primary quality failure'])), \
                    mock.patch.object(run, 'ORIGINAL_WRITE', side_effect=fail_reprobe):
                row = self.execute()
            folder = self.directory / self.trial
            self.assertEqual(ct.load(folder / 'assessment.json')['assessment']['stop_reasons'], ['primary quality failure'])
            self.assertIn('primary quality failure', ct.load(folder / 'coordinator-failure.json')['reason'])
            self.assertIn('later reprobe failure', ct.load(folder / 'coordinator-cleanup-errors.json')['errors'][0])
            self.assertFalse(row['assessment']['can_continue'])


if __name__ == '__main__':
    unittest.main()
