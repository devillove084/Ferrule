"""Reviewer regressions: fresh forged receipts, original GO and earliest STOP.
CPU-only. No source/native archive read, build, GPU query or server launch.
"""
from contextlib import contextmanager
import copy
import io
import json
import os
import signal
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

ACCEPTANCE = HERE / 'artifacts-transfer-acceptance-37b2f40-20261017-01/paired-final.json'


class ReceiptAuthorityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.accepted = ct.load(ACCEPTANCE)['study_layout']

    @contextmanager
    def no_launch(self):
        with mock.patch.object(ex.subprocess, 'Popen', side_effect=AssertionError('no process launch')) as process, \
                mock.patch.object(ct, 'exclusive_owner_lock', side_effect=AssertionError('no UUID locks')) as locks, \
                mock.patch.object(actual.pilot.telemetry, 'authorize', side_effect=AssertionError('no GPU authority')):
            yield
            process.assert_not_called()
            locks.assert_not_called()

    def forged(self, directory, layout, *, nonempty=False):
        path = directory / 'layout.json'
        ct.write(path, layout)
        path.chmod(0o400)
        registration = dict(schema='ferrule.storage-demand.actual-http-transfer-registration.v1',
            study_layout=layout, binding_path=str(path), execution_path=str(path),
            binding_sha256=ct.sha(path), execution_sha256=ct.sha(path), boot_id=actual.boot_id())
        ct.write(directory / 'registration.json', dict(registration, digest=ct.digest(registration)))
        fake = dict(candidate=dict(binary='/usr/bin/true', binary_sha256=ct.sha('/usr/bin/true'),
            runtime_pin_manifest='/usr/bin/true', artifact_pins={}),
            approval_subject=dict(purpose=layout['purpose'], claimed='self-consistent, not authority'))
        inputs = {str(path): dict(sha256=ct.sha(path), identity=run.preparation_identity(path))} if nonempty else {}
        admission = {str(path): ct.sha(path), str(directory / 'registration.json'): ct.sha(directory / 'registration.json')}
        value = dict(schema='ferrule.storage-demand.http-static-preparation.v2',
            trial=layout['trial_order'][0], boot_id=actual.boot_id(), layout_path=str(path),
            layout_sha256=ct.sha(path), registration_sha256=admission[str(directory / 'registration.json')],
            layout=layout, workflow_pins=layout['artifact_pins'], bound_candidate=fake,
            approval_subject=fake['approval_subject'], admission_pins=admission,
            started=dict(monotonic=1), ended=dict(monotonic=2), deadline=181,
            throughput_denominator=False, verified_inputs=inputs, verified_trees={}, verified_globs={}, verified_directories={})
        receipt = directory / 'new-forged-preparation.json'
        ct.write(receipt, value)
        receipt.chmod(0o400)
        return path, dict(path=str(receipt), sha256=ct.sha(receipt)), value

    def test_new_selfconsistent_receipt_cannot_admit_actual_null_candidate_layout(self):
        layout = run.transfer_plan('public_cli_acceptance', compiled_identity=self.accepted['compiled_identity'],
            http_reference_binding=self.accepted['http_reference_binding'])
        self.assertIsNone(layout['candidate'])
        for nonempty in (False, True):
            with self.subTest(nonempty=nonempty), tempfile.TemporaryDirectory() as temp, self.no_launch():
                path, reference, value = self.forged(Path(temp), layout, nonempty=nonempty)
                with mock.patch.object(run, 'validate_transfer_candidate', wraps=run.validate_transfer_candidate) as candidate:
                    with self.assertRaisesRegex(ValueError, 'candidate pending'):
                        run.validate_static_preparation(reference, path)
                    self.assertGreater(candidate.call_count, 0)

    def test_new_nonempty_fake_cli_receipt_requires_real_purpose_main_go(self):
        layout = run.transfer_plan('public_cli_acceptance', compiled_identity=self.accepted['compiled_identity'],
            http_reference_binding=self.accepted['http_reference_binding'])
        layout['candidate'] = copy.deepcopy(self.accepted['candidate'])
        with tempfile.TemporaryDirectory() as temp, self.no_launch():
            path, reference, value = self.forged(Path(temp), layout, nonempty=True)
            with mock.patch.object(run, 'authorize_transfer_candidate', wraps=run.authorize_transfer_candidate) as authorize:
                with self.assertRaisesRegex(ValueError, 'purpose/layout/workflow/evidence-bound GO'):
                    run.validate_static_preparation(reference, path)
                self.assertGreater(authorize.call_count, 0)

    def test_real_original_manifest_go_and_exact_required_universe(self):
        authority = run.transfer_candidate_authority(self.accepted, historical_workflow=True)
        self.assertEqual(authority['bound_candidate']['candidate']['binary_sha256'],
            'c3db43b8c556a210a77f7fac29dc23c4bc9399d5e0374e6dc9fecd016625ccaa')
        self.assertGreater(len(authority['required_pins']), 18000)
        with tempfile.TemporaryDirectory() as temp:
            path, reference, value = self.forged(Path(temp), self.accepted, nonempty=True)
            with mock.patch.object(run, 'authorize_transfer_candidate', wraps=run.authorize_transfer_candidate) as authorize:
                with self.assertRaisesRegex(ValueError, 'complete required pin coverage'):
                    run.validate_transfer_candidate(self.accepted, freshness=value, admission_pins=value['admission_pins'])
                self.assertGreater(authorize.call_count, 0)
            value['verified_inputs'] = {name: dict(sha256=pin) for name, pin in
                dict(authority['required_pins'], **value['admission_pins']).items()}
            value['verified_trees'] = authority['required_trees']
            value['verified_globs'] = authority['required_globs']
            value['verified_directories'] = dict.fromkeys(run.coverage_directories(authority))
            missing = copy.deepcopy(value)
            missing['bound_candidate'] = authority['bound_candidate']
            missing['approval_subject'] = authority['bound_candidate']['approval_subject']
            dependency = next(iter(ct.load(ct.load(self.accepted['candidate']['inputs'])['inputs']['dependencies'])['pins']))
            del missing['verified_inputs'][dependency]
            with self.assertRaisesRegex(ValueError, 'complete required pin coverage'):
                run.validate_prepared_coverage(missing, authority, missing['admission_pins'])
            value['review_result_sha256'] = ct.digest(value['bound_candidate'])
            with self.assertRaisesRegex(ValueError, 'static candidate authority changed'):
                run.validate_prepared_coverage(value, authority, value['admission_pins'])


class InitialRefusalTests(unittest.TestCase):
    @contextmanager
    def fixture(self):
        with tempfile.TemporaryDirectory(prefix='initial-refusal-') as temp, \
                mock.patch.object(ct, 'exclusive_owner_lock', side_effect=AssertionError('UUID lock forbidden')) as locks, \
                mock.patch.object(run.planned, 'supervise', side_effect=AssertionError('supervisor forbidden')) as supervisor, \
                mock.patch.object(actual.pilot.telemetry, 'authorize', side_effect=AssertionError('GPU forbidden')) as authorize:
            self.directory = Path(temp)
            self.locks, self.supervisor = locks, supervisor
            yield
            locks.assert_not_called()
            supervisor.assert_not_called()
            authorize.assert_not_called()

    def execute(self):
        return actual.execute_next(self.directory, admission=lambda d: run.prepare_transfer_admission(d, d / 'not-read-layout.json'))

    def test_initial_registration_read_is_durable_stop_with_unknown_trial(self):
        with self.fixture(), mock.patch.object(run.historical, 'unseal', side_effect=OSError('primary initial read')):
            with self.assertRaisesRegex(OSError, 'primary initial read'):
                self.execute()
            self.assertIn('primary initial read', ct.load(self.directory / 'study-stop.json')['reason'])
            refusal = ct.load(self.directory / 'static-refusal.json')
            self.assertIsNone(refusal['trial'])
            self.assertIsNone(refusal['deadline'])
            self.assertEqual(refusal['worker_state'], 'never_spawned')

    def test_boot_and_trial_lookup_failures_never_issue_an_arm(self):
        for component in ('boot', 'trial'):
            with self.subTest(component=component), self.fixture():
                value = dict(boot_id='wrong' if component == 'boot' else actual.boot_id())
                with mock.patch.object(run.historical, 'unseal', return_value=value), \
                        mock.patch.object(actual, 'next_trial', side_effect=RuntimeError('primary trial lookup')):
                    with self.assertRaisesRegex(ValueError if component == 'boot' else RuntimeError,
                            'boot changed' if component == 'boot' else 'primary trial lookup'):
                        self.execute()
                self.assertTrue((self.directory / 'study-stop.json').exists())
                self.assertIsNone(ct.load(self.directory / 'static-refusal.json')['trial'])

    def test_first_alarm_entry_failure_is_inside_fail_closed_boundary(self):
        @contextmanager
        def expired(deadline):
            raise TimeoutError('primary first admission timeout')
            yield
        with self.fixture(), mock.patch.object(ex, 'absolute_alarm', side_effect=expired), \
                mock.patch.object(run.historical, 'unseal') as read:
            with self.assertRaisesRegex(TimeoutError, 'primary first admission timeout'):
                self.execute()
            read.assert_not_called()
            self.assertTrue((self.directory / 'study-stop.json').exists())
            self.assertIsNone(ct.load(self.directory / 'static-refusal.json')['deadline'])

    def test_context_factory_failure_is_caught_before_context_entry(self):
        with self.fixture():
            with self.assertRaisesRegex(RuntimeError, 'primary context factory'):
                actual.execute_next(self.directory, admission=mock.Mock(side_effect=RuntimeError('primary context factory')))
            self.assertTrue((self.directory / 'study-stop.json').exists())
            self.assertIsNone(ct.load(self.directory / 'static-refusal.json')['trial'])

    def test_stop_or_refusal_write_failure_preserves_primary_and_attempts_both(self):
        for broken in ('stop', 'refusal', 'both'):
            with self.subTest(broken=broken), self.fixture():
                original_write, writes = ct.write, []
                def write(path, value):
                    writes.append(Path(path).name)
                    if (Path(path).name == 'study-stop.json' and broken in ('stop', 'both')) or (
                            Path(path).name == 'static-refusal.json' and broken in ('refusal', 'both')):
                        raise OSError('secondary journal failure')
                    original_write(path, value)
                primary = RuntimeError('primary initial component')
                with mock.patch.object(run.historical, 'unseal', side_effect=primary), \
                        mock.patch.object(ct, 'write', side_effect=write), \
                        mock.patch.object(actual, 'stop', wraps=actual.stop) as stop:
                    with self.assertRaisesRegex(RuntimeError, 'primary initial component') as caught:
                        self.execute()
                    self.assertIs(caught.exception, primary)
                    self.assertGreater(stop.call_count, 0)
                    self.assertIn('study-stop.json', writes)
                    self.assertIn('static-refusal.json', writes)
                    self.assertTrue(any('journal failure' in note for note in caught.exception.__notes__))
                if broken == 'refusal':
                    self.assertTrue((self.directory / 'study-stop.json').exists())
                if broken == 'stop':
                    self.assertTrue((self.directory / 'static-refusal.json').exists())

    def test_real_initial_registration_timeout_stops_before_trial_is_known(self):
        def slow_read(*args):
            time.sleep(.05)
            return {}
        with self.fixture(), mock.patch.object(run, 'STATIC_PREPARATION_SECONDS', .01), \
                mock.patch.object(run.historical, 'unseal', side_effect=slow_read):
            with self.assertRaisesRegex(TimeoutError, 'absolute pilot phase deadline'):
                self.execute()
            self.assertTrue((self.directory / 'study-stop.json').exists())
            self.assertIsNone(ct.load(self.directory / 'static-refusal.json')['trial'])

    def test_initial_exclusive_preparation_path_refuses_without_fake_arm(self):
        with self.fixture():
            trial = 'A1-blocking-synchronous'
            ct.write(self.directory / ('static-preparation-' + trial + '.json'), {})
            with mock.patch.object(run.historical, 'unseal', return_value=dict(
                    boot_id=actual.boot_id(), pilot_deadline=time.monotonic()+7200)), \
                    mock.patch.object(actual, 'next_trial', return_value=trial):
                with self.assertRaisesRegex(ValueError, 'static preparation exclusive'):
                    self.execute()
            self.assertTrue((self.directory / 'study-stop.json').exists())
            self.assertFalse((self.directory / trial).exists())
            self.assertEqual(ct.load(self.directory / ('static-refusal-' + trial + '.json'))['trial'], trial)

    def test_insufficient_persisted_study_budget_stops_before_active_clock(self):
        with self.fixture():
            trial = 'A1-blocking-synchronous'
            with mock.patch.object(run.historical, 'unseal', return_value=dict(
                    boot_id=actual.boot_id(), pilot_deadline=time.monotonic()+700)), \
                    mock.patch.object(actual, 'next_trial', return_value=trial):
                with self.assertRaisesRegex(ValueError, 'absolute deadline expired'):
                    self.execute()
            refusal = ct.load(self.directory / ('static-refusal-' + trial + '.json'))
            self.assertLess(refusal['deadline'], refusal['started']['monotonic'])
            self.assertTrue((self.directory / 'study-stop.json').exists())
            self.assertFalse((self.directory / trial).exists())

    def test_public_next_entry_initial_failure_preserves_primary_and_stops(self):
        with self.fixture():
            primary = OSError('public next initial registration error')
            previous = signal.getsignal(signal.SIGTERM)
            try:
                with mock.patch.object(ct, 'require_cpu_python'), \
                        mock.patch.object(run.historical, 'unseal', side_effect=primary):
                    with self.assertRaisesRegex(OSError, 'public next initial registration error') as caught:
                        run.main(['next', '--study', 'transfer-pilot-v1', '--layout',
                            str(self.directory / 'unread-layout.json'), '--directory', str(self.directory)])
                    self.assertIs(caught.exception, primary)
            finally:
                signal.signal(signal.SIGTERM, previous)
            self.assertTrue((self.directory / 'study-stop.json').exists())
            self.assertIsNone(ct.load(self.directory / 'static-refusal.json')['trial'])

    def test_refusal_alone_blocks_second_public_next_after_stop_only_write_failure(self):
        with self.fixture():
            original_write = ct.write
            primary = OSError('original admission read failed')
            def stop_write_fails(path, value):
                if Path(path).name == 'study-stop.json':
                    raise PermissionError('STOP-only filesystem failure')
                original_write(path, value)
            with mock.patch.object(ct, 'write', side_effect=stop_write_fails), \
                    mock.patch.object(run.historical, 'unseal', side_effect=primary):
                with self.assertRaises(OSError) as caught:
                    self.execute()
                self.assertIs(caught.exception, primary)
            refusal = self.directory / 'static-refusal.json'
            original = refusal.read_bytes()
            self.assertFalse((self.directory / 'study-stop.json').exists())
            with self.assertRaisesRegex(ValueError, 'admission refused'):
                actual.next_trial(self.directory, dict(trial_order=['A1-blocking-synchronous']))
            previous = signal.getsignal(signal.SIGTERM)
            try:
                with mock.patch.object(ct, 'require_cpu_python'), \
                        mock.patch.object(run.historical, 'unseal', side_effect=AssertionError('no second registration read')) as read:
                    with self.assertRaisesRegex(ValueError, 'admission refused'):
                        run.main(['next', '--study', 'transfer-pilot-v1', '--directory', str(self.directory),
                            '--layout', str(self.directory / 'never-read-layout.json')])
                    read.assert_not_called()
            finally:
                signal.signal(signal.SIGTERM, previous)
            self.assertEqual(refusal.read_bytes(), original)
            self.assertIn('original admission read failed', ct.load(refusal)['reason'])
            self.assertFalse((self.directory / 'study-stop.json').exists())

    def test_direct_preparation_and_discovery_block_partial_or_known_trial_refusals(self):
        for name in ('static-refusal.json', 'static-refusal-A1-blocking-synchronous.json'):
            with self.subTest(name=name), self.fixture():
                path = self.directory / name
                path.write_text('{partial refusal')
                before = path.read_bytes()
                with self.assertRaisesRegex(ValueError, 'admission refused'):
                    actual.next_trial(self.directory, dict(trial_order=['A1-blocking-synchronous']))
                with mock.patch.object(run.historical, 'unseal') as read:
                    with self.assertRaisesRegex(ValueError, 'admission refused'):
                        with run.prepare_transfer_admission(self.directory, self.directory / 'not-read-layout.json'):
                            self.fail('refusal must never yield')
                    read.assert_not_called()
                self.assertEqual(path.read_bytes(), before)


class CompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = ct.load(ACCEPTANCE)
        cls.accepted = cls.report['study_layout']

    def prospective(self):
        return run.prospective_transfer_plan(ACCEPTANCE)

    def test_plan_reuses_exact_sealed_cli_but_has_no_new_main_go_or_approval(self):
        layout = self.prospective()
        self.assertEqual(layout['compiled_identity'], self.accepted['compiled_identity'])
        self.assertEqual((layout['scheduled_starts'], layout['scheduled_requests']), (8, 96))
        self.assertIsNone(layout['candidate']['launch_review'])
        binding = layout['candidate']['acceptance']
        plan = run.workflow_compatibility_plan(layout, binding)
        self.assertIsNone(plan['review'])
        self.assertFalse(plan['gpu_authorized'])
        self.assertEqual(plan['subject']['acceptance']['sha256'], run.COMPATIBILITY_ACCEPTANCE_SHA256)
        with self.assertRaises(ValueError):
            run.validate_transfer_candidate(layout, freshness={})

    def test_old_acceptance_default_remains_strict_and_bridge_plan_is_not_approval(self):
        layout = self.prospective()
        binding = layout['candidate']['acceptance']
        with self.assertRaisesRegex(ValueError, 'exact fixed four-cell plan'):
            run.check_acceptance_layout(binding, layout, self.accepted)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'pending-compatibility.json'
            ct.write(path, run.workflow_compatibility_plan(layout, binding))
            path.chmod(0o400)
            binding['workflow_compatibility'] = dict(path=str(path), sha256=ct.sha(path))
            with self.assertRaisesRegex(ValueError, 'human compatibility review pending'):
                run.check_acceptance_layout(binding, layout, self.accepted)

    def test_explicit_reviewed_fixture_bridge_preserves_original_identities(self):
        layout = self.prospective()
        binding = layout['candidate']['acceptance']
        plan = run.workflow_compatibility_plan(layout, binding)
        # Temporary CPU fixture only; not a real compatibility approval or MAIN-GO.
        plan['status'] = 'reviewed_for_prospective_orchestration_only'
        plan['review'] = dict(decision='APPROVE', reviewer='CPU component fixture, not execution approval',
            subject_sha256=ct.digest(plan['subject']))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'cpu-fixture-compatibility.json'
            ct.write(path, plan)
            path.chmod(0o400)
            binding['workflow_compatibility'] = dict(path=str(path), sha256=ct.sha(path))
            run.check_acceptance_layout(binding, layout, self.accepted)
            run.validate_original_workflow(self.accepted)
            subject = run.approval_subject(layout, dict(artifact_pins={'fixture': 'a'*64}, source={},
                binary_sha256=layout['compiled_identity']['binary_sha256'],
                runtime_pin_manifest=ct.load(layout['candidate']['inputs'])['inputs']['runtime_pin_manifest']))
            self.assertEqual(subject['workflow_compatibility_sha256'], binding['workflow_compatibility']['sha256'])
            with self.assertRaisesRegex(ValueError, 'purpose/layout/workflow/evidence-bound GO'):
                run.transfer_candidate_authority(layout)
            self.assertEqual(plan['subject']['original_workflow_pins'], self.accepted['artifact_pins'])
            self.assertNotEqual(layout['artifact_pins'], self.accepted['artifact_pins'])

    def test_quality_eos_reference_quota_producer_and_compiled_source_mutations_reject(self):
        for field in ('limits', 'http_reference_binding', 'upload_producer_contract', 'production_source_pins'):
            layout = self.prospective()
            with self.subTest(field=field):
                layout[field] = (dict(layout[field], unreviewed_change=True) if isinstance(layout[field], dict)
                    else str(layout[field]) + '-unreviewed')
                with self.assertRaisesRegex(ValueError, 'quality-relevant'):
                    run.workflow_compatibility_plan(layout, layout['candidate']['acceptance'])
        layout = self.prospective()
        quality_path = str(HERE / 'quality.py')
        layout['artifact_pins'][quality_path] = '0'*64
        with self.assertRaisesRegex(ValueError, 'outside orchestration scope'):
            run.workflow_compatibility_plan(layout, layout['candidate']['acceptance'])

    def test_internally_rehashed_bridge_forgery_is_not_a_reviewed_subject(self):
        layout = self.prospective()
        binding = layout['candidate']['acceptance']
        plan = run.workflow_compatibility_plan(layout, binding)
        plan['subject']['original_layout']['sha256'] = '0'*64
        plan['status'] = 'reviewed_for_prospective_orchestration_only'
        plan['review'] = dict(decision='APPROVE', reviewer='forged CPU fixture', subject_sha256=ct.digest(plan['subject']))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'new-selfconsistent-forged-bridge.json'
            ct.write(path, plan)
            path.chmod(0o400)
            binding['workflow_compatibility'] = dict(path=str(path), sha256=ct.sha(path))
            with self.assertRaisesRegex(ValueError, 'human compatibility review pending/changed'):
                run.validate_workflow_compatibility(binding, layout)

    def test_public_plan_routes_to_pending_eight_arm_reuse_without_a_build(self):
        previous = signal.getsignal(signal.SIGTERM)
        output = io.StringIO()
        try:
            with mock.patch.object(ct, 'require_cpu_python'), mock.patch.object(sys, 'stdout', output), \
                    mock.patch.object(run, 'review_transfer_build', side_effect=AssertionError('plan must not review/build/archive')):
                self.assertEqual(run.main(['plan', '--acceptance-final', str(ACCEPTANCE)]), 0)
        finally:
            signal.signal(signal.SIGTERM, previous)
        layout = json.loads(output.getvalue())
        self.assertEqual(layout['scheduled_requests'], 96)
        self.assertIsNone(layout['candidate']['launch_review'])
        self.assertEqual(layout['compiled_identity']['binary_sha256'], self.accepted['compiled_identity']['binary_sha256'])

    def test_bridged_gate_revalidates_actual_four_historical_cells_without_archive_or_gpu(self):
        layout = self.prospective()
        binding = layout['candidate']['acceptance']
        plan = run.workflow_compatibility_plan(layout, binding)
        plan['status'] = 'reviewed_for_prospective_orchestration_only'
        plan['review'] = dict(decision='APPROVE', reviewer='CPU fixture only, not approval',
            subject_sha256=ct.digest(plan['subject']))
        original = run.transfer_candidate_authority(self.accepted, historical_workflow=True)['bound_candidate']['build_review']
        original = copy.deepcopy(original)
        del original['artifact_pins'][self.accepted['candidate']['launch_review']]
        original_sha, original_popen = ct.sha, ex.subprocess.Popen
        def no_archive(name):
            if Path(name).suffix in ('.tar', '.a'):
                raise AssertionError('no archive content review in this CPU regression')
            return original_sha(name)
        def cpu_tools_only(argv, **kwargs):
            if Path(argv[0]).name not in ('git', 'readelf'):
                raise AssertionError('no worker/server or other process: ' + str(argv))
            return original_popen(argv, **kwargs)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'cpu-only-compatibility.json'
            ct.write(path, plan)
            path.chmod(0o400)
            binding['workflow_compatibility'] = dict(path=str(path), sha256=ct.sha(path))
            with mock.patch.object(run, 'review_transfer_build', return_value=copy.deepcopy(original)), \
                    mock.patch.object(ct, 'sha', side_effect=no_archive), \
                    mock.patch.object(ex.subprocess, 'Popen', side_effect=cpu_tools_only), \
                    mock.patch.object(ct, 'exclusive_owner_lock', side_effect=AssertionError('no UUID lock')), \
                    mock.patch.object(actual.pilot.telemetry, 'authorize', side_effect=AssertionError('no GPU')), \
                    mock.patch.object(run.planned, 'supervise', side_effect=AssertionError('no lifecycle')), \
                    mock.patch.object(run, 'validate_acceptance_process', wraps=run.validate_acceptance_process) as cells:
                checked = run.validate_public_cli_acceptance(binding, layout, copy.deepcopy(original))
                self.assertEqual(cells.call_count, 4)
                self.assertEqual(checked, self.report)


class IdentityAnchorTests(unittest.TestCase):
    """Tiny synthetic build graph; real issuance, MAIN-GO binding and freshness.

    Build semantics/layout approval are explicit CPU substitutes, not approvals
    for the real CLI. All anchor/GO files are temporary component fixtures.
    """
    @contextmanager
    def fixture(self, *, anchored=True):
        with tempfile.TemporaryDirectory(prefix='identity-anchor-component-') as temp:
            self.root = root = Path(temp)
            compiled = root / 'immutable-content'
            compiled.mkdir()
            self.leaf = compiled / 'architecture_target.rs'
            self.leaf.write_text('reviewed architecture target\n')
            self.leaf.chmod(0o400)
            control, runtime, descriptor = (root / n for n in ('workflow.py', 'runtime.json', 'descriptor.json'))
            for path in (control, runtime, descriptor):
                path.write_text('{}\n')
                path.chmod(0o400)
            compiled.chmod(0o555)
            pins = {str(p): ct.sha(p) for p in (self.leaf, runtime, descriptor)}
            controls = {str(control): ct.sha(control)}
            self.layout = layout = dict(purpose='public_cli_acceptance', artifact_pins=controls,
                http_expectation_contract='bound-http-reference-v1', http_reference_binding=None,
                input_contract={'cpu': 'fixture'}, candidate=dict(
                    schema='ferrule.storage-demand.http-transfer-candidate-binding.v2',
                    inputs=str(descriptor), inputs_sha256=ct.sha(descriptor), acceptance=None,
                    launch_review=None, launch_review_sha256=None, uuid_inventory=[]))
            base_facts = dict(source={'scope': 'CPU component, not a compiled candidate'},
                binary=str(self.leaf), binary_sha256=pins[str(self.leaf)], runtime_pin_manifest=str(runtime),
                initial_read_observation_available=True, initial_read_producer={})
            def authority(selected, *, approval_required=True, **kwargs):
                facts = dict(base_facts, artifact_pins=dict(pins))
                subject = (run.authorize_transfer_candidate(selected, facts) if approval_required
                    else run.approval_subject(selected, facts))
                if approval_required:
                    facts['artifact_pins'][selected['candidate']['launch_review']] = selected['candidate']['launch_review_sha256']
                bound = dict(candidate=dict(binary=facts['binary'], binary_sha256=facts['binary_sha256'],
                    runtime_pin_manifest=str(runtime), artifact_pins=facts['artifact_pins']),
                    build_review=facts, uuid_inventory=[], approval_subject=subject)
                graph = dict(bound_candidate=bound, required_pins=dict(facts['artifact_pins'], **controls),
                    required_trees={str(compiled): [str(self.leaf)]}, required_globs={}, acceptance_roots=[],
                    identity_anchor=None, always_hash={str(control), str(runtime), str(descriptor)})
                reference = selected['candidate'].get('verified_artifact_identity')
                if reference is not None:
                    graph['identity_anchor'] = run.load_verified_artifact_identity(selected, graph)
                    graph['required_pins'][reference['path']] = reference['sha256']
                    graph['always_hash'].add(reference['path'])
                if approval_required:
                    graph['always_hash'].add(selected['candidate']['launch_review'])
                return graph
            real_validator = run.validate_transfer_candidate
            def content_review(selected, **kwargs):
                if kwargs.get('freshness') is not None:
                    return real_validator(selected, **kwargs)
                self.assertIs(kwargs.get('approval_required'), False)
                # Only build semantics are substituted; actual leaf bytes are
                # independently checked again by the real issuance function.
                ct.check_pins(pins)
                return authority(selected, approval_required=False)['bound_candidate']
            from contextlib import ExitStack
            with ExitStack() as stack:
                for module, name, value in (
                    (run, 'transfer_candidate_authority', authority),
                    (run, 'validate_transfer_candidate', content_review),
                    (run, 'validate_transfer_layout', lambda path, **kwargs: ct.load(path)),
                    (run, 'validate_http_references', lambda *a: None),
                    (run, 'validate_transfer_inputs', lambda *a: None),
                    (ct, 'exclusive_owner_lock', mock.Mock(side_effect=AssertionError('no UUID lock'))),
                    (run.planned, 'supervise', mock.Mock(side_effect=AssertionError('no supervisor'))),
                    (actual.pilot.telemetry, 'authorize', mock.Mock(side_effect=AssertionError('no GPU'))),
                    (ex.subprocess, 'Popen', mock.Mock(side_effect=AssertionError('no server/process'))),
                ):
                    stack.enter_context(mock.patch.object(module, name, value))
                self.authority = authority
                if anchored:
                    result = run.issue_verified_artifact_identity(layout, root / 'verified-identities.json')
                    self.reference = result['verified_artifact_identity']
                    self.anchor_bytes = Path(self.reference['path']).read_bytes()
                    layout['candidate'].update(schema='ferrule.storage-demand.http-transfer-candidate-binding.v3',
                        verified_artifact_identity=self.reference)
                go = root / 'cpu-fixture-main-go.json'
                layout['candidate']['launch_review'] = str(go)
                subject = run.approval_subject(layout, dict(base_facts, artifact_pins=dict(pins)))
                ct.write(go, dict(schema='ferrule.storage-demand.http-transfer-launch-review.v2',
                    decision='GO', subject=subject, scope='temporary CPU fixture; not execution approval'))
                go.chmod(0o400)
                layout['candidate']['launch_review_sha256'] = ct.sha(go)
                try:
                    yield
                finally:
                    compiled.chmod(0o755)

    def mutate(self, *, same_bytes=False):
        before = self.leaf.stat()
        raw = self.leaf.read_bytes()
        self.leaf.chmod(0o600)
        self.leaf.write_bytes(raw if same_bytes else b'X' * len(raw))
        os.utime(self.leaf, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.leaf.chmod(0o400)

    def fresh_receipt(self):
        layout_path = self.root / 'layout.json'
        ct.write(layout_path, self.layout)
        layout_path.chmod(0o400)
        registration = dict(schema='ferrule.storage-demand.actual-http-transfer-registration.v1',
            study_layout=self.layout, binding_path=str(layout_path), execution_path=str(layout_path),
            binding_sha256=ct.sha(layout_path), execution_sha256=ct.sha(layout_path), boot_id=actual.boot_id())
        reg = self.root / 'registration.json'
        ct.write(reg, dict(registration, digest=ct.digest(registration)))
        graph = self.authority(self.layout)
        admission = {str(layout_path): ct.sha(layout_path), str(reg): ct.sha(reg)}
        now = ex.stamp()
        value = dict(schema='ferrule.storage-demand.http-static-preparation.v2',
            layout_path=str(layout_path), layout_sha256=ct.sha(layout_path), registration_sha256=ct.sha(reg),
            boot_id=actual.boot_id(), started=now, ended=now, deadline=now['monotonic']+180,
            throughput_denominator=False, layout=self.layout, workflow_pins=self.layout['artifact_pins'],
            bound_candidate=graph['bound_candidate'], approval_subject=graph['bound_candidate']['approval_subject'],
            admission_pins=admission, verified_inputs={name: dict(sha256=pin, identity=run.preparation_identity(name))
                for name, pin in dict(graph['required_pins'], **admission).items()},
            verified_trees=graph['required_trees'], verified_globs=graph['required_globs'],
            verified_directories={name: run.preparation_identity(name) for name in run.coverage_directories(graph)})
        # Even an arbitrary anchor-like value inside a newly self-hashed receipt
        # must not override the actual candidate/MAIN-GO-bound issuance record.
        value['unapproved_identity_anchor'] = copy.deepcopy(value['verified_inputs'])
        path = self.root / 'fresh-self-consistent-receipt.json'
        ct.write(path, value)
        path.chmod(0o400)
        return layout_path, dict(path=str(path), sha256=ct.sha(path))

    def test_fresh_correct_go_complete_receipt_cannot_hide_mutated_readonly_leaf(self):
        with self.fixture():
            self.mutate()
            layout, reference = self.fresh_receipt()
            original_sha = ct.sha
            with mock.patch.object(ct, 'sha', wraps=original_sha) as sha:
                with self.assertRaisesRegex(ValueError, 'artifact content hash differs.*architecture_target'):
                    run.validate_static_preparation(reference, layout)
                self.assertTrue(any(str(call.args[0]) == str(self.leaf) for call in sha.call_args_list))
            self.assertEqual(Path(self.reference['path']).read_bytes(), self.anchor_bytes)

    def test_missing_anchor_forces_actual_readonly_hash_and_rejects_wrong_bytes(self):
        with self.fixture(anchored=False):
            self.mutate()
            layout, reference = self.fresh_receipt()
            with self.assertRaisesRegex(ValueError, 'artifact content hash differs.*architecture_target'):
                run.validate_static_preparation(reference, layout)

    def test_unchanged_anchor_skips_leaf_but_control_files_are_always_hashed(self):
        with self.fixture():
            layout, reference = self.fresh_receipt()
            with mock.patch.object(ct, 'sha', wraps=ct.sha) as sha:
                checked = run.validate_static_preparation(reference, layout)
                names = [str(call.args[0]) for call in sha.call_args_list]
                self.assertNotIn(str(self.leaf), names)
                self.assertIn(str(self.root / 'workflow.py'), names)
            anchor = ct.load(self.reference['path'])
            self.assertNotIn(str(self.root), anchor['verified_directories'])
            self.assertIn(str(self.leaf.parent), anchor['verified_directories'])
            self.assertEqual(checked['bound_candidate']['approval_subject']['verified_artifact_identity_sha256'], self.reference['sha256'])

    def test_matching_byte_fallback_never_refreshes_issued_identity(self):
        with self.fixture():
            self.mutate(same_bytes=True)
            layout, reference = self.fresh_receipt()
            with mock.patch.object(ct, 'sha', wraps=ct.sha) as sha:
                run.validate_static_preparation(reference, layout)
                first = sum(str(c.args[0]) == str(self.leaf) for c in sha.call_args_list)
                run.validate_static_preparation(reference, layout)
                second = sum(str(c.args[0]) == str(self.leaf) for c in sha.call_args_list)
            self.assertGreater(first, 0)
            self.assertGreater(second, first)
            self.assertEqual(Path(self.reference['path']).read_bytes(), self.anchor_bytes)

    def test_unapproved_new_anchor_reference_cannot_borrow_old_main_go(self):
        with self.fixture():
            forged = ct.load(self.reference['path'])
            self.mutate()
            forged['verified_inputs'][str(self.leaf)]['identity'] = run.preparation_identity(self.leaf)
            path = self.root / 'new-self-attested-identities.json'
            ct.write(path, forged)
            path.chmod(0o400)
            self.layout['candidate']['verified_artifact_identity'] = dict(path=str(path), sha256=ct.sha(path))
            with mock.patch.object(run, 'load_verified_artifact_identity', wraps=run.load_verified_artifact_identity) as loader:
                with self.assertRaisesRegex(ValueError, 'purpose/layout/workflow/evidence-bound GO'):
                    self.authority(self.layout)
                loader.assert_not_called()

    def test_issuance_captures_pre_post_identity_around_successful_hash(self):
        with self.fixture(anchored=False):
            pending = copy.deepcopy(self.layout)
            pending['candidate'].update(launch_review=None, launch_review_sha256=None)
            output = self.root / 'must-not-issue-raced-identities.json'
            original_sha, leaf_reads = ct.sha, []
            def raced_sha(name):
                value = original_sha(name)
                if str(name) == str(self.leaf):
                    leaf_reads.append(name)
                    if len(leaf_reads) == 2:
                        self.mutate()  # Actual hash succeeded; the post-hash ID must fail.
                return value
            with mock.patch.object(ct, 'sha', side_effect=raced_sha), \
                    mock.patch.object(run, 'validate_transfer_candidate', wraps=run.validate_transfer_candidate) as reviewed:
                with self.assertRaisesRegex(ValueError, 'identity changed while verifying bytes'):
                    run.issue_verified_artifact_identity(pending, output)
                self.assertEqual(reviewed.call_count, 1)
                self.assertIs(reviewed.call_args.kwargs['approval_required'], False)
            self.assertFalse(output.exists())

    def test_issuance_refuses_output_inside_immutable_content_tree(self):
        with self.fixture(anchored=False):
            pending = copy.deepcopy(self.layout)
            pending['candidate'].update(launch_review=None, launch_review_sha256=None)
            output = self.leaf.parent / 'must-not-affect-content-directory.json'
            with self.assertRaisesRegex(ValueError, 'outside immutable content scopes'):
                run.issue_verified_artifact_identity(pending, output)
            self.assertFalse(output.exists())

    def test_missing_anchor_fallback_cannot_extend_existing_phase_timer(self):
        with self.fixture(anchored=False):
            layout, reference = self.fresh_receipt()
            original_sha, slow_reads = ct.sha, []
            def slow_leaf(name):
                if str(name) == str(self.leaf):
                    slow_reads.append(name)
                    time.sleep(.2)
                return original_sha(name)
            with mock.patch.object(ct, 'sha', side_effect=slow_leaf), ex.absolute_alarm(time.monotonic()+.05):
                with self.assertRaisesRegex(TimeoutError, 'absolute pilot phase deadline'):
                    run.validate_static_preparation(reference, layout)
            self.assertTrue(slow_reads)
