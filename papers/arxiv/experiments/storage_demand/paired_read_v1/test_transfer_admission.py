"""CPU-only role/admission regressions. Real sealed graph, no GPU/locks/launches.

Approval records in authority unit tests are in-memory test doubles, NOT a new
human review or a persisted MAIN-GO. Transaction tests use tiny control fixtures.
"""
from contextlib import ExitStack, contextmanager
import copy
import json
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
from runnerhelpers import transfer_admission as admission
from runnerhelpers.evidence_v2 import ct

ACCEPTANCE = HERE / 'artifacts-transfer-acceptance-37b2f40-20261017-01/paired-final.json'


class RealGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layout = run.prospective_transfer_plan(ACCEPTANCE, role_partition=True)
        cls.graph = admission.derive(run, cls.layout)

    @contextmanager
    def approved_component(self, *, mutate_proof=None, bad_go=False, legacy_go=False, graph=None):
        """Only approval is substituted; descriptors, graph, source and checks real."""
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            layout, graph = copy.deepcopy(self.layout), self.graph if graph is None else graph
            proof = dict(schema=admission.PROOF, layout_subject_sha256=admission.layout_subject(layout),
                candidate_inputs_sha256=layout['candidate']['inputs_sha256'], source=graph['source'],
                metadata_pins=graph['metadata_pins'], partition=copy.deepcopy(graph['partition']),
                identity_anchor=dict(schema='ferrule.storage-demand.reviewed-role-identity.v1',
                    subject=admission.identity_subject(layout, graph)))
            if mutate_proof:
                mutate_proof(proof)
            path = Path(temp) / 'component-proof.json'
            ct.write(path, proof); path.chmod(0o400)
            layout['candidate']['reviewed_provenance'] = dict(path=str(path), sha256=ct.sha(path))
            go_path = str(Path(temp) / 'in-memory-approval-only')
            layout['candidate'].update(launch_review=go_path, launch_review_sha256='a' * 64)
            layout['candidate']['acceptance']['workflow_compatibility'] = dict(path='/in-memory-human-review-only', sha256='b' * 64)
            go = dict(schema='ferrule.storage-demand.http-transfer-launch-review.v2' if legacy_go else
                'ferrule.storage-demand.http-transfer-launch-review.v3', decision='GO')
            real_load, real_sha, real_immutable = ct.load, ct.sha, run.immutable_input
            def load(name):
                if str(name) == go_path:
                    return go
                return real_load(name)
            def sha(name):
                return 'a' * 64 if str(name) == go_path else real_sha(name)
            def immutable(name):
                return Path(name) if str(name) == go_path else real_immutable(name)
            original_authorize = run.authorize_transfer_candidate
            def authorize(selected, facts):
                go['subject'] = run.approval_subject(selected, facts)
                if bad_go:
                    go['subject']['role_partition_sha256'] = '0' * 64
                return original_authorize(selected, facts)
            stack.enter_context(mock.patch.object(ct, 'load', side_effect=load))
            stack.enter_context(mock.patch.object(ct, 'sha', side_effect=sha))
            stack.enter_context(mock.patch.object(run, 'immutable_input', side_effect=immutable))
            stack.enter_context(mock.patch.object(run, 'authorize_transfer_candidate', side_effect=authorize))
            stack.enter_context(mock.patch.object(run, 'validate_workflow_compatibility', return_value={'unit_test_human_review_substitute': True}))
            for name in ('review_transfer_build', 'review_transfer_provenance', 'compiled_source_review',
                         'transfer_candidate_authority', 'validate_public_cli_acceptance', 'validate_acceptance_evidence'):
                stack.enter_context(mock.patch.object(run, name, side_effect=AssertionError('full review forbidden at launch')))
            stack.enter_context(mock.patch.object(ct, 'exclusive_owner_lock', side_effect=AssertionError('no locks')))
            stack.enter_context(mock.patch.object(run.planned, 'supervise', side_effect=AssertionError('no supervisor')))
            yield layout

    def test_real_universe_is_disjoint_exhaustive_and_runtime_wins_overlap(self):
        entries = self.graph['partition']['entries']
        sets = [{n for n, row in entries.items() if row['role'] == role}
            for role in (admission.LIVE, admission.BUILD, admission.QUALITY)]
        self.assertGreater(len(entries), 19000)
        self.assertEqual(set().union(*sets), set(entries))
        self.assertEqual(sum(map(len, sets)), len(entries))
        self.assertEqual(len(self.graph['acceptance']['artifact_pins']), 589)
        self.assertEqual(len(self.graph['documents']['runtime_pin_manifest']['pins']), 12)
        for path in self.graph['documents']['runtime_pin_manifest']['pins']:
            self.assertEqual(entries[path]['role'], admission.LIVE)
        for relative in ('Cargo.toml', *run.PROOF_PATHS):
            path = str(Path(self.graph['documents']['source_freeze']['frozen_source']) / relative)
            self.assertEqual(entries[path]['role'], admission.LIVE)
        self.assertTrue(all(row['rules'] for row in entries.values()))

    def test_correct_candidate_cannot_select_incomplete_or_wrong_role_coverage(self):
        path = next(iter(self.graph['live_pins']))
        mutations = [lambda p: p['partition']['entries'].pop(path),
            lambda p: p['partition']['entries'][path].update(role=admission.BUILD),
            lambda p: p['partition']['entries'][path].update(sha256='0' * 64),
            lambda p: p['partition']['entries'].update({'/foreign/path': p['partition']['entries'].pop(path)})]
        for mutate in mutations:
            with self.subTest(mutate=mutate), self.approved_component(mutate_proof=mutate) as layout:
                with self.assertRaisesRegex(ValueError, 'independent complete role'):
                    admission.authority(run, layout)

    def test_real_launch_has_no_full_review_or_original_build_leaf_reads(self):
        build = {n for n, x in self.graph['partition']['entries'].items() if x['role'] == admission.BUILD}
        old_stat = Path.stat
        attempted = []
        def sha(path):
            if str(path) in build:
                attempted.append(str(path))
                raise AssertionError('original build-only content must not be read')
            return old_sha(path)
        def stat(path, *a, **kw):
            if str(path) in build:
                attempted.append(str(path))
                raise AssertionError('original build-only identity must not be inspected')
            return old_stat(path, *a, **kw)
        with self.approved_component() as layout:
            old_sha = ct.sha
            with mock.patch.object(ct, 'sha', side_effect=sha), mock.patch.object(Path, 'stat', new=stat):
                graph, bound = admission.authority(run, layout)
                admission.verify_live(run, layout, graph)
                self.assertEqual(bound['candidate']['binary_sha256'], self.layout['compiled_identity']['binary_sha256'])
        self.assertEqual(attempted, [])

    def test_build_header_bytes_may_change_after_review_without_target_change(self):
        # Isolated external-manifest component, retaining the full real graph.
        # Never modify the sealed original headers/evidence or claim a real GO.
        with tempfile.TemporaryDirectory() as temp:
            header = Path(temp) / 'reviewed-build-only.h'
            header.write_text('original compiled header fixture')
            external = copy.deepcopy(self.graph['documents']['external_inputs'])
            external['pins'][str(header)] = ct.sha(header)
            manifest = Path(temp) / 'external-input-manifest.json'
            ct.write(manifest, external); manifest.chmod(0o400)
            identity, descriptor = run.transfer_candidate_descriptor(self.layout)
            descriptor = copy.deepcopy(descriptor)
            descriptor['inputs']['external_inputs'] = str(manifest)
            descriptor['sha256']['external_inputs'] = ct.sha(manifest)
            with mock.patch.object(run, 'transfer_candidate_descriptor', return_value=(identity,descriptor)):
                graph = admission.derive(run, self.layout)
                self.assertEqual(graph['partition']['entries'][str(header)]['role'], admission.BUILD)
                original_pin = graph['partition']['entries'][str(header)]['sha256']
                with self.approved_component(graph=graph) as layout:
                    header.write_text('changed AFTER the review fixture; binary is unchanged')
                    self.assertNotEqual(ct.sha(header), original_pin)
                    checked, bound = admission.authority(run, layout)
                    admission.verify_live(run, layout, checked)
                    self.assertEqual(bound['candidate']['binary_sha256'], self.layout['compiled_identity']['binary_sha256'])

    def test_mutated_live_readonly_inputs_reject_even_with_same_candidate_ids(self):
        docs = self.graph['documents']
        paths = [self.graph['descriptor']['inputs']['binary'],
            next(iter(docs['runtime_pin_manifest']['pins'])), str(Path(sys.executable).resolve()),
            next(iter(self.layout['input_contract']['reference_pins'])),
            next(p for p in self.layout['input_contract']['model_metadata'] if p.endswith('tokenizer.json')),
            next(iter(self.graph['semantic_pins']))]
        for path in paths:
            real = ct.sha
            with self.subTest(path=path), mock.patch.object(ct, 'sha', side_effect=lambda n: '0' * 64 if str(n) == path else real(n)):
                with self.assertRaisesRegex(ValueError, 'live input content'):
                    admission.verify_live(run, self.layout, self.graph)

    def test_go_subject_and_version_are_independent_requirements(self):
        for kwargs in ({'bad_go': True}, {'legacy_go': True}):
            with self.subTest(kwargs=kwargs), self.approved_component(**kwargs) as layout:
                with self.assertRaisesRegex(ValueError, 'separate main'):
                    admission.authority(run, layout)

    def test_missing_required_workflow_path_cannot_reclassify_it(self):
        layout = copy.deepcopy(self.layout)
        layout['artifact_pins'].pop(str(Path(sys.executable).resolve()))
        with self.assertRaisesRegex(ValueError, 'required live workflow coverage'):
            admission.derive(run, layout)

    def test_new_compatibility_plan_explicitly_binds_helper_and_test_but_is_pending(self):
        plan = run.workflow_compatibility_plan(self.layout, self.layout['candidate']['acceptance'])
        self.assertEqual(plan['schema'], 'ferrule.storage-demand.http-workflow-compatibility.v2')
        self.assertIsNone(plan['review'])
        self.assertFalse(plan['gpu_authorized'])
        pins = plan['subject']['required_cpu_test_pins']
        self.assertTrue(all(str(ct.ROOT / n) in pins for n in run.ROLE_COMPATIBILITY_ADDED))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'pending-human-review.json'
            ct.write(path, plan); path.chmod(0o400)
            layout = copy.deepcopy(self.layout)
            binding = layout['candidate']['acceptance']
            binding['workflow_compatibility'] = dict(path=str(path), sha256=ct.sha(path))
            with self.assertRaisesRegex(ValueError, 'human compatibility review pending'):
                run.validate_workflow_compatibility(binding, layout)

    def test_runtime_is_live_at_spawn_and_maps_not_rehashed_in_request_control(self):
        runtime = set(self.graph['documents']['runtime_pin_manifest']['pins'])
        binary = self.graph['descriptor']['inputs']['binary']
        self.assertTrue(runtime <= self.graph['live_pins'].keys())
        self.assertNotIn(binary, self.graph['request_pins'])
        self.assertFalse(runtime & self.graph['request_pins'].keys())
        self.assertEqual(set(self.graph['request_pins']), set(self.graph['live_pins']) - runtime - {binary})
        for name in self.graph['metadata_pins']:
            self.assertEqual(self.graph['partition']['entries'][name]['role'], admission.LIVE)

    def test_unknown_descriptor_category_fails_instead_of_being_omitted(self):
        real_load = ct.load
        descriptor_path = self.layout['candidate']['inputs']
        def load(path):
            value = real_load(path)
            if str(path) == descriptor_path:
                value['inputs']['unclassified_runtime_plugin'] = '/unknown/plugin'
                value['sha256']['unclassified_runtime_plugin'] = '0'*64
            return value
        with mock.patch.object(ct, 'load', side_effect=load):
            with self.assertRaisesRegex(ValueError, 'descriptor manifest coverage'):
                admission.derive(run, self.layout)

    def test_legacy_candidate_receipt_forgery_still_uses_strict_validator(self):
        accepted = self.graph['acceptance']['study_layout']
        with mock.patch.object(admission, 'authority', side_effect=AssertionError('legacy must not enter role path')):
            with self.assertRaises(ValueError):
                run.validate_transfer_candidate(accepted, freshness={'admission_pins': {}, 'verified_inputs': {}}, admission_pins={})


class TransactionTests(unittest.TestCase):
    @contextmanager
    def transaction(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            proof_path = directory / 'proof.json'
            ct.write(proof_path, {'component_fixture_not_production_review': True}); proof_path.chmod(0o400)
            reference = dict(path=str(proof_path), sha256=ct.sha(proof_path))
            layout = dict(purpose='pilot', candidate=dict(schema=admission.CANDIDATE, reviewed_provenance=reference,
                launch_review=str(proof_path), launch_review_sha256=reference['sha256'],
                acceptance={'workflow_compatibility': reference}), artifact_pins={}, cases={'blocking-synchronous': {}},
                input_contract={}, http_reference_binding=None, upload_producer_contract={}, http_expectation_contract='fixture')
            path = directory / 'layout.json'; ct.write(path, layout); path.chmod(0o400)
            registration = directory / 'registration.json'; ct.write(registration, {'component_control': True})
            registered = dict(pilot_deadline=time.monotonic()+7200)
            live = directory / 'live'; live.write_text('actual fixture live bytes'); live.chmod(0o400)
            graph = dict(live_pins={str(live): ct.sha(live)}, descriptor={'inputs': {'binary': str(live)}})
            bound = dict(role_partition_sha256='e'*64, candidate={}, build_review={})
            value = admission.Admission(admission.TOKEN, run, path, layout, graph, bound, directory,
                'B1-blocking-synchronous', registered, time.monotonic()+180, actual.ex.stamp())
            try:
                yield value, live
            finally:
                value.close()

    def test_adapter_callbacks_are_data_access_not_review_or_fresh_admission(self):
        with self.transaction() as (value, live), ExitStack() as stack:
            for name in ('validate_transfer_layout', 'validate_static_preparation', 'validate_transfer_candidate',
                         'review_transfer_build', 'review_transfer_provenance'):
                stack.enter_context(mock.patch.object(run, name, side_effect=AssertionError('recursive validation forbidden')))
            with run.transfer_adapter(value.path, admission=value):
                for unused in range(4):
                    self.assertEqual(actual.binding.validate_binding(value.path), value.bound)
                    self.assertEqual(actual.binding.validate_execution(value.path), value.layout)
                with actual.actual_adapter():
                    self.assertEqual(actual.binding.validate_binding(value.path), value.bound)
            self.assertFalse(value.closed)

    def test_postyield_stop_write_preserves_primary_and_closes_admission(self):
        for stop_fails in (False, True):
            with self.subTest(stop_fails=stop_fails), self.transaction() as (value, live), ExitStack() as stack:
                primary = RuntimeError('primary post-yield failure')
                original_write = ct.write
                def write(path, data):
                    if stop_fails and Path(path).name == 'study-stop.json':
                        raise OSError('forced STOP receipt write failure')
                    return original_write(path, data)
                stack.enter_context(mock.patch.object(admission, 'admit_transfer_launch', return_value=value))
                stack.enter_context(mock.patch.object(ct, 'write', side_effect=write))
                stop = stack.enter_context(mock.patch.object(actual, 'stop', wraps=actual.stop))
                refused = stack.enter_context(mock.patch.object(actual, 'admission_refusal',
                    side_effect=AssertionError('post-yield may not claim never spawned')))
                locks = stack.enter_context(mock.patch.object(ct, 'exclusive_owner_lock', side_effect=AssertionError('no locks')))
                supervise = stack.enter_context(mock.patch.object(run.planned, 'supervise', side_effect=AssertionError('no supervisor')))
                process = stack.enter_context(mock.patch.object(actual.ex.subprocess, 'Popen', side_effect=AssertionError('no launch')))
                with self.assertRaises(RuntimeError) as raised:
                    with admission.prepared_admission(run, value.directory, value.path, value.registered,
                            value.trial, value.started, value.deadline):
                        self.assertIs(actual.binding.role_admission, value)
                        self.assertFalse(value.closed)
                        raise primary
                self.assertIs(raised.exception, primary)
                self.assertTrue(value.closed)
                self.assertEqual(getattr(primary, '__notes__', []),
                    ['STOP journal failure: OSError: forced STOP receipt write failure'] if stop_fails else [])
                self.assertEqual((value.directory / 'study-stop.json').exists(), not stop_fails)
                stop.assert_called_once_with(value.directory, 'RuntimeError: primary post-yield failure')
                refused.assert_not_called()
                locks.assert_not_called()
                supervise.assert_not_called()
                process.assert_not_called()

    def test_completion_stop_expiry_identity_and_scope_reset_refuse(self):
        for kind in ('completion', 'stop', 'expiry', 'process', 'trial-reset', 'deadline-reset', 'live-set-reset', 'proof-reset'):
            with self.subTest(kind=kind), self.transaction() as (value, live), ExitStack() as stack:
                if kind == 'completion':
                    (value.directory / value.trial).mkdir(); (value.directory / value.trial / 'assessment.json').touch()
                elif kind == 'stop':
                    (value.directory / 'study-stop.json').touch()
                elif kind == 'expiry':
                    stack.enter_context(mock.patch.object(time, 'monotonic', return_value=value.deadline+1))
                elif kind == 'process':
                    stack.enter_context(mock.patch.object(actual.pilot.launch_guard, 'identity', return_value=dict(pid=os.getpid(), birth_ticks=-1)))
                elif kind == 'trial-reset':
                    value.trial = 'B2-blocking-synchronous'
                elif kind == 'live-set-reset':
                    value.graph['live_pins'] = {}
                elif kind == 'proof-reset':
                    value.proof['sha256'] = '0'*64
                else:
                    value.deadline += 650
                with self.assertRaises(ValueError): value.check()

    def test_registration_seal_is_explicit_one_way_and_content_remains_normative(self):
        with self.transaction() as (value, live):
            path = value.directory / 'registration.json'
            path.chmod(0o400); value.check()
            self.assertEqual(value.seal_transition, dict(before=0o644, after=0o400, content_sha256=value.registration_pin))
            path.chmod(0o644)
            with self.assertRaisesRegex(ValueError, 'one-way'): value.check()
        with self.transaction() as (value, live):
            path = value.directory / 'registration.json'
            path.write_text('{"component_control":false}'); path.chmod(0o400)
            with self.assertRaisesRegex(ValueError, 'registration content'): value.check()

    def test_original_clock_activation_is_once_and_not_a_sliding_ttl(self):
        with self.transaction() as (value, live):
            started = actual.ex.stamp(); outer = min(started['monotonic']+650,value.registered['pilot_deadline']-100)
            value.activate(started, outer); value.check()
            self.assertEqual(value.deadline, outer)
            with self.assertRaisesRegex(ValueError, 'original active clock'): value.activate(started, outer)

    def test_post_admission_live_mutation_is_rejected_before_spawn(self):
        with self.transaction() as (value, live), mock.patch.object(run, 'validate_transfer_inputs'), \
                mock.patch.object(run, 'validate_http_references'):
            live.chmod(0o600); live.write_text('changed'); live.chmod(0o400)
            with self.assertRaisesRegex(ValueError, 'live input content'): value.before_spawn()


class MaintainedRolePipelineTests(unittest.TestCase):
    @contextmanager
    def fixture(self, mutation=None):
        from paired_read_v1.test_deadline_phase import DeadlinePhaseTests
        components = DeadlinePhaseTests()
        with components.fixture(static_delay=0, mutation=mutation), ExitStack() as stack:
            directory, path = components.directory, components.path
            layout = components.layout
            control = directory / 'role-proof-component.json'
            ct.write(control, dict(component_only_not_approval=True)); control.chmod(0o400)
            reference = dict(path=str(control), sha256=ct.sha(control))
            layout['candidate'].update(schema=admission.CANDIDATE, reviewed_provenance=reference,
                launch_review=str(control), launch_review_sha256=reference['sha256'],
                acceptance={'workflow_compatibility': reference})
            path.chmod(0o600); path.write_text(json.dumps(layout)); path.chmod(0o400)
            registered = ct.load(directory / 'registration.json')
            registered.pop('digest')
            registered.update(study_layout=layout, binding_sha256=ct.sha(path), execution_sha256=ct.sha(path))
            (directory / 'registration.json').unlink()
            ct.write(directory / 'registration.json', dict(registered, digest=ct.digest(registered)))
            components.evidence.chmod(0o500); components.evidence.parent.chmod(0o555)
            pins = {str(components.mutable): ct.sha(components.mutable), str(components.evidence): ct.sha(components.evidence)}
            bound = dict(candidate=dict(binary=str(components.evidence), binary_sha256=ct.sha(components.evidence),
                runtime_pin_manifest=str(control), artifact_pins=pins), uuid_inventory=registered['eight_uuids'],
                build_review={'initial_read_producer': {}}, role_partition_sha256='e'*64)
            graph = dict(live_pins=pins, descriptor={'inputs': {'binary': str(components.evidence)}})
            @contextmanager
            def lock(uuid):
                components.events.append('role-lock-component')
                yield dict(uuid=uuid)
            original_admit, original_before = admission.admit_transfer_launch, admission.Admission.before_spawn
            values, fresh = [], []
            def admit(api, selected, directory, trial, **kwargs):
                value = original_admit(api, selected, directory, trial, **kwargs)
                values.append(value)
                return value
            def before(value):
                fresh.append(value.trial)
                return original_before(value)
            for obj, name, value in ((admission, 'authority', lambda *a: (graph, bound)),
                    (admission, 'admit_transfer_launch', admit), (admission.Admission, 'before_spawn', before),
                    (ct, 'exclusive_owner_lock', lock)):
                stack.enter_context(mock.patch.object(obj, name, value))
            yield components, values, fresh

    def test_existing_coordinator_activates_once_and_callbacks_do_not_readmit(self):
        with self.fixture() as (components, values, fresh):
            actual.execute_next(components.directory,
                admission=lambda d: run.prepare_transfer_admission(d, components.path))
            self.assertEqual(len(values), 1)
            self.assertTrue(values[0].closed)
            self.assertEqual(fresh, [components.trial])
            self.assertEqual(len(components.payloads), 1)
            payload = components.payloads[0]
            attempt = ct.load(components.directory / components.trial / 'attempt.json')
            self.assertEqual(payload['spec']['role_admission'], attempt['role_admission'])
            self.assertEqual(attempt['outer_deadline'] - attempt['admission']['monotonic'], 650)
            self.assertLessEqual(attempt['static_preparation_seconds'], 180)
            self.assertEqual(components.events.count('role-lock-component'), 16)  # eight acquisition + eight cleanup reprobe

    def test_preidle_mutation_refuses_before_maintained_supervisor(self):
        with self.fixture(mutation='before-launch') as (components, values, fresh):
            actual.execute_next(components.directory,
                admission=lambda d: run.prepare_transfer_admission(d, components.path))
            self.assertEqual(components.payloads, [])
            self.assertTrue((components.directory / 'study-stop.json').exists())
            self.assertTrue(values[0].closed)

    def test_larger_existing_timer_is_not_a_new_phase_budget(self):
        with actual.ex.absolute_alarm(time.monotonic()+10):
            with self.assertRaisesRegex(ValueError, 'no timer reset'):
                admission.finite_phase(time.monotonic()+1)


if __name__ == '__main__':
    unittest.main()
