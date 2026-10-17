"""CPU-only actual-build review and fail-closed transfer lifecycle regressions."""
from contextlib import ExitStack, nullcontext
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paired_read_v1 import run_v3 as run, run_actual_v2 as actual, quality
from runnerhelpers import execution_v2 as ex
from runnerhelpers.evidence_v2 import ct

UUIDS = [f'GPU-00000000-0000-0000-0000-{i:012d}' for i in range(8)]


def initial_producer_binding(source_sha256='a' * 64):
    """Parser component fixture, never a compiled-candidate approval."""
    return dict(contract=quality.INITIAL_ENDPOINT_PRODUCER, source=quality.INITIAL_ENDPOINT_SOURCE,
        source_sha256=source_sha256)


def initial_event(options, policy='synchronous', pid=12345):
    from test_quality_v1 import observation
    debug = {'synchronous': 'Synchronous', 'grouped-synchronous': 'GroupedSynchronous'}[policy]
    return dict(level='INFO', target='ferrule_performance', fields=dict(
        **{k: v for k, v in quality.INITIAL_ENDPOINT_PRODUCER.items() if k not in ('target', 'level')},
        process_id=pid, requested_expert_transfer_policy='Some(' + debug + ')', expert_transfer_policy=debug,
        expert_read_pipeline_initial=json.dumps(dict(state='observed', value=observation(options, configured=True))),
        owner_binding=json.dumps({k: dict(state='observed', value=v) for k, v in dict(
            scope=dict(owner=17, generation=3), prepared_plan_id=23, image_generation=29, device_ordinal=0).items()})))


def cpu_capture_child(argv):
    """Real ENTRY/ancestry/registration/worker-contract test, NOT GPU admission.

    Only candidate/layout approval, model I/O, GPU-lock validation and the target
    are component substitutes. No GO file, GPU lock or GPU target is created.
    """
    import _profile_capture as wrapper
    import time
    directory = Path(argv[1])
    payload = ct.load(directory / 'worker-launch.json')
    layout = ct.load(payload['execution_path'])
    bound = payload['cpu_bound_component']
    mutation = payload.get('cpu_mutation')
    real_clock, clock = time.monotonic, [None]
    real_wrapper, real_argv, real_pin = wrapper.run_target, ex.process_argv, ct.pinned_executable
    locks_seen, target_calls = [], []
    def inputs(contract):
        if mutation == 'deadline_after_inputs':
            # The second input check is after entry/registration validation.
            if inputs.calls >= 2:
                clock[0] = float(argv[3]) + 1
            inputs.calls += 1
    inputs.calls = 0
    def candidate(*args, **kwargs):
        if mutation == 'no_go':
            raise ValueError('CPU component: no purpose-scoped main GO')
        return copy.deepcopy(bound)
    def target(directory, command, lifetime, shutdown, **kwargs):
        target_calls.append(dict(command=command, lifetime=lifetime, absolute_deadline=kwargs['absolute_deadline']))
        if kwargs['absolute_deadline'] != float(argv[3]) or lifetime > kwargs['absolute_deadline'] - real_clock() + .1:
            raise AssertionError('entry refreshed the original target deadline')
        cpu_code = 'from pathlib import Path; import time; Path(' + repr(str(directory / 'cpu-target-ready')) + ').touch(); time.sleep(20)'
        def cpu_observer_ack(child, event, deadline, active):
            active()
            ct.write(directory / 'cpu-only-observer-ack.json', dict(scope='CPU component; no GPU observer/HTTP quality claim', event=event))
            return dict(stop_event=event, observer_stopped=True)
        return real_wrapper(directory, [sys.executable, '-S', '-B', '-c', cpu_code], min(lifetime, 5), min(shutdown, 2),
            on_stop=cpu_observer_ack, absolute_deadline=kwargs['absolute_deadline'])
    from contextlib import contextmanager
    @contextmanager
    def verified_elf(path, expected):
        with real_pin(path, expected) as fd:
            if mutation == 'deadline_after_elf':
                clock[0] = float(argv[3]) + 1
            yield fd
    def argv_component(pid):
        if pid == os.getppid():
            return [sys.executable, '-S', '-B', str(run.ENTRY), '--_worker', argv[2], '--layout', payload['execution_path']]
        return real_argv(pid)
    result = dict(scope='CPU component test only; not approval or public quality', target_calls=target_calls,
        component_substitutions=['layout/candidate main-GO', 'model input reads', 'GPU-lock validator',
            'CPU bootstrap worker argv', 'CPU target command and CPU-only stop observer callback'])
    try:
        with mock.patch.object(run, 'validate_transfer_layout', return_value=layout), \
                mock.patch.object(run, 'validate_transfer_candidate', side_effect=candidate), \
                mock.patch.object(run, 'validate_transfer_inputs', side_effect=inputs), \
                mock.patch.object(ct, 'validate_owner_lock', side_effect=lambda lock, uuid, pid: locks_seen.append((uuid, pid))), \
                mock.patch.object(actual.pilot.telemetry, 'authorize'), \
                mock.patch.object(ct, 'pinned_executable', side_effect=verified_elf), mock.patch.object(ex, 'process_argv', side_effect=argv_component), mock.patch.object(wrapper, 'run_target', side_effect=target), \
                mock.patch.object(time, 'monotonic', side_effect=lambda: real_clock() if clock[0] is None else clock[0]):
            result['returncode'] = run.main(argv)
            result['authorized_entry'] = True
    except (Exception, KeyboardInterrupt) as error:
        result.update(authorized_entry=False, error=type(error).__name__ + ': ' + str(error))
    result['lock_checks'] = len(locks_seen)
    ct.write(directory / 'cpu-entry-result.json', result)
    return 0


def cpu_capture_worker(launch_hash):
    """Actual maintained-supervisor root; authenticate before spawning capture."""
    import launch_guard
    import subprocess
    import time
    payload = json.load(sys.stdin)
    authenticated = launch_guard.authenticate(payload, launch_hash)
    directory = Path(payload['directory'])
    ct.write(directory / 'cpu-worker-authenticated.json', authenticated)
    target_started = time.monotonic()
    limits = payload['limits']
    target_deadline = min(target_started + limits['target_lifetime_seconds'],
        payload['outer_deadline'] - limits['shutdown_seconds'] - limits['worker_finalization_seconds'],
        payload['pilot_deadline'] - limits['shutdown_seconds'] - limits['worker_finalization_seconds'])
    original = subprocess.Popen
    bootstrap = 'import sys; sys.path.insert(0, ' + repr(str(HERE)) + '); from test_http_transfer_controls import cpu_capture_child; sys.exit(cpu_capture_child(sys.argv[1:]))'
    def capture(argv, **kwargs):
        ct.write(directory / 'cpu-capture-argv.json', argv)
        return original([sys.executable, '-S', '-B', '-c', bootstrap, *argv[4:]], **kwargs)
    with mock.patch.object(ex.subprocess, 'Popen', side_effect=capture), mock.patch.object(ex.runner, 'require_free_port'):
        child = ex.launch_http_capture(directory, payload, target_deadline, target_started=target_started)
    deadline = time.monotonic() + 10
    while child.poll() is None and time.monotonic() < deadline:
        if (directory / 'cpu-target-ready').exists() and not (directory / 'profile-stop').exists():
            ex.stop_capture(directory)
        time.sleep(.01)
    return child.wait(timeout=max(.1, deadline-time.monotonic()))


class ActualBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.facts = run.review_transfer_build()

    def test_actual_immutable_elf_full_source_dependencies_and_runtime(self):
        self.assertEqual(self.facts['binary_sha256'], run.CLI_SHA256)
        self.assertTrue(self.facts['build_verified'])
        self.assertFalse(self.facts['quality_accepted'])
        self.assertGreater(self.facts['source']['compiled_files'], len(run.PROOF_PATHS))
        self.assertEqual(self.facts['source']['archive_commit'], run.ARCHIVE_COMMIT)
        self.assertEqual(self.facts['source']['production_commit'], run.TRANSFER_COMMIT)
        self.assertFalse(self.facts['source']['freeze_relabelled'])
        self.assertTrue(self.facts['runtime_policy']['required'])
        ct.check_pins(self.facts['artifact_pins'])

    def test_dummy_and_legacy_candidate_booleans_never_qualify(self):
        layout = run.transfer_plan()
        for value in (None, dict(binary='/tmp/fixture-cli', binary_sha256='a' * 64,
                main_source_reviewed=True, build_custody_reviewed=True, same_cut_cli_quality_accepted=True)):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'candidate pending'):
                run.validate_transfer_candidate(dict(layout, candidate=value))

    def test_full_compiled_scope_mutation_rejected_outside_dispatch_list(self):
        original = ct.load(run.CANDIDATE_ROOT / 'source-freeze.json')
        mutated = copy.deepcopy(original)
        name = next(n for n in mutated['pins_relative_to_frozen_source']
            if n.startswith('crates/') and n not in run.PROOF_PATHS)
        mutated['pins_relative_to_frozen_source'][name] = '0' * 64
        with self.assertRaises(ValueError):
            run.compiled_source_review(mutated)

    def test_approval_subject_binds_layout_workflow_evidence_and_purpose(self):
        layout = run.transfer_plan('public_cli_acceptance')
        layout['candidate'] = dict(schema='ferrule.storage-demand.http-transfer-candidate-binding.v2',
            inputs=str(run.CANDIDATE_INPUTS), inputs_sha256=ct.sha(run.CANDIDATE_INPUTS),
            launch_review=None, launch_review_sha256=None, uuid_inventory=UUIDS, acceptance=None)
        facts = copy.deepcopy(self.facts)
        facts['artifact_pins'].update(layout['input_contract']['reference_pins'])
        facts['artifact_pins'].update(layout['input_contract']['model_metadata'])
        facts['artifact_pins'][layout['input_contract']['checkpoint_manifest']] = layout['input_contract']['checkpoint_manifest_sha256']
        subject = run.approval_subject(layout, facts)
        for key in ('purpose', 'artifact_pins', 'input_contract'):
            changed = copy.deepcopy(layout)
            changed[key] = 'mutated'
            with self.subTest(key=key):
                self.assertNotEqual(subject, run.approval_subject(changed, facts))
        with mock.patch.object(run, 'review_transfer_build', return_value=copy.deepcopy(self.facts)):
            review = run.validate_transfer_candidate(layout, approval_required=False)
            self.assertEqual(subject, review['approval_subject'])
            with self.assertRaises((ValueError, TypeError)):
                run.validate_transfer_candidate(layout)
        self.assertFalse((run.CANDIDATE_ROOT / 'launch-review.json').exists())

    def test_real_frozen_cli_initial_endpoint_blocks_before_review_or_gpu(self):
        layout = run.transfer_plan('public_cli_acceptance')
        layout['candidate'] = dict(schema='ferrule.storage-demand.http-transfer-candidate-binding.v2',
            inputs=str(run.CANDIDATE_INPUTS), inputs_sha256=ct.sha(run.CANDIDATE_INPUTS),
            launch_review='/not-issued/GO.json', launch_review_sha256='0'*64, uuid_inventory=UUIDS, acceptance=None)
        self.assertFalse(self.facts['initial_read_observation_available'])
        with mock.patch.object(run, 'review_transfer_build', return_value=copy.deepcopy(self.facts)), \
                mock.patch.object(actual.pilot.telemetry, 'authorize', side_effect=AssertionError('no GPU')) as authorize:
            with self.assertRaisesRegex(ValueError, 'no observed initial read endpoint'):
                run.validate_transfer_candidate(layout)
            authorize.assert_not_called()

    def test_pilot_requires_actual_acceptance_not_review_boolean(self):
        layout = run.transfer_plan()
        for acceptance in (None, True, dict(same_cut_cli_quality_accepted=True)):
            with self.subTest(acceptance=acceptance), self.assertRaisesRegex(ValueError, 'actual four-cell'):
                run.validate_public_cli_acceptance(acceptance, layout, copy.deepcopy(self.facts))

    def test_build_custody_or_features_mutation_rejected(self):
        original_load = ct.load
        for name, mutate in (
            ('build-cli-supervisor.json', lambda v: v['custody']['entries'][0]['signals'].append(15)),
            ('artifact-cli.json', lambda v: v['compiled_artifact']['features'].append('cpu-reference')),
            ('compiled-dependency-pins.json', lambda v: v['packages'].pop(next(iter(v['packages'])))),
            ('runtime-library-pins.json', lambda v: v['per_role'].clear()),
            ('native-cubin-pins.json', lambda v: v.update(compiled_target='sm_80')),
            ('control-helper-pins.json', lambda v: v['pins'].clear()),
            ('executed-helper-copy-pins.json', lambda v: v['pins'].pop(next(iter(v['pins'])))),
        ):
            def load(path, name=name, mutate=mutate):
                value = original_load(path)
                if Path(path).name == name:
                    value = copy.deepcopy(value)
                    mutate(value)
                return value
            with self.subTest(name=name), mock.patch.object(ct, 'load', side_effect=load):
                with self.assertRaises((ValueError, KeyError)):
                    run.review_transfer_build()


class InitialEndpointTests(unittest.TestCase):
    def parse(self, event, options=None, policy='synchronous', lines=None, binding=None, ready=None):
        from test_quality_v1 import options as read_options
        options = options or read_options()
        lines = lines or [json.dumps(event), json.dumps(dict(fields=dict(message='Ferrule OpenAI server listening')))]
        return ex.read_initial_endpoint(lines, options, quality.TRANSFER_UPLOAD_PRODUCER,
            server_identity=dict(pid=12345, birth_ticks=8, ppid=12344), producer_binding=binding or initial_producer_binding(),
            policy=policy, ready_lines=lines if ready is None else ready)

    def test_typed_configured_worker_zero_all_four_policy_schedule_cells(self):
        from test_quality_v1 import options
        for mode in ('blocking', 'lookahead'):
            for policy in ('synchronous', 'grouped-synchronous'):
                with self.subTest(mode=mode, policy=policy):
                    event = initial_event(options(mode), policy)
                    result = self.parse(event, options(mode), policy)
                    read = result['read_endpoint']
                    self.assertEqual(read['status'], 'configured')
                    self.assertFalse(read['worker_alive'])
                    self.assertEqual(read['stats']['workers_created'], 0)
                    self.assertEqual(read['current_charge']['host_charged_bytes'], 0)
                    self.assertEqual(result['owner_binding']['scope']['value'], dict(owner=17, generation=3))
                    self.assertIn('no invented owner join', result['invocation_owner_identity'])

    def test_reader_baseline_is_preserved_and_first_delta_reconciled(self):
        from test_quality_v1 import options
        from test_http_transfer_metrics import invocation
        from runnerhelpers.trace_v2 import transfer_counter_window
        event = initial_event(options())
        envelope = json.loads(event['fields']['expert_read_pipeline_initial'])
        envelope['value']['stats'].update(reader_interval_calls=2, reader_interval_bytes=256)
        event['fields']['expert_read_pipeline_initial'] = json.dumps(envelope)
        read = self.parse(event)['read_endpoint']
        self.assertEqual(read['stats']['reader_interval_calls'], 2)
        with self.assertRaises(ValueError):
            quality.read_observation_v3(read, options())  # Historical configured-zero contract is unchanged.
        fields = invocation(0, 'blocking')
        delta = json.loads(fields['expert_read_pipeline_delta'])
        delta.update(reader_interval_calls=4, reader_interval_bytes=512)
        fields['expert_read_pipeline_delta'] = json.dumps(delta)
        window = transfer_counter_window([json.dumps(dict(fields=fields))], options(), initial_read=read,
            producer_contract=quality.TRANSFER_UPLOAD_PRODUCER)
        self.assertTrue(window['read_continuity_valid'], window)
        delta['reader_interval_calls'] = 6
        fields['expert_read_pipeline_delta'] = json.dumps(delta)
        wrong = transfer_counter_window([json.dumps(dict(fields=fields))], options(), initial_read=read,
            producer_contract=quality.TRANSFER_UPLOAD_PRODUCER)
        self.assertFalse(wrong['read_continuity_valid'])

    def test_missing_unknown_mistyped_crossbound_and_live_worker_initials_rejected(self):
        from test_quality_v1 import options
        for key, value in (('process_id', 12346), ('process_id', True), ('producer_schema_version', True),
                ('producer_schema_version', 1.0), ('source_contract', 'unknown'), ('endpoint', 'after_prediction'),
                ('expert_transfer_policy', 'GroupedSynchronous'), ('requested_expert_transfer_policy', 'None'),
                ('owner_binding', {}), ('expert_read_pipeline_initial', {})):
            event = initial_event(options())
            event['fields'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.parse(event)
        for key in ('target', 'level'):
            event = initial_event(options())
            event[key] = 'unknown'
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.parse(event)
        for mutation in ('unknown', 'unobserved', 'verified', 'missing_stats', 'alive', 'worker_created',
                'idle_retained', 'charged', 'failed_read', 'unknown_upload', 'wrong_owner', 'unknown_owner', 'wrong_device'):
            event = initial_event(options())
            envelope = json.loads(event['fields']['expert_read_pipeline_initial'])
            read = envelope['value']
            if mutation in ('unknown', 'unobserved', 'verified'):
                envelope['state'] = mutation
            elif mutation == 'missing_stats':
                read['stats'].pop('submitted_reads')
            elif mutation == 'alive':
                read['worker_alive'] = True
            elif mutation == 'worker_created':
                read['stats']['workers_created'] = 1
            elif mutation == 'idle_retained':
                read['status'] = 'idle_retained'
            elif mutation == 'charged':
                read['current_charge']['host_charged_bytes'] = 10 << 20
            elif mutation == 'failed_read':
                read['stats']['failed_reads'] = 1
            elif mutation == 'unknown_upload':
                read['current_charge']['unknown_upload_bytes'] = 1
            else:
                owner = json.loads(event['fields']['owner_binding'])
                if mutation == 'wrong_owner':
                    owner['scope']['value']['owner'] = False
                elif mutation == 'unknown_owner':
                    owner['image_generation'] = dict(state='unknown')
                else:
                    owner['device_ordinal']['value'] = 1
                event['fields']['owner_binding'] = json.dumps(owner)
            event['fields']['expert_read_pipeline_initial'] = json.dumps(envelope)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.parse(event)

    def test_initial_is_once_before_listening_ready_and_any_invocation(self):
        from test_quality_v1 import options
        event = initial_event(options())
        initial, listening = json.dumps(event), json.dumps(dict(fields=dict(message='Ferrule OpenAI server listening')))
        work = json.dumps(dict(fields=dict(event='mixed_decoder_invocation')))
        for lines, ready in (([initial, initial, listening], None), ([listening, initial], None),
                ([work, initial, listening], None), ([initial, listening, work], None), ([initial], None),
                ([initial, listening], [])):
            with self.subTest(lines=lines, ready=ready), self.assertRaises(ValueError):
                self.parse(event, lines=lines, ready=ready)
        legacy = dict(fields=dict(event='expert_read_initial_observation', expert_read_pipeline='{}'))
        with self.assertRaisesRegex(ValueError, 'cannot invent a zero baseline'):
            self.parse(legacy)
        binding = initial_producer_binding()
        binding['contract'] = dict(binding['contract'], producer_schema_version=True)
        with self.assertRaises(ValueError):
            self.parse(event, binding=binding)


class CompiledIdentityTests(unittest.TestCase):
    def test_explicit_committed_production_archive_identity_not_d3_forever(self):
        identity = run.historical_compiled_identity()
        identity['production_commit'] = run.ARCHIVE_COMMIT
        freeze = ct.load(run.CANDIDATE_ROOT / 'source-freeze.json')
        result = run.compiled_source_review(freeze, identity)
        self.assertEqual(result['production_commit'], run.ARCHIVE_COMMIT)
        self.assertEqual(result['archive_commit'], run.ARCHIVE_COMMIT)
        self.assertFalse(result['freeze_relabelled'])
        identity['archive_commit'] = run.TRANSFER_COMMIT
        with self.assertRaisesRegex(ValueError, 'honest immutable committed'):
            run.compiled_source_review(freeze, identity)

    def test_dirty_producer_declaration_is_not_a_new_binary_or_quality_pass(self):
        import hashlib
        text = (ct.ROOT / quality.INITIAL_ENDPOINT_SOURCE).read_text()
        identity = run.historical_compiled_identity()
        pin = hashlib.sha256(text.encode()).hexdigest()
        identity['source_proof_pins'][quality.INITIAL_ENDPOINT_SOURCE] = pin
        identity['initial_endpoint_producer'] = initial_producer_binding(pin)
        self.assertEqual(run.review_initial_endpoint_producer(text, identity), identity['initial_endpoint_producer'])
        freeze = ct.load(run.CANDIDATE_ROOT / 'source-freeze.json')
        with self.assertRaisesRegex(ValueError, 'source proof pins differ'):
            run.compiled_source_review(freeze, identity)
        changed = text.replace('|| runner.physical_resource_snapshot()', '|| invented_default_snapshot()')
        identity['source_proof_pins'][quality.INITIAL_ENDPOINT_SOURCE] = hashlib.sha256(changed.encode()).hexdigest()
        identity['initial_endpoint_producer'] = initial_producer_binding(identity['source_proof_pins'][quality.INITIAL_ENDPOINT_SOURCE])
        with self.assertRaisesRegex(ValueError, 'ordering differs'):
            run.review_initial_endpoint_producer(changed, identity)
        old = run.review_transfer_build()
        self.assertFalse(old['initial_read_observation_available'])
        self.assertIsNone(old['initial_read_producer'])

    def test_v2_descriptor_identity_is_hashed_and_cannot_relabel_old_freeze(self):
        with tempfile.TemporaryDirectory(prefix='http-transfer-cpu-descriptor-', dir=ct.ROOT / 'target/validation') as temp:
            path = Path(temp) / 'CANDIDATE-INPUTS.json'
            identity = run.historical_compiled_identity()
            declared = {k: v for k, v in identity.items() if k not in ('candidate_inputs', 'candidate_inputs_sha256')}
            descriptor = dict(ct.load(run.CANDIDATE_INPUTS), schema='ferrule.storage-demand.http-transfer-cpu-candidate-inputs.v2',
                compiled_identity=declared)
            ct.write(path, descriptor)
            path.chmod(0o400)
            result = run.compiled_identity_for_inputs(path)
            self.assertEqual(result['candidate_inputs'], str(path))
            self.assertEqual(result['candidate_inputs_sha256'], ct.sha(path))
            self.assertEqual(run.transfer_plan(compiled_identity=result)['compiled_identity'], result)
            with self.assertRaisesRegex(ValueError, 'honest immutable committed'):
                run.compiled_source_review(ct.load(run.CANDIDATE_ROOT / 'source-freeze.json'), result)
            with self.assertRaisesRegex(ValueError, 'descriptor input differs'):
                run.review_transfer_build(path, compiled_identity=result)


class CaptureEntryTests(unittest.TestCase):
    def test_actual_entry_rejects_caller_authored_seals_before_substituted_target(self):
        import launch_guard
        import _profile_capture as wrapper
        import signal
        with tempfile.TemporaryDirectory(prefix='self-authored-entry-') as temp:
            directory = Path(temp)
            payload, launch_hash = launch_guard.seal(dict(directory=str(directory), spec=dict(argv=['/usr/bin/true'])), directory)
            parent = launch_guard.identity(os.getppid())
            ct.write(directory / 'custody.json', dict(root_identity=parent, helper_identity=dict(pid=parent['ppid']),
                root_session_ready=True, containment_verified=True, method='linux-subreaper-waitid', launch_id='self-authored'))
            previous = signal.getsignal(signal.SIGTERM)
            try:
                with mock.patch.dict(os.environ, FERRULE_PAPER_WORKER_TOKEN=payload['worker_token']), \
                        mock.patch.object(wrapper, 'run_target', side_effect=AssertionError('substituted target reached')) as target:
                    with self.assertRaises((ValueError, KeyError, OSError)):
                        run.main(['--_http_capture', str(directory), launch_hash, str(ex.time.monotonic()+10)])
                    target.assert_not_called()
            finally:
                signal.signal(signal.SIGTERM, previous)

    def test_real_entry_executable_rejects_self_authored_custody_and_token(self):
        import launch_guard
        import subprocess
        with tempfile.TemporaryDirectory(prefix='external-entry-rejection-') as temp:
            directory = Path(temp)
            marker = directory / 'unauthorized-target'
            payload, launch_hash = launch_guard.seal(dict(directory=str(directory),
                spec=dict(argv=['/usr/bin/touch', str(marker)], environment=ex.runner.cpu_environment())), directory)
            parent = launch_guard.identity(os.getpid())
            ct.write(directory / 'custody.json', dict(root_identity=parent, helper_identity=dict(pid=parent['ppid']),
                root_session_ready=True, containment_verified=True, method='linux-subreaper-waitid', launch_id='caller-authored'))
            result = subprocess.run([sys.executable, '-S', '-B', str(run.ENTRY), '--_http_capture',
                str(directory), launch_hash, str(ex.time.monotonic()+10)], env=dict(ex.runner.cpu_environment(),
                FERRULE_PAPER_WORKER_TOKEN=payload['worker_token']), capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('not an authenticated maintained-supervisor worker', result.stderr)
            self.assertFalse(marker.exists())
            self.assertFalse((directory / 'profile-server.json').exists())

    def test_actual_entry_real_authenticated_cpu_worker_exact_contract_and_absolute_cap(self):
        import launch_guard
        for mutation in (None, 'spec', 'grant', 'digest', 'registration', 'locks', 'no_go', 'missing_endpoint', 'deadline_after_inputs', 'deadline_after_elf'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(prefix='artifacts-transfer-acceptance-cpu-', dir=HERE) as temp:
                directory = Path(temp)
                layout = run.transfer_plan('public_cli_acceptance')
                layout['candidate'] = dict(uuid_inventory=UUIDS, cpu_component_not_a_candidate=True)
                from test_http_transfer_metrics import reference_context
                document = reference_context(layout['schedule'])['document']
                references = directory / 'CPU-COMPONENT-REFERENCES.json'
                ct.write(references, document)
                references.chmod(0o400)
                layout['http_reference_binding'] = dict(source=str(references), source_sha256=ct.sha(references),
                    document_sha256=ct.digest(document))
                descriptor = ct.load(run.CANDIDATE_INPUTS)
                bound = dict(candidate=dict(binary=descriptor['inputs']['binary'], binary_sha256=run.CLI_SHA256,
                    runtime_pin_manifest=descriptor['inputs']['runtime_pin_manifest'],
                    artifact_pins={descriptor['inputs']['binary']: run.CLI_SHA256}), uuid_inventory=UUIDS,
                    build_review=dict(initial_read_observation_available=mutation != 'missing_endpoint',
                        initial_read_producer=initial_producer_binding()))
                path = directory / 'layout.json'
                ct.write(path, layout)
                with mock.patch.object(run, 'validate_transfer_layout', return_value=layout):
                    with run.transfer_adapter(path):
                        spec = run.planned.spec_for(bound, UUIDS[0], layout['trial_order'][0].split('-',1)[1])
                now = ex.time.monotonic()
                registered = dict(schema='ferrule.storage-demand.actual-http-transfer-registration.v1', study_layout=layout,
                    execution_path=str(path), binding_path=str(path), execution_sha256=ct.sha(path), binding_sha256=ct.sha(path),
                    boot_id=actual.boot_id(), trial_order=layout['trial_order'], order=layout['order'], limits=layout['limits'],
                    schedule=layout['schedule'], owner_authorization=mutation != 'registration', owner_note=run.transfer_note(layout),
                    eight_uuids=UUIDS, selected_uuid=UUIDS[0], pilot_deadline=now+7200)
                ct.write(directory / 'registration.json', dict(registered, digest=ct.digest(registered)))
                trial = layout['trial_order'][0]
                folder = directory / trial
                folder.mkdir()
                owner = launch_guard.identity(os.getpid())
                locks = [dict(cpu_fixture_uuid=u, coordinator_pid=os.getpid()) for u in UUIDS]
                baseline = dict(scope='CPU fixture only, no GPU observation')
                ct.write(folder / 'baseline.json', baseline)
                ct.write(folder / 'locks-held.json', dict(locks=locks, coordinator=owner, all_eight_local=True, global_reservation=False))
                outer = now+650
                ct.write(folder / 'attempt.json', dict(trial=trial, admission=ex.stamp(), outer_deadline=outer,
                    pilot_deadline=registered['pilot_deadline'], registration_sha256=ct.sha(directory / 'registration.json'),
                    coordinator=owner, replay_permitted=False))
                payload = dict(schema='ferrule.storage-demand.paired-read-worker.v1', directory=str(folder), trial=trial,
                    trial_order=layout['trial_order'], mode=trial.split('-',1)[1], request_ceiling=12, schedule=layout['schedule'],
                    limits=layout['limits'], execution_path=str(path), execution_sha256=ct.sha(path), candidate_path=str(path),
                    candidate_sha256=ct.sha(path), uuid=UUIDS[0], eight_uuids=UUIDS, owner_lock=locks[0],
                    owner_note=run.transfer_note(layout), granted=mutation != 'grant', spec=spec, baseline=baseline,
                    outer_deadline=outer, pilot_deadline=registered['pilot_deadline']-100,
                    adapter_pins=layout['artifact_pins'], all_uuid_locks=locks[:7] if mutation == 'locks' else locks,
                    cpu_bound_component=bound, cpu_mutation=mutation)
                if mutation == 'spec':
                    payload['spec']['argv'] = ['/usr/bin/true']
                payload['contract_sha256'] = ct.digest({k: payload[k] for k in run.planned.CONTRACT_KEYS})
                if mutation == 'digest':
                    payload['contract_sha256'] = '0'*64
                ct.write(folder / 'launch.json', payload)
                payload, launch_hash = launch_guard.seal(payload, folder)
                environment = dict(ex.runner.cpu_environment(), FERRULE_PAPER_WORKER_TOKEN=payload['worker_token'])
                code = 'import sys; sys.path.insert(0, ' + repr(str(HERE)) + '); from test_http_transfer_controls import cpu_capture_worker; sys.exit(cpu_capture_worker(sys.argv[1]))'
                receipt = actual.pilot.supervisor.run_process(ct.isolated_argv(
                    [sys.executable, '-S', '-B', '-c', code, launch_hash], environment), 15, payload)
                self.assertEqual(receipt['returncode'], 0, receipt)
                self.assertTrue(ex.natural_descendant_drain(receipt), receipt)
                result = ct.load(folder / 'cpu-entry-result.json')
                self.assertEqual(result['authorized_entry'], mutation is None, result)
                self.assertEqual(len(result['target_calls']), 1 if mutation is None else 0, result)
                if mutation in ('deadline_after_inputs', 'deadline_after_elf'):
                    self.assertIn('expired during capture input/ELF verification', result['error'])
                auth = ct.load(folder / 'cpu-worker-authenticated.json')
                self.assertEqual(auth['launch_id'], receipt['custody']['launch_id'])
                self.assertEqual(auth['worker'], receipt['custody']['root_identity'])
                if mutation is None:
                    self.assertEqual(result['returncode'], 0)
                    cpu_report = ct.load(folder / 'profile-server.json')
                    self.assertTrue(cpu_report['planned_stop_validated'])
                    self.assertEqual(cpu_report['returncode'], -15)
                    self.assertTrue((folder / 'cpu-target-ready').is_file())
                    self.assertGreaterEqual(result['lock_checks'], 9)
                    context = ct.load(folder / 'http-capture-launch.json')
                    self.assertEqual(result['target_calls'][0]['absolute_deadline'], context['target_deadline'])
                    self.assertEqual(ct.load(folder / 'cpu-capture-argv.json')[-1], repr(context['target_deadline']))


class ProspectivePlanTests(unittest.TestCase):
    def test_acceptance_and_pilot_denominators_are_separate(self):
        acceptance = run.transfer_plan('public_cli_acceptance')
        pilot = run.transfer_plan('pilot')
        self.assertEqual((acceptance['scheduled_starts'], acceptance['scheduled_requests']), (4, 48))
        self.assertEqual((pilot['scheduled_starts'], pilot['scheduled_requests']), (8, 96))
        self.assertEqual(acceptance['cases'], pilot['cases'])
        self.assertEqual(acceptance['schedule'], pilot['schedule'])
        self.assertEqual(len(set(acceptance['trial_order'])), 4)
        self.assertFalse(acceptance['filesystem_cold_claim'])
        self.assertEqual(acceptance['http_expectation_contract'], 'bound-http-reference-v1')
        self.assertIsNone(acceptance['http_reference_binding'])
        self.assertIsNone(acceptance['candidate'])
        self.assertIsNone(pilot['candidate'])

    def test_actual_reference_pins_checkpoint_manifest_and_identity_scope(self):
        contract = run.transfer_input_contract()
        self.assertEqual(len(contract['reference_pins']), 8)
        self.assertEqual(len(contract['shard_stat_identities']), 14)
        self.assertIsNone(contract['checkpoint_content_sha256'])
        layout = run.transfer_plan()
        self.assertIn(contract['checkpoint_manifest'], layout['artifact_pins'])
        self.assertIn(str(HERE.parent / 'runnerhelpers/http_expectations.py'), layout['artifact_pins'])

    def test_unknown_upload_producer_rejected_historical_defaults_unchanged(self):
        sys.path.insert(0, str(HERE))
        from test_quality_v1 import observation, options
        read = observation(options())
        read['current_charge']['unknown_upload_bytes'] = 0
        with self.assertRaises(ValueError):
            quality.read_observation_v3(read, options())
        quality.read_observation_v3(read, options(), producer_contract=quality.TRANSFER_UPLOAD_PRODUCER)
        with self.assertRaisesRegex(ValueError, 'unknown upload-charge producer'):
            quality.read_observation_v3(read, options(), producer_contract='unknown')
        for bad in (1, False, None):
            read['current_charge']['unknown_upload_bytes'] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                quality.read_observation_v3(read, options(), producer_contract=quality.TRANSFER_UPLOAD_PRODUCER)

    def test_missing_initial_observation_does_not_become_zero(self):
        with self.assertRaisesRegex(ValueError, 'cannot invent a zero baseline'):
            ex.read_initial_endpoint([], {}, quality.TRANSFER_UPLOAD_PRODUCER)

    def test_each_reference_and_shard_binding_mutation_fails_without_payload_reads(self):
        contract = run.transfer_input_contract()
        for section in ('reference_pins', 'model_metadata', 'shard_stat_identities'):
            for path in contract[section]:
                changed = copy.deepcopy(contract)
                changed[section][path] = '0'*64
                with self.subTest(section=section, path=path), self.assertRaisesRegex(ValueError, 'contract changed'):
                    run.validate_transfer_inputs(changed)

    def test_next_registration_rechecks_inputs_before_authorization_or_locks(self):
        with tempfile.TemporaryDirectory(prefix='admission-controls-') as temp, ExitStack() as stack:
            directory = Path(temp)
            layout = run.transfer_plan('public_cli_acceptance')
            stack.enter_context(mock.patch.object(run.historical, 'unseal', return_value={}))
            stack.enter_context(mock.patch.object(run, 'validate_transfer_layout', return_value=layout))
            checked = stack.enter_context(mock.patch.object(run, 'validate_transfer_inputs', side_effect=ValueError('input identity changed')))
            authorize = stack.enter_context(mock.patch.object(actual.pilot.telemetry, 'authorize', side_effect=AssertionError('GPU authorization forbidden')))
            locks = stack.enter_context(mock.patch.object(ct, 'exclusive_owner_lock', side_effect=AssertionError('GPU locks forbidden')))
            stack.enter_context(mock.patch.object(actual, 'registration', side_effect=lambda d: run.transfer_registration(d, directory / 'layout.json')))
            with self.assertRaisesRegex(ValueError, 'input identity changed'):
                actual.execute_next(directory)
            checked.assert_called_once_with(layout['input_contract'])
            authorize.assert_not_called()
            locks.assert_not_called()

    def test_fixed_inputs_checked_before_each_launch_without_payload_reads(self):
        contract = run.transfer_input_contract()
        original_load = ct.load
        original_stat = Path.stat
        original_link = Path.is_symlink
        calls = []
        def load(path):
            if str(path).endswith('/model.safetensors.index.json'):
                return dict(weight_map={str(i): Path(n).name for i, n in enumerate(contract['shard_stat_identities'])})
            return original_load(path)
        def stat(path, *args, **kwargs):
            if str(path) in contract['shard_stat_identities']:
                dev, ino, size, mtime, ctime = contract['shard_stat_identities'][str(path)]
                from types import SimpleNamespace
                return SimpleNamespace(st_dev=dev, st_ino=ino, st_size=size, st_mtime_ns=mtime, st_ctime_ns=ctime)
            return original_stat(path, *args, **kwargs)
        with mock.patch.object(ct, 'load', side_effect=load), mock.patch.object(ct, 'check_pins', side_effect=lambda p: calls.append(p)), \
                mock.patch.object(Path, 'stat', stat), mock.patch.object(Path, 'is_symlink', lambda p: False if str(p) in contract['shard_stat_identities'] else original_link(p)):
            for _ in range(2):
                run.validate_transfer_inputs(contract)
        self.assertEqual(sum(p == contract['reference_pins'] for p in calls), 2)
        self.assertEqual(sum(p == contract['model_metadata'] for p in calls), 2)


class HTTPReferenceControlTests(unittest.TestCase):
    def test_prospective_layout_requires_hash_bound_shared_eos_reference_contract(self):
        from runnerhelpers import http_expectations as expected
        import test_http_transfer_metrics as fixtures
        layout = run.transfer_plan('public_cli_acceptance')
        with tempfile.TemporaryDirectory(prefix='http-reference-controls-') as temp:
            directory = Path(temp)
            document = fixtures.reference_context(layout['schedule'], short_qa=True)['document']
            source = directory / 'references.json'
            ct.write(source, document)
            source.chmod(0o400)
            binding = dict(source=str(source), source_sha256=ct.sha(source), document_sha256=ct.digest(document))
            layout = run.transfer_plan('public_cli_acceptance', http_reference_binding=binding)
            self.assertEqual(layout['expected_measured_tokens'], 4 * 5 * 3)
            path = directory / 'layout.json'
            ct.write(path, layout)
            self.assertEqual(run.validate_transfer_layout(path, candidate_required=False), layout)
            context = run.validate_http_references(layout)
            row = layout['schedule'][1]
            output = expected.normalize_http_expectation(row, contract=layout['http_expectation_contract'], reference_context=context)
            self.assertEqual(output['finish_reason'], 'stop')
            self.assertEqual(output['actual_output_tokens'], 3)
            for mutate in ('historical', 'bad_source_hash', 'missing_reference'):
                changed = copy.deepcopy(layout)
                if mutate == 'historical':
                    changed['http_expectation_contract'] = expected.HISTORICAL_LENGTH
                    changed['http_reference_binding'] = None
                    check = lambda: run.validate_transfer_layout(path, candidate_required=False)
                    path.write_text(json.dumps(changed))
                elif mutate == 'bad_source_hash':
                    changed['http_reference_binding']['source_sha256'] = '0'*64
                    check = lambda: run.validate_http_references(changed)
                else:
                    changed['http_reference_binding'] = None
                    check = lambda: run.validate_http_references(changed)
                with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                    check()

    def test_execution_gate_accepts_exact_reviewed_eos_not_forced_length(self):
        from runnerhelpers import http_expectations as expected
        import test_http_transfer_metrics as fixtures
        rows = fixtures.schedule()
        context = fixtures.reference_context(rows, short_qa=True)
        observed = fixtures.arm(context=context)['lifecycle']['requests']['observations'][1]
        trace = dict(observed['trace'], metric_eligible=True, worker_terminal_enqueue=[dict(
            usage_available=True, delivery_boundary='worker_channel_enqueue', **observed['actual']['usage'])])
        args = (rows[1], observed['actual'], trace, observed['logical_counters'], dict(status='shared_client_delta'))
        self.assertTrue(ex.request_gate(*args, expectation_contract=expected.BOUND_HTTP_REFERENCE, reference_context=context))
        self.assertFalse(ex.request_gate(*args, expectation_contract=expected.HISTORICAL_LENGTH))
        for key, value in (('finish_reason', 'length'), ('actual_output_tokens', True), ('usage', {}), ('text', 'wrong')):
            actual_output = copy.deepcopy(observed['actual'])
            actual_output[key] = value
            with self.subTest(key=key):
                self.assertFalse(ex.request_gate(rows[1], actual_output, trace, observed['logical_counters'], args[-1],
                    expectation_contract=expected.BOUND_HTTP_REFERENCE, reference_context=context))


class ObserverAcknowledgementTests(unittest.TestCase):
    def test_controller_stops_observer_before_ack_and_rejects_errors_or_partial_work(self):
        from types import SimpleNamespace
        import threading
        import time
        for failure in (None, 'sticky_error', 'failed_observer', 'alive_thread', 'incomplete_requests', 'wrong_birth'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory(prefix='http-ack-controls-') as temp:
                directory = Path(temp)
                ct.write(directory / 'worker-launch.json', dict(scope='CPU ACK interface only; not launch authority'))
                now = time.monotonic()
                root = ex.observer.proc_identity(os.getpid())
                event = dict(ex._identity_key(root), stop_id='1'*32, stop_reason='matrix_worker_stop_marker',
                    requested_monotonic=now, absolute_deadline=now+2.)
                if failure == 'wrong_birth':
                    event['birth_ticks'] += 1
                ct.write(directory / 'http-stop-request.json', dict(schema='ferrule.storage-demand.http-stop-request.v1',
                    stop_event=event, launch_sha256=ct.sha(directory / 'worker-launch.json'), worker_identity=root))
                (directory / 'http-stop-request.json').chmod(0o400)
                requests = dict(can_continue=True, observations=[dict(metric_eligible=True)] * (11 if failure == 'incomplete_requests' else 12))
                ct.write(directory / 'requests-final.json', requests)
                resources = dict(status='observed_within_targets', errors=['sticky error'] if failure == 'sticky_error' else [])
                monitor = SimpleNamespace(finish=mock.Mock(return_value=resources), failed=threading.Event(),
                    thread=SimpleNamespace(is_alive=lambda: failure == 'alive_thread'))
                if failure == 'failed_observer':
                    monitor.failed.set()
                payload = dict(limits=dict(shutdown_seconds=2), outer_deadline=now+2., pilot_deadline=now+2., spec={})
                with mock.patch.object(ex, 'transfer_runtime_maps', return_value=dict(scope='CPU map interface fixture')) as maps:
                    if failure is None:
                        self.assertEqual(ex.acknowledge_http_stop(directory, payload, root, root, monitor, requests), resources)
                        ack = ct.load(directory / 'http-stop-ack.json')
                        self.assertEqual(ack['stop_event'], event)
                        self.assertTrue(ack['observer_stopped'])
                        self.assertEqual(len(ack['evidence_pins']), 3)
                        self.assertEqual((directory / 'http-stop-ack.json').stat().st_mode & 0o222, 0)
                    else:
                        with self.assertRaises(ValueError):
                            ex.acknowledge_http_stop(directory, payload, root, root, monitor, requests)
                        self.assertFalse((directory / 'http-stop-ack.json').exists())
                    if failure in ('incomplete_requests', 'wrong_birth'):
                        maps.assert_not_called()
                    else:
                        monitor.finish.assert_called_once_with()


class CoordinatorFailureTests(unittest.TestCase):
    def test_forced_descendant_cleanup_is_not_normal_http_completion(self):
        receipt = actual.pilot.supervisor.run_process(ct.isolated_argv(
            [sys.executable, '-S', '-B', '-c', 'pass'], ex.runner.cpu_environment()), 5)
        self.assertTrue(ex.natural_descendant_drain(receipt))
        observed = dict(status='complete', transfer_controls_revision=2, closure=dict(final=dict(can_continue=True)),
            planned_http_stop=dict(schema='ferrule.storage-demand.owned-http-stop-result.v1', disposition='expected_planned_stop',
                actual_returncode=-15), capture=dict(returncode=-15, planned_stop_validated=True, stop_reason='planned_stop'))
        post = dict(status='verified', all_eight_idle=True)
        self.assertTrue(ex.lifecycle_assessment(observed, receipt, post, transfer_required=True)['can_continue'])
        for change in ('signals', 'exit_code', 'natural_drain', 'missing', 'custody_errors', 'reporting_errors', 'session'):
            changed = copy.deepcopy(receipt)
            if change == 'natural_drain':
                changed['custody']['natural_drain'] = dict(outcome='grace_exhausted')
            elif change == 'missing':
                changed['custody'].pop('entries')
            elif change in ('custody_errors', 'reporting_errors'):
                changed['custody']['errors' if change == 'custody_errors' else change] = ['sticky failure']
            elif change == 'session':
                changed['custody']['root_session_ready'] = False
            else:
                changed['custody']['entries'][0][change] = [15] if change == 'signals' else -9
            with self.subTest(change=change):
                self.assertFalse(ex.lifecycle_assessment(observed, changed, post, transfer_required=True)['can_continue'])
        for revision in (None, 1, True):
            changed = dict(observed, transfer_controls_revision=revision)
            with self.subTest(revision=revision):
                self.assertFalse(ex.lifecycle_assessment(changed, receipt, post, transfer_required=True)['can_continue'])

    def test_reprobe_failure_appends_and_disqualifies_return_and_finalizer(self):
        with tempfile.TemporaryDirectory(prefix='transfer-controls-') as temp, ExitStack() as stack:
            directory = Path(temp)
            trial = 'A1-blocking-synchronous'
            layout = run.transfer_plan('public_cli_acceptance')
            registered = dict(study_layout=layout, trial_order=[trial], order=[dict(block='A1', modes=['blocking-synchronous'])],
                limits=layout['limits'], schedule=layout['schedule'], pilot_deadline=ex.time.monotonic()+7200,
                selected_uuid=UUIDS[0], eight_uuids=UUIDS, binding_path=str(directory/'binding.json'),
                execution_path=str(directory/'binding.json'), binding_sha256='a'*64, execution_sha256='a'*64,
                started=ex.stamp(), scheduled_blocks=1)
            ct.write(directory/'registration.json', {})
            ct.write(directory/'binding.json', {})
            summary = ex.censoring_summary([], layout['schedule'])
            for module, name, value in (
                (actual, 'registration', mock.Mock(return_value=registered)),
                (actual.binding, 'validate_binding', mock.Mock(return_value=dict(candidate={}))),
                (actual.binding, 'validate_execution', mock.Mock(return_value=dict(artifact_pins={}))),
                (actual, 'actual_adapter', lambda: nullcontext()),
                (actual, 'ledger', mock.Mock(return_value=dict(can_continue=True))),
                (actual.planned, 'can_admit', mock.Mock(return_value=True)),
                (actual.planned, 'spec_for', mock.Mock(return_value={})),
                (actual.planned, 'supervise', mock.Mock(return_value={})),
                (actual.pilot.telemetry, 'authorize', mock.Mock()),
                (actual.pilot, 'idle', mock.Mock(return_value=dict(status='verified', all_eight_idle=True))),
                (actual.pilot, 'baseline', mock.Mock(return_value={})),
                (actual.pilot, 'remaining', mock.Mock()),
                (ct, 'exclusive_owner_lock', lambda uuid: nullcontext(dict(uuid=uuid))),
                (ct, 'validate_owner_lock', mock.Mock()),
                (ex, 'lifecycle_assessment', mock.Mock(return_value=dict(can_continue=True, stop_reasons=[]))),
                (ex, 'recover_observations', mock.Mock(return_value=[])),
                (ex, 'recover_requests', mock.Mock(return_value=summary)),
            ):
                stack.enter_context(mock.patch.object(module, name, value))
            original_write = ct.write
            def write(path, value):
                if Path(path).name == 'locks-reprobed.json':
                    raise OSError('injected post-assessment reprobe receipt failure')
                original_write(path, value)
            with mock.patch.object(ct, 'write', side_effect=write):
                returned = actual.execute_next(directory)
            folder = directory/trial
            self.assertTrue(ct.load(folder/'assessment.json')['assessment']['can_continue'])
            self.assertFalse(returned['assessment']['can_continue'])
            self.assertTrue((folder/'coordinator-failure.json').is_file())
            self.assertTrue((directory/'study-stop.json').is_file())
            final = actual.finalize(directory)
            self.assertEqual(final['status'], 'stopped')
            self.assertEqual(final['actual_completed_lifecycles'], 0)
            with self.assertRaises(ValueError):
                actual.next_trial(directory, registered)

    def test_partial_assessment_and_request_json_keep_all_fixed_slots_visible(self):
        with tempfile.TemporaryDirectory(prefix='partial-controls-') as temp:
            directory = Path(temp)
            layout = run.transfer_plan('public_cli_acceptance')
            trial = layout['trial_order'][0]
            folder = directory / trial
            folder.mkdir()
            raw = '{"assessment":'
            (folder / 'assessment.json').write_text(raw)
            for name in ('attempt.json', 'launch.json'):
                ct.write(folder / name, dict(scope='CPU partial-journal fixture only'))
            for row in layout['schedule'][:2]:
                request_folder = folder / row['name']
                request_folder.mkdir()
                (request_folder / 'issued.json').write_text('{')
                (request_folder / 'http-terminal.json').write_text('{')
            registered = dict(study_layout=layout, trial_order=layout['trial_order'], order=layout['order'],
                limits=layout['limits'], schedule=layout['schedule'], pilot_deadline=ex.time.monotonic()+7200,
                selected_uuid=UUIDS[0], eight_uuids=UUIDS, binding_path=str(directory/'binding.json'),
                binding_sha256='a'*64, execution_sha256='a'*64, started=ex.stamp(), scheduled_blocks=2)
            with mock.patch.object(actual, 'registration', return_value=registered):
                actual.finalize(directory)
            result = ct.load(directory / 'paired-final.json')
            self.assertEqual(result['status'], 'stopped')
            self.assertEqual(len(result['lifecycles']), 4)
            self.assertEqual(result['scheduled_requests'], 48)
            self.assertEqual(sum(r['summary']['scheduled'] for r in result['lifecycles']), 48)
            self.assertEqual(sum(r['summary']['issued'] for r in result['lifecycles']), 2)
            self.assertEqual(sum(r['summary']['censored'] for r in result['lifecycles']), 2)
            self.assertEqual(result['actual_completed_lifecycles'], 0)
            self.assertEqual((folder / 'assessment.json').read_text(), raw)
            self.assertTrue((folder / 'coordinator-failure.json').is_file())
            self.assertTrue((directory / 'study-stop.json').is_file())

    def test_missing_or_crossbound_terminal_receipts_permanently_stop(self):
        for mutation in ('missing', 'crossbound', 'changed_assessment'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(prefix='terminal-controls-') as temp:
                directory = Path(temp)
                folder = directory / 'A1-blocking-synchronous'
                folder.mkdir()
                row = dict(trial=folder.name, assessment=dict(can_continue=True),
                    summary=dict(measured_qa=dict(eligible_success=5)))
                for name, value in (('attempt.json', {}), ('locks-held.json', {}), ('assessment.json', row)):
                    ct.write(folder / name, value)
                binding = actual.coordinator_terminal_binding(folder)
                if mutation == 'crossbound':
                    binding['trial'] = 'foreign-arm'
                if mutation != 'missing':
                    ct.write(folder / 'locks-released.json', dict(binding, phase='release', context_exited=True, all_eight_local=True))
                    ct.write(folder / 'locks-reprobed.json', dict(binding, phase='reprobe', all_eight_nonblocking_reacquired=True, context_exited=True, local_only=True))
                if mutation == 'changed_assessment':
                    changed = copy.deepcopy(row)
                    changed['injected'] = True
                    (folder / 'assessment.json').write_text(json.dumps(changed))
                result = actual.effective_assessment(folder, row, require_completion=True)
                self.assertFalse(result['assessment']['can_continue'])
                self.assertEqual(result['summary']['measured_qa']['eligible_success'], 0)
                self.assertTrue(ct.load(folder / 'assessment.json')['assessment']['can_continue'])
                self.assertEqual(ct.load(folder / 'coordinator-failure.json')['assessment_sha256'], ct.sha(folder / 'assessment.json'))
                self.assertTrue((directory / 'study-stop.json').is_file())
                with self.assertRaisesRegex(ValueError, 'permanently stopped'):
                    actual.next_trial(directory, dict(trial_order=[folder.name], study_layout={}))



class OwnedCPUStopTests(unittest.TestCase):
    def test_real_owned_planned_stop_exports_actual_code_and_rejects_tampering(self):
        import _profile_capture as wrapper
        import threading
        import time
        for graceful in (False, True):
            with self.subTest(graceful=graceful), tempfile.TemporaryDirectory(prefix='http-stop-cpu-') as temp:
                directory = Path(temp)
                ct.write(directory / 'worker-launch.json', dict(scope='CPU fixture only, not GPU launch authority'))
                launch_hash = ct.sha(directory / 'worker-launch.json')
                ready = directory / 'cpu-ready'
                handler = 'signal.signal(signal.SIGTERM, lambda s,f: sys.exit(0)); ' if graceful else ''
                code = ('import signal,sys,time; from pathlib import Path; ' + handler
                    + 'Path(' + repr(str(ready)) + ').touch(); time.sleep(10)')
                errors = []
                def request_stop():
                    deadline = time.monotonic() + 4
                    while not ready.exists() and time.monotonic() < deadline:
                        time.sleep(.01)
                    if not ready.exists():
                        errors.append('CPU child did not become ready')
                    ex.stop_capture(directory)
                thread = threading.Thread(target=request_stop)
                thread.start()
                def acknowledge(child, event, deadline, active):
                    active()
                    ct.write(directory / 'http-stop-request.json', dict(schema='ferrule.storage-demand.http-stop-request.v1',
                        stop_event=event, launch_sha256=launch_hash))
                    (directory / 'http-stop-request.json').chmod(0o400)
                    ct.write(directory / 'resources-final.json', dict(status='observed_within_targets', errors=[], scope='CPU fixture, no GPU observation'))
                    for name in ('stop-library-maps.json', 'requests-final.json'):
                        ct.write(directory / name, dict(scope='CPU fixture only'))
                    ct.write(directory / 'http-stop-ack.json', dict(schema='ferrule.storage-demand.http-stop-ack.v1',
                        stop_event=event, launch_sha256=launch_hash,
                        request_sha256=ct.sha(directory / 'http-stop-request.json'), observer_stopped=True,
                        qualification=True, evidence_pins=ct.inventory([directory / name for name in
                            ('resources-final.json', 'stop-library-maps.json', 'requests-final.json')]),
                        acknowledged_monotonic=time.monotonic()))
                    (directory / 'http-stop-ack.json').chmod(0o400)
                    return dict(stop_event=event, observer_stopped=True)
                try:
                    target_deadline = time.monotonic() + 5
                    result = wrapper.run_target(directory, [sys.executable, '-S', '-B', '-c', code],
                        5, 2, on_stop=acknowledge, absolute_deadline=target_deadline)
                finally:
                    thread.join(timeout=5)
                self.assertEqual(errors, [])
                self.assertEqual(result, 0)
                report = ct.load(directory / 'profile-server.json')
                root = report['root_identity']
                exported = ex.validate_http_stop(directory, report, root, target_deadline=target_deadline)
                with self.assertRaisesRegex(ValueError, 'original target deadline'):
                    ex.validate_http_stop(directory, dict(report, absolute_deadline=target_deadline+1), root, target_deadline=target_deadline)
                self.assertEqual(exported['actual_returncode'], 0 if graceful else -15)
                self.assertFalse(exported['physical_release_verified'])
                for change in ('returncode', 'planned_stop_validated', 'stop_reason', 'stop_event', 'ended_monotonic'):
                    changed = copy.deepcopy(report)
                    changed[change] = dict(stop_id='stale') if change == 'stop_event' else (
                        -9 if change == 'returncode' else False if change in ('planned_stop_validated', 'ended_monotonic') else 'signal')
                    with self.subTest(change=change), self.assertRaises((ValueError, KeyError)):
                        ex.validate_http_stop(directory, changed, root)
                for change in ('status', 'errors'):
                    resource = dict(status='observed_within_targets', errors=[])
                    resource[change] = 'unobserved' if change == 'status' else ['sticky observer failure']
                    (directory / 'resources-final.json').write_text(json.dumps(resource))
                    ack_path = directory / 'http-stop-ack.json'
                    ack = ct.load(ack_path)
                    ack['evidence_pins'][str(directory / 'resources-final.json')] = ct.sha(directory / 'resources-final.json')
                    ack_path.chmod(0o600)
                    ack_path.write_text(json.dumps(ack))
                    ack_path.chmod(0o400)
                    with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'observer status unobserved/sticky'):
                        ex.validate_http_stop(directory, report, root)


if __name__ == '__main__':
    unittest.main()
