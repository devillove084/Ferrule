"""Prospective transfer provenance and lifecycle admission; no process runner.

A local proof is evidence, never permission. Only an independent MAIN-GO binding
its exact bytes, independently derived partition and compatibility review permits
admission. Writable build inputs are deliberately not launch freshness inputs.
"""
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import time

from .evidence_v2 import ct, require

CANDIDATE = 'ferrule.storage-demand.http-transfer-candidate-binding.v4'
PROOF = 'ferrule.storage-demand.http-transfer-reviewed-provenance.v1'
PARTITION = 'ferrule.storage-demand.http-transfer-role-partition.v1'
LIVE, BUILD, QUALITY = 'live', 'build_only', 'archived_quality'
TOKEN = object()
COMMANDS = ('build-cli', 'cubin-cli-list', 'cubin-cli-extract', 'ptx-cli-list',
            'cli-serve-help', 'cli-version', 'cli-help')


def enabled(layout):
    return (layout.get('candidate') or {}).get('schema') == CANDIDATE


def finite_phase(deadline):
    remaining = signal.getitimer(signal.ITIMER_REAL)[0]
    require(type(deadline) in (float, int) and time.monotonic() < deadline and remaining > 0
        and remaining <= deadline - time.monotonic() + .01,
        'role admission requires the existing finite phase deadline; no timer reset')


def document(api, path, expected=None):
    path = api.immutable_input(path)
    before = api.preparation_identity(path)
    raw = path.read_bytes()
    pin = hashlib.sha256(raw).hexdigest()
    require(api.preparation_identity(path) == before and (expected is None or pin == expected),
        'role manifest content/identity changed: ' + str(path))
    return json.loads(raw), pin


def layout_subject(layout):
    value = copy.deepcopy(layout)
    candidate = value['candidate']
    candidate.update(launch_review=None, launch_review_sha256=None, reviewed_provenance=None)
    candidate['acceptance'] = {k: candidate['acceptance'][k] for k in ('path', 'sha256')}
    return ct.digest(value)


def derive(api, layout):
    """Derive every role from real descriptor categories, never proof-supplied sets.

    Only small pinned metadata is read here. No original header, registry,
    compiler, archive payload, full source tree or historical quality leaf is
    stat'ed/opened. Unknown descriptor categories fail in the descriptor validator;
    uncategorized fixed evidence defaults LIVE, never silently disappears.
    """
    require(enabled(layout) and layout.get('purpose') == 'pilot', 'role partition is prospective pilot-only')
    identity, descriptor = api.transfer_candidate_descriptor(layout)
    root = Path(layout['candidate']['inputs']).parent
    docs, metadata = {}, {layout['candidate']['inputs']: layout['candidate']['inputs_sha256']}
    for category, name in descriptor['inputs'].items():
        if category != 'binary':
            docs[category], metadata[name] = document(api, name, descriptor['sha256'][category])
    rows = {}
    rank = {BUILD: 0, QUALITY: 1, LIVE: 2}
    def add(pins, role, rule):
        require(isinstance(pins, dict) and pins, 'empty role category: ' + rule)
        for name, pin in pins.items():
            require(isinstance(name, str) and Path(name).is_absolute() and '..' not in Path(name).parts
                and re.fullmatch('[0-9a-f]{64}', pin or ''), 'invalid role member: ' + str(name))
            old = rows.get(name)
            require(old is None or old['sha256'] == pin, 'conflicting role content: ' + name)
            if old is None:
                rows[name] = dict(sha256=pin, role=role, rules=[rule])
            else:
                old['role'] = max((old['role'], role), key=rank.__getitem__)
                old['rules'] = sorted(set(old['rules']) | {rule})
    frozen = Path(docs['source_freeze']['frozen_source'])
    freeze = docs['source_freeze']
    relative = freeze['pins_relative_to_frozen_source']
    require(frozen == root / 'frozen-source' and freeze['commit'] == identity['archive_commit']
        and freeze['file_count'] == len(relative) and freeze['archive_sha256'] == identity['archive_sha256']
        and all(n and not Path(n).is_absolute() and '..' not in Path(n).parts for n in relative),
        'role frozen-source descriptor differs')
    originals = {str(ct.ROOT / n): h for n, h in relative.items()}
    require(originals == freeze['original_source_pins'] and ct.digest(originals) == freeze['source_pins_digest']
        and ct.digest(relative) == freeze['source_pins_digest_relative']
        and {n: relative[n] for n in api.PROOF_PATHS} == identity['source_proof_pins'],
        'role compiled source digest/proof differs')
    add({str(frozen / n): h for n, h in relative.items()}, BUILD, 'source_freeze.pins_relative_to_frozen_source')
    add({freeze['archive']: identity['archive_sha256']}, BUILD, 'source_freeze.archive')
    artifact = docs['artifact']
    binary = descriptor['inputs']['binary']
    require(artifact['executable'] == binary and artifact['sha256'] == identity['binary_sha256']
        and artifact['source_digest'] == freeze['source_pins_digest']
        and artifact['compiled_artifact']['executable'] == str(root / 'cargo-target/release/ferrule'),
        'role actual compiled/staged ELF relation differs')
    add({artifact['compiled_artifact']['executable']: identity['binary_sha256']}, BUILD, 'artifact.compiled_artifact')
    for category in ('toolchain_pins', 'external_inputs', 'dependencies', 'executed_helpers', 'native_cubins'):
        add(docs[category]['pins'], BUILD, category + '.pins')
    add({n: x['sha256'] for n, x in docs['toolchain_pins']['linker_search_inputs'].items()}, BUILD,
        'toolchain_pins.linker_search_inputs')
    for key in ('original_pins', 'immutable_copies'):
        add(docs['native_archives'][key], BUILD, 'native_archives.' + key)
    for command in COMMANDS:
        for suffix in ('-command.json', '-receipt.json', '-supervisor.json', '.stdout', '.stderr'):
            name = str(root / (command + suffix))
            metadata[name] = ct.sha(api.immutable_input(name))
    for basename in ('build-environment.json', 'external-input-postverify.json', 'executed-helper-copy-pins.json'):
        name = str(root / basename)
        metadata[name] = ct.sha(api.immutable_input(name))
    copies, _ = document(api, root / 'executed-helper-copy-pins.json', metadata[str(root / 'executed-helper-copy-pins.json')])
    add(copies['pins'], BUILD, 'executed-helper-copy-pins.pins')
    controls = docs['control_helpers']
    add({controls['archive']: controls['archive_sha256']}, BUILD, 'control_helpers.archive')
    add({binary: identity['binary_sha256']}, LIVE, 'descriptor.binary')
    add(docs['runtime_pin_manifest']['pins'], LIVE, 'runtime_pin_manifest.pins')
    require(set(layout['artifact_pins']) == {str(p.resolve()) for p in api.transfer_workflow_paths(role_partition=True)},
        'independent required live workflow coverage differs')
    add(layout['artifact_pins'], LIVE, 'layout.current_workflow')
    contract = layout['input_contract']
    add(contract['reference_pins'], LIVE, 'input_contract.reference_pins')
    add(contract['model_metadata'], LIVE, 'input_contract.model_metadata')
    add({contract['checkpoint_manifest']: contract['checkpoint_manifest_sha256']}, LIVE, 'input_contract.checkpoint_manifest')
    if layout['http_reference_binding'] is not None:
        ref = layout['http_reference_binding']
        add({ref['source']: ref['source_sha256']}, LIVE, 'http_reference_binding.source')
    # Freeze the entire small semantic proof set. No original writable Rust source
    # is consulted by admission; callbacks consume these exact reviewed bytes.
    semantic = {str(frozen / n): relative[n] for n in ('Cargo.toml', *api.PROOF_PATHS)}
    add(semantic, LIVE, 'frozen_semantic_source')
    binding = layout['candidate']['acceptance']
    require(binding['sha256'] == api.COMPATIBILITY_ACCEPTANCE_SHA256, 'role policy limited to exact reviewed Qwen acceptance')
    report = api.acceptance_manifest({k: binding[k] for k in ('path', 'sha256')})
    accepted = report['study_layout']
    require(accepted['compiled_identity'] == identity and accepted['input_contract'] == contract,
        'role original acceptance candidate/input identity differs')
    expected = api.transfer_plan('public_cli_acceptance', compiled_identity=identity,
        http_reference_binding=layout['http_reference_binding'], role_partition=True, workflow_pins=layout['artifact_pins'])
    require(set(accepted) == set(expected) and all(accepted[k] == v for k, v in expected.items()
        if k not in ('candidate', 'artifact_pins')), 'role original acceptance workload differs')
    add(report['artifact_pins'], QUALITY, 'original_acceptance.artifact_pins')
    metadata[binding['path']] = binding['sha256']
    old = accepted['candidate']
    _, metadata[old['launch_review']] = document(api, old['launch_review'], old['launch_review_sha256'])
    for name, pin in accepted['artifact_pins'].items():
        if name in layout['artifact_pins'] and layout['artifact_pins'][name] != pin:
            require(name in {str(ct.ROOT / p) for p in api.ROLE_COMPATIBILITY_FILES},
                'role historical workflow replacement outside review scope')
        else:
            add({name: pin}, LIVE, 'unchanged_original_workflow')
    # Metadata actually parsed at admission is live even when it records build-only
    # bytes. Its fixed name set is code-derived, not a proof-controlled whitelist.
    add(metadata, LIVE, 'independent_descriptor_and_review_metadata')
    cubins = docs['native_cubins']
    require(cubins['cli_sha256'] == identity['binary_sha256'] and cubins['embedded_elf_count'] == 2
        and cubins['embedded_ptx_count'] == 0 and cubins['compiled_target'] == 'sm_86',
        'role minimization limited to exact accepted two-cubin/no-PTX CLI')
    actual = {n: h for n, h in relative.items() if any(n == p or n.startswith(p + '/') for p in api.COMPILED_SCOPE)}
    source = dict(production_commit=identity['production_commit'], archive_commit=identity['archive_commit'],
        compiled_scope=list(api.COMPILED_SCOPE), compiled_files=len(actual), compiled_scope_sha256=ct.digest(actual),
        full_source_sha256=ct.digest(relative), archive_sha256=identity['archive_sha256'], freeze_relabelled=False)
    partition = dict(schema=PARTITION, entries=dict(sorted(rows.items())),
        policy='category-derived; live wins overlaps; unknown metadata live; unknown categories fail',
        workload_sha256=ct.digest(dict(cases=layout['cases'], compiled_identity=identity)),
        freshness=dict(before_spawn='all live content', request='non-ELF live control/reference content',
            runtime='exact ELF at execution; all runtime libraries at admission and ready/after-requests/pre-stop maps'))
    live_pins = {n: x['sha256'] for n, x in rows.items() if x['role'] == LIVE}
    runtime_paths = set(docs['runtime_pin_manifest']['pins']) | {binary}
    return dict(partition=partition, metadata_pins=metadata, source=source, descriptor=descriptor,
        semantic_pins=semantic, acceptance=report, documents=docs,
        live_pins=live_pins, request_pins={n: h for n, h in live_pins.items() if n not in runtime_paths})


def identity_subject(layout, graph):
    return dict(candidate_inputs_sha256=layout['candidate']['inputs_sha256'], source=graph['source'],
        role_partition_sha256=ct.digest(graph['partition']), metadata_sha256=ct.digest(graph['metadata_pins']),
        original_acceptance_sha256=layout['candidate']['acceptance']['sha256'],
        verified_content_sha256=ct.digest({n: x['sha256'] for n, x in graph['partition']['entries'].items()}))


def review_transfer_provenance(api, layout, *, deadline):
    finite_phase(deadline)
    require(enabled(layout) and layout['purpose'] == 'pilot' and layout['candidate']['launch_review'] is None
        and layout['candidate']['reviewed_provenance'] is None, 'new unapproved prospective role review required')
    binding = {k: layout['candidate']['acceptance'][k] for k in ('path', 'sha256')}
    report = api.acceptance_manifest(binding)
    api.validate_original_workflow(report['study_layout'])
    reviewed = api.validate_transfer_candidate(report['study_layout'])
    facts = copy.deepcopy(reviewed['build_review'])
    # Same historical content, custody, quality, requests and original GO validators;
    # only prospective compatibility approval is intentionally still pending.
    api.validate_acceptance_evidence(binding, report, facts, reviewed_candidate=reviewed)
    graph = derive(api, layout)
    require(facts['source'] == graph['source'], 'review actual source/derived role source differs')
    required = {n: x['sha256'] for n, x in graph['partition']['entries'].items()}
    ct.check_pins(required)
    finite_phase(deadline)
    return dict(schema=PROOF, layout_subject_sha256=layout_subject(layout), source=graph['source'],
        candidate_inputs_sha256=layout['candidate']['inputs_sha256'], metadata_pins=graph['metadata_pins'],
        partition=graph['partition'], identity_anchor=dict(
            schema='ferrule.storage-demand.reviewed-role-identity.v1', subject=identity_subject(layout, graph)),
        original_acceptance=dict(path=binding['path'], sha256=binding['sha256'],
            workflow_pins=report['study_layout']['artifact_pins'],
            main_go=report['study_layout']['candidate']['launch_review_sha256']),
        review_completed=api.maintained.ex.stamp(), authority='pending independent compatibility review and MAIN-GO')


def verify_live(api, layout, graph):
    for name, pin in graph['live_pins'].items():
        before = api.preparation_identity(name)
        require(ct.sha(name) == pin and api.preparation_identity(name) == before, 'live input content/identity changed: ' + name)
    binary = graph['descriptor']['inputs']['binary']
    api.immutable_input(binary)
    require(Path(binary).stat().st_mode & 0o111 and Path(binary).parent.stat().st_mode & 0o222 == 0,
        'live immutable executable required')
    api.validate_transfer_inputs(layout['input_contract'])
    api.validate_http_references(layout)


def verify_provenance(api, layout):
    require(enabled(layout), 'explicit prospective role candidate required')
    graph = derive(api, layout)
    reference = layout['candidate']['reviewed_provenance']
    require(isinstance(reference, dict) and set(reference) == {'path', 'sha256'}, 'reviewed provenance pending')
    proof, _ = document(api, reference['path'], reference['sha256'])
    require(proof.get('schema') == PROOF and proof.get('layout_subject_sha256') == layout_subject(layout)
        and proof.get('candidate_inputs_sha256') == layout['candidate']['inputs_sha256']
        and proof.get('source') == graph['source'] and proof.get('metadata_pins') == graph['metadata_pins']
        and proof.get('partition') == graph['partition']
        and proof.get('identity_anchor') == dict(schema='ferrule.storage-demand.reviewed-role-identity.v1',
            subject=identity_subject(layout, graph)), 'independent complete role/provenance proof differs')
    return graph, proof


def authority(api, layout):
    graph, proof = verify_provenance(api, layout)
    reference = layout['candidate']['reviewed_provenance']
    binding = layout['candidate']['acceptance']
    require(binding.get('workflow_compatibility') is not None, 'separate role compatibility review pending')
    api.validate_workflow_compatibility(binding, layout)
    facts = reviewed_facts(api, layout, graph, proof)
    subject = api.authorize_transfer_candidate(layout, facts)
    bound = dict(candidate=dict(binary=facts['binary'], binary_sha256=facts['binary_sha256'],
        runtime_pin_manifest=facts['runtime_pin_manifest'], artifact_pins=graph['request_pins']),
        uuid_inventory=layout['candidate']['uuid_inventory'], build_review=facts, approval_subject=subject,
        role_partition_sha256=facts['role_partition_sha256'])
    return graph, bound


def reviewed_facts(api, layout, graph, proof):
    reference = layout['candidate']['reviewed_provenance']
    for name, pin in graph['semantic_pins'].items():
        require(ct.sha(api.immutable_input(name)) == pin, 'frozen semantic source changed: ' + name)
    frozen = Path(graph['documents']['source_freeze']['frozen_source'])
    producer = api.review_initial_endpoint_producer((frozen / api.quality.INITIAL_ENDPOINT_SOURCE).read_text(), layout['compiled_identity'])
    facts = dict(source=graph['source'], binary=graph['descriptor']['inputs']['binary'],
        binary_sha256=layout['compiled_identity']['binary_sha256'],
        runtime_pin_manifest=graph['descriptor']['inputs']['runtime_pin_manifest'],
        artifact_pins=graph['metadata_pins'], initial_read_producer=producer,
        initial_read_observation_available=producer is not None,
        reviewed_provenance_sha256=reference['sha256'], role_partition_sha256=ct.digest(graph['partition']),
        reviewed_identity_sha256=ct.digest(proof['identity_anchor']))
    return facts


class Admission:
    """Process-local, single-trial capability; never deserialized from a receipt."""
    def __init__(self, token, api, path, layout, graph, bound, directory, trial, registered, deadline, started):
        require(token is TOKEN, 'admission must independently verify authority')
        self.api, self.path, self.directory, self.trial = api, Path(path).resolve(), Path(directory).resolve(), trial
        self.layout, self.graph, self.bound = copy.deepcopy(layout), graph, copy.deepcopy(bound)
        self.registered, self.deadline, self.started = registered, deadline, started
        self.layout_pin, self.registration_pin = ct.sha(self.path), ct.sha(self.directory / 'registration.json')
        self.process = api.maintained.pilot.launch_guard.identity(os.getpid())
        self.boot, self.closed, self.outer = api.maintained.boot_id(), False, None
        self.proof = copy.deepcopy(layout['candidate']['reviewed_provenance'])
        self._scope = (str(self.path), str(self.directory), self.trial, self.boot,
            ct.digest(self.layout), ct.digest(self.bound), ct.digest(self.graph['live_pins']), ct.digest(self.graph['descriptor']),
            ct.digest(self.registered), ct.digest(self.proof))
        self._deadline, self._outer = deadline, None
        self.registration_mode = stat.S_IMODE((self.directory / 'registration.json').stat().st_mode)
        self.seal_transition = None

    def check(self, selected=None):
        require(self._scope == (str(self.path), str(self.directory), self.trial, self.boot,
            ct.digest(self.layout), ct.digest(self.bound), ct.digest(self.graph['live_pins']), ct.digest(self.graph['descriptor']),
            ct.digest(self.registered), ct.digest(self.proof))
            and self.deadline == self._deadline and self.outer == self._outer, 'admission scope/clock reset')
        require(not self.closed and time.monotonic() < self.deadline
            and self.api.maintained.boot_id() == self.boot
            and self.api.maintained.pilot.launch_guard.identity(os.getpid()) == self.process,
            'admission expired/completed/process identity changed')
        require(selected is None or Path(selected).resolve() == self.path, 'admission layout crossbinding')
        self.api.maintained.require_admission_open(self.directory)
        require(not any((self.directory / self.trial / name).exists() for name in ('assessment.json', 'coordinator-failure.json', 'locks-released.json')),
            'admission trial completed; no reuse')
        require(ct.sha(self.path) == self.layout_pin, 'admission layout changed')
        # Registration content is normative. The explicit 0644 -> 0400 seal is
        # allowed; a new ctime is not misrepresented as a cached content proof.
        registration = self.directory / 'registration.json'
        mode = stat.S_IMODE(registration.stat().st_mode)
        require(mode == self.registration_mode or (self.registration_mode == 0o644 and mode == 0o400),
            'registration seal transition must be one-way 0644 to 0400')
        require(mode in (0o644, 0o400) and not registration.is_symlink()
            and ct.sha(registration) == self.registration_pin, 'registration content/mode changed')
        if mode != self.registration_mode:
            self.seal_transition = dict(before=0o644, after=0o400, content_sha256=self.registration_pin)
            self.registration_mode = mode

    def activate(self, active_started, outer):
        self.check()
        require(self.outer is None and self.started['monotonic'] <= active_started['monotonic'] <= time.monotonic()
            and outer == min(active_started['monotonic'] + 650,
            self.registered['pilot_deadline'] - 100), 'admission original active clock changed')
        self.outer = self._outer = outer
        self.deadline = self._deadline = outer

    def reference(self):
        return dict(schema='ferrule.storage-demand.role-admission-binding.v1', trial=self.trial,
            boot_id=self.boot, layout_sha256=self.layout_pin, registration_sha256=self.registration_pin,
            reviewed_provenance=self.proof, role_partition_sha256=self.bound['role_partition_sha256'], outer_deadline=self.outer)

    def before_spawn(self):
        self.check()
        candidate = self.layout['candidate']
        for reference in (self.proof, candidate['acceptance']['workflow_compatibility'],
                dict(path=candidate['launch_review'], sha256=candidate['launch_review_sha256'])):
            document(self.api, reference['path'], reference['sha256'])
        verify_live(self.api, self.layout, self.graph)
        self.check()

    def close(self):
        self.closed = True


def admit_transfer_launch(api, path, directory, trial, *, deadline, payload=None, started=None):
    finite_phase(deadline)
    api.maintained.require_admission_open(directory)
    layout = api.validate_transfer_layout(path, candidate_required=False)
    api.immutable_input(path)
    api.directory_check(Path(directory), transfer=True, purpose=layout['purpose'])
    graph, bound = authority(api, layout)
    verify_live(api, layout, graph)
    registered = api.historical.unseal(Path(directory) / 'registration.json',
        'ferrule.storage-demand.actual-http-transfer-registration.v1')
    require(trial in layout['trial_order'], 'admission unregistered trial')
    if payload is None:
        require(api.maintained.next_trial(directory, registered) == trial, 'admission trial order/reuse differs')
    else:
        attempt = ct.load(Path(directory) / trial / 'attempt.json')
        require(attempt['trial'] == trial and attempt['replay_permitted'] is False
            and attempt['outer_deadline'] == payload['outer_deadline'] ==
                min(attempt['admission']['monotonic'] + 650, registered['pilot_deadline'] - 100)
            and registered['pilot_deadline'] == payload['pilot_deadline'] + 100
            and attempt['registration_sha256'] == ct.sha(Path(directory) / 'registration.json'),
            'role worker/capture original attempt clock/registration changed')
        for prior in layout['trial_order'][:layout['trial_order'].index(trial)]:
            folder = Path(directory) / prior
            require(api.maintained.coordinator_completion(folder)
                and ct.load(folder / 'assessment.json')['assessment']['can_continue'] is True,
                'role worker/capture cannot bypass prior incomplete/failed trial')
    value = Admission(TOKEN, api, path, layout, graph, bound, directory, trial, registered,
        deadline, started or api.maintained.ex.stamp())
    api.transfer_registration(directory, path, selected_layout=layout, admission=value)
    value.check()
    if payload is not None:
        value.outer = value._outer = payload['outer_deadline']
        require(deadline <= min(payload['outer_deadline'], payload['pilot_deadline'])
            and payload['spec'].get('role_admission') == value.reference(), 'worker/capture role admission reset/crossbinding')
    return value


@contextmanager
def prepared_admission(api, directory, path, registered, trial, started, deadline):
    value, entered = None, False
    try:
        with api.maintained.ex.absolute_alarm(deadline):
            value = admit_transfer_launch(api, path, directory, trial, deadline=deadline, started=started)
            ended = api.maintained.ex.stamp()
            api.ct.write(Path(directory) / ('static-preparation-' + trial + '.json'), dict(
                schema='ferrule.storage-demand.role-static-preparation.v1', trial=trial, started=started,
                ended=ended, deadline=deadline, throughput_denominator=False, binding=value.reference(),
                control_input_policy=dict(registration_sha256=value.registration_pin,
                    initial_mode=value.registration_mode, permitted_seal=[0o644, 0o400],
                    freshness='content hash; mode transition is not a reused identity verification'),
                authority='process-local admission only; this receipt cannot authorize or refresh admission'))
            (Path(directory) / ('static-preparation-' + trial + '.json')).chmod(0o400)
        with api.transfer_adapter(path, admission=value):
            entered = True
            yield
    except (Exception, KeyboardInterrupt) as error:
        if entered:
            try:
                api.maintained.stop(directory, type(error).__name__ + ': ' + str(error))
            except (Exception, KeyboardInterrupt) as stop_error:
                error.add_note('STOP journal failure: ' + type(stop_error).__name__ + ': ' + str(stop_error))
        else:
            api.maintained.admission_refusal(directory, error, trial=trial, started=started, deadline=deadline,
                source_proof='role admission failed before yielding; no owner locks or active clock')
        raise
    finally:
        if value is not None:
            value.close()
