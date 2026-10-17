#!/usr/bin/env python3
"""Explicit V3 CPU plan/check and separately gated maintained finite-arm execution."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import io
import os
import subprocess
import tarfile
import tomllib
from types import SimpleNamespace
from contextlib import contextmanager, ExitStack
import json
from pathlib import Path
import re
import signal
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paired_read_v1 import plan_v3 as plan, protocol as historical, current_binding_v3 as binding
from paired_read_v1 import run_paired as planned, run_actual_v2 as maintained, analysis, quality
from runnerhelpers.evidence_v2 import ct, require, validate_schedule

ENTRY = Path(__file__).resolve()
NOTE = 'Separate renewed sole-owner authorization for independent corrective V3 paired study; no V2 resume, probes, retries or profiling'
ORIGINAL_WRITE = ct.write


def directory_check(directory, *, transfer=False, purpose='pilot'):
    directory = Path(directory)
    prefix = ('transfer-acceptance' if purpose == 'public_cli_acceptance' else 'transfer-pilot') if transfer else 'v3'
    require(directory.parent.resolve() == HERE and not directory.is_symlink()
            and re.fullmatch('artifacts-' + prefix + r'-[A-Za-z0-9_-]{1,60}', directory.name),
            'new artifacts-' + prefix + '-NAME directly in paired_read_v1 required; no historical V2 resume')
    return directory


def registration(directory):
    directory_check(directory)
    value = historical.unseal(Path(directory) / 'registration.json',
        'ferrule.storage-demand.actual-paired-registration.v3')
    execution = binding.validate_execution(Path(value['execution_path']))
    candidate = binding.validate_binding(Path(value['binding_path']))
    require(value['protocol_id'] == plan.ID and value['protocol_sha256'] == ct.sha(plan.PROTOCOL_PATH)
            and value['cases_sha256'] == ct.sha(plan.CASES_PATH), 'unknown V3 registration identity')
    require(value['boot_id'] == maintained.boot_id() and value['trial_order'] == execution['trial_order']
            and value['order'] == execution['order'] and value['limits'] == execution['limits'],
            'V3 boot/order/absolute limits changed; no clock reset')
    require(value['binding_sha256'] == ct.sha(value['binding_path'])
            and value['execution_sha256'] == ct.sha(value['execution_path'])
            and value['binding_path'] == str(binding.BINDING_PATH)
            and value['execution_path'] == str(binding.EXECUTION_PATH), 'V3 candidate/workflow changed')
    require(value['schedule'] == ct.load(HERE / 'requests.json')['schedule'], 'entire V3 12-request schedule changed')
    validate_schedule(value['schedule'])
    require(value['owner_authorization'] is True and value['owner_note'] == NOTE
            and value['eight_uuids'] == candidate['uuid_inventory']
            and value['selected_uuid'] == value['eight_uuids'][0], 'V3 authorization/inventory changed')
    maintained.pilot.validate_inventory(value['eight_uuids'], value['selected_uuid'])
    return value


def versioned_write(path, value):
    # Label newly produced output correctly; never rewrite an existing V2 receipt.
    if Path(path).name == 'paired-final.json' and value.get('schema') == 'ferrule.storage-demand.paired-read-actual.v2':
        directory_check(Path(path).parent)
        value = dict(value, schema='ferrule.storage-demand.paired-read-actual.v3', protocol_id=plan.ID,
            protocol_sha256=ct.sha(plan.PROTOCOL_PATH), cases_sha256=ct.sha(plan.CASES_PATH),
            historical_resume_permitted=False)
    return ORIGINAL_WRITE(path, value)


@contextmanager
def historical_adapter():
    """Versioned process-local specialization, no copied engine or lifecycle harness."""
    changes = ((historical, 'validate_prepared', binding.validate_execution),
        (historical, 'validate_candidate', binding.validate_binding),
        (historical, 'PROTOCOL_PATH', plan.PROTOCOL_PATH), (historical, 'CASE_PATH', plan.CASES_PATH),
        (historical, 'case_for_mode', binding.case_for_mode),
        (planned, 'ENTRY', ENTRY), (planned, 'worker_contract', maintained.contract),
        (planned, 'read_ledger_assessment', maintained.ledger),
        (maintained, 'binding', binding), (maintained, 'registration', registration),
        (maintained, 'actual_adapter', adapter), (maintained, 'NOTE', NOTE), (ct, 'write', versioned_write))
    originals = [(module, name, getattr(module, name)) for module, name, value in changes]
    try:
        for module, name, value in changes:
            setattr(module, name, value)
        yield
    finally:
        for module, name, value in reversed(originals):
            setattr(module, name, value)


def start(directory, granted):
    require(granted is True, 'separate renewed sole-GPU-owner authorization required')
    directory = directory_check(directory)
    execution = binding.validate_execution()
    candidate = binding.validate_binding()
    directory.mkdir(exist_ok=False)
    now = time.monotonic()
    value = dict(schema='ferrule.storage-demand.actual-paired-registration.v3', protocol_id=plan.ID,
        protocol_sha256=ct.sha(plan.PROTOCOL_PATH), cases_sha256=ct.sha(plan.CASES_PATH),
        started=maintained.ex.stamp(), pilot_deadline=now + execution['limits']['pilot_seconds'],
        boot_id=maintained.boot_id(), order=execution['order'], trial_order=execution['trial_order'],
        limits=execution['limits'], schedule=ct.load(HERE / 'requests.json')['schedule'],
        binding_path=str(binding.BINDING_PATH), binding_sha256=ct.sha(binding.BINDING_PATH),
        execution_path=str(binding.EXECUTION_PATH), execution_sha256=ct.sha(binding.EXECUTION_PATH),
        selected_uuid=candidate['uuid_inventory'][0], eight_uuids=candidate['uuid_inventory'],
        owner_authorization=True, owner_note=NOTE, scheduled_blocks=5, scheduled_starts=10,
        scheduled_requests=120, scheduled_measured_qa=50, retries=0, replacements=0,
        profiling_captures=0, first_arm_is_first_planned_arm=True, extra_pilot_lifecycles=0,
        historical_resume_permitted=False, physical_nas_backend_bytes=None)
    ct.write(directory / 'registration.json', dict(value, digest=ct.digest(value)))
    return dict(status='registered', protocol_id=plan.ID, directory=str(directory),
                first_arm=execution['trial_order'][0], deadline=value['pilot_deadline'])


def main(argv=None):
    ct.require_cpu_python()
    def cancelled(signum, frame):
        raise KeyboardInterrupt('owned V3 coordinator/worker cancellation')
    signal.signal(signal.SIGTERM, cancelled)
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == '--_http_capture':
        require(len(argv) == 4, 'sealed owned HTTP capture entry and original absolute deadline required')
        return maintained.ex.http_capture_main(Path(argv[1]), argv[2], float(argv[3]))
    if argv and argv[0] == '--_worker':
        require(len(argv) in (2, 4) and (len(argv) == 2 or argv[2] == '--layout'),
                'authenticated worker layout arguments required')
        if len(argv) == 4:
            raw = sys.stdin.read()
            payload = json.loads(raw)
            previous = sys.stdin
            try:
                sys.stdin = io.StringIO(raw)
                with transfer_adapter(Path(argv[3]), prepared=payload['spec'].get('static_preparation'),
                        deadline=min(payload['outer_deadline'], payload['pilot_deadline']), launch=payload):
                    return planned.worker(argv[1])
            finally:
                sys.stdin = previous
        with adapter():
            return planned.worker(argv[1])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'compatibility-plan', 'review', 'check', 'start', 'next', 'finalize'))
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--study', choices=('historical-v3', 'transfer-pilot-v1'), default='historical-v3')
    parser.add_argument('--layout', type=Path)
    parser.add_argument('--build-inputs', type=Path, help='immutable v2 candidate descriptor for a separately committed/compiled cut')
    parser.add_argument('--verified-artifact-identity-output', type=Path, help='review-only exclusive unapproved identity manifest, before MAIN-GO')
    parser.add_argument('--role-partition', action='store_true', help='prospective reviewed-provenance admission; separate human bridge and MAIN-GO required')
    parser.add_argument('--reviewed-provenance-output', type=Path, help='review-only exclusive pending provenance/role proof, never GO')
    parser.add_argument('--workflow-compatibility', type=Path, help='immutable separately reviewed prospective compatibility receipt; no implicit bridge')
    parser.add_argument('--acceptance-final', type=Path, help='sealed original acceptance FINAL for a prospective compatibility review plan')
    parser.add_argument('--purpose', choices=('pilot', 'public_cli_acceptance'), default='pilot')
    parser.add_argument('--sole-gpu-owner-granted', action='store_true')
    args = parser.parse_args(argv)
    if (args.layout is not None or args.study == 'transfer-pilot-v1'
            or args.acceptance_final is not None or args.workflow_compatibility is not None
            or args.mode == 'compatibility-plan' or args.verified_artifact_identity_output is not None
            or args.role_partition or args.reviewed_provenance_output is not None):
        return transfer_main(args)
    if args.mode == 'plan':
        value = dict(plan.plan(), versioned_execution_entry=str(ENTRY), binding_pending=not binding.BINDING_PATH.exists(),
                     integration_does_not_amend_preregistration=True)
    elif args.mode == 'review':
        raise ValueError('review requires --study transfer-pilot-v1')
    elif args.mode == 'check':
        binding.validate_execution()
        value = dict(status='checked', protocol_id=plan.ID, gpu_queries=0, server_runs=0)
    else:
        require(args.directory is not None, '--directory required for execution')
        directory = directory_check(args.directory)
        if args.mode == 'start':
            value = start(directory, args.sole_gpu_owner_granted)
        else:
            registration(directory)
            with adapter():
                value = maintained.execute_next(directory) if args.mode == 'next' else maintained.finalize(directory)
            if args.mode == 'next':
                print(json.dumps(dict(trial=value['trial'], can_continue=value['assessment']['can_continue'],
                    measured_qa=value['summary']['measured_qa'])), flush=True)
                return 0 if value['assessment']['can_continue'] else 1
    print(json.dumps(value, sort_keys=True), flush=True)
    return 0 if value['status'] not in ('stopped', 'incomplete') else 1




TRANSFER_COMMIT = 'd3d643a174268b083657d880ebce33392b16cde1'
TRANSFER_NOTE = 'Renewed sole-owner authorization for fixed HTTP transfer pilot; two pairs per schedule, no probes/retries/profiling'
PROOF_PATHS = (
    'crates/ferrule-cli/src/args.rs', 'crates/ferrule-cli/src/commands/serve.rs',
    'crates/ferrule-composition/src/model_factory/mod.rs',
    'crates/ferrule-composition/src/model_factory/expert_read_policy.rs',
    'crates/ferrule-composition/src/model_factory/mixed_serving.rs',
    'crates/ferrule-composition/src/model_factory/bf16_serving.rs',
    'crates/ferrule-model/src/decoder/runner.rs',
    'crates/ferrule-model/src/transformer/standard/cuda/hybrid.rs',
    'crates/ferrule-model/src/transformer/standard/cuda.rs',
    'crates/ferrule-model/src/transformer/standard/cuda/residency.rs',
    'crates/ferrule-model/src/transformer/standard/cuda/grouped_upload.rs',
    'crates/ferrule-backend/src/cuda/operators/numeric_fp8_group_sync.rs',
    'crates/ferrule-model/src/transformer/expert_reads.rs',
    'crates/ferrule-model/src/transformer/standard/cuda/invocation_trace.rs')


TRANSFER_SOURCE_SHA256 = {
    "crates/ferrule-backend/src/cuda/operators/numeric_fp8_group_sync.rs": "37abf2ad1a1907c7007ccfab66a9deadf33204a0cd2d8dd9533d318f42894dd1",
    "crates/ferrule-cli/src/args.rs": "36177fff729744d3feb6e4cd0736ef035e14d2f5f67f7c06050364a51df3f788",
    "crates/ferrule-cli/src/commands/serve.rs": "6858ebca10e7c7bceb1f6a67839769039dc5077fd3e2f6f454d9ddaae7444d86",
    "crates/ferrule-composition/src/model_factory/bf16_serving.rs": "3cb5b5c4a2282d0bfe95f150e47bbfd359937d50e9e4e169be42aae2346c279c",
    "crates/ferrule-composition/src/model_factory/expert_read_policy.rs": "8afc745af7db8218d31ebbf24f3dc3d262c01fa7ad01cafdcb74f0ef795859fa",
    "crates/ferrule-composition/src/model_factory/mixed_serving.rs": "2287edf8cadd528b0e59ed79a2503eb7afed691f62a3d1e6d2c479a74da4184a",
    "crates/ferrule-composition/src/model_factory/mod.rs": "2968b46125baf09af02867512c97177c1068329c98ffc672a32b18aae1d088a8",
    "crates/ferrule-model/src/decoder/runner.rs": "8e0dcf44a5973ec1a7a04d1a970c42f691eae1f1faaf478e8e4c6dc566728c8b",
    "crates/ferrule-model/src/transformer/standard/cuda.rs": "6fce9fd2e623cccce12ce629706559af68efb786df8a00ca0da357185a89c5a7",
    "crates/ferrule-model/src/transformer/standard/cuda/grouped_upload.rs": "09db08bc534429eb468fa98d023a862ec1f950e854abf9379727d20646fb91d5",
    "crates/ferrule-model/src/transformer/standard/cuda/hybrid.rs": "8e78055f3af8782c6c7b0eb6c2c1aa48fd41bed9ab1f6b4c40cceb42ecc255b0",
    "crates/ferrule-model/src/transformer/standard/cuda/residency.rs": "f8d9c21fa5c4228a1f98f7a81bb44397faabfefc5402f2beb3223c54f64adea5",
    "crates/ferrule-model/src/transformer/expert_reads.rs": "eaa2063cf8861c7917269efe13a4b0ec6cf3927fc46a3d5574d21f9717970372",
    "crates/ferrule-model/src/transformer/standard/cuda/invocation_trace.rs": "be51b69dac63ec4b086d2af9cd665c3f5604cfebf95f304a5347fbb83a69ffb6"
}

def transfer_workflow_paths(*, role_partition=False):
    from newversionedprotocol import protocol as worker_protocol
    paths = [ct.ROOT / p for p in worker_protocol.IMPORTS] + [
        ct.ROOT / 'scripts/gpu_idle_window_v2.py', ct.ROOT / 'scripts/run_pinned_jobs.py',
        ct.ROOT / 'target/validation/qwen35-35b-current/toolchain/env-cuda124.sh',
        Path(sys.executable), *HERE.parent.joinpath('runnerhelpers').glob('*.py'),
        *HERE.parent.joinpath('newversionedprotocol').glob('*.py'),
        *(HERE / name for name in ('run_v3.py', 'run_paired.py', 'run_actual_v2.py',
            'analysis.py', 'quality.py', 'protocol.py', 'test_http_transfer.py',
            'cases-v3.json', 'requests.json', 'oracles.json', 'plan_v3.py', 'current_binding_v2.py',
            'current_binding_v3.py', 'test_http_transfer_controls.py', 'test_deadline_phase.py', 'test_preparation_authority.py')),
        HERE.parent / 'checkpoint-metadata.json', Path('/usr/bin/env'), Path('/usr/bin/readelf'), Path('/usr/bin/git')]
    if role_partition:
        return paths + [HERE / 'test_transfer_admission.py']
    return [p for p in paths if p.name != 'transfer_admission.py']


def transfer_input_contract():
    oracles = ct.load(HERE / 'oracles.json')
    recorded = ct.load(CANDIDATE_ROOT / 'historical-reference-control.json')['pins']
    metadata = HERE.parent / 'checkpoint-metadata.json'
    return dict(schema='ferrule.storage-demand.http-transfer-model-inputs.v1',
        reference_pins=oracles['reference_pins'], model_metadata=recorded['model_metadata'],
        checkpoint_manifest=str(metadata), checkpoint_manifest_sha256=ct.sha(metadata),
        shard_stat_identities={row['path']: row['identity'] for row in ct.load(metadata)['shards']},
        checkpoint_content_sha256=None, identity_scope='stat device/inode/size/mtime/ctime; not checkpoint content hash')


def validate_transfer_inputs(contract):
    require(contract == transfer_input_contract(), 'reference/model/checkpoint-path contract changed')
    ct.check_pins(contract['reference_pins'])
    ct.check_pins(contract['model_metadata'])
    ct.check_pins({contract['checkpoint_manifest']: contract['checkpoint_manifest_sha256']})
    shards = contract['shard_stat_identities']
    model = Path('/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8')
    index = ct.load(model / 'model.safetensors.index.json')
    require(set(shards) == {str(model / name) for name in index['weight_map'].values()},
            'checkpoint manifest must cover exactly the indexed shards')
    for name, expected in shards.items():
        st = Path(name).stat()
        require(not Path(name).is_symlink() and [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns] == expected,
                'checkpoint shard stat identity changed: ' + name)
    return dict(input_contract_sha256=ct.digest(contract), checkpoint_content_sha256=None)


def transfer_plan(purpose='pilot', *, compiled_identity=None, http_reference_binding=None,
        role_partition=False, workflow_pins=None):
    require(workflow_pins is None or (role_partition and set(workflow_pins) ==
        {str(p.resolve()) for p in transfer_workflow_paths(role_partition=True)}), 'exact prospective workflow universe required')
    identity = validate_compiled_identity(compiled_identity or historical_compiled_identity())
    require(purpose in ('pilot', 'public_cli_acceptance'), 'unknown transfer purpose')
    cases = {}
    base = ct.load(plan.CASES_PATH)
    for schedule in ('blocking', 'lookahead'):
        for policy in analysis.POLICIES:
            case = {k: copy.deepcopy(base[schedule][k]) for k in ('model', 'topology',
                'argv_after_binary', 'environment', 'read_options', 'expected_scheduler',
                'limits', 'request_ceiling', 'cache_bytes_include_scratch', 'pagecache_policy')}
            case.update(schema='ferrule.storage-demand.http-transfer-case.v1',
                case_id='transfer-pilot-v1-' + schedule + '-' + policy,
                read_schedule=schedule, transfer_policy=policy)
            case['argv_after_binary'] += ['--expert-transfer-policy', policy]
            case['read_options']['schedule'] = schedule
            cases[schedule + '-' + policy] = case
    order = analysis.transfer_order() if purpose == 'pilot' else [dict(block='A' + str(i),
        read_schedule=schedule, modes=[schedule + '-' + policy for policy in analysis.POLICIES])
        for i, schedule in enumerate(('blocking', 'lookahead'), 1)]
    starts = 8 if purpose == 'pilot' else 4
    value = dict(schema=analysis.TRANSFER_LAYOUT, controls_revision=2, purpose=purpose, status='planned_not_authorized',
        study='transfer-pilot-v1', compiled_identity=identity, production_commit=identity['production_commit'],
        production_source_pins={str(ct.ROOT / p): pin for p, pin in identity['source_proof_pins'].items()},
        cases=cases, cases_sha256=ct.sha(plan.CASES_PATH),
        schedule=ct.load(HERE / 'requests.json')['schedule'], schedule_sha256=ct.sha(HERE / 'requests.json'),
        oracles_sha256=ct.sha(HERE / 'oracles.json'), input_contract=transfer_input_contract(),
        http_expectation_contract='bound-http-reference-v1', http_reference_binding=http_reference_binding,
        upload_producer_contract=quality.TRANSFER_UPLOAD_PRODUCER, order=order,
        trial_order=analysis.transfer_trials(order), limits=base['blocking']['limits'],
        scheduled_blocks=len(order), scheduled_pairs_per_schedule=2 if purpose == 'pilot' else 1, scheduled_starts=starts,
        scheduled_requests=starts * 12, scheduled_measured_qa=starts * 5, expected_measured_tokens=None,
        scheduled_measured_qa_per_policy=starts * 5 // 2, expected_measured_tokens_per_policy=None,
        acceptance_starts_disclosed=4, acceptance_requests_disclosed=48, filesystem_cold_claim=False,
        required_cuda_toolkit='12.4', required_cuda_architecture='sm_86',
        candidate=None, artifact_pins=(workflow_pins if workflow_pins is not None else
            ct.inventory(transfer_workflow_paths(role_partition=role_partition))),
        gpu_authorized=False, gpu_queries=0, gpu_locks=0, gpu_runs=0,
        http_eos='ignore_eos omitted, stop enabled; reviewed exact IDs/text/finish_reason/usage references pending',
        diagnostic_eos='stop_at_eos=false; compatibility requires exact observed IDs/text/finish/usage',
        dispatch_evidence='positive cache uploads plus frozen-source policy dispatch; direct grouped-call counter unavailable',
        routes_observed=False, logits_observed=False, scientific_acceptance=False)
    if http_reference_binding is not None:
        from runnerhelpers.http_expectations import normalize_http_expectation
        context = validate_http_references(value)
        measured = sum(normalize_http_expectation(row, contract=value['http_expectation_contract'],
            reference_context=context)['actual_output_tokens'] for row in value['schedule'] if row['measured'])
        value['expected_measured_tokens'] = starts * measured
        value['expected_measured_tokens_per_policy'] = starts * measured // 2
    return value


def validate_transfer_layout(path, *, candidate_required=True):
    value = ct.load(path)
    require(isinstance(value, dict) and value.get('controls_revision') == 2, 'new versioned transfer controls required')
    expected = transfer_plan(value.get('purpose'), compiled_identity=value.get('compiled_identity'),
        http_reference_binding=value.get('http_reference_binding'),
        role_partition=role_admission_enabled(value),
        workflow_pins=value['artifact_pins'] if role_admission_enabled(value) else None)
    require(isinstance(value, dict) and set(value) == set(expected)
        and all(value[k] == v for k, v in expected.items()
            if k not in ('candidate', 'http_reference_binding')),
        'transfer layout/workload/source pins/workflow changed; old source seals cannot qualify')
    validate_schedule(value['schedule'])
    if value['http_reference_binding'] is not None:
        validate_http_references(value)
    if candidate_required:
        validate_transfer_candidate(value)
        immutable_input(path)
    return value


def http_reference_context(layout):
    from runnerhelpers.http_expectations import BOUND_HTTP_REFERENCE, HISTORICAL_LENGTH
    contract, binding = layout['http_expectation_contract'], layout['http_reference_binding']
    if contract == HISTORICAL_LENGTH:
        require(binding is None, 'historical HTTP expectations cannot bind new reference data')
        return None
    require(contract == BOUND_HTTP_REFERENCE and isinstance(binding, dict), 'unknown/unbound HTTP reference contract')
    source = immutable_input(binding['source'])
    require(ct.sha(source) == binding['source_sha256'], 'reviewed HTTP reference file changed')
    return dict(source=str(source), source_sha256=ct.sha(source), document=ct.load(source), reviewed_binding=binding)


def validate_http_references(layout):
    from runnerhelpers.http_expectations import normalize_http_expectation
    context = http_reference_context(layout)
    for row in layout['schedule']:
        normalize_http_expectation(row, contract=layout['http_expectation_contract'], reference_context=context)
    if context is not None:
        require(set(context['document']['references']) == {r['name'] for r in layout['schedule']},
                'all and only the fixed scheduled HTTP references required')
    return context


def review_transfer_source(texts):
    """Lexical guard of the hash-bound cut, not independent runtime dispatch telemetry."""
    checks = {
        PROOF_PATHS[0]: (r'pub\(crate\) expert_transfer_policy: Option<ServeExpertTransferPolicy>',
            r'ServeExpertTransferPolicy::GroupedSynchronous => Self::GroupedSynchronous',
            r'ServeExpertTransferPolicy::Synchronous => Self::Synchronous'),
        PROOF_PATHS[1]: (r'expert_transfer_policy: args\.expert_transfer_policy\.map\(Into::into\)',),
        PROOF_PATHS[2]: (r'ServingExpertTransferPolicy::Synchronous => Self::Synchronous',
            r'ServingExpertTransferPolicy::GroupedSynchronous => Self::GroupedSynchronous'),
        PROOF_PATHS[3]: (r'serving\.expert_read_pipeline\.is_none\(\)',
            r'Some\(ferrule_model::transformer::NumericFp8Precision::Bf16ModelMixedF32Accumulate\)'),
        PROOF_PATHS[4]: (r'serving\.model_precision = Some\(ModelPrecision::Bf16\)',
            r'storage: NumericStoragePolicy::E4M3FnBf16Scales',
            r'precision: ferrule_model::transformer::NumericFp8Precision::Bf16ModelMixedF32Accumulate'),
        PROOF_PATHS[5]: (r'runner\.set_expert_transfer_policy\([\s\S]*?request\s*\.execution\s*\.expert_transfer_policy\s*\.unwrap_or_default\(\)\s*\.into\(\),\s*\)\?;[\s\S]*?runner\.expert_transfer_policy\(\)[\s\S]*?compose_resident_engine\(\s*runner,',),
        PROOF_PATHS[6]: (r'self\.forward_executor\.set_expert_transfer_policy\(policy\)',),
        PROOF_PATHS[7]: (r'ExpertTransferControl::set_expert_transfer_policy\(\s*&mut module\.operators,\s*policy,',),
        PROOF_PATHS[8]: (r'else if this\.grouped_upload\.is_some\(\) \{\s*this\.grouped_metadata_upload\(metadata, provider, input\.shape\(\)\.rows\(\)\)\?;',),
        PROOF_PATHS[9]: (r'ExpertTransferPolicy::GroupedSynchronous => \{[\s\S]*?self\.grouped_upload = Some\(grouped_upload::State::default\(\)\);',),
        PROOF_PATHS[10]: (r'let artifacts = self\.ops\.upload_validated_numeric_fp8_group_sync\(payloads\)\?;\s*self\.bindings\s*\.publish_numeric_expert\(&key, metadata, artifacts\)\?;',),
        PROOF_PATHS[11]: (r'pub fn upload_validated_numeric_fp8_group_sync\(',
            r'checked_sync\(\|\| Ok\(\(\)\), \|\| false, \|\| cu\(stream\.synchronize\(\)\)\)'),
    }
    checks[PROOF_PATHS[12]] = tuple(
        r'#\[serde\(skip_serializing_if = "zero_host_charge"\)\]\s*pub ' + field + ': u64'
        for field in ('upload_reserved_packed_bytes', 'pinned_upload_bytes', 'upload_ticket_metadata_bytes',
            'upload_policy_headroom_bytes', 'unknown_upload_bytes'))
    checks[PROOF_PATHS[13]] = (r'expert_cache_counters_overflowed = cache\.counters_overflowed',
        r'expert_pending_upload_bytes = cache\.pending_upload_bytes',
        r'expert_unknown_quarantine_bytes = cache\.unknown_quarantine_bytes')
    for name, patterns in checks.items():
        require(name in texts and all(re.search(pattern, texts[name]) for pattern in patterns),
                'frozen source transfer dispatch binding differs: ' + name)
    return dict(scoped_files=list(PROOF_PATHS), lexical_dispatch_binding=True,
        direct_backend_call_counter_available=False, native_fence_counts=None)


CANDIDATE_ROOT = ct.ROOT / 'target/validation/http-transfer-eb3c536-cpu-freeze-20261017-01'
CANDIDATE_INPUTS = CANDIDATE_ROOT / 'CANDIDATE-INPUTS.json'
ARCHIVE_COMMIT = 'eb3c536f0abf9098246c988d07307111701fed8e'
CLI_SHA256 = 'd3660990c2c971e9c1c4e2a244206aa97d6a8f69c194912bdf6df69509a3df1c'
ARCHIVE_SHA256 = 'c6fb5708b6fe52f883a75ccfe92be1eef557902089aedbaba67adc1eede8ff0d'
RUNTIME_SHA256 = '344994c62c74bb7dc10125fa42f03e62e4164d71db2b0927454b2f64b092c316'
COMPILED_SCOPE = ('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', 'README.md', '.cargo', 'crates')


def historical_compiled_identity():
    return dict(schema='ferrule.storage-demand.http-transfer-compiled-identity.v1',
        production_commit=TRANSFER_COMMIT, archive_commit=ARCHIVE_COMMIT, compiled_scope=list(COMPILED_SCOPE),
        candidate_inputs=str(CANDIDATE_INPUTS), candidate_inputs_sha256=ct.sha(CANDIDATE_INPUTS),
        binary_sha256=CLI_SHA256, archive_sha256=ARCHIVE_SHA256, runtime_sha256=RUNTIME_SHA256,
        source_proof_pins=dict(TRANSFER_SOURCE_SHA256), initial_endpoint_producer=None)


def validate_compiled_identity(identity):
    require(isinstance(identity, dict) and set(identity) == {'schema', 'production_commit', 'archive_commit',
        'compiled_scope', 'candidate_inputs', 'candidate_inputs_sha256', 'binary_sha256', 'archive_sha256',
        'runtime_sha256', 'source_proof_pins', 'initial_endpoint_producer'}
        and identity['schema'] == 'ferrule.storage-demand.http-transfer-compiled-identity.v1'
        and identity['compiled_scope'] == list(COMPILED_SCOPE), 'explicit full-scope compiled identity required')
    for key in ('production_commit', 'archive_commit', 'candidate_inputs_sha256', 'binary_sha256', 'archive_sha256', 'runtime_sha256'):
        length = 40 if key.endswith('commit') else 64
        require(isinstance(identity[key], str) and re.fullmatch('[0-9a-f]{' + str(length) + '}', identity[key]),
            'invalid compiled identity: ' + key)
    path = Path(identity['candidate_inputs'])
    require(path.is_absolute() and path.name == 'CANDIDATE-INPUTS.json' and path.is_relative_to(ct.ROOT / 'target/validation')
        and '..' not in path.parts and set(identity['source_proof_pins']) == set(PROOF_PATHS)
        and all(isinstance(pin, str) and re.fullmatch('[0-9a-f]{64}', pin) for pin in identity['source_proof_pins'].values()),
        'compiled candidate path/source proof coverage differs')
    producer = identity['initial_endpoint_producer']
    require(producer is None or ct.digest(producer) == ct.digest(dict(contract=quality.INITIAL_ENDPOINT_PRODUCER,
        source=quality.INITIAL_ENDPOINT_SOURCE, source_sha256=identity['source_proof_pins'][quality.INITIAL_ENDPOINT_SOURCE])),
        'unknown/unbound initial endpoint compiled producer')
    return copy.deepcopy(identity)


def compiled_identity_for_inputs(path):
    path = immutable_input(path)
    inputs = ct.load(path)
    if inputs.get('schema') == 'ferrule.storage-demand.http-transfer-cpu-candidate-inputs.v1':
        require(path == CANDIDATE_INPUTS, 'unrecognized historical descriptor; use versioned compiled identity')
        return historical_compiled_identity()
    require(inputs.get('schema') == 'ferrule.storage-demand.http-transfer-cpu-candidate-inputs.v2'
        and isinstance(inputs.get('compiled_identity'), dict), 'versioned candidate descriptor identity required')
    declared = inputs['compiled_identity']
    require('candidate_inputs' not in declared and 'candidate_inputs_sha256' not in declared,
        'descriptor cannot self-declare its own path/hash')
    return validate_compiled_identity(dict(declared, candidate_inputs=str(path), candidate_inputs_sha256=ct.sha(path)))


def review_initial_endpoint_producer(text, identity):
    declaration = identity['initial_endpoint_producer']
    if declaration is None:
        return None
    require(hashlib.sha256(text.encode()).hexdigest() == declaration['source_sha256'], 'compiled initial producer source pin differs')
    builder = text.split('pub(super) fn build_numeric_bf16(', 1)[-1].split('// Metadata-only builds', 1)[0]
    markers = ['runner.set_expert_transfer_policy(', 'log_initial_expert_read_endpoint(',
        'NUMERIC_BF16_TEST_RUNNER.with(', 'ferrule_runtime::engine::compose_resident_engine(']
    positions = [builder.find(marker) for marker in markers]
    require(all(position >= 0 for position in positions) and positions == sorted(positions)
        and builder.count('log_initial_expert_read_endpoint(') == 1
        and '|| runner.physical_resource_snapshot()' in builder[positions[1]:positions[2]]
        and ')?;' in builder[positions[1]:positions[2]]
        and 'runner.expert_transfer_policy()' in builder[positions[1]:positions[2]]
        and 'if request.execution.expert_read_pipeline.is_some()' in builder[:positions[1]],
        'compiled initial producer setter/getter/configuration ordering differs')
    required = ['event = "expert_read_initial_endpoint"', 'producer_schema_version = 1u64',
        'source_contract = "resident_model_runner.physical_resource_snapshot.expert_read_pipeline"',
        'endpoint = "after_owner_configuration_before_prediction"', 'process_id = std::process::id()',
        'serde_json::to_string(&snapshot.expert_read_pipeline)', 'tracing::info!(target: "ferrule_performance"',
        'snapshot: impl FnOnce() -> ferrule_model::resource_observability::PhysicalResourceObservation',
        'if !tracing::enabled!(target: "ferrule_performance", tracing::Level::INFO)',
        'let snapshot = snapshot();', 'serde_json::to_string(&OwnerBinding {',
        'owner_binding = owner_binding.as_str()', 'expert_read_pipeline_initial = initial.as_str()']
    require(all(marker in text for marker in required), 'compiled initial producer typed source contract differs')
    return copy.deepcopy(declaration)


def immutable_input(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file()
        and not any(p.is_symlink() for p in (path, *path.parents))
        and path.stat().st_mode & 0o222 == 0, 'immutable canonical input required: ' + str(path))
    return path


def archive_inventory(stream):
    pins = {}
    with tarfile.open(fileobj=stream, mode='r|') as archive:
        for member in archive:
            name = Path(member.name)
            require(not name.is_absolute() and '..' not in name.parts and str(name) == member.name,
                    'unsafe archived source member')
            if member.isdir():
                continue
            require(member.isfile() and member.name not in pins, 'linked/duplicate archived source member')
            with archive.extractfile(member) as data:
                pins[member.name] = hashlib.file_digest(data, 'sha256').hexdigest()
    return pins


def committed_inventory(commit, scope):
    """Read committed blobs for comparison; do not create a source archive."""
    listing = subprocess.run(['/usr/bin/git', '--no-pager', 'ls-tree', '-r', '-z', '--full-tree',
        commit, '--', *scope], cwd=ct.ROOT, capture_output=True, timeout=30, check=True).stdout
    rows = []
    for item in listing.split(b'\0'):
        if not item:
            continue
        metadata, name = item.split(b'\t', 1)
        mode, kind, oid = metadata.split()
        require(mode in (b'100644', b'100755') and kind == b'blob', 'linked/non-file committed source member')
        rows.append((name.decode(), oid))
    require(rows, 'committed source scope missing')
    result = subprocess.run(['/usr/bin/git', '--no-pager', 'cat-file', '--batch'], cwd=ct.ROOT,
        input=b''.join(oid + b'\n' for _, oid in rows), capture_output=True, timeout=60, check=True).stdout
    require(len(result) <= 128 << 20, 'committed source comparison exceeds bound')
    pins, offset = {}, 0
    for name, oid in rows:
        end = result.index(b'\n', offset)
        actual_oid, kind, size = result[offset:end].split()
        require(actual_oid == oid and kind == b'blob' and size.isdigit(), 'committed source blob identity differs')
        offset, length = end + 1, int(size)
        raw = result[offset:offset+length]
        require(len(raw) == length and result[offset+length:offset+length+1] == b'\n', 'truncated committed source blob')
        pins[name] = hashlib.sha256(raw).hexdigest()
        offset += length + 1
    require(offset == len(result), 'unexpected committed source comparison suffix')
    return pins


def compiled_source_review(freeze, identity=None):
    from runnerhelpers.build_receipts_v1 import inventory_tree
    identity = validate_compiled_identity(identity or historical_compiled_identity())
    root = Path(identity['candidate_inputs']).parent
    frozen = Path(freeze['frozen_source'])
    require(frozen == root / 'frozen-source' and not frozen.is_symlink()
        and frozen.stat().st_mode & 0o222 == 0 and freeze.get('commit') == identity['archive_commit']
        and freeze.get('schema') == 'ferrule.expert-read.committed-source-freeze.v1'
        and freeze.get('immutable_readonly_inputs') is True, 'honest immutable committed source freeze required')
    pins = freeze['pins_relative_to_frozen_source']
    require(freeze['file_count'] == len(pins) and set(pins) ==
        {str(Path(p).relative_to(frozen)) for p in inventory_tree(frozen)}, 'full frozen source coverage differs')
    for name, pin in pins.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'unsafe frozen source member')
        ct.check_pins({str(immutable_input(frozen / name)): pin})
    originals = {str(ct.ROOT / n): h for n, h in pins.items()}
    require(freeze['original_source_pins'] == originals and freeze['source_pins_digest'] == ct.digest(originals)
        and freeze['source_pins_digest_relative'] == ct.digest(pins), 'full source digest relation differs')
    archive = immutable_input(freeze['archive'])
    require(archive == root / 'source.tar' and ct.sha(archive) == freeze['archive_sha256']
        == identity['archive_sha256'], 'frozen archive identity differs')
    with archive.open('rb') as stream:
        require(archive_inventory(stream) == pins, 'archive/frozen full source bytes differ')
    argv = freeze['archive_argv']
    require(argv[:1] in (['git'], ['/usr/bin/git'])
        and argv[1:6] == ['--no-pager', 'archive', '--format=tar', identity['archive_commit'], '--']
        and set(COMPILED_SCOPE) <= set(argv[6:])
        and all(isinstance(name, str) and name and not name.startswith(':')
            and not Path(name).is_absolute() and '..' not in Path(name).parts for name in argv[6:]),
        'honest source archive command/scope binding differs')
    require(committed_inventory(identity['archive_commit'], argv[6:]) == pins,
        'full archived source is not byte-identical to its declared archive commit')
    expected = committed_inventory(identity['production_commit'], COMPILED_SCOPE)
    actual = {n: h for n, h in pins.items() if any(n == p or n.startswith(p + '/') for p in COMPILED_SCOPE)}
    require(actual == expected and actual, 'full compiled production/archive scope is not byte-identical')
    require({n: pins[n] for n in PROOF_PATHS} == identity['source_proof_pins'], 'compiled source proof pins differ')
    profile = tomllib.loads((frozen / 'Cargo.toml').read_text())['profile']['release']
    require(all(profile.get(k) == v for k, v in {'opt-level': 3, 'lto': 'fat', 'codegen-units': 1,
        'panic': 'unwind', 'strip': 'symbols'}.items()), 'compiled release profile differs')
    review_transfer_source({n: (frozen / n).read_text() for n in PROOF_PATHS})
    return dict(production_commit=identity['production_commit'], archive_commit=identity['archive_commit'],
        compiled_scope=list(COMPILED_SCOPE), compiled_files=len(actual), compiled_scope_sha256=ct.digest(actual),
        full_source_sha256=ct.digest(pins), archive_sha256=identity['archive_sha256'], freeze_relabelled=False)


def review_transfer_build(descriptor=CANDIDATE_INPUTS, *, compiled_identity=None):
    """Read actual immutable build evidence. No build/help/model/GPU execution."""
    from runnerhelpers import library_maps_v1 as maps
    from runnerhelpers.build_receipts_v1 import inventory_tree, RUST_COMMIT, CUTLASS_COMMIT
    identity = compiled_identity_for_inputs(descriptor)
    require(compiled_identity is None or identity == validate_compiled_identity(compiled_identity),
        'candidate descriptor/approved compiled identity differs')
    root = Path(descriptor).parent
    inputs = ct.load(immutable_input(descriptor))
    require(inputs.get('schema') in ('ferrule.storage-demand.http-transfer-cpu-candidate-inputs.v1',
            'ferrule.storage-demand.http-transfer-cpu-candidate-inputs.v2')
        and inputs['commit'] == identity['archive_commit'] and inputs['production_commit'] == identity['production_commit']
        and set(inputs['inputs']) == set(inputs['sha256']), 'candidate descriptor source/coverage differs')
    evidence_pins = {str(Path(descriptor)): ct.sha(descriptor)}
    def remember(path):
        path = immutable_input(path)
        evidence_pins[str(path)] = ct.sha(path)
        return path
    for key, path in inputs['inputs'].items():
        require(Path(path).is_relative_to(root) and ct.sha(remember(path)) == inputs['sha256'][key],
                'candidate descriptor input differs: ' + key)
    def load(key):
        return ct.load(inputs['inputs'][key])
    freeze = load('source_freeze')
    source = compiled_source_review(freeze, identity)
    remember(freeze['archive'])
    binary = inputs['inputs']['binary']
    require(inputs['sha256']['binary'] == identity['binary_sha256'] and inputs['sha256']['runtime_pin_manifest'] == identity['runtime_sha256'],
            'unreviewed CLI/runtime bytes')
    ct.validate_staged_binary(binary, identity['binary_sha256'])
    with Path(binary).open('rb') as stream:
        require(stream.read(4) == b'\x7fELF' and Path(binary).stat().st_mode & 0o111 != 0, 'real executable ELF required')
    frozen = Path(freeze['frozen_source'])
    def command(name):
        spec = ct.load(remember(root / (name + '-command.json')))
        receipt = ct.load(remember(root / (name + '-receipt.json')))
        supervisor = ct.load(remember(root / (name + '-supervisor.json')))
        require(all(receipt.get(k) == v for k, v in spec.items()) and receipt.get('returncode') == 0
            and receipt.get('timed_out') is False and maintained.ex.natural_descendant_drain(supervisor)
            and spec.get('gpu_hidden') is True and spec.get('profiling_enabled') is False
            and type(spec.get('timeout_seconds')) is int and 0 < spec['timeout_seconds'] <= 1800
            and spec['environment']['CUDA_VISIBLE_DEVICES'] == ''
            and spec['environment']['NVIDIA_VISIBLE_DEVICES'] == 'none'
            and receipt['supervisor_sha256'] == ct.sha(frozen / 'scripts/test_vllm_qwen35.py')
            and receipt['subreaper_sha256'] == ct.sha(frozen / 'scripts/_owned_subreaper.py'),
            'recorded command/build custody failed: ' + name)
        stdout, stderr = (remember(root / (name + '.' + suffix)) for suffix in ('stdout', 'stderr'))
        require(receipt['stdout_sha256'] == ct.sha(stdout) and receipt['stderr_sha256'] == ct.sha(stderr),
                'recorded command output hash differs: ' + name)
        return spec, stdout.read_text(), stderr.read_text()
    build, stdout, _ = command('build-cli')
    expected_argv = ['/root/.cargo/bin/cargo', 'build', '--locked', '--offline', '--release',
        '--no-default-features', '-p', 'ferrule-cli', '--features', 'cuda,nvtx-profile', '--message-format=json']
    env = build['environment']
    require(build['argv'] == expected_argv and build['cwd'] == str(frozen)
        and env == ct.load(remember(root / 'build-environment.json'))
        and env['CARGO_TARGET_DIR'] == str(root / 'cargo-target')
        and not any(k in env for k in ('RUSTFLAGS', 'RUSTC_WRAPPER', 'LD_PRELOAD'))
        and env['FERRULE_CUDA_TEST_ORACLE'] == '0' and env['FERRULE_CUDA_ARCH'] == 'sm_86'
        and env['FERRULE_PROFILE_NVTX'] == '0' and env['CARGO_INCREMENTAL'] == '0'
        and env['RUSTUP_TOOLCHAIN'] == 'nightly-2026-08-01' and env['RUSTUP_AUTO_INSTALL'] == '0',
        'exact isolated CUDA/NVTX build environment differs')
    rows = [json.loads(line) for line in stdout.splitlines() if line.startswith('{')]
    require(rows[-1] == dict(reason='build-finished', success=True), 'actual Cargo completion missing')
    artifact = load('artifact')
    cli = ct.release_artifact(rows)
    require(cli == artifact['compiled_artifact'] and cli['features'] == ['cuda', 'nvtx-profile']
        and cli['fresh'] is False and cli['manifest_path'] == str(frozen / 'crates/ferrule-cli/Cargo.toml')
        and cli['target']['src_path'] == str(frozen / 'crates/ferrule-cli/src/main.rs')
        and cli['executable'] == str(root / 'cargo-target/release/ferrule')
        and ct.sha(cli['executable']) == artifact['sha256'] == identity['binary_sha256']
        and artifact['executable'] == binary and Path(binary).stat().st_size == artifact['bytes']
        and not Path(binary).samefile(cli['executable']) and artifact['source_digest'] == freeze['source_pins_digest'],
        'actual Cargo/staged CLI/source relation differs')
    workspace = [r for r in rows if r.get('reason') == 'compiler-artifact'
        and Path(r['manifest_path']).is_relative_to(frozen)]
    require(workspace and all('cpu-reference' not in r['features'] for r in workspace), 'CPU reference build forbidden')
    tools, toolpins = load('toolchain'), load('toolchain_pins')
    require(all(v.get('returncode') == 0 and v.get('stderr') == '' for v in tools.values())
        and 'commit-hash: ' + RUST_COMMIT + '\n' in tools['rustc']['stdout']
        and 'V12.4.99' in tools['nvcc']['stdout'] and tools['cutlass']['stdout'] == CUTLASS_COMMIT + '\n'
        and toolpins['toolkit'] == '12.4.99' and toolpins['architecture'] == 'sm_86', 'compiler/toolkit identity differs')
    for key in ('toolchain_pins', 'external_inputs', 'dependencies'):
        ct.check_pins(load(key)['pins'])
    for path in (build['argv'][0], *(env[k] for k in ('RUSTC', 'RUSTDOC', 'NVCC', 'CC', 'CXX'))):
        require(path in toolpins['pins'], 'actual compiler executable unpinned: ' + path)
    for path, row in toolpins['linker_search_inputs'].items():
        require(str(Path(path).resolve()) == row['resolved'] and ct.sha(path) == row['sha256'], 'linker input drift')
    external = load('external_inputs')['pins']
    for base in (env['FERRULE_CUTLASS_DIR'], env['FERRULE_NVTX_INCLUDE_DIR']):
        require(inventory_tree(base) <= external.keys(), 'full CUTLASS/NVTX input coverage missing')
    post = ct.load(remember(root / 'external-input-postverify.json'))
    require(post['reference_manifest_sha256'] == inputs['sha256']['external_inputs']
        and post['verified_count'] == len(external) and post['all_matched'] is True, 'external post-build relation differs')
    dependencies = load('dependencies')
    selected = {r['manifest_path']: r['package_id'] for r in rows if r.get('reason') == 'compiler-artifact'
        and r['package_id'].startswith('registry+')}
    require(selected == dependencies['packages'] and selected and
        set(dependencies['pins']) == set().union(*(inventory_tree(Path(p).parent) for p in selected)),
        'actual full compiled dependency coverage differs')
    native = load('native_archives')
    actual_archives = {str(p) for r in rows if r.get('reason') == 'build-script-executed'
        for p in Path(r['out_dir']).glob('*.a')}
    require(actual_archives == set(native['original_pins']) and
        sorted(native['original_pins'].values()) == sorted(native['immutable_copies'].values()),
        'actual native archive/copy coverage differs')
    ct.check_pins(native['original_pins']); ct.check_pins(native['immutable_copies'])
    for path in native['immutable_copies']:
        remember(path)
    cubins = load('native_cubins')
    require(cubins['cli_sha256'] == identity['binary_sha256'] and cubins['compiled_target'] == 'sm_86'
        and cubins['embedded_elf_count'] == len(cubins['pins']) == 2 and cubins['embedded_ptx_count'] == 0,
        'embedded CUDA architecture/count relation differs')
    ct.check_pins(cubins['pins'])
    for path in cubins['pins']:
        remember(path)
    for name in ('cubin-cli-list', 'cubin-cli-extract', 'ptx-cli-list'):
        command(name)
    help_spec, help_text, _ = command('cli-serve-help')
    require(help_spec['argv'] == [binary, 'serve', '--help'] and '--expert-transfer-policy' in help_text
        and 'grouped-synchronous' in help_text and 'synchronous' in help_text, 'actual public CLI policy help missing')
    for name in ('cli-version', 'cli-help'):
        spec, _, _ = command(name)
        require(spec['argv'][0] == binary, 'CPU CLI probe bound to foreign executable')
    executed = load('executed_helpers')
    ct.check_pins(executed['pins'])
    copies = ct.load(remember(root / 'executed-helper-copy-pins.json'))
    ct.check_pins(copies['pins'])
    require(copies['original_execution_receipt'] == inputs['inputs']['executed_helpers']
        and copies['prepare_launch_main_executed'] is False
        and all(executed['pins'].get(path) == copies['pins'].get(str(root / 'executed-helpers' / Path(path).name))
            for path in executed['executed_build_helpers'])
        and executed['supervisor_module'] == str(frozen / 'scripts/test_vllm_qwen35.py')
        and executed['module_global_injection'] == dict(OUT=str(root), FROZEN=str(frozen), COMMIT=identity['archive_commit'])
        and executed['old_prepare_or_qualify_called'] is False, 'executed build-helper custody copies differ')
    for path in copies['pins']:
        remember(path)
    controls = load('control_helpers')
    require(controls['commit'] == identity['archive_commit'], 'build-era control identity differs')
    with remember(controls['archive']).open('rb') as stream:
        require(ct.sha(controls['archive']) == controls['archive_sha256'] and
            archive_inventory(stream) == controls['archived_relative_pins'], 'build-era controls archive differs')
    require(all(controls['pins'].get(str(ct.ROOT / name)) == pin
        for name, pin in controls['archived_relative_pins'].items()), 'historical control archive/pin relation differs')
    require(committed_inventory(identity['archive_commit'], controls['archived_relative_pins']) == controls['archived_relative_pins'],
        'historical controls are not exact archived commit bytes')
    # Historical controls remain evidence, not a demand to rewrite their pins to current Python.
    runtime = load('runtime_pin_manifest')
    require(runtime['ld_library_path'] == env['LD_LIBRARY_PATH'], 'build/runtime loader environment differs')
    policy = maps.role_policy(binary, identity['binary_sha256'], runtime, 'cli')
    producer = review_initial_endpoint_producer((frozen / quality.INITIAL_ENDPOINT_SOURCE).read_text(), identity)
    return dict(schema='ferrule.storage-demand.http-transfer-build-review.v2', source=source, compiled_identity=identity,
        binary=binary, binary_sha256=identity['binary_sha256'], runtime_pin_manifest=inputs['inputs']['runtime_pin_manifest'],
        runtime_policy=policy, artifact_pins=evidence_pins, candidate_inputs_sha256=ct.sha(descriptor),
        gpu_queries=0, build_verified=True, quality_accepted=False,
        initial_read_producer=producer, initial_read_observation_available=producer is not None)


def approval_subject(layout, facts):
    candidate = {k: v for k, v in layout['candidate'].items() if k not in ('launch_review', 'launch_review_sha256')}
    subject = dict(purpose=layout['purpose'], layout_sha256=ct.digest(dict(layout, candidate=candidate)),
        workflow_sha256=ct.digest(layout['artifact_pins']), build_evidence_sha256=ct.digest(facts['artifact_pins']),
        source=facts['source'], cli_sha256=facts['binary_sha256'],
        runtime_sha256=ct.sha(facts['runtime_pin_manifest']), input_contract_sha256=ct.digest(layout['input_contract']))
    compatibility = (candidate.get('acceptance') or {}).get('workflow_compatibility')
    if compatibility is not None:
        subject['workflow_compatibility_sha256'] = compatibility['sha256']
    anchor = layout['candidate'].get('verified_artifact_identity')
    if anchor is not None:
        subject['verified_artifact_identity_sha256'] = anchor['sha256']
    if role_admission_enabled(layout):
        subject.update(reviewed_provenance_sha256=facts['reviewed_provenance_sha256'],
            role_partition_sha256=facts['role_partition_sha256'],
            reviewed_identity_sha256=facts['reviewed_identity_sha256'])
    return subject


def role_admission_enabled(layout):
    return (layout.get('candidate') or {}).get('schema') == 'ferrule.storage-demand.http-transfer-candidate-binding.v4'


def transfer_admission_api():
    from runnerhelpers import transfer_admission
    return transfer_admission


def review_transfer_provenance(layout, *, deadline):
    return transfer_admission_api().review_transfer_provenance(sys.modules[__name__], layout, deadline=deadline)


def transfer_candidate_descriptor(layout):
    candidate = layout.get('candidate')
    fields = {'schema', 'inputs', 'inputs_sha256', 'launch_review',
        'launch_review_sha256', 'uuid_inventory', 'acceptance'}
    require(layout.get('purpose') in ('pilot', 'public_cli_acceptance') and isinstance(candidate, dict)
        and ((candidate.get('schema') == 'ferrule.storage-demand.http-transfer-candidate-binding.v2'
              and set(candidate) == fields)
            or (candidate.get('schema') == 'ferrule.storage-demand.http-transfer-candidate-binding.v3'
              and set(candidate) == fields | {'verified_artifact_identity'})
            or (candidate.get('schema') == 'ferrule.storage-demand.http-transfer-candidate-binding.v4'
              and set(candidate) == fields | {'reviewed_provenance'})),
        'candidate pending: versioned actual CLI/build/source/runtime binding required')
    if 'verified_artifact_identity' in candidate:
        reference = candidate['verified_artifact_identity']
        require(isinstance(reference, dict) and set(reference) == {'path', 'sha256'}
            and isinstance(reference['path'], str) and Path(reference['path']).is_absolute()
            and re.fullmatch('[0-9a-f]{64}', reference['sha256'] or ''), 'explicit verified artifact identity reference required')
    identity = validate_compiled_identity(layout['compiled_identity'])
    require(candidate['inputs'] == identity['candidate_inputs']
        and candidate['inputs_sha256'] == identity['candidate_inputs_sha256'] == ct.sha(immutable_input(candidate['inputs']))
        and compiled_identity_for_inputs(candidate['inputs']) == identity, 'real candidate input descriptor changed')
    inputs = ct.load(candidate['inputs'])
    required = {'binary', 'source_freeze', 'runtime_pin_manifest', 'build_receipt', 'build_supervisor',
        'artifact', 'toolchain', 'toolchain_pins', 'external_inputs', 'dependencies', 'native_archives',
        'native_cubins', 'executed_helpers', 'control_helpers', 'cpu_tests'}
    require(set(inputs['inputs']) == set(inputs['sha256']) == required
        and inputs['commit'] == identity['archive_commit']
        and inputs['production_commit'] == identity['production_commit'], 'candidate descriptor manifest coverage differs')
    root = Path(candidate['inputs']).parent
    for key, name in inputs['inputs'].items():
        require(Path(name).is_relative_to(root), 'candidate manifest escaped approved directory')
        immutable_input(name)
        if key != 'binary':
            require(ct.sha(name) == inputs['sha256'][key], 'candidate manifest hash differs: ' + key)
    require(inputs['sha256']['binary'] == identity['binary_sha256']
        and inputs['sha256']['runtime_pin_manifest'] == identity['runtime_sha256'], 'approved CLI/runtime identity differs')
    maintained.pilot.validate_inventory(candidate['uuid_inventory'], candidate['uuid_inventory'][0])
    return identity, inputs


def authorize_transfer_candidate(layout, facts):
    """Independent purpose-scoped MAIN-GO enforcement, never receipt authority."""
    from runnerhelpers.http_expectations import BOUND_HTTP_REFERENCE
    require(layout['http_expectation_contract'] == BOUND_HTTP_REFERENCE,
        'prospective transfer requires the explicit reviewed EOS-aware HTTP reference contract')
    require(facts['initial_read_observation_available'] is True,
        'GPU admission blocked: frozen CLI has no observed initial read endpoint; no synthetic baseline/probe')
    validate_http_references(layout)
    candidate, subject = layout['candidate'], approval_subject(layout, facts)
    require(isinstance(candidate['launch_review'], str)
        and re.fullmatch('[0-9a-f]{64}', candidate['launch_review_sha256'] or ''),
        'separate main purpose/layout/workflow/evidence-bound GO pending')
    review = ct.load(immutable_input(candidate['launch_review']))
    require(ct.sha(candidate['launch_review']) == candidate['launch_review_sha256']
        and review.get('schema') == ('ferrule.storage-demand.http-transfer-launch-review.v3'
            if role_admission_enabled(layout) else 'ferrule.storage-demand.http-transfer-launch-review.v2')
        and review.get('decision') == 'GO' and review.get('subject') == subject,
        'separate main purpose/layout/workflow/evidence-bound GO pending')
    return subject


def transfer_candidate_authority(layout, *, historical_workflow=False, approval_required=True):
    """Derive authority/complete coverage from real hash-bound manifests, not cache.

    Only large leaf content checks are reusable. Small manifests, purpose GO,
    immutable CLI identity and the complete manifest graph are always derived.
    """
    identity, inputs = transfer_candidate_descriptor(layout)
    descriptor, candidate = layout['candidate']['inputs'], layout['candidate']
    root = Path(descriptor).parent
    documents = {key: ct.load(name) for key, name in inputs['inputs'].items() if key != 'binary'}
    artifacts = {descriptor: candidate['inputs_sha256']}
    required, trees, globs, acceptance_roots = {}, {}, {}, []
    def merge(target, pins):
        require(isinstance(pins, dict) and bool(pins), 'empty required manifest pin set')
        for name, pin in pins.items():
            require(Path(name).is_absolute() and re.fullmatch('[0-9a-f]{64}', pin or '')
                and (name not in target or target[name] == pin), 'conflicting/malformed manifest pin: ' + name)
            target[name] = pin
    def remember(name, pin=None):
        name = str(name)
        if pin is None:
            immutable_input(name)
            pin = ct.sha(name)
        merge(artifacts, {name: pin})
    for key, name in inputs['inputs'].items():
        remember(name, inputs['sha256'][key])
    freeze = documents['source_freeze']
    frozen, relative = Path(freeze['frozen_source']), freeze['pins_relative_to_frozen_source']
    require(frozen == root / 'frozen-source' and not frozen.is_symlink() and frozen.stat().st_mode & 0o222 == 0
        and freeze['schema'] == 'ferrule.expert-read.committed-source-freeze.v1'
        and freeze['commit'] == identity['archive_commit'] and freeze['immutable_readonly_inputs'] is True
        and relative and freeze['file_count'] == len(relative), 'approved source manifest differs')
    for name in relative:
        require(name and not Path(name).is_absolute() and '..' not in Path(name).parts
            and str(Path(name)) == name, 'unsafe approved source member')
    source_pins = {str(frozen / name): pin for name, pin in relative.items()}
    originals = {str(ct.ROOT / name): pin for name, pin in relative.items()}
    require(originals == freeze['original_source_pins'] and ct.digest(originals) == freeze['source_pins_digest']
        and ct.digest(relative) == freeze['source_pins_digest_relative']
        and {name: relative[name] for name in PROOF_PATHS} == identity['source_proof_pins'], 'approved full source digest differs')
    actual = {name: pin for name, pin in relative.items()
        if any(name == scope or name.startswith(scope + '/') for scope in COMPILED_SCOPE)}
    source = dict(production_commit=identity['production_commit'], archive_commit=identity['archive_commit'],
        compiled_scope=list(COMPILED_SCOPE), compiled_files=len(actual), compiled_scope_sha256=ct.digest(actual),
        full_source_sha256=ct.digest(relative), archive_sha256=identity['archive_sha256'], freeze_relabelled=False)
    require(freeze['archive'] == str(root / 'source.tar')
        and freeze['archive_sha256'] == identity['archive_sha256'], 'approved source archive binding differs')
    remember(freeze['archive'], identity['archive_sha256'])
    merge(required, source_pins)
    trees[str(frozen)] = sorted(source_pins)
    binary = inputs['inputs']['binary']
    artifact = documents['artifact']
    require(artifact['executable'] == binary and artifact['sha256'] == identity['binary_sha256']
        and artifact['source_digest'] == freeze['source_pins_digest']
        and artifact['compiled_artifact']['executable'] == str(root / 'cargo-target/release/ferrule'),
        'actual approved Cargo/staged CLI relation differs')
    ct.validate_staged_binary(binary, identity['binary_sha256'])
    with immutable_input(binary).open('rb') as stream:
        require(stream.read(4) == b'\x7fELF' and Path(binary).stat().st_mode & 0o111,
            'approved immutable CLI ELF required')
    merge(required, {artifact['compiled_artifact']['executable']: identity['binary_sha256']})
    for name in ('build-cli', 'cubin-cli-list', 'cubin-cli-extract', 'ptx-cli-list',
                 'cli-serve-help', 'cli-version', 'cli-help'):
        for suffix in ('-command.json', '-receipt.json', '-supervisor.json', '.stdout', '.stderr'):
            remember(root / (name + suffix))
    for name in ('build-environment.json', 'external-input-postverify.json', 'executed-helper-copy-pins.json'):
        remember(root / name)
    environment = ct.load(root / 'build-environment.json')
    for key in ('toolchain_pins', 'external_inputs', 'dependencies', 'executed_helpers', 'native_cubins', 'runtime_pin_manifest'):
        merge(required, documents[key]['pins'])
    for name, row in documents['toolchain_pins']['linker_search_inputs'].items():
        require(str(Path(name).resolve()) == row['resolved'], 'approved linker resolution changed')
        merge(required, {name: row['sha256']})
    for key in ('original_pins', 'immutable_copies'):
        merge(required, documents['native_archives'][key])
    copies = ct.load(root / 'executed-helper-copy-pins.json')['pins']
    merge(required, copies)
    for pins in (documents['native_archives']['immutable_copies'], documents['native_cubins']['pins'], copies):
        for name, pin in pins.items():
            remember(name, pin)
    controls = documents['control_helpers']
    remember(controls['archive'], controls['archive_sha256'])
    for base in (environment['FERRULE_CUTLASS_DIR'], environment['FERRULE_NVTX_INCLUDE_DIR']):
        prefix = base.rstrip('/') + '/'
        trees[base] = sorted(name for name in documents['external_inputs']['pins'] if name.startswith(prefix))
    for manifest in documents['dependencies']['packages']:
        base = str(Path(manifest).parent)
        prefix = base.rstrip('/') + '/'
        trees[base] = sorted(name for name in documents['dependencies']['pins'] if name.startswith(prefix))
    rows = [json.loads(line) for line in (root / 'build-cli.stdout').read_text().splitlines() if line.startswith('{')]
    for row in rows:
        if row.get('reason') == 'build-script-executed':
            base = row['out_dir']
            globs[base] = sorted(name for name in documents['native_archives']['original_pins'] if str(Path(name).parent) == base)
    producer = review_initial_endpoint_producer((frozen / quality.INITIAL_ENDPOINT_SOURCE).read_text(), identity)
    facts = dict(schema='ferrule.storage-demand.http-transfer-build-review.v2', source=source, compiled_identity=identity,
        binary=binary, binary_sha256=identity['binary_sha256'], runtime_pin_manifest=inputs['inputs']['runtime_pin_manifest'],
        artifact_pins=artifacts, candidate_inputs_sha256=candidate['inputs_sha256'], gpu_queries=0,
        initial_read_producer=producer, initial_read_observation_available=producer is not None)
    for pins in (layout['input_contract']['reference_pins'], layout['input_contract']['model_metadata'],
                 {layout['input_contract']['checkpoint_manifest']: layout['input_contract']['checkpoint_manifest_sha256']}):
        merge(artifacts, pins)
    if layout['http_reference_binding'] is not None:
        merge(artifacts, {layout['http_reference_binding']['source']: layout['http_reference_binding']['source_sha256']})
    if layout['purpose'] == 'pilot':
        binding = candidate['acceptance']
        report = acceptance_manifest(binding)
        accepted = report['study_layout']
        check_acceptance_layout(binding, layout, accepted)
        original = transfer_candidate_authority(accepted, historical_workflow=binding.get('workflow_compatibility') is not None)
        require(original['bound_candidate']['build_review']['source'] == source, 'accepted compiled source differs')
        merge(required, original['required_pins'])
        trees.update(original['required_trees'])
        globs.update(original['required_globs'])
        merge(artifacts, report['artifact_pins'])
        remember(binding['path'], binding['sha256'])
        acceptance_root = str(Path(binding['path']).parent)
        trees[acceptance_root] = sorted(report['artifact_pins'])
        acceptance_roots.append(acceptance_root)
        if binding.get('workflow_compatibility') is not None:
            bridge = binding['workflow_compatibility']
            remember(bridge['path'], bridge['sha256'])
    else:
        require(candidate['acceptance'] is None, 'acceptance may not inherit a prior quality PASS')
    subject = authorize_transfer_candidate(layout, facts) if approval_required else approval_subject(layout, facts)
    if approval_required:
        remember(candidate['launch_review'], candidate['launch_review_sha256'])
    merge(required, artifacts)
    if historical_workflow:
        validate_original_workflow(layout)
        merge(required, {name: pin for name, pin in layout['artifact_pins'].items()
            if name not in {str(ct.ROOT / relative) for relative in COMPATIBILITY_FILES}})
    else:
        merge(required, layout['artifact_pins'])
    bound = dict(candidate=dict(binary=binary, binary_sha256=identity['binary_sha256'],
        runtime_pin_manifest=facts['runtime_pin_manifest'], artifact_pins=artifacts),
        uuid_inventory=candidate['uuid_inventory'], build_review=facts, approval_subject=subject)
    authority = dict(bound_candidate=bound, required_pins=required, required_trees=trees, required_globs=globs,
        acceptance_roots=acceptance_roots, identity_anchor=None)
    reference = candidate.get('verified_artifact_identity')
    if reference is not None:
        # Loading is permitted only after the independently reconstructed MAIN-GO
        # subject has bound this reference. Issuance never uses a preparation file.
        require(approval_required, 'identity anchor reuse requires actual MAIN-GO')
        authority['identity_anchor'] = load_verified_artifact_identity(layout, authority)
        merge(required, {reference['path']: reference['sha256']})
    authority['always_hash'] = set(layout['artifact_pins']) | set(layout['input_contract']['reference_pins'])
    authority['always_hash'].update(layout['input_contract']['model_metadata'])
    authority['always_hash'].add(layout['input_contract']['checkpoint_manifest'])
    if layout['http_reference_binding'] is not None:
        authority['always_hash'].add(layout['http_reference_binding']['source'])
    if approval_required:
        authority['always_hash'].add(candidate['launch_review'])
    if reference is not None:
        authority['always_hash'].add(reference['path'])
    return authority


def coverage_directories(authority):
    directories = set(authority['required_globs'])
    for root, members in authority['required_trees'].items():
        base = Path(root)
        directories.add(root)
        seen = set()
        for name in members:
            parent = Path(name).parent
            require(parent == base or parent.is_relative_to(base), 'coverage member escaped root')
            while parent != base and parent not in seen:
                seen.add(parent)
                directories.add(str(parent))
                parent = parent.parent
    return directories


def coverage_tree(root, authority):
    from runnerhelpers.build_receipts_v1 import inventory_tree
    found = inventory_tree(root)
    if root in authority['acceptance_roots']:
        found = {name for name in found if Path(name).name not in ('paired-final.json', 'study-coordinator.lock')}
    return sorted(found)


def verified_identity_subject(layout, authority):
    normalized = copy.deepcopy(layout)
    normalized['candidate'].pop('verified_artifact_identity', None)
    normalized['candidate'].update(schema='ferrule.storage-demand.http-transfer-candidate-binding.v2',
        launch_review=None, launch_review_sha256=None)
    excluded = {layout['candidate']['launch_review']}
    reference = layout['candidate'].get('verified_artifact_identity')
    if reference is not None:
        excluded.add(reference['path'])
    pins = {name: pin for name, pin in authority['required_pins'].items() if name not in excluded}
    subject = dict(purpose=layout['purpose'], layout_sha256=ct.digest(normalized),
        candidate_inputs_sha256=layout['candidate']['inputs_sha256'], required_pins_sha256=ct.digest(pins),
        required_trees_sha256=ct.digest(authority['required_trees']),
        required_globs_sha256=ct.digest(authority['required_globs']))
    return subject, pins


def hash_verified_identity(name, expected):
    require(signal.getitimer(signal.ITIMER_REAL)[0] > 0, 'content hashing requires a finite phase deadline')
    before = preparation_identity(name)
    actual = ct.sha(name)
    require(preparation_identity(name) == before, 'artifact identity changed while verifying bytes: ' + name)
    require(actual == expected, 'artifact content hash differs: ' + name)
    return dict(sha256=actual, identity=before)


def load_verified_artifact_identity(layout, authority):
    """Called after actual MAIN-GO enforcement, never from a receipt reference."""
    candidate = layout['candidate']
    reference = candidate['verified_artifact_identity']
    def verified_document(name, expected):
        path = immutable_input(name)
        before = preparation_identity(path)
        raw = path.read_bytes()
        require(preparation_identity(path) == before and hashlib.sha256(raw).hexdigest() == expected,
            'MAIN-GO-bound identity document hash/identity differs: ' + name)
        return json.loads(raw)
    go = verified_document(candidate['launch_review'], candidate['launch_review_sha256'])
    require(go.get('schema') == 'ferrule.storage-demand.http-transfer-launch-review.v2'
        and go.get('decision') == 'GO'
        and go.get('subject') == authority['bound_candidate']['approval_subject']
        and go['subject'].get('verified_artifact_identity_sha256') == reference['sha256'],
        'identity anchor is not bound by the actual MAIN-GO subject')
    scopes = (*authority['required_trees'], *authority['required_globs'])
    require(not any(Path(name).is_relative_to(root) for name in (reference['path'], layout['candidate']['launch_review'])
        for root in scopes), 'identity/MAIN-GO outputs must be outside immutable content scopes')
    value = verified_document(reference['path'], reference['sha256'])
    subject, pins = verified_identity_subject(layout, authority)
    require(value.get('schema') == 'ferrule.storage-demand.verified-artifact-identity.v1'
        and value.get('subject') == subject
        and {name: row['sha256'] for name, row in value['verified_inputs'].items()} == pins,
        'MAIN-GO-bound identity anchor subject/complete coverage differs')
    require(set(value['verified_directories']) <= coverage_directories(authority),
        'identity anchor directory scope differs')
    for name, row in value['verified_inputs'].items():
        identity = row['identity']
        require(isinstance(identity, list) and len(identity) == 9 and identity[0] == str(Path(name).resolve())
            and all(type(part) is int for part in identity[1:]), 'malformed issued artifact identity')
    for identity in value['verified_directories'].values():
        require(isinstance(identity, list) and len(identity) == 9 and identity[3] & 0o222 == 0,
            'identity anchor may bind only immutable content-tree directories')
    return value


def issue_verified_artifact_identity(layout, output):
    """Unapproved CPU issuance, only after full semantic AND actual byte checks."""
    require(isinstance(layout.get('candidate'), dict) and 'verified_artifact_identity' not in layout['candidate']
        and layout['candidate']['launch_review'] is None and layout['candidate']['launch_review_sha256'] is None,
        'issue identities before prospective MAIN-GO; never refresh an approved anchor')
    started = maintained.ex.stamp()
    with maintained.ex.absolute_alarm(started['monotonic'] + STATIC_PREPARATION_SECONDS):
        reviewed = validate_transfer_candidate(layout, approval_required=False)
        authority = transfer_candidate_authority(layout, approval_required=False)
        require(reviewed['candidate'] == authority['bound_candidate']['candidate']
            and reviewed['approval_subject'] == authority['bound_candidate']['approval_subject'],
            'full content review/identity issuance authority differs')
        output = Path(output).absolute()
        require(output.resolve() == output and not output.exists()
            and not any(output.is_relative_to(root) for root in (
                *authority['required_trees'], *authority['required_globs'])),
            'identity issuance must be exclusive and outside immutable content scopes')
        subject, pins = verified_identity_subject(layout, authority)
        directories = {name: preparation_identity(name) for name in coverage_directories(authority)
            if Path(name).stat().st_mode & 0o222 == 0}
        verified = {name: hash_verified_identity(name, pin) for name, pin in pins.items()}
        for root, members in authority['required_trees'].items():
            require(coverage_tree(root, authority) == members, 'identity issuance content inventory differs')
        for root, members in authority['required_globs'].items():
            require(sorted(str(p) for p in Path(root).glob('*.a')) == members, 'identity issuance native inventory differs')
        for name, row in verified.items():
            require(preparation_identity(name) == row['identity'], 'artifact changed during identity issuance: ' + name)
        for name, identity in directories.items():
            require(preparation_identity(name) == identity, 'immutable directory changed during identity issuance')
        value = dict(schema='ferrule.storage-demand.verified-artifact-identity.v1', subject=subject,
            started=started, ended=maintained.ex.stamp(), verified_inputs=verified,
            verified_directories=directories, checkpoint_content_sha256=None,
            status='full_content_verified_pending_MAIN_GO')
        ct.write(output, value)
        output.chmod(0o400)
        reference = dict(path=str(output), sha256=ct.sha(output))
    return dict(status='verified_artifact_identity_pending_MAIN_GO', verified_artifact_identity=reference,
        required_candidate_schema='ferrule.storage-demand.http-transfer-candidate-binding.v3')


def validate_prepared_coverage(value, authority, admission_pins):
    """Exact required universes are independently derived, never receipt-supplied."""
    require(value['admission_pins'] == admission_pins, 'static admission pin coverage differs')
    required = dict(authority['required_pins'], **admission_pins)
    require({name: row['sha256'] for name, row in value['verified_inputs'].items()} == required,
        'static preparation complete required pin coverage differs')
    require(value['verified_trees'] == authority['required_trees']
        and value['verified_globs'] == authority['required_globs'], 'static preparation complete scope coverage differs')
    require(set(value['verified_directories']) == coverage_directories(authority),
        'static preparation complete directory coverage differs')
    require(value['bound_candidate'] == authority['bound_candidate']
        and value['approval_subject'] == authority['bound_candidate']['approval_subject'], 'static candidate authority changed')
    anchor = authority.get('identity_anchor')
    issued = anchor['verified_inputs'] if anchor is not None else {}
    always_hash = set(authority.get('always_hash', ())) | set(admission_pins)
    for name, row in value['verified_inputs'].items():
        current = preparation_identity(name)
        baseline = issued.get(name)
        anchored = (baseline is not None and baseline['sha256'] == required[name]
            and baseline['identity'] == current and current[3] & 0o222 == 0)
        # A new receipt's current stat cannot establish a hash-verification event.
        if name in always_hash or not anchored:
            checked = hash_verified_identity(name, required[name])
            current = checked['identity']  # Local proof only; never refresh the anchor.
        require(current == row['identity'], 'static preparation input changed: ' + name)
    if anchor is not None:
        for name, identity in anchor['verified_directories'].items():
            require(preparation_identity(name) == identity, 'approved immutable directory identity changed: ' + name)
    for root, members in authority['required_trees'].items():
        require(coverage_tree(root, authority) == members, 'static preparation inventory changed: ' + root)
    for root, members in authority['required_globs'].items():
        require(sorted(str(p) for p in Path(root).glob('*.a')) == members, 'static native archive inventory changed')
    for name, identity in value['verified_directories'].items():
        require(preparation_identity(name) == identity, 'static preparation directory changed: ' + name)


def validate_transfer_candidate(layout, *, approval_required=True, freshness=None, admission_pins=None):
    if role_admission_enabled(layout):
        require(approval_required and freshness is None, 'role admission never trusts a legacy preparation receipt; review provenance separately')
        api = transfer_admission_api()
        graph, bound = api.authority(sys.modules[__name__], layout)
        api.verify_live(sys.modules[__name__], layout, graph)
        return bound
    if freshness is not None:
        require(approval_required is True, 'prepared admission cannot bypass MAIN-GO')
        authority = transfer_candidate_authority(layout)
        validate_prepared_coverage(freshness, authority, admission_pins)
        return authority['bound_candidate']
    identity, inputs = transfer_candidate_descriptor(layout)
    candidate = layout['candidate']
    facts = review_transfer_build(candidate['inputs'], compiled_identity=identity)
    facts['artifact_pins'].update(layout['input_contract']['reference_pins'])
    facts['artifact_pins'].update(layout['input_contract']['model_metadata'])
    facts['artifact_pins'][layout['input_contract']['checkpoint_manifest']] = layout['input_contract']['checkpoint_manifest_sha256']
    if layout['http_reference_binding'] is not None:
        facts['artifact_pins'][layout['http_reference_binding']['source']] = layout['http_reference_binding']['source_sha256']
    maintained.pilot.validate_inventory(candidate['uuid_inventory'], candidate['uuid_inventory'][0])
    if layout['purpose'] == 'pilot':
        validate_public_cli_acceptance(candidate['acceptance'], layout, facts)
    else:
        require(candidate['acceptance'] is None, 'acceptance may not inherit a prior quality PASS')
    subject = approval_subject(layout, facts)
    if approval_required:
        subject = authorize_transfer_candidate(layout, facts)
        facts['artifact_pins'][candidate['launch_review']] = candidate['launch_review_sha256']
    return dict(candidate=dict(binary=facts['binary'], binary_sha256=facts['binary_sha256'],
        runtime_pin_manifest=facts['runtime_pin_manifest'], artifact_pins=facts['artifact_pins']),
        uuid_inventory=candidate['uuid_inventory'], build_review=facts, approval_subject=subject)


COMPATIBILITY_ACCEPTANCE_SHA256 = '6f544f9d25bc64b6c37ce0f0ffbe2f8e48ad332d80b38ea5b3221905dd530356'
COMPATIBILITY_FILES = {
    'papers/arxiv/experiments/storage_demand/paired_read_v1/run_v3.py': {
        'main', 'transfer_workflow_paths', 'transfer_plan', 'validate_transfer_layout', 'approval_subject',
        'validate_transfer_candidate', 'validate_public_cli_acceptance', 'transfer_registration',
        'validate_http_capture_entry', 'transfer_adapter', 'transfer_main'},
    'papers/arxiv/experiments/storage_demand/paired_read_v1/run_actual_v2.py': {'contract', 'execute_next', 'finalize', 'next_trial'},
    'papers/arxiv/experiments/storage_demand/runnerhelpers/execution_v2.py': {'http_capture_main'},
}
COMPATIBILITY_TESTS = ('test_deadline_phase.py', 'test_preparation_authority.py')
# Separate prospective scope; legacy compatibility subjects/allowlists stay strict.
ROLE_COMPATIBILITY_FILES = {
    name: functions | ({'transfer_candidate_descriptor', 'authorize_transfer_candidate', 'coverage_directories',
        'prospective_compatibility_subject', 'prospective_transfer_plan', 'workflow_compatibility_plan',
        'validate_workflow_compatibility'} if name.endswith('/run_v3.py') else
        {'http_capture_main'} if name.endswith('/execution_v2.py') else set())
    for name, functions in COMPATIBILITY_FILES.items()
}
ROLE_COMPATIBILITY_ADDED = (
    'papers/arxiv/experiments/storage_demand/runnerhelpers/transfer_admission.py',
    'papers/arxiv/experiments/storage_demand/paired_read_v1/test_transfer_admission.py',
)



def acceptance_manifest(binding):
    require(isinstance(binding, dict) and set(binding) in (
        {'path', 'sha256'}, {'path', 'sha256', 'workflow_compatibility'}),
        'pilot pending: actual four-cell public CLI acceptance required')
    path = immutable_input(binding['path'])
    require(ct.sha(path) == binding['sha256'], 'acceptance final hash differs')
    report = ct.load(path)
    accepted = report.get('study_layout', {})
    require(report.get('schema') == 'ferrule.storage-demand.http-transfer-acceptance.v2'
        and report.get('status') == 'complete' and accepted.get('purpose') == 'public_cli_acceptance'
        and report.get('actual_issued_lifecycles') == report.get('actual_completed_lifecycles') == 4
        and report.get('scheduled_starts') == 4 and report.get('scheduled_requests') == 48
        and report.get('retries') == report.get('replacements') == 0, 'four-cell acceptance incomplete/foreign')
    require(report.get('artifact_pins') and [r['trial'] for r in report['lifecycles']] == accepted['trial_order'],
        'acceptance manifest/ordered cells missing')
    return report


def original_workflow_bytes(name, pin, commit):
    path = Path(name)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() == pin:
        return raw
    relative = str(path.relative_to(ct.ROOT))
    require(relative in COMPATIBILITY_FILES, 'compatibility cannot replace historical quality controls: ' + relative)
    row = subprocess.run(['/usr/bin/git', '--no-pager', 'show', commit + ':' + relative],
        cwd=ct.ROOT, capture_output=True, check=True, timeout=10)
    require(hashlib.sha256(row.stdout).hexdigest() == pin, 'original workflow Git bytes/pin differ: ' + relative)
    return row.stdout


def validate_original_workflow(layout):
    commit = layout['compiled_identity']['archive_commit']
    for name, pin in layout['artifact_pins'].items():
        original_workflow_bytes(name, pin, commit)


def compatibility_quality_identity(layout):
    identity, descriptor = transfer_candidate_descriptor(layout)
    freeze = ct.load(descriptor['inputs']['source_freeze'])
    cargo = Path(freeze['frozen_source']) / 'Cargo.toml'
    require(ct.sha(cargo) == freeze['pins_relative_to_frozen_source']['Cargo.toml'], 'compiled profile source changed')
    fields = ('cases', 'schedule', 'limits', 'input_contract', 'http_expectation_contract',
        'http_reference_binding', 'upload_producer_contract', 'http_eos', 'diagnostic_eos',
        'required_cuda_toolkit', 'required_cuda_architecture', 'production_source_pins')
    return dict(**{key: layout[key] for key in fields}, compiled_identity=identity,
        full_compiled_source_pins=freeze['pins_relative_to_frozen_source'],
        release_profile=tomllib.loads(cargo.read_text())['profile']['release'],
        compiled_artifact=ct.load(descriptor['inputs']['artifact'])['compiled_artifact'],
        runtime_sha256=descriptor['sha256']['runtime_pin_manifest'])


def prospective_compatibility_subject(layout, binding, *, role_partition=False):
    require(layout['purpose'] == 'pilot' and binding['sha256'] == COMPATIBILITY_ACCEPTANCE_SHA256,
        'compatibility is prospective pilot-only for the exact reviewed acceptance final')
    report = acceptance_manifest({key: binding[key] for key in ('path', 'sha256')})
    accepted = report['study_layout']
    require(compatibility_quality_identity(accepted) == compatibility_quality_identity(layout),
        'compatibility quality-relevant CLI/runtime/profile/quota/input/reference/EOS/producer/config differs')
    old, new = accepted['artifact_pins'], layout['artifact_pins']
    scope = ROLE_COMPATIBILITY_FILES if role_partition else COMPATIBILITY_FILES
    allowed = {str(ct.ROOT / name) for name in scope}
    allowed_added = {str(HERE / name) for name in COMPATIBILITY_TESTS}
    if role_partition:
        allowed_added.update(str(ct.ROOT / name) for name in ROLE_COMPATIBILITY_ADDED)
    require(not (set(old) - set(new)) and set(new) - set(old) <= allowed_added,
        'compatibility workflow additions/removals exceed the reviewed orchestration/test scope')
    changed = {name for name in old if old[name] != new[name]}
    require(changed <= allowed, 'compatibility changed quality controls outside orchestration scope')
    semantic = {}
    for relative, mutable_functions in scope.items():
        name = str(ct.ROOT / relative)
        original = original_workflow_bytes(name, old[name], accepted['compiled_identity']['archive_commit'])
        current = Path(name).read_bytes()
        require(hashlib.sha256(current).hexdigest() == new[name], 'prospective workflow source pin changed')
        original_ast, current_ast = ast.parse(original), ast.parse(current)
        def globals_by_name(tree):
            return {ast.dump(target, include_attributes=False): ast.dump(node, include_attributes=False)
                for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign))
                for target in (node.targets if isinstance(node, ast.Assign) else [node.target])}
        old_globals, new_globals = globals_by_name(original_ast), globals_by_name(current_ast)
        require(all(new_globals.get(key) == value for key, value in old_globals.items()),
            'compatibility changed an original configuration/quota global: ' + relative)
        before = {node.name: ast.dump(node, include_attributes=False) for node in original_ast.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name not in mutable_functions}
        after = {node.name: ast.dump(node, include_attributes=False) for node in current_ast.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in before}
        require(before == after, 'compatibility changed protected quality/auth/deadline semantic source: ' + relative)
        semantic[name] = {key: hashlib.sha256(value.encode()).hexdigest() for key, value in before.items()}
        semantic[name]['original_configuration_globals'] = ct.digest(old_globals)
    for name in set(old) - changed:
        require(ct.sha(name) == old[name], 'unchanged original workflow pin changed')
    for name in allowed_added:
        require(name in new and ct.sha(name) == new[name], 'compatibility CPU test pins missing/changed')
    first = Path(binding['path']).parent / report['lifecycles'][0]['trial'] / 'worker-launch.json'
    require(ct.sha(first) == report['artifact_pins'][str(first)], 'original acceptance launch pin differs')
    launch = ct.load(first)
    original_layout = immutable_input(launch['execution_path'])
    require(ct.sha(original_layout) == launch['execution_sha256'] and ct.load(original_layout) == accepted,
        'original acceptance layout identity differs')
    original_go = accepted['candidate']
    require(ct.sha(immutable_input(original_go['launch_review'])) == original_go['launch_review_sha256'],
        'original acceptance MAIN-GO identity differs')
    prospective = copy.deepcopy(layout)
    prospective['candidate'].update(launch_review=None, launch_review_sha256=None)
    if role_partition:
        prospective['candidate']['reviewed_provenance'] = None
    if prospective['candidate'].pop('verified_artifact_identity', None) is not None:
        prospective['candidate']['schema'] = 'ferrule.storage-demand.http-transfer-candidate-binding.v2'
    prospective['candidate']['acceptance'] = {key: binding[key] for key in ('path', 'sha256')}
    return dict(acceptance={key: binding[key] for key in ('path', 'sha256')},
        original_layout=dict(path=str(original_layout), sha256=launch['execution_sha256']),
        original_main_go=dict(path=original_go['launch_review'], sha256=original_go['launch_review_sha256']),
        original_workflow_pins=old, prospective_workflow_pins=new,
        prospective_layout_sha256=ct.digest(prospective),
        quality_identity_sha256=ct.digest(compatibility_quality_identity(layout)),
        protected_semantic_source_pins=semantic, changed_orchestration_paths=sorted(changed),
        required_cpu_test_pins={name: new[name] for name in sorted(allowed_added)},
        scope=('prospective role-partition/provenance admission; explicit helper/test review; unchanged quality/work/deadlines'
            if role_partition else 'prospective orchestration/admission-cache/deadline only; no production or quality-control change'))


def prospective_transfer_plan(acceptance_path, *, build_inputs=None, compatibility_path=None, role_partition=False):
    """CPU plan only: reuse the sealed CLI/HTTP references, never authorize it."""
    acceptance_path = immutable_input(Path(acceptance_path).resolve())
    binding = dict(path=str(acceptance_path), sha256=ct.sha(acceptance_path))
    require(binding['sha256'] == COMPATIBILITY_ACCEPTANCE_SHA256, 'prospective reuse requires exact reviewed acceptance final')
    accepted = acceptance_manifest(binding)['study_layout']
    identity = compiled_identity_for_inputs(build_inputs) if build_inputs else accepted['compiled_identity']
    require(identity == accepted['compiled_identity'], 'prospective CLI/source/runtime descriptor differs from acceptance')
    layout = transfer_plan('pilot', compiled_identity=identity, http_reference_binding=accepted['http_reference_binding'],
        role_partition=role_partition)
    if compatibility_path is not None:
        path = immutable_input(Path(compatibility_path).resolve())
        binding['workflow_compatibility'] = dict(path=str(path), sha256=ct.sha(path))
    layout['candidate'] = dict(accepted['candidate'], acceptance=binding, launch_review=None, launch_review_sha256=None)
    if role_partition:
        layout['candidate'].pop('verified_artifact_identity', None)
        layout['candidate'].update(schema='ferrule.storage-demand.http-transfer-candidate-binding.v4', reviewed_provenance=None)
    return layout


def workflow_compatibility_plan(layout, binding):
    role = role_admission_enabled(layout)
    return dict(schema=('ferrule.storage-demand.http-workflow-compatibility.v2' if role else
        'ferrule.storage-demand.http-workflow-compatibility.v1'),
        subject=prospective_compatibility_subject(layout, binding, role_partition=role), review=None,
        status='pending_separate_human_review', gpu_authorized=False)


def validate_workflow_compatibility(binding, layout):
    reference = binding['workflow_compatibility']
    require(isinstance(reference, dict) and set(reference) == {'path', 'sha256'}, 'explicit compatibility path/hash required')
    path = immutable_input(reference['path'])
    require(ct.sha(path) == reference['sha256'], 'workflow compatibility hash differs')
    value = ct.load(path)
    role = role_admission_enabled(layout)
    subject = prospective_compatibility_subject(layout, binding, role_partition=role)
    review = value.get('review')
    require(value.get('schema') == ('ferrule.storage-demand.http-workflow-compatibility.v2' if role else
        'ferrule.storage-demand.http-workflow-compatibility.v1')
        and value.get('subject') == subject
        and value.get('status') == 'reviewed_for_prospective_orchestration_only'
        and value.get('gpu_authorized') is False and isinstance(review, dict)
        and review.get('decision') == 'APPROVE' and isinstance(review.get('reviewer'), str) and review['reviewer'].strip()
        and review.get('subject_sha256') == ct.digest(subject), 'separate human compatibility review pending/changed')
    # This reference is also inside the new MAIN-GO's purpose/layout subject.
    return value


def check_acceptance_layout(binding, layout, accepted):
    expected = transfer_plan('public_cli_acceptance', compiled_identity=layout['compiled_identity'],
        http_reference_binding=layout['http_reference_binding'])
    if binding.get('workflow_compatibility') is not None:
        bridge = validate_workflow_compatibility(binding, layout)
        require(accepted['artifact_pins'] == bridge['subject']['original_workflow_pins'], 'original acceptance workflow changed')
        expected['artifact_pins'] = bridge['subject']['original_workflow_pins']
    require(set(accepted) == set(expected) and all(accepted[key] == value for key, value in expected.items()
        if key not in ('candidate', 'http_reference_binding')), 'acceptance is not the exact fixed four-cell plan')
    for key in ('cases', 'schedule', 'input_contract', 'http_expectation_contract', 'http_reference_binding',
                'compiled_identity', 'production_source_pins', 'artifact_pins', 'upload_producer_contract'):
        if key == 'artifact_pins' and binding.get('workflow_compatibility') is not None:
            continue  # Only the explicit, validated prospective bridge permits this difference.
        require(accepted.get(key) == layout[key], 'acceptance/pilot configuration/workflow crossbinding: ' + key)
    require(accepted['candidate']['inputs_sha256'] == layout['candidate']['inputs_sha256']
        and accepted['candidate']['uuid_inventory'] == layout['candidate']['uuid_inventory'], 'acceptance CLI/inventory differs')


def validate_public_cli_acceptance(binding, layout, facts):
    report = acceptance_manifest(binding)
    path, accepted = Path(binding['path']), report['study_layout']
    check_acceptance_layout(binding, layout, accepted)
    return validate_acceptance_evidence(binding, report, facts)


def validate_acceptance_evidence(binding, report, facts, *, reviewed_candidate=None):
    path, accepted = Path(binding['path']), report['study_layout']
    if reviewed_candidate is None:
        candidate = validate_transfer_candidate(accepted)
    else:
        # Review owns this real full-validator result. Recheck the original GO;
        # never treat a serialized quality/preparation receipt as permission.
        candidate = reviewed_candidate
        original = copy.deepcopy(candidate['build_review'])
        old = accepted['candidate']
        require(original['artifact_pins'].pop(old['launch_review']) == old['launch_review_sha256']
            and original['compiled_identity'] == accepted['compiled_identity']
            and candidate['uuid_inventory'] == old['uuid_inventory']
            and authorize_transfer_candidate(accepted, original) == candidate['approval_subject'],
            'review reuse is not the exact original candidate/GO')
    require(candidate['build_review']['source'] == facts['source'], 'acceptance full compiled identity differs')
    pins = report['artifact_pins']
    ct.check_pins(pins)
    require(set(pins) == {str(p.resolve()) for p in path.parent.rglob('*') if p.is_file()
        and p.name not in ('paired-final.json', 'study-coordinator.lock')}, 'acceptance evidence inventory incomplete')
    for name in pins:
        require(Path(name).is_relative_to(path.parent), 'foreign acceptance evidence path')
        immutable_input(name)
        require(all(parent.stat().st_mode & 0o222 == 0 for parent in Path(name).parents
            if parent == path.parent or parent.is_relative_to(path.parent)), 'acceptance evidence directory must be sealed')
    require(not (path.parent / 'study-stop.json').exists(), 'acceptance permanently stopped')
    require([row['trial'] for row in report['lifecycles']] == accepted['trial_order'], 'all four ordered acceptance cells required')
    for row in report['lifecycles']:
        folder = path.parent / row['trial']
        require(str(folder / 'assessment.json') in pins and maintained.coordinator_completion(folder)
            and row == ct.load(folder / 'assessment.json'), 'acceptance assessment superseded/changed')
        observed = row['lifecycle']
        validate_acceptance_process(folder, row, accepted, candidate)
        require(ct.sha(folder / 'server.log') == observed['server_log_sha256']
            and str(folder / 'lifecycle.json') in pins and observed == ct.load(folder / 'lifecycle.json'),
            'actual acceptance lifecycle/log binding differs')
        require(maintained.ex.lifecycle_assessment(observed, row['supervisor'], row['post_idle'], transfer_required=True)['can_continue']
            and transfer_ledger(observed, row['mode'])['can_continue'], 'acceptance custody/worker evidence failed')
        require(row['assessment']['can_continue'] is True and row['summary']['issued'] == 12
            and row['summary']['successful'] == 12, 'all fixed acceptance slots must pass')
        validate_request_evidence(folder, observed['requests'], accepted)
    facts['artifact_pins'].update(pins)
    facts['artifact_pins'][str(path)] = binding['sha256']
    if binding.get('workflow_compatibility') is not None:
        reference = binding['workflow_compatibility']
        facts['artifact_pins'][reference['path']] = reference['sha256']
    return report


def validate_acceptance_process(folder, row, layout, bound):
    from runnerhelpers import library_maps_v1 as maps, trace_v2
    launch = ct.load(folder / 'launch.json')
    sealed = ct.load(folder / 'worker-launch.json')
    require(all(launch[k] == sealed[k] for k in planned.CONTRACT_KEYS)
        and sealed['contract_sha256'] == ct.digest({k: sealed[k] for k in planned.CONTRACT_KEYS})
        and sealed['schema'] == 'ferrule.storage-demand.paired-read-worker.v1'
        and sealed['trial'] == row['trial'] and sealed['mode'] == row['mode'] == row['trial'].split('-', 1)[1]
        and sealed['trial_order'] == layout['trial_order'] and sealed['request_ceiling'] == 12
        and sealed['limits'] == layout['limits'] and sealed['schedule'] == layout['schedule'] and sealed['adapter_pins'] == layout['artifact_pins'],
        'actual acceptance sealed launch differs')
    layout_path = immutable_input(sealed['execution_path'])
    require(ct.load(layout_path) == layout and sealed['candidate_path'] == str(layout_path)
        and sealed['candidate_sha256'] == sealed['execution_sha256'] == ct.sha(layout_path),
        'acceptance launch/layout hash binding differs')
    observed = row['lifecycle']
    auth, supervisor = observed['authenticated_worker'], row['supervisor']
    require(auth['launch_id'] == supervisor['custody']['launch_id']
        and all(auth['worker'][k] == supervisor['custody']['root_identity'][k] for k in ('pid', 'birth_ticks', 'ppid'))
        and auth['helper'] == supervisor['custody']['helper_identity']
        and supervisor['custody']['root_session_ready'] is True
        and auth['worker']['pid'] == auth['worker']['pgrp'] == auth['worker']['session']
        and auth['coordinator'] == sealed['coordinator']
        and supervisor == ct.load(folder / 'supervisor.json'), 'acceptance authenticated producer identity differs')
    context = ct.load(immutable_input(folder / 'http-capture-launch.json'))
    limits = sealed['limits']
    require(context['schema'] == 'ferrule.storage-demand.http-capture-launch.v1'
        and context['launch_sha256'] == ct.sha(folder / 'worker-launch.json')
        and context['worker_identity'] == auth['worker']
        and context['target_started'] <= observed['capture']['started_monotonic']
        and context['target_deadline'] == observed['target_deadline'] == min(context['target_started'] + limits['target_lifetime_seconds'],
            sealed['outer_deadline'] - limits['shutdown_seconds'] - limits['worker_finalization_seconds'],
            sealed['pilot_deadline'] - limits['shutdown_seconds'] - limits['worker_finalization_seconds']),
        'acceptance original absolute target deadline/worker binding differs')
    case = layout['cases'][row['mode']]
    spec = sealed['spec']
    require(spec['argv'] == [bound['candidate']['binary'], *case['argv_after_binary']]
        and spec['frozen_cli_sha256'] == bound['candidate']['binary_sha256']
        and spec['input_contract'] == layout['input_contract']
        and spec['transfer_controls_revision'] == 2
        and spec['initial_read_producer_binding'] == bound['build_review']['initial_read_producer']
        and observed['requests']['initial_read_evidence']['server_identity'] == observed['server_identity']
        and observed['requests']['initial_read_evidence']['producer_binding'] == spec['initial_read_producer_binding']
        and spec['upload_producer_contract'] == layout['upload_producer_contract']
        and spec['http_expectation_contract'] == layout['http_expectation_contract']
        and spec['http_reference_context'] == http_reference_context(layout)
        and spec['runtime_binding'] == dict(binary=bound['candidate']['binary'],
            binary_sha256=bound['candidate']['binary_sha256'], manifest=bound['candidate']['runtime_pin_manifest'])
        and spec['environment'] == dict(maintained.ex.runner.cpu_environment(), **dict(case['environment'], HOME='/root',
            CUDA_VISIBLE_DEVICES=layout['candidate']['uuid_inventory'][0],
            NVIDIA_VISIBLE_DEVICES=layout['candidate']['uuid_inventory'][0], CUDA_DEVICE_ORDER='PCI_BUS_ID',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')), 'acceptance public CLI/environment crossbinding')
    runtime = ct.load(bound['candidate']['runtime_pin_manifest'])
    policy = maps.role_policy(bound['candidate']['binary'], bound['candidate']['binary_sha256'], runtime, 'cli')
    for label in ('ready', 'after-requests', 'stop'):
        evidence = folder / (label + '-library-evidence')
        snapshot = ct.load(evidence / 'snapshot.json')
        require(snapshot['raw_maps_path'] == str(evidence / 'maps.raw')
            and ct.load(evidence / 'policy.json') == policy, 'acceptance role-map snapshot/path/policy crossbinding')
        checked = maps.validate_snapshot(snapshot, policy)
        exported = ct.load(folder / (label + '-library-maps.json'))
        require(exported.get('actual_argv') == spec['argv']
            and exported.get('actual_environment') == spec['environment']
            and all(exported.get(k) == checked.get(k) for k in ('status', 'libraries', 'required_pin_check',
                'errors', 'missing_required_paths', 'deleted_library_paths', 'observed_library_count',
                'root_identity', 'raw_maps_path', 'raw_maps_sha256', 'hash_scope')),
            'acceptance actual CLI/environment/maps export differs')
        require(checked['status'] == 'passed' and snapshot['root_identity'] == observed['server_identity']
            and snapshot['parent_identity'] == observed['capture_identity'], 'acceptance raw role-map evidence failed')
    require(observed['capture'] == ct.load(folder / 'profile-server.json')
        and observed['requests'] == ct.load(folder / 'requests-final.json')
        and observed['resources'] == ct.load(folder / 'resources-final.json')
        and observed['planned_http_stop'] == maintained.ex.validate_http_stop(folder, observed['capture'], observed['server_identity'], target_deadline=observed['target_deadline']),
        'acceptance worker/capture/observer/stop exports differ')
    startup = trace_v2.transfer_startup(maintained.ex.log_lines(folder / 'server.log'), case['transfer_policy'], bound['candidate']['binary_sha256'])
    require(startup == observed['requests']['expert_transfers']['startup'], 'actual startup/getter evidence differs')


def validate_request_evidence(folder, requests, layout):
    from runnerhelpers import trace_v2
    from runnerhelpers.http_expectations import http_output_matches, normalize_http_expectation
    ex = maintained.ex
    observations = requests['observations']
    require(len(observations) == 12 and requests['can_continue'] is True, 'fixed 12-request acceptance missing')
    previous = requests['initial_read_endpoint']
    context = validate_http_references(layout)
    first = observations[0]['log_start_byte']
    initial = ex.read_initial_endpoint(ex.log_lines(folder / 'server.log', end=first),
        requests['expert_transfers']['requested_case']['read_options'], layout['upload_producer_contract'],
        server_identity=requests['initial_read_evidence']['server_identity'],
        producer_binding=layout['compiled_identity']['initial_endpoint_producer'],
        policy=requests['expert_transfers']['requested_case']['transfer_policy'],
        ready_lines=ex.log_lines(folder / 'server.log', end=requests['ready_log_end_byte']))
    require(previous == initial['read_endpoint'] and requests['initial_read_evidence'] == initial, 'initial raw read endpoint/evidence differs')
    seen_ids = set()
    for planned_row, row in zip(layout['schedule'], observations):
        request_folder = folder / planned_row['name']
        require(all(row.get(k) == v for k, v in planned_row.items() if k not in ('status', 'result'))
            and row == ct.load(request_folder / 'observed.json') and row['actual'] == ct.load(request_folder / 'http-terminal.json'),
                'raw acceptance request artifacts differ')
        lines = ex.log_lines(folder / 'server.log', row['log_start_byte'], row['log_end_byte'])
        expected = normalize_http_expectation(planned_row, contract=layout['http_expectation_contract'], reference_context=context)
        reference = None if expected['reference_ids'] is None else dict(reference_ids=expected['reference_ids'], reference_text=expected['reference_text'])
        trace = trace_v2.join_token_trace_http(trace_v2.parse_token_trace(lines),
            [dict(planned_row, reference=reference, actual=row['actual'])], lines)[0]
        require(trace['actual_worker_request_id'] not in seen_ids and trace['metric_eligible'] is True,
                'acceptance HTTP IDs duplicate/unbound/terminal failure')
        seen_ids.add(trace['actual_worker_request_id'])
        counters = trace_v2.transfer_counter_window(lines, requests['expert_transfers']['requested_case']['read_options'],
            previous_read=previous, producer_contract=layout['upload_producer_contract'])
        require(counters['valid'] and counters['read_continuity_valid'] and row['logical_counters'] == counters,
                'acceptance raw read windows not reconciled')
        previous = counters['final_read_endpoint']
        require(row['trace'] == trace and row['metric_eligible'] is True
            and row['window_end']['monotonic'] <= row['absolute_deadline']
            and ex.request_gate(planned_row, row['actual'], trace, counters, row['nfs'],
                expectation_contract=layout['http_expectation_contract'], reference_context=context)
            and analysis.request_metrics(planned_row, row, contract=layout['http_expectation_contract'],
                reference_context=context)['reference_qualified']
            and http_output_matches(planned_row, row['actual'], trace,
                contract=layout['http_expectation_contract'], reference_context=context), 'actual acceptance output/reference mismatch')


def transfer_ledger(observed, mode):
    try:
        from runnerhelpers.closure_v2 import evidence
        requests = observed['requests']
        case = requests['expert_transfers']['requested_case']
        require(mode == case['read_schedule'] + '-' + case['transfer_policy'], 'read/policy case crossbinding')
        startup = requests['expert_transfers']['startup']
        require(startup['requested'] == startup['effective'] == case['transfer_policy'], 'startup policy missing/mismatch')
        from runnerhelpers.trace_v2 import transfer_read_continuity
        continuity = transfer_read_continuity([r['logical_counters'] for r in requests['observations']],
            case['read_options'], initial_read=requests.get('initial_read_endpoint'),
            producer_contract=quality.TRANSFER_UPLOAD_PRODUCER)
        require(continuity['can_continue'], continuity['stop_reason'])
        previous = requests['initial_read_endpoint']
        uploads = 0
        for request in requests['observations']:
            counters = request['logical_counters']
            require(counters.get('transfer_health_valid') is True and counters.get('valid') is True,
                    'invocation worker/cache health missing/invalid')
            require(counters['deltas']['expert_uploads'] > 0 and counters['deltas']['h2d_bytes'] > 0,
                    'positive actual cache uploads/H2D missing')
            for inv in counters['rows']:
                read = inv['read_endpoint']
                quality.read_observation_v3(read, case['read_options'], stage='idle_retained',
                    producer_contract=quality.TRANSFER_UPLOAD_PRODUCER)
                if previous is not None:
                    require(all(read['stats'][k] >= previous['stats'][k] for k in quality.V3_COUNTERS + quality.V3_PEAKS),
                            'worker lifetime reset across requests')
                previous = read
            uploads += counters['deltas']['expert_uploads']
        require(previous is not None and uploads > 0 and previous['worker_read_observed'] is True,
                'positive read/upload evidence missing')
        state, final = evidence(observed['closure']['attempts'][-1]['summary']['physical']['expert_read_pipeline'])
        require(state in ('observed', 'verified'), 'final read worker endpoint unobserved')
        quality.read_observation_v3(final, case['read_options'], stage='idle_retained',
            producer_contract=quality.TRANSFER_UPLOAD_PRODUCER)
        require(final == previous, 'shutdown worker endpoint differs from last settled invocation')
        return dict(can_continue=True, final_read_ledger=final, cache_uploads=uploads,
            grouped_backend_call_count=None, dispatch_evidence='positive uploads plus frozen-source binding, not getter alone',
            host_idle_charged_bytes=final['current_charge']['host_charged_bytes'],
            quiescence_certificate=False, physical_release='unobserved; intentional owned HTTP stop')
    except (ValueError, KeyError, TypeError) as error:
        return dict(can_continue=False, stop_reason=str(error), final_read_ledger=None)


STATIC_PREPARATION_SECONDS = 180


def preparation_identity(path):
    path = Path(path)
    st, link = path.stat(), path.lstat()
    return [str(path.resolve()), st.st_dev, st.st_ino, st.st_mode, st.st_size,
            st.st_mtime_ns, st.st_ctime_ns, link.st_ino, link.st_ctime_ns]


def validate_static_preparation(reference, path, *, deadline=None):
    """Re-derive actual candidate/MAIN-GO authority; reuse only content freshness.

    Exact coverage comes from hash-bound manifests, never receipt declarations.
    ctime/inode checks also reject rewrites that restore size/mtime. Mutable
    launch inputs and the pinned executable remain independent requirements.
    """
    limit = time.monotonic() + STATIC_PREPARATION_SECONDS
    if deadline is not None:
        limit = min(limit, deadline)
    remaining = signal.getitimer(signal.ITIMER_REAL)[0]
    require(time.monotonic() < limit, 'absolute static validation deadline expired')
    require(not remaining or remaining <= limit - time.monotonic() + .01,
        'existing phase timer exceeds original static validation deadline; refuse rather than refresh')
    with ExitStack() if remaining else maintained.ex.absolute_alarm(limit):
        receipt_path = immutable_input(reference['path'])
        require(ct.sha(receipt_path) == reference['sha256'], 'static preparation receipt changed')
        value = ct.load(receipt_path)
        require(value['schema'] == 'ferrule.storage-demand.http-static-preparation.v2'
            and value['layout_path'] == str(Path(path).resolve())
            and value['boot_id'] == maintained.boot_id()
            and value['registration_sha256'] == ct.sha(receipt_path.parent / 'registration.json')
            and value['layout_sha256'] == ct.sha(path), 'static preparation registration/layout changed')
        layout = validate_transfer_layout(path, candidate_required=False)
        registered = historical.unseal(receipt_path.parent / 'registration.json',
            'ferrule.storage-demand.actual-http-transfer-registration.v1')
        require(registered['study_layout'] == layout
            and registered['binding_path'] == registered['execution_path'] == str(Path(path).resolve())
            and registered['binding_sha256'] == registered['execution_sha256'] == ct.sha(path)
            and registered['boot_id'] == maintained.boot_id(), 'prepared actual registration/layout binding differs')
        admission_pins = {str(Path(path).resolve()): ct.sha(path),
            str(receipt_path.parent / 'registration.json'): ct.sha(receipt_path.parent / 'registration.json')}
        # Always execute the real candidate authority phase. Cached data is freshness only.
        bound = validate_transfer_candidate(layout, freshness=value, admission_pins=admission_pins)
        require(value['started']['monotonic'] <= value['ended']['monotonic'] <= value['deadline']
            and value['throughput_denominator'] is False, 'static preparation phase clock changed')
        require(value['layout'] == layout and value['workflow_pins'] == layout['artifact_pins'],
            'static preparation actual layout changed')
        # These are live launch inputs, not the static semantic-review result.
        ct.check_pins(layout['artifact_pins'])
        validate_transfer_inputs(layout['input_contract'])
        validate_http_references(layout)
        for name, row in value['verified_inputs'].items():
            require(preparation_identity(name) == row['identity'], 'static input changed during revalidation: ' + name)
        for name, identity in value['verified_directories'].items():
            require(preparation_identity(name) == identity, 'static directory changed during revalidation: ' + name)
        return dict(value, layout=layout, bound_candidate=bound)


@contextmanager
def prepare_transfer_admission(directory, path):
    """Finite CPU-only static review under coordinator serialization, before locks."""
    directory = Path(directory)
    started, trial, deadline = None, None, None
    adapters = ExitStack()
    try:
        started = maintained.ex.stamp()
        with maintained.ex.absolute_alarm(started['monotonic'] + STATIC_PREPARATION_SECONDS):
            maintained.require_admission_open(directory)
            path = Path(path).resolve()
            registered = historical.unseal(directory / 'registration.json',
                'ferrule.storage-demand.actual-http-transfer-registration.v1')
            require(registered['boot_id'] == maintained.boot_id(), 'static preparation boot changed')
            trial = maintained.next_trial(directory, registered)
            require(isinstance(trial, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,96}', trial), 'unsafe/unknown preparation trial')
        deadline = min(started['monotonic'] + STATIC_PREPARATION_SECONDS, registered['pilot_deadline'] - 750)
        output = directory / ('static-preparation-' + trial + '.json')
        require(not output.exists(), 'static preparation exclusive; no retry')
        with maintained.ex.absolute_alarm(deadline):
            role_layout = ct.load(path)
    except (Exception, KeyboardInterrupt) as error:
        maintained.admission_refusal(directory, error, trial=trial, started=started, deadline=deadline,
            source_proof='prepare_transfer_admission initial entry failed before yielding')
        raise
    if role_admission_enabled(role_layout):
        with transfer_admission_api().prepared_admission(sys.modules[__name__], directory, path,
                registered, trial, started, deadline):
            yield
        return
    inputs = {}
    original_sha = ct.sha
    def sha(name):
        name = str(Path(name).absolute())
        before = preparation_identity(name)
        pin = original_sha(name)
        require(preparation_identity(name) == before, 'input changed during static hashing: ' + name)
        row = dict(sha256=pin, identity=before)
        require(name not in inputs or inputs[name] == row, 'input changed during static review: ' + name)
        inputs[name] = row
        return pin
    try:
        with maintained.ex.absolute_alarm(deadline):
            ct.sha = sha
            try:
                ct.sha(path)
                ct.sha(directory / 'registration.json')
                layout = validate_transfer_layout(path, candidate_required=False)
                immutable_input(path)
                reviewed = validate_transfer_candidate(layout)
                authority = transfer_candidate_authority(layout)
                bound = authority['bound_candidate']
                require(reviewed['candidate'] == bound['candidate']
                    and reviewed['approval_subject'] == bound['approval_subject'], 'static review/actual authority differs')
                admission_pins = {str(path): ct.sha(path), str(directory / 'registration.json'): ct.sha(directory / 'registration.json')}
                required = dict(authority['required_pins'], **admission_pins)
                for name, pin in required.items():
                    if name not in inputs:
                        sha(name)
                    require(inputs[name]['sha256'] == pin, 'static verified bytes/required manifest pin differs: ' + name)
                inputs = {name: inputs[name] for name in required}
                trees = authority['required_trees']
                for root, members in trees.items():
                    require(coverage_tree(root, authority) == members, 'static required inventory differs')
                directories = {name: preparation_identity(name) for name in coverage_directories(authority)}
                for root, members in authority['required_globs'].items():
                    require(sorted(str(p) for p in Path(root).glob('*.a')) == members, 'static required native archive inventory differs')
                validate_transfer_inputs(layout['input_contract'])
                require(transfer_registration(directory, path, selected_layout=layout) == registered,
                    'registration changed during static review')
                directory_check(directory, transfer=True, purpose=layout['purpose'])
                for name, row in inputs.items():
                    require(preparation_identity(name) == row['identity'], 'input stale after static review: ' + name)
                for root, members in trees.items():
                    require(coverage_tree(root, authority) == members, 'inventory stale after static review')
                for name, identity in directories.items():
                    require(preparation_identity(name) == identity, 'directory stale after static review')
            finally:
                ct.sha = original_sha
            ended = maintained.ex.stamp()
            require(ended['monotonic'] <= deadline, 'static preparation deadline exceeded')
            value = dict(schema='ferrule.storage-demand.http-static-preparation.v2', trial=trial,
                started=started, ended=ended, deadline=deadline,
                duration_seconds=ended['monotonic'] - started['monotonic'],
                throughput_denominator=False, boot_id=maintained.boot_id(),
                registration_sha256=ct.sha(directory / 'registration.json'), layout_path=str(path),
                layout_sha256=ct.sha(path), layout=layout, bound_candidate=bound,
                workflow_pins=layout['artifact_pins'], approval_subject=bound['approval_subject'],
                admission_pins=admission_pins, verified_inputs=inputs, verified_trees=trees,
                verified_globs=authority['required_globs'], verified_directories=directories)
            ct.write(output, value)
            output.chmod(0o400)
            reference = dict(path=str(output), sha256=ct.sha(output))
            adapters.enter_context(transfer_adapter(path, prepared=reference))
    except (Exception, KeyboardInterrupt) as error:
        try:
            adapters.close()
        except (Exception, KeyboardInterrupt) as cleanup_error:
            error.add_note('adapter cleanup failure: ' + str(cleanup_error))
        maintained.admission_refusal(directory, error, trial=trial, started=started, deadline=deadline,
            source_proof='prepare_transfer_admission failed before yielding to execute_next')
        raise
    try:
        with adapters:
            yield
    except (Exception, KeyboardInterrupt) as error:
        # After yielding, execution may have entered the supervisor. Never label
        # an arbitrary later exception as never_spawned. Its arm journal owns it.
        try:
            maintained.stop(directory, type(error).__name__ + ': ' + str(error))
        except (Exception, KeyboardInterrupt) as journal_error:
            error.add_note('STOP journal failure: ' + str(journal_error))
        raise


def transfer_registration(directory, path, *, selected_layout=None, admission=None):
    value = historical.unseal(Path(directory) / 'registration.json',
        'ferrule.storage-demand.actual-http-transfer-registration.v1')
    layout = validate_transfer_layout(path) if selected_layout is None else selected_layout
    if admission is not None:
        require(isinstance(admission, transfer_admission_api().Admission) and role_admission_enabled(layout),
            'process-local role admission required')
        admission.check(path)
    else:
        validate_transfer_inputs(layout['input_contract'])
    require(value['study_layout'] == layout and value['execution_path'] == value['binding_path'] == str(Path(path).resolve())
        and value['binding_sha256'] == value['execution_sha256'] == ct.sha(path), 'transfer layout/candidate changed')
    require(value['boot_id'] == maintained.boot_id() and value['trial_order'] == layout['trial_order']
        and value['order'] == layout['order'] and value['limits'] == layout['limits']
        and value['schedule'] == layout['schedule'], 'transfer clock/order/workload changed')
    require(value['owner_authorization'] is True and value['owner_note'] == transfer_note(layout)
        and value['eight_uuids'] == layout['candidate']['uuid_inventory']
        and value['selected_uuid'] == value['eight_uuids'][0], 'transfer owner/inventory changed')
    return value


def validate_http_capture_entry(directory, payload, launch_hash, target_deadline, worker):
    """Reuse registration/GO and the maintained exact worker contract at capture entry."""
    from runnerhelpers.evidence_v2 import finite
    directory = Path(directory)
    path = Path(payload['execution_path'])
    prepared = payload['spec'].get('static_preparation')
    with transfer_adapter(path, prepared=prepared, deadline=target_deadline, launch=payload):
        registered = maintained.registration(directory.parent)
        layout = registered['study_layout']
        directory_check(directory.parent, transfer=True, purpose=layout['purpose'])
        require(directory.parent.parent == HERE and directory.name == payload['trial']
            and payload['trial'] in registered['trial_order']
            and not any((directory.parent / name).exists() for name in ('study-stop.json', 'paired-final.json'))
            and not any((directory / name).exists() for name in ('assessment.json', 'profile-server.json')),
            'capture is not the active unrepeated registered transfer attempt')
        candidate = maintained.contract(payload)
        launch = ct.load(directory / 'launch.json')
        attempt = ct.load(directory / 'attempt.json')
        held = ct.load(directory / 'locks-held.json')
        if prepared is not None:
            require(attempt['static_preparation'] == prepared
                and Path(prepared['path']).parent == directory.parent
                and ct.load(prepared['path'])['trial'] == payload['trial']
                and ct.load(prepared['path'])['ended']['monotonic'] <= attempt['admission']['monotonic'],
                'capture static admission/attempt binding differs')
        require(all(launch[k] == payload[k] for k in planned.CONTRACT_KEYS)
            and launch['all_uuid_locks'] == payload['all_uuid_locks'] == held['locks']
            and held['coordinator'] == attempt['coordinator'] == payload['coordinator']
            and held.get('all_eight_local') is True and held.get('global_reservation') is False
            and attempt['trial'] == payload['trial'] and attempt['replay_permitted'] is False
            and attempt['registration_sha256'] == ct.sha(directory.parent / 'registration.json')
            and attempt['outer_deadline'] == payload['outer_deadline']
            and attempt['pilot_deadline'] == registered['pilot_deadline'] == payload['pilot_deadline'] + 100
            and registered['owner_authorization'] is True and payload['granted'] is True
            and payload['owner_note'] == registered['owner_note'] == transfer_note(layout)
            and payload['eight_uuids'] == registered['eight_uuids']
            and payload['uuid'] == registered['selected_uuid']
            and payload['baseline'] == ct.load(directory / 'baseline.json'), 'capture registration/owner/held-lock/attempt binding differs')
        for trial in registered['trial_order'][:registered['trial_order'].index(payload['trial'])]:
            prior = directory.parent / trial
            require(maintained.coordinator_completion(prior) and ct.load(prior / 'assessment.json')['assessment']['can_continue'] is True,
                'capture cannot bypass prior failed/incomplete scheduled slot')
        context_path = immutable_input(directory / 'http-capture-launch.json')
        context = ct.load(context_path)
        limits = payload['limits']
        require(context.get('schema') == 'ferrule.storage-demand.http-capture-launch.v1'
            and context['launch_sha256'] == launch_hash and context['worker_identity'] == worker
            and finite(target_deadline) and finite(context['target_started'])
            and attempt['admission']['monotonic'] <= context['target_started'] < time.monotonic() < target_deadline
            and context['target_deadline'] == target_deadline == min(context['target_started'] + limits['target_lifetime_seconds'],
                payload['outer_deadline'] - limits['shutdown_seconds'] - limits['worker_finalization_seconds'],
                payload['pilot_deadline'] - limits['shutdown_seconds'] - limits['worker_finalization_seconds']),
            'capture original target deadline/worker binding differs')
        require(payload['spec']['argv'][0] == candidate['candidate']['binary']
            and payload['spec']['frozen_cli_sha256'] == candidate['candidate']['binary_sha256']
            and payload['spec']['purpose'] == layout['purpose']
            and candidate['build_review']['initial_read_observation_available'] is True
            and payload['spec']['initial_read_producer_binding'] == candidate['build_review']['initial_read_producer'],
            'capture exact approved candidate/purpose/initial producer unavailable')
        return candidate


def transfer_note(layout):
    return TRANSFER_NOTE if layout['purpose'] == 'pilot' else (
        'Renewed sole-owner authorization for four fixed public CLI acceptance cells; 48 requests, no retries or pilot authority')


@contextmanager
def transfer_adapter(path, *, prepared=None, deadline=None, admission=None, launch=None, finalizing=False):
    path = Path(path).resolve()
    closing_layout = validate_transfer_layout(path, candidate_required=False) if finalizing else None
    closing = finalizing and role_admission_enabled(closing_layout)
    closing_bound = validate_transfer_candidate(closing_layout) if closing else None
    role = admission is not None or (launch is not None and launch['spec'].get('role_admission') is not None)
    require(not finalizing or (admission is None and launch is None and prepared is None), 'closeout cannot grant launch admission')
    owned = role and admission is None
    if owned:
        require(prepared is None and launch is not None, 'role adapter requires a validated lifecycle admission')
        api = transfer_admission_api()
        with ExitStack() if signal.getitimer(signal.ITIMER_REAL)[0] else maintained.ex.absolute_alarm(deadline):
            admission = api.admit_transfer_launch(sys.modules[__name__], path, Path(launch['directory']).parent,
                launch['trial'], deadline=deadline, payload=launch)
    if role:
        require(isinstance(admission, transfer_admission_api().Admission) and prepared is None,
            'role adapter cannot reuse a legacy/static receipt')
        admission.check(path)
    static = validate_static_preparation(prepared, path, deadline=deadline) if prepared else None
    layout = admission.layout if role else closing_layout if closing else static['layout'] if static else validate_transfer_layout(path)
    require(role or closing or not role_admission_enabled(layout), 'role adapter needs one process-local admission; no implicit recursion')
    def execution(selected=path):
        require(Path(selected).resolve() == path, 'unknown transfer execution path')
        if closing:
            return copy.deepcopy(layout)
        if role:
            admission.check(selected)
            return copy.deepcopy(admission.layout)
        return validate_static_preparation(prepared, path, deadline=deadline)['layout'] if prepared else validate_transfer_layout(path)
    def candidate(selected=path):
        require(Path(selected).resolve() == path, 'unknown transfer candidate path')
        if closing:
            return copy.deepcopy(closing_bound)
        if role:
            admission.check(selected)
            return copy.deepcopy(admission.bound)
        if prepared:
            return copy.deepcopy(validate_static_preparation(prepared, path, deadline=deadline)['bound_candidate'])
        selected_layout = execution(selected)
        validate_transfer_inputs(selected_layout['input_contract'])
        return validate_transfer_candidate(selected_layout)
    def case(mode):
        require(mode in layout['cases'], 'unknown transfer read/policy case')
        return copy.deepcopy(layout['cases'][mode])
    original_spec = planned.spec_for
    def spec_for(bound, uuid, mode):
        require(not closing, 'closeout is not launch permission')
        spec = original_spec(bound, uuid, mode)
        if role:
            spec['role_admission'] = admission.reference()
        if prepared is not None:
            spec['static_preparation'] = prepared
        spec['environment']['HOME'] = '/root'
        spec.update(transfer_controls_revision=2, purpose=layout['purpose'], http_capture_entry=str(ENTRY),
            input_contract=layout['input_contract'], upload_producer_contract=layout['upload_producer_contract'],
            http_expectation_contract=layout['http_expectation_contract'],
            http_reference_context=http_reference_context(layout),
            initial_read_producer_binding=bound.get('build_review', {}).get('initial_read_producer'),
            runtime_binding=dict(binary=bound['candidate']['binary'],
                binary_sha256=bound['candidate']['binary_sha256'],
                manifest=bound['candidate']['runtime_pin_manifest']))
        return spec
    facade = SimpleNamespace(validate_execution=execution, validate_binding=candidate,
        raw_options=lambda mode: case(mode)['read_options'], static_preparation=prepared,
        admission_deadline=admission.deadline if role else static['deadline'] if static else None,
        static_preparation_started=admission.started['monotonic'] if role else static['started']['monotonic'] if static else None,
        role_admission=admission, finalization_only=closing)
    def write(selected, value):
        if Path(selected).name == 'paired-final.json':
            schema = ('ferrule.storage-demand.http-transfer-acceptance.v2' if layout['purpose'] == 'public_cli_acceptance'
                else 'ferrule.storage-demand.http-transfer-actual.v2')
            value = dict(value, schema=schema,
                historical_resume_permitted=False)
        return ORIGINAL_WRITE(selected, value)
    changes = ((historical, 'validate_prepared', execution), (historical, 'validate_candidate', candidate),
        (historical, 'PROTOCOL_PATH', path), (historical, 'case_for_mode', case),
        (historical, 'trial_order', lambda: layout['trial_order']),
        (historical, 'expected_order', lambda: layout['order']),
        (planned, 'ENTRY', ENTRY), (planned, 'worker_contract', maintained.contract),
        (planned, 'spec_for', spec_for), (planned, 'read_ledger_assessment', transfer_ledger), (maintained, 'binding', facade),
        (maintained, 'registration', lambda directory: transfer_registration(directory, path, selected_layout=execution(), admission=admission)
            if prepared or role or closing else transfer_registration(directory, path)),
        (maintained, 'actual_adapter', lambda: transfer_adapter(path, prepared=prepared, deadline=deadline, admission=admission)),
        (maintained, 'ledger', transfer_ledger), (maintained, 'NOTE', transfer_note(layout)), (ct, 'write', write))
    if role:
        original_capture_launch = maintained.ex.launch_http_capture
        def launch_capture(directory, payload, target_deadline, *, target_started):
            require(payload['spec']['role_admission'] == admission.reference(), 'worker role launch crossbinding')
            with ExitStack() if signal.getitimer(signal.ITIMER_REAL)[0] else maintained.ex.absolute_alarm(target_deadline):
                admission.before_spawn()
            return original_capture_launch(directory, payload, target_deadline, target_started=target_started)
        changes += ((maintained.ex, 'launch_http_capture', launch_capture),)
    originals = [(module, name, getattr(module, name)) for module, name, value in changes]
    try:
        for module, name, value in changes:
            setattr(module, name, value)
        yield
    finally:
        for module, name, value in reversed(originals):
            setattr(module, name, value)
        if owned:
            admission.close()


def adapter(layout_path=None):
    return historical_adapter() if layout_path is None else transfer_adapter(layout_path)


def transfer_start(directory, path, granted):
    require(granted is True, 'renewed sole-GPU-owner authorization required')
    layout = validate_transfer_layout(path)
    immutable_input(path)
    validate_transfer_inputs(layout['input_contract'])
    directory = directory_check(directory, transfer=True, purpose=layout['purpose']).resolve()
    directory.mkdir(exist_ok=False)
    now = time.monotonic()
    value = dict(schema='ferrule.storage-demand.actual-http-transfer-registration.v1',
        started=maintained.ex.stamp(), pilot_deadline=now + layout['limits']['pilot_seconds'],
        boot_id=maintained.boot_id(), order=layout['order'], trial_order=layout['trial_order'],
        limits=layout['limits'], schedule=layout['schedule'], study_layout=layout,
        binding_path=str(Path(path).resolve()), binding_sha256=ct.sha(path),
        execution_path=str(Path(path).resolve()), execution_sha256=ct.sha(path),
        selected_uuid=layout['candidate']['uuid_inventory'][0], eight_uuids=layout['candidate']['uuid_inventory'],
        owner_authorization=True, owner_note=transfer_note(layout),
        scheduled_blocks=layout['scheduled_blocks'], scheduled_starts=layout['scheduled_starts'],
        scheduled_requests=layout['scheduled_requests'], scheduled_measured_qa=layout['scheduled_measured_qa'],
        retries=0, replacements=0, profiling_captures=0, historical_resume_permitted=False)
    ct.write(directory / 'registration.json', dict(value, digest=ct.digest(value)))
    return dict(status='registered', first_arm=layout['trial_order'][0], directory=str(directory))


def transfer_main(args):
    if getattr(args, 'role_partition', False):
        require(args.mode == 'plan' and args.acceptance_final is not None and args.purpose == 'pilot',
            '--role-partition is prospective plan --acceptance-final only')
    if getattr(args, 'reviewed_provenance_output', None) is not None:
        require(args.mode == 'review' and args.layout is not None, 'role proof output is review-only')
    if getattr(args, 'verified_artifact_identity_output', None) is not None:
        require(args.mode == 'review' and args.layout is not None, 'identity issuance requires review --layout before MAIN-GO')
    if args.mode == 'review' and args.layout is None:
        value = dict(review_transfer_build(args.build_inputs or CANDIDATE_INPUTS), status='build_reviewed_not_authorized')
        print(json.dumps(value, sort_keys=True), flush=True)
        return 0
    if args.mode == 'compatibility-plan':
        require(args.layout is not None and args.acceptance_final is not None, '--layout and --acceptance-final required')
        layout = validate_transfer_layout(args.layout, candidate_required=False)
        acceptance = args.acceptance_final.resolve()
        value = workflow_compatibility_plan(layout, dict(path=str(acceptance), sha256=ct.sha(acceptance)))
    elif args.mode == 'plan':
        if getattr(args, 'acceptance_final', None) is not None:
            require(args.purpose == 'pilot', 'prospective acceptance reuse is pilot-only')
            value = prospective_transfer_plan(args.acceptance_final, build_inputs=args.build_inputs,
                compatibility_path=getattr(args, 'workflow_compatibility', None),
                role_partition=getattr(args, 'role_partition', False))
        else:
            require(getattr(args, 'workflow_compatibility', None) is None, '--workflow-compatibility requires --acceptance-final')
            value = transfer_plan(args.purpose, compiled_identity=compiled_identity_for_inputs(args.build_inputs) if args.build_inputs else None)
    else:
        require(args.layout is not None, '--layout required; no historical candidate reuse')
        if args.mode == 'review':
            layout = validate_transfer_layout(args.layout, candidate_required=False)
            if role_admission_enabled(layout):
                require(args.verified_artifact_identity_output is None, 'role proof is separate from legacy stat identity issuance')
                output = getattr(args, 'reviewed_provenance_output', None)
                require(output is not None and not output.exists(), 'role review requires an exclusive --reviewed-provenance-output; no implicit GO')
                review_deadline = time.monotonic() + STATIC_PREPARATION_SECONDS
                with maintained.ex.absolute_alarm(review_deadline):
                    proof = review_transfer_provenance(layout, deadline=review_deadline)
                ct.write(output, proof)
                output.chmod(0o400)
                print(json.dumps(dict(status='provenance_reviewed_not_authorized', path=str(output.resolve()),
                    sha256=ct.sha(output), role_partition_sha256=ct.digest(proof['partition']))), flush=True)
                return 0
            if args.verified_artifact_identity_output is not None:
                value = issue_verified_artifact_identity(layout, args.verified_artifact_identity_output)
                print(json.dumps(value, sort_keys=True), flush=True)
                return 0
            reviewed = validate_transfer_candidate(layout, approval_required=False)
            print(json.dumps(dict(status='review_ready_not_authorized', **reviewed), sort_keys=True), flush=True)
            return 0
        if args.mode == 'next':
            require(args.directory is not None, '--directory required')
            # Validate and review inside the finite admission phase, not before it.
            value = maintained.execute_next(args.directory,
                admission=lambda directory: prepare_transfer_admission(directory, args.layout))
            print(json.dumps(dict(trial=value['trial'], assessment=value['assessment'], summary=value['summary'])), flush=True)
            return 0 if value['assessment']['can_continue'] else 1
        layout = validate_transfer_layout(args.layout)
        if args.mode == 'check':
            value = dict(status='checked', gpu_queries=0, server_runs=0)
        else:
            require(args.directory is not None, '--directory required')
            directory = directory_check(args.directory, transfer=True, purpose=layout['purpose'])
            if args.mode == 'start':
                value = transfer_start(directory, args.layout, args.sole_gpu_owner_granted)
            else:
                transfer_registration(directory, args.layout)
                with transfer_adapter(args.layout, finalizing=args.mode == 'finalize'):
                    value = maintained.execute_next(directory) if args.mode == 'next' else maintained.finalize(directory)
                if args.mode == 'next':
                    print(json.dumps(dict(trial=value['trial'], assessment=value['assessment'], summary=value['summary'])), flush=True)
                    return 0 if value['assessment']['can_continue'] else 1
    print(json.dumps(value, sort_keys=True), flush=True)
    return 0 if value['status'] not in ('stopped', 'incomplete') else 1


if __name__ == '__main__':
    raise SystemExit(main())
