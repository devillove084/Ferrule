#!/usr/bin/env python3
"""Explicit V3 CPU plan/check and separately gated maintained finite-arm execution."""
from __future__ import annotations

import argparse
import copy
from types import SimpleNamespace
from contextlib import contextmanager
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


def directory_check(directory, *, transfer=False):
    directory = Path(directory)
    prefix = 'transfer-pilot' if transfer else 'v3'
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
    if argv and argv[0] == '--_worker':
        require(len(argv) in (2, 4) and (len(argv) == 2 or argv[2] == '--layout'),
                'authenticated worker layout arguments required')
        with adapter(Path(argv[3]) if len(argv) == 4 else None):
            return planned.worker(argv[1])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'check', 'start', 'next', 'finalize'))
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--study', choices=('historical-v3', 'transfer-pilot-v1'), default='historical-v3')
    parser.add_argument('--layout', type=Path)
    parser.add_argument('--sole-gpu-owner-granted', action='store_true')
    args = parser.parse_args(argv)
    if args.layout is not None or args.study == 'transfer-pilot-v1':
        return transfer_main(args)
    if args.mode == 'plan':
        value = dict(plan.plan(), versioned_execution_entry=str(ENTRY), binding_pending=not binding.BINDING_PATH.exists(),
                     integration_does_not_amend_preregistration=True)
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

def transfer_workflow_paths():
    from newversionedprotocol import protocol as worker_protocol
    return [ct.ROOT / p for p in worker_protocol.IMPORTS] + [
        ct.ROOT / 'scripts/gpu_idle_window_v2.py', ct.ROOT / 'scripts/run_pinned_jobs.py',
        ct.ROOT / 'target/validation/qwen35-35b-current/toolchain/env-cuda124.sh',
        Path(sys.executable), *HERE.parent.joinpath('runnerhelpers').glob('*.py'),
        *HERE.parent.joinpath('newversionedprotocol').glob('*.py'),
        *(HERE / name for name in ('run_v3.py', 'run_paired.py', 'run_actual_v2.py',
            'analysis.py', 'quality.py', 'protocol.py', 'test_http_transfer.py',
            'cases-v3.json', 'requests.json', 'oracles.json'))]


def transfer_plan():
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
    return dict(schema=analysis.TRANSFER_LAYOUT, status='planned_not_authorized',
        study='transfer-pilot-v1', production_commit=TRANSFER_COMMIT,
        production_source_pins={str(ct.ROOT / p): pin for p, pin in TRANSFER_SOURCE_SHA256.items()},
        cases=cases, cases_sha256=ct.sha(plan.CASES_PATH),
        schedule=ct.load(HERE / 'requests.json')['schedule'], schedule_sha256=ct.sha(HERE / 'requests.json'),
        oracles_sha256=ct.sha(HERE / 'oracles.json'), order=analysis.transfer_order(),
        trial_order=analysis.transfer_trials(), limits=base['blocking']['limits'],
        scheduled_blocks=4, scheduled_pairs_per_schedule=2, scheduled_starts=8,
        scheduled_requests=96, scheduled_measured_qa=40, expected_measured_tokens=320,
        scheduled_measured_qa_per_policy=20, expected_measured_tokens_per_policy=160,
        required_cuda_toolkit='12.4', required_cuda_architecture='sm_86',
        candidate=None, artifact_pins=ct.inventory(transfer_workflow_paths()),
        gpu_authorized=False, gpu_queries=0, gpu_locks=0, gpu_runs=0,
        http_eos='ignore_eos omitted, stop enabled; require frozen length=8 references',
        diagnostic_eos='stop_at_eos=false; compatibility requires exact observed IDs/text/finish/usage',
        dispatch_evidence='positive cache uploads plus frozen-source policy dispatch; direct grouped-call counter unavailable',
        routes_observed=False, logits_observed=False, scientific_acceptance=False)


def validate_transfer_layout(path, *, candidate_required=True):
    value = ct.load(path)
    expected = transfer_plan()
    ct.check_pins(expected['production_source_pins'])
    require(isinstance(value, dict) and set(value) == set(expected)
        and all(value[k] == v for k, v in expected.items() if k != 'candidate'),
        'transfer layout/workload/source pins/workflow changed; old source seals cannot qualify')
    validate_schedule(value['schedule'])
    if candidate_required:
        validate_transfer_candidate(value)
    return value


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


def validate_transfer_candidate(layout):
    candidate = layout.get('candidate')
    require(isinstance(candidate, dict) and set(candidate) == {'binary', 'binary_sha256',
        'runtime_pin_manifest', 'source_freeze', 'launch_review', 'artifact_pins', 'uuid_inventory'},
        'candidate pending: new frozen CLI/source/runtime and main launch review required')
    ct.check_pins(candidate['artifact_pins'])
    for key in ('binary', 'runtime_pin_manifest', 'source_freeze', 'launch_review'):
        path = Path(candidate[key])
        require(path.is_absolute() and path.resolve() == path and not path.is_symlink() and path.is_file()
            and path.stat().st_mode & 0o222 == 0 and str(path) in candidate['artifact_pins'],
            'immutable pinned candidate input required: ' + key)
    require(candidate['binary_sha256'] == ct.sha(candidate['binary']), 'frozen CLI hash mismatch')
    freeze = ct.load(candidate['source_freeze'])
    require(freeze.get('schema') == 'ferrule.expert-read.committed-source-freeze.v1'
        and freeze.get('commit') == layout['production_commit']
        and freeze.get('immutable_readonly_inputs') is True, 'new committed source freeze required; no old-cut transfer')
    frozen = Path(freeze['frozen_source'])
    require(frozen.is_absolute() and not frozen.is_symlink() and frozen.stat().st_mode & 0o222 == 0,
            'immutable frozen source directory required')
    texts = {}
    for name in PROOF_PATHS:
        pin = layout['production_source_pins'][str(ct.ROOT / name)]
        source = frozen / name
        require(freeze['pins_relative_to_frozen_source'].get(name) == pin
            and not any(p.is_symlink() for p in (source, *source.parents))
            and source.stat().st_mode & 0o222 == 0 and ct.sha(source) == pin, 'old/mismatched frozen source pin: ' + name)
        texts[name] = source.read_text()
    review_transfer_source(texts)
    review = ct.load(candidate['launch_review'])
    require(review.get('schema') == 'ferrule.storage-demand.http-transfer-launch-review.v1'
        and review.get('production_commit') == layout['production_commit']
        and review.get('cli_binary_sha256') == candidate['binary_sha256']
        and review.get('source_freeze_sha256') == ct.sha(candidate['source_freeze'])
        and review.get('runtime_pin_manifest_sha256') == ct.sha(candidate['runtime_pin_manifest'])
        and all(review.get(k) is True for k in ('main_source_reviewed', 'build_custody_reviewed',
            'same_cut_cli_quality_accepted')), 'main same-cut public CLI/build/source acceptance pending')
    ct.check_pins(ct.load(candidate['runtime_pin_manifest'])['pins'])
    maintained.pilot.validate_inventory(candidate['uuid_inventory'], candidate['uuid_inventory'][0])
    return dict(candidate=candidate, uuid_inventory=candidate['uuid_inventory'])


def transfer_ledger(observed, mode):
    try:
        from runnerhelpers.closure_v2 import evidence
        requests = observed['requests']
        case = requests['expert_transfers']['requested_case']
        require(mode == case['read_schedule'] + '-' + case['transfer_policy'], 'read/policy case crossbinding')
        startup = requests['expert_transfers']['startup']
        require(startup['requested'] == startup['effective'] == case['transfer_policy'], 'startup policy missing/mismatch')
        previous = None
        uploads = 0
        for request in requests['observations']:
            counters = request['logical_counters']
            require(counters.get('transfer_health_valid') is True and counters.get('valid') is True,
                    'invocation worker/cache health missing/invalid')
            require(counters['deltas']['expert_uploads'] > 0 and counters['deltas']['h2d_bytes'] > 0,
                    'positive actual cache uploads/H2D missing')
            for inv in counters['rows']:
                read = inv['read_endpoint']
                quality.read_observation_v3(read, case['read_options'], stage='idle_retained')
                if previous is not None:
                    require(all(read['stats'][k] >= previous['stats'][k] for k in quality.V3_COUNTERS + quality.V3_PEAKS),
                            'worker lifetime reset across requests')
                previous = read
            uploads += counters['deltas']['expert_uploads']
        require(previous is not None and uploads > 0 and previous['worker_read_observed'] is True,
                'positive read/upload evidence missing')
        state, final = evidence(observed['closure']['attempts'][-1]['summary']['physical']['expert_read_pipeline'])
        require(state in ('observed', 'verified'), 'final read worker endpoint unobserved')
        quality.read_observation_v3(final, case['read_options'], stage='idle_retained')
        require(final == previous, 'shutdown worker endpoint differs from last settled invocation')
        return dict(can_continue=True, final_read_ledger=final, cache_uploads=uploads,
            grouped_backend_call_count=None, dispatch_evidence='positive uploads plus frozen-source binding, not getter alone',
            host_idle_charged_bytes=final['current_charge']['host_charged_bytes'],
            quiescence_certificate=False, physical_release='unobserved; intentional owned HTTP stop')
    except (ValueError, KeyError, TypeError) as error:
        return dict(can_continue=False, stop_reason=str(error), final_read_ledger=None)


def transfer_registration(directory, path):
    value = historical.unseal(Path(directory) / 'registration.json',
        'ferrule.storage-demand.actual-http-transfer-registration.v1')
    layout = validate_transfer_layout(path)
    require(value['study_layout'] == layout and value['execution_path'] == value['binding_path'] == str(Path(path).resolve())
        and value['binding_sha256'] == value['execution_sha256'] == ct.sha(path), 'transfer layout/candidate changed')
    require(value['boot_id'] == maintained.boot_id() and value['trial_order'] == layout['trial_order']
        and value['order'] == layout['order'] and value['limits'] == layout['limits']
        and value['schedule'] == layout['schedule'], 'transfer clock/order/workload changed')
    require(value['owner_authorization'] is True and value['owner_note'] == TRANSFER_NOTE
        and value['eight_uuids'] == layout['candidate']['uuid_inventory']
        and value['selected_uuid'] == value['eight_uuids'][0], 'transfer owner/inventory changed')
    return value


@contextmanager
def transfer_adapter(path):
    path = Path(path).resolve()
    layout = validate_transfer_layout(path)
    def execution(selected=path):
        require(Path(selected).resolve() == path, 'unknown transfer execution path')
        return validate_transfer_layout(path)
    def candidate(selected=path):
        return validate_transfer_candidate(execution(selected))
    def case(mode):
        require(mode in layout['cases'], 'unknown transfer read/policy case')
        return copy.deepcopy(layout['cases'][mode])
    facade = SimpleNamespace(validate_execution=execution, validate_binding=candidate,
        raw_options=lambda mode: case(mode)['read_options'])
    def write(selected, value):
        if Path(selected).name == 'paired-final.json':
            value = dict(value, schema='ferrule.storage-demand.http-transfer-actual.v1',
                historical_resume_permitted=False)
        return ORIGINAL_WRITE(selected, value)
    changes = ((historical, 'validate_prepared', execution), (historical, 'validate_candidate', candidate),
        (historical, 'PROTOCOL_PATH', path), (historical, 'case_for_mode', case),
        (historical, 'trial_order', lambda: layout['trial_order']),
        (historical, 'expected_order', lambda: layout['order']),
        (planned, 'ENTRY', ENTRY), (planned, 'worker_contract', maintained.contract),
        (planned, 'read_ledger_assessment', transfer_ledger), (maintained, 'binding', facade),
        (maintained, 'registration', lambda directory: transfer_registration(directory, path)),
        (maintained, 'actual_adapter', lambda: transfer_adapter(path)),
        (maintained, 'ledger', transfer_ledger), (maintained, 'NOTE', TRANSFER_NOTE), (ct, 'write', write))
    originals = [(module, name, getattr(module, name)) for module, name, value in changes]
    try:
        for module, name, value in changes:
            setattr(module, name, value)
        yield
    finally:
        for module, name, value in reversed(originals):
            setattr(module, name, value)


def adapter(layout_path=None):
    return historical_adapter() if layout_path is None else transfer_adapter(layout_path)


def transfer_start(directory, path, granted):
    require(granted is True, 'renewed sole-GPU-owner authorization required')
    layout = validate_transfer_layout(path)
    directory = directory_check(directory, transfer=True).resolve()
    directory.mkdir(exist_ok=False)
    now = time.monotonic()
    value = dict(schema='ferrule.storage-demand.actual-http-transfer-registration.v1',
        started=maintained.ex.stamp(), pilot_deadline=now + layout['limits']['pilot_seconds'],
        boot_id=maintained.boot_id(), order=layout['order'], trial_order=layout['trial_order'],
        limits=layout['limits'], schedule=layout['schedule'], study_layout=layout,
        binding_path=str(Path(path).resolve()), binding_sha256=ct.sha(path),
        execution_path=str(Path(path).resolve()), execution_sha256=ct.sha(path),
        selected_uuid=layout['candidate']['uuid_inventory'][0], eight_uuids=layout['candidate']['uuid_inventory'],
        owner_authorization=True, owner_note=TRANSFER_NOTE,
        scheduled_blocks=4, scheduled_starts=8, scheduled_requests=96, scheduled_measured_qa=40,
        retries=0, replacements=0, profiling_captures=0, historical_resume_permitted=False)
    ct.write(directory / 'registration.json', dict(value, digest=ct.digest(value)))
    return dict(status='registered', first_arm=layout['trial_order'][0], directory=str(directory))


def transfer_main(args):
    if args.mode == 'plan':
        value = transfer_plan()
    else:
        require(args.layout is not None, '--layout required; no historical candidate reuse')
        validate_transfer_layout(args.layout)
        if args.mode == 'check':
            value = dict(status='checked', gpu_queries=0, server_runs=0)
        else:
            require(args.directory is not None, '--directory required')
            directory = directory_check(args.directory, transfer=True)
            if args.mode == 'start':
                value = transfer_start(directory, args.layout, args.sole_gpu_owner_granted)
            else:
                transfer_registration(directory, args.layout)
                with transfer_adapter(args.layout):
                    value = maintained.execute_next(directory) if args.mode == 'next' else maintained.finalize(directory)
                if args.mode == 'next':
                    print(json.dumps(dict(trial=value['trial'], assessment=value['assessment'], summary=value['summary'])), flush=True)
                    return 0 if value['assessment']['can_continue'] else 1
    print(json.dumps(value, sort_keys=True), flush=True)
    return 0 if value['status'] not in ('stopped', 'incomplete') else 1


if __name__ == '__main__':
    raise SystemExit(main())
