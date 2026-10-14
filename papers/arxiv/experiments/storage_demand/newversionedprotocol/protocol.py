#!/usr/bin/env python3
"""Preregister/check/bind the narrow v2 CPU protocol. There is NO run command."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
STORAGE = HERE.parent
sys.path.insert(0, str(STORAGE))
from runnerhelpers.evidence_v2 import (
    PHASE_COUNTS, build_schedule, ct, integer, require, validate_deadlines, validate_schedule,
)

from runnerhelpers.build_receipts_v1 import ADAPTER, review_cli_build, validate_candidate_build
from runnerhelpers.quality_reports_v1 import validate_producer_final, validate_quality_reports

PROTOCOL_PATH = HERE / 'protocol.json'
CASE_PATH = HERE / 'case-v2.json'
PREPARED_PATH = HERE / 'prepared-cpu.json'
WORKFLOW_SCHEMA = 'ferrule.storage-demand.cpu-workflow-review.v1'
REVIEWED_REPLACEMENTS = (HERE / 'protocol.py', STORAGE / 'tests/test_protocol_v2.py')
REVIEW_DOCS = (HERE / 'BUILD_RECEIPTS_V1.md', HERE / 'PAIRED_READ_PROTOCOL_DRAFT_V1.md')
METADATA_LOG = ct.ROOT / 'target/validation/bf16-storage-middle64-preparation-20261014-01/cpu-storage-middle-metadata.log'
IMPORTS = (
    'papers/arxiv/experiments/contract.py', 'papers/arxiv/experiments/runner.py',
    'papers/arxiv/experiments/launch_guard.py', 'papers/arxiv/experiments/cli_audit.py',
    'papers/arxiv/experiments/telemetry.py', 'papers/arxiv/experiments/m7.json',
    'papers/arxiv/experiments/m7-device-exclusive.json',
    'papers/arxiv/experiments/tp2_resident/observer.py',
    'papers/arxiv/experiments/tp2_resident/sampling_v4.py',
    'papers/arxiv/experiments/tp2_resident/sampling_v2.py',
    'scripts/_profile_capture.py', 'scripts/_owned_subreaper.py',
    'scripts/test_vllm_qwen35.py', 'scripts/bench_ferrule_vllm.py',
)


def load_protocol():
    value = ct.load(PROTOCOL_PATH)
    validate_protocol(value)
    return value


def validate_protocol(value):
    require(value.get('schema') == 'ferrule.paper.storage-demand.pilot-protocol.v2', 'wrong protocol schema')
    require(value.get('trial_order') == ['L1', 'L2', 'L3', 'L4', 'L5']
            and value.get('trial_count') == 5 and value.get('retry_count') == 0, 'finite trial order/count drift')
    require(value.get('request_phases') == PHASE_COUNTS and value.get('request_ceiling_per_lifecycle') == 12
            and value.get('scheduled_requests') == 60 and value.get('scheduled_measured_qa') == 25,
            'request phase/count/denominator drift')
    require(value.get('candidate_binary') is None, 'preregistration must not pretend a candidate is sealed')
    limits = value['limits']
    validate_deadlines(limits)
    require(limits['target_lifetime_seconds'] <= 1560 and limits['shutdown_seconds'] <= 45,
            'existing run_target CLI limit exceeded')
    require(limits['worker_preidle_seconds'] + limits['target_lifetime_seconds']
            + limits['shutdown_seconds'] + limits['worker_finalization_seconds'] <= limits['supervisor_seconds'],
            'worker phases do not fit supervisor timeout')
    require(limits['coordinator_preidle_seconds'] + limits['supervisor_seconds']
            + limits['supervisor_call_overhead_seconds'] + limits['coordinator_postidle_seconds']
            + limits['coordinator_finalization_seconds'] <= limits['outer_lifecycle_seconds'],
            'coordinator phases do not fit literal outer deadline')
    require(value['custody']['no_drop_cache'] is True and value['custody']['no_foreign_kills'] is True,
            'passive/custody policy changed')
    return dict(status='pass', starts=5, requests=60, measured_qa=25,
                per_lifecycle_seconds=1440, pilot_seconds=7200)


def fresh_case(protocol):
    """Inherit historical argv controls only, NOT binary, quality, or execution."""
    historical = ct.load(STORAGE / 'case-middle.json')
    metadata = [json.loads(line.split('BF16_FULL40_METADATA_ONLY ', 1)[1])
                for line in METADATA_LOG.read_text().splitlines() if 'BF16_FULL40_METADATA_ONLY ' in line]
    require(len(metadata) == 1, 'exactly one current StorageMiddle64 metadata receipt required')
    config = metadata[0]['test_config']
    require(config['name'] == 'storage-middle64'
            and config['numeric_precision'] == 'Bf16ModelMixedF32Accumulate'
            and config['capacity_and_options']['max_experts'] == 64
            and config['capacity_and_options']['max_bytes'] == 335544320
            and config['capacity_and_options']['scheduler']['max_decode_batch'] == 2,
            'metadata is not current StorageMiddle64')
    return dict(
        schema='ferrule.paper.storage-demand.tp1-case.v2',
        case_id='storage-middle64-repeated-qa-20261014-v2',
        status='prepared_cpu_only_candidate_unbound_not_authorized',
        binary=None, binary_sha256=None, quality_receipt=None,
        model=historical['model'], topology=historical['topology'],
        argv_after_binary=historical['argv'][1:],
        environment=dict(historical['environment'], FERRULE_LOG=protocol['logging']['FERRULE_LOG'],
                         FERRULE_LOG_FORMAT='json'),
        limits=protocol['limits'], request_ceiling=12,
        workload=dict(controls=historical['workload']['controls'], phase_counts=PHASE_COUNTS,
                      requests_per_lifecycle=12, measured_qa_per_lifecycle=5,
                      qa_label='same-server repeated QA, not first-after-Ready or storage-cold'),
        quality_test_config=config, effective_metadata_config=metadata[0]['effective_config'],
        quality_config_scope='current CPU factory metadata; not CLI binary linkage, live CUDA admission or math PASS',
        host_expert_retention=0, physical_host_cap_bytes=None,
        cache_bytes_include_scratch=True, pagecache_policy='uncontrolled; no cache dropping',
        historical_controls=dict(path=str(STORAGE / 'case-middle.json'), sha256=ct.sha(STORAGE / 'case-middle.json'),
                                 inherited='argv_after_binary, topology and frozen canary controls only; no legacy binary/quality acceptance'),
        blocked=['No new release CLI binary/build receipt is bound',
                 'Same-source StorageMiddle64 full40 + retained fresh + cold-owner GPU quality must pass first',
                 'Renewed sole-GPU-owner authorization and reviewed TP1 worker registration are still required'],
        gpu_queries=0, gpu_locks=0, gpu_runs=0,
    )


def build_requirements():
    config = copy.deepcopy(ct.BASE)
    config['build_target'] = 'target/validation/storage-middle64-v2-current-release-20261014-01'
    return dict(
        schema='ferrule.storage-demand.new-baseline-build-requirements.v1',
        status='requirements_only_not_built',
        build_config=config, argv=ct.build_argv(config), environment=ct.build_environment(config),
        build_api='papers/arxiv/experiments/runner.py:build(config,new_directory)',
        validation_api='papers/arxiv/experiments/contract.py:validate_build(receipt,config)',
        additional_requirements=[
            'Fresh CUDA-only release CLI artifact: opt3/fat-LTO/codegen1/no debug assertions/no cpu-reference; no historic binary reuse',
            'Current production sources before/after + compiled-source archive and manifest + native archives/dependencies and compiler/CUDA/CUTLASS/runtime inventories',
            'Immutable staged binary hash; CPU-hidden version/help/serve-help receipts for that exact binary',
            'Compiled token enqueue, HTTP request DEBUG and resource summary INFO code paths; no production instrumentation edits by this protocol',
            'Runtime/driver library pins and actual loaded mappings at Ready/after requests; no unknown NCCL binding for TP1',
            'Same effective StorageMiddle64/maxdecode2 Mixed config and independent reference pins; sealed numerical-test-source to CLI-build-source relation',
            'New instrumented baseline only, not a claimed zero-overhead comparison to -02',
        ],
    )


def local_pins():
    paths = [ct.ROOT / path for path in IMPORTS]
    paths += [STORAGE / name for name in ('case-middle.json', 'canaries.json', 'requests.json', 'checkpoint-metadata.json', 'run_middle.py')]
    paths += [METADATA_LOG, PROTOCOL_PATH]
    paths += sorted((STORAGE / 'runnerhelpers').glob('*.py'))
    paths += sorted(HERE.glob('*.py')) + sorted((STORAGE / 'tests').glob('*.py'))
    paths += [HERE / 'README.md', ct.ROOT / 'docs/STORAGE_DEMAND_PILOT_V2.md']
    return ct.inventory(paths)


def seal_cpu():
    ct.require_cpu_python()
    protocol = load_protocol()
    require(not CASE_PATH.exists() and not PREPARED_PATH.exists(), 'CPU seal is exclusive; create a new version, do not overwrite')
    refs = ct.load(STORAGE / 'canaries.json')
    ct.check_pins(refs['reference_pins'])
    schedule = build_schedule(refs['rows'])
    case = fresh_case(protocol)
    ct.write(CASE_PATH, case)
    ct.write(HERE / 'requests-v2.json', dict(schedule=schedule, total_per_lifecycle=12, total_pilot=60))
    ct.write(HERE / 'new-baseline-build-requirements.json', build_requirements())
    pins = local_pins()
    pins.update(ct.inventory([CASE_PATH, HERE / 'requests-v2.json', HERE / 'new-baseline-build-requirements.json']))
    prepared = dict(
        schema='ferrule.paper.storage-demand.cpu-prepared.v2',
        status='cpu_prepared_candidate_unbound_not_authorized', protocol_sha256=ct.sha(PROTOCOL_PATH),
        artifact_pins=pins, reference_pins=refs['reference_pins'],
        candidate_binary=None, candidate_quality='pending', sealed_final_candidate=False,
        gpu_queries=0, gpu_locks=0, gpu_runs=0, tensor_payload_bytes_read=0,
        observation_results=None, no_launch_entry=True,
    )
    prepared['digest'] = ct.digest(prepared)
    ct.write(PREPARED_PATH, prepared)
    return prepared


def baseline_preparation():
    """Review only the named implementation replacement; preserve every other pin."""
    original = ct.load(PREPARED_PATH)
    require(original.get('schema') == 'ferrule.paper.storage-demand.cpu-prepared.v2'
            and original.get('digest') == ct.digest({k: v for k, v in original.items() if k != 'digest'}),
            'original CPU preparation digest/schema mismatch')
    require(original.get('status') == 'cpu_prepared_candidate_unbound_not_authorized'
            and original.get('candidate_binary') is None and original.get('sealed_final_candidate') is False
            and original.get('gpu_queries') == original.get('gpu_locks') == original.get('gpu_runs') == 0,
            'original CPU-only qualification state changed')
    replaced = {str(p) for p in REVIEWED_REPLACEMENTS}
    ct.check_pins({p: h for p, h in original['artifact_pins'].items() if p not in replaced})
    ct.check_pins(original['reference_pins'])
    require(original['protocol_sha256'] == ct.sha(PROTOCOL_PATH), 'frozen protocol changed')
    validate_protocol(ct.load(PROTOCOL_PATH))
    validate_schedule(ct.load(HERE / 'requests-v2.json')['schedule'])
    return original


def workflow_review():
    """Explicit CPU-only revision: same frozen protocol, newly reviewed helper pins."""
    original = baseline_preparation()
    pins = ct.inventory([ct.absolute(p) for p in original['artifact_pins']])
    pins.update(ct.inventory([PREPARED_PATH, *REVIEW_DOCS,
        *sorted((STORAGE / 'runnerhelpers').glob('*.py')), *sorted(HERE.glob('*.py')),
        *sorted((STORAGE / 'tests').glob('*.py'))]))
    replacements = {str(p): dict(previous=original['artifact_pins'][str(p)], current=ct.sha(p))
                    for p in REVIEWED_REPLACEMENTS}
    return dict(schema=WORKFLOW_SCHEMA, status='cpu_prepared_candidate_unbound_not_authorized',
        adapter_version=ADAPTER, original_prepared_path=str(PREPARED_PATH),
        original_prepared_sha256=ct.sha(PREPARED_PATH), protocol_sha256=ct.sha(PROTOCOL_PATH),
        reviewed_replacements=replacements, artifact_pins=pins, reference_pins=original['reference_pins'],
        candidate_binary=None, candidate_quality='pending', sealed_final_candidate=False,
        gpu_queries=0, gpu_locks=0, gpu_runs=0, tensor_payload_bytes_read=0, no_launch_entry=True)


def validate_prepared(path=PREPARED_PATH):
    prepared = ct.load(path)
    require(prepared.get('digest') == ct.digest({key: value for key, value in prepared.items() if key != 'digest'}),
            'CPU preparation digest mismatch')
    if prepared.get('schema') == WORKFLOW_SCHEMA:
        require({k: v for k, v in prepared.items() if k != 'digest'} == workflow_review(),
                'reviewed CPU workflow changed; explicit new seal required')
        return prepared
    require(prepared.get('schema') == 'ferrule.paper.storage-demand.cpu-prepared.v2', 'unknown CPU preparation schema')
    require(prepared['status'] == 'cpu_prepared_candidate_unbound_not_authorized'
            and prepared.get('candidate_binary') is None and prepared.get('sealed_final_candidate') is False,
            'invalid CPU-only qualification state')
    ct.check_pins(prepared['artifact_pins'])
    ct.check_pins(prepared['reference_pins'])
    validate_protocol(ct.load(PROTOCOL_PATH))
    validate_schedule(ct.load(HERE / 'requests-v2.json')['schedule'])
    require(prepared['gpu_queries'] == prepared['gpu_locks'] == prepared['gpu_runs'] == 0, 'CPU seal claims GPU activity')
    return prepared


def review_build(path, case=None):
    """Read an explicitly selected build descriptor without binding or writing."""
    value = ct.load(path)
    require(value.get('schema') == 'ferrule.storage-demand.cpu-build-review-input.v1', 'unknown CPU build review input')
    facts = review_cli_build(value, ct.load(ct.absolute(value['build_receipt'])),
                             ct.load(CASE_PATH) if case is None else case)
    return dict(schema='ferrule.storage-demand.cpu-build-review.v1',
                status='build_evidence_reviewed_quality_pending_not_authorized',
                build_adapter=facts['adapter'], raw_schema=facts['raw_schema'],
                binary_sha256=facts['binary_sha256'], source_digest=facts['source_digest'],
                coverage=facts['coverage'], quality_approved=False, sealed_final_candidate=False,
                gpu_authorized=False, gpu_queries=0, gpu_locks=0, gpu_runs=0, no_launch_entry=True)


def prepare_candidate(candidate, case=None):
    """Read-only final-candidate validation. Called only after owner supplies quality."""
    case = ct.load(CASE_PATH) if case is None else case
    require(candidate.get('schema') == 'ferrule.storage-demand.candidate-binding.v2'
            and candidate.get('status') == 'quality_passed_cpu_bind_requested', 'unknown candidate qualification state')
    require(candidate.get('protocol_sha256') == ct.sha(PROTOCOL_PATH), 'candidate protocol binding changed')
    ct.check_pins(candidate.get('artifact_pins'))
    for key in ('binary', 'build_receipt', 'runtime_pin_manifest', 'source_archive', 'source_archive_manifest',
                'serve_help_receipt', 'full40', 'fresh', 'cold', 'quality_source_relation'):
        require(str(ct.absolute(candidate.get(key, ''))) in candidate['artifact_pins'], 'candidate required artifact unpinned: ' + key)
    build = ct.load(ct.absolute(candidate['build_receipt']))
    config = ct.load(HERE / 'new-baseline-build-requirements.json')['build_config']
    facts = validate_candidate_build(candidate, build, config, case)
    binary = facts['binary']
    require(binary == str(ct.absolute(candidate['binary']).resolve())
            and candidate.get('binary_sha256') == facts['binary_sha256'], 'CLI build/binary relation mismatch')
    runtime = ct.load(ct.absolute(candidate['runtime_pin_manifest']))
    ct.check_pins(runtime['pins'])
    require(runtime.get('unknown_nccl_enabled') is False, 'unknown NCCL build binding')
    if facts['adapter'] != ADAPTER:
        archive = ct.load(ct.absolute(candidate['source_archive_manifest']))
        require(archive == facts['sources'] == build['sources_after'], 'compiled source archive/build relation differs')
        help_receipt = ct.load(ct.absolute(candidate['serve_help_receipt']))
        require(help_receipt.get('binary_sha256') == facts['binary_sha256']
                and help_receipt.get('status') == 'pass' and help_receipt.get('gpu_hidden') is True
                and help_receipt.get('argv_flags') == case['argv_after_binary'][2:], 'exact-binary CPU serve-help flag receipt missing')
    else:
        require(all(runtime['pins'].get(p) == h for p, h in facts['runtime_pins'].items()),
                'run runtime pins omit CPU-resolved libraries')
    relation = ct.load(ct.absolute(candidate['quality_source_relation']))
    require(relation.get('status') == 'verified_same_compiled_production_sources'
            and relation.get('cli_binary_sha256') == facts['binary_sha256']
            and relation.get('production_source_digest') == facts['source_digest']
            and relation.get('independent_reference') is True, 'test-to-CLI source/independent reference relation unverified')
    reports = {label: ct.load(ct.absolute(candidate[label])) for label in ('full40', 'fresh', 'cold')}
    validate_quality_reports(reports, case)
    for label in ('full40', 'fresh', 'cold'):
        require(relation.get('report_sha256', {}).get(label) == ct.sha(ct.absolute(candidate[label])), 'raw quality report relation missing')
    require(relation.get('cold_primary_report_sha256') == ct.sha(ct.absolute(candidate['fresh'])), 'cold primary receipt relation missing')
    for label in ('full40', 'fresh', 'cold'):
        for group in ('supervisor_receipts', 'test_source_manifests', 'test_executables', 'authenticated_launches', 'producer_finals'):
            path = relation.get(group, {}).get(label)
            require(isinstance(path, str) and str(ct.absolute(path)) in candidate['artifact_pins'], 'missing pinned quality producer relation: ' + group + '/' + label)
        custody = ct.load(ct.absolute(relation['supervisor_receipts'][label]))
        require(ct.successful_process(custody), 'quality process custody/failure: ' + label)
        manifest = ct.load(ct.absolute(relation['test_source_manifests'][label]))
        test_sources = manifest.get('original_source_pins', manifest)
        require(test_sources == facts['sources'], 'quality executable compiled-source manifest mismatch: ' + label)
        executable = str(ct.absolute(relation['test_executables'][label]))
        launch = ct.load(ct.absolute(relation['authenticated_launches'][label]))
        identity = launch.get('identity', {})
        require(all(isinstance(identity.get(role), dict)
                    and all(integer(identity[role].get(key), 1) for key in ('pid', 'birth_ticks', 'ppid'))
                    for role in ('helper', 'worker')), 'quality producer PID/birth/parent identity missing: ' + label)
        launch_argv = launch.get('argv')
        require(isinstance(launch_argv, list) and launch_argv, 'quality producer argv missing: ' + label)
        require(launch.get('binary_sha256') == ct.sha(executable)
                and launch_argv[0] == executable
                and launch.get('exec_preserves_authenticated_root_pid_birth') is True
                and launch.get('all_eight_uuid_locks_validated') is True
                and identity.get('launch_id') == custody.get('handshake', {}).get('launch_id')
                and identity.get('helper') == custody.get('handshake', {}).get('helper_identity')
                and identity.get('worker') == custody.get('custody', {}).get('root_identity'),
                'quality actual producer executable/custody relation missing: ' + label)
        final = ct.load(ct.absolute(relation['producer_finals'][label]))
        validate_producer_final(final, label, str(ct.absolute(candidate[label])), ct.sha(ct.absolute(candidate[label])),
                                executable, ct.sha(executable), facts['source_digest'], launch_argv)
    reference_paths = relation.get('reference_manifests', {})
    for label, reference_key in (('full40', 'reference_manifest_sha256'), ('fresh', 'oracle_manifest_sha256'), ('cold', 'oracle_manifest_sha256')):
        path = reference_paths.get(label)
        require(isinstance(path, str) and str(ct.absolute(path)) in candidate['artifact_pins'] and ct.sha(ct.absolute(path)) == reports[label][reference_key], 'independent reference manifest missing/mismatch: ' + label)
    require(relation.get('supervision') == 'authenticated_drained_all_three', 'quality custody unknown')
    return dict(schema='ferrule.paper.storage-demand.final-candidate.v2',
                status='quality_passed_cpu_prepared_not_authorized', candidate=candidate,
                protocol_sha256=ct.sha(PROTOCOL_PATH), case_sha256=ct.sha(CASE_PATH),
                sealed_final_candidate=True, gpu_authorized=False, gpu_runs_by_this_helper=0,
                build_adapter=facts['adapter'], build_evidence_coverage=facts['coverage'],
                no_launch_entry=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'seal-cpu', 'seal-workflow', 'check', 'review-build', 'bind-candidate'))
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--build-review', type=Path)
    parser.add_argument('--cpu-prepared', type=Path, default=PREPARED_PATH)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    ct.require_cpu_python()
    if args.mode == 'plan':
        value = dict(validation=validate_protocol(load_protocol()), build=build_requirements(), no_launch_entry=True)
    elif args.mode == 'seal-cpu':
        value = seal_cpu()
    elif args.mode == 'seal-workflow':
        require(args.output is not None and args.output.parent.resolve() == HERE.resolve()
                and not args.output.exists(), 'exclusive new workflow output inside newversionedprotocol required')
        value = workflow_review()
        value['digest'] = ct.digest(value)
        ct.write(args.output, value)
    elif args.mode == 'check':
        value = validate_prepared(args.cpu_prepared)
    elif args.mode == 'review-build':
        require(args.build_review is not None and args.output is None, 'read-only review requires --build-review and no --output')
        value = review_build(args.build_review)
    else:
        require(args.candidate is not None and args.output is not None, 'candidate and exclusive new output required')
        require(args.output.parent.resolve() == HERE.resolve(), 'final candidate output must stay in newversionedprotocol')
        validate_prepared(args.cpu_prepared)
        value = prepare_candidate(ct.load(args.candidate))
        value['digest'] = ct.digest(value)
        ct.write(args.output, value)
    if args.mode == 'review-build':
        print(json.dumps(value, sort_keys=True))
        return 0
    print(json.dumps(dict(status=value.get('status', 'planned'), no_launch_entry=True,
                          candidate_bound=value.get('sealed_final_candidate', False)), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
