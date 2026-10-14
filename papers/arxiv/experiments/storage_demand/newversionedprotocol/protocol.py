#!/usr/bin/env python3
"""Preregister/check/bind the narrow v2 CPU protocol. There is NO run command."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
STORAGE = HERE.parent
sys.path.insert(0, str(STORAGE))
from runnerhelpers.evidence_v2 import (
    PHASE_COUNTS, build_schedule, ct, integer, require, validate_deadlines, validate_schedule,
)

PROTOCOL_PATH = HERE / 'protocol.json'
CASE_PATH = HERE / 'case-v2.json'
PREPARED_PATH = HERE / 'prepared-cpu.json'
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


def validate_prepared(path=PREPARED_PATH):
    prepared = ct.load(path)
    require(prepared.get('digest') == ct.digest({key: value for key, value in prepared.items() if key != 'digest'}),
            'CPU preparation digest mismatch')
    require(prepared['status'] == 'cpu_prepared_candidate_unbound_not_authorized'
            and prepared.get('candidate_binary') is None and prepared.get('sealed_final_candidate') is False,
            'invalid CPU-only qualification state')
    ct.check_pins(prepared['artifact_pins'])
    ct.check_pins(prepared['reference_pins'])
    validate_protocol(ct.load(PROTOCOL_PATH))
    validate_schedule(ct.load(HERE / 'requests-v2.json')['schedule'])
    require(prepared['gpu_queries'] == prepared['gpu_locks'] == prepared['gpu_runs'] == 0, 'CPU seal claims GPU activity')
    return prepared


def validate_quality_reports(reports, case):
    """Validate actual existing Rust result schemas; no boolean-only quality gate."""
    require(set(reports) == {'full40', 'fresh', 'cold'}, 'all three current quality gates required')
    for label, report in reports.items():
        schema = ('ferrule.numeric-bf16-full40-profile-gate.v1' if label == 'full40'
                  else 'ferrule.bf16-fresh-quality-result.v1')
        require(report.get('schema') == schema and report.get('passed') is True
                and report.get('complete') is True, 'quality failed/pending/unknown: ' + label)
        require(report.get('test_config') == case['quality_test_config'], 'same StorageMiddle64 config missing: ' + label)
        require(report.get('effective_config', {}).get('assertions_passed') is True,
                'effective factory assertions missing: ' + label)
        require(report.get('failures') == [] and report.get('quarantine') is False,
                'quality failures/quarantine missing or nonempty: ' + label)
        require(report.get('ordinary_no_observer_claim') is False and report.get('concurrent_scheduler_claim') is False,
                'unknown quality claim scope: ' + label)
        if label != 'full40':
            require(report.get('cold_owner_restart') is (label == 'cold'), 'fresh/cold owner order mismatch')
            cases = report.get('cases')
            require(isinstance(cases, list) and len(cases) == (1 if label == 'cold' else 3), 'fresh/cold case coverage')
            require(all(row.get('free_greedy_equal') is True and row.get('top1_failures') == [] for row in cases),
                    'actual fresh generated IDs failed/unknown')
        else:
            require(report.get('diagnostic_capture_only') is False and report.get('status') == 'passed', 'partial full40 is not PASS')
    require(reports['fresh'].get('oracle_manifest_sha256') == reports['cold'].get('oracle_manifest_sha256')
            and re.fullmatch(r'[0-9a-f]{64}', reports['fresh'].get('oracle_manifest_sha256') or ''), 'fresh/cold reference mismatch')
    require(re.fullmatch(r'[0-9a-f]{64}', reports['full40'].get('reference_manifest_sha256') or ''), 'full40 reference pin missing')
    return dict(status='pass_same_StorageMiddle64', performance_acceptance=False)


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
    binary = ct.validate_build(build, config)
    require(binary == str(ct.absolute(candidate['binary']).resolve())
            and candidate.get('binary_sha256') == build['binary_sha256'], 'CLI build/binary relation mismatch')
    runtime = ct.load(ct.absolute(candidate['runtime_pin_manifest']))
    ct.check_pins(runtime['pins'])
    require(runtime.get('unknown_nccl_enabled') is False, 'unknown NCCL build binding')
    archive = ct.load(ct.absolute(candidate['source_archive_manifest']))
    require(archive == build['sources_before'] == build['sources_after'], 'compiled source archive/build relation differs')
    help_receipt = ct.load(ct.absolute(candidate['serve_help_receipt']))
    require(help_receipt.get('binary_sha256') == build['binary_sha256']
            and help_receipt.get('status') == 'pass' and help_receipt.get('gpu_hidden') is True
            and help_receipt.get('argv_flags') == case['argv_after_binary'][2:], 'exact-binary CPU serve-help flag receipt missing')
    relation = ct.load(ct.absolute(candidate['quality_source_relation']))
    require(relation.get('status') == 'verified_same_compiled_production_sources'
            and relation.get('cli_binary_sha256') == build['binary_sha256']
            and relation.get('production_source_digest') == ct.digest(build['sources_before'])
            and relation.get('independent_reference') is True, 'test-to-CLI source/independent reference relation unverified')
    reports = {label: ct.load(ct.absolute(candidate[label])) for label in ('full40', 'fresh', 'cold')}
    validate_quality_reports(reports, case)
    for label in ('full40', 'fresh', 'cold'):
        require(relation.get('report_sha256', {}).get(label) == ct.sha(ct.absolute(candidate[label])), 'raw quality report relation missing')
    require(relation.get('cold_primary_report_sha256') == ct.sha(ct.absolute(candidate['fresh'])), 'cold primary receipt relation missing')
    for label in ('full40', 'fresh', 'cold'):
        for group in ('supervisor_receipts', 'test_source_manifests', 'test_executables'):
            path = relation.get(group, {}).get(label)
            require(isinstance(path, str) and str(ct.absolute(path)) in candidate['artifact_pins'], 'missing pinned quality producer relation: ' + group + '/' + label)
        require(ct.successful_process(ct.load(ct.absolute(relation['supervisor_receipts'][label]))), 'quality process custody/failure: ' + label)
        require(ct.load(ct.absolute(relation['test_source_manifests'][label])) == build['sources_before'], 'quality executable compiled-source manifest mismatch: ' + label)
    reference_paths = relation.get('reference_manifests', {})
    for label, reference_key in (('full40', 'reference_manifest_sha256'), ('fresh', 'oracle_manifest_sha256'), ('cold', 'oracle_manifest_sha256')):
        path = reference_paths.get(label)
        require(isinstance(path, str) and str(ct.absolute(path)) in candidate['artifact_pins'] and ct.sha(ct.absolute(path)) == reports[label][reference_key], 'independent reference manifest missing/mismatch: ' + label)
    require(relation.get('supervision') == 'authenticated_drained_all_three', 'quality custody unknown')
    return dict(schema='ferrule.paper.storage-demand.final-candidate.v2',
                status='quality_passed_cpu_prepared_not_authorized', candidate=candidate,
                protocol_sha256=ct.sha(PROTOCOL_PATH), case_sha256=ct.sha(CASE_PATH),
                sealed_final_candidate=True, gpu_authorized=False, gpu_runs_by_this_helper=0,
                no_launch_entry=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'seal-cpu', 'check', 'bind-candidate'))
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    ct.require_cpu_python()
    if args.mode == 'plan':
        value = dict(validation=validate_protocol(load_protocol()), build=build_requirements(), no_launch_entry=True)
    elif args.mode == 'seal-cpu':
        value = seal_cpu()
    elif args.mode == 'check':
        value = validate_prepared()
    else:
        require(args.candidate is not None and args.output is not None, 'candidate and exclusive new output required')
        require(args.output.parent.resolve() == HERE.resolve(), 'final candidate output must stay in newversionedprotocol')
        validate_prepared()
        value = prepare_candidate(ct.load(args.candidate))
        value['digest'] = ct.digest(value)
        ct.write(args.output, value)
    print(json.dumps(dict(status=value.get('status', 'planned'), no_launch_entry=True,
                          candidate_bound=value.get('sealed_final_candidate', False)), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
