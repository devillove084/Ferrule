#!/usr/bin/env python3
"""One failed-study publication, not a general curator: reads this directory only."""
from __future__ import annotations

import json
from pathlib import Path
import re

from check import load, no_credentials, regular, require, sha, verify

HERE = Path(__file__).resolve().parent
SCOPE = HERE.parent
RUN = 'artifacts-current-20261015-01'
ARM = RUN + '/P1-blocking/'
V1 = ['README.md', '__init__.py', 'analysis.py', 'build_adapter.py', 'cases.json',
      'cpu-sealed.json', 'cpu-tests.log', 'cpu-validation.json', 'oracles.json', 'order.json',
      'profile_audit.py', 'protocol.json', 'protocol.py', 'quality.py', 'requests.json',
      'run_paired.py', 'source-cut.json', 'stage-manifest.json', 'test_paired_read_v1.py', 'test_quality_v1.py']
V2 = ['EXECUTION_V2.md', 'RESULT_V2.md', 'actual-binding-v2.json', 'actual-build-review-v2.json',
      'actual-stage-manifest-v2.json', 'cpu-tests-v2.log', 'cpu-validation-v2.json',
      'current_binding_v2.py', 'execution-sealed-v2.json', 'run_actual_v2.py',
      'test_actual_v2.py', 'test_stopped_closeout_v2.py']
V3 = ['plan_v3.py', 'protocol-v3.json', 'cases-v3.json', 'PLAN_V3.md', 'test_plan_v3.py',
      'cpu-sealed-v3.json', 'v2-configuration-rejection-v3.json', 'cpu-validation-v3.json']
PROJECTIONS = ['study.json', 'supervision.json', 'closure.json', 'nfs-client.json', 'custody-idle.json', 'bindings.json']
PUB = ['publication_v1/' + name for name in ['README.md', 'check.py', 'curate.py',
       'startup.log', *PROJECTIONS, 'STAGE_FILES.json', 'manifest.json', 'test_publication.py', 'stage.py', 'STAGE_FILES.tsv']]


def pick(value, keys):
    return {key: value[key] for key in keys}


def read(name):
    return load(regular(SCOPE, name))


def write(name, value):
    with (HERE / name).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def projected(name, sources, data, omitted):
    value = dict(schema='ferrule.storage-demand.redacted-publication-projection.v1',
        redacted_projection=True, replaces_original_receipt=False,
        original_sources=[dict(path=path, original_sha256=sha(regular(SCOPE, path))) for path in sources],
        omitted_fields=omitted, authentication_scope='Recorded outcome projection; original private authentication is not replayed', data=data)
    no_credentials(value)
    write(name, value)


def identity(value):
    return pick(value, ['pid', 'birth_ticks', 'ppid'])


def idle_projection(value):
    result = pick(value, ['schema', 'status', 'all_eight_idle', 'deadline_seconds', 'required_consecutive',
        'interval_seconds', 'zero_memory_mib', 'zero_utilization_percent', 'no_compute_apps_required',
        'physical_owner_release', 'mode', 'authenticated_echild', 'recorded_owners_absent',
        'accepted_sample_indices', 'consecutive_idle', 'elapsed_seconds'])
    result['samples'] = []
    for sample in value['samples']:
        rows = []
        for line in sample['gpu'].splitlines():
            index, physical_id, memory, utilization = [part.strip() for part in line.split(',')]
            require(re.fullmatch(r'\d+ MiB', memory) and re.fullmatch(r'\d+ %', utilization), 'idle units unknown')
            rows.append(dict(index=int(index), memory_used_mib=int(memory.split()[0]), utilization_percent=int(utilization.split()[0])))
        result['samples'].append(dict(**pick(sample, ['monotonic_started', 'monotonic_finished', 'recorded_unix_ns',
            'classification', 'consecutive_idle']), gpu_rows=rows,
            compute_application_count=len([line for line in sample['apps'].splitlines() if line.strip()])))
    return result


def build():
    require(not (HERE / 'manifest.json').exists(), 'publication exclusive; never replace projections/manifest')
    old68 = read('protocol-v3.json')['historical_preservation']['artifact_pins']
    require(len(old68) == 68, 'frozen preservation scope changed')
    # Preserve the original scope, not subsequently added integration/CPU files.
    original_names = [Path(path).relative_to(SCOPE).as_posix() for path in old68] + V3
    require(len(original_names) == len(set(original_names)) == 76, 'original preservation scope changed')
    preserved = {}
    for name in sorted(original_names):
        path = regular(SCOPE, name)
        preserved[name] = dict(bytes=path.stat().st_size, original_sha256=sha(path))
    for path, pin in old68.items():
        name = Path(path).relative_to(SCOPE).as_posix()
        require(preserved[name]['original_sha256'] == pin, 'preexisting frozen receipt changed')
    # Credential values are read only to reject accidental publication, never serialized.
    private = read(ARM + 'worker-launch.json')
    private_literals = []
    def credentials(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key.lower() in ('worker_token', 'nonce', 'launch_id') and isinstance(child, str):
                    private_literals.append(child)
                credentials(child)
        elif isinstance(value, list):
            for child in value:
                credentials(child)
    credentials(private)
    sources = {name: read(name) for name in [RUN + '/paired-final.json', RUN + '/closeout.json',
        ARM + 'lifecycle.json', ARM + 'supervisor.json', ARM + 'read-ledger-assessment.json',
        RUN + '/final-lock-release.json', RUN + '/final-lock-reprobe.json']}
    final, closeout = sources[RUN + '/paired-final.json'], sources[RUN + '/closeout.json']
    lifecycle = sources[ARM + 'lifecycle.json']
    supervisor = sources[ARM + 'supervisor.json']
    rows = []
    for row in final['lifecycles']:
        summary = pick(row['summary'], ['scheduled', 'issued', 'terminal', 'successful', 'failed', 'censored',
            'unissued', 'conservation', 'counts', 'by_phase', 'reasons', 'measured_qa', 'denominator_policy'])
        rows.append(dict(trial=row['trial'], block=row['block'], mode=row['mode'], summary=summary,
            assessment=pick(row['assessment'], ['can_continue', 'stop_reasons']) if 'assessment' in row else None,
            status=row.get('status'), stop_reason=row.get('stop_reason')))
    data = dict(**pick(final, ['schema', 'status', 'order', 'scheduled_blocks', 'scheduled_starts', 'scheduled_requests',
        'scheduled_measured_qa', 'started', 'ended', 'pilot_deadline', 'retries', 'replacements', 'extra_pilot_lifecycles',
        'profile_lifecycles', 'actual_issued_lifecycles', 'actual_completed_lifecycles', 'scientific_acceptance']),
        binding_sha256=final['candidate_sha256'], execution_sha256=final['execution_sha256'],
        denominators=final['analysis']['denominators'], measured_qa=final['analysis']['measured_qa'],
        paired_blocks=final['analysis']['blocks'], paired_aggregate=final['analysis']['aggregate'], lifecycles=rows,
        http_requests=closeout['actual_http_requests'], actual_http_token_id_join=closeout['actual_http_token_id_join'],
        client_e2e_seconds=closeout['client_e2e_seconds'], client_ttft_seconds=closeout['client_ttft_seconds'],
        qualified_goodput_tokens_per_second=closeout['quality_qualified_goodput_tokens_per_second'],
        pipeline_positive_worker_read=closeout['pipeline_positive_worker_read'], pipeline_wasted_bytes=closeout['pipeline_wasted_bytes'],
        pipeline_host_credit=closeout['pipeline_host_credit'], permanent_stop=pick(final['stop'], ['status', 'reason', 'recorded', 'retries_permitted', 'remaining_slots']))
    projected('study.json', [RUN + '/paired-final.json', RUN + '/closeout.json'], data,
        ['Nested raw lifecycle/observer environments', 'physical UUIDs, private paths, launch material', 'raw artifact inventories'])
    custody = supervisor['custody']
    filtered = pick(custody, ['status', 'method', 'containment_verified', 'echild', 'errors', 'reporting_errors'])
    filtered.update(root_identity=identity(custody['root_identity']), helper_identity=identity(custody['helper_identity']),
        entries=[dict(identity=identity(row['identity']), **pick(row, ['role', 'signals', 'reaped', 'exit_code',
            'first_observed_at', 'reaped_at'])) for row in custody['entries']])
    projected('supervision.json', [ARM + 'supervisor.json'], dict(**pick(supervisor,
        ['returncode', 'timed_out', 'wall_seconds', 'cleanup_errors', 'cleanup_scope', 'descendant_custody',
         'gpu_quiescence', 'stdout_truncated', 'stderr_truncated']), custody=filtered),
        ['Raw stdout/stderr channels', 'Handshake/launch identifiers and authentication material', 'boot ID'])
    closure = dict(status=lifecycle['status'], error=lifecycle['error'], cleanup_error=lifecycle['cleanup_error'],
        capture=pick(lifecycle['capture'], ['schema', 'started_monotonic', 'started_unix_ns', 'pid', 'returncode',
                                          'ended_monotonic', 'ended_unix_ns', 'descendant_custody']),
        shutdown_summary_present=bool(lifecycle['closure']['attempts']), owner_resource_ledger='unknown',
        resource_contract_complete=final['resource_contract_complete'],
        universal_final_release_verified=final['universal_final_release_verified'],
        read_ledger=pick(sources[ARM + 'read-ledger-assessment.json'], ['can_continue', 'stop_reason', 'final_read_ledger', 'resource_contract']),
        cli_ready=closeout['cli_ready'], actual_ready_runtime_maps=closeout['actual_ready_runtime_maps'],
        nvtx_runtime_spec=closeout['nvtx_runtime_spec'], nvtx_process_observation=closeout['nvtx_actual_process_environment_observation'],
        physical_release_certificate=False)
    projected('closure.json', [ARM + 'lifecycle.json', ARM + 'read-ledger-assessment.json', RUN + '/closeout.json'], closure,
        ['Capture/process environments', 'Raw process mappings/observer records', 'Private authentication and host paths'])
    nfs_sources = [ARM + name + '-mountstats.json' for name in ('prelaunch', 'shutdown-pre', 'shutdown-post')]
    samples = []
    for name in nfs_sources:
        sample = read(name)
        mount = sample['selected_mount']
        redacted_identity = json.dumps([sample['boot_id'], sample['mount_namespace'], mount['device'], mount['mountpoint'],
                                      mount['fstype'], mount['statvers']], sort_keys=True).encode()
        import hashlib
        public_mount = dict(**pick(mount, ['fstype', 'statvers', 'age_seconds', 'counters', 'errors']),
                            redacted_identity_sha256=hashlib.sha256(redacted_identity).hexdigest())
        samples.append(dict(**pick(sample, ['label', 'acquisition_begin', 'acquisition_end', 'captured_unix_ns',
            'captured_monotonic', 'acquisition_status', 'shared_client_traffic', 'owned_server_physical_bytes',
            'physical_nas_backend_bytes']), selected_mount=public_mount))
    projected('nfs-client.json', nfs_sources + [ARM + 'lifecycle.json'], dict(scope='Shared NFS client counters only; not server-attributable NAS disk/network IO',
        shared_client_traffic=True, physical_nas_backend_bytes=None, ready_startup_window=None,
        http_request_windows=[], negative_deltas_clamped=False, samples=samples, shutdown_window=lifecycle['shutdown_nfs']),
        ['Raw mountstats/mount lines', 'Export/server device and mountpoint paths', 'Mount namespace/boot/reader identities'])
    idle_names = dict(**{'coordinator-post-idle': ARM + 'coordinator-post-idle.json',
                       'final-all-eight-idle': RUN + '/final-all-eight-idle.json'})
    projected('custody-idle.json', list(idle_names.values()) + [RUN + '/final-lock-release.json', RUN + '/final-lock-reprobe.json'],
        dict(observations={label: idle_projection(read(name)) for label, name in idle_names.items()},
             final_lock_release=pick(sources[RUN + '/final-lock-release.json'], ['context_exited', 'all_eight_local_locks_released',
                 'recorded_owners_absent', 'authenticated_echild', 'owner_resource_ledger', 'physical_release_certificate', 'study_permanently_stopped']),
             final_lock_reprobe=pick(sources[RUN + '/final-lock-reprobe.json'], ['all_eight_nonblocking_reacquired',
                 'recorded_owners_absent', 'context_exits_after_receipt', 'workloads', 'not_a_resume'])),
        ['Physical GPU UUIDs and raw telemetry strings', 'Raw flock FDs/device/inode and coordinator identifiers'])
    b2, review, p1, p3 = (read(name) for name in ('actual-binding-v2.json', 'actual-build-review-v2.json',
                                               'protocol.json', 'protocol-v3.json'))
    projected('bindings.json', ['actual-binding-v2.json', 'actual-build-review-v2.json', 'execution-sealed-v2.json',
        'protocol.json', 'protocol-v3.json', 'cpu-sealed-v3.json'], dict(
        v1=dict(protocol_id=p1['id'], protocol_sha256=sha(regular(SCOPE, 'protocol.json')),
                cases_sha256=sha(regular(SCOPE, 'cases.json')), cpu_seal_sha256=sha(regular(SCOPE, 'cpu-sealed.json')),
                order_sha256=sha(regular(SCOPE, 'order.json')), requests_sha256=sha(regular(SCOPE, 'requests.json')),
                oracles_sha256=sha(regular(SCOPE, 'oracles.json')), quality_tolerances=p1['quality_tolerances']),
        v2=dict(**pick(b2, ['schema', 'status', 'adapter', 'cli_binary_sha256', 'source_digest', 'protocol_sha256',
                           'cases_sha256', 'cli_serving_accepted', 'old_adapter_acceptance_transferred']),
                binding_sha256=sha(regular(SCOPE, 'actual-binding-v2.json')),
                execution_sha256=sha(regular(SCOPE, 'execution-sealed-v2.json')),
                build_review_sha256=sha(regular(SCOPE, 'actual-build-review-v2.json')),
                source_archive_sha256=review['source_archive_sha256'], source_members=review['source_members'],
                separate_quality_gate_reports=len(review['gates']), build_custody=review['build_custody'],
                cli_qualification=review['cli_qualification'], source_native_runtime_relation=review['source_native_runtime_relation']),
        v3=dict(protocol_id=p3['id'], protocol_sha256=sha(regular(SCOPE, 'protocol-v3.json')),
                cases_sha256=sha(regular(SCOPE, 'cases-v3.json')), cpu_seal_sha256=sha(regular(SCOPE, 'cpu-sealed-v3.json')),
                candidate_binary=p3['candidate_binary'], candidate_source_cut=p3['candidate_source_cut'],
                actual_lifecycles=p3['actual_lifecycles'], gpu_authorized=False,
                replacement_or_resume=p3['historical_study']['replacement_or_resume'],
                candidate_prerequisites=p3['candidate_prerequisites'], corrective_flags=p3['corrective_flags'])),
        ['Huge external source/dependency/runtime pin path inventories', 'CLI/native/archive/map bytes',
         'Private raw producer/launch channels; old PASS cannot promote new V3'])
    startup_source = regular(SCOPE, ARM + 'server.log')
    startup = startup_source.read_text()
    require('prepared source profile changed' in startup and not any(value in startup for value in private_literals),
            'startup is not the expected safe small failure log')
    require(not re.search(r'worker_token|FERRULE_PAPER_WORKER_TOKEN|password|api_key|authorization', startup, re.I),
            'credential-like assignment in startup')
    require(not re.search(r'/(?:home|root|mnt|proc|opt)/', startup), 'private path in startup; explicit reviewed redaction required')
    with (HERE / 'startup.log').open('xb') as stream:
        stream.write(startup_source.read_bytes())
    groups = dict(initial_preregistration_v1=V1, binding_failed_study_v2=V2,
                  prospective_unbound_v3=V3, curated_publication_v1=PUB,
                  v3_execution_integration=['current_binding_v3.py', 'run_v3.py', 'test_run_v3.py',
                                            'EXECUTION_V3.md', 'cpu-validation-integration-v3.json'])
    paths = [path for group in groups.values() for path in group]
    write('STAGE_FILES.json', dict(schema='ferrule.storage-demand.safe-stage-whitelist.v1',
        project_prefix='papers/arxiv/experiments/storage_demand/paired_read_v1', path_basis='paired_read_v1_relative',
        size_limit_bytes=10 << 20, groups=groups, files=paths,
        policy='Stage only these literal regular UTF-8 text paths, per group if desired. No folder glob, raw study/worker launch, target binary, archive or outside path.'))
    from stage import render
    with (HERE / 'STAGE_FILES.tsv').open('x') as stream:
        stream.write(render(load(HERE / 'STAGE_FILES.json'), 'tsv'))
    records = {}
    for name in paths:
        if name == 'publication_v1/manifest.json':
            continue
        path = regular(SCOPE, name)
        content = path.read_bytes()
        text = content.decode('utf-8')
        require(b'\0' not in content and not any(value in text for value in private_literals),
                'binary or private launch credential leaked into staged path: ' + name)
        if path.suffix == '.json':
            no_credentials(json.loads(text))
        records[name] = dict(bytes=len(content), publication_sha256=sha(path))
    manifest = dict(schema='ferrule.storage-demand.curated-publication-manifest.v1',
        redacted_projections_are_not_originals=True, manifest_self_hash_excluded=True,
        path_basis='paired_read_v1_relative', stage_artifacts=records,
        stage_bytes_excluding_manifest=sum(row['bytes'] for row in records.values()),
        preserved_originals=preserved, preserved_original_file_count=len(preserved),
        first_68_original_files_unchanged=True, v3_eight_files_unchanged=True,
        projections=PROJECTIONS, verbatim_startup=dict(path='publication_v1/startup.log',
            original_path=ARM + 'server.log', original_sha256=sha(startup_source), publication_sha256=sha(HERE / 'startup.log'),
            redacted=False, scope='Only credential/path-audited small textual startup log copied verbatim'),
        raw_copy_policy='No original JSON receipts replaced. Projection hashes address published bytes; original hashes address private raw bytes, which remain unchanged and excluded from stage.',
        never_ship=['Raw worker launch/handshake tokens', 'Arbitrary process/build environments or ENV secrets',
                    'Actual CLI/native libraries/source archives/weights', 'Raw maps, raw mountstats export/server paths',
                    'All originals under admission-20261015-01 and artifacts-current-20261015-01'],
        no_zip_or_gzip_needed=True, candidate_v3_unbound=True, gpu_queries=0, server_runs=0)
    write('manifest.json', manifest)
    print(json.dumps(verify(), sort_keys=True))


if __name__ == '__main__':
    build()
