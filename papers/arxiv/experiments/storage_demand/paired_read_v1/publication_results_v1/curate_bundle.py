"""Fixed-source publication builder; no modification of inputs or other agents' work."""
from __future__ import annotations

import collections
import json
from pathlib import Path

from common import *
from curate import (SCOPE, ROOT, QUAL, HISTORY, REQUEST_KEYS, read, locator, source,
                    projection, exact_copy, identity, idle, supervision, public_argv,
                    public_controls, resource, mount, maps, log_events, collect_private_literals)


def metadata(final, audit):
    for name in ('order.json', 'requests.json'):
        exact_copy('design/' + name, SCOPE / name)
    oracle = read(SCOPE / 'oracles.json')
    oracle['reference_pins'] = {locator(k): v for k, v in oracle['reference_pins'].items()}
    projection('design/oracles.json', [SCOPE / 'oracles.json'], oracle)
    protocol = read(SCOPE / 'protocol-v3.json')
    public = {k: v for k, v in protocol.items() if k not in ('historical_preservation', 'historical_study')}
    projection('design/protocol.json', [SCOPE / 'protocol-v3.json'], public)
    cases = read(SCOPE / 'cases-v3.json')
    for case in cases.values():
        case['public_runtime_controls'] = public_controls(case.pop('environment'))
        case['model'] = 'CHECKPOINT_DIRECTORY'
        case['argv_after_binary'] = public_argv(case['argv_after_binary'])
    projection('design/cases.json', [SCOPE / 'cases-v3.json'], cases)
    prereg = read(SCOPE / PLAN / 'preregistration.json')
    public = {k: v for k, v in prereg.items() if k not in ('artifact_pins', 'artifacts_directory', 'historical_attempts')}
    projection('design/preregistration.json', [SCOPE / PLAN / 'preregistration.json'], public)
    review = read(SCOPE / PLAN / 'actual-build-review-v3.json')
    identity_value = {k: v for k, v in review['candidate_identity'].items() if k != 'qualification_directory'}
    require(identity_value['commit'] == SOURCE_COMMIT and identity_value['cli_binary_sha256'] == CLI_SHA256,
            'unexpected actual source or binary')
    roles = {}
    for role, value in review['build_review']['per_role'].items():
        roles[role] = pick(value, 'sha256 feature_argv common_nvtx_compiled native_snapshot_sha256 native_copy_count runtime_library_count build_custody')
    commands = {}
    command_paths = []
    for role in roles:
        path = QUAL / ('build-final-' + role + '-command.json')
        command_paths.append(path)
        cmd = read(path)
        commands[role] = dict(argv=['cargo', *cmd['argv'][1:]], working_directory='immutable-source-archive',
                             **pick(cmd, 'timeout_seconds gpu_hidden profiling_enabled'))
    env = read(command_paths[0])['environment']
    build_controls = {k: env[k] for k in ('RUSTUP_TOOLCHAIN', 'RUSTUP_AUTO_INSTALL', 'CARGO_BUILD_JOBS',
                      'CARGO_INCREMENTAL', 'FERRULE_CUDA_ARCH', 'FERRULE_CUDA_TEST_ORACLE',
                      'CUBLAS_WORKSPACE_CONFIG', 'FERRULE_PROFILE_NVTX')}
    quality_review = review['quality_review']
    for gate in quality_review['gates']:
        for key in ('report', 'supervisor'):
            gate[key] = locator(gate[key])
    projection('design/source-and-build.json', [SCOPE / PLAN / 'actual-build-review-v3.json', *command_paths],
               dict(candidate_identity=identity_value, build_roles=roles, build_commands=commands,
                    public_build_controls=build_controls, source_disposition=review['source_disposition'],
                    source_archive_included=False, executables_included=False, native_libraries_included=False,
                    weights_included=False, build_custody=review['build_custody'],
                    quality_review=quality_review,
                    runtime_pins={Path(k).name: v for k, v in review['runtime_pins'].items()},
                    current_source_not_initial_preregistration_source=True))
    tools = read(QUAL / 'toolchain.json')
    artifact = read(QUAL / 'artifact-final-cli.json')
    projection('design/build-tools.json', [QUAL / 'toolchain.json', QUAL / 'artifact-final-cli.json'],
               dict(tool_versions={k: v['stdout'] for k, v in tools.items()},
                    compiled_profile=artifact['compiled_artifact']['profile'], binary_sha256=artifact['sha256'],
                    nvtx_compiled=artifact['nvtx_compiled'],
                    qualification_nvtx_runtime_enabled=artifact['qualification_nvtx_runtime_enabled']))
    study = {k: v for k, v in final.items() if k not in ('lifecycles', 'artifact_pins', 'binding_path',
             'physical_uuid', 'eight_uuids', 'final_idle_path', 'final_lock_release_path', 'final_lock_reprobe_path')}
    study['trials'] = [a['trial'] for a in final['lifecycles']]
    projection('study.json.gz', [SCOPE / RUN / 'paired-final.json'], study)
    audit = dict(audit)
    audit.pop('artifact_pins')
    audit.pop('paired_final_path')
    audit['source_identity'] = identity_value
    for old in audit['historical_attempts']:
        old['path'] = locator(old['path'])
    projection('poststudy-audit.json.gz', [SCOPE / PLAN / 'poststudy-audit.json'], audit)
    exact_copy('final-cleanup.json', SCOPE / RUN / 'final-cleanup.json')
    history = []
    for name in HISTORY:
        path = SCOPE / name / 'paired-final.json'
        old = read(path)
        history.append(dict(study=name, **pick(old, 'status order scheduled_blocks scheduled_starts scheduled_requests '
                        'scheduled_measured_qa actual_issued_lifecycles actual_completed_lifecycles stop analysis'),
                        lifecycles=[{k: a[k] for k in ('trial', 'block', 'mode', 'summary', 'status', 'stop_reason', 'assessment') if k in a}
                                    for a in old['lifecycles']], pooled=False, resumed=False,
                        private_source_warning='Original reports contain nested credentials/environment/maps; projection only'))
    projection('historical-failures.json', [SCOPE / n / 'paired-final.json' for n in HISTORY], history)


def arm_files(arm, uuids, mount_ids):
    trial = arm['trial']
    directory = SCOPE / RUN / trial
    dest = 'arms/' + trial + '/'
    life = arm['lifecycle']
    require(read(directory / 'lifecycle.json') == life, 'raw lifecycle differs from final journal')
    journal = pick(arm, 'trial block mode summary assessment read_ledger outer_deadline coordinator_ended')
    journal.update(supervisor=supervision(arm['supervisor']), post_idle=idle(arm['post_idle']))
    projection(dest + 'journal.json', [SCOPE / RUN / 'paired-final.json', directory / 'supervisor.json',
               directory / 'coordinator-post-idle.json'], journal)
    projection(dest + 'assessment.json', [directory / 'assessment.json'], arm['assessment'])
    for name in ('attempt.json', 'read-ledger-assessment.json'):
        exact_copy(dest + name, directory / name)
    locks = read(directory / 'locks-reprobed.json')
    projection(dest + 'locks.json', [directory / 'locks-reprobed.json'],
               pick(locks, 'all_eight_nonblocking_reacquired'))
    launch = read(directory / 'launch.json')
    projection(dest + 'configuration.json', [directory / 'launch.json'],
               dict(**pick(launch, 'trial mode request_ceiling limits execution_sha256 candidate_sha256'),
                    argv=public_argv(launch['spec']['argv']),
                    public_runtime_controls=public_controls(launch['spec']['environment']),
                    selected_device_alias='device_' + str(uuids.index(life['physical_uuid']))))
    public_life = pick(life, 'status trial started_monotonic physical_nas_backend_bytes server_identity '
                       'startup_seconds ready target_deadline startup_nfs startup_logical_counters requests_nfs shutdown_nfs '
                       'closure server_log_sha256 whole_log_token_join ended')
    public_life['closure'] = resource(life['closure'], uuids)
    public_life['resources'] = resource(life['resources'], uuids)
    public_life['capture'] = {k: v for k, v in life['capture'].items() if k != 'environment'}
    public_life['requests'] = {k: v for k, v in life['requests'].items() if k != 'observations'}
    projection(dest + 'lifecycle.json.gz', [directory / 'lifecycle.json'], public_life)
    for request in life['requests']['observations']:
        name = request['name']
        raw = directory / name
        require(read(raw / 'observed.json') == request, 'observed copy differs from lifecycle')
        public = pick(request, REQUEST_KEYS)
        public['logical_counters'] = request['logical_counters']
        projection(dest + name + '/observed.json.gz', [raw / 'observed.json'], public)
        for small in ('issued.json', 'http-terminal.json'):
            exact_copy(dest + name + '/' + small, raw / small)
        http = read(raw / 'http.json')
        public_http = pick(http, 'schema engine served_model created_unix status workload corpus definitions warmups runs '
                           'errors comparison_speedup aggregate finalized checkpoint_unix measurements '
                           'measurement_in_progress active_requests server_drain')
        public_http['client'] = pick(http['client'], 'script_sha256 python transport checkpointing')
        projection(dest + name + '/http.json.gz', [raw / 'http.json'], public_http)
    snapshots = {}
    snapshot_paths = sorted(directory.glob('*-mountstats.json'))
    for path in snapshot_paths:
        snapshots[path.name.removesuffix('-mountstats.json')] = mount(path, mount_ids)
    projection(dest + 'nfs-snapshots.json.gz', snapshot_paths, snapshots,
               'Selected numeric mount counters only; equal raw identities share a sequential alias; no mount/export/network paths')
    map_paths, map_rows = [], {}
    for phase in ('ready', 'after-requests'):
        folder = directory / (phase + '-library-evidence')
        map_rows[phase] = maps(folder)
        map_paths.extend(folder / name for name in ('snapshot.json', 'policy.json', 'validation.json', 'maps.raw'))
    projection(dest + 'library-evidence.json', map_paths, map_rows,
               'Required library basenames and original content hashes; raw maps and path/overlay identities withheld')
    events = log_events(directory / 'server.log')
    text = b''.join((json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode() for row in events)
    write(HERE, dest + 'server-events.jsonl.gz', text, raw=True)
    projection(dest + 'server-events-source.json', [directory / 'server.log'],
               dict(selected_events=len(events), original_bytes=(directory / 'server.log').stat().st_size,
                    event_counts=dict(collections.Counter(v['record']['fields']['event'] for v in events)),
                    omitted='Non-JSON startup text and non-request startup/listener records; not required for token/read reconciliation',
                    source_byte_offsets_refer_to_original=True))


def inventory_and_scan():
    raw_files = sorted({p for directory in (SCOPE / RUN, SCOPE / PLAN, *(SCOPE / n for n in HISTORY))
                        for p in directory.rglob('*') if p.is_file()})
    require(all(not p.is_symlink() for p in raw_files), 'raw symlink unexpected')
    original_files = [source(p) for p in raw_files]
    hashes = {row['locator']: row['original_sha256'] for row in original_files}
    private = set()
    for path in raw_files:
        if path.suffix == '.json':
            collect_private_literals(read(path), private)
    pin_sets = {}
    inputs = [SCOPE / RUN / 'paired-final.json', SCOPE / PLAN / 'poststudy-audit.json',
              *(SCOPE / PLAN / n for n in ('preregistration.json', 'actual-binding-v3.json', 'actual-build-review-v3.json', 'execution-sealed-v3.json')),
              *(SCOPE / n for n in ('protocol-v3.json', 'cases-v3.json', 'oracles.json', 'order.json', 'requests.json', 'analysis.py', 'quality.py'))]
    def pins(value, prefix):
        if isinstance(value, dict):
            if value and all(isinstance(k, str) and k.startswith('/') and isinstance(v, str)
                             and re.fullmatch('[0-9a-f]{64}', v) for k, v in value.items()):
                public = {locator(k): v for k, v in value.items()}
                require(len(public) == len(value), 'pin locator collision')
                pin_sets[prefix] = public
                for k, v in public.items():
                    if k in hashes:
                        require(hashes[k] == v, 'original global raw pin changed')
            else:
                for k, v in value.items():
                    pins(v, prefix + '/' + k)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                pins(v, prefix + '/' + str(i))
    import re
    for path in inputs:
        if path.suffix == '.json':
            pins(read(path), locator(path))
    provenance = dict(schema='ferrule.paired-http-original-input-inventory.v1',
                      immutable_source_pins=PINNED, raw_files=original_files,
                      input_sources=[source(p) for p in inputs], original_pin_sets=pin_sets,
                      original_hashes_are_not_projection_hashes=True,
                      raw_maps_environment_authentication_withheld=True,
                      external_location_alias_is_not_a_content_hash=True,
                      private_artifacts_inventoried_not_copied=True)
    safe_data(provenance)
    write(HERE, 'provenance.json.gz', provenance)
    for path in HERE.rglob('*'):
        if path.is_file():
            data = unpack(path)
            require(not any(v.encode() in data for v in private), 'private literal found; contents suppressed')
    write(HERE, 'private-scan-receipt.json', dict(
        status='passed', original_json_files_scanned=sum(p.suffix == '.json' for p in raw_files),
        distinct_private_literals_checked=len(private), nested_json_strings_scanned=True,
        original_and_nested_credential_field_variants_checked=True, raw_sources_modified=False,
        scan_scope='All selected current and two historical raw trees; decompressed public members',
        offline_limit='Checker repeats structural marker scan, not private-literal scan without originals'))
    return original_files


def build():
    require(not (HERE / 'provenance.json.gz').exists(), 'one-shot curation; do not overwrite an archive')
    for name, expected in PINNED.items():
        require(sha((SCOPE / name).read_bytes()) == expected, 'immutable input hash differs: ' + name)
    final, audit = read(SCOPE / RUN / 'paired-final.json'), read(SCOPE / PLAN / 'poststudy-audit.json')
    metadata(final, audit)
    mount_ids = {}
    for arm in final['lifecycles']:
        arm_files(arm, final['eight_uuids'], mount_ids)
    require(len(mount_ids) == 1, 'unexpected mount identity changes')
    originals = inventory_and_scan()
    require(all(sha((ROOT / row['locator']).read_bytes()) == row['original_sha256'] for row in originals),
            'original inputs changed during curation')
    print('Curated ten arms; all original input hashes unchanged; private-literal scan passed')


if __name__ == '__main__':
    build()
