#!/usr/bin/env python3
"""One-shot CPU curation helpers. No runner imports or GPU/network operations."""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil

from common import HERE, RUN, PLAN, PINNED, SECRET_KEY, decode, pick, require, safe_data, sha, write

SCOPE = HERE.parent
ROOT = SCOPE.parents[4]
HISTORY = ('artifacts-current-20261015-01', 'artifacts-v3-final-20261015-01')
QUAL = ROOT / 'target/validation/fixesmetadataCLIcurrent-quality-20261015-01'
SOURCE_RULE = 'Explicit field projection; original bytes are private and not replaced by this copy'
REQUEST_KEYS = ('index name phase population measured prompt input_tokens max_tokens reference_id reference '
                'payload status issued actual trace duplicate_lifecycle_request_id quality_passed metric_eligible '
                'process_io nfs log_start_byte log_end_byte window_begin window_end absolute_deadline')
EVENTS = {'request_admitted', 'request_http_admitted', 'mixed_decoder_invocation',
          'request_worker_token_enqueued', 'request_worker_terminal',
          'request_worker_terminal_enqueue', 'request_http_terminal', 'resource_shutdown_summary'}


def read(path):
    return decode(Path(path).read_bytes())


def locator(path):
    path = Path(path)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        # This is a location alias, explicitly NOT a content hash.
        return 'external-location-withheld/' + sha(str(path).encode())[:20] + '/' + path.name


def source(path):
    path = Path(path)
    return dict(locator=locator(path), original_sha256=sha(path.read_bytes()),
                original_bytes=path.stat().st_size)


def projection(name, paths, data, rule=SOURCE_RULE):
    obj = dict(schema='ferrule.paired-http-publication-projection.v1',
               redacted_projection=True, original_receipt_replaced=False,
               original_sources=[source(path) for path in paths], rule=rule, data=data)
    safe_data(obj)
    write(HERE, name, obj)


def exact_copy(name, path):
    data = path.read_bytes()
    safe_data(decode(data))
    target = HERE / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        require(target.read_bytes() == data, 'existing copy differs; never overwrite')
        return
    shutil.copyfile(path, target)
    require(target.read_bytes() == data, 'copy differs')


def identity(value):
    return pick(value, 'pid birth_ticks ppid')


def idle(value):
    result = pick(value, 'status all_eight_idle deadline_seconds required_consecutive interval_seconds '
                  'zero_memory_mib zero_utilization_percent no_compute_apps_required physical_owner_release '
                  'mode authenticated_echild recorded_owners_absent accepted_sample_indices consecutive_idle elapsed_seconds')
    result['samples'] = []
    for row in value['samples']:
        devices = []
        for line in row['gpu'].splitlines():
            index, _, memory, utilization = [v.strip() for v in line.split(',')]
            require(re.fullmatch(r'\d+ MiB', memory) and re.fullmatch(r'\d+ %', utilization), 'idle units')
            devices.append(dict(index=int(index), memory_used_mib=int(memory.split()[0]),
                                utilization_percent=int(utilization.split()[0])))
        result['samples'].append(dict(**pick(row, 'monotonic_started monotonic_finished recorded_unix_ns classification consecutive_idle'),
                                      devices=devices, compute_applications=len(row['apps'].splitlines())))
    return result


def supervision(value):
    result = pick(value, 'returncode timed_out wall_seconds cleanup_errors cleanup_scope descendant_custody '
                  'gpu_quiescence stdout_truncated stderr_truncated')
    c = value['custody']
    result['custody'] = dict(**pick(c, 'status method containment_verified echild errors reporting_errors'),
                            root_identity=identity(c['root_identity']), helper_identity=identity(c['helper_identity']),
                            entries=[dict(**pick(v, 'role signals reaped exit_code first_observed_at reaped_at'),
                                          identity=identity(v['identity'])) for v in c['entries']])
    result['private_authentication_replayed'] = False
    return result


def public_argv(argv):
    result = list(argv)
    for i, value in enumerate(result):
        if i and result[i - 1] == '--host':
            result[i] = 'LOOPBACK_ADDRESS'
        elif value.startswith('/'):
            result[i] = 'CHECKPOINT_DIRECTORY' if '/hf/' in value else 'CLI_ARTIFACT'
    return result


def public_controls(env):
    keys = ('CUBLAS_WORKSPACE_CONFIG', 'FERRULE_LOG', 'FERRULE_LOG_FORMAT', 'FERRULE_PROFILE_NVTX')
    return {key: env[key] for key in keys if key in env}


def resource(value, uuids):
    def walk(v):
        if isinstance(v, dict):
            return {('device_' + str(uuids.index(k)) if k in uuids else k): walk(child)
                    for k, child in v.items() if k not in ('uuid', 'final_process_diagnostics')}
        if isinstance(v, list):
            return [walk(c) for c in v]
        if isinstance(v, str) and v in uuids:
            return 'device_' + str(uuids.index(v))
        return v
    return walk(value)


def mount(path, ids):
    value = read(path)
    m = value['selected_mount']
    private_identity = (value['boot_id'], value['mount_namespace'], m['device'], m['mountpoint'], m['fstype'], m['statvers'])
    if private_identity not in ids:
        ids[private_identity] = 'shared-client-mount-' + str(len(ids) + 1)
    return dict(**pick(value, 'label acquisition_begin acquisition_end captured_monotonic captured_unix_ns '
                      'acquisition_status shared_client_traffic owned_server_physical_bytes physical_nas_backend_bytes raw_sha256'),
                reader_identity=identity(value['reader_identity']),
                mount_identity_alias=ids[private_identity],
                selected_mount=pick(m, 'fstype statvers age_seconds counters errors'))


def maps(folder):
    snapshot, policy, valid = (read(folder / n) for n in ('snapshot.json', 'policy.json', 'validation.json'))
    raw = source(folder / 'maps.raw')
    require(raw['original_sha256'] == snapshot['raw_maps_sha256'] == valid['raw_maps_sha256'], 'map source changed')
    result = dict(snapshot=pick(snapshot, 'schema root_identity parent_identity acquisition_begin acquisition_end '
                                'errors expected_executable_sha256 before parent_before raw_saved after parent_after '
                                'executable_sha256 after_executable raw_maps_sha256 hash_scope'),
                  policy=dict(**pick(policy, 'role executable_sha256 hash_scope'),
                              pins={Path(k).name: v for k, v in policy['pins'].items()},
                              required=[Path(k).name for k in policy['required']],
                              optional=[Path(k).name for k in policy['optional']]),
                  validation=dict(**pick(valid, 'status required_pin_check errors missing_required_paths deleted_library_paths '
                                          'observed_library_count hash_scope live_mapped_memory_sha256 root_identity captured raw_maps_sha256'),
                                  libraries={Path(k).name: v for k, v in valid['libraries'].items()},
                                  ignored_deleted_nonlibraries_count=len(valid['ignored_deleted_nonlibraries']),
                                  overlay_backing_identity='unobserved'), original_raw_map=raw)
    require(len(result['policy']['pins']) == len(policy['pins']), 'library basename alias collision')
    return result


def log_events(path):
    result, offset = [], 0
    for line_number, line in enumerate(path.read_bytes().splitlines(keepends=True), 1):
        try:
            obj = decode(line)
        except (ValueError, UnicodeDecodeError):
            obj = {}
        if obj.get('fields', {}).get('event') in EVENTS:
            safe_data(obj)
            result.append(dict(source_line=line_number, source_byte_start=offset,
                               source_byte_end=offset + len(line), record=obj))
        offset += len(line)
    return result


def collect_private_literals(value, output):
    """Includes original_worker_token and nested JSON in stdout/context fields."""
    if isinstance(value, dict):
        for key, child in value.items():
            if SECRET_KEY.search(key) and isinstance(child, str) and len(child) >= 8:
                output.add(child)
            collect_private_literals(child, output)
    elif isinstance(value, list):
        for child in value:
            collect_private_literals(child, output)
    elif isinstance(value, str):
        candidates = [value] + value.splitlines()
        for candidate in candidates:
            if candidate.lstrip().startswith(('{', '[')):
                try:
                    collect_private_literals(decode(candidate), output)
                except (ValueError, TypeError):
                    pass


if __name__ == '__main__':
    from curate_bundle import build
    build()
