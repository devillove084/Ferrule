#!/usr/bin/env python3
"""Passive, post-hoc Linux inventory; no GPU, model, network or subprocess calls."""

import argparse
import collections
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import time
import uuid

BASE = Path('papers/arxiv/experiments/storage_demand/paired_read_v1')
RUN = 'artifacts-v3-mapfixed-20261015-01'
PUBLIC = 'publication_results_v1'
V1_UNLIMITED_THRESHOLD = 1 << 60
RESOURCE_KEYS = {
    'cpuinfo', 'meminfo', 'memtotal', 'model name', 'physical id', 'core id',
    'cpus_allowed_list', 'mems_allowed_list', 'cpuset', 'cgroup', 'affinity',
    'kernel', 'kernel_release', 'uname', 'memory.limit_in_bytes', 'memory.max',
    'cpu.cfs_quota_us', 'cpu.max', 'cpuset.cpus', 'cpuset.mems',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def canonical_hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())


def read_bytes(path, limit=64 * 1024 * 1024):
    with Path(path).open('rb') as stream:
        raw = stream.read(limit + 1)
    require(len(raw) <= limit, 'input exceeds bounded read size')
    return raw


def read_text(path):
    return read_bytes(path, 2 * 1024 * 1024).decode('utf-8')


def read_json(path):
    if str(path).endswith('.gz'):
        with gzip.open(path, 'rb') as stream:
            raw = stream.read(64 * 1024 * 1024 + 1)
        require(len(raw) <= 64 * 1024 * 1024, 'expanded JSON exceeds read bound')
    else:
        raw = read_bytes(path)
    return json.loads(raw)


def clock():
    ns = time.time_ns()
    return {'unix_ns': ns, 'utc': utc(ns), 'monotonic_ns': time.monotonic_ns()}


def utc(ns):
    seconds, nanos = divmod(ns, 1_000_000_000)
    base = datetime.datetime.fromtimestamp(seconds, datetime.timezone.utc)
    return base.strftime('%Y-%m-%dT%H:%M:%S') + f'.{nanos:09d}Z'


def boot_hash(value):
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    try:
        parsed = str(uuid.UUID(value))
    except ValueError:
        return None
    if parsed != value.lower():
        return None
    return sha256(value.encode('ascii'))


def compare_boot_ids(values, current_hash):
    hashes = [boot_hash(value) for value in values]
    counts = collections.Counter(hashes)
    matched = counts.get(current_hash, 0) if current_hash else 0
    missing = counts.get(None, 0)
    return {
        'status': 'same_boot' if hashes and current_hash and matched == len(hashes)
        else 'cannot_bind_posthoc_to_study',
        'occurrences': len(hashes), 'matches': matched,
        'missing_or_invalid': missing, 'mismatches': len(hashes) - matched - missing,
        'recorded_boot_id_sha256_counts': dict(sorted(
            (key, count) for key, count in counts.items() if key is not None)),
    }


def cpu_list(text):
    text = text.strip()
    result = set()
    if not text:
        return {'list': '', 'count': 0}
    for part in text.split(','):
        require(re.fullmatch(r'\d+(?:-\d+)?', part), 'invalid CPU/node list')
        ends = [int(x) for x in part.split('-')]
        first, last = ends[0], ends[-1]
        require(0 <= first <= last <= 1_000_000, 'invalid CPU/node range')
        require(last - first <= 65536, 'CPU/node range exceeds bound')
        result.update(range(first, last + 1))
    return {'list': text, 'count': len(result)}


def parse_cpuinfo(text):
    records, duplicates = {}, 0
    for block in re.split(r'\n\s*\n', text.strip()):
        fields = {}
        for line in block.splitlines():
            if ':' in line:
                key, value = line.split(':', 1)
                if key.strip() in {'processor', 'model name', 'physical id', 'core id'}:
                    key, value = key.strip(), value.strip()
                    require(key not in fields or fields[key] == value,
                            'conflicting duplicate CPU field')
                    fields[key] = value
        if 'processor' not in fields:
            continue
        require(fields['processor'].isdigit(), 'invalid processor ID')
        processor = int(fields['processor'])
        for key in ('physical id', 'core id'):
            if key in fields:
                require(fields[key].isdigit(), 'invalid CPU topology ID')
                fields[key] = int(fields[key])
        fields['processor'] = processor
        if processor in records:
            require(records[processor] == fields, 'conflicting duplicate processor ID')
            duplicates += 1
        records[processor] = fields
    require(records, 'no logical processors in cpuinfo')
    rows = list(records.values())
    complete = all('physical id' in row and 'core id' in row for row in rows)
    sockets = {row['physical id'] for row in rows if 'physical id' in row}
    pairs = {(row['physical id'], row['core id']) for row in rows
             if 'physical id' in row and 'core id' in row}
    return {
        'source': '/proc/cpuinfo', 'logical_processor_count': len(records),
        'duplicate_identical_processor_records': duplicates,
        'model_counts': dict(sorted(collections.Counter(
            row.get('model name', 'unrecorded') for row in rows).items())),
        'topology_complete_in_visible_records': complete,
        'visible_socket_id_count': len(sockets) if complete else None,
        'visible_socket_core_pair_count': len(pairs) if complete else None,
        'topology_columns': ['processor', 'physical_id', 'core_id'],
        'topology_rows': [[row['processor'], row.get('physical id'), row.get('core id')]
                          for row in sorted(rows, key=lambda row: row['processor'])],
        'interpretation': 'Visible proc topology, not physical-host socket/core attestation; '
                          'logical count is not CPU quota or experiment affinity.',
    }


def parse_meminfo(text):
    lines = [line.strip() for line in text.splitlines() if line.startswith('MemTotal:')]
    require(len(lines) == 1, 'expected exactly one MemTotal field')
    match = re.fullmatch(r'MemTotal:\s+(\d+)\s+kB', lines[0])
    require(match is not None, 'MemTotal must use Linux proc kB units')
    kib = int(match.group(1))
    require(kib > 0, 'MemTotal must be positive')
    return {
        'source': '/proc/meminfo:MemTotal', 'source_value': kib, 'source_unit': 'kB',
        'bytes_per_source_unit': 1024, 'bytes': kib * 1024,
        'GiB': kib / 1024 ** 2, 'GB': kib * 1024 / 1000 ** 3,
        'interpretation': 'Kernel-visible MemTotal; not DIMM-installed RAM, '
                          'free memory, or a cgroup allowance.',
    }


def memory_limit(value, version):
    raw = value.strip()
    if version == 2 and raw == 'max':
        return {'raw': raw, 'status': 'unlimited_at_read_scope', 'finite_bytes': None}
    number = int(raw)
    require(number >= 0 or (version == 1 and number == -1), 'invalid memory limit')
    unlimited = version == 1 and (number == -1 or number >= V1_UNLIMITED_THRESHOLD)
    return {'raw': raw, 'status': 'unlimited_sentinel_at_read_scope' if unlimited else 'finite',
            'finite_bytes': None if unlimited else number,
            'v1_unlimited_threshold_bytes': V1_UNLIMITED_THRESHOLD if version == 1 else None}


def cpu_quota(quota, period):
    require(int(period) > 0, 'invalid CPU quota period')
    if quota in ('max', '-1'):
        return {'quota': quota, 'period_us': int(period),
                'status': 'unlimited_at_read_scope', 'cpu_time_ratio': None}
    require(int(quota) > 0, 'invalid CPU quota')
    return {'quota': quota, 'period_us': int(period), 'status': 'finite',
            'cpu_time_ratio': int(quota) / int(period)}


def unescape_mount(value):
    return re.sub(r'\\([0-7]{3})', lambda match: chr(int(match.group(1), 8)), value)


def safe_absolute(value):
    path = PurePosixPath(value)
    require(path.is_absolute() and '..' not in path.parts, 'unsafe resource path')
    return path


def resolve_cgroups(cgroup_text, mountinfo_text):
    memberships = []
    for line in cgroup_text.splitlines():
        hierarchy, controllers, group = line.split(':', 2)
        memberships.append((hierarchy, set(controllers.split(',')), safe_absolute(group)))
    mounts = []
    for line in mountinfo_text.splitlines():
        left, right = line.split(' - ', 1)
        before, after = left.split(), right.split()
        if after[0] not in ('cgroup', 'cgroup2'):
            continue
        root = safe_absolute(unescape_mount(before[3]))
        mount = safe_absolute(unescape_mount(before[4]))
        # Ignore duplicate bind/overlay aliases outside the canonical cgroup tree.
        if mount != PurePosixPath('/sys/fs/cgroup') and PurePosixPath('/sys/fs/cgroup') not in mount.parents:
            continue
        mounts.append((root, mount, after[0], set(after[2].split(','))))
    result = {}
    for controller in ('memory', 'cpu', 'cpuset'):
        candidates = []
        for hierarchy, controllers, group in memberships:
            for root, mount, kind, options in mounts:
                match = (kind == 'cgroup2' and hierarchy == '0' and controllers == {''}) or (
                    kind == 'cgroup' and controller in controllers and controller in options)
                if not match or (root != group and root not in group.parents):
                    continue
                candidates.append({'version': 2 if kind == 'cgroup2' else 1,
                                   'root': root, 'mount': mount, 'group': group,
                                   'directory': mount / group.relative_to(root)})
        if len(candidates) == 1:
            result[controller] = candidates[0]
    return result


def read_optional(path):
    try:
        return {'status': 'read', 'value': read_text(path).strip()}
    except OSError as error:
        return {'status': 'unavailable', 'errno': error.errno}


def collect_cgroups():
    cgroup_text = read_text('/proc/self/cgroup')
    mount_text = read_text('/proc/self/mountinfo')
    resolved = resolve_cgroups(cgroup_text, mount_text)
    result = {'resolution_sources': ['/proc/self/cgroup', '/proc/self/mountinfo'],
              'membership_sha256': sha256(cgroup_text.encode()),
              'mountinfo_sha256': sha256(mount_text.encode()),
              'raw_membership_and_mount_paths_published': False,
              'scope': 'Current collector only; experiment groups and hidden parents unknown.',
              'controllers': {}}
    for controller in ('memory', 'cpu', 'cpuset'):
        entry = resolved.get(controller)
        if entry is None:
            result['controllers'][controller] = {'status': 'unknown_unresolved_or_ambiguous'}
            continue
        version = entry['version']
        if controller == 'memory':
            names = ['memory.limit_in_bytes', 'memory.memsw.limit_in_bytes',
                     'memory.use_hierarchy', 'memory.stat'] if version == 1 else [
                         'memory.max', 'memory.high', 'memory.swap.max']
        elif controller == 'cpu':
            names = ['cpu.cfs_quota_us', 'cpu.cfs_period_us'] if version == 1 else ['cpu.max']
        else:
            names = ['cpuset.cpus', 'cpuset.mems'] + (
                ['cpuset.effective_cpus', 'cpuset.effective_mems'] if version == 1 else
                ['cpuset.cpus.effective', 'cpuset.mems.effective'])
        directory, mount = Path(entry['directory']), Path(entry['mount'])
        levels = []
        while True:
            fields = {name: read_optional(directory / name) for name in names}
            if 'memory.stat' in fields and fields['memory.stat']['status'] == 'read':
                fields['memory.stat']['value'] = {
                    k: int(v) for k, v in (line.split() for line in fields['memory.stat']['value'].splitlines())
                    if k in ('hierarchical_memory_limit', 'hierarchical_memsw_limit')}
            level = {'depth_above_current': len(levels), 'fields': fields}
            if controller == 'memory':
                name = 'memory.limit_in_bytes' if version == 1 else 'memory.max'
                if fields[name]['status'] == 'read':
                    level['local_memory_limit'] = memory_limit(fields[name]['value'], version)
            elif controller == 'cpu' and all(field['status'] == 'read' for field in fields.values()):
                if version == 1:
                    quota, period = (fields[name]['value'] for name in names)
                else:
                    quota, period = fields['cpu.max']['value'].split()
                level['local_cpu_quota'] = cpu_quota(quota, period)
            levels.append(level)
            if directory == mount:
                break
            require(mount in directory.parents, 'resource path escaped cgroup mount')
            directory = directory.parent
        result['controllers'][controller] = {
            'status': 'resolved_from_current_membership', 'version': version,
            'membership_path_sha256': sha256(str(entry['group']).encode()),
            'mount_root_sha256': sha256(str(entry['root']).encode()),
            'current_equals_visible_mount_root': entry['directory'] == entry['mount'],
            'hidden_ancestors_unobserved': str(entry['root']) != '/',
            'effective_limit_claim': 'not_attested; local observations only', 'levels': levels,
        }
    result['membership_unchanged_during_collection'] = cgroup_text == read_text('/proc/self/cgroup')
    return result


def inspect_fields(value, boot_values, resource_keys):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == 'boot_id':
                boot_values.append(child)
            if key.lower() in RESOURCE_KEYS:
                resource_keys[key] += 1
            inspect_fields(child, boot_values, resource_keys)
    elif isinstance(value, list):
        for child in value:
            inspect_fields(child, boot_values, resource_keys)


def inspect_study(repo, current_hash):
    base, public = repo / BASE, repo / BASE / PUBLIC
    manifest = read_json(public / 'manifest.json')
    provenance = read_json(public / 'provenance.json.gz')
    pins = {entry['locator']: entry for entry in provenance['raw_files']}
    public_records, public_keys = [], collections.Counter()
    public_json = sorted(p for p in public.rglob('*') if p.is_file() and (
        p.name.endswith('.json') or p.name.endswith('.json.gz')))
    for path in public_json:
        relative = str(path.relative_to(public))
        raw = read_bytes(path)
        if relative != 'manifest.json':
            pin = manifest['files'].get(relative)
            require(pin and pin['sha256'] == sha256(raw) and pin['bytes'] == len(raw),
                    'public JSON does not match manifest')
            if path.suffix == '.gz':
                expanded = gzip.decompress(raw)
                require(pin['uncompressed_sha256'] == sha256(expanded)
                        and pin['uncompressed_bytes'] == len(expanded), 'public expanded JSON changed')
        public_records.append([relative, len(raw), sha256(raw)])
        inspect_fields(read_json(path), [], public_keys)
    order = read_json(public / 'design/order.json')
    expected_arms = sorted(block['block'] + '-' + mode for block in order['blocks'] for mode in block['modes'])
    run = base / RUN
    actual_arms = sorted(p.name for p in run.iterdir() if p.is_dir())
    require(actual_arms == expected_arms and len(expected_arms) == 10, 'completed ten-arm scope differs')
    originals, all_boots, private_keys = [], [], collections.Counter()
    original_json = sorted(run.rglob('*.json'))
    for path in original_json:
        raw = read_bytes(path)
        locator = str(path.relative_to(repo))
        pin = pins.get(locator)
        require(pin and pin['original_sha256'] == sha256(raw) and pin['original_bytes'] == len(raw),
                'original JSON does not match public provenance')
        originals.append([locator, len(raw), sha256(raw)])
        inspect_fields(json.loads(raw), all_boots, private_keys)
    per_arm, phases, times, monotonic = [], set(), [], []
    raw_text_records = []
    for arm in expected_arms:
        snapshots = sorted((run / arm).glob('*-mountstats.json'))
        projected = read_json(public / 'arms' / arm / 'nfs-snapshots.json.gz')
        public_snapshots = projected['data']
        require({p.name.removesuffix('-mountstats.json') for p in snapshots} == set(public_snapshots),
                'original/public NFS phase coverage differs')
        require(len(snapshots) == 30, 'incomplete NFS phase coverage')
        projection_pins = {entry['locator']: entry for entry in projected['original_sources']}
        boots, arm_times, arm_monotonic = [], [], []
        for path in snapshots:
            value = read_json(path)
            label = path.name.removesuffix('-mountstats.json')
            locator = str(path.relative_to(repo))
            require(projection_pins.get(locator) == pins.get(locator), 'NFS projection pin mismatch')
            require(value['label'] == label, 'NFS label differs from source name')
            for key in ('captured_unix_ns', 'captured_monotonic', 'acquisition_begin',
                        'acquisition_end', 'raw_sha256'):
                require(value[key] == public_snapshots[label][key], 'NFS projection field differs')
            raw_path = path.with_suffix('.txt')
            raw = read_bytes(raw_path)
            raw_pin = pins.get(str(raw_path.relative_to(repo)))
            require(raw_pin and raw_pin['original_sha256'] == sha256(raw) == value['raw_sha256']
                    and raw_pin['original_bytes'] == len(raw), 'original mount text hash mismatch')
            raw_text_records.append([str(raw_path.relative_to(repo)), len(raw), sha256(raw)])
            boots.append(value.get('boot_id'))
            arm_times.append(value['captured_unix_ns'])
            arm_monotonic.append(value['captured_monotonic'])
            phases.add(label)
        times.extend(arm_times)
        monotonic.extend(arm_monotonic)
        per_arm.append({'arm': arm, 'phase_snapshots': len(snapshots),
                        'boot_comparison': compare_boot_ids(boots, current_hash),
                        'captured_utc_range': [utc(min(arm_times)), utc(max(arm_times))]})
    study = read_json(public / 'study.json.gz')['data']
    all_comparison = compare_boot_ids(all_boots, current_hash)
    return {
        'scope': str(BASE / RUN), 'public_provenance': str(BASE / PUBLIC / 'provenance.json.gz'),
        'public_manifest_sha256': sha256(read_bytes(public / 'manifest.json')),
        'public_provenance_sha256': sha256(read_bytes(public / 'provenance.json.gz')),
        'public_json_files_checked_against_manifest': len(public_records) - 1,
        'public_json_files_inspected': len(public_records),
        'public_json_inventory_sha256': canonical_hash(public_records),
        'original_json_files_checked_against_public_provenance': len(originals),
        'original_json_inventory_sha256': canonical_hash(originals),
        'original_mount_text_files_checked': len(raw_text_records),
        'original_mount_text_inventory_sha256': canonical_hash(raw_text_records),
        'inventory_hash_rule': 'SHA256 UTF-8 compact sorted-key JSON of sorted '
                               '[relative_locator, byte_length, raw_sha256] rows',
        'all_original_json_boot_ids': all_comparison,
        'phase_snapshots': sum(row['phase_snapshots'] for row in per_arm),
        'per_arm': per_arm, 'phase_labels': sorted(phases),
        'snapshot_captured_utc_range': [utc(min(times)), utc(max(times))],
        'snapshot_captured_monotonic_seconds_range': [min(monotonic), max(monotonic)],
        'study_started': study['started'], 'study_ended': study['ended'],
        'study_utc_range': [utc(study['started']['unix_ns']), utc(study['ended']['unix_ns'])],
        'resource_field_search': {
            'method': 'Recursive JSON keys only, fixed key allowlist; not a claim about all repository archives.',
            'searched_case_insensitive_keys': sorted(RESOURCE_KEYS),
            'original_matches': dict(private_keys), 'public_matches': dict(public_keys),
            'mounts_are_not_limits': 'Original mount snapshots include cgroup mount metadata, '
                                     'not recorded cgroup resource-control file values.',
        },
        'binding_status': 'POSTHOC_INVENTORY_SAME_BOOT' if all_comparison['status'] == 'same_boot'
        and all(row['boot_comparison']['status'] == 'same_boot' for row in per_arm)
        else 'cannot_bind_posthoc_to_study',
        'other_experiment_groups_binding': 'not_checked; do not extrapolate to pressure/profile/reference runs',
    }


def collect_current():
    begin = clock()
    cpu_text, memory_text = read_text('/proc/cpuinfo'), read_text('/proc/meminfo')
    status = dict(line.split(':', 1) for line in read_text('/proc/self/status').splitlines() if ':' in line)
    uname = os.uname()
    nodes = []
    for path in sorted(Path('/sys/devices/system/node').glob('node[0-9]*')):
        if re.fullmatch(r'node\d+', path.name):
            value = read_optional(path / 'cpulist')
            nodes.append({'node': path.name, 'cpus': cpu_list(value['value'])
                          if value['status'] == 'read' else value})
    online = read_optional('/sys/devices/system/cpu/online')
    result = {
        'begin': begin, 'cpu': parse_cpuinfo(cpu_text), 'memory': parse_meminfo(memory_text),
        'cpuinfo_sha256': sha256(cpu_text.encode()), 'meminfo_sha256': sha256(memory_text.encode()),
        'collector_affinity': {key: cpu_list(status[key])
                               for key in ('Cpus_allowed_list', 'Mems_allowed_list') if key in status},
        'online_cpus': cpu_list(online['value']) if online['status'] == 'read' else online,
        'numa_nodes': nodes, 'numa_source': '/sys/devices/system/node/node*/cpulist',
        'uname': {'sysname': uname.sysname, 'kernel_release': uname.release, 'architecture': uname.machine},
        'cgroups': collect_cgroups(), 'end': clock(),
    }
    return result


def collect(repo):
    begin = clock()
    current_hash = boot_hash(read_text('/proc/sys/kernel/random/boot_id'))
    require(current_hash is not None, 'current boot ID unavailable or malformed')
    study = inspect_study(repo, current_hash)
    result = {
        'schema': 'ferrule.posthoc-platform-inventory.v1',
        'collection_started': begin, 'boot_id_sha256': current_hash,
        'boot_hash_rule': 'SHA256 of stripped original UUID text, UTF-8, no newline; no raw UUID published',
        'study_binding': study, 'current_inventory': None,
        'status': study['binding_status'],
        'limits_of_inference': [
            'CPU/RAM/kernel/affinity/resource-limit inventory was collected post hoc, not during study arms.',
            'Same-boot evidence supports the study platform only assuming no relevant hotplug/reconfiguration; '
            'unchanged hardware or resource controls is not established.',
            'Visible CPU topology is not physical-host topology; MemTotal is not installed DIMM capacity.',
            'Current collector affinity/cgroup files do not attest experiment-time or other experiment groups.',
            'Hidden parent cgroup limits remain unknown even when visible local limits are unlimited.',
            'Boot hashes and provenance pins are integrity/binding evidence, not signatures or independent '
            'public access to the withheld original boot IDs.',
            'No GPU/driver/NVML, model, network, hostname, machine ID, MAC, serial or IP probe performed.',
        ],
        'collector_source_sha256': sha256(read_bytes(Path(__file__))),
    }
    if result['status'] == 'POSTHOC_INVENTORY_SAME_BOOT':
        result['current_inventory'] = collect_current()
    after = boot_hash(read_text('/proc/sys/kernel/random/boot_id'))
    result['boot_hash_unchanged_during_collection'] = current_hash == after
    if after != current_hash:
        result['status'] = 'cannot_bind_posthoc_to_study'
        result['current_inventory'] = None
    result['collection_finished'] = clock()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument('--output', type=Path, required=True,
                        help='New JSON receipt inside this directory; never overwrites')
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.parent == Path(__file__).resolve().parent, 'output must stay inside inventory directory')
    require(not output.exists(), 'refusing to overwrite an existing receipt')
    result = collect(args.repo.resolve())
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'receipt': output.name,
                      'boot_id_sha256': result['boot_id_sha256'],
                      'boot_comparison': result['study_binding']['all_original_json_boot_ids']}))
    return 0 if result['status'] == 'POSTHOC_INVENTORY_SAME_BOOT' else 2


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, TypeError) as error:
        # Do not print exception values: a source failure may carry a private path/value.
        print('Inventory stopped: input validation/read failed (' + type(error).__name__ + ').', file=sys.stderr)
        sys.exit(2)
