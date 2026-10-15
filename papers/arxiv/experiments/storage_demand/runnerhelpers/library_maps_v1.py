"""Private proc-map evidence and fail-closed, role-aware ELF pathname validation.

A pathname SHA is NOT a hash of live mapped memory. Inode/device observations
are checked, with explicit overlay-device ambiguity, not a resident-page proof.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import time

from .evidence_v2 import ct, require

LIMIT = 'pathname SHA256 and inode check; overlay backing identity and live mapped memory hash unobserved'


def stamp():
    return dict(unix_ns=time.time_ns(), monotonic=time.monotonic())


def private_write(path, value):
    data = value if isinstance(value, bytes) else (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def parse_maps(raw):
    rows = []
    for number, line in enumerate(raw.splitlines(), 1):
        fields = line.split(maxsplit=5)
        require(len(fields) >= 5, f'malformed maps line {number}')
        address, permissions, offset, device, inode = fields[:5]
        require(re.fullmatch(r'[0-9a-f]+-[0-9a-f]+', address)
                and re.fullmatch(r'[r-][w-][x-][ps]', permissions)
                and re.fullmatch(r'[0-9a-f]+', offset)
                and re.fullmatch(r'[0-9a-f]+:[0-9a-f]+', device)
                and inode.isdecimal(), f'malformed maps fields at line {number}')
        begin, end = (int(v, 16) for v in address.split('-'))
        require(begin < end, f'empty/reversed maps range at line {number}')
        pathname = fields[5] if len(fields) == 6 else ''
        deleted = pathname.endswith(' (deleted)')
        path = pathname[:-10] if deleted else pathname
        # proc escapes newlines but cannot distinguish a literal backslash-012.
        require('\\012' not in path, f'ambiguous proc pathname at line {number}')
        rows.append(dict(line=number, path=path, deleted=deleted, permissions=permissions,
                         device=[int(v, 16) for v in device.split(':')], inode=int(inode)))
    return rows


def elf_dynamic(path):
    with Path(path).open('rb') as stream:
        require(stream.read(4) == b'\x7fELF', 'not an ELF file: ' + str(path))
    result = subprocess.run(['readelf', '-W', '-d', '-l', str(path)],
                            capture_output=True, text=True, timeout=10, check=True)
    text = result.stdout
    sonames = re.findall(r'\(SONAME\).*\[([^\]]+)\]', text)
    interpreters = re.findall(r'Requesting program interpreter: ([^\]]+)\]', text)
    require(len(sonames) <= 1 and len(interpreters) <= 1, 'ambiguous ELF dynamic identity')
    return dict(needed=re.findall(r'\(NEEDED\).*\[([^\]]+)\]', text),
                soname=sonames[0] if sonames else None,
                interpreter=interpreters[0] if interpreters else None)


def role_policy(executable, executable_sha256, runtime, role):
    """Keep every pin; all build-role pins and recursive DT_NEEDED are required."""
    pins = runtime['pins']
    role_pins = runtime['per_role'][role]
    require(pins and role_pins and all(pins.get(p) == h for p, h in role_pins.items()),
            'missing/inconsistent explicit build-role runtime pins')
    require(ct.sha(executable) == executable_sha256, 'role executable hash differs')
    ct.check_pins(pins)
    objects = {p: elf_dynamic(p) for p in pins}
    binary = elf_dynamic(executable)
    names = {}
    for path, info in objects.items():
        require(str(Path(path).resolve()) == path, 'runtime pin must name canonical file: ' + path)
        for name in {Path(path).name, info['soname']} - {None}:
            require(name not in names or names[name] == path, 'ambiguous pinned ELF SONAME: ' + name)
            names[name] = path
    required = set(role_pins)
    reasons = {p: ['build-role:' + role] for p in role_pins}
    edges = {}
    pending = [(str(executable), binary), *[(p, objects[p]) for p in role_pins]]
    seen = set()
    while pending:
        path, info = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        dependencies = []
        for name in info['needed']:
            require(name in names, 'DT_NEEDED lacks pinned ELF: ' + path + ' -> ' + name)
            dependencies.append(names[name])
        if info['interpreter']:
            interpreter = str(Path(info['interpreter']).resolve())
            require(interpreter in pins, 'ELF interpreter not pinned: ' + interpreter)
            dependencies.append(interpreter)
        edges[path] = sorted(set(dependencies))
        for dependency in dependencies:
            required.add(dependency)
            reasons.setdefault(dependency, []).append('ELF dependency of ' + path)
            pending.append((dependency, objects[dependency]))
    return dict(role=role, executable=str(Path(executable).resolve()),
                executable_sha256=executable_sha256, pins=dict(pins),
                required={p: sorted(set(reasons[p])) for p in sorted(required)},
                optional={p: 'other-role pin, not this role or its DT_NEEDED closure; checked if mapped'
                          for p in sorted(set(pins) - required)}, elf_dependencies=edges,
                hash_scope=LIMIT)


def _identity_matches(actual, expected):
    return (isinstance(actual, dict) and isinstance(expected, dict)
            and all(type(expected.get(k)) is int and expected[k] > 0
                    and actual.get(k) == expected[k] for k in ('pid', 'birth_ticks', 'ppid'))
            and actual.get('state') not in ('Z', 'X'))


def capture(directory, root, parent, executable_sha256, identity):
    """Persist bytes before ELF/library validation, including on later failures.

    Authenticate before reading proc to avoid inspecting a recycled/foreign PID;
    if that fails, preserve an unavailable receipt rather than inventing raw maps.
    """
    directory = Path(directory)
    directory.mkdir(mode=0o700, exist_ok=False)
    record = dict(schema='ferrule.private-runtime-map-snapshot.v1', root_identity=dict(root),
                  parent_identity=dict(parent), acquisition_begin=stamp(), raw_maps_path=None,
                  raw_maps_sha256=None, errors=[], expected_executable_sha256=executable_sha256,
                  hash_scope=LIMIT)
    try:
        record['before'] = identity(root['pid'])
        record['parent_before'] = identity(parent['pid'])
        require(_identity_matches(record['before'], root), 'owned server PID/birth/parent differs before capture')
        require(_identity_matches(record['parent_before'], parent)
                and root['ppid'] == parent['pid'], 'owned capture parent/wrong child differs')
        raw = Path(f'/proc/{root["pid"]}/maps').read_bytes()
        private_write(directory / 'maps.raw', raw)
        record.update(raw_maps_path=str((directory / 'maps.raw').resolve()),
                      raw_maps_sha256=hashlib.sha256(raw).hexdigest(), raw_saved=stamp())
        record['after'] = identity(root['pid'])
        record['parent_after'] = identity(parent['pid'])
        record['executable_path'] = str(Path(f'/proc/{root["pid"]}/exe').readlink())
        record['executable_sha256'] = ct.sha(f'/proc/{root["pid"]}/exe')
        record['after_executable'] = identity(root['pid'])
    except Exception as error:
        record['errors'].append(type(error).__name__ + ': ' + str(error))
    finally:
        record['acquisition_end'] = stamp()
        private_write(directory / 'snapshot.json', record)
    return record


def overlay_mount(path, device):
    """Only the actual longest-prefix overlay mount can explain st_dev aliasing.

    Linux overlayfs may expose the overlay superblock's st_dev via stat while
    proc maps exposes the backing file's device. This does not prove backing
    identity: report the discrepancy, never claim a mapped-memory hash.
    """
    matches = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left, separator, right = line.partition(' - ')
        require(separator, 'malformed mountinfo')
        fields = left.split()
        mountpoint = re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), fields[4])
        if Path(path).is_relative_to(mountpoint):
            matches.append((len(mountpoint), mountpoint, fields[2], right.split()[0]))
    require(matches, 'pathname mount unobserved: ' + path)
    _, mountpoint, major_minor, filesystem = max(matches)
    if filesystem == 'overlay' and [int(v) for v in major_minor.split(':')] == device:
        return dict(mountpoint=mountpoint, filesystem=filesystem, stat_device=device,
                    observed=stamp(), scope='validator mount namespace, not live mapped-file identity',
                    mount_namespace=str(Path('/proc/self/ns/mnt').readlink()))
    return None


def assess(raw, policy):
    """Offline assessment of exactly the captured text, never reads live proc."""
    pins, required = policy['pins'], set(policy['required'])
    require(required and required <= pins.keys()
            and set(policy['optional']) == pins.keys() - required,
            'incomplete required/optional pin classification')
    result = dict(status='failed', libraries={}, missing_required_paths=[], deleted_library_paths=[],
                  ignored_deleted_nonlibraries=[], overlay_device_aliases={}, errors=[], observed_library_count=0,
                  required_pin_check=None, hash_scope=LIMIT, live_mapped_memory_sha256=None)
    seen = set()
    try:
        rows = parse_maps(raw)
        for row in rows:
            path = row['path']
            if not path or path.startswith('['):
                continue
            absolute = path.startswith('/')
            canonical = str(Path(path).resolve()) if absolute else path
            named_library = bool(re.search(r'\.so(?:\.|$)', Path(path).name))
            is_library = canonical in pins or named_library
            if (absolute and not row['deleted'] and canonical not in pins
                    and canonical != policy['executable'] and 'x' in row['permissions']):
                try:
                    # Never probe model-data mappings or read device nodes.
                    if stat.S_ISREG(Path(canonical).stat().st_mode):
                        with Path(canonical).open('rb') as stream:
                            is_library = is_library or (stream.read(4) == b'\x7fELF'
                                                        and canonical != policy['executable'])
                except OSError:
                    if 'x' in row['permissions'] or named_library:
                        result['errors'].append('unreadable mapped executable/library: ' + path)
            if row['deleted']:
                special = (path.startswith(('/memfd:', 'memfd:', '/dev/shm/', '/SYSV'))
                           or path == '/dev/zero')
                if is_library or 'x' in row['permissions'] or not special:
                    result['deleted_library_paths'].append(path)
                else:
                    result['ignored_deleted_nonlibraries'].append(path)
                continue
            if not is_library:
                continue
            seen.add(canonical)
            if canonical not in pins:
                result['errors'].append('unbound loaded library: ' + path)
                continue
            try:
                metadata = Path(canonical).stat()
                require(row['inode'] == metadata.st_ino, 'mapped device/inode differs: ' + path)
                device = [os.major(metadata.st_dev), os.minor(metadata.st_dev)]
                if row['device'] != device:
                    mount = overlay_mount(canonical, device)
                    require(mount is not None, 'mapped device differs outside overlay mount: ' + path)
                    result['overlay_device_aliases'][canonical] = dict(mount,
                        mapped_device=row['device'], inode=metadata.st_ino,
                        backing_file_identity='unobserved; pathname pin check only')
                if canonical not in result['libraries']:
                    require(ct.sha(canonical) == pins[canonical], 'mapped pathname hash differs: ' + path)
                    with Path(canonical).open('rb') as stream:
                        require(stream.read(4) == b'\x7fELF', 'mapped pin is not ELF: ' + path)
                    result['libraries'][canonical] = pins[canonical]
            except (OSError, ValueError) as error:
                result['errors'].append(str(error))
    except (OSError, ValueError) as error:
        result['errors'].append(str(error))
    result['missing_required_paths'] = sorted(required - seen)
    for key in ('deleted_library_paths', 'ignored_deleted_nonlibraries', 'errors'):
        result[key] = sorted(set(result[key]))
    result['observed_library_count'] = len(result['libraries'])
    if not result['observed_library_count']:
        result['errors'].append('zero observed bound ELF runtime libraries')
    if not result['errors'] and not result['missing_required_paths'] and not result['deleted_library_paths']:
        result.update(status='passed', required_pin_check=True)
    elif result['observed_library_count']:
        result['required_pin_check'] = False
    return result


def validate_snapshot(snapshot, policy):
    raw_path = snapshot.get('raw_maps_path')
    result = dict(status='failed', required_pin_check=None, errors=list(snapshot['errors']),
                  missing_required_paths=[], deleted_library_paths=[], libraries={},
                  observed_library_count=0, hash_scope=LIMIT, live_mapped_memory_sha256=None)
    if raw_path:
        require(ct.sha(raw_path) == snapshot['raw_maps_sha256'], 'saved raw maps hash differs')
        result.update(assess(Path(raw_path).read_text(), policy))
        result['errors'].extend(snapshot['errors'])
    for phase in ('before', 'after', 'after_executable'):
        if not _identity_matches(snapshot.get(phase), snapshot['root_identity']):
            result['errors'].append('owned server PID/birth/parent differs: ' + phase)
    for phase in ('parent_before', 'parent_after'):
        if not _identity_matches(snapshot.get(phase), snapshot['parent_identity']):
            result['errors'].append('owned capture PID/birth differs: ' + phase)
    if snapshot['root_identity']['ppid'] != snapshot['parent_identity']['pid']:
        result['errors'].append('wrong child: server is not owned capture direct child')
    if (snapshot.get('executable_sha256') != policy['executable_sha256']
            or snapshot.get('expected_executable_sha256') != policy['executable_sha256']
            or snapshot.get('executable_path') != policy['executable']):
        result['errors'].append('actual server executable differs from pinned build role (not wrapper)')
    if result['errors']:
        result.update(status='failed', required_pin_check=False if result['observed_library_count'] else None)
    result.update(root_identity=snapshot['root_identity'], captured=snapshot['acquisition_end'],
                  raw_maps_path=raw_path, raw_maps_sha256=snapshot['raw_maps_sha256'])
    return result


def legacy_assess(raw, pins):
    """Literal old predicate for same-snapshot calibration comparison only."""
    paths = sorted({line.split()[-1] for line in raw.splitlines()
                    if '.so' in line and line.split()[-1].startswith('/')})
    if not paths or '(deleted)' in raw:
        return dict(status='failed', error='loaded library mappings missing/deleted')
    for path in paths:
        actual = str(Path(path).resolve())
        if actual not in pins or ct.sha(actual) != pins[actual]:
            return dict(status='failed', error='unbound loaded library: ' + actual)
    return dict(status='passed', error=None)


def library_maps(root, pins, *, directory, parent, executable, executable_sha256, runtime, role, identity):
    snapshot = capture(directory, root, parent, executable_sha256, identity)
    try:
        require(runtime['pins'] == pins, 'caller/manifest runtime pin set differs')
        policy = role_policy(executable, executable_sha256, runtime, role)
        private_write(Path(directory) / 'policy.json', policy)
        result = validate_snapshot(snapshot, policy)
    except Exception as error:
        result = dict(status='failed', required_pin_check=None, errors=[type(error).__name__ + ': ' + str(error)],
                      raw_maps_path=snapshot['raw_maps_path'], raw_maps_sha256=snapshot['raw_maps_sha256'])
    private_write(Path(directory) / 'validation.json', result)
    require(result['status'] == 'passed', 'runtime library maps rejected: ' + json.dumps({
        k: result.get(k) for k in ('errors', 'missing_required_paths', 'deleted_library_paths')}))
    return result
