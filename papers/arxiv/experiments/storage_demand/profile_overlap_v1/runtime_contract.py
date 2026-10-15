"""CPU-only, profile-local ELF provenance; never admit arbitrary unknown maps."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import re
import subprocess
import tarfile

from runnerhelpers import library_maps_v1 as maps
from runnerhelpers.evidence_v2 import ct, require

SDK_LIBRARIES = ('libToolsInjection64.so', 'libToolsInjectionProxy64.so',
                 'libToolsInjectionMemoryAllocator.so', 'libcupti.so.12.9',
                 'libLinuxKeyboardInterceptorProxy.so')
KEYBOARD_SHA = '1a5c1b1c7be874be5bd02a6b30834a2062f169e6e28371c2ebe695672a0e44fa'
NSS_SHA = 'b95591b74a1e9cc22803bf805042ad6c6173435ac873e912ed1291bdaf9a2bb7'


def package_members(package, wanted):
    """Stream the existing Debian ar/tar container, without extracting or executing it."""
    result, control = {}, None
    with Path(package).open('rb') as stream:
        require(stream.read(8) == b'!<arch>\n', 'SDK package is not Debian ar')
        while header := stream.read(60):
            require(len(header) == 60 and header[-2:] == b'`\n', 'invalid SDK ar header')
            size, start = int(header[48:58]), stream.tell()
            member = header[:16].decode().strip().rstrip('/')
            if member in ('control.tar.gz', 'data.tar.gz'):
                with tarfile.open(fileobj=stream, mode='r|gz') as archive:
                    for item in archive:
                        name = item.name.removeprefix('./')
                        if member == 'control.tar.gz' and name == 'control':
                            require(item.isfile() and item.size < 65536, 'invalid SDK package control')
                            control = archive.extractfile(item).read().decode()
                        if member == 'data.tar.gz' and name in wanted:
                            require(item.isfile() and name not in result, 'ambiguous SDK ELF member')
                            digest = hashlib.sha256()
                            with archive.extractfile(item) as source:
                                while chunk := source.read(1024 * 1024):
                                    digest.update(chunk)
                            result[name] = digest.hexdigest()
            stream.seek(start + size + size % 2)
    require(control is not None and set(result) == set(wanted), 'SDK package members missing')
    return control, result


def resolve_system(soname):
    command = ['/sbin/ldconfig', '-p']
    output = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10).stdout
    paths = {str(Path(m[1]).resolve()) for line in output.splitlines()
             if (m := re.fullmatch(r'\s*' + re.escape(soname) + r'\s+\(libc6,x86-64[^)]*\) => (\S+)', line))}
    require(len(paths) == 1, 'missing/ambiguous ldconfig system SONAME: ' + soname)
    path = paths.pop()
    require(maps.elf_dynamic(path)['soname'] == soname, 'ldconfig ELF SONAME differs')
    return path, dict(argv=command, matching_lines=[line for line in output.splitlines() if soname in line])


def inventory(tool, version, prior, candidate):
    """An explicit capability list verified against package, loader and captured ELF origin.

    The old rejection is immutable. Its maps are provenance evidence, not an
    allowlist: an extra unknown path still fails the unchanged role validator.
    """
    target = tool / 'target-linux-x64'
    package = tool.parents[3] / 'NsightSystems-linux-cli-public-2025.3.1.90-3582212.deb'
    relative = 'opt/nvidia/nsight-systems-cli/2025.3.1/target-linux-x64/'
    control, members = package_members(package, [relative + n for n in (*SDK_LIBRARIES, 'nsys')])
    require('Package: nsight-systems-cli-2025.3.1\n' in control
            and 'Version: ' + version + '\n' in control, 'official SDK package version differs')
    for name in (*SDK_LIBRARIES, 'nsys'):
        require(ct.sha(target / name) == members[relative + name], 'installed SDK ELF differs from package: ' + name)
    require(members[relative + 'libLinuxKeyboardInterceptorProxy.so'] == KEYBOARD_SHA,
            'unreviewed keyboard injection ELF')
    nss, nss_loader = resolve_system('libnss_files.so.2')
    util, util_loader = resolve_system('libutil.so.1')
    require(nss == '/usr/lib64/libnss_files-2.32.so' and ct.sha(nss) == NSS_SHA, 'unreviewed system NSS ELF')
    nss_config = Path('/etc/nsswitch.conf')
    databases = {line.split(':', 1)[0]: line.split(':', 1)[1].split('#', 1)[0].split()
                 for line in nss_config.read_text().splitlines() if ':' in line and not line.lstrip().startswith('#')}
    require(any('files' in databases.get(name, []) for name in ('passwd', 'group', 'hosts')),
            'glibc NSS files auto-dlopen configuration absent')
    require(maps.elf_dynamic(nss)['needed'] == ['libc.so.6'], 'NSS dependency closure changed')
    runtime = copy.deepcopy(ct.load(candidate['candidate']['runtime_pin_manifest']))
    original = dict(runtime['per_role']['cli'])
    require(len(original) == 12, 'original twelve CLI requirements changed')
    additions = ct.inventory([*(target / n for n in SDK_LIBRARIES), Path(util), Path(nss)])
    runtime['pins'].update(additions)
    for name in ('libToolsInjection64.so', 'libcupti.so.12.9'):
        path = str(target / name)
        runtime['per_role']['cli'][path] = additions[path]
    policy = maps.role_policy(candidate['candidate']['binary'], candidate['candidate']['binary_sha256'], runtime, 'cli')
    require(all(policy['pins'].get(path) == sha and path in policy['required'] for path, sha in original.items()),
            'original CLI requirements weakened')
    policy['optional'][str(target / 'libLinuxKeyboardInterceptorProxy.so')] = (
        'official SDK injected keyboard/OSRT side capability, checked if mapped; not a global runtime exemption')
    policy['optional'][nss] = 'glibc NSS files runtime auto-dlopen via nsswitch.conf, checked if mapped'
    folder = prior / 'runs/blocking/session/ready-library-evidence'
    snapshot = ct.load(folder / 'snapshot.json')
    assessment = maps.validate_snapshot(snapshot, policy)
    require(assessment['status'] == 'passed', 'verified explicit profile inventory does not cover captured ELF maps')
    # No unused nvperf/nvlog allowance and no admission inferred from unknown paths.
    require(len(policy['pins']) == 19 and len(policy['required']) == 16, 'profile ELF closure changed')
    require(set(assessment['libraries']) == set(policy['pins']), 'profile inventory is not the verified loaded set')
    proof = dict(schema='ferrule.storage-demand.profile-runtime.v2', package=str(package),
        package_sha256=ct.sha(package), package_control=control, package_member_sha256=members,
        signature_verification='not performed; installed official SDK/package byte equality, existing tool pins retained',
        system_loader=dict(nss=nss_loader, util=util_loader), nss_config=str(nss_config),
        nss_config_sha256=ct.sha(nss_config), nss_databases=databases,
        nss_dlopen='glibc files service selected by NSS configuration; not asserted to be a direct executable DT_NEEDED',
        source_snapshot=str(folder / 'snapshot.json'), source_snapshot_sha256=ct.sha(folder / 'snapshot.json'),
        historical_validation=ct.load(folder / 'validation.json'), new_policy_raw_assessment=assessment,
        original_cli_checks={path: dict(expected_sha256=sha, observed_sha256=assessment['libraries'].get(path),
                                        passed=assessment['libraries'].get(path) == sha)
                             for path, sha in original.items()},
        injected_capabilities={str(target / n): dict(sha256=additions[str(target / n)],
            required=str(target / n) in policy['required'], elf=maps.elf_dynamic(target / n)) for n in SDK_LIBRARIES},
        historical_full_trace_accepted=False, acceptance_transferred=False,
        hash_scope=maps.LIMIT, live_mapped_memory_sha256=None)
    return runtime, policy, proof
