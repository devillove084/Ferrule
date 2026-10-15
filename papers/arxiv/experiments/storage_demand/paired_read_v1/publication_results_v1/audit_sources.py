#!/usr/bin/env python3
"""Optional owner-side read-only source/hash/privacy audit; not needed offline."""
from common import HERE, decode, encode, load, require, sha, unpack
from curate import ROOT, collect_private_literals


def audit():
    inventory = load(HERE, 'provenance.json.gz')
    private = set()
    json_count = 0
    for item in inventory['raw_files']:
        path = ROOT / item['locator']
        require(path.is_relative_to(ROOT) and not path.is_symlink(), 'unsafe original source')
        content = path.read_bytes()
        require(sha(content) == item['original_sha256'] and len(content) == item['original_bytes'],
                'original input changed: ' + item['locator'])
        if path.suffix == '.json':
            collect_private_literals(decode(content), private)
            json_count += 1
    members, expanded = 0, 0
    for path in sorted(HERE.rglob('*')):
        if path.is_file():
            content = unpack(path)
            require(not any(value.encode() in content for value in private), 'private literal found (value suppressed)')
            members += 1
            expanded += len(content)
    return dict(status='passed', original_files_verified=len(inventory['raw_files']),
                original_json_files_deep_scanned=json_count, distinct_private_literals_checked=len(private),
                public_members_scanned=members, public_uncompressed_bytes_scanned=expanded,
                nested_json_strings_and_original_credential_variants_checked=True,
                raw_sources_modified=False, gpu_workloads=0, network_requests=0,
                scope='All inventoried current and two historical raw files; all decompressed public files')


if __name__ == '__main__':
    print(encode(audit()).decode(), end='')
