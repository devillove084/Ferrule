#!/usr/bin/env python3
"""CPU-only membership checks for refactor-sensitive GPU selections.

Use --nextest-json with `cargo nextest list --profile PROFILE --features cuda
--run-ignored all --message-format json` (or --run-ignored only for nightly), or --cargo-artifacts with Cargo's CUDA --no-run
--message-format=json output. The latter only invokes test binaries with --list;
it never builds, runs a test body, or writes into the Cargo target directory.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    'gpu-smoke': ('ferrule-backend', r'^cuda::operator_owner::pr01_tests::'),
    'gpu-nightly': ('ferrule-composition', r'::tiny_ep2_full_hybrid_matches_local_and_lifecycle$'),
}


def load_config(source):
    """Extract the audited literal filter blocks, using stdlib even on Python 3.10.

    Nextest remains responsible for full TOML validation. Changed/duplicate
    sections or a different filter-string representation fail this source guard.
    """
    profiles = {}
    for name, body in re.findall(r'(?ms)^\[profile\.([^\]]+)\][ \t]*\n(.*?)(?=^\[|\Z)', source):
        if name not in REQUIRED:
            continue
        filters = re.findall(r"(?ms)^default-filter\s*=\s*'''(.*?)'''[ \t]*(?:\n|$)", body)
        if name in profiles or len(filters) != 1:
            raise ValueError('Need one literal default-filter for ' + name)
        profiles[name] = {'default-filter': filters[0]}
    if set(profiles) != set(REQUIRED):
        raise ValueError('Missing audited GPU profile')
    return {'profile': profiles}


def profile_filter(config, profile):
    """Evaluate only the small selector grammar used by these checked profiles.

    Unknown syntax fails closed; this is not a replacement for Nextest's parser.
    """
    source = config['profile'][profile]['default-filter']
    atom = r'(?:package|binary|kind)\([^)]*\)|test\(/.*?/\)|test\(=[^)]*\)'
    tokens = re.findall(atom + r'|[()&|]|\S+', source)
    index = 0

    def primary():
        nonlocal index
        token = tokens[index]
        index += 1
        if token == '(':
            result = expression()
            if tokens[index] != ')':
                raise ValueError('Unclosed selector group')
            index += 1
            return result
        match = re.fullmatch(r'(package|binary|kind|test)\((.*)\)', token)
        if not match:
            raise ValueError('Unsupported Nextest selector: ' + token)
        field, pattern = match.groups()
        if pattern.startswith('/') and pattern.endswith('/') and field == 'test':
            regex = re.compile(pattern[1:-1])
            return lambda row: regex.search(row[field]) is not None
        if pattern.startswith('=') or field == 'kind':
            value = pattern.removeprefix('=')
            return lambda row: row[field] == value
        raise ValueError('Unsupported Nextest matching mode: ' + token)

    def conjunction():
        nonlocal index
        parts = [primary()]
        while index < len(tokens) and tokens[index] == '&':
            index += 1
            parts.append(primary())
        return lambda row: all(part(row) for part in parts)

    def expression():
        nonlocal index
        parts = [conjunction()]
        while index < len(tokens) and tokens[index] == '|':
            index += 1
            parts.append(conjunction())
        return lambda row: any(part(row) for part in parts)

    result = expression()
    if index != len(tokens):
        raise ValueError('Unexpected selector tokens: ' + repr(tokens[index:]))
    return result


def package_name(package_id):
    for name in ('ferrule-backend', 'ferrule-composition', 'ferrule-runtime', 'ferrule-model'):
        if re.search(r'(?:^|[/# ])' + re.escape(name) + r'(?:[@/# ]|$)', package_id):
            return name
    return package_id


def nextest_rows(data):
    rows = []
    for suite in data['rust-suites'].values():
        for name, case in suite['testcases'].items():
            rows.append(dict(package=package_name(suite['package-id']),
                             binary=suite['binary-name'], kind=suite['kind'], test=name,
                             selected=case['filter-match']['status'] == 'matches'))
    return rows


def cargo_rows(path, profile):
    package, _ = REQUIRED[profile]
    binaries = set()
    for line in path.read_text().splitlines():
        if not line.startswith('{'):
            continue
        item = json.loads(line)
        if (item.get('reason') == 'compiler-artifact' and item.get('executable')
                and item.get('profile', {}).get('test') is True
                and package_name(item['package_id']) == package
                and 'lib' in item['target']['kind'] and 'cuda' in item.get('features', [])):
            binaries.add((str(Path(item['executable']).resolve(strict=True)), item['target']['name']))
    if len(binaries) != 1:
        raise ValueError(f'Need exactly one CUDA {package} library test artifact, found {len(binaries)}')
    binary, name = binaries.pop()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='void')
    listed = subprocess.run([binary, '--list', '--format', 'terse'], env=env,
                            text=True, capture_output=True, check=True, timeout=60)
    return [dict(package=package, binary=name, kind='lib', test=line[:-6])
            for line in listed.stdout.splitlines() if line.endswith(': test')]


def check(config, profile, rows):
    selected = profile_filter(config, profile)
    package, pattern = REQUIRED[profile]
    matches = [row for row in rows if row['package'] == package and row['kind'] == 'lib'
               and re.search(pattern, row['test'])]
    if not matches:
        raise ValueError(f'{profile}: required branch has zero compiled tests: {package} / {pattern}')
    for row in matches:
        if not selected(row) or row.get('selected') is False:
            raise ValueError(f'{profile}: required test excluded: {row["package"]} / {row["test"]}')
    return [row['test'] for row in matches]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=REQUIRED, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / '.config/nextest.toml')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--nextest-json', type=Path)
    source.add_argument('--cargo-artifacts', type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config.read_text())
    rows = (nextest_rows(json.loads(args.nextest_json.read_text())) if args.nextest_json
            else cargo_rows(args.cargo_artifacts, args.profile))
    matches = check(config, args.profile, rows)
    print(f'{args.profile}: {len(matches)} required compiled tests selected')
    for name in matches:
        print(name)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
