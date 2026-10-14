#!/usr/bin/env python3
"""Offline publication/whitelist verifier: stdlib files only, no imports of runners."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
SCOPE = HERE.parent
MAX_BYTES = 10 << 20
SUFFIXES = {'.py', '.md', '.json', '.log', '.tsv'}
PRIVATE_KEYS = {'worker_token', 'worker_launch_hash', 'launch_id', 'nonce', 'password',
    'passwd', 'secret', 'api_key', 'access_token', 'refresh_token', 'authorization',
    'ld_preload', 'raw_maps', 'raw_text', 'raw_lines', 'mount_namespace', 'boot_id'}
PUBLIC_ENV_KEYS = {'CUBLAS_WORKSPACE_CONFIG', 'CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES',
                  'FERRULE_LOG', 'FERRULE_LOG_FORMAT', 'LD_LIBRARY_PATH', 'FERRULE_PROFILE_NVTX'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def regular(root, name):
    require(isinstance(name, str) and name and not Path(name).is_absolute()
            and '..' not in Path(name).parts and str(Path(name)) == name, 'nonportable/unsafe path: ' + str(name))
    root = Path(root).resolve()
    current = root
    for part in Path(name).parts:
        current = current / part
        require(not current.is_symlink(), 'symlink forbidden: ' + name)
    require(current.is_file() and current.resolve().is_relative_to(root), 'not a scoped regular file: ' + name)
    return current


def no_credentials(value, prefix=''):
    if isinstance(value, dict):
        for key, child in value.items():
            # Source pin inventories use paths as keys, never credential field names.
            name = key.lower()
            require('/' in key or name not in PRIVATE_KEYS, 'private JSON field: ' + prefix + key)
            require('/' in key or not re.fullmatch(r'.*(password|passwd|api_key|access_token|refresh_token)$', name),
                    'secret JSON field: ' + prefix + key)
            if name == 'environment':
                require(isinstance(child, dict) and set(child) <= PUBLIC_ENV_KEYS,
                        'unreviewed environment dump: ' + prefix)
            no_credentials(child, prefix + key + '.')
    elif isinstance(value, list):
        for child in value:
            no_credentials(child, prefix)


def projection(name):
    value = load(regular(HERE, name))
    require(value.get('schema') == 'ferrule.storage-demand.redacted-publication-projection.v1'
            and value.get('redacted_projection') is True and value.get('replaces_original_receipt') is False,
            'projection must not impersonate original receipt: ' + name)
    require(value.get('original_sources') and value.get('omitted_fields'), 'projection provenance/redaction missing')
    for origin in value['original_sources']:
        require(origin['path'].startswith(('artifacts-current-20261015-01/', 'actual-', 'execution-', 'protocol-', 'protocol.json', 'cpu-')),
                'unexpected original scope')
        require(re.fullmatch(r'[0-9a-f]{64}', origin['original_sha256']) is not None, 'original hash missing')
    no_credentials(value)
    return value['data']


def validate_denominators(study):
    require(study['status'] == 'stopped' and study['actual_issued_lifecycles'] == 1
            and study['actual_completed_lifecycles'] == 0, 'actual failed lifecycle count changed')
    require(study['scheduled_starts'] == 10 and study['scheduled_blocks'] == 5, 'planned paired scope changed')
    require(study['denominators'] == dict(scheduled=120, issued=0, terminal=0, successful=0,
                                        failed=0, censored=0, unissued=120), 'request denominator conservation failed')
    require(study['measured_qa'] == dict(scheduled=50, success=0, eligible_success=0), 'measured denominator changed')
    require(study['paired_aggregate'] is None and study['scientific_acceptance'] is False, 'false performance acceptance')
    blocks = study['paired_blocks']
    order = [['blocking', 'lookahead'], ['blocking', 'lookahead'], ['lookahead', 'blocking'],
             ['lookahead', 'blocking'], ['blocking', 'lookahead']]
    require(len(blocks) == 5 and [(b['block'], b['execution_order']) for b in blocks]
            == [(f'P{i}', modes) for i, modes in enumerate(order, 1)], 'five paired blocks/order incomplete')
    for block in blocks:
        require(block['qualified'] is False and block['failure_pair_retained'] is True
                and block['log_ratios'] is None and block['arms'] == dict(blocking=None, lookahead=None), 'failed pair dropped/fabricated')
    rows = study['lifecycles']
    require(len(rows) == 10 and len({r['trial'] for r in rows}) == 10, 'ten lifecycle denominators missing')
    require([row['trial'] for row in rows] == [f'P{i}-{mode}' for i, modes in enumerate(order, 1) for mode in modes],
            'lifecycle order changed')
    for row in rows:
        require(row['summary']['scheduled'] == row['summary']['unissued'] == 12
                and row['summary']['issued'] == 0 and row['summary']['measured_qa']['scheduled'] == 5,
                'lifecycle slots silently excluded')
    require(study['http_requests'] == 0 and study['actual_http_token_id_join'] == [], 'fabricated HTTP evidence')
    for key in ('client_e2e_seconds', 'client_ttft_seconds', 'qualified_goodput_tokens_per_second',
                'pipeline_positive_worker_read', 'pipeline_wasted_bytes', 'pipeline_host_credit'):
        require(study[key] is None, 'unobserved outcome must stay null: ' + key)


def verify(scope=SCOPE, check_sources=True, check_private_originals=True):
    scope = Path(scope).resolve()
    pub = scope / 'publication_v1'
    require(pub == HERE or not check_sources, 'custom scope permitted only for publication-only checking')
    manifest = load(regular(pub, 'manifest.json'))
    require(manifest.get('schema') == 'ferrule.storage-demand.curated-publication-manifest.v1'
            and manifest.get('redacted_projections_are_not_originals') is True, 'unknown publication manifest')
    whitelist = load(regular(pub, 'STAGE_FILES.json'))
    require(whitelist['path_basis'] == 'paired_read_v1_relative' and whitelist['size_limit_bytes'] == MAX_BYTES,
            'whitelist root/budget changed')
    grouped = [path for group in whitelist['groups'].values() for path in group]
    paths = whitelist['files']
    require(len(grouped) == len(set(grouped)) == len(paths) == len(set(paths)) and set(grouped) == set(paths),
            'duplicate/inconsistent staging whitelist')
    require(all(len(Path(name).parts) == 1 or name.startswith('publication_v1/') for name in paths),
            'raw experiment/admission tree is not publication-safe')
    require(set(manifest['stage_artifacts']) == set(paths) - {'publication_v1/manifest.json'}, 'manifest/whitelist coverage differs')
    total = 0
    checked = 0
    for name in paths:
        require(Path(name).suffix in SUFFIXES, 'only approved UTF-8 text files permitted')
        if not check_sources and not name.startswith('publication_v1/'):
            continue
        path = regular(scope, name)
        raw = path.read_bytes()
        text = raw.decode('utf-8')
        require(b'\0' not in raw, 'binary/NUL payload forbidden')
        total += len(raw)
        checked += 1
        if name != 'publication_v1/manifest.json':
            expected = manifest['stage_artifacts'][name]
            require(len(raw) == expected['bytes'] and sha(path) == expected['publication_sha256'], 'published hash/size mismatch: ' + name)
        if path.suffix == '.json':
            no_credentials(json.loads(text))
    require(total < MAX_BYTES, 'publication exceeds 10 MiB')
    expected_tsv = 'group\tpath\n' + ''.join(group + '\t' + whitelist['project_prefix'] + '/' + name + '\n'
        for group, names in whitelist['groups'].items() for name in names)
    require(regular(pub, 'STAGE_FILES.tsv').read_text() == expected_tsv, 'TSV staging list differs')
    require(manifest['stage_bytes_excluding_manifest'] == sum(v['bytes'] for v in manifest['stage_artifacts'].values()),
            'stage byte total changed')
    source_checks = 0
    if check_private_originals:
        for name, expected in manifest['preserved_originals'].items():
            path = regular(scope, name)
            require(path.stat().st_size == expected['bytes'] and sha(path) == expected['original_sha256'], 'original mutated: ' + name)
            source_checks += 1
    # All checks below need only the curated publication directory.
    study = projection('study.json')
    validate_denominators(study)
    supervision = projection('supervision.json')
    require(supervision['returncode'] == 1 and supervision['timed_out'] is False
            and supervision['descendant_custody'] == 'proven_drained'
            and supervision['custody']['echild'] is True
            and all(not row['signals'] for row in supervision['custody']['entries']), 'failed natural-exit custody changed')
    closure = projection('closure.json')
    require(closure['shutdown_summary_present'] is False and closure['owner_resource_ledger'] == 'unknown'
            and closure['resource_contract_complete'] == 'unverified' and closure['universal_final_release_verified'] is False,
            'idle falsely repairs missing owner ledger')
    nfs = projection('nfs-client.json')
    require(nfs['shared_client_traffic'] is True and nfs['physical_nas_backend_bytes'] is None
            and nfs['ready_startup_window'] is None and nfs['http_request_windows'] == []
            and nfs['negative_deltas_clamped'] is False, 'physical IO/coverage claim fabricated')
    samples = nfs['samples']
    require([row['label'] for row in samples] == ['prelaunch', 'shutdown-pre', 'shutdown-post'], 'NFS sample coverage changed')
    for row in samples:
        require(row['acquisition_status'] == 'captured' and row['selected_mount']['fstype'] in ('nfs', 'nfs4')
                and row['acquisition_begin']['monotonic'] <= row['acquisition_end']['monotonic'], 'raw NFS window incomplete')
    before, after = samples[-2:]
    require(before['selected_mount']['redacted_identity_sha256'] == after['selected_mount']['redacted_identity_sha256'], 'NFS identity/reset')
    delta = nfs['shutdown_window']
    require(delta['status'] == 'shared_client_delta' and delta['shared_client_traffic'] is True
            and delta['owned_server_physical_bytes'] is None and delta['physical_nas_backend_bytes'] is None, 'NFS scope changed')
    for name, value in delta['deltas'].items():
        a, b = before['selected_mount']['counters'][name], after['selected_mount']['counters'][name]
        require(type(a) is int and type(b) is int and b >= a and value == b - a, 'negative/reset/invented NFS delta')
    idle = projection('custody-idle.json')
    require(idle['final_lock_release']['all_eight_local_locks_released'] is True
            and idle['final_lock_release']['physical_release_certificate'] is False
            and idle['final_lock_reprobe']['all_eight_nonblocking_reacquired'] is True, 'final lock evidence changed')
    observations = idle['observations']
    for name in ('coordinator-post-idle', 'final-all-eight-idle'):
        row = observations[name]
        require(row['status'] == 'verified' and row['authenticated_echild'] is True
                and row['recorded_owners_absent'] is True and row['required_consecutive'] == 3, 'post-ECHILD idle missing')
        for i in row['accepted_sample_indices']:
            sample = row['samples'][i]
            require(sample['classification'] == 'idle' and sample['compute_application_count'] == 0
                    and [gpu['index'] for gpu in sample['gpu_rows']] == list(range(8))
                    and all(gpu['memory_used_mib'] == gpu['utilization_percent'] == 0 for gpu in sample['gpu_rows']),
                    'all-eight idle sample not actually zero')
    bindings = projection('bindings.json')
    require(bindings['v2']['cli_serving_accepted'] is False and bindings['v2']['old_adapter_acceptance_transferred'] is False,
            'old adapter/server acceptance fabricated')
    require(bindings['v2']['binding_sha256'] == study['binding_sha256']
            and bindings['v2']['execution_sha256'] == study['execution_sha256'], 'failed study binding hashes differ')
    v3 = bindings['v3']
    require(v3['candidate_binary'] is None and v3['candidate_source_cut'] is None and v3['actual_lifecycles'] == 0
            and v3['gpu_authorized'] is False and v3['replacement_or_resume'] is False, 'V3 bound or historical resume')
    startup = regular(pub, 'startup.log').read_text()
    require('prepared source profile changed' in startup and 'cuInit' not in startup
            and 'OpenAI API listening' not in startup, 'actual startup failure changed')
    require(manifest['verbatim_startup']['publication_sha256'] == manifest['verbatim_startup']['original_sha256'],
            'verbatim startup claimed with differing hash')
    return dict(status='passed', checked_stage_files=checked, checked_bytes=total,
        original_files_verified_unchanged=source_checks, original_authentication_reperformed=False,
        offline_checks_gpu_queries=0, offline_checks_server_runs=0,
        projection_policy='Redacted recorded evidence, not original receipt replacement or new authentication proof')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publication-only', action='store_true', help='Verify copied publication without any private original/runner artifacts')
    args = parser.parse_args()
    print(json.dumps(verify(check_sources=not args.publication_only, check_private_originals=not args.publication_only), sort_keys=True))
