#!/usr/bin/env python3
"""Offline allowlist projection/check of the completed private experiment; never launches."""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import run

ct, require = run.ct, run.require
HERE, PRIVATE = run.HERE, run.PRIVATE
OUTPUT = HERE / 'receipts' / 'result.json'
TERMINAL_KEYS = ('event', 'request_id', 'session_id', 'status', 'finish_reason', 'generated_tokens',
    'emitted_token_events', 'last_emission_ordinal', 'token_events', 'terminal_event_enqueued', 'usage_available',
    'prompt_tokens', 'completion_tokens', 'total_tokens', 'cancellation_requested', 'cancellation_submitted',
    'cancellation_failed', 'requires_restore_or_shutdown')
CACHE_KEYS = ('expert_resident_entries', 'expert_resident_bytes', 'expert_pending_upload_bytes',
    'expert_scratch_bytes', 'expert_owner_peak_bytes', 'expert_unknown_quarantine_bytes')


def choose(value, keys):
    return {key: value[key] for key in keys if key in value}


def invocation(value):
    if value is None:
        return None
    pipeline = json.loads(value['expert_read_pipeline'])
    return dict(**choose(value, ('transaction', 'status', 'quarantined', 'rows', 'sequences', 'prefill_rows', 'decode_rows')),
                cache=choose(value, CACHE_KEYS), read_status=pipeline['status'],
                logical_host_charge=pipeline['current_charge'], cumulative_read_stats=pipeline['stats'],
                per_request_transaction_identity_join=None)


def derive():
    value = ct.load(PRIVATE / 'result.json')
    prepared = ct.load(run.PREPARED)
    require(value['status'] == 'complete' and value['assessment']['can_continue'] is True, 'lifecycle did not qualify')
    ct.check_pins(value['raw_pins'])
    require(prepared['plan_sha256'] == ct.sha(HERE / 'plan.json'), 'preregistered plan changed')
    ct.check_pins(prepared['artifact_pins'])
    lifecycle = value['lifecycle']
    folder = PRIVATE / run.TRIAL
    lines = run.ex.log_lines(folder / 'server.log')
    events = [run.trace.fields(event) for _, event in run.trace.json_events(lines)]
    observations = lifecycle['requests']['observations']
    joined = run.join(run.trace.parse_token_trace(lines), observations, lines)
    require(len(joined) == 2 and all(row['protocol_accepted'] for row in joined), 'actual terminal join differs')
    requests, previous = [], None
    for row, bound in zip(observations, joined):
        actual = row['actual']
        worker_id = bound['actual_worker_request_id']
        terminals = [event for event in events if event.get('event') == 'request_worker_terminal' and event.get('request_id') == worker_id]
        require(len(terminals) == 1 and terminals[0]['session_id'] == worker_id, 'runtime terminal identity absent')
        invocations = [entry['fields'] for entry in row['logical_counters']['rows']]
        require(row['logical_counters']['valid'] is True and invocations, 'counter window incomplete')
        current = dict(name=row['name'], outcome=row['status'], protocol_accepted=row['protocol_accepted'],
            quality_passed=row['quality_passed'], metric_eligible=row['metric_eligible'],
            http_completion_id=actual['completion_id'], worker_request_id=worker_id, session_id=terminals[0]['session_id'],
            actual_worker_enqueued_ids=bound['actual_generated_ids'], enqueue_ordinals=bound['enqueue_ordinals'],
            worker_enqueue_count=len(bound['actual_generated_ids']), client_content_event_count=actual['content_event_count'],
            client_text=actual['text'], client_usage=actual['usage'], client_finish_reason=actual['finish_reason'],
            client_DONE_observed=actual['done_at'] is not None, client_token_ids=None,
            runtime_terminal=choose(terminals[0], TERMINAL_KEYS),
            terminal_enqueue=[choose(e, TERMINAL_KEYS) for e in bound['worker_terminal_enqueue']],
            http_terminal=[choose(e, TERMINAL_KEYS) for e in bound['http_response_terminal']],
            logical_counters=row['logical_counters']['deltas'], invocation_count=len(invocations),
            previous_completed_invocation=invocation(previous), first_invocation=invocation(invocations[0]),
            last_invocation=invocation(invocations[-1]), per_request_pending_operations=None,
            per_request_KV_snapshot=None, per_request_epoch=None,
            counters_scope='serial log-byte window, not transaction/request identity; logical reads/transfers, not NAS backend bytes',
            gauges_scope='last completed observations, not exact before/after live request ledger; cumulative/peak values are not summed')
        if row['controlled_cancel']:
            current.update(cancellation_accepted=bound['cancellation_accepted'], controlled_censor=True,
                prefix_matches_frozen=bound['prefix_matches_frozen'], exact_eight_id_quality_pass=None,
                tcp_shutdown=actual['tcp_shutdown'],
                observed_content_to_socket_shutdown_seconds=actual['closed_at'] - actual['raw_sse_events'][-1]['at'],
                worker_terminal_wait_seconds=actual['cancellation_evidence']['wait_seconds'],
                terminal_wait_status=actual['cancellation_evidence']['wait_status'])
        else:
            require(terminals[0]['status'] == 'finished' and terminals[0]['cancellation_requested'] is False,
                    'recovery runtime terminal not natural finished')
            current.update(exact_eight_id_quality_pass=bound['exact_token_pass'], new_request_and_session=True)
        requests.append(current)
        previous = invocations[-1]
    final = lifecycle['closure']['attempts'][-1]['summary']
    physical = final['physical']
    maps = [ct.load(folder / name) for name in ('ready-library-maps.json', 'after-requests-library-maps.json')]
    require(all(m['status'] == 'passed' and m['observed_library_count'] == 12 and m['required_pin_check'] is True for m in maps), 'twelve-library identity proof incomplete')
    runtime_pins = ct.load(prepared['candidate']['runtime_pin_manifest'])
    manifest = {str(Path(path).relative_to(PRIVATE)): digest for path, digest in value['raw_pins'].items()}
    manifest['result.json'] = ct.sha(PRIVATE / 'result.json')
    event_counts = {name: sum(e.get('event') == name for e in events) for name in (
        'request_admitted', 'request_http_admitted', 'request_worker_token_enqueued', 'request_worker_terminal',
        'request_worker_terminal_enqueue', 'request_http_terminal', 'mixed_decoder_invocation', 'resource_shutdown_summary')}
    return dict(schema=run.SCHEMA + '.public-result', status='complete',
        protocol_sha256=prepared['plan_sha256'], prepared_sha256=ct.sha(run.PREPARED),
        candidate=choose(prepared['candidate_record']['candidate_identity'],
                         ('commit', 'cli_binary_sha256', 'source_digest', 'source_archive_sha256', 'source_members')),
        maintained_execution_role_root_sha256=ct.sha(run.maintained.roles.ROLE_PATH),
        qualified_runtime_manifest_sha256=ct.sha(prepared['candidate']['runtime_pin_manifest']),
        runtime_library_pins=[dict(name=Path(path).name, sha256=digest) for path, digest in sorted(runtime_pins['pins'].items())],
        scheduled_lifecycles=1, actual_lifecycles=1, scheduled_requests=2, actual_requests=len(observations),
        initial_smokes=0, retries=0, profiler=False, NVTX=False, schedule='lookahead',
        lifecycle_seconds=value['ended']['monotonic'] - value['started']['monotonic'], within_270_seconds=value['within_270_seconds'],
        requests=requests, summary=lifecycle['requests']['summary'], raw_event_counts=event_counts,
        final_owner=dict(custody_state=lifecycle['closure']['final']['custody_state'],
            runtime=final['runtime'], physical=choose(physical, ('scope', 'image_generation', 'expert_cache', 'expert_read_pipeline',
                'live_model_state_bytes', 'default_state', 'quiescence', 'residency', 'close', 'release', 'quarantine')),
            KV_broker_domains=[row for row in final['broker_domains'] if row.get('value', {}).get('kind') == 'KvPage'],
            per_request_ledger=None, physical_final_release=None),
        maps=[dict(phase=label, **choose(m, ('status', 'required_pin_check', 'observed_library_count', 'hash_scope')),
                   raw_maps_sha256=m['raw_maps_sha256'], ignored_deleted_nonlibrary_count=len(m['ignored_deleted_nonlibraries']),
                   overlay_alias_count=len(m['overlay_device_aliases']), live_mapped_memory_sha256=None)
              for label, m in zip(('ready', 'after-requests'), maps)],
        cleanup=dict(authenticated_ECHILD=value['final_authenticated_echild'], recorded_owners_absent=value['final_recorded_owners_absent'],
            all_eight_stable_idle=value['final_all_eight_idle'], all_eight_locks_released_and_reprobed=value['all_eight_locks_released_and_reprobed'],
            capture_returncode=lifecycle['capture']['returncode'], resource_observer_status=lifecycle['resources']['status'],
            idle_schema=value['post_idle']['schema'], required_consecutive=value['post_idle']['required_consecutive']),
        scope=dict(normal_production_HTTP_cancellation=True, fresh_request_recovery=True, same_request_resume=False,
            stale_continuation_observed_in_frozen_eight_tokens=False, arbitrary_KV_equivalence_proved=False,
            Unknown_proved=False, integrated_transaction_ledger_proved=False, per_request_epoch=None,
            physical_release_proved=False, request_cancellation_is_not_read_cancellation_count=True),
        raw_sha256=manifest, raw_not_published=True,
        provenance='Offline allowlist projection after one frozen run; no new inference, telemetry, native load or source recapture')


def credentials():
    """Known private authentication strings are forbidden even under an innocent key."""
    result = set()
    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if re.search('token|secret|nonce|challenge|cookie', key, re.I) and isinstance(child, str) and len(child) >= 16:
                    result.add(child)
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    for path in PRIVATE.rglob('*.json'):
        walk(ct.load(path))
    require(result, 'no authenticated private credentials found for redaction check')
    return result


def check_public_files(paths):
    secrets = credentials()
    for path in paths:
        require(path.is_file() and not path.is_symlink() and not path.is_relative_to(PRIVATE), 'unsafe publication path')
        text = path.read_text()
        require(not any(secret in text for secret in secrets), 'private credential in publication: ' + path.name)
    return dict(status='passed', known_private_credentials_checked=len(secrets), publication_files=len(paths), raw_files_published=0)


def main():
    require(sys.argv[1:] in (['derive'], ['check']), 'offline derive/check only')
    result = derive()
    if sys.argv[1] == 'derive':
        ct.write(OUTPUT, result)
    require(ct.load(OUTPUT) == result, 'curated result differs from preserved raw evidence')
    print(json.dumps(dict(status='passed', result_sha256=ct.sha(OUTPUT), raw_hashes=len(result['raw_sha256']),
                         actual_requests=result['actual_requests'], cleanup=result['cleanup']), indent=2))


if __name__ == '__main__':
    raise SystemExit(main())
