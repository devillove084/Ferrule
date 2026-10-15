#!/usr/bin/env python3
"""Offline checker: only public bundle files, Python standard library, no GPU/NAS."""
from __future__ import annotations

import argparse
import collections
import itertools
import json
from pathlib import Path
import statistics

from common import (HERE, RUN, PLAN, PINNED, CLI_SHA256, SOURCE_COMMIT, SOURCE_ARCHIVE_SHA256,
                    Invalid, decode, encode, load, pick, require, sha, unpack, verify_members)
import prespecified
from reconcile import (summed, read_ledger, invocations, client_tokens, pid_io, nfs_delta,
                       map_evidence, same_birth, close)


PREFIX = 'papers/arxiv/experiments/storage_demand/paired_read_v1/'


def data(root, name):
    obj = load(root, name)
    require(obj['schema'] == 'ferrule.paired-http-publication-projection.v1'
            and obj['redacted_projection'] is True and obj['original_receipt_replaced'] is False,
            'projection not explicitly labeled')
    return obj['data']


def provenance(root, manifest):
    original = load(root, 'provenance.json.gz')
    require(original['immutable_source_pins'] == PINNED, 'provenance source pin mismatch')
    hashes = {}
    for entry in original['raw_files'] + original['input_sources']:
        name, digest = entry['locator'], entry['original_sha256']
        require(name not in hashes or hashes[name] == digest, 'inconsistent original source hash')
        hashes[name] = digest
    for pins in original['original_pin_sets'].values():
        for name, digest in pins.items():
            require(name not in hashes or hashes[name] == digest, 'inconsistent global raw hash pin')
            hashes[name] = digest
    for name, pin in PINNED.items():
        require(hashes[PREFIX + name] == pin, 'global original pin omitted')
    for name, entry in manifest['files'].items():
        if name.endswith(('.json', '.json.gz')):
            obj = load(root, name)
            if isinstance(obj, dict) and obj.get('redacted_projection') is True:
                for source in obj['original_sources']:
                    require(hashes[source['locator']] == source['original_sha256'], 'projection original source differs: ' + name)
        original_name = None
        if name in ('design/order.json', 'design/requests.json'):
            original_name = PREFIX + name.split('/', 1)[1]
        elif name == 'final-cleanup.json':
            original_name = PREFIX + RUN + '/' + name
        elif name.startswith('arms/') and name.endswith(('/attempt.json', '/read-ledger-assessment.json', '/issued.json', '/http-terminal.json')):
            original_name = PREFIX + RUN + '/' + name.split('/', 1)[1]
        if original_name is not None:
            require(hashes[original_name] == entry['sha256'], 'byte-identical raw copy hash differs: ' + name)
    return dict(original_raw_files=len(original['raw_files']), original_pin_sets=len(original['original_pin_sets']),
                distinct_original_pointers=len(hashes))


def qualify_close(journal, life, events, ledger):
    supervisor = journal['supervisor']
    c = supervisor['custody']
    require(supervisor['returncode'] == 0 and supervisor['timed_out'] is False
            and supervisor['cleanup_errors'] == [] and supervisor['descendant_custody'] == 'proven_drained'
            and c['status'] == 'proven_drained' and c['containment_verified'] is True and c['echild'] is True
            and c['errors'] == [] and c['reporting_errors'] == []
            and all(e['reaped'] is True for e in c['entries']), 'recorded authenticated drain failed')
    require(supervisor['private_authentication_replayed'] is False, 'projection cannot replay private authentication')
    idle = journal['post_idle']
    require(idle['all_eight_idle'] is True and idle['authenticated_echild'] is True and idle['recorded_owners_absent'] is True
            and idle['consecutive_idle'] >= idle['required_consecutive'] >= 3, 'independent idle absent')
    for index in idle['accepted_sample_indices']:
        sample = idle['samples'][index]
        require(len(sample['devices']) == len({d['index'] for d in sample['devices']}) == 8
                and all(d['memory_used_mib'] == d['utilization_percent'] == 0 for d in sample['devices'])
                and sample['compute_applications'] == 0, 'all-eight idle sample mismatch')
    closure = life['closure']
    require(closure['errors'] == [] and closure['sticky_unknown'] is False, 'sticky unknown closure')
    raw = [e['record'] for e in events if e['record']['fields']['event'] == 'resource_shutdown_summary']
    require(len(raw) == len(closure['attempts']) == 1, 'shutdown journal count changed')
    summary = decode(raw[0]['fields']['summary_json'])
    require(summary == closure['attempts'][0]['summary'] and raw[0] == closure['attempts'][0]['raw'], 'shutdown summary differs from log')
    require(summary['outcome'] == 'complete' and summary['error'] is None
            and summary['runtime']['closure'] == {'state': 'observed', 'value': 'closed'}, 'runtime not closed')
    physical = summary['physical']
    require(physical['expert_read_pipeline'] == {'state': 'observed', 'value': ledger}
            and physical['close'] == {'state': 'observed', 'value': 'closed'}
            and physical['residency'] == {'state': 'observed', 'value': 'retained'}
            and physical['quiescence'] == {'state': 'verified', 'value': 'quiescent'}
            and physical['release'] == {'state': 'unobserved'}, 'physical retained/closed/unobserved-release evidence changed')
    final = closure['final']
    require(final['can_continue'] is True and final['custody_state'] == 'known_closed_owner_local'
            and final['owner_retained'] is True and final['unknown_reasons'] == []
            and final['release'] == {'state': 'unobserved', 'value': None}
            and final['resource_contract_complete'] == 'unverified' and final['universal_final_release_verified'] is False,
            'retained close incorrectly upgraded to physical release')
    same_birth(closure['source_binding']['server_identity'], life['server_identity'])
    require(life['physical_nas_backend_bytes'] is None, 'backend data invented')
    return final


def request_summary(requests, original):
    counts = collections.Counter(r['status'] for r in requests)
    require(counts == {'success': 12}, 'request failure/incomplete; cannot publish complete-study aggregate')
    measured = [r for r in requests if r['measured']]
    phases = collections.Counter(r['phase'] for r in requests)
    for name, count in phases.items():
        require(original['by_phase'][name] == dict(success=count, failed=0, censored=0, unissued=0), 'phase denominator mismatch')
    expected = dict(scheduled=len(requests), issued=sum(r['issued'] for r in requests), terminal=len(requests),
                    successful=counts['success'], failed=0, censored=0, unissued=0)
    require(all(original[k] == v for k, v in expected.items()) and original['conservation'] is True
            and original['counts'] == dict(success=12, failed=0, censored=0, unissued=0)
            and original['reasons'] == {} and len(measured) == 5
            and original['measured_qa'] == dict(scheduled=5, success=5, eligible_success=5, request_only_eligible_success=5),
            'request/QA denominator conservation')


def rebuild(root):
    study = data(root, 'study.json.gz')
    design_order = load(root, 'design/order.json')
    schedule = load(root, 'design/requests.json')['schedule']
    prereg = data(root, 'design/preregistration.json')
    protocol = data(root, 'design/protocol.json')
    cases = data(root, 'design/cases.json')
    build = data(root, 'design/source-and-build.json')
    oracle_data = data(root, 'design/oracles.json')
    oracles = {r['id']: r for r in oracle_data['references']}
    expected_order = [row['block'] + '-' + mode for row in design_order['blocks'] for mode in row['modes']]
    require(design_order['seed'] == 20261015 and study['order'] == prereg['order'] == design_order['blocks'], 'order/design changed')
    require(study['trials'] == expected_order and len(set(study['trials'])) == 10, 'missing pair/duplicate arm/wrong execution order')
    rank = sorted(range(1, 6), key=lambda i: sha(('20261015:P' + str(i)).encode()))[:3]
    require(all(b['modes'] == (['blocking', 'lookahead'] if i in rank else ['lookahead', 'blocking'])
                for i, b in enumerate(design_order['blocks'], 1)), 'counterbalancing seed mismatch')
    require(study['status'] == 'complete' and study['actual_completed_lifecycles'] == study['actual_issued_lifecycles'] == 10,
            'study not fully complete')
    require(study['retries'] == study['replacements'] == study['extra_pilot_lifecycles'] == study['profile_lifecycles'] == 0,
            'retries/replacements/profiling contaminates study')
    require(study['physical_nas_backend_bytes'] is None and study['universal_final_release_verified'] is False
            and study['resource_contract_complete'] == 'unverified' and study['scientific_acceptance'] is False,
            'unsupported physical/scientific claim')
    current = build['candidate_identity']
    require(current['commit'] == SOURCE_COMMIT and current['cli_binary_sha256'] == CLI_SHA256
            and current['source_archive_sha256'] == SOURCE_ARCHIVE_SHA256 and current['source_members'] == 1484,
            'actual immutable binary/source mismatch')
    require(protocol['scheduled_requests'] == 120 and protocol['scheduled_measured_qa'] == 50 and len(schedule) == 12,
            'preregistered denominator changed')
    for mode in ('blocking', 'lookahead'):
        require(cases[mode]['public_runtime_controls']['FERRULE_PROFILE_NVTX'] == '0', 'runtime NVTX enabled')
    configs, rows, arms, requests, maps = [], [], [], [], []
    for trial in expected_order:
        prefix = 'arms/' + trial + '/'
        journal = data(root, prefix + 'journal.json')
        life = data(root, prefix + 'lifecycle.json.gz')
        config = data(root, prefix + 'configuration.json')
        configs.append(config)
        mode = journal['mode']
        require(trial == journal['trial'] == life['trial'] == config['trial'] and trial == journal['block'] + '-' + mode, 'journal arm identity')
        require(config['argv'][1:] == cases[mode]['argv_after_binary']
                and config['public_runtime_controls'] == cases[mode]['public_runtime_controls']
                and config['candidate_sha256'] == study['candidate_sha256']
                and config['execution_sha256'] == study['execution_sha256'], 'actual configuration differs from frozen design')
        ledger_report = load(root, prefix + 'read-ledger-assessment.json')
        require(journal['assessment'] == data(root, prefix + 'assessment.json')
                and journal['assessment']['can_continue'] is True and journal['assessment']['stop_reasons'] == []
                and journal['read_ledger'] == ledger_report and ledger_report['can_continue'] is True, 'lifecycle assessment differs')
        ledger = ledger_report['final_read_ledger']
        read_ledger(ledger, mode)
        events = [decode(line) for line in unpack(root / (prefix + 'server-events.jsonl.gz')).splitlines()]
        event_source = data(root, prefix + 'server-events-source.json')
        counts = dict(collections.Counter(e['record']['fields']['event'] for e in events))
        require(counts == event_source['event_counts'] and len(events) == event_source['selected_events'], 'server event inventory differs')
        require(all(e['source_byte_start'] < e['source_byte_end'] <= event_source['original_bytes'] for e in events)
                and all(a['source_byte_end'] <= b['source_byte_start'] and a['source_line'] < b['source_line'] for a, b in zip(events, events[1:])), 'log source positions invalid')
        closure = qualify_close(journal, life, events, ledger)
        nfs = data(root, prefix + 'nfs-snapshots.json.gz')
        expected_snapshots = {r['name'] + '-' + suffix for r in schedule for suffix in ('pre', 'post')} | {
            'prelaunch', 'ready', 'requests-pre', 'requests-post', 'shutdown-pre', 'shutdown-post'}
        require(set(nfs) == expected_snapshots, 'NFS request/phase endpoint coverage')
        phase_deltas = {}
        for phase, before, after in (('startup_nfs', 'prelaunch', 'ready'), ('requests_nfs', 'requests-pre', 'requests-post'),
                                     ('shutdown_nfs', 'shutdown-pre', 'shutdown-post')):
            phase_deltas[phase] = nfs_delta(nfs[before], nfs[after], life[phase])
        public_maps = data(root, prefix + 'library-evidence.json')
        require(set(public_maps) == {'ready', 'after-requests'}, 'map phase coverage')
        for phase in ('ready', 'after-requests'):
            maps.append(map_evidence(public_maps[phase], trial, phase, life['server_identity'], build['runtime_pins']))
        previous, transaction = None, 0
        arm_requests, observations = [], []
        for slot in schedule:
            request_prefix = prefix + slot['name'] + '/'
            r = data(root, request_prefix + 'observed.json.gz')
            actual = r['actual']
            tokens = client_tokens(r, slot, load(root, request_prefix + 'http-terminal.json'),
                                   data(root, request_prefix + 'http.json.gz'), load(root, request_prefix + 'issued.json'), events, oracles)
            numeric, previous, transaction = invocations(r, events, mode, previous, transaction)
            io = pid_io(r['process_io'], life['server_identity'], actual)
            nfs_change = nfs_delta(nfs[r['name'] + '-pre'], nfs[r['name'] + '-post'], r['nfs'])
            record = dict(trial=trial, name=r['name'], phase=r['phase'], measured=r['measured'],
                          actual_http_ids=r['trace']['actual_http_completion_ids'], worker_request_id=r['trace']['actual_worker_request_id'],
                          actual_generated_ids=tokens, enqueue_ordinals=r['trace']['enqueue_ordinals'],
                          exact_frozen_oracle=r['trace']['exact_token_pass'], usage=actual['usage'],
                          ttft_seconds=actual['ttft_seconds'], request_e2e_seconds=actual['request_e2e_seconds'],
                          quality_passed=True, **numeric, pid_io_delta=io, nfs_client_delta=nfs_change,
                          pid_io_birth_bound=True, log_window=[r['log_start_byte'], r['log_end_byte']],
                          binding_scope='Serial request log-byte window; not transaction-to-request causal attribution')
            arm_requests.append(record)
            requests.append(record)
            observations.append(r)
        require(transaction == counts['mixed_decoder_invocation'] == 102
                and counts['request_worker_token_enqueued'] == sum(r['usage']['completion_tokens'] for r in arm_requests),
                'unaccounted invocation/token events')
        require(previous == ledger['stats'], 'last invocation differs from shutdown owner ledger')
        total = summed([r['pipeline_nonpeak_deltas'] for r in arm_requests])
        require(total == {k: v for k, v in ledger['stats'].items() if not k.startswith('peak_')}, 'request deltas do not sum to owner lifetime')
        require(all(v == 0 for v in ledger['current_charge'].values()), 'final logical charge not zero')
        request_summary(observations, journal['summary'])
        request_only_summary = dict(journal['summary'], measured_qa={k: v for k, v in journal['summary']['measured_qa'].items()
                                                                  if k != 'request_only_eligible_success'})
        require(life['requests']['summary'] == request_only_summary and life['whole_log_token_join'] == [r['trace'] for r in observations],
                'whole-log token join/request denominator differs')
        require(life['startup_logical_counters']['rows'] == [] and life['startup_logical_counters']['valid'] is False
                and all(v is None for v in life['startup_logical_counters']['deltas'].values()), 'startup logical missingness concealed')
        require(life['status'] == 'complete' and life['capture']['returncode'] == 0, 'server lifecycle incomplete')
        res = life['resources']
        selected = config['selected_device_alias']
        require(res['status'] == 'observed_within_targets' and res['errors'] == []
                and all(v == 0 for k, v in res['delta_high_water_bytes'].items() if k != selected)
                and res['gpu_peak_bytes'][selected] <= 7168 << 20, 'resource target/other-device violation')
        require(res['owned_gpu_attribution'] == 'unknown' and res['owned_process_gpu_peak_bytes'] is None
                and res['exact_peaks'] is False and res['continuous_coverage'] is False and res['hard_memory_partition'] is False,
                'sampled memory upgraded to exact owned/hard-cap proof')
        attempt = load(root, prefix + 'attempt.json')
        require(attempt['trial'] == trial and attempt['replay_permitted'] is False
                and journal['coordinator_ended']['monotonic'] <= attempt['outer_deadline'] <= attempt['pilot_deadline'], 'lifecycle deadline/replay')
        require(data(root, prefix + 'locks.json')['all_eight_nonblocking_reacquired'] is True, 'lock reprobe missing')
        journal['lifecycle'] = dict(life, requests=dict(life['requests'], observations=observations))
        rows.append(journal)
        arms.append(dict(trial=trial, mode=mode, metrics=prespecified.arm_metrics(journal),
                         quality=dict(requests=len(observations), exact_oracle_requests=sum(r['reference'] is not None for r in observations),
                                      length_only_smoke_requests=sum(r['reference'] is None for r in observations),
                                      actual_generated_tokens=sum(r['usage']['completion_tokens'] for r in arm_requests),
                                      measured_qa=sum(r['measured'] for r in observations), passed=True),
                         startup_seconds=life['startup_seconds'],
                         coordinator_seconds=journal['coordinator_ended']['monotonic'] - attempt['admission']['monotonic'],
                         logical_counters=summed([r['logical_counters'] for r in arm_requests]), pipeline_stats=ledger['stats'],
                         final_logical_charge=ledger['current_charge'], peak_device_sample_bytes=res['gpu_peak_bytes'][selected],
                         peak_tree_pss_sample_bytes=res['tree_pss_peak_bytes'], memory_sample_count=res['memory_components']['sample_count'],
                         nonselected_delta_bytes=0, owned_gpu_byte_attribution=res['owned_gpu_attribution'], nfs_phase_client_deltas=phase_deltas,
                         nfs_phase_coverage=len(phase_deltas), nfs_request_coverage=len(observations), pid_io_request_pairs=len(observations),
                         pid_io_startup_shutdown_coverage='unobserved: maintained collector only samples request boundaries',
                         runtime_closed=True, owner_local_close_known=True, physical_residency=closure['owner_retained'],
                         physical_release=closure['release'], final_authenticated_echild=True, final_all_eight_idle=True, locks_reprobed=True))
    for config in configs:
        same = list(config['argv'])
        same[same.index('--expert-read-schedule') + 1] = 'MODE'
        first = list(configs[0]['argv'])
        first[first.index('--expert-read-schedule') + 1] = 'MODE'
        require(same == first and config['selected_device_alias'] == configs[0]['selected_device_alias'], 'more than schedule differs across arms')
    analysis = prespecified.summarize(dict(order=study['order'], lifecycles=rows))
    require(analysis == study['analysis'], 'full five-pair prespecified analysis differs')
    audit = data(root, 'poststudy-audit.json.gz')
    require(analysis == audit['analysis'], 'poststudy analysis differs')
    for key, rebuilt in (('arms', arms), ('requests', requests), ('map_evidence', maps)):
        require(rebuilt == audit[key], 'poststudy ' + key + ' recomputation differs')
    cleanup = load(root, 'final-cleanup.json')
    require(cleanup == audit['cleanup'] and all(cleanup[k] is True for k in ('final_authenticated_echild',
            'all_recorded_owned_pid_births_absent', 'all_eight_idle', 'all_eight_nonblocking_reacquired', 'locks_released', 'within_overall_deadline')),
            'final cleanup evidence mismatch')
    close(audit['elapsed_study_seconds'], study['ended']['monotonic'] - study['started']['monotonic'], 'study duration differs')
    history = data(root, 'historical-failures.json')
    require(len(history) == 2, 'historical attempts missing')
    for old in history:
        require(old['status'] == 'stopped' and old['actual_issued_lifecycles'] == 1 and old['actual_completed_lifecycles'] == 0
                and old['pooled'] is False and old['resumed'] is False and old['analysis']['aggregate'] is None
                and old['analysis']['denominators']['unissued'] == 120 and old['analysis']['measured_qa']['scheduled'] == 50
                and prespecified.summarize(old) == old['analysis'], 'historical failed denominators/pooling changed')
    coverage = dict(requests=len(requests), exact_eight_token_oracles=sum(r['exact_frozen_oracle'] is True for r in requests),
                    length_only_two_token_smoke=sum(r['exact_frozen_oracle'] is None for r in requests),
                    actual_worker_generated_tokens=sum(r['usage']['completion_tokens'] for r in requests),
                    measured_qa=sum(r['measured'] for r in requests), pipeline_invocation_jsons=sum(r['decoder_invocations'] for r in requests),
                    request_pid_io_pairs=len(requests), request_pid_io_snapshots=2 * len(requests),
                    nfs_request_deltas=len(requests), nfs_phase_deltas=sum(a['nfs_phase_coverage'] for a in arms),
                    library_snapshots=len(maps), required_libraries_per_snapshot=12, startup_shutdown_pid_io_deltas=0)
    require(coverage == audit['coverage'], 'audit coverage differs')
    enumerated = [{metric: statistics.median(analysis['blocks'][i]['log_ratios'][metric] for i in draw)
                   for metric in prespecified.METRICS} for draw in itertools.product(range(5), repeat=5)]
    return dict(schema='ferrule.paired-http-offline-reconstruction.v1', status='passed', analysis=analysis,
                arms=arms, requests=requests, map_evidence=maps, coverage=coverage,
                bootstrap_draw_count=len(enumerated), bootstrap_draw_order='lexicographic P1..P5 indices, repeat=5',
                bootstrap_draws_sha256=sha(encode(enumerated)), elapsed_study_seconds=audit['elapsed_study_seconds'],
                historical_studies=[dict(study=h['study'], attempted_lifecycles=1, completed_lifecycles=0,
                                        unissued_requests=h['analysis']['denominators']['unissued'],
                                        measured_qa=h['analysis']['measured_qa'], pooled=False) for h in history],
                startup_pid_io_delta=None, shutdown_pid_io_delta=None, physical_nas_backend_bytes=None,
                physical_release_verified=False, private_authentication_replayed=False,
                limitations=audit['limitations'], collection_deviations=audit['collection_deviations'],
                scientific_acceptance=False)


def verify_bootstrap(root, result):
    import math
    saved = load(root, 'bootstrap-enumeration.json.gz')
    require(saved['count'] == len(saved['draws']) == 3125 and saved['nearest_rank_endpoints'] == [79, 3047],
            'bootstrap enumeration coverage changed')
    logs = []
    for indices, draw in zip(itertools.product(range(5), repeat=5), saved['draws']):
        expected = {metric: statistics.median(result['analysis']['blocks'][i]['log_ratios'][metric] for i in indices)
                    for metric in prespecified.METRICS}
        require(draw['blocks'] == ['P' + str(i + 1) for i in indices] and draw['log_ratios'] == expected
                and draw['ratios'] == {k: math.exp(v) for k, v in expected.items()}, 'bootstrap draw differs')
        logs.append(expected)
    require(sha(encode(logs)) == saved['log_ratio_rows_sha256'] == result['bootstrap_draws_sha256'], 'bootstrap draw hash differs')


def verify(root=HERE, compare_result=True):
    members = verify_members(root)
    pins = provenance(root, load(root, 'manifest.json'))
    result = rebuild(root)
    verify_bootstrap(root, result)
    if compare_result:
        require(load(root, 'recomputed.json.gz') == result, 'saved offline reconstruction stale')
    return result, dict(**members, **pins)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=HERE)
    parser.add_argument('--json', action='store_true', help='print full independently recomputed report to stdout')
    args = parser.parse_args()
    result, inventory = verify(args.root.resolve())
    if args.json:
        print(encode(result).decode(), end='')
    else:
        print(json.dumps(dict(status=result['status'], inventory=inventory, coverage=result['coverage'],
                              aggregate=result['analysis']['aggregate'], bootstrap_draw_count=result['bootstrap_draw_count'],
                              physical_nas_backend_bytes=None, startup_pid_io_delta=None, shutdown_pid_io_delta=None), indent=2))


if __name__ == '__main__':
    try:
        main()
    except (Invalid, KeyError, TypeError, ValueError, OSError) as error:
        raise SystemExit('FAIL: ' + str(error)) from error
