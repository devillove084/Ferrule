"""Retained-policy extension of the maintained v3 endpoint control, CPU only."""
import quality as lifecycle
from runnerhelpers.evidence_v2 import ct, integer, require

POLICY = dict(mode='retained', policy_headroom_bytes=16 << 20, readiness_timeout_ms=30_000,
              host_quota_unchanged=True, physical_overhead_measured=False, gpu_overlap_claim=False)
PAYLOAD_BYTES = 3 * (512 * 2048 + 64 * 2)
TICKET_BYTES = 584
MINIMUM_AGGREGATE_BYTES = 3 * PAYLOAD_BYTES + TICKET_BYTES + (16 << 20) + (8 << 20) + (2 << 20)
assert MINIMUM_AGGREGATE_BYTES == 36_701_896
UPLOAD_FIELDS = ('upload_reserved_packed_bytes', 'pinned_upload_bytes',
    'upload_ticket_metadata_bytes', 'upload_policy_headroom_bytes', 'unknown_upload_bytes')


def validate_policy(report, mode):
    config = report.get('test_config', {})
    effective = report.get('effective_config', {})
    for container in (config, effective):
        policy = container.get('expert_transfer_policy')
        require(isinstance(policy, dict) and policy == POLICY
            and all(type(policy[k]) is type(value) for k, value in POLICY.items()),
            'Retained policy must have exact values and JSON types')
    require(config.get('name') == 'storage-middle64-retained-' + mode
        and config.get('expert_transfer_policy') == POLICY
        and effective.get('expert_transfer_policy') == POLICY
        and effective.get('expert_transfer_selection') == 'explicit existing-owner capability before first prediction',
        'Retained opt-in policy missing/default-off/mismatched')


def validated_cache_pairs(report):
    """Shared settled-cache evidence; no retained policy or reservation minimum."""
    pairs = []
    previous_uploads = None
    for call in report['expert_read_quality']['calls']:
        caches = []
        for endpoint in ('before', 'after'):
            physical = call[endpoint]
            item = physical.get('expert_cache', {})
            require(isinstance(item, dict) and item.get('state') == 'observed'
                and isinstance(item.get('value'), dict), 'Actual cache-upload evidence unobserved')
            cache = item['value']
            require(all(integer(cache.get(k)) for k in ('uploads', 'evictions', 'hits'))
                and cache.get('quarantined') is False and cache.get('counters_overflowed') is False
                and type(cache.get('pending_upload_bytes')) is int and cache['pending_upload_bytes'] == 0
                and type(cache.get('unknown_quarantine_bytes')) is int and cache['unknown_quarantine_bytes'] == 0,
                'Cache pending/Unknown/overflowed/missing')
            require(all(integer(cache.get(k)) for k in
                ('resident_entries', 'resident_bytes', 'scratch_bytes', 'charged_bytes', 'peak_bytes'))
                and cache['resident_entries'] <= 64 and cache['scratch_bytes'] <= 8 << 20
                and cache['charged_bytes'] == cache['resident_bytes'] + cache['pending_upload_bytes'] + cache['scratch_bytes']
                and cache['charged_bytes'] <= cache['peak_bytes'] <= 320 << 20,
                'Cache charged limits/conservation differ')
            require(previous_uploads is None or cache['uploads'] >= previous_uploads,
                    'Cache-upload counters reset')
            previous_uploads = cache['uploads']
            caches.append(cache)
        pairs.append(caches)
    return pairs


def validate(report, mode):
    validate_policy(report, mode)
    lifecycle.read_quality_v3(report, mode)
    pairs = validated_cache_pairs(report)
    positive_upload_calls = 0
    for call, caches in zip(report['expert_read_quality']['calls'], pairs):
        reads = []
        for endpoint in ('before', 'after'):
            physical = call[endpoint]
            read = physical['expert_read_pipeline']['value']
            charge = read['current_charge']
            # These fields are omitted only when zero by the committed
            # serializer; a present charge is never treated as refunded.
            require(all(type(charge.get(k, 0)) is int and charge.get(k, 0) == 0 for k in UPLOAD_FIELDS),
                    'Outstanding/Unknown retained upload charge at proof endpoint')
            reads.append(read)
        if call['positive_worker_read']:
            require(reads[1]['stats']['peak_host_charged_bytes'] >= MINIMUM_AGGREGATE_BYTES,
                    'No positive retained aggregate reservation evidence')
            if caches[1]['uploads'] > caches[0]['uploads']:
                positive_upload_calls += 1
    require(positive_upload_calls > 0, 'Retained upload path not actually exercised')


GROUPED_POLICY = dict(mode='grouped_synchronous', source_custody='existing_bounded_read',
    host_quota_unchanged=True, additional_pinned_bytes=0, physical_overhead_measured=False, gpu_overlap_claim=False)
GROUPED_MINIMUM_AGGREGATE_BYTES = 2 * PAYLOAD_BYTES + (8 << 20) + (2 << 20)
assert GROUPED_MINIMUM_AGGREGATE_BYTES == 16_777_984


def validate_grouped(report, mode):
    for container in (report.get('test_config', {}), report.get('effective_config', {})):
        policy = container.get('expert_transfer_policy')
        require(isinstance(policy, dict) and policy == GROUPED_POLICY
            and all(type(policy[k]) is type(v) for k, v in GROUPED_POLICY.items()),
            'Exact grouped policy values/types required; no synchronous/retained crossbinding')
    require(report['test_config'].get('name') == 'storage-middle64-grouped-' + mode
        and report['effective_config'].get('expert_transfer_selection') ==
            'explicit existing-owner capability before first prediction', 'Grouped opt-in/default-off selection differs')
    lifecycle.read_quality_v3(report, mode)
    positive_upload_calls = 0
    for call, caches in zip(report['expert_read_quality']['calls'], validated_cache_pairs(report)):
        for endpoint in ('before', 'after'):
            charge = call[endpoint]['expert_read_pipeline']['value']['current_charge']
            require(all(type(charge.get(k, 0)) is int and charge.get(k, 0) == 0 for k in UPLOAD_FIELDS),
                    'Grouped source/Unknown charge outstanding at proof endpoint')
        if call['positive_worker_read']:
            peak = call['after']['expert_read_pipeline']['value']['stats']['peak_host_charged_bytes']
            require(GROUPED_MINIMUM_AGGREGATE_BYTES <= peak <= 64 << 20,
                    'Grouped bounded-read aggregate reservation missing/over quota')
            positive_upload_calls += caches[1]['uploads'] > caches[0]['uploads']
    require(positive_upload_calls > 0, 'Grouped actual cache-upload path not exercised')


GROUPED_STAGE = ct.ROOT / 'target/validation/grouped-upload-ce6d4ff-gpu-20261017-01'
GROUPED_COMMIT = 'ce6d4ff28b96d4057d0de6dfa858707104aad16b'
GROUPED_SOURCE_SHA = '4e15b7301f678424878ba97c3a3df8f8533a696053d2eb58b36bb35e6895e135'
GROUPED_SOURCE_DIGEST = '7405b6fd931dc5986a5a1cc745e9979f1ceac64b029e50e82708b4ef3ed68294'
GROUPED_BINARIES = dict(backend='9ecac19e5f0e9929134ae6ebda3aedb66493f46fcdde36269ef662c18265d722',
    model='ffde27858ad59584ac7bd58309792a3865db3bc21ab72d56b23cb6def2355a9d',
    composition='d8c0bae4872ffbbb6828dd7f49acecb4de0f69ab32748662845ddaa6d0b30557')


def grouped_quality(plan):
    return plan is not None and plan.get('schema') == 'ferrule.grouped-upload.current-quality-plan.v1'


def grouped_jobs():
    rows = []
    def add(name, role, selector, kind='model_contract', schedule=None):
        rows.append(dict(id=f'{len(rows)+1:02d}-{name}', kind=kind, schedule=schedule, role=role,
            selector=selector, executable=str(GROUPED_STAGE / 'staged' / role),
            binary_sha256=GROUPED_BINARIES[role], seconds=1200,
            phase='primary' if kind == 'fresh' else 'quality', transfer_mode='grouped', repetition=None))
    add('backend-group-sync', 'backend',
        'cuda::operator_owner::numeric_fp8::group_sync::contract_tests::gpu::public_group_sync_matches_three_baseline_uploads_both_scale_types')
    retained = 'transformer::standard::cuda::residency::retained_upload::tests::'
    for name, selector in (('model-root-route', 'real_root_route_rejects_metadata_less_provider_before_payload_read'),
            ('model-worker-route', 'real_worker_route_rejects_metadata_less_provider_before_payload_read'),
            ('model-owner-lifecycle', 'real_owner_opt_in_matches_default_and_reloads_only_after_consumer_proof')):
        add(name, 'model', retained + selector)
    grouped = 'transformer::standard::cuda::residency::grouped_upload::tests::'
    add('model-group-policy', 'model', grouped + 'real_owner_group_policy_rejects_unsupported_duplicate_revert_and_cross_policy')
    add('model-group-admission', 'model', grouped + 'real_owner_group_admits_all_six_destinations_plus_scratch_before_payload_read')
    prefix = 'model_factory::dense_bf16_tests::full40::'
    for kind in ('full40', 'fresh', 'ordinary', 'cold'):
        for mode in ('blocking', 'lookahead'):
            selector = (prefix + 'numeric_bf16_mixed_storage_middle64_grouped_' + mode +
                '_public_factory_full40_logits_routes_states_match_profile_reference' if kind == 'full40' else
                prefix + 'fresh_quality::mixed_storage_middle64_grouped_' + mode + '_fresh_quality_' +
                dict(fresh='retained_generation', ordinary='no_observer_equality', cold='cold_owner_restart')[kind])
            add(mode + '-' + kind, 'composition', selector, kind, mode)
    return rows


GROUPED_FIXED_INPUTS = {'composition_capacity_and_options': {'allocator_margin_bytes': 536870912,
                                      'ctx_size': 128,
                                      'device_bytes': 7516192768,
                                      'enable_native_proposals': False,
                                      'expert_cache': 'ExpertCacheOptions { host_entries: 256, '
                                                      'host_mebibytes: 0, pinned_entries: 64, '
                                                      'pinned_mebibytes: 0 }',
                                      'expert_reader_max_tensor_mebibytes': 64,
                                      'kv_bytes': 16777216,
                                      'kv_cache_mebibytes': 16,
                                      'max_batch_tokens': 8,
                                      'max_bytes': 335544320,
                                      'max_experts': 64,
                                      'max_layers': None,
                                      'max_positions': 128,
                                      'max_sequences': 2,
                                      'max_tensor_mebibytes': 1024,
                                      'moe_hotset_experts': 0,
                                      'numeric_fp8_profile': 'None',
                                      'output_head_chunk_rows': 4096,
                                      'qwen35_host_cache': 'None',
                                      'scheduler': {'allow_mixed_batches': True,
                                                    'decode_cohort_max_deferrals': 0,
                                                    'decode_cohort_target': 1,
                                                    'max_active_sequences': 2,
                                                    'max_batch_tokens': 8,
                                                    'max_decode_batch': 2,
                                                    'prefill_chunk_size': 8,
                                                    'prefill_reserve_tokens': 1,
                                                    'prefix_cache_capacity_pages': 0},
                                      'scratch_bytes': 8388608,
                                      'stop_at_eos': False,
                                      'workspace_bytes': 2147483648},
 'runtime_base_environment': {'PATH': '/usr/bin:/bin',
                              'HOME': '/root',
                              'CUDA_VISIBLE_DEVICES': 'GPU-b980f22b-58a3-fc88-21c1-ae86053d8535',
                              'NVIDIA_VISIBLE_DEVICES': 'GPU-b980f22b-58a3-fc88-21c1-ae86053d8535',
                              'CUBLAS_WORKSPACE_CONFIG': ':0:0',
                              'LD_LIBRARY_PATH': '/home/admin/wxw403883/anaconda3/envs/tao/lib:/usr/lib64',
                              'FERRULE_PROFILE_NVTX': '0',
                              'TMPDIR': '/tmp'},
 'uuid_inventory': ['GPU-b980f22b-58a3-fc88-21c1-ae86053d8535',
                    'GPU-63f1bd01-ddf9-a742-ee48-9389892493d9',
                    'GPU-0bc01d70-2ade-0714-8959-b25ae75469b6',
                    'GPU-dc825838-7c6c-2683-71e7-9ee0ac61b011',
                    'GPU-6c86e874-0ba8-b45a-a604-bca3b934bb86',
                    'GPU-ff61a997-b3fc-fd3d-800d-63bad94fd685',
                    'GPU-2cdfe4c5-38b1-ef88-fbdd-f6848ddbae9d',
                    'GPU-0ac006a6-932d-f77d-9abd-ea242dd3ba4f'],
 'fresh_reference_sha256': '102c798baeb5200bc8ed8386eabed6146870b3b6da2a90fc6010677041352c7e',
 'full40_reference_sha256': '351495921233e0158eb2a4507c415d173c895a6086d5905b7a72d82de71849f9'}


def validate_grouped_plan(plan):
    require(grouped_quality(plan) and plan.get('commit') == GROUPED_COMMIT
        and plan.get('source_archive_sha256') == GROUPED_SOURCE_SHA
        and plan.get('source_digest') == GROUPED_SOURCE_DIGEST, 'Wrong immutable grouped cut/archive/digest')
    require(ct.digest(plan.get('jobs')) == ct.digest(grouped_jobs()), 'Grouped exact14 selectors/order/artifact/deadline differs')
    artifacts = {role: dict(executable=str(GROUPED_STAGE / 'staged' / role), sha256=sha)
                 for role, sha in GROUPED_BINARIES.items()}
    require(ct.digest(plan.get('artifacts')) == ct.digest(artifacts), 'Grouped artifact role/path/hash differs')
    require(ct.digest(plan.get('host_options')) == ct.digest(dict(max_host_bytes=64 << 20,
        max_metadata_bytes=8 << 20, worker_stack_bytes=2 << 20)), 'Grouped host option limits differ')
    for name, value in GROUPED_FIXED_INPUTS.items():
        require(ct.digest(plan.get(name)) == ct.digest(value), 'Grouped fixed quota/runtime/reference differs: ' + name)
    require(plan.get('atol') == .00025 and plan.get('rtol') == .012,
            'Grouped numerical tolerance drift')
    require(ct.digest(plan.get('transfer_policy')) == ct.digest(GROUPED_POLICY), 'Grouped policy drift')
    baseline = str(ct.ROOT / 'target/validation/retained-upload-57e4629-gpu-20261016-01/retained-plan.json')
    require(plan.get('pins', {}).get(baseline) == '6041cae37cdd08795a980e912af3ab82b52cefe3a795f42fa2631513e284da76',
            'Grouped unchanged-quota/reference baseline must remain pinned')
    require(ct.digest(plan.get('capacity_and_options')) == ct.digest(GROUPED_FIXED_INPUTS['composition_capacity_and_options']),
            'Grouped capacity alias differs')
    for name in ('retries', 'extra_warmups', 'calibration', 'profiling_captures'):
        require(type(plan.get(name)) is int and plan[name] == 0, 'Grouped retries/probes/profile prohibited')
    require(plan.get('profiling') is False and plan.get('stop_on_first_failure') is True
        and plan.get('independent_review_required') is True
        and plan.get('measurement') == 'quality_only_no_performance_claim', 'Grouped scope/approval differs')
    return plan


def grouped_job(plan, gate):
    wanted = next((row for row in plan['jobs'] if row['id'] == gate.get('id')), None)
    require(wanted is not None and ct.digest(wanted) == ct.digest(gate), 'Grouped job/policy/source/deadline crossbinding')
    return wanted
