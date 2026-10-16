"""Retained-policy extension of the maintained v3 endpoint control, CPU only."""
import quality as lifecycle
from runnerhelpers.evidence_v2 import integer, require

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
