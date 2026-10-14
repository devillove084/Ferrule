"""CPU-only full-export bidirectional correlation; closure is not completeness."""
from collections import Counter, defaultdict
import re

API_KINDS = {
    'cudaLaunchKernel': 'kernel', 'cuLaunchKernel': 'kernel',
    'cuMemcpyHtoDAsync': 'memcpy', 'cuMemcpyDtoHAsync': 'memcpy',
    'cuMemcpyDtoDAsync': 'memcpy', 'cudaMemcpyAsync': 'memcpy',
    'cuMemsetD8Async': 'memset', 'cudaMemsetAsync': 'memset',
}


def process_id(global_id):
    return global_id // 0x1000000 % 0x1000000


def api_kind(name):
    return API_KINDS.get(re.sub(r'_v\d+$', '', name))


def containing_window(row, windows):
    matches = [label for label, w in windows.items()
               if w['start_ns'] <= row['start'] and row['end'] <= w['end_ns']]
    if len(matches) > 1:
        raise ValueError('Overlapping request windows')
    return matches[0] if matches else None


def closure_status(counts, issues, overlaps):
    closed = (all(counts.get(kind+'_observed', 0) > 0 for kind in ('kernel', 'memcpy', 'memset'))
              and not issues and overlaps == 0)
    return dict(status='observed_correlation_closed' if closed else 'observed_correlation_not_closed',
                observed_correlation_closed=closed, full_trace_accepted=False,
                paired_api_activity_loss_excluded=False)


def audit_correlations(activities, api_rows, windows, model_pid):
    """One activity per direct submission for the API_KINDS used in this report.

    Counterparts are sought in the full export by (PID, correlationId), then
    checked for uniqueness, kind, success and same-window membership. Unsupported
    submission kinds cannot qualify activity rows. All API intervals participate
    in same-thread overlap checks, including intervals crossing window bounds.
    """
    by_activity, by_api = defaultdict(list), defaultdict(list)
    counts = {label: Counter() for label in windows}
    issues = {label: [] for label in windows}
    intervals = defaultdict(list)
    submissions = []

    def check_bounds(row, label, direction):
        if row['end'] < row['start']:
            raise ValueError('Negative interval')
        for other, w in windows.items():
            if label != other and row['start'] < w['end_ns'] and row['end'] > w['start_ns']:
                issues[other].append(dict(direction=direction, reason='row_crosses_request_boundary',
                    correlation_id=row['correlationId'], start_ns=row['start'], end_ns=row['end']))

    for row in activities:
        key = (process_id(row['globalPid']), row['correlationId'])
        by_activity[key].append(row)
        if key[0] != model_pid:
            continue
        label = containing_window(row, windows)
        check_bounds(row, label, 'activity')
        if label:
            counts[label][row['kind']+'_observed'] += 1

    for row in api_rows:
        key = (process_id(row['globalTid']), row['correlationId'])
        kind = api_kind(row['name'])
        if key in by_activity or kind:
            by_api[key].append(row)
        if key[0] != model_pid:
            continue
        label = containing_window(row, windows)
        check_bounds(row, label, 'api')
        intervals[row['globalTid']].append((row['start'], row['end'], row['correlationId']))
        if label:
            counts[label]['runtime_api_rows'] += 1
        if kind and row['returnValue'] == 0:
            submissions.append((key, row, kind, label))
            if label:
                counts[label][kind+'_successful_api'] += 1
                if kind == 'kernel':
                    counts[label]['launch_runtime' if row['name'].startswith('cuda') else 'launch_driver'] += 1

    for key, group in by_activity.items():
        if key[0] != model_pid:
            continue
        for row in group:
            label = containing_window(row, windows)
            if label is None:
                continue
            candidates = by_api.get(key, [])
            reason = None
            if not candidates:
                reason = 'missing_api_in_full_export'
            elif len(candidates) != 1:
                reason = 'nonunique_api_correlation'
            elif candidates[0]['returnValue'] != 0:
                reason = 'api_not_successful'
            elif api_kind(candidates[0]['name']) != row['kind']:
                reason = 'unsupported_or_wrong_api_kind'
            elif len(group) != 1:
                reason = 'nonunique_activity_correlation'
            elif containing_window(candidates[0], windows) != label:
                reason = 'api_outside_request_window'
            if reason:
                issues[label].append(dict(direction='activity_to_api', reason=reason, kind=row['kind'],
                    pid=key[0], correlation_id=key[1], start_ns=row['start'], end_ns=row['end'],
                    api_matches=len(candidates), activity_matches=len(group)))
            else:
                counts[label][row['kind']+'_activity_to_api_unique'] += 1

    for key, row, kind, label in submissions:
        if label is None:
            continue
        matches = by_activity.get(key, [])
        reason = None
        if not matches:
            reason = 'missing_activity_in_full_export'
        elif len(matches) != 1:
            reason = 'nonunique_activity_correlation'
        elif matches[0]['kind'] != kind:
            reason = 'wrong_activity_kind'
        elif len(by_api[key]) != 1:
            reason = 'nonunique_api_correlation'
        elif containing_window(matches[0], windows) != label:
            reason = 'activity_outside_request_window'
        if reason:
            issues[label].append(dict(direction='api_to_activity', reason=reason, kind=kind,
                api=row['name'], pid=key[0], correlation_id=key[1], start_ns=row['start'], end_ns=row['end'],
                activity_matches=len(matches)))
        else:
            counts[label][kind+'_api_to_activity_unique'] += 1

    overlaps, examples = Counter(), defaultdict(list)
    for tid, rows in intervals.items():
        active_end = active_corr = None
        for start, end, corr in sorted(rows):
            if active_end is not None and start < active_end:
                for label, w in windows.items():
                    if max(start, w['start_ns']) < min(active_end, end, w['end_ns']):
                        overlaps[label] += 1
                        if len(examples[label]) < 20:
                            examples[label].append(dict(globalTid=tid, previous_correlation_id=active_corr,
                                correlation_id=corr, start_ns=start, previous_end_ns=active_end))
            if active_end is None or end > active_end:
                active_end, active_corr = end, corr

    return {label: dict(closure_status(counts[label], issues[label], overlaps[label]),
        request_window_ns=[w['start_ns'], w['end_ns']], counts=dict(counts[label]), issues=issues[label],
        same_thread_runtime_overlaps=overlaps[label], overlap_examples=examples[label],
        matching_policy='Full-export (PID, correlation ID), unique successful API, correct activity kind, same request window',
        limitation='Exported-row closure only; simultaneous API/activity loss remains possible. Not a complete-report claim.')
        for label, w in windows.items()}
