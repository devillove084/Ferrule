"""Pure offline evidence rules, not a model, ledger implementation, or executor."""
import copy
import math

NAMES = ("hello2", "canary-qa", "canary-arithmetic", "canary-chinese")
POINT_IDS = ("small", "middle", "big")
COUNTERS = ("checkpoint_read_bytes", "checkpoint_read_calls", "checkpoint_source_checks",
            "checkpoint_path_metadata_calls", "checkpoint_handle_metadata_calls",
            "expert_hits", "expert_uploads", "expert_evictions", "artifact_uploads",
            "artifact_upload_bytes", "h2d_copies", "h2d_bytes", "source_extents_read")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value):
    return type(value) is int and value >= 0


def positive_payload_entries(entries, cache_bytes, scratch_bytes, expert_payload_bytes):
    """Logical upper bound, not physical admission, occupancy, or working-set fit."""
    require(all(integer(v) for v in (entries, cache_bytes, scratch_bytes, expert_payload_bytes)),
            "nonnegative integer cache quantities required")
    require(entries > 0 and expert_payload_bytes > 0 and cache_bytes >= scratch_bytes,
            "positive payload/entries and nonnegative payload capacity required")
    return min(entries, (cache_bytes - scratch_bytes) // expert_payload_bytes)


def can_admit(phase_remaining_seconds):
    require(type(phase_remaining_seconds) in (int, float)
            and math.isfinite(phase_remaining_seconds), "finite remaining time required")
    return phase_remaining_seconds >= 240 + 180


def denominator_summary(records):
    """Every frozen slot remains; absent terminal after issue is censored, not unissued."""
    keys = {(point, name) for point in POINT_IDS for name in NAMES}
    require(set(records) <= keys, "unknown/replacement request slot")
    counts = dict(scheduled=12, issued=0, terminal=0, successful=0, failed=0, censored=0, unissued=0)
    for key in keys:
        row = records.get(key)
        if row is None:
            counts["unissued"] += 1
            continue
        require(type(row.get("issued")) is bool, "explicit issued flag required")
        status = row.get("status")
        if not row["issued"]:
            require(status in (None, "unissued", "scheduled"), "terminal without issue")
            counts["unissued"] += 1
            continue
        counts["issued"] += 1
        require(status in (None, "success", "failed", "censored"), "invalid issued outcome")
        if status in ("success", "failed"):
            counts["terminal"] += 1
            counts["successful" if status == "success" else "failed"] += 1
        else:
            counts["censored"] += 1
    return counts


def counter_totals(invocations):
    """Missing fields stay null; actual zero is not missing. Never invent misses."""
    result = {}
    for field in COUNTERS:
        values = [r.get(field) for r in invocations]
        result[field] = sum(values) if values and all(integer(v) for v in values) else None
    result.update(expert_cache_misses=None, cache_hit_rate=None,
                  working_set_unique_expert_ids=None, physical_nas_backend_bytes=None)
    return result


def charged_bytes(snapshot):
    """Quarantine is a subset of already charged bytes, not a fourth partition."""
    parts = [snapshot.get(k) for k in ("resident_bytes", "pending_upload_bytes", "scratch_bytes")]
    quarantine = snapshot.get("unknown_quarantine_bytes")
    if not all(integer(v) for v in parts) or not integer(quarantine):
        return dict(charged_bytes=None, unknown_quarantine_bytes=quarantine,
                    quarantine_subset_verified=None)
    charged = sum(parts)
    require(quarantine <= charged, "quarantine cannot exceed its charged superset")
    return dict(charged_bytes=charged, unknown_quarantine_bytes=quarantine,
                quarantine_subset_verified=True)


def copy_observation(snapshot):
    """Accept only serialized data; no live owner, poll, join, drain, or fence."""
    def validate(value):
        if value is None or type(value) in (bool, str, int):
            return
        if type(value) is float:
            require(math.isfinite(value), "nonfinite observation")
            return
        if type(value) is list:
            for item in value:
                validate(item)
            return
        require(type(value) is dict and all(type(k) is str for k in value),
                "plain serialized snapshot required, not an active owner")
        for item in value.values():
            validate(item)
    require(type(snapshot) is dict, "plain serialized snapshot required")
    validate(snapshot)
    return copy.deepcopy(snapshot)
