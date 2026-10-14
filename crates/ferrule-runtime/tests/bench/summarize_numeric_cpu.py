#!/usr/bin/env python3
"""Summarize an ignored numeric_process_plain_cpu report; stdlib only, no GPU."""
import argparse
import collections
import json
import statistics
from pathlib import Path


def summarize(report):
    records = report["records"]
    buckets = []
    for tokens in sorted({r["tokens"] for r in records}):
        rows = [r for r in records if r["tokens"] == tokens]
        buckets.append({
            "tokens": tokens,
            "buckets": len(rows),
            "command_bytes_mean": statistics.mean(r["command_bytes"] for r in rows),
            "frame_bytes_mean": statistics.mean(r["frame_bytes"] for r in rows),
            "reply_frame_bytes_mean": statistics.mean(r["reply_frame_bytes"] for r in rows),
            "median_ns": {k: statistics.median(r["ns"][k] for r in rows) for k in rows[0]["ns"]},
            "domain_median_ns": {k: statistics.median(r["domain_ns"][k] for r in rows) for k in rows[0]["domain_ns"]},
        })
    cases = []
    for case, stage in sorted({(r["case"], r["stage"]) for r in records}):
        rows = [r for r in records if (r["case"], r["stage"]) == (case, stage)]
        histogram = collections.Counter(r["tokens"] for r in rows)
        histogram[0] = 40 * 8 - len(rows)
        owners = []
        for owner in range(8):
            selected = [r for r in rows if r["owner"] == owner]
            owners.append({
                "owner_slot": owner,
                "buckets": len(selected),
                "tokens": sum(r["tokens"] for r in selected),
                "frame_bytes": sum(r["frame_bytes"] + r["reply_frame_bytes"] for r in selected),
                "old_ns": sum(r["ns"]["roundtrip_nonoverlap"] for r in selected),
                "raw_ns": sum(r["ns"]["raw_roundtrip_nonoverlap"] for r in selected),
            })
        old = sum(r["old_ns"] for r in owners)
        raw = sum(r["raw_ns"] for r in owners)
        domain = {k: sum(r["domain_ns"][k] for r in rows) for k in rows[0]["domain_ns"]}
        cases.append({
            "case": case, "stage": stage, "histogram": dict(sorted(histogram.items())),
            "owners": owners, "old_ns": old, "raw_ns": raw, "saved_ns": old - raw,
            "domain_ns": domain,
            "serial_framing_plus_domain_model_old_ns": old + sum(domain.values()),
            "serial_framing_plus_domain_model_raw_ns": raw + sum(domain.values()),
        })
    return {
        "schema": "ferrule.numeric-cpu-summary.v1",
        "scope": "Sum of per-bucket median CPU-only work, NOT measured end-to-end/GPU latency. Domain model uses representative full-40 metadata; validation remains unchanged.",
        "buckets": buckets, "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(json.loads(args.report.read_text()))
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
