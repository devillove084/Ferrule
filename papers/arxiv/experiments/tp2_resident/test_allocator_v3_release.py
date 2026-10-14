"""CPU-only fixed release, historical dispatch and allocator-boundary regressions."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import allocator_v3_release as release
import operator_retirement_ab as ab
import optimized_release
import paired
import retirement_profile as profile


class AllocatorV3Release(unittest.TestCase):
    def link(self):
        return dict(path=str(release.RELEASE/"handoff.json"), sha256=release.HANDOFF_SHA)

    def test_actual_final_release_and_all_96_builder_pins(self):
        evidence = release.audit(self.link())
        pins = ab.pre.checked_document(dict(path=str(release.RELEASE/"handoff-file-pins.json"),
                                            sha256=release.FILE_PINS_SHA))
        self.assertEqual(len(pins), 96)
        self.assertEqual(evidence["binary_sha256"], release.BINARY_SHA)
        self.assertEqual(evidence["quality"]["schema"], release.QUALITY_SCHEMA)
        self.assertEqual((evidence["quality"]["calls"], evidence["quality"]["metrics"]), (37, 40293))
        self.assertFalse(evidence["quality"]["numerical_pass_inherited"])
        self.assertEqual(ab.reviewed_build(ab.REVIEWED_BUILD)["release_handoff"], self.link())

    def test_unreviewed_release_path_and_hash_rejected(self):
        for link in (dict(self.link(), sha256="0"*64), dict(self.link(), path="/fixture/handoff.json")):
            with self.assertRaisesRegex(ValueError, "Unreviewed V3"):
                release.audit(link)

    def test_only_fixed_handoff_dispatches_to_v3(self):
        m = dict(build_receipt=self.link())
        with mock.patch.object(optimized_release, "validate", return_value={}) as new, \
                mock.patch.object(ab.pre, "validate_release", side_effect=AssertionError("frozen adapter")):
            paired.validate_release(m)
            new.assert_called_once_with(m)
        for link in (dict(self.link(), sha256="0"*64), dict(self.link(), path="/fixture/handoff.json"), {}):
            m = dict(build_receipt=link)
            with mock.patch.object(optimized_release, "validate", side_effect=AssertionError("not fixed")), \
                    mock.patch.object(ab.pre, "validate_release", return_value={}) as old:
                paired.validate_release(m)
                old.assert_called_once_with(m)

    def test_legacy_pre_serving_bytes_remain_authenticated(self):
        snapshot = ab.pre.HERE/"history/pre-operator-retirement-release-20261011/pre_serving.py.txt"
        self.assertEqual((ab.pre.HERE/"pre_serving.py").read_bytes(), snapshot.read_bytes())
        self.assertEqual(ab.ct.sha(snapshot), "649f1ca9535ffa721d51f9bc527b08135c348f7a41ce61369c9dba6088599f96")
        self.assertFalse(ab.indexed.frozen_audit()["current_tree_qualified"])
        self.assertFalse(ab.frozen_audit()["current_tree_qualified"])

    def log(self, v3=False):
        log = (ab.pre.HERE/"runs/arena-index-c1-20261010-01/runs/AFTER/session/server.log").read_text()
        if not v3: return log
        lines = []
        for line in log.splitlines():
            if "CudaAllocatorMetrics {" in line:
                r = json.loads(line)
                r["fields"]["allocator"] = r["fields"]["allocator"].replace(" }", ", retired_blocks: 200, retirement_batches: 50, retirement_created_fences: 150, pending_retirement_batches: 0, pending_unfenced_bytes: 0, shutdown_quiescent: false, shutdown_retained: false }")
                line = json.dumps(r)
            lines.append(line)
        return "\n".join(lines)

    def test_before_metrics_stay_historical_and_peaks_unknown(self):
        report = ab.allocator_rows(self.log(), ab.BEFORE_SHA)
        self.assertEqual(report["metric_schema"], "historical_9ef")
        self.assertEqual(len(report["rows"]), 4)
        self.assertIsNone(report["pending_peak"])
        self.assertIsNone(report["unknown_peak"])
        for row in report["rows"]:
            self.assertNotIn("retirement_created_fences", row["metrics"])
            self.assertNotIn("pending_retirement_batches", row["metrics"])

    def test_v3_success_boundaries_are_not_high_watermarks(self):
        report = ab.allocator_rows(self.log(True), release.BINARY_SHA)
        self.assertTrue(report["quiescent_pending_zero"])
        self.assertEqual(report["arena_bytes_per_owner"], 3 << 30)
        self.assertEqual(report["ordinary_pressure_failure_count"], 0)
        self.assertEqual(report["observed_boundary_pending_bytes"], 0)
        self.assertIsNone(report["pending_peak"])
        self.assertIsNone(report["unknown_peak"])
        self.assertEqual(report["rows"][0]["metrics"]["retirement_created_fences"], 150)

    def test_v3_requires_new_fields_but_before_does_not(self):
        with self.assertRaises(ValueError): ab.allocator_rows(self.log(), release.BINARY_SHA)
        for field in ("pending_retirement_bytes", "pending_retirement_batches", "pending_unfenced_bytes",
                      "capture_retirement_bytes", "allocation_failures"):
            log = self.log(True).replace(field+": 0", field+": 1")
            with self.subTest(field=field), self.assertRaises(ValueError):
                ab.allocator_rows(log, release.BINARY_SHA)

    def test_pressure_quarantine_duplicate_and_bad_metrics_rejected(self):
        log = self.log(True)
        line = next(x for x in log.splitlines() if "CudaAllocatorMetrics {" in x)
        for bad in (log+"\n"+line, log+"\ndriver growth deferred by ordinary retirement pressure",
                    log.replace("shutdown_retained: false", "shutdown_retained: true"),
                    log.replace("pending_unfenced_bytes: 0", "pending_unfenced_bytes: false"),
                    log.replace("retirement_batches: 50", "retirement_batches: -1")):
            with self.assertRaises(ValueError): ab.allocator_rows(bad, release.BINARY_SHA)
        with self.assertRaises(ValueError): ab.allocator_rows(log, "0"*64)

    def test_missing_and_foreign_boundaries_rejected(self):
        rows = [json.loads(x) for x in self.log(True).splitlines() if x.startswith("{")]
        index = next(i for i,r in enumerate(rows) if "allocator" in r["fields"])
        for change in ("missing", "owner", "device", "quarantine", "target", "arena"):
            changed = deepcopy(rows)
            if change == "missing": changed.pop(index)
            if change == "owner": changed[index]["fields"]["owner"] = True
            if change == "device": changed[index]["fields"]["device"] = 7
            if change == "quarantine": changed[index]["fields"]["device_quarantine"] = True
            if change == "target": changed[index]["target"] = "foreign"
            if change == "arena":
                cache = next(r["fields"] for r in changed if r["fields"].get("event") == "mixed_decoded_cache_snapshot")
                s = json.loads(cache["snapshot_json"]); s["arena_backing_grants"] = 0
                cache["snapshot_json"] = json.dumps(s)
            with self.subTest(change=change), self.assertRaises(ValueError):
                ab.allocator_rows("\n".join(map(json.dumps, changed)), release.BINARY_SHA)

    def test_fixed_profile_policy_and_event_api_layers(self):
        self.assertFalse(profile.CONFIG["rankperf"])
        self.assertEqual(profile.CONFIG["log"], "info")
        self.assertEqual(profile.CONFIG["after_binary_sha256"], release.BINARY_SHA)
        self.assertFalse(profile.selected(dict(profiler=dict(profile.CONFIG, rankperf=True))))
        pid = 17
        rows = [dict(name=n, start=10, end=20, globalTid=pid << 24, returnValue=0)
                for n in ("cudaEventCreateWithFlags", "cuEventRecord", "cuStreamWaitEvent", "cuLaunchKernel")]
        rows += [dict(rows[0], start=0), dict(rows[0], globalTid=18 << 24)]
        counts = profile.api_counts(rows, {"measured":dict(start_ns=5, end_ns=30)}, pid)["measured"]
        self.assertEqual(counts["events"]["runtime"]["create"]["count"], 1)
        self.assertEqual(counts["events"]["driver"]["record"]["count"], 1)
        self.assertEqual(len(counts["boundary_crossing_calls"]), 1)
        self.assertIn("cuStreamWaitEvent", counts["synchronization"])
        self.assertIn("cuLaunchKernel", counts["launches"])


if __name__ == "__main__": unittest.main()
