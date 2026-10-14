"""CPU-only regression tests against the actual curated receipts, not fake PASS fixtures."""
import copy
import gzip
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import offline_check as oc

HERE = Path(__file__).resolve().parent


class QualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.archive = oc.Archive(HERE)
        cls.plan = cls.archive.load("qualification-plan-v2.json")
        cls.final = cls.archive.load("FINAL.json")
        cls.audit = cls.archive.load("receipt-audit.json")
        cls.gates = {g["id"]: g for g in cls.plan["gates"]}

    def report(self, gate="06-blocking-ordinary"):
        suffix = "numeric-report/full40-bf16-profile.json" if "full40" in gate else "quality.json"
        return copy.deepcopy(self.archive.load("gates/" + gate + "/" + suffix))

    def final_copy(self):
        value = dict(self.final)
        value["results"] = [dict(r) for r in self.final["results"]]
        return value

    def reject_report(self, report, message, gate="06-blocking-ordinary"):
        with self.assertRaisesRegex(oc.InvalidEvidence, message):
            oc.check_report(report, self.gates[gate])

    def test_actual_complete_archive_passes_without_subprocess_or_gpu(self):
        with mock.patch("subprocess.Popen", side_effect=AssertionError("No process launches")):
            result = oc.verify(HERE)
        self.assertEqual(result["factory_gates_passed"], 9)
        self.assertIs(result["cli_serving_qualified"], False)
        self.assertEqual(result["new_gpu_launches"], 0)
        self.assertEqual(result["physical_release"], "unobserved")

    def test_gzip_is_lossless_and_original_hashed_separately(self):
        name = "gates/04-blocking-fresh/quality.json"
        entry = self.archive.entries[name]
        self.assertEqual(entry["encoding"], "gzip")
        self.assertEqual(oc.sha(self.archive.raw(name)), entry["original_sha256"])
        self.assertNotEqual(entry["stored_sha256"], entry["original_sha256"])

    def test_stale_stored_hash_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "archive"
            shutil.copytree(HERE, root)
            path = root / "receiptarchive/gates/01-constructor/test.stdout"
            path.write_bytes(path.read_bytes() + b"stale\n")
            with self.assertRaisesRegex(oc.InvalidEvidence, "Stale stored hash"):
                oc.Archive(root).raw("gates/01-constructor/test.stdout")

    def test_rehashed_tampered_copy_still_fails_original_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "archive"
            shutil.copytree(HERE, root)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            entry = next(e for e in manifest["entries"] if e["path"] == "receiptarchive/SUMMARY.md")
            data = (root / entry["path"]).read_bytes() + b"Changed claim\n"
            (root / entry["path"]).write_bytes(data)
            for field in ("stored_sha256", "content_sha256", "original_sha256"):
                entry[field] = oc.sha(data)
            for field in ("stored_bytes", "content_bytes", "original_bytes"):
                entry[field] = len(data)
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(oc.InvalidEvidence, "Unsealed/stale copy input"):
                oc.Archive(root).integrity()

    def test_missing_report_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "archive"
            shutil.copytree(HERE, root)
            (root / self.archive.entries["gates/08-blocking-cold/quality.json"]["path"]).unlink()
            with self.assertRaisesRegex(oc.InvalidEvidence, "Missing or symlink record"):
                oc.Archive(root).raw("gates/08-blocking-cold/quality.json")

    def test_external_candidate_hash_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "archive"
            shutil.copytree(HERE, root)
            manifest = json.loads((root / "manifest.json").read_text())
            next(r for r in manifest["external_artifacts"] if r["role"] == "cli")["sha256"] = "0" * 64
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(oc.InvalidEvidence, "External candidate path/hash"):
                oc.Archive(root).integrity()

    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "archive"
            shutil.copytree(HERE, root)
            manifest = json.loads((root / "manifest.json").read_text())
            manifest["entries"][0]["path"] = "receiptarchive/../../outside"
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(oc.InvalidEvidence, "Unsafe archive path"):
                oc.Archive(root)

    def test_source_pin_digest_rejected(self):
        source = copy.deepcopy(self.archive.load("source-freeze.json"))
        members = self.archive.load("archive-exact-members.json")
        first = next(iter(source["original_source_pins"]))
        source["original_source_pins"][first] = "0" * 64
        with self.assertRaisesRegex(oc.InvalidEvidence, "Stale source-pin digest"):
            oc.check_source(source, members)

    def test_stale_source_cut_rejected(self):
        source = dict(self.archive.load("source-freeze.json"))
        source["commit"] = "old"
        with self.assertRaisesRegex(oc.InvalidEvidence, "Stale source commit"):
            oc.check_source(source, self.archive.load("archive-exact-members.json"))

    def test_candidate_binary_changed_rejected(self):
        artifacts = {r:copy.deepcopy(self.archive.load("artifact-" + r + ".json")) for r in oc.CANDIDATES}
        artifacts["cli"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(oc.InvalidEvidence, "Stale candidate"):
            oc.check_candidates(self.plan, artifacts, self.archive.load("build-environment.json"))

    def test_nine_gate_count_rejected(self):
        final = self.final_copy()
        final["actual_gpu_tests_issued"] = 8
        with self.assertRaisesRegex(oc.InvalidEvidence, "Nine-gate final PASS incomplete"):
            oc.check_counts(final, self.plan, self.audit)

    def test_duplicate_or_old_gate_pass_cannot_be_pooled(self):
        for gate_id in (oc.GATES[0], "04-blocking-fresh-from-prior-01"):
            with self.subTest(gate_id=gate_id):
                final = self.final_copy()
                final["results"][1]["gate_id"] = gate_id
                with self.assertRaisesRegex(oc.InvalidEvidence, "Nine exact-once gates"):
                    oc.check_counts(final, self.plan, self.audit)

    def test_unissued_or_retry_gate_rejected(self):
        for key, value in (("actual_test_issued", False), ("retries", 1)):
            with self.subTest(key=key):
                final = self.final_copy()
                final["results"][0][key] = value
                with self.assertRaisesRegex(oc.InvalidEvidence, "exactly once"):
                    oc.check_counts(final, self.plan, self.audit)

    def test_skipped_test_and_pooled_stdout_rejected(self):
        gate = self.gates["01-constructor"]
        prefix = "gates/01-constructor/"
        stdout = self.archive.raw(prefix + "test.stdout").decode()
        listed = self.archive.load(prefix + "exact-list-receipt.json")
        for changed in (stdout.replace("1 passed; 0 failed; 0 ignored", "0 passed; 0 failed; 1 ignored"), stdout + stdout):
            with self.subTest(stdout=changed[:20]):
                with self.assertRaisesRegex(oc.InvalidEvidence, "Skipped/ignored/failed"):
                    oc.check_test(changed, listed, gate["selector"])

    def test_exact_selector_list_missing_rejected(self):
        listed = dict(self.archive.load("gates/01-constructor/exact-list-receipt.json"))
        listed["stdout"] = "0 tests, 0 benchmarks\n"
        with self.assertRaisesRegex(oc.InvalidEvidence, "exactly one selected test"):
            oc.check_test("", listed, self.gates["01-constructor"]["selector"])

    def test_metric_count_rejected(self):
        report = self.report()
        report["metrics"].pop()
        self.reject_report(report, "Numerical metric count")

    def test_nonfinite_or_failed_numerical_metric_rejected(self):
        for key, value in (("nonfinite", 1), ("failures", 1), ("max_tolerance_ratio", 1.1)):
            with self.subTest(key=key):
                report = self.report()
                report["metrics"][0][key] = value
                self.reject_report(report, "Numerical metric|Tolerance exceeded")

    def test_fingerprint_64_and_28_coverage_rejected(self):
        for gate in ("06-blocking-ordinary", "08-blocking-cold"):
            with self.subTest(gate=gate):
                report = self.report(gate)
                report["prediction_fingerprints"]["calls"].popitem()
                self.reject_report(report, "64/28 fingerprint coverage", gate)

    def test_fingerprint_shape_or_greedy_unknown_rejected(self):
        for key, value in (("values", 0), ("next_token_id", -1), ("f32_le_sha256", "unknown")):
            with self.subTest(key=key):
                report = self.report()
                next(iter(report["prediction_fingerprints"]["calls"].values()))[key] = value
                self.reject_report(report, "fingerprint words/shape/greedy")

    def test_current_primary_hash_rejected(self):
        report = self.report()
        worker = self.archive.load("gates/06-blocking-ordinary/worker-launch.public.json")
        primary = self.archive.load("gates/04-blocking-fresh/quality.json")
        report["diagnostic_primary_report_sha256"] = "0" * 64
        with self.assertRaisesRegex(oc.InvalidEvidence, "Stale current primary SHA"):
            oc.check_primary(report, primary, worker, self.archive.content_sha("gates/04-blocking-fresh/quality.json"),
                             worker["environment"]["FERRULE_BF16_FRESH_PRIMARY_REPORT"], "ordinary")

    def test_current_primary_fingerprint_mismatch_rejected(self):
        report = self.report()
        worker = self.archive.load("gates/06-blocking-ordinary/worker-launch.public.json")
        primary = self.archive.load("gates/04-blocking-fresh/quality.json")
        next(iter(report["prediction_fingerprints"]["calls"].values()))["f32_le_sha256"] = "0" * 64
        with self.assertRaisesRegex(oc.InvalidEvidence, "fingerprints differ"):
            oc.check_primary(report, primary, worker, self.archive.content_sha("gates/04-blocking-fresh/quality.json"),
                             worker["environment"]["FERRULE_BF16_FRESH_PRIMARY_REPORT"], "ordinary")

    def test_cold_primary_config_mismatch_rejected(self):
        report = self.report("08-blocking-cold")
        worker = self.archive.load("gates/08-blocking-cold/worker-launch.public.json")
        report["effective_config"]["effective"]["ctx_size"] = 256
        with self.assertRaisesRegex(oc.InvalidEvidence, "primary config/oracle"):
            oc.check_primary(report, self.archive.load("gates/04-blocking-fresh/quality.json"), worker,
                             self.archive.content_sha("gates/04-blocking-fresh/quality.json"),
                             worker["environment"]["FERRULE_BF16_FRESH_PRIMARY_REPORT"], "cold")

    def test_per_stream_denominator_or_pooling_rejected(self):
        for pooled in (True, False):
            with self.subTest(pooled=pooled):
                report = self.report()
                if pooled:
                    report["prediction_denominators"]["no_cross_stream_metric_sums"] = False
                else:
                    report["prediction_denominators"]["streams"]["free_feedback"]["logit_values_compared"] += 1
                self.reject_report(report, "Pooled/missing|Per-stream denominator")

    def test_logical_memory_charge_not_rss_rejected(self):
        report = self.report()
        report["expert_read_quality"]["calls"][0]["after"]["expert_read_pipeline"]["value"]["charge_basis"] = "physical_rss"
        self.reject_report(report, "Logical charge/RSS")

    def test_both_memory_endpoints_bounded(self):
        for endpoint in ("before", "after"):
            with self.subTest(endpoint=endpoint):
                report = self.report()
                report["expert_read_quality"]["calls"][0][endpoint]["expert_read_pipeline"]["value"]["current_charge"]["host_charged_bytes"] = 67108865
                self.reject_report(report, "Logical host charge exceeds")

    def test_component_peak_memory_bound_rejected(self):
        report = self.report()
        report["expert_read_quality"]["calls"][0]["after"]["expert_read_pipeline"]["value"]["stats"]["peak_packed_bytes"] = 67108864
        self.reject_report(report, "Logical host charge exceeds")

    def test_effective_schedule_unknown_rejected(self):
        report = self.report()
        report["expert_read_quality"]["calls"][-1]["after"]["expert_read_pipeline"]["value"]["effective_mode"] = None
        self.reject_report(report, "Effective mode/positive worker read")

    def test_last_sampled_lifetime_counters_not_reinterpreted(self):
        report = self.report()
        report["expert_read_quality"]["calls"][-1]["after"]["expert_read_pipeline"]["value"]["stats"]["verified_reads"] += 2089
        self.reject_report(report, "owner-lifetime counters")

    def test_source_verified_read_and_timestamps_required(self):
        for key, value in (("outcome", "unverified"), ("io_bytes", 0), ("worker_started_ns", None)):
            with self.subTest(key=key):
                report = self.report()
                report["expert_read_quality"]["calls"][-1]["after"]["expert_read_pipeline"]["value"]["last_receipt"][key] = value
                self.reject_report(report, "numerical source read|timestamps unknown")

    def test_physical_release_must_remain_unknown(self):
        report = self.report()
        report["expert_read_quality"]["calls"][0]["after"]["release"] = {"state": "observed", "value": "released"}
        self.reject_report(report, "Physical release must remain unobserved")
        final = self.final_copy()
        final["physical_release"] = "certified"
        with self.assertRaisesRegex(oc.InvalidEvidence, "Physical release not unknown"):
            oc.check_counts(final, self.plan, self.audit)

    def test_idle_acceptance_is_actual_all_eight_zero(self):
        idle = copy.deepcopy(self.archive.load("final-idle.json"))
        idle["samples"][1]["gpu"] = idle["samples"][1]["gpu"].replace("0 MiB", "1 MiB", 1)
        with self.assertRaisesRegex(oc.InvalidEvidence, "inventory/memory mismatch"):
            oc.check_idle(idle, self.plan["uuid_inventory"])

    def test_missing_authenticated_echild_rejected(self):
        prefix = "gates/01-constructor/"
        supervisor = copy.deepcopy(self.archive.load(prefix + "supervisor.json"))
        supervisor["custody"]["echild"] = False
        with self.assertRaisesRegex(oc.InvalidEvidence, "Authenticated ECHILD"):
            oc.check_custody(supervisor, self.archive.load(prefix + "authenticated-launch.json"),
                             self.archive.load(prefix + "worker-launch.public.json"), supervisor["custody"])

    def test_pid_birth_authentication_mismatch_rejected(self):
        prefix = "gates/01-constructor/"
        launch = copy.deepcopy(self.archive.load(prefix + "authenticated-launch.json"))
        launch["identity"]["worker"]["birth_ticks"] += 1
        with self.assertRaisesRegex(oc.InvalidEvidence, "PID/birth"):
            oc.check_custody(self.archive.load(prefix + "supervisor.json"), launch,
                             self.archive.load(prefix + "worker-launch.public.json"), self.archive.load(prefix + "custody.json"))

    def test_duplicate_and_nonfinite_json_rejected(self):
        for data in ('{"passed":true,"passed":false}', '{"value":NaN}'):
            with self.subTest(data=data):
                with self.assertRaises(oc.InvalidEvidence):
                    oc.parse_json(data)

    def test_numerical_stderr_failure_or_count_rejected(self):
        name = "gates/06-blocking-ordinary/test.stderr"
        stderr = self.archive.raw(name).decode()
        report = self.archive.load("gates/06-blocking-ordinary/quality.json")
        for changed in (stderr.replace("failures=0", "failures=1", 1), stderr + stderr):
            with self.subTest(changed=changed[:20]):
                with self.assertRaisesRegex(oc.InvalidEvidence, "Numerical stderr|numerical stderr"):
                    oc.check_log_metrics(changed, report, "ordinary")

    def test_final_cleanup_unknown_rejected(self):
        held = self.archive.load("uuid-locks-held.json")
        released = self.archive.load("final-lock-release.json")
        probe = self.archive.load("final-lock-probe.json")
        idle = self.archive.load("final-idle.json")
        final = self.final_copy()
        final["final_recorded_owners_absent"] = False
        with self.assertRaisesRegex(oc.InvalidEvidence, "cleanup incomplete"):
            oc.check_final_cleanup(final, idle, self.plan["uuid_inventory"], held, released, probe)
        probe = copy.deepcopy(probe)
        probe["all_eight_nonblocking_reacquired"] = False
        with self.assertRaisesRegex(oc.InvalidEvidence, "release/probe unknown"):
            oc.check_final_cleanup(self.final, idle, self.plan["uuid_inventory"], held, released, probe)

    def test_no_raw_worker_credentials_or_executable_payloads(self):
        for name, entry in self.archive.entries.items():
            self.assertNotEqual(Path(name).name, "dispatch.json")
            self.assertNotEqual(Path(name).name, "worker-launch.json")
            self.assertFalse(name.endswith((".so", ".a", ".tar", ".safetensors")))
            if name.endswith("worker-launch.public.json"):
                self.assertNotIn("worker_token", self.archive.load(name))
                self.assertEqual(entry["transform"], "remove_worker_token_only")
                self.assertTrue(entry["original_relpath"].endswith("worker-launch.json"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
