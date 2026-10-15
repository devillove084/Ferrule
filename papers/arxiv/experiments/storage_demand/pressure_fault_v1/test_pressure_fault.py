"""CPU-only plan/evidence tests; no CUDA, HTTP, subprocess, or harness imports."""
import ast
import copy
import unittest
from pathlib import Path

import evidence as e
import prepare as p


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.plan = p.load(p.HERE / "plan.json")

    def test_preregistered_denominators_and_canonical_oracles(self):
        p.validate_design(self.plan)
        rows = p.frozen_requests(self.plan)
        d = self.plan["pressure"]["denominators"]
        self.assertEqual(d, dict(lifecycles=3, requests_per_lifecycle=4, requests=12,
                                 hello_smokes=3, exact_eight_id_canaries=9,
                                 planned_completion_tokens=78, exact_oracle_tokens=72,
                                 measured_repeated_qa=0))
        self.assertEqual(sum(r["max_tokens"] for r in rows) * 3, d["planned_completion_tokens"])
        self.assertEqual(sum(len(r["reference"]["reference_ids"]) for r in rows[1:]) * 3,
                         d["exact_oracle_tokens"])
        self.assertEqual(p.denominator_summary({})["unissued"], 12)

    def test_point_capacity_arithmetic_is_not_admission(self):
        rows = p.cases(self.plan)
        self.assertEqual([r["byte_only_payload_entries"] for r in rows], [18, 103, 210])
        self.assertEqual([r["logical_positive_payload_entry_bound"] for r in rows], [8, 64, 128])
        for row in rows:
            self.assertEqual(row["model_expert_id_count"], 10240)
            self.assertEqual(row["max_source_extents_per_whole_expert"], 6)
            self.assertIsNone(row["effective_admitted_entries"])
            self.assertIsNone(row["physical_admission"])
            self.assertIsNone(row["actual_working_set_unique_expert_ids"])
            self.assertFalse(row["point_full_numerical_verified"])
            self.assertTrue(all(v is None for v in row["measurements"].values()))

    def test_only_two_cache_knobs_change_and_all_fixed_options_stay(self):
        rows = p.cases(self.plan)
        base = p.load(p.ROOT / self.plan["pressure"]["case_source"])["blocking"]["argv_after_binary"]
        allowed = {base.index(k) + 1 for k in self.plan["pressure"]["vary_only"]}
        for row in rows:
            actual = row["requested_argv_after_binary"]
            self.assertEqual(len(actual), len(base))
            self.assertTrue({i for i, (a, b) in enumerate(zip(actual, base)) if a != b} <= allowed)

    def test_order_budget_and_numerical_scope_cannot_drift(self):
        for key, changed in (("schedule", "lookahead"), ("retry_count", 1),
                             ("extra_qa_requests", 1), ("reuse_middle_baseline_samples", True),
                             ("order", ["middle", "small", "big"])):
            value = copy.deepcopy(self.plan)
            value["pressure"][key] = changed
            with self.assertRaises(ValueError):
                p.validate_design(value)
        value = copy.deepcopy(self.plan)
        value["pressure"]["limits"]["phase_seconds"] = 901
        with self.assertRaises(ValueError):
            p.validate_design(value)
        value = copy.deepcopy(self.plan)
        value["pressure"]["quality"]["full_numerical_per_point"] = "passed"
        with self.assertRaises(ValueError):
            p.validate_design(value)

    def test_hard_reserve_and_nonadaptive_admission(self):
        self.assertTrue(e.can_admit(420))
        self.assertFalse(e.can_admit(419.999))
        self.assertFalse(e.can_admit(-1))
        for value in (None, float("inf"), True):
            with self.assertRaises(ValueError):
                e.can_admit(value)
        self.assertEqual(3 * p.LIMITS["outer_lifecycle_seconds"] +
                         p.LIMITS["study_cleanup_reserve_seconds"], 900)

    def test_existing_worker_is_incompatible_not_silently_reused(self):
        base = p.HERE.parent
        source = (base / "runnerhelpers/evidence_v2.py").read_text()
        serial = (base / "runnerhelpers/execution_v2.py").read_text()
        worker = (base / "newversionedprotocol/run_pilot.py").read_text()
        self.assertIn("ceiling == 12", source)
        self.assertIn("len(observations) == 12", serial)
        self.assertIn("len(result['whole_log_token_join']) == 12", worker)
        self.assertFalse(self.plan["launch_entry"])

    def test_fault_selectors_and_component_scope_are_real_source(self):
        faults = self.plan["faults"]
        self.assertEqual(len(faults["minimum_gpu_selectors"]), 2)
        for item in faults["minimum_gpu_selectors"]:
            text = (p.ROOT / item["source"]).read_text()
            self.assertIn("fn " + item["selector"].rsplit("::", 1)[1] + "(", text)
            self.assertIn("#[ignore", text)
            self.assertFalse(item["same_serving_owner"])
        self.assertFalse(faults["current_public_cli_fault_gate"])
        self.assertEqual(faults["gpu_execution"], "not_run_not_authorized")

    def test_source_bindings_and_prepared_remain_cpu_only(self):
        value = p.prepared()
        self.assertEqual(value["denominators"]["unissued"], 12)
        self.assertIsNone(value["pressure_comparison"])
        self.assertFalse(value["launch_entry"])
        self.assertEqual(value["gpu_runs"], 0)
        self.assertEqual(value["candidate"]["recorded_passed_gates"], 9)


class EvidenceTests(unittest.TestCase):
    def test_issued_without_terminal_is_not_unissued(self):
        records = {
            ("small", "hello2"): dict(issued=True, status="success"),
            ("small", "canary-qa"): dict(issued=True),
            ("middle", "hello2"): dict(issued=False, status="unissued"),
        }
        self.assertEqual(e.denominator_summary(records), dict(scheduled=12, issued=2, terminal=1,
                         successful=1, failed=0, censored=1, unissued=10))
        records[("small", "canary-qa")]["status"] = "failed"
        self.assertEqual(e.denominator_summary(records)["failed"], 1)
        with self.assertRaises(ValueError):
            e.denominator_summary({("retry", "hello2"): dict(issued=True, status="success")})
        with self.assertRaises(ValueError):
            e.denominator_summary({("small", "hello2"): dict(issued=False, status="success")})

    def test_missing_null_zero_and_no_invented_misses_or_working_set(self):
        rows = [dict(checkpoint_read_bytes=0, expert_hits=2, expert_uploads=1),
                dict(checkpoint_read_bytes=0, expert_hits=0)]
        result = e.counter_totals(rows)
        self.assertEqual(result["checkpoint_read_bytes"], 0)
        self.assertEqual(result["expert_hits"], 2)
        self.assertIsNone(result["expert_uploads"])
        for key in ("source_extents_read", "expert_cache_misses", "cache_hit_rate",
                    "working_set_unique_expert_ids", "physical_nas_backend_bytes"):
            self.assertIsNone(result[key])
        self.assertTrue(all(v is None for v in e.counter_totals([]).values()))
        self.assertIsNone(e.counter_totals([dict(expert_hits=True)])["expert_hits"])

    def test_quarantine_is_subset_not_double_charge(self):
        row = dict(resident_bytes=100, pending_upload_bytes=20, scratch_bytes=8,
                   unknown_quarantine_bytes=128)
        result = e.charged_bytes(row)
        self.assertEqual(result["charged_bytes"], 128)
        self.assertTrue(result["quarantine_subset_verified"])
        row["unknown_quarantine_bytes"] = 129
        with self.assertRaises(ValueError):
            e.charged_bytes(row)
        self.assertIsNone(e.charged_bytes({})["charged_bytes"])
        row["scratch_bytes"] = None
        self.assertIsNone(e.charged_bytes(row)["quarantine_subset_verified"])

    def test_snapshot_is_data_only_and_never_polls_or_heals(self):
        class Owner:
            def poll(self):
                self.fail("poll must not be called")
            def join(self):
                self.fail("join must not be called")
        with self.assertRaises(ValueError):
            e.copy_observation(Owner())
        with self.assertRaises(ValueError):
            e.copy_observation({"owner": Owner()})
        row = {"status": "unknown", "charge": {"bytes": 128}, "io_bytes": None}
        before = copy.deepcopy(row)
        self.assertEqual(e.copy_observation(row), e.copy_observation(row))
        e.copy_observation(row)["charge"]["bytes"] = 0
        self.assertEqual(row, before)
        self.assertEqual(e.copy_observation(row)["status"], "unknown")

    def test_payload_invalid_or_zero_room_not_clamped_to_one(self):
        self.assertEqual(e.positive_payload_entries(8, 8, 8, 3), 0)
        for args in ((8, 7, 8, 3), (0, 64, 8, 3), (8, 64, 8, 0), (True, 64, 8, 3)):
            with self.assertRaises(ValueError):
                e.positive_payload_entries(*args)

    def test_new_sources_have_no_execution_or_gpu_imports(self):
        allowed = {"__future__", "argparse", "copy", "hashlib", "json", "pathlib", "math", "evidence"}
        for filename in ("prepare.py", "evidence.py"):
            tree = ast.parse((Path(__file__).parent / filename).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertTrue(all(n.name in allowed for n in node.names))
                elif isinstance(node, ast.ImportFrom):
                    self.assertIn(node.module, allowed)


if __name__ == "__main__":
    unittest.main()
