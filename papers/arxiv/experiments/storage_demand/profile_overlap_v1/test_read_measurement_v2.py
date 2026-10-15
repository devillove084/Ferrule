"""CPU-only OSRT timing/typed-byte regression fixtures."""
import copy
import unittest

import test_scoped_contract as scoped_tests
from profile_overlap_v1 import analyze as a, scope_contract as s, protocol as p
from profile_overlap_v1.test_profile_overlap import COMPUTE


class ReadMeasurementTests(unittest.TestCase):
    def setUp(self):
        fixture = scoped_tests.ScopedSQLTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.db, self.context = fixture.db, fixture.context

    def audit(self):
        return a.audit_sqlite(self.db, self.context)

    def rejected(self):
        try:
            report = self.audit()
        except ValueError:
            return
        self.assertFalse(report["scoped_sql_evidence_consistent"], report)
        self.assertFalse(s.acceptance(dict(report, log_issues=[]), scoped_tests.GATES)["scoped_overlap_accepted"])

    def test_zero_return_large_verified_bytes_has_no_osrt_byte_observation(self):
        size = 11_571_399_936
        self.db.execute("UPDATE StringIds SET value=? WHERE id=5",
                        (f"read_subset operation=1 source_generation=7 requested_bytes={size}",))
        self.context["read_stats"].update(proved_read_bytes=size, reader_interval_bytes=size)
        self.context["last_receipt"].update(requested_bytes=size, io_bytes=size)
        report = self.audit()
        self.assertTrue(report["scoped_sql_evidence_consistent"], report["scoped_issues"])
        self.assertTrue(s.acceptance(dict(report, log_issues=[]), scoped_tests.GATES)["scoped_overlap_accepted"])
        self.assertEqual(report["source_verified_read_bytes"], size)
        self.assertIsNone(report["osrt_read_bytes"])
        self.assertEqual(report["osrt_read_bytes_reason"], "unsupported_schema_no_size_or_proved_ssize_t")
        self.assertEqual(report["osrt_read_return_values"], {0: 2})
        self.assertFalse(report["per_call_success_proved"])
        self.assertFalse(report["physical_io_bytes_claim"])
        row = report["read_range_metrics"][0]
        self.assertEqual((row["operation"], row["source_generation"], row["pread_calls"]), (1, 7, 2))
        self.assertIsNone(row["osrt_read_bytes"])
        self.assertEqual(row["pread_kernel_intersection_ns"], 70)
        self.assertEqual(row["read_envelope_kernel_intersection_ns"], 110)
        self.assertEqual(report["pread_calls_with_kernel_overlap"], 2)
        self.assertEqual(report["expert_range_metrics"][0]["kernel_union_intervals_ns"], [(260, 370)])

    def test_missing_ledger_or_unverified_source_or_failed_worker_rejects(self):
        original = copy.deepcopy(self.context)
        for key in ("read_stats", "read_source_boundary", "model_source_generation", "last_receipt"):
            with self.subTest(missing=key):
                self.context = copy.deepcopy(original)
                del self.context[key]
                self.rejected()
        for key in ("failed_reads", "unknown_reads", "cancelled_reads", "unobserved_reader_intervals",
                    "unobserved_terminal_reads", "unknown_bytes", "verified_reads", "completed_reads"):
            with self.subTest(counter=key):
                self.context = copy.deepcopy(original)
                self.context["read_stats"][key] += 1
                self.rejected()
        self.context = copy.deepcopy(original)
        self.context["last_receipt"]["outcome"] = "failed"
        self.rejected()

    def test_count_loss_rejects_even_when_official_counter_is_adjusted(self):
        self.db.execute("DELETE FROM OSRT_API WHERE start=250")
        self.db.execute("UPDATE DIAGNOSTIC_EVENT SET text=? WHERE text LIKE ?",
                        ("Number of OS runtime libraries events collected: 1.", "Number of OS%"))
        self.rejected()

    def test_no_synthesized_bytes_or_success_from_nonzero_return(self):
        for code in (-1, 1, 8, 16, "unknown"):
            with self.subTest(code=code):
                self.db.execute("UPDATE OSRT_API SET returnValue=?", (code,))
                self.rejected()

    def test_orphan_on_other_model_thread_is_not_ignored(self):
        self.db.execute("UPDATE OSRT_API SET globalTid=? WHERE start=250", (COMPUTE,))
        self.rejected()

    def test_zero_overlap_is_valid_negative_with_positive_ranges(self):
        self.db.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET start=450,end=460")
        report = self.audit()
        self.assertTrue(report["scoped_sql_evidence_consistent"], report["scoped_issues"])
        self.assertEqual(report["read_ranges_with_kernel_overlap"], 0)
        self.assertEqual(report["read_ranges_with_pread_kernel_overlap"], 0)
        self.assertEqual(report["pread_calls_with_kernel_overlap"], 0)
        self.assertGreater(report["worker_pread_union_ns"], 0)
        self.assertEqual(report["pread_kernel_intersection_ns"], 0)
        self.assertTrue(s.acceptance(dict(report, log_issues=[]), scoped_tests.GATES)["scoped_overlap_accepted"])


class FrozenSchemaTests(unittest.TestCase):
    def test_frozen_04_remains_failed_with_opaque_osrt_return_field(self):
        import sqlite3
        old = p.HERE / "artifacts-go-20261015-04"
        case = old / "runs/blocking/session"
        if not (case / "nsys.sqlite").exists():
            self.skipTest("private frozen -04 capture not installed")
        verdict = p.ct.load(case / "worker-profile-audit.json")
        self.assertEqual(verdict["status"], "rejected")
        self.assertFalse(verdict["scoped_overlap_accepted"])
        self.assertEqual(set(verdict["scoped_issues"]), {
            "failed/empty worker pread", "worker pread bytes/counts missing or differ from NVTX requested bytes"})
        self.assertEqual(p.ct.load(old / "prepared.json")["generation"], old.name)
        with sqlite3.connect((case / "nsys.sqlite").resolve().as_uri() + "?mode=ro", uri=True) as db:
            self.assertEqual({r[1] for r in db.execute("PRAGMA table_info(OSRT_API)")},
                {"start", "end", "eventClass", "globalTid", "nameId", "returnValue", "nestingLevel", "callchainId"})
            rows = db.execute("SELECT DISTINCT returnValue FROM OSRT_API JOIN StringIds ON nameId=id "
                              "WHERE value IN (?,?)", ("pread", "pread64")).fetchall()
            self.assertEqual(rows, [(0,)])


if __name__ == "__main__":
    unittest.main()
