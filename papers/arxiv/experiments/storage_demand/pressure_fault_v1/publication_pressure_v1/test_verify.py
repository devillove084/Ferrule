"""CPU-only positive and adversarial tests of the published evidence checker."""
import copy
import json
from pathlib import Path
import unittest

import verify


class PublicEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = verify.load_bundle()

    def changed(self, kind, point="small", predicate=lambda row: True):
        rows = copy.deepcopy(self.records)
        value = next(row["data"] for row in rows if row["kind"] == kind
                     and row["point"] == point and predicate(row["data"]))
        return rows, value

    def test_public_records_reproduce_summary(self):
        self.assertEqual(verify.derive(self.records), json.loads((verify.HERE / "summary.json").read_text()))

    def test_actual_token_corruption_is_not_hidden_by_reference_ids(self):
        rows, event = self.changed("server_event", predicate=lambda r: r["fields"].get("event") == "request_worker_token_enqueued"
                                  and r["fields"]["request_id"] == 2)
        event["fields"]["actual_token_id"] += 1
        with self.assertRaisesRegex(ValueError, "oracle"):
            verify.derive(rows)

    def test_complete_worker_trace_does_not_replace_http_usage(self):
        rows, request = self.changed("request", predicate=lambda r: r["name"] == "canary-chinese")
        request["actual"]["raw_sse_events"] = [row for row in request["actual"]["raw_sse_events"]
                                               if '"usage"' not in row["data"]]
        with self.assertRaisesRegex(ValueError, "usage"):
            verify.derive(rows)

    def test_missing_done_is_not_complete_http(self):
        rows, request = self.changed("request")
        request["actual"]["raw_sse_events"].pop()
        with self.assertRaisesRegex(ValueError, "DONE"):
            verify.derive(rows)

    def test_hello2_has_no_exact_token_oracle(self):
        rows, request = self.changed("request", predicate=lambda r: r["name"] == "hello2")
        request["trace"]["exact_token_pass"] = True
        with self.assertRaisesRegex(ValueError, "token join"):
            verify.derive(rows)

    def test_fifth_admission_without_tokens_is_detected(self):
        rows = copy.deepcopy(self.records)
        event = copy.deepcopy(next(row for row in rows if row["kind"] == "server_event"
                                  and row["point"] == "small" and row["data"]["fields"].get("event") == "request_admitted"))
        event["data"]["fields"].update(request_id=99, session_id=99)
        rows.append(event)
        with self.assertRaisesRegex(ValueError, "whole-log request"):
            verify.derive(rows)

    def test_reader_delta_corruption_is_detected(self):
        rows, event = self.changed("server_event", predicate=lambda r: r["fields"].get("event") == "mixed_decoder_invocation")
        event["fields"]["checkpoint_read_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "read delta/cumulative"):
            verify.derive(rows)

    def test_counter_missing_is_not_zero(self):
        rows, event = self.changed("server_event", predicate=lambda r: r["fields"].get("event") == "mixed_decoder_invocation")
        event["fields"]["expert_hits"] = None
        with self.assertRaisesRegex(ValueError, "typed invocation"):
            verify.derive(rows)

    def test_owner_peak_must_not_be_summed(self):
        rows, event = self.changed("server_event", predicate=lambda r: r["fields"].get("event") == "resource_shutdown_summary")
        event["fields"]["summary_json"]["physical"]["expert_cache"]["value"]["peak_bytes"] *= 30
        with self.assertRaisesRegex(ValueError, "final cache bounds"):
            verify.derive(rows)

    def test_missing_memory_is_not_a_zero_sample(self):
        rows, sample = self.changed("memory_sample")
        sample["gpu"]["gpu_bytes"] = None
        with self.assertRaisesRegex(ValueError, "missing memory"):
            verify.derive(rows)

    def test_unpaired_helper_span_is_rejected(self):
        rows = copy.deepcopy(self.records)
        index = next(i for i, row in enumerate(rows) if row["kind"] == "phase_timing"
                     and row["data"]["phase"] == "candidate_pins" and row["data"]["status"] == "complete")
        rows.pop(index)
        with self.assertRaisesRegex(ValueError, "open timing spans"):
            verify.derive(rows)

    def test_physical_admission_is_not_filled_from_requested_caps(self):
        rows, options = self.changed("source_options")
        options["effective_admitted_entries"] = 8
        with self.assertRaisesRegex(ValueError, "physical admission"):
            verify.derive(rows)

    def test_idle_does_not_establish_physical_release(self):
        rows, idle = self.changed("coordinator-post-idle")
        idle["physical_owner_release"] = "verified"
        with self.assertRaisesRegex(ValueError, "idle must not prove release"):
            verify.derive(rows)

    def test_history_is_separate_and_censored(self):
        history = json.loads((verify.HERE / "history.json").read_text())
        self.assertFalse(history["pooled"])
        old = history["censored_study"]["analysis"]
        self.assertEqual(old["historical_denominators"], dict(censored=1, failed=0, issued=4,
                         scheduled=12, successful=3, terminal=3, unissued=8))
        self.assertEqual(old["raw_enqueued_tokens"], 26)
        self.assertEqual(old["exact_http_oracle_passes"], 2)
        self.assertIsNone(old["historical_helper_durations_seconds"])
        blocked = history["blocked_start_study"]["observation"]
        self.assertFalse(blocked["registration_created"])
        self.assertEqual(blocked["denominators"]["requests_unissued"], 12)
        self.assertEqual(blocked["server_starts"], 0)


if __name__ == "__main__":
    unittest.main()
