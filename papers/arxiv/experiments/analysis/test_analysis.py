import json
import sys
import tempfile
import struct
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from cache_plan import cache_capacity, counterfactuals, expert_headers  # noqa: E402
from performance import analyze_http, analyze_lines, analyze_log  # noqa: E402


def event(target, fields):
    return json.dumps({"timestamp": "2026-10-08T00:00:00Z", "target": target, "fields": fields})


def perf(transaction, *, rows=8, prefill=0, decode=8, reset=False, available=True,
         status="completed", quarantined=False, host=1000, hits=2):
    fields = {
        "event": "mixed_decoder_invocation", "transaction": transaction, "device": 0,
        "status": status, "quarantined": quarantined, "timing_scope": (
            "decoder_validate_through_finish_or_failure_not_device_kernel_or_publication"),
        "rows": rows, "sequences": 2, "prefill_rows": prefill, "decode_rows": decode,
        "counters_available": available, "counter_reset_detected": reset,
        "host_invocation_us": host,
        "facade_kernel_launches": 10, "h2d_copies": 1, "h2d_bytes": 20,
        "d2h_copies": 2, "d2h_bytes": 40, "artifact_uploads": 3,
        "artifact_upload_bytes": 60, "allocation_attempts": 4, "allocation_bytes": 80,
        "facade_stream_syncs": 5, "facade_stream_sync_failures": 0,
        "expert_hits": hits, "expert_uploads": 1, "expert_evictions": 1,
        "checkpoint_read_calls": 2, "checkpoint_read_bytes": 100,
        "checkpoint_source_checks": 3, "checkpoint_path_metadata_calls": 4,
        "checkpoint_handle_metadata_calls": 5,
        "consumer_event_record_sync_calls": 6, "consumer_event_record_sync_failures": 0,
        "consumer_event_record_sync_host_us": 7, "semantic_compute_sync_calls": 8,
        "semantic_compute_sync_failures": 0, "semantic_compute_sync_host_us": 9,
        "semantic_upload_sync_calls": 10, "semantic_upload_sync_failures": 0,
        "semantic_upload_sync_host_us": 11,
        "expert_resident_entries": 8, "expert_resident_bytes": 1000,
        "expert_pending_upload_bytes": 0, "expert_scratch_bytes": 8,
        "expert_owner_peak_bytes": 1008, "expert_unknown_quarantine_bytes": 0,
    }
    return event("ferrule_performance", fields)


def worker(count, *, enabled=True, cancelled=False):
    first, last = ((100, 250) if count > 1 else (100, 100)) if count else (None, None)
    fields = {
        "event": "request_worker_terminal", "request_id": 4, "session_id": 9,
        "status": "cancelled" if cancelled else "finished", "finish_reason": "cancelled" if cancelled else "length",
        "worker_request_us": 500, "generated_tokens": count, "emitted_token_events": count,
        "emission_timing_enabled": enabled, "worker_first_emission_elapsed_us": first,
        "worker_last_emission_elapsed_us": last, "worker_emission_interval_count": max(count - 1, 0) if enabled else None,
        "worker_emission_interval_sum_us": 150 if enabled and count > 1 else 0 if enabled else None,
        "worker_emission_interval_min_us": 150 // (count - 1) if enabled and count > 1 else None,
        "worker_emission_interval_max_us": 150 // (count - 1) if enabled and count > 1 else None,
        "cancellation_requested": cancelled, "cancellation_submitted": cancelled,
        "cancellation_failed": False, "requires_restore_or_shutdown": False,
    }
    return event("ferrule_request", fields)


class PerformanceAnalysisTests(unittest.TestCase):
    def test_shared_cost_is_summed_once_and_phase_partitioned(self):
        report = analyze_lines([perf(1, rows=8, prefill=8, decode=0, host=100),
                                perf(2, rows=2, prefill=0, decode=2, host=50)])
        self.assertEqual(report["instrumentation"], "observed")
        self.assertEqual(report["invocation_summary"]["rows"]["total"], 10)
        self.assertEqual(report["invocation_summary"]["metrics"]["host_invocation_us"]["total"], 150)
        self.assertEqual(report["invocation_summary"]["metrics"]["host_invocation_us"]["per_row"], 15)
        self.assertEqual(report["phases"]["prefill"]["metrics"]["host_invocation_us"]["total"], 100)
        self.assertEqual(report["phases"]["decode"]["metrics"]["host_invocation_us"]["total"], 50)
        self.assertEqual(report["invocations"][0]["phase"], "prefill")
        self.assertEqual(report["invocations"][1]["phase"], "decode")
        self.assertEqual(report["invocation_summary"]["gauges"]["expert_owner_peak_bytes"]["max"], 1008)
        self.assertEqual(report["request_cost_attribution"], None)

    def test_missing_nan_and_reset_are_unknown_not_zero(self):
        reset = json.loads(perf(2))
        reset["fields"]["counter_reset_detected"] = True
        reset["fields"]["facade_kernel_launches"] = 0
        missing = json.loads(perf(3))
        missing["fields"].pop("h2d_bytes")
        missing["fields"]["host_invocation_us"] = float("nan")
        lines = [json.dumps(reset), json.dumps(missing)]
        report = analyze_lines(lines)
        metrics = report["invocation_summary"]["metrics"]
        self.assertIsNone(report["invocations"][0]["values"]["facade_kernel_launches"])
        self.assertIsNone(report["invocations"][1]["values"]["h2d_bytes"])
        self.assertIsNone(report["invocations"][1]["values"]["host_invocation_us"])
        self.assertIsNone(metrics["facade_kernel_launches"]["total"])
        self.assertEqual(metrics["facade_kernel_launches"]["known"], 1)
        self.assertEqual(metrics["facade_kernel_launches"]["observed_sum"], 10)
        self.assertEqual(report["diagnostics"].get("non_json_lines", 0), 0)

    def test_worker_intervals_are_distinct_from_client_tpot(self):
        report = analyze_lines([worker(3, cancelled=True)])
        item = report["worker_requests"][0]
        self.assertEqual(item["worker_observed_mean_interval_us"], 75)
        self.assertIsNone(item["client_exact_token_tpot_seconds"])
        self.assertTrue(item["intervals_consistent"])
        http = analyze_http({"runs": [{"requests": [{
            "status": "success", "actual_output_tokens": 3, "ttft_seconds": 0.1,
            "mean_tpot_usage_proxy_seconds": 0.2, "exact_token_tpot_seconds": None,
        }]}]})
        self.assertEqual(http["requests"][0]["mean_tpot_usage_proxy_seconds"], 0.2)
        self.assertFalse(http["joined_to_worker_requests"])

    def test_disabled_baseline_log_has_no_causal_counters(self):
        path = Path(__file__).parents[1] / "runs/m7-feasibility-exclusive-002/ferrule/server.log"
        if not path.is_file():
            self.skipTest("retained local M7 artifact unavailable")
        report = analyze_log(path)
        self.assertEqual(report["instrumentation"], "not_observed")
        self.assertEqual(report["invocations"], [])
        self.assertEqual(report["worker_requests"], [])
        self.assertIsNone(report["invocation_summary"]["metrics"]["host_invocation_us"]["total"])

    def test_duplicate_identity_is_not_summed(self):
        one = perf(1)
        report = analyze_lines([one, one])
        self.assertEqual(report["diagnostics"]["duplicate_records"], 1)
        self.assertEqual(report["invocation_summary"]["invocations"], 1)

    def test_unknown_allocation_cost_is_explicit(self):
        report = analyze_lines([perf(1)])
        metric = report["invocation_summary"]["metrics"]["allocation_host_us"]
        self.assertIsNone(metric["total"])
        self.assertIsNone(report["invocations"][0]["values"]["allocation_host_us"])


    def test_zero_one_and_disabled_worker_intervals(self):
        for count, enabled in [(0, True), (1, True), (3, False)]:
            with self.subTest(count=count, enabled=enabled):
                item = analyze_lines([worker(count, enabled=enabled)])["worker_requests"][0]
                self.assertIsNone(item["worker_observed_mean_interval_us"])
                if enabled:
                    self.assertTrue(item["intervals_consistent"])
                    self.assertEqual(item["values"]["worker_emission_interval_count"], 0)
                else:
                    self.assertIsNone(item["values"]["worker_emission_interval_count"])

    def test_missing_worker_flag_is_unknown_not_false(self):
        record = json.loads(worker(2))
        del record["fields"]["emission_timing_enabled"]
        item = analyze_lines([json.dumps(record)])["worker_requests"][0]
        self.assertIsNone(item["emission_timing_enabled"])
        self.assertIsNone(item["worker_observed_mean_interval_us"])

    def test_inconsistent_worker_endpoints_reject_mean(self):
        record = json.loads(worker(2))
        record["fields"]["worker_last_emission_elapsed_us"] = 99
        item = analyze_lines([json.dumps(record)])["worker_requests"][0]
        self.assertFalse(item["intervals_consistent"])
        self.assertIsNone(item["worker_observed_mean_interval_us"])

    def test_invalid_numbers_are_never_coerced(self):
        for invalid in [None, float("nan"), float("inf"), -1, True, "12", 1.5, 2**65]:
            with self.subTest(invalid=invalid):
                record = json.loads(perf(1))
                record["fields"]["h2d_bytes"] = invalid
                report = analyze_lines([json.dumps(record)])
                self.assertIsNone(report["invocations"][0]["values"]["h2d_bytes"])
                json.dumps(report, allow_nan=False)

    def test_valid_zero_remains_zero(self):
        report = analyze_lines([perf(1, hits=0)])
        self.assertEqual(report["invocation_summary"]["metrics"]["expert_hits"]["total"], 0)

    def test_mixed_is_not_counted_in_both_phases(self):
        report = analyze_lines([perf(1, rows=8, prefill=6, decode=2)])
        self.assertEqual(report["phases"]["mixed"]["invocations"], 1)
        self.assertEqual(report["phases"]["prefill"]["invocations"], 0)
        self.assertEqual(report["phases"]["decode"]["invocations"], 0)
        self.assertIsNone(report["first_observed_prefill"])

    def test_first_prefill_means_first_observed_not_per_request(self):
        report = analyze_lines([perf(1), perf(2, prefill=8, decode=0), perf(3, prefill=8, decode=0)])
        self.assertEqual(report["first_observed_prefill"]["transaction"], 2)
        self.assertIsNone(report["request_cost_attribution"])

    def test_conflicting_identity_blocks_totals(self):
        report = analyze_lines([perf(1, host=100), perf(1, host=200)])
        self.assertFalse(report["aggregate_identity_valid"])
        self.assertIsNone(report["invocation_summary"])
        self.assertEqual(report["diagnostics"]["conflicting_invocation_identities"], 1)

    def test_failure_quarantine_and_no_snapshot_are_preserved(self):
        report = analyze_lines([perf(1, available=False, status="failed", quarantined=True)])
        item = report["invocations"][0]
        self.assertTrue(item["quarantined"])
        self.assertEqual(item["status"], "failed")
        self.assertIsNone(item["values"]["expert_hits"])
        self.assertEqual(item["values"]["consumer_event_record_sync_host_us"], 7)
        self.assertEqual(report["completed_invocation_summary"]["invocations"], 0)

    def test_wrong_targets_and_malformed_lines_do_not_manufacture_events(self):
        wrong = json.loads(perf(1)); wrong["target"] = "not_ferrule_performance"
        weird = event("ferrule_request", {"event": []})
        report = analyze_lines(["plain stderr", "{broken", json.dumps(wrong), weird, "[]"])
        self.assertEqual(report["instrumentation"], "not_observed")
        self.assertIsNone(report["invocation_summary"]["metrics"]["h2d_bytes"]["total"])
        self.assertEqual(report["diagnostics"]["malformed_json_lines"], 1)

    def test_proof_times_are_separate_nonoverlapping_observed_call_groups(self):
        item = analyze_lines([perf(1)])["invocations"][0]
        self.assertEqual(item["observed_proof_host_us"], 27)
        self.assertEqual(item["observed_proof_host_fraction"], 0.027)
        self.assertEqual(item["per_row"]["consumer_event_record_sync_calls"], 0.75)

    def test_gauges_are_max_last_not_sum(self):
        second = json.loads(perf(2)); second["fields"]["expert_resident_bytes"] = 500
        report = analyze_lines([perf(1), json.dumps(second)])
        gauge = report["invocation_summary"]["gauges"]["expert_resident_bytes"]
        self.assertEqual(gauge["max"], 1000)
        self.assertEqual(gauge["last"], 500)
        self.assertNotIn("total", gauge)

    def test_invalid_rows_disable_normalization(self):
        report = analyze_lines([perf(1, rows=0, prefill=0, decode=0)])
        self.assertEqual(report["invocations"][0]["phase"], "unknown")
        self.assertIsNone(report["invocations"][0]["per_row"]["h2d_bytes"])

    def test_http_nan_and_missing_remain_unknown(self):
        report = analyze_http({"runs": [{"requests": [{"ttft_seconds": float("nan")}]}]})
        self.assertIsNone(report["requests"][0]["ttft_seconds"])
        self.assertIsNone(report["requests"][0]["mean_tpot_usage_proxy_seconds"])
        json.dumps(report, allow_nan=False)


class CachePlanTests(unittest.TestCase):
    def setUp(self):
        self.metadata = {"model": "header-only", "layers": 40, "experts_per_layer": 256,
                         "top_k": 8, "expert_count": 40 * 256,
                         "expert_payload_bytes": 3_146_112, "header_bytes_read": 1,
                         "payload_bytes_read": 0, "header_sha256": {}, "config_sha256": "x",
                         "index_sha256": "y"}

    def test_cross_layer_entry_and_byte_bounds(self):
        cap = cache_capacity(entries=8, cache_bytes=320 * 2**20, scratch_bytes=8 * 2**20,
                             expert_bytes=self.metadata["expert_payload_bytes"])
        self.assertEqual(cap["resident_expert_upper_bound"], 8)
        self.assertEqual(cap["byte_only_expert_upper_bound"], 103)
        requested = cache_capacity(entries=256, cache_bytes=320 * 2**20, scratch_bytes=8 * 2**20,
                                   expert_bytes=self.metadata["expert_payload_bytes"])
        self.assertEqual(requested["resident_expert_upper_bound"], 103)

    def test_counterfactual_flags_are_nonlaunching_and_2g_is_under_10g(self):
        plan = counterfactuals(self.metadata, 6_846_743_296)
        self.assertEqual(plan["single_row_layer_expert_visits"], 320)
        self.assertEqual(plan["full_catalog_layer_experts"], 10_240)
        self.assertFalse(plan["launch_authorized"])
        self.assertFalse(plan["protocol_modified"])
        case = plan["cases"][2]
        self.assertEqual(case["resident_expert_upper_bound"], 512)
        self.assertLess(case["configured_device_budget_bytes"], 10 * 2**30 + 1)
        self.assertEqual(case["factory_metadata_admission"], "not_run")
        self.assertEqual(case["quality"], "unverified_for_candidate")


    def test_invalid_cache_caps_are_rejected(self):
        for entries, cache, scratch in [(0, 10, 1), (True, 10, 1), (8, 10, 10), (8, 10, -1)]:
            with self.assertRaises(ValueError):
                cache_capacity(entries=entries, cache_bytes=cache, scratch_bytes=scratch, expert_bytes=3)

    def test_header_only_numeric_payload_accounting(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root = Path(directory)
            config = {"text_config": {"hidden_size": 2048, "moe_intermediate_size": 512,
                      "num_hidden_layers": 1, "num_experts": 1, "num_experts_per_tok": 1}}
            header, index, offset = {}, {}, 0
            for role in ["gate", "up", "down"]:
                for kind in ["weight", "weight_scale_inv"]:
                    name = f"model.language_model.layers.0.mlp.experts.0.{role}_proj.{kind}"
                    shape = [2048, 512] if role == "down" else [512, 2048]
                    scale = kind == "weight_scale_inv"
                    if scale: shape = [n // 128 for n in shape]
                    size = shape[0] * shape[1] * (2 if scale else 1)
                    header[name] = {"dtype": "BF16" if scale else "F8_E4M3", "shape": shape,
                                    "data_offsets": [offset, offset + size]}
                    offset += size
                    index[name] = "model.safetensors"
            raw = json.dumps(header).encode()
            (root / "config.json").write_text(json.dumps(config))
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
            # Deliberately contains no payload: metadata analysis must not try reading one.
            (root / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)
            result = expert_headers(root)
            self.assertEqual(result["expert_payload_bytes"], 3_146_112)
            self.assertEqual(result["header_bytes_read"], 8 + len(raw))
            self.assertEqual(result["payload_bytes_read"], 0)
            index.pop(next(iter(index)))
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                expert_headers(root)


if __name__ == "__main__":
    unittest.main()
