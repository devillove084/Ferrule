"""Canonical-counter checks: malformed/missing accounting is not a baseline."""
import copy
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.append(str(SCRIPTS))

from bench_process_ipc import summarize_ipc, validate_counters


class CounterContracts(unittest.TestCase):
    def setUp(self):
        phases = {}
        for name in ["encode", "payload_decode", "frame_decode", "frame_write", "frame_read", "pipe_wait", "copy"]:
            phases[name] = {"calls": 146, "completed": 146, "completed_bytes": 1024, "wall_ns": 100}
        phases["pipe_wait"]["completed_bytes"] = 0
        self.records = [{"schema": "ferrule.ipc-timing.v1", "pid": pid, "phases": copy.deepcopy(phases)} for pid in [1, 2]]
        self.report = {"pid": 1, "result": {"child_pid": 2}}

    def test_valid_echo_and_separate_codec(self):
        validate_counters(self.records, self.report, "echo")
        validate_counters([], {}, "codec")

    def test_missing_child_is_not_verified(self):
        with self.assertRaises(AssertionError):
            validate_counters(self.records[:1], self.report, "echo")

    def test_frame_count_bytes_failure_and_overlapping_costs(self):
        for phase, field, value in [
            ("frame_write", "calls", 147),
            ("frame_write", "completed_bytes", 1025),
            ("frame_read", "completed", 145),
            ("copy", "completed_bytes", 2048),
            ("copy", "wall_ns", 101),
            ("pipe_wait", "wall_ns", 201),
        ]:
            with self.subTest(phase=phase, field=field):
                records = copy.deepcopy(self.records)
                records[0]["phases"][phase][field] = value
                with self.assertRaises(AssertionError):
                    validate_counters(records, self.report, "echo")

    def test_aggregate_null_timing_and_mixed_records_remain_missing(self):
        for record in self.records:
            record['thread'] = 'ThreadId(1)'
        self.records[0]['schema'] = 'ferrule.ipc-timing.v2'
        self.records[0]['phases']['copy']['wall_ns'] = None
        validate_counters(self.records, self.report, 'echo')
        for records in (self.records, list(reversed(self.records))):
            summary = summarize_ipc(records)
            self.assertIsNone(summary['phases']['copy']['wall_ns'])
            self.assertIsNone(summary['phases']['copy']['inclusive_thread_wall_seconds'])
            self.assertEqual(summary['phases']['copy']['completed_bytes'], 2048)
        self.assertIsNone(summarize_ipc([])['phases']['copy']['calls'])
        self.assertIsNone(summarize_ipc([])['phases']['encode']['wall_ns'])

    def test_null_is_only_valid_for_v2_copy_duration(self):
        record = copy.deepcopy(self.records[0])
        record['thread'] = 'ThreadId(1)'
        record['phases']['copy']['wall_ns'] = None
        with self.assertRaises(ValueError):
            summarize_ipc([record])
        record['schema'] = 'ferrule.ipc-timing.v2'
        summarize_ipc([record])
        for phase, field in [('encode', 'wall_ns'), ('copy', 'calls')]:
            bad = copy.deepcopy(record)
            bad['phases'][phase][field] = None
            with self.assertRaises(ValueError):
                summarize_ipc([bad])

    def test_codec_cannot_claim_transport(self):
        with self.assertRaises(AssertionError):
            validate_counters(self.records, {}, "codec")


if __name__ == "__main__":
    unittest.main()
