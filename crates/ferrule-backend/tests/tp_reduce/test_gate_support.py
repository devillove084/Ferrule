"""CPU-only tests of evidence parsing/bounds. Never queries or initializes a GPU."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import gate_support as s


class EvidenceTests(unittest.TestCase):
    def test_exact_selectors_are_distinct_and_unknown_is_last(self):
        self.assertEqual(len(set(s.CASES.values())), 4)
        self.assertEqual(list(s.CASES)[-1], 'unknown')
        self.assertEqual(len(s.UUIDS), 2)
        self.assertIn('m1_m5_rows8', s.CASES['finite'])

    def test_ownership_capabilities_are_sealed_separately(self):
        self.assertEqual(len(s.case_rows('ownership')), 6)
        self.assertEqual(len(s.case_rows('numerical')), 4)
        self.assertTrue(all('ownership::control_' in v for v in s.case_rows('ownership').values()))
        with self.assertRaises(ValueError):
            s.case_rows('public')

    def test_native_probe_never_forces_primary_cleanup(self):
        source = (s.ROOT / 'crates/ferrule-backend/native/cuda/core/tp_ownership_probe.cuh').read_text()
        self.assertIn('cuDevicePrimaryCtxGetState', source)
        for forbidden in ('cudaDeviceReset(', 'cuDevicePrimaryCtxRetain(', 'cuDevicePrimaryCtxRelease(', 'cuDevicePrimaryCtxReset('):
            self.assertNotIn(forbidden, source)

    def test_rust_test_name_prefix_and_nvml_precision(self):
        event = dict(stage='beforecontext', all_eight_nvml_csv='0, GPU-test, 24576, 24125, 0, 0\n', selected_cu=[])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.log'
            path.write_text('test selector ... TP_MEMORY ' + json.dumps(event) + '\nTP_NCCL_ERROR result=1\n')
            rows = s.parse_log(path)
        self.assertEqual(rows, [dict(kind='TP_MEMORY', **event)])
        report = s.memory_report(rows)
        self.assertEqual(report['deltas'], [])
        self.assertEqual(report['private_device_bound'], 'Unknown')
        self.assertFalse(report['hard_cap_claim'])
        self.assertIn('NVML MiB', report['beforecontext_precision'])

    def test_global_delta_is_not_an_allocation_bound(self):
        def stage(name, free, reserved):
            return dict(kind='TP_MEMORY', stage=name, selected_cu=[dict(ordinal=0, cu_free_bytes=free, reserved_bytes=reserved)])
        report = s.memory_report([
            stage('aftercontext', 1000000, 0),
            stage('beforeNccl_afterOwnedGrant', 900000, 262144),
            stage('afterinit_async_ready_beforeWarm', 800000, 262144),
        ])
        self.assertEqual(report['deltas'][1]['measured_allocator_reserved_delta_bytes'], 0)
        self.assertEqual(report['deltas'][1]['measured_device_free_drop_bytes'], 100000)
        self.assertTrue(report['measured_only'])
        self.assertFalse(report['hard_cap_claim'])
        self.assertEqual(report['private_channel_sliding_buffer_bound'], 'Unknown')

    def test_idle_rejects_nonselected_usage_even_if_framework_says_verified(self):
        row = dict(status='verified', baseline={str(i):dict(index=i, gpu_bytes=0) for i in range(8)})
        row['baseline']['7']['gpu_bytes'] = 1
        with mock.patch.object(s.observer, 'idle', return_value=row):
            with self.assertRaises(ValueError):
                s.idle('test note; no GPU query')


if __name__ == '__main__':
    unittest.main()
