"""CPU-only timestamp, UUID remapping, interval and conservative naming checks."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import profile_trace_audit as audit


class Audit(unittest.TestCase):
    def test_iso_timestamp_is_not_decimal_and_keeps_microseconds(self):
        self.assertEqual(audit.epoch_ns('1970-01-01T00:00:01.123456Z'),1123456000)
        self.assertEqual(audit.epoch_ns('1970-01-01T08:00:01.123456+08:00'),1123456000)

    def test_process_local_zero_maps_to_distinct_uuid(self):
        uuids=sorted(audit.EXPECTED)
        mapping={(10,0):0,(20,0):1};gpus=dict(enumerate(uuids))
        self.assertEqual(audit.physical((1<<48)+(10<<24),0,mapping,gpus),uuids[0])
        self.assertEqual(audit.physical((1<<48)+(20<<24),0,mapping,gpus),uuids[1])
        with self.assertRaises(ValueError):audit.physical(30<<24,0,mapping,gpus)

    def test_union_does_not_double_count_overlap_or_invent_idle(self):
        self.assertEqual(audit.interval_union([(1,5),(2,3),(4,9),(10,11)]),[[1,9],[10,11]])
        self.assertEqual(audit.interval_union([]),[])

    def test_gemm_names_are_not_api_measurements(self):
        self.assertIn('not_GemmEx_API',audit.kernel_group('bf16_gemmex::split_pair'))
        self.assertIn('not_GemmEx_API',audit.kernel_group('ampere_bf16_gemm'))
        self.assertEqual(audit.kernel_group('numeric_fp8::decode_weights'),'decode_weights')


if __name__=='__main__':unittest.main()
