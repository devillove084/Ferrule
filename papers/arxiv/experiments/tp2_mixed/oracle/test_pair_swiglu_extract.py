"""Stdlib CPU tests for byte extraction and fail-closed paper handoff."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

import pair_swiglu_extract as e

RUN = e.HERE / "runs/pair-swiglu-m1-cpu-01"
RUN_PIN = "47a6c82c5a3baf9609a2d13369d556390e4904944eb956c17b017ccfc9a35d2d"


class ByteExtractionTests(unittest.TestCase):
    def test_rank_row_partition_and_scale_grid(self):
        weight = b"".join(bytes([i // 128]) * 2048 for i in range(512))
        self.assertEqual(e.local_rows(weight, 0, 512, 2048, 1), weight[:524288])
        self.assertEqual(e.local_rows(weight, 1, 512, 2048, 1), weight[524288:])
        scales = struct.pack("<64H", *range(64))
        self.assertEqual(struct.unpack("<32H", e.local_rows(scales, 1, 4, 16, 2)), tuple(range(32, 64)))

    def test_invalid_extents_and_rank_rejected(self):
        for args in ((b"", 0, 512, 2048, 1), (bytes(128), 2, 4, 16, 2), (bytes(96), 0, 3, 16, 2)):
            with self.assertRaises(ValueError):
                e.local_rows(*args)

    def test_bf16_validation(self):
        e.finite_bf16(struct.pack("<4H", 0, 0x8000, 0x3f80, 0xbf80))
        e.finite_bf16(struct.pack("<2H", 1, 0x3f80), positive=True)
        for raw, positive in ((b"x", False), (struct.pack("<H", 0x7f80), False),
                              (struct.pack("<H", 0x7fc1), False), (bytes(2), True),
                              (struct.pack("<H", 0x8001), True)):
            with self.assertRaises(ValueError):
                e.finite_bf16(raw, positive)

    def test_checksum_tamper_and_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").write_bytes(b"original")
            sums = root / "SHA256SUMS"
            sums.write_text(e.sha(root / "data") + "  data\n")
            pin = e.sha(sums)
            self.assertEqual(e.check_sums(root, sums.name, pin), {"data"})
            (root / "data").write_bytes(b"modified")
            with self.assertRaises(ValueError):
                e.check_sums(root, sums.name, pin)
            sums.write_text("0" * 64 + "  ../outside\n")
            with self.assertRaises(ValueError):
                e.check_sums(root, sums.name, e.sha(sums))

    def test_existing_release_cannot_be_overwritten(self):
        with self.assertRaisesRegex(ValueError, "Fresh immediate"):
            e.extract(RUN)


class FrozenPreparationTests(unittest.TestCase):
    def test_release_checksums_and_explicit_missing_input(self):
        e.check_sums(RUN, "SHA256SUMS", RUN_PIN)
        release = e.load(RUN / "manifest.json")
        self.assertFalse(release["fixture_ready"])
        self.assertEqual(release["complete_rank_fixtures"], 0)
        self.assertEqual(release["prepared_weight_activation_rank_pairs"], 8)
        self.assertEqual(e.sha(RUN / "extractor-source.py"), e.sha(Path(e.__file__)))
        for item in release["fixtures"]:
            root = RUN / item["directory"]
            e.check_sums(root, "SHA256SUMS", item["sha256sums_sha256"])
            manifest = e.load(root / "manifest.json")
            self.assertEqual((manifest["rows"], manifest["n"], manifest["k"], manifest["packed_n"]), (1, 256, 2048, 512))
            self.assertFalse(manifest["fixture_ready"])
            self.assertEqual(manifest["normalized_input"]["shape"], [1, 2048])
            self.assertIsNone(manifest["normalized_input"]["file"])
            self.assertIsNone(manifest["normalized_input"]["sha256"])
            self.assertEqual(len(manifest["observed_tensor_inventory"]), 13)
            self.assertFalse(list(root.glob("rank*.input.bf16")))

    def test_all_rank_bytes_independently_match_immutable_sources(self):
        cpu = e.load(e.CPU / "manifest.json")
        count = 0
        for item in e.load(RUN / "manifest.json")["fixtures"]:
            name = item["directory"]
            root = RUN / name
            manifest = e.load(root / "manifest.json")
            for rank_record in manifest["ranks"]:
                rank = rank_record["rank"]
                for kind, spec in rank_record["files"].items():
                    actual = (root / spec["file"]).read_bytes()
                    self.assertEqual(len(actual), spec["bytes"])
                    self.assertEqual(e.digest(actual), spec["sha256"])
                    if kind == "expected_activation":
                        original = (e.CUDA / name / ("rank%d.input.bf16" % rank)).read_bytes()
                        self.assertEqual(len(original), 512)
                        self.assertEqual(actual, original)
                        count += 1
                    else:
                        source = cpu["original_payloads"][spec["source"]["tensor"]]
                        original = (e.CPU / source["file"]).read_bytes()
                        half = len(original) // 2
                        self.assertEqual(actual, original[rank * half:(rank + 1) * half])
                        self.assertEqual(spec["shape"], [2, 16] if kind.endswith("scale") else [256, 2048])
        self.assertEqual(count, 8)

    def test_m5_actual_routes_not_tiled_m1(self):
        evidence = e.load(RUN / "m5-evidence.json")
        self.assertEqual(evidence, e.m5_inventory(e.load(e.REF / "manifest.json")))
        self.assertEqual((evidence["selected_layer"], evidence["selected_expert"]), (9, 95))
        layer = evidence["route_inventory"][9]
        self.assertTrue(all(95 in row for row in layer["selected_expert_ids"]))
        self.assertEqual(layer["active_expert_count"], 24)
        self.assertFalse(evidence["route_inventory"][0]["all_five_experts"])
        self.assertFalse(evidence["route_inventory"][3]["all_five_experts"])
        self.assertEqual(evidence["full_control_calls"]["native_experts_forward"], 2)
        self.assertEqual(evidence["full_control_calls"]["packed_gate_up_linear"], 48)
        self.assertFalse(evidence["cuda_executed"])


if __name__ == "__main__":
    unittest.main()
