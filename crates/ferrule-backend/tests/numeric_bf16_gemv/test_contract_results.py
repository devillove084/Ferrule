"""CPU-only negative tests for the durable, zero-tolerance BF16 output audit."""
import copy
import json
import unittest
from pathlib import Path

import contract_results as results

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = ROOT / "target/validation/capital-gemv-contract-20261006"


class Results(unittest.TestCase):
    def test_bits_not_numeric_tolerance(self):
        self.assertEqual(results.differences(b"\x00\x00", b"\x00\x80"),
                         [dict(index=0, actual="0000", hf="8000", abs_diff=0.0)])
        self.assertEqual(results.differences(b"\xe2\x32", b"\xe1\x32")[0],
                         dict(index=0, actual="32e2", hf="32e1",
                              abs_diff=1.1641532182693481e-10))
        self.assertEqual(results.differences(b"\xe1\x32", b"\xe1\x32"), [])

    def test_extent_errors(self):
        for a, b in [(b"a", b"a"), (b"aa", b""), (b"aa", b"aaaa")]:
            with self.assertRaises(ValueError):
                results.differences(a, b)

    def test_receipt_rejects_false_success_and_unauthenticated_drain(self):
        receipt = json.loads((EVIDENCE / "native-124-supervisor.json").read_text())
        pin = results.contract_run.sha(EVIDENCE / "PREPARED.json")
        results.validate_receipt(receipt, pin)
        for field, value in [("returncode", 1), ("timed_out", True),
                             ("stdout_truncated", True), ("cleanup_errors", ["failure"])]:
            changed = copy.deepcopy(receipt)
            changed["supervisor"][field] = value
            with self.assertRaises(ValueError):
                results.validate_receipt(changed, pin)
        changed = copy.deepcopy(receipt)
        changed["supervisor"]["custody"]["launch_id"] = "different-launch"
        with self.assertRaises(ValueError):
            results.validate_receipt(changed, pin)
        changed = copy.deepcopy(receipt)
        changed["gpu_after"].pop()
        with self.assertRaises(ValueError):
            results.validate_receipt(changed, pin)

    def test_frozen_byte_matrix(self):
        result = results.audit(EVIDENCE)
        self.assertTrue(result["matched_bf16_gemmex_exact"])
        self.assertTrue(result["native_repetitions_stable"])
        self.assertTrue(result["binary_equal_between_versions"])
        self.assertEqual(result["words_per_matched_pass"], 14336)
        # These are regression assertions on frozen payloads, not dispatch rules.
        counts = dict(expert146_down=1, expert200_down=1, expert238_down=3)
        for by_case in result["matrix"].values():
            for case, by_config in by_case.items():
                for config, value in by_config.items():
                    expected = counts.get(case, 0) if config in (
                        "gemmex-no-workspace", "tst-prior") else 0
                    self.assertEqual(value["max_unequal_hf"], expected)


if __name__ == "__main__":
    unittest.main()
