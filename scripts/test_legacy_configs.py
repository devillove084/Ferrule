"""Syntax-only checks for inert legacy examples; not a runtime config parser.

Run with Python 3.11+ (stdlib tomllib). No Cargo/TOML dependency is added.
"""
from pathlib import Path
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LegacyConfigContracts(unittest.TestCase):
    def test_examples_remain_valid_toml_without_claiming_execution(self):
        for name in ("agent", "train", "rollout"):
            with self.subTest(example=name):
                text = (ROOT / "configs" / f"{name}.toml").read_text()
                self.assertTrue(text.startswith("# LEGACY / NON-EXECUTABLE EXAMPLE\n"))
                header = text.split("[observability]", 1)[0]
                self.assertIn("CLI does not load this TOML file", header)
                self.assertIn("do not configure", header)
                parsed = tomllib.loads(text)
                expected = {"observability", "model", "rollout"}
                if name != "rollout":
                    expected.add(name)
                self.assertEqual(set(parsed), expected)
                # These old endpoint fields remain inert documentary data.
                self.assertTrue(parsed["observability"]["metrics_enabled"])
                self.assertEqual(parsed["observability"]["metrics_bind"], "127.0.0.1:9000")


if __name__ == "__main__":
    unittest.main()
