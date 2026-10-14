"""CPU-only checks for the narrow routed-fusion qualification contract."""
import unittest

import routed_fusion_gate as gate


class SelectedContract(unittest.TestCase):
    def test_math_is_narrow_and_direct_tail_has_real_target(self):
        rows = gate.selected_gates()
        small = [row for row in rows if row.reference is None]
        self.assertEqual(len(small), 10)
        self.assertEqual(len({row.name for row in rows}), len(rows))
        direct = next(row for row in rows if row.name == 'backend-direct-tail')
        self.assertEqual(direct.binary, 'backend-tail')
        self.assertIn('::gpu::direct_tail_words_', direct.selector)
        hf = next(row for row in rows if row.name == 'real-hf-pair')
        self.assertEqual(hf.binary, 'model-hf')
        self.assertEqual(hf.evidence, 'independent_real_hf_every_bf16_word')

    def test_six_unobserved_recipes_reuse_one_executor_with_cache(self):
        recipes = [row for row in gate.selected_gates() if row.reference in gate.base.RECIPES]
        self.assertEqual({row.reference for row in recipes}, set(gate.base.RECIPES))
        self.assertEqual(len(recipes), 6)
        self.assertEqual(len({row.selector for row in recipes}), 1)
        self.assertTrue(all(row.observer is False and row.cache_bytes == 3 << 30 for row in recipes))

    def test_logits_reject_nonfinite_shape_failure_and_wrong_greedy(self):
        for actual, expected, token, vocab in [
            ([float('nan'), 1.0], [0.0, 1.0], 1, 2),
            ([0.0, 1.0], [float('inf'), 1.0], 1, 2),
            ([0.0], [0.0, 1.0], 1, 2),
            ([0.0, 1.0], [0.0, 1.0], 0, 2),
            ([0.0, 1.0], [0.0, 2.0], 1, 2),
        ]:
            with self.subTest(actual=actual, expected=expected):
                with self.assertRaises(ValueError):
                    gate.checked_finite_logits(actual, expected, token, vocab)
        result = gate.checked_finite_logits([0.0, 1.0], [0.0, 1.0], 1, 2)
        self.assertTrue(result['all_exact'])
        self.assertEqual(result['failures'], 0)

    def test_diagnostic_cannot_claim_cached_fusion(self):
        row = next(row for row in gate.selected_gates() if row.name == 'fused-cache-full40')
        contract = dict(schema='ferrule.paper.routed-fusion-execution-path.v1',
                        path='unobserved_fused', numeric_observer_installed=False,
                        committed_tensor_export=False, cache_grants_per_owner=3 << 30,
                        device_cap_bytes=22 << 30, atol=.00025, rtol=.012)
        gate.validate_execution_claim(row, contract)
        for changed in (dict(numeric_observer_installed=True), dict(cache_grants_per_owner=None),
                        dict(path='diagnostic_fallback'), dict(committed_tensor_export=True)):
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    gate.validate_execution_claim(row, dict(contract, **changed))


if __name__ == '__main__':
    unittest.main()
