"""CPU-only checks for diagnostic reduction replay and BF16-bin attribution."""
import unittest

import torch
import torch.nn.functional as F

import diagnose_numeric_first as diagnose


class FirstDivergenceTests(unittest.TestCase):
    def test_f32_reduction_order_is_not_f64_sum(self):
        x = torch.ones(1, 128)
        weight = torch.zeros(1, 128)
        weight[0, :3] = torch.tensor([2.**24, 1., -2.**24])
        # Four per-lane accumulators add (large + 1) before cancellation.
        actual = diagnose.packed_vector_reduction(x, weight)
        self.assertEqual(actual.item(), 0.)
        self.assertEqual(F.linear(x.double(), weight.double()).item(), 1.)

    def test_reduction_identity_multiple_rows_and_k_tiles(self):
        x = torch.arange(512, dtype=torch.float32).reshape(2, 256).bfloat16().float()
        weight = torch.eye(256)[:4]
        self.assertTrue(torch.equal(diagnose.packed_vector_reduction(x, weight), x[:, :4]))

    def test_replay_rejects_non_bf16_operands(self):
        x, weight = torch.ones(1, 128), torch.ones(1, 128)
        x[0, 0] = 1.001
        with self.assertRaises(AssertionError):
            diagnose.packed_vector_reduction(x, weight)

    def test_tiny_upstream_error_crosses_rne_bin_and_explains_projection(self):
        # Adjacent F32 values at a BF16 midpoint straddle different BF16 bins.
        midpoint = torch.tensor(1.00390625)
        above = torch.nextafter(midpoint, torch.tensor(float('inf')))
        x, reference = above.reshape(1, 1), midpoint.reshape(1, 1)
        weight = torch.tensor([[2.], [-3.]])
        actual = F.linear(diagnose.ref.independent_bf16_rne(x), weight)
        expected = F.linear(diagnose.ref.independent_bf16_rne(reference), weight)
        self.assertEqual(diagnose.metric(x, reference)['outside'], 0)
        self.assertGreater(diagnose.metric(actual, expected)['outside'], 0)
        effect = diagnose.bin_effect(x, reference, weight, actual, expected)
        self.assertEqual(effect['changed_elements'], 1)
        self.assertEqual(effect['gpu_vs_oracle_plus_operand_delta']['max_abs'], 0.)
        self.assertEqual(effect['examples'][0]['reference_minus_midpoint'], 0.)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
