#!/opt/conda310/bin/python
"""Small CPU-only adapter regression checks; no checkpoint reads or model load."""
import os
import sys
import unittest

sys.dont_write_bytecode = True
os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
                  OPENBLAS_NUM_THREADS='8')

import qwen35_fp8_reference as ref


class MemoryStore:
    def __init__(self, values):
        self.values = values
        self.read_count = 0

    def meta(self, name):
        value = self.values[name]
        dtype = {ref.torch.float32: 'F32', ref.torch.bfloat16: 'BF16',
                 ref.torch.float8_e4m3fn: 'F8_E4M3'}[value.dtype]
        return dict(dtype=dtype, shape=list(value.shape))

    def read(self, name):
        self.read_count += 1
        return self.values[name].clone()


class PrecisionAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapters = ref.initialize_torch(8)
        cls.torch, cls.F = ref.torch, ref.F

    def test_rne_signed_midpoint_ties(self):
        t = self.torch
        value = t.tensor([1.00390625, 1.01171875, -1.00390625, -1.01171875])
        expected = t.tensor([1., 1.015625, -1., -1.015625])
        self.assertTrue(t.equal(ref.independent_bf16_rne(value), expected))
        self.assertTrue(t.equal(ref.bf16_f32(value), expected))

    def test_numeric_scale_orientation_partial_blocks_and_not_reciprocal(self):
        t = self.torch
        weight = t.full((129, 257), 1.5).to(t.float8_e4m3fn)
        scale = t.tensor([[.0311, .0573, .0917], [.1257, .2411, .4013]]).bfloat16()
        actual = ref.dequantize(weight, scale)
        self.assertTrue(t.equal(actual, ref.independent_dequantize(weight, scale)))
        self.assertEqual(actual[128, 256].item(), ref.bf16_f32(t.tensor(1.5) * scale[1, 2].float()).item())
        self.assertFalse(t.equal(actual, ref.independent_dequantize(weight, 1 / scale)))

    def test_invalid_scale_shape_dtype_and_values_rejected(self):
        t = self.torch
        weight = t.ones((129, 129)).to(t.float8_e4m3fn)
        for scale in (t.ones(1, 1).bfloat16(), t.ones(2, 2), t.zeros(2, 2).bfloat16(),
                      -t.ones(2, 2).bfloat16(), t.full((2, 2), float('nan')).bfloat16()):
            with self.subTest(shape=scale.shape, dtype=scale.dtype), self.assertRaises(RuntimeError):
                ref.dequantize(weight, scale)
        with self.assertRaises(RuntimeError):
            ref.dequantize(t.full((1, 1), float('nan')).to(t.float8_e4m3fn), t.ones(1, 1).bfloat16())

    def test_quantized_linear_rounds_activation_but_not_output(self):
        t = self.torch
        _, Linear, _ = self.adapters
        weight = t.tensor([[1.5, -2.25], [.75, 3.5]]).to(t.float8_e4m3fn)
        scale = t.tensor([[.0311]]).bfloat16()
        x = t.tensor([[1.005, -.1271]])
        loader = ref.WeightLoader(MemoryStore({'p.weight': weight, 'p.weight_scale_inv': scale}), 1024)
        actual = Linear(loader, 'p.weight')(x)
        expected = self.F.linear(ref.independent_bf16_rne(x), ref.independent_dequantize(weight, scale))
        self.assertTrue(t.equal(actual, expected))
        self.assertEqual(actual.dtype, t.float32)
        self.assertFalse(t.equal(actual, ref.bf16_f32(actual)))
        self.assertFalse(t.equal(actual, self.F.linear(x, ref.dequantize(weight, scale))))

    def test_nonquantized_linear_does_not_round_activation(self):
        t = self.torch
        _, Linear, _ = self.adapters
        for dtype in (t.bfloat16, t.float32):
            weight = t.tensor([[.3127, -.7531]], dtype=dtype)
            x = t.tensor([[1.005, .2491]])
            loader = ref.WeightLoader(MemoryStore({'p.weight': weight}), 1024)
            actual = Linear(loader, 'p.weight')(x)
            self.assertTrue(t.equal(actual, self.F.linear(x, weight.float())))
            self.assertFalse(t.equal(actual, self.F.linear(ref.bf16_f32(x), weight.float())))

    def test_tensor_tag_does_not_patch_global_linear_or_propagate_to_output(self):
        t = self.torch
        Tagged, _, _ = self.adapters
        linear = self.F.linear
        x = t.tensor([[1.005, .2511]])
        weight = t.tensor([[.5, -.75]])
        actual = linear(x, Tagged.wrap(weight))
        self.assertIs(self.F.linear, linear)
        self.assertIs(type(actual), t.Tensor)
        self.assertTrue(t.equal(actual, linear(ref.independent_bf16_rne(x), weight)))
        self.assertFalse(t.equal(actual, linear(x, weight)))

    def test_cache_hits_eviction_and_oversize_bypass(self):
        t = self.torch
        store = MemoryStore({'a': t.ones(4, 4), 'b': t.ones(4, 4) * 2, 'large': t.ones(16, 16)})
        cache = ref.WeightLoader(store, 80)
        cache.matrix('a')
        cache.matrix('a')
        self.assertEqual(store.read_count, 1)
        cache.matrix('b')
        self.assertEqual(list(cache.cache), ['b'])
        self.assertEqual(cache.evictions, 1)
        cache.matrix('large')
        self.assertNotIn('large', cache.cache)
        self.assertLessEqual(cache.peak, 80)
        self.assertLessEqual(cache.bytes, 80)
        self.assertEqual(cache.hits, 1)

    def test_expert_bank_concatenation_and_native_formula(self):
        t = self.torch
        _, _, Bank = self.adapters
        values = {}
        for expert in range(2):
            for part in ('gate_proj', 'up_proj', 'down_proj'):
                name = f'experts.{expert}.{part}.weight'
                values[name] = t.full((2, 2), 1.5 + expert).to(t.float8_e4m3fn)
                values[name + '_scale_inv'] = t.tensor([[.0573 if part == 'up_proj' else .0311]]).bfloat16()
        loader = ref.WeightLoader(MemoryStore(values), 64)
        bank = Bank(loader, 'experts', True, 2)
        matrix = bank[t.tensor(1)].as_subclass(t.Tensor)
        expected = t.cat([ref.independent_dequantize(values[f'experts.1.{p}.weight'],
                           values[f'experts.1.{p}.weight_scale_inv']) for p in ('gate_proj', 'up_proj')])
        self.assertTrue(t.equal(matrix, expected))
        with self.assertRaises(RuntimeError):
            bank[2]
        config = ref.q.Qwen3_5MoeTextConfig(hidden_size=2, moe_intermediate_size=2, num_experts=2)
        experts = ref.q.Qwen3_5MoeExperts(config)
        native_forward = experts.forward.__func__
        for name, gate_up in [('gate_up_proj', True), ('down_proj', False)]:
            del experts._parameters[name]
            setattr(experts, name, Bank(loader, 'experts', gate_up, 2))
        x = t.tensor([[1.005, -.1371]])
        output = experts(x, t.tensor([[1]]), t.ones(1, 1))
        self.assertIs(experts.forward.__func__, native_forward)
        self.assertEqual(output.shape, x.shape)
        self.assertTrue(bool(t.isfinite(output).all()))

    def test_f32_dequantization_preserves_product_without_bf16_round(self):
        t = self.torch
        weight = t.full((129, 257), 1.5).to(t.float8_e4m3fn)
        scale = t.tensor([[.0311, .0573, .0917], [.1257, .2411, .4013]]).bfloat16()
        actual = ref.dequantize(weight, scale, 'f32')
        rows = t.arange(129) // 128
        cols = t.arange(257) // 128
        expected = weight.float() * scale.float()[rows[:, None], cols[None, :]]
        self.assertTrue(t.equal(actual, expected))
        self.assertFalse(t.equal(actual, ref.bf16_f32(actual)))
        self.assertTrue(t.equal(ref.dequantize(weight, scale), ref.bf16_f32(actual)))

    def test_f32_linear_preserves_activation_weight_and_output(self):
        t = self.torch
        Tagged, Linear, _ = ref.make_adapter_types('f32')
        weight = t.tensor([[1.5, -2.25], [.75, 3.5]]).to(t.float8_e4m3fn)
        scale = t.tensor([[.0311]]).bfloat16()
        x = t.tensor([[1.005, -.1271]])
        store = MemoryStore({'p.weight': weight, 'p.weight_scale_inv': scale})
        loader = ref.WeightLoader(store, 1024, precision='f32')
        expected_weight = weight.float() * scale.float()
        expected = self.F.linear(x, expected_weight)
        self.assertTrue(t.equal(Linear(loader, 'p.weight')(x), expected))
        self.assertTrue(t.equal(self.F.linear(x, Tagged.wrap(expected_weight)), expected))
        self.assertFalse(t.equal(expected, self.F.linear(ref.bf16_f32(x), expected_weight)))
        self.assertFalse(t.equal(expected, self.F.linear(x, ref.bf16_f32(expected_weight))))
        self.assertFalse(t.equal(expected, ref.bf16_f32(expected)))

    def test_precision_profiles_cannot_mix_cache_and_adapter(self):
        t = self.torch
        _, LegacyLinear, LegacyBank = self.adapters
        loader = ref.WeightLoader(MemoryStore({'p.weight': t.ones(2, 2)}), 1024, precision='f32')
        with self.assertRaises(RuntimeError):
            LegacyLinear(loader, 'p.weight')
        with self.assertRaises(RuntimeError):
            LegacyBank(loader, 'experts', True, 2)
        for create in (lambda: ref.make_adapter_types('unknown'),
                       lambda: ref.WeightLoader(loader.store, 1024, 'unknown')):
            with self.assertRaises(RuntimeError):
                create()
        self.assertEqual(self.adapters[0].precision_profile, 'bf16_rne')

    @classmethod
    def tearDownClass(cls):
        assert not cls.torch.cuda.is_initialized(), 'Unit tests must not initialize CUDA'


if __name__ == '__main__':
    unittest.main(verbosity=2)
