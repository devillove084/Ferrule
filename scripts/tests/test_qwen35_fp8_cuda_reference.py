#!/usr/bin/env python3
"""CPU-only independent CUDA-oracle contract tests. Never initializes CUDA."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import sys
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.append(str(SCRIPTS))

os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8')
import qwen35_fp8_cuda_reference as ref


class ContractTests(unittest.TestCase):
    def test_reference_keeps_native_f32_and_rope_f32_contract(self):
        source=(SCRIPTS / 'qwen35_fp8_cuda_reference.py').read_text()
        self.assertNotIn('model.to(torch.bfloat16)', source)
        self.assertNotIn('model.to(t.bfloat16)', source)
        self.assertIn("p.dtype == torch.float32", source)
        self.assertIn("model.model.rotary_emb = q.Qwen3_5MoeTextRotaryEmbedding", source)
        self.assertIn("weight='F32 product -> BF16 RNE -> F32'", source)
        self.assertIn("activation='BF16 RNE -> F32 for quantized linear only'", source)

    def test_exact_versions_no_alternative_build(self):
        good = dict(ref.REQUIRED_VERSIONS, safetensors='0.8.0', tokenizers='0.22.2', numpy='2.3.5')
        ref.validate_dependencies(good)
        for name, version in (('torch', '2.13.0'), ('torch', '2.13.0+cu130'),
                              ('transformers', '5.5.3'), ('transformers', None), ('safetensors', None)):
            with self.subTest(name=name, version=version), self.assertRaises(ValueError):
                ref.validate_dependencies(dict(good, **{name: version}))

    def test_no_production_or_cpu_oracle_imports(self):
        tree = ast.parse(Path(ref.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or '']
            else:
                continue
            self.assertFalse(any('ferrule' in n.lower() or 'cutlass' in n.lower()
                                 or n == 'qwen35_fp8_reference' for n in names), names)


    def test_static_preflight_never_queries_cuda(self):
        args = SimpleNamespace(model=Path('/unused'), execute_cuda=False)
        manifest = dict(diagnostics=[])
        config = SimpleNamespace(to_dict=lambda: {})
        torch = SimpleNamespace(cuda=mock.Mock())
        torch.cuda.is_initialized.return_value = False
        with mock.patch.object(ref, 'dependency_versions', return_value={}), \
             mock.patch.object(ref, 'validate_dependencies'), mock.patch.object(ref, 'initialize'), \
             mock.patch.object(ref, 'read_config', return_value=config), mock.patch.object(ref, 'sha256'), \
             mock.patch.object(ref, 'torch', torch, create=True), mock.patch.object(ref, 'gpu_admission') as gpu:
            ref.run(args, manifest)
        gpu.assert_not_called()
        torch.cuda.is_available.assert_not_called()
        self.assertEqual(manifest['preflight'], 'passed')
        self.assertIn('NOT attempted', manifest['diagnostics'][0])

    def test_full_model_failure_propagates_and_preserves_incomplete_provenance(self):
        for failure in ('build', 'capture'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                args = SimpleNamespace(model=root, output=root, execute_cuda=True,
                                       devices='cuda:0', cache_mib=1, case='both')
                manifest = ref.new_manifest(args)
                config = SimpleNamespace(to_dict=lambda: {})
                torch = SimpleNamespace(cuda=mock.Mock())
                torch.cuda.is_initialized.return_value = False
                checkpoint = mock.Mock()
                checkpoint.provenance.return_value = {'used_payloads': {}}
                weights = mock.Mock(device='cuda:0')
                model = SimpleNamespace(model=SimpleNamespace(rotary_emb=object()))
                with mock.patch.object(ref, 'dependency_versions', return_value={}), \
                     mock.patch.object(ref, 'validate_dependencies'), mock.patch.object(ref, 'initialize'), \
                     mock.patch.object(ref, 'read_config', return_value=config), \
                     mock.patch.object(ref, 'sha256', return_value='a'*64), \
                     mock.patch.object(ref, 'torch', torch, create=True), \
                     mock.patch.object(ref, 'gpu_admission', return_value=[dict(device='cuda:0', free=16*ref.GIB)]), \
                     mock.patch.object(ref, 'Checkpoint', return_value=checkpoint), \
                     mock.patch.object(ref, 'DeviceWeights', return_value=weights), \
                     mock.patch.object(ref, 'audit_rope_f32'), \
                     mock.patch.object(ref, 'build_model', return_value=(model, ['cuda:0']),
                                       side_effect=RuntimeError('build failed') if failure == 'build' else None), \
                     mock.patch.object(ref, 'full_case', side_effect=RuntimeError('capture failed')) as capture:
                    with self.assertRaisesRegex(RuntimeError, failure + ' failed'):
                        ref.run(args, manifest)
                self.assertEqual(manifest['status'], 'incomplete')
                self.assertEqual(manifest['reference_kind'], 'incomplete')
                self.assertFalse(manifest['full_reference_complete'])
                self.assertIn(failure + ' failed', manifest['diagnostics'][-1])
                self.assertEqual(manifest['cases'], [])
                self.assertEqual(manifest['source_payloads']['file'], 'source-payloads.json')
                self.assertEqual(json.loads((root/'source-payloads.json').read_text()), {'used_payloads': {}})
                weights.clear.assert_called_once()
                torch.cuda.empty_cache.assert_called_once()
                if failure == 'build':
                    capture.assert_not_called()

    def test_retired_options_are_rejected_before_output_or_execution(self):
        options = (['--mode', 'layers'], ['--frozen-reference', '/unused'],
                   ['--frozen-manifest-sha256', 'a'*64], ['--diagnostic-modules'],
                   ['--diagnostic-max-gib', '2'])
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'not-created'
            for option in options:
                with self.subTest(option=option), mock.patch.object(sys, 'argv',
                        ['reference', '--output', str(output), *option]), \
                     mock.patch.object(ref, 'run') as run, contextlib.redirect_stderr(io.StringIO()) as error:
                    with self.assertRaises(SystemExit) as caught:
                        ref.main()
                    self.assertEqual(caught.exception.code, 2)
                    self.assertIn('unrecognized arguments', error.getvalue())
                    self.assertFalse(output.exists())
                    run.assert_not_called()

    def test_worker_failure_writes_incomplete_manifest_and_nonzero_status(self):
        with tempfile.TemporaryDirectory() as temp:
            argv = ['reference', '--output', temp, '--worker', '--execute-cuda']
            with mock.patch.object(sys, 'argv', argv), mock.patch.dict(os.environ), \
                 mock.patch.object(ref, 'run', side_effect=RuntimeError('capture failed')):
                self.assertEqual(ref.main(), 2)
            manifest = json.loads((Path(temp)/'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'incomplete')
            self.assertEqual(manifest['reference_kind'], 'incomplete')
            self.assertFalse(manifest['full_reference_complete'])
            self.assertIn('RuntimeError: capture failed', manifest['diagnostics'])


class NumericTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global np
        import numpy as np
        ref.initialize()
        cls.t = ref.torch
        cls.t.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        assert not cls.t.cuda.is_initialized()

    def test_all_e4m3_encodings_partial_blocks_numeric_scale_rne(self):
        t = self.t
        bits = t.arange(256, dtype=t.int32).to(t.uint8)
        bits[(bits == 127) | (bits == 255)] = 0
        weight = bits.repeat(129, 2).view(t.float8_e4m3fn)
        scales = t.tensor([[.0311, .0573, .0917, .1257], [.2411, .4013, .0613, .1817]]).bfloat16()
        result = ref.dequantize(weight, scales)
        # Scalar indexed F64 product and integer RNE: independent of expansion and Torch cast.
        w = weight.float().numpy()
        scale = scales.float().numpy()
        products = np.empty(w.shape, dtype=np.float32)
        for r in range(w.shape[0]):
            for c in range(w.shape[1]):
                products[r, c] = float(w[r, c]) * float(scale[r//128, c//128])
        raw = products.view(np.uint32)
        expected = ((raw + np.uint32(0x7fff) + ((raw >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)
        np.testing.assert_array_equal(result.numpy(), expected)
        self.assertFalse(np.array_equal(expected, products))
        for bad in (t.ones(1, 1).bfloat16(), scales.float(), scales*0, -scales, scales*float('nan')):
            with self.assertRaises(ValueError):
                ref.dequantize(weight, bad)
        with self.assertRaisesRegex(ValueError, 'Nonfinite FP8'):
            ref.dequantize(t.full((1, 1), float('nan')).to(t.float8_e4m3fn), t.ones(1, 1).bfloat16())

    def test_rope_never_inherits_checkpoint_bf16_dtype(self):
        t=self.t
        c=ref.q.Qwen3_5MoeTextConfig(hidden_size=2048,num_attention_heads=16,head_dim=256,
            rope_parameters=dict(rope_type='default',rope_theta=10000000.,partial_rotary_factor=.25))
        rope=ref.q.Qwen3_5MoeTextRotaryEmbedding(c,device='cpu')
        self.assertTrue(ref.audit_rope_f32(rope)['native_f32_exact'])
        original=rope.inv_freq.clone()
        self.assertFalse(t.equal(original,original.bfloat16().float()))
        rope.to(t.bfloat16)
        with self.assertRaisesRegex(ValueError,'must stay F32'): ref.audit_rope_f32(rope)
        rope.float()
        with self.assertRaisesRegex(ValueError,'possible blanket BF16'): ref.audit_rope_f32(rope)
        self.assertFalse(t.cuda.is_initialized())

    def test_signed_ties_linear_boundaries_and_cache(self):
        t = self.t
        values = {'q.weight': t.tensor([[1.5, -2.25, .75, 3.5]]).to(t.float8_e4m3fn),
                  'q.weight_scale_inv': t.tensor([[.0311]]).bfloat16(),
                  'plain.weight': t.tensor([[.3127, -.7531, .1317, .9991]])}
        store = SimpleNamespace(read=lambda name: values[name].clone())
        weights = ref.DeviceWeights(store, 'cpu', 16)
        Linear, _ = ref.binding_types()
        x = t.tensor([[1.00390625, 1.01171875, -1.00390625, -1.01171875]])
        rounded = t.tensor([[1., 1.015625, -1., -1.015625]])
        self.assertTrue(t.equal(x.bfloat16().float(), rounded))
        output = Linear(weights, 'q.weight')(x)
        self.assertTrue(t.equal(output, ref.F.linear(rounded, ref.dequantize(values['q.weight'], values['q.weight_scale_inv']))))
        self.assertFalse(t.equal(output, output.bfloat16().float()))
        self.assertIs(type(output), t.Tensor)
        self.assertTrue(t.equal(Linear(weights, 'plain.weight')(x), ref.F.linear(x, values['plain.weight'])))
        self.assertEqual(list(weights.cache), ['plain.weight'])
        self.assertLessEqual(weights.bytes, weights.capacity)
        weights.clear()
        self.assertEqual(weights.bytes, 0)
        with self.assertRaisesRegex(ValueError, 'boundary'):
            Linear(weights, 'q.weight')(x.bfloat16())

    def test_native_tiny_model_binding_and_cached_states_on_cpu(self):
        t = self.t
        q = ref.q
        c = q.Qwen3_5MoeTextConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, head_dim=16, num_experts=4,
            num_experts_per_tok=2, moe_intermediate_size=24, shared_expert_intermediate_size=24,
            linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8,
            linear_value_head_dim=8, linear_conv_kernel_dim=4,
            layer_types=['linear_attention', 'full_attention'], max_position_embeddings=32,
            tie_word_embeddings=False, dtype='float32', rope_parameters=dict(rope_type='default',
            rope_theta=10000., partial_rotary_factor=.5, mrope_interleaved=True, mrope_section=[1, 1, 2]))
        c._attn_implementation = c._experts_implementation = 'eager'
        t.manual_seed(3513)
        native = q.Qwen3_5MoeForCausalLM(c).eval()
        values = {}
        # Synthetic checkpoint at BF16/FP8 boundaries, not real model weights.
        def put(name, tensor, quantized):
            if quantized:
                values[name] = (tensor.detach()*16).clamp(-448, 448).to(t.float8_e4m3fn)
                values[name+'_scale_inv'] = t.full((1, 1), .0625, dtype=t.bfloat16)
            else:
                values[name] = tensor.detach().bfloat16()
        for name, parameter in native.named_parameters():
            source = ref.source_name(name)
            if '.experts.' in name:
                prefix, part = source.rsplit('.', 1)
                for expert in range(c.num_experts):
                    parts = parameter[expert].chunk(2, 0) if part == 'gate_up_proj' else [parameter[expert]]
                    labels = ('gate_proj', 'up_proj') if part == 'gate_up_proj' else ('down_proj',)
                    for label, value in zip(labels, parts):
                        put(f'{prefix}.{expert}.{label}.weight', value, True)
            else:
                owner = native.get_submodule(name.rsplit('.', 1)[0])
                put(source, parameter, isinstance(owner, t.nn.Linear) and name != 'lm_head.weight')
        checkpoint = SimpleNamespace(read=lambda name: values[name].clone())
        weights = ref.DeviceWeights(checkpoint, 'cpu', 1024)
        with t.device('meta'):
            lazy = q.Qwen3_5MoeForCausalLM(c)
        forwards = [type(m).forward for m in lazy.modules() if isinstance(m, (q.Qwen3_5MoeExperts, q.Qwen3_5MoeDecoderLayer))]
        ref.bind(lazy, '', weights, c)
        lazy.model.rotary_emb = q.Qwen3_5MoeTextRotaryEmbedding(c, device='cpu')
        self.assertEqual(forwards, [type(m).forward for m in lazy.modules() if isinstance(m, (q.Qwen3_5MoeExperts, q.Qwen3_5MoeDecoderLayer))])
        # Independent eager parameters and pre-hooks, no tensor tag or lazy cache for experts.
        for name, parameter in list(native.named_parameters()):
            source = ref.source_name(name)
            if '.experts.' in name:
                prefix, part = source.rsplit('.', 1)
                labels = ('gate_proj', 'up_proj') if part == 'gate_up_proj' else ('down_proj',)
                decoded = []
                for expert in range(c.num_experts):
                    decoded.append(t.cat([(values[f'{prefix}.{expert}.{label}.weight'].float()*.0625).bfloat16().float() for label in labels]))
                parameter.data.copy_(t.stack(decoded))
            else:
                raw = values[source]
                parameter.data.copy_((raw.float()*.0625).bfloat16().float() if raw.dtype == t.float8_e4m3fn else raw.float())
                if raw.dtype == t.float8_e4m3fn:
                    native.get_submodule(name.rsplit('.', 1)[0]).register_forward_pre_hook(lambda _m, a: (a[0].bfloat16().float(),))
        # Exercise native expert formula separately; eager experts need operand-only rounding
        # at two calls, so use a local F.linear test spy rather than editing native forward.
        original = ref.F.linear
        expert_storages = {p.untyped_storage().data_ptr() for n, p in native.named_parameters() if '.experts.' in n}
        def eager_linear(x, w, *args, **kwargs):
            if w.untyped_storage().data_ptr() in expert_storages:
                x = x.bfloat16().float()
            return original(x, w, *args, **kwargs)
        class CpuCapture(ref.Capture):
            def put(self, key, value):
                self.tensors[f'{self.stage}.{key}'] = value.detach().cpu().contiguous().clone()
        capture = CpuCapture()
        capture.observe(lazy.model.embed_tokens, 'embedding')
        capture.observe(lazy.model.norm, 'final_norm')
        for i, layer in enumerate(lazy.model.layers):
            capture.layer(layer, i)
        cache_a = cache_b = None
        next_id = None
        for start in (0, 3):
            ids = [4, 7, 9] if start == 0 else [next_id]
            capture.stage = 'prefill' if start == 0 else 'decode.0'
            inputs = t.tensor([ids])
            positions = t.arange(start, start+len(ids))
            kwargs = dict(input_ids=inputs, cache_position=positions, position_ids=positions[None],
                          attention_mask=t.ones(1, start+len(ids), dtype=t.long), use_cache=True, return_dict=True)
            with t.inference_mode():
                a = lazy(**kwargs, past_key_values=cache_a)
                with mock.patch.object(ref.F, 'linear', eager_linear):
                    b = native(**kwargs, past_key_values=cache_b)
            t.testing.assert_close(a.logits, b.logits, atol=0, rtol=0)
            cache_a, cache_b = a.past_key_values, b.past_key_values
            next_id = int(a.logits[0, -1].argmax())
            for key, value in (('logits', a.logits), ('input_ids', inputs), ('position_ids', positions[None]),
                               ('cache_position', positions), ('attention_mask', kwargs['attention_mask'])):
                capture.put(key, value)
            capture.cache(cache_a, range(2))
            for field in ref.CACHE_FIELDS:
                for va, vb in zip(getattr(cache_a, field), getattr(cache_b, field)):
                    if va is not None:
                        self.assertEqual(va.dtype, t.float32)
                        t.testing.assert_close(va, vb, atol=0, rtol=0)
        self.assertEqual(cache_a.get_seq_length(), 4)
        capture.close()

    def test_checkpoint_payload_hash_and_mutation(self):
        from safetensors.torch import save_file
        t = self.t
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root/'model.safetensors'
            save_file({'weight': t.ones(2, 3).bfloat16()}, str(path))
            ref.write_json(root/'model.safetensors.index.json', dict(weight_map={'weight': path.name}))
            store = ref.Checkpoint(root)
            self.assertEqual(store.read('weight').shape, (2, 3))
            self.assertEqual(store.provenance()['used_payloads']['weight']['dtype'], 'torch.bfloat16')
            save_file({'weight': t.zeros(2, 3).bfloat16()}, str(path))
            with self.assertRaisesRegex(ValueError, 'changed'):
                store.read('weight')

    def test_capture_rejects_cpu_and_narrow_boundary_contract(self):
        with self.assertRaisesRegex(ValueError, 'Non-CUDA'):
            ref.Capture().put('hidden', self.t.ones(1))
        self.assertFalse(self.t.cuda.is_initialized())



if __name__ == '__main__':
    unittest.main(verbosity=2)
