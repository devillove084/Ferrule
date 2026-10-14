"""CPU-only native BF16 reference contracts. Never runs the whole model/GPU."""
import copy
import json
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

os.environ.update(CUDA_VISIBLE_DEVICES='', ATEN_CPU_CAPABILITY='avx512', CUBLAS_WORKSPACE_CONFIG=':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import qwen35_fp8_bf16_reference as ref


class NativeBf16Reference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ref.initialize()
        cls.t = ref.torch

    def tearDown(self):
        self.assertFalse(self.t.cuda.is_initialized())

    def test_loading_experiment_matches_frozen_independent_evidence(self):
        actual = ref.loading_evidence()
        fixture = Path(__file__).resolve().parents[2]/'crates/ferrule-model/tests/hybrid_bf16/hf_loading.json'
        self.assertEqual(actual,json.loads(fixture.read_text()))
        self.assertEqual(actual['cache_runs']['hf_default'][1]['ssm'],'torch.bfloat16')
        self.assertEqual(actual['cache_runs']['explicit_f32_ssm'][1]['ssm'],'torch.float32')

    def test_profile_cannot_mutate_or_alias_legacy_w8_f32(self):
        self.assertNotEqual(ref.SCHEMA, ref.legacy.SCHEMA)
        self.assertEqual(ref.legacy.CONTRACT['boundary'],'F32')
        self.assertEqual(ref.legacy.REQUIRED_VERSIONS['transformers'],'5.2.0')
        self.assertEqual(ref.CONTRACT['model_boundary'],'bf16')
        with self.assertRaises(TypeError): ref.CONTRACT['ssm']='BF16'
        with mock.patch.object(ref,'SOURCE_PINS',dict(ref.SOURCE_PINS,cache_utils='0'*64)):
            with self.assertRaisesRegex(ValueError,'Pinned HF source'): ref.initialize()

    def test_raw_partial_blocks_and_bounded_bf16_weight_cache(self):
        t=self.t
        raw=t.arange(129*136,dtype=t.int64).remainder(126).to(t.uint8).reshape(129,136).view(t.float8_e4m3fn)
        scales=t.tensor([[.123,.321],[.234,.432]],dtype=t.bfloat16)
        output=ref.dequantize_bf16(raw,scales)
        self.assertEqual(output.dtype,t.bfloat16)
        # Scalar coordinate mapping, independent of the exporter tile/grid path.
        expected=t.empty_like(output)
        for r in range(129):
            for c in range(136): expected[r,c]=float(raw[r,c])*float(scales[r//128,c//128])
        self.assertTrue(t.equal(output,expected))
        class Store:
            def read(self,name): return scales.clone() if name.endswith('_scale_inv') else raw.clone()
        loader=ref.DeviceWeights(Store(),'cpu',output.numel()*2)
        a=loader.matrix('a')
        self.assertTrue(t.equal(a,output))
        self.assertEqual(loader.matrix('a').data_ptr(),a.data_ptr())
        loader.matrix('b')
        self.assertEqual(list(loader.cache),['b'])
        self.assertEqual(loader.peak_bytes,output.numel()*2)
        loader.clear()
        self.assertEqual(loader.bytes,0)
        with self.assertRaisesRegex(ValueError,'paired BF16'): ref.dequantize_bf16(raw,scales.float())

    def test_lazy_bind_uses_native_bf16_forward_and_preserves_f32_rope(self):
        t=self.t
        t.manual_seed(3501635)
        c=ref.tiny_config()
        source=ref.hf.Qwen3_5MoeForCausalLM(c).eval()
        tensors={}
        eager_state={}
        def put(name,value,numeric):
            v=value.detach().bfloat16()
            if numeric:
                scale=(v.float().abs().max()/300).clamp_min(1e-8).bfloat16().reshape(1,1)
                raw=(v.float()/scale.float()).to(t.float8_e4m3fn)
                tensors[name]=raw
                tensors[name+'_scale_inv']=scale
                return ref.dequantize_bf16(raw,scale)
            tensors[name]=v.float() if name.endswith(('.A_log','.linear_attn.norm.weight')) else v
            return v
        for name,value in source.state_dict().items():
            external=ref.legacy.source_name(name)
            if '.experts.gate_up_proj' in name:
                experts=[]
                for i in range(c.num_experts):
                    parts=[]
                    for part,v in zip(['gate_proj','up_proj'],value[i].chunk(2,dim=0)):
                        parts.append(put(external.removesuffix('gate_up_proj')+f'{i}.{part}.weight',v,True))
                    experts.append(t.cat(parts))
                eager_state[name]=t.stack(experts)
            elif '.experts.down_proj' in name:
                eager_state[name]=t.stack([put(external.removesuffix('down_proj')+f'{i}.down_proj.weight',value[i],True) for i in range(c.num_experts)])
            else:
                numeric=isinstance(source.get_submodule(name.rsplit('.',1)[0]),t.nn.Linear) and name!='lm_head.weight'
                eager_state[name]=put(external,value,numeric)
        eager=copy.deepcopy(source)
        # Assign parameter-only BF16 state; never cast model buffers.
        eager.load_state_dict(eager_state,assign=True)
        class Store:
            def read(self,name): return tensors[name].clone()
        loader=ref.DeviceWeights(Store(),'cpu',2048)
        with t.device('meta'): lazy=ref.hf.Qwen3_5MoeForCausalLM(c).eval()
        ref.bind(lazy,'',loader,c)
        lazy.model.rotary_emb=ref.hf.Qwen3_5MoeTextRotaryEmbedding(c,device='cpu')
        self.assertTrue(ref.legacy.audit_rope_f32(lazy.model.rotary_emb)['native_f32_exact'])
        caches=[ref.f32_ssm_cache(c,['cpu','cpu'],batch=8) for _ in range(2)]
        with t.inference_mode():
            for tokens in ([1,3,2],[5]):
                outputs=[m(t.tensor([tokens]*8),past_key_values=cache,use_cache=True).logits for m,cache in zip([eager,lazy],caches)]
                self.assertEqual(outputs[1].dtype,t.bfloat16)
                self.assertTrue(t.equal(*outputs))
                for attr in ['conv_states','recurrent_states']:
                    self.assertTrue(t.equal(getattr(caches[0].layers[0],attr),getattr(caches[1].layers[0],attr)))
        self.assertLessEqual(loader.peak_bytes,loader.capacity)

    def test_router_policy_only_equal_raw_logits_preserves_native_weights(self):
        t = self.t
        for raw in [t.tensor([[-.875, -.10009765625, -.875, -.3046875]], dtype=t.bfloat16),
                    t.tensor([[0., -1000., -992., -984.]], dtype=t.bfloat16),
                    t.tensor([[0., 1.e-8, 2.e-8, 3.e-8]], dtype=t.float32)]:
            p = raw.softmax(-1, dtype=t.float32)
            weights, ids = p.topk(3, dim=-1)
            weights = weights / weights.sum(-1, keepdim=True)
            out = (p, weights, ids)
            result = ref.canonicalize_router_ties(raw, out)
            self.assertIs(result[0], p)
            self.assertIs(result[1], weights)
            self.assertTrue(t.equal(raw.gather(-1, ids), raw.gather(-1, result[2])))
            if len(set(raw[0].tolist())) == 4:
                self.assertLess(len(set(p[0].tolist())), 4, 'negative witness must actually collide in F32 softmax')
                self.assertIs(result, out, 'softmax collisions are not raw ties')
            else:
                self.assertEqual(result[2][0].tolist(), [1, 3, 0])
        self.assertNotEqual(ref.SCHEMA, ref.PROFILE_SCHEMA)
        self.assertTrue(ref.router_contract(ref.ROUTER_NATIVE)['unmodified_native_router'])
        self.assertEqual(ref.router_contract(ref.ROUTER_PROFILE)['canonicalization'], 'equal_raw_logits_only')

    def test_native_router_hook_is_observational_and_profile_hook_is_explicit(self):
        t = self.t
        c = ref.tiny_config()
        router = ref.hf.Qwen3_5MoeTopKRouter(c).bfloat16()
        x = t.zeros(8, 16, dtype=t.bfloat16)
        with t.no_grad():
            out = router(x)
            raw, native = ref.router_observation(router, (x,), out, ref.ROUTER_NATIVE)
            _, profile = ref.router_observation(router, (x,), out, ref.ROUTER_PROFILE)
        self.assertIs(native, out)
        self.assertEqual(raw.dtype, t.bfloat16)
        self.assertEqual(profile[2][0].tolist(), [0, 1, 2])
        self.assertIs(profile[1], out[1])
        with self.assertRaisesRegex(ValueError, 'probability'):
            ref.router_observation(router, (x,), (out[0]+.01, out[1], out[2]), ref.ROUTER_PROFILE)

    def test_full40_expected_tensors_has_typed_state_and_both_route_identities(self):
        c = ref.tiny_config().to_dict()
        expected = ref.expected_tensors(c, [1, 3, 2])
        for stage in ('prefill', 'decode.0'):
            self.assertEqual(expected[stage+'.logits'][1], 'BF16')
            self.assertEqual(expected[stage+'.cache.conv_states.00'][1], 'BF16')
            self.assertEqual(expected[stage+'.cache.recurrent_states.00'][1], 'F32')
            self.assertEqual(expected[stage+'.cache.key_cache.01'][1], 'BF16')
            self.assertEqual(expected[stage+'.layers.01.raw_router_logits'][1], 'BF16')
            self.assertEqual(expected[stage+'.layers.01.router_probabilities'][1], 'F32')
            self.assertEqual(expected[stage+'.layers.01.native_expert_ids'][1], 'I64')
            self.assertEqual(expected[stage+'.layers.01.selected_expert_ids'][1], 'I64')

if __name__=='__main__': unittest.main()
