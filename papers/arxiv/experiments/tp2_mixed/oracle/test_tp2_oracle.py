"""Synthetic CPU validation only; never opens original checkpoint/model files."""
import ast
import copy
import os
from pathlib import Path
import sys
import unittest

os.environ.update(CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none', ATEN_CPU_CAPABILITY='avx512',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import tp2_oracle as o
import torch as t
import torch.nn.functional as F


class Store:
    def __init__(self, model):
        self.values = {}
        for name, value in model.state_dict().items():
            value = value.detach().bfloat16().clone()
            if name.endswith('.experts.gate_up_proj'):
                for e, expert in enumerate(value):
                    for part, matrix in zip(('gate_proj','up_proj'), expert.chunk(2,0)):
                        self.values[o.native.legacy.source_name(name.removesuffix('gate_up_proj')+f'{e}.{part}.weight')] = matrix
            elif name.endswith('.experts.down_proj'):
                for e, matrix in enumerate(value):
                    self.values[o.native.legacy.source_name(name.removesuffix('down_proj')+f'{e}.down_proj.weight')] = matrix
            else:
                self.values[o.native.legacy.source_name(name)] = value

    def read(self,name):
        return self.values[name].clone()


def tiny():
    c = o.native.tiny_config()
    c.num_key_value_heads = 2
    return c


class OracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        o.initialize()

    def tearDown(self):
        self.assertFalse(t.cuda.is_initialized())

    def test_code_fingerprint_ignores_reference_counts_not_code_changes(self):
        import inspect
        code=inspect.unwrap(o.native.hf.Qwen3_5MoeAttention.forward).__code__
        before=o.code_fingerprint(code)
        refs=list(code.co_consts)
        self.assertEqual(before,o.code_fingerprint(code))
        changed=code.replace(co_consts=code.co_consts+('different constant',))
        self.assertNotEqual(before,o.code_fingerprint(changed))
        self.assertEqual(len(refs),len(code.co_consts))

    def test_imports_are_not_production(self):
        tree = ast.parse(Path(o.__file__).read_text())
        names = [n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        names += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        self.assertFalse(any(n.startswith(('vllm','ferrule')) for n in names))

    def test_column_vocab_order_and_tp1(self):
        c = tiny()
        x = t.arange(32,dtype=t.bfloat16).reshape(2,16)/32
        w = t.arange(32*16,dtype=t.bfloat16).reshape(32,16)/128
        local = [F.linear(x,o.select(w,*o.layout('lm_head.weight',w.shape,c,r,2))) for r in range(2)]
        self.assertTrue(t.equal(t.cat(local,-1), F.linear(x,w)))
        self.assertFalse(t.equal(t.cat(local[::-1],-1),F.linear(x,w)))
        self.assertTrue(t.equal(o.rank_sum([F.linear(x,w)]),F.linear(x,w)))
        with self.assertRaises(ValueError): o.bounds(3,0,2)
        with self.assertRaises(ValueError): o.bounds(4,2,2)
        with self.assertRaises(ValueError): o.rank_sum([x.float(),x.float()])

    def test_row_rounding_witness_rejects_unsharded_or_late_round(self):
        # Each exact partial is halfway above 1 and rounds down; unsplit is
        # halfway above 2 with an odd BF16 mantissa and rounds up.
        x = t.ones(1,4,dtype=t.bfloat16)
        w = t.tensor([[1.,1/256,1.,3/256]],dtype=t.bfloat16)
        parts = [F.linear(x[:,:2],w[:,:2]),F.linear(x[:,2:],w[:,2:])]
        expected = (parts[0].float()+parts[1].float()).bfloat16()
        self.assertTrue(t.equal(o.rank_sum(parts),expected))
        # A cancellation fixture makes local BF16 rounding observable.
        w = t.tensor([[1.,1/256,-1.,0.]],dtype=t.bfloat16)
        parts = [F.linear(x[:,:2],w[:,:2]),F.linear(x[:,2:],w[:,2:])]
        self.assertEqual(float(o.rank_sum(parts)),0.)
        self.assertEqual(float(F.linear(x,w)),1/256)

    def test_gdn_group_order_and_head_repeat(self):
        c = tiny()
        groups = o.gdn_groups(c)
        x = t.arange(sum(groups)).reshape(1,-1,1)
        pieces = [o.select(x,1,o.grouped_ranges(groups,r,2)) for r in range(2)]
        self.assertTrue(t.equal(o.merge_gdn(pieces,c,2),x))
        self.assertFalse(t.equal(t.cat(pieces,1),x))
        q = t.arange(c.linear_num_key_heads)
        global_repeat = q.repeat_interleave(c.linear_num_value_heads//c.linear_num_key_heads)
        local_repeat = t.cat([s.repeat_interleave(c.linear_num_value_heads//c.linear_num_key_heads) for s in q.chunk(2)])
        self.assertTrue(t.equal(global_repeat,local_repeat))

    def test_gate_up_concat_shards_each_half(self):
        c = tiny()
        gate = t.arange(c.moe_intermediate_size*c.hidden_size,dtype=t.float32).reshape(c.moe_intermediate_size,c.hidden_size).bfloat16()
        up = gate+512
        for r in range(2):
            actual = t.cat([o.select(w,*o.layout('model.layers.0.mlp.experts.0.'+part+'.weight',w.shape,c,r,2))
                            for part,w in [('gate_proj',gate),('up_proj',up)]])
            lo,hi = o.bounds(c.moe_intermediate_size,r,2)
            self.assertTrue(t.equal(actual,t.cat([gate[lo:hi],up[lo:hi]])))
            self.assertFalse(t.equal(actual,t.cat([gate,up]).chunk(2)[r]))

    def test_native_expert_order_is_not_slot_order_or_f32_accumulation(self):
        c = tiny(); c.num_experts=3; c.num_experts_per_tok=3
        expert = o.native.hf.Qwen3_5MoeExperts(c).bfloat16()
        expert.act_fn = t.nn.Identity()
        with t.no_grad():
            expert.gate_up_proj.zero_(); expert.down_proj.zero_()
            expert.gate_up_proj[:,0,0]=1
            expert.gate_up_proj[:,c.moe_intermediate_size,0]=1
            expert.down_proj[0,:,0]=256
            expert.down_proj[1,:,0]=1
            expert.down_proj[2,:,0]=-256
        x = t.zeros(1,c.hidden_size,dtype=t.bfloat16); x[:,0]=1
        ids = t.tensor([[0,2,1]])
        out = expert(x,ids,t.ones(1,3,dtype=t.float32))
        self.assertTrue(t.equal(out,t.zeros_like(out)))
        slot_order = ((t.tensor(256,dtype=t.bfloat16)+t.tensor(-256,dtype=t.bfloat16))+t.tensor(1,dtype=t.bfloat16))
        self.assertEqual(float(slot_order),1.)
        self.assertEqual(256.+1.-256.,1.)

    def test_raw_ties_only_not_softmax_collisions(self):
        raw=t.tensor([[4.,4.,1.,-1000.]],dtype=t.bfloat16)
        probs=F.softmax(raw,dim=-1,dtype=t.float32)
        ids=t.tensor([[1,0]]); weights=probs.gather(-1,ids); weights/=weights.sum(-1,keepdim=True)
        selected=o.native.canonicalize_router_ties(raw,(probs,weights,ids))
        self.assertEqual(selected[2].tolist(),[[0,1]])
        self.assertTrue(t.equal(selected[1],weights))
        raw=t.tensor([[0.,-1000.,-1008.,-1016.]],dtype=t.bfloat16)
        probs=F.softmax(raw,dim=-1,dtype=t.float32)
        ids=t.tensor([[0,3]])
        weights=t.tensor([[1.,0.]],dtype=t.float32)
        self.assertTrue(t.equal(o.native.canonicalize_router_ties(raw,(probs,weights,ids))[2],ids))

    def test_combine_must_add_locally_before_reduce(self):
        shared=[t.tensor([256.],dtype=t.bfloat16),t.tensor([-256.],dtype=t.bfloat16)]
        routed=[t.tensor([1.],dtype=t.bfloat16),t.tensor([0.],dtype=t.bfloat16)]
        good=o.rank_sum([s+r for s,r in zip(shared,routed)])
        bad=o.rank_sum(shared)+o.rank_sum(routed)
        self.assertEqual(float(good),0.); self.assertEqual(float(bad),1.)

    def test_original_fp8_scales_global_128_coordinates_and_checksums(self):
        c=tiny(); c.moe_intermediate_size=258; c.hidden_size=130
        raw=t.arange(258*130,dtype=t.int64).remainder(120).to(t.uint8).reshape(258,130).view(t.float8_e4m3fn)
        scale=t.tensor([[.1,.2],[.3,.4],[.5,.6]],dtype=t.bfloat16)
        class Encoded:
            def read(self,name):return scale.clone() if name.endswith('_scale_inv') else raw.clone()
        operands=o.Operands(Encoded(),c,'cpu')
        name='model.layers.0.mlp.experts.0.gate_proj.weight'
        for rank in range(2):
            got=operands.get(name,rank,matrix=True)
            lo,hi=o.bounds(258,rank)
            expected=t.empty_like(got)
            for row in range(hi-lo):
                for col in range(130):
                    expected[row,col]=float(raw[row+lo,col])*float(scale[(row+lo)//128,col//128])
            self.assertTrue(t.equal(got,expected))
            entry=operands.audit[f'rank.{rank}.{name}']
            self.assertEqual(entry['decoded_bf16_sha256'],o.digest_tensor(expected))
        self.assertIn((1,0),operands.audit['rank.1.'+name]['global_128_block_coordinates'])

    def test_tp1_matches_native_prefill_cached_logits_and_cache(self):
        t.manual_seed(3501635); c=tiny()
        native=o.native.hf.Qwen3_5MoeForCausalLM(c).eval()
        store=Store(native)
        # Parameter-only casting, NOT model.bfloat16() (RoPE stays F32).
        for p in native.parameters(): p.data=p.data.bfloat16()
        handles=[layer.mlp.gate.register_forward_hook(
            lambda m,a,out:o.native.router_observation(m,a,out,o.native.ROUTER_PROFILE)[1]) for layer in native.model.layers]
        capture=o.Capture(); operands=o.Operands(store,c,'cpu',world=1)
        sharded=o.build_model(c,operands,capture)
        try:
            o.run_case(sharded,c,operands,capture,[1,3,2],decode_token=5)
            cache=o.native.f32_ssm_cache(c,['cpu']*2)
            with t.inference_mode():
                for stage,ids,start in [('prefill',[1,3,2],0),('decode.0',[5],3)]:
                    logits=native(t.tensor([ids]),position_ids=t.arange(start,start+len(ids))[None],
                                  attention_mask=t.ones(1,start+len(ids),dtype=t.int64),past_key_values=cache,use_cache=True).logits
                    self.assertTrue(t.equal(logits,capture.tensors[stage+'.logits']))
                    for i,layer in enumerate(cache.layers):
                        for source,key in [('conv_states','conv_states'),('recurrent_states','recurrent_states'),('keys','key_cache'),('values','value_cache')]:
                            value=getattr(layer,source,None)
                            if value is not None:self.assertTrue(t.equal(value,capture.tensors[f'{stage}.cache.{key}.{i:02d}']))
        finally:
            capture.close()
            for handle in handles:handle.remove()

    def test_tp2_native_local_modules_full_inventory_both_stages(self):
        t.manual_seed(3512); c=tiny()
        original=o.native.hf.Qwen3_5MoeForCausalLM(c).eval()
        capture=o.Capture(); operands=o.Operands(Store(original),c,'cpu')
        model=o.build_model(c,operands,capture)
        try:
            calls=o.run_case(model,c,operands,capture,[1,3,2],decode_token=5)
            self.assertEqual(len(calls),2)
            self.assertEqual(set(capture.tensors),set(o.expected_tensors(c,[1,3,2])))
            for stage in ('prefill','decode.0'):
                for i in range(2):
                    partials=[capture.tensors[f'{stage}.rank.{r}.layers.{i:02d}.moe_partial'] for r in range(2)]
                    self.assertTrue(t.equal(o.rank_sum(partials),capture.tensors[f'{stage}.layers.{i:02d}.moe']))
                cache=capture.tensors[stage+'.cache.recurrent_states.00']
                self.assertEqual(cache.dtype,t.float32)
                self.assertTrue(t.equal(cache,t.cat([capture.tensors[f'{stage}.rank.{r}.cache.recurrent_states.00'] for r in range(2)],1)))
        finally:capture.close()

    def test_tolerance_is_unchanged_and_ids_are_exact(self):
        a=t.tensor([1.],dtype=t.bfloat16)
        self.assertEqual(o.tensor_metric(a,a)['atol'],.00025)
        self.assertEqual(o.tensor_metric(a,a)['rtol'],.012)
        self.assertEqual(o.tensor_metric(t.tensor([2]),t.tensor([1]),exact=True)['failures'],1)
        with self.assertRaises(ValueError):o.tensor_metric(t.tensor([float('nan')]),t.tensor([1.]))

    def test_capture_bound_and_unknown_operand_fail_closed(self):
        with self.assertRaises(ValueError):o.Capture(max_bytes=1).put('logits',t.ones(1,dtype=t.bfloat16))
        with self.assertRaises(ValueError):o.layout('model.unknown.weight',(2,2),tiny(),0,2)


if __name__=='__main__':unittest.main()
