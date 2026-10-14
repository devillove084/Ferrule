"""Independent native HF tiny hybrid; explicit BF16 history/F32 SSM cache policy.
No Ferrule math or replacement model forward. CPU only; eight identical batches
avoid the documented BF16 CPU rsqrt scalar tail. Eager attention and experts.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['ATEN_CPU_CAPABILITY'] = 'avx512'
import copy
import sys
NUMERIC = "--numeric" in sys.argv
PROFILE_TIES = "--profile-ties" in sys.argv
assert not PROFILE_TIES or NUMERIC, "profile tie evidence is numeric-only"
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import torch
import transformers
from transformers import DynamicCache
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf

torch.set_num_threads(1)
torch.manual_seed(3501635)
config = hf.Qwen3_5MoeTextConfig(
    vocab_size=32, hidden_size=16, num_hidden_layers=2,
    num_attention_heads=2, num_key_value_heads=1, head_dim=8,
    layer_types=['linear_attention', 'full_attention'],
    linear_num_key_heads=2, linear_num_value_heads=4,
    linear_key_head_dim=4, linear_value_head_dim=4, linear_conv_kernel_dim=4,
    moe_intermediate_size=16, shared_expert_intermediate_size=16,
    num_experts=4, num_experts_per_tok=3, rms_norm_eps=1e-6,
    rope_parameters={'rope_type': 'default', 'rope_theta': 10000.0,
                     'partial_rotary_factor': 0.5, 'mrope_section': [1, 1, 0]},
    tie_word_embeddings=False,
)
with tempfile.TemporaryDirectory(prefix='ferrule-native-hybrid-bf16-') as directory:
    source = hf.Qwen3_5MoeForCausalLM(config).eval()
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.12)
        source.model.layers[0].linear_attn.norm.weight.fill_(1.0)
    source.save_pretrained(directory)
    model = hf.Qwen3_5MoeForCausalLM.from_pretrained(directory, dtype=torch.bfloat16,
        attn_implementation='eager', experts_implementation='eager').eval()
assert {p.dtype for p in model.parameters()} == {torch.bfloat16}
assert model.model.rotary_emb.inv_freq.dtype == torch.float32
assert model.config._attn_implementation == model.config._experts_implementation == 'eager'
gdn = model.model.layers[0].linear_attn
assert gdn.causal_conv1d_fn is None
assert gdn.chunk_gated_delta_rule is hf.torch_chunk_gated_delta_rule
assert gdn.recurrent_gated_delta_rule is hf.torch_recurrent_gated_delta_rule
BATCH = 8
probe = torch.full((32,), 0.00811767578125, dtype=torch.bfloat16)
assert torch.equal(probe.rsqrt(), probe.float().rsqrt().bfloat16())

def values(t): return t.detach().float().reshape(-1).tolist()
def tensor(t): return {'shape': list(t.shape), 'values': values(t)}
tensors = {}
def put(name, value, role, layer=None, expert=None):
    numeric = NUMERIC and value.ndim == 2 and role not in {
        'TokenEmbedding', 'OutputHead', 'RouterLogits', 'SharedExpertOutputGate'}
    raw = None
    if numeric:
        # Oracle-only decoding: serialize original FP8/scales, then load exactly
        # their BF16 decoded words into the unmodified native HF model.
        raw = torch.empty_like(value, dtype=torch.float8_e4m3fn)
        scales = torch.empty(((value.shape[0]+127)//128, (value.shape[1]+127)//128), dtype=torch.bfloat16)
        with torch.no_grad():
            for row in range(0, value.shape[0], 128):
                for col in range(0, value.shape[1], 128):
                    block = value[row:row+128, col:col+128]
                    scale = (block.float().abs().max() / 300).clamp_min(1e-8).bfloat16()
                    encoded = (block.float()/scale.float()).to(torch.float8_e4m3fn)
                    raw[row:row+128, col:col+128] = encoded
                    scales[row//128, col//128] = scale
                    block.copy_((encoded.float()*scale.float()).bfloat16())
    tensors[name] = dict(tensor(value), role=role, layer=layer, expert=expert)
    if NUMERIC and role in {'LinearAttentionALog', 'LinearAttentionNorm'}:
        original = value.detach().float() * (1 + 1/8192)
        assert torch.equal(original.bfloat16(), value)
        tensors[name].update(source_dtype='float32', source_values=values(original))
    if numeric:
        tensors[name].update(raw_fp8=raw.view(torch.uint8).reshape(-1).tolist(),
            scale_words=scales.view(torch.int16).reshape(-1).tolist(), scale_shape=list(scales.shape))
put('token_embedding.weight', model.model.embed_tokens.weight, 'TokenEmbedding')
put('final_norm.weight', model.model.norm.weight, 'OutputNorm')
put('output.weight', model.lm_head.weight, 'OutputHead')
for i, layer in enumerate(model.model.layers):
    def p(name, value, role, expert=None): put(f'layers.{i}.{name}', value, role, i, expert)
    p('input_norm.weight', layer.input_layernorm.weight, 'AttentionNorm')
    p('post_attention_norm.weight', layer.post_attention_layernorm.weight, 'FeedForwardNorm')
    if i == 0:
        a = layer.linear_attn
        for field, name, role in [('in_proj_qkv','qkv','LinearAttentionQkv'), ('in_proj_z','z','LinearAttentionZ'),
            ('in_proj_b','beta','LinearAttentionBeta'), ('in_proj_a','a','LinearAttentionA'),
            ('out_proj','output','AttentionOutput'), ('norm','norm','LinearAttentionNorm')]:
            p(f'attention.{name}.weight', getattr(a,field).weight, role)
        p('attention.conv.weight', a.conv1d.weight, 'LinearAttentionConv')
        p('attention.a_log.weight', a.A_log, 'LinearAttentionALog')
        p('attention.dt_bias.weight', a.dt_bias, 'LinearAttentionDtBias')
    else:
        a = layer.self_attn
        for field,name,role in [('q_proj','query','AttentionQuery'),('k_proj','key','AttentionKey'),
            ('v_proj','value','AttentionValue'),('o_proj','output','AttentionOutput'),
            ('q_norm','query_norm','AttentionQueryNorm'),('k_norm','key_norm','AttentionKeyNorm')]:
            p(f'attention.{name}.weight', getattr(a,field).weight, role)
    m = layer.mlp
    p('feed_forward.router.weight', m.gate.weight, 'RouterLogits')
    p('feed_forward.shared_gate.weight', m.shared_expert_gate.weight, 'SharedExpertOutputGate')
    for field,name,role in [('gate_proj','gate','SharedExpertGate'),('up_proj','up','SharedExpertUp'),('down_proj','down','SharedExpertDown')]:
        p(f'feed_forward.shared.{name}.weight',getattr(m.shared_expert,field).weight,role)
    for e in range(4):
        gate,up = m.experts.gate_up_proj[e].chunk(2,dim=0)
        for name,value,role in [('gate',gate,'RoutedExpertGate'),('up',up,'RoutedExpertUp'),('down',m.experts.down_proj[e],'RoutedExpertDown')]:
            p(f'feed_forward.experts.{e}.{name}.weight',value,role,e)

residuals = []
routes = []
handles = []
router_ties = []
if PROFILE_TIES:
    from hf_router_ties import canonicalize_router_ties
    for index, layer in enumerate(model.model.layers):
        def canonicalize(module, args, out, index=index):
            raw = torch.nn.functional.linear(args[0].reshape(-1, module.hidden_dim), module.weight)
            assert torch.equal(raw.softmax(-1, dtype=torch.float32), out[0])
            result = canonicalize_router_ties(raw, out)
            if not torch.equal(out[2], result[2]):
                router_ties.append({'layer': index, 'input': values(args[0][:1]),
                    'raw_logits': values(raw[:1]), 'probabilities': values(out[0][:1]),
                    'native_ids': out[2][:1].tolist(),
                    'profile_ids': result[2][:1].tolist(),
                    'weights': values(out[1][:1])})
            return result
        handles.append(layer.mlp.gate.register_forward_hook(canonicalize))
for i, layer in enumerate(model.model.layers):
    handles.append(layer.post_attention_layernorm.register_forward_pre_hook(
        lambda module,args: residuals.append(values(args[0][:1]))))
    handles.append(layer.register_forward_hook(
        lambda module,args,out: residuals.append(values(out[:1]))))
    handles.append(layer.mlp.gate.register_forward_hook(
        lambda module,args,out: routes.append({'weights': values(out[1][:out[1].shape[0]//BATCH]),
            'ids': out[2][:out[2].shape[0]//BATCH].tolist()})))

def cache():
    result = DynamicCache(config=model.config)
    # Storage policy only: HF's default LinearAttentionLayer lazily allocates SSM
    # in conv dtype. Preallocate F32 SSM explicitly before native update methods;
    # no HF compute function is changed and history remains native BF16.
    for i, kind in enumerate(model.config.layer_types):
        if kind == 'linear_attention':
            result.layers[i].recurrent_states = torch.zeros(BATCH,4,4,4,dtype=torch.float32)
            result.layers[i].is_recurrent_states_initialized = True
    return result

@torch.no_grad()
def run(name, tokens, state):
    residuals.clear(); routes.clear(); router_ties.clear()
    out = model(torch.tensor([tokens]*BATCH), past_key_values=state, use_cache=True)
    gdn, gqa = out.past_key_values.layers
    assert gdn.conv_states.dtype == torch.bfloat16
    assert gdn.recurrent_states.dtype == torch.float32
    assert gqa.keys.dtype == gqa.values.dtype == torch.bfloat16
    assert out.logits.dtype == torch.bfloat16
    return {'name':name,'tokens':tokens,'logits':out.logits[0].float().tolist(),
        **({'router_ties':copy.deepcopy(router_ties)} if PROFILE_TIES else {}),
        'residuals':copy.deepcopy(residuals),'routes':copy.deepcopy(routes),
        'conv':values(gdn.conv_states[:1]),'ssm':values(gdn.recurrent_states[:1]),
        'key':values(gqa.keys[:1].transpose(1,2)), 'value':values(gqa.values[:1].transpose(1,2))}
state = cache()
cases = [run('prefill',[1,3,2],state)]
fork = copy.deepcopy(state)
cases.append(run('decode',[5],state))
cases.append(run('continuation',[7],state))
cases.append(run('fork',[9],fork))
cases.append(run('full',[1,3,2,5,7],cache()))
for handle in handles: handle.remove()
record = {'schema':'ferrule.native-hybrid-bf16.v1','torch':torch.__version__,
    'transformers':transformers.__version__,'device':'cpu','batch_replicas':BATCH,
    'hf_source_sha256':hashlib.sha256(Path(inspect.getfile(hf)).read_bytes()).hexdigest(),
    'ssm_policy':'explicit F32 cache preallocation; native HF update/forward unchanged',
    'parameter_dtype':'bfloat16','inv_freq_dtype':'float32','conv_dtype':'bfloat16','ssm_dtype':'float32',
    'tensors':tensors,'cases':cases}
if NUMERIC:
    record.update(schema='ferrule.numeric-fp8-model-bf16.v1',
        profile='cuda-hybrid-numeric-fp8-model-bf16-f32-ssm',
        numeric_weight_dtype='float8_e4m3fn', scale_dtype='bfloat16',
        accumulator_dtype='float32', activation_dtype='bfloat16', kv_dtype='bfloat16',
        router_weight_dtype='float32', logits_dtype='bfloat16',
        decode_policy='raw E4M3FN times original BF16 scale in F32 then BF16 RNE; HF native forward')
path = Path(__file__).with_name('hf_numeric_model.json') if NUMERIC else Path(__file__).with_suffix('.json')
if PROFILE_TIES:
    # Keep the pinned, unmodified native HF oracle immutable. Only an explicit
    # supplemental reference may choose the profile's documented top-k tie order.
    native = json.loads(path.read_text())
    assert record['tensors'] == native['tensors']
    for key in ('torch', 'transformers', 'hf_source_sha256'):
        assert record[key] == native[key], f'pinned oracle mismatch: {key}'
    record.pop('tensors')
    record.update(native_oracle_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        router_tie_policy='equal raw logits: ascending expert ID; native F32 probabilities/weights unchanged',
        oracle_contract='native HF forward with explicit router tie canonicalization; not unmodified HF',
        router_hook_sha256=hashlib.sha256(Path(__file__).with_name('hf_router_ties.py').read_bytes()).hexdigest())
    path = path.with_name('hf_numeric_profile.json')
path.write_text(json.dumps(record,separators=(',',':'))+'\n')
print('native HF complete GDN+gated GQA+shared+routed BF16:', [c['name'] for c in cases])
