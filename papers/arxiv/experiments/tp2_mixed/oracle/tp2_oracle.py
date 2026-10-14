#!/opt/conda310/bin/python
"""Pinned HF forwards on independently partitioned logical TP ranks.

Only the paper launcher may request a CUDA export. Importing this module does
not read model weights, initialize CUDA, or import production implementations.
"""
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT / 'scripts'))
import qwen35_fp8_bf16_reference as native

SCHEMA = 'ferrule.paper.tp2-mixed-independent-oracle.v1'
PROFILE = 'SystemMixedTP2EP1'
PROMPTS = {'hello': [9419], 'capital': [760, 6511, 314, 9338, 369]}
ATOL, RTOL = .00025, .012
require, sha256 = native.require, native.sha256


def initialize():
    source = native.initialize()
    require(torch.version.cuda == '12.9', 'Wrong CUDA build')
    return source


def digest_tensor(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def bounds(size, rank, world=2):
    require(world in (1, 2) and 0 <= rank < world and size > 0 and size % world == 0,
            'Invalid/divisibility-unsupported TP geometry')
    return rank * (size // world), (rank + 1) * (size // world)


def rank_sum(parts):
    require(len(parts) in (1, 2), 'Only TP1/TP2 admitted')
    require(all(p.dtype == torch.bfloat16 and p.shape == parts[0].shape and p.device == parts[0].device for p in parts),
            'Reduction must consume like-shaped local BF16 partials')
    if len(parts) == 1:
        return parts[0]
    return (parts[0].float() + parts[1].float()).bfloat16()


def grouped_ranges(groups, rank, world):
    """Shard EACH semantic segment, never split the concatenated QKV in half."""
    result, offset = [], 0
    for size in groups:
        lo, hi = bounds(size, rank, world)
        result.append((offset + lo, offset + hi))
        offset += size
    return result


def gdn_groups(c):
    k = c.linear_num_key_heads * c.linear_key_head_dim
    return (k, k, c.linear_num_value_heads * c.linear_value_head_dim)


def local_config(config, world):
    c = copy.deepcopy(config)
    for name in ('num_attention_heads', 'num_key_value_heads', 'linear_num_key_heads',
                 'linear_num_value_heads', 'moe_intermediate_size', 'shared_expert_intermediate_size'):
        value = getattr(c, name)
        bounds(value, 0, world)
        setattr(c, name, value // world)
    require(config.num_attention_heads % config.num_key_value_heads == 0 and
            config.linear_num_value_heads % config.linear_num_key_heads == 0, 'Unsupported head grouping')
    require(not config.attention_bias and config.hidden_act == 'silu', 'Unsupported bias/activation')
    bounds(config.vocab_size, 0, world)
    c._attn_implementation = c._experts_implementation = 'eager'
    return c


def layout(name, shape, config, rank, world):
    """Config-derived checkpoint coordinates; no production ParameterLayout."""
    if name.endswith(('.in_proj_qkv.weight', '.conv1d.weight')):
        groups = gdn_groups(config)
        require(shape[0] == sum(groups), 'GDN QKV/conv source geometry')
        return 0, grouped_ranges(groups, rank, world)
    rows = {
        '.q_proj.weight': config.num_attention_heads * config.head_dim * 2,
        '.k_proj.weight': config.num_key_value_heads * config.head_dim,
        '.v_proj.weight': config.num_key_value_heads * config.head_dim,
        '.in_proj_z.weight': config.linear_num_value_heads * config.linear_value_head_dim,
        '.in_proj_a.weight': config.linear_num_value_heads,
        '.in_proj_b.weight': config.linear_num_value_heads,
        '.A_log': config.linear_num_value_heads, '.dt_bias': config.linear_num_value_heads,
    }
    for suffix, size in rows.items():
        if name.endswith(suffix):
            require(shape[0] == size, 'Source head geometry: ' + name)
            return 0, [bounds(size, rank, world)]
    if name == 'lm_head.weight':
        require(tuple(shape) == (config.vocab_size, config.hidden_size), 'Vocabulary geometry')
        return 0, [bounds(shape[0], rank, world)]
    if name.endswith(('.gate_proj.weight', '.up_proj.weight')):
        size = config.shared_expert_intermediate_size if '.shared_expert.' in name else config.moe_intermediate_size
        require(shape[0] == size, 'MLP intermediate source geometry')
        return 0, [bounds(size, rank, world)]
    if name.endswith(('.down_proj.weight', '.o_proj.weight', '.out_proj.weight')):
        if name.endswith('.down_proj.weight'):
            size = config.shared_expert_intermediate_size if '.shared_expert.' in name else config.moe_intermediate_size
        elif name.endswith('.o_proj.weight'):
            size = config.num_attention_heads * config.head_dim
        else:
            size = config.linear_num_value_heads * config.linear_value_head_dim
        require(shape[1] == size, 'Row source geometry')
        return 1, [bounds(size, rank, world)]
    require(name.endswith(('.gate.weight', '.shared_expert_gate.weight', '.norm.weight',
                           '.q_norm.weight', '.k_norm.weight', '.input_layernorm.weight',
                           '.post_attention_layernorm.weight', '.embed_tokens.weight')),
            'Unclassified operand: ' + name)
    return None, []


def select(value, axis, ranges):
    if axis is None:
        return value
    return torch.cat([value.narrow(axis, lo, hi - lo) for lo, hi in ranges], dim=axis).contiguous()


class Operands:
    """Lazy original host checkpoint; every expert remains logically available.

    The existing independent storage reader hashes raw bytes. Decode before
    selecting global coordinates, so partial 128 blocks never get re-based.
    No runtime cache/offload/performance claim is attached to this strategy.
    """
    def __init__(self, checkpoint, config, device, world=2, cache_bytes=256*1024**2):
        self.checkpoint, self.config, self.world = checkpoint, config, world
        self.device = torch.device(device)
        self.decoded = native.DeviceWeights(checkpoint, self.device, cache_bytes)
        self.audit = {}

    def get(self, name, rank, matrix=False):
        external = native.legacy.source_name(name)
        if matrix:
            value = self.decoded.matrix(external)
        else:
            raw = self.checkpoint.read(external)
            allowed_f32 = name.endswith(('.linear_attn.A_log', '.linear_attn.norm.weight'))
            require(raw.dtype == torch.bfloat16 or (allowed_f32 and raw.dtype == torch.float32),
                    'Non-native parameter dtype: ' + name)
            value = raw.to(device=self.device, dtype=torch.bfloat16)
        axis, ranges = layout(name, value.shape, self.config, rank, self.world)
        output = select(value, axis, ranges)
        require(bool(torch.isfinite(output).all()), 'Nonfinite operand: ' + name)
        key = f'rank.{rank}.{name}'
        if key not in self.audit:
            coords = []
            # Bind global tile coordinates to a hash independently of slicing.
            if value.ndim == 2:
                rows = ranges if axis == 0 else [(0, value.shape[0])]
                cols = ranges if axis == 1 else [(0, value.shape[1])]
                coords = sorted({(r // 128, c // 128)
                    for r0, r1 in rows for c0, c1 in cols
                    for r in range((r0 // 128)*128, r1, 128)
                    for c in range((c0 // 128)*128, c1, 128)})
            self.audit[key] = dict(source=external, source_shape=list(value.shape), axis=axis,
                ranges=ranges, shape=list(output.shape), decoded_bf16_sha256=digest_tensor(output),
                global_128_block_coordinates=coords,
                block_coordinates_sha256=hashlib.sha256(json.dumps(coords).encode()).hexdigest())
        return output


class Linear(nn.Module):
    def __init__(self, name, operands, rank, shape):
        super().__init__()
        self.name, self.operands, self.rank, self.shape = name, operands, rank, tuple(shape)

    def forward(self, x):
        require(x.dtype == torch.bfloat16, 'Linear input boundary must be BF16')
        weight = self.operands.get(self.name, self.rank, matrix=True)
        require(tuple(weight.shape) == self.shape, 'Local linear geometry: ' + self.name)
        result = F.linear(x, weight)
        require(result.dtype == torch.bfloat16, 'Linear partial must be BF16')
        return result


class ExpertBank:
    def __init__(self, prefix, parts, operands, rank, shape):
        self.prefix, self.parts, self.operands, self.rank, self.shape = prefix, parts, operands, rank, tuple(shape)

    def __getitem__(self, expert):
        expert = int(expert)
        require(0 <= expert < self.operands.config.num_experts, 'Expert out of range')
        value = torch.cat([self.operands.get(f'{self.prefix}.{expert}.{part}.weight', self.rank, matrix=True)
                           for part in self.parts], dim=0)
        require(tuple(value.shape) == self.shape, 'Local expert shape')
        return value


def bind(module, prefix, operands, rank):
    """Replace operands, not any HF mathematical forward implementation."""
    hf = native.hf
    for name, child in list(module.named_modules()):
        full = '.'.join(p for p in (prefix, name) if p)
        if type(child) is hf.Qwen3_5MoeExperts:
            for attr, parts in [('gate_up_proj', ('gate_proj', 'up_proj')), ('down_proj', ('down_proj',))]:
                shape = child._parameters[attr].shape[1:]
                del child._parameters[attr]
                setattr(child, attr, ExpertBank(full, parts, operands, rank, shape))
        elif isinstance(child, nn.Linear):
            require(child.bias is None and name, 'Unexpected biased/root linear')
            parent, _, attr = name.rpartition('.')
            setattr(module.get_submodule(parent), attr, Linear(full + '.weight', operands, rank, child.weight.shape))
    for name, parameter in list(module.named_parameters()):
        full = '.'.join(p for p in (prefix, name) if p)
        value = operands.get(full, rank)
        require(value.shape == parameter.shape, 'Local native parameter shape: ' + full)
        parent, _, attr = name.rpartition('.')
        setattr(module.get_submodule(parent), attr, nn.Parameter(value, requires_grad=False))
    return module.eval()


class Capture:
    def __init__(self, max_bytes=512*1024**2, device='cpu'):
        self.stage, self.tensors, self.handles = '', {}, []
        self.bytes, self.max_bytes, self.device = 0, max_bytes, torch.device(device)

    def put(self, key, value):
        require(value.device == self.device, 'Capture device differs: ' + key)
        expected = (torch.int64 if key.endswith(('input_ids', 'position_ids', 'cache_position', 'attention_mask',
                    'selected_expert_ids', 'native_expert_ids')) else
                    torch.float32 if 'recurrent_states.' in key or key.endswith(('routing_weights', 'router_probabilities'))
                    else torch.bfloat16)
        require(value.dtype == expected, 'Boundary dtype: ' + key + ': ' + str(value.dtype))
        require(not value.is_floating_point() or bool(torch.isfinite(value).all()), 'Nonfinite boundary: ' + key)
        key = self.stage + '.' + key
        size = value.numel()*value.element_size()
        require(key not in self.tensors and self.bytes + size <= self.max_bytes, 'Duplicate/capture cap: ' + key)
        self.tensors[key] = value.detach().cpu().contiguous().clone()
        self.bytes += size

    def observe(self, module, key):
        self.handles.append(module.register_forward_hook(
            lambda m, a, out: self.put(key, out[0] if isinstance(out, tuple) else out)))

    def router(self, module, prefix):
        def hook(m, args, out):
            raw, selected = native.router_observation(m, args, out, native.ROUTER_PROFILE)
            for name, value in [('raw_router_logits', raw), ('router_probabilities', out[0]),
                ('native_expert_ids', out[2]), ('selected_expert_ids', selected[2]), ('routing_weights', selected[1])]:
                self.put(prefix + '.' + name, value)
            return selected
        self.handles.append(module.register_forward_hook(hook))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def merge_gdn(values, config, world, axis=1):
    widths = [s // world for s in gdn_groups(config)]
    split = [value.split(widths, dim=axis) for value in values]
    return torch.cat([torch.cat([rank[group] for rank in split], dim=axis) for group in range(3)], dim=axis)


class Attention(nn.Module):
    def __init__(self, branches, config, local, index, capture):
        super().__init__()
        self.branches = nn.ModuleList(branches)
        self.config, self.local, self.index, self.capture = config, local, index, capture
        self.caches = None

    def forward(self, hidden_states, **kwargs):
        linear = self.config.layer_types[self.index] == 'linear_attention'
        cache_key = 'cache_params' if linear else 'past_key_values'
        global_cache = kwargs.pop(cache_key, None)
        require(global_cache is not None and self.caches is not None, 'Explicit reset/cache required')
        partials = []
        for rank, branch in enumerate(self.branches):
            args = dict(kwargs, **{cache_key: self.caches[rank]})
            if linear:
                args = {k: v for k, v in args.items() if k in ('cache_params', 'attention_mask')}
            out = branch(hidden_states=hidden_states, **args)
            value = out if linear else out[0]
            partials.append(value)
        states = [c.layers[self.index] for c in self.caches]
        if linear:
            global_cache.update_conv_state(merge_gdn([s.conv_states for s in states], self.config, len(states)), self.index)
            global_cache.update_recurrent_state(torch.cat([s.recurrent_states for s in states], dim=1), self.index)
        else:
            n = hidden_states.shape[1]
            global_cache.update(torch.cat([s.keys[:, :, -n:] for s in states], dim=1),
                                torch.cat([s.values[:, :, -n:] for s in states], dim=1), self.index)
        result = rank_sum(partials)
        return result if linear else (result, None)


class Moe(nn.Module):
    def __init__(self, branches):
        super().__init__()
        self.branches = nn.ModuleList(branches)

    def forward(self, hidden_states):
        # HF SparseMoeBlock performs expert-ID-ascending BF16 accumulation,
        # BF16 sigmoid shared gating and local BF16 add. Reduce ONLY afterwards.
        return rank_sum([branch(hidden_states) for branch in self.branches])


class Vocab(nn.Module):
    def __init__(self, branches, capture):
        super().__init__()
        self.branches, self.capture = nn.ModuleList(branches), capture

    def forward(self, x):
        values = [branch(x) for branch in self.branches]
        for rank, value in enumerate(values):
            self.capture.put(f'rank.{rank}.logits', value)
        return torch.cat(values, dim=-1)


def build_model(config, operands, capture):
    hf, world = native.hf, operands.world
    local = local_config(config, world)
    with torch.device('meta'):
        model = hf.Qwen3_5MoeForCausalLM(config)
    for index, layer in enumerate(model.model.layers):
        attention_attr = 'linear_attn' if layer.layer_type == 'linear_attention' else 'self_attn'
        branches, moes = [], []
        for rank in range(world):
            with torch.device('meta'):
                attn = (hf.Qwen3_5MoeGatedDeltaNet(local, index) if attention_attr == 'linear_attn'
                        else hf.Qwen3_5MoeAttention(local, index))
                moe = hf.Qwen3_5MoeSparseMoeBlock(local)
            prefix = f'model.layers.{index}'
            branches.append(bind(attn, prefix + '.' + attention_attr, operands, rank))
            moes.append(bind(moe, prefix + '.mlp', operands, rank))
            p = f'rank.{rank}.layers.{index:02d}'
            capture.observe(attn, p + '.attention_partial')
            capture.observe(moe, p + '.moe_partial')
            capture.observe(moe.experts, p + '.routed_partial')
            capture.observe(moe.shared_expert, p + '.shared_partial')
            capture.observe(moe.shared_expert_gate, p + '.shared_gate_logit')
            capture.router(moe.gate, p)
            if attention_attr == 'linear_attn':
                require(attn.chunk_gated_delta_rule is hf.torch_chunk_gated_delta_rule and
                        attn.recurrent_gated_delta_rule is hf.torch_recurrent_gated_delta_rule and
                        attn.causal_conv1d_update is hf.torch_causal_conv1d_update and
                        attn.causal_conv1d_fn is None,
                        'GDN source interception detected')
        setattr(layer, attention_attr, Attention(branches, config, local, index, capture))
        layer.mlp = Moe(moes)
        for name in ('input_layernorm', 'post_attention_layernorm'):
            bind(getattr(layer, name), f'model.layers.{index}.{name}', operands, 0)
        p = f'layers.{index:02d}'
        capture.observe(layer, p + '.hidden')
        capture.observe(getattr(layer, attention_attr), p + '.attention')
        capture.observe(layer.mlp, p + '.moe')
    bind(model.model.embed_tokens, 'model.embed_tokens', operands, 0)
    bind(model.model.norm, 'model.norm', operands, 0)
    model.model.rotary_emb = hf.Qwen3_5MoeTextRotaryEmbedding(config, device=operands.device)
    native.legacy.audit_rope_f32(model.model.rotary_emb)
    model.lm_head = Vocab([Linear('lm_head.weight', operands, r,
                                  (config.vocab_size//world, config.hidden_size)) for r in range(world)], capture)
    capture.observe(model.model.embed_tokens, 'embedding')
    capture.observe(model.model.norm, 'final_norm')
    require(all(p.dtype == torch.bfloat16 and p.device == operands.device for p in model.parameters()),
            'Unmaterialized/non-BF16 parameter')
    require(all(b.device == operands.device for b in model.buffers()), 'Unmaterialized buffer')
    return model.eval()


def reset_caches(model, config, device, world):
    local = local_config(config, world)
    rank_caches = [native.f32_ssm_cache(local, [device]*config.num_hidden_layers) for _ in range(world)]
    for layer in model.model.layers:
        attn = layer.linear_attn if layer.layer_type == 'linear_attention' else layer.self_attn
        attn.caches = rank_caches
    return native.f32_ssm_cache(config, [device]*config.num_hidden_layers), rank_caches


def expected_tensors(c, ids, world=2, stages=None):
    """Independent complete inventory; missing rank/layer/stage is fatal."""
    result = {}
    for stage, n, start in (stages if stages is not None else [('prefill', len(ids), 0), ('decode.0', 1, len(ids))]):
        def add(key, shape, dtype='torch.bfloat16'):
            result[stage + '.' + key] = dict(shape=list(shape), dtype=dtype)
        for key, shape in [('input_ids', (1,n)), ('position_ids', (1,n)), ('cache_position', (n,)),
                           ('attention_mask', (1,start+n))]:
            add(key, shape, 'torch.int64')
        add('logits', (1,n,c.vocab_size))
        for key in ('embedding', 'final_norm'):
            add(key, (1,n,c.hidden_size))
        for r in range(world):
            add(f'rank.{r}.logits', (1,n,c.vocab_size//world))
        for i, kind in enumerate(c.layer_types):
            for key in ('hidden', 'attention', 'moe'):
                add(f'layers.{i:02d}.{key}', (1,n,c.hidden_size))
            for r in range(world):
                p = f'rank.{r}.layers.{i:02d}.'
                for key in ('attention_partial', 'moe_partial'):
                    add(p+key, (1,n,c.hidden_size))
                for key in ('routed_partial', 'shared_partial'):
                    add(p+key, (n,c.hidden_size))
                add(p+'shared_gate_logit', (n,1))
                add(p+'raw_router_logits', (n,c.num_experts))
                add(p+'router_probabilities', (n,c.num_experts), 'torch.float32')
                add(p+'routing_weights', (n,c.num_experts_per_tok), 'torch.float32')
                for key in ('native_expert_ids', 'selected_expert_ids'):
                    add(p+key, (n,c.num_experts_per_tok), 'torch.int64')
            for rank in [None, *range(world)]:
                p = '' if rank is None else f'rank.{rank}.'
                div = 1 if rank is None else world
                if kind == 'linear_attention':
                    add(p+f'cache.conv_states.{i:02d}', (1,sum(gdn_groups(c))//div,c.linear_conv_kernel_dim))
                    add(p+f'cache.recurrent_states.{i:02d}',
                        (1,c.linear_num_value_heads//div,c.linear_key_head_dim,c.linear_value_head_dim), 'torch.float32')
                else:
                    for key in ('key_cache', 'value_cache'):
                        add(p+f'cache.{key}.{i:02d}', (1,c.num_key_value_heads//div,start+n,c.head_dim))
    return result


def run_call(model, config, operands, capture, cache, rank_caches, fed, start, stage):
    """One native HF invocation, shared by legacy and bounded lifecycle captures."""
    require(fed and all(type(i) is int and 0 <= i < config.vocab_size for i in fed), 'Invalid input IDs')
    require(cache.get_seq_length() == start, 'Cache before-call frontier mismatch')
    for layer in model.model.layers:
        attn = layer.linear_attn if layer.layer_type == 'linear_attention' else layer.self_attn
        attn.caches = rank_caches
    capture.stage = stage
    with torch.inference_mode():
        inputs = torch.tensor([fed], dtype=torch.int64, device=operands.device)
        positions = torch.arange(start, start+len(fed), device=operands.device)
        mask = torch.ones((1,start+len(fed)), dtype=torch.int64, device=operands.device)
        output = model(input_ids=inputs, position_ids=positions[None], attention_mask=mask,
                       past_key_values=cache, use_cache=True, logits_to_keep=0, return_dict=True)
        require(output.past_key_values is cache and cache.get_seq_length() == start+len(fed), 'Cache frontier mismatch')
        for key,value in [('input_ids',inputs), ('position_ids',positions[None]), ('cache_position',positions),
                          ('attention_mask',mask), ('logits',output.logits)]:
            capture.put(key,value)
        for rank,state in [(None,cache), *enumerate(rank_caches)]:
            prefix = '' if rank is None else f'rank.{rank}.'
            for i,layer in enumerate(state.layers):
                for field,key in [('conv_states','conv_states'), ('recurrent_states','recurrent_states'),
                                  ('keys','key_cache'), ('values','value_cache')]:
                    value = getattr(layer,field,None)
                    if value is not None: capture.put(prefix+f'cache.{key}.{i:02d}',value)
        predicted = int(output.logits[0,-1].argmax())
    return dict(stage=stage,input_token_ids=list(fed),position_start=start,
                sequence_length_after=start+len(fed),next_token_id=predicted)


def run_case(model, config, operands, capture, ids, decode_token=None):
    capture.tensors.clear(); capture.bytes = 0
    cache, rank_caches = reset_caches(model, config, operands.device, operands.world)
    calls, fed = [], list(ids)
    for stage in ('prefill', 'decode.0'):
        start = 0 if stage == 'prefill' else len(ids)
        call = run_call(model,config,operands,capture,cache,rank_caches,fed,start,stage)
        calls.append(call)
        fed = [call['next_token_id'] if decode_token is None else decode_token]
    expected = expected_tensors(config, ids, operands.world)
    actual = {k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in capture.tensors.items()}
    require(actual == expected, 'Incomplete/incorrect full boundary inventory')
    for stage in ('prefill', 'decode.0'):
        logits = [capture.tensors[f'{stage}.rank.{r}.logits'] for r in range(operands.world)]
        require(torch.equal(torch.cat(logits,dim=-1),capture.tensors[stage+'.logits']), 'Vocab gather order mismatch')
        if operands.world == 2:
            for i in range(config.num_hidden_layers):
                for field in ('raw_router_logits','router_probabilities','routing_weights','selected_expert_ids'):
                    require(torch.equal(capture.tensors[f'{stage}.rank.0.layers.{i:02d}.{field}'],
                                        capture.tensors[f'{stage}.rank.1.layers.{i:02d}.{field}']), 'Replica route disagreement')
    return calls


def code_fingerprint(code):
    """Canonical bytecode identity, independent of marshal reference-table flags."""
    from types import CodeType
    def constant(value):
        if isinstance(value, CodeType):return dict(code=describe(value))
        if isinstance(value, tuple):return dict(tuple=[constant(v) for v in value])
        if isinstance(value, frozenset):return dict(frozenset=sorted([constant(v) for v in value],key=repr))
        return dict(type=type(value).__name__,value=repr(value))
    def describe(c):
        return dict(bytecode=c.co_code.hex(),constants=[constant(v) for v in c.co_consts],
            names=c.co_names,varnames=c.co_varnames,freevars=c.co_freevars,cellvars=c.co_cellvars,
            args=c.co_argcount,posonly=c.co_posonlyargcount,kwonly=c.co_kwonlyargcount,
            locals=c.co_nlocals,stack=c.co_stacksize,flags=c.co_flags,
            exceptiontable=getattr(c,'co_exceptiontable',b'').hex())
    return hashlib.sha256(json.dumps(describe(code),sort_keys=True).encode()).hexdigest()


def source_audit():
    objects = ['Qwen3_5MoeDecoderLayer', 'Qwen3_5MoeAttention', 'Qwen3_5MoeGatedDeltaNet',
               'Qwen3_5MoeSparseMoeBlock', 'Qwen3_5MoeExperts', 'Qwen3_5MoeTopKRouter',
               'Qwen3_5MoeRMSNorm', 'Qwen3_5MoeRMSNormGated']
    functions = {name+'.forward': getattr(native.hf,name).forward for name in objects}
    functions.update({name:getattr(native.hf,name) for name in (
        'torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule','torch_causal_conv1d_update',
        'eager_attention_forward','apply_rotary_pos_emb','repeat_kv','l2norm','apply_mask_to_padding_states')})
    live = {}
    for name, function in functions.items():
        unwrapped = inspect.unwrap(function)
        live[name] = dict(source_sha256=hashlib.sha256(inspect.getsource(unwrapped).encode()).hexdigest(),
                          code_sha256=code_fingerprint(unwrapped.__code__),
                          code_file=unwrapped.__code__.co_filename,
                          wrapper_code_sha256=code_fingerprint(function.__code__))
    return dict(live_functions=live,
                native_forwards={name:hashlib.sha256(inspect.getsource(getattr(native.hf,name).forward).encode()).hexdigest()
                                for name in objects},
                replacements=['linear operands', 'expert bank operands', 'raw-ties-only router forward hook',
                              'two native local attention calls + sum/cache assembly', 'two native local MoE calls + sum',
                              'vocabulary shards + gather'],
                unchanged=['HF norm/attention/GDN/experts/router/decoder forward source', 'F32 RoPE construction'],
                production_imports=False, hf_implementation='eager', actual_compute_device='recorded in manifest',
                simulation='two logical ranks on one granted CUDA device; no NCCL or performance claim',
                vllm_bitexact=False, routed_order='HF ascending expert ID; each F32 weighted product cast BF16 then BF16 index_add')


def tensor_metric(actual, expected, exact=False):
    require(actual.shape == expected.shape and actual.dtype == expected.dtype, 'Comparison shape/dtype mismatch')
    require(bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()), 'Nonfinite comparison')
    error = (actual.double()-expected.double()).abs()
    tolerance = torch.zeros_like(error) if exact else ATOL+RTOL*expected.double().abs()
    return dict(max_abs=float(error.max()), mean_abs=float(error.mean()), rmse=float(error.square().mean().sqrt()),
                exact=torch.equal(actual,expected), failures=int((error>tolerance).sum()), elements=error.numel(),
                atol=0 if exact else ATOL, rtol=0 if exact else RTOL)



def file_record(path):
    path = Path(path)
    return dict(file=path.name,sha256=sha256(path),bytes=path.stat().st_size)


def save_case(output, name, config, ids, capture, calls, drift=None, inventory=None):
    """Shared CPU-smoke/CUDA serialization path; never promotes run completion."""
    from safetensors.torch import save_file
    import re
    require(name in PROMPTS if inventory is None else bool(re.fullmatch(r'[a-z][a-z0-9_-]{0,79}',name)), 'Unknown/unsafe case name')
    inventory = expected_tensors(config, ids) if inventory is None else inventory
    require({k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in capture.tensors.items()} == inventory,
            'Incomplete case cannot be serialized')
    path = Path(output)/(name+'.safetensors')
    require(not path.exists() and not (Path(output)/(name+'.json')).exists(), 'Refusing overwrite')
    stats = {k: dict(sha256=digest_tensor(v), elements=v.numel(), bytes=v.numel()*v.element_size(),
                     finite=bool(torch.isfinite(v).all()), minimum=float(v.min()), maximum=float(v.max()))
             for k,v in capture.tensors.items()}
    require(all(v['finite'] for v in stats.values()), 'Nonfinite serialized case')
    native.write_json(Path(output)/(name+'.stats.json'), stats)
    save_file(capture.tensors, str(path), metadata={'schema': SCHEMA, 'profile': PROFILE,
        'device': str(capture.device), 'logical_tp': '2', 'ssm': 'F32', 'model_boundary': 'BF16'})
    report = dict(id=name, calls=calls, tensors=inventory, capture_bytes=capture.bytes,
                  unsharded_drift=drift or dict(status='not_supplied', acceptance='unchanged'), **file_record(path))
    native.write_json(Path(output)/(name+'.json'), report)
    return dict(id=name, **file_record(path), report=file_record(Path(output)/(name+'.json')),
                statistics=file_record(Path(output)/(name+'.stats.json')))


def authenticate_export(payload, launch_hash):
    """Verify a live descendant of the maintained authenticated paper worker.

    A Python import, owner flag or environment variable alone is not custody.
    This runs before dependency initialization, checkpoint reads or CUDA queries.
    """
    import os
    sys.path.insert(0, str(ROOT/'papers/arxiv/experiments'))
    import contract as ct
    import launch_guard as guard
    import telemetry
    telemetry.authorize(payload.get('sole_gpu_owner_granted'), payload.get('gpu_uuid'),
                        payload.get('owner_note'), 'device_exclusive_local_custody')
    require(payload.get('execute_cuda') is True, 'Export-only child')
    directory = Path(payload['directory']).resolve()
    require(directory.is_relative_to(HERE), 'Export must stay paper-local')
    receipt = directory/'worker-launch.json'
    require(receipt.is_file() and not receipt.is_symlink() and receipt.stat().st_mode & 0o222 == 0,
            'Missing sealed launch receipt')
    require(launch_hash and sha256(receipt) == launch_hash and ct.load(receipt) == payload,
            'Launch receipt/hash differs')
    require(os.environ.get('FERRULE_PAPER_WORKER_TOKEN') == payload['worker_token'], 'Child token differs')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == payload['gpu_uuid'], 'Physical grant/environment mismatch')
    auth = ct.load(directory/'authentication.json')
    parent = guard.identity(os.getppid())
    current = guard.identity(os.getpid())
    require(parent == auth['worker'] and current['pgrp'] == parent['pgrp'] and current['session'] == parent['session'],
            'Numerical child is not in authenticated worker custody')
    require(guard.identity(parent['ppid']) == auth['helper'] and
            guard.identity(payload['coordinator']['pid']) == auth['coordinator'] == payload['coordinator'],
            'Custodian/coordinator identity changed')
    journal = ct.load(directory/'custody.json')
    # The subreaper registers the child before its setsid(). Journal pgrp/session
    # may therefore be birth-time values; live session/ancestry was checked above.
    root = journal.get('root_identity') or {}
    require(journal.get('containment_verified') is True and journal.get('method') == 'linux-subreaper-waitid' and
            journal.get('launch_id') == auth['launch_id'] and journal.get('helper_identity') == auth['helper'] and
            all(root.get(k) == parent[k] for k in ('pid','birth_ticks','ppid')), 'Authenticated custody journal differs')
    require(os.readlink('/proc/self/ns/pid') == os.readlink(f'/proc/{parent["pid"]}/ns/pid'),
            'Child PID namespace differs')
    ct.validate_owner_lock(payload['owner_lock'], payload['gpu_uuid'], payload['coordinator']['pid'])
    return dict(child=current, worker=parent, helper=auth['helper'], coordinator=auth['coordinator'],
                launch_sha256=launch_hash, launch_id=auth['launch_id'])


def baseline_metrics(directory, pinned_hash, name, capture, checkpoint, config_hash, calls, cap):
    """Read-only optional old-HF drift, never an oracle acceptance redefinition."""
    from safetensors import safe_open
    directory = Path(directory)
    require(sha256(directory/'manifest.json') == pinned_hash, 'Unsharded manifest pin mismatch')
    manifest = json.loads((directory/'manifest.json').read_text())
    require(manifest['schema'] in (native.SCHEMA,native.PROFILE_SCHEMA) and manifest['status']=='complete'
            and manifest['full_reference_complete'], 'Not a completed unsharded BF16 reference')
    require(manifest['checkpoint']['config_sha256']==config_hash and manifest['versions']==dict(native.VERSIONS),
            'Unsharded config/version mismatch')
    payload_record=manifest['source_payloads']
    payload_path=directory/payload_record['file']
    require(payload_path.parent.resolve()==directory.resolve() and sha256(payload_path)==payload_record['sha256'],
            'Unsharded payload provenance differs')
    previous=json.loads(payload_path.read_text())
    require(previous['index_sha256']==checkpoint.index_hash,'Different checkpoint index')
    common=checkpoint.payloads.keys() & previous['used_payloads'].keys()
    require(bool(common) and all(checkpoint.payloads[k]==previous['used_payloads'][k] for k in common),
            'Shared original checkpoint payload differs')
    entry=next(c for c in manifest['cases'] if c['id']==name)
    path=directory/entry['file']
    require(path.parent.resolve()==directory.resolve() and path.stat().st_size<=cap and sha256(path)==entry['sha256'],
            'Unsharded case escapes path/cap/hash')
    records={}
    with safe_open(path,framework='pt',device='cpu') as archive:
        require(tensor_metric(capture.tensors['prefill.input_ids'],archive.get_tensor('prefill.input_ids'),True)['failures']==0,
                'Different original prompt')
        for stage in ('prefill','decode.0'):
            same_inputs=torch.equal(capture.tensors[stage+'.input_ids'],archive.get_tensor(stage+'.input_ids'))
            if not same_inputs:
                records[stage]=dict(status='different_autoregressive_token_not_same_input',metrics=None)
                continue
            rows={}
            for key in archive.keys():
                if not key.startswith(stage+'.'):continue
                mapped=key
                if key not in capture.tensors and '.layers.' in key:
                    mapped=key.replace(stage+'.layers.',stage+'.rank.0.layers.',1)
                if mapped in capture.tensors:
                    expected=archive.get_tensor(key)
                    exact=expected.dtype==torch.int64
                    rows[key]=tensor_metric(capture.tensors[mapped],expected,exact)
            records[stage]=dict(status='same_input',metrics=rows,
                full_vocab_argmax_matches=torch.equal(capture.tensors[stage+'.logits'].argmax(-1),
                                                       archive.get_tensor(stage+'.logits').argmax(-1)))
    return dict(manifest_sha256=pinned_hash,reference_schema=manifest['schema'],stages=records,
                acceptance='descriptive drift only; original tolerance unchanged; does not gate TP-reference validity',
                provenance_scope='config/index and common used payload hashes matched; historical unused weights not certified',
                matched_payloads=len(common),calls=calls)


def checkpoint_metadata(checkpoint, shard_paths):
    """Original safetensors headers only; no requantization or tensor materialization."""
    import struct
    result = dict(index_sha256=checkpoint.index_hash, shards={}, tensors={})
    for path in shard_paths:
        size = path.stat().st_size
        with path.open('rb') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            require(0 < length <= 64*1024**2 and 8+length <= size, 'Invalid safetensors header size')
            raw = stream.read(length)
        header = json.loads(raw)
        result['shards'][path.name] = dict(bytes=size, header_bytes=length,
            header_sha256=hashlib.sha256(raw).hexdigest(), metadata=header.pop('__metadata__', {}))
        for name,entry in header.items():
            require(name not in result['tensors'] and 0 <= entry['data_offsets'][0] <= entry['data_offsets'][1] <= size-8-length,
                    'Duplicate/invalid original tensor header')
            result['tensors'][name] = dict(shard=path.name, **entry)
    require(set(result['tensors']) == set(checkpoint.mapping), 'Checkpoint index/header tensor inventory differs')
    require(all(result['tensors'][name]['shard'] == shard for name,shard in checkpoint.mapping.items()), 'Index/header shard differs')
    result.update(tensor_count=len(result['tensors']), shard_count=len(shard_paths),
                  total_shard_bytes=sum(s['bytes'] for s in result['shards'].values()),
                  policy='Original file headers/names/dtypes/offsets plus whole-file hashes; no float-to-FP8 quantization')
    return result


def export(payload, launch_hash=None):
    """Only called below the authenticated paper worker, after a fresh grant."""
    import gc
    import os
    import time
    authentication = authenticate_export(payload, launch_hash)
    require(payload['execute_cuda'] is True and payload['sole_gpu_owner_granted'] is True and payload['owner_note'].strip(),
            'Export requires an explicit fresh owner grant')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==payload['gpu_uuid'], 'Physical grant/environment mismatch')
    output=Path(payload['directory'])
    native.write_json(output/'numerical-authentication.json', authentication)
    manifest=dict(schema=SCHEMA,profile=PROFILE,status='incomplete',full_reference_complete=False,
        strict_ferrule_full40=False,versions=dict(native.VERSIONS),cases=[],
        numerical_authentication=file_record(output/'numerical-authentication.json'),
        protocol=payload['protocol'],workflow_pins=payload['pins'],
        execution=dict(device='cuda:0',physical_gpu_uuid=payload['gpu_uuid'],logical_tp=2,ep=1,pp=1,
            tf32=False,bf16_reduced_precision_reduction=False,production_imports=False,
            transport='rank-order Torch simulation on one granted device; no NCCL/transport/performance proof'),
        runtime_gate=dict(status='not_run',requires='caller-supplied reference path plus SHA256 and fresh Ferrule full40 comparison'),
        open_claims=['No vLLM fused-MoE bit equivalence', 'No CPU-to-CUDA exponential/state equality claim',
                     'No full40 Ferrule equivalence until caller compares every required tensor'])
    checkpoint,operands,capture,model=None,None,None,None
    start=time.monotonic()
    if payload.get('scenario_plan') is not None:
        import multistep
        multistep.validate_plan(payload['scenario_plan'],full_model=True)
        manifest['schema']=multistep.SCHEMA
    try:
        for path,pin in payload['pins'].items():require(sha256(path)==pin,'Workflow source changed: '+path)
        manifest['native_source']=initialize()
        manifest['interception_audit']=source_audit()
        import marshal
        marshal_before={name:hashlib.sha256(marshal.dumps(inspect.unwrap(getattr(native.hf,name).forward).__code__)).hexdigest()
                        for name in ('Qwen3_5MoeDecoderLayer','Qwen3_5MoeAttention','Qwen3_5MoeGatedDeltaNet','Qwen3_5MoeExperts')}
        require(not torch.cuda.is_initialized(),'Unexpected early CUDA initialization')
        manifest['loading_evidence']=native.loading_evidence()
        import cuda_gate
        print('Authenticated CUDA small-fixture gate before full checkpoint reads', flush=True)
        gate=cuda_gate.run(output)
        require(gate['status']=='pass', 'Small CUDA gate failed')
        manifest['cuda_small_gate']=file_record(output/'cuda-small-gate.json')
        print('CUDA small-fixture gate passed; hashing full checkpoint', flush=True)
        model_dir=Path(payload['model'])
        config=native.read_config(model_dir)
        local_config(config,2)
        checkpoint=native.legacy.Checkpoint(model_dir)
        # Full shard hashes are taken only in an authorized export, NEVER CPU preflight.
        shard_paths=sorted({(model_dir/p).resolve() for p in checkpoint.mapping.values()})
        require(all(p.parent==model_dir.resolve() for p in shard_paths),'Checkpoint shard escape')
        stamps={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in shard_paths}
        metadata=checkpoint_metadata(checkpoint,shard_paths)
        native.write_json(output/'checkpoint-metadata.json',metadata)
        manifest['checkpoint_metadata']=file_record(output/'checkpoint-metadata.json')
        manifest['checkpoint']=dict(directory=str(model_dir.resolve()),config_sha256=sha256(model_dir/'config.json'),
            effective_text_config=config.to_dict(),index_sha256=checkpoint.index_hash,
            full_shards={p.name:sha256(p) for p in shard_paths},
            shard_bytes={p.name:p.stat().st_size for p in shard_paths},
            tensor_count=metadata['tensor_count'],total_shard_bytes=metadata['total_shard_bytes'],
            tokenizer_files={name:sha256(model_dir/name) for name in ('tokenizer.json','tokenizer_config.json')})
        identity=json.dumps(manifest['checkpoint']['full_shards'],sort_keys=True).encode()
        manifest['checkpoint']['full_shard_set_sha256']=hashlib.sha256(identity).hexdigest()
        manifest['checkpoint']['hash_scope']='all original indexed shard files, including unused experts/vision/MTP'
        free,total=torch.cuda.mem_get_info('cuda:0')
        require(free>=4*1024**3,'Need at least 4GiB oracle workspace/embedding/cache headroom')
        manifest['device']=dict(free_before=free,total=total,name=torch.cuda.get_device_name(0))
        torch.cuda.reset_peak_memory_stats(0)
        operands=Operands(checkpoint,config,'cuda:0',cache_bytes=payload['cache_mib']*1024**2)
        capture=Capture(payload['capture_mib']*1024**2,'cuda:0')
        model=build_model(config,operands,capture)
        if payload.get('scenario_plan') is not None:
            import multistep
            manifest['suite']=multistep.run(payload['scenario_plan'],model,config,operands,capture,output)
            manifest['cases']=manifest['suite']['cases']
        else:
            for name,ids in PROMPTS.items():
                print('Full40 case '+name+' prefill + cached decode',flush=True)
                calls=run_case(model,config,operands,capture,ids)
                drift = None
                if payload.get('unsharded_reference'):
                    drift=baseline_metrics(payload['unsharded_reference'],payload['unsharded_sha256'],
                        name,capture,checkpoint,manifest['checkpoint']['config_sha256'],calls,payload['capture_mib']*1024**2)
                manifest['cases'].append(save_case(output,name,config,ids,capture,calls,drift))
                print('Captured '+name+': '+str(calls),flush=True)
                capture.tensors.clear();capture.bytes=0
        torch.cuda.synchronize()
        require(all((p.stat().st_size,p.stat().st_mtime_ns)==stamps[str(p)] for p in shard_paths),'Source shard modified')
        require(sha256(model_dir/'config.json')==manifest['checkpoint']['config_sha256'],'Config changed')
        for name,pin in manifest['checkpoint']['tokenizer_files'].items():require(sha256(model_dir/name)==pin,'Tokenizer changed')
        for path,pin in payload['pins'].items():require(sha256(path)==pin,'Workflow source changed during export')
        after_audit=source_audit()
        native.write_json(output/'interception-after.json',after_audit)
        manifest['interception_after']=file_record(output/'interception-after.json')
        manifest['marshal_diagnostic_only']=dict(before=marshal_before,after={name:hashlib.sha256(marshal.dumps(inspect.unwrap(getattr(native.hf,name).forward).__code__)).hexdigest() for name in marshal_before},
            policy='Python marshal serializes object-sharing references; canonical bytecode/source identity is the acceptance check')
        require(after_audit==manifest['interception_audit'],'HF forward interception changed')
        native.write_json(output/'operand-layout.json',operands.audit)
        manifest['operand_layout']=file_record(output/'operand-layout.json')
        manifest['device'].update(peak_allocated=torch.cuda.max_memory_allocated(0),
                                  peak_reserved=torch.cuda.max_memory_reserved(0),peak_operand_cache=operands.decoded.peak_bytes)
        manifest.update(status='export_complete_pending_custody',full_reference_complete=False,
                        boundary_inventory_complete=True,reference_kind='independent_profile_matched_tp2_hf_eager')
    except BaseException as error:
        manifest.update(status='incomplete',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if checkpoint is not None:
            native.write_json(output/'source-payloads.json',checkpoint.provenance())
            manifest['source_payloads']=file_record(output/'source-payloads.json')
        if capture is not None:capture.close()
        if operands is not None:operands.decoded.clear()
        del model
        gc.collect()
        if torch.cuda.is_initialized():torch.cuda.empty_cache()
        import resource
        manifest['peak_host_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        manifest['host_memory_policy']='lazy one-tensor reads; no all-model GPU residency; peak RSS reported, not a hard partition'
        manifest['elapsed_seconds']=time.monotonic()-start
        with (output/'export.json').open('x') as stream:json.dump(manifest,stream,indent=2,allow_nan=False)


if __name__=='__main__':
    raise SystemExit('Use launch.py; direct CUDA worker invocation is not an authorized export')
