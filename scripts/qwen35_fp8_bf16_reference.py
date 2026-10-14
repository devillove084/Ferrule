#!/usr/bin/env python3
"""Independent native-BF16 model-boundary oracle for raw Qwen3.5 FP8 storage.

No CUDA work without --execute-cuda. Separate from the legacy W8/F32 oracle.
Native eager HF family/expert forwards; only lazy checkpoint operands are bound.
The declared F32 SSM cache is NOT HF 5.5.3's default BF16 cache allocation.
"""
import argparse
from collections import OrderedDict
from enum import Enum
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import MappingProxyType

import qwen35_fp8_cuda_reference as legacy

SCHEMA = 'ferrule.qwen35-fp8-native-bf16-cuda-reference.v1'
PROFILE_SCHEMA = 'ferrule.qwen35-fp8-profile-bf16-cuda-reference.v1'
ROUTER_NATIVE = 'native_unstable'
ROUTER_PROFILE = 'lower_expert_id'
PROFILE = 'cuda-hybrid-numeric-fp8-model-bf16-f32-ssm'
VERSIONS = MappingProxyType({'torch': '2.13.0+cu129', 'transformers': '5.5.3'})
SOURCE_PINS = MappingProxyType({
    'modeling_qwen3_5_moe': '0f8ca77e875c31d448137c9ac291bed505c6b9927a85b5b8460af29d1e1ee647',
    'cache_utils': 'a51d2cd525f4458941a20cb5b3b8a8d989d8ca5bcd451855a27bb41743fed586',
    'modeling_utils': 'b8467e1ada952862d2e4d76632475ac2f9fa198121d0e1ce64fa393ea3773e16',
})

class ModelBoundary(str, Enum):
    BF16 = 'bf16'

CONTRACT = MappingProxyType(dict(
    profile=PROFILE, model_boundary=ModelBoundary.BF16.value,
    fp8='E4M3FN', scale='original numeric BF16 multiplicative 128x128',
    weight='FP8 times BF16 scale in F32, then BF16 RNE; native BF16 words preserved',
    activation='BF16', accumulation='F32', output='BF16', kv='BF16', conv='BF16',
    ssm='F32 explicitly preallocated; NOT the HF default cache dtype',
    fp32_parameters='only GDN A_log/norm: once BF16 RNE on load, as pinned native from_pretrained(dtype=BF16)',
    rope_frequency='native F32 construction, never blanket model.to(BF16)', router_weights='F32 native full softmax/topk/renormalize',
    tf32=False, bf16_reduced_precision_reduction=False,
    baseline='decoded-storage native eager BF16 model, NOT the HF dynamic-FP8 quantizer',
))
require, sha256, write_json = legacy.require, legacy.sha256, legacy.write_json


def canonicalize_router_ties(raw_logits, out):
    """Only substitute IDs at exactly equal raw logits; retain native F32 weights.

    Distinct logits can collide after F32 softmax rounding/underflow. Those are
    NOT profile ties and must retain the native selection, even at the cutoff.
    """
    import torch
    probabilities, weights, native_ids = out
    require(raw_logits.shape == probabilities.shape and raw_logits.ndim == 2,
            'Router raw logits/probability shape mismatch')
    require(bool(torch.isfinite(raw_logits).all()), 'Nonfinite raw router logits')
    require(probabilities.dtype == weights.dtype == torch.float32, 'Native router weights must be F32')
    selected = raw_logits.gather(-1, native_ids)
    ids = native_ids.clone()
    candidates = torch.arange(raw_logits.shape[-1], device=raw_logits.device).expand_as(raw_logits)
    for slot in range(native_ids.shape[-1]):
        value = selected[:, slot:slot+1]
        rank = (selected[:, :slot] == value).sum(-1, keepdim=True)
        equal_ids = candidates.masked_fill(raw_logits != value, raw_logits.shape[-1]).sort(-1).values
        ids[:, slot:slot+1] = equal_ids.gather(-1, rank)
    require(torch.equal(raw_logits.gather(-1, ids), selected), 'Tie substitution changed raw logits')
    require(torch.equal(probabilities.gather(-1, ids), probabilities.gather(-1, native_ids)),
            'Tie substitution changed native probabilities')
    if torch.equal(ids, native_ids):
        return out
    return probabilities, weights, ids


def router_observation(module, args, out, policy):
    # Pinned HF's router exposes probabilities, not raw logits. Repeat its exact
    # BF16 F.linear on the SAME producer input/weight for observation only; do
    # not replace its forward or recompute the returned route weights.
    raw = F.linear(args[0].reshape(-1, module.hidden_dim), module.weight)
    require(torch.equal(F.softmax(raw, dtype=torch.float32, dim=-1), out[0]),
            'Native router probability/observed raw-logit mismatch')
    require(policy in (ROUTER_NATIVE, ROUTER_PROFILE), 'Unknown explicit router tie policy')
    return raw, canonicalize_router_ties(raw, out) if policy == ROUTER_PROFILE else out


def router_contract(policy):
    require(policy in (ROUTER_NATIVE, ROUTER_PROFILE), 'Unknown router tie policy')
    return dict(policy=policy, native_policy=ROUTER_NATIVE,
        canonicalization='equal_raw_logits_only' if policy == ROUTER_PROFILE else 'none',
        unique_raw_logits='native selection unchanged, including softmax collisions',
        probabilities='native F32 full softmax unchanged', weights='native F32 selected renormalized weights unchanged',
        raw_logits='same-input BF16 F.linear observation, checked against native probabilities',
        unmodified_native_router=policy == ROUTER_NATIVE)


def initialize():
    global torch, F, hf, DynamicCache
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import DynamicCache, cache_utils, modeling_utils
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf
    require(torch.__version__ == VERSIONS['torch'] and transformers.__version__ == VERSIONS['transformers'],
            f'Pinned native BF16 dependencies required: {dict(VERSIONS)}; no installation/fallback')
    require(torch.version.cuda == '12.9', 'Pinned Torch CUDA build differs')
    modules = {'modeling_qwen3_5_moe': hf, 'cache_utils': cache_utils, 'modeling_utils': modeling_utils}
    for name, module in modules.items():
        require(sha256(module.__file__) == SOURCE_PINS[name], f'Pinned HF source differs: {name}')
    for symbol in ('causal_conv1d_fn', 'causal_conv1d_update', 'chunk_gated_delta_rule',
                   'fused_recurrent_gated_delta_rule', 'FusedRMSNormGated'):
        require(getattr(hf, symbol, None) is None, f'Accelerated HF path not admitted: {symbol}')
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    # Reuse storage/provenance and RoPE validation utilities, never the legacy
    # precision adapter, loader, model, manifest, comparator or initialization.
    legacy.torch = torch
    return {name: {'file': m.__file__, 'sha256': sha256(m.__file__)} for name, m in modules.items()}


def tiny_config():
    c = hf.Qwen3_5MoeTextConfig(vocab_size=32, hidden_size=16, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        layer_types=['linear_attention', 'full_attention'], linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=4, linear_value_head_dim=4,
        linear_conv_kernel_dim=4, moe_intermediate_size=16, shared_expert_intermediate_size=16,
        num_experts=4, num_experts_per_tok=3, rms_norm_eps=1e-6, tie_word_embeddings=False,
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000.,
            'partial_rotary_factor': .5, 'mrope_section': [1, 1, 0]})
    c._attn_implementation = c._experts_implementation = 'eager'
    return c


def f32_ssm_cache(config, devices, batch=1):
    cache = DynamicCache(config=config)
    for i, kind in enumerate(config.layer_types):
        if kind == 'linear_attention':
            state = cache.layers[i]
            state.recurrent_states = torch.zeros(batch, config.linear_num_value_heads,
                config.linear_key_head_dim, config.linear_value_head_dim, dtype=torch.float32, device=devices[i])
            state.is_recurrent_states_initialized = True
    return cache


def loading_evidence():
    """CPU execution of the pinned loader and native cache updates, not guesses.

    Deliberately does not rewrite the frozen tiny oracle used by the GPU owner.
    F32 source values are non-BF16; mixed native words must survive exact loading.
    """
    require(not torch.cuda.is_initialized(), 'Loading evidence must precede CUDA initialization')
    torch.manual_seed(3501635)
    config = tiny_config()
    source = hf.Qwen3_5MoeForCausalLM(config).eval()
    state = {}
    for name, value in source.state_dict().items():
        state[name] = value.detach().bfloat16()
        if name.endswith(('.A_log', '.linear_attn.norm.weight')):
            state[name] = torch.linspace(.12345, .45678, value.numel(), dtype=torch.float32).reshape(value.shape)
    with tempfile.TemporaryDirectory(prefix='qwen35-bf16-loading-evidence-') as directory:
        source.save_pretrained(directory, state_dict=state)
        loaded = hf.Qwen3_5MoeForCausalLM.from_pretrained(directory, dtype=torch.bfloat16,
            attn_implementation='eager', experts_implementation='eager', local_files_only=True).eval()
    report = dict(parameter_load={}, source_pins=dict(SOURCE_PINS), versions=dict(VERSIONS), device='cpu',
        loader='save_pretrained mixed native BF16/F32 state_dict -> from_pretrained(dtype=torch.bfloat16)',
        baseline=CONTRACT['baseline'])
    for name, parameter in loaded.named_parameters():
        raw = state[name]
        require(parameter.dtype == torch.bfloat16, f'Pinned loader parameter dtype: {name}')
        require(torch.equal(parameter, raw.bfloat16()), f'Pinned loader values differ: {name}')
        if raw.dtype == torch.float32:
            require(not torch.equal(raw, raw.bfloat16().float()), 'Witness must distinguish source F32 from BF16')
            report['parameter_load'][name] = dict(source_dtype=str(raw.dtype), loaded_dtype=str(parameter.dtype),
                source_values=raw.flatten().tolist(), loaded_values=parameter.float().flatten().tolist(),
                equals_source_bf16_rne=True, differs_from_source_f32=True)
        else:
            require(torch.equal(parameter.view(torch.int16), raw.view(torch.int16)), 'Native BF16 words changed')
    report['native_bf16_words_preserved'] = True
    report['rope'] = legacy.audit_rope_f32(loaded.model.rotary_emb)
    captures = []
    # Observe native recurrent input/output without substituting computation.
    from torch.overrides import TorchFunctionMode
    class Observe(TorchFunctionMode):
        def __torch_function__(self, function, types, args=(), kwargs=None):
            result = function(*args, **(kwargs or {}))
            if function.__name__ == 'exp' and args and isinstance(args[0], torch.Tensor):
                captures.append({'input_dtype': str(args[0].dtype), 'output_dtype': str(result.dtype)})
            return result
    report['cache_runs'] = {}
    for label in ('hf_default', 'explicit_f32_ssm'):
        cache = DynamicCache(config=config) if label == 'hf_default' else f32_ssm_cache(config, ['cpu']*2, batch=8)
        records = []
        with torch.inference_mode(), Observe():
            for tokens in ([1, 3, 2], [5]):
                result = loaded(torch.tensor([tokens]*8), past_key_values=cache, use_cache=True)
                gdn, gqa = cache.layers
                records.append(dict(tokens=tokens, conv=str(gdn.conv_states.dtype), ssm=str(gdn.recurrent_states.dtype),
                    kv=str(gqa.keys.dtype), logits=str(result.logits.dtype)))
        expected = 'torch.bfloat16' if label == 'hf_default' else 'torch.float32'
        require(all(r['ssm'] == expected and r['conv'] == r['kv'] == r['logits'] == 'torch.bfloat16' for r in records),
                'Pinned native cache behavior changed')
        report['cache_runs'][label] = records
    require(captures and all(c['input_dtype'] == 'torch.float32' for c in captures), 'Native exponential is F32 after loading')
    report['native_exp_observations'] = captures
    report['ssm_policy'] = CONTRACT['ssm']
    require(not torch.cuda.is_initialized(), 'CPU evidence initialized CUDA')
    return report


def dequantize_bf16(raw, scale):
    require(raw.ndim == 2 and raw.dtype == torch.float8_e4m3fn, 'Expected raw E4M3FN matrix')
    require(scale.dtype == torch.bfloat16 and tuple(scale.shape) == tuple((n+127)//128 for n in raw.shape), 'Expected paired BF16 block scales')
    require(bool(torch.isfinite(scale).all() and (scale > 0).all()), 'Invalid numeric scale')
    output = torch.empty(raw.shape, dtype=torch.bfloat16, device=raw.device)
    # Bounded transient F32 tile, not an entire F32 decoded matrix/grid.
    for row in range(0, raw.shape[0], 128):
        block = raw[row:row+128].float()
        require(bool(torch.isfinite(block).all()), 'Nonfinite FP8 payload')
        grid = scale[row//128].float().repeat_interleave(128)[:raw.shape[1]]
        output[row:row+128] = (block * grid).bfloat16()
    require(bool(torch.isfinite(output).all()), 'Decoded BF16 overflow')
    return output


class DeviceWeights:
    def __init__(self, checkpoint, device, capacity):
        self.checkpoint, self.device, self.capacity = checkpoint, torch.device(device), capacity
        self.cache, self.bytes, self.peak_bytes = OrderedDict(), 0, 0
    def matrix(self, name):
        if name in self.cache:
            self.cache.move_to_end(name)
            return self.cache[name]
        raw = self.checkpoint.read(name)
        size = raw.numel()*2
        while self.cache and self.bytes + size > self.capacity:
            self.bytes -= self.cache.popitem(last=False)[1].numel()*2
        if raw.dtype == torch.float8_e4m3fn:
            value = dequantize_bf16(raw.to(self.device), self.checkpoint.read(name+'_scale_inv').to(self.device))
        else:
            require(raw.dtype == torch.bfloat16, 'Native matrix must preserve BF16 checkpoint words')
            value = raw.to(self.device)
        if size <= self.capacity:
            self.cache[name] = value
            self.bytes += size
            self.peak_bytes = max(self.peak_bytes, self.bytes)
        return value
    def clear(self):
        self.cache.clear()
        self.bytes = 0


def bind(module, prefix, weights, config):
    class Linear(torch.nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name
        def forward(self, x):
            require(x.dtype == torch.bfloat16 and x.device == weights.device, 'Native BF16 linear input mismatch')
            output = F.linear(x, weights.matrix(self.name))
            require(output.dtype == torch.bfloat16, 'Native BF16 linear output mismatch')
            return output
    class Bank:
        def __init__(self, prefix, parts):
            self.prefix, self.parts = prefix, parts
        def __getitem__(self, index):
            index = int(index)
            require(0 <= index < config.num_experts, 'Expert index out of range')
            return torch.cat([weights.matrix(f'{self.prefix}.{index}.{part}.weight') for part in self.parts], dim=0)
    for name, child in list(module.named_modules()):
        full = '.'.join(p for p in (prefix, name) if p)
        if type(child) is hf.Qwen3_5MoeExperts:
            for attr, parts in [('gate_up_proj', ('gate_proj', 'up_proj')), ('down_proj', ('down_proj',))]:
                del child._parameters[attr]
                setattr(child, attr, Bank(legacy.source_name(full), parts))
        elif isinstance(child, torch.nn.Linear):
            require(child.bias is None and name, 'Unexpected bias/root linear')
            parent, _, attr = name.rpartition('.')
            setattr(module.get_submodule(parent), attr, Linear(legacy.source_name(full)+'.weight'))
    for name, parameter in list(module.named_parameters()):
        full = '.'.join(p for p in (prefix, name) if p)
        raw = weights.checkpoint.read(legacy.source_name(full))
        f32_gdn = full.endswith(('.linear_attn.A_log', '.linear_attn.norm.weight'))
        require(raw.shape == parameter.shape and (raw.dtype == torch.bfloat16 or (f32_gdn and raw.dtype == torch.float32)), f'Unexpected native parameter: {full}')
        require(bool(torch.isfinite(raw).all()), f'Nonfinite native parameter: {full}')
        parent, _, attr = name.rpartition('.')
        # This parameter-only loading rule is independently checked above. Do
        # not call model.to(BF16): rotary buffers must never pass through BF16.
        setattr(module.get_submodule(parent), attr, torch.nn.Parameter(raw.to(device=weights.device, dtype=torch.bfloat16), requires_grad=False))
    return module.eval()


def read_config(directory):
    raw = json.loads((directory/'config.json').read_text())
    require(raw['model_type'] == 'qwen3_5_moe' and raw['quantization_config']['quant_method'] == 'fp8'
        and raw['quantization_config']['weight_block_size'] == [128,128], 'Wrong checkpoint encoding')
    c = hf.Qwen3_5MoeTextConfig(**dict(raw['text_config'], dtype='bfloat16', tie_word_embeddings=False))
    require(c.num_hidden_layers == 40 and c.num_experts == 256 and c.vocab_size == 248320, 'Expected actual 35B full40 text model')
    require(c.layer_types == (['linear_attention']*3+['full_attention'])*10, 'Wrong hybrid layers')
    c._attn_implementation = c._experts_implementation = 'eager'
    return c


def build_model(config, weights):
    require(len(weights) == 8, 'Whole-35B oracle requires exactly eight devices, five layers each')
    with torch.device('meta'):
        model = hf.Qwen3_5MoeForCausalLM(config)
    placement = [weights[i//5].device for i in range(40)]
    for i, layer in enumerate(model.model.layers):
        bind(layer, f'model.layers.{i}', weights[i//5], config)
        layer.register_forward_pre_hook(legacy.move_inputs(placement[i]), with_kwargs=True)
    bind(model.model.embed_tokens, 'model.embed_tokens', weights[0], config)
    bind(model.model.norm, 'model.norm', weights[-1], config)
    model.model.norm.register_forward_pre_hook(legacy.move_inputs(weights[-1].device), with_kwargs=True)
    # Root LM head is lazy too, so loading never expands all routed experts.
    holder = torch.nn.Module()
    holder.lm_head = model.lm_head
    bind(holder, '', weights[-1], config)
    model.lm_head = holder.lm_head
    model.model.rotary_emb = hf.Qwen3_5MoeTextRotaryEmbedding(config, device=weights[0].device)
    legacy.audit_rope_f32(model.model.rotary_emb)
    require(all(p.device.type == 'cuda' and p.dtype == torch.bfloat16 for p in model.parameters()), 'Unmaterialized/non-BF16 parameter')
    require(all(b.device.type == 'cuda' for b in model.buffers()), 'Meta/CPU buffer remains')
    for layer in model.model.layers:
        if layer.layer_type == 'linear_attention':
            g = layer.linear_attn
            require(g.chunk_gated_delta_rule is hf.torch_chunk_gated_delta_rule and g.recurrent_gated_delta_rule is hf.torch_recurrent_gated_delta_rule, 'Native GDN changed')
    return model.eval(), placement


class Capture(legacy.Capture):
    def __init__(self, max_bytes, router_policy):
        super().__init__()
        self.bytes, self.max_bytes = 0, max_bytes
        self.router_policy = router_policy
    def put(self, key, value):
        require(value.device.type == 'cuda', f'Non-CUDA boundary: {key}')
        expected = torch.int64 if key.endswith(('input_ids','position_ids','cache_position','attention_mask','selected_expert_ids','native_expert_ids')) else (
            torch.float32 if '.recurrent_states.' in key or key.endswith(('routing_weights', 'router_probabilities')) else torch.bfloat16)
        require(value.dtype == expected, f'Native dtype mismatch: {key}: {value.dtype}, expected {expected}')
        require(not value.is_floating_point() or bool(torch.isfinite(value).all()), f'Nonfinite boundary: {key}')
        size = value.numel()*value.element_size()
        require(self.bytes + size <= self.max_bytes, 'Host capture byte cap exceeded')
        full = f'{self.stage}.{key}'
        require(full not in self.tensors, f'Duplicate boundary: {full}')
        self.tensors[full] = value.detach().cpu().contiguous().clone()
        self.bytes += size
    def layer(self, layer, index):
        prefix = f'layers.{index:02d}'
        def route(module, args, result):
            raw, selected = router_observation(module, args, result, self.router_policy)
            self.put(prefix + '.raw_router_logits', raw)
            self.put(prefix + '.router_probabilities', result[0])
            self.put(prefix + '.native_expert_ids', result[2])
            return selected
        # The inherited observer runs afterwards and records the actual selected
        # IDs consumed by HF experts; the unmodified native IDs remain separate.
        self.handles.append(layer.mlp.gate.register_forward_hook(route))
        super().layer(layer, index)

    def cache(self, cache, indices):
        for i in indices:
            layer = cache.layers[i]
            for source, field in [('conv_states','conv_states'), ('recurrent_states','recurrent_states'), ('keys','key_cache'), ('values','value_cache')]:
                value = getattr(layer, source, None)
                if value is not None:
                    self.put(f'cache.{field}.{i:02d}', value)


def full_case(model, config, placement, output, name, ids, max_bytes, router_policy):
    from safetensors.torch import save_file
    capture = Capture(max_bytes, router_policy)
    cache = f32_ssm_cache(config, placement)
    capture.observe(model.model.embed_tokens, 'embedding')
    capture.observe(model.model.norm, 'final_norm')
    for i, layer in enumerate(model.model.layers): capture.layer(layer, i)
    calls, fed = [], ids
    try:
        with torch.inference_mode():
            for stage in ('prefill','decode.0'):
                capture.stage = stage
                start = 0 if stage == 'prefill' else len(ids)
                inputs = torch.tensor([fed], device=placement[0], dtype=torch.int64)
                position = torch.arange(start,start+len(fed), device=placement[0])
                mask = torch.ones((1,start+len(fed)), device=placement[0], dtype=torch.int64)
                result = model(input_ids=inputs, position_ids=position[None], attention_mask=mask,
                    past_key_values=cache, use_cache=True, logits_to_keep=0, return_dict=True)
                require(result.past_key_values is cache, 'Unexpected cache replacement')
                for key,value in [('input_ids',inputs),('position_ids',position[None]),('cache_position',position),('attention_mask',mask),('logits',result.logits)]: capture.put(key,value)
                capture.cache(cache, range(40))
                predicted = int(result.logits[0,-1].argmax())
                require(cache.get_seq_length() == start+len(fed), 'Cache frontier mismatch')
                calls.append(dict(stage=stage,input_token_ids=fed,position_start=start,sequence_length_after=start+len(fed),next_token_id=predicted))
                fed = [predicted]
        expected = expected_tensors(config.to_dict(), ids)
        require(capture.tensors.keys() == expected.keys(), 'Full40 boundary coverage incomplete')
        for key,(shape,_) in expected.items(): require(tuple(capture.tensors[key].shape)==shape, f'Shape mismatch: {key}')
        path = output/f'{name}.safetensors'
        save_file(capture.tensors,str(path),metadata={'schema':PROFILE_SCHEMA if router_policy == ROUTER_PROFILE else SCHEMA,'profile':PROFILE,'model_boundary':'bf16','router_tie_policy':router_policy})
        report = dict(id=name,file=path.name,sha256=sha256(path),calls=calls,profile=PROFILE,router=router_contract(router_policy),
            tensors={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in capture.tensors.items()})
        write_json(output/f'{name}.json',report)
        return dict(id=name,file=path.name,sha256=sha256(path),report=f'{name}.json',report_sha256=sha256(output/f'{name}.json'))
    finally: capture.close()


def expected_tensors(config, ids):
    expected = legacy.expected_tensors(config, ids)
    for key, (shape, dtype) in list(expected.items()):
        if dtype == 'F32' and '.recurrent_states.' not in key and not key.endswith('routing_weights'):
            expected[key] = (shape, 'BF16')
    for stage, rows in [('prefill', len(ids)), ('decode.0', 1)]:
        for index in range(config['num_hidden_layers']):
            prefix = f'{stage}.layers.{index:02d}'
            expected[prefix+'.raw_router_logits'] = ((rows, config['num_experts']), 'BF16')
            expected[prefix+'.router_probabilities'] = ((rows, config['num_experts']), 'F32')
            expected[prefix+'.native_expert_ids'] = ((rows, config['num_experts_per_tok']), 'I64')
    return expected


def run(args, manifest):
    require(args.model_boundary == ModelBoundary.BF16.value, 'Explicit BF16 ModelBoundary required')
    require(args.ssm_policy == 'f32', 'Explicit F32 SSM policy required')
    manifest['native_source'] = initialize()
    evidence = loading_evidence()
    write_json(args.output/'loading-semantics.json',evidence)
    manifest['loading_evidence'] = dict(file='loading-semantics.json',sha256=sha256(args.output/'loading-semantics.json'))
    config = read_config(args.model)
    manifest['checkpoint'] = dict(directory=str(args.model.resolve()),config_sha256=sha256(args.model/'config.json'),effective_text_config=config.to_dict())
    manifest['tokenizer_files'] = {name:sha256(args.model/name) for name in ('tokenizer.json','tokenizer_config.json')}
    manifest['model_loading'] = 'meta native HF skeleton; lazy per-matrix actual checkpoint reads; parameter-only BF16 materialization checked against pinned from_pretrained; no blanket model.to; native F32 RoPE rebuilt'
    require(not torch.cuda.is_initialized(),'Preflight initialized CUDA')
    if not args.execute_cuda:
        manifest.update(status='preflight_only',full_reference_complete=False,reference_kind='not_executed')
        return
    if args.router_tie_policy == ROUTER_PROFILE:
        require(args.native_reference_dir is not None and args.native_reference_sha256 is not None,
                'Profile export requires a separately completed immutable native reference and SHA256')
        native_path = args.native_reference_dir/'manifest.json'
        require(sha256(native_path) == args.native_reference_sha256, 'Native reference manifest pin mismatch')
        native = json.loads(native_path.read_text())
        require(native['schema'] == SCHEMA and native['router']['policy'] == ROUTER_NATIVE
            and native['status'] == 'complete' and native['full_reference_complete'] is True,
            'Native reference is not a complete unmodified-router BF16 run')
        require(native['checkpoint']['config_sha256'] == manifest['checkpoint']['config_sha256']
            and native['versions'] == manifest['versions'] and native['tokenizer_files'] == manifest['tokenizer_files'],
            'Native/profile source mismatch')
        manifest['native_reference'] = dict(directory=str(args.native_reference_dir.resolve()),
            manifest_sha256=args.native_reference_sha256, router_policy=ROUTER_NATIVE,
            purpose='separate unmodified native router baseline; own autoregressive trajectory, not replaced')
    devices = args.devices.split(',')
    require(len(devices)==8 and len(set(devices))==8 and all(d.startswith('cuda:') and d[5:].isdigit() for d in devices),'Eight distinct CUDA ordinals required')
    manifest['devices'] = []
    for d in devices:
        free,total = torch.cuda.mem_get_info(d)
        require(free >= (2*1024+args.cache_mib)*1024**2, f'{d}: need 2 GiB work/native/headroom plus cache')
        manifest['devices'].append(dict(device=d,free_before=free,total=total))
        torch.cuda.reset_peak_memory_stats(d)
    checkpoint = legacy.Checkpoint(args.model)
    weights = [DeviceWeights(checkpoint,d,args.cache_mib*1024**2) for d in devices]
    model = None
    try:
        model, placement = build_model(config, weights)
        manifest['layer_devices'] = {str(i):str(d) for i,d in enumerate(placement)}
        manifest['rope_audit'] = legacy.audit_rope_f32(model.model.rotary_emb)
        for name,ids in legacy.CASES.items():
            manifest['cases'].append(full_case(model,config,placement,args.output,name,ids,args.capture_mib*1024**2,args.router_tie_policy))
        manifest['execution']['device'] = 'cuda'
        manifest.update(status='complete',reference_kind='independent_profile_bf16_full_model' if args.router_tie_policy == ROUTER_PROFILE else 'independent_native_bf16_full_model',full_reference_complete=True)
    finally:
        write_json(args.output/'source-payloads.json',checkpoint.provenance())
        manifest['source_payloads'] = dict(file='source-payloads.json',sha256=sha256(args.output/'source-payloads.json'))
        for info,w in zip(manifest['devices'],weights):
            info.update(peak_allocated=torch.cuda.max_memory_allocated(w.device),peak_reserved=torch.cuda.max_memory_reserved(w.device),peak_operand_cache=w.peak_bytes)
            w.clear()
        del model
        gc.collect()
        torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,default=Path('/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model-boundary',choices=['bf16'],required=True)
    p.add_argument('--ssm-policy',choices=['f32'],required=True)
    p.add_argument('--router-tie-policy',choices=[ROUTER_NATIVE,ROUTER_PROFILE],default=ROUTER_NATIVE)
    p.add_argument('--native-reference-dir',type=Path)
    p.add_argument('--native-reference-sha256')
    p.add_argument('--execute-cuda',action='store_true')
    p.add_argument('--devices',default='cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7')
    p.add_argument('--cache-mib',type=int,default=256,choices=range(1,1025))
    p.add_argument('--capture-mib',type=int,default=512,choices=range(128,2049))
    p.add_argument('--timeout',type=int,default=1800,choices=range(1,1801))
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',
        TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8',ATEN_CPU_CAPABILITY='avx512')
    if not args.execute_cuda: os.environ['CUDA_VISIBLE_DEVICES']=''
    if not args.worker:
        args.output.mkdir(parents=True,exist_ok=False)
        from test_vllm_qwen35 import run_process
        result = run_process([sys.executable,'-B',__file__,*sys.argv[1:],'--worker'], args.timeout)
        (args.output/'run.log').write_text(result['stdout'] + result['stderr'])
        write_json(args.output/'custody.json', result)
        if result.get('error') or result['custody'].get('status') != 'proven_drained' or not result['custody'].get('echild'):
            return 1
        return result['returncode'] if result['returncode'] is not None else 124
    manifest=dict(schema=PROFILE_SCHEMA if args.router_tie_policy == ROUTER_PROFILE else SCHEMA,profile=PROFILE,
        router=router_contract(args.router_tie_policy),precision_contract=dict(CONTRACT),versions=dict(VERSIONS),
        script_sha256=sha256(__file__),storage_utility_sha256=sha256(legacy.__file__),arguments=vars(args),
        status='incomplete',reference_kind='not_executed',full_reference_complete=False,strict_ferrule_full40=False,
        execution=dict(device='not_run',model_boundary=args.model_boundary,tf32=False,production_imports=False,reduction_emulation=False),cases=[])
    started=time.monotonic()
    try:
        run(args,manifest)
    except Exception as error:
        manifest.update(status='incomplete',full_reference_complete=False,error=f'{type(error).__name__}: {error}')
        raise
    finally:
        manifest['elapsed_seconds']=time.monotonic()-started
        # A run never overwrites its immutable final manifest; progress/failure
        # evidence uses other files. Failed/partial reference is never complete.
        path=args.output/'manifest.json'
        with path.open('x') as f: json.dump(manifest,f,indent=2,default=str,allow_nan=False)
        (args.output/'manifest.sha256').write_text(sha256(path)+'\n')
    return 0

if __name__=='__main__': raise SystemExit(main())
