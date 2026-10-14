#!/usr/bin/env python3
"""Offline native HF CUDA oracle; no production imports, kernels or reduction replay.

Requires the *existing* torch 2.13.0+cu129 / Transformers 5.2.0 environment.
Without --execute-cuda only static preflight runs. Full model uses native HF
forwards with lazy checkpoint operands, not an all-BF16 from_pretrained load.
Full-model failures remain incomplete; there is no replay or CPU fallback.
See qwen35_fp8_cuda_reference.md for the artifact contract.
"""
import argparse
from collections import OrderedDict
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

SCHEMA = 'ferrule.qwen35-fp8-cuda-reference.v1'
REQUIRED_VERSIONS = {'torch': '2.13.0+cu129', 'transformers': '5.2.0'}
CASES = {'hello': [9419], 'capital': [760, 6511, 314, 9338, 369]}
STAGES = ('prefill', 'decode.0')
CACHE_FIELDS = ('conv_states', 'recurrent_states', 'key_cache', 'value_cache')
CONTRACT = dict(profile='bf16_rne', fp8='E4M3FN', scale='numeric BF16 multiplicative 128x128',
                weight='F32 product -> BF16 RNE -> F32', activation='BF16 RNE -> F32 for quantized linear only',
                accumulation='F32', output='F32', boundary='F32', kv='F32', gdn='F32', tf32=False)
NATIVE_SHA256 = 'b3001ab72dbddce268207a5771365e09961505e5e9fd092244fa17d362f57da5'
GIB = 1024 ** 3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + '\n')
    temporary.replace(path)


def dependency_versions():
    versions = {}
    for name in (*REQUIRED_VERSIONS, 'safetensors', 'tokenizers', 'numpy'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def validate_dependencies(versions):
    for name, version in REQUIRED_VERSIONS.items():
        require(versions.get(name) == version,
                f'Exact existing dependency required: {name}=={version}; found {versions.get(name)!r}; no install permitted')
    require(all(versions.get(n) for n in ('safetensors', 'tokenizers', 'numpy')), 'Missing existing capture dependency')


def initialize():
    global torch, F, q
    validate_dependencies(dependency_versions())
    import torch
    import torch.nn.functional as F
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as q
    import transformers
    require(transformers.__version__ == REQUIRED_VERSIONS['transformers'], 'Wrong imported Transformers version')
    require(torch.__version__ == REQUIRED_VERSIONS['torch'] and torch.version.cuda == '12.9', 'Wrong imported Torch/CUDA build')
    require(sha256(q.__file__) == NATIVE_SHA256, 'Native HF source differs from the audited, frozen 5.2.0 source')
    for symbol in ('causal_conv1d_fn', 'causal_conv1d_update', 'chunk_gated_delta_rule',
                   'fused_recurrent_gated_delta_rule', 'FusedRMSNormGated'):
        require(getattr(q, symbol, None) is None, f'Unexpected accelerated HF path: {symbol}')
    torch.set_num_threads(8)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return dict(file=q.__file__, sha256=sha256(q.__file__))


def dequantize(weight, scale):
    """Documented checkpoint block format, not backend address/reduction helpers."""
    require(weight.ndim == 2 and weight.dtype == torch.float8_e4m3fn, 'Expected E4M3FN matrix')
    require(scale.dtype == torch.bfloat16 and tuple(scale.shape) == tuple(math.ceil(n / 128) for n in weight.shape),
            'Expected BF16 numeric block scales')
    require(bool(torch.isfinite(scale).all() and (scale > 0).all()), 'Nonpositive/nonfinite scale')
    values = weight.float()
    require(bool(torch.isfinite(values).all()), 'Nonfinite FP8 weight')
    # Expand the public block grid before cropping the partial edge blocks.
    grid = scale.float().repeat_interleave(128, 0).repeat_interleave(128, 1)
    result = (values * grid[:weight.shape[0], :weight.shape[1]]).bfloat16().float()
    require(bool(torch.isfinite(result).all()), 'Dequantized weight overflow')
    return result


class Checkpoint:
    """Read one tensor at a time; hash every actually used payload, never vendor code."""
    def __init__(self, directory):
        self.directory = directory.resolve()
        self.index_path = directory / 'model.safetensors.index.json'
        self.index_hash = sha256(self.index_path)
        self.mapping = json.loads(self.index_path.read_text())['weight_map']
        self.payloads = {}
        self.shards = {}

    def read(self, name):
        from safetensors import safe_open
        path = (self.directory / self.mapping[name]).resolve()
        require(path.parent == self.directory, 'Checkpoint shard escapes model directory')
        stat = path.stat()
        identity = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        require(self.shards.get(path.name, identity) == identity, 'Checkpoint shard changed during run')
        self.shards[path.name] = identity
        with safe_open(path, framework='pt', device='cpu') as shard:
            value = shard.get_tensor(name).clone()
        require(value.dtype in (torch.float8_e4m3fn, torch.bfloat16, torch.float32), f'Unsupported storage: {name}')
        digest = hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        entry = dict(sha256=digest, shape=list(value.shape), dtype=str(value.dtype), shard=path.name)
        require(self.payloads.get(name, entry) == entry, f'Checkpoint payload changed: {name}')
        self.payloads[name] = entry
        return value

    def provenance(self):
        require(sha256(self.index_path) == self.index_hash, 'Checkpoint index changed')
        return dict(index_sha256=self.index_hash, shards=self.shards, used_payloads=self.payloads,
                    scope='Only actually read tensors; unused experts, vision and MTP are not certified')


class DeviceWeights:
    def __init__(self, checkpoint, device, capacity):
        self.checkpoint, self.device, self.capacity = checkpoint, torch.device(device), capacity
        self.cache = OrderedDict()
        self.bytes = 0

    def matrix(self, name):
        if name in self.cache:
            self.cache.move_to_end(name)
            return self.cache[name]
        raw = self.checkpoint.read(name)
        quantized = raw.dtype == torch.float8_e4m3fn
        size = raw.numel() * 4
        while self.cache and self.bytes + size > self.capacity:
            old, _ = self.cache.popitem(last=False)[1]
            self.bytes -= old.numel() * 4
        value = raw.to(self.device)
        value = dequantize(value, self.checkpoint.read(name + '_scale_inv').to(self.device)) if quantized else value.float()
        require(bool(torch.isfinite(value).all()), f'Nonfinite operand: {name}')
        if size <= self.capacity:
            self.cache[name] = (value, quantized)
            self.bytes += size
        return value, quantized

    def clear(self):
        self.cache.clear()
        self.bytes = 0


def binding_types():
    class RoundedWeight(torch.Tensor):
        @staticmethod
        def wrap(value):
            return torch.Tensor._make_subclass(RoundedWeight, value, False)

        @classmethod
        def __torch_function__(cls, function, types, args=(), kwargs=None):
            if function is F.linear:
                x, w, *rest = args
                return F.linear(x.bfloat16().float(), w.as_subclass(torch.Tensor), *rest, **(kwargs or {}))
            return super().__torch_function__(function, types, args, kwargs or {})

    class Linear(torch.nn.Module):
        def __init__(self, weights, name):
            super().__init__()
            self.weights, self.name = weights, name

        def forward(self, x):
            require(x.dtype == torch.float32 and x.device == self.weights.device, 'Linear boundary/device mismatch')
            w, quantized = self.weights.matrix(self.name)
            return F.linear(x.bfloat16().float() if quantized else x, w)

    class Bank:
        def __init__(self, weights, prefix, parts, count):
            self.weights, self.prefix, self.parts, self.count = weights, prefix, parts, count

        def __getitem__(self, index):
            index = int(index)
            require(0 <= index < self.count, 'Expert index out of range')
            matrices = [self.weights.matrix(f'{self.prefix}.{index}.{part}.weight') for part in self.parts]
            require(all(quantized for _, quantized in matrices), 'Expected FP8 expert operands')
            return RoundedWeight.wrap(torch.cat([w for w, _ in matrices], dim=0))

    return Linear, Bank


def source_name(name):
    return 'model.language_model.' + name[len('model.'):] if name.startswith('model.') else name


def bind(module, prefix, weights, config):
    """Only bind parameters/linear operands; leave all native HF family forwards intact."""
    Linear, Bank = binding_types()
    for name, child in list(module.named_modules()):
        full = '.'.join(p for p in (prefix, name) if p)
        if type(child) is q.Qwen3_5MoeExperts:
            for attribute, parts in (('gate_up_proj', ('gate_proj', 'up_proj')), ('down_proj', ('down_proj',))):
                del child._parameters[attribute]
                setattr(child, attribute, Bank(weights, source_name(full), parts, config.num_experts))
        elif isinstance(child, torch.nn.Linear):
            require(child.bias is None, f'Unexpected linear bias: {full}')
            parent, _, attribute = name.rpartition('.')
            setattr(module.get_submodule(parent), attribute, Linear(weights, source_name(full) + '.weight'))
    for name, parameter in list(module.named_parameters()):
        full = '.'.join(p for p in (prefix, name) if p)
        value = weights.checkpoint.read(source_name(full))
        require(value.dtype in (torch.bfloat16, torch.float32) and value.shape == parameter.shape,
                f'Unbound quantized parameter or wrong shape: {full}')
        require(bool(torch.isfinite(value).all()), f'Nonfinite parameter: {full}')
        parent, _, attribute = name.rpartition('.')
        setattr(module.get_submodule(parent), attribute, torch.nn.Parameter(value.float().to(weights.device), requires_grad=False))
    for child in module.modules():
        if isinstance(child, q.Qwen3_5MoeGatedDeltaNet):
            require(child.chunk_gated_delta_rule is q.torch_chunk_gated_delta_rule
                    and child.recurrent_gated_delta_rule is q.torch_recurrent_gated_delta_rule
                    and child.causal_conv1d_update is q.torch_causal_conv1d_update, 'Non-native GDN implementation')
    return module.eval()


def read_config(directory):
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    raw = json.loads((directory / 'config.json').read_text())
    require(raw['model_type'] == 'qwen3_5_moe' and raw['quantization_config']['quant_method'] == 'fp8'
            and raw['quantization_config']['weight_block_size'] == [128, 128], 'Wrong model/quantization format')
    config = Qwen3_5MoeTextConfig(**dict(raw['text_config'], dtype='float32', tie_word_embeddings=False))
    require(config.num_hidden_layers == 40 and config.num_experts == 256 and config.vocab_size == 248320,
            'Expected native 35B full40 text checkpoint')
    require(config.layer_types == (['linear_attention'] * 3 + ['full_attention']) * 10, 'Wrong 30 GDN / 10 attention layout')
    config._attn_implementation = 'eager'
    config._experts_implementation = 'eager'
    return config


def move_inputs(device):
    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, tuple):
            return tuple(move(v) for v in value)
        return value

    def hook(_module, args, kwargs):
        return tuple(move(v) for v in args), {k: move(v) for k, v in kwargs.items()}
    return hook



def audit_rope_f32(rope):
    """Reject BF16-then-F32 frequency loss, not merely a BF16 buffer dtype."""
    for name in ('inv_freq', 'original_inv_freq'):
        value=getattr(rope,name)
        require(value.dtype==torch.float32, 'RoPE frequency buffer must stay F32: '+name)
    expected,_=rope.compute_default_rope_parameters(rope.config,rope.inv_freq.device)
    require(torch.equal(rope.inv_freq,expected) and torch.equal(rope.original_inv_freq,expected),
            'RoPE frequencies differ from native F32 construction; possible blanket BF16 cast')
    return dict(inv_freq_dtype=str(rope.inv_freq.dtype),original_inv_freq_dtype=str(rope.original_inv_freq.dtype),
                native_f32_exact=True,inv_freq_sha256=hashlib.sha256(rope.inv_freq.detach().cpu().numpy().tobytes()).hexdigest())


def build_model(config, weights):
    with torch.device('meta'):
        model = q.Qwen3_5MoeForCausalLM(config)
    # Contiguous layer groups; native cache is already per-layer/device aware.
    placement = {i: weights[min(i * len(weights) // config.num_hidden_layers, len(weights)-1)] for i in range(40)}
    for i, layer in enumerate(model.model.layers):
        bind(layer, f'model.layers.{i}', placement[i], config)
        layer.register_forward_pre_hook(move_inputs(placement[i].device), with_kwargs=True)
    bind(model.model.embed_tokens, 'model.embed_tokens', weights[0], config)
    bind(model.model.norm, 'model.norm', weights[-1], config)
    model.model.norm.register_forward_pre_hook(move_inputs(weights[-1].device), with_kwargs=True)
    Linear, _ = binding_types()
    model.lm_head = Linear(weights[-1], 'lm_head.weight')
    model.model.rotary_emb = q.Qwen3_5MoeTextRotaryEmbedding(config, device=weights[0].device)
    audit_rope_f32(model.model.rotary_emb)
    require(all(p.device.type == 'cuda' and p.dtype == torch.float32 for p in model.parameters()), 'Non-CUDA/F32 parameter')
    require(all(b.device.type == 'cuda' for b in model.buffers()), 'Unmaterialized buffer')
    return model.eval(), {str(i): str(w.device) for i, w in placement.items()}


class Capture:
    def __init__(self):
        self.stage, self.tensors, self.handles = '', {}, []

    def put(self, key, value):
        require(value.device.type == 'cuda', f'Non-CUDA capture: {key}')
        require(value.dtype in (torch.float32, torch.int64), f'Boundary/cache must be F32 or I64: {key}')
        require(not value.is_floating_point() or bool(torch.isfinite(value).all()), f'Nonfinite capture: {key}')
        self.tensors[f'{self.stage}.{key}'] = value.detach().cpu().contiguous().clone()

    def observe(self, module, key):
        self.handles.append(module.register_forward_hook(
            lambda _m, _a, result: self.put(key, result[0] if isinstance(result, tuple) else result)))

    def layer(self, layer, index):
        prefix = f'layers.{index:02d}'
        for module, key in ((layer, 'hidden'), (layer.mlp, 'moe'),
                            (layer.mlp.shared_expert_gate, 'shared_gate_logit'),
                            (layer.linear_attn if layer.layer_type == 'linear_attention' else layer.self_attn, 'attention')):
            self.observe(module, prefix + '.' + key)

        def route(_module, _args, result):
            self.put(prefix + '.selected_expert_ids', result[2])
            self.put(prefix + '.routing_weights', result[1])
        self.handles.append(layer.mlp.gate.register_forward_hook(route))

    def cache(self, cache, indices):
        for field in CACHE_FIELDS:
            for index in indices:
                value = getattr(cache, field)[index]
                if value is not None:
                    self.put(f'cache.{field}.{index:02d}', value)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def expected_tensors(config, ids):
    """Shared fixed-case tensor inventory, available without importing Torch."""
    c = config
    result = {}
    for stage, length, start in (('prefill', len(ids), 0), ('decode.0', 1, len(ids))):
        def add(key, shape, dtype='F32'):
            result[f'{stage}.{key}'] = (tuple(shape), dtype)
        add('input_ids', (1, length), 'I64')
        add('position_ids', (1, length), 'I64')
        add('cache_position', (length,), 'I64')
        add('attention_mask', (1, start + length), 'I64')
        add('logits', (1, length, c['vocab_size']))
        for name in ('embedding', 'final_norm'):
            add(name, (1, length, c['hidden_size']))
        for i, kind in enumerate(c['layer_types']):
            prefix = f'layers.{i:02d}'
            for name in ('hidden', 'attention', 'moe'):
                add(prefix + '.' + name, (1, length, c['hidden_size']))
            add(prefix + '.shared_gate_logit', (length, 1))
            add(prefix + '.selected_expert_ids', (length, c['num_experts_per_tok']), 'I64')
            add(prefix + '.routing_weights', (length, c['num_experts_per_tok']))
            if kind == 'linear_attention':
                channels = 2*c['linear_num_key_heads']*c['linear_key_head_dim'] + c['linear_num_value_heads']*c['linear_value_head_dim']
                add(f'cache.conv_states.{i:02d}', (1, channels, c['linear_conv_kernel_dim']))
                add(f'cache.recurrent_states.{i:02d}', (1, c['linear_num_value_heads'], c['linear_key_head_dim'], c['linear_value_head_dim']))
            else:
                for field in ('key_cache', 'value_cache'):
                    add(f'cache.{field}.{i:02d}', (1, c['num_key_value_heads'], start+length, c['head_dim']))
    return result


def save_capture(output, name, capture, calls, ids):
    from safetensors.torch import save_file
    path = output / f'{name}.safetensors'
    save_file(capture.tensors, str(path), metadata={'schema': SCHEMA})
    report = dict(id=name, request_token_ids=ids, calls=calls, file=path.name, sha256=sha256(path),
                  tensors={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in capture.tensors.items()})
    write_json(output / f'{name}.json', report)
    entry = dict(id=name, file=path.name, sha256=report['sha256'], report=f'{name}.json',
                 report_sha256=sha256(output / f'{name}.json'))
    return entry


def full_case(model, config, device, output, name, ids):
    capture, calls, cache = Capture(), [], None
    capture.observe(model.model.embed_tokens, 'embedding')
    capture.observe(model.model.norm, 'final_norm')
    for index, layer in enumerate(model.model.layers):
        capture.layer(layer, index)
    fed = ids
    try:
        for stage in STAGES:
            capture.stage = stage
            start = 0 if stage == 'prefill' else len(ids)
            inputs = torch.tensor([fed], dtype=torch.int64, device=device)
            position = torch.arange(start, start+len(fed), device=device)
            mask = torch.ones((1, start+len(fed)), dtype=torch.int64, device=device)
            with torch.inference_mode():
                result = model(input_ids=inputs, position_ids=position[None], cache_position=position,
                               attention_mask=mask, past_key_values=cache, use_cache=True, logits_to_keep=0, return_dict=True)
            cache = result.past_key_values
            for key, value in (('input_ids', inputs), ('position_ids', position[None]), ('cache_position', position),
                               ('attention_mask', mask), ('logits', result.logits)):
                capture.put(key, value)
            capture.cache(cache, range(40))
            predicted = int(result.logits[0, -1].argmax())
            require(cache.get_seq_length() == start+len(fed), 'Wrong cached continuation length')
            calls.append(dict(stage=stage, input_token_ids=fed, position_start=start,
                              sequence_length_after=start+len(fed), next_token_id=predicted,
                              cache_source=None if stage == 'prefill' else 'prefill'))
            fed = [predicted]
        expected = expected_tensors(config.to_dict(), ids)
        require(capture.tensors.keys() == expected.keys(), 'Incomplete full40 capture')
        for key, (shape, dtype) in expected.items():
            value = capture.tensors[key]
            require(tuple(value.shape) == shape and value.dtype == (torch.float32 if dtype == 'F32' else torch.int64),
                    f'Capture dtype/shape mismatch: {key}')
        return save_capture(output, name, capture, calls, ids)
    finally:
        capture.close()




def gpu_admission(devices):
    require(torch.cuda.is_available(), 'CUDA unavailable in the exact existing environment')
    result = []
    for device in devices:
        free, total = torch.cuda.mem_get_info(device)
        result.append(dict(device=device, name=torch.cuda.get_device_name(device), free=free, total=total))
    return result


def run(args, manifest):
    manifest['versions'] = dependency_versions()
    validate_dependencies(manifest['versions'])
    manifest['native_source'] = initialize()
    config = read_config(args.model)
    config_hash = sha256(args.model / 'config.json')
    manifest['checkpoint'] = dict(directory=str(args.model.resolve()), config_sha256=config_hash,
                                  effective_text_config=config.to_dict())
    manifest['tokenizer_files'] = {n: sha256(args.model / n) for n in ('tokenizer.json', 'tokenizer_config.json')}
    require(not torch.cuda.is_initialized(), 'Static preflight initialized CUDA')
    if not args.execute_cuda:
        manifest['diagnostics'].append('GPU execution not authorized: model loading, memory admission and forwards NOT attempted')
        manifest['preflight'] = 'passed'
        return
    devices = args.devices.split(',')
    require(len(devices) == len(set(devices)) and all(d.startswith('cuda:') and d[5:].isdigit() for d in devices),
            'Use distinct explicit CUDA devices, e.g. cuda:0,cuda:1')
    manifest['devices'] = gpu_admission(devices)
    checkpoint = Checkpoint(args.model)
    weights = [DeviceWeights(checkpoint, d, args.cache_mib*1024**2) for d in devices]
    manifest['execution'].update(device='cuda', devices=devices)
    write_json(args.output / 'manifest.json', manifest)
    model = None
    try:
        for gpu in manifest['devices']:
            require(gpu['free'] >= 8*GIB + args.cache_mib*1024**2,
                    f"{gpu['device']}: full-model admission needs 8 GiB + operand cache; free={gpu['free']}")
        model, placement = build_model(config, weights)
        manifest['layer_devices'] = placement
        manifest['rope_audit'] = audit_rope_f32(model.model.rotary_emb)
        manifest['model_loading'] = 'native HF text model; lazy CUDA operands, no CPU compute offload; loaded'
        for name, ids in CASES.items():
            if getattr(args, 'case', 'both') not in ('both', name):
                continue
            manifest['cases'].append(full_case(model, config, weights[0].device, args.output, name, ids))
            write_json(args.output / 'manifest.json', manifest)
    except Exception as error:
        manifest.update(status='incomplete', reference_kind='incomplete', full_reference_complete=False)
        manifest['diagnostics'].append(f'Full native CUDA model failed: {type(error).__name__}: {error}')
        manifest['model_loading'] = 'failed or forward/capture incomplete; see diagnostics'
        raise
    finally:
        del model
        for w in weights:
            w.clear()
        gc.collect()
        torch.cuda.empty_cache()
        provenance = args.output / 'source-payloads.json'
        write_json(provenance, checkpoint.provenance())
        manifest['source_payloads'] = dict(file=provenance.name, sha256=sha256(provenance))
    manifest.update(status='complete', reference_kind='independent_cuda_full_model', full_reference_complete=True)


def new_manifest(args):
    return dict(schema=SCHEMA, status='incomplete', reference_kind='incomplete', full_reference_complete=False,
                script_sha256=sha256(__file__), executable=sys.executable, python=sys.version,
                arguments=vars(args), precision_contract=CONTRACT, cases=[], diagnostics=[],
                model_loading='not attempted', execution=dict(device='not_run', tf32=False,
                precision_profile='bf16_rne', native='HF 5.2.0 unmodified family/expert forward',
                production_imports=False, reduction_emulation=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--model', type=Path, default=Path('/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8'))
    parser.add_argument('--output', type=Path, required=True, help='Fresh directory; never overwrite a prior run')
    parser.add_argument('--execute-cuda', action='store_true')
    parser.add_argument('--case',choices=('both','hello','capital'),default='both')
    parser.add_argument('--devices', default='cuda:0')
    parser.add_argument('--cache-mib', type=int, default=256, choices=range(0, 1025))
    parser.add_argument('--timeout', type=int, default=900, choices=range(1, 901))
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                      TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    path = args.output / 'manifest.json'
    if args.worker:
        manifest = new_manifest(args)
        started = time.monotonic()
        try:
            run(args, manifest)
        except Exception as error:
            manifest.update(status='incomplete', reference_kind='incomplete', full_reference_complete=False)
            manifest['diagnostics'].append(f'{type(error).__name__}: {error}')
        finally:
            manifest['elapsed_seconds'] = time.monotonic()-started
            write_json(path, manifest)
        return 0 if manifest['full_reference_complete'] else 2
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(path, new_manifest(args))
    command = [sys.executable, '-B', str(Path(__file__).resolve()), *sys.argv[1:], '--worker']
    with (args.output / 'run.log').open('w') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=args.timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            manifest = json.loads(path.read_text())
            manifest.update(status='incomplete', reference_kind='incomplete', full_reference_complete=False)
            manifest['diagnostics'].append(f'Worker stopped: {type(error).__name__}; no full-model certification')
            write_json(path, manifest)
            code = 124
    # A crash must never leave a stale successful/full manifest.
    if code not in (0, 2, 124):
        manifest = json.loads(path.read_text())
        manifest.update(status='incomplete', reference_kind='incomplete', full_reference_complete=False)
        manifest['diagnostics'].append(f'Worker exited {code}; see run.log')
        write_json(path, manifest)
    print(path)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
