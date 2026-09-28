#!/opt/conda310/bin/python
"""Offline TF 5.2 Qwen3.5 MoE numeric-FP8 reference, NOT a production model.

/opt/conda310/bin/python -B scripts/qwen35_fp8_reference.py --tiny-only
/opt/conda310/bin/python -B scripts/qwen35_fp8_reference.py --timeout 900

Only writes fresh target/validation/qwen35-fp8-reference-* directories. Native
Transformers attention/GDN/norm/router/shared-gate/experts forwards are retained.
See scripts/qwen35_fp8_reference.md for the precision and artifact contracts.
"""
import argparse
from collections import OrderedDict
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import struct
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
NAS = Path('/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8')
SCHEMA = 'ferrule.qwen35-fp8-reference.v1'
F32_SCHEMA = 'ferrule.qwen35-fp8-reference.f32.v1'
PREFIX = 'model.language_model.'
GIB = 1024 ** 3
BLOCK = 128


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def write_json(path, value):
    # Atomic checkpoint so SIGKILL cannot leave a half-written status document.
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + '\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def memory_info():
    info = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        name, value = line.split(':', 1)
        if name in ('MemTotal', 'MemAvailable', 'SwapTotal'):
            info[name] = int(value.split()[0]) * 1024
    # Container-relative mounts, v2 first, then v1. Record hierarchy's limit too.
    for name, path in {
        'cgroup_v2_limit': '/sys/fs/cgroup/memory.max',
        'cgroup_v2_current': '/sys/fs/cgroup/memory.current',
        'cgroup_v1_limit': '/sys/fs/cgroup/memory/memory.limit_in_bytes',
        'cgroup_v1_current': '/sys/fs/cgroup/memory/memory.usage_in_bytes',
    }.items():
        if Path(path).exists():
            value = Path(path).read_text().strip()
            info[name] = None if value == 'max' or int(value) >= 2 ** 60 else int(value)
    stat = Path('/sys/fs/cgroup/memory/memory.stat')
    if stat.exists():
        values = dict(line.split() for line in stat.read_text().splitlines())
        limit = int(values.get('hierarchical_memory_limit', 2 ** 63 - 1))
        info['cgroup_v1_hierarchical_limit'] = limit if limit < 2 ** 60 else None
    return info


def resident_bytes(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
    except FileNotFoundError:
        pass
    return 0


def source_name(name):
    return PREFIX + name[len('model.'):] if name.startswith('model.') else name


def initialize_torch(threads, precision='bf16_rne'):
    global torch, F, q, save_file
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as q
    import transformers
    require(transformers.__version__ == '5.2.0', 'This adapter is audited for Transformers 5.2.0')
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    require(precision in ('bf16_rne', 'f32'), f'Unknown precision profile: {precision}')
    torch.use_deterministic_algorithms(True)
    require(q.FusedRMSNormGated is None and q.causal_conv1d_fn is None
            and q.causal_conv1d_update is None and q.chunk_gated_delta_rule is None
            and q.fused_recurrent_gated_delta_rule is None,
            'Reference requires native torch GDN/conv/norm fallbacks, not FLA')
    return make_adapter_types(precision)


def bf16_f32(value):
    return value.to(torch.bfloat16).float()


def independent_bf16_rne(value):
    """Finite-only integer RNE oracle, independent of torch's BF16 conversion."""
    require(bool(torch.isfinite(value).all()), 'RNE oracle input must be finite')
    bits = value.float().contiguous().view(torch.int32).to(torch.int64)
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return bits.to(torch.int32).view(torch.float32)


def dequantize(weight, scale, precision='bf16_rne'):
    require(precision in ('bf16_rne', 'f32'), f'Unknown precision profile: {precision}')
    require(weight.dtype == torch.float8_e4m3fn and scale.dtype == torch.bfloat16,
            'Expected E4M3FN weight and BF16 numeric scale')
    require(weight.ndim == 2 and tuple(scale.shape) == tuple(math.ceil(d / BLOCK) for d in weight.shape),
            'Expected one scale per [128,128] weight block')
    require(bool(torch.isfinite(scale).all() and (scale > 0).all()), 'Invalid numeric scale')
    value = weight.float()
    require(bool(torch.isfinite(value).all()), 'Non-finite FP8 weight')
    # Do not expand scales to a full-sized tensor. Handle partial last blocks.
    for row in range(scale.shape[0]):
        for col in range(scale.shape[1]):
            value[row * BLOCK:(row + 1) * BLOCK, col * BLOCK:(col + 1) * BLOCK].mul_(scale[row, col].float())
    return bf16_f32(value) if precision == 'bf16_rne' else value.contiguous()


def independent_dequantize(weight, scale, precision='bf16_rne'):
    require(precision in ('bf16_rne', 'f32'), f'Unknown precision profile: {precision}')
    rows = torch.arange(weight.shape[0]) // BLOCK
    cols = torch.arange(weight.shape[1]) // BLOCK
    value = weight.float() * scale.float()[rows[:, None], cols[None, :]]
    return independent_bf16_rne(value) if precision == 'bf16_rne' else value.contiguous()


def make_adapter_types(precision='bf16_rne'):
    require(precision in ('bf16_rne', 'f32'), f'Unknown precision profile: {precision}')
    class ProductWeight(torch.Tensor):
        """Tag ONLY expert matrices; do not monkey-patch global torch functions."""
        precision_profile = precision

        @staticmethod
        def wrap(value):
            return torch.Tensor._make_subclass(ProductWeight, value, False)

        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            if func is F.linear:
                activation, weight, *rest = args
                activation = bf16_f32(activation) if precision == 'bf16_rne' else activation.float()
                return F.linear(activation, weight.as_subclass(torch.Tensor), *rest, **(kwargs or {}))
            return super().__torch_function__(func, types, args, kwargs or {})

    class LazyLinear(torch.nn.Module):
        def __init__(self, loader, name):
            super().__init__()
            require(loader.precision == precision, 'Linear/cache precision mismatch')
            self.loader, self.name = loader, name
            self.quantized = loader.store.meta(name)['dtype'] == 'F8_E4M3'

        def forward(self, value):
            weight = self.loader.matrix(self.name)
            activation = bf16_f32(value) if self.quantized and precision == 'bf16_rne' else value.float()
            return F.linear(activation, weight)

    class ExpertBank:
        """The unchanged HF expert forward indexes these in place of 3D tensors."""
        def __init__(self, loader, prefix, gate_up, count):
            require(loader.precision == precision, 'Expert/cache precision mismatch')
            self.loader, self.prefix, self.gate_up, self.count = loader, prefix, gate_up, count

        def __getitem__(self, index):
            index = int(index)
            require(0 <= index < self.count, 'Invalid expert index')
            prefix = f'{self.prefix}.{index}'
            if self.gate_up:
                # The native forward expects gate then up, concatenated along rows.
                value = torch.cat((self.loader.matrix(prefix + '.gate_proj.weight'),
                                   self.loader.matrix(prefix + '.up_proj.weight')), dim=0)
            else:
                value = self.loader.matrix(prefix + '.down_proj.weight')
            return ProductWeight.wrap(value)

    return ProductWeight, LazyLinear, ExpertBank


class TensorStore:
    """Headers + bounded open file handles; no full shard mmap or model load."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.index_path = self.directory / 'model.safetensors.index.json'
        self.index = json.loads(self.index_path.read_text())['weight_map']
        self.headers, self.files, self.reads = {}, OrderedDict(), {}
        self.bytes_read = 0
        for filename in sorted(set(self.index.values())):
            path = (self.directory / filename).resolve()
            require(path.parent == self.directory.resolve(), 'Shard must be inside checkpoint directory')
            with path.open('rb') as stream:
                size = struct.unpack('<Q', stream.read(8))[0]
                require(size <= 64 * 1024 * 1024, 'Oversized safetensors header')
                raw = stream.read(size)
            self.headers[filename] = (8 + size, json.loads(raw), hashlib.sha256(raw).hexdigest())

    def meta(self, name):
        require(name in self.index, f'Missing tensor: {name}')
        return self.headers[self.index[name]][1][name]

    def read(self, name):
        meta = self.meta(name)
        filename = self.index[name]
        if filename not in self.files:
            if len(self.files) >= 2:
                self.files.popitem(last=False)[1].close()
            self.files[filename] = (self.directory / filename).open('rb')
        self.files.move_to_end(filename)
        stream = self.files[filename]
        begin, end = meta['data_offsets']
        count = end - begin
        require(0 < count <= 2 * GIB, f'Tensor exceeds bounded payload read: {name}')
        stream.seek(self.headers[filename][0] + begin)
        payload = bytearray(count)
        require(stream.readinto(payload) == count, f'Short tensor read: {name}')
        digest = hashlib.sha256(payload).hexdigest()
        if name in self.reads:
            require(digest == self.reads[name]['sha256'], f'Checkpoint changed during run: {name}')
        else:
            self.reads[name] = dict(file=filename, shape=meta['shape'], dtype=meta['dtype'],
                                    bytes=count, sha256=digest)
        self.bytes_read += count
        dtype = {'F8_E4M3': torch.float8_e4m3fn, 'BF16': torch.bfloat16, 'F32': torch.float32}[meta['dtype']]
        return torch.frombuffer(payload, dtype=dtype).reshape(meta['shape'])

    def validate_text_pairs(self, output):
        pairs = []
        for name in sorted(self.index):
            if not name.startswith(PREFIX):
                continue
            meta = self.meta(name)
            if name.endswith('.weight_scale_inv'):
                weight = name.removesuffix('_scale_inv')
                require(self.meta(weight)['dtype'] == 'F8_E4M3', f'Orphan scale: {name}')
            if meta['dtype'] != 'F8_E4M3':
                continue
            require(name.endswith('.weight') and len(meta['shape']) == 2, f'Unsupported FP8 tensor: {name}')
            scale_name = name + '_scale_inv'
            scale_meta = self.meta(scale_name)
            require(scale_meta['dtype'] == 'BF16', f'Expected BF16 scale: {scale_name}')
            require(scale_meta['shape'] == [math.ceil(d / BLOCK) for d in meta['shape']],
                    f'Invalid block scale shape: {scale_name}')
            # Read EVERY text pair's actual scale, not just a representative layer.
            scale = self.read(scale_name).float()
            require(bool(torch.isfinite(scale).all() and (scale > 0).all()), f'Invalid numeric scale: {scale_name}')
            pairs.append(dict(weight=name, scale=scale_name, weight_shape=meta['shape'],
                              scale_shape=scale_meta['shape'], scale_dtype='BF16',
                              minimum=scale.min().item(), maximum=scale.max().item(),
                              scale_sha256=self.reads[scale_name]['sha256']))
        require(pairs, 'No quantized text weights')
        write_json(output / 'scale-pairs.json', dict(
            count=len(pairs), block_shape=[128, 128], pairs=pairs,
            definition='E4M3FN_value * BF16_numeric_scale[row//128,col//128], NOT reciprocal or exponent',
            coverage='All text pairs: headers and full scale payload; FP8 payload checked only when loaded'))
        return len(pairs)

    def provenance(self):
        return dict(index_sha256=sha256(self.index_path), bytes_read=self.bytes_read,
                    shards={name: dict(header_sha256=header[2],
                                       bytes=(self.directory / name).stat().st_size,
                                       mtime_ns=(self.directory / name).stat().st_mtime_ns)
                            for name, header in self.headers.items()},
                    payloads=self.reads,
                    note='Not whole-shard hashes: all used payloads are hashed, untouched weights are not certified')

    def close(self):
        for stream in self.files.values():
            stream.close()
        self.files.clear()


class WeightLoader:
    def __init__(self, store, capacity_bytes, precision='bf16_rne'):
        require(precision in ('bf16_rne', 'f32'), f'Unknown precision profile: {precision}')
        self.store, self.capacity, self.precision = store, capacity_bytes, precision
        self.cache = OrderedDict()
        self.bytes = self.peak = self.loads = self.hits = self.evictions = 0
        self.checked_pairs = set()

    def matrix(self, name):
        if name in self.cache:
            self.cache.move_to_end(name)
            self.hits += 1
            return self.cache[name]
        weight = self.store.read(name)
        if weight.dtype == torch.float8_e4m3fn:
            scale = self.store.read(name + '_scale_inv')
            value = dequantize(weight, scale, self.precision)
            # Actual per-pair numeric samples on both sides of block boundaries.
            rows = sorted({0, min(127, weight.shape[0]-1), min(128, weight.shape[0]-1), weight.shape[0]-1})
            cols = sorted({0, min(127, weight.shape[1]-1), min(128, weight.shape[1]-1), weight.shape[1]-1})
            for row in rows:
                for col in cols:
                    product = weight[row, col].float() * scale[row//128, col//128].float()
                    expected = (independent_bf16_rne(product) if self.precision == 'bf16_rne' else product)
                    require(torch.equal(value[row, col], expected), f'Numeric pair sample failed: {name}')
            self.checked_pairs.add(name)
        else:
            require(weight.dtype in (torch.bfloat16, torch.float32), f'Unsupported nonquant weight: {name}')
            value = weight.float()
            require(bool(torch.isfinite(value).all()), f'Non-finite weight: {name}')
        size = value.numel() * value.element_size()
        self.loads += 1
        while self.cache and self.bytes + size > self.capacity:
            old = self.cache.popitem(last=False)[1]
            self.bytes -= old.numel() * old.element_size()
            self.evictions += 1
        if size <= self.capacity:
            self.cache[name] = value
            self.bytes += size
            self.peak = max(self.peak, self.bytes)
        return value

    def statistics(self):
        return dict(capacity_bytes=self.capacity, resident_bytes=self.bytes, peak_bytes=self.peak,
                    loads=self.loads, hits=self.hits, evictions=self.evictions,
                    actual_quantized_pairs_sample_checked=len(self.checked_pairs))


def text_config(directory):
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    nested = json.loads((directory / 'config.json').read_text())
    require(nested['model_type'] == 'qwen3_5_moe', 'Expected nested MoE config')
    quant = nested['quantization_config']
    require(quant['quant_method'] == 'fp8' and quant['weight_block_size'] == [128, 128], 'Unsupported quantization config')
    values = dict(nested['text_config'])
    values.update(dtype='float32', tie_word_embeddings=False)
    config = Qwen3_5MoeTextConfig(**values)
    config._attn_implementation = 'eager'
    config._experts_implementation = 'eager'
    return config


def build_lazy(config, store, loader, adapter_types):
    _, LazyLinear, ExpertBank = adapter_types
    with torch.device('meta'):
        model = q.Qwen3_5MoeForCausalLM(config)
    bindings = []
    for i, layer in enumerate(model.model.layers):
        experts = layer.mlp.experts
        require(type(experts) is q.Qwen3_5MoeExperts, 'Unexpected experts implementation')
        prefix = f'{PREFIX}layers.{i}.mlp.experts'
        for e in range(config.num_experts):
            for projection, shape in [('gate_proj', [config.moe_intermediate_size, config.hidden_size]),
                                      ('up_proj', [config.moe_intermediate_size, config.hidden_size]),
                                      ('down_proj', [config.hidden_size, config.moe_intermediate_size])]:
                meta = store.meta(f'{prefix}.{e}.{projection}.weight')
                require(meta['dtype'] == 'F8_E4M3' and meta['shape'] == shape, 'Expert layout mismatch')
        for name, gate_up in [('gate_up_proj', True), ('down_proj', False)]:
            del experts._parameters[name]
            setattr(experts, name, ExpertBank(loader, prefix, gate_up, config.num_experts))
        # experts.forward is intentionally NEVER replaced.
    for name, module in list(model.named_modules()):
        if isinstance(module, torch.nn.Linear):
            require(module.bias is None, f'Unsupported bias: {name}')
            source = source_name(name) + '.weight'
            require(store.meta(source)['shape'] == [module.out_features, module.in_features], f'Linear shape: {name}')
            parent, _, child = name.rpartition('.')
            setattr(model.get_submodule(parent) if parent else model, child, LazyLinear(loader, source))
            bindings.append(dict(module=name, source=source, quantized=store.meta(source)['dtype'] == 'F8_E4M3'))
    for name, parameter in list(model.named_parameters()):
        source = source_name(name)
        raw = store.read(source)
        require(raw.dtype in (torch.bfloat16, torch.float32), f'Unbound FP8 parameter: {name}')
        require(tuple(raw.shape) == tuple(parameter.shape), f'Parameter shape: {name}')
        value = raw.float()
        require(bool(torch.isfinite(value).all()), f'Non-finite parameter: {name}')
        parent, _, child = name.rpartition('.')
        setattr(model.get_submodule(parent), child, torch.nn.Parameter(value, requires_grad=False))
    # Regenerate nonpersistent RoPE buffers using the native module; never to_empty()
    # a meta inv_freq buffer (that would silently introduce uninitialized values).
    model.model.rotary_emb = q.Qwen3_5MoeTextRotaryEmbedding(config, device='cpu')
    require(all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in model.parameters()), 'Parameter profile mismatch')
    require(all(b.device.type == 'cpu' for b in model.buffers()), 'Unmaterialized model buffer')
    for layer in model.model.layers:
        if layer.layer_type == 'linear_attention':
            attn = layer.linear_attn
            require(attn.chunk_gated_delta_rule is q.torch_chunk_gated_delta_rule
                    and attn.recurrent_gated_delta_rule is q.torch_recurrent_gated_delta_rule
                    and attn.causal_conv1d_update is q.torch_causal_conv1d_update,
                    'Non-native GDN path')
    return model.eval(), bindings


def dump_tensors(output, name, tensors, precision='bf16_rne'):
    for key, value in tensors.items():
        require(not value.is_floating_point() or bool(torch.isfinite(value).all()), f'Non-finite output: {key}')
    path = output / f'{name}.safetensors'
    metadata = {'schema': SCHEMA} if precision == 'bf16_rne' else {'schema': F32_SCHEMA, 'precision_profile': precision}
    save_file(tensors, str(path), metadata=metadata)
    return dict(file=path.name, sha256=sha256(path), tensors={
        key: dict(shape=list(value.shape), dtype=str(value.dtype)) for key, value in tensors.items()})


def run_case(model, ids, output, name, decode_text, cache_states=False, prompt=None, precision='bf16_rne'):
    tensors, calls, handles = {}, [], []
    stage = ''

    def put(key, value):
        tensors[f'{stage}.{key}'] = value.detach().cpu().contiguous().clone()

    def observe(key):
        def hook(_module, _args, result):
            put(key, result[0] if isinstance(result, tuple) else result)
        return hook

    handles.append(model.model.embed_tokens.register_forward_hook(observe('embedding')))
    handles.append(model.model.norm.register_forward_hook(observe('final_norm')))
    for i, layer in enumerate(model.model.layers):
        prefix = f'layers.{i:02d}'
        handles.append(layer.register_forward_hook(observe(prefix + '.hidden')))
        handles.append(layer.mlp.register_forward_hook(observe(prefix + '.moe')))
        handles.append(layer.mlp.shared_expert_gate.register_forward_hook(observe(prefix + '.shared_gate_logit')))
        attn = layer.linear_attn if layer.layer_type == 'linear_attention' else layer.self_attn
        handles.append(attn.register_forward_hook(observe(prefix + '.attention')))

        def route(_module, _args, result, prefix=prefix):
            put(prefix + '.selected_expert_ids', result[2])
            put(prefix + '.routing_weights', result[1])
        handles.append(layer.mlp.gate.register_forward_hook(route))
    cache, fed_ids = None, list(ids)
    try:
        for stage in ('prefill', 'decode.0'):
            started = time.monotonic()
            start = 0 if cache is None else cache.get_seq_length()
            inputs = torch.tensor([fed_ids], dtype=torch.long)
            positions = torch.arange(start, start + len(fed_ids))
            mask = torch.ones((1, start + len(fed_ids)), dtype=torch.long)
            with torch.inference_mode():
                result = model(input_ids=inputs, attention_mask=mask, position_ids=positions[None, :],
                               cache_position=positions, past_key_values=cache, use_cache=True,
                               logits_to_keep=0, return_dict=True)
            require(result.logits.dtype == torch.float32, 'Logits must be F32, no BF16 output rounding')
            put('input_ids', inputs)
            put('position_ids', positions[None, :])
            put('attention_mask', mask)
            put('cache_position', positions)
            put('logits', result.logits)
            next_id = int(result.logits[0, -1].argmax())
            cache = result.past_key_values
            if cache_states:
                for field in ('conv_states', 'recurrent_states', 'key_cache', 'value_cache'):
                    for i, value in enumerate(getattr(cache, field)):
                        if value is not None:
                            put(f'cache.{field}.{i:02d}', value)
            calls.append(dict(stage=stage, input_token_ids=fed_ids, position_start=start,
                              cache_source=None if stage == 'prefill' else 'prefill',
                              sequence_length_after=cache.get_seq_length(), next_token_id=next_id,
                              next_token_text=decode_text([next_id]), elapsed_seconds=time.monotonic()-started))
            print(f'{name}/{stage}: input={fed_ids}, next={next_id} {decode_text([next_id])!r}, '
                  f'{time.monotonic()-started:.2f}s, RSS={resident_bytes(os.getpid())/GIB:.2f}GiB', flush=True)
            fed_ids = [next_id]
        artifact = dump_tensors(output, name, tensors, precision)
        report = dict(id=name, prompt=prompt, request_token_ids=ids, calls=calls,
                      add_special_tokens=False, chat_template=False, **artifact)
        write_json(output / f'{name}.json', report)
        return report, tensors
    finally:
        for handle in handles:
            handle.remove()


def precision_golden(output, adapter_types):
    ProductWeight, _, _ = adapter_types
    require(ProductWeight.precision_profile == 'bf16_rne', 'Legacy precision golden requires BF16-RNE')
    torch.manual_seed(350535)
    bits = torch.randint(0, 256, (257, 259), dtype=torch.uint8)
    bits[(bits == 127) | (bits == 255)] = 0  # E4M3FN NaN encodings.
    weight = bits.view(torch.float8_e4m3fn)
    scale = torch.tensor([[.0311, .0573, .0917], [.1257, .2411, .4013], [.0613, .1817, .3211]]).bfloat16()
    x = torch.randn(3, 259)
    # Positive/negative midpoints test tie-to-even, not truncation or away-from-zero.
    x[0, :6] = torch.tensor([1.00390625, 1.01171875, -1.00390625, -1.01171875, 1.005, -1.005])
    actual_weight = dequantize(weight, scale, 'bf16_rne')
    expected_weight = independent_dequantize(weight, scale, 'bf16_rne')
    require(torch.equal(actual_weight, expected_weight), 'Independent dequant RNE mismatch')
    require(torch.equal(bf16_f32(x), independent_bf16_rne(x)), 'Activation RNE mismatch')
    actual = F.linear(x, ProductWeight.wrap(actual_weight))
    reference = F.linear(independent_bf16_rne(x).double(), expected_weight.double()).float()
    torch.testing.assert_close(actual, reference, atol=1e-3, rtol=2e-5)
    require(actual.dtype == torch.float32 and not torch.equal(actual, bf16_f32(actual)), 'Output incorrectly rounded to BF16')
    require(not torch.equal(actual, F.linear(x, actual_weight)), 'Fixture must detect missing activation rounding')
    artifact = dump_tensors(output, 'precision-adapter', dict(
        activation=x, fp8_bytes=bits, scale=scale, dequant_bf16_f32=actual_weight,
        output=actual, independent_f64_product_sum=reference))
    report = dict(status='complete', seed=350535, **artifact,
                  rne='Integer mantissa oracle, including signed ties',
                  max_abs_vs_f64=(actual-reference).abs().max().item(),
                  tolerance=dict(atol=1e-3, rtol=2e-5), product='BF16 operands represented in F32; F32 accumulation/output')
    write_json(output / 'precision-adapter.json', report)
    return report


def f32_precision_golden(output, adapter_types):
    ProductWeight, _, _ = adapter_types
    require(ProductWeight.precision_profile == 'f32', 'F32 golden requires the F32 adapter')
    torch.manual_seed(350535)
    bits = torch.randint(0, 256, (257, 259), dtype=torch.uint8)
    bits[(bits == 127) | (bits == 255)] = 0
    weight = bits.view(torch.float8_e4m3fn)
    scale = torch.tensor([[.0311, .0573, .0917], [.1257, .2411, .4013], [.0613, .1817, .3211]]).bfloat16()
    x = torch.randn(3, 259)
    x[0, :4] = torch.tensor([1.00390625, 1.01171875, -1.00390625, -1.01171875])
    actual_weight = dequantize(weight, scale, 'f32')
    expected_weight = independent_dequantize(weight, scale, 'f32')
    require(torch.equal(actual_weight, expected_weight), 'Independent F32 block dequantization mismatch')
    require(not torch.equal(actual_weight, bf16_f32(actual_weight)), 'Fixture must detect weight BF16 rounding')
    actual = F.linear(x, ProductWeight.wrap(actual_weight))
    require(torch.equal(actual, F.linear(x, expected_weight)), 'F32 adapter changed activation or output precision')
    reference = F.linear(x.double(), expected_weight.double()).float()
    torch.testing.assert_close(actual, reference, atol=1e-3, rtol=2e-5)
    require(not torch.equal(actual, F.linear(bf16_f32(x), expected_weight)), 'Fixture must detect activation rounding')
    require(not torch.equal(actual, bf16_f32(actual)), 'Fixture must detect BF16 output rounding')
    artifact = dump_tensors(output, 'precision-adapter', dict(
        activation=x, fp8_bytes=bits, scale=scale, dequant_f32=actual_weight,
        output=actual, independent_f64_product_sum=reference), precision='f32')
    report = dict(status='complete', precision_profile='f32', seed=350535, **artifact,
                  max_abs_vs_f64=(actual-reference).abs().max().item(),
                  tolerance=dict(atol=1e-3, rtol=2e-5),
                  product='Unrounded F32 dequantized weight and F32 activation; strict F32 accumulation/output')
    write_json(output / 'precision-adapter.json', report)
    return report


def quantize_tiny(value):
    scale = torch.empty(tuple(math.ceil(d / BLOCK) for d in value.shape), dtype=torch.bfloat16)
    fp8 = torch.empty_like(value, dtype=torch.float8_e4m3fn)
    for row in range(scale.shape[0]):
        for col in range(scale.shape[1]):
            block = value[row*BLOCK:(row+1)*BLOCK, col*BLOCK:(col+1)*BLOCK]
            scale[row, col] = (block.abs().max() / 400).clamp_min(1e-8)
            fp8[row*BLOCK:(row+1)*BLOCK, col*BLOCK:(col+1)*BLOCK] = (block / scale[row, col].float()).clamp(-448, 448).to(torch.float8_e4m3fn)
    return fp8, scale


def make_tiny_checkpoint(output):
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    torch.manual_seed(350535)
    config = Qwen3_5MoeTextConfig(
        vocab_size=64, hidden_size=256, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=64, num_experts=4, num_experts_per_tok=2,
        moe_intermediate_size=192, shared_expert_intermediate_size=192,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
        linear_value_head_dim=32, linear_conv_kernel_dim=4,
        layer_types=['linear_attention', 'full_attention'], max_position_embeddings=128,
        tie_word_embeddings=False, dtype='float32',
        rope_parameters=dict(rope_type='default', rope_theta=10000000.0,
                             partial_rotary_factor=.25, mrope_interleaved=True, mrope_section=[2, 3, 3]))
    config._attn_implementation = 'eager'
    model = q.Qwen3_5MoeForCausalLM(config).eval()
    weights = {}

    def add(name, value, quantized):
        value = value.detach().contiguous()
        if quantized:
            weights[name], weights[name + '_scale_inv'] = quantize_tiny(value)
        else:
            weights[name] = value.float() if name.endswith(('.A_log', '.linear_attn.norm.weight')) else value.bfloat16()

    for name, value in model.state_dict().items():
        if '.experts.gate_up_proj' in name:
            for i in range(config.num_experts):
                gate, up = value[i].chunk(2, dim=0)
                prefix = source_name(name).removesuffix('gate_up_proj') + str(i)
                add(prefix + '.gate_proj.weight', gate, True)
                add(prefix + '.up_proj.weight', up, True)
        elif '.experts.down_proj' in name:
            for i in range(config.num_experts):
                prefix = source_name(name).removesuffix('down_proj') + str(i)
                add(prefix + '.down_proj.weight', value[i], True)
        else:
            module = model.get_submodule(name.rsplit('.', 1)[0])
            quantized = isinstance(module, torch.nn.Linear) and name != 'lm_head.weight' and not name.endswith(
                ('.in_proj_a.weight', '.in_proj_b.weight', '.shared_expert_gate.weight'))
            add(source_name(name), value, quantized)
    directory = output / 'tiny-checkpoint'
    directory.mkdir()
    save_file(weights, str(directory / 'model.safetensors'))
    write_json(directory / 'model.safetensors.index.json', dict(weight_map={k: 'model.safetensors' for k in weights}))
    write_json(directory / 'config.json', dict(model_type='qwen3_5_moe', text_config=config.to_dict(),
               quantization_config=dict(quant_method='fp8', weight_block_size=[128, 128])))
    return directory


def eager_tiny(config, store, adapter_types, precision='bf16_rne'):
    """Small independent eager binding: native forwards, integer-RNE dequant oracle."""
    ProductWeight, _, _ = adapter_types
    model = q.Qwen3_5MoeForCausalLM(config).eval()
    for name, parameter in list(model.named_parameters()):
        source = source_name(name)
        quantized = False
        if '.experts.' in name:
            prefix, projection = source.rsplit('.', 1)
            matrices = []
            for i in range(config.num_experts):
                parts = ['gate_proj', 'up_proj'] if projection == 'gate_up_proj' else ['down_proj']
                matrices.append(torch.cat([independent_dequantize(
                    store.read(f'{prefix}.{i}.{part}.weight'), store.read(f'{prefix}.{i}.{part}.weight_scale_inv'), precision)
                    for part in parts], dim=0))
            value = torch.stack(matrices)
            quantized = True
        else:
            raw = store.read(source)
            quantized = raw.dtype == torch.float8_e4m3fn
            value = independent_dequantize(raw, store.read(source + '_scale_inv'), precision) if quantized else raw.float()
        require(tuple(value.shape) == tuple(parameter.shape), f'Tiny eager shape: {name}')
        parent, _, child = name.rpartition('.')
        if '.experts.' in name:
            # The native indexing operation preserves the tag on the 3D tensor.
            del model.get_submodule(parent)._parameters[child]
            setattr(model.get_submodule(parent), child, ProductWeight.wrap(value))
        else:
            setattr(model.get_submodule(parent), child, torch.nn.Parameter(value, requires_grad=False))
            if quantized and precision == 'bf16_rne':
                model.get_submodule(parent).register_forward_pre_hook(
                    lambda _module, args: (independent_bf16_rne(args[0]),))
    return model


def tiny_golden(output, adapter_types, precision='bf16_rne'):
    directory = make_tiny_checkpoint(output)
    config = text_config(directory)
    store = TensorStore(directory)
    loader = WeightLoader(store, 128 * 1024, precision)  # Force eviction while testing both expert banks.
    try:
        model, _ = build_lazy(config, store, loader, adapter_types)
        report, actual = run_case(model, [4, 7, 9], output, 'tiny-lazy', str, cache_states=True, precision=precision)
        del model
        eager = eager_tiny(config, store, adapter_types, precision)
        _, expected = run_case(eager, [4, 7, 9], output, 'tiny-eager', str, cache_states=True, precision=precision)
        require(actual.keys() == expected.keys(), 'Tiny capture schema mismatch')
        worst = 0.0
        for key in actual:
            if actual[key].is_floating_point():
                torch.testing.assert_close(actual[key], expected[key], atol=2e-5, rtol=2e-5, msg=key)
                worst = max(worst, (actual[key]-expected[key]).abs().max().item())
            else:
                require(torch.equal(actual[key], expected[key]), f'Tiny integer mismatch: {key}')
        require(loader.evictions > 0, 'Tiny test did not exercise LRU eviction')
        result = dict(status='complete', checkpoint='tiny-checkpoint', lazy_report='tiny-lazy.json',
                      eager_report='tiny-eager.json', tensor_count=len(actual), max_abs=worst,
                      checked='Native miniMoE + shared sigmoid gate + GDN chunk/recurrent + full attention; prefill + 1 decode',
                      cache=loader.statistics(), seed=350535,
                      note='Random tiny checkpoint, not a language-quality oracle')
        write_json(output / 'tiny-golden.json', result)
        return result
    finally:
        store.close()


def worker(args, output, cap):
    started = time.monotonic()
    manifest = dict(schema=SCHEMA if args.precision == 'bf16_rne' else F32_SCHEMA,
                    status='running', full_reference_complete=False,
                    arguments=vars(args), host_memory=memory_info(), cases=[],
                    execution=dict(device='cpu', threads=args.threads, interop_threads=1,
                                   tf32=False, batch=1, memory_cap_bytes=cap,
                                   precision_profile=args.precision,
                                   precision=('FP8 numeric BF16 scale -> BF16-RNE weight + BF16-RNE activation -> F32 linear'
                                              if args.precision == 'bf16_rne' else
                                              'FP8 numeric BF16 scale -> F32 weight + F32 activation -> strict F32 linear'),
                                   native='Transformers qwen3_5_moe; no custom family/expert forward; no NCCL'))
    write_json(output / 'manifest.json', manifest)
    store = None
    try:
        adapter_types = initialize_torch(args.threads, args.precision)
        # The imports reserve address space but no checkpoint data. Refuse allocations
        # beyond this hard limit BEFORE headers/weights or tiny models are loaded.
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        manifest.update(script_sha256=sha256(__file__), python=sys.version, executable=sys.executable,
                        versions=dict(torch=torch.__version__, transformers='5.2.0'),
                        native_source=dict(file=q.__file__, sha256=sha256(q.__file__)),
                        precision_contract={
                            'profile': args.precision,
                            'fp8_storage': 'F8_E4M3',
                            'numeric_scale': 'BF16 weight_scale_inv, row-major [ceil(M/128),ceil(K/128)], multiplicative',
                            'dequantized_weight': 'BF16-RNE represented as F32' if args.precision == 'bf16_rne' else 'F32 without BF16 rounding',
                            'activation': 'BF16-RNE represented as F32 for quantized Linear' if args.precision == 'bf16_rne' else 'F32 unchanged for quantized Linear',
                            'linear_product_accumulation_output': 'F32 strict CPU torch.matmul/linear; no TF32',
                            'native_tf_layers': 'F32 native attention/GDN/norm/router/shared-expert formulas',
                            'cuda_kernel_reduction_order': 'not emulated; this is an independent CPU oracle contract',
                            'legacy_default_compatibility': 'default bf16_rne preserves the original profile',
                        })
        if args.precision == 'bf16_rne':
            manifest['precision_golden'] = precision_golden(output, adapter_types)
            manifest['tiny_golden'] = tiny_golden(output, adapter_types)
        else:
            manifest['precision_golden'] = f32_precision_golden(output, adapter_types)
            manifest['tiny_golden'] = tiny_golden(output, adapter_types, precision='f32')
            manifest['consumer_comparison'] = dict(
                atol=2e-4, rtol=2e-4, metric_dtype='F64',
                rule='Every finite element: abs(actual-reference) <= atol + rtol*abs(reference)',
                scope='Initial Ferrule TF32x3 vs CPU F32 criterion; not run or certified by this exporter')
        write_json(output / 'manifest.json', manifest)
        if not args.tiny_only:
            from transformers import AutoTokenizer
            config = text_config(args.model)
            require(config.num_hidden_layers == 40 and config.num_experts == 256,
                    'Full reference requires the real 40-layer, 256-expert checkpoint')
            manifest['checkpoint'] = dict(directory=str(args.model.resolve()),
                                          config_sha256=sha256(args.model / 'config.json'),
                                          effective_text_config=config.to_dict())
            print('Validating every text FP8/scale pair (headers + numeric scale payloads)...', flush=True)
            store = TensorStore(args.model)
            manifest['scale_pair_count'] = store.validate_text_pairs(output)
            write_json(output / 'manifest.json', manifest)
            loader = WeightLoader(store, args.cache_mib * 1024 ** 2, args.precision)
            model, bindings = build_lazy(config, store, loader, adapter_types)
            write_json(output / 'bindings.json', bindings)
            tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
            manifest['tokenizer_files'] = {name: sha256(args.model / name) for name in
                                          ('tokenizer.json', 'tokenizer_config.json')}
            for name, prompt in [('hello', 'Hello')] + ([('capital', 'The capital of France is')] if args.capital else []):
                ids = tokenizer.encode(prompt, add_special_tokens=False)
                require(0 < len(ids) <= 16, 'Reference only supports short fixed prompts')
                report, tensors = run_case(model, ids, output, name,
                    lambda ids: tokenizer.decode(ids, clean_up_tokenization_spaces=False),
                    cache_states=args.cache_states, prompt=prompt, precision=args.precision)
                manifest['cases'].append(dict(id=name, report=f'{name}.json', file=report['file'], sha256=report['sha256']))
                manifest['weight_cache'] = loader.statistics()
                write_json(output / 'manifest.json', manifest)
                del tensors
            write_json(output / 'source-payloads.json', store.provenance())
            manifest['full_reference_complete'] = True
            del model
        require(not torch.cuda.is_initialized(), 'CPU reference unexpectedly initialized CUDA')
        manifest.update(status='complete', elapsed_seconds=time.monotonic()-started,
                        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                        validation_scope='Reference generation + tiny adapter self-checks, NOT a Ferrule comparison')
        write_json(output / 'manifest.json', manifest)
        print(f'COMPLETE: {output}', flush=True)
    except BaseException as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started)
        write_json(output / 'manifest.json', manifest)
        raise
    finally:
        if store is not None:
            store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=NAS)
    parser.add_argument('--precision', choices=('bf16_rne', 'f32'), default='bf16_rne',
                        help='Explicit oracle profile; default preserves legacy BF16-RNE behavior; f32 means F32 dequantized weight and F32 activation')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--memory-cap-gib', type=int, default=16)
    parser.add_argument('--cache-mib', type=int, default=256)
    parser.add_argument('--tiny-only', action='store_true')
    parser.add_argument('--capital', action='store_true', help='Also run the short capital prompt')
    parser.add_argument('--cache-states', action='store_true', help='Export all full-model conv/recurrent/KV states')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(1 <= args.threads <= 8 and 1 <= args.timeout <= 900, 'At most 8 threads and 900s per worker')
    require(0 <= args.cache_mib <= 1024 and 10 <= args.memory_cap_gib <= 32, 'cache <=1024MiB; memory cap 10..32GiB')
    cap = args.memory_cap_gib * GIB
    validation = ROOT / 'target' / 'validation'
    require(validation.is_dir(), 'target/validation must exist')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = (args.output or validation / f'qwen35-fp8-reference-{stamp}').resolve()
    require(output.parent == validation.resolve() and output.name.startswith('qwen35-fp8-reference-'), 'Output outside allowed scope')
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false',
                      PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS=str(args.threads),
                      MKL_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads))
    if args.worker:
        require(output.is_dir(), 'Supervisor-created output directory required')
        worker(args, output, cap)
        return
    memory = memory_info()
    require(memory['MemAvailable'] > cap, 'Insufficient available host memory for configured cap')
    for version in ('v1', 'v2'):
        limit = memory.get(f'cgroup_{version}_limit')
        if version == 'v1':
            hierarchy = memory.get('cgroup_v1_hierarchical_limit')
            limit = min(limit, hierarchy) if limit and hierarchy else limit or hierarchy
        if limit is not None:
            require(limit - memory.get(f'cgroup_{version}_current', 0) > cap, 'Insufficient cgroup headroom')
    print(f'Preflight: {json.dumps(memory)}; worker cap={args.memory_cap_gib}GiB, cache={args.cache_mib}MiB', flush=True)
    output.mkdir(exist_ok=False)
    command = [sys.executable, '-B', str(Path(__file__).resolve()), *sys.argv[1:], '--worker', '--output', str(output)]
    with (output / 'run.log').open('w') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + args.timeout
        failure = None
        try:
            while child.poll() is None:
                if time.monotonic() >= deadline:
                    failure = f'Hard worker timeout ({args.timeout}s); no automatic retry'
                    break
                if resident_bytes(child.pid) > cap:
                    failure = f'Worker RSS exceeded {cap} bytes'
                    break
                time.sleep(.2)
        except KeyboardInterrupt:
            failure = 'Interrupted; worker process group killed'
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        if failure or child.returncode:
            path = output / 'manifest.json'
            report = json.loads(path.read_text()) if path.exists() else dict(schema=SCHEMA)
            report.update(status='failed', supervisor_error=failure or f'Worker exit {child.returncode}')
            write_json(path, report)
    print((output / 'run.log').read_text(), end='', flush=True)
    print(f'Artifacts: {output}', flush=True)
    raise SystemExit(124 if failure and 'timeout' in failure else 1 if failure or child.returncode else 0)


if __name__ == '__main__':
    main()
