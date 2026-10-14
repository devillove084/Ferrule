"""Bounded pinned-HF Hello stage replay; original references are never replaced."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts'))
import qwen35_fp8_bf16_reference as ref


def main(directory):
    assert not (directory / "hf-layers5-11.safetensors").exists(), "fresh HF output required"
    assert not (directory / "hf-replay.json").exists(), "fresh HF report required"
    pins = ref.initialize()
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    reference = ROOT / 'target/validation/numeric-bf16-gate-20261004-v2/profile-reference'
    assert ref.sha256(reference / 'manifest.json') == '351495921233e0158eb2a4507c415d173c895a6086d5905b7a72d82de71849f9'
    assert ref.sha256(reference / 'hello.safetensors') == '224217500f282bd3366e5c7743d1aea68499877ca668e3bf344007fd83091274'
    producer = directory / 'producer/hello/prefill'
    events = [json.loads(line) for line in (producer / 'events.jsonl').read_text().splitlines()]
    store = ref.legacy.Checkpoint(Path('/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8'))
    weights = ref.DeviceWeights(store, 'cuda:0', 256 * 1024**2)
    config = ref.read_config(store.directory)
    rope = ref.hf.Qwen3_5MoeTextRotaryEmbedding(config, device='cuda:0')
    ref.legacy.audit_rope_f32(rope)
    saved, checks, comparisons = {}, [], []

    def actual(event):
        return torch.frombuffer(bytearray((producer / event['file']).read_bytes()), dtype=torch.float32).reshape(event['shape'])

    def delta(a, b):
        a = a.detach().float().cpu().reshape(-1)
        b = b.detach().float().cpu().reshape(-1)
        assert a.shape == b.shape
        indices = (a.view(torch.int32) != b.view(torch.int32)).nonzero().reshape(-1)
        error = (a-b).abs()
        return dict(elements=a.numel(), unequal=indices.numel(), max_abs=float(error.max()), outside=int((error > (.00025 + .012*b.abs())).sum()), first=[dict(index=int(i), actual=float(a[i]), expected=float(b[i])) for i in indices[:3]])

    def capture(key, value):
        if isinstance(value, tuple):
            value = value[0]
        if value is None:
            return
        assert key not in saved, key
        saved[key] = value.detach().cpu().contiguous().clone()

    with torch.inference_mode(), safe_open(str(reference / 'hello.safetensors'), framework='pt') as frozen:
        # Reuse the validated independent capture for layers2..4. Each new layer
        # starts from original HF hidden, not a possibly divergent producer row.
        for index in range(5, 12):
            cache = ref.f32_ssm_cache(config, ['cuda:0'] * 40)
            hidden = frozen.get_tensor(f'prefill.layers.{index-1:02}.hidden').cuda()
            capture(f'{index}.layer.input', hidden)
            with torch.device('meta'):
                layer = ref.hf.Qwen3_5MoeDecoderLayer(config, index)
            ref.bind(layer, f'model.layers.{index}', weights, config)
            layer_events = [e for e in events if e['layer'] == index]
            for name, module in layer.named_modules():
                if not name:
                    continue
                observed = name in ('input_layernorm', 'post_attention_layernorm', 'linear_attn.norm', 'linear_attn', 'self_attn', 'self_attn.q_norm', 'self_attn.k_norm', 'mlp', 'mlp.experts', 'mlp.shared_expert', 'mlp.shared_expert_gate') or module.__class__.__name__ == 'Linear'
                if observed:
                    module.register_forward_pre_hook(lambda m, args, kwargs, key=f'{index}.{name}.input': capture(key, args[0] if args else kwargs['hidden_states']), with_kwargs=True)
                    module.register_forward_hook(lambda m, args, result, key=f'{index}.{name}.output': capture(key, result))
                if module.__class__.__name__ == 'Linear':
                    def replay(m, args, result, key=name, i=index):
                        output = [e for e in layer_events if e['event'] == 'linear.output' and (e.get('weight') or {}).get('name') == m.name]
                        assert len(output) == 1, (i, key)
                        inputs = [e for e in layer_events if e['event'] == 'linear.input' and (e.get('weight') or {}).get('name') == m.name]
                        assert len(inputs) == 1
                        x, y = actual(inputs[0]), actual(output[0])
                        entry = dict(layer=i, module=key, file=output[0]['file'], input=delta(x, args[0]), output=delta(y, result))
                        if entry['output']['unequal']:
                            replayed = ref.F.linear(x.cuda().bfloat16(), weights.matrix(m.name))
                            entry['sameinput'] = delta(y, replayed)
                        comparisons.append(entry)
                    module.register_forward_hook(replay)
            def route(m, args, result, i=index):
                raw, selected = ref.router_observation(m, args, result, ref.ROUTER_PROFILE)
                capture(f'{i}.router.raw', raw)
                capture(f'{i}.router.ids', selected[2])
                capture(f'{i}.router.weights', selected[1])
                return selected
            layer.mlp.gate.register_forward_hook(route)
            if hasattr(layer, 'linear_attn'):
                for attr in ['chunk_gated_delta_rule', 'recurrent_gated_delta_rule']:
                    native = getattr(layer.linear_attn, attr)
                    def observe(*args, _fn=native, **kwargs):
                        for name, value in zip(['q', 'k', 'v'], args):
                            capture(f'{index}.core.{name}', value)
                        for name in ['g', 'beta', 'initial_state']:
                            if kwargs.get(name) is not None:
                                capture(f'{index}.core.{name}', kwargs[name])
                        result = _fn(*args, **kwargs)
                        capture(f'{index}.core.output', result[0])
                        capture(f'{index}.core.final_state', result[1])
                        return result
                    setattr(layer.linear_attn, attr, observe)
            hidden = layer(hidden, position_embeddings=rope(hidden, torch.tensor([[0]], device='cuda:0')), past_key_values=cache, position_ids=torch.tensor([[0]], device='cuda:0'), attention_mask=None)
            capture(f'{index}.layer.output', hidden)
            for short, source in [('attention', 'self_attn' if config.layer_types[index] == 'full_attention' else 'linear_attn'), ('moe', 'mlp'), ('hidden', 'layer')]:
                check = dict(layer=index, boundary=short, **delta(saved[f'{index}.{source}.output'], frozen.get_tensor(f'prefill.layers.{index:02}.{short}')))
                checks.append(check)
                assert check['unequal'] == 0, check
            for field in ['conv_states', 'recurrent_states', 'keys', 'values']:
                value = getattr(cache.layers[index], field, None)
                if value is not None:
                    capture(f'{index}.cache.{field}', value)
                    original = {'keys': 'key_cache', 'values': 'value_cache'}.get(field, field)
                    check = dict(layer=index, boundary=field, **delta(value, frozen.get_tensor(f'prefill.cache.{original}.{index:02}')))
                    checks.append(check)
                    assert check['unequal'] == 0, check
            # Export loaded parameters explicitly: native BF16 words and the
            # parameter-only F32->BF16 loading rule, never blanket model casts.
            for name, parameter in layer.named_parameters():
                if 'norm' in name or name.endswith('A_log'):
                    capture(f'{index}.loaded.{name}', parameter)
            del layer, cache
            weights.clear()
    save_file(saved, str(directory / 'hf-layers5-11.safetensors'))
    (directory / 'hf-source-payloads.json').write_text(json.dumps(store.provenance(), indent=2)+'\n')
    result = dict(checks=checks, comparisons=comparisons, source_pins=pins, script_sha256=ref.sha256(__file__), capture_sha256=ref.sha256(directory/'hf-layers5-11.safetensors'), native_versions=dict(ref.VERSIONS), expected_source='original HF predecessor hidden; independently validated outputs/state; SAMEINPUT replays diagnostic only', cache_peak=weights.peak_bytes)
    (directory / 'hf-replay.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(original_checks=len(checks), failures=[c for c in checks if c['unequal']], differing_linears=[c for c in comparisons if c['output']['unequal']]), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    main(parser.parse_args().directory.resolve())
