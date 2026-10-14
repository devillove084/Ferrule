#!/opt/conda310/bin/python
"""Supplemental Hello M1 pinned-HF layers2..4 capture; no full-model regeneration."""
import argparse
import json
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts'))
import qwen35_fp8_bf16_reference as ref
REFERENCE = ROOT / 'target/validation/numeric-bf16-gate-20261004-v2/profile-reference'
PIN = '351495921233e0158eb2a4507c415d173c895a6086d5905b7a72d82de71849f9'
HELLO = '224217500f282bd3366e5c7743d1aea68499877ca668e3bf344007fd83091274'


def main(out):
    assert not out.exists(), 'fresh output required'
    assert ref.sha256(REFERENCE / 'manifest.json') == PIN
    assert ref.sha256(REFERENCE / 'hello.safetensors') == HELLO
    out.mkdir(parents=True)
    pins = ref.initialize()
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    store = ref.legacy.Checkpoint(Path('/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8'))
    weights = ref.DeviceWeights(store, 'cuda:0', 256 * 1024**2)
    config = ref.read_config(store.directory)
    cache = ref.f32_ssm_cache(config, ['cuda:0'] * 40)
    rope = ref.hf.Qwen3_5MoeTextRotaryEmbedding(config, device='cuda:0')
    ref.legacy.audit_rope_f32(rope)
    position = torch.tensor([[0]], device='cuda:0')
    saved, checks, linears = {}, [], []

    def capture(key, value):
        if isinstance(value, tuple): value = value[0]
        assert key not in saved and value.numel() * value.element_size() <= 4 << 20
        saved[key] = value.detach().cpu().contiguous().clone()

    with torch.inference_mode(), safe_open(str(REFERENCE / 'hello.safetensors'), framework='pt') as frozen:
        hidden = frozen.get_tensor('prefill.layers.01.hidden').cuda()
        assert hidden.dtype == torch.bfloat16 and list(hidden.shape) == [1, 1, 2048]
        for i in [2, 3, 4]:
            capture(f'{i}.layer.input', hidden)
            with torch.device('meta'): layer = ref.hf.Qwen3_5MoeDecoderLayer(config, i)
            ref.bind(layer, f'model.layers.{i}', weights, config)
            for name, module in layer.named_modules():
                if not name: continue
                if name in ('input_layernorm', 'post_attention_layernorm', 'linear_attn.norm', 'linear_attn', 'self_attn', 'self_attn.q_norm', 'self_attn.k_norm', 'mlp', 'mlp.experts', 'mlp.shared_expert', 'mlp.shared_expert_gate') or module.__class__.__name__ == 'Linear':
                    module.register_forward_pre_hook(lambda m,args,kwargs,key=f'{i}.{name}.input': capture(key,args[0] if args else kwargs['hidden_states']),with_kwargs=True)
                    module.register_forward_hook(lambda m,args,result,key=f'{i}.{name}.output': capture(key,result))
                if module.__class__.__name__ == 'Linear':
                    linears.append(dict(layer=i, module=name, source=module.name))
            def route(module,args,result,index=i):
                raw, selected = ref.router_observation(module,args,result,ref.ROUTER_PROFILE)
                capture(f'{index}.router.raw',raw)
                capture(f'{index}.router.weights',selected[1]);capture(f'{index}.router.ids',selected[2])
                return selected
            layer.mlp.gate.register_forward_hook(route)
            hidden = layer(hidden,position_embeddings=rope(hidden,position),past_key_values=cache,position_ids=position,attention_mask=None)
            capture(f'{i}.layer.output',hidden)
            for short,source in [('attention','self_attn' if i==3 else 'linear_attn'),('moe','mlp'),('hidden','layer')]:
                actual = saved[f'{i}.{source}.output']; expected = frozen.get_tensor(f'prefill.layers.{i:02}.{short}')
                checks.append(dict(layer=i,boundary=short,unequal=int((actual!=expected).sum()),max_abs=float((actual.float()-expected.float()).abs().max())))
                assert torch.equal(actual,expected), checks[-1]
            for field in ['conv_states','recurrent_states','keys','values']:
                value=getattr(cache.layers[i],field,None)
                if value is not None:
                    capture(f'{i}.cache.{field}',value)
                    native={'keys':'key_cache','values':'value_cache'}.get(field,field)
                    expected=frozen.get_tensor(f'prefill.cache.{native}.{i:02}')
                    assert torch.equal(saved[f'{i}.cache.{field}'],expected), (i,field)
            del layer;weights.clear()
    save_file(saved,str(out/'layers234.safetensors'))
    provenance=store.provenance()
    (out/'source-payloads.json').write_text(json.dumps(provenance,indent=2)+'\n')
    # Source descriptors include actual checkpoint header offsets, never a production trace.
    index=json.loads((store.directory/'model.safetensors.index.json').read_text())['weight_map']
    for item in linears:
        descriptors={}
        for kind,name in [('weight',item['source']),('scale',item['source']+'_scale_inv')]:
            if name not in index: continue
            path=store.directory/index[name]
            with path.open('rb') as f:
                size=int.from_bytes(f.read(8),'little'); header=json.loads(f.read(size))
            entry=header[name];start,end=entry['data_offsets']
            descriptors[kind]=dict(path=str(path),name=name,offset=8+size+start,bytes=end-start,shape=entry['shape'],dtype=entry['dtype'],sha256=provenance['used_payloads'][name]['sha256'])
        item.update(descriptors)
    manifest=dict(schema='ferrule.supplemental-bf16-layers234.v1',input_source='immutable HF prefill.layers.01.hidden, original Hello [9419]',layers=[2,3,4],model_boundary='bf16',ssm='f32',router=ref.router_contract(ref.ROUTER_PROFILE),versions=dict(ref.VERSIONS),native_sources=pins,original_manifest_sha256=PIN,original_hello_sha256=HELLO,script_sha256=ref.sha256(__file__),exporter_sha256=ref.sha256(ref.__file__),capture_sha256=ref.sha256(out/'layers234.safetensors'),payloads_sha256=ref.sha256(out/'source-payloads.json'),tf32=torch.backends.cuda.matmul.allow_tf32,bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,linears=linears,checks=checks,tensors=len(saved),cache_capacity=weights.capacity,cache_peak=weights.peak_bytes,production_inputs_used=False)
    with (out/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    assert ref.sha256(REFERENCE/'manifest.json')==PIN and ref.sha256(REFERENCE/'hello.safetensors')==HELLO
    print(json.dumps(dict(checks=checks,tensors=len(saved),manifest_sha256=ref.sha256(out/'manifest.json')),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.output)
