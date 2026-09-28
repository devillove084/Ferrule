"""Independent BF16-RNE operand / F32-output oracle (torch CPU, no Ferrule).

Default: write the tiny 2-layer GDN/GQA + top-k=2 MoE fixture next to this file.
--model DIR: read only layer 0/expert 0 gate/up/down projections, not a model.
"""
import argparse
import json
from pathlib import Path

import torch

PRECISION = 'bf16_rne'


def numeric(w, scale):
    expanded = scale.float().repeat_interleave(128, 0).repeat_interleave(128, 1)
    product = w.float() * expanded[:w.shape[0], :w.shape[1]]
    return product.bfloat16().float() if PRECISION == 'bf16_rne' else product


def linear(x, w):
    # These BF16 values are exact in F32; CPU F32 GEMM leaves accumulation/output
    # in F32, unlike torch BF16 matmul, which rounds its output to BF16.
    return (x.bfloat16().float() if PRECISION == 'bf16_rne' else x.float()) @ w.T


class NumericLinear(torch.nn.Module):
    def __init__(self, original, number):
        super().__init__()
        n, k = original.weight.shape
        scale = torch.full(((n+127)//128, (k+127)//128), 0.0071 + number*0.000013).bfloat16()
        expanded = scale.float().repeat_interleave(128, 0).repeat_interleave(128, 1)[:n, :k]
        raw = (original.weight.detach() / expanded).to(torch.float8_e4m3fn)
        self.register_buffer('raw', raw)
        self.register_buffer('scale', scale)
        self.register_buffer('decoded', numeric(raw, scale))

    def forward(self, x):
        return linear(x, self.decoded)


class Expert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = torch.nn.Linear(8, 12, bias=False)
        self.up = torch.nn.Linear(8, 12, bias=False)
        self.down = torch.nn.Linear(12, 8, bias=False)

    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


class Moe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.router = torch.nn.Linear(8, 3, bias=False)
        self.experts = torch.nn.ModuleList([Expert() for _ in range(3)])
        self.shared = Expert()
        self.shared_gate = torch.nn.Linear(8, 1, bias=False)

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, 8)
        scores, ids = torch.topk(self.router(x), 2, dim=-1)
        weights = torch.softmax(scores.float(), -1)
        result = torch.zeros_like(x)
        for row in range(x.shape[0]):
            for slot in range(2):
                result[row] += weights[row, slot] * self.experts[ids[row, slot]](x[row:row+1])[0]
        result += self.shared(x) * torch.sigmoid(self.shared_gate(x))
        return result.reshape(shape)


def canonical(name):
    if name == 'model.embed_tokens.weight': return 'token_embedding.weight'
    if name == 'model.norm.weight': return 'final_norm.weight'
    if name == 'lm_head': return 'output'
    _, _, layer, suffix = name.split('.', 3)
    suffix = suffix.replace('input_layernorm', 'input_norm').replace('post_attention_layernorm', 'post_attention_norm')
    renames = {'self_attn.q_proj': 'attention.query', 'self_attn.k_proj': 'attention.key',
        'self_attn.v_proj': 'attention.value', 'self_attn.o_proj': 'attention.output',
        'self_attn.q_norm': 'attention.query_norm', 'self_attn.k_norm': 'attention.key_norm',
        'linear_attn.in_proj_qkv': 'attention.qkv', 'linear_attn.in_proj_z': 'attention.z',
        'linear_attn.in_proj_b': 'attention.beta', 'linear_attn.in_proj_a': 'attention.a',
        'linear_attn.conv1d': 'attention.conv', 'linear_attn.norm': 'attention.norm',
        'linear_attn.out_proj': 'attention.output', 'linear_attn.A_log': 'attention.a_log.weight',
        'linear_attn.dt_bias': 'attention.dt_bias.weight', 'mlp': 'feed_forward'}
    for old, new in renames.items():
        if suffix == old or suffix.startswith(old+'.'):
            suffix = new + suffix[len(old):]
            break
    return f'layers.{layer}.{suffix}'


def tiny():
    import transformers
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    config = Qwen3_5TextConfig(vocab_size=11, hidden_size=8, intermediate_size=12,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        linear_conv_kernel_dim=3, linear_key_head_dim=3, linear_value_head_dim=2,
        linear_num_key_heads=1, linear_num_value_heads=2,
        layer_types=['linear_attention', 'full_attention'],
        rope_parameters={'rope_type':'default', 'rope_theta':10000.,
                         'partial_rotary_factor':0.5, 'mrope_section':[1,1,0]},
        rms_norm_eps=1e-6, tie_word_embeddings=False, dtype='float32', attn_implementation='eager')
    model = Qwen3_5ForCausalLM(config).float().eval()
    for layer in model.model.layers:
        layer.mlp = Moe()
    with torch.no_grad():
        for number, (name, p) in enumerate(model.named_parameters()):
            x = torch.arange(p.numel(), dtype=torch.float32).reshape(p.shape)
            value = torch.sin(x*.17 + number*.31)*.2
            if name.endswith('linear_attn.norm.weight'): value += .8
            if name.endswith('A_log'): value -= .5
            p.copy_(value)
    tensors = {}
    for number, (name, module) in enumerate(list(model.named_modules())):
        if not isinstance(module, torch.nn.Linear): continue
        replacement = NumericLinear(module, number)
        parent, _, child = name.rpartition('.')
        setattr(model.get_submodule(parent), child, replacement)
        tensors[canonical(name)+'.weight'] = dict(shape=list(module.weight.shape),
            dtype='F8_E4M3', raw=replacement.raw.view(torch.uint8).flatten().tolist(),
            scales=replacement.scale.view(torch.uint16).flatten().tolist())
    for name, p in model.named_parameters():
        tensors[canonical(name)] = dict(shape=list(p.shape), dtype='F32', values=p.detach().flatten().tolist())
    tokens = [1,3,2,5,7,4,6,8,2,9,1,10]
    with torch.no_grad():
        result = model(torch.tensor([tokens[:3]]), use_cache=True)
        logits = result.logits[0].tolist()
        cache = result.past_key_values
        for token in tokens[3:]:
            result = model(torch.tensor([[token]]), past_key_values=cache, use_cache=True)
            logits.extend(result.logits[0].tolist())
        ffn_cases = []
        for step in range(8):
            x = torch.sin(torch.arange(16).float()*.29 + step*.47).reshape(2,8)*.7
            layers = []
            for layer in model.model.layers:
                y = layer.mlp(x)
                layers.append(y.flatten().tolist())
                x = x+y
            ffn_cases.append(dict(input=(torch.sin(torch.arange(16).float()*.29 + step*.47)*.7).tolist(), layers=layers))
        projection_cases = {}
        for name, module in model.named_modules():
            if isinstance(module, NumericLinear) and '.experts.' not in name:
                x = torch.sin(torch.arange(3 * module.decoded.shape[1]).float()*.31).reshape(3, -1)*.83
                projection_cases[canonical(name)+'.weight'] = dict(input=x.flatten().tolist(), expected=module(x).flatten().tolist())
    data = dict(torch=torch.__version__, transformers=transformers.__version__, tensors=tensors,
                tokens=tokens, logits=logits, ffn_cases=ffn_cases, projection_cases=projection_cases)
    if PRECISION == 'f32':
        data['precision_profile'] = PRECISION
    path = Path(__file__).with_name('numeric_standard_oracle.json' if PRECISION == 'bf16_rne' else 'numeric_standard_f32_oracle.json')
    path.write_text(json.dumps(data, separators=(',', ':'))+'\n')
    print('Wrote', len(tensors), 'tiny tensors and', len(tokens), 'token logits to', path)


def real(directory):
    from numeric_fp8_oracle import tensor_metadata, rectangle
    directory = Path(directory).resolve()
    index = json.loads((directory/'model.safetensors.index.json').read_text())['weight_map']
    projections = []
    for projection in ['gate_proj', 'up_proj', 'down_proj']:
        name = f'model.language_model.layers.0.mlp.experts.0.{projection}.weight'
        weight = tensor_metadata(directory, index, name)
        scale = tensor_metadata(directory, index, name.removesuffix('.weight')+'.weight_scale_inv')
        n,k = weight['shape']
        assert weight['bytes'] <= 8*1024*1024
        w = rectangle(weight, [0,n], [0,k], 1, torch.float8_e4m3fn)
        s = rectangle(scale, [0,(n+127)//128], [0,(k+127)//128], 2, torch.bfloat16)
        x = torch.sin(torch.arange(2*k).float()*.0137).reshape(2,k)*.73
        expected = linear(x, numeric(w,s))
        projections.append(dict(weight=weight, scale=scale, input=x.flatten().tolist(), expected=expected.flatten().tolist()))
    print(json.dumps(dict(torch=torch.__version__, projections=projections), separators=(',', ':')))


if __name__ == '__main__':
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--precision', choices=['bf16_rne', 'f32'], default='bf16_rne')
    args = parser.parse_args()
    PRECISION = args.precision
    if args.model: real(args.model)
    else: tiny()
