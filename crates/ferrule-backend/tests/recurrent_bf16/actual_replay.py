"""Pinned Hello decode GDN operator replay; never loads a model or runs projections.

Run hf and native in separate supervised processes (Torch CUDA12.9 vs native
CUDA12.4). The original independent capture is read-only. Intermediate expected
values are observed from the pinned native HF function, not replacement math.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
CAPTURE = ROOT / 'target/validation/bf16-layer01-supported-retry-20261005'
NATIVE = ROOT / 'crates/ferrule-backend/native/cuda'
STRIDE = 524288
FIELDS = ['qnorm', 'knorm', 'qscaled', 'g', 'beta', 'decay', 'decayed', 'memory', 'correction', 'dot']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def hf_replay(out):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    sys.path.insert(0, str(ROOT / 'scripts'))
    import qwen35_fp8_bf16_reference as ref
    pins = ref.initialize()
    assert sha(CAPTURE / 'hf-manifest.json') == '281ad7cc617bd9b9b69413adb426170a86c883f019864c49f9f585209b0e5c26'
    assert sha(CAPTURE / 'hf-layer01.safetensors') == '6752faa07d618e884c6f4be51694aa071866fa88f4e1d0b7860e3423c9b21b07'
    frozen = safe_open(str(CAPTURE / 'hf-layer01.safetensors'), framework='pt', device='cpu')
    get = lambda key: frozen.get_tensor('hello/decode.0.0.' + key)
    source = json.loads((CAPTURE / 'hf-source-payloads.json').read_text())['used_payloads']
    checkpoint = Path('/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8')
    used = {}
    def parameter(name):
        key = 'model.language_model.layers.0.linear_attn.' + name
        meta = source[key]
        with safe_open(str(checkpoint / meta['shard']), framework='pt', device='cpu') as f:
            value = f.get_tensor(key)
        raw = value.contiguous().view(torch.uint8).numpy().tobytes()
        assert hashlib.sha256(raw).hexdigest() == meta['sha256']
        used[key] = meta
        return value.bfloat16().contiguous()
    prod = CAPTURE / 'producer/hello/decode.0'
    def actual(name):
        return torch.frombuffer(bytearray((prod / name).read_bytes()), dtype=torch.float32).clone()
    def actual_bf16(name):
        value = actual(name)
        assert torch.equal(value, value.bfloat16().float()), 'Not an exact BF16 expansion: ' + name
        return value.bfloat16()
    tensors = dict(raw=actual_bf16('0004.f32'), a=actual_bf16('0008.f32'),
                   b=actual_bf16('0010.f32'), state=actual('before-0-ssm.f32'),
                   history=actual_bf16('before-0-conv.f32'),
                   a_log=parameter('A_log'), dt_bias=parameter('dt_bias'), weight=parameter('conv1d.weight'))
    assert torch.equal(tensors['raw'], get('linear_attn.in_proj_qkv.output').flatten())
    assert torch.equal(tensors['a'], get('linear_attn.in_proj_a.output').flatten())
    assert torch.equal(tensors['b'], get('linear_attn.in_proj_b.output').flatten())
    assert torch.equal(tensors['state'].view(torch.int32), get('before.recurrent_states').flatten().view(torch.int32))
    assert torch.equal(tensors['history'], get('before.conv_states').flatten())
    input_pins = {}
    for name, tensor in tensors.items():
        path = out / (name + ('.bf16' if tensor.dtype == torch.bfloat16 else '.f32'))
        path.write_bytes(tensor.view(torch.uint8).numpy().tobytes())
        input_pins[path.name] = sha(path)
    tensors = {k:v.cuda() for k,v in tensors.items()}
    hf = ref.hf
    expected = {}
    with torch.inference_mode():
        history = tensors['history'].reshape(1,8192,4).clone()
        conv = hf.torch_causal_conv1d_update(tensors['raw'].reshape(1,8192,1), history,
                                            tensors['weight'].reshape(8192,4), None, 'silu').transpose(1,2)
        assert torch.equal(conv.cpu().flatten(), actual('0011.f32').bfloat16())
        q,k,v = conv.split([2048,2048,4096],dim=-1)
        q=q.reshape(1,1,16,128).repeat_interleave(2,dim=2)
        k=k.reshape(1,1,16,128).repeat_interleave(2,dim=2)
        v=v.reshape(1,1,32,128)
        beta=tensors['b'].reshape(1,1,32).sigmoid()
        g=-tensors['a_log'].float().exp()*ref.F.softplus(tensors['a'].reshape(1,1,32).float()+tensors['dt_bias'])
        assert torch.equal(g.cpu(),get('core.g')) and torch.equal(beta.cpu(),get('core.beta'))
        expected.update(conv=conv, history=history, g=g, beta=beta)
        fn=hf.torch_recurrent_gated_delta_rule
        lines,start=inspect.getsourcelines(fn)
        # sys.settrace observes tensor locals at statement boundaries without
        # replacing HF ops, reduction dispatch, or cache dtype.
        locations={start+i:line.strip() for i,line in enumerate(lines)}
        norm_count=0
        def observer(frame,event,arg):
            nonlocal norm_count
            if frame.f_code is hf.l2norm.__code__ and event=='return':
                expected[['qnorm','knorm'][norm_count]]=arg.clone(); norm_count+=1
            if frame.f_code is fn.__code__ and event=='line':
                line=locations[frame.f_lineno]; loc=frame.f_locals
                if line.startswith('kv_mem ='):
                    expected.update(decayed=loc['last_recurrent_state'].clone(),qscaled=loc['q_t'].clone(),decay=loc['g_t'].clone())
                if line.startswith('delta ='): expected['memory']=loc['kv_mem'].clone()
                if line.startswith('last_recurrent_state = last_recurrent_state +'): expected['correction']=loc['delta'].clone()
                if line.startswith('if not output_final_state:'): expected['dot']=loc['core_attn_out'].clone()
            return observer
        assert sys.gettrace() is None
        sys.settrace(observer)
        try:
            output,state=fn(q,k,v,g,beta,tensors['state'].reshape(1,32,128,128),True,True)
        finally: sys.settrace(None)
        assert torch.equal(output.cpu(),get('core.output'))
        assert torch.equal(state.cpu().view(torch.int32),get('core.final_state').view(torch.int32))
        assert norm_count == 2
        expected.update(output=output,state=state)
        for name in ['conv','history','output','state']:
            (out/('expected-'+name+'.f32')).write_bytes(expected[name].float().cpu().contiguous().numpy().tobytes())
        save_file({k:v.detach().cpu().contiguous() for k,v in expected.items()},str(out/'hf-intermediates.safetensors'))
    (out/'hf-replay.json').write_text(json.dumps(dict(source_pins=pins,parameter_pins=used,input_pins=input_pins,
        independent_capture_sha256=sha(CAPTURE/'hf-layer01.safetensors'),expected_sha256=sha(out/'hf-intermediates.safetensors'),
        script_sha256=sha(__file__),hf_manifest_sha256=sha(CAPTURE/'hf-manifest.json'),
        replay_output_and_f32_state_exact=True,actual_input_checks_exact=True,observed_fields=list(expected)),indent=2)+'\n')
    print('Actual HF replay: output and F32 state bit-exact to independent retained capture')

def instrument(out):
    source=(NATIVE/'core/recurrent_bf16.cuh').read_text()
    begin=source.index('__global__ void delta(')
    end=source.index('\n__global__ void gated_norm',begin)
    kernel=source[begin:end].replace('void delta(FerruleCoreBf16DeltaArgs a)', 'void delta_observed(FerruleCoreBf16DeltaArgs a, float *trace)')
    def inject(anchor,extra):
        nonlocal kernel
        assert kernel.count(anchor)==1,anchor
        kernel=kernel.replace(anchor,anchor+'\n'+extra)
    inject('const float beta = bf16_round(1.0f / (1.0f + expf(-load(a.b_bf16, g))));', '''    if (column == 0) {
      trace[3*524288+h] = __fmul_rn(-A, softplus);
      trace[4*524288+h] = beta;
      trace[5*524288+h] = decay;
      for (uint32_t d=0; d<a.key_dim; ++d) {
        float nq=bf16_round(__fmul_rn(load(a.qkv_bf16,q+d),qn));
        trace[h*a.key_dim+d]=nq;
        trace[524288+h*a.key_dim+d]=bf16_round(__fmul_rn(load(a.qkv_bf16,k+d),kn));
        trace[2*524288+h*a.key_dim+d]=__fmul_rn(nq,scale);
      }
    }''')
    inject('s[off] = __fmul_rn(s[off], decay);','      trace[6*524288+uint64_t(h)*a.key_dim*a.value_dim+off+column]=s[off];')
    inject('const float correction = __fmul_rn(__fsub_rn(v, memory), beta);','    trace[7*524288+i]=memory; trace[8*524288+i]=correction;')
    inject('pointer<uint16_t>(a.output_bf16)[uint64_t(t) * vw + i] = bf16_rne(result);','    trace[9*524288+i]=result;')
    (out/'observed.cuh').write_text('namespace ferrule::cuda::core::hf_bf16 {\n'+kernel+'\n}\n')
    (out/'instrumentation.json').write_text(json.dumps(dict(source_sha256=sha(NATIVE/'core/recurrent_bf16.cuh'),generated_sha256=sha(out/'observed.cuh'),fields=FIELDS,stride=STRIDE),indent=2)+'\n')

def compare(out,label,require_exact=False):
    import torch
    from safetensors import safe_open
    torch.set_num_threads(1)
    manifest=json.loads((out/'hf-replay.json').read_text())
    assert sha(out/'hf-intermediates.safetensors') == manifest['expected_sha256']
    for file,pin in manifest['input_pins'].items(): assert sha(out/file)==pin
    hf=safe_open(str(out/'hf-intermediates.safetensors'),framework='pt',device='cpu')
    def load(name,dtype=torch.float32):
        return torch.frombuffer(bytearray((out/name).read_bytes()),dtype=dtype).clone()
    trace=load(label+'-trace.f32')
    metrics={}
    def check(name,a,b):
        a=a.float().flatten(); b=b.float().flatten(); assert a.shape==b.shape
        indices=(a.view(torch.int32)!=b.view(torch.int32)).nonzero().flatten()
        metrics[name]=dict(elements=a.numel(),unequal=indices.numel(),max_abs=float((a-b).abs().max()),first=[dict(index=int(i),actual=float(a[i]),expected=float(b[i]),actual_bits=hex(int(a[i].view(torch.int32))&0xffffffff),expected_bits=hex(int(b[i].view(torch.int32))&0xffffffff)) for i in indices[:4]])
    for i,name in enumerate(FIELDS):
        b=hf.get_tensor(name);check(name,trace[i*STRIDE:i*STRIDE+b.numel()],b)
    for name,dtype in [('output',torch.bfloat16),('state',torch.float32),('conv',torch.bfloat16),('history',torch.bfloat16)]:
        check(name,load(label+'-'+name+('.bf16' if dtype==torch.bfloat16 else '.f32'),dtype),hf.get_tensor(name))
    prod=CAPTURE/'producer/hello/decode.0'
    for name,file,dtype in [('output','0012.f32',torch.bfloat16),('state','0015.f32',torch.float32)]:
        check('original_production_'+name,load(label+'-'+name+('.bf16' if dtype==torch.bfloat16 else '.f32'),dtype),torch.frombuffer(bytearray((prod/file).read_bytes()),dtype=torch.float32))
    (out/(label+'-comparison.json')).write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))
    if require_exact:
        assert all(v["unequal"] == 0 for k,v in metrics.items() if not k.startswith("original_production_"))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['hf','instrument','compare']);p.add_argument('output',type=Path);p.add_argument('--label',default='baseline');p.add_argument('--require-exact',action='store_true');a=p.parse_args()
    if a.mode=='hf': hf_replay(a.output)
    elif a.mode=='instrument': instrument(a.output)
    else: compare(a.output,a.label,a.require_exact)
