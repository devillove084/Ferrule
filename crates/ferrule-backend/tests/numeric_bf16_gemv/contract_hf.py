"""Pinned native F.linear repeat/call-contract observation, never F32 oracle replacement."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[4]

def main(fixture,out,interposed):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0', 'requires explicit GPU grant'
    out.mkdir(exist_ok=False)
    meta=json.loads((fixture/'manifest.json').read_text())
    for name,pin in meta['payload_pins'].items():
        assert hashlib.sha256((fixture/name).read_bytes()).hexdigest()==pin
    sys.path.insert(0,str(ROOT/'scripts'))
    import qwen35_fp8_bf16_reference as ref
    source_pins=ref.initialize()
    import torch
    assert torch.version.git_version=='cf30153c4c131c8164ee7798e5022d810682e2cb'
    assert torch.cuda.get_device_capability()==(8,6)
    records=[]
    def load(name,shape):return torch.frombuffer(bytearray((fixture/name).read_bytes()),dtype=torch.bfloat16).reshape(shape).cuda()
    def raw(t):return t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    with torch.inference_mode():
        for c in meta['cases']:
            id,n,k=c['id'],c['n'],c['k']
            x=load(id+'.input.bf16',(1,k));w=load(id+'.decoded.bf16',(n,k))
            expected=(fixture/(id+'.hf.bf16')).read_bytes()
            outputs=[];diff=[]
            for i in range(16):
                y=torch.nn.functional.linear(x,w);torch.cuda.synchronize()
                b=raw(y);outputs.append(hashlib.sha256(b).hexdigest())
                diff.append(sum(b[j:j+2]!=expected[j:j+2] for j in range(0,len(b),2)))
                (out/(id+'-'+str(i)+'.bf16')).write_bytes(b)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
                y=torch.nn.functional.linear(x,w)
            torch.cuda.synchronize()
            kernels=sorted(set(e.name for e in p.events() if e.device_type==torch.autograd.DeviceType.CUDA))
            p.export_chrome_trace(str(out/(id+'-trace.json')))
            # This remains diagnostic only: never overwrite the actual BF16 expected.
            f32=torch.mm(x,w.T,out_dtype=torch.float32);b=raw(f32.bfloat16())
            record=dict(case=id,n=n,k=k,input_stride=list(x.stride()),weight_stride=list(w.stride()),output_stride=list(y.stride()),
                input_mod256=x.data_ptr()%256,weight_mod256=w.data_ptr()%256,output_mod256=y.data_ptr()%256,
                bf16_unequal=diff,unique_outputs=len(set(outputs)),output_sha256=outputs,
                f32_output_rounded_unequal=sum(b[j:j+2]!=expected[j:j+2] for j in range(0,len(b),2)),kernels=kernels)
            records.append(record)
    maps=Path('/proc/self/maps').read_text()
    libraries=sorted({line.split()[-1] for line in maps.splitlines() if any(x in line for x in ['libcublas','libcudart']) and '/' in line})
    report=dict(torch=torch.__version__,cuda=torch.version.cuda,source_pins=source_pins,interposed=interposed,
        libraries=libraries,library_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in libraries},
        workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),deterministic=torch.are_deterministic_algorithms_enabled(),
        tf32=torch.backends.cuda.matmul.allow_tf32,bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,cases=records,
        exact=all(not any(r['bf16_unequal']) for r in records),stable=all(r['unique_outputs']==1 for r in records))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    assert report['exact'] and report['stable'], 'native BF16 frozen replay changed or nondeterministic'
    if interposed:
        calls=[json.loads(l) for l in Path(os.environ['FERRULE_GEMV_CALL_LOG']).read_text().splitlines()]
        assert len(calls)>=8*16, 'interposer did not observe native calls; do not infer call contract'

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('fixture',type=Path);p.add_argument('output',type=Path);p.add_argument('--execute-cuda',action='store_true');p.add_argument('--interposed',action='store_true');a=p.parse_args();assert a.execute_cuda;main(a.fixture,a.output,a.interposed)
