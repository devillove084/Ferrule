"""CPU-only real-operand fixture and reduction experiment; never a quality gate.

Select one captured numeric linear output. Freeze BF16 operands, original FP8 and
numeric BF16 scales, saved production output, and independent F64 dot products.
No GPU initialization, model forward, production import or new error threshold.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def rne(x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    require(np.isfinite(x).all(), 'Nonfinite operand')
    bits = x.view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)


def read_weight(meta):
    with Path(meta['path']).open('rb') as file:
        file.seek(meta['offset'])
        raw = file.read(meta['bytes'])
    require(len(raw)==meta['bytes'], 'Truncated checkpoint slice')
    return raw


def decode_fp8(raw):
    b=np.frombuffer(raw,dtype=np.uint8)
    require(np.all((b & 127)!=127),'Nonfinite E4M3FN')
    e=((b>>3)&15).astype(np.int32)
    mant=(b&7).astype(np.float32)
    values=np.where(e==0,mant/512.,np.ldexp(8.+mant,e-10)).astype(np.float32)
    return np.where(b&128,-values,values).astype(np.float32)


def metrics(a, truth, norm):
    error=np.abs(a.astype(np.float64)-truth)
    bound=1e-5+2e-6*norm
    return dict(max_abs=float(error.max()), rms=float(np.sqrt(np.mean(error*error))),
                bound_failures=int(np.count_nonzero(error>bound)),max_bound_fraction=float((error/bound).max()),
                bf16_bins_vs_rounded_f64=int(np.count_nonzero(rne(a).view(np.uint32)!=rne(truth).view(np.uint32))))


def pair(a,b):
    return dict(max_abs=float(np.abs(a.astype(np.float64)-b).max()),
                unequal=int(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32))),
                bf16_bins=int(np.count_nonzero(rne(a).view(np.uint32)!=rne(b).view(np.uint32))))


def tree_reduce(values, axis):
    """Fixed F32 balanced addition; no F64 summation hidden in the candidate."""
    values=np.moveaxis(values,axis,-1)
    n=values.shape[-1]
    if n & (n-1):
        values=np.pad(values,[(0,0)]*(values.ndim-1)+[(0,(1 << (n-1).bit_length())-n)])
    while values.shape[-1]>1:
        values=(values[...,::2]+values[...,1::2]).astype(np.float32)
    return values[...,0]


def reductions(x,w):
    """Bound memory by N=256; emulate production only to identify its order.

    Serial mirrors the independent scalar's RN FMA for verified exact BF16
    products. `vector_two_bank` and `pairwise_k` are CPU experiments only.
    WMMA internal accumulation is intentionally NOT emulated.
    """
    require(x.dtype==w.dtype==np.float32 and np.array_equal(x,rne(x)) and np.array_equal(w,rne(w)), 'Expected BF16 values represented in F32')
    m,k=x.shape; n=w.shape[0]
    require(k%128==0 and w.shape[1]==k,'Packed real projection requires K multiple of 128')
    results={key:np.empty((m,n),dtype=np.float32) for key in ('serial','vector_replay','vector_two_bank','pairwise_k')}
    truth=np.empty((m,n),dtype=np.float64); norms=truth.copy()
    for col in range(0,n,256):
        weight=w[col:col+256]
        product=x[:,None,:].astype(np.float64)*weight[None,:,:].astype(np.float64)
        require(np.array_equal(product,product.astype(np.float32).astype(np.float64)), 'BF16 products not exact F32; serial replay assumption invalid')
        truth[:,col:col+len(weight)]=product.sum(-1,dtype=np.float64)
        norms[:,col:col+len(weight)]=np.abs(product).sum(-1,dtype=np.float64)
        serial=np.zeros(product.shape[:2],dtype=np.float32)
        for inner in range(k):
            serial=(serial.astype(np.float64)+product[:,:,inner]).astype(np.float32)
        results['serial'][:,col:col+len(weight)]=serial
        results['pairwise_k'][:,col:col+len(weight)]=tree_reduce(product.astype(np.float32),-1)
        parts=product.reshape(m,len(weight),k//128,32,4)
        sums=np.zeros((m,len(weight),32,4),dtype=np.float32)
        banks=np.zeros((2,m,len(weight),32,4),dtype=np.float32)
        for tile in range(k//128):
            sums=(sums.astype(np.float64)+parts[:,:,tile]).astype(np.float32)
            bank=tile%2
            banks[bank]=(banks[bank].astype(np.float64)+parts[:,:,tile]).astype(np.float32)
        for name,acc in [('vector_replay',sums),('vector_two_bank',banks[0]+banks[1])]:
            lanes=(acc[...,0]+acc[...,1])+(acc[...,2]+acc[...,3])
            for offset in (16,8,4,2,1):
                lanes[...,:32-offset]=lanes[...,:32-offset]+lanes[...,offset:]
            results[name][:,col:col+len(weight)]=lanes[...,0]
    return results,truth,norms


def prepare(args):
    events=[json.loads(line) for line in (args.trace/'events.jsonl').read_text().splitlines()]
    event=next(e for e in events if e['sequence']==args.sequence)
    require(event['event']=='linear.output' and event['numeric'] and event['precision']=='bf16_rne','Select a numeric W8A16 linear.output')
    prior=[e for e in events if e['sequence']<event['sequence'] and e['stage']==event['stage'] and e['event']=='linear.input' and e['parameter']==event['parameter']]
    require(prior,'No captured input for selected projection')
    input_event=prior[-1]
    m,n=event['shape']; nw,k=event['weight']['shape']
    require(nw==n and input_event['shape']==[m,k] and event['scale']['dtype']=='Bf16' and event['weight']['dtype']=='F8E4M3','Unexpected captured layout/storage')
    def read(e):
        path=args.trace/e['file']
        require(path.resolve().parent==args.trace.resolve(),'Escaping capture path')
        a=np.fromfile(path,dtype='<f4')
        require(a.size==np.prod(e['shape']),'Truncated capture')
        return a.reshape(e['shape'])
    original=read(input_event); x=rne(original); actual=read(event)
    raw=read_weight(event['weight']); scales_raw=read_weight(event['scale'])
    scale_words=np.frombuffer(scales_raw,dtype='<u2').reshape(event['scale']['shape'])
    scale=(scale_words.astype(np.uint32)<<16).view(np.float32)
    require(np.all(np.isfinite(scale)&(scale>0)),'Invalid numeric scale')
    w=decode_fp8(raw).reshape(n,k)
    for row in range(scale.shape[0]):
        for col in range(scale.shape[1]):
            w[row*128:(row+1)*128,col*128:(col+1)*128]*=scale[row,col]
    w=rne(w)
    results,truth,norm=reductions(x,w)
    report=dict(schema='ferrule.w8a16-captured-reductions.v1',scope='CPU reduction experiment on real operands; not CUDA execution or a model gate',
                source=dict(trace=str(args.trace),events_sha256=sha256(args.trace/'events.jsonl'),sequence=event['sequence'],stage=event['stage'],layer=event['layer'],parameter=event['parameter'],role=event['role'],
                            input_sha256=sha256(args.trace/input_event['file']),output_sha256=sha256(args.trace/event['file']),weight=event['weight'],scale=event['scale']),
                m=m,n=n,k=k,scale_dtype='Bf16',row_origin=0,column_origin=0,
                bound=dict(absolute=1e-5,sum_abs_products_factor=2e-6),
                production=metrics(actual,truth,norm),variants={},
                source_sha256={str(p):sha256(p) for p in [Path(__file__),ROOT/'crates/ferrule-backend/native/cuda/cutlass/numeric_w8a16.cuh',ROOT/'crates/ferrule-backend/tests/numeric_w8a16/reference.cu']})
    for name,value in results.items():
        report['variants'][name]=dict(vs_f64=metrics(value,truth,norm),vs_production=pair(value,actual))
    require(report['variants']['vector_replay']['vs_production']['unequal']==0,'Saved production does not match current packed vector order; investigate before changing it')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'shape.txt').write_text(f'{m} {n} {k}\n')
    ((x.view(np.uint32)>>16).astype('<u2')).tofile(args.output/'input.bf16')
    (args.output/'weight.fp8').write_bytes(raw)
    (args.output/'scale.bf16').write_bytes(scales_raw)
    actual.astype('<f4').tofile(args.output/'captured.f32')
    truth.astype('<f8').tofile(args.output/'truth.f64')
    norm.astype('<f8').tofile(args.output/'sumabs.f64')
    for name in ('serial','pairwise_k','vector_two_bank'):
        results[name].astype('<f4').tofile(args.output/(name+'.f32'))
    (args.output/'cpu-reductions.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    files=sorted(args.output.iterdir())
    (args.output/'SHA256SUMS').write_text(''.join(sha256(p)+'  '+p.name+'\n' for p in files))
    print(json.dumps(dict(m=m,n=n,k=k,parameter=event['parameter'],production=report['production'],variants=report['variants']),indent=2))
    require(not torch.cuda.is_initialized(),'CPU fixture generation initialized CUDA')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--sequence',type=int,required=True,help='Exact existing linear.output event number')
    p.add_argument('--output',type=Path,required=True)
    prepare(p.parse_args())


if __name__=='__main__': main()
