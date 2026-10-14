"""CPU-only source/captured-stage evidence, not a CUDA fallback or GPU pass.

Requires the host-only chunk_math_cpu binary compiled from the production header.
Frozen HF/native captures are read-only; no model/reference regeneration.
"""
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import unittest

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['NVIDIA_VISIBLE_DEVICES']='void'
import numpy as np
from safetensors import safe_open
from performance import ROOT, CUDA_LIB, sha

CAPTURE=ROOT/'target/validation/chunk-continuation-performance-gpu-20261006'
STAGES=CAPTURE/'stage-handoff'
SCAN=Path('/opt/conda310/lib/python3.10/site-packages/torch/include/ATen/native/cuda/ScanUtils.cuh')
HF=Path('/opt/conda310/lib/python3.10/site-packages/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py')
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',required=True,type=Path)
p.add_argument('--report',required=True,type=Path)
args,rest=p.parse_known_args()
REPORT=dict(gpu_execution=False,scope='CPU arithmetic/source evidence only',historical_serial_reduction='Retained pre-fix K512 serial FMA differs from HF; current split-K production gates are checked separately by test_chunk_splitk.py')


def scan_window(data,window):
    """F32 emulation of the source operation order, never used by production."""
    result=np.empty_like(data)
    carry=np.zeros(data.shape[:-1],np.float32)
    for start in range(0,64,window):
        row=data[...,start:start+window].copy()
        row[...,0]=row[...,0]+carry
        stride=1
        while stride<window:
            old=row.copy()
            for t in range(window):
                if t&stride:row[...,t]=old[...,t]+old[...,(t&~(2*stride-1))+stride-1]
            stride*=2
        result[...,start:start+window]=row
        carry=row[...,-1]
    return result


def metric(a,b):
    return dict(elements=a.size,unequal=int(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32))),max_abs=float(np.max(np.abs(a-b))))


def bf16_rne(x):
    bits=x.copy().view(np.uint32)
    bits+=np.uint32(0x7fff)+((bits>>16)&1)
    bits&=np.uint32(0xffff0000)
    return bits.view(np.float32)


class ChunkMathCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert sha(CAPTURE/'BACKEND-HANDOFF.json')=='27f3ec44237fbea9af4cf79c030cac7aedd5b38b16197d63c1b861ba39093c27'
        handoff=json.loads((CAPTURE/'BACKEND-HANDOFF.json').read_text())
        for name,m in handoff['cases'].items():
            assert sha(STAGES/name/'dense-stage-comparison.json')==m['comparison_sha256']
            assert sha(STAGES/name/'hf-observed-stages.safetensors')==m['observed_hf_stages_sha256']
        assert sha(SCAN)=='d323f3dcb5e175feed840294e67e82300fdd6dabdad516072f5497bab103d206'
        assert sha(HF)=='0f8ca77e875c31d448137c9ac291bed505c6b9927a85b5b8460af29d1e1ee647'
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='void',LD_LIBRARY_PATH=CUDA_LIB+':/usr/lib64')
        r=subprocess.run([str(args.binary.resolve())],env=env,capture_output=True,text=True,timeout=15)
        assert r.returncode==0,r.stderr
        cls.host=json.loads(r.stdout)
        REPORT['production_host_helpers']=cls.host
        REPORT['source_pins']={str(p):sha(p) for p in [SCAN,HF,Path(__file__),args.binary,ROOT/'crates/ferrule-backend/native/cuda/core/recurrent_bf16_chunk.cuh']}

    def test_all_admitted_query_scales_match_python_double_then_f32(self):
        expected=np.array([1/(dk**0.5) for dk in range(1,513)],np.float32).view(np.uint32).tolist()
        self.assertEqual(self.host['scales'],expected)
        self.assertEqual(expected[510],0x3d353245)
        self.assertEqual(expected[127],0x3db504f3)

    def test_scan_geometry_matches_pinned_uint32_source(self):
        for rows,heads,window in self.host['windows']:
            logical=heads*((rows+63)//64)
            log_y=(logical-1).bit_length()
            # ScanUtils instantiates get_log_num_threads_x_inner_scan<uint32_t>.
            diff=(6-log_y)&0xffffffff
            log_x=min(max(((9+diff)&0xffffffff)//2,4),9)
            self.assertEqual(window,min(64,2*(1<<log_x)),(rows,heads))
        windows={(r,h):w for r,h,w in self.host['windows']}
        self.assertEqual(windows[64,32],64)
        self.assertEqual(windows[65,32],32)
        self.assertEqual(windows[129,32],32)
        self.assertEqual(windows[65,2],64)

    def test_dense129_source_tree_predicts_all_frozen_hf_prefix_words(self):
        d=STAGES/'dense129'
        with safe_open(str(d/'hf-observed-stages.safetensors'),framework='numpy') as f:
            g=f.get_tensor('g');hf=f.get_tensor('g_prefix')
        old=scan_window(g,64);fixed=scan_window(g,32)
        metrics=[]
        for chunk in range(3):
            native=np.fromfile(d/('chunk'+str(chunk)+'-g_prefix.f32'),dtype=np.float32).reshape(1,32,64)
            np.testing.assert_array_equal(old[:,:,chunk].view(np.uint32),native.view(np.uint32))
            self.assertEqual(metric(fixed[:,:,chunk],hf[:,:,chunk])['unequal'],0)
            metrics.append(dict(old=metric(old[:,:,chunk],hf[:,:,chunk]),candidate=metric(fixed[:,:,chunk],hf[:,:,chunk])))
        self.assertEqual([m['old']['unequal'] for m in metrics],[378,372,0])
        REPORT['dense129_prefix']=metrics

    def test_511_double_scale_explains_every_observed_scaled_query(self):
        d=STAGES/'near-511-512'
        with safe_open(str(d/'hf-observed-stages.safetensors'),framework='numpy') as f:hf=f.get_tensor('qscaled').reshape(-1)
        actual=np.fromfile(d/'chunk0-qscaled.f32',dtype=np.float32)
        old=np.float32(1)/np.sqrt(np.float32(511))
        new=np.array(self.host['scales'][510],np.uint32).view(np.float32)
        # The observation omitted qnorm. Recover its BF16 candidate and require
        # exact reproduction of every native old-scale word before using it.
        qnorm=bf16_rne(actual/old)
        np.testing.assert_array_equal((qnorm*old).view(np.uint32),actual.view(np.uint32))
        predicted=qnorm*new
        self.assertEqual(metric(actual,hf)['unequal'],1912)
        self.assertEqual(metric(predicted,hf)['unequal'],0)
        REPORT['Dk511_scale']=dict(old_bits=hex(int(old.view(np.uint32))),new_bits=hex(int(new.view(np.uint32))),
            original=metric(actual,hf),candidate=metric(predicted,hf),
            qualification='BF16 qnorm reconstructed and all original scaled words reproduced; not a newly observed HF qnorm tensor')

    def test_k512_serial_fma_reproduces_native_but_not_hf(self):
        fma=ctypes.CDLL('libm.so.6').fmaf
        fma.argtypes=[ctypes.c_float]*3;fma.restype=ctypes.c_float
        results={}
        for case in ['near-512-511','limit-512-512']:
            d=STAGES/case
            with safe_open(str(d/'hf-observed-stages.safetensors'),framework='numpy') as f:
                a=f.get_tensor('k_beta').reshape(64,512);b=f.get_tensor('knorm').reshape(64,512);hf=f.get_tensor('matmul0').reshape(64,64)
            native=np.fromfile(d/'chunk0-matmul0.f32',dtype=np.float32).reshape(64,64)
            np.testing.assert_array_equal(a.reshape(-1).view(np.uint32),np.fromfile(d/'chunk0-k_beta.f32',dtype=np.uint32))
            np.testing.assert_array_equal(b.reshape(-1).view(np.uint32),np.fromfile(d/'chunk0-knorm.f32',dtype=np.uint32))
            self.assertFalse(np.any(a[2:]));self.assertFalse(np.any(b[2:]))
            serial=np.zeros((64,64),np.float32)
            for r in range(2):
                for c in range(2):
                    value=0.0
                    for k in range(512):value=fma(float(a[r,k]),float(b[c,k]),value)
                    serial[r,c]=value
            np.testing.assert_array_equal(serial.view(np.uint32),native.view(np.uint32))
            self.assertEqual(metric(serial,hf)['unequal'],4)
            candidates={}
            for kind in ['contiguous','interleaved']:
                for lanes in [2,4,8,16,32]:
                    out=np.zeros((64,64),np.float32)
                    for r in range(2):
                        for c in range(2):
                            accum=[0.0]*lanes
                            for k in range(512):
                                lane=k%lanes if kind=='interleaved' else k//(512//lanes)
                                accum[lane]=fma(float(a[r,k]),float(b[c,k]),accum[lane])
                            total=np.float32(0)
                            for value in accum:total=np.float32(total+np.float32(value))
                            out[r,c]=total
                    candidates[kind+str(lanes)]=metric(out,hf)
            self.assertTrue(all(m['unequal'] for m in candidates.values()))
            results[case]=dict(serial_vs_hf=metric(serial,hf),serial_native_exact=True,unproven_candidates=candidates)
        REPORT['historical_K512_serial_reduction']=results


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ChunkMathCPU)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    REPORT['tests_passed']=result.wasSuccessful()
    REPORT['tests_run']=result.testsRun
    with args.report.open('x') as f:json.dump(REPORT,f,indent=2);f.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
