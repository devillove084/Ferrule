"""CPU-only F32 operation-order check on frozen HF stage operands.

No Torch import, device query, oracle regeneration, or tolerance. The companion
symbolic check compiles the actual helper with symbolic ordered operators;
this check connects that order to retained numeric intermediate evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np


def check(fixture):
    manifest = json.loads((fixture / 'hf-replay.json').read_text())
    blob = (fixture / 'hf-intermediates.safetensors').read_bytes()
    assert hashlib.sha256(blob).hexdigest() == manifest['expected_sha256']
    for name, pin in manifest['input_pins'].items():
        assert hashlib.sha256((fixture / name).read_bytes()).hexdigest() == pin
    size = struct.unpack('<Q', blob[:8])[0]
    header = json.loads(blob[8:8 + size])
    def tensor(name):
        meta = header[name]
        start, end = meta['data_offsets']
        data = blob[8 + size + start:8 + size + end]
        if meta['dtype'] == 'BF16':
            a = (np.frombuffer(data, dtype='<u2').astype('<u4') << 16).view('<f4')
        else:
            assert meta['dtype'] == 'F32'
            a = np.frombuffer(data, dtype='<f4')
        assert np.isfinite(a).all()
        return a.reshape(meta['shape']).copy()
    state = np.frombuffer((fixture / 'state.f32').read_bytes(), dtype='<f4').reshape(32,128,128).copy()
    k = tensor('knorm').reshape(32,128)
    q = tensor('qscaled').reshape(32,128)
    decay = tensor('decay').reshape(32,1)
    beta = tensor('beta').reshape(32,1)
    v = tensor('conv').reshape(-1)[4096:].reshape(32,128)
    metrics = {}
    def equal(name, actual):
        expected = tensor(name).reshape(actual.shape)
        indices = np.flatnonzero(actual.view('<u4').reshape(-1) != expected.view('<u4').reshape(-1))
        metrics[name] = dict(elements=actual.size, unequal=int(indices.size))
        assert not indices.size, (name, indices[:8].tolist())
    def reduce(update, factor):
        partial = []
        for lane in range(4):
            acc = [np.zeros((32,128), dtype='<f4') for _ in range(4)]
            for base in range(lane,128,16):
                for j in range(4):
                    d = base + j * 4
                    if d >= 128:
                        continue
                    kd = k[:,d,None]
                    if update:
                        state[:,d,:] = np.add(state[:,d,:], np.multiply(kd, factor))
                        x = q[:,d,None]
                    else:
                        state[:,d,:] = np.multiply(state[:,d,:], factor)
                        x = np.multiply(kd, np.float32(1))
                    acc[j] = np.add(acc[j], np.multiply(state[:,d,:], x))
            partial.append(np.add(np.add(np.add(acc[0],acc[1]),acc[2]),acc[3]))
        return np.add(np.add(partial[0],partial[2]),np.add(partial[1],partial[3]))
    memory = reduce(False,decay)
    equal('memory',memory); equal('decayed',state)
    correction = np.multiply(np.subtract(v,memory),beta)
    equal('correction',correction)
    dot = reduce(True,correction)
    equal('state',state); equal('dot',dot)
    bits = dot.view('<u4')
    rounded = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16) << 16
    equal('output',rounded.view('<f4'))
    return dict(gpu_execution=False, expected_sha256=manifest['expected_sha256'], stages=metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fixture',type=Path)
    print(json.dumps(check(parser.parse_args().fixture),indent=2))
