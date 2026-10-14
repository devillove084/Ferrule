"""CPU exact-rational diagnosis of frozen failed coordinates; never an oracle replacement."""
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path
import struct
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pow2(e):
    return F(2**e) if e >= 0 else F(1, 2**-e)


def rne(x, precision):
    if not x:
        return F(0)
    sign = -1 if x < 0 else 1
    x = abs(x)
    e = x.numerator.bit_length() - x.denominator.bit_length()
    if x < pow2(e):
        e -= 1
    step = pow2(e - precision + 1)
    q, rem = divmod(x / step, 1)
    n = int(q)
    if rem > F(1, 2) or (rem == F(1, 2) and n % 2):
        n += 1
    return sign * n * step


def bf16(w):
    return F(struct.unpack('<f', struct.pack('<I', w << 16))[0])


def word(x):
    return struct.unpack('<I', struct.pack('<f', float(rne(x, 8))))[0] >> 16


def fp8(w):
    sign = -1 if w & 128 else 1
    e, mantissa = (w >> 3) & 15, w & 7
    assert not (e == 15 and mantissa == 7)
    return sign * (F(mantissa, 8) * pow2(-6) if e == 0 else F(8 + mantissa, 8) * pow2(e - 7))


def words(path):
    raw = path.read_bytes()
    return struct.unpack('<' + 'H' * (len(raw) // 2), raw)


def main(run):
    root = Path(__file__).resolve().parents[4]
    release = root / 'papers/arxiv/experiments/tp2_mixed/oracle/runs/routed-down-m1-cuda-01'
    assert sha(release / 'manifest.json') == '4f3081e16cc496121817bd49e4732cb5b49a030ca9853f616751c17d23ba1704'
    records = []
    for case in json.loads((release / 'manifest.json').read_text())['fixtures']:
        directory = release / case['directory']
        assert sha(directory / 'SHA256SUMS') == case['sha256']
        for line in (directory / 'SHA256SUMS').read_text().splitlines():
            digest, name = line.split()
            assert sha(directory / name) == digest
        for report in json.loads((run / case['directory'] / 'exact.json').read_text()):
            if report['comparison'] != 'w8-vs-hf':
                continue
            lane = report['rank']
            for mismatch in report['mismatches']:
                col = mismatch['column']
                inputs = [bf16(w) for w in words(directory / f'rank{lane}.input.bf16')]
                raw = (directory / f'rank{lane}.weight.fp8').read_bytes()[col * 256:(col + 1) * 256]
                scales = words(directory / f'rank{lane}.scale.bf16')
                weights = [rne(fp8(w) * bf16(scales[(col // 128) * 2 + k // 128]), 8) for k, w in enumerate(raw)]
                products = [x * w for x, w in zip(inputs, weights)]
                lanes = []
                for lane_id in range(32):
                    sums = [F(0)] * 4
                    for k in range(lane_id * 4, 256, 128):
                        for j in range(4):
                            sums[j] = rne(products[k + j] + sums[j], 24)
                    lanes.append(rne(rne(sums[0] + sums[1], 24) + rne(sums[2] + sums[3], 24), 24))
                for offset in (16, 8, 4, 2, 1):
                    old = lanes[:]
                    for lane_id in range(32 - offset):
                        lanes[lane_id] = rne(old[lane_id] + old[lane_id + offset], 24)
                simt = lanes[0]
                assert word(simt) == mismatch['actual_word'], 'CPU reduction must reproduce actual GPU W8 word'
                exact = sum(products, F(0))
                sequential = F(0)
                for value in products:
                    sequential = rne(sequential + value, 24)
                pairwise = products[:]
                while len(pairwise) > 1:
                    pairwise = [rne(a + b, 24) for a, b in zip(pairwise[::2], pairwise[1::2])]
                mid = (bf16(mismatch['actual_word']) + bf16(mismatch['expected_word'])) / 2
                record = dict(case=case['directory'], rank=lane, row=0, column=col,
                    gpu_w8_word=hex(mismatch['actual_word']), hf_native_word=hex(mismatch['expected_word']),
                    cpu_simt_word=hex(word(simt)), cpu_simt_f32=float(simt), exact_dot=float(exact),
                    exact_dot_rational=str(exact), bf16_midpoint=float(mid),
                    simt_minus_midpoint=float(simt - mid), exact_minus_midpoint=float(exact - mid),
                    exact_direct_bf16=hex(word(exact)), exact_f32_then_bf16=hex(word(rne(exact, 24))),
                    sequential_f32_word=hex(word(sequential)), sequential_f32=float(sequential),
                    pairwise_f32_word=hex(word(pairwise[0])), pairwise_f32=float(pairwise[0]),
                    inference='CPU SIMT schedule reproduces GPU word; alternative sums are diagnostic, not HF acceptance')
                records.append(record)
    result = dict(cpu_only=True, expected_modified=False, production_modified=False,
        script_sha256=sha(Path(__file__)), fixture_release_sha256=sha(release / 'manifest.json'),
        gpu_release_sha256=sha(run / 'release.json'), coordinates=records)
    with (run / 'reduction-diagnosis.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
