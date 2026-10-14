"""CPU source/extent guards for the authorized fusion; never claims GPU parity."""
import argparse
import json
import re
from pathlib import Path
import unittest
from performance import HERE, NATIVE, ROOT, sha
from chunk_fusion import ARCHIVE_SHA

parser = argparse.ArgumentParser()
parser.add_argument('--prepared', type=Path, required=True)
args, rest = parser.parse_known_args()
SOURCE = (NATIVE / 'core/recurrent_bf16_chunk.cuh').read_text()
OLD = (HERE / 'chunk_unfused.cuh').read_text()


def definition(text, marker):
    start = text.index(marker)
    begin = text.index('{', start)
    depth = 1
    end = begin + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


class FusionCPU(unittest.TestCase):
    def test_archive_and_all_nonfused_math_unchanged(self):
        self.assertEqual(sha(HERE / 'chunk_unfused.cuh'), ARCHIVE_SHA)
        for marker in ['struct Workspace', 'inline bool span(',
                       '__device__ inline float inv_l2(', 'inline float query_scale(',
                       'inline uint32_t prefix_window(', '__global__ void prepare(',
                       '__global__ void prefix(', '__global__ void matmul(',
                       '__global__ void mask_matrix(', '__global__ void triangular(']:
            self.assertEqual(definition(SOURCE, marker), definition(OLD, marker), marker)

    def test_launch_graph_and_rust_metadata(self):
        launch = definition(SOURCE, 'ferrule_core_bf16_chunk_launch_impl(')
        self.assertEqual(len(re.findall(r'CHUNK_MM\(w\.', launch)), 8)
        names = re.findall(r'\b(\w+)<<<', launch)
        self.assertEqual(names, ['matmul', 'prepare', 'prefix',
            'decay_and_initial_exponents', 'mask_matrix', 'triangular',
            'attention_delta_and_end_keys', 'output_and_state'])
        self.assertEqual(len(names) - 1 + 8, 15)
        self.assertIn('launches_per_chunk = 15;', SOURCE)
        plan = (ROOT / 'crates/ferrule-backend/src/recurrent_bf16_chunk_plan.rs').read_text()
        self.assertIn('.checked_mul(15)', plan)
        self.assertIn('launches: 1,', plan)
        self.assertLess(launch.index('CHUNK_MM(w.kc,'), launch.index('attention_delta_and_end_keys<<<'))
        self.assertLess(launch.index('CHUNK_MM(w.ke,'), launch.index('output_and_state<<<'))

    def test_fused_independent_bounds_and_no_barriers(self):
        for kernel, bounds in [('decay_and_initial_exponents', ['nm', 'nk']),
                              ('attention_delta_and_end_keys', ['nm', 'nv', 'nk']),
                              ('output_and_state', ['nv', 'ns'])]:
            body = definition(SOURCE, '__global__ void ' + kernel + '(')
            for bound in bounds:
                self.assertIn('if (i < w.' + bound + ')', body)
            for forbidden in ['return;', '__syncthreads', '__fmaf', 'atomic']:
                self.assertNotIn(forbidden, body)
        for forbidden in ['cudaMalloc', 'cudaMemcpy', 'cudaDeviceSynchronize',
                          'cudaStreamSynchronize', 'cudaFree', 'cublas', '__float_to_tf32']:
            self.assertNotIn(forbidden, SOURCE)

    def test_scalar_expressions_are_retained_verbatim(self):
        # Whitespace normalization only, not algebraic normalization.
        old = re.sub(r'\s+', '', OLD)
        new = re.sub(r'\s+', '', SOURCE)
        for expression in [
            'r >= c ? expf(__fsub_rn(w.prefix[h*C+r],w.prefix[h*C+c])) : 0.0f',
            '__fmul_rn(w.kb[i],e)', '__fmul_rn(w.q[i],e)',
            'r >= c ? __fmul_rn(w.attn[i],w.decay[i]) : 0.0f',
            '__fsub_rn(w.value[i],w.vp[i])',
            '__fmul_rn(w.k[i],expf(__fsub_rn(w.prefix[h*C+C-1],g)))',
            'bf16_rne(__fadd_rn(w.inter[i],w.intra[i]))',
            '__fadd_rn(__fmul_rn(s[i],expf(w.prefix[h*C+C-1])),w.state_product[i])']:
            expression = re.sub(r'\s+', '', expression)
            self.assertIn(expression, old)
            self.assertIn(expression, new)

    def test_independent_extent_and_tail_coverage(self):
        for dk, dv in [(1,3), (4,7), (16,9), (17,65), (65,17),
                       (128,128), (511,512), (512,511)]:
            nm, nk, nv, ns = 2*64*64, 2*64*dk, 2*64*dv, 2*dk*dv
            for extents in [(nm,nk), (nm,nv,nk), (nv,ns)]:
                threads = ((max(extents)+255)//256)*256
                self.assertGreaterEqual(threads, max(extents))
            for rows in [2,5,63,64,65,129]:
                start = (rows-1)//64*64
                self.assertEqual(sum(start+t < rows for t in range(64)), rows-start)
        # ns>nv (Dk128) ensures padded output threads must still update S.
        self.assertGreater(2*128*128, 2*64*128)

    def test_prepared_pins_contracts_and_no_gpu_claim(self):
        out = args.prepared.resolve()
        manifest = json.loads((out / 'fusion-prepared.json').read_text())
        self.assertEqual(manifest['status'], 'CPU_PREPARED_GPU_NOT_RUN')
        for path, pin in manifest['source_pins'].items():
            self.assertEqual(sha(path), pin, path)
        for path, pin in manifest['artifact_pins'].items():
            self.assertEqual(sha(out / path), pin, path)
        for path in ['fusion-build.json', 'math_cpu-build.json',
                     'baseline/build.json', 'baseline/contracts.json',
                     'continuation/build.json', 'continuation/contracts.json',
                     'matrices/build.json']:
            self.assertEqual(json.loads((out / path).read_text())['returncode'], 0, path)
        self.assertEqual(len(json.loads((out/'extra-jobs.json').read_text())), 6)
        self.assertFalse((out / 'stage-comparison.json').exists())


if __name__ == '__main__':
    unittest.main(argv=[__file__, *rest])
