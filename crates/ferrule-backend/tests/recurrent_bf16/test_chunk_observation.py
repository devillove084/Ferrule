"""CPU-only integrity/stage checks for the bounded actual-input chunk replay."""
import argparse
import json
import os
from pathlib import Path
import unittest

# Never initialize CUDA while checking retained GPU evidence.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from safetensors import safe_open
import chunk_observation as replay

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--evidence', required=True, type=Path)
parser.add_argument('--without-matrix-observer', required=True, type=Path)
options, remaining = parser.parse_known_args()
OUT = options.evidence.resolve()
BASELINE = options.without_matrix_observer.resolve()
torch.set_num_threads(1)


def read(name):
    return json.loads((OUT/name).read_text())


class ActualChunkEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.observation = read('observation.json')
        cls.comparison = read('comparison.json')
        cls.stages = safe_open(str(OUT/'hf-stages.safetensors'), framework='pt', device='cpu')

    def same(self, a, b):
        self.assertEqual(a.dtype, b.dtype)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16)))

    def test_pins_inputs_and_production_unchanged(self):
        replay.verify()
        self.assertEqual(replay.sha(OUT/'hf-stages.safetensors'), self.observation['stage_sha256'])
        self.assertEqual(replay.sha(replay.__file__), self.observation['script_sha256'])
        self.assertEqual(replay.sha(replay.CAPTURE/'GDN-HANDOFF.json'), self.observation['handoff_sha256'])
        self.assertEqual(replay.sha(replay.FROZEN/'inputs.json'), self.observation['input_manifest_sha256'])
        launch = read('launch.json')
        for name in ['native','hf']:
            receipt = read(name+'-supervisor.json')
            self.assertEqual(receipt['returncode'], 0)
            self.assertFalse(receipt['timed_out'])
            self.assertFalse(receipt['cleanup_errors'])
            self.assertEqual(receipt['descendant_custody'], 'proven_drained')
            self.assertTrue(receipt['custody']['containment_verified'])
            self.assertTrue(receipt['custody']['echild'])
        self.assertEqual(launch['production_sha256'], replay.sha(replay.HEADER))
        self.assertEqual(read('release.json')['release'], 'released')
        self.assertTrue(read('release.json')['gpu_quiescence'])
        self.assertEqual(len(read('release.json')['after']), 3)

    def test_actual_inputs_normalization_and_independent_reference_exact(self):
        prefixes = ('producer.', 'independent.', 'normalization.', 'scan_vs_producer.',
                    'fresh_native_vs_production.', 'fresh_trace.')
        for key, value in self.comparison.items():
            if key.startswith(prefixes):
                self.assertEqual(value['unequal'], 0, key)
        for key in ['qnorm','knorm','qscaled']:
            self.assertEqual(self.comparison['scan_vs_chunk.'+key]['unequal'], 0)
        for t in range(5):
            self.assertEqual(self.comparison['scan_vs_recurrent.'+str(t)+'.decay']['unequal'], 0)
        self.assertEqual(self.stages.get_tensor('chunk.state').dtype, torch.float32)
        self.assertEqual(self.stages.get_tensor('chunk.output').dtype, torch.bfloat16)

    def test_first_comparable_recurrent_difference_and_chunk_boundary(self):
        for key in ['decayed_state','memory','correction']:
            self.assertEqual(self.comparison['scan_vs_recurrent.0.'+key]['unequal'], 0)
        self.assertEqual(self.comparison['scan_vs_recurrent.0.dot']['unequal'], 3333)
        self.assertEqual(self.comparison['scan_vs_recurrent.1.decayed_state']['unequal'], 0)
        self.assertEqual(self.comparison['scan_vs_recurrent.1.memory']['unequal'], 3388)
        self.assertEqual(self.comparison['recurrent_vs_chunk.output']['unequal'], 0)
        self.assertEqual(self.comparison['recurrent_vs_chunk.state']['unequal'], 297454)
        self.assertEqual(self.comparison['production_vs_chunk.output']['unequal'], 1)
        self.assertEqual(self.comparison['production_vs_chunk.state']['unequal'], 302175)
        # Fixture regression only, never an output correction or production rule.
        focus = self.observation['focus']
        self.assertEqual([focus[k]['bits'] for k in ['scan','recurrent','chunk']],
                         ['0x33f17ffe','0x33f18002','0x33f18001'])

    def test_matrix_observation_does_not_change_original_stages(self):
        baseline_manifest = json.loads((BASELINE/'observation.json').read_text())
        self.assertEqual(replay.sha(BASELINE/'hf-stages.safetensors'), baseline_manifest['stage_sha256'])
        baseline = safe_open(str(BASELINE/'hf-stages.safetensors'), framework='pt', device='cpu')
        self.assertEqual(len(baseline.keys()), 53)
        for name in baseline.keys():
            self.same(self.stages.get_tensor(name), baseline.get_tensor(name))
        matrices = [k for k in self.stages.keys() if '.matmul.' in k]
        self.assertEqual(len(matrices), 24)
        for name in matrices:
            self.assertEqual(self.stages.get_tensor(name).dtype, torch.float32)
        for index, name in [(1,'value'), (2,'k_cumdecay'), (4,'v_prime'), (5,'attn_inter')]:
            self.same(self.stages.get_tensor('chunk.matmul.'+str(index)+'.result'),
                      self.stages.get_tensor('chunk.'+name))

    def test_real_chunk_padding_and_reassociated_decay(self):
        stages = self.stages
        # Fixed 64-token padding, even though only five tokens are real.
        q = stages.get_tensor('chunk.qscaled')
        self.assertEqual(tuple(q.shape), (1,32,64,128))
        self.assertEqual(torch.count_nonzero(q[:,:,5:]).item(), 0)
        for name in ['chunk.v_prime','chunk.attn_inter']:
            self.assertEqual(torch.count_nonzero(stages.get_tensor(name)).item(), 0)
        d = stages.get_tensor('chunk.decay_mask').reshape(32,64,64)
        native = torch.frombuffer(bytearray((replay.FROZEN/'scan-decay.f32').read_bytes()),
                                  dtype=torch.float32).reshape(5,32)
        # Compare original observed values, no CPU reproduction of CUDA math.
        self.assertEqual((d[:,1,0].contiguous().view(torch.int32) != native[1].view(torch.int32)).sum().item(), 4)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main(argv=[__file__, *remaining])
