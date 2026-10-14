"""CPU-only pinned-HF regression for the actual cached-decode router tie."""
import hashlib
import inspect
import itertools
import json
import os
from pathlib import Path
import unittest

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['ATEN_CPU_CAPABILITY'] = 'avx512'
import torch
import transformers
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf
from hf_router_ties import canonicalize_router_ties

ROOT = Path(__file__).parent


class RouterTieContract(unittest.TestCase):
    def test_actual_decode_native_contract_and_profile_reference(self):
        native_bytes = (ROOT / 'hf_numeric_model.json').read_bytes()
        native = json.loads(native_bytes)
        profile = json.loads((ROOT / 'hf_numeric_profile.json').read_text())
        self.assertEqual(hashlib.sha256(native_bytes).hexdigest(), profile['native_oracle_sha256'])
        self.assertEqual(torch.__version__, native['torch'])
        self.assertEqual(transformers.__version__, native['transformers'])
        self.assertEqual(hashlib.sha256(Path(inspect.getfile(hf)).read_bytes()).hexdigest(), native['hf_source_sha256'])
        self.assertEqual(hashlib.sha256((ROOT / 'hf_router_ties.py').read_bytes()).hexdigest(), profile['router_hook_sha256'])
        self.assertIn('not guaranteed to be stable', torch.topk.__doc__)
        tie = profile['cases'][1]['router_ties'][0]
        self.assertEqual(tie['layer'], 1)
        x = torch.tensor(tie['input'], dtype=torch.bfloat16).reshape(1, 16).repeat(8, 1)
        router = hf.Qwen3_5MoeTopKRouter(hf.Qwen3_5MoeTextConfig(
            hidden_size=16, num_experts=4, num_experts_per_tok=3)).bfloat16()
        weight = native['tensors']['layers.1.feed_forward.router.weight']
        with torch.no_grad():
            router.weight.copy_(torch.tensor(weight['values']).reshape(4, 16))
            logits = torch.nn.functional.linear(x, router.weight)
            out = router(x)
        self.assertEqual(logits[0].float().tolist(), [-0.875, -0.10009765625, -0.875, -0.3046875])
        self.assertEqual(out[0][0].tolist(), tie['probabilities'])
        self.assertEqual(out[2][0].tolist(), [1, 3, 2])
        result = canonicalize_router_ties(logits, out)
        self.assertEqual(result[2][0].tolist(), [1, 3, 0])
        self.assertIs(result[0], out[0])
        self.assertIs(result[1], out[1])
        self.assertEqual(out[0][0, 0], out[0][0, 2])
        self.assertEqual(native['cases'][1]['routes'][1]['ids'], [[1, 3, 2]])
        self.assertEqual(profile['cases'][1]['routes'][1]['ids'], [[1, 3, 0]])
        self.assertEqual(native['cases'][1]['residuals'][3][0], -0.12451171875)
        self.assertEqual(profile['cases'][1]['residuals'][3][0], -0.1318359375)
        # Prefill/fork never encounter a tie: the original complete native oracle
        # remains exact, including conv/SSM/KV and every residual/logit.
        for index in (0, 3):
            for key, value in native['cases'][index].items():
                self.assertEqual(profile['cases'][index][key], value)
            self.assertEqual(profile['cases'][index]['router_ties'], [])

    def test_all_tiny_tie_patterns_only_substitute_equal_scores(self):
        for logits in itertools.product((-1., 0., 1.), repeat=4):
            probabilities = torch.tensor([logits]).softmax(-1)
            for k in range(1, 5):
                scores, ids = probabilities.topk(k, dim=-1)
                weights = scores / scores.sum(-1, keepdim=True)
                out = (probabilities, weights, ids)
                result = canonicalize_router_ties(torch.tensor([logits]), out)
                expected = sorted(range(4), key=lambda i: (-logits[i], i))[:k]
                self.assertEqual(result[2][0].tolist(), expected)
                self.assertTrue(torch.equal(probabilities.gather(-1, ids), probabilities.gather(-1, result[2])))
                self.assertIs(result[0], probabilities)
                self.assertIs(result[1], weights)
                if torch.equal(ids, result[2]):
                    self.assertIs(result, out)

    def test_unique_scores_leave_native_hf_result_untouched(self):
        for logits in itertools.permutations((-2., -1., 0., 1.)):
            p = torch.tensor([logits]).softmax(-1)
            weights, ids = p.topk(3, dim=-1)
            out = (p, weights / weights.sum(-1, keepdim=True), ids)
            self.assertIs(canonicalize_router_ties(torch.tensor([logits]), out), out)


if __name__ == '__main__':
    unittest.main()
