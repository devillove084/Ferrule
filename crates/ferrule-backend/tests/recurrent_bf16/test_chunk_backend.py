"""CPU checks of current gate code against retained release evidence; no CUDA.

Historical provenance is anchored to the immutable release manifest/artifacts,
not today's mutable source tree. No archived original sources were retained for
this release: their recorded hashes identify provenance but cannot prove current
source equivalence. Current-code acceptance is tested separately below.
"""
import argparse
import ast
import contextlib
import ctypes
import hashlib
import io
import json
import math
import os
from pathlib import Path
import unittest
from unittest import mock

os.environ['CUDA_VISIBLE_DEVICES']=''
from performance import ROOT, HERE, sha
import chunk_gate

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--prepared',required=True,type=Path)
parser.add_argument('--evidence',required=True,type=Path)
args,rest=parser.parse_known_args()
OUT=args.prepared.resolve()
EVIDENCE=args.evidence.resolve()

# Independent trust anchors for the retained 20261006 release. Never regenerate
# these from current sources or rewrite the historical manifest to accept edits.
RETAINED_PREPARED_SHA256 = 'e2fa7041c255738c0c852fc79789411e9d2c15f6ccbba7a427524bf9b6309f34'
RETAINED_BINARY_SHA256 = '7d6f19b12d6314a21e73dd42287195ba2e1c5f1b04910afc272da89b32d2372b'
RETAINED_FIXTURE_PINS_SHA256 = 'c6d821d54ed8a7fe94197cfee75b79eac388d8ae793fc3043463a33c9ab9dc99'


class ChunkBackendTests(unittest.TestCase):
    def test_retained_provenance_and_immutable_fixture(self):
        # Hashing the manifest anchors every recorded source/artifact pin without
        # claiming the old source hash describes the code being tested today.
        self.assertEqual(sha(OUT/'prepared.json'), RETAINED_PREPARED_SHA256)
        launch=json.loads((EVIDENCE/'launch.json').read_text())
        self.assertEqual(launch['prepared_sha256'], RETAINED_PREPARED_SHA256)
        m=json.loads((OUT/'prepared.json').read_text())
        self.assertEqual(m['artifact_pins']['chunk'], RETAINED_BINARY_SHA256)
        for path,pin in m['artifact_pins'].items():self.assertEqual(sha(OUT/path),pin,path)
        self.assertEqual(sha(HERE/'chunk.sha256'), RETAINED_FIXTURE_PINS_SHA256)
        for line in (HERE/'chunk.sha256').read_text().splitlines():
            pin,name=line.split();self.assertEqual(sha(OUT/'actual'/name),pin,name)

    def test_current_compare_accepts_retained_numerical_evidence(self):
        # Exercise today's comparisons and receipt gate, preserving all evidence.
        with mock.patch.object(chunk_gate, 'write_json') as write, contextlib.redirect_stdout(io.StringIO()):
            result = chunk_gate.compare(OUT)
        self.assertEqual(len(result), 10)
        self.assertEqual(write.call_count, 2)
        self.assertEqual(write.call_args_list[0].args, (OUT/'comparison.json', result))
        path, summary = write.call_args_list[1].args
        self.assertEqual(path, OUT/'exact-gates.json')
        self.assertIs(summary['all_cases_exact'], True)
        self.assertIs(summary['dense_observations_exact'], True)
        self.assertIs(summary['tolerance_used'], False)
        receipt = summary['dense_stage_receipt']
        self.assertEqual(receipt['sha256'], chunk_gate.DENSE_STAGE_RECEIPT_SHA256)
        self.assertEqual(receipt['stages'], 30)

    def test_actual_all_observed_stages_and_matrices_exact(self):
        c=json.loads((OUT/'comparison.json').read_text())['actual']
        self.assertEqual(c['output']['elements'],20480)
        self.assertEqual(c['state']['elements'],524288)
        for name,metric in c.items():self.assertEqual(metric['unequal'],0,name)
        self.assertEqual(len([n for n in c if n.startswith('isolated_matmul')]),8)

    def test_structured_tails_nonzero_state_and_dense_inputs_exact(self):
        c=json.loads((OUT/'comparison.json').read_text())
        for case in ['synthetic-2','synthetic-5','synthetic-63','synthetic-64','synthetic-65','synthetic-67','synthetic-129','dense-5']:
            self.assertEqual(c[case]['output']['unequal'],0,case)
            self.assertEqual(c[case]['state']['unequal'],0,case)
        self.assertEqual(c['dense-65']['output']['unequal'],0)
        self.assertEqual(c['dense-65']['state']['unequal'],0)
        self.assertEqual(c['dense-65']['state']['max_abs'],0)
        status=json.loads((OUT/'exact-gates.json').read_text())
        receipt = chunk_gate.validate_dense_stage_receipt()
        self.assertEqual(receipt['stages'], 30)
        if 'dense_stage_receipt' in status:
            self.assertEqual(status['dense_stage_receipt'], receipt)
        self.assertTrue(status['dense_observations_exact'])
        self.assertTrue(status['all_cases_exact'])
        self.assertFalse(status['tolerance_used'])

    def test_native_contracts_resources_and_authenticated_release(self):
        self.assertEqual(json.loads((OUT/'contracts.json').read_text())['returncode'],0)
        m=json.loads((OUT/'prepared.json').read_text())
        for name in ['hf',*[c['name'] for c in m['cases']]]:
            r=json.loads((EVIDENCE/(name+'-supervisor.json')).read_text())
            self.assertEqual(r['returncode'],0,name)
            self.assertEqual(r['descendant_custody'],'proven_drained',name)
            self.assertTrue(r['custody']['containment_verified'])
            self.assertTrue(r['custody']['echild'])
            self.assertFalse(r['cleanup_errors'])
            if name!='hf':
                v=json.loads(r['stdout']);self.assertTrue(v['guards']);self.assertTrue(v['inputs_immutable']);self.assertTrue(v['observation_exact'])
        self.assertEqual(json.loads((EVIDENCE/'release.json').read_text())['release'],'released')

    def test_source_no_launch_allocation_or_synchronization(self):
        native=(ROOT/'crates/ferrule-backend/native/cuda/core/recurrent_bf16_chunk.cuh').read_text()
        for forbidden in ['cudaMalloc','cudaFree','cudaMemcpy','cudaDeviceSynchronize','cudaStreamSynchronize','cublas','__float_to_tf32']:
            self.assertNotIn(forbidden,native)
        self.assertIn('__fmaf_rn',native)
        rust=(ROOT/'crates/ferrule-backend/src/cuda/operators/recurrent_bf16_chunk.rs').read_text()
        self.assertIn('check_buffer_owner',rust)
        self.assertIn('disjoint(',rust)
        self.assertIn('submit_operator(plan.launch_count()',rust)
        for path in [HERE/'chunk_gate.py',HERE/'test_chunk_backend.py']:
            ast.parse(path.read_text())


class DenseStageReceiptTests(unittest.TestCase):
    def test_retained_receipt_valid(self):
        receipt = chunk_gate.validate_dense_stage_receipt()
        self.assertEqual(receipt['sha256'], chunk_gate.DENSE_STAGE_RECEIPT_SHA256)
        self.assertEqual(receipt['stages'], 30)

    def test_missing_receipt_rejected(self):
        with mock.patch.object(Path, 'read_bytes', side_effect=FileNotFoundError):
            with self.assertRaisesRegex(ValueError, 'receipt unavailable'):
                chunk_gate.validate_dense_stage_receipt()

    def test_wrong_hash_rejected(self):
        payload = chunk_gate.DENSE_STAGE_RECEIPT.read_bytes() + b'\n'
        with mock.patch.object(Path, 'read_bytes', return_value=payload):
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                chunk_gate.validate_dense_stage_receipt()

    def reject_mutation(self, mutate, message):
        receipt = json.loads(chunk_gate.DENSE_STAGE_RECEIPT.read_bytes())
        mutate(receipt)
        payload = json.dumps(receipt).encode()
        # Re-pin only inside this test so semantic guards, not just the digest,
        # must reject a malformed receipt. Production has no pin override flag.
        with mock.patch.object(Path, 'read_bytes', return_value=payload), mock.patch.object(
                chunk_gate, 'DENSE_STAGE_RECEIPT_SHA256', hashlib.sha256(payload).hexdigest()):
            with self.assertRaisesRegex(ValueError, message):
                chunk_gate.validate_dense_stage_receipt()

    def test_missing_stage_rejected(self):
        self.reject_mutation(lambda r: r['comparisons'].pop('chunk1-matmul5'), 'inventory')

    def test_extra_stage_rejected(self):
        self.reject_mutation(lambda r: r['comparisons'].update(extra={}), 'inventory')

    def test_wrong_element_count_rejected(self):
        self.reject_mutation(
            lambda r: r['comparisons']['chunk1-matmul5'].update(elements=1151), 'element count')

    def test_nonzero_mismatch_rejected(self):
        self.reject_mutation(
            lambda r: r['comparisons']['chunk1-matmul5'].update(unequal=1), 'not exact')

    def test_inconsistent_error_metrics_rejected(self):
        for fields in [{'max_abs': 1e-9}, {'max_abs': float('nan')}, {'first_indices': [0]}]:
            with self.subTest(fields=fields):
                self.reject_mutation(
                    lambda r: r['comparisons']['chunk1-matmul5'].update(fields), 'not exact')

    def test_changed_reference_rejected(self):
        self.reject_mutation(
            lambda r: r.update(original_synthetic_hf_output_state_unchanged=False), 'original HF')

    def test_compare_requires_receipt_before_writing_success(self):
        with mock.patch.object(Path, 'read_bytes', side_effect=FileNotFoundError), mock.patch.object(
                chunk_gate, 'write_json') as write:
            with self.assertRaisesRegex(ValueError, 'receipt unavailable'):
                chunk_gate.compare(OUT)
            write.assert_not_called()


class ExceptionalArithmeticPolicyTests(unittest.TestCase):
    """CPU IEEE examples, not a CUDA oracle or a test of native kernel results."""

    def test_finite_gate_parameter_can_overflow_f32(self):
        a_log = 100.0  # Exactly representable in BF16; exp fits F64, not F32.
        self.assertTrue(math.isfinite(a_log))
        exponential = ctypes.c_float(math.exp(a_log)).value
        self.assertEqual(exponential, math.inf)
        g = -exponential * math.log1p(math.exp(0.0))
        self.assertEqual(g, -math.inf)
        difference = g - g
        self.assertTrue(math.isnan(difference))
        self.assertTrue(math.isnan(0.0 * difference))

    def test_finite_gates_can_overflow_f32_prefix(self):
        gate = ctypes.c_float(-1e38).value
        self.assertTrue(math.isfinite(gate))
        pair = ctypes.c_float(gate + gate).value
        self.assertTrue(math.isfinite(pair))
        prefix = ctypes.c_float(pair + pair).value
        self.assertEqual(prefix, -math.inf)
        self.assertTrue(math.isnan(prefix - prefix))


if __name__=='__main__':unittest.main(argv=[__file__,*rest])
