"""Tiny CPU artifact round trips; these fixtures are NOT production evidence."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

os.environ.update(CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none', ATEN_CPU_CAPABILITY='avx512')
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare as c
from test_tp2_oracle import Store, tiny
from safetensors.torch import save_file

o = c.o


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        o.initialize()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=o.HERE)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ref, self.actual = self.root/'reference', self.root/'actual'
        self.ref.mkdir(); self.actual.mkdir()
        self.config, self.ids = tiny(), [1, 3, 2]
        o.torch.manual_seed(934)
        original = o.native.hf.Qwen3_5MoeForCausalLM(self.config).eval()
        operands = o.Operands(Store(original), self.config, 'cpu')
        self.capture = o.Capture()
        self.addCleanup(self.capture.close)
        model = o.build_model(self.config, operands, self.capture)
        self.calls = o.run_case(model, self.config, operands, self.capture, self.ids)
        self.r = o.save_case(self.ref, 'hello', self.config, self.ids, self.capture, self.calls)
        self.values = {k:v.clone() for k,v in self.capture.tensors.items() if not k.endswith('.native_expert_ids')}
        self.report = json.loads((self.ref/'hello.json').read_text())
        self.report['tensors'] = c.candidate_inventory(self.config, self.ids)
        self.a = self.write_actual()

    def tearDown(self):
        self.assertFalse(o.torch.cuda.is_initialized())

    def write_actual(self):
        path = self.actual/'hello.safetensors'
        save_file(self.values, str(path))
        self.report.update(o.file_record(path))
        (self.actual/'hello.json').write_text(json.dumps(self.report))
        return dict(id='hello', **o.file_record(path), report=o.file_record(self.actual/'hello.json'))

    def compare(self):
        return c.compare_case(self.ref, self.r, self.actual, self.a, self.config, self.ids)

    def test_logits_claim_receipt_is_not_a_full_candidate(self):
        from claim_receipt import inspect_receipt
        path=o.ROOT/'target/validation/tp2-mixed-claim-gates/shared-full40-01'
        if not (path/'result.json').is_file():self.skipTest('claim receipt not present')
        claim=inspect_receipt(path)
        self.assertFalse(claim['full_inventory_compatible'])
        self.assertIn('per-rank route IDs/weights',claim['missing_provenance'])
        self.assertEqual(sum(v['compared'] for v in claim['metrics'].values()),1986560)

    def test_complete_tiny_serialization_and_comparison(self):
        row = self.compare()
        self.assertEqual(row['greedy_ids_match'], [True, True])
        self.assertTrue(all(m['exact'] and m['failures'] == 0 for m in row['metrics'].values()))
        self.assertEqual(set(row['metrics']), set(c.candidate_inventory(self.config, self.ids)))
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            o.save_case(self.ref, 'hello', self.config, self.ids, self.capture, self.calls)
        del self.capture.tensors['prefill.rank.1.logits']
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            o.save_case(self.ref, 'capital', self.config, self.ids, self.capture, self.calls)

    def test_missing_rank_boundary_rejected(self):
        del self.values['decode.0.rank.1.cache.recurrent_states.00']
        self.a = self.write_actual()
        with self.assertRaisesRegex(ValueError, 'Missing/extra'):
            self.compare()

    def test_route_ids_exact_and_float_drift_not_hidden(self):
        self.values['prefill.rank.0.layers.00.selected_expert_ids'][0, 0] += 1
        self.values['decode.0.rank.0.cache.recurrent_states.00'] += 1
        self.a = self.write_actual()
        row = self.compare()
        self.assertEqual(row['metrics']['prefill.rank.0.layers.00.selected_expert_ids']['failures'], 1)
        metric = row['metrics']['decode.0.rank.0.cache.recurrent_states.00']
        self.assertGreater(metric['failures'], 0)
        self.assertEqual((metric['atol'], metric['rtol']), (.00025, .012))

    def test_candidate_must_replay_recorded_cached_token(self):
        token = (self.calls[1]['input_token_ids'][0]+1) % self.config.vocab_size
        self.values['decode.0.input_ids'][0, 0] = token
        self.report['calls'][1]['input_token_ids'] = [token]
        self.a = self.write_actual()
        with self.assertRaisesRegex(ValueError, 'replay oracle'):
            self.compare()

    def test_gather_order_and_report_argmax_rejected(self):
        key = 'prefill.rank.0.logits'
        self.values[key] += 1
        self.a = self.write_actual()
        with self.assertRaisesRegex(ValueError, 'gather'):
            self.compare()
        self.values[key] = self.capture.tensors[key]
        self.report['calls'][0]['next_token_id'] = -1
        self.a = self.write_actual()
        with self.assertRaisesRegex(ValueError, 'greedy ID'):
            self.compare()

    def test_stale_hash_escape_and_nonfinite_rejected(self):
        with (self.actual/'hello.safetensors').open('ab') as stream:
            stream.write(b'bad')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.compare()
        with self.assertRaisesRegex(ValueError, 'path/size'):
            c.checked_file(self.actual, dict(file='../reference/hello.safetensors', sha256=self.r['sha256']))
        self.values['prefill.layers.00.hidden'][0, 0, 0] = float('nan')
        self.a = self.write_actual()
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            self.compare()

    def test_dtype_contract_rejected(self):
        self.values['prefill.layers.00.hidden'] = self.values['prefill.layers.00.hidden'].float()
        self.a = self.write_actual()
        with self.assertRaisesRegex(ValueError, 'geometry'):
            self.compare()

    def test_live_source_audit_and_free_export_rejection(self):
        audit = o.source_audit()
        self.assertIn('torch_recurrent_gated_delta_rule', audit['live_functions'])
        self.assertEqual(audit, o.source_audit())
        with mock.patch.object(o.native, 'read_config') as read, mock.patch.object(o, 'initialize') as init:
            with self.assertRaisesRegex(ValueError, 'Explicit'):
                o.export({})
            read.assert_not_called(); init.assert_not_called()
            # Owner booleans and visible UUID alone still do not authenticate.
            payload = dict(sole_gpu_owner_granted=True, gpu_uuid='GPU-'+'a'*36,
                           owner_note='synthetic CPU test only', execute_cuda=True, directory=str(self.root))
            with self.assertRaisesRegex(ValueError, 'sealed launch'):
                o.export(payload)
            read.assert_not_called(); init.assert_not_called()

    def test_old_unsharded_mapping_is_descriptive_and_input_checked(self):
        old = self.root/'old'; old.mkdir()
        values = {}
        for key in o.native.expected_tensors(self.config.to_dict(), self.ids):
            mapped = key if key in self.capture.tensors else key.replace('.layers.', '.rank.0.layers.')
            values[key] = self.capture.tensors[mapped].clone()
        values['prefill.logits'] += .25
        values['decode.0.input_ids'][0, 0] = (int(values['decode.0.input_ids'][0, 0])+1) % self.config.vocab_size
        save_file(values, str(old/'hello.safetensors'))
        payloads = dict(index_sha256='index', used_payloads={'synthetic': {'sha256': 'payload'}})
        (old/'source-payloads.json').write_text(json.dumps(payloads))
        manifest = dict(schema=o.native.SCHEMA, status='complete', full_reference_complete=True,
                        checkpoint=dict(config_sha256='config'), versions=dict(o.native.VERSIONS),
                        source_payloads=o.file_record(old/'source-payloads.json'),
                        cases=[dict(id='hello', **o.file_record(old/'hello.safetensors'))])
        (old/'manifest.json').write_text(json.dumps(manifest))
        checkpoint = mock.Mock(index_hash='index', payloads=payloads['used_payloads'])
        drift = o.baseline_metrics(old, o.sha256(old/'manifest.json'), 'hello', self.capture,
                                  checkpoint, 'config', self.calls, 1024**2)
        self.assertGreater(drift['stages']['prefill']['metrics']['prefill.logits']['failures'], 0)
        self.assertEqual(drift['stages']['decode.0']['status'], 'different_autoregressive_token_not_same_input')
        self.assertIn('descriptive drift only', drift['acceptance'])


if __name__ == '__main__':
    unittest.main()
