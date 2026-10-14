"""CPU regressions: routing alignment, combine order, and discontinuous quantization."""
import unittest
import json
import tempfile
from pathlib import Path
import torch
import torch.nn.functional as F
import diagnose_w8a16_causal as d


class CausalTests(unittest.TestCase):
    def test_route_permutation_is_aligned_by_expert(self):
        logits=torch.tensor([[3.,2.,1.,0.]])
        ids=torch.tensor([[1,0]])
        weights=logits.softmax(-1).gather(1,ids)
        weights/=weights.sum(-1,keepdim=True)
        result=d.route_metrics(logits,ids,weights)
        self.assertTrue(result['same_set'])
        self.assertEqual(result['slot_mismatch_rows'],1)
        self.assertEqual(result['same_selected_weights']['outside'],0)

    def test_combine_order_changes_f32_not_route_membership(self):
        table=torch.tensor([[[2.**24],[1.],[-2.**24]]])
        ids=torch.tensor([[0,2,1]])
        weights=torch.ones(1,3)
        self.assertEqual(d.combine(table,ids,weights).item(),0.)
        self.assertEqual(d.combine(table,ids,weights,expert_order=True).item(),1.)

    def test_operator_bound_does_not_imply_end_to_end_tolerance(self):
        # Both first operators satisfy the *existing* triangle bound on identical
        # BF16 operands; one ULP at a BF16 midpoint changes the next linear.
        midpoint=torch.tensor(1.00390625)
        above=torch.nextafter(midpoint,torch.tensor(float('inf')))
        x=torch.tensor([[1.,1.]])
        w=torch.tensor([[1.,2.**-8]])
        for output in (midpoint,above):
            self.assertEqual(d.operator_bound(output.reshape(1,1),x,w)['outside'],0)
        a=d.ref.independent_bf16_rne(above.reshape(1,1))
        b=d.ref.independent_bf16_rne(midpoint.reshape(1,1))
        downstream=torch.tensor([[2.]])
        actual,expected=F.linear(a,downstream),F.linear(b,downstream)
        self.assertEqual(d.operator_bound(actual,a,downstream)['outside'],0)
        self.assertEqual(d.operator_bound(expected,b,downstream)['outside'],0)
        self.assertEqual(d.metric(above,midpoint)['outside'],0)
        self.assertEqual(d.metric(actual,expected)['outside'],1)
        self.assertEqual((actual-expected).item(),0.015625)

    def test_native_capture_ids_hashes_and_same_operand_classification(self):
        with tempfile.TemporaryDirectory() as root:
            directory=Path(root)/'operator-trace-00'; directory.mkdir()
            events=[]
            for name,value in [('input',torch.tensor([[1.001]])),('output',torch.tensor([[2.]]))]:
                path=directory/(name+'.bin'); path.write_bytes(value.numpy().tobytes())
                events.append(dict(key='prefill.trace.model.layers.0.test.'+name+'.0',file=path.name,
                                   dtype='torch.float32',shape=list(value.shape),sha256=d.sha256(path)))
            index=directory/'events.jsonl'; index.write_text(''.join(json.dumps(e)+'\n' for e in events))
            trace=d.NativeTrace(Path(root),dict(directory=directory.name,events_sha256=d.sha256(index),tensors=2))
            result=trace.compare('prefill.trace.model.layers.0.test',torch.tensor([[1.]]),torch.tensor([[2.]]),True)
            self.assertTrue(result['rounded_operands_identical'])
            result=trace.compare('prefill.trace.model.layers.0.test',torch.tensor([[1.]]),torch.tensor([[2.]]),False)
            self.assertFalse(result['rounded_operands_identical'])
            (directory/'input.bin').write_bytes(torch.zeros(1).numpy().tobytes())
            with self.assertRaises(ValueError): trace.get(events[0]['key'])

    def test_replay_authorization_and_cpu_math(self):
        with self.assertRaisesRegex(ValueError,'requires --execute-cuda'):
            d.SameInputReplay('cuda:0',False)
        replay=d.SameInputReplay()
        x=torch.tensor([[1.,2.]])
        self.assertTrue(torch.equal(replay.linear(x,torch.eye(2)),x))
        table=torch.tensor([[[2.**24],[1.],[-2.**24]]])
        ids=torch.tensor([[0,2,1]])
        self.assertEqual(replay.native_combine(table,ids,torch.ones(1,3)).item(),1.)
        self.assertFalse(torch.cuda.is_initialized())

    def test_full40_completeness_rejects_missing_decode_and_state(self):
        events=[]
        for stage in ('prefill','decode.0'):
            for layer in range(40):
                names=['router']+['residual.output']*3
                if layer%4!=3:
                    names+=['gdn.conv_state','gdn.recurrent_state','gdn.convolved','gdn.recurrent_output','gdn.normalized','gdn.gated']
                events.extend(dict(stage=stage,layer=layer,event=name,role=None) for name in names)
                roles=['AttentionNorm','FeedForwardNorm']+(['AttentionQueryNorm','AttentionKeyNorm'] if layer%4==3 else [])
                events.extend(dict(stage=stage,layer=layer,event='norm.output',role=role) for role in roles)
            events.extend([dict(stage=stage,layer=None,event='embedding',role=None),
                           dict(stage=stage,layer=None,event='norm.output',role='OutputNorm'),
                           dict(stage=stage,layer=None,event='linear.output',role='OutputHead')])
        d.require_full40_capture(events)
        with self.assertRaises(ValueError): d.require_full40_capture([e for e in events if e['stage']=='prefill'])
        with self.assertRaises(ValueError): d.require_full40_capture([e for e in events if e['event']!='gdn.recurrent_state'])

    def test_nonfinite_and_shape_rejected(self):
        with self.assertRaises(ValueError): d.metric(torch.tensor([float('nan')]),torch.ones(1))
        with self.assertRaises(ValueError): d.metric(torch.ones(2),torch.ones(3))

    def test_wrong_route_set_is_not_hidden_by_weight_alignment(self):
        result=d.route_metrics(torch.tensor([[3.,2.,1.,0.]]),torch.tensor([[0,2]]),torch.tensor([[.8,.2]]))
        self.assertFalse(result['same_set'])
        self.assertEqual(result['set_mismatch_rows'],1)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
