"""CPU-only immutable-field mapping tests."""
import unittest
import hybrid_mixed_capital_full_compare as compare

class MappingTests(unittest.TestCase):
    def event(self,name,layer,role=None):return dict(event=name,layer=layer,role=role)
    def test_every_hidden_frontier(self):
        for layer in range(40):
            expected='prefill.embedding' if layer==0 else f'prefill.layers.{layer-1:02d}.hidden'
            self.assertEqual(compare.mapping(self.event('mixed.layer.input',layer),1),expected)
            self.assertEqual(compare.mapping(self.event('mixed.layer.output',layer),1),f'prefill.layers.{layer:02d}.hidden')
    def test_rank_state_and_route(self):
        for rank in range(2):
            self.assertEqual(compare.mapping(self.event('gdn.recurrent_state',2),rank),f'prefill.rank.{rank}.cache.recurrent_states.02')
            self.assertEqual(compare.mapping(self.event('router',3),rank),f'prefill.rank.{rank}.layers.03.raw_router_logits')
    def test_unknown_is_not_invented(self):
        self.assertIsNone(compare.mapping(self.event('mixed.pair.gate',4,'RoutedExpertGate'),0))
        self.assertIsNone(compare.mapping(self.event('norm.output',4,'FeedForwardNorm'),0))
    def test_static_output_fields(self):
        self.assertEqual(compare.mapping(self.event('norm.output',None,'OutputNorm'),0),'prefill.final_norm')
        self.assertEqual(compare.mapping(self.event('mixed.linear.output',None,'OutputHead'),1),'prefill.rank.1.logits')
    def test_gqa_append_fields(self):
        self.assertEqual(compare.mapping(self.event('gqa.key',3),0),'prefill.rank.0.cache.key_cache.03')
        self.assertEqual(compare.mapping(self.event('gqa.value',3),1),'prefill.rank.1.cache.value_cache.03')

if __name__=='__main__':unittest.main()
