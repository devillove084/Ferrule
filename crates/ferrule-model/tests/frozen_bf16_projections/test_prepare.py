import json
import struct
import tempfile
import unittest
from pathlib import Path
import prepare as p


class FrozenExtraction(unittest.TestCase):
    def event(self):
        return dict(layer=0, parameter='layers.0.linear_attention.output.weight', shape=[1, 4096],
                    weight=dict(shape=[2048, 4096], dtype='F8E4M3'),
                    scale=dict(shape=[16, 32], dtype='Bf16'))

    def test_out_projection_identity(self):
        self.assertEqual(p.identify(self.event()), (0, 'AttentionOutput', 'linear_attn.out_proj', 2048, 4096))

    def test_unknown_role_rejected(self):
        e = self.event()
        e['parameter'] = 'layers.0.unknown.weight'
        with self.assertRaisesRegex(ValueError, 'unknown numeric role'):
            p.identify(e)

    def test_shapes_and_layer_rejected(self):
        for key, value in [('shape', [2, 4096]), ('layer', 1)]:
            e = self.event()
            e[key] = value
            with self.assertRaises(ValueError):
                p.identify(e)
        e = self.event()
        e['scale']['shape'] = [32, 16]
        with self.assertRaisesRegex(ValueError, 'scale shape'):
            p.identify(e)

    def test_experts_not_promoted_to_oracle(self):
        e = self.event()
        e.update(parameter='layers.0.feed_forward.experts.56.down.weight', shape=[1, 512])
        e['weight']['shape'] = [2048, 512]
        e['scale']['shape'] = [16, 4]
        self.assertEqual(p.identify(e), (0, 'RoutedExpertDown', None, 2048, 512))

    def test_missing_hf_expected_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'hf.safetensors'
            h = json.dumps({'0.linear_attn.out_proj.input': dict(dtype='BF16', shape=[1, 1, 4096], data_offsets=[0, 8192])}).encode()
            path.write_bytes(struct.pack('<Q', len(h)) + h + bytes(8192))
            with self.assertRaises(KeyError):
                p.pair(path, 'linear_attn.out_proj', 0, 2048, 4096)

    def test_pin_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'source').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                p.check_pins(path, {'source': '0' * 64})

    def test_non_bf16_trace_never_silently_rounded(self):
        with self.assertRaises(AssertionError):
            p.util.exact_words(struct.pack('<I', 0x3f800001))


if __name__ == '__main__':
    unittest.main()
