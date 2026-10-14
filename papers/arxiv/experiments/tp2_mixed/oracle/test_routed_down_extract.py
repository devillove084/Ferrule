"""CPU byte-layout/provenance tests; no model arithmetic or CUDA expectation."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

import routed_down_extract as e


class RoutedDownExtractionTests(unittest.TestCase):
    def test_two_rank_down_columns_and_original_scale_coordinates(self):
        n, k = 256, 512
        raw = bytes((r * 17 + c // 128) % 251 for r in range(n) for c in range(k))
        scales = struct.pack('<8H', *range(8))
        pieces = [e.down_slices(raw, scales, n, k, r) for r in (0, 1)]
        for rank, (weight, scale) in enumerate(pieces):
            self.assertEqual(len(weight), n * 256)
            self.assertEqual(struct.unpack('<4H', scale), tuple(row * 4 + rank * 2 + col for row in range(2) for col in range(2)))
            for row in range(n):
                self.assertEqual(weight[row * 256:(row + 1) * 256], raw[row * k + rank * 256:row * k + (rank + 1) * 256])
        self.assertEqual(b''.join(pieces[0][0][r*256:(r+1)*256] + pieces[1][0][r*256:(r+1)*256] for r in range(n)), raw)

    def test_rejects_incomplete_blocks_and_invalid_rank(self):
        for n, k, rank in ((127, 512, 0), (128, 384, 0), (128, 512, 2), (0, 512, 0)):
            with self.subTest(n=n, k=k, rank=rank), self.assertRaises(ValueError):
                e.down_slices(b'', b'', n, k, rank)

    def test_rejects_wrong_byte_geometry(self):
        for weight, scales in ((b'\x01' * 65535, b'\x01' * 8), (b'\x01' * 65536, b'\x01' * 7)):
            with self.assertRaisesRegex(ValueError, 'geometry'):
                e.down_slices(weight, scales, 128, 512, 0)

    def test_bf16_scale_words_are_not_reencoded(self):
        words = [0x0000, 0x8000, 0x3F81, 0x3F7F]
        _, rank1 = e.down_slices(bytes(128 * 512), struct.pack('<4H', *words), 128, 512, 1)
        self.assertEqual(rank1, struct.pack('<2H', *words[2:]))

    def test_payload_requires_old_authenticated_hash_and_header(self):
        with tempfile.TemporaryDirectory(dir=e.o.HERE) as tmp:
            directory = Path(tmp)
            raw = b'\x80\x3f\x81\x3f'
            tensor = dict(dtype='BF16', shape=[2], data_offsets=[0, len(raw)])
            header = json.dumps({'weight': tensor}).encode()
            path = directory / 'original.safetensors'
            path.write_bytes(struct.pack('<Q', len(header)) + header + raw)
            metadata = dict(tensors={'weight': dict(shard=path.name, **tensor)},
                            shards={path.name: dict(bytes=path.stat().st_size, header_bytes=len(header), header_sha256=e.digest(header))})
            checkpoint = dict(full_shards={path.name: e.digest(path.read_bytes())})
            used = {'weight': dict(sha256=e.digest(raw), shard=path.name, shape=[2])}
            reader = e.PinnedPayloads(directory, checkpoint, metadata, used)
            self.assertEqual(reader.read('weight'), raw)
            reader.finish()
            bad = {'weight': dict(used['weight'], sha256='0' * 64)}
            with self.assertRaisesRegex(ValueError, 'payload hash'):
                e.PinnedPayloads(directory, checkpoint, metadata, bad).read('weight')
            with self.assertRaises(KeyError):
                e.PinnedPayloads(directory, checkpoint, metadata, {}).read('weight')
            path.write_bytes(path.read_bytes() + b'changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                reader.finish()

    def test_cpu_module_does_not_initialize_cuda(self):
        self.assertEqual(e.os.environ['CUDA_VISIBLE_DEVICES'], '')
        self.assertFalse(e.o.torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
