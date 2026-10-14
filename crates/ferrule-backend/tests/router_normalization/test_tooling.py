"""CPU-only harness tests; these are not CUDA arithmetic evidence."""
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Tooling(unittest.TestCase):
    def test_comparator_counts_bits_not_tolerance(self):
        compare = load('compare').compare
        data = struct.pack('<3f', 0., 1., 2.)
        changed = struct.pack('<3I', 0x80000000, 0x3f800001, 0x40000000)
        result = compare(data, changed)
        self.assertEqual((result['elements'], result['unequal'], result['first']), (3, 2, 0))
        self.assertEqual(result['max_abs'], 2**-23)

    def test_preparation_rejects_non_sameinput_without_torch(self):
        prepare = load('prepare').prepare
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            source.mkdir()
            (source / 'manifest.json').write_text(json.dumps(dict(cases=[dict(id='case', m=1, n=1, module='mlp.gate')])))
            (source / 'case.hf.bf16').write_bytes(b'\x80\x3f')
            (source / 'case.direct.bf16').write_bytes(b'\x00\x40')
            with self.assertRaisesRegex(AssertionError, 'not SAMEINPUT'):
                prepare(source, root / 'out')

    def test_preparation_refuses_overwrite(self):
        prepare = load('prepare').prepare
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                prepare(Path(tmp), Path(tmp))


if __name__ == '__main__':
    unittest.main()
