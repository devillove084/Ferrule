"""Capture-parser guardrails; CPU-only, no Torch import or GPU queries."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

import hybrid_mixed_gdn_full40_replay as replay


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.events = []
        for key, count in (replay.BF16_FIELDS | replay.F32_FIELDS | {'normalized_input': 2048}).items():
            file = f'{len(self.events):04}.f32'
            (self.directory / file).write_bytes(bytes(4 * count))
            self.events.append(dict(event='norm.output' if key == 'normalized_input' else 'gdn.'+key,
                                    role='AttentionNorm', layer=0, file=file, shape=[1,count]))
        self.events.append(dict(self.events[-1], event='mixed.layer.output'))
        self.save()

    def save(self):
        (self.directory / 'events.jsonl').write_text('\n'.join(json.dumps(e) for e in self.events))

    def test_valid_capture(self):
        values, pins = replay.read_events(self.directory)
        self.assertEqual(len(values), len(replay.BF16_FIELDS)+len(replay.F32_FIELDS)+1)
        self.assertIn('events.jsonl', pins)

    def test_non_bf16_expansion_rejected(self):
        path = self.directory / self.events[0]['file']
        path.write_bytes(struct.pack('<I', 0x3f800001)+path.read_bytes()[4:])
        with self.assertRaises(AssertionError):
            replay.read_events(self.directory)

    def test_duplicate_operand_rejected(self):
        self.events.insert(0, self.events[0])
        self.save()
        with self.assertRaises(AssertionError):
            replay.read_events(self.directory)

    def test_nonzero_frontier_rejected(self):
        event = next(e for e in self.events if e['event'] == 'gdn.before.ssm')
        path = self.directory / event['file']
        path.write_bytes(struct.pack('<f', 1.)+path.read_bytes()[4:])
        with self.assertRaises(AssertionError):
            replay.read_events(self.directory)

    def test_path_escape_rejected(self):
        self.events[0]['file'] = '../escape.f32'
        self.save()
        with self.assertRaises(AssertionError):
            replay.read_events(self.directory)

    def test_later_layer_rejected(self):
        self.events[0]['layer'] = 1
        self.save()
        with self.assertRaises(AssertionError):
            replay.read_events(self.directory)


if __name__ == '__main__':
    unittest.main()
