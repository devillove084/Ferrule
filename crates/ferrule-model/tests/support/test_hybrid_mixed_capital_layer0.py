"""CPU-only M5 capture guardrails; does not import Torch or query CUDA."""
import json
from pathlib import Path
import struct
import tempfile
import unittest
import hybrid_mixed_capital_layer0 as replay

class CaptureTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.path=Path(temp.name);self.events=[]
        for key,width,rows in [('gdn.input.qkv',4096,5),('gdn.input.a',16,5),('gdn.input.b',16,5),('gdn.input.z',2048,5),('gdn.convolved',4096,5),('gdn.recurrent_output',2048,5),('gdn.gated',2048,5),('norm.output',2048,5),('gdn.before.ssm',262144,1),('gdn.before.conv',16384,1)]:
            file=f'{len(self.events):04}.f32';(self.path/file).write_bytes(bytes(rows*width*4))
            self.events.append(dict(event=key,role='AttentionNorm',file=file,layer=0,shape=[rows,width]))
        self.events.append(dict(self.events[-1],event='mixed.layer.output'));self.save()
    def save(self):
        (self.path/'events.jsonl').write_text('\n'.join(json.dumps(e) for e in self.events))
    def test_valid_m5(self):
        _,v=replay.capture(self.path);self.assertEqual(v['gdn.input.qkv'][1],[5,4096])
    def test_wrong_rows(self):
        self.events[0]['shape']=[1,20480];self.save()
        with self.assertRaises(AssertionError):replay.capture(self.path)
    def test_non_bf16(self):
        p=self.path/self.events[0]['file'];p.write_bytes(struct.pack('<I',0x3f800001)+p.read_bytes()[4:])
        with self.assertRaises(AssertionError):replay.capture(self.path)
    def test_nonzero_state(self):
        e=next(e for e in self.events if e['event']=='gdn.before.ssm');p=self.path/e['file'];p.write_bytes(struct.pack('<f',1)+p.read_bytes()[4:])
        with self.assertRaises(AssertionError):replay.capture(self.path)
    def test_no_later_layer(self):
        self.events[-1]['layer']=1;self.save()
        with self.assertRaises(AssertionError):replay.capture(self.path)

if __name__=='__main__':unittest.main()
