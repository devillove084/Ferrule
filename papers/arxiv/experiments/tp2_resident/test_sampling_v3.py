"""CPU-only tests for complete UUID baselines and delta admission."""
from pathlib import Path
import sys
import unittest
from unittest import mock
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment_v3 as experiment
import sampling_v3 as sampling

UUIDS=tuple('GPU-'+str(i).zfill(36) for i in range(8))

def baseline():
    return {u:dict(index=i,uuid=u,total_bytes=24*(1<<30),gpu_bytes=0,utilization=0) for i,u in enumerate(UUIDS)}

def sample(values=None, apps=None):
    values=values or {}
    return dict(devices=[dict(baseline()[u],gpu_bytes=values.get(u,0)) for u in UUIDS],apps=apps or [])

class BaselineDelta(unittest.TestCase):
    def test_stable_one_mib_residue_is_recorded_and_allowed(self):
        b=baseline(); s=sample({UUIDS[2]:1<<20,UUIDS[3]:1<<20,UUIDS[4]:1<<20,UUIDS[5]:1<<20})
        result=sampling.device_deltas(s,b,UUIDS[:2])
        self.assertEqual(result['violations'],[])
        self.assertEqual(result['per_uuid'][UUIDS[2]]['delta_bytes'],1<<20)
        self.assertEqual(result['per_uuid'][UUIDS[2]]['tolerance_bytes'],1<<20)

    def test_positive_delta_above_tolerance_blocks_without_signal(self):
        b=baseline(); s=sample({UUIDS[2]:(1<<20)+1})
        result=sampling.device_deltas(s,b,UUIDS[:2])
        self.assertEqual(len(result['violations']),1)
        self.assertIn(UUIDS[2],result['violations'][0])

    def test_negative_delta_and_selected_peak_are_retained(self):
        b=baseline(); b[UUIDS[2]]['gpu_bytes']=2<<20
        s=sample({UUIDS[0]:20<<30,UUIDS[2]:1<<20})
        result=sampling.device_deltas(s,b,UUIDS[:2])
        self.assertEqual(result['violations'],[])
        self.assertEqual(result['per_uuid'][UUIDS[2]]['delta_bytes'],-1<<20)
        self.assertEqual(result['per_uuid'][UUIDS[0]]['observed_bytes'],20<<30)

    def test_nonselected_pid_is_evidence_not_a_foreign_signal(self):
        b=baseline(); s=sample(apps=[dict(pid=123,uuid=UUIDS[2],used_bytes=1<<20)])
        result=sampling.device_deltas(s,b,UUIDS[:2])
        self.assertEqual(result['violations'],[])
        self.assertEqual(s['apps'][0]['pid'],123)

    def test_incomplete_or_remapped_baseline_rejected(self):
        b=baseline(); del b[UUIDS[-1]]
        with self.assertRaises(ValueError): sampling.device_deltas(sample(),b,UUIDS[:2])
        b=baseline(); s=sample(); s['devices'][2]['index']=7
        with self.assertRaises(ValueError): sampling.device_deltas(s,b,UUIDS[:2])

    def test_idle_records_baseline_and_after_relative_to_anchor(self):
        b=baseline(); calls=[sample({UUIDS[2]:1<<20}),sample({UUIDS[2]:1<<20})]
        result=sampling.idle({'limits':{'idle_seconds':7},'resources':{'gpu_expected_total_bytes':24*(1<<30)}},UUIDS[:2],'test',anchor=b,query=mock.Mock(side_effect=calls))
        self.assertEqual(result['status'],'verified')
        self.assertEqual(result['baseline'][UUIDS[2]]['gpu_bytes'],1<<20)
        self.assertEqual(result['observations'][-1]['baseline_delta_check']['per_uuid'][UUIDS[2]]['delta_bytes'],1<<20)

    def test_inference_transport_and_canary_references_unchanged(self):
        import experiment_v2
        c=experiment.config(); old=experiment_v2.config()
        self.assertEqual(c['vllm_flags'],old['vllm_flags'])
        self.assertEqual(c['transport'],old['transport'])
        self.assertEqual(c['workloads'],old['workloads'])
        self.assertEqual(c['vllm_flags']['--cpu-offload-gb'],'0')
        self.assertNotIn('--enable-expert-parallel',c['vllm_flags'])
        rows,pins=experiment.references()
        self.assertEqual([r['reference_text'] for r in rows],
            [' Tokyo.\nThe capital of Japan is',' ?\n\n7 + 8 = ','北京，北京是中国的首都。那么'])
        self.assertIn('102c798baeb5200bc8ed8386eabed6146870b3b6da2a90fc6010677041352c7e',pins.values())

    def test_stable_nonzero_baseline_is_not_zeroed(self):
        b=baseline(); b[UUIDS[2]]['gpu_bytes']=9<<20
        result=sampling.device_deltas(sample({UUIDS[2]:10<<20}),b,UUIDS[:2])
        self.assertEqual(result['violations'],[])
        self.assertEqual(result['per_uuid'][UUIDS[2]]['baseline_bytes'],9<<20)
        self.assertEqual(result['per_uuid'][UUIDS[2]]['delta_bytes'],1<<20)

    def test_tolerance_cannot_silently_increase(self):
        with self.assertRaises(ValueError):
            sampling.device_deltas(sample(),baseline(),UUIDS[:2],tolerance=2<<20)

if __name__=='__main__': unittest.main()
