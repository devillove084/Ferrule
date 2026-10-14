"""CPU-only readiness tests; no timing result or CUDA execution claim."""
import argparse
import ast
import json
import os
from pathlib import Path
import unittest
from unittest import mock
os.environ.update(CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='void')
import chunk_fusion_performance as gate

p=argparse.ArgumentParser();p.add_argument('--prepared',required=True,type=Path)
a,rest=p.parse_known_args();OUT=a.prepared.resolve()


class FusionPerformanceCPU(unittest.TestCase):
    def test_verified_current_and_correct_archived_math(self):
        m=gate.verify(OUT)
        self.assertEqual(m['status'],'CPU_READY_GPU_NOT_RUN')
        self.assertEqual(m['archive_sha256'],gate.ARCHIVE_SHA)
        self.assertIn('inline uint32_t prefix_window(',m['unchanged_math'])
        self.assertIn('inline float query_scale(',m['unchanged_math'])
        self.assertIn('__global__ void matmul(',m['unchanged_math'])
        self.assertEqual(json.loads((OUT/'build.json').read_text())['returncode'],0)
        self.assertNotIn('warning:',json.loads((OUT/'build.json').read_text())['stderr'])
        self.assertIn('matched 15/19 benchmark contracts; no CUDA calls',
                      json.loads((OUT/'contracts.json').read_text())['stdout'])

    def test_exactly_three_bounded_jobs_no_scan_or_model(self):
        jobs=json.loads((OUT/'jobs.json').read_text())
        self.assertEqual(len(jobs),3)
        for i,(name,rows) in enumerate(gate.CASES):
            self.assertEqual(jobs[i]['name'],name)
            self.assertEqual(jobs[i]['argv'],[str(OUT/'benchmark'),str(OUT/name),str(rows),str(i%2)])
            self.assertEqual(jobs[i]['seconds'],60)
            self.assertEqual(jobs[i]['pins'][str(gate.WRAPPER)],gate.sha(gate.WRAPPER))
        source=(gate.HERE/'chunk_fusion_performance.cu').read_text()
        self.assertIn('ferrule_core_prior_bf16_chunk_launch(&a)',source)
        self.assertIn('ferrule_core_bf16_chunk_prefill_launch(&a)',source)
        self.assertNotIn('ferrule_core_bf16_delta_launch',source)
        self.assertNotIn('launch(call,',source)

    def test_fixed_pinned_original_inputs_and_independent_references(self):
        original=json.loads((gate.PRIOR/'prepared.json').read_text())
        expected=json.loads((gate.EXPECTED/'manifest.json').read_text())
        self.assertEqual(gate.sha(gate.EXPECTED/'manifest.json'),gate.EXPECTED_SHA)
        self.assertFalse(expected['native_results_used'])
        for name,rows in gate.CASES:
            for file in ['state.f32','a_log.bf16','dt_bias.bf16',
                         'step0/qkv.bf16','step0/a.bf16','step0/b.bf16']:
                self.assertEqual(gate.sha(OUT/name/file),original['artifact_pins'][name+'/'+file])
            self.assertEqual((OUT/name/'expected-output.bf16').read_bytes(),
                             gate.bf16((gate.EXPECTED/name/'carry-0-output.f32').read_bytes()))
            self.assertEqual(gate.sha(OUT/name/'expected-state.f32'),
                             expected['artifacts'][name+'/carry-0-state.f32'])
            self.assertEqual((OUT/name/'expected-output.bf16').stat().st_size,rows*4096*2)

    def test_thirty_balanced_pairs_and_checks_outside_events(self):
        source=(gate.HERE/'chunk_fusion_performance.cu').read_text()
        self.assertIn('pair < 30',source)
        self.assertIn('warmup < 10',source)
        loop=source[source.index('for (int pair = 0;'):]
        self.assertLess(loop.index('copy(state, seed, stream)'),loop.index('cudaEventRecord(start.value'))
        self.assertLess(loop.index('cudaEventSynchronize(stop.value)'),loop.index('check();'))
        for text in ['exact(call.output.host(), expected_y', 'exact(state.host(), expected_s',
                     'call.immutable(); seed.immutable(); al.immutable(); dt.immutable(); scratch.host();']:
            self.assertIn(text,source)
        for offset in [0,1]:
            orders=[(pair+offset)%2 for pair in range(30)]
            self.assertEqual(orders.count(0),15)
            self.assertEqual(orders.count(1),15)

    def test_resource_and_custody_sources_pinned(self):
        m=gate.verify(OUT)
        for path in [gate.WRAPPER,gate.ROOT/'scripts/test_vllm_qwen35.py',
                     gate.ROOT/'scripts/_owned_subreaper.py',gate.NVCC]:
            self.assertEqual(m['source_pins'][str(path)],gate.sha(path))
        for lib in m['library_pins']:
            self.assertEqual(m['source_pins'][lib['realpath']],lib['sha256'])
        wrapper=gate.WRAPPER.read_text()
        self.assertIn('assert idle(before)',wrapper)
        self.assertIn("c.get('echild')",wrapper)
        self.assertIn('for _ in range(3)',wrapper)
        self.assertIn("h.get('launch_id')==c.get('launch_id')",wrapper)
        self.assertIn('Function properties',json.loads((OUT/'build.json').read_text())['stderr'])

    def test_run_requires_explicit_ownership_before_any_subprocess(self):
        with mock.patch.object(gate,'execute') as run:
            with self.assertRaisesRegex(RuntimeError,'sole GPU ownership'):
                gate.run(OUT,OUT/'never-run',False)
            with self.assertRaisesRegex(RuntimeError,'sole GPU ownership'):
                gate.run(OUT,OUT/'never-run',True)
            run.assert_not_called()
        self.assertFalse((OUT/'never-run').exists())
        ast.parse(Path(gate.__file__).read_text())


if __name__=='__main__':unittest.main(argv=[__file__,*rest])
