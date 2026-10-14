"""Fresh-interpreter import regression; no CUDA import, query, lock or server."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parent))
import paired as p
import instrumentation_capture as target


class ImportIsolation(unittest.TestCase):
    def fresh(self, code, visible=None):
        selected=','.join(p.shared.UUIDS) if visible is None else visible
        env=dict(p.base.cpu_environment(),CUDA_VISIBLE_DEVICES=selected,NVIDIA_VISIBLE_DEVICES=selected,
                 FERRULE_IMPORT_SENTINEL='retained')
        r=subprocess.run([sys.executable,'-S','-B','-c',code],cwd=p.HERE,env=env,
                         capture_output=True,text=True,timeout=15)
        self.assertEqual(r.returncode,0,r.stderr)
        return json.loads(r.stdout)

    def test_real_transitive_import_does_not_mutate_any_environment_key(self):
        r=self.fresh("import os,json; before=dict(os.environ); import instrumentation_capture; print(json.dumps(dict(before=before,after=dict(os.environ))))")
        self.assertEqual(r['before'],r['after'])
        self.assertEqual(r['after']['CUDA_VISIBLE_DEVICES'],','.join(p.shared.UUIDS))

    def test_explicit_cpu_entry_hides_visibility_but_not_arbitrary_environment(self):
        r=self.fresh("import os,json; import instrumentation_capture as t; t.ct.require_cpu_python(); print(json.dumps(dict(cuda=os.environ['CUDA_VISIBLE_DEVICES'],nvidia=os.environ['NVIDIA_VISIBLE_DEVICES'],sentinel=os.environ['FERRULE_IMPORT_SENTINEL'])))")
        self.assertEqual(r,dict(cuda='',nvidia='none',sentinel='retained'))

    def test_cpu_subprocess_allowlist_does_not_leak_parent_visibility(self):
        r=self.fresh("import os,json; import instrumentation_capture as t; before=dict(os.environ); env=t.live.baseline.base.cpu_environment(); print(json.dumps(dict(env=env,unchanged=before==dict(os.environ))))")
        self.assertEqual(r['env']['CUDA_VISIBLE_DEVICES'],'')
        self.assertEqual(r['env']['NVIDIA_VISIBLE_DEVICES'],'none')
        self.assertNotIn('FERRULE_IMPORT_SENTINEL',r['env']);self.assertTrue(r['unchanged'])

    def test_collector_exact_selected_survives_real_import(self):
        r=self.fresh("import os,json; import instrumentation_capture as t; t.require_selected_visibility(t.live.shared.UUIDS); print(json.dumps(dict(cuda=os.environ['CUDA_VISIBLE_DEVICES'],nvidia=os.environ['NVIDIA_VISIBLE_DEVICES'])))")
        self.assertEqual(set(r.values()),{','.join(p.shared.UUIDS)})

    def test_foreign_reordered_hidden_and_all_visibility_rejected_without_mutation(self):
        expected=','.join(p.shared.UUIDS)
        for value in ('','none','all','0,1',p.shared.UUIDS[0],','.join(reversed(p.shared.UUIDS)),expected+',GPU-foreign'):
            with mock.patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':value,'NVIDIA_VISIBLE_DEVICES':value}):
                before=dict(os.environ)
                with self.assertRaisesRegex(ValueError,'exactly selected'):target.require_selected_visibility(p.shared.UUIDS)
                self.assertEqual(before,dict(os.environ))
        with mock.patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':expected,'NVIDIA_VISIBLE_DEVICES':'all'}):
            with self.assertRaises(ValueError):target.require_selected_visibility(p.shared.UUIDS)

    def test_wrong_interpreter_cannot_enter_cpu_mode(self):
        with mock.patch.object(p.ct.sys,'executable','/unapproved/python'):
            with self.assertRaisesRegex(ValueError,'exact base Python'):p.ct.require_cpu_python()


if __name__=='__main__':unittest.main()
