"""Run one fresh, pinned backend gate under existing authenticated custody."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import contract_run

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('label',choices=['blocked','matrix','lifecycle','unknown','runtime129','existing-graph'])
    p.add_argument('--execute-cuda',action='store_true')
    p.add_argument('--sole-owner-permission',action='store_true')
    a=p.parse_args()
    if not a.execute_cuda or not a.sole_owner_permission:
        raise ValueError('Explicit sole-GPU permission required')
    directory=a.directory.resolve()
    prepared=json.loads((directory/'PREPARED.json').read_text())
    contract_run.verify(prepared)
    plan=prepared['plans'][a.label]
    env=os.environ.copy();env.update(plan['environment'])
    link=subprocess.check_output(['ldd',plan['command'][0]],env=env,text=True,timeout=15)
    if 'not found' in link:raise ValueError('Unresolved test libraries')
    base=plan.get('library_directory','/home/admin/wxw403883/anaconda3/envs/tao/lib')+'/'
    for lib in ['libcublas.so.12','libcublasLt.so.12','libcudart.so.12']:
        expected=(plan.get('runtime_directory',base.rstrip('/'))+'/' if lib=='libcudart.so.12' else base)+lib
        if expected not in link:raise ValueError('Wrong pinned library: '+lib)
    with (directory/(a.label+'-ldd.txt')).open('x') as f:f.write(link)
    contract_run.run(directory,a.label,True,True)
