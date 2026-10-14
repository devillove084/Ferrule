#!/opt/conda310/bin/python
"""CPU-only strict full-inventory comparison against an authenticated TP2 oracle."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.update(CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='none',ATEN_CPU_CAPABILITY='avx512')
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import tp2_oracle as o

CANDIDATE_SCHEMA='ferrule.paper.tp2-mixed-candidate.v1'


def checked_file(directory, record, cap=2048*1024**2):
    path=(directory/record['file']).resolve()
    o.require(path.parent==directory.resolve() and path.is_file() and path.stat().st_size<=cap,'Artifact path/size invalid')
    o.require(o.sha256(path)==record['sha256'],'Artifact hash mismatch: '+str(path))
    return path


def load_manifest(directory,pin,reference,lifecycle=False):
    o.require(o.sha256(directory/'manifest.json')==pin,'Manifest pin mismatch')
    m=json.loads((directory/'manifest.json').read_text())
    expected_schema=o.SCHEMA if reference else CANDIDATE_SCHEMA
    if lifecycle:
        import multistep
        expected_schema=multistep.SCHEMA if reference else 'ferrule.paper.tp2-mixed-lifecycle-candidate.v1'
    o.require(m['schema']==expected_schema and m['profile']==o.PROFILE,'Wrong reference/candidate schema')
    o.require(m['status']=='complete','Incomplete run')
    o.require(m['protocol']==json.loads((o.HERE/'protocol.json').read_text()),'Unsupported/stale numerical protocol')
    if lifecycle:
        multistep.validate_plan(m['suite']['plan'],full_model=True)
        case_ids=multistep.call_ids(m['suite']['plan'])
        o.require(m['suite']['cases']==m['cases'],'Lifecycle case index differs')
    else:case_ids=list(o.PROMPTS)
    o.require([c['id'] for c in m['cases']]==case_ids,'Missing/duplicate/reordered cases')
    if reference:
        o.require(m['full_reference_complete'] is True and m['execution']['logical_tp']==2 and m['execution']['ep']==1 and m['execution']['pp']==1 and m['boundary_inventory_complete'] is True,
                  'Not a completed TP2 reference')
        o.require(m['versions']==dict(o.native.VERSIONS),'Unpinned reference versions')
        for key,source in m['native_source'].items():
            o.require(source['sha256']==o.native.SOURCE_PINS[key],'HF source pin differs')
        o.require(set(m['native_source'])==set(o.native.SOURCE_PINS),'Missing HF source pins')
        sys.path.insert(0,str(o.ROOT/'papers/arxiv/experiments'))
        import contract as ct
        receipt=json.loads(checked_file(directory,m['custody'],1024*1024).read_text())
        o.require(ct.successful_process(receipt),'Reference custody unproven')
        for field in ('source_payloads','operand_layout','authentication','numerical_authentication','launch','idle_before','idle_after','cuda_small_gate','checkpoint_metadata','interception_after'):
            checked_file(directory,m[field])
        gate=json.loads(checked_file(directory,m['cuda_small_gate']).read_text())
        o.require(gate['status']=='pass' and gate['metrics'] and all(v['failures']==0 for v in gate['metrics'].values()),'CUDA small gate incomplete')
        o.require(json.loads(checked_file(directory,m['interception_after']).read_text())==m['interception_audit'],'Interception audit changed')
        for case in m['cases']:checked_file(directory,case['statistics'])
        sealed=json.loads(checked_file(directory,m['launch']).read_text())
        o.require(sealed['protocol']==m['protocol'] and sealed['pins']==m['workflow_pins'], 'Sealed source/protocol differs')
        if lifecycle:o.require(sealed['scenario_plan']==m['suite']['plan'],'Sealed lifecycle plan differs')
        numerical=json.loads(checked_file(directory,m['numerical_authentication']).read_text())
        authenticated=json.loads(checked_file(directory,m['authentication']).read_text())
        o.require(numerical['launch_sha256']==m['launch']['sha256'] and all(
            numerical[k]==authenticated[k] for k in ('worker','helper','coordinator','launch_id')), 'Child custody binding differs')
        payloads=json.loads(checked_file(directory,m['source_payloads']).read_text())
        o.require(payloads['index_sha256']==m['checkpoint']['index_sha256'], 'Payload/index identity differs')
        shards=m['checkpoint']['full_shards']
        import hashlib
        o.require(hashlib.sha256(json.dumps(shards,sort_keys=True).encode()).hexdigest()==m['checkpoint']['full_shard_set_sha256'],
                  'Full checkpoint shard-set digest differs')
        o.require(payloads['used_payloads'] and all(p['shard'] in shards for p in payloads['used_payloads'].values()),
                  'Used payload has no whole-shard provenance')
    return m


def candidate_inventory(config, ids):
    # Torch's pre-canonicalization unstable top-k IDs have no production counterpart.
    # Preserve them in the oracle, but gate only canonical selected IDs exactly.
    return {k:v for k,v in o.expected_tensors(config,ids).items() if not k.endswith('.native_expert_ids')}


def validate_case(archive, report, config, ids, reference):
    inventory=o.expected_tensors(config,ids) if reference else candidate_inventory(config,ids)
    o.require(set(archive.keys())==set(inventory) and report['tensors']==inventory,
              'Missing/extra required tensors; partial comparison forbidden')
    o.require(len(report['calls'])==2, 'Two calls required')
    for key,spec in inventory.items():
        value=archive.get_tensor(key)
        o.require(list(value.shape)==spec['shape'] and str(value.dtype)==spec['dtype'], 'Boundary geometry differs: '+key)
        o.require(bool(o.torch.isfinite(value).all()), 'Nonfinite boundary: '+key)
    for call,stage,n,start in zip(report['calls'],('prefill','decode.0'),(len(ids),1),(0,len(ids))):
        o.require(call['stage']==stage and call['position_start']==start and call['sequence_length_after']==start+n,
                  'Wrong stage/cache frontier')
        fed=archive.get_tensor(stage+'.input_ids')
        o.require(fed.tolist()==[call['input_token_ids']], 'Call/tensor input differs')
        if stage=='prefill':o.require(call['input_token_ids']==ids, 'Wrong original prompt')
        elif reference:o.require(call['input_token_ids']==[report['calls'][0]['next_token_id']], 'Reference continuation differs')
        positions=o.torch.arange(start,start+n,dtype=o.torch.int64)
        for key,expected in [('position_ids',positions[None]),('cache_position',positions),
                             ('attention_mask',o.torch.ones((1,start+n),dtype=o.torch.int64))]:
            o.require(o.torch.equal(archive.get_tensor(stage+'.'+key),expected), 'Position/mask contract differs')
        logits=archive.get_tensor(stage+'.logits')
        gathered=o.torch.cat([archive.get_tensor(f'{stage}.rank.{r}.logits') for r in range(2)],-1)
        o.require(o.torch.equal(logits,gathered), 'Vocabulary gather differs')
        o.require(call['next_token_id']==int(logits[0,-1].argmax()), 'Reported greedy ID differs from full vocabulary')
    return inventory


def compare_case(reference, expected_case, actual, actual_case, config, ids):
    from safetensors import safe_open
    name=expected_case['id']
    o.require(actual_case['id']==name, 'Case identity differs')
    rp=checked_file(reference,expected_case)
    ap=checked_file(actual,actual_case)
    rr=json.loads(checked_file(reference,expected_case['report']).read_text())
    ar=json.loads(checked_file(actual,actual_case['report']).read_text())
    o.require(rr['id']==ar['id']==name, 'Report identity differs')
    metrics={}
    with safe_open(rp,framework='pt',device='cpu') as r, safe_open(ap,framework='pt',device='cpu') as a:
        validate_case(r,rr,config,ids,True)
        inventory=validate_case(a,ar,config,ids,False)
        for rc,ac in zip(rr['calls'],ar['calls']):
            for field in ('stage','input_token_ids','position_start','sequence_length_after'):
                o.require(rc[field]==ac[field], 'Candidate must replay oracle inputs/frontier, including cached token')
        for key in inventory:
            rv,av=r.get_tensor(key),a.get_tensor(key)
            metrics[key]=o.tensor_metric(av,rv,exact=rv.dtype==o.torch.int64)
    return dict(metrics=metrics,reference_calls=rr['calls'],actual_calls=ar['calls'],
                oracle_only_diagnostics=['native_expert_ids'],
                greedy_ids_match=[a['next_token_id']==r['next_token_id'] for r,a in zip(rr['calls'],ar['calls'])])


def compare(reference,pin,actual,actual_pin):
    o.initialize()
    ref=load_manifest(reference,pin,True)
    got=load_manifest(actual,actual_pin,False)
    for key in ('config_sha256','full_shard_set_sha256','index_sha256','effective_text_config'):
        o.require(ref['checkpoint'][key]==got['checkpoint'][key], 'Candidate checkpoint identity differs: '+key)
    config=o.native.hf.Qwen3_5MoeTextConfig(**ref['checkpoint']['effective_text_config'])
    o.require(config.num_hidden_layers==40 and config.num_experts==256 and config.vocab_size==248320 and
              config.layer_types==(['linear_attention']*3+['full_attention'])*10, 'Not full40')
    o.local_config(config,2)
    rows={r['id']:compare_case(reference,r,actual,a,config,o.PROMPTS[r['id']]) for r,a in zip(ref['cases'],got['cases'])}
    passed=all(m['failures']==0 for c in rows.values() for m in c['metrics'].values()) and all(
               all(c['greedy_ids_match']) for c in rows.values())
    return dict(schema='ferrule.paper.tp2-mixed-comparison.v1',status='pass' if passed else 'fail',passed=passed,
                reference_manifest_sha256=pin,actual_manifest_sha256=actual_pin,cases=rows,
                tolerance=dict(atol=o.ATOL,rtol=o.RTOL,ids='exact',greedy_ids='exact',
                    states='unchanged tolerance plus exactness reported separately; no silent loosening'),
                claim='Numerical artifact comparison only; not production execution/custody/factory admission by itself')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('reference','actual','output'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('reference-sha256','actual-sha256'):p.add_argument('--'+key,required=True)
    args=p.parse_args()
    o.require(args.output.resolve().is_relative_to(o.HERE),'Comparison output must stay paper-local')
    result=compare(args.reference,args.reference_sha256,args.actual,args.actual_sha256)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
