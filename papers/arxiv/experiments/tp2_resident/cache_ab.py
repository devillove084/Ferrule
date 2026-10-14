"""Matched-ceiling Ferrule A/B admission/report hooks, never a second launcher.

Only paired.py owns prepare/run/authentication/locks/HTTP. The frozen exception
is exactly the already qualified 4d40 binary, never the current worktree.
"""
from copy import deepcopy
from pathlib import Path

import pre_serving as pre
import profile_2025 as historical

ct=pre.ct
PROTOCOL='tp2-cacheoff-cacheon-matched4g-qa8-descriptive'
SLOTS=('OFF','ON')
POLICY='frozen_qualified_4d40_workspace4096_only_v1'
PREFLIGHT=pre.HERE/'runs/cache-ab-4g-cpu-20261010-001/workspace4096-preflight-02.json'


def frozen_metadata():
    ct.require(ct.sha(historical.ADMISSION)==historical.ADMISSION_SHA,'Historical admission changed')
    original=ct.load(historical.ADMISSION)['engines']['ferrule']
    ct.require(original['binary']['sha256']==historical.BINARY_SHA,'Frozen policy is only for qualified 4d40')
    result=deepcopy(original)
    argv=result['spec']['argv']
    ct.require(argv.count('--cuda-workspace-mb')==1,'Ambiguous historical workspace')
    i=argv.index('--cuda-workspace-mb')+1
    ct.require(argv[i]=='1024','Historical scope changed')
    argv[i]='4096'
    result['historical_serving_smoke']=result.pop('serving_smoke')
    result['frozen_binary_policy']=dict(policy=POLICY,binary_sha256=historical.BINARY_SHA,
        historical_admission_sha256=historical.ADMISSION_SHA,workspace_mib_before=1024,workspace_mib_now=4096,
        cache_enabled=False,current_tree_qualified=False,workspace4096_serving_prequalified=False,
        cache_snapshot_support=False,profiled_activity=None)
    return result


def frozen_audit():
    """No current-source substitution, widened drift list or new binary exception."""
    expected=frozen_metadata()
    old=ct.load(historical.ADMISSION)['engines']['ferrule']
    ct.check_pins({str(historical.BINARY):historical.BINARY_SHA,**old['runtime_pins']})
    ct.validate_staged_binary(old['binary']['path'],historical.BINARY_SHA)
    build=pre.checked_document(old['build_receipt'])
    quality=pre.checked_document(old['test_quality'])
    ct.require(build['artifact']['sha256']==historical.BINARY_SHA
        and build['release_state']['status']=='complete'
        and quality['status']=='pass' and (quality['calls'],quality['metrics'])==(37,40293)
        and pre.numerical_sources(old['source_pins'])==quality['numerical_source_pins'],
        'Frozen binary lacks its existing build/numerical qualification')
    # Skip only the exact historical build-source identities, not arbitrary paths
    # under crates/. Immutable receipts, test binaries and comparator data still hash.
    pins={str(historical.ADMISSION):historical.ADMISSION_SHA}
    for group in (old['build_evidence_pins'],quality['artifact_pins']):
        evidence={path:digest for path,digest in group.items() if path not in old['source_pins']}
        ct.check_pins(evidence);pins.update(evidence)
    pre.validate_smoke(old)  # historical 1024 MiB only; NOT a 4096 MiB smoke certificate
    check=ct.load(PREFLIGHT)
    ct.require(check['binary_sha256']==historical.BINARY_SHA
        and check['status']=='workspace4096_cli_and_pre_model_arguments_accepted'
        and check['returncode']!=0 and check['full_model_resource_admission_proven'] is False
        and check['environment']['CUDA_VISIBLE_DEVICES']=='' and check['environment']['NVIDIA_VISIBLE_DEVICES']=='none'
        and check['argv'][check['argv'].index('--cuda-workspace-mb')+1]=='4096'
        and check['argv'][check['argv'].index('--cuda-device-budget-mb')+1]=='22528',
        'Actual hidden old-binary CPU preflight missing/different')
    for suffix in ('stdout','stderr'):
        path=PREFLIGHT.with_suffix('.'+suffix)
        ct.require(ct.sha(path)==check[suffix+'_sha256'],'CPU preflight raw evidence changed')
        pins[str(path)]=check[suffix+'_sha256']
    ct.require('has neither config.json nor a .gguf file' in PREFLIGHT.with_suffix('.stderr').read_text(),
               'CPU preflight did not stop before model construction')
    pins[str(PREFLIGHT)]=ct.sha(PREFLIGHT)
    return dict(policy=POLICY,binary_sha256=historical.BINARY_SHA,
        historical_source_pins=old['source_pins'],artifact_pins=pins,
        current_tree_qualified=False,rebuild=False,workspace4096_serving_prequalified=False,
        cache_snapshot_support=False,profiled_activity=None,
        scope_change='Only workspace 1024 -> 4096 MiB; historical HTTP receipt remains scoped to 1024',
        qualification='Existing frozen binary/test/source relationship only; no qualification inherited by candidate or current sources')


def common_specs(off,on):
    import paired
    a,b=off['spec'],on['spec']
    ct.require(a['engine']==b['engine']=='ferrule' and a['argv'][1:]==b['argv'][1:]
        and a['environment']==b['environment'] and off['runtime_pins']==on['runtime_pins'],
        'A/B requires identical inference flags/environment/runtime, except executable path')
    ct.require(b==paired.ferrule_spec(on['binary']['path'],paired.CACHE_WORKSPACE_SCOPE),
               'A/B requires exactly workspace4096/device22528, fixed TP2 profile')
    for key in ('topology','boundaries','workload','resources','limits','model','cache'):
        ct.require(off[key]==on[key],'A/B common contract differs: '+key)
    ct.require(b['environment'].get('FERRULE_LOG')=='info'
        and not any(k in b['environment'] for k in ('RUST_LOG','LD_PRELOAD')),
        'No diagnostic/profile injection in measured A/B')


def validate(admission):
    import paired
    ct.require(set(admission)=={'protocol','engines','cacheoff','cache_ab'}
        and admission['protocol']==PROTOCOL and admission['cache_ab']==contract(),
        'Unreviewed A/B protocol; no profiler or policy overrides')
    ct.require(set(admission['engines'])=={'ferrule'},'No vLLM execution in cache A/B')
    ct.require(admission['cacheoff']==frozen_metadata(),'Frozen cacheoff identity/source/spec changed')
    frozen_audit()
    # Bootstrap-level CURRENT release audit is mandatory, even though these two
    # short lifecycles do not claim the separate two-owner cancel/reuse smoke.
    paired.validate_admission(dict(protocol=pre.PROTOCOL,engines=admission['engines']),'serving_smoke')
    on=admission['engines']['ferrule']
    ct.require(on.get('decoded_cache',{}).get('enabled') is True
        and on.get('serving_scope')==paired.CACHE_WORKSPACE_SCOPE
        and on['binary']['sha256']!=historical.BINARY_SHA,'Qualified distinct cache-on candidate required')
    common_specs(admission['cacheoff'],on)
    return admission


def contract():
    return dict(order=list(SLOTS),fresh_lifecycles=2,workspace_mib=4096,device_cap_mib=22528,
        qa8_unmeasured=3,qa8_unmeasured_layout='existing excluded first QA8 + 2 warmups',qa8_measured=5,
        canaries='existing Hello2 + three frozen canaries before QA8',rankperf=False,profiler=False,
        cacheoff_policy=POLICY,cacheoff_snapshots=None,activity_when_off=None,
        comparison='matched deployment ceilings/flags; release/public/source differences remain, not isolated cache-only effect',
        candidate_serving_qualification='fresh readiness/canaries and measured requests only; not S1/S2 cancel/reuse smoke',
        vllm='historical same-machine baseline descriptive only; no V slots; full F/V after candidate wins')


def plan():
    import paired
    audit=frozen_audit()
    return dict(protocol=PROTOCOL,status='CPU_preparation_only_candidate_release_and_GPU_authorization_pending',
        execute_allowed=False,contract=contract(),cacheoff_binary_sha256=audit['binary_sha256'],
        candidate_binary=None,common_flags=paired.ferrule_flags(paired.CACHE_WORKSPACE_SCOPE),
        old_binary_preflight='actual help + hidden empty-model preflight passed CLI/pre-model boundary; full resource/GPU admission untested',
        snapshot_policy='candidate ready/shutdown physical component totals and capacity; old unavailable remains null; off-mode hit/miss activity stays null',
        supplementary_activity='future separately scoped model/profile evidence only; never enable debug in one timed variant or borrow prior dadf counters',
        current_tree_qualified=False,fairmatch=False,exact_speedup=None)


def admission(output,candidate_path):
    candidate=ct.load(candidate_path)
    ct.require(candidate['protocol']==pre.PROTOCOL,'Use the final candidate bootstrap smoke admission')
    result=dict(protocol=PROTOCOL,engines=deepcopy(candidate['engines']),cacheoff=frozen_metadata(),cache_ab=contract())
    validate(result)
    output=output.resolve()
    ct.require(output.is_relative_to(pre.HERE/'admission') and not output.exists() and output.parent.is_dir(),
               'Fresh existing paper admission location required')
    with output.open('x') as stream:
        import json
        json.dump(result,stream,indent=2);stream.write('\n')
    return output


def finish(directory,prepared):
    import paired
    rows=[]
    for slot in SLOTS:
        cell=directory/'runs'/slot
        assessed=paired.assess_slot(cell,'ferrule')
        case=ct.load(cell/'session/case.json')
        http=ct.load(cell/'session/qa8-V1/http.json')
        log=(cell/'session/server.log').read_text()
        ct.require('mixed_rank_performance' not in log and 'request_worker_terminal' not in log,
                   'A/B benchmark diagnostics unexpectedly enabled')
        for phase in ('hello2','canary8','qa8-V1'):
            ct.require(ct.load(cell/'session'/phase/'gate.json')['status']=='pass','A/B request gate failed')
        ct.require(len(http['warmups'])==2 and http['first_after_ready']['status']=='success'
            and len(http['runs'])==1 and len(http['runs'][0]['requests'])==5
            and all(r['status']=='success' and r['actual_input_tokens']==5 and r['actual_output_tokens']==8
                    for r in [http['first_after_ready'],*http['warmups'],*http['runs'][0]['requests']]),
            'A/B must retain 3 unmeasured + 5 measured QA8')
        m=paired.slot_metadata(prepared['admission'],slot)
        rows.append(dict(slot=slot,binary=m['binary'],build_source_pins=m['source_pins'],
            startup_seconds=case['startup_seconds'],summary=http['runs'][0]['summary'],
            resources=case['resources'],runtime=assessed['runtime'],launch_id=assessed['launch_id'],
            cache_snapshots=assessed['runtime'].get('decoded_cache_snapshots'),
            activity=None,activity_status='not collected in no-trace benchmark'))
    ct.require(rows[0]['launch_id']!=rows[1]['launch_id'],'A/B requires fresh server owners')
    result=dict(protocol=PROTOCOL,status='complete_descriptive_only',contract=contract(),rows=rows,
        fairmatch=False,exact_speedup=None,isolated_cache_effect=False,
        limitation='matched flags are necessary but old/new public and source implementations still differ; one lifecycle per variant, no statistics or automatic promotion',
        artifact_pins=ct.inventory([f for f in directory.rglob('*') if f.is_file()]))
    ct.write(directory/'cache-ab-summary.json',result)
    return result


def main():
    import argparse,json
    ct.require_cpu_python()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('plan','admission'))
    parser.add_argument('--candidate-admission',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.mode=='plan':print(json.dumps(plan(),indent=2))
    else:
        ct.require(args.output is not None and args.candidate_admission is not None,'Both admission paths required')
        print(admission(args.output,args.candidate_admission))


if __name__=='__main__':main()
