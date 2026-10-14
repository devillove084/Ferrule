"""Typed frozen-arena/indexed-arena A/B hooks; paired.py is the only launcher."""
from copy import deepcopy
import json
from pathlib import Path

import pre_serving as pre
import cache_ab
import decoded_cache_release as release

ct = pre.ct
PROTOCOL = 'tp2-arena-index-matched4g-qa8-descriptive-v1'
SLOTS = ('BEFORE','AFTER')
POLICY = 'frozen_qualified_ae3e_arena_before_v1'
BEFORE_SHA = release.ARENA_BINARY_SHA
BASELINE = pre.HERE/'admission/arena-public-20261010-01/cache-ab-qualified-candidate.json'
BASELINE_SHA = '1b0f093c39ffbb91befe96db954a79314449c28d34a95dc2440f949105d86da6'
BASELINE_AB = pre.HERE/'runs/arena-cache-ab-20261010-01/cache-ab-summary.json'
BASELINE_AB_SHA = '28aedb7441b2362cb0862bcc32b6f8b818198a68ee3c326445ae026258d756ed'
WORKFLOW_ARCHIVE = pre.HERE/'history/pre-arena-index-ab-20261010/paired.py.txt'
# These qualification-source files are already being updated by the model agent.
# Keep their authenticated historical identities, not claims about current bytes.
CLOSED_WORKFLOW_SOURCES = {
    str(pre.HERE/'decoded_cache_gate.py'): '595993f45ef13ce65138e1e43b3d9faad310996e33914d993e0d69698fa85ffc',
    str(pre.HERE/'test_decoded_cache_gate.py'): '1caeb4b9fa103f49f85da693c9f4107bdd71c37ebf83e49c28025b20072203d4',
}
DELTA_SCHEMA = 'ferrule.paper.arena-index-production-delta.v1'
ALLOWED_PRODUCTION = tuple(str(ct.ROOT/'crates/ferrule-model/src/transformer/standard'/name)
                           for name in ('expert_cache.rs','mixed_decoded.rs'))


def slots(cycle):
    ct.require(type(cycle) is int and cycle in (1,2), 'Arena index cycle must be 1 or 2')
    return SLOTS if cycle == 1 else tuple(reversed(SLOTS))


def contract(cycle):
    return dict(policy=POLICY,cycle=cycle,order=list(slots(cycle)),fresh_lifecycles=2,
        arena_bytes_per_owner=3<<30,workspace_mib=4096,device_cap_mib=22528,
        concurrency=1,qa8_unmeasured=3,qa8_unmeasured_layout='excluded first + 2 warmups',qa8_measured=5,
        actual_input_tokens=5,actual_output_tokens=8,rankperf=False,profiler=False,
        scope='TP2 ctx128 batch8 seq2 resident/no-offload; identical argv/environment/runtime except executable',
        source_delta='Exactly expert_cache.rs and standard/mixed_decoded.rs production hashes; tests reported separately',
        attribution='Two crossed-order cycles remain descriptive; neither an all-differences attribution nor statistical significance',
        cpu_observation=dict(affinity=None,thread_count=None,
            reason='Existing sampler captures memory/process identity, not affinity/thread counts; observer unchanged to preserve timing'))


def frozen_metadata():
    ct.require(ct.sha(BASELINE)==BASELINE_SHA, 'Frozen arena admission changed')
    return deepcopy(ct.load(BASELINE)['engines']['ferrule'])


def production_sources(m):
    build = pre.checked_document(m['build_receipt'])
    before = pre.checked_document(build['source_pins_before'])
    after = pre.checked_document(build['source_pins_after'])
    ct.require(before==after, 'Builder source inventory changed during build')
    return pre.release_sources(before), build


def frozen_audit():
    """Closed historical source inventories, never substituted with the current tree."""
    m = frozen_metadata()
    ct.require(m['binary']['sha256']==BEFORE_SHA and m['decoded_cache']==release.CACHE,
               'Frozen baseline is the qualified cache-ON ae3e arena, not cache-OFF')
    ct.validate_staged_binary(m['binary']['path'],BEFORE_SHA)
    ct.check_pins(m['runtime_pins'])
    sources, build = production_sources(m)
    ct.require(m['build_receipt']['sha256']==release.ARENA_HANDOFF_SHA
        and build['artifact']['sha256']==BEFORE_SHA, 'Frozen arena builder differs')
    quality = pre.checked_document(m['test_quality'])
    qualification = pre.checked_document(build['qualification_handoff'])
    summary = pre.checked_document(qualification['acceptance_summary'])
    ct.require(len(summary['rows'])==23 and {r['gate']:r['selector'] for r in summary['rows']}
        == {r['gate']:r['selector'] for r in quality['rows']}, 'Frozen named gate inventory differs')
    prepared = pre.checked_document(summary['prepared'])
    ct.require(all(prepared['pins'].get(p)==h for p,h in CLOSED_WORKFLOW_SOURCES.items()),
               'Historical qualification workflow source identities differ')
    ct.require(quality['status']=='pass' and len(quality['rows'])==23
        and (quality['calls'],quality['metrics'])==(37,40293), 'Frozen arena quality differs')
    # Only exact entries in the authenticated historical source maps are closed.
    # All other evidence, including executed workflow bytes, still hashes.
    historical = dict(summary['sources'])
    for name,digest in pre.checked_document(build['source_pins_before']).items():
        path = str((ct.ROOT/name).resolve())
        ct.require(path not in historical or historical[path]==digest, 'Conflicting historical source pin')
        historical[path] = digest
    historical.update(CLOSED_WORKFLOW_SOURCES)
    pins = {str(BASELINE):BASELINE_SHA,str(BASELINE_AB):BASELINE_AB_SHA}
    for group in (m['build_evidence_pins'],quality['artifact_pins']):
        for name,digest in group.items():
            if name in historical:
                ct.require(historical[name]==digest, 'Historical source pin relabelled')
                continue
            actual = str(WORKFLOW_ARCHIVE) if name==str(pre.HERE/'paired.py') else name
            ct.require(actual not in pins or pins[actual]==digest, 'Historical evidence pin conflict')
            pins[actual] = digest
    ct.check_pins(pins)
    smoke = pre.validate_smoke(m)
    ab = ct.load(BASELINE_AB)
    ct.check_pins(ab['artifact_pins'])
    on = next(row for row in ab['rows'] if row['slot']=='ON')
    ct.require(ab['status']=='complete_descriptive_only' and on['binary']==m['binary']
        and on['build_source_pins']==m['source_pins'] and on['summary']['success']==5
        and smoke['binary_sha256']==BEFORE_SHA, 'Frozen same-binary serving evidence differs')
    return dict(policy=POLICY,binary_sha256=BEFORE_SHA,admission_sha256=BASELINE_SHA,
        source_pins=m['source_pins'],production_source_digest=ct.digest(sources),
        historical_source_digest=ct.digest(historical),evidence_pins=pins,
        closed_historical_workflow_source_pins=dict(CLOSED_WORKFLOW_SOURCES),
        current_tree_qualified=False,cache_enabled=True,smoke_status='pass',historical_ab_status=ab['status'])


def validate_source_delta(before, after, delta, after_sha):
    ct.require(set(before)==set(after), 'Production source inventory added/removed files')
    changes = [dict(path=name,before_sha256=before[name],after_sha256=after[name])
               for name in sorted(before) if before[name]!=after[name]]
    ct.require({row['path'] for row in changes}==set(ALLOWED_PRODUCTION),
               'Production source delta must be exactly the two reviewed index files')
    expected = dict(schema=DELTA_SCHEMA,before_binary_sha256=BEFORE_SHA,after_binary_sha256=after_sha,
        before_production_source_digest=ct.digest(before),after_production_source_digest=ct.digest(after),
        production_changes=changes)
    ct.require(delta==expected, 'Builder source delta does not match actual production pins')
    return changes


def require_fresh_quality(q, prior):
    rows = q['rows']; gates = release.gates.ALL_GATES
    ct.require(q['schema']==release.QUALITY_SCHEMA and q['status']=='pass'
        and q['kind']=='authenticated_test_binary_not_serving'
        and q['numerical_pass_inherited'] is False and q['source_relation']['inherited_gpu_pass'] is False
        and (q['calls'],q['metrics'])==(37,40293), 'AFTER fresh numerical quality required')
    ct.require(len(rows)==len(gates)==23 and {r['gate'] for r in rows}==set(gates)
        and all(r['selector']==gates[r['gate']] and pre.contract._sha(r['test_binary_sha256']) for r in rows),
        'AFTER requires all 23 named selectors, not aggregate count')
    ids = {r['launch_id'] for r in rows}
    ct.require(len(ids)==23 and all(isinstance(x,str) and x for x in ids)
        and not ids.intersection(r['launch_id'] for r in prior['rows'])
        and pre.contract._sha(q.get('qualification_handoff_sha256'))
        and q['qualification_handoff_sha256']!=prior['qualification_handoff_sha256'],
        'AFTER cannot inherit baseline GPU gates/qualification')


def validate(admission):
    import paired
    ct.require(set(admission)=={'protocol','engines','before','cycle','arena_index_ab','source_delta','after_admission'}
        and admission['protocol']==PROTOCOL and set(admission['engines'])=={'ferrule'},
        'Unreviewed arena index protocol/engine/override')
    ct.require(admission['arena_index_ab']==contract(admission['cycle'])
        and admission['before']==frozen_metadata(), 'Arena index contract/frozen baseline differs')
    frozen_audit()
    after = admission['engines']['ferrule']; before = admission['before']
    supplied = pre.checked_document(admission['after_admission'])
    ct.require(supplied['engines']['ferrule']==after, 'AFTER admission identity differs')
    ct.require(after['binary']['sha256']!=BEFORE_SHA, 'AFTER must be a distinct indexed release')
    # This still requires the reviewed release adapter. An unknown/new handoff
    # schema is deliberately blocked until its builder receipt is integrated.
    paired.validate_admission(dict(protocol=pre.PROTOCOL,engines={'ferrule':after}),'serving_smoke')
    pre.validate_smoke(after)
    q = pre.checked_document(after['test_quality']); prior = pre.checked_document(before['test_quality'])
    require_fresh_quality(q,prior)
    cache_ab.common_specs(before,after)
    ct.require(before['decoded_cache']==after['decoded_cache']==release.CACHE
        and before['serving_scope']==after['serving_scope']==paired.CACHE_WORKSPACE_SCOPE,
        'Both variants must keep the same enabled 3-GiB arena policy')
    before_sources, before_build = production_sources(before)
    after_sources, after_build = production_sources(after)
    ct.check_pins(after_sources)
    delta_link = admission['source_delta']
    ct.require(after['build_evidence_pins'].get(delta_link['path'])==delta_link['sha256'],
               'Source delta must be pinned in AFTER builder evidence')
    validate_source_delta(before_sources,after_sources,pre.checked_document(delta_link),after['binary']['sha256'])
    old_inputs = pre.checked_document(before_build['receipts']['build-inputs'])
    new_inputs = pre.checked_document(after_build['receipts']['build-inputs'])
    for key in ('environment','release_profile'):
        ct.require(new_inputs[key]==old_inputs[key], 'Build environment/profile differs beyond index change')
    for key in ('cargo','rustc','nvcc','cxx','cutlass_head'):
        ct.require(new_inputs['toolchain'][key]==old_inputs['toolchain'][key], 'Build toolchain differs: '+key)
    plan = paired.contract.http_plan('ferrule',after['spec'],'qa8',Path('/not-executed'))
    argv = plan['argv']
    ct.require(argv[argv.index('--concurrency')+1]=='1' and argv[argv.index('--warmup')+1]=='2'
        and '--first-after-ready' in argv, 'Arena index must retain serial 3-unmeasured/5-measured QA8')
    ct.require(paired.shared.config()['workloads']['qa8']==dict(max_tokens=8,requests=5,warmup=2,first_after_ready=True),
               'Frozen QA8 request contract differs')
    return admission


def admission(output, after_path, after_sha, delta_path, delta_sha, cycle):
    ct.require(pre.contract._sha(after_sha) and pre.contract._sha(delta_sha), 'Explicit final admission/delta hashes required')
    after_link = dict(path=str(after_path.resolve()),sha256=after_sha)
    candidate = pre.checked_document(after_link)
    result = dict(protocol=PROTOCOL,engines={'ferrule':deepcopy(candidate['engines']['ferrule'])},
        before=frozen_metadata(),cycle=cycle,arena_index_ab=contract(cycle),after_admission=after_link,
        source_delta=dict(path=str(delta_path.resolve()),sha256=delta_sha))
    validate(result)
    output = output.resolve()
    ct.require(output.is_relative_to(pre.HERE/'admission') and output.parent.is_dir() and not output.exists(),
               'Fresh existing paper admission location required')
    with output.open('x') as stream:
        json.dump(result,stream,indent=2);stream.write('\n')
    return output


def finish(directory, prepared):
    import paired
    rows = []; a = prepared['admission']; refs = paired.shared.references()[0]
    for slot in slots(a['cycle']):
        cell=directory/'runs'/slot; session=cell/'session'
        assessed=paired.assess_slot(cell,'ferrule'); case=ct.load(session/'case.json')
        log=(session/'server.log').read_text()
        ct.require('mixed_rank_performance' not in log and 'request_worker_terminal' not in log,
                   'Benchmark diagnostics unexpectedly enabled')
        snapshots=release.snapshot_rows(log,profiling=False)
        ct.require(all(r['schema_version']==2 and r['snapshot']['arena_backing_grants']==3<<30 for r in snapshots),
                   'Both variants require observed v2 3-GiB arena roots')
        http=ct.load(session/'qa8-V1/http.json')
        for phase in ('hello2','canary8','qa8-V1'):
            gate=ct.load(session/phase/'gate.json')
            ct.require(gate['status']=='pass' and gate['http_receipt_sha256']==ct.sha(session/phase/'http.json'),
                       'Arena index HTTP gate/report differs')
        ct.require(len(http['warmups'])==2 and len(http['runs'])==1 and len(http['runs'][0]['requests'])==5,
                   'Require three unmeasured and five measured QA8')
        requests=[http['first_after_ready'],*http['warmups'],*http['runs'][0]['requests']]
        ct.require(all(r['status']=='success' and r['actual_input_tokens']==5 and r['actual_output_tokens']==8
            and r['text']==refs[0]['reference_text'] for r in requests), 'Arena index actual tokens/canary differ')
        with (session/'samples.jsonl').open() as stream:
            samples=[r for r in map(json.loads,stream) if r.get('kind')=='sample']
        resources=case['resources']
        ct.require(resources['status']=='observed_within_targets' and not resources['errors'], 'Arena index resource failure')
        rows.append(dict(slot=slot,binary=paired.slot_metadata(a,slot)['binary'],launch_id=assessed['launch_id'],
            startup_seconds=case['startup_seconds'],summary=http['runs'][0]['summary'],resources=resources,
            cache_snapshots=snapshots,activity=None,cpu_observation=contract(a['cycle'])['cpu_observation'],
            gpu_samples=sum(r.get('gpu') is not None for r in samples),missing_gpu_samples=sum(r.get('gpu') is None for r in samples)))
    ct.require(len({r['launch_id'] for r in rows})==2, 'Fresh owners required')
    result=dict(protocol=PROTOCOL,status='complete_descriptive_only',cycle=a['cycle'],order=list(slots(a['cycle'])),
        comparison_identity=ct.digest(dict(before_sha=BEFORE_SHA,after_sha=a['engines']['ferrule']['binary']['sha256'],
            delta_sha=a['source_delta']['sha256'])),source_delta=a['source_delta'],rows=rows,
        source_pins={s:paired.slot_metadata(a,s)['source_pins'] for s in SLOTS},
        rate_window='Original runner-observed fixed-work window includes terminal checkpoint/executor exit; rates unchanged',
        limitations=['Two crossed cycles are descriptive, not statistical significance or exclusive causal attribution.',
            'Whole-lifecycle cache counts do not identify measured hot decode; activity null is not zero.',
            'No affinity/thread-count observation; no CPU oversubscription or GPU thermal/clock attribution.',
            'NVML host PID mapping may be unavailable; local authenticated drain does not prove absence of foreign interference.'],
        isolated_cache_effect=False,statistical_significance_claim=False,fairmatch=False,exact_speedup=None,
        artifact_pins=ct.inventory([p for p in directory.rglob('*') if p.is_file()]))
    ct.require(not (directory/'arena-index-summary.json').exists(), 'No summary overwrite')
    ct.write(directory/'arena-index-summary.json',result)
    return result


def plan():
    return dict(protocol=PROTOCOL,status='CPU_plan_only_AFTER_release_quality_smoke_and_delta_pending',
        execute_allowed=False,before=dict(path=str(BASELINE),sha256=BASELINE_SHA,binary_sha256=BEFORE_SHA),
        cycles=[contract(1),contract(2)],after_binary_sha256=None,
        builder_delta_schema=dict(schema=DELTA_SCHEMA,required_fields=['before_binary_sha256','after_binary_sha256',
            'before_production_source_digest','after_production_source_digest','production_changes'],
            production_changes_fields=['path','before_sha256','after_sha256'],allowed_paths=list(ALLOWED_PRODUCTION),
            digest='contract.digest(pre_serving.release_sources(builder source_pins_before)); includes native .cc',
            binding='Exact delta file path/hash must be included in AFTER build_evidence_pins'),
        after_requirements=['Reviewed release adapter, current source pins, stable before/after build inputs',
            'Fresh authenticated 23 named gates, 37 calls / 40293 exact; no inherited baseline launch IDs',
            'Actual same-binary S1/S2 PASS before A/B admission',
            'Exactly two production files changed; identical build environment/profile/toolchain and runtime/serving flags'],
        launcher='paired.py --purpose arena_index_ab; no new harness',gpu_started=False)


def main(argv=None):
    import argparse
    ct.require_cpu_python()
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('mode',choices=('plan','admission'))
    cli.add_argument('--after-admission',type=Path)
    cli.add_argument('--after-admission-sha256')
    cli.add_argument('--source-delta',type=Path)
    cli.add_argument('--source-delta-sha256')
    cli.add_argument('--cycle',type=int,choices=(1,2))
    cli.add_argument('--output',type=Path)
    args=cli.parse_args(argv)
    if args.mode=='plan':
        print(json.dumps(plan(),indent=2))
    else:
        ct.require(all(x is not None for x in (args.after_admission,args.source_delta,args.cycle,args.output)),
                   'AFTER admission, builder source delta, cycle and output required')
        print(admission(args.output,args.after_admission,args.after_admission_sha256,args.source_delta,args.source_delta_sha256,args.cycle))


if __name__=='__main__':
    main()
