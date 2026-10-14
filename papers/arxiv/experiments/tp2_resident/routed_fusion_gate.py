"""Routed-fusion qualification using the existing authenticated paper supervisor.

Typed gates separate diagnostic tensors from actual cache-on unobserved fusion.
Selected GPU gates bind the immutable independent HF release and fresh binaries,
require explicit sole-owner authorization, and preserve exclusive receipts.
Preparation is GPU-hidden; no new execution loop or arithmetic oracle is added.
"""
import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import decoded_cache_gate as base
ct = base.ct
ROOT = base.ROOT
FREEZES = {
    'crates/ferrule-backend/tests/numeric_bf16_gemv/pair_swiglu.sha256':
        '5e16d782e1781e86e06e0aa2743a7392f6538aab893250af9accf19b5bb8ae02',
    'crates/ferrule-model/tests/support/mixed_routed_fusion.sha256':
        '15c8e46222eb11ab6a2ea6e679d15a96864223622210560016387c7d493403fb',
}
RELEASE = ROOT / 'target/validation/tp2-mixed-allocator-v3-release-20261011-01/handoff.json'
PAIR = 'cuda::providers::bf16_gemmex::pair_swiglu_tests::'
MODEL = base.MODEL_PREFIX

@dataclass(frozen=True)
class Gate:
    name: str
    binary: str
    selector: str
    evidence: str
    devices: int
    reference: str | None = None
    cache_bytes: int | None = None
    observer: bool | None = None


def gates():
    rows = [Gate(name, binary, selector, 'existing_backend_regression' if name.startswith('backend-')
                 else 'source_pinned_model_small', devices)
            for name, (binary, selector, devices) in base.SMALL_GATES.items()]
    rows += [Gate('backend-' + name, 'backend', PAIR + name, 'shared_helper_small_not_independent_hf', 1)
             for name in ('pair_swiglu_rankpair_m1_m2_m5_exact', 'pair_swiglu_bf16_edges_m1_m2_m5_exact',
                          'pair_swiglu_validation_and_poison')]
    rows += [Gate('backend-down-nan-c', 'backend-output',
        'routed_down_nan_c_packed_offset_beta_zero_m1_m2_m5', 'backend_beta_zero_not_model_path', 1)]
    rows += [Gate('model-' + name, 'model', MODEL + 'routed_fusion::' + name, 'source_pinned_model_small', 2)
             for name in ('routed_fusion_noobs_and_obs_exact_with_launch_and_allocation_counts',
                          'routed_pack_words_nonfinite_contract_and_removed_launches',
                          'routed_fusion_unknown_retains_leases_and_never_replays',
                          'routed_fusion_after_down_failure_does_not_replay')]
    rows += [Gate('model-down-canaries', 'model', MODEL +
        'routed_down_staged_slices_match_standalone_and_preserve_neighbors', 'source_pinned_model_small', 2)]
    rows += [Gate('diagnostic-' + name, 'composition', selector, 'diagnostic_fallback_full_inventory', 2,
                  name, 3 << 30, True) for name, selector in base.GATES.items()]
    rows += [Gate('fused-native-full40', 'composition', 'full40_independent_logits_prefill_one_decode',
                  'unobserved_fused_logits_publication', 2, 'full40', None, False),
             Gate('fused-cache-full40', 'composition',
                  'decoded_cache::full40_cache_unobserved_fused_logits_two_cases_two_phases',
                  'unobserved_fused_logits_publication', 2, 'full40', 3 << 30, False)]
    rows += [Gate('fused-cache-' + name, 'composition',
                  'lifecycle::full40_decoded_cache_unobserved_fused_lifecycle',
                  'unobserved_fused_logits_publication', 2, name, 3 << 30, False) for name in base.RECIPES]
    ct.require(len({r.name for r in rows}) == len(rows), 'Duplicate named gate')
    return rows


def plan():
    return dict(schema='ferrule.paper.routed-fusion-quality-plan.v1',
        gates=[asdict(g) for g in gates()], gate_count=len(gates()),
        build_timeout_seconds=600, cuda='12.4', arch='sm_86', cublas_workspace=':0:0',
        oracle_enabled=False, arena_bytes=3 << 30, device_cap_bytes=22 << 30, snapshot_version=2,
        numerical_thresholds=dict(atol=.00025, rtol=.012, nonfinite_logits='reject', greedy='independent reference tokens'),
        diagnostic_claim='Original 2178 metrics per full40 case / 1089 per lifecycle call; split observer fallback, not fused proof',
        fused_claim='Only no-observer full-vocabulary logits, independent per-call tokens, and host publication/frontier checks; no internal tensor claim',
        independent_hf_fixture=dict(status='ready_for_selected_gates',
            selector='real_hf_pair_swiglu_cold_and_resident_every_word',
            manifest=str(HF_ROOT/'manifest.json'), manifest_sha256=HF_SHA, required_before_gpu=True,
            reason='Ten independent HF fixtures compare every cold/resident BF16 word'),
        model_nan_c_gap='Frozen model canary preloads a finite sentinel. New NaN-C gate is backend-only; model-level NaN prefill still needs a separately reviewed test hook/fixture.',
        gpu_execution=False, gpu_authorization=False, release_rebuilt=False, performance_claim=False)


def freeze():
    for name, expected in FREEZES.items():
        ct.require(ct.sha(ROOT/name) == expected, 'Freeze manifest changed: ' + name)
        for line in (ROOT/name).read_text().splitlines():
            digest, path = line.split(maxsplit=1)
            ct.require(ct.sha(ROOT/path) == digest, 'Frozen backend/model input changed: ' + path)


def environment():
    env = ct.build_environment(ct.EXCLUSIVE)
    env.update(CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='void', FERRULE_NO_CUDA='1',
               FERRULE_CUDA_TEST_ORACLE='0', CUBLAS_WORKSPACE_CONFIG=':0:0')
    ct.require(env['CUDA_HOME'] == '/home/admin/wxw403883/anaconda3/envs/tao', 'Need known tao CUDA12.4')
    return env


def checked_finite_logits(actual, expected, next_token, vocab):
    ct.require(len(actual) == len(expected) and len(actual) > 0 and len(actual) % vocab == 0,
               'Full-vocabulary shape mismatch')
    ct.require(all(math.isfinite(x) for x in actual) and all(math.isfinite(x) for x in expected),
               'Nonfinite logits cannot qualify')
    errors = [abs(float(a)-float(b)) for a,b in zip(actual, expected)]
    failures = sum(d > .00025 + .012*abs(float(b)) for d,b in zip(errors,expected))
    greedy = max(range(vocab), key=lambda i:(actual[-vocab+i], -i))
    ct.require(failures == 0 and greedy == next_token, 'Independent logits/tokens failed')
    return dict(compared=len(actual), failures=failures, nonfinite=0, max_abs=max(errors),
                rmse=math.sqrt(sum(d*d for d in errors)/len(errors)), greedy=greedy,
                all_exact=all(a == b for a,b in zip(actual,expected)))


def validate_execution_claim(gate, contract):
    ct.require(gate.reference is not None, 'Not a full-model gate')
    expected = 'diagnostic_fallback' if gate.observer else 'unobserved_fused'
    ct.require(contract['schema'] == 'ferrule.paper.routed-fusion-execution-path.v1'
        and contract['path'] == expected
        and contract['numeric_observer_installed'] is gate.observer
        and contract['committed_tensor_export'] is gate.observer
        and contract['cache_grants_per_owner'] == gate.cache_bytes
        and contract['device_cap_bytes'] == 22 << 30
        and contract['atol'] == .00025 and contract['rtol'] == .012,
        'Observer/path/cache/threshold mismatch; diagnostic results cannot qualify fusion')


def compare_unobserved(directory, gate):
    """Read immutable serialized oracle logits; never imports the arithmetic helper/exporter."""
    ct.require(gate.evidence == 'unobserved_fused_logits_publication', 'Wrong evidence type')
    validate_execution_claim(gate, ct.load(directory/'execution-path.json'))
    ct.require(not (directory/'candidate').exists() and not (directory/'lifecycle-raw').exists(),
               'Unobserved gate has diagnostic tensor capture')
    ref, digest = base.FULL if gate.reference == 'full40' else base.RECIPES[gate.reference]
    reference = base.ORACLE/'runs'/ref
    ct.require(ct.sha(reference/'manifest.json') == digest, 'Immutable oracle manifest drift')
    manifest = ct.load(reference/'manifest.json')
    def artifact(row):
        path=(reference/row['file']).resolve()
        ct.require(path.parent == reference.resolve() and ct.sha(path) == row['sha256'], 'Oracle artifact drift/escape')
        return path
    reports=[]
    for entry in manifest['cases']:
        tensor, report = artifact(entry), ct.load(artifact(entry['report']))
        with tensor.open('rb') as stream:
            size=struct.unpack('<Q',stream.read(8))[0]
            ct.require(size < 64<<20,'Oversized tensor header')
            header=json.loads(stream.read(size))
            for call in report['calls']:
                stage=call['stage'];spec=header[stage+'.logits'];tokens=call['input_token_ids']
                ct.require(spec['shape'] == [1,len(tokens),248320], 'Oracle full-vocab geometry')
                start,end=spec['data_offsets'];stream.seek(8+size+start);raw=stream.read(end-start)
                ct.require(len(raw) == end-start,'Truncated reference')
                if spec['dtype']=='F32': expected=[x[0] for x in struct.iter_unpack('<f',raw)]
                else:
                    ct.require(spec['dtype']=='BF16','Unknown logit type')
                    expected=[struct.unpack('<f',struct.pack('<I',x[0]<<16))[0] for x in struct.iter_unpack('<H',raw)]
                actual_path=directory/(entry['id']+'.'+stage+'.actual.f32le')
                actual=[x[0] for x in struct.iter_unpack('<f',actual_path.read_bytes())]
                reports.append(dict(case=entry['id'],stage=stage,input_token_ids=tokens,
                    actual=base.record(actual_path),**checked_finite_logits(actual,expected,call['next_token_id'],248320)))
    if gate.reference != 'full40':
        result=ct.load(directory/'lifecycle-unobserved/result.json')
        ct.require(result['schema']=='ferrule.mixed-lifecycle-unobserved.v1'
            and result['status']=='executed_unobserved' and result['shutdown_ok'] is True
            and result['writer_complete'] is None and result['numeric_observer_installed'] is False,
            'Unobserved lifecycle incomplete or captured')
        operations=manifest['suite']['plan']['operations']
        ct.require(len(result['events'])==len(operations),'Missing operations')
        for op,event in zip(operations,result['events']):
            ct.require(all(event.get(k)==v for k,v in op.items()),'Changed frozen recipe')
            if op['op']=='call':
                call=next(r for r in reports if r['case']==op['id'])
                ct.require(event['call']['input_token_ids']==call['input_token_ids']
                    and event['call']['next_token_id']==call['greedy'],'Per-call independent tokens differ')
                validate_publication(event['publication'])
    else:
        pubs=sorted(directory.glob('case-*.publication.json'))
        ct.require(len(pubs)==len(reports),'Missing full40 publication checks')
        for path in pubs: validate_publication(ct.load(path))
    return dict(schema='ferrule.paper.routed-fusion-unobserved-logits.v1',gate=gate.name,
        oracle_manifest_sha256=digest,calls=reports,passed=True,internal_tensor_comparison=False)


def validate_publication(row):
    ct.require(row['schema']=='ferrule.paper.unobserved-publication.v1'
        and row['publications_after']==row['publications_before']+1
        and row['submitted_rows']>0 and row['committed_tokens']==row['position_start']+row['submitted_rows']
        and row['outstanding']==0 and row['quarantined'] is False and row['tensor_readback'] is False,
        'Publication/frontier evidence incomplete')


def prepare(identity):
    out=base.run_directory(identity,True);freeze();env=environment()
    sources=ct.source_pins();tools=ct.required_inventory(ct.EXCLUSIVE)
    ct.write(out/'sources-before.json',sources);ct.write(out/'environment.json',env);ct.write(out/'plan.json',plan())
    release=ct.load(RELEASE);ct.require(ct.sha(release['artifact']['path'])==release['artifact']['sha256'], 'Old serving artifact changed')
    ct.write(out/'preserved-release.json',dict(handoff=base.record(RELEASE),artifact=release['artifact']))
    ct.require('release 12.4' in subprocess.check_output([env['NVCC'],'--version'],env=env,text=True,timeout=10),'Wrong CUDA compiler')
    target=ROOT/'target/pair-swiglu-cuda124'
    targets=[('backend','ferrule-backend','ferrule_backend',['--lib']),
        ('backend-slice','ferrule-backend','cuda_bf16_slice',['--test','cuda_bf16_slice']),
        ('backend-arena','ferrule-backend','cuda_bf16_arena',['--test','cuda_bf16_arena']),
        ('backend-output','ferrule-backend','cuda_routed_fusion_output',['--test','cuda_routed_fusion_output']),
        ('model','ferrule-model','ferrule_model',['--lib']),
        ('model-output','ferrule-model','mixed_routed_output',['--test','mixed_routed_output']),
        ('model-layer','ferrule-model','hybrid_mixed_layer',['--test','hybrid_mixed_layer']),
        ('model-handoff','ferrule-model','mixed_decoded_arena_handoff',['--test','mixed_decoded_arena_handoff']),
        ('runtime','ferrule-runtime','mixed_decoded_snapshot',['--test','mixed_decoded_snapshot']),
        ('composition','ferrule-composition','tp2_mixed_execution',['--test','tp2_mixed_execution'])]
    binaries={}
    for key,package,name,flags in targets:
        text=base.cpu_command(out,'build-'+key,[ct.EXCLUSIVE['cargo'],'test','--locked','--offline','-p',package,
            '--features','cuda',*flags,'--no-run','--message-format=json','--target-dir',target],env,600)
        artifacts=[json.loads(line) for line in text.splitlines() if line.startswith('{')]
        matches=[a['executable'] for a in artifacts if a.get('reason')=='compiler-artifact'
            and a['target']['name']==name and a.get('executable') and a['profile']['test']]
        ct.require(len(matches)==1,'Missing exact test artifact '+name)
        stage=out/('binary-'+key);stage.mkdir();binaries[key]=base.record(ct.stage_binary(matches[0],stage))
    inventory=[]
    for gate in gates():
        argv=[binaries[gate.binary]['path'],gate.selector,'--exact','--ignored','--list']
        text=subprocess.check_output(argv,env=env,text=True,timeout=20)
        ct.require(text.splitlines().count(gate.selector+': test')==1 and '1 test, 0 benchmarks' in text,
            'Missing exact ignored selector: '+gate.name)
        inventory.append(dict(**asdict(gate),listed=True,executed=False,binary_sha256=binaries[gate.binary]['sha256']))
    ct.write(out/'selector-inventory.json',inventory)
    cpu=[(key,selector,(key,selector) in base.IGNORED_HOST) for key,selector in base.HOST_SMALL]
    cpu += [('composition',selector,ignored) for selector,ignored in base.CPU.items()]
    cpu += [('backend',PAIR+name,False) for name in ('pair_swiglu_cold_preflight_without_device_calls',
        'pair_swiglu_resident_preflight_without_device_calls','pair_swiglu_rank_payloads_are_admitted_without_device_calls')]
    cpu += [('composition','quality_path::'+name,False) for name in ('diagnostic_capture_never_claims_fused_quality',
        'unobserved_cache_contract_installs_no_tensor_observer','publication_requires_one_commit_and_exact_frontier')]
    cpu += [('composition','lifecycle::recipe::canonical_recipe_fixtures_budget_invocations_and_recreation_without_gpu',False)]
    receipts=[]
    for index,(key,selector,ignored) in enumerate(cpu):
        argv=[binaries[key]['path'],selector,'--exact','--nocapture','--test-threads=1']+(['--ignored'] if ignored else [])
        text=base.cpu_command(out,'cpu-'+str(index),argv,dict(env,FERRULE_MIXED_HEADER_DIR=base.MODEL),600)
        ct.require('1 passed; 0 failed; 0 ignored' in text,'CPU selector absent/skipped')
        receipts.append(dict(binary=key,selector=selector,receipt=base.record(out/('cpu-'+str(index)+'.json'))))
    for key in ('model-output','model-layer'):
        text=base.cpu_command(out,'cpu-'+key,[binaries[key]['path'],'--test-threads=1'],env,600)
        ct.require('test result: ok.' in text and '0 passed;' not in text,'Empty CPU phase')
    base.cpu_command(out,'numeric-admission',[ct.EXCLUSIVE['cargo'],'test','--locked','--offline','-p','ferrule-backend',
        '--test','bf16_gemmex','--test','numeric_bf16_gemv','--test','numeric_fp8_payload','--test','numeric_w8a16',
        '--target-dir',target],env,600)
    base.cpu_command(out,'paper-policy',[sys.executable,'-S','-B','-m','unittest','discover','-s',HERE,
        '-p','test_routed_fusion_gate.py','-v'],env,60)
    base.cpu_command(out,'lifecycle-bridge-policy',[sys.executable,'-S','-B','-m','unittest','discover','-s',
        ROOT/'crates/ferrule-composition/tests/tp2_mixed_execution','-p','test_lifecycle_bridge.py','-v'],env,60)
    for package,flags in [('ferrule-backend',['--lib','--test','cuda_routed_fusion_output']),
                          ('ferrule-model',['--lib','--tests']),
                          ('ferrule-composition',['--test','tp2_mixed_execution'])]:
        base.cpu_command(out,'clippy-'+package,[ct.EXCLUSIVE['cargo'],'clippy','--locked','--offline','-p',package,
            '--features','cuda',*flags,'--target-dir',target,'--','-D','warnings'],env,600)
    freeze();ct.require(sources==ct.source_pins(),'Source changed during preparation')
    ct.validate_inventory(ct.EXCLUSIVE,tools)
    ct.require(ct.sha(release['artifact']['path'])==release['artifact']['sha256'],'Old release changed')
    for ref,digest in [base.FULL,*base.RECIPES.values()]:
        ct.require(ct.sha(base.ORACLE/'runs'/ref/'manifest.json')==digest,'Immutable raw oracle changed')
    pins=ct.inventory([Path(__file__),HERE/'test_routed_fusion_gate.py',*base.workflow_paths(),
        *[out/name for name in ('plan.json','environment.json','selector-inventory.json','preserved-release.json')],
        *[path for path in out.iterdir() if path.is_file() and path.suffix in ('.json','.stdout','.stderr')]])
    previous=ct.load(base.checked(release['source_pins_after']))
    delta={path:dict(before=previous.get(path),after=sources.get(path)) for path in previous.keys()|sources.keys()
           if previous.get(path)!=sources.get(path)}
    ct.write(out/'source-difference-from-release.json',delta)
    ct.write(out/'prepared.json',dict(schema='ferrule.paper.routed-fusion-prepared.v1',sources=sources,tools=tools,
        binaries=binaries,pins=pins,cpu_exact=receipts,freezes=FREEZES,source_difference=base.record(out/'source-difference-from-release.json'),
        status='CPU_PREPARED_GPU_BLOCKED_PENDING_HF_FIXTURE_AND_NEW_OWNER_GRANT',gpu_run=False,gpu_owner_lock_acquired=False,
        independent_fixture_ready=False,model_nan_c_covered=False,old_release_preserved=True,release_rebuilt=False))
    print(out/'prepared.json',flush=True)


HF_ROOT = base.ORACLE/'runs/pair-swiglu-m1-m5-cuda-01'
HF_SHA = '807e5b5643ecce603088052e39fbee36ce15dca4a3ba319b51be120e2922d40c'
DIRECT = 'pair_swiglu_direct_tail::gpu::direct_tail_words_old_literal_vs_fused_m1_m2_m5'

def selected_gates(route_batch=False):
    if route_batch:
        return route_batch_gates()
    names = ['backend-'+n for n in ('pair_swiglu_rankpair_m1_m2_m5_exact',
        'pair_swiglu_bf16_edges_m1_m2_m5_exact', 'pair_swiglu_validation_and_poison')]
    names += ['model-'+n for n in ('routed_fusion_noobs_and_obs_exact_with_launch_and_allocation_counts',
        'routed_pack_words_nonfinite_contract_and_removed_launches',
        'routed_fusion_unknown_retains_leases_and_never_replays', 'routed_fusion_after_down_failure_does_not_replay')]
    names += ['model-down-canaries', 'fused-native-full40', 'fused-cache-full40']
    names += ['fused-cache-'+n for n in base.RECIPES]
    rows = [g for g in gates() if g.name in names]
    rows.insert(3, Gate('backend-direct-tail', 'backend-tail', DIRECT, 'independent_literal_tail_words_not_hf', 1))
    rows.insert(4, Gate('real-hf-pair', 'model-hf', 'real_hf_pair_swiglu_cold_and_resident_every_word',
        'independent_real_hf_every_bf16_word', 1))
    return rows


def prepare_selected(identity, e2e=False, route_batch=False):
    out = base.run_directory(identity, not e2e)
    if route_batch:
        route_batch_freeze()
    else:
        freeze()
        ct.require(ct.sha(HF_ROOT/'manifest.json') == HF_SHA, 'Independent HF release changed')
    env = environment()
    env['FERRULE_CUDA_TEST_ORACLE'] = '0' if route_batch else '1'
    sources = ct.source_pins()
    tools = ct.required_inventory(ct.EXCLUSIVE)
    targets = [('composition', 'ferrule-composition', 'tp2_mixed_execution', ['--test','tp2_mixed_execution'])] if e2e else [
        ('backend', 'ferrule-backend', 'ferrule_backend', ['--lib']),
        ('backend-tail', 'ferrule-backend', 'numeric_bf16_gemv', ['--test','numeric_bf16_gemv']),
        ('model', 'ferrule-model', 'ferrule_model', ['--lib']),
        ('model-hf', 'ferrule-model', 'pair_swiglu_hf_fixtures', ['--test','pair_swiglu_hf_fixtures'])]
    if route_batch and not e2e:
        targets = [('backend','ferrule-backend','ferrule_backend',['--lib']), ('model','ferrule-model','ferrule_model',['--lib'])]
    if e2e:
        prior = ct.load(out/'math-prepared.json')
        ct.check_pins(prior['pins'])
        ct.require(prior['sources'] == sources, 'Math source drift')
        for gate in selected_gates(route_batch):
            if gate.reference is None:
                ct.require(ct.load(out/gate.name/'result.json')['passed'] is True, 'Small gate did not pass: '+gate.name)
    binaries = {}
    for key, package, name, flags in targets:
        text = base.cpu_command(out, 'build-'+key, [ct.EXCLUSIVE['cargo'], 'test', '--locked','--offline',
            '-p',package,'--features','cuda',*flags,'--no-run','--message-format=json',
            '--target-dir',ROOT/'target/pair-swiglu-cuda124'], env, 600)
        artifacts = [json.loads(line) for line in text.splitlines() if line.startswith('{')]
        matches = [a['executable'] for a in artifacts if a.get('reason') == 'compiler-artifact'
            and a['target']['name'] == name and a.get('executable') and a['profile']['test']]
        ct.require(len(matches) == 1, 'Missing fresh exact test binary: '+name)
        stage = out/('binary-'+key)
        stage.mkdir()
        binaries[key] = base.record(ct.stage_binary(matches[0], stage))
    for gate in selected_gates(route_batch):
        if gate.binary in binaries:
            listing = subprocess.check_output([binaries[gate.binary]['path'],gate.selector,'--exact','--list'],
                env=env, text=True, timeout=20)
            ct.require(listing.splitlines().count(gate.selector+': test') == 1 and '1 test, 0 benchmarks' in listing,
                'Selector must be exactly one, never zero: '+gate.name)
    ct.require(sources == ct.source_pins(), 'Source changed during build')
    release = ct.load(RELEASE)
    artifact = ct.load(FUSED_ARTIFACT) if route_batch else release['artifact']
    ct.check_pins({artifact['path']:artifact['sha256']})
    pins = ct.inventory([Path(__file__),*base.workflow_paths(),HF_ROOT/'manifest.json',
        ROOT/'crates/ferrule-backend/tests/numeric_bf16_gemv/pair_swiglu_direct_tail.sha256',RELEASE,
        *[p for p in out.iterdir() if p.is_file() and p.suffix in ('.json','.stdout','.stderr')]])
    if route_batch:
        pins.update(ct.inventory([ROUTE_MANIFEST,ROUTE_ARCHIVE,FUSED_ARTIFACT]))
    ct.write(out/('e2e-prepared.json' if e2e else 'math-prepared.json'), dict(sources=sources,tools=tools,route_batch=route_batch,
        binaries=binaries,pins=pins,environment=env,hf_manifest_sha256=HF_SHA,
        selected_gates=[asdict(g) for g in selected_gates(route_batch)],old_release_preserved=artifact,
        performance_claim=False))
    print(out, flush=True)


def selected_worker(pin):
    p = json.load(sys.stdin)
    route_batch = p.get('route_batch',False)
    auth = base.guard.authenticate(p,pin)
    ct.check_pins(p['pins']); ct.check_pins(p['sources'])
    ct.validate_staged_binary(p['binary'],p['binary_sha256'])
    gate = next(g for g in selected_gates(route_batch) if g.name == p['gate'])
    selected = base.UUIDS[:gate.devices]
    ct.require(selected == p['selected'], 'Sealed selected UUID mismatch')
    for uuid in selected:
        ct.validate_owner_lock(p['locks'][uuid],uuid,p['coordinator']['pid'])
    out = Path(p['directory'])
    ct.write(out/'authentication.json',auth)
    env = dict(PATH='/usr/bin:/bin',HOME=str(Path.home()),CUDA_VISIBLE_DEVICES=','.join(selected),
        NVIDIA_VISIBLE_DEVICES=','.join(selected),CUDA_DEVICE_ORDER='PCI_BUS_ID',
        LD_LIBRARY_PATH='/home/admin/wxw403883/anaconda3/envs/tao/lib:/usr/lib64',
        CUBLAS_WORKSPACE_CONFIG=':0:0',FERRULE_MIXED_GPU_RUN='1',FERRULE_RUN_MIXED_RANK_GPU='1',
        FERRULE_MIXED_MODEL=base.MODEL,FERRULE_MIXED_ORACLE_DIR=str(base.ORACLE/'runs'/base.FULL[0]),
        FERRULE_MIXED_ORACLE_SHA256=base.FULL[1],FERRULE_MIXED_EVIDENCE_DIR=str(out),
        FERRULE_PAIR_SWIGLU_FIXTURE_ROOT=str(HF_ROOT),FERRULE_PAPER_WORKER_TOKEN=p['worker_token'],
        FERRULE_LOG='info',FERRULE_LOG_FORMAT='json')
    if gate.reference in base.RECIPES:
        ref,sha = base.RECIPES[gate.reference]
        env.update(FERRULE_MIXED_LIFECYCLE_ORACLE_DIR=str(base.ORACLE/'runs'/ref),FERRULE_MIXED_LIFECYCLE_ORACLE_SHA256=sha)
    ct.write(out/'environment.json',env)
    with (out/'test.log').open('x') as log:
        with ct.pinned_executable(p['binary'],p['binary_sha256']) as fd:
            proc = subprocess.Popen(['/proc/self/fd/'+str(fd),gate.selector,'--exact','--ignored','--nocapture','--test-threads=1'],
                env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,pass_fds=(fd,))
        config = dict(base.CONFIG, resources=dict(base.CONFIG['resources'], sample_interval_seconds=.05 if gate.reference is None else .5))
        monitor = base.observer.Monitor(out,proc,config,auth,selected,p['owner_note'],p['baseline'],
            ('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG','LD_LIBRARY_PATH'))
        monitor.start()
        try:
            code = base.wait_observed_test(proc,monitor,1600 if gate.reference else 120)
        finally:
            resources = monitor.finish()
            ct.write(out/'resources.json',resources)
    text = (out/'test.log').read_text()
    one = code == 0 and 'running 1 test' in text and '1 passed; 0 failed; 0 ignored' in text
    ct.write(out/'test-exit.json',dict(returncode=code,one_test_passed=one,selector=gate.selector,
        binary_sha256=p['binary_sha256'],log_sha256=ct.sha(out/'test.log')))
    ct.check_pins(p['sources']);ct.check_pins(p['pins'])
    if one and gate.reference:
        ct.write(out/'comparison.json',compare_unobserved(out,gate))
    return 0 if one and resources['status'] == 'observed_within_targets' else 1


def run_selected(identity,name,granted,note,route_batch=False):
    ct.require(granted and bool(note), 'Fresh explicit sole GPU owner grant required')
    root = base.run_directory(identity)
    gate = next(g for g in selected_gates(route_batch) if g.name == name)
    prepared_path = root/('e2e-prepared.json' if gate.reference else 'math-prepared.json')
    prepared = ct.load(prepared_path)
    ct.require(prepared.get('route_batch',False) == route_batch, 'Wrong sealed qualification profile')
    ct.check_pins(prepared['pins']);ct.check_pins(prepared['sources'])
    ct.validate_inventory(ct.EXCLUSIVE,prepared['tools'])
    ct.require(prepared['sources'] == ct.source_pins(), 'Fresh build source set drift')
    if gate.reference:
        for small in selected_gates(route_batch):
            if small.reference is None:
                ct.require(ct.load(root/small.name/'result.json')['passed'] is True, 'Do not run E2E after a small/numeric failure')
    out = root/name
    out.mkdir(exist_ok=False)
    binary = prepared['binaries'][gate.binary]
    selected = base.UUIDS[:gate.devices]
    p = dict(directory=str(out),gate=name,binary=binary['path'],binary_sha256=binary['sha256'],
        sources=prepared['sources'],pins=dict(prepared['pins'],**{str(prepared_path):ct.sha(prepared_path)}),
        selected=selected,owner_note=note,route_batch=route_batch)
    with base.contextlib.ExitStack() as stack:
        p['locks'] = {u:stack.enter_context(ct.exclusive_owner_lock(u)) for u in sorted(selected)}
        before = base.idle(note)
        ct.write(out/'before.json',before)
        p['baseline'] = before['baseline']
        p,pin = base.guard.seal(p,out)
        env = dict(PATH='/usr/bin:/bin',HOME=str(Path.home()),CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='none',
            FERRULE_PAPER_WORKER_TOKEN=p['worker_token'],PYTHONDONTWRITEBYTECODE='1')
        result = base.supervisor.run_process(ct.isolated_argv([sys.executable,'-S','-B',str(Path(__file__).resolve()),
            'selected-worker','--pin',pin],env),1700 if gate.reference else 180,p)
        ct.write(out/'supervisor.json',result)
        ct.require(ct.drained(result), 'GPU custody unknown; do not release/continue')
        after = base.idle(note)
        ct.write(out/'after.json',after)
        passed = ct.successful_process(result)
        ct.write(out/'result.json',dict(passed=passed,gate=asdict(gate),proven_drained=True,
            all_eight_idle=True,performance_claim=False))
    ct.write(out/'release.json',dict(gpu_owner_released=True,local_locks_released=True,all_eight_idle=True))
    print(json.dumps(ct.load(out/'result.json')),flush=True)
    print('\n'.join((out/'test.log').read_text().splitlines()[-90:]),flush=True)
    ct.require(passed,'Exact gate failed; no threshold changes, E2E blocked')

ROUTE_MANIFEST = ROOT/'crates/ferrule-model/tests/support/mixed_route_batch.sha256'
ROUTE_MANIFEST_SHA = 'e35bf8c3641048454fbe8a0c024285b3cac4daee700f90e0dc436c3fcae0570c'
ROUTE_ARCHIVE = ROOT/'target/validation/route-batch-incremental-20261012-01/mixed_route_batch_gpu.before.rs'
FUSED_ARTIFACT = ROOT/'target/validation/pair-swiglu-fused-release-20261011-01/release-artifact.json'


def route_batch_freeze():
    ct.require(ct.sha(ROUTE_MANIFEST) == ROUTE_MANIFEST_SHA, 'Original route-batch freeze changed')
    for line in ROUTE_MANIFEST.read_text().splitlines():
        pin,path = line.split(maxsplit=1)
        actual = ROUTE_ARCHIVE if path == 'crates/ferrule-model/tests/unit/transformer/mixed_route_batch_gpu.rs' else ROOT/path
        ct.require(ct.sha(actual) == pin, 'Route-batch original source/archive changed: '+path)


def route_batch_gates():
    backend = 'cuda::operators::standard_bf16::tests::'
    model = MODEL+'route_batch::'
    rows = [Gate(name,'backend',backend+selector,'route_batch_backend_words_counts_custody',1) for name,selector in (
        ('backend-route-exact','packed_routes_gpu_old_expert_gathers_and_combine_match_batch'),
        ('backend-route-alias-h2d-fault','packed_routes_gpu_reject_invalid_owners_aliases_and_after_h2d_fault'),
        ('backend-route-finite-bits','packed_routes_gpu_finite_bits_and_preflight_no_work'))]
    rows += [Gate(name,'model',model+selector,'actual_route_batch_model_synthetic_partials_not_hf_R64',devices) for name,selector,devices in (
        ('model-route-partials','raw_partial_legacy_vs_batch_exact_after_source_retirement',2),
        ('model-route-after-gather-unknown','unknown_after_gather_retains_both_roots_and_never_replays',1),
        ('model-route-observer-input','observer_input_views_use_the_single_gather_root',1),
        ('model-route-late-consumer-unknown','late_consumer_unknown_retains_metadata_gather_and_image',1),
        ('model-route-foreign-source-preflight','foreign_model_sources_reject_before_payload_cuda_work',1))]
    rows += [Gate('fused-cache-full40','composition','decoded_cache::full40_cache_unobserved_fused_logits_two_cases_two_phases',
        'unobserved_fused_logits_publication',2,'full40',3<<30,False)]
    rows += [Gate('fused-cache-'+name,'composition','lifecycle::full40_decoded_cache_unobserved_fused_lifecycle',
        'unobserved_fused_logits_publication',2,name,3<<30,False) for name in base.RECIPES]
    return rows


def main():
    ct.require_cpu_python()
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=('plan','prepare','validate','prepare-math','prepare-e2e','run-selected','selected-worker'))
    ap.add_argument('--id')
    ap.add_argument('--route-batch',action='store_true')
    ap.add_argument('--gate')
    ap.add_argument('--pin')
    ap.add_argument('--sole-gpu-owner-granted',action='store_true')
    ap.add_argument('--owner-note')
    args=ap.parse_args()
    if args.mode=='prepare-math':prepare_selected(args.id,route_batch=args.route_batch);return
    if args.mode=='prepare-e2e':prepare_selected(args.id,True,args.route_batch);return
    if args.mode=='run-selected':run_selected(args.id,args.gate,args.sole_gpu_owner_granted,args.owner_note,args.route_batch);return
    if args.mode=='selected-worker':sys.exit(selected_worker(args.pin))
    if args.mode=='plan':print(json.dumps(plan(),indent=2));return
    if args.mode=='prepare':prepare(args.id);return
    out=base.run_directory(args.id);p=ct.load(out/'prepared.json');freeze()
    ct.check_pins(p['pins']);ct.check_pins(p['sources']);ct.validate_inventory(ct.EXCLUSIVE,p['tools'])
    for row in p['binaries'].values():ct.validate_staged_binary(row['path'],row['sha256'])
    ct.require(p['sources']==ct.source_pins(),'Source inventory drift')
    print('CPU preparation valid; GPU blocked: independent fixture/consumer and fresh authorization required')

if __name__=='__main__':main()
