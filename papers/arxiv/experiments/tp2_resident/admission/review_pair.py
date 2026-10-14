"""CPU-only receipt projection for this authorized pilot; never launch a server."""
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paired as p

ad = Path(__file__).resolve().parent
smoke_dir = p.pair_path('tp2-serving-smoke-20261009-003')
smoke_path = smoke_dir/'serving-smoke.json'
F = copy.deepcopy(p.ct.load(ad/'tp2-serving-smoke-admission.json')['engines']['ferrule'])
F['serving_smoke'] = dict(path=str(smoke_path), sha256=p.ct.sha(smoke_path))
p.pre_serving.validate_smoke(F)
F['arithmetic'].update(
    scales='FP8 E4M3 source / BF16 scales; source-bound BF16 compatibility profile',
    dequant_rounding='BF16 boundary qualification from test binary, not HTTP numerical observation',
    reduction='source-bound F32 accumulation; no per-operation runtime tracing',
    gdn='source-bound Mixed implementation / F32 SSM; not Marlin or Triton GDN',
    effective_evidence='S1/S2 profile + precision logs, completed serving canaries; full operation closure unobserved')
F['transport'].update(
    observed_channels='Thread TP2 observed; CPU/host collective source-bound; per-collective channel tracing unavailable, not NCCL',
    evidence_sha256=p.ct.sha(smoke_dir/'runs/S1/session/server.log'))
F['backend_differences'] = dict(
    scheduler='actual requested decode batch 2 lowers to 1; mixed batching disabled; c1 workload only',
    transport='CPU collective vs vLLM NCCL SHM; CPU collectives are not model CPU offload',
    arithmetic='HF-eager TP2 reference is not Marlin arithmetic',
    trace='numerical/state trace reserves zero; smoke request DEBUG only, paired FERRULE_LOG=info',
    cache='Ferrule KV 16 MiB / workspace 1024 MiB / max tensor 2048 MiB; not vLLM KV allocation parity')
old_dir = p.HERE.parent/'runs/tp2-resident-single-lifecycle-20261009-001'
old = p.ct.load(old_dir/'prepared.json')
r = p.ct.load(old_dir/'execution.json')
p.ct.require(r['status'] == 'complete' and r['authenticated_drained'] is True
    and r['supervisor_success'] is True and r['after']['status'] == 'verified'
    and p.ct.successful_process(p.ct.load(old_dir/'session/supervisor.json'))
    and p.ct.load(old_dir/'lock-release.json')['released'] is True, 'Historical baseline custody not proven')
p.ct.require(not r['runtime']['contradictions'], 'Historical baseline runtime contradicted')
config = p.shared.config()
p.ct.require(config == old['config'], 'Baseline config changed')
p.ct.check_pins(old['runtime_pins'])
p.ct.require(p.ct.inventory(p.shared.runtime_paths(config)) == old['runtime_pins'], 'Runtime inventory changed')
cli_audit = p.shared.audit(config)
spec = p.shared.spec_for(config, old['spec']['argv'][0])
p.ct.require(spec == old['spec'] and p.model_contract()['checkpoint_identity'] ==
    {k:old['model_identity'][k] for k in ('path','metadata_sha256','shard_stats')}, 'Baseline spec/model changed')
refs, ref_pins = p.shared.references()
canary = p.ct.load(old_dir/'session/canary8/http.json')
rows = [row for run in canary['runs'] for row in run['requests']]
p.ct.require(len(rows) == 3, 'Canary scope changed')
for row, ref in zip(rows, refs):
    p.ct.require(row['status'] == 'success' and row['text'] == ref['reference_text']
        and row['actual_input_tokens'] == ref['input_tokens'] and row['actual_output_tokens'] == 8,
        'Historical visible canary differs')
for stage in ('hello2','canary8','qa8-V1'):
    p.ct.require(p.ct.load(old_dir/'session'/stage/'gate.json')['status'] == 'pass', 'Historical quality failed')
source_pins = {name:h for name,h in old['runtime_pins'].items() if
    ('/site-packages/vllm/' in name and Path(name).suffix == '.py') or
    (Path(name).name == 'METADATA' and any(x in name for x in ('vllm-0.28.0','torch-2.13.0')))}
p.ct.require(source_pins, 'Installed vLLM sources missing')
base = [old_dir/name for name in ('prepared.json','execution.json','lock-release.json',
    'session/supervisor.json','session/server.log','session/before.json','session/after.json')]
quality_paths = base + [old_dir/'session'/stage/name for stage in ('hello2','canary8','qa8-V1')
                        for name in ('gate.json','http.json')]
V = dict(p.contract.common_contract(),engine='vllm',origin='fresh_preparation',
    binary=dict(path=spec['argv'][0],sha256=old['executable_sha256'],kind='public_serving'),
    source_pins=source_pins,runtime_pins=old['runtime_pins'],environment_policy='isolated_allowlist',
    spec=spec,model=p.model_contract(),cache=copy.deepcopy(F['cache']),custody_policy=copy.deepcopy(F['custody_policy']),
    arithmetic=dict(profile=config['profile'],scales='FP8 checkpoint scales / Marlin backend',
        dequant_rounding='Marlin internal rounding not instrumented; no arithmetic parity assertion',
        reduction='TP2 NCCL plus Marlin kernels; exact operation arithmetic closure unobserved',
        gdn='logged Triton/FLA prefill and Triton decode fallback; not Ferrule Mixed',
        effective_evidence='historical exact-runtime logs; fresh V1/V2 must independently pass logged-operation checks'),
    transport=dict(declared_environment=config['transport'],observed_channels='NCCL 2.29.7+cu129 SHM/direct/direct, P2P disabled',
        evidence_sha256=p.ct.sha(old_dir/'session/server.log')),
    backend_differences=copy.deepcopy(F['backend_differences']),gates={})
for gate in p.required_gates('vllm'):
    path = ad/('vllm-'+gate+'.json')
    evidence = p.ct.inventory(quality_paths if gate == 'serving_quality' else base)
    evidence.update(ref_pins)
    if gate == 'public_serving_cli':
        evidence.update(source_pins)
        evidence.update(p.ct.inventory([p.HERE/'experiment_v4.py',p.ct.absolute(config['vllm_fix_environment'])]))
    receipt = dict(schema='ferrule.paper.tp2-serving-admission.v1',engine='vllm',gate=gate,status='pass',
        binary_sha256=V['binary']['sha256'],source_pins=source_pins,spec_digest=p.ct.digest(spec),evidence_pins=evidence,
        origin='reviewed existing authenticated baseline; NOT fresh paired timing',
        audit_source_sha256=p.ct.sha(__file__),static_cli_audit=cli_audit if gate=='public_serving_cli' else None,
        limits='visible text/usage and authenticated lifecycle only; no full Marlin numerical oracle or historical cancellation claim')
    p.ct.write(path,receipt)
    V['gates'][gate] = dict(status='pass',receipt_path=str(path),receipt_sha256=p.ct.sha(path),
        binary_sha256=V['binary']['sha256'],source_pins=source_pins)
pair = dict(protocol=p.PROTOCOL,engines=dict(ferrule=F,vllm=V))
p.validate_admission(pair,'paired')
path = ad/'tp2-paired-admission.json'
p.ct.write(path,pair)
print('Verified paired admission:', path)
print('Fresh order F1,V1,V2,F2; trace disabled; no fairmatch/speedup; smoke excluded from performance')
