"""Offline report for the bounded pilot. No lock, GPU query or server launch."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paired as p

D = p.pair_path('tp2-fv-qa8-20261009-001')
S = p.pair_path('tp2-serving-smoke-20261009-003')
prepared = p.validate_pair(D)
p.validate_pair(S)
p.ct.require(p.ct.load(D/'execution.json')['status'] == 'complete_descriptive_only', 'Pair incomplete')
smoke = p.pre_serving.validate_smoke(prepared['admission']['engines']['ferrule'])
p.ct.require(p.ct.load(D/'final-owner-release.json')['all_eight_idle'] is True, 'Final idle missing')
rows = []
row_keys = ('prompt_id','population','status','actual_input_tokens','actual_output_tokens',
    'text','finish_reason','ttft_seconds','request_e2e_seconds','output_token_ids')
for slot in p.SLOTS:
    directory = D/'runs'/slot
    assessed = p.assess_slot(directory, p.slot_engine(slot))
    case = p.ct.load(directory/'session/case.json')
    resources = case['resources']
    phases = {}
    for phase in ('hello2','canary8','qa8-V1'):
        report = p.ct.load(directory/'session'/phase/'http.json')
        gate = p.ct.load(directory/'session'/phase/'gate.json')
        p.ct.require(gate['status'] == 'pass' and report['status'] == 'complete', 'Request phase failed')
        measured = [row for run in report['runs'] for row in run['requests']]
        populations = dict(first=[report['first_after_ready']] if 'first_after_ready' in report else [],
                           warmups=report['warmups'],measured=measured)
        phases[phase] = dict(gate=gate['status'],summaries=[run['summary'] for run in report['runs']],
            **{name:[{key:row.get(key) for key in row_keys} for row in values] for name,values in populations.items()})
    log_path = directory/'session/server.log'
    log = log_path.read_text(errors='replace')
    if slot.startswith('F'):
        trace_lines = [line for line in log.splitlines() if 'Mixed optional diagnostic reserves' in line]
        p.ct.require(len(trace_lines) == 1 and json.loads(trace_lines[0])['fields']['trace_host'] == 0
            and json.loads(trace_lines[0])['fields']['trace_device_per_owner'] == 0
            and 'request_worker_terminal' not in log, 'Paired Ferrule trace/debug unexpectedly active')
        p.ct.require('max_decode_batch: 1' in log and 'allow_mixed_batches: false' in log, 'Scheduler evidence missing')
        runtime = dict(profile=p.PROFILE,trace_disabled=True,request_debug_disabled=True,
            scheduler='effective decode batch 1, mixed batches false',
            collective='thread TP2 observed; CPU/host collective established by source, not per-op trace',
            operation_evidence='profile/precision/provider/resource logs; full per-kernel arithmetic closure not instrumented')
    else:
        for literal in ('MarlinFP8ScaledMMLinearKernel','Using MARLIN Fp8 MoE','Using Triton/FLA GDN prefill',
                'Falling back to the Triton GDN decode path','NCCL_P2P_DISABLE set by environment to 1',
                'via SHM/direct/direct','collect_detailed_traces=None','enable_layerwise_nvtx_tracing=False'):
            p.ct.require(literal in log, 'vLLM runtime marker missing: '+literal)
        runtime = dict(profile=p.shared.config()['profile'],trace_disabled=True,
            kernels='Marlin FP8 linear/MoE; Triton/FLA GDN prefill; Triton GDN decode fallback',
            transport='NCCL 2.29.7+CUDA12.9 SHM/direct/direct, NCCL_P2P_DISABLE=1',
            arithmetic_equivalence=False)
    library_paths = sorted({lib for process in resources['final_process_diagnostics'] for lib in process['libraries']})
    runtime.update(log_path=str(log_path),log_sha256=p.ct.sha(log_path),observed_library_paths=library_paths,
        library_mapping_caveat='filtered snapshots, not exhaustive loader closure; vLLM also maps CUDA13.2 runtime/NVRTC alongside cu129')
    summary = phases['qa8-V1']['summaries'][0]
    p.ct.require(summary['success'] == 5 and summary['successful_output_tokens'] == 40
        and len(phases['qa8-V1']['first']) == 1 and len(phases['qa8-V1']['warmups']) == 2, 'Wrong QA scope')
    selected = {key:value for key,value in resources.items() if key != 'final_process_diagnostics'}
    rows.append(dict(slot=slot,engine=p.slot_engine(slot),launch_id=assessed['launch_id'],
        server_identity=case['server_identity'],startup_seconds=case['startup_seconds'],
        worker_wall_seconds=case['wall_seconds'],shutdown=case['shutdown'],resources=selected,
        runtime=runtime,phases=phases,measured_summary=summary,
        custody='authenticated subreaper ECHILD/drain + separate pre/post idle and UUID release verified'))
# Revalidate old run files, not old mutable workflow/source globs.
old = p.ct.load(p.HERE.parent/'runs/tp2-resident-single-lifecycle-20261009-001/prepared.json')
p.ct.check_pins(old['historical_pins'])
pins = p.ct.inventory([file for directory in (D,S) for file in directory.rglob('*') if file.is_file()])
pins.update(p.ct.inventory([Path(__file__), p.HERE/'admission/review_pair.py', p.HERE/'model-runtime-quality.json']))
result = dict(schema='ferrule.paper.tp2-serving-pilot.v1',status='complete_descriptive_only',
    run=D.name,smoke_run=S.name,protocol=p.PROTOCOL,order=list(p.SLOTS),
    release_binary_sha256=prepared['admission']['engines']['ferrule']['binary']['sha256'],
    common_contract=p.contract.common_contract(),scope=p.contract.SCOPE,
    rows=rows,smoke=dict(status=smoke['status'],lifecycles=smoke['lifecycles'],
        policy='diagnostic-only; excluded entirely from performance; normal shutdown/cancel/reuse/fresh CPU owner'),
    prepared_not_run=['tp2-serving-smoke-20261009-001','tp2-serving-smoke-20261009-002'],
    denominators=dict(paired_servers_planned=4,paired_servers_started=4,paired_servers_success=4,
        paired_http_planned=48,paired_http_success=48,qa_measured_planned=20,qa_measured_success=20,
        qa_actual_output_tokens=160,paired_failed=0,paired_unknown=0,paired_not_run=0,
        smoke_servers_started=2,smoke_normal_http_success=20,smoke_intentional_cancellations_verified=2),
    fairmatch=False,exact_speedup=None,arithmetic_equivalence=False,e2_complete=False,
    backend_differences=prepared['admission']['engines']['ferrule']['backend_differences'],
    limitations=['Only two balanced order blocks, five measured requests per fresh server; no CI/statistical claim',
        'No longer outputs/load extrapolation; no optimal tuning claim; not arithmetic-equivalent',
        'Visible text and usage only; HTTP token IDs/logits/routes/state not observed',
        '22 GiB/card and 128 GiB PSS are sampled ceilings, not hard partitions; peaks not exact',
        'OS/JIT/page caches uncontrolled, fresh process is not storage cold',
        'Whole-device NVML PID mapping unavailable: owned GPU attribution unknown; no foreign signal',
        'Catalog build has source/log hashes but no authenticated build supervisor; runtime custody authenticated'],
    quality=dict(full40_fields_each=2178,lifecycle_calls=33,lifecycle_metrics=35937,
                 old_greedy_failure_preserved=True,fresh_greedy4_02_closed=True),
    validation=dict(original_tests=42,tp2_tests=78,historical_run_files_rehashed=len(old['historical_pins'])),
    final_release=p.ct.load(D/'final-owner-release.json'),artifact_pins=pins)
p.ct.write(D/'serving-pilot-summary.json',result)
for row in rows:
    s=row['measured_summary'];dist=s['distributions'];r=row['resources']
    print(row['slot'],f"startup={row['startup_seconds']:.3f}s wall={s['wall_seconds']:.6f}s tok/s={s['successful_output_tokens_per_second']:.4f}",
          f"TTFT={dist['success_ttft_seconds']['mean']*1000:.3f}ms E2E={dist['success_request_e2e_seconds']['mean']*1000:.3f}ms",
          'GPU GiB', [round(r['gpu_peak_bytes'][u]/2**30,4) for u in p.shared.UUIDS], 'PSS GiB',round(r['tree_pss_peak_bytes']/2**30,4))
print('Summary and raw receipts pinned:',D/'serving-pilot-summary.json')
