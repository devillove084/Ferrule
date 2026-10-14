"""CPU-only preflight for future TP2 serving; NOT launch authorization.

Reuse the maintained baseline's config, frozen references and HTTP schema.
No executable/CLI probing, lock acquisition, GPU query or HTTP call occurs here.
Declarations passing this check still need pinned receipt verification and a
reviewed Ferrule adapter in the authenticated supervisor; execution stays closed.
"""
from copy import deepcopy
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment_current as baseline

shared = baseline.shared
REFERENCE_SHA256 = '102c798baeb5200bc8ed8386eabed6146870b3b6da2a90fc6010677041352c7e'
SCOPE = 'boundary-aligned system-level deployment comparison; not arithmetic-equivalent'
SHA256 = re.compile(r'[0-9a-f]{64}')
REQUIRED_GATES = ('public_serving_cli', 'all_boundary', 'multistep', 'service_lifecycle', 'serving_quality')


def common_contract():
    c = shared.config()
    protocol = c['protocol']
    return dict(
        topology=dict(tp=2, pp=1, dp=1, ep=1, enable_expert_parallel=False,
                      gpu_uuids=list(shared.UUIDS)),
        boundaries={k: protocol[k] for k in ('activation', 'kv', 'conv', 'ssm')},
        workload=dict(context=protocol['context'], batch_tokens=protocol['batch_tokens'],
            active_sequences=protocol['active_sequences'], concurrency=protocol['concurrency'],
            seed=protocol['seed'], eos=protocol['eos'], prefix_cache=protocol['prefix_cache'],
            endpoint=protocol['endpoint'], eager=True, cpu_offload_gb=0,
            reference_manifest_sha256=REFERENCE_SHA256, qa=deepcopy(c['workloads']['qa8'])),
        resources=dict(gpu_target_bytes=c['resources']['gpu_target_bytes'],
            host_target_bytes=c['resources']['host_target_bytes'], hard_partition=False,
            policy='sampled whole-device GPU and server-tree PSS; missing is null, not zero'),
        limits={k: baseline.PROTOCOL[k] for k in ('startup_seconds', 'request_seconds',
            'all_requests_seconds', 'teardown_seconds', 'supervisor_seconds', 'supervisor_ceiling_seconds')})


def _sha(value):
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def _pins(value):
    return isinstance(value, dict) and bool(value) and all(
        isinstance(path, str) and path and _sha(digest) for path, digest in value.items())


def _known(value):
    return baseline.http.known(value)


def _engine_blockers(engine, metadata, common):
    """Check declarations, never infer authenticity from a caller's pass flag."""
    reasons = []
    if metadata.get('engine') != engine:
        reasons.append('engine identity missing/different')
    if metadata.get('origin') != 'fresh_preparation':
        reasons.append('fresh preparation required; archived timings cannot fill a pair')
    for key, expected in common.items():
        if metadata.get(key) != expected:
            reasons.append(key + ' missing/different from common contract')
    for key in ('source_pins', 'runtime_pins'):
        if not _pins(metadata.get(key)):
            reasons.append(key + ' missing/malformed')
    binary = metadata.get('binary') or {}
    if not binary.get('path') or not _sha(binary.get('sha256')) or binary.get('kind') != 'public_serving':
        reasons.append('pinned public serving binary required; staged test binary is insufficient')
    spec = metadata.get('spec') or {}
    env = spec.get('environment')
    if (spec.get('engine') != engine or not isinstance(spec.get('argv'), list)
            or not spec['argv'] or spec['argv'][0] != binary.get('path')):
        reasons.append('engine-specific documented argv/binary binding missing')
    if (not isinstance(env, dict) or not env
            or env.get('CUDA_VISIBLE_DEVICES') != ','.join(shared.UUIDS)
            or metadata.get('environment_policy') != 'isolated_allowlist'):
        reasons.append('isolated engine-specific environment/UUID binding missing')
    if engine == 'vllm' and isinstance(env, dict) and any(
            key in env for key in ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'CUBLAS_WORKSPACE_CONFIG')):
        reasons.append('Ferrule CUDA environment must not leak into vLLM')
    for group, fields in dict(
            model=('checkpoint_identity', 'identity_strength', 'tokenizer_identity'),
            arithmetic=('profile', 'scales', 'dequant_rounding', 'reduction', 'gdn', 'effective_evidence'),
            transport=('declared_environment', 'observed_channels', 'evidence_sha256'),
            cache=('state_reset', 'os_jit_page_cache_policy')).items():
        value = metadata.get(group) or {}
        if not all(key in value and _known(value[key]) for key in fields):
            reasons.append(group + ' unknown/incomplete')
    if (metadata.get('cache') or {}).get('state_reset') != 'per_request':
        reasons.append('per-request state reset missing')
    custody = metadata.get('custody_policy') or {}
    expected = dict(identity_mode='device_exclusive_local_custody',
        sorted_physical_uuid_locks=True, atomic_release_on_failure=True,
        supervisor='authenticated_linux_subreaper', root_identity='pid_starttime',
        pre_post_idle=True, nonselected_baseline_delta_bytes=1 << 20,
        foreign_signals=False, owned_gpu_attribution='unknown',
        stop_on_unknown_drain=True, sole_gpu_owner=True)
    if custody != expected:
        reasons.append('exclusive authenticated custody policy missing/different')
    gates = metadata.get('gates') or {}
    for name in REQUIRED_GATES:
        gate = gates.get(name) or {}
        if (gate.get('status') != 'pass' or not _sha(gate.get('receipt_sha256'))
                or gate.get('binary_sha256') != binary.get('sha256')
                or gate.get('source_pins') != metadata.get('source_pins')):
            reasons.append(name + ' missing/failed or not bound to this serving binary/source')
    return [engine + ': ' + reason for reason in reasons]


def review_pair(ferrule=None, vllm=None):
    """Deterministic metadata review only; even complete declarations cannot run."""
    engines = dict(ferrule=deepcopy(ferrule or {}), vllm=deepcopy(vllm or {}))
    common = common_contract()
    reasons = [reason for engine, value in engines.items()
               for reason in _engine_blockers(engine, value, common)]
    for group in ('model', 'cache'):
        a, b = (value.get(group) for value in engines.values())
        if not _known(a) or not _known(b) or a != b:
            reasons.append('paired ' + group + ' differs/unknown')
    return dict(status='blocked', execute_allowed=False, fairmatch=False, exact_speedup=None,
        scope=SCOPE, arithmetic_equivalence=False,
        metadata_complete=not reasons, metadata_blockers=sorted(set(reasons)),
        execution_blockers=['Public Ferrule TP2 serving CLI/profile not admitted',
            'Pinned gate receipts require independent verification, not caller pass flags',
            'Engine-neutral adapter must be wired and tested inside existing authenticated supervisor'],
        common=common, engines=engines,
        proposed_order=[list(block) for block in shared.ORDER_BLOCKS],
        schedule_policy='sequential fresh servers; one owner; stop on quality/failure/unknown drain; never reuse archived V timings',
        request_policy='Hello2 then 3 frozen canaries in each server; only on pass QA8 first QA-phase request +2 warmups +5 measured',
        denominator_policy='retain planned/dispatched/terminal/success/quality-fail/OOM/timeout/cancel/unknown/not_run; missing samples null')


def http_plan(engine, spec, stage, directory):
    """Build arguments for the EXISTING HTTP helper, not a server CLI or request.

    The helper already supports ferrule/vllm. Unlike shared.request_block's
    historical vLLM-only wrapper, engine identity is explicit and checked here.
    The caller must persist corpus/metadata, then use the existing HTTP helper
    only inside the admitted worker, after readiness and phase gates.
    """
    if engine not in ('ferrule', 'vllm') or spec.get('engine') != engine:
        raise ValueError('Explicit matching HTTP engine required')
    endpoint = urlsplit(spec.get('url', ''))
    if (endpoint.scheme != 'http' or endpoint.hostname != '127.0.0.1'
            or endpoint.path != '/v1/completions' or endpoint.username or endpoint.password
            or endpoint.query or endpoint.fragment or not endpoint.port or not spec.get('model')):
        raise ValueError('Local raw-completions endpoint and exact model ID required')
    if stage not in ('hello2', 'canary8', 'qa8'):
        raise ValueError('Unknown request phase')
    refs, pins = shared.references()
    if REFERENCE_SHA256 not in pins.values():
        raise ValueError('Frozen original HF manifest changed')
    workload = shared.config()['workloads'][stage]
    directory = Path(directory)
    args = ['--engine', engine, '--url', spec['url'], '--model', spec['model'],
        '--seed', '1', '--max-tokens', str(workload['max_tokens']), '--concurrency', '1',
        '--requests', str(workload['requests']), '--warmup', str(workload['warmup']),
        '--repeat', '1', '--timeout', '120', '--window-kind', 'fixed-work',
        '--measurement-seconds', '600', '--metadata', str(directory/'metadata.json'),
        '--output', str(directory/'http.json')]
    corpus = None
    if stage == 'hello2':
        args += ['--prompt', 'Hello', '--input-tokens', '1']
    else:
        corpus = [{k: row[k] for k in ('id', 'prompt', 'input_tokens')}
                  for row in (refs if stage == 'canary8' else refs[:1])]
        args += ['--prompts-json', str(directory/'prompts.json')]
    if stage == 'qa8':
        args += ['--first-after-ready']
    return dict(argv=args, corpus=corpus, reference_pins=pins,
        canary_expectations=deepcopy(refs) if stage == 'canary8' else None,
        first_label='excluded first QA-phase request, not first after server readiness',
        execution='plan_only; no HTTP issued', effective_token_ids='unobserved',
        outer_request_deadline_seconds=baseline.PROTOCOL['all_requests_seconds'])
