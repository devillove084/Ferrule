"""Opt-in official Nsight 2025.3 tools, separate from release qualification.

This is a receipt/flag adapter, not a server launcher. Current-source admission
remains required with TOOL_CONFIG. CONFIG alone selects the historical frozen
4d40 artifact policy; that artifact is not rebuilt.
"""
from pathlib import Path
import pre_serving as pre

ct = pre.ct
ROOT = ct.ROOT
ADMISSION = pre.HERE/'admission/optimized-20261009/profile-admission.json'
ADMISSION_SHA = 'ccde84e24fcfe37630364d192dd2ed68748142be951f5729c69699460eecccf0'
BINARY_SHA = '4d40c34738d36a1c78f62c0e400dd732da40e34c316ce4bd3c356dc4d29b3173'
BINARY = ROOT/'target/validation/tp2-mixed-optimized-release-20261009/ferrule-optimized'
TOOL = ROOT/'target/vendor/nsight-systems-cli-2025.3.1/opt/nvidia/nsight-systems-cli/2025.3.1'
CONFIG = dict(release='2025.3.1.90-253135822126v0', policy='frozen_qualified_4d40_no_rebuild',
              binary_sha256=BINARY_SHA, admission_sha256=ADMISSION_SHA)
TOOL_CONFIG = dict(release=CONFIG['release'], policy='current_release_same_binary_smoke')
TOOL_PINS = {
    str(TOOL/'target-linux-x64/nsys'): '525f35a10de49b396e91a3831ae40e98a8631d619b7c8bbb4c54469b8c08a72e',
    str(TOOL/'host-linux-x64/QdstrmImporter'): '23e4185b7b20d00c27dc2e521c1048a75abe04f04dbbd88e411b6dfba2cc8053',
    str(TOOL/'target-linux-x64/libcupti.so.12.9'): '19f0f06e0c5579113840169d5f6d6d4222d4a4fc2d66403cbe3cad2d91db1ace'}
DRIFT_PATHS = {str(ROOT/'crates/ferrule-model/src/transformer/standard'/p) for p in
               ('cuda.rs','cuda/bf16_hybrid.rs','cuda/hybrid_rank/mixed.rs','cuda/hybrid_rank/mixed_layer.rs')}


def is_frozen(admission):
    return admission.get('profiler') == CONFIG


def configure(config, module=pre):
    if config is None:
        module.NSYS = '/opt/nvidia/nsight-systems/2026.1.3/target-linux-x64/nsys'
        module.PROFILE_TRACE = 'cuda-sw,nvtx,osrt'
        module.PROFILE_SCOPE_FLAGS = (*module.instrumentation_scope.SCOPE_FLAGS, '--cuda-trace-all-apis=true')
        return
    ct.require(config in (CONFIG, TOOL_CONFIG), 'Unreviewed profiler/fixed-binary policy')
    module.NSYS = str(TOOL/'target-linux-x64/nsys')
    module.PROFILE_TRACE = 'cuda,nvtx,osrt,cublas'
    module.PROFILE_SCOPE_FLAGS = ('--gpu-metrics-devices=none','--gpuctxsw=false')


def check_fixed_shape(admission, purpose, historical):
    ct.require(purpose == 'profiling', 'Frozen binary policy is profiling-only, never benchmark/build admission')
    ct.require(admission == dict(historical, profiler=CONFIG), 'Fixed admission/spec/evidence altered')
    ct.require(historical['engines']['ferrule']['binary']['sha256'] == BINARY_SHA, 'Wrong serving binary')


def fixed_binary_audit(admission):
    ct.require(ct.sha(ADMISSION) == ADMISSION_SHA, 'Historical qualified admission changed')
    historical = ct.load(ADMISSION)
    check_fixed_shape(admission, 'profiling', historical)
    m = historical['engines']['ferrule']
    ct.check_pins({str(BINARY): BINARY_SHA, **TOOL_PINS, **m['runtime_pins']})
    ct.validate_staged_binary(m['binary']['path'], BINARY_SHA)
    build = pre.checked_document(m['build_receipt'])
    quality = pre.checked_document(m['test_quality'])
    smoke = pre.checked_document(m['serving_smoke'])
    ct.require(build['artifact']['sha256'] == BINARY_SHA and build['release_state']['status'] == 'complete',
               'Historical build artifact not complete')
    ct.require(quality['status'] == 'pass' and quality['calls'] == 37 and quality['metrics'] == 40293,
               'Historical exact numerical qualification missing')
    ct.require(pre.numerical_sources(m['source_pins']) == quality['numerical_source_pins'],
               'Historical quality/build source relation differs')
    sources = m['source_pins']
    # Source paths describe the OLD binary. Check all other historical evidence
    # normally; never overwrite old hashes with the working tree's new content.
    for pins in (m['build_evidence_pins'], quality['artifact_pins']):
        ct.check_pins({p:h for p,h in pins.items() if p not in sources})
    pre.validate_smoke(m)
    ct.require(smoke['binary_sha256'] == BINARY_SHA, 'Smoke is for a different binary')
    current = ct.inventory([Path(p) for p in sources])
    drift = {p:dict(build_sha256=h,current_sha256=current[p]) for p,h in sources.items() if current[p] != h}
    ct.require(set(drift) <= DRIFT_PATHS, 'Additional unreviewed source drift')
    return dict(policy=CONFIG['policy'], historical_admission_sha256=ADMISSION_SHA,
        binary_path=str(BINARY), staged_identical_path=m['binary']['path'], binary_sha256=BINARY_SHA,
        build_source_pins=sources, current_source_drift=drift,
        current_tree_qualified=False, rebuild=False,
        runtime_pins=m['runtime_pins'], tool_pins=TOOL_PINS,
        build_receipt=m['build_receipt'], quality_receipt=m['test_quality'], smoke_receipt=m['serving_smoke'],
        qualification='Prior exact quality and two serving lifecycles belong to this immutable binary, not current edited sources',
        logging_override='Only FERRULE_LOG=info,ferrule_request=debug; same setting already used by serving smoke')


def validate_fixed_admission(admission, purpose):
    import paired
    ct.require(ct.sha(ADMISSION) == ADMISSION_SHA, 'Historical admission hash differs')
    check_fixed_shape(admission, purpose, ct.load(ADMISSION))
    m = admission['engines']['ferrule']
    ct.require(m['spec'] == paired.ferrule_spec(m['binary']['path']), 'Fixed argv/environment changed')
    ct.require(m['model'] == paired.model_contract(), 'Checkpoint/tokenizer identity changed')
    ct.require(all(m['source_pins'].get(p) == h for p,h in paired.source_audit()['source_pins'].items()),
               'Public quality constant/CLI admission sources changed')
    fixed_binary_audit(admission)
    return admission
