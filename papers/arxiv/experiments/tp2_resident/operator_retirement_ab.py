"""Typed allocator V3 comparison; paired.py remains the only GPU launcher.

The final AFTER release and fixed review manifest are registered.
Same-binary smoke still requires fresh authorization; no CLI can widen source exceptions.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import arena_index_ab as indexed
import cache_ab
import pre_serving as pre
import decoded_cache_release as release

ct = pre.ct
PROTOCOL = 'tp2-operator-retirement-v3-matched4g-qa8-descriptive-v1'
POLICY = 'frozen_qualified_9ef_operator_retirement_v3'
SLOTS = ('BEFORE', 'AFTER')
BEFORE_SHA = release.INDEXED_BINARY_SHA
BASELINE = pre.HERE/'admission/indexed-lru-20261010-01/paired-admission.json'
BASELINE_SHA = 'bb804493bac2d1f17a3200dd015bf453610ac99ae2997631c2c2a82ee3cd0939'
CLOSEOUT = pre.HERE/'runs/arena-index-crossed-closeout-20261010-01/owner-closeout.json'
CLOSEOUT_SHA = '8e4e8c85ce642274deedebd7208874a5d28220b7d07b1c25f46a6fd3bcc8d753'
WORKFLOW_ARCHIVE = pre.HERE/'history/pre-operator-retirement-ab-20261011/paired.py.txt'
CLOSED_WORKFLOW_SOURCES = {
    str(pre.HERE/'decoded_cache_gate.py'): '41c355ac74b4ad3b12ee083cfab9d5f2b5b07c9290d4310c3042a9b0a1fbb308',
    str(pre.HERE/'test_decoded_cache_gate.py'): '55ceca42c058e9a97cebd919492490d3525099a1bc6b7ef27ae8f3dc2a3fc854',
}
QUALIFICATION = dict(path=str(pre.HERE/'runs/decoded-allocator-v3-20261011-03/handoff.json'),
    sha256='faa71361682e8b454b2e85feffd456c06598bcc4ff4572a34bfaa54367e83fd4')
QUALIFICATION_SCHEMA = 'ferrule.paper.allocator-v3-handoff.v1'
DELTA_SCHEMA = 'ferrule.paper.operator-retirement-source-delta.v1'
REVIEW_SCHEMA = 'ferrule.paper.operator-retirement-build-review.v1'
# One reviewed path/SHA, never a caller-supplied source allowlist.
REVIEWED_BUILD = dict(path=str(pre.HERE/'plans/operator-retirement-v3-release-20261011-01/review-manifest.json'),
    sha256='5519bdbaf6089ebba5bbac9640f84875fa8e90698e53b3464916b3c0564b4b71')
PRODUCTION = tuple(str(ct.ROOT/'crates/ferrule-backend/src/cuda'/p) for p in ('allocator.rs', 'runtime.rs'))
TEST_PATCH = dict(path=str(ct.ROOT/'crates/ferrule-backend/src/cuda/providers/cutlass.rs'),
    before_sha256='cf567b42135279ea1de45fb61a7ba935ab9be03497713067d4d0c7d2d393a3b5',
    after_sha256='389ee1596b3af387fcb4782bceed8ce54d6870de56a93f4979f8df97209dbce4')


def slots(cycle):
    ct.require(type(cycle) is int and cycle in (1, 2), 'Retirement cycle must be 1 or 2')
    return SLOTS if cycle == 1 else tuple(reversed(SLOTS))


def contract(cycle):
    result = indexed.contract(cycle)
    result.update(policy=POLICY, source_delta='Exactly allocator.rs/runtime.rs plus the fixed cfg(test) manifest patch; complete inventories required',
        unknown_pressure=None, pending_peak=None,
        retirement_ledger_scope='Complete successfully created event sets only; not total CUDA events; partial-failure driver calls uncounted')
    return result


def frozen_metadata():
    return deepcopy(pre.checked_document(dict(path=str(BASELINE), sha256=BASELINE_SHA))['engines']['ferrule'])


def frozen_audit():
    """Audit immutable 9ef receipts, never substitute the current source tree."""
    m = frozen_metadata()
    ct.require(m['binary']['sha256'] == BEFORE_SHA and m['decoded_cache'] == release.CACHE,
               'BEFORE must be the qualified cache-on 9ef indexed release')
    ct.validate_staged_binary(m['binary']['path'], BEFORE_SHA)
    ct.check_pins(m['runtime_pins'])
    sources, build = indexed.production_sources(m)
    ct.require(m['build_receipt']['sha256'] == release.INDEXED_HANDOFF_SHA
        and build['artifact']['sha256'] == BEFORE_SHA, 'Frozen indexed build differs')
    ct.check_pins({build['artifact']['path']: BEFORE_SHA})
    q = pre.checked_document(m['test_quality'])
    qualification = pre.checked_document(build['qualification_handoff'])
    summary = pre.checked_document(qualification['acceptance_summary'])
    prepared = pre.checked_document(qualification['prepared'])
    ct.require(q['status'] == 'pass' and (q['calls'], q['metrics']) == (37, 40293)
        and len(q['rows']) == 23 and {r['gate']:r['selector'] for r in q['rows']} == release.gates.ALL_GATES,
        'Frozen indexed named numerical qualification differs')
    ct.require(all(prepared['pins'].get(p) == h for p,h in CLOSED_WORKFLOW_SOURCES.items()),
               'Frozen indexed workflow identities differ')
    historical = dict(summary['sources'])
    for name, digest in pre.checked_document(build['source_pins_before']).items():
        path = str((ct.ROOT/name).resolve())
        ct.require(path not in historical or historical[path] == digest, 'Conflicting historical source pin')
        historical[path] = digest
    historical.update(CLOSED_WORKFLOW_SOURCES)
    pins = {str(BASELINE): BASELINE_SHA, str(CLOSEOUT): CLOSEOUT_SHA}
    for group in (m['build_evidence_pins'], q['artifact_pins']):
        for name, digest in group.items():
            if name in historical:
                ct.require(historical[name] == digest, 'Historical source identity relabelled')
                continue
            archives = {str(pre.HERE/n):str(pre.HERE/'history/pre-operator-retirement-release-20261011'/(n+'.txt'))
                        for n in ('pre_serving.py','optimized_release.py')}
            archives[str(pre.HERE/'paired.py')] = str(WORKFLOW_ARCHIVE)
            actual = archives.get(name, name)
            ct.require(actual not in pins or pins[actual] == digest, 'Historical evidence conflict')
            pins[actual] = digest
    ct.check_pins(pins)
    smoke = pre.validate_smoke(m)
    closeout = ct.load(CLOSEOUT)
    ct.require(closeout['status'] == 'completed_descriptive_only' and closeout['smoke'] == m['serving_smoke']
        and closeout['promoted_admission'] == dict(path=str(BASELINE), sha256=BASELINE_SHA), 'Frozen indexed promotion differs')
    for cycle in closeout['cycles']:
        report = pre.checked_document(cycle['summary'])
        ct.check_pins(report['artifact_pins'])
        row = next(r for r in report['rows'] if r['slot'] == 'AFTER')
        ct.require(report['status'] == 'complete_descriptive_only' and row['binary'] == m['binary']
            and row['summary']['success'] == 5, 'Frozen indexed cycle evidence differs')
    return dict(policy=POLICY, binary_sha256=BEFORE_SHA, production_source_digest=ct.digest(sources),
        evidence_pins=pins, closed_historical_workflow_source_pins=dict(CLOSED_WORKFLOW_SOURCES),
        current_tree_qualified=False, smoke_status=smoke['status'], official_profile_available=False)


def qualification_identity():
    """Check the pinned eligibility document; this is NOT a serving release audit."""
    h = pre.checked_document(QUALIFICATION)
    summary = pre.checked_document(h['acceptance_summary'])
    ct.require(h['schema'] == QUALIFICATION_SCHEMA
        and h['status'] == 'FULL_MODEL_QUALIFICATION_PASS_RELEASE_BUILD_ELIGIBLE'
        and h['fresh_gpu_gates'] == 23 and h['supplemental_replaces_named_gates'] is False,
        'Allocator V3 eligibility differs')
    ct.require(len(h['named_gpu_gates']) == 23
        and {r['gate']:r['selector'] for r in h['named_gpu_gates']} == release.gates.ALL_GATES
        and len(summary['rows']) == 23
        and {r['gate']:r['selector'] for r in summary['rows']} == release.gates.ALL_GATES
        and len({r['launch_id'] for r in summary['rows']}) == 23
        and (summary['full_model_calls'], summary['quality_metrics']) == (37, 40293)
        and summary['all_exact'] is True and summary['failures'] == 0
        and summary['prior_gpu_passes_inherited'] is False and summary['reused_backend_gates'] == [],
        'Fresh named 23 gates / 37 calls / 40293 exact required; supplement cannot substitute')
    supplement = pre.checked_document(h['supplemental_allocator_18'])
    ct.require(supplement['gpu_passed'] == 18 and supplement['gpu_failed'] == 0
        and supplement['production_freeze_drift'] is False, 'Pinned supplemental allocator evidence differs')
    relation = pre.checked_document(h['source_relation'])
    ct.require(set(relation['production_changes']) == set(PRODUCTION), 'Qualification production scope differs')
    patch = relation['test_only_source_patch']
    ct.require({k:patch[v] for k,v in (('path','path'),('before_sha256','before'),('after_sha256','after'))} == TEST_PATCH,
               'Qualification cfg(test) patch differs')
    return h, summary, relation


def validate_test_patch():
    data = Path(TEST_PATCH['path']).read_bytes()
    ct.require(hashlib.sha256(data).hexdigest() == TEST_PATCH['after_sha256'], 'Unreviewed cfg(test) file bytes')
    prefix, marker, tests = data.partition(b'#[cfg(test)]')
    entry = b'                CutlassKernelId::NumericBf16M1Gemv,\n'
    ct.require(marker and tests.count(entry) == 1, 'Fixed cfg(test) manifest entry missing/ambiguous')
    restored = prefix + marker + tests.replace(entry, b'', 1)
    ct.require(hashlib.sha256(restored).hexdigest() == TEST_PATCH['before_sha256'],
               'cfg(test) reverse patch must restore the exact BEFORE file')


def changes(before, after):
    return [dict(path=k, before_sha256=before.get(k), after_sha256=after.get(k))
            for k in sorted(set(before) | set(after)) if before.get(k) != after.get(k)]


def source_delta(before_inventory, after_inventory, after_sha):
    """Recompute all production/build bytes, including native .cc and test changes."""
    ct.require(pre.contract._sha(after_sha) and after_sha != BEFORE_SHA, 'Distinct final AFTER binary hash required')
    before = pre.release_sources(before_inventory); after = pre.release_sources(after_inventory)
    ct.require(set(before) == set(after), 'Production inventory added/removed files')
    _, _, relation = qualification_identity()
    production = [dict(path=k, before_sha256=v['before'], after_sha256=v['after'])
                  for k,v in sorted(relation['production_changes'].items())]
    ct.require(changes(before, after) == sorted([*production, TEST_PATCH], key=lambda r:r['path']),
               'Only the exact qualified allocator/runtime and cfg(test) bytes may change')
    def tests(inventory):
        return {str((ct.ROOT/p).resolve()):h for p,h in inventory.items()
                if (ct.ROOT/p).resolve().is_relative_to(ct.ROOT/'crates')
                and len((ct.ROOT/p).resolve().relative_to(ct.ROOT).parts) > 3
                and (ct.ROOT/p).resolve().relative_to(ct.ROOT).parts[2] == 'tests'}
    return dict(schema=DELTA_SCHEMA, before_binary_sha256=BEFORE_SHA, after_binary_sha256=after_sha,
        before_inventory_digest=ct.digest(before_inventory), after_inventory_digest=ct.digest(after_inventory),
        before_production_source_digest=ct.digest(before), after_production_source_digest=ct.digest(after),
        production_changes=production, test_only_source_changes=[dict(TEST_PATCH)],
        test_source_changes=changes(tests(before_inventory), tests(after_inventory)),
        qualification_handoff=dict(QUALIFICATION))


def reviewed_build(link):
    ct.require(REVIEWED_BUILD is not None, 'AFTER fixed review manifest/hash and release adapter pending; CPU preparation only')
    ct.require(link == REVIEWED_BUILD, 'Unregistered retirement review manifest; no arbitrary allowlist')
    review = pre.checked_document(link)
    ct.require(set(review) == {'schema','optimization_class','before_binary_sha256','after_binary_sha256',
        'qualification_handoff','release_handoff','source_delta','rationale','safety_invariants'}
        and review['schema'] == REVIEW_SCHEMA and review['optimization_class'] == 'operator_retirement_v3'
        and review['before_binary_sha256'] == BEFORE_SHA and review['qualification_handoff'] == QUALIFICATION
        and pre.contract._sha(review['after_binary_sha256']) and review['after_binary_sha256'] != BEFORE_SHA
        and review['rationale'] and review['safety_invariants'], 'Reviewed retirement build identity differs')
    return review


def require_fresh_quality(q, prior):
    rows = q['rows']; gates = release.gates.ALL_GATES
    ct.require(q['schema']=='ferrule.paper.tp2-allocator-v3-test-quality.v1' and q['status']=='pass'
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
    ct.require(set(admission) == {'protocol','engines','before','cycle','operator_retirement_ab','review_manifest','after_admission'}
        and admission['protocol'] == PROTOCOL and set(admission['engines']) == {'ferrule'},
        'Unreviewed retirement protocol/engine/override')
    ct.require(admission['operator_retirement_ab'] == contract(admission['cycle'])
        and admission['before'] == frozen_metadata(), 'Retirement contract/frozen BEFORE differs')
    review = reviewed_build(admission['review_manifest'])
    frozen_audit()
    before = admission['before']; after = admission['engines']['ferrule']
    supplied = pre.checked_document(admission['after_admission'])
    ct.require(supplied['engines']['ferrule'] == after
        and after['binary']['sha256'] == review['after_binary_sha256']
        and after['build_receipt'] == review['release_handoff'], 'AFTER reviewed identity differs')
    # Unknown builder schemas remain rejected by the existing release auditor.
    # Registering a manifest alone never bypasses that auditor or same-binary smoke.
    paired.validate_admission(dict(protocol=pre.PROTOCOL, engines={'ferrule':after}), 'serving_smoke')
    pre.validate_smoke(after)
    q = pre.checked_document(after['test_quality'])
    require_fresh_quality(q, pre.checked_document(before['test_quality']))
    h, summary, _ = qualification_identity()
    ct.require(q['qualification_handoff_sha256'] == QUALIFICATION['sha256']
        and {r['gate']: (r['selector'], r['launch_id'], r['test_binary_sha256']) for r in q['rows']}
        == {r['gate']: (r['selector'], r['launch_id'], r['binary_sha256']) for r in summary['rows']},
        'AFTER quality must audit the specific fresh allocator V3 selectors and test binaries')
    cache_ab.common_specs(before, after)
    ct.require(before['decoded_cache'] == after['decoded_cache'] == release.CACHE
        and before['serving_scope'] == after['serving_scope'] == paired.CACHE_WORKSPACE_SCOPE,
        'Both variants require identical cache-on arena geometry')
    old_sources, old_build = indexed.production_sources(before)
    new_sources, new_build = indexed.production_sources(after)
    ct.check_pins(new_sources)
    ct.require(all(new_sources.get(p) == v for p,v in h['required_production_sources'].items()),
               'AFTER required qualified source hashes differ')
    delta = pre.checked_document(review['source_delta'])
    ct.require(delta == source_delta(pre.checked_document(old_build['source_pins_before']),
        pre.checked_document(new_build['source_pins_before']), after['binary']['sha256']), 'Reviewed complete source delta differs')
    validate_test_patch()
    for link in (admission['review_manifest'], review['source_delta']):
        ct.require(after['build_evidence_pins'].get(link['path']) == link['sha256'], 'Review/delta absent from AFTER evidence pins')
    old_inputs = pre.checked_document(old_build['receipts']['build-inputs'])
    new_inputs = pre.checked_document(new_build['receipts']['build-inputs'])
    import allocator_v3_release
    allocator_v3_release.validate_build_inputs(old_inputs, new_inputs)
    argv = paired.contract.http_plan('ferrule', after['spec'], 'qa8', Path('/not-executed'))['argv']
    ct.require(argv[argv.index('--concurrency')+1] == '1' and argv[argv.index('--warmup')+1] == '2'
        and '--first-after-ready' in argv and paired.shared.config()['workloads']['qa8']
        == dict(max_tokens=8, requests=5, warmup=2, first_after_ready=True), 'Frozen serial QA8 contract differs')
    return admission


def admission(output, after_path, after_sha, review_path, review_sha, cycle):
    ct.require(pre.contract._sha(after_sha) and pre.contract._sha(review_sha), 'Explicit final admission/review hashes required')
    review_link = dict(path=str(review_path.resolve()), sha256=review_sha)
    reviewed_build(review_link)
    after_link = dict(path=str(after_path.resolve()), sha256=after_sha)
    candidate = pre.checked_document(after_link)
    result = dict(protocol=PROTOCOL, engines={'ferrule':deepcopy(candidate['engines']['ferrule'])},
        before=frozen_metadata(), cycle=cycle, operator_retirement_ab=contract(cycle),
        review_manifest=review_link, after_admission=after_link)
    validate(result)
    output = output.resolve()
    ct.require(output.is_relative_to(pre.HERE/'admission') and output.parent.is_dir() and not output.exists(),
               'Fresh paper admission location required')
    with output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    return output



def allocator_rows(log, binary_sha256):
    """Ready/shutdown host observations, never a peak or CUDA-event trace."""
    import allocator_v3_release
    ct.require(binary_sha256 in (BEFORE_SHA, allocator_v3_release.BINARY_SHA),
               "Unreviewed allocator observation binary")
    v3 = binary_sha256 == allocator_v3_release.BINARY_SHA
    snapshots = release.snapshot_rows(log, profiling=False)
    ct.require(all(r["schema_version"] == 2 and r["snapshot"]["arena_backing_grants"] == 3 << 30
        for r in snapshots), "Retirement requires observed v2 3-GiB arena roots")
    messages = {"Mixed owner preparation claim completion": "ready",
                "Mixed owner shutdown fence; image retained until worker retirement": "shutdown"}
    required = ("allocation_failures", "reserved_bytes", "peak_reserved_bytes",
                "pending_retirement_blocks", "pending_retirement_bytes", "capture_retirement_bytes")
    rows, stages = [], {0: [], 1: []}
    ct.require("driver growth deferred by ordinary retirement pressure" not in log,
               "Normal owner hit retirement pressure; no threshold relaxation")
    for line in log.splitlines():
        if "CudaAllocatorMetrics {" not in line: continue
        record = json.loads(line[line.find("{"):]); f = record["fields"]
        owner = f.get("owner"); stage = messages.get(f.get("message"))
        ct.require(type(owner) is int and owner in stages and type(f.get("device")) is int
            and f["device"] == owner and stage is not None, "Foreign allocator owner/boundary")
        ct.require(record.get("level") == "INFO"
            and record.get("target") == "ferrule_runtime::parallel::pipeline::hybrid_cuda_tensor::mixed",
            "Unexpected allocator observation target/level")
        raw = f["allocator"]
        ct.require(raw.startswith("CudaAllocatorMetrics { ") and raw.endswith(" }"), "Allocator Debug format changed")
        parts = [part.split(": ", 1) for part in raw[len("CudaAllocatorMetrics { "):-2].split(", ")]
        values = {k: json.loads(v) for k,v in parts}
        ct.require(len(values) == len(parts), "Duplicate allocator metric")
        ct.require(all(type(values.get(k)) is int and values[k] >= 0 for k in required),
                   "Missing/invalid allocator boundary metric")
        ct.require(values["allocation_failures"] == 0
            and values["reserved_bytes"] <= values["peak_reserved_bytes"] <= 22 << 30,
            "Allocator allocation failure/reservation cap exceeded")
        ct.require(all(values[k] == 0 for k in required[-3:]), "Owner still has pending retirement")
        ct.require(f.get("device_quarantine") is False
            and f.get("claim_finished" if stage == "ready" else "quiescent") is True
            and (stage != "shutdown" or f.get("operator_quarantine") is False),
            "Owner completion/quarantine proof missing")
        if v3:
            ct.require(type(values.get("shutdown_quiescent")) is bool
                and values.get("shutdown_retained") is False, "Invalid V3 shutdown state")
        stages[owner].append(stage)
        rows.append(dict(owner=owner, stage=stage, metrics=values))
    ct.require(all(order == ["ready", "shutdown"] for order in stages.values()),
               "Missing/duplicate allocator owner boundaries")
    if v3:
        release.gates.allocator_v3_rows(log, 1)
    return dict(schema="ferrule.paper.operator-retirement-boundaries.v1", binary_sha256=binary_sha256,
        metric_schema="allocator_v3" if v3 else "historical_9ef", rows=rows,
        arena_bytes_per_owner=3 << 30, quiescent_pending_zero=True, ordinary_pressure_failure_count=0,
        observed_boundary_pending_bytes=max(r["metrics"]["pending_retirement_bytes"] for r in rows),
        pending_peak=None, unknown_peak=None, unknown_pressure=None,
        observation_scope="Ready/shutdown only; absent high-watermarks remain null, never zero. "
            "V3 fences count successful complete event sets, not all CUDA event calls; BEFORE fields stay historical.")


def finish(directory, prepared):
    import paired
    a = prepared['admission']; rows = []; reference = paired.shared.references()[0][0]['reference_text']
    for slot in slots(a['cycle']):
        cell = directory/'runs'/slot; session = cell/'session'
        assessed = paired.assess_slot(cell, 'ferrule'); case = ct.load(session/'case.json')
        log = (session/'server.log').read_text()
        ct.require('mixed_rank_performance' not in log and 'request_worker_terminal' not in log,
                   'Retirement benchmark diagnostics unexpectedly enabled')
        snapshots = release.snapshot_rows(log, profiling=False)
        ct.require(all(r['schema_version'] == 2 and r['snapshot']['arena_backing_grants'] == 3 << 30 for r in snapshots),
                   'Retirement requires observed v2 3-GiB arena roots')
        for phase in ('hello2','canary8','qa8-V1'):
            gate = ct.load(session/phase/'gate.json')
            ct.require(gate['status'] == 'pass' and gate['http_receipt_sha256'] == ct.sha(session/phase/'http.json'),
                       'Retirement HTTP gate/report differs')
        http = ct.load(session/'qa8-V1/http.json')
        ct.require(len(http['warmups']) == 2 and len(http['runs']) == 1
            and len(http['runs'][0]['requests']) == 5, 'Require 3 unmeasured and 5 measured QA8')
        requests = [http['first_after_ready'], *http['warmups'], *http['runs'][0]['requests']]
        ct.require(all(r['status'] == 'success' and r['actual_input_tokens'] == 5 and r['actual_output_tokens'] == 8
            and r['text'] == reference for r in requests), 'Retirement actual tokens/canary differ')
        resources = case['resources']
        ct.require(resources['status'] == 'observed_within_targets' and not resources['errors'], 'Retirement resource failure')
        with (session/'samples.jsonl').open() as stream:
            samples = [r for r in map(json.loads, stream) if r.get('kind') == 'sample']
        rows.append(dict(slot=slot, binary=paired.slot_metadata(a, slot)['binary'], launch_id=assessed['launch_id'],
            startup_seconds=case['startup_seconds'], summary=http['runs'][0]['summary'], resources=resources,
            cache_snapshots=snapshots, allocator_retirement=assessed["runtime"]["allocator_retirement"],
            activity=None, unknown_pressure=None, pending_peak=None, unknown_peak=None,
            retirement_event_counts=None, cuda_api_event_counts=None,
            cpu_observation=contract(a['cycle'])['cpu_observation'],
            gpu_samples=sum(r.get('gpu') is not None for r in samples), missing_gpu_samples=sum(r.get('gpu') is None for r in samples)))
    ct.require(len({r['launch_id'] for r in rows}) == 2, 'Fresh retirement owners required')
    result = dict(protocol=PROTOCOL, status='complete_descriptive_only', cycle=a['cycle'], order=list(slots(a['cycle'])),
        comparison_identity=ct.digest(dict(before_sha=BEFORE_SHA, after_sha=a['engines']['ferrule']['binary']['sha256'],
            review_sha=a['review_manifest']['sha256'])), review_manifest=a['review_manifest'], rows=rows,
        source_pins={s:paired.slot_metadata(a, s)['source_pins'] for s in SLOTS},
        rate_window='Unchanged runner fixed-work window includes terminal checkpoint/executor exit; not the Nsight request window',
        limitations=['Two crossed cycles are descriptive, not statistical significance or exclusive retirement causality.',
            'Whole-lifecycle cache counts do not locate hot decode; null activity/pressure/peaks are not zero.',
            'Retirement ledger counts complete created event sets, not total CUDA events; partial-failure driver calls uncounted.',
            'No measured CPU affinity/thread counts or GPU clocks/thermals; local drain does not exclude foreign interference.',
            'No official 9ef profile exists yet; old 4d40 traces are not a retirement baseline.'],
        isolated_retirement_effect=False, statistical_significance_claim=False, fairmatch=False, exact_speedup=None,
        artifact_pins=ct.inventory([p for p in directory.rglob('*') if p.is_file()]))
    ct.require(not (directory/'operator-retirement-summary.json').exists(), 'No retirement summary overwrite')
    ct.write(directory/'operator-retirement-summary.json', result)
    return result


def plan():
    import profile_2025
    import retirement_profile
    return dict(protocol=PROTOCOL, status='CPU_adapter_ready_same_binary_smoke_and_GPU_authorization_pending',
        execute_allowed=False, gpu_started=False, gpu_queried=False, locks_acquired=False,
        before=dict(path=str(BASELINE), sha256=BASELINE_SHA, binary_sha256=BEFORE_SHA),
        qualification_handoff=dict(QUALIFICATION), qualification_schema=QUALIFICATION_SCHEMA,
        qualification_counts=dict(fresh_named_gpu_gates=23, calls=37, exact_metrics=40293,
            supplemental_gpu_small=18, supplemental_replaces_named_gates=False),
        after_binary_sha256=retirement_profile.release.BINARY_SHA, registered_build_review=REVIEWED_BUILD, cycles=[contract(1), contract(2)],
        launcher='paired.py --purpose operator_retirement_ab; existing admission/prepare/run/worker/supervisor',
        review_manifest_schema=dict(schema=REVIEW_SCHEMA, optimization_class='operator_retirement_v3',
            required_fields=['before_binary_sha256','after_binary_sha256','qualification_handoff','release_handoff',
                'source_delta','rationale','safety_invariants'], registration='One code-reviewed exact manifest path/SHA; no CLI override'),
        source_delta_schema=dict(schema=DELTA_SCHEMA, production_paths=list(PRODUCTION), test_only_source_changes=[dict(TEST_PATCH)],
            required_fields=['before_binary_sha256','after_binary_sha256','before_inventory_digest','after_inventory_digest',
                'before_production_source_digest','after_production_source_digest','production_changes',
                'test_only_source_changes','test_source_changes','qualification_handoff'],
            inventory='Stable full builder before/after maps, including native .cc and build/config inputs; test changes separately',
            qualification_native_cc_coverage_gaps=4, qualification_gap_is_not_permission_to_omit_build_inventory=True),
        promotion_order=['Review final release handoff, binary SHA, full source/dependency inventories, exact fixed review manifest',
            'Integrate actual typed release audit; reject unknown schema; fully audit named qualification receipts',
            'Fresh explicit main-agent GPU authorization, then actual SAME-AFTER-binary S1/S2',
            'Promote only after both smoke owners pass; then crossed cycles BEFORE/AFTER and AFTER/BEFORE',
            'Separately authorized official short profile AFTER; BEFORE needs its own same-9ef-artifact short trace'],
        profile=dict(status='plan_only_not_admitted', release=profile_2025.TOOL_CONFIG['release'], tool_pins=profile_2025.TOOL_PINS,
            launcher='existing paired.py --purpose profiling collector-first lifecycle',
            supervision='Reuse eight UUID locks, stable B0, selected 0/1 B1, authenticated ECHILD/drain back to B0; no new framework',
            actual_request=dict(input_tokens=5, output_tokens=8, warmup=1, measured=1), rankperf=False,
            log='info', cache_activity_required=False, policy=dict(retirement_profile.CONFIG),
            pending_evidence='Fixed info-only rankperf-off adapter registered; actual same-artifact traces and fresh authorization required',
            correlation='Existing profile_trace_audit_v2 full-export bidirectional (PID, correlationId), same actual-request window',
            cuda_api_counts=dict(runtime=['cudaEventCreate','cudaEventRecord','cudaEventQuery','cudaEventDestroy'],
                driver=['cuEventCreate','cuEventRecord','cuEventQuery','cuEventDestroy'],
                additional=['stream-wait-event','event/stream/context synchronization'],
                scope='Separate API layers, owner/thread/request/phase, durations and return status; boundary-crossing calls disclosed',
                missing_coverage=None, unique_events=None, completed_retirements=None),
            evidence='Preserve raw nsys-rep/SQLite, clock anchors, diagnostics/loss/coverage; correlation closure is not trace completeness',
            before_official_profile=None, historical_4d40_is_retirement_baseline=False,
            cpu_stacks='Not requested; stripped artifacts are not stack evidence; no replacement build or symbol scan'),
        unknown_pressure=None, pending_peak=None, fairmatch=False, exact_speedup=None)


def main(argv=None):
    import argparse
    ct.require_cpu_python()
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('mode', choices=('plan','admission'))
    cli.add_argument('--after-admission', type=Path)
    cli.add_argument('--after-admission-sha256')
    cli.add_argument('--review-manifest', type=Path)
    cli.add_argument('--review-manifest-sha256')
    cli.add_argument('--cycle', type=int, choices=(1,2))
    cli.add_argument('--output', type=Path)
    args = cli.parse_args(argv)
    if args.mode == 'plan':
        print(json.dumps(plan(), indent=2))
    else:
        ct.require(all(x is not None for x in (args.after_admission,args.review_manifest,args.cycle,args.output)),
                   'AFTER admission, fixed review manifest, cycle and output required')
        print(admission(args.output, args.after_admission, args.after_admission_sha256,
            args.review_manifest, args.review_manifest_sha256, args.cycle))


if __name__ == '__main__':
    main()
