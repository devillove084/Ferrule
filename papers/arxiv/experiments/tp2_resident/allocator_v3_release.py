"""Exact V3 serving-build auditor. No old-schema aliases or GPU launch authority."""
from pathlib import Path
import json
import re
import shlex

import pre_serving as pre
import decoded_cache_release as d
import operator_retirement_ab as ab

ct = pre.ct
gates = d.gates
Evidence = d.Evidence
pin_links = d.pin_links
arena_gate_inventory = d.arena_gate_inventory
CALLS = d.CALLS
QUALITY_SCHEMA = 'ferrule.paper.tp2-allocator-v3-test-quality.v1'
SCHEMA = 'ferrule.build.allocator-v3-serving-release.v1'
RELEASE = ct.ROOT/'target/validation/tp2-mixed-allocator-v3-release-20261011-01'
HANDOFF_SHA = '8362e938840955840114c59c8092be19eec0dc3b5c9237f10b39832bdda23b28'
FILE_PINS_SHA = 'bc1d9ceee571562fc085e5c915b39dbf4f72b07a0fdead4bcaf0d3168328152d'
BINARY_SHA = '7e1c23a88143cb6455e3c6e10bfe43a567fc97b5fd45745800eae3a1e39a0e73'
QUALITY = Path(ab.QUALIFICATION['path']).parent
HISTORY = pre.HERE/'history/pre-operator-retirement-release-20261011'
WORKFLOW_ARCHIVES = {str(pre.HERE/name): HISTORY/(name+'.txt') for name in ('pre_serving.py','optimized_release.py','test_indexed_release.py')}
WORKFLOW_ARCHIVES[str(pre.HERE/'paired.py')] = ab.WORKFLOW_ARCHIVE
EXPECTED_BUILD = ['cargo','build','--locked','--offline','--release','-p','ferrule-cli',
    '--no-default-features','--features','cuda','--bin','ferrule','--target-dir',
    'target/catalog-tp2-cuda','--message-format=json-render-diagnostics','-vv']


def validate_build_inputs(old, new):
    expected = dict(old['environment'], FERRULE_CUDA_TEST_ORACLE='0')
    ct.require('FERRULE_CUDA_TEST_ORACLE' not in old['environment'] and new['environment'] == expected,
               'Only reviewed unset-to-0 oracle build environment difference is permitted')
    ct.require(new['release_profile'] == old['release_profile'] ==
        dict(opt_level=3,lto='fat',codegen_units=1,panic='unwind',strip='symbols'), 'V3 release profile differs')
    for key in ('cargo','rustc','nvcc','cxx','cutlass_head','cutlass_status'):
        ct.require(new['toolchain'][key] == old['toolchain'][key], 'V3 toolchain differs: '+key)
    build = (ct.ROOT/'crates/ferrule-backend/build.rs').read_text()
    ct.require('env::var(CUDA_TEST_ORACLE_ENV).is_ok_and(|value| value == "1")' in build,
               'Oracle disabled semantics changed')


def qualification_extras(e, h, summary, prepared):
    ct.require((h['fresh_gpu_gates'],h['fresh_cpu_selectors'],h['full_model_calls'],h['quality_metrics']) == (23,38,37,40293)
        and h['supplemental_replaces_named_gates'] is False and h['prior_gpu_passes_inherited'] is False,
        'V3 qualification grouping/lineage differs')
    ct.require(all(summary['sources'].get(p) == v for p,v in h['required_production_sources'].items()),
               'V3 required production hashes differ')
    for key in ('check','strict_clippy','rustfmt'):
        receipt = e.linked(h[key])
        ct.require(ct.successful_process(receipt['supervisor']), 'V3 CPU validation failed: '+key)
        if key == 'strict_clippy':
            ct.require(receipt['argv'][-3:] == ['--','-D','warnings'], 'V3 qualification Clippy not strict')
    for key in ('backend_unit_release','paper_cpu_policy'):
        receipt = e.linked(h[key]['receipt'])
        ct.require(ct.successful_process(receipt['supervisor']), 'V3 qualification CPU suite failed')
    ledger = e.linked(h['allocator_ledger']); pin_links(e, ledger)
    released = e.linked(h['final_release'])
    ct.require(released['gpu_owner_released'] is True and released['all_eight_idle'] is True,
               'V3 qualification final release missing')


def prepared_plan(e, root, prepared):
    """A historical preparation binds its own plan, not today's mutable runner."""
    path = root/'plan.json'
    ct.require(str(path) in prepared['pins'], 'Qualification plan is not pinned')
    plan = e.read(path, prepared['pins'][str(path)])
    ct.require(plan['schema']=='ferrule.paper.allocator-v3-plan.v1'
        and plan['gpu_selectors']==gates.GATES
        and {k:tuple(v) for k,v in plan['small_selectors'].items()}==gates.SMALL_GATES,
        'Pinned named GPU gate plan differs')
    ignored = {tuple(row) for row in plan.get('ignored_host_selectors',[])}
    selectors = [(kind,selector,(kind,selector) in ignored) for kind,selector in plan['host_small_selectors']]
    selectors += [('composition',selector,skip) for selector,skip in plan['cpu_selectors'].items()]
    receipts = prepared['cpu_receipts']
    ct.require(len(receipts)==len(selectors) and len({r['path'] for r in receipts})==len(receipts),
               'Pinned CPU gate receipts missing/duplicated')
    for index, (ref,(kind,selector,skip)) in enumerate(zip(receipts,selectors)):
        receipt = e.linked(ref,root/('cpu-'+str(index)+'.json'))
        argv = [prepared['binaries'][kind]['path'],selector,'--exact','--nocapture','--test-threads=1']
        if skip: argv.append('--ignored')
        ct.require(receipt['argv']==argv and ct.successful_process(receipt['supervisor']),
                   'Pinned CPU selector/receipt differs')
    return plan


def quality(e, qualification):
    ct.require(qualification['schema'] == ab.QUALIFICATION_SCHEMA, 'Wrong V3 qualification schema')
    quality_root = QUALITY
    summary_sha = '8719eb72f0a042b8eb5514c5033ba85877de1beceabc96eb24adb52ee52893e4'
    qualification_sha = ab.QUALIFICATION['sha256']
    e.pin(quality_root/'handoff.json', qualification_sha)
    summary = e.linked(qualification['acceptance_summary'], quality_root/'acceptance-summary.json')
    e.pin(quality_root/'acceptance-summary.json', summary_sha)
    ct.require(summary['schema'] == 'ferrule.paper.allocator-v3-acceptance.v1'
        and summary['all_exact'] is True and summary['failures'] == 0
        and summary['prior_gpu_passes_inherited'] is False and summary['reused_backend_gates'] == []
        and summary['public_e2e_accepted'] is False and summary['performance_accepted'] is False,
        'Arena qualification failed/overclaimed')
    arena_gate_inventory(summary)
    prepared = e.linked(summary['prepared'], quality_root/'prepared.json')
    ct.require(summary['prepared'] == qualification['prepared']
        and prepared['schema'] == 'ferrule.paper.allocator-v3-prepared.v1'
        and prepared['sources'] == summary['sources'] and prepared['binaries'] == summary['binaries']
        and prepared['tools'] == summary['tools'] and qualification['binaries'] == summary['binaries']
        and ct.digest(summary['sources']) == qualification['source_digest'], 'Arena preparation differs')
    prepared_plan(e, quality_root, prepared)
    history = {}
    for path, digest in prepared['pins'].items():
        original = Path(path)
        archived = path in WORKFLOW_ARCHIVES
        actual = WORKFLOW_ARCHIVES.get(path, original)
        e.pin(actual, digest)
        if archived: history[path] = dict(path=str(actual), sha256=digest)
    for path, digest in summary['sources'].items(): e.pin(path, digest)
    ct.validate_inventory(ct.config(ct.EXCLUSIVE_PATH), prepared['tools'])
    for pins in prepared['tools'].values():
        for path, digest in pins.items():
            ct.require(path not in e.pins or e.pins[path] == digest, 'Arena tool pin collision')
            e.pins[path] = digest
    for binary in summary['binaries'].values(): e.pin(binary['path'], binary['sha256'])
    for link in prepared['cpu_receipts']:
        receipt = e.linked(link)
        ct.require(ct.successful_process(receipt['supervisor']), 'Arena CPU qualification failed')
    pin_links(e, qualification)
    calls = metrics = 0
    audited = []
    for row in summary['rows']:
        name = row['gate']; directory = quality_root/name
        docs = {key: e.linked(row[key], directory/(key+'.json')) for key in (
            'supervisor','authentication','worker-launch','test-exit','resources','release','before','after','result')}
        launch = docs['supervisor']; auth = docs['authentication']; worker = docs['worker-launch']
        exit = docs['test-exit']; release = docs['release']; result = docs['result']
        ct.require(ct.successful_process(launch) and launch['custody']['echild'] is True
            and auth['launch_id'] == launch['handshake']['launch_id'] == row['launch_id']
            and result['passed'] is True and result['drained'] is True
            and all(release[k] is True for k in ('gpu_owner_released','drained','all_eight_idle')),
            'Arena gate authenticated closure failed: '+name)
        ct.require(exit['one_test_passed'] is True and exit['returncode'] == 0
            and exit['selector'] == row['selector'] and exit['binary_sha256'] == row['binary_sha256']
            and worker['gate'] == name and worker['binary_sha256'] == row['binary_sha256']
            and worker['sources'] == summary['sources'], 'Arena executed gate/source differs: '+name)
        gates.validate_execution_devices(name, worker['devices'])
        ct.require(row['selected_and_visible'] == gates.execution_devices(name)
            and docs['resources']['status'] == 'observed_within_targets' and not docs['resources']['errors']
            and all(docs[k]['status'] == 'verified' for k in ('before','after')), 'Arena devices/resources/idle differ')
        log = e.pin(row['log']['path'], row['log']['sha256'])
        ct.require(log == directory/'test.log', 'Arena test log path differs')
        if name in gates.SMALL_GATES:
            small = e.linked(row['assertions'], directory/'small-receipt.json')
            ct.require(small['assertions_passed'] is True and small['selector'] == row['selector']
                and small['binary_sha256'] == row['binary_sha256']
                and small['source_digest'] == qualification['source_digest'], 'Arena small assertions differ')
            pin_links(e, small)
        else:
            comparison = e.linked(row['comparison'], quality_root/(name+'-comparison.json'))
            gates.exact_inventory(comparison, name)
            groups = comparison['cases'] if name == 'full40' else comparison['calls']
            n = sum(len(g['metrics']) for g in groups.values())
            ct.require(all(m['rmse'] == 0 for g in groups.values() for m in g['metrics'].values())
                and (row['calls'],row['metrics']) == (CALLS[name],n), 'Arena exact comparator differs')
            reference, expected = gates.FULL if name == 'full40' else gates.RECIPES[name]
            receipt = e.linked(row['comparison_receipt'], directory/'comparison-receipt.json')
            ct.require(receipt['report'] == row['comparison'] and receipt['all_exact'] is True
                and receipt['reference_sha256'] == comparison['reference_manifest_sha256'] == expected,
                'Arena comparator/reference binding differs')
            e.pin(gates.ORACLE/'runs'/reference/'manifest.json', expected)
            e.pin(quality_root/(name+'-canonical')/'manifest.json', comparison['actual_manifest_sha256'])
            bindings = gates.trace_budgets(directory, name)
            pin_links(e, bindings)
            snapshots = e.linked(row['snapshots'], directory/'cache-snapshots.json')
            ct.require(snapshots == gates.internal_snapshot_rows(log.read_text(), bindings), 'Arena snapshot evidence differs')
            calls += CALLS[name]; metrics += n
        audited.append(dict(gate=name, selector=row['selector'], test_binary_sha256=row['binary_sha256'],
                            launch_id=row['launch_id']))
    ct.require((calls,metrics) == (summary['full_model_calls'],summary['quality_metrics']) == (37,40293)
        and (summary['fresh_gpu_gates'],summary['fresh_small_gates']) == (23,16), 'Arena quality totals differ')
    qualification_extras(e, qualification, summary, prepared)
    return summary, dict(schema=QUALITY_SCHEMA, status='pass', kind='authenticated_test_binary_not_serving',
        calls=calls, metrics=metrics, rows=audited, numerical_pass_inherited=False,
        historical_workflow_pins=history, qualification_handoff_sha256=qualification_sha,
        qualification='23 named fresh V3 gates; separate 18-gate supplement is not a replacement; actual serving smoke remains required')



def supplement(e, h):
    import allocator_v3_gate as small
    relation = e.read(RELEASE/'allocator18-supplement-relation.json')
    ct.require(relation['closeout'] == h['supplemental_allocator_18'] and relation['gpu_passed'] == 18
        and relation['replaces_named_23'] is False and relation['inherited_as_release_pass'] is False,
        'V3 supplemental grouping differs')
    close = e.linked(relation['closeout']); prepared = e.linked(relation['prepared']); result = e.linked(relation['result'])
    ct.require(close['gpu_passed'] == 18 and close['gpu_failed'] == 0 and result['passed'] is True
        and result['owner_lock_released'] is True and len(result['phases']) == 18, 'V3 supplement failed')
    expected = ['tests::'+s for s in small.NEW_GPU]
    expected += ['cuda::runtime::context_drop_gpu_tests::'+s for s in small.OLD_GPU]
    expected += [s for s,_ in small.PUBLIC]
    rows = relation['rows']
    ct.require(len(rows) == 18 and [r['selector'] for r in rows] == expected, 'Specific supplemental selectors differ')
    for path, digest in relation['verified_artifact_pins'].items(): e.pin(path,digest)
    pin_links(e,prepared)
    for row, phase in zip(rows,result['phases']):
        receipt = e.linked(row['receipt']); binary = row['binary']; e.pin(binary['path'],binary['sha256'])
        ct.require(row['passed'] is True and phase['passed'] is True and phase['selector'] == row['selector']
            and prepared['binaries'][phase['binary']] == binary
            and receipt['argv'][:2] == [binary['path'],row['selector']]
            and '--exact' in receipt['argv'] and '--test-threads=1' in receipt['argv']
            and ct.successful_process(receipt['supervisor']) and receipt['supervisor']['custody']['echild'] is True
            and '1 passed; 0 failed' in receipt['supervisor']['stdout']
            and receipt['environment']['CUDA_VISIBLE_DEVICES'] == small.UUID,
            'Supplemental selector/binary/custody differs')
    return dict(gates=18, rows=rows, replaces_named_23=False, release_smoke=False)


def cpu_evidence(e, receipts):
    inventory = receipts['cpu-test-inventory']
    ct.require((inventory['passed'],inventory['ignored'],inventory['gpu_visible']) == (26,3,False), 'V3 CPU count differs')
    commands = {
        'allocator-v3-cpu':['test','--locked','--offline','-p','ferrule-backend','--features','cuda','--test','allocator_batch_v3','v3_','--','--test-threads=1','--nocapture'],
        'runtime-snapshot':['test','--locked','--offline','-p','ferrule-runtime','--features','cuda','--test','mixed_decoded_snapshot','--','--test-threads=1','--nocapture'],
        'composition-cache':['test','--locked','--offline','-p','ferrule-composition','--features','cuda','--test','tp2_mixed_execution','decoded_cache::','--','--test-threads=1','--nocapture'],
        'composition-public':['test','--locked','--offline','-p','ferrule-composition','--features','cuda','--lib','mixed_tp2_','--','--test-threads=1','--nocapture'],
        'cli-public':['test','--locked','--offline','-p','ferrule-cli','--features','cuda','--bin','ferrule','mixed_tp2_','--','--test-threads=1','--nocapture'],
    }
    for name, argv in commands.items():
        r = e.read(RELEASE/(name+'.json')); log = e.pin(RELEASE/(name+'.stdout')).read_text()
        ct.require(r['status'] == 'passed' and r['returncode'] == 0 and r['timed_out'] is False
            and r['argv'] == ['/root/.cargo/bin/cargo',*argv] and r['environment']['CUDA_VISIBLE_DEVICES'] == ''
            and r['environment']['NVIDIA_VISIBLE_DEVICES'] == 'none', 'V3 CPU selector/environment failed: '+name)
        rows = [r for r in inventory['tests'] if r['group'] == name]
        ct.require(re.findall(r'^test (\S+) \.\.\. (.*)$',log,re.M) == [(r['selector'],r['status']) for r in rows]
            and re.search(r'test result: ok\. [1-9][0-9]* passed; 0 failed;',log), 'V3 CPU actual inventory differs')
    for ref in receipts['strict-clippy-summary']['receipts']:
        r = e.linked(ref)
        ct.require(r['status'] == 'passed' and r['returncode'] == 0 and r['timed_out'] is False
            and r['argv'][1:4] == ['clippy','--locked','--offline'] and r['argv'][-3:] == ['--','-D','warnings']
            and r['environment']['CUDA_VISIBLE_DEVICES'] == '' and r['environment']['NVIDIA_VISIBLE_DEVICES'] == 'none',
            'V3 strict Clippy failed')
    ct.require(len(receipts['strict-clippy-summary']['receipts']) == 4, 'Missing strict Clippy groups')


def audit(link):
    ct.require(link == dict(path=str(RELEASE/'handoff.json'),sha256=HANDOFF_SHA), 'Unreviewed V3 serving handoff')
    e = Evidence(); h = e.linked(link)
    ct.require(h['schema'] == SCHEMA and h['strict_clippy_passed'] is True
        and all(h[k] is False for k in ('inherited_gpu_pass','release_is_quality_test_binary','release_smoke_passed',
            'public_e2e_accepted','performance_accepted','profile_run','debug_build_created','gpu_workload_started','server_started')),
        'V3 release schema/claims differ')
    files = e.read(RELEASE/'handoff-file-pins.json', FILE_PINS_SHA)
    ct.require(files['handoff.json'] == HANDOFF_SHA and len(files) == 96, 'V3 final builder inventory differs')
    for name, digest in files.items():
        ct.require(Path(name).name == name, 'Unsafe V3 builder filename')
        e.pin(RELEASE/name,digest)
    receipts = {name:e.linked(ref,RELEASE/(name+'.json')) for name,ref in h['receipts'].items()}
    pin_links(e,receipts)
    before = e.linked(h['source_pins_before']); after = e.linked(h['source_pins_after'])
    full = pre.release_sources(before)
    sources = {p:v for p,v in full.items() if Path(p).suffix not in ('.md','.cc')}
    ct.require(before == after and full == e.linked(h['production_source_pins_before'])
        == e.linked(h['production_source_pins_after']) and len(full) == 584
        and sources == pre.current_release_sources(), 'V3 complete release source inventory drift')
    for path, digest in full.items(): e.pin(path,digest)
    old_m = ab.frozen_metadata(); old = pre.checked_document(old_m['build_receipt'])
    old_sources = pre.release_sources(pre.checked_document(old['source_pins_before']))
    delta = e.linked(h['production_delta'])
    ct.require(delta['schema'] == 'ferrule.build.allocator-v3-production-delta.v1'
        and delta['before_binary_sha256'] == ab.BEFORE_SHA and delta['after_binary_sha256'] == BINARY_SHA
        and set(full) == set(old_sources) and delta['production_changes'] == ab.changes(old_sources,full)
        and len(delta['production_changes']) == 3 and delta['added'] == delta['removed'] == []
        and delta['before_production_source_digest'] == ct.digest(old_sources)
        and delta['after_production_source_digest'] == ct.digest(full), 'V3 builder delta differs')
    ab.validate_test_patch()
    review = ab.reviewed_build(ab.REVIEWED_BUILD)
    paper_delta = e.linked(review['source_delta'])
    ct.require(paper_delta == ab.source_delta(pre.checked_document(old['source_pins_before']),before,BINARY_SHA),
               'Fixed review source delta differs from full actual inventories')
    e.linked(ab.REVIEWED_BUILD)
    state = receipts['release']; inputs = receipts['build-inputs']; env = inputs['environment']
    ct.require(state['argv'] == EXPECTED_BUILD and state['status'] == 'passed' and state['returncode'] == 0
        and state['timed_out'] is False and state['environment'] == env and env['CUDA_VISIBLE_DEVICES'] == ''
        and env['NVIDIA_VISIBLE_DEVICES'] == 'none' and env['FERRULE_CUDA_ARCH'] == 'sm_86' and 'RUSTFLAGS' not in env,
        'V3 release build/environment failed')
    validate_build_inputs(pre.checked_document(old['receipts']['build-inputs']),inputs)
    flags = receipts['observed-build-flags']; backend = flags['backend_build_script']
    ct.require(['FERRULE_BACKEND_CUDA_COMPILED_TARGET','sm_86'] in backend['env']
        and 'ferrule_cuda_test_oracle' not in backend['cfgs'], 'V3 compiled target/oracle differs')
    for flag in ('-C opt-level=3','-C lto=fat','-C codegen-units=1','-C strip=symbols'):
        ct.require(flag in flags['rustc_invocations']['ferrule']['command'], 'V3 release flag differs')
    log = e.pin(RELEASE/'release.stderr').read_text()
    ct.require(all(row['command'] in log for row in flags['rustc_invocations'].values()), 'Build commands absent from actual log')
    events = [json.loads(line) for line in (RELEASE/'release.stdout').read_text().splitlines() if line.startswith('{')]
    ct.require(any(r.get('reason') == 'build-finished' and r['success'] is True for r in events)
        and backend in events, 'V3 actual Cargo build completion missing')
    cpu_evidence(e,receipts)
    qualification = e.linked(h['qualification_handoff'])
    ct.require(h['qualification_handoff'] == ab.QUALIFICATION and h['qualification_source_digest'] == qualification['source_digest'],
               'V3 serving-to-qualification relation differs')
    summary, q = quality(e,qualification)
    ct.require(all(before.get(str(Path(path).relative_to(ct.ROOT))) == value for path,value in summary['sources'].items()
        if path in full), 'V3 tested-to-release production mismatch')
    supp = supplement(e,qualification)
    deps = receipts['actual-release-dependencies']
    actual_names = shlex.split((RELEASE/'ferrule.d').read_text().splitlines()[0].split(': ',1)[1])
    ct.require([row['path'] for row in deps] == actual_names, 'V3 actual compiler dependency inventory differs')
    prepared = e.linked(qualification['prepared'])
    for row in deps:
        path = Path(row['path'])
        if row.get('kind') == 'native_directory':
            actual = ct.inventory([f for f in path.rglob('*') if f.is_file()])
            ct.require(actual == row['files'] and all(full.get(p) == v for p,v in actual.items()), 'V3 native dependency tree differs')
            for p,v in actual.items(): e.pin(p,v)
        else:
            e.pin(path,row['sha256'])
            ct.require(row['sha256'] == before.get(str(path.relative_to(ct.ROOT)),
                prepared['tools']['cutlass_tree'].get(str(path))), 'V3 compile dependency/source binding differs')
    gaps = receipts['actual-dependency-quality-relation']['native_cc_qualification_coverage_gaps']
    ct.require(len(gaps) == 4 and {r['path'] for r in gaps} == {p for p in full if p.endswith('.cc')}
        and all(r['sha256'] == full[r['path']] == old_sources[r['path']] and r['qualification_sha256'] is None for r in gaps),
        'V3 native qualification gap disclosure differs')
    runtime = {}
    for path, item in receipts['runtime-library-pins']['libraries'].items():
        load = Path(item['load_path'])
        ct.require(str(load.resolve()) == path and load.stat().st_size == item['bytes'], 'V3 runtime identity differs')
        e.pin(path,item['sha256']); runtime[str(load)] = item['sha256']
    ct.require(runtime == old_m['runtime_pins'], 'V3 runtime libraries differ from BEFORE')
    artifact = h['artifact']; binary = e.pin(artifact['path'],BINARY_SHA)
    ct.require(artifact['sha256'] == BINARY_SHA and binary.parent == RELEASE and binary.stat().st_size == artifact['bytes']
        and artifact['independent_copy'] is True and not binary.samefile(artifact['source'])
        and BINARY_SHA not in {v['sha256'] for v in summary['binaries'].values()}, 'V3 serving artifact identity differs')
    e.pin(h['before_artifact']['path'],ab.BEFORE_SHA)
    for group in e.read(RELEASE/'preserved-artifacts-on-entry.json'):
        if 'path' in group:
            e.pin(ct.ROOT/group['path'],group['sha256'])
        else:
            e.linked(group['manifest'])
            for name,digest in group['files'].items(): e.pin(Path(group['receipt_directory'])/name,digest)
    launch = receipts['owner-launch-plan']
    ct.require(launch['artifact_sha256'] == BINARY_SHA and launch['decoded_cache_bytes'] == 3 << 30
        and launch['workspace_mib'] == 4096 and launch['device_ceiling_mib'] == 22528 and launch['executed'] is False,
        'V3 launch geometry differs')
    symbols = receipts['elf-symbol-status']
    ct.require(symbols['stripped'] is True and symbols['ordinary_symtab_present'] is False
        and symbols['dwarf_debug_sections_present'] is False, 'V3 symbol disclosure differs')
    q.update(source_relation=dict(inherited_gpu_pass=False, production_delta=h['production_delta'],
            complete_production_inventory_count=len(full), native_cc_qualification_pin_gaps=gaps,
            build_environment_difference={'FERRULE_CUDA_TEST_ORACLE':dict(before=None,after='0')},
            source_transfer='Exact fresh V3 source hashes; no inherited numerical pass'),
        numerical_source_pins=pre.numerical_sources(summary['sources']), release_numerical_source_pins=pre.numerical_sources(sources),
        supplemental_allocator=supp, public_metadata_ledger=d.metadata_ledger(e.read(RELEASE/'public-metadata-ledger.json')),
        public_cpu_relation=dict(passed=26,ignored=3,reused_prior_tests=False), artifact_pins=dict(e.pins),
        pending_peak=None,unknown_pressure=None,fairmatch=False,exact_speedup=None)
    import paired
    return dict(binary_path=str(binary),binary_sha256=BINARY_SHA,source_pins=sources,runtime_pins=runtime,
        evidence_pins=dict(e.pins),quality=q,serving_scope=dict(paired.CACHE_WORKSPACE_SCOPE),decoded_cache=dict(d.CACHE),
        build_custody='Actual pinned builder subprocess exit/logs, full source maps and dependencies; no invented authenticated builder supervisor; same-binary S1/S2 required')
