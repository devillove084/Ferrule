"""CPU-only typed adapter for the optimized builder's actual receipt schema.

Not a build/quality/server runner. Exact artifacts are pinned, never paper globs.
The old 33/35937 release policy remains separate and unchanged.
"""
from pathlib import Path
import pre_serving as pre

ct = pre.ct
SCHEMA = 'ferrule.optimized-mixed-tp2-release-handoff.v1'
QUALITY_SCHEMA = 'ferrule.paper.tp2-optimized-test-quality.v1'
RELEASE = ct.ROOT/'target/validation/tp2-mixed-optimized-release-20261009'
QUALITY = ct.ROOT/'target/validation/tp2-mixed-production-opts-01'


FUSED_SCHEMA = 'ferrule.build.pair-swiglu-fused-release.v1'
FUSED_RELEASE = ct.ROOT/'target/validation/pair-swiglu-fused-release-20261011-01'


ROUTEBATCH_READY = dict(path=str(ct.ROOT/'target/validation/route-batch-incremental-20261012-01/ready-release-builder.json'),
    sha256='4d870c9aadd427fc5a52eda03de474ed0b5bc9c01e5f1a2f9741810115430b0b')
REVIEWED_FUSED_CASES = {
    'fused': dict(binary_sha256='f6a65e1b6f09c0a8385fcfb0ee0b1cab5f02d2b09a341a6cd4d9e7aa81ed06f4',
        handoff=dict(path=str(FUSED_RELEASE/'handoff.json'),sha256='e482fab64d721d928c26bf3a0fde5634ec5d3d6234e9530fbb8a87cffdac50b8'),
        qualification=dict(path=str(ct.ROOT/'target/validation/pair-swiglu-fused-validation-20261011-01/ready-release-builder.json'),
            sha256='fe77dcb66142f370d4de7445265c4cb8db50d52e863eb88eedb8d80bf6014c43'),
        before_sha256='7e1c23a88143cb6455e3c6e10bfe43a567fc97b5fd45745800eae3a1e39a0e73',production_delta_counts=(9,1,0)),
    # Artifact/handoff/delta pins are deliberately absent until the actual builder receipt is reviewed.
    'routebatch': {'qualification': {'path': '/home/admin/luhuanbing.lhb/Ferrule/target/validation/route-batch-incremental-20261012-01/ready-release-builder.json', 'sha256': '4d870c9aadd427fc5a52eda03de474ed0b5bc9c01e5f1a2f9741810115430b0b'}, 'before_sha256': 'f6a65e1b6f09c0a8385fcfb0ee0b1cab5f02d2b09a341a6cd4d9e7aa81ed06f4', 'binary_sha256': 'e674eda31c0b1379e19afbd1446662278d98047c27b77c0f1233670e601c87f7', 'handoff': {'path': '/home/admin/luhuanbing.lhb/Ferrule/target/validation/route-batch-release-20261012-01/handoff.json', 'sha256': '955c4d0f9d4ac111bc2dcbf46ebc51612114d97c164606598903c90424065c4a'}, 'production_delta_counts': (15, 12, 0), 'source_relation': {'path': '/home/admin/luhuanbing.lhb/Ferrule/target/validation/route-batch-release-20261012-01/sourceCPUdelta.json', 'sha256': '9a6a1cbe09f10e5b64017fb6ca40111665591f9636e6dc8f6427d2058cfc52b4'}},
}


STARTUP_RELEASE = ct.ROOT/'target/validation/storage-paper-current-release-20261013-01'
REVIEWED_FUSED_CASES['startup_metadata'] = dict(
    binary_sha256='36e6aaa54d1aff7c8cc35d63dcee0df1cad91fcc8f6d3e746369504f2c70a58c',
    handoff=dict(path=str(STARTUP_RELEASE/'handoff.json'),sha256='b782207cc06f7317e1a28498afd4d76134c439e544370cbba50b260730245444'),
    qualification=ROUTEBATCH_READY, before_sha256=REVIEWED_FUSED_CASES['routebatch']['binary_sha256'],
    production_delta_counts=(7,0,0), source_relation=dict(path=str(STARTUP_RELEASE/'sourceRelationProof.json'),
        sha256='a5eb95ce2a9d7814475b119e31caaeb828f26b62e8b77fab11c4ac57d413eb2e'))


def startup_metadata_sources():
    import hashlib, json
    h = pre.checked_document(reviewed_fused_case('startup_metadata')['handoff'])
    proof = pre.checked_document(reviewed_fused_case('startup_metadata')['source_relation'])
    sources = pre.checked_document(h['source_relation']['source_snapshot_manifest'])
    production = pre.checked_document(h['source_relation']['production_pins'])
    before = pre.release_sources(pre.checked_document(proof['baseline_source_pins']))
    allowed = {str(ct.ROOT/p) for p in h['source_relation']['behavior_paths']}
    hook = proof['cfg_test_only_mod_rs']; allowed.add(hook['path'])
    delta = proof['raw_production_delta']
    ct.require(len(sources) == 3255 and len(production) == 597 and set(before) == set(production)
        and len(allowed) == 7 and set(delta['modified']) == allowed and not delta['added'] and not delta['removed']
        and {p for p in before if before[p] != production[p]} == allowed
        and all(before[p] == row['before'] and production[p] == row['after'] for p,row in delta['modified'].items())
        and (proof['composition_delta_count'],proof['checkpoint_representation_delta_count'],proof['production_behavior_delta_count']) == (3,3,6),
        'Startup metadata fixed seven-path relation differs')
    ct.require(all(v['byte_exact'] is True and v['baseline_digest'] == v['current_digest']
        for v in proof['unchanged_production_scopes'].values()), 'Execution/native/runtime source scope changed')
    suffix = '\n#[cfg(test)]\n#[path = "tests/read_plan_shared_path.rs"]\nmod read_plan_shared_path_tests;\n'
    text = Path(hook['path']).read_text()
    ct.require(text.endswith(suffix) and hashlib.sha256(text[:-len(suffix)].encode()).hexdigest() == hook['before'],
               'Checkpoint hook is not the exact cfg(test)-only suffix')
    ct.require(hashlib.sha256(json.dumps(sources,sort_keys=True,separators=(',',':')).encode()).hexdigest() == proof['source_digest'],
               'Sealed 3255-source generation differs')
    ct.check_pins(sources); ct.check_pins(production)
    current = pre.current_release_sources()
    ct.require(all(sources.get(p) == v for p,v in {**production,**current}.items())
        and set(current)-set(production) == {proof['new_cpu_test_file']['path']},
        'Current production inputs differ from exact release')
    return sources, current


def audit_startup_metadata(link):
    import paired, decoded_cache_release, json, tomllib
    reviewed = reviewed_fused_case('startup_metadata')
    ct.require(link == reviewed['handoff'], 'Unreviewed startup metadata handoff')
    h = pre.checked_document(link)
    ct.require(h['schema'] == 'storage-paper-current-release-handoff-v1' and h['sealed'] is True
        and h['build']['release_invocations'] == 1 and h['build']['unknown_nccl_enabled'] is False,
        'Unsealed startup metadata build/schema')
    pins = {link['path']:link['sha256']}
    for item in h['receipts'].values():
        ct.require(Path(item['path']).resolve().is_relative_to(STARTUP_RELEASE), 'Receipt outside fixed release')
        ct.check_pins({item['path']:item['sha256']}); pins[item['path']] = item['sha256']
    def read(name):
        path = str(STARTUP_RELEASE/name)
        ct.require(path in pins, 'Mandatory current builder receipt missing: '+name)
        return ct.load(path)
    frozen, sources = startup_metadata_sources()
    relation = read('qualification-relationship.json')
    ct.require(relation['quality_ready'] == reviewed['qualification'] and relation['historical_gpu_quality_rerun'] is False
        and relation['historical_receipt_pin_count'] == 172 and relation['arithmetic_bodies_unchanged_vs_e674'] is True,
        'Inherited qualification/current relation differs')
    route = audit_routebatch_ready(reviewed['qualification'], startup_metadata=True)
    q = route['ready']; ct.check_pins(q['pins']); pins.update(q['pins']); pins[reviewed['qualification']['path']] = reviewed['qualification']['sha256']
    artifact = read('release-artifact.json')
    ct.require(artifact['path'] == str(STARTUP_RELEASE/'ferrule-storage-paper-current')
        and artifact['sha256'] == reviewed['binary_sha256'] and ct.sha(artifact['path']) == artifact['sha256']
        and Path(artifact['path']).stat().st_size == artifact['bytes'] == 25409968
        and artifact['independent_copy'] is True and artifact['compiler_artifact']['profile']['test'] is False,
        'Exact current serving artifact differs')
    state_link = h['artifact']['release_receipt']; state_path = ct.ROOT/state_link['path']
    ct.require(ct.sha(state_path) == state_link['sha256'], 'Release command receipt changed'); pins[str(state_path)] = state_link['sha256']
    state = ct.load(state_path)
    ct.require(state['passed'] is True and state['returncode'] == 0 and state['timed_out'] is False
        and state['source_stable'] is True and state['source_digest_before'] == state['source_digest_after'] == artifact['source_digest']
        and state['environment']['CUDA_VISIBLE_DEVICES'] == '' and state['environment']['FERRULE_CUDA_ARCH'] == 'sm_86',
        'Current bounded/offline build failed or source generation changed')
    ct.require(ct.sha(state['log']) == state['log_sha256'], 'Actual release log changed'); pins[state['log']] = state['log_sha256']
    cpu = read('cpu-test-summary.json')
    ct.require((cpu['actual_unique_passed'],cpu['failed'],cpu['ignored']) == (111,0,4)
        and all(read(n)['passed'] is True and read(n)['returncode'] == 0 for n in ('clippy-libs-strict.json','clippy-cli-strict.json')),
        'Current CPU/strict CLI/library checks failed')
    dep = read('dependency-reverification.json'); auth = pre.checked_document(dep['authenticated_entry'])
    lock = tomllib.loads((ct.ROOT/'Cargo.lock').read_text()); checksums = {(p['name'],p['version']):p['checksum'] for p in lock['package'] if 'checksum' in p}
    ct.require(dep['packages'] == 217 and dep['actual_source_files'] == 9855 and dep['cargo_lock_sha256'] == ct.sha(ct.ROOT/'Cargo.lock')
        and dep['network_used'] is False and dep['generated_metadata_overwritten'] is False, 'Current dependency identity differs')
    manifests = set(); count = 0
    for package in auth['packages']:
        ct.require(ct.sha(package['archive_path']) == package['archive_sha256'] == checksums[(package['name'],package['version'])], 'Dependency archive/lock mismatch')
        ct.check_pins(package['source_files']); count += len(package['source_files']); manifests.add(package['canonical_manifest_path'])
    compiled = read('actual-compiled-dependencies.json')
    ct.require(count == 9855 and compiled['count'] == 158 and compiled['offline'] is True,
        'Actual compiled dependency coverage differs')
    native = read('native-archives-verification.json'); ct.check_pins(native['archives'])
    ct.require(native['native_archive_count'] == 3 and native['same_paths_and_hashes_as_e674'] is True, 'Linked native archives differ')
    libraries = read('runtime-library-verification.json'); runtime = {p:v['sha256'] for p,v in libraries['libraries'].items()}
    ct.require(libraries['count'] == 12 and libraries['all_match_route_batch'] is True, 'Current runtime libraries differ'); ct.check_pins(runtime)
    for key,value in h['preserved_baselines'].items():
        if isinstance(value,dict):
            path = str(ct.ROOT/value['path']); ct.check_pins({path:value['sha256']}); pins[path] = value['sha256']
    quality = dict(schema=QUALITY_SCHEMA,status='pass',kind='authenticated_test_binary_not_serving',qualification=reviewed['qualification'],
        qualification_schema=q['schema'],artifact_pins=pins,numerical_source_pins=pre.numerical_sources(sources),numerical_mode=route['numerical_mode'],
        source_relation=reviewed['source_relation'],new_binary_gpu_numerical_validation=False,
        observability='Inherited 37 calls/14154240 exact logits only; new binary CPU metadata/IO qualified; actual S1/S2 required, not new numerical equivalence',
        fairmatch=False,exact_speedup=None)
    return dict(binary_path=artifact['path'],binary_sha256=artifact['sha256'],source_pins=sources,runtime_pins=runtime,evidence_pins=pins,quality=quality,
        serving_scope=dict(paired.CACHE_WORKSPACE_SCOPE),decoded_cache=dict(decoded_cache_release.CACHE),optimization='startup_metadata',
        build_custody='Exact sealed current executable/3255 sources/seven-path relation/three native archives/current dependencies; real TP2 S1/S2 pending')


def reviewed_fused_case(optimization):
    case = REVIEWED_FUSED_CASES.get(optimization)
    ct.require(case is not None, 'Unknown reviewed fused-after optimization')
    ct.require({'binary_sha256','handoff','production_delta_counts'} <= case.keys(),
               'Pending actual reviewed release artifact/handoff/source relation: '+optimization)
    return case


def routebatch_source_pins(q, mode, startup_metadata=False):
    import re, difflib
    d = pre.checked_document(reviewed_fused_case('routebatch')['source_relation']); p = d['sourceRelationProof']
    channel = str(ct.ROOT/'crates/ferrule-model/src/transformer/hybrid_parallel/mixed_channel.rs')
    ct.require(p['actual_new_source_path'] == channel and not d['blockers'] and d['byte_exact_reverse_verified'] is True
        and d['six_new_CPU_tests_passed'] is True and d['same_qualification_source_pin'] is False, 'Unreviewed CPU source relocation')
    ct.require(ct.sha(p['qualified_old_source_path']) == p['qualified_old_sha256'] == q['source_pins'][channel]
        and ct.sha(channel) == p['actual_new_sha256'] == d['current_file_sha256']
        and ct.sha(p['new_cfg_test_file_path']) == p['new_cfg_test_file_sha256']
        and ct.sha(p['diff_path']) == p['diff_sha256'], 'CPU source relation pins differ')
    old = Path(p['qualified_old_source_path']).read_text(); new = Path(channel).read_text(); tests = Path(p['new_cfg_test_file_path']).read_text()
    names = ['permit_device','check_device_permit','device_enqueued','complete_device_payloads']
    ct.require(d['moved_methods'] == names and len(d['new_cpu_tests']) == 6, 'Private helper/test scope differs')
    def method(text, name):
        start = re.search(r'(?m)^    (?:pub\(crate\) )?fn '+name+r'\b', text)
        ct.require(start is not None, 'Relocated private helper missing')
        j = text.index('{',start.start())+1; depth = 1
        while depth:
            depth += (text[j]=='{')-(text[j]=='}'); j += 1
        return start.start(), j, text[start.start():j]
    spans = []
    for name in names:
        a,b,body = method(old,name); _,_,moved = method(tests,name)
        ct.require(body.replace('pub(crate) ','') == moved, 'Relocated private helper body changed'); spans.append((a,b))
    retained = old
    for a,b in reversed(spans):retained = retained[:a]+retained[b:]
    suffix = '\n// The current leaf uses host exchange; the unbound adapter is test-only.\n#[cfg(test)]\n#[path = "mixed_channel_device_tests.rs"]\nmod device_tests;\n'
    ct.require(new.endswith(suffix), 'Device helper is not cfg(test)-only')
    lines = lambda text: [line for line in text.splitlines() if line.strip()]
    ct.require(lines(retained) == lines(new[:-len(suffix)]), 'Production Host behavior source changed')
    diff = ''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='qualification/mixed_channel.rs',tofile='current/mixed_channel.rs'))
    diff += ''.join(difflib.unified_diff([],tests.splitlines(True),fromfile='/dev/null',tofile='current/mixed_channel_device_tests.rs'))
    ct.require(diff == Path(p['diff_path']).read_text(), 'Authenticated reverse source diff differs')
    sources = dict(q['source_pins']); sources[channel] = p['actual_new_sha256']
    mode_pins = dict(mode['pins']); mode_pins[channel] = p['actual_new_sha256']
    if startup_metadata:
        frozen, _ = startup_metadata_sources()
        h = pre.checked_document(reviewed_fused_case('startup_metadata')['handoff'])
        relation = pre.checked_document(h['receipts']['qualification-relationship.json'])
        for original, actual in ((q['source_pins'], sources), (mode['pins'], mode_pins)):
            for path, digest in original.items():
                if path in relation['quality_source_delta']:
                    row = relation['quality_source_delta'][path]
                    ct.require(digest == row['qualification'] and frozen[path] == row['current'], 'Fixed qualification transfer mismatch')
                    actual[path] = row['current']
                else:
                    ct.require(path not in frozen or digest == frozen[path], 'Unreviewed qualification source delta')
    ct.check_pins(sources); ct.check_pins(mode_pins)
    return sources


def routebatch_dependencies(read):
    import tomllib, json
    d = read('dependency-authentication.json'); compiled = read('actual-compiled-dependencies.json')
    ct.require(d['count'] == 217 and d['verified_files'] == 9855 and d['network_used'] is False
        and d['source_files_overwritten'] is False and d['restored_generated_metadata_count'] == 0
        and d['cargo_lock_sha256'] == ct.sha(ct.ROOT/'Cargo.lock'), 'Actual dependency authentication differs')
    lock = tomllib.loads((ct.ROOT/'Cargo.lock').read_text())
    checksums = {(p['name'],p['version']):p['checksum'] for p in lock['package'] if 'checksum' in p}
    count = 0; manifests = set()
    for package in d['packages']:
        ct.require(package['archive_matches_actual_Cargo_lock'] is True and package['all_actual_source_files_equal_archive'] is True
            and ct.sha(package['archive_path']) == package['archive_sha256'] == checksums[(package['name'],package['version'])], 'Archive/lock identity differs')
        ct.check_pins(package['source_files']); count += len(package['source_files']); manifests.add(package['canonical_manifest_path'])
    ct.require(count == 9855 and compiled['registry_package_count'] == 158
        and compiled['all_registry_compiler_artifacts_authenticated'] is True, 'Actual compiled dependency coverage differs')
    ct.require(all(v['manifest_path'] in manifests for k,v in compiled['cargo_compiler_artifact_packages'].items() if k.startswith('registry+')), 'Unauthenticated compiled registry manifest')
    actual = set()
    for line in Path(read('release.json')['log']).read_text().splitlines():
        if line.startswith('{'):
            event = json.loads(line)
            if event.get('reason') == 'compiler-artifact':actual.add(event['package_id'])
    ct.require(actual == set(compiled['cargo_compiler_artifact_packages']), 'Compiler-artifact receipt/log differs')
    return d['historical_coverage_gap']


def audit_routebatch_ready(link, startup_metadata=False):
    """Exact Host-mode qualification transfer; compiled disabled NCCL is not budget acceptance."""
    import re
    ct.require(link == ROUTEBATCH_READY, 'Unreviewed routebatch ready pin')
    q = pre.checked_document(link)
    ct.require(q['schema'] == 'ferrule.route-batch.incremental-validation.v1' and q['ready_for_release_builder'] is True
        and (q['small_gates'],q['e2e_gates'],q['e2e_calls'],q['full_vocabulary_logits']) == (8,7,37,14154240)
        and q['all_exact'] is True and q['nccl_provider_enabled'] is False
        and q['cache_arena_bytes'] == 3 << 30 and q['device_cap_bytes'] == 22528 << 20, 'Routebatch qualification/mode/scope differs')
    ct.require(q['maximum_independent_real_route_count_covered'] == 40
        and q['model_R64_independent_partial_fixture_available'] is False and q['model_R64_numeric_acceptance_claim'] is False
        and q['synthetic_R64_route_root_budget_cpu_passed'] is True and q['real_header_maxrows8_full_budget_passed'] is True,
        'Routebatch R64/public Rows8 coverage overclaim or missing budget gate')
    rows = q['rows']
    ct.require(len(rows) == 15 and all(r['passed'] is True for r in rows)
        and sum(r['gate'].startswith('backend-') for r in rows) == 3
        and sum(r['gate'].startswith('model-') for r in rows) == 5, 'Routebatch actual gate coverage differs')
    ct.check_pins(q['pins']); ct.check_pins(q['default_off_audit'])
    ct.require(len(q['default_off_audit']) == 1, 'Routebatch default-off audit missing')
    mode = ct.load(next(iter(q['default_off_audit'])))
    expected = dict(default_data_location='Host',device_participation_default=False,nccl_provider_enabled=False,
        nccl_private_device_bound='Unknown',nccl_private_host_bound='Unknown',hard_cap_claim_rejected=True,default_host_cpu_exact_passed=True)
    ct.require(all(mode.get(k) == v for k,v in expected.items()), 'Routebatch numerical mode/private NCCL bounds differ')
    partial = next(r for r in rows if r['gate'] == 'model-route-partials')
    log = Path(partial['directory'])/'test.log'
    ct.require(q['pins'].get(str(log)) == ct.sha(log), 'Actual routebatch counter log not pinned')
    pattern = r'ROUTE_BATCH_COUNTER lane=(\d+) M=(\d+) cached=(true|false) E=(\d+) R=(\d+) h2d_new=(\d+) h2d_old=(\d+) kernels_new=(\d+) kernels_old=(\d+)'
    counters = [dict(lane=int(v[0]),rows=int(v[1]),cached=v[2]=='true',experts=int(v[3]),routes=int(v[4]),
        h2d_new=int(v[5]),h2d_old=int(v[6]),kernels_new=int(v[7]),kernels_old=int(v[8])) for v in re.findall(pattern,log.read_text())]
    ct.require(counters == q['true_model_counters'] and len(counters) == 32
        and all(r['h2d_old'] == r['experts']+2 and r['h2d_new'] == 1
            and r['kernels_old']-r['kernels_new'] == r['experts']-1 for r in counters), 'Routebatch actual counters differ')
    ct.check_pins({v['path']:v['sha256'] for v in q['binaries'].values()})
    sources = routebatch_source_pins(q,mode,startup_metadata)
    return dict(ready=q,numerical_mode=expected,qualification=link,source_pins=sources)


def audit_fused(link):
    """Audit actual reviewed builder schemas; never synthesize historical dependency identity."""
    import decoded_cache_release, paired, json, hashlib
    h = pre.checked_document(link)
    is_route = link == reviewed_fused_case('routebatch')['handoff']
    optimization = 'routebatch' if is_route else h.get('optimization','fused')
    reviewed = reviewed_fused_case(optimization); release = Path(reviewed['handoff']['path']).parent
    ct.require(link == reviewed['handoff'] and (h.get('status') == 'built' and h['blockers'] == [] if is_route else h.get('schema') == FUSED_SCHEMA), 'Unreviewed serving build')
    pins = {link['path']:link['sha256']}
    for name,digest in h['receipts'].items():
        path = Path(name) if Path(name).is_absolute() else release/name
        ct.require(path.resolve().is_relative_to(release) and ct.sha(path) == digest, 'Builder receipt changed: '+str(path)); pins[str(path)] = digest
    def read(name):
        path = release/name
        ct.require(str(path) in pins, 'Mandatory builder receipt missing: '+name)
        return ct.load(path)
    qualification = {k:h['qualification'][k] for k in ('path','sha256')}
    ct.require(qualification == reviewed['qualification'], 'Reviewed qualification differs')
    route = audit_routebatch_ready(qualification) if is_route else None
    q = route['ready'] if route else pre.checked_document(qualification)
    if not route:
        ct.require(q['schema'] == 'ferrule.pair-swiglu.fused-validation.v1' and q['ready_for_release_builder'] is True
            and (q['math_gates'],q['e2e_gates'],q['e2e_calls'],q['logit_values'],q['hf_compared_words'],q['hf_mismatching_words']) == (10,7,37,14154240,9216,0)
            and q['all_logits_exact'] is True, 'Fused qualification differs')
        ct.check_pins(q['source_pins'])
    ct.check_pins(q['pins']); pins.update(q['pins']); pins[qualification['path']] = qualification['sha256']
    frozen = read('source-pins-before.json'); production = read('production-source-pins-sealed.json' if route else 'production-source-pins-before.json')
    qualified = route['source_pins'] if route else q['source_pins']; sources = pre.current_release_sources()
    ct.require(all(frozen.get(p) == v for p,v in qualified.items()) and all(production.get(p) == v for p,v in sources.items()), 'Build/current/qualification source relation differs')
    ct.check_pins(frozen); ct.check_pins(production)
    if route:
        ct.require(frozen == read('source-pins-sealed.json') and h['source_atomic_digest'] == hashlib.sha256(json.dumps(frozen,sort_keys=True,separators=(',',':')).encode()).hexdigest(), 'Post-clippy source seal differs')
    delta = read('source-delta-vs-before.json' if route else 'production-delta.json')
    ct.require((len(delta['modified' if route else 'changed']),len(delta['added']),len(delta['removed'])) == reviewed['production_delta_counts'], 'Production delta differs')
    before = q['before_fused_artifact_preserved'] if route else q['old_before_fixed_artifact_preserved']
    ct.require(h['before_artifact'] == before and ct.sha(before['path']) == before['sha256'] == reviewed['before_sha256'], 'BEFORE artifact changed')
    pins[before['path']] = before['sha256']; artifact = read('release-artifact.json'); state = read('release.json'); inputs = read('build-inputs.json')
    ct.require(artifact == h['artifact'] and artifact['sha256'] == reviewed['binary_sha256'] and ct.sha(artifact['path']) == artifact['sha256']
        and Path(artifact['path']).stat().st_size == artifact['bytes'] and artifact['independent_copy'] is True and artifact['is_quality_test_binary'] is False
        and artifact['sha256'] not in {v['sha256'] for v in q['binaries'].values()}, 'Serving artifact identity differs')
    ct.require(state['status'] == 'passed' and state['returncode'] == 0 and state['environment'] == inputs['environment']
        and inputs['quality_ready_receipt'] == qualification and inputs['environment']['CUDA_VISIBLE_DEVICES'] == ''
        and inputs['environment']['FERRULE_CUDA_ARCH'] == 'sm_86' and 'release 12.4' in inputs['toolchain']['nvcc']['stdout'], 'Actual build/environment differs')
    if route:
        ct.require(h['release_invocations'] == artifact['release_invocations'] == 1 and state['timed_out'] is False
            and h['cuda']['release_profile'] == inputs['release_profile'] == pre.checked_document(inputs['previous_build_inputs'])['release_profile'], 'Single build/profile differs')
        gap = routebatch_dependencies(read)
        ct.require(read('strict-clippy-summary.json')['passed'] is True and read('cpu-owner-test-summary.json')['passed'] is True, 'Fresh CPU checks failed')
        for check in h['checks'].values():
            r = pre.checked_document(check['receipt']); ct.require(r['returncode'] == 0 and r['status'] == 'passed' and r['timed_out'] is False, 'Actual builder CPU check failed')
    else:
        ct.require(state['build_invocations'] == 1 and read('strict-clippy-summary.json')['all_passed'] is True, 'Fused build/check differs')
    if route:
        cpu_help = pre.checked_document({'path': '/home/admin/luhuanbing.lhb/Ferrule/papers/arxiv/experiments/tp2_resident/route-batch-cpu-help-20261012-01.json', 'sha256': '7fea58ede124e51d20025cefc5bb03663f549a193e004a20e3141138c48dd4d1'})
        ct.require(cpu_help['binary_sha256'] == artifact['sha256'] and cpu_help['gpu_hidden'] is True and len(cpu_help['checks']) == 3
            and all(c['returncode'] == 0 for c in cpu_help['checks']), 'Actual CPU help/loader follow-up failed')
        pins['/home/admin/luhuanbing.lhb/Ferrule/papers/arxiv/experiments/tp2_resident/route-batch-cpu-help-20261012-01.json'] = '7fea58ede124e51d20025cefc5bb03663f549a193e004a20e3141138c48dd4d1'
    else:
        ct.require(all(read(n)['returncode'] == 0 for n in ('release-help.json','release-serve-help.json','release-ldd.json')), 'Release help/loader check failed')
    libraries = read('runtime-library-pins.json')
    ct.require(len(libraries['libraries']) == 12 and (libraries['all_12_BEFORE_library_hashes_match'] is True if route else libraries['all_12_match_before'] is True), 'Runtime libraries differ')
    runtime = {p:v['sha256'] for p,v in libraries['libraries'].items()}; ct.check_pins(runtime)
    scope = dict(paired.CACHE_WORKSPACE_SCOPE)
    if route:
        ct.require(h['limits'] == dict(batch_tokens=8,cache_arena_bytes=3<<30,context=128,device_cap_bytes=22528<<20,logging='info',profiler=False,rankperf=False,sequences=2,tp=2,workspace_bytes=4<<30)
            and h['nccl']['default_disabled'] is True and h['nccl']['private_device_bound'] == h['nccl']['private_host_bound'] == 'Unknown'
            and all(h[k] is False for k in ('new_performance_measured','server_started','gpu_workload_started','simple_release_e2e_performed')), 'Mode/budget or unearned runtime claim')
    else:
        ct.require(h['serving_scope'] == scope and h['cache_arena_bytes'] == 3<<30 and all(h[k] is False for k in ('release_is_quality_test_binary','release_smoke_passed','performance_measured','server_started','gpu_workload_started')), 'Fused scope/claims differ')
    quality = dict(schema=QUALITY_SCHEMA,status='pass',kind='authenticated_test_binary_not_serving',qualification_schema=q['schema'],qualification=qualification,
        coverage=h['qualification_coverage'],artifact_pins=pins,numerical_source_pins=pre.numerical_sources(sources),
        observability='37 cache-on no-observer calls/14154240 exact logits; not internal 2178 states, independent real-model R64 or this serving binary',fairmatch=False,exact_speedup=None)
    if route:quality.update(numerical_mode=route['numerical_mode'],source_relation=reviewed['source_relation'],historical_dependency_identity=gap)
    return dict(binary_path=artifact['path'],binary_sha256=artifact['sha256'],source_pins=sources,runtime_pins=runtime,evidence_pins=pins,quality=quality,
        serving_scope=scope,decoded_cache=dict(decoded_cache_release.CACHE),optimization=optimization,build_custody='Actual fixed builder receipts/current dependency authentication; authenticated CPU source relation; same-binary S1/S2 still required')


def exact_comparison(report, full40=False):
    ct.require(report.get('status') == 'pass' and report.get('passed') is True, 'Numerical comparator failed')
    groups = report['cases'] if full40 else report['calls']
    if full40:
        ct.require(set(groups) == {'hello','capital'}, 'Full40 case coverage differs')
    for group in groups.values():
        ct.require(len(group['metrics']) == (2178 if full40 else 1089)
            and all(m['exact'] is True and m['failures'] == 0 and m['max_abs'] == 0 and m['rmse'] == 0
                    for m in group['metrics'].values()), 'Numerical metric is not exact/missing')
    return (4 if full40 else len(groups)), sum(len(g['metrics']) for g in groups.values())


def audit(link):
    handoff = pre.checked_document(link)
    if link == reviewed_fused_case('startup_metadata')['handoff']:
        return audit_startup_metadata(link)
    if link == reviewed_fused_case('routebatch')['handoff']:
        return audit_fused(link)
    if handoff.get('schema') == FUSED_SCHEMA:
        return audit_fused(link)
    import decoded_cache_release
    import allocator_v3_release
    if handoff.get('schema') == allocator_v3_release.SCHEMA:
        return allocator_v3_release.audit(link)
    if handoff.get('schema') in decoded_cache_release.SCHEMAS:
        return decoded_cache_release.audit(link)
    ct.require(handoff.get('schema') == SCHEMA and Path(link['path']).resolve() == RELEASE/'handoff.json',
               'Unreviewed optimized handoff schema/location')
    pins = {}
    def pin(path, expected=None):
        path = Path(path)
        path = path if path.is_absolute() else ct.ROOT/path
        digest = ct.sha(path)
        ct.require(expected is None or digest == expected, 'Optimized evidence changed: '+str(path))
        pins[str(path)] = digest
        return path
    def read(path, expected=None):
        return ct.load(pin(path,expected))
    pin(link['path'], link['sha256'])
    state = read(RELEASE/'release-state.json')
    inputs = read(RELEASE/'build-inputs.json')
    expected_argv = ['cargo','build','--locked','--offline','--release','-p','ferrule-cli',
        '--no-default-features','--features','cuda','--bin','ferrule','--target-dir',
        'target/catalog-tp2-cuda','--message-format=json-render-diagnostics','-vv']
    ct.require(state == handoff['release_state'] and state['status'] == 'complete'
        and state['returncode'] == 0 and state['retries'] == 0 and state['source_pins_unchanged'] is True
        and state['argv'] == expected_argv and state['terminal_timeout_ms'] == 480000,
        'Optimized release build failed/changed')
    ct.require(inputs['release_profile'] == dict(opt_level=3,lto='fat',codegen_units=1,panic='unwind',strip='symbols'),
               'Optimized artifact is not the declared CUDA release profile')
    ct.require(inputs == handoff['build_inputs'] and state['environment'] == inputs['environment']
        and inputs['environment']['CUDA_VISIBLE_DEVICES'] == ''
        and inputs['environment']['NVIDIA_VISIBLE_DEVICES'] == 'none'
        and handoff['server_started'] is False and handoff['gpu_workload_started'] is False,
        'Build environment/CPU-hidden provenance differs')
    before = read(RELEASE/'production-source-pins-before.json',handoff['production_pins_before_sha256'])
    after = read(RELEASE/'production-source-pins-after.json',handoff['production_pins_after_sha256'])
    ct.require(before == after, 'Production source changed during build')
    sources = {name:digest for name,digest in pre.release_sources(before).items()
               if Path(name).suffix != '.md'}
    ct.require(sources == pre.current_release_sources(), 'Release production/build inputs changed')
    ct.check_pins(sources)
    dependencies = read(RELEASE/'actual-release-dependencies.json')
    # Includes actual compile-time doc inputs, not arbitrary adjacent paper files.
    for name, digest in dependencies.items(): pin(name,digest)
    relation = read(RELEASE/'quality-pin-relation.json')
    read(RELEASE/'actual-dependency-quality-relation.json')
    summary = read(QUALITY/'acceptance-summary.json',handoff['quality']['summary_sha256'])
    frozen = read(QUALITY/'frozen.json',handoff['quality']['frozen_sha256'])
    ct.require(relation == handoff['quality']['production_pin_relation']
        and relation['production_mismatches'] == [] and relation['production_pins_compared'] == 2415
        and relation['quality_summary_sha256'] == pins[str(QUALITY/'acceptance-summary.json')]
        and relation['frozen_sha256'] == pins[str(QUALITY/'frozen.json')], 'Quality/build source relation differs')
    for filename,key in [('build/sources-before.json','quality_build_sources_before_sha256'),
                         ('build/sources-after.json','quality_build_sources_after_sha256')]:
        read(QUALITY/filename,relation[key])
    ct.require(pre.numerical_sources(sources) == pre.numerical_sources(frozen['sources']),
               'Optimized numerical sources not the freshly tested ones')
    for name,digest in handoff['quality']['verified_receipts'].items():pin(name,digest)
    ct.require(summary['schema'] == 'ferrule.production-opt-quality-summary.v1'
        and (summary['successful_gpu_gates'],summary['quality_calls'],summary['quality_metrics'],summary['failures']) == (19,37,40293,0)
        and summary['all_exact'] is True and summary['max_abs'] == 0
        and len(summary['rows']) == 19 and summary['source_set_unchanged'] is True,
        'Optimized fresh quality closure differs')
    rows=[]; calls=metrics=0
    binaries={item['sha256'] for item in frozen['binaries'].values()}
    for item in frozen['binaries'].values():pin(item['path'],item['sha256'])
    for row in summary['rows']:
        directory = QUALITY/row['run']
        prepared = read(directory/'prepared.json')
        result = read(directory/'result.json',row['result_sha256'])
        supervisor = read(directory/'supervisor.json')
        auth = read(directory/'authentication.json')
        release = read(directory/'release.json')
        exit_receipt = read(directory/'test-exit.json')
        ct.require(ct.successful_process(supervisor) and result['passed'] is True
            and result['echild'] is True and result['all_eight_idle'] is True
            and release['gpu_owner_released'] is True and release['all_eight_idle'] is True
            and auth == row['authentication'] and auth['launch_id'] == supervisor['handshake']['launch_id']
            and exit_receipt['one_test_passed'] is True and exit_receipt['returncode'] == 0,
            'Optimized gate lacks authenticated successful closure: '+row['run'])
        ct.require(pre.numerical_sources(prepared['sources']) == pre.numerical_sources(sources)
            and prepared['binary_sha256'] == row['binary_sha256'] and row['binary_sha256'] in binaries,
            'Gate test binary/source differs')
        pin(prepared['binary'],prepared['binary_sha256']);pin(directory/'test.log',exit_receipt['log_sha256'])
        for name in ('before.json','after.json'):
            ct.require(read(directory/name)['status'] == 'verified','Quality idle missing')
        if 'comparison' in row:
            comparison = row['comparison']
            report = read(comparison['report'],comparison['report_sha256'])
            n,m = exact_comparison(report,row['run']=='full40')
            ct.require((n,m)==(row['calls'],row['metrics']), 'Comparator count differs')
            canonical = Path(comparison['report'].replace('-comparison.json','-canonical'))/'manifest.json'
            pin(canonical,comparison['actual_sha256'])
            ct.require(report['actual_manifest_sha256'] == comparison['actual_sha256']
                and report['reference_manifest_sha256'] == comparison['reference_sha256'], 'Comparator manifest binding differs')
            calls+=n;metrics+=m
        rows.append(dict(run=row['run'],test_binary_sha256=row['binary_sha256'],launch_id=auth['launch_id']))
    ct.require((calls,metrics)==(37,40293), 'Audited optimized quality totals differ')
    failed = read(QUALITY/summary['failed_attempt']['run']/'result.json')
    ct.require(failed['passed'] is False, 'Historical failed attempt must remain failed')
    artifact = handoff['artifact']
    binary = pin(artifact['preserved_path'],artifact['sha256'])
    ct.require(binary.stat().st_size == artifact['bytes'] and artifact['sha256'] not in binaries,
               'Serving release is not a distinct build artifact')
    runtime={}
    for name,item in handoff['dynamic_library_pins'].items():
        path=Path(name)
        ct.require(str(path.resolve()) == item['resolved'] and path.stat().st_size == item['bytes'], 'Runtime library identity differs')
        pin(path,item['sha256']);runtime[str(path)]=item['sha256']
    pin(RELEASE/'release-help.txt',handoff['release_help']['sha256'])
    pin(RELEASE/'ldd.txt',handoff['ldd']['sha256'])
    ct.require(handoff['release_help']['returncode']==0 and handoff['ldd']['missing_libraries']==[], 'Release help/link evidence failed')
    quality=dict(schema=QUALITY_SCHEMA,status='pass',kind='authenticated_test_binary_not_serving',
        calls=calls,metrics=metrics,rows=rows,numerical_source_pins=pre.numerical_sources(sources),
        artifact_pins=pins,old_failed_attempt_preserved=True,
        observability=summary['observability'],fairmatch=False,exact_speedup=None)
    return dict(binary_path=str(binary),binary_sha256=artifact['sha256'],source_pins=sources,
        runtime_pins=runtime,evidence_pins=pins,quality=quality,
        build_custody='builder completion/environment/source/binary hashes; raw Cargo terminal output not on disk; no authenticated build supervisor')


def validate(m):
    evidence=audit(m['build_receipt'])
    quality=pre.checked_document(m['test_quality'])
    ct.require(quality==evidence['quality'] and m['source_pins']==evidence['source_pins']
        and m['runtime_pins']==evidence['runtime_pins'] and m['binary']['sha256']==evidence['binary_sha256']
        and m['build_evidence_pins']==evidence['evidence_pins'], 'Optimized admission identity differs')
    if quality.get('schema') != QUALITY_SCHEMA:
        ct.require(m.get('serving_scope') == evidence['serving_scope']
            and m.get('decoded_cache') == evidence['decoded_cache'], 'Cache-on serving metadata differs')
    return dict(serving_binary_sha256=m['binary']['sha256'],test_quality_sha256=m['test_quality']['sha256'],
        build_receipt_sha256=m['build_receipt']['sha256'],build_custody=evidence['build_custody'],serving_observation='pending')


def bootstrap(output, handoff, handoff_sha256=None):
    """Produce CPU admission only. HTTP qualification is deliberately downstream."""
    import copy
    import paired
    paired.strict_gate()
    output=output.resolve()
    ct.require(output.is_relative_to(pre.HERE/'admission') and not output.exists(), 'Fresh paper admission directory required')
    link=dict(path=str(handoff.resolve()),sha256=ct.sha(handoff))
    if pre.ct.load(handoff).get('schema') != SCHEMA:
        ct.require(handoff_sha256 == link['sha256'], 'New release requires explicit final --handoff-sha256; no mutable builder artifact')
    evidence=audit(link)
    output.mkdir(parents=True)
    binary=ct.stage_binary(evidence['binary_path'],output)
    quality_path=output/'model-runtime-quality.json'
    ct.write(quality_path,evidence['quality'])
    m=copy.deepcopy(ct.load(pre.HERE/'admission/tp2-serving-smoke-admission.json')['engines']['ferrule'])
    m.update(binary=dict(path=binary,sha256=evidence['binary_sha256'],kind='public_serving'),
        spec=paired.ferrule_spec(binary),source_pins=evidence['source_pins'],runtime_pins=evidence['runtime_pins'],
        test_quality=dict(path=str(quality_path),sha256=ct.sha(quality_path)),build_receipt=link,
        build_evidence_pins=evidence['evidence_pins'],gates={},model=paired.model_contract())
    if 'serving_scope' in evidence:
        m.update(serving_scope=evidence['serving_scope'],decoded_cache=evidence['decoded_cache'],
                 spec=paired.ferrule_spec(binary,evidence['serving_scope']))
    if evidence.get('optimization') in ('routebatch','startup_metadata'):
        m['optimization'] = evidence['optimization']
    m.pop('serving_smoke',None)
    admission=dict(protocol=pre.PROTOCOL,engines=dict(ferrule=m))
    if evidence.get('optimization') == 'startup_metadata':
        admission['proc_timeline'] = paired.baseline.observer.PROC_TIMELINE_POLICY
    paired.validate_admission(admission,'serving_smoke')
    path=output/'smoke-admission.json';ct.write(path,admission)
    return path


def promote(smoke_admission, smoke_receipt):
    """After same-release smoke, reuse V pins and emit pair/profile admissions."""
    import copy
    import paired
    paired.strict_gate()
    admission=ct.load(smoke_admission)
    paired.validate_admission(admission,'serving_smoke')
    m=copy.deepcopy(admission['engines']['ferrule'])
    m['serving_smoke']=dict(path=str(smoke_receipt.resolve()),sha256=ct.sha(smoke_receipt))
    smoke=pre.validate_smoke(m)
    # Only source-qualified distinctions, not numerical HTTP observation.
    m['arithmetic'].update(effective_evidence='same-release S1/S2 profile/precision logs and visible canaries; kernel closure unobserved',
        dequant_rounding='test-binary BF16 boundaries exact; not HTTP logits; not Marlin arithmetic')
    first=Path(smoke['lifecycles'][0]['directory'])/'session/server.log'
    m['transport'].update(observed_channels='local thread TP2 CPU collective, source-qualified; no per-collective trace or NCCL claim',
                          evidence_sha256=ct.sha(first))
    old=ct.load(pre.HERE/'admission/tp2-paired-admission.json')
    m['backend_differences']=copy.deepcopy(old['engines']['ferrule']['backend_differences'])
    if m.get('optimization') == 'startup_metadata':
        output=smoke_admission.parent
        ordinary=dict(protocol=paired.PROTOCOL,engines={'ferrule':m},proc_timeline=paired.baseline.observer.PROC_TIMELINE_POLICY)
        paired.validate_admission(ordinary,'fused_after')
        path=output/'fused-after-admission.json'
        ct.require(not path.exists(), 'No admission overwrite')
        ct.write(path,ordinary)
        return path
    pair=dict(protocol=paired.PROTOCOL,engines=dict(ferrule=m,vllm=old['engines']['vllm']))
    profile=dict(protocol=pre.PROFILE_PROTOCOL,engines=dict(ferrule=m))
    if m.get('decoded_cache'):
        import profile_2025
        profile['profiler']=dict(profile_2025.TOOL_CONFIG)
    paired.validate_admission(pair,'paired');paired.validate_admission(profile,'profiling')
    output=smoke_admission.parent
    ct.require(not (output/'paired-admission.json').exists() and not (output/'profile-admission.json').exists(), 'No admission overwrite')
    ct.write(output/'paired-admission.json',pair);ct.write(output/'profile-admission.json',profile)
    return output


def main(argv=None):
    import argparse
    ct.require_cpu_python()
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('mode',choices=('audit','audit-qualification','bootstrap','promote'))
    cli.add_argument('--handoff',type=Path,default=RELEASE/'handoff.json')
    cli.add_argument('--output',type=Path)
    cli.add_argument('--handoff-sha256',help='Final owner-reviewed handoff hash required for decoded-cache/arena bootstrap; no GPU acceptance')
    cli.add_argument('--smoke-admission',type=Path)
    cli.add_argument('--smoke-receipt',type=Path)
    args=cli.parse_args(argv)
    if args.mode=='audit-qualification':
        import decoded_cache_release
        import json
        ct.require(args.handoff_sha256 is not None, 'Explicit qualification --handoff-sha256 required')
        quality=decoded_cache_release.audit_qualification(dict(path=str(args.handoff.resolve()),sha256=args.handoff_sha256))
        print(json.dumps(dict(status='cpu_audited_qualification_not_serving',quality=quality),indent=2))
    elif args.mode=='audit':
        import json
        link=dict(path=str(args.handoff.resolve()),sha256=ct.sha(args.handoff))
        if args.handoff_sha256 is not None:
            ct.require(link['sha256']==args.handoff_sha256,'Handoff hash differs')
        evidence=audit(link)
        print(json.dumps(dict(binary_sha256=evidence['binary_sha256'],
            quality=evidence['quality'],status='cpu_audited_not_gpu_accepted'),indent=2))
    elif args.mode=='bootstrap':
        ct.require(args.output is not None,'--output required')
        print(bootstrap(args.output,args.handoff,args.handoff_sha256))
    else:
        ct.require(args.smoke_admission is not None and args.smoke_receipt is not None,'Both smoke paths required')
        print(promote(args.smoke_admission,args.smoke_receipt))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
