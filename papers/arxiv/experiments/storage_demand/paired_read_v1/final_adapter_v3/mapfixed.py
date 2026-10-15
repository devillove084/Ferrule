#!/usr/bin/env python3
"""Independent map-fixed generation of the maintained final-layout V3 adapter."""
from contextlib import contextmanager
from pathlib import Path
import os
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE.parent))
from paired_read_v1 import run_v3, plan_v3 as plan, protocol as historical
from paired_read_v1.final_adapter_v3 import binding
from paired_read_v1.map_evidence_v1 import adapter as maps
from runnerhelpers.evidence_v2 import ct, require

GENERATION = 'mapfixed-20261015-01'
OUT = HERE / ('plan-' + GENERATION)
DIRECTORY = HERE / ('artifacts-v3-' + GENERATION)
PREREGISTRATION = OUT / 'preregistration.json'
SCHEMA = 'ferrule.storage-demand.independent-mapfixed-preregistration.v3'
SOURCES = ('final_adapter_v3/mapfixed.py', 'final_adapter_v3/test_mapfixed.py',
           'final_adapter_v3/MAPFIXED.md')
PRIOR = ('artifacts-current-20261015-01', 'artifacts-v3-final-20261015-01')


def preregistration_value():
    prepared = plan.validate_prepared()
    prior = []
    for name in PRIOR:
        path = HERE / name / 'paired-final.json'
        value = ct.load(path)
        require(value['status'] == 'stopped' and value['actual_issued_lifecycles'] == 1
                and value['actual_completed_lifecycles'] == 0
                and value['analysis']['denominators']['unissued'] == 120,
                'historical corrective-attempt denominators changed')
        prior.append(dict(path=str(path), sha256=ct.sha(path), status=value['status'],
            issued_lifecycles=1, completed_lifecycles=0, unissued_requests=120,
            scheduled_measured_qa=50, pooled=False, resumed=False))
    return dict(schema=SCHEMA, generation=GENERATION, artifacts_directory=str(DIRECTORY),
        correction='Readiness map evidence adapter from 7d3bee2 only; no CLI/config/design change',
        protocol_id=plan.ID, protocol_sha256=ct.sha(plan.PROTOCOL_PATH),
        cases_sha256=ct.sha(plan.CASES_PATH), order=prepared['order'],
        trial_order=prepared['trial_order'], limits=prepared['limits'],
        scheduled_blocks=5, scheduled_starts=10, scheduled_requests=120, scheduled_measured_qa=50,
        historical_attempts=prior, retries=0, replacements=0, pooling=False,
        additional_startup_probes=0, profiling_captures=0,
        artifact_pins=ct.inventory([plan.PROTOCOL_PATH, plan.CASES_PATH,
            *(HERE / name for name in ('requests.json', 'oracles.json', 'order.json', *SOURCES)),
            *(HERE / name / 'paired-final.json' for name in PRIOR)]),
        claim_limits=['Exploratory n=5; unchanged paired estimands and interval',
            'Required mapped path/hash/ELF roles, not mapped-memory byte equivalence',
            'Overlay backing identity may be unobserved; raw maps remain private',
            'NFS client deltas are not NAS backend bytes',
            'Owner-local known close is not post-drop physical release proof',
            'Actual token-ID TRACE and worker metadata remain private'])


def validate_preregistration():
    value = historical.unseal(PREREGISTRATION, SCHEMA)
    require({k: v for k, v in value.items() if k != 'digest'} == preregistration_value(),
            'independent map-fixed preregistration stale')
    ct.check_pins(value['artifact_pins'])
    return value


@contextmanager
def generation():
    """Rebind only private destinations and seals; reuse all admission/lifecycle checks."""
    original_execution = binding.execution_value
    original_directory_check = run_v3.directory_check

    def execution_value():
        validate_preregistration()
        value = original_execution()
        value.update(execution_generation=GENERATION,
                     preregistration_path=str(PREREGISTRATION),
                     preregistration_sha256=ct.sha(PREREGISTRATION))
        return value

    def directory_check(directory):
        directory = original_directory_check(directory)
        require(directory == DIRECTORY, 'only the new map-fixed artifact directory is enabled')
        return directory

    paths = dict(OUT=OUT, BINDING_PATH=OUT / 'actual-binding-v3.json',
                 REVIEW_PATH=OUT / 'actual-build-review-v3.json',
                 EXECUTION_PATH=OUT / 'execution-sealed-v3.json',
                 ADAPTER=binding.ADAPTER + '.' + GENERATION)
    changes = [(binding, key, value) for key, value in paths.items()]
    changes += [(binding._base, key, value) for key, value in paths.items() if key != 'OUT']
    changes += [(binding, 'SOURCE_FILES', binding.SOURCE_FILES + SOURCES
                 + (str(PREREGISTRATION.relative_to(HERE)),)),
        (binding, 'execution_value', execution_value),
        (binding._base, 'execution_value', execution_value),
        (run_v3, 'binding', binding), (run_v3, 'ENTRY', Path(__file__).resolve()),
        (run_v3, 'directory_check', directory_check), (run_v3.planned, 'worker', maps.worker)]
    originals = [(module, key, getattr(module, key)) for module, key, value in changes]
    try:
        for module, key, value in changes:
            setattr(module, key, value)
        yield
    finally:
        for module, key, value in reversed(originals):
            setattr(module, key, value)


def main(argv=None):
    ct.require_cpu_python()
    os.umask(0o077)
    argv = list(sys.argv[1:] if argv is None else argv)
    require(argv, 'explicit preregister, binding, or maintained run command required')
    with generation():
        if argv == ['preregister']:
            require(OUT.is_dir() and not OUT.is_symlink() and OUT.resolve() == OUT
                    and not DIRECTORY.exists(), 'new private plan directory required before start')
            value = preregistration_value()
            ct.write(PREREGISTRATION, dict(value, digest=ct.digest(value)))
            validate_preregistration()
            print('Preregistered ' + GENERATION + '; no GPU workload issued', flush=True)
            return 0
        if argv[0] in ('review-and-bind', 'seal-execution'):
            validate_preregistration()
            return binding.main(argv)
        return run_v3.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
