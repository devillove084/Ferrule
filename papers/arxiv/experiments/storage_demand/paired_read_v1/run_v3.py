#!/usr/bin/env python3
"""Explicit V3 CPU plan/check and separately gated maintained finite-arm execution."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import re
import signal
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paired_read_v1 import plan_v3 as plan, protocol as historical, current_binding_v3 as binding
from paired_read_v1 import run_paired as planned, run_actual_v2 as maintained
from runnerhelpers.evidence_v2 import ct, require, validate_schedule

ENTRY = Path(__file__).resolve()
NOTE = 'Separate renewed sole-owner authorization for independent corrective V3 paired study; no V2 resume, probes, retries or profiling'
ORIGINAL_WRITE = ct.write


def directory_check(directory):
    directory = Path(directory)
    require(directory.parent.resolve() == HERE and not directory.is_symlink()
            and re.fullmatch(r'artifacts-v3-[A-Za-z0-9_-]{1,60}', directory.name),
            'new artifacts-v3-NAME directly in paired_read_v1 required; no historical V2 resume')
    return directory


def registration(directory):
    directory_check(directory)
    value = historical.unseal(Path(directory) / 'registration.json',
        'ferrule.storage-demand.actual-paired-registration.v3')
    execution = binding.validate_execution(Path(value['execution_path']))
    candidate = binding.validate_binding(Path(value['binding_path']))
    require(value['protocol_id'] == plan.ID and value['protocol_sha256'] == ct.sha(plan.PROTOCOL_PATH)
            and value['cases_sha256'] == ct.sha(plan.CASES_PATH), 'unknown V3 registration identity')
    require(value['boot_id'] == maintained.boot_id() and value['trial_order'] == execution['trial_order']
            and value['order'] == execution['order'] and value['limits'] == execution['limits'],
            'V3 boot/order/absolute limits changed; no clock reset')
    require(value['binding_sha256'] == ct.sha(value['binding_path'])
            and value['execution_sha256'] == ct.sha(value['execution_path'])
            and value['binding_path'] == str(binding.BINDING_PATH)
            and value['execution_path'] == str(binding.EXECUTION_PATH), 'V3 candidate/workflow changed')
    require(value['schedule'] == ct.load(HERE / 'requests.json')['schedule'], 'entire V3 12-request schedule changed')
    validate_schedule(value['schedule'])
    require(value['owner_authorization'] is True and value['owner_note'] == NOTE
            and value['eight_uuids'] == candidate['uuid_inventory']
            and value['selected_uuid'] == value['eight_uuids'][0], 'V3 authorization/inventory changed')
    maintained.pilot.validate_inventory(value['eight_uuids'], value['selected_uuid'])
    return value


def versioned_write(path, value):
    # Label newly produced output correctly; never rewrite an existing V2 receipt.
    if Path(path).name == 'paired-final.json' and value.get('schema') == 'ferrule.storage-demand.paired-read-actual.v2':
        directory_check(Path(path).parent)
        value = dict(value, schema='ferrule.storage-demand.paired-read-actual.v3', protocol_id=plan.ID,
            protocol_sha256=ct.sha(plan.PROTOCOL_PATH), cases_sha256=ct.sha(plan.CASES_PATH),
            historical_resume_permitted=False)
    return ORIGINAL_WRITE(path, value)


@contextmanager
def adapter():
    """Versioned process-local specialization, no copied engine or lifecycle harness."""
    changes = ((historical, 'validate_prepared', binding.validate_execution),
        (historical, 'validate_candidate', binding.validate_binding),
        (historical, 'PROTOCOL_PATH', plan.PROTOCOL_PATH), (historical, 'CASE_PATH', plan.CASES_PATH),
        (historical, 'case_for_mode', binding.case_for_mode),
        (planned, 'ENTRY', ENTRY), (planned, 'worker_contract', maintained.contract),
        (planned, 'read_ledger_assessment', maintained.ledger),
        (maintained, 'binding', binding), (maintained, 'registration', registration),
        (maintained, 'actual_adapter', adapter), (maintained, 'NOTE', NOTE), (ct, 'write', versioned_write))
    originals = [(module, name, getattr(module, name)) for module, name, value in changes]
    try:
        for module, name, value in changes:
            setattr(module, name, value)
        yield
    finally:
        for module, name, value in reversed(originals):
            setattr(module, name, value)


def start(directory, granted):
    require(granted is True, 'separate renewed sole-GPU-owner authorization required')
    directory = directory_check(directory)
    execution = binding.validate_execution()
    candidate = binding.validate_binding()
    directory.mkdir(exist_ok=False)
    now = time.monotonic()
    value = dict(schema='ferrule.storage-demand.actual-paired-registration.v3', protocol_id=plan.ID,
        protocol_sha256=ct.sha(plan.PROTOCOL_PATH), cases_sha256=ct.sha(plan.CASES_PATH),
        started=maintained.ex.stamp(), pilot_deadline=now + execution['limits']['pilot_seconds'],
        boot_id=maintained.boot_id(), order=execution['order'], trial_order=execution['trial_order'],
        limits=execution['limits'], schedule=ct.load(HERE / 'requests.json')['schedule'],
        binding_path=str(binding.BINDING_PATH), binding_sha256=ct.sha(binding.BINDING_PATH),
        execution_path=str(binding.EXECUTION_PATH), execution_sha256=ct.sha(binding.EXECUTION_PATH),
        selected_uuid=candidate['uuid_inventory'][0], eight_uuids=candidate['uuid_inventory'],
        owner_authorization=True, owner_note=NOTE, scheduled_blocks=5, scheduled_starts=10,
        scheduled_requests=120, scheduled_measured_qa=50, retries=0, replacements=0,
        profiling_captures=0, first_arm_is_first_planned_arm=True, extra_pilot_lifecycles=0,
        historical_resume_permitted=False, physical_nas_backend_bytes=None)
    ct.write(directory / 'registration.json', dict(value, digest=ct.digest(value)))
    return dict(status='registered', protocol_id=plan.ID, directory=str(directory),
                first_arm=execution['trial_order'][0], deadline=value['pilot_deadline'])


def main(argv=None):
    ct.require_cpu_python()
    def cancelled(signum, frame):
        raise KeyboardInterrupt('owned V3 coordinator/worker cancellation')
    signal.signal(signal.SIGTERM, cancelled)
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == '--_worker':
        require(len(argv) == 2, 'authenticated V3 worker launch hash required')
        with adapter():
            return planned.worker(argv[1])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'check', 'start', 'next', 'finalize'))
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--sole-gpu-owner-granted', action='store_true')
    args = parser.parse_args(argv)
    if args.mode == 'plan':
        value = dict(plan.plan(), versioned_execution_entry=str(ENTRY), binding_pending=not binding.BINDING_PATH.exists(),
                     integration_does_not_amend_preregistration=True)
    elif args.mode == 'check':
        binding.validate_execution()
        value = dict(status='checked', protocol_id=plan.ID, gpu_queries=0, server_runs=0)
    else:
        require(args.directory is not None, '--directory required for execution')
        directory = directory_check(args.directory)
        if args.mode == 'start':
            value = start(directory, args.sole_gpu_owner_granted)
        else:
            registration(directory)
            with adapter():
                value = maintained.execute_next(directory) if args.mode == 'next' else maintained.finalize(directory)
            if args.mode == 'next':
                print(json.dumps(dict(trial=value['trial'], can_continue=value['assessment']['can_continue'],
                    measured_qa=value['summary']['measured_qa'])), flush=True)
                return 0 if value['assessment']['can_continue'] else 1
    print(json.dumps(value, sort_keys=True), flush=True)
    return 0 if value['status'] not in ('stopped', 'incomplete') else 1


if __name__ == '__main__':
    raise SystemExit(main())
