"""Explicit TP1 specialization of the maintained authenticated collector.

Private module instances keep the historical TP2 policy and timing modules intact.
Only the grant's model cardinality and finite wrapper budget differ. No copied
Popen, signal, descendant-reaping, baseline, or UUID-lock engine lives here.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import sys
from types import SimpleNamespace

STORAGE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STORAGE))
from runnerhelpers.evidence_v2 import ct, require

TP2 = ct.HERE / 'tp2_resident'
sys.path.insert(0, str(TP2))
# These are private instances, not patches to sys.modules' maintained modules.
_scope_spec = importlib.util.spec_from_file_location('_overlap_tp1_scope', TP2 / 'instrumentation_scope.py')
scope = importlib.util.module_from_spec(_scope_spec)
_scope_spec.loader.exec_module(scope)
_live_spec = importlib.util.spec_from_file_location('_overlap_tp1_live', TP2 / 'instrumentation_live.py')
live = importlib.util.module_from_spec(_live_spec)
_live_spec.loader.exec_module(live)
_capture_spec = importlib.util.spec_from_file_location('_overlap_tp1_capture', TP2 / 'instrumentation_capture.py')
capture = importlib.util.module_from_spec(_capture_spec)
_capture_spec.loader.exec_module(capture)

UUID = 'GPU-b980f22b-58a3-fc88-21c1-ae86053d8535'
SCOPE = 'storage_tp1_all_eight_instrumentation_probe_v1'


def tp1_grant_scope(inventory, selected, grant):
    require(len(inventory) == len(set(inventory)) == 8 and all(
        re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', u) for u in inventory), 'eight distinct physical UUIDs required')
    require(tuple(selected) == (UUID,) and UUID in inventory, 'explicit TP1 GPU0 binding required')
    require(grant.get('scope') == SCOPE and grant.get('granted') is True
            and grant.get('note', '').strip() and grant.get('gpu_uuids') == sorted(inventory)
            and grant.get('model_gpu_uuids') == [UUID], 'separate TP1 instrumentation grant required')
    return sorted(inventory)


scope.grant_scope = tp1_grant_scope
scope.PROBE_SECONDS = 120
live.scope = scope
live.shared = SimpleNamespace(UUIDS=(UUID,))
capture.live = live
capture.ct, capture.observer = live.ct, live.observer
_original_load_wrapper = live.load_wrapper


def bounded_wrapper():
    wrapper = _original_load_wrapper()
    run_target = wrapper.run_target

    def run(directory, command, _historical_lifetime, _historical_shutdown):
        return run_target(directory, command, 150, 45)

    wrapper.run_target = run
    return wrapper


live.load_wrapper = bounded_wrapper


def adapter_binding():
    """Catch stale imported aliases without mutating maintained module globals."""
    require(live.scope is scope and live.shared.UUIDS == (UUID,)
            and capture.live is live and capture.ct is live.ct and capture.observer is live.observer
            and live.load_wrapper is bounded_wrapper, 'private collector context binding differs')
    require(live.CollectorMonitor.start.__globals__ is vars(live)
            and capture.run.__globals__ is vars(capture), 'collector functions bound to foreign module context')
    return dict(private_scope=True, private_live=True, private_capture=True,
                capture_aliases_bound=True, maintained_wrapper='scripts/_profile_capture.py',
                lifetime_seconds=150, shutdown_seconds=45, new_execution_loop=False)


def model_identity(server, directory, deadline, authenticated):
    """The model is the gated target's child, not the nsys CLI's direct child."""
    import time
    directory = Path(directory)
    collector = live.observer.proc_identity(server.pid)
    require(collector['ppid'] == authenticated['worker']['pid'], 'collector not owned by authenticated worker')
    while time.monotonic() < deadline:
        require(server.poll() is None, 'collector exited before model identity')
        try:
            manifest = ct.load(directory / 'collector-manifest.json')
            live.live_auth(manifest)
            target = ct.load(directory / 'collector-target-ready.json')['identity']
            live.descendant(target, collector)
            report = ct.load(directory / 'profile-server.json')
            root = live.observer.proc_identity(report['pid'])
            require(root['ppid'] == target['pid'], 'model not maintained wrapper direct child')
            live.descendant(root, target)
            return target, root
        except FileNotFoundError:
            time.sleep(.05)
    raise TimeoutError('model PID/birth handshake exceeded startup budget')
