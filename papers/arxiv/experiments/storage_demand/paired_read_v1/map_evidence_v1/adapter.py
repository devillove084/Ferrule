"""Append-only map evidence adapter; historical lifecycle sources stay pinned."""
from contextlib import contextmanager
import io
import json
from pathlib import Path
import sys

from paired_read_v1 import run_paired as planned
from runnerhelpers import execution_v2 as ex, library_maps_v1 as maps
from runnerhelpers.evidence_v2 import ct, require

ORIGINAL_WORKER = planned.worker


@contextmanager
def mapped_worker(payload):
    original_maps, original_root = ex.library_maps, ex.capture_root
    owned, observations = {}, []

    def capture_root(*args, **kwargs):
        parent, root, report = original_root(*args, **kwargs)
        owned.update(parent=parent, root=root)
        return parent, root, report

    def library_maps(root, pins):
        require(root == owned.get('root'), 'map target is not the authenticated owned server')
        candidate = ct.load(payload['candidate_path'])['candidate']
        label = 'ready' if not observations else 'after-requests'
        result = maps.library_maps(root, pins,
            directory=Path(payload['directory']) / (label + '-library-evidence'),
            parent=owned['parent'], executable=candidate['binary'],
            executable_sha256=candidate['binary_sha256'],
            runtime=ct.load(candidate['runtime_pin_manifest']), role='cli',
            identity=ex.observer.proc_identity)
        observations.append(result)
        return result

    try:
        ex.library_maps, ex.capture_root = library_maps, capture_root
        yield
    finally:
        ex.library_maps, ex.capture_root = original_maps, original_root


def worker(launch_hash):
    raw = sys.stdin.read()
    payload = json.loads(raw)
    previous = sys.stdin
    try:
        sys.stdin = io.StringIO(raw)
        with mapped_worker(payload):
            return ORIGINAL_WORKER(launch_hash)
    finally:
        sys.stdin = previous
