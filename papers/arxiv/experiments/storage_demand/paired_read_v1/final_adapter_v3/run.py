#!/usr/bin/env python3
"""Run exactly the sealed V3 plan using the reviewed final-layout binding."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE.parent))
from paired_read_v1 import run_v3
from paired_read_v1.final_adapter_v3 import binding
from paired_read_v1.map_evidence_v1 import adapter as map_adapter

# The maintained subreaper must re-enter this same adapter for authenticated workers.
run_v3.binding = binding
run_v3.ENTRY = Path(__file__).resolve()
run_v3.planned.worker = map_adapter.worker

if __name__ == '__main__':
    raise SystemExit(run_v3.main())
