"""Tiny oracle uses the same explicit raw-logit tie policy as the 35B exporter.

No native oracle values/weights are replaced. F32 probability collisions between
unequal raw logits must keep native top-k selection.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'scripts'))
from qwen35_fp8_bf16_reference import canonicalize_router_ties
