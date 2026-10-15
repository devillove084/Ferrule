"""Shared plotting primitives; no repository discovery or measurement-file reads."""
import hashlib
import json

ANALYSIS = 'papers/arxiv/experiments/storage_demand/paired_read_v1/analysis_v1/'
FIG = 'papers/arxiv/figures/paired-read-'
MIB, GIB = 2**20, 2**30
MODES = ('blocking', 'lookahead')
METRICS = ('request_e2e_seconds', 'ttft_seconds', 'qualified_goodput_tokens_per_second')


class EvidenceError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def digest(content):
    return hashlib.sha256(content).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + '\n').encode()
