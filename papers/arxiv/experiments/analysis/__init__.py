"""Paper-local, read-only Ferrule performance analysis."""

from .cache_plan import cache_capacity, counterfactuals, expert_headers
from .performance import analyze_http, analyze_lines, analyze_log

__all__ = [
    "analyze_http", "analyze_lines", "analyze_log", "cache_capacity",
    "counterfactuals", "expert_headers",
]
