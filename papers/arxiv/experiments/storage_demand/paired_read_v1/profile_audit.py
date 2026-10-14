"""CPU-only completeness layer for optional official exports; never launches nsys.

The maintained auditor handles direct API/activity correlations. Graph launches
need explicit exported node-to-launch relations and may not be treated as lost
one-to-one direct submissions. Missing coverage is unverified, not PASS.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tp2_resident'))
import profile_trace_audit_v2 as direct
from runnerhelpers.evidence_v2 import require

GRAPH_APIS = {'cuGraphLaunch', 'cudaGraphLaunch'}


def audit(activities, api_rows, windows, model_pid, graph_expansions=None, completeness=None):
    require(isinstance(model_pid, int) and model_pid > 0, 'actual model PID required')
    graph_expansions = graph_expansions or []
    graph_apis = [r for r in api_rows if r.get('name', '').split('_v')[0] in GRAPH_APIS
                  and direct.process_id(r['globalTid']) == model_pid]
    graph_keys = {(direct.process_id(r['globalTid']), r['correlationId']) for r in graph_apis}
    issues, consumed = [], set()
    require(len(graph_keys) == len(graph_apis), 'duplicate graph launch correlation key')
    expansions = {(r.get('pid'), r.get('launch_correlation_id')): r for r in graph_expansions}
    require(len(expansions) == len(graph_expansions), 'duplicate graph expansion relation')
    graph_nodes = set()
    for launch in graph_apis:
        key = (model_pid, launch['correlationId'])
        expansion = expansions.get(key)
        if not expansion or expansion.get('official_export_relation') is not True:
            issues.append('missing official exported graph expansion: ' + str(key))
            continue
        nodes = expansion.get('activities', [])
        if launch.get('returnValue') != 0 or not nodes:
            issues.append('failed/empty graph launch expansion: ' + str(key))
            continue
        label = direct.containing_window(launch, windows)
        if label is None:
            issues.append('graph launch outside registered request window')
        for node in nodes:
            node_key = (node.get('globalPid'), node.get('kind'), node.get('start'), node.get('end'), node.get('correlationId'))
            matching = [r for r in activities if (r.get('globalPid'), r.get('kind'), r.get('start'), r.get('end'), r.get('correlationId')) == node_key]
            if len(matching) != 1 or node_key in graph_nodes or direct.process_id(node.get('globalPid', 0)) != model_pid:
                issues.append('graph node missing/duplicate/foreign process')
                continue
            if direct.containing_window(matching[0], windows) != label:
                issues.append('graph node/request window mismatch')
            graph_nodes.add(node_key)
        if expansion.get('expected_node_count') != len(nodes):
            issues.append('graph expansion node denominator unknown/mismatch')
        consumed.add(key)
    if set(expansions) != consumed:
        issues.append('unbound graph expansion relation')
    ordinary = [r for r in activities if (r.get('globalPid'), r.get('kind'), r.get('start'), r.get('end'), r.get('correlationId')) not in graph_nodes]
    result = direct.audit_correlations(ordinary, [r for r in api_rows if r not in graph_apis], windows, model_pid)
    completeness = completeness or {}
    complete = (completeness.get('schema') == 'ferrule.storage-demand.official-nsys-completeness.v1'
        and completeness.get('official_full_export') is True
        and completeness.get('collector_custody_drained') is True
        and completeness.get('clock_pid_uuid_request_bindings_verified') is True
        and completeness.get('loss_diagnostics_available') is True
        and completeness.get('dropped_events') == 0
        and completeness.get('unknown_event_loss') is False
        and completeness.get('warnings') == []
        and completeness.get('independent_collection_coverage_verified') is True)
    # This deliberately cannot manufacture collection coverage from correlated rows.
    return dict(direct_correlation=result, graph_launches=len(graph_apis), graph_nodes=len(graph_nodes),
        graph_issues=issues, official_collection_completeness_verified=complete,
        full_trace_accepted=False, scientific_acceptance=False,
        interpretation='Requires independent official coverage and collector acceptance; this layer alone never authorizes a trace or overlap claim')
