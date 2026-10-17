"""Pure HTTP output semantics; callers authenticate reference provenance, not this module."""
from __future__ import annotations

import copy
from pathlib import Path
import re

from .evidence_v2 import ct, integer, require

HISTORICAL_LENGTH = 'historical-length-v1'
BOUND_HTTP_REFERENCE = 'bound-http-reference-v1'
REFERENCE_SCHEMA = 'ferrule.storage-demand.http-output-references.v1'


def normalize_http_expectation(planned, *, contract=HISTORICAL_LENGTH, reference_context=None):
    """Resolve one expectation, never infer an EOS reference from observed output.

    For BOUND_HTTP_REFERENCE the caller supplies an authenticated reviewed_binding
    and independently computed source_sha256. document_sha256 is ct.digest of the
    parsed source document, not its file bytes. This checks their exact binding;
    it does not establish review authority or read/rewrite a reference file.
    """
    require(integer(planned.get('input_tokens'), 1) and integer(planned.get('max_tokens'), 1),
            'HTTP request token bounds missing/invalid')
    if contract == HISTORICAL_LENGTH:
        require(reference_context is None, 'historical contract cannot consume HTTP reference context')
        reference = planned.get('reference')
        ids = reference['reference_ids'] if reference is not None else None
        text = reference['reference_text'] if reference is not None else None
        count, reason, binding = planned['max_tokens'], 'length', None
        usage = dict(prompt_tokens=planned['input_tokens'], completion_tokens=count,
                     total_tokens=planned['input_tokens'] + count)
    else:
        require(contract == BOUND_HTTP_REFERENCE, 'unknown HTTP expectation contract')
        context = reference_context
        require(isinstance(context, dict) and set(context) == {
            'source', 'source_sha256', 'document', 'reviewed_binding'}, 'reviewed HTTP reference context required')
        binding, document = context['reviewed_binding'], context['document']
        require(isinstance(binding, dict) and set(binding) == {
            'source', 'source_sha256', 'document_sha256'}, 'reviewed HTTP source/hash binding required')
        require(isinstance(context['source'], str) and Path(context['source']).is_absolute()
            and context['source'] == binding['source']
            and isinstance(context['source_sha256'], str)
            and re.fullmatch(r'[0-9a-f]{64}', context['source_sha256'])
            and context['source_sha256'] == binding['source_sha256']
            and isinstance(binding['document_sha256'], str)
            and re.fullmatch(r'[0-9a-f]{64}', binding['document_sha256'])
            and ct.digest(document) == binding['document_sha256'], 'HTTP reference source/hash binding mismatch')
        require(isinstance(document, dict) and set(document) == {'schema', 'references'}
            and document['schema'] == REFERENCE_SCHEMA and isinstance(document['references'], dict),
            'unknown HTTP reference document')
        reference = document['references'].get(planned['name'])
        require(isinstance(reference, dict) and set(reference) == {
            'prompt', 'input_tokens', 'max_tokens', 'payload', 'reference_ids',
            'reference_text', 'finish_reason', 'usage'}, 'complete named HTTP reference required')
        require(all(ct.digest(reference[k]) == ct.digest(planned[k]) for k in
            ('prompt', 'input_tokens', 'max_tokens', 'payload')), 'HTTP reference request/payload binding mismatch')
        ids, text = reference['reference_ids'], reference['reference_text']
        require(isinstance(ids, list), 'HTTP reference IDs must be a list')
        count, reason, usage = len(ids), reference['finish_reason'], reference['usage']
    require(ids is None or (isinstance(ids, list) and len(ids) == count
        and all(integer(token) for token in ids)), 'invalid HTTP reference IDs/count')
    require((contract == HISTORICAL_LENGTH and text is None) or isinstance(text, str),
            'invalid HTTP reference text')
    require(count <= planned['max_tokens'] and reason in ('length', 'stop')
        and (reason != 'length' or count == planned['max_tokens']), 'invalid HTTP reference finish/count')
    expected_usage = dict(prompt_tokens=planned['input_tokens'], completion_tokens=count,
                          total_tokens=planned['input_tokens'] + count)
    require(isinstance(usage, dict) and set(usage) == set(expected_usage)
        and all(type(usage[k]) is int and usage[k] == v for k, v in expected_usage.items()),
        'invalid HTTP reference usage/count')
    return copy.deepcopy(dict(reference_ids=ids, reference_text=text, finish_reason=reason,
        usage=usage, actual_output_tokens=count, source_binding=binding))


def http_output_matches(planned, actual, trace, *, contract=HISTORICAL_LENGTH, reference_context=None):
    """Exact output/usage check shared by execution and analysis; malformed input fails closed."""
    try:
        expected = normalize_http_expectation(planned, contract=contract, reference_context=reference_context)
        usage = actual.get('usage')
        ids = trace.get('actual_generated_ids')
        return bool(isinstance(usage, dict) and set(usage) == set(expected['usage'])
            and all(type(usage[k]) is int and usage[k] == v for k, v in expected['usage'].items())
            and type(actual.get('actual_output_tokens')) is int
            and actual['actual_output_tokens'] == expected['actual_output_tokens']
            and actual.get('finish_reason') == expected['finish_reason']
            and (expected['reference_ids'] is None or (isinstance(ids, list)
                and all(integer(token) for token in ids) and ids == expected['reference_ids']))
            and (expected['reference_text'] is None or actual.get('text') == expected['reference_text']))
    except (ValueError, KeyError, TypeError, AttributeError):
        return False
