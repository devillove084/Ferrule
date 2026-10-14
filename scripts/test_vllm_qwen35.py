#!/opt/conda310/bin/python
"""Opt-in vLLM diagnostics, CUDA smoke, offline generation and existing-server HTTP.

No installs, environment repair, CPU inference fallback or Ferrule imports.
See test_vllm_qwen35.md. Without a mode, only `check` runs.
"""
import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import inspect
import json
import math
import os
from pathlib import Path
import re
import signal
import shlex
import subprocess
import sys
import tempfile
import uuid
import socket
import time
import urllib.parse
import urllib.request

sys.dont_write_bytecode = True
SCRIPT = Path(__file__).resolve()
MODEL = '/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8'
OUTPUT = SCRIPT.parents[1] / 'target/validation/vllm-qwen35-check'
LIMIT = 64 * 1024
NOTES = [
    'Import success is not CUDA, Qwen3.5/FP8/SM86 compatibility or oracle acceptance.',
    'Offline walltime is not HTTP TTFT. HTTP content events are not token boundaries.',
    'Precision, kernels, physical topology and shard identity remain unverified; no P-match claim.',
    'vLLM TP/EP is not Ferrule thread EP equivalence; capacity failures remain failures.',
    'Process termination/client timeout is not proof of GPU quiescence or server drain.',
]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def scrub(value):
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if not isinstance(value, str):
        return value
    # Never archive a full environment. Remove known credential values from errors/logs too.
    for key, secret in os.environ.items():
        if secret and re.search(r'(?i)token|password|secret|api.?key|credential|private.?key', key):
            value = value.replace(secret, '<redacted>')
    value = re.sub(r'-----BEGIN [^-]*PRIVATE KEY-----.*?(?:-----END [^-]*PRIVATE KEY-----|$)',
                   '<private key redacted>', value, flags=re.S)
    value = re.sub(r'(?i)(bearer\s+)[^\s\"\',;]+', r'\1<redacted>', value)
    value = re.sub(r'(?i)((?:api[_-]?key|password|secret|token)[\"\']?\s*[:=]\s*[\"\']?)[^\s\"\',;]+',
                   r'\1<redacted>', value)
    return re.sub(r'(https?://)[^/\s@]+@', r'\1<redacted>@', value)


def custody_capabilities():
    """Probe an isolated helper, never set subreaper state in this caller."""
    result = run_process(None, 1)
    ready = result.get('handshake', {})
    return dict(containment=ready.get('containment_verified') is True
                and result.get('custody', {}).get('status') == 'proven_drained'
                and not result.get('error'), method='linux-subreaper-waitid',
                probe=result, privileges='none; no cgroup/pidfd requirement')


def run_process(argv, timeout, payload=None):
    """Stable public interface; the only subprocess we own/reap here is the helper.

    Its verified handshake precedes application launch. Helper loss invalidates
    every drain claim; never chase application PIDs from this threaded caller.
    """
    started = time.monotonic()
    result = dict(returncode=None, timed_out=False, wall_seconds=None, stdout='', stderr='',
                  cleanup_errors=[], cleanup_scope='isolated_subreaper_direct_children',
                  descendant_custody='unknown', gpu_quiescence='unverified', custody=dict(status='unknown'))
    primary, process = None, None
    launch_id = uuid.uuid4().hex
    parent, child = socket.socketpair()
    with tempfile.TemporaryDirectory(prefix='ferrule-custody-') as temporary, \
            tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err, tempfile.TemporaryFile() as inp:
        report_path = Path(temporary) / 'result.json'
        if payload is not None:
            inp.write(json.dumps(payload).encode())
        inp.seek(0)
        handshake = None
        try:
            helper = SCRIPT.with_name('_owned_subreaper.py')
            process = subprocess.Popen([sys.executable, '-B', str(helper), str(child.fileno()),
                                        str(inp.fileno()), str(report_path), launch_id],
                                       stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                       pass_fds=(child.fileno(), inp.fileno()), start_new_session=True)
            child.close()
            handshake_deadline = time.monotonic() + 5
            wire = bytearray()
            while not wire.endswith(b'\n'):
                remaining = handshake_deadline - time.monotonic()
                require(remaining > 0, 'Helper handshake deadline exceeded')
                parent.settimeout(remaining)
                chunk = parent.recv(4096)
                require(chunk, 'Helper exited before containment handshake')
                wire.extend(chunk)
                require(len(wire) <= 16384, 'Oversized helper handshake')
            candidate = json.loads(wire)
            require(candidate.get('kind') == 'ready' and candidate.get('protocol') == 1
                    and candidate.get('launch_id') == launch_id
                    and candidate.get('helper_identity', {}).get('pid') == process.pid
                    and candidate.get('containment_verified') is True
                    and candidate.get('method') == 'linux-subreaper-waitid', 'Invalid helper containment handshake')
            handshake = candidate
            result['handshake'] = handshake
            journal = (str(Path(payload['directory']) / 'custody.json')
                       if isinstance(payload, dict) and 'manifest' in payload and 'directory' in payload else None)
            parent.settimeout(5)
            parent.sendall((json.dumps(dict(launch_id=launch_id, argv=argv, timeout=timeout,
                                            journal=journal)) + '\n').encode())
            process.wait(timeout=timeout + 8)
        except subprocess.TimeoutExpired as exc:
            result.update(timed_out=True, primary_error=f'{type(exc).__name__}: {exc}',
                          error='Supervisor helper deadline; custody unknown')
        except (OSError, RuntimeError, ValueError) as exc:
            result.update(primary_error=f'{type(exc).__name__}: {exc}', error='Supervisor helper failure')
        except BaseException as exc:
            primary = exc
            result['primary_error'] = f'{type(exc).__name__}: {exc}'
        finally:
            if process is not None:
                if process.returncode is None:
                    try:
                        parent.settimeout(.1)
                        parent.sendall(b'cancel\n')
                    except OSError:
                        pass
                    try:
                        process.wait(timeout=8)
                    except Exception as exc:
                        result['cleanup_errors'].append(f'Helper cleanup wait: {type(exc).__name__}: {exc}')
                        # This is our unreaped direct Popen helper only; its loss
                        # leaves application custody unknown, not permission to kill others.
                        try:
                            process.kill()
                            process.wait(timeout=2)
                        except Exception as error:
                            result['cleanup_errors'].append(f'Helper reap: {type(error).__name__}: {error}')
                try:
                    data = json.loads(report_path.read_text())
                    custody = data['custody']
                    if 'primary_error' in data:
                        result['helper_primary_error'] = data['primary_error']  # diagnostic, not authority
                    require(handshake is not None and process.returncode == 0
                            and custody.get('launch_id') == launch_id
                            and custody.get('helper_identity') == handshake.get('helper_identity'),
                            'Helper exit/identity invalidates cleanup evidence')
                    if custody.get('status') == 'proven_drained':
                        require(custody.get('echild') is True and custody.get('containment_verified') is True,
                                'Drain requires verified subreaper and kernel ECHILD')
                    result.update(returncode=data['returncode'], custody=custody,
                                  descendant_custody=custody['status'])
                    result['timed_out'] |= data['timed_out']
                    result['cleanup_errors'].extend(data['cleanup_errors'])
                    if data.get('reporting_errors'):
                        result.setdefault('reporting_errors', []).extend(data['reporting_errors'])
                    if 'primary_error' in data:
                        result.setdefault('primary_error', data['primary_error'])
                        result['helper_primary_error'] = data['primary_error']
                    if 'error' in data:
                        result.setdefault('error', data['error'])
                except Exception as exc:
                    result['cleanup_errors'].append(f'Helper evidence: {type(exc).__name__}: {exc}')
                    result['custody'] = dict(status='unknown')
                if result['cleanup_errors'] or result['custody']['status'] != 'proven_drained':
                    result.setdefault('error', 'cleanup incomplete; descendant custody unknown')
            parent.close()
            child.close()
            for name, stream in (('stdout', out), ('stderr', err)):
                try:
                    size = stream.tell()
                    stream.seek(max(0, size - LIMIT))
                    result[name] = scrub(stream.read(LIMIT).decode('utf-8', 'replace'))
                    result[name + '_truncated'] = size > LIMIT
                except Exception as exc:
                    result.setdefault('reporting_errors', []).append(f'{name}: {type(exc).__name__}: {exc}')
    result['wall_seconds'] = time.monotonic() - started
    if primary is not None:
        primary.supervisor_result = result
        raise primary
    return result


def dependencies(root, get_dist=None, packaging=None):
    """Evaluate actual installed metadata including propagated extras; never guess PEP 508."""
    get_dist = get_dist or metadata.distribution
    result = dict(status='pass', versions={}, metadata={}, missingdeps=[], errors=[], unevaluated=[])
    try:
        root_dist = get_dist(root)
    except metadata.PackageNotFoundError:
        result['missingdeps'].append(dict(name=root, reason='distribution metadata absent'))
        result['status'] = 'fail'
        return result

    if packaging is None:
        try:
            packaging = (importlib.import_module('packaging.requirements').Requirement,
                         importlib.import_module('packaging.specifiers').SpecifierSet)
        except ImportError as exc:
            result['versions'][root] = root_dist.version
            result['metadata'][root] = dict(requires_python=root_dist.metadata.get('Requires-Python'),
                                            requires_dist=list(root_dist.requires or []))
            result['unevaluated'] = list(root_dist.requires or [])
            result['errors'].append(f'packaging unavailable ({exc}); Requires-Python and Requires-Dist '
                                    'markers/versions cannot be evaluated. Install/repair packaging yourself; '
                                    'no dependency names or installation commands were guessed.')
            result['missingdeps'].append(dict(name='packaging', reason='requirement evaluator unavailable'))
            result['status'] = 'fail'
            return result
    Requirement, SpecifierSet = packaging
    pending, visited = [(root, frozenset())], set()
    while pending:
        name, extras = pending.pop()
        key = (re.sub(r'[-_.]+', '-', name).lower(), extras)
        if key in visited:
            continue
        visited.add(key)
        try:
            dist = get_dist(name)
        except metadata.PackageNotFoundError:
            result['missingdeps'].append(dict(name=name, reason='distribution metadata absent'))
            continue
        result['versions'][name] = dist.version
        raw = list(dist.requires or [])
        python_requirement = dist.metadata.get('Requires-Python')
        result['metadata'][name] = dict(requires_python=python_requirement, requires_dist=raw)
        try:
            if python_requirement and not SpecifierSet(python_requirement).contains(
                    '.'.join(map(str, sys.version_info[:3])), prereleases=True):
                result['errors'].append(f'{name} Requires-Python {python_requirement} excludes this interpreter')
            for text in raw:
                req = Requirement(text)
                if req.marker and not any(req.marker.evaluate({'extra': extra}) for extra in ('', *extras)):
                    continue
                if req.url:
                    result['unevaluated'].append(dict(parent=name, requirement=text,
                                                       reason='direct URL identity not verified'))
                    continue
                try:
                    installed = get_dist(req.name)
                except metadata.PackageNotFoundError:
                    result['missingdeps'].append(dict(name=req.name, parent=name, requirement=text,
                                                      reason='active requirement not installed'))
                    continue
                if not req.specifier.contains(installed.version, prereleases=True):
                    result['missingdeps'].append(dict(name=req.name, parent=name, requirement=text,
                                                      installed=installed.version, reason='version mismatch'))
                pending.append((req.name, frozenset(req.extras)))
        except Exception as exc:
            result['errors'].append(f'{name}: metadata evaluation failed: {type(exc).__name__}: {exc}')
    if result['missingdeps'] or result['errors'] or result['unevaluated']:
        result['status'] = 'fail'
    return result


def model_check(model):
    path = Path(model).expanduser().resolve()
    require(path.is_dir(), f'Local model directory absent: {path}')
    config = json.loads((path / 'config.json').read_text())
    require(isinstance(config, dict) and config.get('model_type'), 'config.json lacks model_type')
    assets = [name for name in ('tokenizer.json', 'tokenizer.model') if (path / name).is_file()]
    require(assets, 'Missing tokenizer.json/tokenizer.model (tokenizer_config alone is not a tokenizer)')
    require((path / 'tokenizer_config.json').is_file(), 'Missing tokenizer_config.json')
    names = ['config.json', 'tokenizer_config.json', *assets]
    names += [name for name in ('model.safetensors.index.json', 'chat_template.jinja',
                                'generation_config.json', 'special_tokens_map.json') if (path / name).is_file()]
    return dict(path=str(path), model_type=config['model_type'], architectures=config.get('architectures'),
                files={name: dict(bytes=(path / name).stat().st_size, sha256=digest(path / name)) for name in names},
                checkpoint_revision=None, shard_hashes=None, full_checkpoint_verified=False)


def add_phase(report, name, action):
    try:
        value = action()
        report['phases'][name] = dict(status='pass', details=value)
        return value
    except Exception as exc:
        message = f'{type(exc).__name__}: {exc}'
        report['phases'][name] = dict(status='fail', errors=[message])
        report['errors'].append(dict(phase=name, message=message))
        return None


def dep_phase(report, root):
    value = add_phase(report, 'dependencies', lambda: dependencies(root))
    if value is None:
        return False
    report['versions'].update(value['versions'])
    report['missingdeps'].extend(value['missingdeps'])
    report['phases']['dependencies']['status'] = value['status']
    if value['status'] != 'pass':
        report['errors'].append(dict(phase='dependencies', message=
            'Resolve reported missing/mismatched/unevaluated dependencies manually in ' + sys.executable +
            '; no installation or relaxed validation was attempted.'))
    return value['status'] == 'pass'


def command_phase(report, name, argv, timeout):
    value = add_phase(report, name, lambda: run_process(argv, timeout))
    if value is not None and (value['returncode'] != 0 or value['timed_out'] or value.get('error')):
        report['phases'][name]['status'] = 'timeout' if value['timed_out'] else 'fail'
        report['errors'].append(dict(phase=name, message=value.get('error') or value['stderr'] or
                                     value['stdout'] or 'command failed'))
    return value


def preflight(report, args):
    dep_phase(report, 'vllm')
    identity = add_phase(report, 'model', lambda: model_check(args.model))
    report['sourceidentity']['model'] = identity
    command_phase(report, 'driver', ['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total',
                                     '--format=csv,noheader,nounits'], min(args.timeout, 10))
    if not args.skip_pip_check:
        command_phase(report, 'pip_check', [sys.executable, '-B', '-m', 'pip', 'check'], args.timeout)
    if not args.skip_import_probes:
        for module in ('torch', 'vllm'):
            code = f'import importlib; m = importlib.import_module({module!r}); print(getattr(m, "__version__", None))'
            value = command_phase(report, 'import_' + module, [sys.executable, '-B', '-c', code], args.timeout)
            if value and value['returncode'] != 0:
                missing = re.findall(r"No module named ['\"]([^'\"]+)", value['stderr'])
                report['missingdeps'].extend(dict(module=name, reason='import failure; module is not a pip package name')
                                             for name in missing)


def accepted_parameters(callable_obj):
    return {name for name, p in inspect.signature(callable_obj).parameters.items()
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}


def generation_options(config, llm_type, engine_args, sampling_type):
    llm_names = accepted_parameters(llm_type)
    forwards = any(p.kind == p.VAR_KEYWORD for p in inspect.signature(llm_type).parameters.values())
    supported = llm_names | (accepted_parameters(engine_args) if forwards else set())
    options = dict(model=str(Path(config['model']).expanduser().resolve()), tokenizer=str(Path(config['model']).expanduser().resolve()),
                   tensor_parallel_size=config['tp'], enable_expert_parallel=config['ep'],
                   max_model_len=config['ctx'], max_num_seqs=config['max_seqs'],
                   gpu_memory_utilization=config['memory'], seed=config['seed'],
                   enable_prefix_caching=False, trust_remote_code=False, cpu_offload_gb=0,
                   generation_config='vllm', pipeline_parallel_size=1, data_parallel_size=1,
                   speculative_config=None)
    sampling = dict(temperature=0.0, max_tokens=config['max_tokens'], min_tokens=config['max_tokens'],
                    ignore_eos=True, seed=config['seed'])
    require(set(options) <= supported, f'Installed LLM/EngineArgs API lacks {sorted(set(options) - supported)}')
    require(set(sampling) <= accepted_parameters(sampling_type),
            f'Installed SamplingParams API lacks {sorted(set(sampling) - accepted_parameters(sampling_type))}')
    return options, sampling


def gpu_worker(config):
    import torch
    require(torch.cuda.is_available(), 'CUDA unavailable/incompatible; no CPU fallback')
    started = time.monotonic()
    result = (torch.tensor([1.0], device='cuda') + 2).sum()
    torch.cuda.synchronize()
    require(result.item() == 3.0, 'CUDA tensor arithmetic mismatch')
    return dict(wall_seconds=time.monotonic() - started, torch=torch.__version__, cuda_runtime=torch.version.cuda,
                device=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()), value=3.0)


def generate_worker(config):
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import EngineArgs
    from vllm.platforms import current_platform
    import torch
    require(current_platform.is_cuda(), 'vLLM platform is not CUDA; no CPU fallback or environment repair')
    require(torch.cuda.is_available(), 'CUDA unavailable/incompatible; no CPU fallback')
    options, sampling = generation_options(config, LLM, EngineArgs, SamplingParams)
    started = time.monotonic()
    llm = LLM(**options)
    ready = time.monotonic()
    outputs = llm.generate([config['prompt']], SamplingParams(**sampling))
    finished = time.monotonic()
    require(len(outputs) == 1 and len(outputs[0].outputs) == 1, 'Expected one request and one completion')
    completion = outputs[0].outputs[0]
    tokens = list(completion.token_ids)
    value = dict(token_ids=tokens, text=completion.text, finish_reason=completion.finish_reason,
                 prompt_token_ids=list(outputs[0].prompt_token_ids), actual_output_tokens=len(tokens),
                 requested_output_tokens=config['max_tokens'], fixed_token_count_met=len(tokens) == config['max_tokens'],
                 load_seconds=ready-started, generate_seconds=finished-ready, total_seconds=finished-started,
                 llm_options=options, sampling_options=sampling, http_ttft_seconds=None, tpot_seconds=None,
                 torch=torch.__version__, cuda_runtime=torch.version.cuda, precision_equivalence='unverified')
    return value


def isolated_worker(mode, config, timeout):
    with tempfile.TemporaryDirectory(prefix='vllm-qwen35-') as temp:
        result_path = Path(temp) / 'result.json'
        command = [sys.executable, '-B', str(SCRIPT), '--worker', mode, str(result_path)]
        process = run_process(command, timeout, config)
        value = json.loads(result_path.read_text()) if result_path.is_file() else None
        return dict(process=process, result=value)


def explicit_worker(report, args):
    root = 'torch' if args.mode == 'gpu' else 'vllm'
    if not dep_phase(report, root):
        return
    if args.mode == 'generate':
        report['sourceidentity']['model'] = add_phase(report, 'model', lambda: model_check(args.model))
        if report['sourceidentity']['model'] is None:
            return
    child = add_phase(report, args.mode, lambda: isolated_worker(args.mode, vars(args), args.timeout))
    if child is None:
        return
    process, result = child['process'], child['result']
    report['measurements'][args.mode] = result
    good = process['returncode'] == 0 and not process['timed_out'] and not process.get('error') and result is not None
    if args.mode == 'generate' and result is not None:
        good = good and result.get('fixed_token_count_met', False) and result.get('finish_reason') == 'length'
    if not good:
        report['phases'][args.mode]['status'] = 'timeout' if process['timed_out'] else 'fail'
        report['errors'].append(dict(phase=args.mode, message=process.get('error') or process['stderr'] or
                                     'Worker failed or fixed-token/terminal contract not met'))


def parse_sse(lines, started, clock=time.monotonic):
    """Strict single-choice OpenAI chat stream; SSE event times, not token times."""
    data, event_type, done, terminal, usage = [], '', False, None, None
    text, arrivals, chunks = [], [], 0
    for raw in lines:
        line = raw.decode('utf-8') if isinstance(raw, bytes) else raw
        line = line.rstrip('\r\n')
        if line:
            if line.startswith(':'):
                continue
            field, _, value = line.partition(':')
            value = value[1:] if value.startswith(' ') else value
            if field == 'data':
                data.append(value)
            elif field == 'event':
                event_type = value
            elif field not in ('id', 'retry'):
                raise RuntimeError('Unsupported SSE field: ' + field)
            continue
        if not data:
            require(event_type != 'error', 'Server SSE error event')
            event_type = ''
            continue
        received = clock()
        require(not done, 'Data after SSE [DONE]')
        require(event_type in ('', 'message'), 'Unexpected/error SSE event: ' + event_type)
        payload = '\n'.join(data)
        data, event_type = [], ''
        if payload == '[DONE]':
            require(terminal is not None and usage is not None, '[DONE] before finish_reason/usage')
            done = True
            ended = received
            continue
        value = json.loads(payload)
        require(isinstance(value, dict) and 'error' not in value, 'Server SSE error/non-object')
        choices = value.get('choices')
        require(isinstance(choices, list), 'Missing choices list')
        chunks += 1
        if value.get('usage') is not None:
            require(terminal is not None and choices == [] and usage is None, 'Invalid/duplicate usage terminal')
            usage = value['usage']
            require(isinstance(usage, dict), 'Invalid usage object')
            names = ('prompt_tokens', 'completion_tokens', 'total_tokens')
            require(all(type(usage.get(k)) is int and usage[k] >= 0 for k in names), 'Invalid usage token counts')
            require(usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens'], 'Inconsistent usage')
            continue
        require(terminal is None and len(choices) == 1, 'Choices after finish or wrong choice count')
        choice = choices[0]
        require(isinstance(choice, dict) and choice.get('index') == 0, 'Invalid choice index')
        delta = choice.get('delta')
        require(isinstance(delta, dict), 'Missing delta')
        content = delta.get('content')
        require(content is None or isinstance(content, str), 'Non-text content delta')
        require(not delta.get('tool_calls') and not delta.get('refusal'), 'Tool/refusal response unsupported')
        if content:
            text.append(content)
            arrivals.append(received - started)
        if choice.get('finish_reason') is not None:
            require(choice['finish_reason'] in ('stop', 'length'), 'Error/unsupported finish_reason')
            terminal = choice['finish_reason']
    require(not data and not event_type, 'Unterminated SSE event')
    require(done, 'Missing terminal SSE [DONE]')
    require(arrivals and usage['completion_tokens'] > 0, 'No non-empty content/completion tokens')
    wall = ended - started
    require(wall > 0, 'Non-positive request walltime')
    return dict(text=''.join(text), finish_reason=terminal, usage=usage, sse_chunks=chunks,
                nonempty_content_events=len(arrivals), content_event_seconds=arrivals,
                content_chunk_gaps_seconds=[b-a for a, b in zip(arrivals, arrivals[1:])],
                ttft_seconds=arrivals[0], ttft_definition='send to first non-empty delta.content SSE event; '
                    'excludes role, keepalive, empty and reasoning-only events; not exact first token',
                wall_seconds=wall, actual_output_tokens=usage['completion_tokens'],
                e2e_output_tokens_per_second=usage['completion_tokens']/wall,
                tpot_seconds=None, token_itl_seconds=None, token_ids=None)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError('HTTP redirect refused; supply the exact server URL')


def http_worker(config):
    payload = dict(model=config['served_model'], messages=[dict(role='user', content=config['prompt'])],
                   stream=True, stream_options=dict(include_usage=True), temperature=0, seed=config['seed'],
                   n=1, max_tokens=config['max_tokens'])
    request = urllib.request.Request(config['url'], data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json', 'Accept': 'text/event-stream'})
    started = time.monotonic()
    with urllib.request.build_opener(NoRedirect()).open(request, timeout=config['timeout']) as response:
        require(response.status == 200, f'HTTP status {response.status}')
        require(response.headers.get_content_type() == 'text/event-stream', 'Expected text/event-stream')

        def lines():
            total = 0
            while True:
                require(time.monotonic() - started < config['timeout'], 'HTTP request deadline exceeded')
                line = response.readline(LIMIT + 1)
                if not line:
                    return
                total += len(line)
                require(len(line) <= LIMIT and total <= 16 * 1024 * 1024, 'SSE size limit exceeded')
                yield line

        value = parse_sse(lines(), started)
    value['requested_max_tokens'] = config['max_tokens']
    return value


def http_command(report, args):
    records = []
    report['measurements']['http'] = records
    total = args.warmup + args.repeat
    for index in range(total):
        label = 'warmup' if index < args.warmup else 'measured'
        child = isolated_worker('http', vars(args), args.timeout)
        process, result = child['process'], child['result']
        good = process['returncode'] == 0 and not process['timed_out'] and not process.get('error') and result is not None
        records.append(dict(index=index, population=label, status='pass' if good else 'fail',
                            measurements=result, process=process))
        if not good:
            report['errors'].append(dict(phase='http', message=f'{label} request {index} failed; no retry'))
            break
    report['phases']['http'] = dict(status='fail' if report['errors'] else 'pass',
                                    scheduled=total, offered=len(records), not_issued=total-len(records),
                                    scope='sequential smoke; no aggregate throughput/full paper protocol')


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('must be finite and positive')
    return number


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def parser():
    cli = argparse.ArgumentParser(description=__doc__, epilog='No packages are installed or modified. '
        'Default check uses read-only nvidia-smi, bounded import probes and pip check, never CUDA arithmetic/LLM. '
        'Use --skip-import-probes to omit torch/vllm imports. Reports go into a fresh private directory.')
    cli.add_argument('mode', nargs='?', default='check', choices=('check', 'gpu', 'generate', 'http'))
    cli.add_argument('--model', default=MODEL, help='local checkpoint, not a Hub download name')
    cli.add_argument('--output', help='new output directory; existing paths are refused before work')
    cli.add_argument('--quiet', action='store_true', help='print only status/report path instead of the default summary')
    cli.add_argument('--timeout', type=positive, default=None, help='seconds per subprocess/request; check/gpu 30, generate 900, http 300')
    cli.add_argument('--skip-import-probes', action='store_true', help='check only; leave import tests notrun')
    cli.add_argument('--skip-pip-check', action='store_true', help='check only; does not relax metadata validation')
    cli.add_argument('--prompt', default='Explain why the sky is blue in one sentence.', help='non-sensitive text only; output is archived')
    cli.add_argument('--max-tokens', type=positive_int, default=32)
    cli.add_argument('--seed', type=int, default=0)
    cli.add_argument('--tp', type=positive_int, default=1, help='generate: tensor_parallel_size')
    cli.add_argument('--ep', action='store_true', help='generate: enable_expert_parallel within vLLM parallel group')
    cli.add_argument('--ctx', type=positive_int, default=2048, help='generate: max_model_len')
    cli.add_argument('--max-seqs', type=positive_int, default=1)
    cli.add_argument('--memory', type=positive, default=0.8, help='generate: gpu_memory_utilization in (0,1]')
    cli.add_argument('--url', default='http://127.0.0.1:8000/v1/chat/completions', help='http: already-running server; no credentials/query/redirects')
    cli.add_argument('--served-model', help='http: exact server model ID (required; not inferred from local metadata)')
    cli.add_argument('--warmup', type=int, default=1)
    cli.add_argument('--repeat', type=positive_int, default=3)
    return cli


def _one_line(value, limit=300):
    text = 'N/A (not measured/available)' if value is None else scrub(str(value))
    text = ' '.join(text.split())
    text = ''.join(char for char in text if char.isprintable())
    return text if len(text) <= limit else text[:limit - 3] + '...'


def _phase_status(report, name):
    return report.get('phases', {}).get(name, {}).get('status', 'notrun')


def _measurement_summary(name, value):
    if value is None:
        return 'N/A (not run)'
    if name == 'http' and isinstance(value, list):
        passed = sum(item.get('status') == 'pass' for item in value if isinstance(item, dict))
        rows = []
        for item in value[:3]:
            metric = _measurement_summary('request', item.get('measurements'))
            rows.append(f"{item.get('population', 'request')} {item.get('status', 'unknown')}: {metric}")
        if len(value) > 3:
            rows.append(f'{len(value) - 3} more request records in JSON')
        return f'{passed}/{len(value)} requests passed; ' + '; '.join(rows)
    if isinstance(value, dict):
        fields = []
        for key in ('wall_seconds', 'total_seconds', 'actual_output_tokens', 'fixed_token_count_met',
                    'finish_reason', 'ttft_seconds', 'tpot_seconds', 'e2e_output_tokens_per_second', 'value'):
            if key in value:
                fields.append(f'{key}={_one_line(value[key], 70)}')
        return ', '.join(fields) if fields else 'recorded (see JSON)'
    return _one_line(value)


def print_report_summary(report, report_path):
    """Render existing evidence only: no imports, probes or environment changes."""
    report = scrub(report)
    report_path = _one_line(str(report_path), 2000)
    phases = report.get('phases', {})
    statuses = [item.get('status', 'notrun') for item in phases.values()]
    passed = sum(status == 'pass' for status in statuses)
    failed = sum(status in ('fail', 'timeout', 'error') for status in statuses)
    skipped = sum(status == 'notrun' for status in statuses)
    other = len(statuses) - passed - failed - skipped
    versions = report.get('versions', {})
    config = report.get('config', {})
    model = config.get('served_model') if config.get('mode') == 'http' else config.get('model')
    driver_phase = phases.get('driver', {})
    driver_details = driver_phase.get('details') or {}
    driver = driver_details.get('stdout') if driver_phase.get('status') == 'pass' else None
    runtime = report.get('measurements', {}).get('generate') or report.get('measurements', {}).get('gpu') or {}
    cuda = runtime.get('cuda_runtime')
    print('vLLM Qwen3.5 validation summary')
    print(f"Status: {str(report.get('status', 'unknown')).upper()}")
    print(f'Report: {report_path}')
    missing = report.get('missingdeps', [])
    dep_details = phases.get('dependencies', {}).get('details') or {}
    failures = ['; '.join(f'{key}={value}' for key, value in item.items()) if isinstance(item, dict)
                else str(item) for item in missing]
    failures += list(dep_details.get('errors', []))
    failures += [f'Unevaluated requirement: {item}' for item in dep_details.get('unevaluated', [])]
    for error in report.get('errors', []):
        if isinstance(error, dict):
            failures.append(f"{error.get('phase', 'unknown')}: {error.get('message', '')}")
        else:
            failures.append(str(error))
    for record in report.get('measurements', {}).get('http') or []:
        if record.get('status') != 'pass':
            process = record.get('process') or {}
            failures.append(process.get('stderr') or process.get('error') or 'HTTP worker failed/timeout')
    if report.get('status') == 'fail':
        print('Key failures:')
        # Keep traceback conclusions, not just the frame list at the beginning.
        for error in failures[:8]:
            lines = scrub(str(error)).splitlines()
            print('  - ' + _one_line(' '.join(lines[-3:])))
        if len(failures) > 8:
            print(f'  ... {len(failures) - 8} more failures in JSON')
        print('Next steps (manual only; no packages were changed):')
        python = shlex.quote(versions.get('executable') or sys.executable)
        print('  - Environment check: ' + _one_line(f'{python} -B -m pip check'))
        print('  - Installed packages: ' + _one_line(f'{python} -B -m pip show vllm torch packaging'))
        if missing or any(name.startswith('import_') and entry.get('status') in ('fail', 'timeout')
                          for name, entry in phases.items()):
            print('  - Manually satisfy the recorded metadata requirements, then rerun check; inspect ABI errors before any install.')
        if _phase_status(report, 'model') in ('fail', 'error'):
            print('  - Verify --model contains config.json, tokenizer_config.json, and tokenizer.json or tokenizer.model.')
        if _phase_status(report, 'driver') in ('fail', 'timeout', 'error') or _phase_status(report, 'gpu') in ('fail', 'timeout'):
            print('  - Run nvidia-smi and verify the active driver/CUDA/PyTorch environment without changing loader variables.')
        if _phase_status(report, 'http') in ('fail', 'timeout'):
            print('  - Verify the already-running server URL/model and its SSE finish_reason, usage, and [DONE] contract.')
        print('  - Rerun the same mode after resolving failures, using a fresh --output directory.')
    print(f"Mode: {report.get('config', {}).get('mode', 'N/A')} | Model: {_one_line(model)}")
    print(f"Python: {_one_line(versions.get('python', 'N/A'), 100)}")
    print(f"Versions: vllm={_one_line(versions.get('vllm'))} | "
          f"torch={_one_line(versions.get('torch') or runtime.get('torch'))} | CUDA runtime={_one_line(cuda)} | "
          f"driver={_one_line(driver)}")
    counts = f'passed={passed} failed={failed} skipped={skipped}'
    if other:
        counts += f' other={other}'
    print(f'Phases: {counts}')
    for label, selected in (('Passed', ('pass',)), ('Failed', ('fail', 'error', 'timeout')), ('Skipped', ('notrun',))):
        names = [name for name, value in phases.items() if value.get('status') in selected]
        print(f'  {label}: ' + _one_line(', '.join(names) or 'none'))

    missing = report.get('missingdeps', [])
    import_errors = []
    for name, value in phases.items():
        if name.startswith('import_') and value.get('status') in ('fail', 'timeout', 'error'):
            details = value.get('details') or {}
            error = details.get('stderr') or value.get('errors') or value.get('status')
            import_errors.append(f'{name}: {_one_line(" ".join(str(error).splitlines()[-3:]))}')
    print('Missing dependencies/import errors:')
    if not missing and not import_errors:
        print('  none reported (see skipped phases; not proof of installed/compatible dependencies)')
    else:
        for item in missing[:5]:
            if isinstance(item, dict):
                print('  - ' + _one_line('; '.join(f'{key}={value}' for key, value in item.items())))
            else:
                print('  - ' + _one_line(item))
        for item in import_errors[:5]:
            print('  - ' + item)
        if len(missing) > 5:
            print(f'  ... {len(missing) - 5} more dependency issues in JSON')

    model_phase = phases.get('model', {})
    model_details = model_phase.get('details') or {}
    model_type = model_details.get('model_type', 'N/A') if isinstance(model_details, dict) else 'N/A'
    model_errors = model_phase.get('errors', []) if isinstance(model_phase, dict) else []
    print(f"Model checks: {_phase_status(report, 'model')} | model_type={_one_line(model_type)}")
    if model_details.get('files'):
        print('  Files checked: ' + _one_line(', '.join(model_details['files'])))
        print('  Full checkpoint/shard verification: not established by basic checks')
    for error in model_errors[:2]:
        print('  - ' + _one_line(error))

    print('Measurements:')
    measurements = report.get('measurements', {})
    for name in ('gpu', 'generate', 'http', 'oracle', 'gpu_quiescence'):
        print(f'  {name}: {_measurement_summary(name, measurements.get(name))}')

    if report.get('status') == 'pass':
        print('Next steps: selected scope passed; compatibility, precision, topology, and oracle acceptance remain unverified.')


def reserve_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False, mode=0o700)
    return path


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.timeout is None:
        args.timeout = {'check': 30, 'gpu': 30, 'generate': 900, 'http': 300}[args.mode]
    if args.memory > 1 or args.warmup < 0:
        cli.error('--memory must be <= 1 and --warmup >= 0')
    url = urllib.parse.urlsplit(args.url)
    if args.mode == 'http' and (not args.served_model or url.scheme not in ('http', 'https') or
                               not url.hostname or url.username or url.password or url.query or url.fragment):
        cli.error('http requires --served-model and an http(s) URL without credentials/query/fragment')
    output = Path(args.output) if args.output else OUTPUT / (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) +
                                                            f'-{args.mode}-{os.getpid()}-{time.time_ns()}')
    try:
        reserve_output(output)
    except OSError as exc:
        print(f'Refusing output path before any work: {exc}', file=sys.stderr)
        return 2
    config = vars(args).copy()
    config['prompt'] = dict(sha256=hashlib.sha256(args.prompt.encode()).hexdigest(), utf8_bytes=len(args.prompt.encode()))
    report = dict(schema='ferrule.vllm-qwen35-check.v1', status='running', phases={}, errors=[], missingdeps=[],
                  versions=dict(python=sys.version, executable=sys.executable), config=config,
                  sourceidentity=dict(script_sha256=digest(SCRIPT), model=None, server_identity=None),
                  measurements=dict(gpu=None, generate=None, http=None, oracle=None, gpu_quiescence=None),
                  notes=NOTES, environment={key: dict(present=key in os.environ,
                      value_sha256=hashlib.sha256(os.environ[key].encode()).hexdigest() if key in os.environ else None)
                      for key in ('CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'LD_LIBRARY_PATH', 'VLLM_TARGET_DEVICE',
                                  'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')})
    for name in ('dependencies', 'model', 'driver', 'pip_check', 'import_torch', 'import_vllm', 'gpu', 'generate', 'http'):
        report['phases'][name] = dict(status='notrun', details=None)
    try:
        if args.mode == 'check':
            preflight(report, args)
        elif args.mode in ('gpu', 'generate'):
            explicit_worker(report, args)
        else:
            http_command(report, args)
    except (Exception, KeyboardInterrupt) as exc:
        report['errors'].append(dict(phase=args.mode, message=f'{type(exc).__name__}: {exc}'))
    report['status'] = 'fail' if report['errors'] or report['missingdeps'] else 'pass'
    with (output / 'report.json').open('x') as stream:
        json.dump(scrub(report), stream, indent=2, allow_nan=False)
        stream.write('\n')
    if args.quiet:
        print(scrub(f"{report['status']}: {output / 'report.json'}"))
    else:
        print_report_summary(report, output / 'report.json')
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--worker':
        # Worker output has a separate channel: vLLM logs cannot corrupt JSON.
        config = json.load(sys.stdin)
        workers = {'gpu': gpu_worker, 'generate': generate_worker, 'http': http_worker}
        result = workers[sys.argv[2]](config)
        with Path(sys.argv[3]).open('x') as stream:
            json.dump(scrub(result), stream, allow_nan=False)
    else:
        sys.exit(main())
