"""Current paper observer: partial proc snapshots are not root-custody failures.

Historical observers stay immutable. Pure UUID-delta/idle policy is reused;
GPU queries, leaf races, coverage deadlines and diagnostics live here once.
"""
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback

import contract as ct
import launch_guard as guard
import telemetry
from sampling_v4 import device_deltas, NONSELECTED_DELTA_TOLERANCE_BYTES
from sampling_v4 import idle as _idle

VALID_SAMPLE_DEADLINE_SECONDS = 60


class CustodyLost(ValueError):
    pass


def proc_identity(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, birth_ticks=int(fields[19]), ppid=int(fields[1]), state=fields[0])


def root_key(identity):
    return identity['pid'], identity['birth_ticks']


PROC_TIMELINE_SCHEMA = 'ferrule.proc-process-timeline.v1'
_PROC_STATUS_BYTES = ('VmSize', 'VmRSS', 'RssAnon', 'RssFile', 'RssShmem')
_PROC_STATUS_INTS = ('Threads', 'voluntary_ctxt_switches', 'nonvoluntary_ctxt_switches')
_PROC_STATUS_TEXT = ('Cpus_allowed_list', 'Mems_allowed_list')


def _parse_proc_status(text):
    result = {key: None for key in (*_PROC_STATUS_BYTES, *_PROC_STATUS_INTS, *_PROC_STATUS_TEXT)}
    for line in text.splitlines():
        key, _, value = line.partition(':')
        if key in _PROC_STATUS_BYTES:
            fields = value.split()
            ct.require(len(fields) == 2 and fields[1] == 'kB' and fields[0].isdigit(),
                       'Malformed proc status value: '+key)
            result[key] = int(fields[0]) * 1024
        elif key in _PROC_STATUS_INTS:
            value = value.strip()
            ct.require(value.isdigit(), 'Malformed proc status counter: '+key)
            result[key] = int(value)
        elif key in _PROC_STATUS_TEXT:
            result[key] = value.strip() or None
    return result


def _parse_proc_stat(text):
    fields = text.rsplit(')', 1)[1].split()
    ct.require(len(fields) > 19, 'Malformed proc stat')
    return dict(state=fields[0], minor_faults=int(fields[7]), major_faults=int(fields[9]),
                num_threads=int(fields[17]), starttime_ticks=int(fields[19]))


def process_timeline_snapshot(pid, phase, clock=time.monotonic, wall_clock_ns=time.time_ns):
    """Read-only PID/TID status/stat snapshot for existing lifecycle anchors.

    This is intentionally not a sampler: callers take one snapshot at launch,
    readiness, request anchors, or shutdown. PSS subcomponents and live heap
    remain null unless the kernel exposes them through the existing rollup path.
    """
    sample_start = clock()
    root = proc_identity(pid)
    task_dir = Path('/proc') / str(pid) / 'task'
    errors, threads = [], []
    try:
        tids = sorted(int(path.name) for path in task_dir.glob('[0-9]*'))
    except (FileNotFoundError, ProcessLookupError, PermissionError) as error:
        tids = [pid]
        errors.append(type(error).__name__+': '+str(error))
    for tid in tids:
        try:
            stat_path = Path('/proc') / str(tid) / 'stat'
            before = _parse_proc_stat(stat_path.read_text())
            status = _parse_proc_status((Path('/proc') / str(tid) / 'status').read_text())
            stat = _parse_proc_stat(stat_path.read_text())
            ct.require(before['starttime_ticks'] == stat['starttime_ticks'], 'Thread identity changed during snapshot')
            threads.append(dict(tid=tid, is_main=tid == pid, **status, **stat))
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError) as error:
            errors.append(dict(tid=tid, error=type(error).__name__+': '+str(error)))
    ct.require(root_key(proc_identity(pid)) == root_key(root), 'Process identity changed during snapshot')
    if not any(row['is_main'] for row in threads):
        errors.append(dict(tid=pid, error='Main thread unavailable during snapshot'))
    return dict(schema=PROC_TIMELINE_SCHEMA, phase=phase, pid=pid, root_identity=root,
        sample_start_monotonic=sample_start, sample_end_monotonic=clock(), wall_clock_ns=wall_clock_ns(), threads=threads,
        thread_count=len(threads), errors=errors, read_only=True,
        pss_anon_bytes=None, pss_file_bytes=None, heap_live_bytes=None, numa_resident_pages=None,
        timing_semantics='CPU proc read bounds only; phase is caller label, not CUDA synchronization or a production-stage timer',
        memory_unit='bytes; task Vm/RSS describe shared process address space and must not be summed across TIDs',
        memory_semantics='status Vm/RssAnon/RssFile plus stat faults only; no PSS subcomponent or heap-live proof')


PROC_TIMELINE_PHASES = ('main_pid_first_identified','ready_before_first',
    'afterwarm_before_measured','postmeasured','shutdown_after_exit')
PROC_TIMELINE_POLICY = dict(schema=PROC_TIMELINE_SCHEMA, phases=list(PROC_TIMELINE_PHASES),
    max_snapshots=5, measured_window_scans=False, periodic_scans=False)


class ProcTimeline:
    """Finite read-only anchor records; optional metrics never become custody proof."""
    def __init__(self, directory, root):
        self.path = Path(directory)/'proc-timeline.json'
        self.root, self.records = root, []
        ct.require(not self.path.exists(), 'No proc timeline overwrite/resume')

    def capture(self, phase, anchor=None):
        ct.require(phase in PROC_TIMELINE_PHASES and phase not in {r['phase'] for r in self.records}
            and len(self.records) < 5, 'Invalid/repeated proc timeline anchor')
        started = time.monotonic()
        try:
            data = process_timeline_snapshot(self.root['pid'],phase)
            ct.require(root_key(data['root_identity']) == root_key(self.root), 'Proc timeline root identity changed')
            try:
                data['process_memory'] = memory_row(self.root['pid'])
            except (OSError,ValueError) as failure:
                data['process_memory'] = None
                data['errors'].append(dict(metric='smaps_rollup',error=type(failure).__name__+': '+str(failure)))
            availability = 'partial' if data['errors'] else 'available'
            error = None
        except (OSError,ValueError) as failure:
            data, availability, error = None, 'unavailable', type(failure).__name__+': '+str(failure)
        ended = time.monotonic()
        self.records.append(dict(phase=phase,raw_anchor=anchor,expected_root=self.root,
            read_start_monotonic=started,read_end_monotonic=ended,read_cost_seconds=ended-started,
            availability=availability,data=data,error=error))
        receipt = dict(policy=PROC_TIMELINE_POLICY,records=self.records,
            metric_semantics='proc status/stat snapshots only; no weights-load progress, heap-live, CUDA sync or device quiescence proof',
            cost_semantics='proc read/validation bounds; receipt writes also occur outside measured window, persistence cost not separately measured')
        encoded = json.dumps(receipt,allow_nan=False,sort_keys=True,indent=2)+'\n'
        if len(self.records) == 1:
            ct.write(self.path,receipt)
        else:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w',dir=self.path.parent,prefix='.proc-timeline-',delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.replace(temporary,self.path)
            finally:
                temporary.unlink(missing_ok=True)


def measure_with_proc_timeline(measure, timeline):
    """Keep every scan outside the original measure function's complete call."""
    def wrapped(*args, **kwargs):
        timeline.capture('afterwarm_before_measured',dict(kind='original measure call entry, after warmup'))
        result = measure(*args, **kwargs)
        timeline.capture('postmeasured',dict(kind='original measure returned',
            started_at=result.get('started_at'),observed_end_at=result.get('observed_end_at'),
            measurements=result.get('measurements')))
        return result
    return wrapped


def verify_root(expected, identity=proc_identity):
    try:
        current = identity(expected['pid'])
    except (FileNotFoundError, ProcessLookupError) as error:
        raise CustodyLost('Authenticated server root disappeared') from error
    if root_key(current) != root_key(expected) or current['ppid'] != expected['ppid']:
        raise CustodyLost('Authenticated server root PID/start-time/parent changed')
    if current['state'] in ('Z', 'X'):
        raise CustodyLost('Authenticated server root exited')
    return current


MEMORY_COMPONENTS = {'Anonymous':'anonymous_bytes', 'Pss_Anon':'pss_anon_bytes',
                     'Pss_File':'pss_file_bytes', 'Private_Dirty':'private_dirty_bytes'}


def parse_memory(smaps, status):
    """smaps components are optional across kernels; status must not replace them."""
    data = {name:None for name in MEMORY_COMPONENTS}
    for text, allowed in ((smaps, {'Rss','Pss',*MEMORY_COMPONENTS}), (status, {'VmLck','VmPin'})):
        for line in text.splitlines():
            key, _, value = line.partition(':')
            if key in allowed:
                fields = value.split()
                ct.require(len(fields) == 2 and fields[1] == 'kB' and fields[0].isdigit(),
                           'Malformed proc memory value: '+key)
                data[key] = int(fields[0])*1024
    ct.require('Rss' in data and 'Pss' in data, 'Live process RSS/PSS unavailable')
    return data


def memory_row(pid):
    directory = Path('/proc')/str(pid)
    return dict(pid=pid, **parse_memory((directory/'smaps_rollup').read_text(),
                                      (directory/'status').read_text()))


def host_snapshot(root, identity=proc_identity, memory=memory_row, pids=None):
    """Bind root once at launch; each leaf has independent birth-time evidence."""
    verify_root(root, identity)
    snapshots, enumeration_gaps = {}, []
    pids = pids if pids is not None else [int(p.name) for p in Path('/proc').glob('[0-9]*')]
    for pid in pids:
        try:
            snapshots[pid] = identity(pid)
        except (FileNotFoundError, ProcessLookupError):
            enumeration_gaps.append(pid)
    snapshots[root['pid']] = verify_root(root, identity)
    owned = {root['pid']}
    while True:
        grown = owned | {pid for pid, row in snapshots.items() if row['ppid'] in owned}
        if grown == owned:
            break
        owned = grown
    rows, omitted = {}, {}
    for pid in sorted(owned):
        try:
            before = snapshots[pid]
            if before['state'] in ('Z', 'X'):
                omitted[pid] = 'departed/zombie'
                continue
            row = memory(pid)
            after = identity(pid)
            if (root_key(before) != root_key(after) or before['ppid'] != after['ppid']
                    or after['state'] in ('Z', 'X')):
                omitted[pid] = 'leaf departed/reused/reparented during sample'
                continue
            rows[pid] = dict(row, identity=before)
        except (OSError, ValueError) as error:
            omitted[pid] = type(error).__name__ + ': ' + str(error)
    verify_root(root, identity)
    # Exclude children of a vanished intermediate process as an incomplete branch.
    while True:
        children = {pid: 'ancestor branch unavailable' for pid in owned if pid not in omitted
                    and pid != root['pid'] and snapshots[pid]['ppid'] in omitted}
        if not children:
            break
        omitted.update(children)
    for pid in omitted:
        rows.pop(pid, None)
    complete = not omitted and root['pid'] in rows
    partial_pss = sum(row['Pss'] for row in rows.values())
    partial_rss = sum(row['Rss'] for row in rows.values())
    components = {}
    coverage = {}
    for name, output in MEMORY_COMPONENTS.items():
        values = [row.get(name) for row in rows.values()]
        known = [value for value in values if value is not None]
        components[output] = sum(known) if complete and len(known) == len(values) else None
        coverage[output] = dict(observed_processes=len(known), sampled_processes=len(values),
                               missing_pids=[pid for pid,row in rows.items() if row.get(name) is None])
    return dict(status='complete' if complete else 'incomplete', root_identity=root,
                **components, component_coverage=coverage,
                pss_bytes=partial_pss if complete else None, rss_bytes=partial_rss if complete else None,
                partial_tree_pss_lower_bound_bytes=partial_pss, partial_tree_rss_lower_bound_bytes=partial_rss,
                processes=list(rows.values()), omitted_branches=omitted,
                unclassified_proc_enumeration_disappearances=enumeration_gaps,
                scope='snapshot descendants only; departed branch omitted; missing PSS never zero')


def all_gpu_sample(uuid, note, event=lambda row: None, run=subprocess.run, clock=time.monotonic):
    """Normal 3s query and at most one 15s TOTAL diagnostic retry, in observer thread."""
    telemetry.authorize(True, uuid, note, 'device_exclusive_local_custody')
    fields = ('gpu=index,uuid,memory.total,memory.used,utilization.gpu',
              'compute-apps=pid,gpu_uuid,used_gpu_memory')
    failures = []
    for attempt in range(2):
        deadline = clock() + (6 if attempt == 0 else 15)
        values = []
        try:
            for field in fields:
                remaining = deadline - clock()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired('nvidia-smi diagnostic pair', 15)
                out = run(['/usr/bin/nvidia-smi', '--query-' + field, '--format=csv,noheader,nounits'],
                          capture_output=True, text=True, check=True,
                          timeout=min(3, remaining) if attempt == 0 else remaining)
                values.append(list(csv.reader(io.StringIO(out.stdout))))
            break
        except subprocess.TimeoutExpired as error:
            failure = dict(kind='gpu_sample_missing', attempt=attempt+1, monotonic=clock(),
                           error=str(error), gpu_bytes=None, diagnostic=bool(attempt))
            failures.append(failure)
            event(failure)
            if attempt == 1:
                raise
    devices = []
    for row in values[0]:
        ct.require(len(row) == 5, 'Malformed GPU sample')
        index, actual, total, used, util = [x.strip() for x in row]
        devices.append(dict(index=int(index), uuid=actual, total_bytes=int(total)*(1<<20),
                            gpu_bytes=int(used)*(1<<20), utilization=int(util)))
    ct.require(len(devices) == 8 and len({d['uuid'] for d in devices}) == 8, 'Incomplete eight-GPU inventory')
    apps = []
    for row in values[1]:
        ct.require(len(row) == 3, 'Malformed GPU app sample')
        pid, actual, memory = [x.strip() for x in row]
        apps.append(dict(pid=int(pid), uuid=actual, used_bytes=int(memory)*(1<<20)))
    selected = [d for d in devices if d['uuid'] == uuid]
    ct.require(len(selected) == 1, 'Selected UUID missing')
    return dict(selected[0], devices=devices, apps=apps, monotonic=clock(),
                diagnostic_retry=bool(failures), missing_attempts=failures)


def idle(config, uuids, note, anchor=None):
    missing = []
    result = _idle(config, uuids, note, anchor,
                   query=lambda: all_gpu_sample(uuids[0], note, event=missing.append))
    result['missing_query_attempts'] = missing
    return result


def process_diagnostics(processes, environment_keys):
    rows = []
    for process in processes:
        pid, expected = process['pid'], process['identity']
        try:
            if root_key(proc_identity(pid)) != root_key(expected):
                continue
            command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            environment = dict(x.split('=', 1) for x in Path(f'/proc/{pid}/environ').read_text().split('\0') if '=' in x)
            libraries = sorted({line.split()[-1] for line in Path(f'/proc/{pid}/maps').read_text().splitlines()
                                if '/' in line and any(name in line for name in
                                ('libcuda', 'libnvidia', 'libnccl', 'libtorch', 'libc10_cuda', 'libnvrtc'))})
            if root_key(proc_identity(pid)) != root_key(expected):
                continue
            rows.append(dict(identity=expected, command=command,
                             environment={k:environment[k] for k in environment_keys if k in environment}, libraries=libraries))
        except (OSError, ValueError):
            continue
    return rows


def memory_peaks(samples, uuids):
    gpu = [sample['gpu'] for sample in samples if sample.get('gpu') is not None]
    hosts = [sample['host'] for sample in samples if (sample.get('host') or {}).get('status') == 'complete']
    def peak(key):
        return max((host[key] for host in hosts if host.get(key) is not None), default=None)
    return dict(sample_count=len(samples), complete_host_samples=len(hosts), gpu_samples=len(gpu),
        gpu_peak_bytes={u:max((d['gpu_bytes'] for sample in gpu for d in sample['devices'] if d['uuid']==u),default=None) for u in uuids},
        tree_pss_peak_bytes=peak('pss_bytes'), tree_rss_peak_bytes=peak('rss_bytes'),
        **{'tree_'+key.removesuffix('_bytes')+'_peak_bytes':peak(key) for key in MEMORY_COMPONENTS.values()},
        component_sample_counts={key:sum(host.get(key) is not None for host in hosts) for key in MEMORY_COMPONENTS.values()},
        peak_semantics='independent component maxima, not an additive decomposition of PSS peak',
        anonymous_semantics='Anonymous is resident anonymous memory, not proportional PSS; Pss_Anon/Pss_File may omit shmem')


def phase_memory(samples, uuids, ready=None, first_sent=None, first_terminal=None):
    """Only wholly contained samples belong to a phase; boundary-straddlers stay raw."""
    def finite(value):
        import math
        return isinstance(value,(int,float)) and math.isfinite(value)
    ct.require(all(value is None or finite(value) for value in (ready,first_sent,first_terminal)), 'Invalid phase timestamp')
    ct.require(first_terminal is None or (first_sent is not None and first_terminal >= first_sent), 'Invalid first-request bounds')
    if ready is not None and first_sent is not None:
        ct.require(first_sent >= ready, 'First request precedes readiness')
    phases = {name:[] for name in ('startup','ready_before_first','first_request','after_first_request')}
    unknown=[]
    for row in samples:
        start,end = row['monotonic'],row.get('sample_end_monotonic')
        if not finite(end) or end < start:
            unknown.append(row); continue
        if ready is not None and end < ready:
            phases['startup'].append(row)
        elif ready is not None and first_sent is not None and start >= ready and end < first_sent:
            phases['ready_before_first'].append(row)
        elif first_sent is not None and first_terminal is not None and start >= first_sent and end < first_terminal:
            phases['first_request'].append(row)
        elif first_terminal is not None and start >= first_terminal:
            phases['after_first_request'].append(row)
        else:
            unknown.append(row)
    return dict(bounds=dict(ready_monotonic=ready,first_sent_monotonic=first_sent,first_terminal_monotonic=first_terminal),
        phases={name:memory_peaks(rows,uuids) for name,rows in phases.items()},
        unassigned_or_boundary_samples=len(unknown),
        steady_state_claim=False, policy='after first request includes later canaries/warmup/requests; not a steady-state proof')


class Monitor:
    def __init__(self, directory, server, config, authenticated, uuids, note, baseline, environment_keys=()):
        self.directory, self.server, self.config = directory, server, config
        self.authenticated, self.uuids, self.note, self.baseline = authenticated, uuids, note, baseline
        self.root = proc_identity(server.pid)
        ct.require(self.root['ppid'] == authenticated['worker']['pid'], 'Server not authenticated worker child')
        self.environment_keys = environment_keys
        self.stop, self.failed = threading.Event(), threading.Event()
        self.thread = threading.Thread(target=self.collect, daemon=True)
        self.samples, self.events, self.errors = [], [], []
        self.library_sets = {}
        self.started = self.last_gpu = self.last_host = time.monotonic()
        self.ready_monotonic = None

    def mark_ready(self, at):
        ct.require(self.ready_monotonic is None and at >= self.started, 'Readiness marker changed')
        self.ready_monotonic = at

    def start(self):
        self.thread.start()

    def emit(self, stream, row):
        stream.write(json.dumps(row, allow_nan=False)+'\n')
        stream.flush()

    def fatal(self, reason):
        self.errors.append(reason)
        self.failed.set()
        if not self.stop.is_set() and self.server.poll() is None:
            # Validate launch-time identity before signaling only the direct child.
            try:
                verify_root(self.root)
                self.server.terminate()
            except CustodyLost:
                pass

    def gpu_delta_check(self, gpu):
        return device_deltas(gpu, self.baseline, self.uuids)

    def collect(self):
        try:
            with (self.directory/'samples.jsonl').open('x') as stream:
                def event(row):
                    self.events.append(row)
                    self.emit(stream, row)
                while not self.stop.is_set() and time.monotonic() - self.started < self.config['limits']['case_seconds']:
                    if self.server.poll() is not None:
                        break
                    try:
                        verify_root(self.root)
                        if guard.identity(self.authenticated['worker']['pid']) != self.authenticated['worker']:
                            raise CustodyLost('Authenticated worker identity changed')
                        row = dict(kind='sample', monotonic=time.monotonic(), gpu=None, host=None)
                        try:
                            row['gpu'] = all_gpu_sample(self.uuids[0], self.note, event)
                            self.last_gpu = time.monotonic()
                            row['gpu']['baseline_delta_check'] = self.gpu_delta_check(row['gpu'])
                            row['gpu']['ownership'] = telemetry.app_ownership(
                                [a for a in row['gpu']['apps'] if a['uuid'] in self.uuids], self.server.pid,
                                self.authenticated, 'device_exclusive_local_custody')
                        except subprocess.TimeoutExpired:
                            pass  # missing rows already persisted; wait for the bounded coverage deadline
                        except (FileNotFoundError, ProcessLookupError) as error:
                            verify_root(self.root)
                            event(dict(kind='gpu_identity_sample_incomplete', error=str(error), monotonic=time.monotonic()))
                            row['gpu'] = None
                        row['host'] = host_snapshot(self.root)
                        if row['host']['status'] == 'complete':
                            self.last_host = time.monotonic()
                        else:
                            event(dict(kind='host_sample_incomplete', monotonic=time.monotonic(),
                                       pss_bytes=None, omitted_branches=row['host']['omitted_branches']))
                        if not self.samples or time.monotonic() - self.samples[-1].get('diagnostics_at', 0) > 5:
                            row['process_diagnostics'] = process_diagnostics(row['host']['processes'], self.environment_keys)
                            row['diagnostics_at'] = time.monotonic()
                            for process in row['process_diagnostics']:
                                self.library_sets[str(root_key(process['identity']))] = process
                        else:
                            row['diagnostics_at'] = self.samples[-1]['diagnostics_at']
                        row['sample_end_monotonic'] = time.monotonic()
                        self.samples.append(row)
                        self.emit(stream, row)
                        gpu = row['gpu']
                        if gpu:
                            ct.require(not gpu['baseline_delta_check']['violations'], str(gpu['baseline_delta_check']['violations']))
                            ct.require(all(d['gpu_bytes'] <= self.config['resources']['gpu_target_bytes'] for d in gpu['devices']
                                           if d['uuid'] in self.uuids), 'Selected device memory ceiling exceeded')
                        ct.require(row['host']['partial_tree_pss_lower_bound_bytes'] <= self.config['resources']['host_target_bytes'],
                                   'Host PSS ceiling exceeded even for sampled subtree')
                        ct.require(time.monotonic() - self.last_gpu < VALID_SAMPLE_DEADLINE_SECONDS
                                   and time.monotonic() - self.last_host < VALID_SAMPLE_DEADLINE_SECONDS,
                                   'No valid GPU/complete PSS sample within 60-second coverage deadline')
                    except (CustodyLost, ValueError) as error:
                        if self.server.poll() is None:
                            self.fatal(type(error).__name__+': '+str(error))
                        else:
                            event(dict(kind='server_exited_during_sample', error=str(error), monotonic=time.monotonic()))
                        break
                    self.stop.wait(self.config['resources']['sample_interval_seconds'])
                self.emit(stream, dict(kind='monitor_terminal', errors=self.errors, samples=len(self.samples), monotonic=time.monotonic()))
        except Exception:
            self.fatal(traceback.format_exc())

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=24)
        if self.thread.is_alive():
            self.errors.append('Observer thread did not stop within bounded query reserve')
        gpu_rows = [s['gpu'] for s in self.samples if s['gpu'] is not None]
        host_rows = [s['host'] for s in self.samples if s['host']['status'] == 'complete']
        peaks = {u:max((d['gpu_bytes'] for s in gpu_rows for d in s['devices'] if d['uuid']==u), default=None) for u in self.uuids}
        pss = max((s['pss_bytes'] for s in host_rows), default=None)
        rss = max((s['rss_bytes'] for s in host_rows), default=None)
        valid = (not self.errors and bool(gpu_rows) and bool(host_rows)
                 and all(0 < v <= self.config['resources']['gpu_target_bytes'] for v in peaks.values())
                 and pss <= self.config['resources']['host_target_bytes'])
        first_sent = first_terminal = None
        receipt = self.directory/'hello2/http.json'
        phase_error = None
        if receipt.is_file():
            try:
                report = ct.load(receipt)
                requests = [row for run in report.get('runs',[]) for row in run.get('requests',[])]
                if len(requests) == 1 and requests[0].get('status') == 'success':
                    first_sent = requests[0].get('sent_at')
                    first_terminal = requests[0].get('ended_at')
            except (OSError,ValueError,KeyError,TypeError) as error:
                phase_error = str(error)
        try:
            phases = phase_memory(self.samples,self.uuids,self.ready_monotonic,first_sent,first_terminal)
        except ValueError as error:
            phase_error = str(error)
            phases = phase_memory(self.samples,self.uuids)
        return dict(status='observed_within_targets' if valid else 'blocked', errors=self.errors,
                    memory_components=memory_peaks(self.samples,self.uuids),
                    phase_memory=phases, phase_metadata_error=phase_error,
                    gpu_peak_bytes=peaks, tree_pss_peak_bytes=pss, tree_rss_peak_bytes=rss,
                    complete_host_samples=len(host_rows), incomplete_host_samples=len(self.samples)-len(host_rows),
                    gpu_samples=len(gpu_rows), missing_gpu_attempts=sum(e['kind']=='gpu_sample_missing' for e in self.events),
                    baseline=self.baseline,
                    delta_high_water_bytes={u:max((s['baseline_delta_check']['per_uuid'][u]['delta_bytes'] for s in gpu_rows),default=None)
                                            for u in self.baseline},
                    final_process_diagnostics=list(self.library_sets.values()),
                    owned_gpu_attribution='unknown', owned_process_gpu_peak_bytes=None,
                    exact_peaks=False, hard_memory_partition=False, continuous_coverage=False)
