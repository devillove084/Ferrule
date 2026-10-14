"""Finite, permission-gated external GPU queries and read-only process-tree samples."""
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from contract import require, resource_evidence, identity_mode, IDENTITY_MODES
from launch_guard import identity


def authorize(granted, uuid, note, mode='hostpid_verified'):
    require(mode in IDENTITY_MODES, 'Unknown identity mode')
    import re
    require(granted is True and isinstance(note, str) and bool(note.strip()),
            'Explicit --sole-gpu-owner-granted and --owner-note required; no GPU query/launch')
    require(isinstance(uuid, str) and re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', uuid), 'Need physical --gpu-uuid')


def gpu_sample(tool, uuid, granted, note):
    authorize(granted, uuid, note)
    def query(fields):
        result = subprocess.run([tool, '--id=' + uuid, '--query-' + fields,
                                 '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=3, check=True)
        return list(csv.reader(io.StringIO(result.stdout)))
    rows = query('gpu=index,uuid,name,driver_version,pci.bus_id,memory.total,memory.used,utilization.gpu')
    require(len(rows) == 1 and len(rows[0]) == 8, 'Missing/ambiguous GPU telemetry')
    index, actual, name, driver, pci, total, used, util = [s.strip() for s in rows[0]]
    require(actual == uuid, 'Physical GPU mapping changed')
    apps = []
    for row in query('compute-apps=pid,gpu_uuid,used_gpu_memory'):
        require(len(row) == 3, 'Malformed process GPU telemetry')
        pid, device, memory = [s.strip() for s in row]
        if device == uuid:
            apps.append(dict(pid=int(pid), uuid=device, used_bytes=int(memory) * (1 << 20)))
    return dict(index=int(index), uuid=actual, name=name, driver=driver, pci_bus_id=pci,
                total_bytes=int(total) * (1 << 20), gpu_bytes=int(used) * (1 << 20),
                utilization=int(util), apps=apps, monotonic=time.monotonic())


def all_gpu_sample(tool, uuid, granted, note):
    authorize(granted, uuid, note, 'device_exclusive_local_custody')
    def query(fields):
        result = subprocess.run([tool, '--query-' + fields, '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=3, check=True)
        return list(csv.reader(io.StringIO(result.stdout)))
    devices = []
    for row in query('gpu=index,uuid,memory.total,memory.used,utilization.gpu'):
        require(len(row) == 5, 'Malformed all-device telemetry')
        index, device, total, used, util = [s.strip() for s in row]
        devices.append(dict(index=int(index), uuid=device, total_bytes=int(total)*(1 << 20),
                            gpu_bytes=int(used)*(1 << 20), utilization=int(util)))
    require(len(devices) == 8 and len({d['uuid'] for d in devices}) == 8,
            'Expected complete eight-device idle inventory')
    selected = [d for d in devices if d['uuid'] == uuid]
    require(len(selected) == 1, 'Selected physical UUID absent/ambiguous')
    apps = []
    for row in query('compute-apps=pid,gpu_uuid,used_gpu_memory'):
        require(len(row) == 3, 'Malformed all-device application telemetry')
        pid, device, memory = [s.strip() for s in row]
        apps.append(dict(pid=int(pid), uuid=device, used_bytes=int(memory)*(1 << 20)))
    return dict(selected[0], devices=devices, apps=apps, monotonic=time.monotonic())


def idle(tool, uuid, granted, note, config):
    deadline = time.monotonic() + config['limits']['idle_seconds'] - 6
    records, consecutive = [], 0
    exclusive = identity_mode(config) == 'device_exclusive_local_custody'
    scope = 'all eight physical GPUs' if exclusive else 'selected physical UUID only'
    while time.monotonic() < deadline:
        s = (all_gpu_sample if exclusive else gpu_sample)(tool, uuid, granted, note)
        records.append(s)
        good = (not s['apps'] and s['utilization'] == 0
                and s['gpu_bytes'] <= config['resources']['idle_max_device_mib'] * (1 << 20)
                and s['total_bytes'] == config['resources']['gpu_expected_total_bytes'])
        if exclusive:
            good = good and all(d['utilization'] == 0 and d['gpu_bytes'] == 0 for d in s['devices'])
        consecutive = consecutive + 1 if good else 0
        if consecutive == 2:
            return dict(status='verified', observations=records, scope=scope)
        time.sleep(.25)
    return dict(status='blocked', observations=records, scope=scope)


def host_sample(root_pid):
    parents = {}
    for stat in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = stat.read_text().rsplit(')', 1)[1].split()
            parents[int(stat.parent.name)] = int(fields[1])
        except FileNotFoundError:
            continue
    require(root_pid in parents, 'Server root missing during host sample')
    owned = {root_pid}
    while True:
        grown = owned | {pid for pid, parent in parents.items() if parent in owned}
        if grown == owned:
            break
        owned = grown
    rows = []
    for pid in sorted(owned):
        try:
            data = {}
            for name in ('smaps_rollup', 'status'):
                for line in (Path('/proc') / str(pid) / name).read_text().splitlines():
                    key, _, value = line.partition(':')
                    if key in ('Rss', 'Pss', 'VmLck', 'VmPin'):
                        data[key] = int(value.split()[0]) * 1024
            require('Rss' in data and 'Pss' in data, 'Missing RSS/PSS for owned process')
            rows.append(dict(pid=pid, **data))
        except FileNotFoundError:
            continue  # departed child; never turn a missing live root into zero memory
    require(any(row['pid'] == root_pid for row in rows), 'No live server root memory evidence')
    return dict(pss_bytes=sum(row['Pss'] for row in rows), rss_bytes=sum(row['Rss'] for row in rows),
                processes=rows, host_coverage='sampled server descendants; sharing deduplicated via PSS')


def app_ownership(apps, server_pid, authenticated_worker, mode='hostpid_verified'):
    require(mode in IDENTITY_MODES, 'Unknown identity mode')
    if mode == 'device_exclusive_local_custody':
        worker = authenticated_worker['worker']
        require(identity(worker['pid']) == worker, 'Authenticated worker identity changed')
        require(identity(server_pid)['ppid'] == worker['pid'],
                'Server is not authenticated worker direct child')
        if os.readlink('/proc/self/ns/pid') != 'pid:[4026531836]':
            return dict(status='unmapped_host_pids', matched=[], nvml_pids=[a['pid'] for a in apps],
                        owned_gpu_attribution='unknown', performance_basis='exclusive_operator_allocation_assumption',
                        reason='No proven host-to-local PID translation; local custody checked independently',
                        absence_of_foreign_interference_proven=False)
        # When host PID identity is available, keep the strict foreign-app check.

    if not apps:
        return dict(status='no_apps_observed', matched=[])
    # NVML reports host PIDs. In nested namespaces even one NSpid column is not
    # proof of a host mapping. This implementation supports only the known Linux
    # initial PID namespace; containers remain explicitly blocked, never guessed.
    if os.readlink('/proc/self/ns/pid') != 'pid:[4026531836]':
        raise ValueError('GPU app PID namespace mapping unprovable; ownership unknown')
    worker = authenticated_worker['worker']
    require(identity(worker['pid']) == worker, 'Authenticated worker identity changed')
    server = identity(server_pid)
    require(server['ppid'] == worker['pid'], 'Server is not authenticated worker direct child')
    matched = []
    for app in apps:
        pid, chain = app['pid'], []
        while pid > 1 and pid not in {row['pid'] for row in chain}:
            row = identity(pid)
            chain.append(row)
            if pid == server_pid:
                break
            pid = row['ppid']
        require(chain and chain[-1]['pid'] == server_pid, 'Foreign GPU application detected; no foreign signal permitted')
        require(all(identity(row['pid']) == row for row in chain), 'GPU app ancestry changed during sample')
        matched.append(dict(gpu_pid=app['pid'], ancestry=chain))
    return dict(status='matched_authenticated_descendants', matched=matched,
                pid_namespace='initial Linux namespace; no cross-namespace PID inference')


class Monitor:
    def __init__(self, directory, server, tool, uuid, note, config, authenticated_worker):
        self.directory, self.server, self.tool = directory, server, tool
        self.uuid, self.note, self.config = uuid, note, config
        self.authenticated_worker = authenticated_worker
        self.samples, self.errors = [], []
        self.stop = threading.Event()
        self.failed = threading.Event()
        self.thread = threading.Thread(target=self.collect, daemon=True)

    def collect(self):
        deadline = time.monotonic() + self.config['limits']['case_seconds']
        stream = None
        try:
            stream = (self.directory / 'samples.jsonl').open('x')
            while not self.stop.is_set() and time.monotonic() < deadline:
                if self.server.poll() is not None:
                    break
                try:
                    require((self.directory / 'server.log').stat().st_size <= 64 * 1024 * 1024,
                            'Server log exceeded 64 MiB observation limit')
                    row = gpu_sample(self.tool, self.uuid, True, self.note)
                    row['ownership'] = app_ownership(row['apps'], self.server.pid, self.authenticated_worker,
                                                     identity_mode(self.config))
                    row['nvml_pid_set_changed'] = bool(self.samples and
                        {a['pid'] for a in row['apps']} != {a['pid'] for a in self.samples[-1]['apps']})
                    row.update(host_sample(self.server.pid))
                    self.samples.append(row)
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                    stream.flush()
                    require(row['gpu_bytes'] <= self.config['resources']['gpu_target_bytes'], 'Sampled GPU target exceeded')
                    require(row['pss_bytes'] <= self.config['resources']['host_target_bytes'], 'Sampled host PSS target exceeded')
                except Exception as error:
                    self.errors.append(str(error))
                    self.failed.set()
                    # An in-flight query failure is evidence even if finish()
                    # requested stop concurrently. Only signal our direct child.
                    if not self.stop.is_set() and self.server.poll() is None:
                        self.server.terminate()
                    break
                self.stop.wait(self.config['resources']['sample_interval_seconds'])
        except Exception as error:
            self.errors.append('monitor thread failure: ' + str(error))
            self.failed.set()
        finally:
            terminal = dict(kind='monitor_terminal', status='failed' if self.errors else 'stopped',
                            errors=list(self.errors), samples=len(self.samples), monotonic=time.monotonic())
            if stream is not None:
                stream.write(json.dumps(terminal, allow_nan=False) + '\n')
                stream.flush()
                stream.close()

    def start(self):
        self.thread.start()

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=8)
        if self.thread.is_alive():
            self.errors.append('Sampler did not stop within its bounded query timeout')
        result = resource_evidence(self.samples, self.errors, self.config['resources'])
        result.update(identity_mode=identity_mode(self.config), measurement_scope='whole_selected_device',
                      owned_process_gpu_peak_bytes=None, owned_gpu_attribution='unknown',
                      nvml_pid_set_changes=sum(s.get('nvml_pid_set_changed', False) for s in self.samples),
                      absence_of_foreign_interference_proven=False)
        return result
