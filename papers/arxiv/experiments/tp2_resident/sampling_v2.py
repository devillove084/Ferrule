"""Two-device telemetry using existing queries/ownership/PSS helpers.

Only ESRCH during proc enumeration is resampled, with the live server root
kernel identity held constant. No missing observation is ever replaced by zero.
"""
import json
import threading
import time
import traceback

import contract as ct
import launch_guard as guard
import telemetry


def host_sample(root_pid, identity=None, sampler=None):
    identity = guard.identity if identity is None else identity
    sampler = telemetry.host_sample if sampler is None else sampler
    root = identity(root_pid)
    races = []
    for attempt in range(5):
        ct.require(identity(root_pid) == root, 'Server root identity changed before PSS sample')
        try:
            result = sampler(root_pid)
        except ProcessLookupError:
            races.append(dict(attempt=attempt+1, error=traceback.format_exc(), monotonic=time.monotonic()))
            ct.require(identity(root_pid) == root, 'Server root disappeared during PSS sample')
            if attempt == 4:
                raise
            time.sleep(.01)
            continue
        ct.require(identity(root_pid) == root, 'Server root identity changed during PSS sample')
        ct.require(result['pss_bytes'] > 0 and any(row['pid'] == root_pid for row in result['processes']),
                   'Live root PSS missing; never substitute zero')
        result['proc_enumeration_resamples'] = races
        return result
    raise AssertionError('Unreachable bounded sampler')


class Monitor:
    def __init__(self, directory, server, config, authenticated, uuids, note):
        self.directory, self.server, self.config = directory, server, config
        self.authenticated, self.uuids, self.note = authenticated, uuids, note
        self.samples, self.errors = [], []
        self.stop, self.failed = threading.Event(), threading.Event()
        self.thread = threading.Thread(target=self.collect, daemon=True)

    def is_set(self):
        return self.failed.is_set()

    def start(self):
        self.thread.start()

    def collect(self):
        deadline = time.monotonic() + self.config['limits']['case_seconds']
        stream = None
        try:
            stream = (self.directory / 'samples.jsonl').open('x')
            while not self.stop.is_set() and time.monotonic() < deadline:
                if self.server.poll() is not None:
                    break
                ct.require((self.directory / 'server.log').stat().st_size <= 64*1024*1024, 'Server log limit exceeded')
                row = telemetry.all_gpu_sample('/usr/bin/nvidia-smi', self.uuids[0], True, self.note)
                selected = [d for d in row['devices'] if d['uuid'] in self.uuids]
                ct.require(len(selected) == 2, 'Two-device telemetry missing')
                for index, uuid in enumerate(self.uuids):
                    ct.require(any(d['index'] == index and d['uuid'] == uuid and
                               d['total_bytes'] == self.config['resources']['gpu_expected_total_bytes'] for d in selected),
                               'Selected device mapping changed')
                row['ownership'] = telemetry.app_ownership([a for a in row['apps'] if a['uuid'] in self.uuids],
                    self.server.pid, self.authenticated, 'device_exclusive_local_custody')
                row['gpu_bytes'] = max(d['gpu_bytes'] for d in selected)
                row.update(host_sample(self.server.pid))
                row['nvml_pid_set_changed'] = bool(self.samples and
                    {a['pid'] for a in row['apps']} != {a['pid'] for a in self.samples[-1]['apps']})
                self.samples.append(row)
                stream.write(json.dumps(row, allow_nan=False)+'\n')
                stream.flush()
                ct.require(all(d['gpu_bytes'] == 0 and d['utilization'] == 0 for d in row['devices']
                               if d['uuid'] not in self.uuids) and not any(a['uuid'] not in self.uuids for a in row['apps']),
                           'Nonselected GPU activity under sole-owner allocation; never signal foreign process')
                ct.require(row['gpu_bytes'] <= self.config['resources']['gpu_target_bytes'], 'Sampled GPU target exceeded')
                ct.require(row['pss_bytes'] <= self.config['resources']['host_target_bytes'], 'Sampled host PSS target exceeded')
                self.stop.wait(self.config['resources']['sample_interval_seconds'])
        except Exception:
            self.errors.append(traceback.format_exc())
            self.failed.set()
            if not self.stop.is_set() and self.server.poll() is None:
                self.server.terminate()
        finally:
            if stream is not None:
                stream.write(json.dumps(dict(kind='monitor_terminal', errors=self.errors, samples=len(self.samples),
                                              monotonic=time.monotonic()))+'\n')
                stream.close()

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=8)
        if self.thread.is_alive():
            self.errors.append('Sampler did not stop within bounded query deadline')
        result = ct.resource_evidence(self.samples, self.errors, self.config['resources'])
        rows = {}
        for uuid in self.uuids:
            samples = [dict(s, gpu_bytes=next(d['gpu_bytes'] for d in s['devices'] if d['uuid'] == uuid)) for s in self.samples]
            rows[uuid] = ct.resource_evidence(samples, self.errors, self.config['resources'])
        result.update(per_device=rows, identity_mode='device_exclusive_local_custody',
                      measurement_scope='all eight physical devices; selected two independently budgeted',
                      owned_process_gpu_peak_bytes=None, owned_gpu_attribution='unknown',
                      absence_of_foreign_interference_proven=False,
                      proc_enumeration_resamples=sum(len(s['proc_enumeration_resamples']) for s in self.samples),
                      nvml_pid_set_changes=sum(s['nvml_pid_set_changed'] for s in self.samples),
                      pss_policy='server tree sampled once; never sum per-device copies')
        return result
