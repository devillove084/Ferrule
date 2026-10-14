"""Baseline-relative two-device telemetry using existing query/ownership/PSS helpers.

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


# Preserve v2 ESRCH behavior exactly; no shared/helper changes.
from sampling_v2 import host_sample

NONSELECTED_DELTA_TOLERANCE_BYTES = 1 << 20


def device_deltas(sample, baseline, uuids, tolerance=NONSELECTED_DELTA_TOLERANCE_BYTES):
    ct.require(type(tolerance) is int and tolerance == NONSELECTED_DELTA_TOLERANCE_BYTES,
               'Only the predeclared 1 MiB nonselected delta tolerance is supported')
    devices = {d['uuid']: d for d in sample['devices']}
    ct.require(len(devices) == len(sample['devices']) == len(baseline) == 8
               and set(devices) == set(baseline), 'Complete matching eight-UUID baseline required')
    rows, reasons = {}, []
    for uuid, device in devices.items():
        before = baseline[uuid]
        ct.require((device['index'], device['total_bytes']) == (before['index'], before['total_bytes']),
                   'Physical UUID/index/capacity changed')
        ct.require(type(device['gpu_bytes']) is int and device['gpu_bytes'] >= 0
                   and type(before['gpu_bytes']) is int and before['gpu_bytes'] >= 0, 'Invalid memory observation')
        delta = device['gpu_bytes'] - before['gpu_bytes']
        selected = uuid in uuids
        rows[uuid] = dict(index=device['index'], selected=selected, baseline_bytes=before['gpu_bytes'],
                          observed_bytes=device['gpu_bytes'], delta_bytes=delta, positive_delta_bytes=max(0, delta),
                          utilization=device['utilization'], tolerance_bytes=None if selected else tolerance)
        if not selected and delta > tolerance:
            reasons.append('Nonselected positive memory delta exceeds 1 MiB: ' + uuid)
    for app in sample['apps']:
        ct.require(app['uuid'] in devices, 'Compute PID on unknown GPU')
    return dict(per_uuid=rows, violations=reasons,
                policy='Nonselected memory rejects only for positive delta > 1 MiB; PID/utilization retained without foreign signals; pre/post idle still require no apps or utilization')


def idle(config, uuids, note, anchor=None, query=None):
    query = query or (lambda: telemetry.all_gpu_sample('/usr/bin/nvidia-smi', uuids[0], True, note))
    deadline = time.monotonic() + config['limits']['idle_seconds'] - 6
    records, consecutive, previous = [], 0, None
    while time.monotonic() < deadline:
        sample = query()
        current = {d['uuid']: dict(d) for d in sample['devices']}
        ct.require(len(current) == 8, 'Incomplete idle inventory')
        for index, uuid in enumerate(uuids):
            ct.require(uuid in current and current[uuid]['index'] == index
                       and current[uuid]['total_bytes'] == config['resources']['gpu_expected_total_bytes'],
                       'Selected UUID/index/capacity differs')
        comparison = device_deltas(sample, anchor if anchor is not None else current, uuids)
        sample['baseline_delta_check'] = comparison
        records.append(sample)
        good = (not sample['apps'] and all(d['utilization'] == 0 for d in current.values())
                and all(current[u]['gpu_bytes'] == 0 for u in uuids) and not comparison['violations'])
        stable = previous is not None and current == previous
        consecutive = consecutive + 1 if good and stable else (1 if good else 0)
        if consecutive >= 2:
            return dict(status='verified', observations=records, baseline=current,
                        anchor=anchor, scope='all eight GPUs; selected zero; stable nonselected baseline-relative memory',
                        nonselected_delta_tolerance_bytes=NONSELECTED_DELTA_TOLERANCE_BYTES)
        previous = current
        time.sleep(.25)
    return dict(status='blocked', observations=records, anchor=anchor,
                nonselected_delta_tolerance_bytes=NONSELECTED_DELTA_TOLERANCE_BYTES)


class Monitor:
    def __init__(self, directory, server, config, authenticated, uuids, note, baseline):
        self.directory, self.server, self.config = directory, server, config
        self.authenticated, self.uuids, self.note = authenticated, uuids, note
        self.baseline = baseline
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
                row['baseline_delta_check'] = device_deltas(row, self.baseline, self.uuids)
                row['gpu_bytes'] = max(d['gpu_bytes'] for d in selected)
                row.update(host_sample(self.server.pid))
                row['nvml_pid_set_changed'] = bool(self.samples and
                    {a['pid'] for a in row['apps']} != {a['pid'] for a in self.samples[-1]['apps']})
                self.samples.append(row)
                stream.write(json.dumps(row, allow_nan=False)+'\n')
                stream.flush()
                ct.require(not row['baseline_delta_check']['violations'],
                           '; '.join(row['baseline_delta_check']['violations']))
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
        result.update(per_device=rows, baseline=self.baseline,
                      nonselected_delta_tolerance_bytes=NONSELECTED_DELTA_TOLERANCE_BYTES,
                      all_device_delta_high_water_bytes={uuid:max((s['baseline_delta_check']['per_uuid'][uuid]['delta_bytes']
                         for s in self.samples), default=None) for uuid in self.baseline},
                      identity_mode='device_exclusive_local_custody',
                      measurement_scope='all eight physical devices; selected two independently budgeted',
                      owned_process_gpu_peak_bytes=None, owned_gpu_attribution='unknown',
                      absence_of_foreign_interference_proven=False,
                      proc_enumeration_resamples=sum(len(s['proc_enumeration_resamples']) for s in self.samples),
                      nvml_pid_set_changes=sum(s['nvml_pid_set_changed'] for s in self.samples),
                      pss_policy='server tree sampled once; never sum per-device copies')
        return result
