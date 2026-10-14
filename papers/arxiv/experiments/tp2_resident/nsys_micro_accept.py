#!/usr/bin/env python3
"""Read-only Nsight 2025.3 SQLite micro acceptance; never launches GPU work."""
import argparse
import json
from pathlib import Path
import sqlite3

EXPECTED = {'GPU-b980f22b-58a3-fc88-21c1-ae86053d8535', 'GPU-63f1bd01-ddf9-a742-ee48-9389892493d9'}


def require(value, message):
    if not value:
        raise ValueError(message)


def accept(connection):
    connection.row_factory = sqlite3.Row
    rows = lambda sql: [dict(r) for r in connection.execute(sql)]
    processes = {r['globalPid']: r['pid'] for r in rows('SELECT globalPid,pid FROM PROCESSES')}
    gpu_rows = rows('SELECT id,uuid FROM TARGET_INFO_GPU')
    require(len(gpu_rows) == 8 and len({r['id'] for r in gpu_rows}) == 8, 'Incomplete physical GPU inventory')
    gpus = {r['id']: 'GPU-'+r['uuid'].removeprefix('GPU-') for r in gpu_rows}
    mappings = {}
    for row in rows('SELECT pid,cudaId,gpuId FROM TARGET_INFO_CUDA_DEVICE'):
        key = row['pid'], row['cudaId']
        require(key not in mappings, 'Ambiguous per-process CUDA mapping')
        mappings[key] = row['gpuId']
    strings = {r['id']: r['value'] for r in rows('SELECT id,value FROM StringIds')}

    def mapped(row):
        pid = row['globalPid'] // 0x1000000 % 0x1000000
        require(processes.get(row['globalPid']) == pid, 'Serialized PID and process table disagree')
        key = pid, row['deviceId']
        require(key in mappings and mappings[key] in gpus, 'Unmapped CUDA device; never guess physical ID')
        uuid = gpus[mappings[key]]
        require(uuid in EXPECTED, 'Kernel/memcpy on nonselected physical GPU')
        require(row['end'] > row['start'], 'Invalid GPU event interval')
        return dict(row, pid=pid, physical_gpu_id=mappings[key], physical_uuid=uuid)

    kernels = [mapped(r) for r in rows('SELECT globalPid,deviceId,start,end,correlationId,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL')]
    copies = [mapped(r) for r in rows('SELECT globalPid,deviceId,start,end,correlationId,copyKind,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY')]
    apis = rows('SELECT globalTid,start,end,correlationId,nameId,returnValue FROM CUPTI_ACTIVITY_KIND_RUNTIME')
    require(len(kernels) >= 2 and {r['physical_uuid'] for r in kernels} == EXPECTED,
            'Need actual kernel activity on both physical UUIDs, not API-only/exit-zero')
    api_keys = {}
    for row in apis:
        key = row['globalTid']//0x1000000%0x1000000, row['correlationId']
        api_keys.setdefault(key, []).append(row)
    for row in kernels + copies:
        matches = api_keys.get((row['pid'], row['correlationId']), [])
        require(len(matches) == 1 and matches[0]['returnValue'] == 0, 'Missing/failed correlated runtime API')
    copy_kinds = {r['id']: r['name'] for r in rows('SELECT id,name FROM ENUM_CUDA_MEMCPY_OPER')}
    for uuid in EXPECTED:
        directions = {copy_kinds[r['copyKind']] for r in copies if r['physical_uuid'] == uuid and r['bytes'] > 0}
        require({'CUDA_MEMCPY_KIND_HTOD','CUDA_MEMCPY_KIND_DTOH'} <= directions, 'Missing H2D/D2H on selected GPU')
    for row in kernels:
        row['kernel_name'] = strings[row['demangledName']]
        require('vectorAdd' in row['kernel_name'], 'Unexpected micro kernel')
    diags = rows('SELECT * FROM DIAGNOSTIC_EVENT')
    severity = {r['id']: r['name'] for r in rows('SELECT id,name FROM ENUM_DIAGNOSTIC_SEVERITY_LEVEL')}
    fatal = [r for r in diags if severity.get(r['severity'], '').lower() in ('error','fatal')
             or any(s in r['text'].lower() for s in ('numtpcs','stringstorage','targetprofilingfailed'))]
    require(not fatal, 'Fatal collector/importer diagnostics')
    warnings = [r for r in diags if severity.get(r['severity']) == 'Warning']
    kernel_pids = {r['globalPid'] for r in kernels}
    require(not any('profiling might have not been started' in r['text'] and r['globalPid'] in kernel_pids for r in warnings),
            'CUDA-start warning on an actual CUDA workload process')
    loaded = [r for r in diags if 'Loaded CUPTI library:' in r['text']]
    require({r['globalPid'] for r in loaded} >= kernel_pids, 'Missing actual loaded CUPTI evidence')
    return dict(status='pass_sqlite', kernel_rows=len(kernels), kernel_events=kernels,
        distinct_physical_kernel_uuids=sorted({r['physical_uuid'] for r in kernels}),
        distinct_physical_kernel_gpu_ids=sorted({r['physical_gpu_id'] for r in kernels}),
        logical_kernel_device_ids=sorted({r['deviceId'] for r in kernels}),
        memcpy_rows=len(copies), memcpy_events=copies, runtime_rows=len(apis),
        fatal_diagnostics=fatal, warnings=warnings, loaded_cupti=loaded,
        mapping='CUPTI globalPid -> PROCESSES.pid; (pid, deviceId) -> TARGET_INFO_CUDA_DEVICE.(pid,cudaId) -> gpuId -> TARGET_INFO_GPU.id/uuid',
        schema_reference='https://docs.nvidia.com/nsight-systems/2025.3/UserGuide/index.html#sqlite-schema-reference')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pair', type=Path)
    args = parser.parse_args()
    pair = args.pair.resolve()
    case = pair/'runs/P1/session'
    with sqlite3.connect((case/'micro.sqlite').as_uri()+'?mode=ro', uri=True) as connection:
        result = accept(connection)
    release = json.loads((pair/'gpu-release.json').read_text())
    require(all(release.get(k) is True for k in ('authenticated_drained','supervisor_success',
        'original_b0_verified','b1_returned','local_contexts_closed')), 'Capture lifecycle/release is not successful')
    worker = json.loads((case/'micro-worker.json').read_text())
    require(worker['status'] == 'work_complete' and worker['shutdown']['returncode'] == 0
        and not worker['shutdown']['timed_out'] and not worker['monitor']['errors'], 'Worker/CLI capture failed')
    text = (case/'server.log').read_text()
    require(not any(s in text.lower() for s in ('fatal','error','failed','numtpcs','stringstorage')), 'CLI/importer log failure')
    result.update(status='pass', release=release, sqlite=str(case/'micro.sqlite'))
    with (pair/'acceptance.json').open('x') as out:
        json.dump(result, out, indent=2)
        out.write('\n')
    print(json.dumps({k: result[k] for k in ('status','kernel_rows','distinct_physical_kernel_uuids',
          'distinct_physical_kernel_gpu_ids','logical_kernel_device_ids','memcpy_rows','runtime_rows')}, indent=2))


if __name__ == '__main__':
    main()
