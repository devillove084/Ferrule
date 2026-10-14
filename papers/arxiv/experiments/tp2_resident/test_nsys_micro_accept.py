"""CPU SQLite fixtures: two logical device-zero processes are distinct physical GPUs."""
from pathlib import Path
import sqlite3
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nsys_micro_accept as audit
import nsys_micro as micro


class Acceptance(unittest.TestCase):
    def setUp(self):
        self.c = sqlite3.connect(':memory:')
        self.addCleanup(self.c.close)
        self.c.executescript('''
            CREATE TABLE PROCESSES(globalPid,pid);
            CREATE TABLE TARGET_INFO_GPU(id,uuid);
            CREATE TABLE TARGET_INFO_CUDA_DEVICE(pid,cudaId,gpuId);
            CREATE TABLE StringIds(id,value);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(globalPid,deviceId,start,end,correlationId,demangledName);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(globalPid,deviceId,start,end,correlationId,copyKind,bytes);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(globalTid,start,end,correlationId,nameId,returnValue);
            CREATE TABLE ENUM_CUDA_MEMCPY_OPER(id,name);
            CREATE TABLE DIAGNOSTIC_EVENT(globalPid,severity,text);
            CREATE TABLE ENUM_DIAGNOSTIC_SEVERITY_LEVEL(id,name);
            INSERT INTO StringIds VALUES(1,'vectorAdd(float*,float*,float*,int)');
            INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES(1,'CUDA_MEMCPY_KIND_HTOD'),(2,'CUDA_MEMCPY_KIND_DTOH');
            INSERT INTO ENUM_DIAGNOSTIC_SEVERITY_LEVEL VALUES(1,'Info'),(2,'Warning'),(3,'Error');
        ''')
        uuids = list(micro.shared.UUIDS)+[u for u in micro.INVENTORY if u not in micro.shared.UUIDS]
        self.c.executemany('INSERT INTO TARGET_INFO_GPU VALUES(?,?)', enumerate(u.removeprefix('GPU-') for u in uuids))
        for gpu,pid in enumerate((10,20)):
            global_pid = (1<<48)+(pid<<24)
            self.c.execute('INSERT INTO PROCESSES VALUES(?,?)',(global_pid,pid))
            self.c.execute('INSERT INTO TARGET_INFO_CUDA_DEVICE VALUES(?,0,?)',(pid,gpu))
            self.c.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,0,10,20,1,1)',(global_pid,))
            for correlation,kind in ((2,1),(3,2)):
                self.c.execute('INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(?,0,10,20,?,?,200000)',(global_pid,correlation,kind))
            for correlation in (1,2,3):
                self.c.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,1,2,?,1,0)',(global_pid+pid,correlation))
            self.c.execute('INSERT INTO DIAGNOSTIC_EVENT VALUES(?,1,?)',(global_pid,'Loaded CUPTI library: /official/libcupti.so.12.9'))

    def test_distinct_physical_uuids_not_distinct_logical_ordinals(self):
        result = audit.accept(self.c)
        self.assertEqual(result['logical_kernel_device_ids'], [0])
        self.assertEqual(result['distinct_physical_kernel_gpu_ids'], [0,1])
        self.assertEqual(set(result['distinct_physical_kernel_uuids']), audit.EXPECTED)

    def test_api_only_is_not_pass(self):
        self.c.execute('DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL')
        with self.assertRaises(ValueError): audit.accept(self.c)

    def test_missing_pid_mapping_cannot_be_guessed(self):
        self.c.execute('DELETE FROM TARGET_INFO_CUDA_DEVICE WHERE pid=20')
        with self.assertRaises(ValueError): audit.accept(self.c)

    def test_same_physical_gpu_is_not_two_device_pass(self):
        self.c.execute('UPDATE TARGET_INFO_CUDA_DEVICE SET gpuId=0')
        with self.assertRaises(ValueError): audit.accept(self.c)

    def test_fatal_importer_diagnostic_rejected(self):
        self.c.execute("INSERT INTO DIAGNOSTIC_EVENT VALUES(0,3,'NumTpcs not initialized')")
        with self.assertRaises(ValueError): audit.accept(self.c)

    def test_missing_d2h_rejected(self):
        self.c.execute('DELETE FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind=2')
        with self.assertRaises(ValueError): audit.accept(self.c)

    def test_failed_correlated_api_rejected(self):
        self.c.execute('UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET returnValue=1')
        with self.assertRaises(ValueError): audit.accept(self.c)


if __name__ == '__main__':
    unittest.main()
