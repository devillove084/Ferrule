#!/usr/bin/env python3
"""Internal single-thread Linux child custodian for test_vllm_qwen35.run_process.

No benchmark/model logic. Not a security sandbox against hostile same-UID code,
ptrace, external launch services or privileged namespace manipulation.
"""
import ctypes
import json
import os
from pathlib import Path
import select
import signal
import socket
import sys
import time

WAIT = os.WEXITED | os.WNOHANG | os.WNOWAIT | 0x40000000  # Linux __WALL: clone children too
CLEANUP_SECONDS = 5


def identity(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, birth_ticks=int(fields[19]), ppid=int(fields[1]),
                pgrp=int(fields[2]), session=int(fields[3]))


def checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, allow_nan=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def enable_subreaper():
    if sys.platform != 'linux':
        raise RuntimeError('Linux subreaper required')
    # A fresh interpreter has no other waiter/thread. Installing a real handler
    # clears inherited SIG_IGN/SA_NOCLDWAIT; the handler must never reap.
    signal.signal(signal.SIGCHLD, lambda signum, frame: None)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), 'PR_SET_CHILD_SUBREAPER')
    value = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(value), 0, 0, 0) != 0 or value.value != 1:
        raise RuntimeError('PR_GET_CHILD_SUBREAPER did not verify enabled state')
    if len(list(Path('/proc/self/task').iterdir())) != 1:
        raise RuntimeError('Custodian must have exactly one thread')
    # With no initial children, ECHILD is also a waitid/__WALL capability check.
    try:
        os.waitid(os.P_ALL, 0, WAIT)
    except ChildProcessError:
        return
    raise RuntimeError('Unexpected pre-launch child in isolated custodian')


class Custodian:
    def __init__(self, launch_id, journal=None):
        self.pid = os.getpid()
        self.journal = journal
        self.root_pid = None
        self.root_code = None
        self.entries = []
        self.active = {}
        self.custody = dict(status='tracking', method='linux-subreaper-waitid',
                            containment_verified=True, launch_id=launch_id,
                            helper_identity=identity(self.pid), root_identity=None, root_session_ready=False,
                            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                            entries=self.entries, errors=[], reporting_errors=[], echild=False)

    def persist(self):
        if self.journal:
            try:
                checkpoint(self.journal, self.custody)
            except OSError as exc:
                error = f'Custody checkpoint: {type(exc).__name__}: {exc}'
                if error not in self.custody['reporting_errors']:
                    self.custody['reporting_errors'].append(error)

    def register(self, pid):
        # This is the authority check, NOT procfs membership/name matching. No
        # other thread or SIGCHLD handler reaps, so a child exiting here stays a
        # zombie and cannot have its PID recycled before our explicit reap.
        os.waitid(os.P_PID, pid, WAIT)
        current = identity(pid)
        if current['ppid'] != self.pid:
            raise RuntimeError('waitid/procfs direct-child identity mismatch')
        if pid not in self.active:
            is_root = pid == self.root_pid and self.custody['root_identity'] is None
            row = dict(identity=current, role='root' if is_root else 'adopted',
                       signals=[], reaped=False, exit_code=None, first_observed_at=time.monotonic())
            self.entries.append(row)
            self.active[pid] = row
            if is_root:
                self.custody['root_identity'] = current
            self.persist()
        elif current['birth_ticks'] != self.active[pid]['identity']['birth_ticks']:
            raise RuntimeError('Direct-child PID birth changed before reap')
        return self.active[pid]

    def signal_child(self, pid, sig):
        row = self.register(pid)
        # register has proved an unreaped, stable, direct child. Never use this
        # numeric-PID signaling procedure in a threaded/general-purpose parent.
        try:
            os.kill(pid, sig)
            row['signals'].append(int(sig))
        except ProcessLookupError:
            # Exiting child remains waitable; next waitid determines its state.
            pass

    def children(self):
        # Only our kernel-maintained child list, never arbitrary /proc PIDs.
        return [int(pid) for pid in Path(f'/proc/self/task/{self.pid}/children').read_text().split()]

    def reap_available(self):
        # A steady stream of exits must not starve the cleanup deadline/signals.
        for _ in range(256):
            try:
                event = os.waitid(os.P_ALL, 0, WAIT)
            except ChildProcessError:
                if self.active:
                    raise RuntimeError('ECHILD with unreconciled child identities')
                self.custody['echild'] = True
                return True
            if event is None:
                return False
            row = self.register(event.si_pid)
            # Record identity before the one and only reap. No signaling may use
            # this row again once consumed, even if the numeric PID reappears.
            consumed = os.waitid(os.P_PID, event.si_pid, WAIT & ~os.WNOWAIT)
            if consumed is None or consumed.si_pid != event.si_pid:
                raise RuntimeError('waitid observation/reap mismatch')
            code = event.si_status if event.si_code == os.CLD_EXITED else -event.si_status
            row.update(reaped=True, exit_code=code, reaped_at=time.monotonic())
            del self.active[event.si_pid]
            if row['role'] == 'root' and self.root_code is None:
                self.root_code = code
            self.persist()
        return False

    def launch(self, argv, input_fd, ready_seconds=5):
        read_gate, write_gate = os.pipe()
        read_ready, write_ready = os.pipe()
        self.root_pid = os.fork()
        if self.root_pid == 0:
            try:
                os.close(write_gate)
                os.close(read_ready)
                os.setsid()
                os.write(write_ready, b'R')
                os.close(write_ready)
                if os.read(read_gate, 1) != b'G':
                    os._exit(125)
                os.close(read_gate)
                os.dup2(input_fd, 0)
                os.close(input_fd)
                # Do not inherit helper handlers into application code.
                for sig in (signal.SIGCHLD, signal.SIGTERM, signal.SIGINT):
                    signal.signal(sig, signal.SIG_DFL)
                os.execvpe(argv[0], argv, os.environ)
            except BaseException as exc:
                os.write(2, f'Root exec: {type(exc).__name__}: {exc}\n'.encode())
                os._exit(127)
        os.close(read_gate)
        os.close(write_ready)
        self.custody['echild'] = False
        try:
            # The child cannot exec before this parent has recorded its final
            # session identity. Waiting for a byte also bounds setsid failure.
            if not select.select([read_ready], [], [], ready_seconds)[0] or os.read(read_ready, 1) != b'R':
                raise RuntimeError('Root post-setsid readiness missing or expired')
            row = self.register(self.root_pid)
            current = row['identity']
            if current['pid'] != self.root_pid or current['pgrp'] != self.root_pid or current['session'] != self.root_pid:
                raise RuntimeError('Root post-setsid group/session identity mismatch')
            self.custody['root_session_ready'] = True
            self.persist()
            if self.custody['reporting_errors']:
                raise RuntimeError('Root post-setsid custody checkpoint failed before exec')
            os.write(write_gate, b'G')
        finally:
            os.close(read_ready)
            os.close(write_gate)

    def cleanup(self, seconds=CLEANUP_SECONDS):
        deadline = time.monotonic() + seconds
        self.custody['status'] = 'cleaning'
        term_at = {}
        # Freeze current direct fork sources before reaping or graceful requests.
        for pid in self.children():
            self.signal_child(pid, signal.SIGSTOP)
        while True:
            if self.reap_available():
                self.custody['status'] = 'proven_drained'
                self.persist()
                return
            if time.monotonic() >= deadline:
                self.custody['status'] = 'unknown'
                self.custody['errors'].append('Cleanup deadline before kernel ECHILD')
                self.persist()
                return
            # Stop root/current fork sources before requesting exit. TERM+CONT
            # permits graceful cleanup, including late forks; subsequent sweeps
            # stop/kill sources and handle children adopted on every parent exit.
            for pid in self.children():
                row = self.register(pid)
                key = (pid, row['identity']['birth_ticks'])
                if key not in term_at:
                    self.signal_child(pid, signal.SIGSTOP)
                    self.signal_child(pid, signal.SIGTERM)
                    self.signal_child(pid, signal.SIGCONT)
                    term_at[key] = time.monotonic()
                elif time.monotonic() - term_at[key] >= .15:
                    self.signal_child(pid, signal.SIGSTOP)
                    self.signal_child(pid, signal.SIGKILL)
            time.sleep(.01)


def main():
    control_fd, input_fd, result_path, launch_id = sys.argv[1:]
    control = socket.socket(fileno=int(control_fd))
    control.set_inheritable(False)
    os.set_inheritable(int(input_fd), False)
    stopped = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: stopped.append(signal.Signals(signum).name))
    result = dict(returncode=None, timed_out=False, cleanup_errors=[], custody=dict(status='unknown'))
    owner = None
    try:
        enable_subreaper()
        owner = Custodian(launch_id)
        control.sendall((json.dumps(dict(kind='ready', protocol=1, launch_id=launch_id,
                            helper_identity=owner.custody['helper_identity'], containment_verified=True,
                            method='linux-subreaper-waitid')) + '\n').encode())
        # No application may launch before the parent has validated the handshake.
        authorization_deadline = time.monotonic() + 5
        data = bytearray()
        while not data.endswith(b'\n'):
            remaining = authorization_deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Launch authorization deadline')
            control.settimeout(remaining)
            chunk = control.recv(65536)
            if not chunk:
                raise RuntimeError('Parent closed before launch authorization')
            data.extend(chunk)
            if len(data) > 4*1024*1024:
                raise RuntimeError('Launch configuration too large')
        config = json.loads(data)
        if config['launch_id'] != launch_id:
            raise RuntimeError('Launch identity mismatch')
        owner.journal = config.get('journal')
        owner.persist()
        if config['argv'] is not None:
            deadline = time.monotonic() + config['timeout']
            owner.launch(config['argv'], int(input_fd), min(5, config['timeout']))
            while owner.root_code is None:
                owner.reap_available()
                for pid in owner.children():
                    owner.register(pid)
                if owner.root_code is not None:
                    break
                if stopped:
                    raise RuntimeError('Owner/helper interrupted: ' + stopped[0])
                if time.monotonic() >= deadline:
                    result.update(timed_out=True, primary_error='TimeoutExpired: owned root deadline')
                    break
                if select.select([control], [], [], .01)[0]:
                    message = control.recv(65536)
                    raise RuntimeError('Owner cancellation' if message else 'Owner connection lost')
        else:
            result['returncode'] = 0  # capability probe: no application launched
    except BaseException as exc:
        result['primary_error'] = f'{type(exc).__name__}: {exc}'
        result['error'] = 'Helper launch/execution failed'
    finally:
        if owner is not None:
            try:
                owner.cleanup()
            except BaseException as exc:
                owner.custody.update(status='unknown', echild=False)
                owner.custody['errors'].append(f'{type(exc).__name__}: {exc}')
            result['custody'] = owner.custody
            result['cleanup_errors'] = owner.custody['errors']
            result['reporting_errors'] = owner.custody['reporting_errors']
            if result['reporting_errors']:
                result.setdefault('error', 'Custody reporting failed')
            if owner.root_pid is not None:
                result['returncode'] = owner.root_code
        if result['custody']['status'] != 'proven_drained':
            result['error'] = 'Helper could not prove complete descendant drain'
        checkpoint(result_path, result)
        control.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
