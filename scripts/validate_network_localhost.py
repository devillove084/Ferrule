#!/usr/bin/env python3
"""Opt-in, single-host acceptance of the real CUDA-linear network CLI.

No build, model download, remote host, mock endpoint, or CPU production fallback.
Credentials live only in a private temporary directory and are removed on exit.
See scripts/validate_network_localhost.md for the entry point and custody limits.
This runner's PID observations are not an authenticated custodian drain receipt.
"""
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def credentials(root):
    def run(*args):
        subprocess.run(["openssl", *map(str, args)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=10)
    # Do not inherit host openssl.cnf extensions (some hosts duplicate -addext).
    config = root / "openssl.cnf"
    config.write_text("[req]\ndistinguished_name=dn\n[dn]\n[ca]\n"
                      "basicConstraints=critical,CA:TRUE\n"
                      "keyUsage=critical,keyCertSign,cRLSign\nsubjectKeyIdentifier=hash\n")
    run("req", "-config", config, "-extensions", "ca", "-x509", "-sha256",
        "-newkey", "rsa:2048", "-nodes", "-days", "1",
        "-subj", "/CN=Ferrule TEST ONLY CA", "-keyout", root / "ca.key",
        "-out", root / "ca.pem")
    for serial, name in enumerate(("server", "client", "other"), 10):
        run("req", "-config", config, "-sha256", "-newkey", "rsa:2048", "-nodes", "-subj", f"/CN=TEST ONLY {name}",
            "-keyout", root / f"{name}.key", "-out", root / f"{name}.csr")
        ext = root / f"{name}.ext"
        ext.write_text("basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature\n"
                       "subjectAltName=DNS:localhost\nextendedKeyUsage=" +
                       ("serverAuth" if name == "server" else "clientAuth") + "\n")
        run("x509", "-req", "-in", root / f"{name}.csr", "-CA", root / "ca.pem",
            "-CAkey", root / "ca.key", "-set_serial", str(serial), "-sha256",
            "-days", "1", "-extfile", ext, "-out", root / f"{name}.pem")
        run("verify", "-CAfile", root / "ca.pem", root / f"{name}.pem")
    for key in root.glob("*.key"):
        key.chmod(0o600)


def tls(root, own, peer):
    return {"ca": str(root / "ca.pem"), "certificate_chain": str(root / f"{own}.pem"),
            "private_key": str(root / f"{own}.key"), "peer_leaf": str(root / f"{peer}.pem")}


def proc_identity(pid):
    try:
        # starttime prevents treating a recycled PID as our child during cleanup.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19], fields[0]
    except FileNotFoundError:
        return None


def children(pid):
    try:
        return [int(p) for p in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()]
    except FileNotFoundError:
        return []


class Run:
    def __init__(self, binary, output, env):
        self.binary, self.output, self.env = binary, output, env
        self.processes, self.seen, self.files = [], {}, []
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.forced = False
        self.observer = threading.Thread(target=self.observe, daemon=True)
        self.observer.start()

    def spawn(self, name, args):
        stdout = self.output / f"{name}.stdout"
        stderr = self.output / f"{name}.stderr"
        out, err = stdout.open("wb"), stderr.open("wb")
        self.files.extend([out, err])
        command = [str(self.binary), *args]
        p = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                             env=self.env, start_new_session=True)
        entry = {"name": name, "pid": p.pid, "argv": command,
                 "stdout": str(stdout), "stderr": str(stderr), "process": p}
        with self.lock:
            self.processes.append(entry)
        return entry

    def observe(self):
        while not self.stop.wait(0.002):
            with self.lock:
                endpoints = [e for e in self.processes if e["name"].startswith("rank-")]
            for endpoint in endpoints:
                for pid in children(endpoint["pid"]):
                    identity = proc_identity(pid)
                    if identity:
                        with self.lock:
                            self.seen.setdefault(pid, {"pid": pid, "parent": endpoint["pid"],
                                                      "starttime": identity[0]})

    def bound(self, entry):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            text = Path(entry["stderr"]).read_text()
            for line in text.splitlines():
                if line.startswith("rank-serve bound "):
                    address = line.split()[2].rstrip(";")
                    require(address.startswith("127.0.0.1:"), "non-loopback bind")
                    return address
            require(entry["process"].poll() is None, f"endpoint failed before bind: {text}")
            time.sleep(0.005)
        raise RuntimeError("bind-status deadline exceeded (no TCP probe used)")

    def wait(self, entry, deadline):
        entry["exit_code"] = entry["process"].wait(timeout=max(0.001, deadline - time.monotonic()))
        return entry["exit_code"]

    def wait_children(self, controller):
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            with self.lock:
                live = [v.copy() for v in self.seen.values() if proc_identity(v["pid"])]
            if len(live) == 2:
                return live
            require(controller["process"].poll() is None, "controller exited before two children")
            time.sleep(0.002)
        raise RuntimeError("two real rank children not observed")

    def cleanup(self):
        # Only sessions created by this Run are signalled, never name-based pkill.
        for entry in reversed(self.processes):
            p = entry["process"]
            if p.poll() is None:
                self.forced = True
                os.killpg(p.pid, signal.SIGKILL)
                p.wait(timeout=5)
            entry["exit_code"] = p.returncode
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            remaining = []
            with self.lock:
                seen = list(self.seen.values())
            for item in seen:
                pid = item["pid"]
                identity = proc_identity(pid)
                if identity and identity[0] == item["starttime"]:
                    try:
                        os.waitpid(pid, os.WNOHANG)  # only adopted children, if any
                    except ChildProcessError:
                        pass
                    identity = proc_identity(pid)
                    if identity and identity[0] == item["starttime"]:
                        remaining.append(item)
            if not remaining:
                break
            time.sleep(0.01)
        for item in remaining:
            self.forced = True
            os.kill(item["pid"], signal.SIGKILL)
            try:
                os.waitpid(item["pid"], 0)
            except ChildProcessError:
                pass
        self.stop.set()
        self.observer.join(timeout=1)
        for f in self.files:
            f.close()
        records = [{k: v for k, v in e.items() if k != "process"} for e in self.processes]
        for item in self.seen.values():
            identity = proc_identity(item["pid"])
            item["absent"] = identity is None or identity[0] != item["starttime"]
        return {"processes": records, "rank_children": list(self.seen.values()),
                "forced_cleanup": self.forced,
                "children_absent": all(v["absent"] for v in self.seen.values())}


def case(binary, output, secret, env, mode, fault):
    name = mode + "-" + fault
    folder = output / name
    folder.mkdir()
    local = secret / name
    local.mkdir()
    fixture = {"rows": 2, "in_features": 3, "out_features": 5,
               "weight": [n / 8 for n in range(1, 16)], "input": list(range(1, 7))}
    fixture_path = folder / "fixture.json"
    write_json(fixture_path, fixture)
    oracle = [sum(fixture["input"][r * 3 + k] * fixture["weight"][c * 3 + k]
                  for k in range(3)) for r in range(2) for c in range(5)]
    limits = dict(max_frame_bytes=8388608, max_config_bytes=4194304,
                  max_command_bytes=4194304, max_output_bytes=4194304, max_error_bytes=4096)
    run = Run(binary, folder, env)
    started = time.monotonic()
    evidence = {"case": name, "mode": mode, "fault": fault, "passed": False}
    try:
        ranks, endpoints = [], []
        for rank in range(2):
            binding = {"run_id": f"TEST-{os.getpid()}-{name}", "controller_id": "TEST-controller",
                       "node_id": "localhost", "identity": {"epoch": 991, "rank": rank,
                       "owner_instance": rank + 1}, "target": {"mode": mode.replace("-", "_"),
                       "world": 2, "device": rank, "in_features": 3, "out_features": 5, "max_rows": 2}}
            config = {"binding": binding, "startup_timeout_ms": 4000,
                      "command_timeout_ms": 1500, "limits": limits,
                      "fixture": str(fixture_path), "tls": tls(secret, "server", "client")}
            if fault == "client-pin":
                config["tls"]["peer_leaf"] = str(secret / "other.pem")
            config_path = local / f"endpoint-{rank}.json"
            write_json(config_path, config)
            entry = run.spawn(f"rank-{rank}", ["rank-serve", "--config", str(config_path),
                                               "--bind", "127.0.0.1:0"])
            address = run.bound(entry)
            endpoints.append(entry)
            remote = {k: v for k, v in config.items() if k not in ("fixture", "tls")}
            remote.update(address=address, server_name="localhost", tls=tls(secret, "client", "server"))
            if fault == "server-pin":
                remote["tls"]["peer_leaf"] = str(secret / "other.pem")
            elif fault == "ca":
                remote["tls"]["ca"] = str(secret / "other.pem")
            elif fault == "san":
                remote["server_name"] = "wrong.invalid"
            ranks.append(remote)
        evidence["endpoints"] = [{"address": r["address"], "binding": r["binding"]} for r in ranks]
        manifest = local / "manifest.json"
        write_json(manifest, {"ranks": ranks})
        inject = fault in ("child-stop", "child-exit", "controller-disconnect")
        controller = run.spawn("controller", ["bench-parallel", "--mode", mode, "--ranks", "2",
                    "--fixture", str(fixture_path), "--remote-manifest", str(manifest),
                    "--rank-timeout-ms", "1500", "--warmup", "1", "--iterations",
                    "100000" if inject else "20", "--emit-values", "--json"])
        if inject:
            observed = run.wait_children(controller)
            time.sleep(0.15)
            require(controller["process"].poll() is None, "workload ended before injection")
            victim = controller["pid"] if fault == "controller-disconnect" else observed[0]["pid"]
            sig = signal.SIGSTOP if fault == "child-stop" else signal.SIGKILL
            identity = proc_identity(victim)
            require(identity is not None, "victim exited before injection")
            os.kill(victim, sig)
            evidence["injection"] = {"pid": victim, "signal": sig.name,
                                     "seconds_after_start": time.monotonic() - started}
        deadline = time.monotonic() + 12
        code = run.wait(controller, deadline)
        exits = [run.wait(e, deadline) for e in endpoints]
        stdout = Path(controller["stdout"]).read_text()
        stderr = Path(controller["stderr"]).read_text()
        if fault == "healthy":
            require(code == 0 and exits == [0, 0], "healthy CLI subprocess failed")
            report = json.loads(stdout)
            require(len(report["outputs"]) == 2, "missing rank outputs")
            errors = []
            for rank, result in enumerate(report["outputs"]):
                require(result["rank"] == rank and len(result["values"]) == len(oracle), "rank/shape mismatch")
                for actual, expected in zip(result["values"], oracle):
                    require(math.isfinite(actual) and abs(actual - expected) <= 1e-5 + 1e-5 * abs(expected), "oracle mismatch")
                    errors.append(abs(actual - expected))
            if mode != "dp":
                require(report["outputs"][0]["values"] == report["outputs"][1]["values"], "TP Apply differs")
            require(report["transport"] == "host-staged-mtls" and report["rank_backend"] == "network",
                    "not a network benchmark")
            evidence.update(report=report, max_abs_error=max(errors), oracle=oracle)
        else:
            require(code != 0 and stdout == "", "failed run published a success report")
            if fault in ("child-stop", "child-exit"):
                require("unknown" in stderr.lower(), "lost child was not reported Unknown")
            if not inject:
                require("authentication" in stderr.lower() or "initialization" in stderr.lower()
                        or "startup" in stderr.lower(), "unexpected authentication failure")
        evidence["passed"] = True
    except Exception as error:
        evidence["error"] = str(error)
    finally:
        evidence.update(run.cleanup())
        evidence["elapsed_seconds"] = time.monotonic() - started
        if evidence["forced_cleanup"] or not evidence["children_absent"]:
            evidence.update(passed=False, cleanup_error="bounded normal cleanup failed")
        if fault == "healthy" and len(evidence["rank_children"]) != 2:
            evidence.update(passed=False, topology_error="did not observe two separate rank children")
        write_json(folder / "evidence.json", evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True, help="already-built Unix+CUDA ferrule (never builds)")
    parser.add_argument("--devices", required=True, help="two explicitly assigned physical GPU ordinals, e.g. 2,3")
    parser.add_argument("--output", type=Path, required=True, help="new evidence directory")
    parser.add_argument("--approve-linear-gpu", action="store_true", required=True)
    args = parser.parse_args()
    require(os.name == "posix" and Path("/proc/self/task").exists(), "Linux /proc required for PID custody evidence")
    devices = args.devices.split(",")
    require(len(devices) == 2 and all(d.isdigit() for d in devices) and devices[0] != devices[1], "assign two distinct GPUs")
    binary = args.binary.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Reap only descendants of this harness if an injected parent exits first.
    require(ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0, "cannot establish test subreaper")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.devices)
    evidence = {"schema": "ferrule.localhost-network.v1", "scope": "single-host real CLI linear ranks",
                "binary": str(binary), "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "devices": devices, "multi_host": False, "model_serving": False,
                "nvcc_invoked": False, "runtime_fault_ledger": "separate cargo network tests, not inferred from CLI exits",
                "cases": []}
    try:
        with tempfile.TemporaryDirectory(prefix="ferrule-network-TEST-ONLY-") as temp:
            secret = Path(temp)
            credentials(secret)
            matrix = [(m, "healthy") for m in ("dp", "tp-column", "tp-row")]
            matrix += [("tp-row", f) for f in ("server-pin", "client-pin", "ca", "san",
                                               "child-stop", "child-exit", "controller-disconnect")]
            for mode, fault in matrix:
                result = case(binary, output, secret, env, mode, fault)
                evidence["cases"].append(result)
                print(f"{result['case']}: {'PASS' if result['passed'] else 'FAIL'}", flush=True)
                write_json(output / "summary.json", evidence)
                if not result["passed"]:
                    break
    except Exception as error:
        evidence["error"] = str(error)
    evidence["passed"] = len(evidence["cases"]) == 10 and all(c["passed"] for c in evidence["cases"])
    evidence["credentials_removed"] = True
    write_json(output / "summary.json", evidence)
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
