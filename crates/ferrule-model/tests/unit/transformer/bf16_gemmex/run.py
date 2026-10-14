"""Run one pinned integration gate only after explicit sole-GPU permission."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[6]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("label")
    parser.add_argument("--execute-cuda", action="store_true")
    parser.add_argument("--sole-owner-permission", action="store_true")
    args = parser.parse_args()
    if not args.execute_cuda or not args.sole_owner_permission:
        parser.error("Both explicit GPU permission flags are required")
    directory = args.directory.resolve()
    prepared = json.loads((directory / "PREPARED.json").read_text())
    if args.label not in prepared["plans"]:
        parser.error("Unknown gate")
    spec = importlib.util.spec_from_file_location(
        "gemmex_custody", ROOT / "crates/ferrule-backend/tests/numeric_bf16_gemv/contract_run.py"
    )
    custody = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custody)
    custody.verify(prepared)
    plan = prepared["plans"][args.label]
    env = os.environ.copy()
    env.update(plan["environment"])
    # Full40 retains its maintained default Hello/prefill capture configuration.
    os.environ.pop("FERRULE_NUMERIC_BF16_CAPTURE_PHASES", None)
    linked = subprocess.check_output(["ldd", plan["command"][0]], env=env, text=True, timeout=15)
    for lib in ("libcublas.so.12", "libcublasLt.so.12", "libcudart.so.12"):
        if "/home/admin/wxw403883/anaconda3/envs/tao/lib/" + lib not in linked:
            raise ValueError("Wrong linked library: " + lib)
    if "not found" in linked:
        raise ValueError("Unresolved runtime library")
    with (directory / (args.label + "-ldd.txt")).open("x") as stream:
        stream.write(linked)
    custody.run(directory, args.label, True, True)


if __name__ == "__main__":
    main()
