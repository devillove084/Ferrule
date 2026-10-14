"""CPU-only audit of frozen BF16 call-contract outputs; never synthesize an oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import contract_run

CONFIGS = (
    "gemmex-matched", "gemmex-no-workspace", "gemmex-default-math",
    "gemmex-atomics-allowed", "tst-matched", "tst-prior",
    "tss-matched-diagnostic", "gemmex-f32-diagnostic",
)
LABELS = ("hf-plain", "hf-logged", "native-129", "native-124")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def differences(actual, expected):
    require(len(actual) == len(expected) and len(actual) % 2 == 0,
            "BF16 output extent differs")
    result = []
    for index, (a, e) in enumerate(zip(struct.iter_unpack("<H", actual),
                                      struct.iter_unpack("<H", expected))):
        if a != e:
            af = struct.unpack("<f", struct.pack("<I", a[0] << 16))[0]
            ef = struct.unpack("<f", struct.pack("<I", e[0] << 16))[0]
            result.append(dict(index=index, actual=f"{a[0]:04x}",
                               hf=f"{e[0]:04x}", abs_diff=abs(af - ef)))
    return result


def validate_receipt(record, prepared_pin):
    require(record["passed"] and record["custody_verified"] and
            record["pins_unchanged"], "Failed execution receipt")
    require(record["prepared_sha256"] == prepared_pin, "Prepared pin differs")
    result = record["supervisor"]
    require(result["returncode"] == 0 and not result.get("timed_out") and
            not result.get("error") and not result.get("cleanup_errors") and
            not result.get("stdout_truncated"), "Execution incomplete")
    h, c = result["handshake"], result["custody"]
    require(h["containment_verified"] and c["containment_verified"] and
            c["status"] == "proven_drained" and c["echild"] and
            h["launch_id"] == c["launch_id"] and
            h["helper_identity"] == c["helper_identity"], "Custody differs")
    require(record["gpu_before"]["idle"] and len(record["gpu_after"]) == 3 and
            all(x["idle"] for x in record["gpu_after"]), "Idle checks failed")


def audit(directory):
    directory = Path(directory)
    prepared = json.loads((directory / "PREPARED.json").read_text())
    contract_run.verify(prepared)
    pins = {}

    def payload(path):
        data = path.read_bytes()
        pins[str(path)] = hashlib.sha256(data).hexdigest()
        return data

    prepared_pin = hashlib.sha256(payload(directory / "PREPARED.json")).hexdigest()
    receipts = {}
    for label in LABELS:
        receipt = json.loads(payload(directory / (label + "-supervisor.json")))
        validate_receipt(receipt, prepared_pin)
        require(receipt["label"] == label, "Receipt label differs")
        receipts[label] = receipt
    calls = [json.loads(l) for l in payload(directory / "hf-calls.jsonl").splitlines()]
    observed = contract_run.contract(calls, prepared["cases"])
    require(observed["calls"] == 8 * 17 and len(calls) == 8 * 18,
            "Observed call count differs")
    cases = prepared["cases"]
    require(len(cases) == 8 and len({c["id"] for c in cases}) == 8,
            "Expected eight distinct fixtures")
    hf = {}
    for label in LABELS[:2]:
        report = json.loads(payload(directory / label / "report.json"))
        require(report["exact"] and report["stable"], "HF gate failed")
        require([c["case"] for c in report["cases"]] == [c["id"] for c in cases],
                "HF cases differ")
        for case, logged in zip(cases, report["cases"]):
            expected = payload(directory / "fixture" / (case["id"] + ".hf.bf16"))
            require(len(expected) == case["n"] * 2, "Oracle extent differs")
            for repeat in range(16):
                path = directory / label / f"{case['id']}-{repeat}.bf16"
                require(payload(path) == expected, "HF BF16 dump differs")
                require(pins[str(path)] == logged["output_sha256"][repeat],
                        "HF output hash differs")
        hf[label] = report
    matrix = {}
    binary_equal_between_versions = True
    for label, version in (("native-129", 120901), ("native-124", 120405)):
        stdout = payload(directory / (label + ".stdout")).decode()
        require(stdout == receipts[label]["supervisor"]["stdout"], "Stdout differs")
        rows = [json.loads(l) for l in stdout.splitlines()]
        require(rows[0]["cublas"] == version and rows[0]["sm"] == 86,
                "Wrong native runtime")
        rows = rows[1:]
        require(len(rows) == len(cases) * len(CONFIGS), "Native matrix incomplete")
        indexed = {(r["case"], r["config"]): r for r in rows}
        require(len(indexed) == len(rows), "Duplicate native result")
        matrix[label] = {}
        for case in cases:
            name = case["id"]
            expected = payload(directory / "fixture" / (name + ".hf.bf16"))
            matrix[label][name] = {}
            for config in CONFIGS:
                record = indexed[name, config]
                require((record["n"], record["k"], record["repeats"]) ==
                        (case["n"], case["k"], 16), "Native geometry differs")
                require(all(record[key] == 0 for key in
                            ("a_mod256", "b_mod256", "c_mod256")), "Alignment differs")
                counts, outputs, first_diff = [], [], None
                for repeat in range(16):
                    filename = f"{name}-{config}-{repeat}.bf16"
                    data = payload(directory / label / filename)
                    diff = differences(data, expected)
                    counts.append(len(diff))
                    outputs.append(data)
                    if repeat == 0:
                        first_diff = diff
                    if label == "native-124":
                        binary_equal_between_versions &= (
                            data == payload(directory / "native-129" / filename))
                drift = sum(data != outputs[0] for data in outputs)
                require(max(counts) == record["max_unequal_hf"] and
                        drift == record["runs_different_from_first"],
                        "Native summary differs from binary dumps")
                matrix[label][name][config] = dict(
                    max_unequal_hf=max(counts), repeat_drift=drift,
                    first_output_diff=first_diff,
                    diagnostic_only=config.endswith("diagnostic"))
    matched_exact = all(matrix[label][c["id"]]["gemmex-matched"]["max_unequal_hf"] == 0
                        for label in matrix for c in cases)
    stable = all(value["repeat_drift"] == 0 for by_case in matrix.values()
                 for by_config in by_case.values() for value in by_config.values())
    return dict(status="AUDITED", matched_bf16_gemmex_exact=matched_exact,
                native_repetitions_stable=stable,
                binary_equal_between_versions=binary_equal_between_versions,
                original_bf16_oracle_only=True, case_order=[c["id"] for c in cases],
                config_order=CONFIGS, matrix=matrix, observed_contract=observed,
                words_per_matched_pass=sum(c["n"] for c in cases), repeats=16,
                hf_kernels=sorted({k for c in hf["hf-plain"]["cases"] for k in c["kernels"]}),
                evidence_pins=pins)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "Fresh audit output required")
    result = audit(args.directory)
    result["audit_source_sha256"] = contract_run.sha(__file__)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("matrix", "evidence_pins", "hf_kernels")}, indent=2))
