"""Losslessly curate the sealed CPU/GPU evidence; never executes an artifact.

Run only against the original owner-local evidence. Existing archive entries must
match the original or the documented projection; no unrelated file is replaced.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

ORIGINAL = "target/validation/expert-read-current-quality-20261014-02"
SEAL_SHA = "0b6311f25951d4bb89724cf53caa57d47475dab6c46b70025d37f451c30b7ceb"
TOP = (
    "SUMMARY.md FINAL.json evidence-manifest.json receipt-audit.json "
    "qualification-plan-v2.json source-freeze.json archive-exact-members.json "
    "reference-pins.json model-shard-identities.json toolchain.json "
    "build-environment.json native-archive-pins.json runtime-library-pins.json "
    "compiled-dependency-pins.json external-input-pins.json uuid-locks-held.json "
    "final-idle.json final-lock-release.json final-lock-probe.json"
).split()
GATE_FILES = (
    "authenticated-launch.json authenticated-pre-cuda-idle.json custody.json "
    "exact-list-command.json exact-list-receipt.json exact-list.stdout exact-list.stderr "
    "gate-result.json pre-idle.json post-idle.json runtime-environment.json "
    "runtime-mapping-observations.json supervisor.json test.stdout test.stderr"
).split()
PROOFS = {
    "constructor.rs": "crates/ferrule-model/tests/expert_read_constructor.rs",
    "expert_reads.rs": "crates/ferrule-model/src/transformer/expert_reads.rs",
    "numeric_fp8.rs": "crates/ferrule-model/src/checkpoint/numeric_fp8.rs",
    "bf16_full40.rs": "crates/ferrule-composition/tests/unit/engine/model_factory/bf16_full40.rs",
    "bf16_fresh_quality.rs": "crates/ferrule-composition/tests/unit/engine/model_factory/bf16_fresh_quality/mod.rs",
    "expert_read_reporting.rs": "crates/ferrule-composition/tests/unit/engine/model_factory/expert_read_reporting.rs",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def curate(repo, output):
    repo = repo.resolve()
    base = repo / ORIGINAL
    require(output.resolve() == repo / "papers/arxiv/experiments/storage_demand/qualification_v1",
            "Output must be the exclusive qualification_v1 directory")
    archive = output / "receiptarchive"
    require(archive.is_dir(), "Create and verify receiptarchive first")
    seal_bytes = (base / "evidence-manifest.json").read_bytes()
    require(sha(seal_bytes) == SEAL_SHA, "Original evidence seal changed")
    seal = json.loads(seal_bytes)["pins_relative_to_directory"]
    require(len(seal) == 1772, "Original sealed-file count changed")
    # Hash all sealed inputs, not just the selected reports. Do not chmod them.
    for name, expected in seal.items():
        p = base / name
        require(p.is_file() and not p.is_symlink(), "Missing original: " + name)
        require(sha(p.read_bytes()) == expected, "Stale original: " + name)
        require(not p.stat().st_mode & 0o222, "Original is not read-only: " + name)
    require(not (base / "evidence-manifest.json").stat().st_mode & 0o222,
            "Original seal is not read-only")
    entries = []

    def add(source, name, projection=None):
        source = Path(source)
        require(source.is_file() and not source.is_symlink(), "Missing copy input: " + str(source))
        raw = source.read_bytes()
        original_rel = str(source.relative_to(repo))
        if source.is_relative_to(base):
            key = str(source.relative_to(base))
            require(sha(raw) == (SEAL_SHA if key == "evidence-manifest.json" else seal[key]),
                    "Copy input changed: " + key)
        content = raw if projection is None else json_bytes(projection)
        encoding = "gzip" if len(content) > 65536 else "identity"
        destination = archive / (name + (".gz" if encoding == "gzip" else ""))
        require(destination.parent.is_dir(), "Unverified copy parent: " + str(destination.parent))
        stored = gzip.compress(content, compresslevel=9, mtime=0) if encoding == "gzip" else content
        if destination.exists():
            require(destination.read_bytes() in (raw, stored), "Refuse to replace unrelated file: " + str(destination))
            if destination.read_bytes() != stored:
                destination.chmod(0o644)
                destination.write_bytes(stored)
        elif encoding == "identity" and projection is None:
            shutil.copyfile(source, destination)
        else:
            destination.write_bytes(stored)
        destination.chmod(0o644)
        uncompressed_copy = archive / name
        if encoding == "gzip" and uncompressed_copy.exists():
            require(uncompressed_copy.read_bytes() == raw, "Unexpected uncompressed copy")
            uncompressed_copy.unlink()
        entries.append(dict(path=str(destination.relative_to(output)), original_relpath=original_rel,
                            original_sha256=sha(raw), original_bytes=len(raw), encoding=encoding,
                            stored_sha256=sha(stored), stored_bytes=len(stored),
                            content_sha256=sha(content), content_bytes=len(content),
                            transform="identity" if projection is None else "remove_worker_token_only"))

    for name in TOP:
        add(base / name, name)
    for role in ("cli", "constructor", "composition"):
        add(base / ("artifact-" + role + ".json"), "artifact-" + role + ".json")
        for family in ("build", "elf", "ldd"):
            for suffix in ("-command.json", "-receipt.json", ".stdout", ".stderr"):
                name = family + "-" + role + suffix
                add(base / name, name)
    for family in ("cli-help", "cli-serve-help", "cli-version"):
        for suffix in ("-command.json", "-receipt.json", ".stdout", ".stderr"):
            name = family + suffix
            add(base / name, name)
    plan = json.loads((base / "qualification-plan-v2.json").read_bytes())
    for gate in plan["gates"]:
        prefix = "gates/" + gate["id"] + "/"
        for name in GATE_FILES:
            add(base / (prefix + name), prefix + name)
        name = "numeric-report/full40-bf16-profile.json" if gate["kind"] == "full40" else "quality.json"
        add(base / (prefix + name), prefix + name)
        worker = json.loads((base / (prefix + "worker-launch.json")).read_bytes())
        require("worker_token" in worker, "Worker projection no longer understood")
        del worker["worker_token"]
        add(base / (prefix + "worker-launch.json"), prefix + "worker-launch.public.json", worker)
        # Remove only the exact raw copies made during initial curation.
        for excluded in ("dispatch.json", "worker-launch.json"):
            path = archive / (prefix + excluded)
            if path.exists():
                require(path.read_bytes() == (base / (prefix + excluded)).read_bytes(), "Unexpected excluded copy")
                path.unlink()
    for name, member in PROOFS.items():
        add(base / "frozen-source" / member, "proof-source/" + name)
    refs = json.loads((base / "reference-pins.json").read_bytes())
    for path, expected in refs["reference_files"].items():
        if not path.endswith(".json"):
            continue
        source = Path(path)
        require(sha(source.read_bytes()) == expected, "Reference input changed")
        family = "full40" if "profile-reference" in source.parts else "fresh"
        add(source, "references/" + family + "-" + source.name)
    external = []
    for role, artifact in plan["artifacts"].items():
        external.append(dict(role=role, path=artifact["executable"], sha256=artifact["sha256"],
                             bytes=artifact["bytes"], archived=False))
    source = json.loads((base / "source-freeze.json").read_bytes())
    external.append(dict(role="committed_source_archive", path=source["archive"],
                         sha256=source["archive_sha256"], bytes=(base / "source.tar").stat().st_size, archived=False))
    for path, expected in refs["reference_files"].items():
        if not path.endswith(".json"):
            external.append(dict(role="reference_tensor_payload", path=path, sha256=expected, archived=False))
    manifest = dict(schema="ferrule.storage-demand.qualification-curation.v1", source_commit=plan["commit"],
                    source_digest=plan["source_digest"], source_archive_sha256=source["archive_sha256"],
                    original_root=ORIGINAL, original_evidence_manifest_sha256=SEAL_SHA,
                    original_sealed_files=1772, original_files_including_seal=1773,
                    originals_verified_readonly=True, candidate_sha256={r:a["sha256"] for r,a in plan["artifacts"].items()},
                    gates=[g["id"] for g in plan["gates"]], entries=sorted(entries, key=lambda e:e["path"]),
                    external_artifacts=external,
                    gate_checks=[{k:r[k] for k in ("gate_id", "status", "exact_test_passed", "constructor_cells",
                        "constructor_finite_logit_values", "main_report_metrics", "main_report_logit_values",
                        "actual_fingerprint_calls", "positive_worker_read_calls", "sampled_read_operations",
                        "report_sha256", "authenticated_echild", "recorded_pid_birth_owners_absent", "natural_exit")}
                        for r in json.loads((base / "receipt-audit.json").read_bytes())["gates"]],
                    omissions=["all executable/library/native archive/model/source archive/build object bytes",
                               "all but six numerical/counter proof source members",
                               "credential-bearing dispatch.json; worker_token removed from worker projections"],
                    qualification_scope="same-source CLI candidate quality, not same-binary CLI serving qualification",
                    physical_release="unobserved", cross_stream_sums=False)
    (output / "manifest.json").write_bytes(json_bytes(manifest))
    # Verify that nothing outside the explicit archive inventory survived.
    actual = {str(p.relative_to(output)) for p in archive.rglob("*") if p.is_file()}
    require(actual == {e["path"] for e in entries}, "Unexpected archive files")
    print("CURATED",len(entries),"records;",sum(e["stored_bytes"] for e in entries),"stored bytes; 1772 originals + seal verified read-only")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    curate(args.repo, Path(__file__).resolve().parent)
