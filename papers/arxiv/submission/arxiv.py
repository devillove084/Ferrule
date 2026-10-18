#!/usr/bin/env python3
"""Prepare reviewed arXiv sources; never authenticate, upload, or submit."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

CLEANER_VERSION = "1.0.11"
PAPER = Path(__file__).resolve().parents[1]
POLICY = Path(__file__).with_name("source-files.json")
MAX_FILE = 20 * 1024 * 1024
MAX_TOTAL = 50 * 1024 * 1024
TEXT_SUFFIXES = {".tex", ".sty", ".cls", ".bst", ".def", ".cfg", ".clo"}
FIGURE_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg"}
FORBIDDEN = {
    "build", "experiments", "experiment", "weights", "models", "vendor",
    "target", "data", "submission", "node_modules", "__pycache__",
}
AUTHOR_CHECKS = [
    "Confirm author spelling/order, affiliations, email, title and abstract with all authors.",
    "Choose the category/cross-list and obtain endorsement if required.",
    "Confirm rights for every source/figure and explicitly choose the arXiv license in the browser.",
    "Review the exact archive manifest, local validation PDF, and arXiv-generated PDF.",
    "Complete author declarations and personally confirm the final Submit Article action in the browser.",
]


class SubmissionError(Exception):
    """An actionable preparation or validation failure."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def safe_name(name: str) -> str:
    if not isinstance(name, str) or not name or not re.fullmatch(r"[A-Za-z0-9_./-]+", name):
        raise SubmissionError(f"Unsafe relative path: {name!r}")
    path = PurePosixPath(name)
    if not path.parts or path.is_absolute() or path.as_posix() != name or any(
        part in {".", ".."} or part.startswith(".") for part in path.parts
    ):
        raise SubmissionError(f"Unsafe relative path: {name!r}")
    if any(part.lower() in FORBIDDEN for part in path.parts):
        raise SubmissionError(f"Forbidden research/build path: {name}")
    if "preview" in path.stem.lower() or "candidate" in path.stem.lower():
        raise SubmissionError(f"Preview/candidate files are not submission inputs: {name}")
    return name


def local_path(root: Path, name: str) -> Path:
    safe_name(name)
    path = root
    for part in PurePosixPath(name).parts:
        path = path / part
        if path.is_symlink():
            raise SubmissionError(f"Symlinks are not permitted: {name}")
        if not path.exists():
            raise SubmissionError(f"Missing dependency: {name}")
        if part not in os.listdir(path.parent):
            raise SubmissionError(f"Dependency name has incorrect case: {name}")
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise SubmissionError(f"Dependency must be a regular file inside the paper: {name}")
    if path.stat().st_size > MAX_FILE:
        raise SubmissionError(f"Dependency exceeds the {MAX_FILE // 1024 // 1024} MiB per-file limit: {name}")
    return path


def output_path(root: Path, requested: str | None, *, must_exist=False) -> Path:
    base = root / "build" / "submission"
    path = Path(requested) if requested else base
    if not path.is_absolute():
        path = root / path
    if ".." in path.parts or not path.is_relative_to(base):
        raise SubmissionError("Output must be build/submission or a new directory below it.")
    current = root
    for part in path.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            raise SubmissionError(f"Output path contains a symlink: {current}")
    if must_exist:
        if not path.is_dir():
            raise SubmissionError(f"Prepared output directory does not exist: {path}")
    elif path.exists():
        raise SubmissionError(f"Refusing to overwrite {path}; choose --output build/submission/NEW-RUN.")
    return path


def uncomment(text: str) -> str:
    # TeX comments start at an unescaped percent, including after an even run of backslashes.
    return re.sub(r"(?<!\\)((?:\\\\)*)%[^\n]*", r"\1", text)


def collect(root: Path, policy: dict) -> tuple[dict[str, bytes], list[str]]:
    if not isinstance(policy, dict):
        raise SubmissionError("Source policy must be a JSON object.")
    names = policy.get("files", [])
    licenses = policy.get("licenses", [])
    if not isinstance(names, list) or not isinstance(licenses, list) or not names:
        raise SubmissionError("Policy requires nonempty files and a licenses list.")
    for name in names + licenses:
        safe_name(name)
    if len(set(names + licenses)) != len(names + licenses):
        raise SubmissionError("Policy contains duplicate file names.")
    allowed = set(names)
    entry = safe_name(policy.get("entrypoint", ""))
    if entry not in allowed or not entry.endswith(".tex") or "/" in entry:
        raise SubmissionError("The entrypoint must be an allowlisted root-level .tex file.")
    for name in names:
        suffix = PurePosixPath(name).suffix
        if suffix not in TEXT_SUFFIXES | FIGURE_SUFFIXES | {".bib"}:
            raise SubmissionError(f"Unsupported source type: {name}")
        if suffix in FIGURE_SUFFIXES and not name.startswith("figures/"):
            raise SubmissionError(f"Only referenced figures under figures/ may be packaged: {name}")
    for name in licenses:
        if not re.fullmatch(r"(?:licenses/)?(?:LICENSE|COPYING|NOTICE)(?:[A-Za-z0-9_.-]*)", name):
            raise SubmissionError(f"Use an explicitly reviewed LICENSE/COPYING/NOTICE file: {name}")
    macros = policy.get("figure_macros", [])
    if not isinstance(macros, list) or any(not re.fullmatch(r"[A-Za-z]+", m) for m in macros):
        raise SubmissionError("figure_macros must be a list of literal TeX command names.")
    commands = ["input", "include", "includegraphics", "bibliography", "bibliographystyle",
                "usepackage", "RequirePackage", "documentclass", "LoadClass", *macros]
    pattern = re.compile(r"\\(" + "|".join(commands) + r")\*?\s*(?:\[[^\]]*\]\s*)?\{([^{}]*)\}")
    files: dict[str, bytes] = {}
    system = set()

    def add(name: str):
        if name in files:
            return
        safe_name(name)
        if name not in allowed and name not in licenses:
            raise SubmissionError(f"Dependency is not in the reviewed allowlist: {name}")
        data = local_path(root, name).read_bytes()
        files[name] = data
        if sum(map(len, files.values())) > MAX_TOTAL:
            raise SubmissionError("Source package exceeds the 50 MiB safety limit.")
        if PurePosixPath(name).suffix not in TEXT_SUFFIXES:
            return
        try:
            text = uncomment(data.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise SubmissionError(f"Source must be UTF-8: {name}") from exc
        if re.search(r"\\(?:graphicspath|addbibresource|import|subimport|subfile|includesvg|includepdf|openin|openout|write18)\b", text):
            raise SubmissionError(f"Unsupported dynamic/external dependency in {name}; review the source and policy first.")
        matches = list(pattern.finditer(text))
        literal_inputs = sum(m.group(1) in {"input", "include"} for m in matches)
        if len(re.findall(r"\\(?:input|include)\b", text)) != literal_inputs:
            raise SubmissionError(f"Use literal braced input/include paths in {name}.")
        for match in matches:
            command, value = match.groups()
            value = value.strip()
            if command == "includegraphics" and value == "#1" and macros:
                # This paper's reviewed paperplot wrapper passes its first literal argument here.
                continue
            values = value.split(",") if command in {"bibliography", "usepackage", "RequirePackage"} else [value]
            for value in values:
                value = value.strip()
                safe_name(value)
                if command in {"includegraphics", *macros}:
                    choices = [value] if PurePosixPath(value).suffix else [value + ext for ext in (".pdf", ".png", ".jpg", ".jpeg")]
                    selected = next((p for p in choices if p in allowed and (root / p).exists()), None)
                    if selected is None:
                        raise SubmissionError(f"Missing or unallowlisted figure in {name}: {value}")
                    if PurePosixPath(selected).suffix not in FIGURE_SUFFIXES:
                        raise SubmissionError(f"Unsupported pdflatex figure: {selected}")
                    add(selected)
                elif command in {"input", "include", "bibliography"}:
                    suffix = ".bib" if command == "bibliography" else ".tex"
                    add(value if PurePosixPath(value).suffix else value + suffix)
                else:
                    suffix = {"bibliographystyle": ".bst", "documentclass": ".cls", "LoadClass": ".cls"}.get(command, ".sty")
                    dependency = value if value.endswith(suffix) else value + suffix
                    if dependency in allowed or (root / dependency).exists():
                        add(dependency)
                    else:
                        if "/" in dependency:
                            raise SubmissionError(f"Missing local style dependency: {dependency}")
                        system.add(dependency)

    add(entry)
    for name in licenses:
        add(name)
    unused = allowed - files.keys()
    if unused:
        raise SubmissionError("Unreferenced allowlist entries; review/remove them: " + ", ".join(sorted(unused)))
    return dict(sorted(files.items())), sorted(system)


def run(command: list[str], cwd: Path, timeout: int, env=None):
    try:
        result = subprocess.run(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise SubmissionError(f"Missing executable: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise SubmissionError(f"Timed out after {timeout}s: {' '.join(command)}") from exc
    if result.returncode:
        detail = result.stdout.decode(errors="replace").splitlines()[-30:]
        raise SubmissionError(f"Command failed ({result.returncode}): {' '.join(command)}\n" + "\n".join(detail))
    return result.stdout


def clean(files: dict[str, bytes], policy: dict, timeout: int) -> dict[str, bytes]:
    try:
        version = importlib.metadata.version("arxiv-latex-cleaner")
    except importlib.metadata.PackageNotFoundError as exc:
        raise SubmissionError("Missing arxiv-latex-cleaner==1.0.11. Install submission/requirements.txt in the isolated venv, or explicitly use --no-clean (sources/comments unchanged).") from exc
    if version != CLEANER_VERSION:
        raise SubmissionError(f"Expected arxiv-latex-cleaner=={CLEANER_VERSION}, found {version}.")
    with tempfile.TemporaryDirectory(prefix="ferrule-arxiv-clean-") as temporary:
        stage = Path(temporary) / "source"
        stage.mkdir()
        for name, data in files.items():
            path = stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        run([sys.executable, "-m", "arxiv_latex_cleaner", str(stage), "--keep_bib"], Path(temporary), timeout)
        cleaned = Path(str(stage) + "_arXiv")
        # Reconstitute only our reviewed file set, not the cleaner's directory traversal results.
        result = {}
        for name, original in files.items():
            if PurePosixPath(name).suffix == ".tex":
                result[name] = local_path(cleaned, name).read_bytes()
                notices = re.findall(r"(?im)^%[^\n]*(?:copyright|SPDX-License-Identifier|licensed under)[^\n]*", original.decode())
                if notices:
                    raise SubmissionError(f"License notice in {name}: use --no-clean to preserve it verbatim.")
            else:
                result[name] = original
        for name, data in result.items():
            (stage / name).write_bytes(data)
        verified, _ = collect(stage, policy)
        return verified


def archives(files: dict[str, bytes]) -> tuple[bytes, bytes]:
    tar_buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=tar_buffer, mode="wb", filename="", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for name, data in sorted(files.items()):
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = len(data), 0o644, 0
                archive.addfile(info, io.BytesIO(data))
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system, info.external_attr, info.compress_type = 3, 0o100644 << 16, zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    return tar_buffer.getvalue(), zip_buffer.getvalue()


def prepare(args):
    root = Path(args.source_root).resolve(strict=True)
    output = output_path(root, args.output)
    policy_data = Path(args.policy).read_bytes()
    policy = json.loads(policy_data)
    original, system = collect(root, policy)
    files = original if args.no_clean else clean(original, policy, args.timeout)
    tgz, zipped = archives(files)
    manifest = {
        "schema": 1, "status": "prepared-only-not-validated-not-submitted",
        "entrypoint": policy["entrypoint"], "policy_sha256": digest(policy_data),
        "cleaner": None if args.no_clean else f"arxiv-latex-cleaner=={CLEANER_VERSION}",
        "bibliography": "source .bib; no cached .bbl; validation runs fresh BibTeX",
        "system_tex_dependencies": system,
        "files": [{"path": name, "bytes": len(data), "sha256": digest(data),
                   "original_sha256": digest(original[name])} for name, data in files.items()],
        "archives": {"source.tar.gz": digest(tgz), "source.zip": digest(zipped)},
        "author_checks": AUTHOR_CHECKS,
    }
    # Never clear or reuse an existing directory, including one created by a concurrent process.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    (output / "source.tar.gz").write_bytes(tgz)
    (output / "source.zip").write_bytes(zipped)
    (output / "manifest.json").write_bytes(json_bytes(manifest))
    print(f"Prepared {len(files)} source files in {output}; NOT validated, uploaded, or submitted.")
    print("Review manifest.json, run validate, then upload ONE archive in the arXiv website.")


def unpack(output: Path, destination: Path) -> dict:
    manifest_path = output / "manifest.json"
    if manifest_path.is_symlink():
        raise SubmissionError("Manifest must not be a symlink.")
    manifest = json.loads(manifest_path.read_bytes())
    if not isinstance(manifest, dict) or manifest.get("schema") != 1:
        raise SubmissionError("Unsupported manifest schema.")
    records = manifest["files"]
    if not isinstance(records, list) or not records or any(not isinstance(record, dict) for record in records):
        raise SubmissionError("Manifest files must be a nonempty list of file records.")
    expected = {safe_name(record["path"]): record for record in records}
    if len(expected) != len(records):
        raise SubmissionError("Duplicate manifest paths.")
    if sum(record["bytes"] for record in records) > MAX_TOTAL:
        raise SubmissionError("Manifest exceeds the source size limit.")
    entry = safe_name(manifest["entrypoint"])
    if "/" in entry or not entry.endswith(".tex") or entry not in expected:
        raise SubmissionError("Invalid manifest entrypoint.")
    for archive_name in ("source.tar.gz", "source.zip"):
        archive = output / archive_name
        if archive.is_symlink() or archive.stat().st_size > MAX_TOTAL + 1024 * 1024:
            raise SubmissionError(f"Archive is a symlink or exceeds the source size limit: {archive_name}")
        if digest(archive.read_bytes()) != manifest["archives"][archive_name]:
            raise SubmissionError(f"Archive checksum does not match manifest.json: {archive_name}")
    blob = (output / "source.tar.gz").read_bytes()
    seen = set()
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        for member in tf:
            name = safe_name(member.name)
            if name in seen or name not in expected or not member.isfile():
                raise SubmissionError(f"Unexpected, duplicate, or non-regular archive member: {name}")
            record = expected[name]
            if member.size != record["bytes"] or not 0 <= member.size <= MAX_FILE:
                raise SubmissionError(f"Archive member size mismatch: {name}")
            data = tf.extractfile(member).read()
            if digest(data) != record["sha256"]:
                raise SubmissionError(f"Archive member checksum mismatch: {name}")
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            seen.add(name)
    if seen != set(expected):
        raise SubmissionError("Archive is missing files listed in manifest.json.")
    return manifest


def check_final_log(log: str, bibtex_log: str) -> int:
    issue = re.compile(r"\bwarning\b(?=\s*(?::|--|\())|Overfull \\[hv]box|Underfull \\[hv]box|undefined references|undefined citations|Citation .* undefined|Reference .* undefined|Label\(s\) may have changed|There were multiply-defined labels", re.IGNORECASE)
    problems = [line for line in (log + "\n" + bibtex_log).splitlines() if issue.search(line)]
    if problems:
        raise SubmissionError("Final compilation/BibTeX log has warnings or unresolved references:\n" + "\n".join(problems[:10]))
    pages = re.search(r"Output written on .*?\((\d+) pages?\b", log, re.DOTALL)
    if not pages:
        raise SubmissionError("Final compilation log does not report the PDF page count.")
    return int(pages.group(1))


def validate(args):
    root = Path(args.source_root).resolve(strict=True)
    output = output_path(root, args.output, must_exist=True)
    report = output / "validation.json"
    pdf = output / "validated-main.pdf"
    evidence = [output / name for name in ("validation-final.log", "validation-bibtex.log", "validated-main.bbl")]
    if any(path.exists() or path.is_symlink() for path in [report, pdf, *evidence]):
        raise SubmissionError("Refusing to overwrite validation results; prepare a new output directory.")
    tools = {}
    for tool in ("pdflatex", "bibtex"):
        tools[tool] = shutil.which(tool)
        if not tools[tool]:
            raise SubmissionError(f"Missing {tool}; install a TeX distribution before validating.")
    with tempfile.TemporaryDirectory(prefix="ferrule-arxiv-validate-") as temporary:
        work = Path(temporary)
        source = work / "source"
        source.mkdir()
        manifest = unpack(output, source)
        home = work / "home"
        home.mkdir()
        # Direct tools, no latexmk hooks, user texmf tree, shell escape, or network/package installer.
        env = {key: os.environ[key] for key in ("PATH", "SYSTEMROOT") if key in os.environ}
        env.update({"HOME": str(home), "TEXMFHOME": str(home / "texmf"),
                    "TEXMFCONFIG": str(home / "config"), "TEXMFVAR": str(home / "var"),
                    # Absolute TEXINPUTS makes pdftex.def's local figure lookup fail under openin_any=p.
                    "TEXINPUTS": "." + os.pathsep, "BIBINPUTS": "." + os.pathsep,
                    "BSTINPUTS": "." + os.pathsep, "openin_any": "p", "openout_any": "p",
                    "shell_escape": "f", "SOURCE_DATE_EPOCH": "0", "FORCE_SOURCE_DATE": "1"})
        entry = manifest["entrypoint"]
        stem = Path(entry).stem
        latex = [tools["pdflatex"], "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", entry]
        run(latex, source, args.timeout, env)
        bibtex_log = b""
        if any(record["path"].endswith(".bib") for record in manifest["files"]):
            bibtex_log = run([tools["bibtex"], stem], source, args.timeout, env)
            if (source / (stem + ".blg")).exists():
                bibtex_log = (source / (stem + ".blg")).read_bytes()
        run(latex, source, args.timeout, env)
        run(latex, source, args.timeout, env)
        log = (source / (stem + ".log")).read_bytes()
        pages = check_final_log(log.decode(errors="replace"), bibtex_log.decode(errors="replace"))
        bbl_path = source / (stem + ".bbl")
        bbl = bbl_path.read_bytes() if bbl_path.exists() else b""
        bibliography_items = len(re.findall(rb"\\bibitem(?:\[[^\]]*\])?\s*\{", bbl))
        data = (source / (stem + ".pdf")).read_bytes()
        with pdf.open("xb") as handle:
            handle.write(data)
        result = {"status": "local-compile-passed-not-submitted", "archive_sha256": manifest["archives"]["source.tar.gz"],
                  "pdf_sha256": digest(data), "page_count": pages, "bibliography_items": bibliography_items,
                  "final_log_sha256": digest(log), "bibtex_log_sha256": digest(bibtex_log), "bbl_sha256": digest(bbl),
                  "validator_sha256": digest(Path(__file__).read_bytes()),
                  "shell_escape": False, "commands": ["pdflatex", "bibtex if .bib", "pdflatex", "pdflatex"],
                  "tools": {name: run([path, "--version"], source, args.timeout, env).decode(errors="replace").splitlines()[0]
                            for name, path in tools.items()}, "author_checks": AUTHOR_CHECKS}
        for path, content in zip(evidence, (log, bibtex_log, bbl)):
            with path.open("xb") as handle:
                handle.write(content)
        with report.open("xb") as handle:
            handle.write(json_bytes(result))
    print(f"Local archive compilation passed: {pdf}. Review it AND the arXiv-generated PDF; NOT submitted.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "validate"):
        sub = subparsers.add_parser(command, help="Clean/package sources only" if command == "prepare" else "Compile a prepared archive in a fresh temporary directory")
        sub.add_argument("--source-root", default=str(PAPER), help="Paper root (defaults to papers/arxiv)")
        sub.add_argument("--output", help="New output under the paper's build/submission (default: build/submission)")
        sub.add_argument("--timeout", type=int, default=180, help="Per-command timeout in seconds (default: 180)")
        if command == "prepare":
            sub.add_argument("--policy", default=str(POLICY), help="Reviewed file allowlist JSON")
            sub.add_argument("--no-clean", action="store_true", help="Explicit stdlib-only fallback: preserve source bytes, including ALL comments")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    try:
        (prepare if args.command == "prepare" else validate)(args)
    except (SubmissionError, OSError, ValueError, KeyError, TypeError, tarfile.TarError, zipfile.BadZipFile) as exc:
        print(f"arxiv: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
