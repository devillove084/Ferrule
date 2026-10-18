"""Small synthetic sources only: no manuscript compilation or experiment execution."""

import contextlib
import importlib.metadata
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock
import zipfile

SCRIPT = Path(__file__).resolve().parents[1] / "arxiv.py"
SPEC = importlib.util.spec_from_file_location("submission_arxiv", SCRIPT)
arxiv = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(arxiv)


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ferrule-submission-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.policy = {
            "entrypoint": "main.tex", "figure_macros": ["paperplot"],
            "files": ["main.tex", "sections/body.tex", "references.bib", "figures/chart.pdf"],
            "licenses": [],
        }
        self.write("main.tex", r"""% Private draft note: preserved only with --no-clean.
\documentclass{article}
\usepackage{graphicx}
\newcommand{\paperplot}[1]{\includegraphics{#1}}
\begin{document}
\input{sections/body}
\bibliographystyle{plain}
\bibliography{references}
\end{document}
""")
        self.write("sections/body.tex", r"A citation \cite{example}.\paperplot{figures/chart.pdf}")
        self.write("references.bib", "@article{example, author={Author}, title={Title}, year={2026}}\n")
        self.write("figures/chart.pdf", b"%PDF-1.4\nsynthetic figure, not a real rendered PDF\n")
        self.policy_file = self.root / "policy.json"
        self.save_policy()

    def write(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data.encode() if isinstance(data, str) else data)
        return path

    def save_policy(self):
        self.policy_file.write_text(json.dumps(self.policy))

    def collect(self):
        return arxiv.collect(self.root, self.policy)

    def cli(self, *args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as stderr:
            result = arxiv.main([*args, "--source-root", str(self.root)])
        self.last_error = stderr.getvalue()
        return result

    def prepare(self, output="build/submission", *, clean=False):
        self.save_policy()
        flags = [] if clean else ["--no-clean"]
        code = self.cli("prepare", "--policy", str(self.policy_file), "--output", output, *flags)
        self.assertEqual(code, 0, self.last_error)
        return self.root / output

    def test_success_minimal_fixture_and_no_source_changes(self):
        before = {name: (self.root / name).read_bytes() for name in self.policy["files"]}
        output = self.prepare()
        manifest = json.loads((output / "manifest.json").read_bytes())
        self.assertEqual(manifest["status"], "prepared-only-not-validated-not-submitted")
        self.assertIsNone(manifest["cleaner"])
        with zipfile.ZipFile(output / "source.zip") as archive:
            self.assertEqual(set(archive.namelist()), set(before))
            for name, data in before.items():
                self.assertEqual(archive.read(name), data)
                self.assertEqual((self.root / name).read_bytes(), data)
        unpacked = self.root / "unpacked"
        unpacked.mkdir()
        arxiv.unpack(output, unpacked)
        self.assertEqual((unpacked / "references.bib").read_bytes(), before["references.bib"])
        self.assertFalse((output / "validation.json").exists())

    def test_excludes_sensitive_large_data_previews_and_old_bbl(self):
        for name in ["experiments/raw.json", "weights/model.safetensors", "vendor/private.tex",
                     ".git/config", ".env", "build/main.pdf", "build/main.bbl", "build/main.log",
                     "figures/chart.png", "figures/paper-preview.tex", "reference-candidates.bib",
                     "sections/unused.tex"]:
            self.write(name, "MUST NOT SHIP")
        with (self.root / "weights/model.safetensors").open("r+b") as handle:
            handle.truncate(arxiv.MAX_TOTAL * 3)
        output = self.prepare()
        with tarfile.open(output / "source.tar.gz") as archive:
            self.assertEqual(set(archive.getnames()), set(self.policy["files"]))
            self.assertTrue(all(member.isfile() for member in archive))

    def test_archives_and_manifest_are_stable_across_mtime_and_runs(self):
        first = self.prepare()
        for name in self.policy["files"]:
            import os
            os.utime(self.root / name, (1700000000, 1700000000))
        second = self.prepare("build/submission/repeat")
        for name in ("source.tar.gz", "source.zip", "manifest.json"):
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

    def test_output_never_overwrites_or_writes_outside_build_submission(self):
        output = self.prepare()
        sentinel = output / "user-note.txt"
        sentinel.write_text("keep")
        self.assertEqual(self.cli("prepare", "--policy", str(self.policy_file), "--no-clean"), 1)
        self.assertIn("Refusing to overwrite", self.last_error)
        self.assertEqual(sentinel.read_text(), "keep")
        for name in (".", "build", "build/elsewhere", "build/submission/../escape", str(self.root.parent / "escape")):
            with self.subTest(name=name), self.assertRaises(arxiv.SubmissionError):
                arxiv.output_path(self.root, name)
        (output / "link").symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(arxiv.SubmissionError, "symlink"):
            arxiv.output_path(self.root, "build/submission/link/new")

    def test_unsafe_dependency_paths_and_non_source_files_are_rejected(self):
        for name in (".", "../secret.tex", "/etc/passwd", "./main.tex", "a//b.tex", ".env", "a/../main.tex",
                     "experiments/private.tex", "weights/model.tex", "figures/paper-preview.png", "main.pdf", "auth.cookies"):
            with self.subTest(name=name):
                policy = {**self.policy, "files": [*self.policy["files"], name]}
                with self.assertRaises(arxiv.SubmissionError):
                    arxiv.collect(self.root, policy)

    def test_file_and_parent_symlinks_are_rejected(self):
        (self.root / "figures/chart.pdf").unlink()
        (self.root / "figures/chart.pdf").symlink_to(self.root / "main.tex")
        with self.assertRaisesRegex(arxiv.SubmissionError, "Symlink"):
            self.collect()
        (self.root / "figures/chart.pdf").unlink()
        (self.root / "figures").rmdir()
        (self.root / "figures").symlink_to(self.root / "sections", target_is_directory=True)
        self.write("sections/chart.pdf", b"synthetic")
        with self.assertRaisesRegex(arxiv.SubmissionError, "Symlink"):
            self.collect()

    def test_missing_tex_and_bib_and_figure_errors(self):
        for name in ("sections/body.tex", "references.bib", "figures/chart.pdf"):
            path = self.root / name
            data = path.read_bytes()
            path.unlink()
            with self.subTest(name=name), self.assertRaisesRegex(arxiv.SubmissionError, "Missing"):
                self.collect()
            path.write_bytes(data)

    def test_new_figure_not_allowlisted_and_unused_allowlist_fail(self):
        self.write("sections/body.tex", r"\includegraphics{figures/unreviewed.pdf}")
        self.write("figures/unreviewed.pdf", b"synthetic")
        with self.assertRaisesRegex(arxiv.SubmissionError, "unallowlisted figure"):
            self.collect()
        self.write("sections/body.tex", r"No figure.")
        with self.assertRaisesRegex(arxiv.SubmissionError, "Unreferenced"):
            self.collect()

    def test_dynamic_inputs_and_graphicspath_are_rejected(self):
        for text in (r"\input sections/body", r"\input{../secret}", r"\input{\generated}",
                     r"\graphicspath{{private/}}", r"\includesvg{foo}"):
            self.write("sections/body.tex", text)
            with self.subTest(text=text), self.assertRaises(arxiv.SubmissionError):
                self.collect()

    def test_local_style_closure_and_reviewed_license_preserved(self):
        main = (self.root / "main.tex").read_text().replace(r"\usepackage{graphicx}", r"\usepackage{graphicx,local}")
        self.write("main.tex", main)
        self.write("local.sty", r"\ProvidesPackage{local}\input{macros.tex}")
        self.write("macros.tex", "% Copyright Example; do not remove\n")
        self.write("LICENSE-local", "Existing license, not chosen by this tool.")
        self.policy["files"] += ["local.sty", "macros.tex"]
        self.policy["licenses"] = ["LICENSE-local"]
        files, system = self.collect()
        self.assertIn("macros.tex", files)
        self.assertEqual(files["LICENSE-local"], b"Existing license, not chosen by this tool.")
        self.assertIn("graphicx.sty", system)
        self.assertNotIn("local.sty", system)

    def test_missing_cleaner_and_wrong_version_are_actionable(self):
        files, _ = self.collect()
        with mock.patch.object(arxiv.importlib.metadata, "version", side_effect=importlib.metadata.PackageNotFoundError):
            with self.assertRaisesRegex(arxiv.SubmissionError, "--no-clean"):
                arxiv.clean(files, self.policy, 2)
        with mock.patch.object(arxiv.importlib.metadata, "version", return_value="0.0"):
            with self.assertRaisesRegex(arxiv.SubmissionError, "Expected"):
                arxiv.clean(files, self.policy, 2)

    def test_cleaner_only_receives_staged_allowlist_and_no_resize_options(self):
        original, _ = self.collect()

        def fake_clean(command, cwd, timeout):
            self.assertEqual(command[:3], [sys.executable, "-m", "arxiv_latex_cleaner"])
            self.assertEqual(command[4:], ["--keep_bib"])
            stage = Path(command[3])
            self.assertNotEqual(stage, self.root)
            self.assertEqual({p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file()}, set(original))
            destination = Path(str(stage) + "_arXiv")
            shutil.copytree(stage, destination)
            (destination / "main.tex").write_text(arxiv.uncomment((stage / "main.tex").read_text()))
            (destination / "leak.txt").write_text("must not ship")

        with mock.patch.object(arxiv.importlib.metadata, "version", return_value=arxiv.CLEANER_VERSION), mock.patch.object(arxiv, "run", side_effect=fake_clean):
            files = arxiv.clean(original, self.policy, 2)
        self.assertEqual(set(files), set(original))
        self.assertNotIn(b"Private draft note", files["main.tex"])
        self.assertEqual(files["figures/chart.pdf"], original["figures/chart.pdf"])
        self.assertEqual((self.root / "main.tex").read_bytes(), original["main.tex"])

    def test_license_notice_prevents_destructive_cleaning(self):
        self.write("sections/body.tex", "% Copyright Example\n" + (self.root / "sections/body.tex").read_text())
        files, _ = self.collect()

        def fake_clean(command, cwd, timeout):
            shutil.copytree(command[3], command[3] + "_arXiv")

        with mock.patch.object(arxiv.importlib.metadata, "version", return_value=arxiv.CLEANER_VERSION), mock.patch.object(arxiv, "run", side_effect=fake_clean):
            with self.assertRaisesRegex(arxiv.SubmissionError, "License notice"):
                arxiv.clean(files, self.policy, 2)

    def test_size_limit(self):
        with mock.patch.object(arxiv, "MAX_FILE", 5):
            with self.assertRaisesRegex(arxiv.SubmissionError, "per-file limit"):
                self.collect()

    def test_tampered_archive_is_rejected(self):
        output = self.prepare()
        for name in ("source.tar.gz", "source.zip"):
            path = output / name
            original = path.read_bytes()
            path.write_bytes(original + b"tamper")
            with self.subTest(name=name), self.assertRaisesRegex(arxiv.SubmissionError, "checksum"):
                arxiv.unpack(output, self.root / "unpacked")
            path.write_bytes(original)

    def test_malformed_policy_and_manifest_have_readable_errors(self):
        self.policy_file.write_text("[]")
        self.assertEqual(self.cli("prepare", "--policy", str(self.policy_file), "--no-clean"), 1)
        self.assertIn("JSON object", self.last_error)
        self.assertFalse((self.root / "build/submission").exists())
        output = self.prepare()
        (output / "manifest.json").write_text("[]")
        with self.assertRaisesRegex(arxiv.SubmissionError, "schema"):
            arxiv.unpack(output, self.root / "unpacked")

    def test_archive_traversal_and_symlinks_are_rejected(self):
        output = self.prepare()
        manifest = json.loads((output / "manifest.json").read_bytes())
        for name, symlink in (("../escape.tex", False), ("main.tex", True)):
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
                member = tarfile.TarInfo(name)
                if symlink:
                    member.type, member.linkname = tarfile.SYMTYPE, "../outside"
                archive.addfile(member)
            blob = buffer.getvalue()
            (output / "source.tar.gz").write_bytes(blob)
            manifest["archives"]["source.tar.gz"] = arxiv.digest(blob)
            (output / "manifest.json").write_text(json.dumps(manifest))
            with self.subTest(name=name), self.assertRaises(arxiv.SubmissionError):
                arxiv.unpack(output, self.root / "unpacked")
        self.assertFalse((self.root / "escape.tex").exists())

    def test_subprocess_timeout_and_missing_executable(self):
        with self.assertRaisesRegex(arxiv.SubmissionError, "Timed out"):
            arxiv.run([sys.executable, "-c", "import time; time.sleep(2)"], self.root, 1)
        with self.assertRaisesRegex(arxiv.SubmissionError, "Missing executable"):
            arxiv.run([str(self.root / "not-an-executable")], self.root, 1)

    def test_validation_commands_use_fresh_extraction_and_no_shell_escape(self):
        output = self.prepare()
        (self.root / "build/main.bbl").write_text("STALE BIBLIOGRAPHY MUST NOT BE USED")
        commands = []

        def fake_run(command, cwd, timeout, env):
            if "--version" in command:
                return b"synthetic TeX tool 1.0\n"
            commands.append(command)
            self.assertNotEqual(cwd, self.root)
            self.assertNotIn("TEXMFOUTPUT", env)
            self.assertEqual(env["shell_escape"], "f")
            self.assertEqual(env["openin_any"], "p")
            for key in ("TEXINPUTS", "BIBINPUTS", "BSTINPUTS"):
                self.assertEqual(env[key], "." + arxiv.os.pathsep)
            self.assertNotEqual(env["HOME"], str(Path.home()))
            self.assertTrue((cwd / "references.bib").is_file())
            if len(commands) == 1:
                self.assertFalse((cwd / "main.bbl").exists())
            if command[0].endswith("bibtex"):
                (cwd / "main.bbl").write_text(r"\bibitem{example} Fresh synthetic bbl")
            else:
                self.assertIn("-no-shell-escape", command)
                (cwd / "main.log").write_text("Output written on main.pdf (1 page, 30 bytes).\n")
                (cwd / "main.pdf").write_bytes(b"%PDF-synthetic validation result")
            return b""

        with mock.patch.object(arxiv.shutil, "which", side_effect=lambda name: "/tools/" + name), mock.patch.object(arxiv, "run", side_effect=fake_run):
            self.assertEqual(self.cli("validate"), 0, self.last_error)
        self.assertEqual([Path(command[0]).name for command in commands], ["pdflatex", "bibtex", "pdflatex", "pdflatex"])
        self.assertTrue((output / "validation.json").exists())
        report = json.loads((output / "validation.json").read_bytes())
        self.assertEqual(report["page_count"], 1)
        self.assertEqual(report["bibliography_items"], 1)
        self.assertEqual(arxiv.digest((output / "validation-final.log").read_bytes()), report["final_log_sha256"])
        self.assertEqual(arxiv.digest((output / "validated-main.bbl").read_bytes()), report["bbl_sha256"])
        self.assertEqual((self.root / "build/main.bbl").read_text(), "STALE BIBLIOGRAPHY MUST NOT BE USED")
        self.assertEqual(self.cli("validate"), 1)
        self.assertIn("overwrite", self.last_error)

    def test_final_log_rejects_all_warning_and_box_diagnostics(self):
        clean = "Output written on main.pdf (41 pages, 900000 bytes).\n"
        self.assertEqual(arxiv.check_final_log(clean, "BibTeX clean"), 41)
        self.assertEqual(arxiv.check_final_log(clean + "Package: infwarerr 2019/12/03 Providing info/warning/error messages (HO)\n", "warning$ -- 0\n"), 41)
        for problem in ("LaTeX Warning: Citation `missing' undefined.", "Package rerunfilecheck Warning: Rerun required.",
                        "pdfTeX warning (ext4): destination with the same identifier has been already used",
                        r"Overfull \hbox (1.0pt too wide)", r"Underfull \vbox (badness 10000)",
                        "There were undefined references.", "Label(s) may have changed."):
            with self.subTest(problem=problem), self.assertRaisesRegex(arxiv.SubmissionError, "Final compilation"):
                arxiv.check_final_log(clean + problem, "")
        with self.assertRaisesRegex(arxiv.SubmissionError, "Final compilation"):
            arxiv.check_final_log(clean, "Warning--empty year in example")
        with self.assertRaisesRegex(arxiv.SubmissionError, "page count"):
            arxiv.check_final_log("Incomplete log", "")

    def test_validation_missing_tools(self):
        self.prepare()
        with mock.patch.object(arxiv.shutil, "which", return_value=None):
            self.assertEqual(self.cli("validate"), 1)
        self.assertIn("Missing pdflatex", self.last_error)

    def test_cli_help_and_no_clean_subprocess(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 0)
        self.assertIn(b"prepare", result.stdout)
        self.assertIn(b"validate", result.stdout)
        result = subprocess.run([sys.executable, str(SCRIPT), "prepare", "--source-root", str(self.root),
                                 "--policy", str(self.policy_file), "--no-clean"], capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertIn(b"NOT validated, uploaded, or submitted", result.stdout)


if __name__ == "__main__":
    unittest.main()
