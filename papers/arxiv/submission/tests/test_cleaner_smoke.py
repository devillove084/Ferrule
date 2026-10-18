"""Optional real-cleaner smoke test; synthetic sources only, never runs TeX."""

import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
import zipfile

import test_arxiv as fixtures

try:
    CLEANER_VERSION = importlib.metadata.version("arxiv-latex-cleaner")
except importlib.metadata.PackageNotFoundError:
    CLEANER_VERSION = None


@unittest.skipUnless(CLEANER_VERSION, "Install the pinned submission requirements for real-cleaner smoke testing")
class RealCleanerSmoke(unittest.TestCase):
    def test_real_cli_and_wrapper_on_minimal_fixture(self):
        self.assertEqual(CLEANER_VERSION, fixtures.arxiv.CLEANER_VERSION)
        fixture = fixtures.SubmissionTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        original = {name: (fixture.root / name).read_bytes() for name in fixture.policy["files"]}
        fixture.write(".env", "PRIVATE SENTINEL")
        fixture.write("experiments/private.json", "PRIVATE SENTINEL")

        stage = fixture.root / "direct-source"
        stage.mkdir()
        for name in original:
            destination = stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fixture.root / name, destination)
        command = [sys.executable, "-m", "arxiv_latex_cleaner", str(stage), "--keep_bib"]
        result = subprocess.run(command, cwd=fixture.root, stdin=subprocess.DEVNULL,
                                capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, (result.stdout + result.stderr).decode(errors="replace"))
        cleaned = Path(str(stage) + "_arXiv")
        self.assertTrue(cleaned.is_dir(), "Real cleaner must produce the wrapper's expected _arXiv directory")
        self.assertEqual({path.relative_to(cleaned).as_posix() for path in cleaned.rglob("*") if path.is_file()}, set(original))
        self.assertNotIn(b"Private draft note", (cleaned / "main.tex").read_bytes())
        for name in ("references.bib", "figures/chart.pdf"):
            self.assertEqual((cleaned / name).read_bytes(), original[name])

        first = fixture.prepare(clean=True)
        second = fixture.prepare("build/submission/repeat", clean=True)
        manifest = json.loads((first / "manifest.json").read_bytes())
        self.assertEqual(manifest["cleaner"], "arxiv-latex-cleaner==" + CLEANER_VERSION)
        self.assertEqual({record["path"] for record in manifest["files"]}, set(original))
        with zipfile.ZipFile(first / "source.zip") as archive:
            self.assertEqual(set(archive.namelist()), set(original))
            for name in original:
                self.assertEqual(archive.read(name), (cleaned / name).read_bytes())
        for name in ("source.tar.gz", "source.zip", "manifest.json"):
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
        unpacked = fixture.root / "unpacked"
        unpacked.mkdir()
        fixtures.arxiv.unpack(first, unpacked)
        self.assertNotIn(b"Private draft note", (unpacked / "main.tex").read_bytes())
        for name, data in original.items():
            self.assertEqual((fixture.root / name).read_bytes(), data)
        self.assertFalse((first / "validation.json").exists())
        self.assertFalse((first / "validated-main.pdf").exists())


if __name__ == "__main__":
    unittest.main()
