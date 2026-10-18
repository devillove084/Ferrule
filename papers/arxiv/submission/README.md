# arXiv source preparation CLI

**Preparation only, never submission.** No account, password, token, cookies,
browser automation, upload request, license choice, or final submission action
is implemented. Source preparation is not scientific approval or arXiv acceptance.

## Official workflow and tool choice

Checked against the following official pages on **2026-10-18**:

- [Submitting an article](https://info.arxiv.org/help/submit/index.html): start a
  submission on the registered author's user page, upload sources, check files,
  compile, review metadata and the generated PDF, and finally confirm **Submit
  Article**. It accepts `zip` and `tar.gz`, including relative subdirectories.
- [TeX/LaTeX submissions](https://info.arxiv.org/help/submit_tex.html), especially
  **Include .bib or .bbl files if you use BibTeX/Biber**: the current submission
  system processes `.bib` files. A supplied `.bbl` takes precedence and must match
  the main TeX filename. The CLI therefore ships the current `references.bib`,
  not an old `build/main.bbl`. Local archive validation runs fresh BibTeX.
- [Official API](https://info.arxiv.org/help/api/index.html) and its
  [manual](https://info.arxiv.org/help/api/user-manual.html) describe article
  retrieval/search and metadata, not a public article-submission API. No supported
  official submission CLI/public posting API was found in these documents.
  A metadata API client or third-party paper downloader cannot submit a paper.
- [Licenses](https://info.arxiv.org/help/license/index.html) and
  [TeX Live at arXiv](https://info.arxiv.org/help/faq/texlive.html): the author must
  choose an authorized license and inspect the PDF compiled on arXiv's system;
  a local compilation cannot certify its output.

The default cleaner is Google's mature community tool
[`google-research/arxiv-latex-cleaner`](https://github.com/google-research/arxiv-latex-cleaner),
[`arxiv-latex-cleaner==1.0.11`](https://pypi.org/project/arxiv-latex-cleaner/1.0.11/),
the current PyPI release when checked. It removes comments and unnecessary
sources; it **does not submit papers**. Its published source distribution contains
an **Apache License 2.0** license. Its license does not select the manuscript's
arXiv license or relicense bundled third-party sources.

## Isolated installation

Use Python **3.12**. From the repository root:

```sh
/home/admin/wxw403883/anaconda3/bin/python3.12 -m venv papers/arxiv/build/submission-venv
papers/arxiv/build/submission-venv/bin/python -m pip --isolated --disable-pip-version-check install --index-url https://mirrors.aliyun.com/pypi/simple/ --timeout 60 --retries 2 --no-input -r papers/arxiv/submission/requirements.txt
```

`requirements.txt` pins the cleaner and all four runtime dependencies (absl-py,
Pillow, PyYAML, regex). Do not install into the user's base conda environment or
upgrade unrelated packages. Installation needs network access to the selected
package index and its package host. The command above uses the Aliyun PyPI mirror
with normal TLS verification; `https://pypi.org/simple` can also be used. No
`trusted-host`, global pip configuration change, or dependency downgrade is
needed. Preparation and validation do not perform network requests or install
TeX packages. On 2026-10-18, installation completed in this isolated environment
from Aliyun after the original PyPI download timed out and the Tsinghua mirror
returned HTTP 403. Installed versions: cleaner 1.0.11, absl-py 2.5.1, Pillow
13.0.0, PyYAML 6.0.3, regex 2026.9.29. The explicit `--no-clean` fallback below
needs only the standard library.

## Commands

After the other manuscript edits are complete and the main build is approved,
from the repository root:

```sh
# 1. Standard manuscript build; unchanged existing Makefile behavior.
make -C papers/arxiv

# 2. Clean only an allowlisted temporary copy; package without compiling.
make -C papers/arxiv submission

# 3. Later, compile the extracted tar.gz, not the working source directory.
make -C papers/arxiv submission-validate
```

The preparation command produces these **new** files under
`papers/arxiv/build/submission/`:

- `source.tar.gz` and `source.zip`: equivalent source payloads; upload **one**.
- `manifest.json`: original and packaged SHA-256 hashes, sizes, archive hashes,
  cleaner version, system TeX dependencies, policy hash, pending author checks,
  and an explicit **not validated / not submitted** status.

Successful validation additionally produces `validated-main.pdf`,
`validation.json`, `validation-final.log`, `validation-bibtex.log`, and the fresh
`validated-main.bbl`. All are local review artifacts **outside both source
archives**, not substitutes for the arXiv-generated PDF. The report records the
page count, generated bibliography-item count, validator/source archive/PDF and
log hashes. Manifest preparation status stays unchanged; the separate validation
report identifies the archive hash actually checked. No validation artifacts are
written on compilation or final-log-check failure.

For the 2026-10-18 coordinated release check, a separate read-only PDF/ZIP/TAR
inspection produced `build/submission/audit.json`: 20 identical archive members,
41 PDF pages, Huanbing Lu and the approved email, references numbered 1–34, and
all ten figures loaded. This supplemental audit is not generated automatically
by `validate` and does not constitute arXiv submission. PDF text inspection used
`pypdf==6.1.1` in a separate ignored review venv, leaving the submission runtime
requirements unchanged:

```sh
/home/admin/wxw403883/anaconda3/bin/python3.12 -m venv papers/arxiv/build/submission-review-venv
papers/arxiv/build/submission-review-venv/bin/python -m pip --isolated --disable-pip-version-check install --index-url https://mirrors.aliyun.com/pypi/simple/ --timeout 60 --retries 2 --no-input pypdf==6.1.1
```

For an explicitly dependency-free preparation that retains **all comments**:

```sh
make -C papers/arxiv submission SUBMISSION_PYTHON=/home/admin/wxw403883/anaconda3/bin/python3.12 SUBMISSION_ARGS=--no-clean
make -C papers/arxiv submission-validate SUBMISSION_PYTHON=/home/admin/wxw403883/anaconda3/bin/python3.12
```

For this fallback, manually inspect source comments for private notes. The tool
never silently switches from cleaning to this mode. To make a new revision,
choose a new directory rather than deleting or overwriting a prior package:

```sh
make -C papers/arxiv submission SUBMISSION_ARGS='--output build/submission/revision-2'
make -C papers/arxiv submission-validate SUBMISSION_ARGS='--output build/submission/revision-2 --timeout 240'
```

Direct CLI help is available without installed cleaner dependencies:

```sh
/home/admin/wxw403883/anaconda3/bin/python3.12 papers/arxiv/submission/arxiv.py --help
/home/admin/wxw403883/anaconda3/bin/python3.12 papers/arxiv/submission/arxiv.py prepare --help
/home/admin/wxw403883/anaconda3/bin/python3.12 papers/arxiv/submission/arxiv.py validate --help
```

## Scope and safety

- `source-files.json` is an explicit, reviewed file allowlist, not a recursive
  copy rule. Dependencies are followed from `main.tex`: literal `input/include`,
  `bibliography`, `includegraphics`, the manuscript's `paperplot` wrapper, and
  named local classes/styles. The current policy covers the main file, eight
  sections, `references.bib` and ten actually referenced PDF figures.
- Every allowlisted source must be referenced; new/missing dependencies or unused
  entries stop preparation. When the manuscript's dependencies change, review
  and update the policy, not a glob. Literal braced paths and UTF-8 sources are
  required. Macro-generated filenames, `graphicspath`, external documents,
  BibLaTeX, SVG conversion, and arbitrary TeX dependency discovery are deliberately
  unsupported. The scanner handles this manuscript, not the whole TeX language.
- Local `.sty`, `.cls`, `.bst`, `.def`, `.cfg`, and `.clo` dependencies may be
  explicitly allowlisted. Standard TeX packages are recorded by name, not copied
  from system `texmf`. No `.fls` scan ever imports the entire system installation.
- No experiment/weights/vendor/git/build tree, logs, manuscript PDF, unreferenced
  figures/PNG previews, temporary candidate bibliography, or unused figure source
  is included. Referenced raster figures could be explicitly reviewed, but this
  manuscript uses PDF figures only. Sources are capped at 20 MiB per file and
  50 MiB total as conservative local safety limits, not a promise of acceptance.
- Relative paths are preserved. Traversal, absolute/dynamic paths, symlinks,
  hidden files, duplicate archive entries, and checksum/size mismatches are
  rejected. Output is restricted to `build/submission` and its descendants;
  existing output directories and validation artifacts are never overwritten.
  Original files are never cleaned, modified, or removed.
- The cleaner sees only a fresh allowlisted temporary directory. No image resize,
  PDF recompression, PNG conversion or user config is enabled. Figures, `.bib`,
  local styles and reviewed license files retain their original bytes. Cleaned
  TeX dependencies are rechecked before packaging. If a TeX file contains a
  detected copyright/SPDX/license notice, cleaning stops and requests `--no-clean`
  rather than stripping the notice.
- `licenses` in the policy is intentionally empty: no new license is imposed on
  the manuscript or third-party files, and the repository's software license is
  not automatically copied. When a necessary local third-party style is added,
  add its existing required `LICENSE*`/`COPYING*`/`NOTICE*` file (root or `licenses/`)
  explicitly and verify its distribution rights before packaging.
- Stable sorted names, timestamps, permissions, gzip headers and canonical JSON
  make repeated preparations byte-identical for identical sources, policy,
  Python/zlib and cleaner environment. Runtime logs and timestamps are not added
  to the source archives. Across different toolchains, compression may differ.
- Validation verifies the tar.gz against the manifest, safely extracts it into a
  clean temporary directory, and runs `pdflatex -no-shell-escape
  -interaction=nonstopmode -halt-on-error -file-line-error main.tex`, `bibtex main`,
  then two more pdfLaTeX passes. Only the final pdfLaTeX log is checked for
  unresolved citations/references, rerun requests, any warning, and overfull or
  underfull boxes; transient first-pass diagnostics are not grounds for failure.
  BibTeX warnings also fail validation. It uses no stale `.bbl`, latexmk hook,
  shell command, network installer or inherited user TeX tree. Each process has a timeout
  (default 180 seconds). This is for trusted manuscript sources: TeX is not a
  sandbox and this is **not OS-level network isolation**. Do not validate hostile
  downloaded sources; use an external network-disabled sandbox if required.

## Tests and final human gates

```sh
make -C papers/arxiv submission-test SUBMISSION_PYTHON=/home/admin/wxw403883/anaconda3/bin/python3.12
```

Tests use small synthetic fixtures and mocked TeX commands; they never build the
real manuscript or run experiments, Rust, GPU code, or an upload. The unit suite
checks the cleaner's staging contract with a mocked cleaner. With the pinned
venv installed, run the real-cleaner smoke alone without repeating the unit suite:

```sh
papers/arxiv/build/submission-venv/bin/python papers/arxiv/submission/tests/test_cleaner_smoke.py -v
```

This smoke passed on 2026-10-18 with cleaner 1.0.11. It invokes the real
`python -m arxiv_latex_cleaner SOURCE --keep_bib`, verifies the expected
`SOURCE_arXiv` output, comment removal and retained bibliography/figure bytes,
then exercises the wrapper twice and checks equivalent, stable archives and
unchanged originals. No TeX compiler is invoked. It skips when the cleaner is
not installed; real manuscript packaging/compilation remains a separate,
coordinated step.

Before browser upload, the author must confirm **Huanbing Lu** and the approved
original email `luhuanbing084@gmail.com` in the final TeX/PDF and metadata. This
CLI does not edit or infer author fields. Also confirm title/abstract,
affiliations and author order, category/cross-list, endorsement, rights and the
chosen arXiv license, author declarations, the file manifest, and the PDF produced
by **arXiv itself**. Only the author should authorize the website's final
**Submit Article** action. Nothing in this directory has uploaded or submitted
this manuscript.
