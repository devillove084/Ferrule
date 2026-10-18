.PHONY: all pdf check-tools clean help submission submission-validate submission-test

SUBMISSION_PYTHON ?= build/submission-venv/bin/python
SUBMISSION_ARGS ?=

all: pdf

check-tools:
	@command -v latexmk >/dev/null 2>&1 || { printf '%s\n' 'Missing latexmk. Install a TeX distribution with latexmk before building.' >&2; exit 1; }
	@command -v pdflatex >/dev/null 2>&1 || { printf '%s\n' 'Missing pdflatex. Install pdfLaTeX before building.' >&2; exit 1; }

pdf: check-tools
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex

clean:
	@command -v latexmk >/dev/null 2>&1 || { printf '%s\n' 'Missing latexmk; no files were removed.' >&2; exit 1; }
	latexmk -C -outdir=build main.tex

submission:
	$(SUBMISSION_PYTHON) submission/arxiv.py prepare $(SUBMISSION_ARGS)

submission-validate:
	$(SUBMISSION_PYTHON) submission/arxiv.py validate $(SUBMISSION_ARGS)

submission-test:
	$(SUBMISSION_PYTHON) -m unittest discover -s submission/tests -v

help:
	@printf '%s\n' 'make       Build build/main.pdf with latexmk and pdfLaTeX' 'make clean Remove generated main-document files only' 'make submission          Prepare source archives only; see submission/README.md for setup' 'make submission-validate Compile a prepared archive; never uploads/submits' 'make submission-test     Run submission CLI unit tests, not the manuscript build' 'No target downloads packages or runs model experiments.'
