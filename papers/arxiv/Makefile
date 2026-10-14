.PHONY: all pdf check-tools clean help

all: pdf

check-tools:
	@command -v latexmk >/dev/null 2>&1 || { printf '%s\n' 'Missing latexmk. Install a TeX distribution with latexmk before building.' >&2; exit 1; }
	@command -v pdflatex >/dev/null 2>&1 || { printf '%s\n' 'Missing pdflatex. Install pdfLaTeX before building.' >&2; exit 1; }

pdf: check-tools
	latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex

clean:
	@command -v latexmk >/dev/null 2>&1 || { printf '%s\n' 'Missing latexmk; no files were removed.' >&2; exit 1; }
	latexmk -C -outdir=build main.tex

help:
	@printf '%s\n' 'make       Build build/main.pdf with latexmk and pdfLaTeX' 'make clean Remove generated main-document files only' 'No target downloads packages or runs model experiments.'
