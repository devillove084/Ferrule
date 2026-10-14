.PHONY: all preview render clean

all: architecture.pdf transaction-lifecycle.pdf operation-state.pdf

%.pdf: %-source.tex %-diagram.tex figure-driver.tex figure-style.tex
	latexmk -pdf -pdflatex="pdflatex -no-shell-escape %O %S" -interaction=nonstopmode -halt-on-error -outdir=build -jobname=$* $<
	cp build/$@ $@

# Match the manuscript's page/font setup without editing or compiling main.tex.
# This preview includes real captions, leaves and cross-references.
preview: all
	make -C .. -f figures/Makefile preview-from-paper

.PHONY: preview-from-paper
preview-from-paper:
	latexmk -pdf -pdflatex="pdflatex -no-shell-escape %O %S" -interaction=nonstopmode -halt-on-error -outdir=figures/build -jobname=figure-preview figures/paper-preview.tex

# Ghostscript is optional; no installation or GPU is used.
render: all
	gs -q -dSAFER -dBATCH -dNOPAUSE -sDEVICE=png16m -r120 -sOutputFile=build/architecture-preview.png architecture.pdf
	gs -q -dSAFER -dBATCH -dNOPAUSE -sDEVICE=png16m -r120 -sOutputFile=build/transaction-lifecycle-preview.png transaction-lifecycle.pdf
	gs -q -dSAFER -dBATCH -dNOPAUSE -sDEVICE=png16m -r120 -sOutputFile=build/operation-state-preview.png operation-state.pdf

clean:
	latexmk -c -outdir=build -jobname=architecture architecture-source.tex
	latexmk -c -outdir=build -jobname=transaction-lifecycle transaction-lifecycle-source.tex
	latexmk -c -outdir=build -jobname=operation-state operation-state-source.tex
