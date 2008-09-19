# Generic makefile

PDFLATEX = pdflatex
BIBTEX = bibtex

FILENAME = paper
BIBFILE = paper

SOURCES =  paper.bib  $(wildcard *.tex) $(wildcard figures/*.png)

all: $(FILENAME).pdf

clean:
	rm -f table*.eps *.dvi *.ps *.ps.Z *.jps *.log *.blg *.aux *.bbl *pdf

.SUFFIXES: .bib .tex .dvi .fig .ps .pdf


$(FILENAME).pdf: $(SOURCES)
	$(PDFLATEX) $(FILENAME)
	- $(BIBTEX) $(FILENAME)
	$(PDFLATEX) $(FILENAME)
	$(PDFLATEX) $(FILENAME)

texclean:
	rm -f *.dvi *.aux *.blg *.log


