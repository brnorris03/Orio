DVIPSFLAGS = -f -tletter -Ppdf -G0 -O 0.0in,0.0in

# For the Mac only, comment out for other architectures
#LATEXFLAGS = "-output-format=0"

all: paper.pdf

paper.dvi: $(wildcard *.tex) $(wildcard *.bib) makefile
	latex $(LATEXFLAGS) paper
	bibtex paper
	latex $(LATEXFLAGS) paper
	latex $(LATEXFLAGS) paper

paper.pdf: paper.dvi
	dvips $(DVIPSFLAGS) -o paper.ps paper.dvi
	ps2pdf paper.ps paper.pdf

clean:
	$(RM) *.aux *.log paper.dvi paper.ps paper.pdf paper.bbl paper.blg

.PHONY: clean
