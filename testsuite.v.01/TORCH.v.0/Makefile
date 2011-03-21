include Makefile.in

SUBDIRS = denseLinearAlgebra/C sparseLinearAlgebra/C spectralMethods/C structuredGrid/C monteCarlo/C sort nBody/C/2D nBody/C/3D

subdirs: $(SUBDIRS)

.PHONY: all clean subdirs

$(SUBDIRS): Makefile
	@for d in $(SUBDIRS); do (cd $$d; $(MAKE);); done

all: subdirs

clean: 
	@for d in $(SUBDIRS); do (cd $$d; $(MAKE) clean;); done

