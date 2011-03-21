all: structuredGridMain

# makefile for structured grid codes
# Alex Kaiser, LBNL, 7/2010


INCDIRS = -iquote../../sparseLinearAlgebra/C -iquote../../spectralMethods/C

LOCALOBJS = structuredGridMain.o heatEqnSolvers.o testFunctions.o differentialOps.o gridUtil.o
FFTOBJS = ../../spectralMethods/C/complexUtil.o ../../spectralMethods/C/ffts.o 
SLAOBJS = ../../sparseLinearAlgebra/C/sparseUtil.o ../../sparseLinearAlgebra/C/sparseArithmetic.o ../../sparseLinearAlgebra/C/generators.o

# other directories in which code must be built on which the current files depend
DEPDIRS = ../../spectralMethods/C ../../sparseLinearAlgebra/C


include ../../Makefile.in


structuredGridMain: $(LOCALOBJS) dependentDirs
	$(CC)  $(FFTOBJS) $(SLAOBJS) $(LOCALOBJS) $(CFLAGS) $(CLIBS) -o structuredGridMain

# ensure that non-local files are built. 
dependentDirs:
	@for d in $(DEPDIRS); do (cd $$d; $(MAKE);); done


structuredGridMain.o: 
	$(CC) $(CFLAGS) $(INCDIRS) -c structuredGridMain.c
	
heatEqnSolvers.o:
	$(CC) $(CFLAGS) $(INCDIRS) -c heatEqnSolvers.c

testFunctions.o: 
	$(CC) $(CFLAGS) -c testFunctions.c

differentialOps.o: 
	$(CC) $(CFLAGS) -c differentialOps.c
	
gridUtil.o: 
	$(CC) $(CFLAGS) $(INCDIRS) -c gridUtil.c

clean:
	rm *.o structuredGridMain
