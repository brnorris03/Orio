F77=f77                                            
FFLAGS = -c -O3                                      
#
# compiler must be defined on line 1 w/ whitespace to fill 30 chars
# then flags, also to 30 chars
#
#F77 = /soft/apps/packages/pgi-3.1/linux86/bin/pgf77
F77 = f77
#FFLAGS = -c -n32 -mips4 -S -O3 -show
#FFLAGS = -c -n32 -mips4 -O3 -show
#LDFLAGS = -Vaxlib
LDFLAGS = 
.SUFFIXES: .f .o

RTE    = /home/research/faculty/jessup/PAPERS/BOYANA/CODE
RTEb    = /home/research/faculty/jessup/PAPERS/BOYANA/CODE/BLAS
FILES  =  $(RTE)/testprogram.o $(RTE)/vadd.o $(RTE)/waxpby.o $(RTE)/bicgkernel.o $(RTE)/gesummv.o $(RTE)/gemver.o
FILESb = $(RTEb)/dblas1.o $(RTEb)/dblas2.o $(RTEb)/dblas3.o $(RTEb)/blashelp.o

.f.o:
	@echo $(F77) $(FFLAGS) $<
	@sh -c 'if { $(F77) $(FFLAGS) $< ; S=$$? ; test "$$S" = "1" ; \
	} then { exit 0 ; } else { exit $$S ; } fi';

prog: $(FILES) $(FILESb)
	$(F77) $(LDFLAGS) -o $(RTE)/prog $(FILES) $(FILESb)
