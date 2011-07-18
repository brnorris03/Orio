
# ---------------------------------------------------------------------
# Compiler selection 
# ---------------------------------------------------------------------

CCC = nvcc

# ---------------------------------------------------------------------
# Compiler options 
# ---------------------------------------------------------------------

OMP = -Xcompiler -fopenmp
BLAS = -lcublas
CCOPT = -O3 $(OMP) $(BLAS) -arch=sm_20

# ---------------------------------------------------------------------
# Link options and libraries
# ---------------------------------------------------------------------


all:
	make all_cu

execute: all
	make execute_cu

EXDIR         = .
EXSRC         = $(EXDIR)
NVINC	      = /usr/local/cuda/include

#------------------------------------------------------------
#  make all      : to compile the examples. 
#  make execute  : to compile and execute the examples.
#------------------------------------------------------------

CEX = main 

all_cu: $(CEX)

execute_cu: $(CEX)
	 ./main
# ------------------------------------------------------------
clean :
	/bin/rm -rf *.o
	/bin/rm -rf $(CEX)

# ------------------------------------------------------------


main: main.o 
	$(CCC) $(CCOPT) main.o -o main
main.o: $(EXSRC)/main.cu
	$(CCC) $(CCOPT) -c $(EXSRC)/main.cu

