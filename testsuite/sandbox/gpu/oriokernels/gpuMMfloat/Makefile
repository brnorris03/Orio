
# ---------------------------------------------------------------------
# Compiler selection 
# ---------------------------------------------------------------------

CCC = /usr/local/cuda/bin/nvcc -O3

# ---------------------------------------------------------------------
# Compiler options 
# ---------------------------------------------------------------------

#CCOPT = -O -fPIC -fexceptions -DNDEBUG -DIL_STD

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
	$(CCC) -Xcompiler -fopenmp -arch=sm_13 main.o -o main
main.o: $(EXSRC)/main.cu
	$(CCC) -Xcompiler -fopenmp -arch=sm_13 -c $(EXSRC)/main.cu

