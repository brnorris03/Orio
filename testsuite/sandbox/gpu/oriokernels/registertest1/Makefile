
# ---------------------------------------------------------------------
# Compiler selection 
# ---------------------------------------------------------------------

CCC = nvcc -O3

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


main: main.o init_kernel.o
	$(CCC) -arch=sm_20 init_kernel.o main.o -o main
main.o: $(EXSRC)/main.cu
	$(CCC) -arch=sm_20 -c $(EXSRC)/main.cu
init_kernel.o: $(EXSRC)/init_kernel.cu
	$(CCC) -arch=sm_20 -c $(EXSRC)/init_kernel.cu

