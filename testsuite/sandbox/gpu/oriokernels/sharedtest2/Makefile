
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


main: dot.o
	@echo " "
	@echo "-----------------------------------------------------------"
	@echo "run with: ./main <array size>  <optional rseed (default=1)>"
	@echo "-----------------------------------------------------------"
	@echo " "
	$(CCC) -arch=sm_20 dot.o -o main
dot.o: $(EXSRC)/dot.cu
	$(CCC) -arch=sm_20 -c $(EXSRC)/dot.cu
