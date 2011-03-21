This directory contains routines and scripts for the problems of the 
Spectral Methods motif in Matlab. The following is a summary of the files 
and their use. 

Files:

	1. test.c - This test routine performs the following computations:
	 
		1. Computes a complex to complex FFT, inverts the result
		and calculates the RMS error.
	 
		2. Computes a convolution of two integer vectors
		using real-to-complex and complex-to-real FFTs.
	 
		3. Computes a three dimensional FFT, inverts the result
		and calculates the RMS error
	 
		4. Solves a Poisson PDE as in the NAS FT benchmark. Confirmation
		is hard-coded to a specific size, though computation works for any 
		dimension. 

	2. ffts.c and ffts.h - Rountines for FFTs
	
	3. complexUtil.c and complexUtil.h - Rountines for complex arithmetic. 


Compiling: 

	The included Makefile builds and links all included files. 