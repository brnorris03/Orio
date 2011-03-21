This directory contains routines and scripts for the problems of the 
Spectral Methods motif in Matlab. The following is a summary of the files 
and their use. 

Scripts: 

    1. checkFFT.m - computes a complex to complex FFT, inverts the result
    and calculates RMS error. The optional last line compares results 
    to built in Matlab functions.

    2. checkConvolution.m - computes a convolution of two integer vectors
    using real-to-complex and complex-to-real FFTs. 

    3. checkFFT3D.m - computes a three dimensional FFT, inverts the results
	and calculates RMS error. 

    4. checkPoisson.m - solves a Poisson PDE as in the NAS FT benchmark. 
	Operation of this file requires that the C file "generateData.c"
	has been run, as described below. 

C files:

    1. generateData.c - generates data to solve the Poisson PDE. Compile
	with any C compiler. Takes no input and writes output to appropriately
	named file.  

Functions:

  FFT routines:

    1. function res = fourStepFFT( x, n, n1, n2, sign)

    % Computes FFT(x) by the four step method. 
    % Primary one-dimensional FFT algorithm

    % input - 
    % n = length(x). Must be power of two. 
    % n = n1*n2. Best if n1 is close to sqrt(n). 
    % Sign - sign of exponent in FFT, -1 for standard FFT. 
    % to calc inv. FFT, use sign = 1 and devide y by n after function call. 
    % individual FFT's are called by Stockham's algorithm. 

    % output - 
    % res = fft of input


    2. function y = stockhamFFT( x , n, sign) 

    % computes FFT of x using the Stockham FFT algorithm
    % secondary one-dimensional FFT called by fourStep fft. 
    % n = length(x). Must be power of two. 
    % Sign - sign of exponent in FFT.


    3. function y = r_to_cFFT( x, n, n1, n2, sign)

    % input: 
    % x - input vector, must be real. 
    % n = n1*n2 , length of x
    % n must be a multiple of four and power of two or errors will result. 
    % sign is the sign of the transform

    % output:
    % y - result of the FFT interleaved as length n/2 complex


    4. function W = c_to_rFFT( x, n, n1, n2, sign)

    % input: 
    % x - complex input vector of length n/2 + 1. First and last values must be
    %   real for this to represent the forward transform of real data. 
    % n = n1*n2 , length of original real input
    % n must be a multiple of four or errors will result. 
    % sign is the sign of the transform

    % output:
    % W - result of the FFT interleaved as length n/2 complex


    5. function res = FFT2D( x, m, m1, m2, n, n1, n2, sign)

    % computes a 2d fft of the m by n input array x. 
    % m = m1 * m2, n = n1 * n2
    % sign is the sign of the transform


    6. function res = FFT3D( x, m, m1, m2, n, n1, n2, p, p1, p2, sign)


    % computes a 3D fft of the m by n input array x. 
    % dimensions m = m1 * m2, n = n1 * n2, p = p1 * p2
    % sign is the sign of the transform


  Related computations:

    1. function F = convolution(a, b, n, n1, n2)

    % input: 
    % a, b - real data
    % n = n1*n2 - length of a and b

    %output - F = the acyclic convolution of a and b


    2. function checkSums = solvePoisson(m, m1, m2, n, n1, n2, p, p1, p2, NSteps)

        % solves Poisson PDE as in the NAS FT benchmark. 
        % input data is read from a file, so correct parameters are required
        % set m = n = p = 64, m1 = n1 = p1 = 8, m2 = n2 = p2 = 8
        % NSteps = 6


Utilities: 

    1. function a = getRandomInt(m, n, b)

    % returns m*n matrix of random int's in the range [0, 2^b - 1]


    2. function err = rmsError(x, y)

    % calculates the rms error between vectors x and y.


    3. function err = rmsError3(x, y)

    % calculates the rms error between 3D arrays x and y. 





