#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "complexUtil.h"
#include "ffts.h"



/*
 Spectral methods main. 
 
 Alex Kaiser, LBNL, 9/2/09.
*/ 

// test routines
char checkStockhamFFT(int n, struct complex *table, int tableSize) ;
char checkCooleyTukeyFFT(int n, struct complex *table, int tableSize) ;
char checkFourStepFFT(int n, int n1, int n2, struct complex *table, int tableSize) ;
char checkConvolution(int n, int n1, int n2, int b, struct complex *table, int tableSize) ;
char check3dFFT(int m, int m1, int m2, int n, int n1, int n2, int p, int p1, int p2, struct complex *table, int tableSize);

int main(){
	/*
	 This test routine performs the following computations:

	 1. Computes a complex to complex FFT three ways.
			b. Cooley Tukey
			c. Stockham's
			d. Four Step

	 2. Inverts the result of each and calculates the RMS error.

	 3. Computes a convolution of two integer vectors
		using real-to-complex and complex-to-real FFTs.

	 4. Computes a three dimensional FFT using FourStep FFTs, inverts the result
		and calculates the RMS error
	 */

	printf("Beginning spectral methods tests.\n\n"); 
	
	double startTime, endTime; 
	startTime = read_timer();
	
	// sizes for 1D fft
	int n = 0x1 << 20;
	int n1, n2;
	n1 = 0x1 << 10;
	n2 = 0x1 << 10;


	// initialize lookup table for exponential factors
	// tableSize must be equal or larger than the size of the largest fft to be computed
	int tableSize = 0x1 << 20;
	struct complex *table = initTable(tableSize) ;

	int i;
	int numTests = 5;
	char *pass = (char *) malloc(numTests * sizeof(char));
	char allPass = 1;

	// 1d FFT tests
	pass[0] = checkStockhamFFT(n, table, tableSize) ;
	pass[1] = checkCooleyTukeyFFT(n, table, tableSize) ;
	pass[2] = checkFourStepFFT(n, n1, n2, table, tableSize) ;

	// parameter for convolution test
	n = 0x1 << 18;
	n1 = 0x1 << 9;
	n2 = 0x1 << 9;
	int b = 5;

	
	// convolution test
	pass[3] = checkConvolution(n, n1, n2, b, table, tableSize) ;


	// check the 3d fft routine.
	int m, m1, m2;
	int p, p1, p2;

	m = 0x1 << 6;
	m1 = 0x1 << 3;
	m2 = 0x1 << 3;

	n = 0x1 << 6;
	n1 = 0x1 << 3;
	n2 = 0x1 << 3;

	p = 0x1 << 6;
	p1 = 0x1 << 3;
	p2 = 0x1 << 3;

	pass[4] = check3dFFT(m, m1, m2, n, n1, n2, p, p1, p2, table, tableSize);


	endTime = read_timer(); 
	printf("Total time = %f seconds.\n\n", endTime - startTime) ;
	
	// Check whether all tests passed.
	for(i=0; i<numTests; i++){
		if(pass[i]){
			printf("Test %d passed.\n", i) ;
			allPass &= 1;
		}
		else{
			fprintf(stderr, "Test %d failed.\n", i) ;
			allPass = 0;
		}
	}

	if( allPass )
		printf("\nAll tests passed.\n\n") ;
	else
		fprintf(stderr, "\nTests failed!\n\n") ;

	printf("End of spectal methods tests.\n\n");
	
	free(pass) ;

	return allPass;
}


char checkStockhamFFT(int n, struct complex *table, int tableSize){
	/*
	 Generates data and computes the FFT of the data using a Stockham FFT.
	 Inverts the result and ensures that output data compared to original data
	     has an RMS error of under 10^-10.

	 Input:
	 int n						Length of the FFT to perform.
	 struct complex *table		Lookup table of exponential factors
	 int tableSize				Size of lookup table. Must be at least n.

	 Output:
	 char pass (returned)		Whether RMS error upon inversion is under 10^-10.
	 */


	struct complex * in = allocComplex(n);
	struct complex * inCopy = allocComplex(n);
	struct complex * out ;
	struct complex * inverted ;
	double err;

	char pass;

	// error tolerance for all basic tests
	double tol = 1e-10;

	int k;
	double realRand, imagRand;

	for(k=0; k<n; k++){
		realRand = bcnrand();
		imagRand = bcnrand();
		setComplex(&in[k], realRand, imagRand); // keep copy of input array for comparisons
		setComplex(&inCopy[k], realRand, imagRand);
	}

	double startTime, endTime; 
	startTime = read_timer();
	
	// evaluate FFT
	out = stockhamFFT(in, n, -1, table, tableSize);
	
	endTime = read_timer(); 
	printf("Elapsed time = %f seconds.\n", endTime - startTime) ;

	//invert and compute RMS error with Stockham FFT
	inverted = stockhamFFT(out, n, 1, table, tableSize);
	for(k=0; k<n; k++) inverted[k] = multComplexReal(inverted[k], 1.0 / (double) n );

	struct complex rmsErr = rmsError(inCopy, inverted, n);
	printf("rms error of Stockham FFT on inversion = ");
	printComplex(rmsErr);
	printf("\n");
	err = sqrt(rmsErr.real * rmsErr.real + rmsErr.imag * rmsErr.imag) ;
	if( err < tol ){
		printf("Inversion of Stockham FFT passed.\n\n");
		pass = 1;
	}
	else{
		fprintf(stderr, "Inversion of Stockham FFT failed.\n\n");
		pass = 0;
	}

	free(in);
	free(inCopy);

	return pass ;
}

char checkCooleyTukeyFFT(int n, struct complex *table, int tableSize){
	/*
	 Generates data and computes the FFT of the data using a Cooley Tukey FFT.
	 Inverts the result and ensures that output data compared to original data
	     has an RMS error of under 10^-10.

	 Input:
	 int n						Length of the FFT to perform.
	 struct complex *table		Lookup table of exponential factors
	 int tableSize				Size of lookup table. Must be at least n.

	 Output:
	 char pass (returned)		Whether RMS error upon inversion is under 10^-10.
	 */


	struct complex * in = allocComplex(n);
	struct complex * inCopy = allocComplex(n);
	struct complex * out ;
	struct complex * inverted ;
	double err;

	char pass;

	// error tolerance for all basic tests
	double tol = 1e-10;

	int k;
	double realRand, imagRand;

	for(k=0; k<n; k++){
		realRand = bcnrand();
		imagRand = bcnrand();
		setComplex(&in[k], realRand, imagRand); // keep copy of input array for comparisons
		setComplex(&inCopy[k], realRand, imagRand);
	}

	double startTime, endTime; 
	startTime = read_timer();

	// evaluate FFT
	out = cooleyTukeyFFT(in, n, -1, table, tableSize);

	endTime = read_timer(); 
	printf("Elapsed time = %f seconds.\n", endTime - startTime) ;
	
	//invert and compute RMS error with Stockham FFT
	inverted = cooleyTukeyFFT(out, n, 1, table, tableSize);
	for(k=0; k<n; k++) inverted[k] = multComplexReal(inverted[k], 1.0 / (double) n );

	struct complex rmsErr = rmsError(inCopy, inverted, n);
	printf("rms error of Cooley Tukey FFT on inversion = ");
	printComplex(rmsErr);
	printf("\n");
	err = sqrt(rmsErr.real * rmsErr.real + rmsErr.imag * rmsErr.imag) ;
	if( err < tol ){
		printf("Inversion of Cooley Tukey FFT passed.\n\n");
		pass = 1;
	}
	else{
		fprintf(stderr, "Inversion of Cooley Tukey FFT failed.\n\n");
		pass = 0;
	}

	free(in);
	free(inCopy);

	return pass ;
}


char checkFourStepFFT(int n, int n1, int n2, struct complex *table, int tableSize){
	/*
	 Generates data and computes the FFT of the data using a Four Step FFT.
	 Inverts the result and ensures that output data compared to original data
	     has an RMS error of under 10^-10.

	 Input:
	 int n						Length of the FFT to perform.
	 int n1, n2					Integers such that n = n1 * n2. 
	 struct complex *table		Lookup table of exponential factors. 
	 int tableSize				Size of lookup table. Must be at least n.

	 Output:
	 char pass (returned)		Whether RMS error upon inversion is under 10^-10.
	 */


	struct complex * in = allocComplex(n);
	struct complex * inCopy = allocComplex(n);
	struct complex * out ;
	struct complex * inverted ;
	double err;

	char pass;

	// error tolerance for all basic tests
	double tol = 1e-10;

	int k;
	double realRand, imagRand;

	for(k=0; k<n; k++){
		realRand = bcnrand();
		imagRand = bcnrand();
		setComplex(&in[k], realRand, imagRand); // keep copy of input array for comparisons
		setComplex(&inCopy[k], realRand, imagRand);
	}

	double startTime, endTime; 
	startTime = read_timer();

	// evaluate FFT
	out = fourStepFFT(in, n, n1, n2, -1, table, tableSize);

	endTime = read_timer(); 
	printf("Elapsed time = %f seconds.\n", endTime - startTime) ;
	
	//invert and compute RMS error with Four Step FFT
	inverted = fourStepFFT(out, n, n1, n2, 1, table, tableSize);
	for(k=0; k<n; k++) inverted[k] = multComplexReal(inverted[k], 1.0 / (double) n );

	struct complex rmsErr = rmsError(inCopy, inverted, n);
	printf("rms error of Four Step FFT on inversion = ");
	printComplex(rmsErr);
	printf("\n");
	err = sqrt(rmsErr.real * rmsErr.real + rmsErr.imag * rmsErr.imag) ;
	if( err < tol ){
		printf("Inversion of Four Step FFT passed.\n\n");
		pass = 1;
	}
	else{
		fprintf(stderr, "Inversion of Four Step FFT failed.\n\n");
		pass = 0;
	}

	free(in);
	free(inCopy);

	return pass ;
}

char checkConvolution(int n, int n1, int n2, int b, struct complex *table, int tableSize){
	/* 
	 Checks convolution of real, integer data. 
	 Input is randomly generated in the range [0, 2^b - 1]. 
	 A bound for the validity of the verification scheme is checked, namely that:
		2*b + log2(n) <= 53. 
	 All output data must be within the specified tolerance of an integer for test to pass. 
	 The first 'numToCompute' elements of the convolution will be also computed with a naive convolution algorithm. 
	 Output must match these numbers also within the specified tolerance. 
	 
	 Input:
	 int n						Length of the convolution to perform.
	 int n1, n2					Integers such that n = n1 * n2. 
	 int b						Bound for random number generator. 
	 struct complex *table		Lookup table of exponential factors. 
	 int tableSize				Size of lookup table. Must be at least n.
	 
	 Output:
	 char pass (returned)		Whether above checks are within specified tolerance. 
	 */ 

	printf("Convolution test:\n");

	int numToCompute = 50 ;
	char pass;
	char passNearestInt = 1; 
	char passNaiveComparison = 1; 

	double tol = 1e-6;
	double maxDiff = 0; 
	double currentDiff ; 
	// test for invalid parameters and quick exit
	if(2*b + log2(n) > 53){
		fprintf(stderr, "2*b + log2(n) must be less than or equal to 53 for verification scheme to be valid.\nTest failed.\n") ;
		pass = 0;
		return pass; 
	}

	double *firstReal = (double *) malloc(n * sizeof(double));
	double *secondReal = (double *) malloc(n * sizeof(double));
	double *convResult ;
	double *naiveConvResult ;

	int k;
	for(k=0; k<n; k++){
		firstReal[k] = (double) getRandomInt(b);
		secondReal[k] = (double) getRandomInt(b);
	}

	double startTime, endTime; 
	startTime = read_timer();
	
	convResult = convolution(firstReal, secondReal, n, n1, n2, table, tableSize);

	endTime = read_timer(); 
	printf("Elapsed time = %f seconds.\n", endTime - startTime) ;
	
	for(k=0; k<n; k++){
		currentDiff = fabs(round(convResult[k]) - convResult[k]) ; 
	    if( currentDiff > tol ){
	        fprintf(stderr,"convResult[%d] is more than the tolerance of %e from the nearest integer.\nTest failed.\n", k, tol) ;
	        passNearestInt = 0;
	        break;
	    }
		if( currentDiff > maxDiff ) 
			maxDiff = currentDiff ; 
	}

	printf("Max difference from integer in convolution result:  %e\n", maxDiff);

	maxDiff = 0; 
	
	startTime = read_timer();
	naiveConvResult = naiveConvolution(firstReal, secondReal, n, numToCompute);
	endTime = read_timer(); 
	printf("Naive convolution for verifictaion time = %f seconds.\n", endTime - startTime) ;
	
	
	for(k=0; k<numToCompute; k++){
		currentDiff = fabs(convResult[k] - naiveConvResult[k]) ; 
	    if( fabs(convResult[k] - naiveConvResult[k]) > tol ){
	        fprintf(stderr,"convResult[%d] is more than the tolerance of %e from naive convolution result.\nTest failed.\n", k, tol) ;
	        passNaiveComparison = 0;
	        break;
	    }
		if( currentDiff > maxDiff ) 
			maxDiff = currentDiff ; 
	}

	printf("Max difference from naive convolution result:  %e\n", maxDiff);
	
	if(passNearestInt && passNaiveComparison){
		pass = 1; 
		printf("Convolution test passed.\n\n");
	}
	else{
		pass = 0; 
		printf("Convolution test failed.\n\n");
	}

	free(firstReal) ;
	free(secondReal) ;
	free(convResult) ;
	free(naiveConvResult) ;

	return pass;
}


char check3dFFT(int m, int m1, int m2, int n, int n1, int n2, int p, int p1, int p2, struct complex *table, int tableSize){
	/*
	Checks three dimensional FFT. 
	Computes the 3D FFT of randomly generated, complex input data. 
	Inverts the result and ensures that output data compared to original data
	has an RMS error of under 10^-10.
	 
	Input:
		int m							Length of input in first dimension, must be power of two. 
		int m1, m2						Integers such that m = m1 * m2. Each must be a power of two.
		int n							Length of input in second dimension, must be power of two. 
		int n1, n2						Integers such that n = n1 * n2. Each must be a power of two. 
		int p							Length of input in third dimension, must be power of two. 
		int p1, p2						Integers such that p = p1 * p2. Each must be a power of two.
		int sign						Sign of exponent in FFT. -1 for forward FFT, 1 for inverse. 
		struct complex *table			Table of precompted exponential factors. 	 
		int tableSize					Size of table, greater than or equal to m,n and p, and a power of two.  

 	 Output:
		char pass (returned)			Whether RMS error upon inversion is under 10^-10.
	 */ 

	int j,k,l;

	struct complex *** in3D = allocComplex3d(m, n, p);
	struct complex *** x3D = allocComplex3d(m, n, p);
	struct complex *** inverted3D ;

	char pass;

	double tol = 1e-10;

	double realRand, imagRand;

	for(j=0; j<m; j++){
		for(k=0; k<n; k++){
			for(l=0; l<p; l++){
				realRand = bcnrand();
				imagRand = bcnrand();
				setComplex(&in3D[j][k][l], realRand, imagRand); // keep copy of input array for comparisons
				setComplex(&x3D[j][k][l], realRand, imagRand);
			}
		}
	}
	
	double startTime, endTime; 
	startTime = read_timer();

	x3D = fft3D(x3D, m, m1, m2, n, n1, n2, p, p1, p2, -1, table, tableSize);
	
	endTime = read_timer(); 
	printf("Elapsed time = %f seconds.\n", endTime - startTime) ;
	
	inverted3D = fft3D(x3D, m, m1, m2, n, n1, n2, p, p1, p2, 1, table, tableSize);
	double inversionCoefficient = 1.0 / (double) (m*n*p);

	for(j=0; j<m; j++){
		for(k=0; k<n; k++){
			for(l=0; l<p; l++){
				inverted3D[j][k][l] = multComplexReal(inverted3D[j][k][l], inversionCoefficient);
			}
		}
	}

	struct complex rmsErr3D = rmsError3D(in3D, inverted3D, m, n, p);
	printf("rms error of 3D FFT on inversion = \n");
	printComplex(rmsErr3D);
	printf("\n");
	double err = sqrt( rmsErr3D.real * rmsErr3D.real + rmsErr3D.imag * rmsErr3D.imag ) ;
	if( err < tol){
		printf("Inversion of 3D FFT passed.\n\n");
		pass = 1;
	}
	else{
		printf("Inversion of 3D FFT failed.\n\n");
		pass = 0;
	}

	freeComplex3d(in3D, m, n, p);
	freeComplex3d(x3D, m, n, p);

	return pass;
}

