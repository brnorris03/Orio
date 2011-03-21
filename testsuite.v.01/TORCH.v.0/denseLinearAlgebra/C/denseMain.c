

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "denseUtil.h"
#include "denseArithmetic.h"
#include "generators.h"

/*
 Dense Linear Algebra tests.

 Alex Kaiser, LBNL, 8/2010.
 */

char checkLU();
char checkQR();




int main(){

	printf("Dense Linear Algebra Tests.\n") ;

	int i;
	int numTests = 2;
	char *pass = (char *) malloc(numTests * sizeof(char));
	char allPass = 1;

	double startTime, endTime; 
	
	startTime = read_timer();
	
	pass[0] = checkLU() ;
	pass[1] = checkQR() ;
	
	endTime = read_timer(); 
	
	printf("Total time for all tests = %f seconds.\n\n", endTime - startTime); 
	
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
		fprintf(stderr, "\nTests failed!\n") ;
	
	free(pass) ; 
	
	return allPass;
}


char checkLU(){
	/*
	 Tests LU factorization.

	 Parameters:
	 int n     Matrix dimension.
	 double tolMax    Tolerance for max difference.

	 Output:
	 char pass (returned) Whether all norms are under supplied tolerances.
	 */

	int j;
	int n = 100 ;
	double tolMax = 1e-10 ;
	char pass = 1;

	double cond[3] = {10.0, 1.0 / sqrt(1e-14), 1.0 / 1e-14} ;
	int numConds = 3;
	int type = 4;
	
	double startTime, endTime; 

	matrix A = allocAndInitRandomizedMatrix(n,n) ;

	matrix L = allocAndInitZeroMatrix(n, n) ;
	matrix U = allocAndInitZeroMatrix(n, n) ;

	matrix product = allocAndInitZeroMatrix(n,n) ;

	for(j=0; j<numConds; j++){

		printf("Tests for cond = %e :\n\n", cond[j]) ;

		for(type = 0; type < 5; type++){

			getCondNumberMatrix(&A, cond[j], type) ;
			// perform the matrix factorization

			startTime = read_timer(); 
			LUFactorize(A, &L, &U) ;
			endTime = read_timer(); 
			
			printf("Elapsed time = %f seconds.\n", endTime - startTime); 

			matrixMatrixMultiply(L, U, &product) ;

			char output = 0;
			if(output){
				printf("Input matrix:\n");
				printMatrix(A);
				printf("U = \n") ;
				printMatrix(U);
				printf("L = \n") ;
				printMatrix(L);
				printMatrix(product) ;
			}


			double maxDiff = maxDiffMatrix(product, A) ;

			printf("maxDiff = %e\n", maxDiff) ;

			if(maxDiff < tolMax){
				printf("LU factorization test passed on matrix of type %d.\n\n", type) ;
			}
			else{
				printf("LU factorization test failed on matrix of type %d.\n\n", type) ;
				pass = 0;
			}
		}
	}

	if(pass)
		printf("LU factorization test passed.\n\n") ;
	else
		printf("LU factorization test failed.\n\n") ;


	// free resources
	freeMatrix(A) ;
	freeMatrix(L) ;
	freeMatrix(U) ;
	freeMatrix(product) ;

	return pass;
}


char checkQR(){
	 /*
	 Tests symmetric QR factorization for computation of eigenvalues.

	 Parameters:
	 int n     Matrix dimension.
	 double tolMax    Tolerance for max difference.
	 int numTypes   Number of matrix types to check. Maximum supported is currently 9.
	 char output    Whether to print data to standard out about input and computed eigenvalues.

	 Output:
	 char pass (returned) Whether all norms are under supplied tolerances.
	 */

	int n = 100;
	int type ;
	double tolMax = 1e-12 ;
	char pass = 1;
	int numTypes = 9 ;
	double maxDiff ;
	char output = 0;

	double startTime, endTime; 
	
	double *eigsOriginal = (double *) malloc(n * sizeof(double)) ;
	matrix A = allocAndInitZeroMatrix(n,n);

	double *eigs ;

	for(type=0; type < numTypes; type++){
		getEigenvalueTestMatrix(n, type, &A, eigsOriginal) ;

		qsort(eigsOriginal, n, sizeof(double), compareDoubles) ;

		if(output){
			printf("Input matrix:\n") ;
			printMatrix(A) ;
		}


		// compute eigenvalues
		startTime = read_timer(); 
		eigs = symmetricQR(A) ;
		endTime = read_timer(); 
		
		printf("Elapsed time = %f seconds.\n", endTime - startTime); 

		if(output){
			printf("Known eigenvalues = \n") ;
			printVector(eigsOriginal, n) ;

			printf("Computed eigenvalues = \n") ;
			printVector(eigs, n) ;
		}

		maxDiff = maxDiffVector(eigs, eigsOriginal, n) ;
		printf("maxDiff = %e\n", maxDiff);

		if (maxDiff < tolMax) {
			printf("QR factorization for eigenvalues test passed on matrix of type %d.\n\n", type);
		} else {
			printf("QR factorization for eigenvalues test failed on matrix of type %d.\n\n", type);
			pass = 0;
		}
	}

	if(pass)
		printf("QR factorization for eigenvalues test passed.\n\n") ;
	else
		printf("QR factorization for eigenvalues test failed.\n\n") ;

	free(eigs) ;
	free(eigsOriginal) ;
	freeMatrix(A) ;

	return pass;
}
