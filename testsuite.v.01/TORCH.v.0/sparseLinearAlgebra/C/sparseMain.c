#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "sparseUtil.h"
#include "sparseArithmetic.h"
#include "generators.h"


/*
 
 Spare matrix tests. 
 Generates matrices and performs arithmetic tests as described below. 
 
 
 Alex Kaiser, LBNL, 7/2010
 
 */ 


// csr matrix structure
// copy struct definition for reference
/*
typedef struct csrFiller{
	int m;
	int n;
	int nnz;
	int *rowPtr;
	int *columnIndices;
	double *values ;
} csrMatrix;
*/


// headers 

// Solver verification routines. 
char checkSpmv();
char checkSpts();
char checkCG();



int main(){

	
	int i;
	int numTests = 3;
	char *pass = (char *) malloc(numTests * sizeof(char));
	char allPass = 1;
	
	
	printf("Beginning Sparse Linear Algebra tests.\n\n") ;
	
	double startTime, endTime; 
	startTime = read_timer_local(); 
	
	pass[0] = checkSpmv(); 
	pass[1] = checkSpts(); 
	pass[2] = checkCG(); 
	
	endTime = read_timer_local(); 
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
		printf("\nAll Sparse Linear Algebra tests passed.\n") ;
	else
		fprintf(stderr, "\nAt least one Sparse Linear Algebra test failed!\n") ;
	
	printf("\nEnd of Sparse Linear Algebra tests.\n\n\n");
	
	return allPass ; 
}




char checkSpmv(){
	/*
	 See Structured Grid for more rigorous verification of SpMV kernels. 
 
	 Checks sparse matrix vector multiplication for basic correctness.
	 Generates random sparse matrix and multiplies, then compares
		results to dense matrix version of same.
	 Not a full or true verification scheme because verification is more
		computationally intensive than kernel in question, but usefull
		for basic correctness of computation.
	 
	 
	 Parameters:

		int n						Size of matrix, select such that (n mod(10) == 1) to ensure that corner of matrix isn't left unfilled
		int nnz						Approximate number of nonzeros. Actual number may be slightly lower because of possibility of duplicate entries
										which are removed in generator
		double distribution[10]		Matrix is divided into ten bands, each including approximately distribution[i]*nnz nonzero entires
		double tol					Relative error must be less that this value or test will return failure

	 Output:
		char (returned)      If relative error is less than specified tolerance
	 */

	printf("Testing Sparse Matrix Vector Multiply.\n");
	
	int n = 1001;
	int nnz = (int) ceil(n * n * 0.25) ;
	double distribution[10] = {0.658, 0.114, 0.0584, 0.0684, 0.0285, 0.0186, 0.0144, 0.0271, 0.00774, 0.00387} ;
	csrMatrix A = getSymmetricDiagonallyDominantCSR(n, nnz, distribution) ;

	double *x = (double *) malloc(n * sizeof(double)) ;
	double *resSparse = (double *) malloc(n * sizeof(double)) ;

	double tol = 1e-10;

	double startTime, endTime; 
	startTime = read_timer_local(); 
	
	spmv(A, x, resSparse) ;

	endTime = read_timer_local(); 
	printf("Elapsed time = %f seconds.\n", endTime - startTime) ;
	
	double ** dense = toDense(A) ;
	double *resDense = (double *) malloc(n * sizeof(double)) ;
	resDense = denseMV(dense, n, n, x);

	double relErr = l2RelativeErrDouble_local(resSparse, resDense, n) ;

	printf("relative error = %.14f\n", relErr) ;

	// free resources
	free(x) ;
	free(resSparse) ;
	freeDouble2d(dense, n, n) ;
	free(resDense) ;


	if(relErr < tol){
		printf("SpMV test passed.\n\n\n");
		return 1;
	}
	else{
		fprintf(stderr, "SpMV test failed.\n\n");
		return 0;
	}
}



char checkSpts(){
	/*
	 Verification routine for sparse triangular solve routine.
		- Generates sparse lower triangular matrix according to parameters
		- Randomly generates solution to linear system
		- Uses matrix multiply to generate right hand side
		- Solves system and compares for errors with L2 norm
	 
	 Parameters

		int n						Size of matrix, select such that (n mod(10) == 1) to ensure that corner of matrix isn't left unfilled
		int nnz						Approximate number of non-zeros. Actual number may be slightly lower because of possibility of duplicate entries
										which are removed in generator
		double distribution[10]		Matrix is divided into ten bands, each including approximately distribution[i]*nnz nonzero entires
		double tol					Relative error must be less that this value or test will return failure 
	 */ 

	
	printf("Testing Sparse Triangular solve.\n");
	
	int n = 1001;
	int nnz = (int) ceil(n * n * 0.2) ;
	double distribution[10] = {0.658, 0.114, 0.0584, 0.0684, 0.0285, 0.0186, 0.0144, 0.0271, 0.00774, 0.00387} ;
	csrMatrix A = getStructuredLowerTriangularCSR(n, nnz, distribution) ;

	// view structure of matrix if desired
	// spyPrimitive(toDense(A), A.m, A.n) ;


	int j;

	double *solution = (double *) malloc(n * sizeof(double)) ;
	double *x ;  // allocated in spts function
	double *b = (double *) malloc(n * sizeof(double)) ;

	// generate the solution to the linear system
	for(j=0; j<n; j++)
		solution[j] = bcnrand_local() ;
	
	// multiply by the matrix to get the RHS
	spmv(A, solution, b) ;

	double startTime, endTime; 
	startTime = read_timer_local();
	
	// solve the system, solution memory allocated in spts function
	x = spts(A, b) ;

	endTime = read_timer_local(); 
	printf("Elapsed time = %f seconds.\n", endTime - startTime) ;
	
	double relErr = l2RelativeErrDouble_local(x, solution, n) ;
	double tol = 10e-10;

	printf("relative error with original solution = %.14f\n", relErr) ;

	// free resources
	free(solution);
	free(b);
	freeCSRMatrixArrays(A);
	free(x); //allocated in spts function


	if(relErr < tol){
		printf("SpTS test passed.\n\n\n");
		return 1;
	}
	else{
		fprintf(stderr, "SpTS test failed.\n\n\n");
		return 0;
	}
}


char checkCG(){
	/*
	 Verification rountine for sparse triangular solve routine. 
	 - Generates sparse symmetic diagonally-dominant matrix according to parameters
	 - Randomly generates solution to linear system
	 - Uses matrix multiply to generate right hand side
	 - Solves system and compares for errors with L2 norm
	 
	 Parameters
	 
		int n						Size of matrix, select such that (n mod(10) == 1) to ensure that corner of matrix isn't left unfilled
		int nnz						Approximate number of nonzeros. Actual number may be slightly lower because of possibility of duplicate entries
											which are removed in generator
		double distribution[10]		Matrix is divided into ten bands, each including approximately distribution[i]*nnz nonzero entires
		double tol					Relative error must be less that this value or test will return failure
	 */ 
	
	printf("Testing Conjugate Gradient solve.\n");
	
	int n = 1001;
	int nnz = (int) ceil(n * n * 0.2) ;
	double distribution[10] = {0.658, 0.114, 0.0584, 0.0684, 0.0285, 0.0186, 0.0144, 0.0271, 0.00774, 0.00387} ;

	csrMatrix A = getSymmetricDiagonallyDominantCSR(n, nnz, distribution) ;

	int j;

	double *solution = (double *) malloc(n * sizeof(double)) ;
	double *x ; //= (double *) malloc(n * sizeof(double)) ; // allocated in function
	double *b = (double *) malloc(n * sizeof(double)) ;

	// generate a zero initial guess
	double *guess = (double *) malloc(n * sizeof(double)) ;
	for(j=0; j<n; j++)
		guess[j] = 0.0 ;

	// generate the solution to the linear system
	for(j=0; j<n; j++)
		solution[j] = bcnrand_local() ;

	// multiply by the matrix to get the RHS
	spmv(A, solution, b) ;


	// solve the system, solution memory allocated in CG function
	double eps = 10e-12;
	int maxIt = n;
	
	double startTime, endTime; 
	startTime = read_timer_local();
	
	x = conjugateGradient(A, b, guess, eps, maxIt) ;

	endTime = read_timer_local(); 
	printf("Elapsed time = %f seconds.\n", endTime - startTime) ;
	
	double relErr = l2RelativeErrDouble_local(x, solution, n) ;
	double tol = 10e-10;

	printf("relative error with original solution = %.14f\n", relErr) ;

	// free resources
	free(solution);
	free(b);
	free(guess) ;
	freeCSRMatrixArrays(A);
	free(x); //allocated in CG function

	if(relErr < tol){
		printf("CG test passed.\n\n\n");
		return 1;
	}
	else{
		fprintf(stderr, "CG test failed.\n\n\n");
		return 0;
	}
}


