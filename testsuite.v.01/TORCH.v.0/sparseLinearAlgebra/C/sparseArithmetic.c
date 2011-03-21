
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "sparseUtil.h"
#include "sparseArithmetic.h"

/*
 Arithmetic routines for sparse matrix operations.

 Alex Kaiser, LBNL, 2010
 */



void spmv(csrMatrix A, double * x, double *y){
	/*
	 Sparse matrix vector multiply.
	 Computes y = A * x.
	 Space for product must be preallocated.

	 Input:
		csrMatrix A		Martix to multiply
		double *x		Vector to multiply by
		double *y		Their product

	 Output:
		double *y		The matrix-vector product
	 */

	int rowNum,j;
	double yTemp ;

	for( rowNum = 0; rowNum < A.m; rowNum++){
		yTemp = 0.0;
		for( j = A.rowPtr[rowNum]; j < A.rowPtr[rowNum+1]; j++){
			yTemp += A.values[j] * x[ A.columnIndices[j] ] ;
		}
		y[rowNum] = yTemp ;
	}
}


double * spts(csrMatrix A, double *b){
	/*
	 Sparse triangular solve.
	 Solves the linear system A * x = b.
	 Result is allocated in this function.

	 Input:
		csrMatrix A				Martix of system to solve
		double *b				RHS of system

	 Output:
		double * (returned)		Solution to the linear system
	 */


	double * x = (double *) malloc(A.m * sizeof(double));
	int rowNum, j;

	x[0] = b[0] / A.values[0] ;

	double sum;

	for(rowNum = 1; rowNum < A.m; rowNum++){
		sum = 0.0;

		for( j = A.rowPtr[rowNum]; j < A.rowPtr[rowNum+1] - 1; j++)
			sum += A.values[j] * x[ A.columnIndices[j] ] ;

		x[rowNum] = ( b[rowNum] - sum) / A.values[ A.rowPtr[rowNum + 1] - 1 ] ;
	}

	return x;
}


double * conjugateGradient(csrMatrix A, double *b, double *guess, double eps, int maxIt){
	/*
	 Conjugate gradient solve.
	 Solves the linear system A * x = b.
	 Result is allocated in this function.

	 Input:
		csrMatrix A				Martix of system to solve.
		double *b				RHS of system.
		double *guess			Initial guess for the solution of the system.
		double eps				Epsilon used to determine termination of the algorithm.
		int maxIt				Maximum number of iterations to perform.
									Algorithm is guaranteed to find a solution (if it exists)
									if this parameter is the dimension of the system or larger.

	 Output:
		double * (returned)		Solution to the linear system.
	 */


	if(A.m != A.n){
		fprintf(stderr, "Input matrix must be square. Terminating.\n") ;
		exit(-1);
	}

	int n = A.n;
	int j, iter;
	double rho, lastRho, alpha, beta;

	// allocate arrays
	double * x = (double *) malloc(n * sizeof(double));
	double * r = (double *) malloc(n * sizeof(double));
	double * w = (double *) malloc(n * sizeof(double));
	double * p = (double *) malloc(n * sizeof(double));

	for(j=0; j<n; j++)
		x[j] = guess[j] ;

	// calulate residual r = b - A*x ;
	spmv(A, x, r) ;
	vectorPlusConstantByVector(r, b, -1.0, r, n);

	rho = normDoublel2(r, n) ;
	rho = rho * rho ; // l2 norm squared
	lastRho = rho ; // junk initial value

	double stoppingValue = eps * normDoublel2(b, n) ;

	iter = 0;
	while( sqrt(rho) > stoppingValue ){

		iter++;

		if(iter == 1)
			for(j=0; j<n; j++)
				p[j] = r[j] ;
		else{
			beta = rho / lastRho;
			vectorPlusConstantByVector(p, r, beta, p, n) ;
		}

		spmv(A, p, w);
		alpha = rho / innerProduct(p, w, n);
		vectorPlusConstantByVector(x, x, alpha, p, n) ;
		vectorPlusConstantByVector(r, r, -alpha, w, n) ;

		lastRho = rho;
		rho = normDoublel2(r, n) ;
		rho = rho * rho ;

		if(iter >= maxIt){
	        fprintf(stderr, "cg hit max iterations. %d iterations performed.\n", iter );
	        free(r);
	        free(w);
	        free(p);
	        return x;
		}
	}

	printf("cg converged in %d iterations.\n", iter) ;
    free(r);
    free(w);
    free(p);
	return x;
}

void matrixPowers(csrMatrix A, int k, double **x){
	/*
	 Computes the following set of matrix vector products:
	 [x, A*x, ... A^(n-2) * x, A^(n-1) * x ]

	 Input:
	 csrMatrix A      Matrix to exponentiate.
	 int k            Computes powers through A^k-1
	 double **x       Input vector and results.
	                  Must be preallocated to k * A.n double precision array.
	                  x[0] = input vector.

	 Output:
	 double **x       Output vectors also stored here.
	                  x[i] = A^i * x[0].
	 */
	int i;
	for(i=1; i<k; i++)
		spmv(A, x[i-1], x[i]) ;
}


double innerProduct(double x[], double y[], int n){
	/*
	 Performs inner product of two double precision vectors.

	 Input:
		double x[]			Vector to inner product
		double y[]			Vector to inner product
		int n				Length of above vectors

	 Output:
		double (returned)	Their inner product
	 */

	int j;
	double sum = 0.0;
	for(j=0; j<n; j++)
		sum += x[j] * y[j] ;

	return sum ;
}

void vectorPlusConstantByVector(double toReturn[], double x[], double alpha, double y[], int n){

	/*
	 Vector plus constant times vector.
	 toReturn = x + alpha * y
	 No memory allocated.
	 May be performed in place, with either input vector equal to output vector

	 Input:
		double toReturn[]	Vector in which to store result
		double x[]			Vector, not multiplied by constant alpha
		double alpha		Constant coefficient for y
		double y[]			Vector to be multiplied by constant alpha
		int n				Length of vectors

	 Output:
		double toReturn[]	x + alpha * y

	 */

	int j;

	for(j=0; j<n; j++)
		toReturn[j] = x[j] + alpha * y[j] ;

}


double * denseMV(double ** dense, int m, int n, double * x){
	/*
	 Naive dense matrix vector multiply.
	 For debugging purposes only.

	 Input:
		double ** dense			Matrix to multiply
		int m					Number of rows
		int n					Number of columns
		double *x				Vector to multiply by

	 Output:
		double * (returned)		Their product

	 */

	double * y = (double *) malloc(m * sizeof(double));
	int i,j;

	for(i=0; i<m; i++){
		y[i] = 0.0 ;
		for(j=0; j<n; j++)
			y[i] += dense[i][j] * x[j] ;
	}

	return y;
}
