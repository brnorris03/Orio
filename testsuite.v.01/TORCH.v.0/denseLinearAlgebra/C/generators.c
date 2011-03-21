

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "denseUtil.h"
#include "denseArithmetic.h"

/*
 Generators for dense linear algebra operations.

 Alex Kaiser, LBNL, 8/2010.
 */


matrix get1_2_1Tridiagonal(int size){
	/*
	 Returns a 1-2-1 tri-diagonal matrix of the specified size.

	 Intput:
	 int size  Matrix dimension.

	 Output:
	 matrix (returned)   size x size 1-2-1 tri-diagonal matrix.
	 */

	matrix A = allocAndInitZeroMatrix(size, size) ;

	int i;
	for(i=0; i<size-1; i++){
		A.values[i][i] = 2.0 ;
		A.values[i+1][i] = 1.0 ;
		A.values[i][i+1] = 1.0 ;
	}
	A.values[size-1][size-1] = 2.0 ;

	return A;
}

void getCondNumberMatrix(matrix *A, double cond, int type){
	/*
	Modifies randomized matrix following a simplified version of the
	LAPACK generator function 'DLATMR'

	Off diagonal entries not modified and should be passed in randomized.
	Diagonal entries are specified from the parameters 'cond' and 'type'

	Source:
	LAPACK Working Note 9
	A Test Matrix Generation Suite
	James Demmel, Alan McKenney
	http://www.netlib.org/lapack/lawnspdf/lawn09.pdf

	Input:
	matrix A              Randomized matrix. Diagonal is overwritten. Must be square.
	double cond           Parameter to influence condition number.
	int type              Specifies the distribution of entries on the matrix diagonal.
							Suppose D is an array of diagonal entries.

							Type 0:
								D(1) = 1
								D(2:n) = 1 / cond

							Type 1:
								D(1:n-1) = 1
								D(n) = 1 / cond

							Type 2:
								D(i) form a geometric sequence from 1 to 1/cond

							Type 3:
								D(i) form an arithmetic sequence from 1 to 1/cond

							Type 4:
								D(i) are random from the same distribution
									as the off diagonal entries.
								This is default if type 1-4 are not selected.


	Output:
	matrix A              Input matrix with diagonal entries overwritten.
	 */

	if(A->n != A->m){
		fprintf(stderr, "Input matrix must be square. Exiting.\n");
		exit(-1);
	}

	int i;
	double temp;

	int n = A->n ;


	if(type == 0){
		A->values[0][0] = 1.0 ;
		for(i=1; i<n; i++)
			A->values[i][i] = 1.0 / cond ;
	}
	else if(type == 1){
		for(i=0; i<n-1; i++)
			A->values[i][i] = 1.0 ;
		A->values[n-1][n-1] = 1.0 / cond ;
	}
	else if(type == 2){
		temp = pow( (1.0 / cond) , 1.0 / (double) (n-1) ) ; // 1 / n-1st root for constructing geometric sequences.
		A->values[0][0] =1.0 ;
		for(i=1; i<n; i++)
			A->values[i][i] = temp * A->values[i-1][i-1] ;
	}
	else if(type == 3){
		temp = -(1.0 - (1.0/cond)) / (double) (n-1);
		A->values[0][0] =1.0 ;
		for(i=1; i<n; i++)
			A->values[i][i] = temp + A->values[i-1][i-1] ;
	}
	else{
		for(i=0; i<n; i++)
			A->values[i][i] = bcnrand();
	}

}


matrix getRandomOrthogonalMatrix(int n){
	/*
	 Randomly generates an orthogonal matrix Q.

	 Method is as follows:
	 Peform a series of householder transformations on a random symmetric matrix.
     Calculate their product, which is then a randomized orthogonal martix.
     This method is preferred for transparency and efficiency and thus default.
     This loosely follows the scheme of "The Efficient Generation of Randomized
     Orthogonal Matrices with an Application to Condition Estimators"
     R. W. Stewart, SIAM Journal on Numerical Analysis, Vol 17.

     Input:
     int n   Size of matrix.

     Output:
     matrix A (returned)   Randomized orthogonal matrix.
	 */

	int i,j,k;
	int subI, subJ;

	matrix T = allocAndInitZeroMatrix(n, n) ;

	// initialize a random symmetric matrix
	for(i=0; i<n; i++)
		T.values[i][i] = bcnrand();

	for(i=0; i < n-1; i++){
		for(j=i+1; j < n; j++){
			T.values[i][j] = bcnrand();
			T.values[j][i] = T.values[i][j]  ;
		}
	}

	double *betaArray = (double *) malloc((n - 1 ) * sizeof(double)) ;

	// allocate non-rectangular 2D array for Householder vectors
	double **houseHolderVectors = (double **) malloc((n-1) * sizeof(double *)) ;
	j = n; // length of current vector
	for(i=0; i < n-1; i++){
		houseHolderVectors[i] = (double *) malloc(j * sizeof(double)) ;  // Seg fault city! be careful!
		j--;
	}

	for(i=0; i<n-1; i++)
		betaArray[i] = householder(T, i, i, houseHolderVectors[i]) ;

	matrix identity = allocAndInitZeroMatrix(n,n) ;
	for(i=0; i<n; i++)
		identity.values[i][i] = 1.0 ;

	matrix Q = allocAndInitZeroMatrix(n,n) ;
	for(i=0; i<n; i++)
		Q.values[i][i] = 1.0 ;

	matrix outerProductRes = allocAndInitZeroMatrix(n,n) ;
	matrix temp = allocAndInitZeroMatrix(n,n) ;
	matrix temp2 = allocAndInitZeroMatrix(n,n) ;


	outerProductRes.m = 2;
	outerProductRes.n = 2;
	identity.m = 2;
	identity.n = 2;
	temp.m = 2;
	temp.n = 2;
	temp2.m = 2;
	temp2.n = 2;

	for(i = n-3; i >= 0; i--){

		// book keeping for matrix sizes
		outerProductRes.m++;
		outerProductRes.n++;
		identity.m++;
		identity.n++;
		temp.m++;
		temp.n++;
		temp2.m++;
		temp2.n++;

		// copy submatrix to temp array
		for(k = i, subI = 0; k < n; k++, subI++){
			for(j = i, subJ = 0; j < n; j++, subJ++){
				temp.values[subI][subJ] = Q.values[k][j] ;
			}
		}

		outerProduct(&outerProductRes, houseHolderVectors[i], n-i, houseHolderVectors[i], n-i) ;
		matrixPlusConstantByMatrix(&outerProductRes, identity, -betaArray[i], outerProductRes) ;
		matrixMatrixMultiply(outerProductRes, temp, &temp2) ;

		// copy temp array back to submatrix
		for(k = i, subI = 0; k < n; k++, subI++){
			for(j = i, subJ = 0; j < n; j++, subJ++){
				Q.values[k][j] = temp2.values[subI][subJ] ;
			}
		}


	}

	// free resources
	free(betaArray) ;
	freeMatrix(outerProductRes) ;
	freeMatrix(temp) ;
	freeMatrix(temp2) ;
	// free non-rectangular array for householder vectors
	for(i=0; i < n-1; i++){
		free(houseHolderVectors[i]) ;
	}
	free(houseHolderVectors) ;


	return Q ;
}

void getEigenvalueTestMatrix(int n, int type, matrix *A, double *eigs){
	/*
	 Generates randomized matrices with specified eigenvalue distribution.
	 First, generates eigenvalues and stores them in a diagonal matrix D.
	 Generates a randomized orthogonal matrix Q, then returns the product
	   a = Q' * D * Q.

	 Source: http://www.netlib.org/lapack/lawnspdf/lawn182.pdf

	 Input:
	 int n               Dimension of matrix returned.
	 int t               Type number.
	 double *eigs        Preallocated space for eigenvalues.

	 Output:
	 matrix A            Randomized matrix with known eigenvalues.
	 double *eigs        Eigenvalues of matrix A.
	 */


	matrix D = allocAndInitZeroMatrix(n,n) ;
	matrix Q = getRandomOrthogonalMatrix(n) ;
	matrix QTranspose = allocAndInitZeroMatrix(n,n) ;

	transpose(Q, &QTranspose) ;

	matrix tempProduct = allocAndInitZeroMatrix(n,n) ;


	double ulp = 1.0e-15; 
	double k;
	int i;


	if(type == 0){
		k = 1.0 / ulp ;
		eigs[0] = 1.0 ;
		for(i=1; i<n; i++)
			eigs[i] = 1.0 / k ;
	}

	else if(type == 1){
		k = 1.0 / ulp ;
		for(i=0; i<n-1; i++)
			eigs[i] = 1.0 ;

		eigs[n-1] = 1.0 / k ;
	}

	else if(type == 2){
		k = 1.0 / ulp ;
		for(i=1; i<=n; i++)
			eigs[i-1] = pow(k, (double) -(i-1) / (double) (n-1) ) ;
	}

	else if(type == 3){
		k = 1.0 / ulp ;
		for(i=1; i<=n; i++)
			eigs[i-1] = 1.0 - ( (double) (i-1) / (double) (n-1)) * (1.0 - 1.0 / (double) k) ;
	}

	else if(type == 4){
		for(i=0; i<n; i++){
			eigs[i] = log(ulp) * bcnrand(); //  n random numbers, unif in [log(ulp), 0)
			eigs[i] = exp( eigs[i] ) ;  // their exponentials, which are dist on (1/k, 1)
		}
	}

	else if(type == 5){
		for(i=0; i<n; i++){
			eigs[i] = bcnrand() ;
		}
	}

	else if(type == 6){
		for(i=1; i <= n-1; i++)
			eigs[i-1] = ulp * i ;
		eigs[n-1] = 1.0 ;
	}

	else if(type == 7){
		eigs[0] = ulp ;
		for(i=2; i < n-1; i++)
			eigs[i-1] = 1.0 + sqrt(ulp) * i ;
		eigs[n-1] = 2.0 ;
	}

	else if(type == 8){
		eigs[0] = 1.0 ;
		for(i=0; i<n; i++)
			eigs[i] = eigs[i-1] + 100.0 * ulp ;
	}

	else{
		fprintf(stderr, "Unsupported type. Exiting.\n") ;
		exit(-1);
	}


	for(i=0; i<n; i++)
		D.values[i][i] = eigs[i] ;


	// compute A = Q' * D * Q
	matrixMatrixMultiply(QTranspose, D, &tempProduct) ;
	matrixMatrixMultiply(tempProduct, Q, A) ;

	freeMatrix(Q) ;
	freeMatrix(D) ;
	freeMatrix(QTranspose) ;
	freeMatrix(tempProduct) ;
}
