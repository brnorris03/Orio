
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "denseUtil.h"
#include "denseArithmetic.h"


/*
 Arithmetic routines for dense linear algebra operations.

 Alex Kaiser, LBNL, 8/2010.
 */



void LUFactorize(matrix A, matrix *L, matrix *U){
	/*
	 Perform LU factorization of matrix A.
	 Returns permutation of lower triangular matrix and upper triangular matrix such that L*U = A.

	 Input:
	 matrix A   Matrix to factor.
	 matrix L   Filler for returned matrix, must be preallocated.
	 matrix U   Filler for returned matrix, must be preallocated.

	 Output:
	 matrix L   Permutation of lower triangular factor.
	 matrix U   Upper triangular factor.
	 */


	if(A.m != A.n){
		fprintf(stderr,"Input matrix must be square. Other dimensions not supported.\n") ;
		exit(-1);
	}
	if((L->m != A.m) || (L->n != A.n)){
		fprintf(stderr, "Matrix dimensions must agree. Exiting.\n") ;
		exit(-1) ;
	}
	if((U->m != A.m) || (U->n != A.n)){
		fprintf(stderr, "Matrix dimensions must agree. Exiting.\n") ;
		exit(-1) ;
	}

	int i,j,k;
	int n = A.n ;

	// zero L
	for(i=0; i<n; i++){
		for(j=0; j<n; j++){
			L->values[i][j] = 0.0 ;
		}
	}

	// copy A for non-destructive editing
	copyMatrix(A, *U) ;




	int *pivots = (int *) malloc(n * sizeof(int)) ;

	for(i=0; i < n-1; i++){

		pivots[i] = getIndexOfMax(*U, i, i) ;

		if(i != pivots[i])
			swapRow(*U, i, pivots[i]) ;

		for(j=i+1; j < U->n; j++)
			U->values[j][i] /= U->values[i][i] ;

		// trailing matrix update
		for(j=i+1; j<n; j++){
			for(k=i+1; k<n; k++){
				U->values[j][k] -= U->values[j][i] * U->values[i][k] ;
			}
		}

	}

	pivots[n-1] = n - 1 ;

	// place the result in the two separate arrays
	// copy diagonal entries
	for(i=0; i<n; i++)
		L->values[i][i] = 1.0 ;
	// copy relevant data from U to L
	for(j=0; j < n-1; j++){
		for(i=j+1; i<n; i++){
			L->values[i][j] = U->values[i][j] ;
			U->values[i][j] = 0.0 ;
		}
	}

	// permute L to make L*U = A
	for(i=n-1; i>=0; i--){
		if(pivots[i] != i)
			swapRow(*L, pivots[i], i) ;
	}

	free(pivots) ;
}


double * symmetricQR(matrix A){
	/*
	Symmetric QR for computation of eigenvalues.

	Source: Golub and Van Loan, Matrix Computations p. 421.

	Input:
	matrix A   Square, symmetric matrix of which to compute eigenvalues

	Output:
	double *eigs (returned)   Eigenvalues of the matrix
	*/

	double *eigs = (double *) malloc(A.n * sizeof(double)) ;
	A = tridiagonalize(A) ;

	int i, k;
	int start;

	int n = A.n ;
	int maxIt = 10000;
	int p,q = 0 ;

	double tol = 1e-14 ;
	int iteration = 0 ;

	while(q < n-1){
	    for (i = 0; i<n-1; i++){
	       if ( fabs( A.values[i+1][i] ) <= tol * (fabs( A.values[i][i] ) + fabs( A.values[i+1][i+1] )) ) {
	           A.values[i+1][i] = 0.0;
	           A.values[i][i+1] = 0.0;
	       }
	    }

	    // find largest diagonal block in lower right of matrix
	    start = n-q; //loop index starts with previously found diagonal block
	    for( k = start - 1; k >= 0; k--){
	        //check off-diagonal entries
	        //only above diagonal is checked because of symmetry
	    	if(k != 0){
				if(A.values[k][k-1] == 0.0)
					q++;
				else
					break ;
	    	}
	    }

	    p = n - q - 1;
		for (k = start - 1; k > 0; k--) {
			if (A.values[k][k - 1] != 0)
				p--;
			else
				break;
		}


	    if (q < n - 1 ){
	        A = symmetricQRStep(A, p, n-q)  ;
	    }

	    iteration++ ;
	    if (iteration > maxIt){
	        fprintf(stderr, "Max it reached. Returning.\n");
	        break ;
	    }
	}


	for(i=0; i<n; i++)
		eigs[i] = A.values[i][i] ;


	qsort(eigs, n, sizeof(double), compareDoubles) ;
	return eigs ;
}



matrix symmetricQRStep(matrix A, int startIndex, int stopIndex){
	/*
	 Single step of a QR factorization. Uses givens rotations.
	 Does not explicitly store information on Q.
	 Performed on the square submatrix from i,j = startIndex ... stopIndex - 1

	 Input:
	 matrix A    Working matrix.
	 int startIndex  First index to modify.
	 int stopIndex  Upper boundary

	 Output:
	 Matrix A    Modified matrix.
	 */

	int k;
	double x, z;

    double d = 0.5 * (A.values[stopIndex-2][stopIndex-2] - A.values[stopIndex-1][stopIndex-1] ) ;

    double mu = A.values[stopIndex-1][stopIndex-1] - pow(A.values[stopIndex-1][stopIndex-2], 2) /
                        (d + sign(d) * sqrt(d*d + pow(A.values[stopIndex-1][stopIndex-2], 2))) ;

    x = A.values[startIndex][startIndex] - mu;
    z = A.values[startIndex + 1][startIndex] ;


    for(k = startIndex; k<stopIndex-1; k++){
    	A = applyGivens(A, x, z, k, k+1, startIndex, stopIndex);
        if (k < stopIndex-2){
            x = A.values[k+1][k];
            z = A.values[k+2][k];
        }
    }

    return A;
}

matrix applyGivens(matrix A, double a, double b, int i, int k, int startIndex, int stopIndex){
	/*
	 Applies givens rotation to matrix A, computing G'*A*G, and returns the
	 rotated matrix.


	 Input:
	   matrix A						Matrix to rotate.
	   double aa, bb				Specifies the angle of the Givens rotations.
	   int i, k						Modifies row and column i and k of the input matrix.
	   int startIndex, stopIndex	Processes the submatrix from index startIndex:stopIndex-1

	 Output:
	   matrix A             Rotated matrix.
	*/

	double c, s ;
	double tau1, tau2 ;


	int j;

    if(b == 0.0){
        c = 1.0;
        s = 0.0;
    }
    else{
        if(fabs(b) > fabs(a)){
            tau1 = -a/b;
            s = 1.0 / sqrt(1.0 + tau1*tau1);
            c = s*tau1;
        }
        else{
            tau1 = -b/a;
            c = 1.0 / sqrt(1.0 + tau1*tau1);
            s = c*tau1;
        }
    }

    for(j = startIndex; j<stopIndex; j++){
        tau1 = A.values[i][j];
        tau2 = A.values[k][j];

        A.values[i][j] = c*tau1 - s*tau2;
        A.values[k][j] = s*tau1 + c*tau2;
    }

    for(j = startIndex; j<stopIndex; j++){
        tau1 = A.values[j][i];
        tau2 = A.values[j][k];

        A.values[j][i] = c*tau1 - s*tau2 ;
        A.values[j][k] = s*tau1 + c*tau2 ;
    }


    return A;
}


int sign(double d){
	/*
	 Returns sign of input. Zero has sign of one.

	 Input:
		double d			Number to obtain sign of.

	 output
		int (returned)		Its sign.
	 */

	if(fabs(d) == d)
		return 1;
	else
		return -1;
}



matrix tridiagonalize(matrix A){
	/*
	Tridiagonalize a symmetric matrix using Householder transformations.
	Transformations are performed in place with Householder vectors and coefficients,
	rather than explicitly forming matrices and explicitly computing their products.

	Returns T = Q^t * A * Q

	Source: Golub and Van Loan, "Matrix Computations" p. 415

	Input:
	    Matrix A                  Matrix to tridiagonalize.

	  Output:
	    Matrix T (returned)       Tridiagonal matrix similar to input matrix.
	*/


	double *v = (double *) malloc(A.n * sizeof(double)) ;
	double *p = (double *) malloc(A.n * sizeof(double)) ;
	double *w = (double *) malloc(A.n * sizeof(double)) ;
 	double beta ;
 	double temp ;

 	double t1, t2;

 	int currentVectorLength = A.n - 1 ;
 	int n = A.n ;

 	int iVector, jVector;

	int i,j,k;
	for(k=0; k < n-2; k++){
		beta = householder(A, k+1, k, v) ;
		matrixVectorMultiply(A, k+1, k+1, v, beta, p) ;

		temp = - 0.5 * beta * innerProduct(p, v, currentVectorLength ) ;
		vectorPlusConstantByVector(w, p, temp, v, currentVectorLength ) ;

		temp = 0.0;
		for(j=k+1; j<n; j++){
			temp += A.values[j][k] * A.values[j][k] ;
		}
		temp = sqrt(temp) ;
		A.values[k+1][k] = temp ;
		A.values[k][k+1] = temp ;

		for(i=k+1, iVector=0; i<n; i++, iVector++){
			for(j=k+1, jVector=0; j<n; j++, jVector++){
				t1 = v[i]*w[j] ;
				t2 = v[j]*w[i] ;
				A.values[i][j] -= (v[iVector]*w[jVector] + v[jVector]*w[iVector]) ;
			}
		}

		currentVectorLength-- ;
	}

	// zero off diagonal multipliers
	for(i=0; i<n-1; i++){
		for(j=i+2; j<n; j++){
			A.values[i][j] = 0.0 ;
			A.values[j][i] = 0.0 ;
		}
	}

	free(v) ;
	free(p) ;
	free(w) ;

	return A;
}



double householder(matrix A, int startIndex, int columnNum, double *v){
	/*
	 Computes the Householder Vector for use with QR algorithm for computing
	 eigenvalues.

	 Upon completion, the following conditions hold about the matrix P, which
	 is not explicitly formed:

		P = I - Beta*v*v', where P is a Householder reflection matrix
		P is orthogonal, v(1) = 1,
		Px = ||x|| * e_1
		where ||x|| is the Euclidean norm, and e_1 is the first
		element of the standard basis in R_n, [1,0 ... 0]

	 Source: Golub and Van Loan, "Matrix Computations" p. 210


	 Input:
	 matrix A   Relevant matrix.
	 int startIndex  Index of column at which to start modifying vector.
	 int columnNum  Column to modify.
	 double *v  Preallocated space for Householder vector.

	 Output:
	 double beta (returned)  Coefficient for Householder vector.
	 double *v  Householder vector.
	 */

	int i, vIndex;
	double beta, mu, temp;


	v[0] = 1.0 ;
	double sigma = 0.0;

	for(i=startIndex + 1, vIndex=1; i < A.n; i++, vIndex++){
		sigma += A.values[i][columnNum] * A.values[i][columnNum] ;
		v[vIndex] = A.values[i][columnNum] ;
	}

	if(sigma == 0.0)
		beta = 0.0 ;
	else{
		mu = sqrt( A.values[startIndex][columnNum]*A.values[startIndex][columnNum] + sigma ) ;
		if(A.values[startIndex][columnNum] <= 0)
			v[0] = A.values[startIndex][columnNum] - mu ;
		else{
			v[0] = -sigma / (A.values[startIndex][columnNum] + mu) ;
		}
		beta = 2.0 * v[0]*v[0] / (sigma + v[0]*v[0]) ;

		temp = 1.0 / v[0] ;
		for(i=0; i < (A.n - startIndex); i++)
			v[i] *= temp ;

	}

	return beta ;
}



void matrixVectorMultiply(matrix A, int startIndexRow, int startIndexColumn, double *v, double constant, double *out){
	/*
	 Matrix vector multiply.
	 Computes
	     out = constant * A * v ;
	 Allows for multiplication by subarray.


	 Input:
	 matrix A   Matrix to multiply.
	 int startIndexRow   First row index.
	 int startIndexColumn   First column index.
	 double *v   Vector to multiply.
	 double constant   Coefficient.
	 double *out    Output vector, preallocated.

	 Output:
	 double *out     constant * A * v
	 */

	int i,j ;
	int rowNum, colNum;

	for(i=startIndexRow, rowNum=0; i<A.m; i++, rowNum++){
		out[rowNum] = 0.0 ;
		for(j=startIndexColumn, colNum=0; j<A.n; j++, colNum++){
			out[rowNum] += constant * A.values[i][j] * v[colNum] ;
		}
	}

}

void scalarVectorProduct(double alpha, double x[], int n, double toReturn[]){
	/*
	 Performs scalar-vector product

	 Input:
	    double alpha        Scalar to product
		double x[]			Vector to product
		int n				Length of above vectors
		double toReturn[]   Their element-wise product

	 Output:
		double toReturn[] 	Scalar-vector product
	 */

	int j;
	for(j=0; j<n; j++)
		toReturn[j] = alpha * x[j] ;
}


void elementWiseVectorProduct(double x[], double y[], int n, double toReturn[]){
	/*
	 Performs element-wise product of two double precision vectors.

	 Input:
		double x[]			Vector to product
		double y[]			Vector to product
		int n				Length of above vectors
		double toReturn[]   Their element-wise product

	 Output:
		double toReturn[] 	Their element-wise product
	 */

	int j;
	for(j=0; j<n; j++)
		toReturn[j] = x[j] * y[j] ;
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

void outerProduct(matrix *toReturn, double *column, int lengthCol, double *row, int lengthRow){
	/*
	 Vector outer product.
	 toReturn = column * row

	 Input:
	 matrix *toReturn   Outer product of input. Must be preallocated.
	 double *column     Column vector for product.
	 int lengthCol      Column length.
	 double *row        Row vector for product.
	 int lengthRow      Row length.

	 Output:
	 matrix *toReturn   Result of outer product.
	 */

	if((toReturn->m != lengthCol) || (toReturn->n != lengthRow) ){
		fprintf(stderr, "Matrix dimensions must agree with vector lengths for outer product. Exiting.\n") ;
		exit(-1) ;
	}

	int i,j;

	for(i=0; i<lengthCol; i++){
		for(j=0; j<lengthRow; j++){
			toReturn->values[i][j] = column[i] * row[j] ;
		}
	}
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


void matrixPlusConstantByMatrix(matrix *toReturn, matrix A, double alpha, matrix B){

	/*
	 Matrix plus constant times matrix.
	 toReturn = A + alpha * B
	 No memory allocated.
	 May be performed in place, with either input matrix equal to output matrix.
	 All matrices must be the same dimension.

	 Input:
		matrix *toReturn	Matrix in which to store result
		matrix A			Matrix, not multiplied by constant alpha
		double alpha		Constant coefficient for y
		matrix y			Matrix to be multiplied by constant alpha

	 Output:
		matrix *toReturn	A + alpha * B
	 */

	if((toReturn->m != A.m) || (toReturn->n != A.n)){
		fprintf(stderr, "Matrix dimensions must agree. Exiting.\n") ;
		exit(-1) ;
	}
	if((toReturn->m != B.m) || (toReturn->n != B.n)){
		fprintf(stderr, "Matrix dimensions must agree. Exiting.\n") ;
		exit(-1) ;
	}

	int i,j;

	for(i=0; i<A.m; i++){
		for(j=0; j<A.n; j++){
			toReturn->values[i][j] = A.values[i][j] + alpha * B.values[i][j] ;
		}
	}
}


void matrixMatrixMultiply(matrix A, matrix B, matrix *res){
	/*
	 Naive matrix matrix multiply.
	 Not blocked and thus low performing.
	 Resizes output matrix if needed.
	 For debugging use.

	 Input:
	 matrix A, B   Matrices to multiply

	 Output:
	 matrix *res   Matrix product, res = A*B ;
	 */


	if( (res->m != A.m) || (res->n != B.n) ){
		printf("Resizing output matrix.\n") ;
		freeMatrix(*res) ;
		*res = allocAndInitZeroMatrix(A.m, B.n) ;
	}

	if(A.n != B.m){
		fprintf(stderr, "Matrix dimensions must agree. Exiting.\n");
		exit(-1);
	}

	int m,n,p ;
	m = A.m ;
	n = A.n ;
	p = B.n ;


	int i,j,k;

	double temp = 0.0;

	for(i=0; i < m; i++){
		for(j=0; j < p; j++){
			res->values[i][j] = 0.0 ;
			for(k=0; k < n; k++){
				res->values[i][j] += A.values[i][k] * B.values[k][j] ;
				temp = res->values[i][j] ;
			}
		}
	}

}

void transpose(matrix in, matrix *out){
	/*
	 Naive, out of place, matrix transpose.
	 Matrices must be preallocated.

	 Input:
	 matrix In  Matrix to transpose.
	 matrix *Out  Result, must be preallocated.

	 Output:
	 matrix *out  Transpose of original matrix.
	 */

	int i,j ;

	if( (in.m != out->n) || (in.n != out->m)){
		fprintf(stderr, "Matrix dimensions must match for transpose. Exiting.\n") ;
		exit(-1) ;
	}

	for(i=0; i<in.m; i++){
		for(j=0; j<in.n; j++){
			out->values[j][i] = in.values[i][j] ;
		}
	}

}




