
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include "denseUtil.h"


/*
 Utilities for dense linear algebra operations.

 Alex Kaiser, LBNL, 8/2010.
 */


double ** allocDouble2d(int m, int n){
	/*
	 Returns ptr to and (m by n) array of double precision real

	 Input:
	 int m,n                  Matrix dimensions.

	 Output:
	 double ** (returned)     m by n double precision array
	 */

	double ** temp = (double **) malloc( m * sizeof(double *) ) ;
	int j;
	for(j=0; j<m; j++){
		temp[j] = (double *) malloc( n * sizeof(double) ) ;
	}
	return temp;
}


void freeDouble2d(double **z, int m, int n){
	/*
	 Frees (m by n) array of double precision reals.

	 Input:
	 double **z               Array to free.
	 int m,n                  Matrix dimensions.
	 */
	int j;
	for(j=0; j<m; j++){
		free( z[j] );
	}
	free(z) ;
}

void freeMatrix(matrix A){
	/*
	 Frees the associated arrays of a double precision matrix.

	 Input:
	 matrix A  Matrix to free.
	 */

	freeDouble2d(A.values, A.m, A.n) ;
}

matrix allocAndInitZeroMatrix(int m, int n){
	/*
	 Allocates and zeros matrix to specified dimensions.

	 Input:
	 int m Number of rows.
	 int n Number of columns.

	 Output:
	 matrix (returned)  Double precision zero matrix.
	*/

	matrix A ;
	A.m = m;
	A.n = n;

	A.values = allocDouble2d(m,n) ;

	int i, j;
	for(i=0; i<m; i++){
		for(j=0; j<n; j++){
			A.values[i][j] = 0.0 ;
		}
	}

	return A;
}

matrix allocAndInitRandomizedMatrix(int m, int n){
	/*
	 Allocates and initializes randomized matrix to specified dimensions.
	 Uses random (0,1) generator.

	 Input:
	 int m Number of rows.
	 int n Number of columns.

	 Output:
	 matrix (returned)  Double precision zero matrix.
	*/

	matrix A ;
	A.m = m;
	A.n = n;

	A.values = allocDouble2d(m,n) ;

	int i, j;
	for(i=0; i<m; i++){
		for(j=0; j<n; j++){
			A.values[i][j] = bcnrand() ;
		}
	}

	return A;
}


double read_timer( ){
	/*
	 Timer. 
	 Returns elapsed time since initialization as a double. 
	 
	 Output:
	 double (returned)   Elapsed time. 
	 */ 
	
    static char initialized = 0;
    static struct timeval start;
    struct timeval end;
	
    if( !initialized ){
        gettimeofday( &start, NULL );
        initialized = 1;
    }
	
    gettimeofday( &end, NULL );
    return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
}


void swapRow(matrix A, int first, int second){
	/*
	 Swaps two rows of matrix A.

	 Input:
	 martix A  Matrix to swap.

	 Output:
	 int i Row to swap.
	 int j Other row to swap.
	 */

	int i;
	double temp;

	for(i=0; i < A.n; i++){
		temp = A.values[first][i] ;
		A.values[first][i] = A.values[second][i] ;
		A.values[second][i] = temp ;
	}
}

int compareDoubles(const void *x, const void *y){
	/*
	 Comparator function for doubles for built in library function qsort.
	 Both input arrays will be cast to double.

	 Input:
	 const void *x  Address of first double to compare
	 const void *y  Address of second double to compare

	 Output:
	 int (returned)   1 if  x > y
	                 -1 if  x < y
	                  0 if x == y
	 */
	double *xCopy = (double *) x;
	double *yCopy = (double *) y;


	if(*xCopy > *yCopy)
		return 1;
	if(*xCopy < *yCopy)
		return -1;

	return 0;
}

void copyMatrix(matrix in, matrix out){
	/*
	 Copies entries of a matrix.
	 Both matrices must be the same dimension.

	 Input:
	 matrix in   Input matrix.
	 matrix out  Output matrix.

	 Output:
	 matrix out  Output matrix.
	 */

	int i,j;

	for(i=0; i < in.m; i++){
		for(j=0; j < in.n; j++){
			out.values[i][j] = in.values[i][j] ;
		}
	}

}

void printMatrix(matrix A){
	/*
	 Prints matrix to stdout.

	 Input:
		matrix A		Matrix to print.
	 */

	int j,k;
	for(j=0; j<A.m; j++){
		for(k=0; k<A.n; k++){
				printf("%f ", A.values[j][k]);
		}
		printf("\n");
	}
	printf("\n");
}


double maxDiffMatrix(matrix guess, matrix true){
	/*
	 Returns the maximum difference in absolute value between two matrices.

	 Input:
	 matrix guess    First matrix to compare.
	 matrix true     Second matrix to compare.

	 Output:
	 double (returned)  Their maximum difference in absolute value.
	 */

	int i,j;

	double maxFound = 0.0;
	double currentMax ;

	for(i=0; i < guess.m; i++){
		for(j=0; j < guess.n; j++){
			currentMax = fabs( guess.values[i][j] - true.values[i][j]) ;
			if(currentMax > maxFound)
				maxFound = currentMax ;
		}
	}

	return maxFound ;
}

double maxDiffVector(double *guess, double *true, int n){
	/*
	 Returns the maximum difference in absolute value between two double precision arrays.

	 Input:
	 double *guess    First array to compare.
	 double *true     Second array to compare.

	 Output:
	 double (returned)  Their maximum difference in absolute value.
	 */

	int i;

	double maxFound = 0.0;
	double currentMax ;

	for(i=0; i < n; i++){
		currentMax = fabs( guess[i] - true[i]) ;
		if(currentMax > maxFound)
			maxFound = currentMax ;
	}

	return maxFound ;
}

int getIndexOfMax(matrix A, int startIndex, int columnNum){
	/*
	 Finds the index of the maximum value below the specified index in the specified column.

	 For use with pivoting in LU factorization.

	 Input:
	 matrix A  Matrix to find the max index of.
	 int startIndex  First index to search.
	 int columnNum   Which column to search.

	 Output:
	 int (returned)   Index of the maximum value.
	 */

	int indexOfMax = startIndex ;

	int k;
	for(k=startIndex+1; k<A.n; k++){
		if( fabs(A.values[k][columnNum]) > fabs(A.values[indexOfMax][columnNum]))
			indexOfMax = k ;
	}

	return indexOfMax;
}

void printVector(double *z, int n){
	/*
	 Prints double array to stdout.

	 Input:
		double * z			Array to print.
		int n				Length.
	 */

	int i;
	for(i=0; i<n; i++)
		printf("%e ", z[i]) ;

	printf("\n");
}

double bcnrand( ){
	/*
	 This routine generates a sequence of IEEE 64-bit floating-point pseudorandom
	 numbers in the range (0, 1), based on the recently discovered class of
	 normal numbers described in the paper "Random Generators and Normal Numbers"
	 by David H. Bailey and Richard Crandall, available at http://www.nersc.gov/~dhbailey.
	 
	 Internal, private variables are allocated as static and should not be modified by the user.
	 User should simply call the routine for each random number desired. 
	 
	 Parameter: 
	 static double startingIndex       Users may modify this parameter to control where in the sequence the generator starts. 
	 This facility is useful for parallel implementations of this generator
	 to ensure that each processor gets the correct portion of the sequence.  
	 */
	
	static char initialized = 0;
	
	// variables
	static double d1, d2, d3;
	static dd_real dd1, dd2, dd3;
	
	// constants
	static double threeToTheThirtyThree ;
	static double reciprocalOfThreeToThirtyThree ;
	static double twoToTheFiftyThree ;
	
	static double startingIndex ;
	
	static unsigned long long int n;
	
	if( !initialized ){
		
		n = 0;
		startingIndex = pow(3.0, 33) + 10000.0 ;
		
		// parameters
		threeToTheThirtyThree = pow(3, 33) ;
		reciprocalOfThreeToThirtyThree = 1.0 / threeToTheThirtyThree ;
		twoToTheFiftyThree = pow(2, 53);
		
		// check inputs
		if( (startingIndex < threeToTheThirtyThree + 100) || (startingIndex > twoToTheFiftyThree) ){
			fprintf(stderr, "bcnrand error: startingIndex must satisfy 3^33 + 100 <= startingIndex <= 2^53\nstartingIndex = %.15f", startingIndex);
			exit(-1) ;
		}
		
		// calculate starting element
		d2 = expm2(startingIndex - threeToTheThirtyThree, threeToTheThirtyThree) ;
		d3 = trunc(0.5 * threeToTheThirtyThree) ;
		dd1 = ddMult(d2, d3) ;
		d1 = trunc(reciprocalOfThreeToThirtyThree * dd1.x[0]) ;
		dd2 = ddMult(d1, threeToTheThirtyThree) ;
		dd3 = ddSub(dd1, dd2) ;
		d1 = dd3.x[0] ;
		
		if(d1 < 0.0)
			d1 += threeToTheThirtyThree ;
		
		initialized = 1;
	}
	
	n++ ;
	if( n > (2.0 * threeToTheThirtyThree / 3.0) ){
		fprintf(stderr, "bcnrand warning: number of elements exceeds period.\nn = %llu\nperiod = %f", n, 2.0 * twoToTheFiftyThree / 3.0) ;
	}
	
	// calculate next element of sequence
	dd1.x[0] = twoToTheFiftyThree * d1 ;
	dd1.x[1] = 0.0;
	d2 = trunc(twoToTheFiftyThree * d1 / threeToTheThirtyThree) ;
	dd2 = ddMult(threeToTheThirtyThree, d2) ;
	dd3 = ddSub(dd1, dd2);
	d1 = dd3.x[0];
	
	if(d1 < 0.0)
		d1 += threeToTheThirtyThree ;
	
	return reciprocalOfThreeToThirtyThree * d1 ;
}

double expm2(double p, double modulus){
	/*
	 Returns 2^p mod am.  This routine uses a left-to-right binary scheme.
	 
	 Input:
	 double p            Exponent.
	 double modulus      Modulus.
	 
	 Output:
	 double (returned) 2^p mod am
	 */
	
	double reciprocalOfModulus;
	double d2;
	dd_real dd1, dd2, dd3, ddModulus;
	double p1, pt1, r;
	
	
	double twoToTheFiftyThree = pow(2, 53);
	
	reciprocalOfModulus = 1.0 / modulus;
	p1 = p ;
	pt1 = twoToTheFiftyThree;
	r = 1.0 ;
	ddModulus.x[0] = modulus;
	ddModulus.x[1] = 0.0;
	
	while(1){
		
		if(p1 >= pt1){
			// r = mod (2.d0 * r, am)
			dd1 = ddMult(2.0, r) ;
			if( dd1.x[0] > modulus){
				dd2 = ddSub(dd1, ddModulus) ;
				dd1 = dd2 ;
			}
			r = dd1.x[0] ;
			p1 -= pt1 ;
		}
		
		pt1 *= 0.5 ;
		
		if(pt1 >= 1.0){
			//r = mod (r * r, am)
			dd1 = ddMult(r, r) ;
			dd2.x[0] = reciprocalOfModulus * dd1.x[0] ;
			d2 = trunc(dd2.x[0]) ;
			dd2 = ddMult(modulus, d2) ;
			dd3 = ddSub(dd1, dd2) ;
			r = dd3.x[0] ;
			
			if(r < 0.0)
				r += modulus ;
		}
		else
			break;
	}
	
	return r;
}


dd_real ddMult(double da, double db){
	/*
	 Returns res = a * b, where res is double double precision
	 
	 Input:
	 double da, db         Values to multiply
	 
	 Output:
	 returned (dd_real)    Their products in double double precision
	 */
	
	dd_real res ;
	
	double split = 134217729.0 ;
	
	double a1, a2, b1, b2, cona, conb ;
	
	cona = da * split ;
	conb = db * split ;
	a1 = cona - (cona - da) ;
	b1 = conb - (conb - db) ;
	a2 = da - a1 ;
	b2 = db - b1 ;
	
	res.x[0] = da * db ;
	res.x[1] = (((a1 * b1 - res.x[0]) + a1 * b2) + a2 * b1) + a2 * b2 ;
	
	return res;
	
}

dd_real ddSub(dd_real a, dd_real b){
	/*
	 Returns res = a - b, where res is double double precision
	 
	 Input:
	 double da, db         Values to subtract
	 
	 Output:
	 returned (dd_real)    Their difference in double double precision
	 */
	
	double e, temp1, temp2;
	dd_real res ;
	
	temp1 = a.x[0] - b.x[0] ;
	e = temp1 - a.x[0] ;
	temp2 = ((-b.x[0] - e) + (a.x[0] - (temp1 - e))) + a.x[1] - b.x[1] ;
	
	res.x[0] = temp1 + temp2 ;
	res.x[1] = temp2 - (res.x[0] - temp1) ;
	
	return res;
}

