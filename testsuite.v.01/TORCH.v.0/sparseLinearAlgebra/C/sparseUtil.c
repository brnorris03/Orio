

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#include "sparseUtil.h"

/*
 Sparse matrix utility routines.

 Alex Kaiser, LBNL, 2010
*/


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

csrMatrix allocCSRMatrix(int m, int n, int nnz){
	/*
	 Allocates space for an m x n matrix with nnz non-zeros in csr format.

	 Input:
		int m					Number of rows.
		int n					Number of columns.
		int nnz					Number of non-zeros to include.

	 Output
		csrMatrix (returned)	csrMatrix struct with specified values.

	 */

	csrMatrix temp ;
	temp.m = m;
	temp.n = n;
	temp.nnz = nnz;
	temp.rowPtr = (int *) malloc( (m+1) * sizeof(int)) ;
	temp.columnIndices = (int *) malloc(nnz * sizeof(int)) ;
 	temp.values = (double *) malloc(nnz * sizeof(double)) ;
 	return temp;
}

void freeCSRMatrixArrays(csrMatrix A){
	/*
	 Frees arrays from a csr matrix
	 Doesn't touch the struct itself

	 Input:
		csrMartix A			Matrix from which to free arrays
	 */

	free(A.rowPtr) ;
	free(A.columnIndices) ;
	free(A.values);
}

void printCSRMatrix(csrMatrix A){
	/*
	 Prints the rowPtr, columnIndices and values arrays of a csr matrix to stdout.

	 Input:
		csrMatrix A		Matrix to print.

	 Output:
		Arrays of A printed to stdout

	 */

	int i;

	printf("rowPtr = \n") ;
	for(i=0; i < (A.m + 1); i++)
		printf("%d ", A.rowPtr[i] ) ;
	printf("\n\n") ;

	printf("columnIndices = \n") ;
	for(i=0; i < A.nnz; i++)
		printf("%d ", A.columnIndices[i] ) ;
	printf("\n\n") ;

	printf("values = \n") ;
	for(i=0; i < A.nnz; i++)
		printf("%f ", A.values[i] ) ;
	printf("\n\n") ;

}


csrMatrix getCSRfromRowColumn(int m, int n, int row[], int column[], double valuesOrig[], int origLength){

	/*
	 Generates a CSR matrix from a matrix in (row, col, val) format.
	 Values are sorted and duplicates are removed, so the (row, col, val) arrays
		can come in any order.

	 Input:
		int m					Number of rows.
		int n					Number of columns.
		int row[]				Row indices.
		int column[]			Column indices.
		double valuesOrig[]		Values of matrix entries
		int origLength			Original length of array, which may contain duplicate entries

	 Output:
		csrMatrix A				The original matrix in CSR format

	 */
	if( !isSortedAndNoDuplicates(row, column, valuesOrig, origLength))
		quickSortTriples(row, column, valuesOrig, 0, origLength-1) ;

	// copy buffers
	int *rowCopy = (int *) malloc(origLength * sizeof(int));
	int *columnCopy = (int *) malloc(origLength * sizeof(int));
	double *valuesOrigCopy = (double *) malloc(origLength * sizeof(double)) ;

	rowCopy[0] = row[0] ;
	columnCopy[0] = column[0];
	valuesOrigCopy[0] = valuesOrig[0];

	int nnz = 1 ;

	int j;
	for(j=1; j<origLength; j++){

		// copy if the last indices are not the same, that is, don't take duplicates
		if( (row[j] != rowCopy[nnz-1]) || (column[j] != columnCopy[nnz-1]) ){
			rowCopy[nnz] = row[j] ;
			columnCopy[nnz] = column[j] ;
			valuesOrigCopy[nnz] = valuesOrig[j] ;
			nnz++ ;
		}

	}

	csrMatrix A = allocCSRMatrix(m, n, nnz) ;

	for(j=0; j < A.nnz; j++){
		A.columnIndices[j] = columnCopy[j] ;
		A.values[j] = valuesOrigCopy[j] ;
	}

	int rowIndex = 0;
	A.rowPtr[0] = 0;

	for(j=0; j < A.nnz; j++){
		if(rowCopy[j] > rowIndex){
			rowIndex++;
			A.rowPtr[rowIndex] = j ;
		}
	}

	A.rowPtr[A.m] = A.nnz ;

	// free resources
	free(rowCopy);
	free(columnCopy);
	free(valuesOrigCopy);

	return A;
}



void quickSortTriples(int row[], int column[], double valuesOrig[], int left, int right){
	/*
	 Simple quick sort.
	 Taken directly from "The C Programming Language", Kernighan and Ritchie.
	 Sort triples into increasing lexicographical order by (row, column, value).

	 Input:
		int row[]				Row array.
		int column[]			Column array.
		double valuesOrig[]		Values array.
		int left				First index of array to sort.
		int right				Last index of array to sort.

	 Output:
		int row[]				Row array, sorted in the relevant region.
		int column[]			Column array, sorted in the relevant region.
		double valuesOrig[]		Values array, sorted in the relevant region.
	 */

	int i, last;

	if (left >= right) /* do nothing if array contains */
		return;        /* fewer than two elements */

	swapTriples(row, column, valuesOrig, left, (left + right)/2); /* move partition elem */
	last = left;                     /* to v[0] */

	for (i = left + 1; i <= right; i++)  /* partition */
		if (compareTriple(row, column, valuesOrig, i, left) < 0)
			swapTriples(row, column, valuesOrig, ++last, i);

	swapTriples(row, column, valuesOrig, left, last);            /* restore partition  elem */

	quickSortTriples(row, column, valuesOrig, left, last-1);
	quickSortTriples(row, column, valuesOrig, last+1, right);
}


void swapTriples(int row[], int column[], double valuesOrig[], int i, int j) {
	/*
	 Swap entries in two positions of all three input arrays.

	 Input:
		int row[]				Row array.
		int column[]			Column array.
		double valuesOrig[]		Values array.
		int i					First index to swap.
		int j					Second index to swap.

	 Output:
		int row[]				Row array, swapped.
		int column[]			Column array, swapped.
		double valuesOrig[]		Values array, swapped.

	 */

	int temp;
	temp = row[i];
	row[i] = row[j];
	row[j] = temp;

	temp = column[i];
	column[i] = column[j];
	column[j] = temp;

	double tempDouble ;
	tempDouble = valuesOrig[i];
	valuesOrig[i] = valuesOrig[j];
	valuesOrig[j] = tempDouble;
}

int compareTriple(int row[], int column[], double valuesOrig[], int i, int j){
	/*
	 Compares (row, col, val) triples in lexicographical order.

	 Input:
		int row[]				Row array.
		int column[]			Column array.
		double valuesOrig[]		Values array.
		int i					First index to compare.
		int j					Second index to compare.

	 Output:
		int (returned)			Value is
									-1 if entry[i] < entry[j]
									 1 if entry[i] > entry[j]
									 0 if equal
	 */


    if(row[i] < row[j])
        return -1 ;
    else if (row[i] > row[j])
        return 1 ;
    else{  //if row(i) == row(j)
            if (column[i] < column[j])
                return -1 ;
            else if (column[i] > column[j])
                return  1 ;
            else
                if (valuesOrig[i] < valuesOrig[j])
                	return -1;
                else if( valuesOrig[i] > valuesOrig[j])
                	return 1;
                else
                	return 0;
    }

}

char isSortedAndNoDuplicates(int row[], int column[], double valuesOrig[], int origLength){
	/*
	 Checks whether a given group of arrays are sorted in lexicographical order and duplicate free

	 Input:
	 	int row[]				Row indices.
		int column[]			Column indices.
		double valuesOrig[]		Values of matrix entries.
		int origLength			Original length of array, which may contain duplicate entries.

	Output:
	char (returned)  Whether array is sorted and duplicate free.
	 */

	int j;
	for(j=0; j < origLength - 1; j++){

		// check for out of order element
		if( compareTriple(row, column, valuesOrig, j+1, j) != 1)
			return 0;

		// check for duplicate entry
		if( (column[j+1] == column[j]) && (row[j+1] == row[j]) )
			return 0;
	}

	return 1;
}


double ** allocDouble2d(int m, int n){
	// returns ptr to and (m by n) array of double precision
	double **temp = (double **) malloc( m * sizeof(double *) ) ;
	int k;
	for(k=0; k<m; k++){
		temp[k] = (double *) malloc( n * sizeof(double));
	}
	return temp;
}



void freeDouble2d(double ** z, int m, int n){
	// frees ptr to an (m by n) array of double precision complex
	int k;
	for(k=0; k<m; k++){
		free( z[k] );
	}
	free(z) ;
}

double ** toDense(csrMatrix A){
	/*
	 Converts csrMatrix A to dense format.
	 Dense matrix is allocated.

	 Input:
		csrMatrix A				Matrix to be converted

	 Output:
		double ** (returned)	Dense matrix with same entries as A

	 */

	double ** dense = allocDouble2d(A.m, A.n) ;

	int i, j;

	// zero matrix before using
	for(i=0; i<A.m; i++){
		for(j=0; j<A.n; j++){
			dense[i][j] = 0.0 ;
		}
	}

	// take values from CSR matrix
	for(i=0; i<A.m; i++)
		for(j = A.rowPtr[i]; j < A.rowPtr[i+1]; j++)
			dense[i][A.columnIndices[j] ] = A.values[j] ;

	return dense;
}


double ** toDenseFromRowColumn(int m, int n, int row[], int column[], double valuesOrig[], int nnz){
	/*
	 Converts (row, col, val) matrix to dense format.
	 Dense matrix is allocated.

	 Input:
		int m					Number of rows.
		int n					Number of columns.
		int row[]				Row indices.
		int column[]			Column indices.
		double valuesOrig[]		Values of matrix entries
		int nnz					Original length of arrays

	 Output:
		double ** (returned)	Dense matrix with same entries as A

	 */
	double ** dense = allocDouble2d(m, n) ;

	int i, j;

	int rowTemp, colTemp;

	for(i=0; i<m; i++){
		for(j=0; j<n; j++){
			dense[i][j] = 0.0 ;
		}
	}

	for( i=0; i < nnz; i++){
		rowTemp = row[i] ;
		colTemp = column[i] ;
		dense[rowTemp][colTemp] = valuesOrig[i] ;
	}

	return dense;
}


void printMatrix(double ** zz, int m, int n){
	/*
	 Prints 2D array to stdout.

	 Input:
		double ** zz		Array to print.
		int m				Number of rows.
		int n				Number of columns.
	 */

	int j,k;
	for(j=0; j<m; j++){
		for(k=0; k<n; k++){
			//if( fabs(zz[j][k]) < 1e-10)
			//	printf("0       ");
			//else
				printf("%f ", zz[j][k]);
		}
		printf("\n");
	}
	printf("\n");
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
		printf("%f ", z[i]) ;

	printf("\n");
}

void printIntegerVector(int *z, int n){
	/*
	 Prints integer array to stdout.

	 Input:
		int * z			Array to print.
		int n				Length.
	 */

	int i;
	for(i=0; i<n; i++)
		printf("%d ", z[i]) ;

	printf("\n");
}

void spyPrimitive(double **x, int m, int n){
	/*
	 Prints a primitive plot of the non-zeros of x

	 Input:
		double **x		Matrix to print.
		int m				Number of rows.
		int n				Number of columns.

	 */

	int i, j;

	for(i=0; i<m; i++){
		for(j=0; j<n; j++){
			if(x[i][j] == 0.0)
				putchar('-');
			else
				putchar('X');
		}
		putchar('\n') ;
	}


}

double normDoublel2(double * x, int n){
	/*
	 Calculates l2 norm of double array.

	 Input:
		double *x			Array to calculate norm.
		int n				Length.
	 */


	double runningNorm = 0.0;
	int j;

	for (j=0; j<n; j++)
		runningNorm += x[j] * x[j] ;

	return sqrt(runningNorm);
}


double l2RelativeErrDouble_local(double * guess, double * trueValue, int n){
	/*
	 Computes l2 relative error between two real vectors.
	 norm( guess - trueValue ) / norm(trueValue)

	 Input:
		double *guess			Guess array.
		double *trueValue		True array.
		int n					Length.

	 Output:
		double (returned)		l2 relative error between two input vectors.
	 */


	int j;

	double * diffs = (double *) malloc(n * sizeof(double));

	for (j=0; j<n; j++)
		diffs[j] = guess[j] - trueValue[j] ;

	double normTrue = normDoublel2(trueValue,n);
	double normDiffs = normDoublel2(diffs, n);

	// if true norm is equal, then difference must be exactly zero.
	// else return NaN
	if(normTrue == 0.0 ){
		if(normDiffs == 0.0)
			return 0.0;
		else
			return 0.0/0.0;
	}

	free(diffs) ;
	return normDiffs / normTrue;
}


int getRandomInt_local(int maxVal){
	/*
	 Returns random int in range [0, maxVal).
	 Uses frand().
	 */
	return floor( maxVal * bcnrand_local()) ;
}

double bcnrand_local( ){
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
	static dd_real_local dd1, dd2, dd3;
	
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
		d2 = expm2_local(startingIndex - threeToTheThirtyThree, threeToTheThirtyThree) ;
		d3 = trunc(0.5 * threeToTheThirtyThree) ;
		dd1 = ddMult_local(d2, d3) ;
		d1 = trunc(reciprocalOfThreeToThirtyThree * dd1.x[0]) ;
		dd2 = ddMult_local(d1, threeToTheThirtyThree) ;
		dd3 = ddSub_local(dd1, dd2) ;
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
	dd2 = ddMult_local(threeToTheThirtyThree, d2) ;
	dd3 = ddSub_local(dd1, dd2);
	d1 = dd3.x[0];
	
	if(d1 < 0.0)
		d1 += threeToTheThirtyThree ;
	
	return reciprocalOfThreeToThirtyThree * d1 ;
}

double expm2_local(double p, double modulus){
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
	dd_real_local dd1, dd2, dd3, ddModulus;
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
			dd1 = ddMult_local(2.0, r) ;
			if( dd1.x[0] > modulus){
				dd2 = ddSub_local(dd1, ddModulus) ;
				dd1 = dd2 ;
			}
			r = dd1.x[0] ;
			p1 -= pt1 ;
		}
		
		pt1 *= 0.5 ;
		
		if(pt1 >= 1.0){
			//r = mod (r * r, am)
			dd1 = ddMult_local(r, r) ;
			dd2.x[0] = reciprocalOfModulus * dd1.x[0] ;
			d2 = trunc(dd2.x[0]) ;
			dd2 = ddMult_local(modulus, d2) ;
			dd3 = ddSub_local(dd1, dd2) ;
			r = dd3.x[0] ;
			
			if(r < 0.0)
				r += modulus ;
		}
		else
			break;
	}
	
	return r;
}


dd_real_local ddMult_local(double da, double db){
	/*
	 Returns res = a * b, where res is double double precision
	 
	 Input:
	 double da, db         Values to multiply
	 
	 Output:
	 returned (dd_real_local)    Their products in double double precision
	 */
	
	dd_real_local res ;
	
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

dd_real_local ddSub_local(dd_real_local a, dd_real_local b){
	/*
	 Returns res = a - b, where res is double double precision
	 
	 Input:
	 double da, db         Values to subtract
	 
	 Output:
	 returned (dd_real_local)    Their difference in double double precision
	 */
	
	double e, temp1, temp2;
	dd_real_local res ;
	
	temp1 = a.x[0] - b.x[0] ;
	e = temp1 - a.x[0] ;
	temp2 = ((-b.x[0] - e) + (a.x[0] - (temp1 - e))) + a.x[1] - b.x[1] ;
	
	res.x[0] = temp1 + temp2 ;
	res.x[1] = temp2 - (res.x[0] - temp1) ;
	
	return res;
}


double read_timer_local( ){
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
