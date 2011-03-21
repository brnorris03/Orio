
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "sparseUtil.h"
#include "generators.h"

/*
  Functions to generate test martices.


  Alex Kaiser, LBNL 7/2010
 */

csrMatrix getStructuredLowerTriangularCSR(int n, int nnz, double distribution[]){
	/*
	 Generates randomized lower triangular matrices in a banded structure.
	 Roughly follows the scheme of

	 "Benchmarking Sparse Matrix-Vector Multiply in Five Minutes"
	 Hormozd Gahvari, Mark Hoemmen, James Demmel, and Katherine Yelick
	 UC Berkeley.

	 Inputs:
		int n						Size of matrix, select such that (n mod(10) == 1) to ensure that corner of matrix isn't left unfilled
		int nnz						Approximate number of nonzeros. Actual number may be slightly lower because of possibility of duplicate entries
										which are removed
		double distribution[10]		Matrix is divided into ten bands, each including approximately distribution[i]*nnz nonzero entries
										Warning! distribution must be of length ten or severe errors will result!

	 Output:
		csrMatrix A					Structured lower triangular matrix in compressed sparse-row format
	*/


	int j;
	double sumDist = 0.0;

    if (nnz < n){
        fprintf(stderr, "must store at least n non-zeros for this routine, as n non zeros are placed on the diagonal");
        exit(-1);
    }

	for(j=0; j<10; j++)
		sumDist += distribution[j] ;

	if(sumDist > 1.0){
		fprintf(stderr, "distribution must sum to less or equal to one.\n") ;
		exit(-1);
	}

	int *row = (int *) malloc(nnz * sizeof(int));
	int *column = (int *) malloc(nnz * sizeof(int));
	double *valuesOrig = (double *) malloc(nnz * sizeof(double)) ;

	int nnzSoFar = 0;

	for(j=0; j<n; j++){
		row[nnzSoFar] = j;
		column[nnzSoFar] = j;
		valuesOrig[nnzSoFar] = 10.0 ; 
		nnzSoFar++;
	}

	int width = (int) floor((n-1) / 10) ;
	int band, nnzThisBand;
	int rowInd, colInd;

	for(band = 0; band < 10; band++){
		nnzThisBand = (int) floor( distribution[band] * (nnz - n)) ;

		for(j=0; j<nnzThisBand; j++){
			getRandomIndicesFromBand(&rowInd, &colInd, n, band, width);
			row[nnzSoFar] = rowInd ;
			column[nnzSoFar] = colInd ;
			valuesOrig[nnzSoFar] = bcnrand_local() ;
			nnzSoFar++;
		}
	}


	csrMatrix A = getCSRfromRowColumn(n, n, row, column, valuesOrig, nnzSoFar) ;

	free(row);
	free(column);
	free(valuesOrig);

	return A ;
}


csrMatrix getSymmetricDiagonallyDominantCSR(int n, int nnz, double distribution[]){
	/*
	 Generates randomized symmetric diagonally-dominant matrices in a banded structure.
	 Roughly follows the scheme of

	 "Benchmarking Sparse Matrix-Vector Multiply in Five Minutes"
	 Hormozd Gahvari, Mark Hoemmen, James Demmel, and Katherine Yelick
	 UC Berkeley.

	 Inputs:
		int n						Size of matrix, select such that (n mod(10) == 1) to ensure that corner of matrix isn't left unfilled
		int nnz						Approximate number of non-zeros. Actual number may be slightly lower because of possibility of duplicate entries
										which are removed
		double distribution[10]		Matrix is divided into ten bands, each including approximately distribution[i]*nnz nonzero entries
										Warning! distribution must be of length ten or severe errors will result!

	 Output:
		csrMatrix A					Structured lower triangular matrix in compressed sparse-row format
	 */


	int j;
	double sumDist = 0.0;

    if (nnz < n)
        fprintf(stderr, "must store at least n non-zeros for this routine, as n non zeros are placed on the diagonal");

	for(j=0; j<10; j++)
		sumDist += distribution[j] ;

	if(sumDist > 1.0){
		fprintf(stderr, "distribution must sum to less than one.\n") ;
		exit(-1);
	}

	int *row = (int *) malloc(nnz * sizeof(int));
	int *column = (int *) malloc(nnz * sizeof(int));
	double *valuesOrig = (double *) malloc(nnz * sizeof(double)) ;

	double *rowSum = (double *) malloc(n * sizeof(double)) ;
	for(j=0; j<n; j++)
		rowSum[j] = 0.0;

	int nnzSoFar = 0;

	int width = (int) floor((n-1) / 10) ;
	int band, nnzThisBand;
	int rowInd, colInd;
	double currentValue;


	for(band = 0; band < 10; band++){
		nnzThisBand = (int) floor( distribution[band] * (nnz - n) / 2) ;

		for(j=0; j<nnzThisBand; j++){

			// place lower diagonal element
			getRandomIndicesFromBand(&rowInd, &colInd, n, band, width);
			row[nnzSoFar] = rowInd ;
			column[nnzSoFar] = colInd ;
			currentValue = bcnrand_local();
			valuesOrig[nnzSoFar] = currentValue;
			nnzSoFar++;

			// place above diagonal element
			row[nnzSoFar] = colInd ;
			column[nnzSoFar] = rowInd ;
			valuesOrig[nnzSoFar] = currentValue;
			nnzSoFar++;

			// adjust the current value placed on the row
			rowSum[rowInd] += currentValue ;
			rowSum[colInd] += currentValue ;
		}
	}

	// add diagonal elements
	// set value to ensure that matrix is diagonally dominant
	for(j=0; j<n; j++){
		row[nnzSoFar] = j;
		column[nnzSoFar] = j;
		valuesOrig[nnzSoFar] = 2 * fabs(rowSum[j]) + 1.0 ;
		nnzSoFar++;
	}

	csrMatrix A = getCSRfromRowColumn(n, n, row, column, valuesOrig, nnzSoFar) ;

	free(row);
	free(column);
	free(valuesOrig);

	return A ;
}

csrMatrix getHeatEqnMatrix(int n, double lambda){
	/*
	   Returns an n x n x n heat equation matrix for a 3D explicit heat equation.
	   Assumes problem is homogeneous and ignores boundaries accordingly.

	   Input:
	         int n           Dimension of original heat equation grid.
	                            Matrix will be of dimension n^3 by n^3
	         double lambda   Constant for heat equation solves. lambda = dt / (h*h)

	   Output:
		csrMatrix A				Matrix for heat equation solves in CSR format

	*/

	int i, j, k;

	int nnzSoFar = 0;
	int nCubed = n * n * n ;
	int *row = (int *) malloc(7 * nCubed * sizeof(int));
	int *column = (int *) malloc(7 * nCubed * sizeof(int));
	double *valuesOrig = (double *) malloc(7 * nCubed *  sizeof(double)) ;

	int diagIndex;


	// this would traverse an n x n x n 3D grid array...
	for(i=0; i<n; i++){
	    for(j=0; j<n; j++){
	        for(k=0; k<n; k++){

	            diagIndex = k + n*j + n*n*i ;

	            if(i != 0){
	                row[nnzSoFar] = diagIndex ;
	                column[nnzSoFar] = diagIndex - n*n;
	                valuesOrig[nnzSoFar] = lambda ;
	                nnzSoFar++ ;
	            }

	            if(j != 0){
	                row[nnzSoFar] = diagIndex ;
	                column[nnzSoFar] = diagIndex - n;
	                valuesOrig[nnzSoFar] = lambda ;
	                nnzSoFar++ ;
	            }

	            if(k != 0){
	                row[nnzSoFar] = diagIndex ;
	                column[nnzSoFar] = diagIndex - 1;
	                valuesOrig[nnzSoFar] = lambda ;
	                nnzSoFar++ ;
	            }


	            //always place the diagonal element
	            row[nnzSoFar] = diagIndex ;
	            column[nnzSoFar] = diagIndex ;
	            valuesOrig[nnzSoFar] = 1 - 6 * lambda ;
	            nnzSoFar++ ;

	            if(k !=  n-1){
	                row[nnzSoFar] = diagIndex ;
	                column[nnzSoFar] = diagIndex + 1;
	                valuesOrig[nnzSoFar] = lambda ;
	                nnzSoFar++ ;
	            }

	            if(j !=  n-1){
	                row[nnzSoFar] = diagIndex ;
	                column[nnzSoFar] = diagIndex + n;
	                valuesOrig[nnzSoFar] = lambda ;
	                nnzSoFar++ ;
	            }

	            if(i !=  n-1){
	                row[nnzSoFar] = diagIndex ;
	                column[nnzSoFar] = diagIndex + n*n;
	                valuesOrig[nnzSoFar] = lambda ;
	                nnzSoFar++ ;
	            }
	        }
	    }
	}

	csrMatrix A = getCSRfromRowColumn(nCubed, nCubed, row, column, valuesOrig, nnzSoFar) ;

	free(row) ;
	free(column) ;
	free(valuesOrig) ;

	return A;
}

csrMatrix getHeatEqnMatrixImplicit(int n, double lambda){
	/*
	   Returns an n x n x n heat equation matrix for a 3D implicit heat equation.
	   Assumes problem is homogeneous and ignores boundaries accordingly.

	   Input:
	         int n           Dimension of original heat equation grid.
	                            Matrix will be of dimension n^3 by n^3
	         double lambda   Constant for heat equation solves. lambda = dt / (h*h)

	   Output:
		csrMatrix A			Matrix for heat equation solves in CSR format

	*/

	int i, j, k;

	int nnzSoFar = 0;
	int nCubed = n * n * n ;
	int *row = (int *) malloc(7 * nCubed * sizeof(int));
	int *column = (int *) malloc(7 * nCubed * sizeof(int));
	double *valuesOrig = (double *) malloc(7 * nCubed *  sizeof(double)) ;

	int diagIndex;


	// this would traverse an n x n x n 3D grid array...
	for(i=0; i<n; i++){
	    for(j=0; j<n; j++){
	        for(k=0; k<n; k++){

	            diagIndex = k + n*j + n*n*i ;

	            if(i != 0){
	                row[nnzSoFar] = diagIndex ;
	                column[nnzSoFar] = diagIndex - n*n;
	                valuesOrig[nnzSoFar] = -lambda ;
	                nnzSoFar++ ;
	            }

	            if(j != 0){
	                row[nnzSoFar] = diagIndex ;
	                column[nnzSoFar] = diagIndex - n;
	                valuesOrig[nnzSoFar] = -lambda ;
	                nnzSoFar++ ;
	            }

	            if(k != 0){
	                row[nnzSoFar] = diagIndex ;
	                column[nnzSoFar] = diagIndex - 1;
	                valuesOrig[nnzSoFar] = -lambda ;
	                nnzSoFar++ ;
	            }


	            //always place the diagonal element
	            row[nnzSoFar] = diagIndex ;
	            column[nnzSoFar] = diagIndex ;
	            valuesOrig[nnzSoFar] = 1 + 6 * lambda ;
	            nnzSoFar++ ;

	            if(k !=  n-1){
	                row[nnzSoFar] = diagIndex ;
	                column[nnzSoFar] = diagIndex + 1;
	                valuesOrig[nnzSoFar] = -lambda ;
	                nnzSoFar++ ;
	            }

	            if(j !=  n-1){
	                row[nnzSoFar] = diagIndex ;
	                column[nnzSoFar] = diagIndex + n;
	                valuesOrig[nnzSoFar] = -lambda ;
	                nnzSoFar++ ;
	            }

	            if(i !=  n-1){
	                row[nnzSoFar] = diagIndex ;
	                column[nnzSoFar] = diagIndex + n*n;
	                valuesOrig[nnzSoFar] = -lambda ;
	                nnzSoFar++ ;
	            }
	        }
	    }
	}

	csrMatrix A = getCSRfromRowColumn(nCubed, nCubed, row, column, valuesOrig, nnzSoFar) ;

	free(row) ;
	free(column) ;
	free(valuesOrig) ;

	return A;
}

csrMatrix getLaplacianMatrix(int n, double height){
	/*
	   Returns an n x n x n matrix for computing Laplacian.
	   Assumes problem is homogeneous and ignores boundaries accordingly.

	   Input:
	         int n           Dimension of original Laplacian grid.
	                            Matrix will be of dimension n^3 by n^3
	         double height   Grid height.

	   Output:
		csrMatrix A				Matrix for heat equation solves in CSR format

	*/

	int i, j, k;

	int nnzSoFar = 0;
	int nCubed = n * n * n ;
	int *row = (int *) malloc(7 * nCubed * sizeof(int));
	int *column = (int *) malloc(7 * nCubed * sizeof(int));
	double *valuesOrig = (double *) malloc(7 * nCubed *  sizeof(double)) ;

	int diagIndex;

	double lambda = 1 / (height * height);


	// this would traverse an n x n x n 3D grid array...
	for(i=0; i<n; i++){
	    for(j=0; j<n; j++){
	        for(k=0; k<n; k++){

	            diagIndex = k + n*j + n*n*i ;

	            if(i != 0){
	                row[nnzSoFar] = diagIndex ;
	                column[nnzSoFar] = diagIndex - n*n;
	                valuesOrig[nnzSoFar] = lambda ;
	                nnzSoFar++ ;
	            }

	            if(j != 0){
	                row[nnzSoFar] = diagIndex ;
	                column[nnzSoFar] = diagIndex - n;
	                valuesOrig[nnzSoFar] = lambda ;
	                nnzSoFar++ ;
	            }

	            if(k != 0){
	                row[nnzSoFar] = diagIndex ;
	                column[nnzSoFar] = diagIndex - 1;
	                valuesOrig[nnzSoFar] = lambda ;
	                nnzSoFar++ ;
	            }


	            //always place the diagonal element
	            row[nnzSoFar] = diagIndex ;
	            column[nnzSoFar] = diagIndex ;
	            valuesOrig[nnzSoFar] = - 6 * lambda ;
	            nnzSoFar++ ;

	            if(k !=  n-1){
	                row[nnzSoFar] = diagIndex ;
	                column[nnzSoFar] = diagIndex + 1;
	                valuesOrig[nnzSoFar] = lambda ;
	                nnzSoFar++ ;
	            }

	            if(j !=  n-1){
	                row[nnzSoFar] = diagIndex ;
	                column[nnzSoFar] = diagIndex + n;
	                valuesOrig[nnzSoFar] = lambda ;
	                nnzSoFar++ ;
	            }

	            if(i !=  n-1){
	                row[nnzSoFar] = diagIndex ;
	                column[nnzSoFar] = diagIndex + n*n;
	                valuesOrig[nnzSoFar] = lambda ;
	                nnzSoFar++ ;
	            }
	        }
	    }
	}

	csrMatrix A = getCSRfromRowColumn(nCubed, nCubed, row, column, valuesOrig, nnzSoFar) ;

	free(row) ;
	free(column) ;
	free(valuesOrig) ;

	return A;
}


void getRandomIndicesFromBand(int *rowInd, int *colInd, int n, int bandNumber, int width){
	/*
	Returns randomized pair of indices in the specified band off the diagonal
	Does not ever return i=j, so when using this function to construct a
		matrix diagonal entries should be handled manually

	Band should be indexed from (bandNumber * width) : (bandNumber * width + width)

	Input
		int *rowInd			Pointer to row index, returned
		int *colInd			Pointer to column index, returned
		int n				Matrix size
		int bandNumber		Band of matrix from which to generate a pair.
								Numbered from 0 to 9
		int width			Width, in matrix entries, of each band

	 Output
		int *rowInd			Pointer to row index
		int *colInd			Pointer to column index
	 */

	int firstRand, randVal, offDiagIndex, diagIndex ;
	int rowIndTemp, colIndTemp;

	firstRand = getRandomInt_local(width) + 1;
	offDiagIndex = bandNumber * width + firstRand ;

	randVal = getRandomInt_local( 2*n - ( 2 * (bandNumber * width + firstRand) ) )  ;
	diagIndex = bandNumber * width + firstRand + randVal ;

	rowIndTemp = round( (offDiagIndex + diagIndex) / 2 ) ;
	colIndTemp = round( ( diagIndex - offDiagIndex ) / 2 ) ;


	// this should never be called if range of parameters is computed correctly
	int k = 0 ;
	while( (rowIndTemp >= n) || (colIndTemp >= n) || (rowIndTemp < 0) || (colIndTemp < 0) || (rowIndTemp == colIndTemp) ){
	    k = k + 1;

		firstRand = getRandomInt_local(width) + 1;
		offDiagIndex = bandNumber * width + firstRand ;

		randVal = getRandomInt_local( 2*n - ( 2 * (bandNumber * width + firstRand) ) )  ;
		diagIndex = bandNumber * width + firstRand + randVal ;

		rowIndTemp = round( (offDiagIndex + diagIndex) / 2 ) ;
		colIndTemp = round( ( diagIndex - offDiagIndex ) / 2 ) ;
	}


	if(k > 0){
	    fprintf(stderr, "Hit problem loop %d times\n", k) ;
	}

	if ( (rowIndTemp >= n) || (colIndTemp >= n) || (rowIndTemp < 0) || (colIndTemp < 0) || (rowIndTemp == colIndTemp) ){
		fprintf(stderr, "Paramters not correct. Errors. Exiting.\n") ;
		exit(-1);
	}

	// assign values for return
	*rowInd = rowIndTemp ;
	*colInd = colIndTemp ;
}



