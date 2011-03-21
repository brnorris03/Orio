#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "complexUtil.h"
#include "ffts.h"

/*
 FFT algorithms and associated other routines. 
 
 Alex Kaiser, LBNL, 9/09
*/ 

struct complex *initTable(int tableSize){
	/*
	 Initializes lookup table for exponential factors for FFTs
	 All FFTs computed using the table must be of size less than or equal to tableSize

	 Input:
	 int tableSize							Maximum length of FFT to calculate

	 Output:
	 struct complex * table (returned)		Table of exponential factors.
	 */

	int k;
	double pi = acos(-1) ;
	struct complex * table = allocComplex(tableSize);


	for(k=0; k<tableSize; k++){
		setComplex(&table[k], cos(k * 2 * pi / (double) tableSize ), sin( k * 2 * pi / (double) tableSize ) );
	}

	return table;
}


struct complex * stockhamFFT( struct complex *x, int n, int sign, struct complex *table, int tableSize){
    /*
	 Computes FFT of x using the Stockham FFT algorithm. 
	 Secondary one-dimensional FFT called by fourStep fft.
	 
	 Input:
	 struct complex *x				Input array, overwritten in function. 
	 int n							Length of input, must be power of two. 
	 int sign						Sign of exponent in FFT. -1 for forward FFT, 1 for inverse. 
	 struct complex *table			Table of precompted exponential factors. 
	 int tableSize					Size of table, must be at least n and a power of two. 
	 
	 Output:
	 struct complex * (returned)	FFT of x. 
	 */

	int l = n/2;
	int m = 1;

	int iteration;
	int j,k;

	int index;

	struct complex c0, c1, w;
	struct complex *y = allocComplex(n);


	for(iteration=0; iteration < log2(n); iteration++){

		for(j=0; j<l; j++){

			// get the index of exponential factor from the table
			if(sign == -1){  // must count backwards
				index = tableSize - ( tableSize /(l*2) * j) ;
			}
			else
				index = ( tableSize / (l*2) ) * j ; // division here is an exact integer division

			if(index == tableSize) index = 0; // if out of bounds, take the first

			w = table[index] ;

			for(k=0; k<m; k++){
				c0 = x[ k + j*m ] ;
				c1 = x[ k + j*m + l*m ] ;
				y[ k + 2*j*m ] = addComplex(c0, c1);
				y[ k + 2*j*m + m ] = multComplex(w, subComplex(c0, c1)) ;
			}
		}

		for(j=0; j<n; j++)  x[j] = y[j] ;

		l = l/2 ; // this is exact division, as l is initialized to a power of two
		m = m*2 ;
	}

	free(y);
	return x;
}


struct complex * cooleyTukeyFFT(struct complex *x, int n, int sign, struct complex *table, int tableSize){
    /*
	 Cooley-Tukey FFT
	 Algorithm 1.6.1 in Van Loan's 'Computational Frameworks for the Fast Fourier Transform'
	 Secondary one-dimensional FFT called by fourStep fft.
	 
	 Input:
	 struct complex *x				Input array, overwritten in function. 
	 int n							Length of input, must be power of two. 
	 int sign						Sign of exponent in FFT. -1 for forward FFT, 1 for inverse. 
	 struct complex *table			Table of precompted exponential factors. 
	 int tableSize					Size of table, must be at least n and a power of two. 
	 
	 Output:
	 struct complex * (returned)	FFT of x. 
	 */
	
	unsigned int j, k, m, q, s;
	int L, r, lStar;
	int index;
	struct complex w, temp;

	int t = log2(n);


	// Algorithm 1.5.2 in Van Loan's 'Computational Frameworks for the
    // Fast Fourier Transform'
	// permute by bit reversal
	for(k=0; k<n; k++){

		// bit reverse index
		j=0;
		m=k;
		for(q=0; q<t; q++){
			s = floor(m/2); 
			j = 2*j + (m - 2*s);
			m = s;
		}

		if(j > k){
            temp = x[j];
			x[j] = x[k];
			x[k] = temp;
        }
	}


	for (q=1; q<=t; q++) {

		L = 0x1 << q ;			// 2^q
		r = n/L;				// exact division
		lStar = L/2;

		for (j=0; j<lStar; j++) {

			// get the index of exponential factor from the table
			if(sign == -1){  // must count backwards
				index = tableSize - ( tableSize / L * j) ;
			}
			else
				index = ( tableSize / L ) * j ; // division here is an exact integer division

			if(index == tableSize) index = 0; // if out of bounds, take the first

			w = table[index] ;

			for (k=0; k<r; k++) {
				temp = multComplex( w, x[k*L + j + lStar] );
				x[ k*L + j + lStar ] = subComplex( x[k*L + j] , temp);
				x[ k*L + j ] = addComplex( x[ k*L + j ] , temp);
			}
		}
	}

	return x;
}


struct complex * fourStepFFT( struct complex *x, int n, int n1, int n2, int sign, struct complex *table, int tableSize){
	/*
	 Computes FFT(x) by the four step method.
	 Primary one-dimensional FFT algorithm. 
	 To calculate the inverse FFT, use sign = 1 and devide output by n after function call.
	 
	 Input:
	 struct complex *x				Input array, overwritten in function. 
	 int n							Length of input, must be power of two. 
	 int n1, n2						Integers such that n = n1 * n2. Each must be a power of two. 
	 int sign						Sign of exponent in FFT. -1 for forward FFT, 1 for inverse. 
	 struct complex *table			Table of precompted exponential factors. 
	 int tableSize					Size of table, must be at least n and a power of two. 
	 
	 Output:
	 struct complex * (returned)	FFT of x. 
	 */
	 

	int j, k;
	int index;

	// copy into 2d array
	// allocate large enough to transpose in place
	struct complex ** zz = allocComplex2d( max(n1, n2), max(n1, n2) ) ;

	// allocate column-wise as in algorithm statement
	// regardless, one array must non-unit stride if basic algorithm statement is followed
	for(k=0; k<n2; k++){
		for(j=0; j<n1; j++){
			zz[j][k] = x[ j + k*n1 ] ;
		}
	}

	for(j=0; j<n1; j++){
		// may select Stockham or Cooley-Tukey FFT here.  
		zz[j] = stockhamFFT(zz[j], n2, sign, table, tableSize);
		//zz[j] = cooleyTukeyFFT(zz[j], n2, sign, table, tableSize);
	}

	//twiddles
	for(j=0; j<n1; j++){
		for(k=0; k<n2; k++){

			// get the index of exponential factor from the table
			if(sign == -1){  // must count backwards
				index = tableSize - ( (tableSize /(n)) * j * k) ;
			}
			else
				index = ( tableSize /(n)) * j * k ; // division here is an exact integer division

			if(index == tableSize) index = 0; // if out of bounds, take the first

			zz[j][k] = multComplex( zz[j][k] , table[index] );
		}
	}

	zz = transpose(zz, n2, n1);

	for(j=0; j<n2; j++){
		// may select Stockham or Cooley-Tukey FFT here. 
		zz[j] = stockhamFFT(zz[j], n1, sign, table, tableSize);
		//zz[j] = cooleyTukeyFFT(zz[j], n1, sign, table, tableSize);
	}

	// n2 down the down dimension.
	for(j=0; j<n1; j++){
		for(k=0; k<n2; k++){
			x[ k + j*n2 ] = zz[k][j];
		}
	}

	freeComplex2d(zz, max(n1, n2), max(n1, n2) );

	return x;
}


struct complex * r_to_cFFT(double x[], int n, int n1, int n2, int sign, struct complex *table, int tableSize){
	/*
	 Performs FFT of real data, interleaving data and using a Four Step FFT after doing so. 
	
	 Input:
	 double x[]						Real input array, overwritten in function. 
	 int n							Length of input, must be power of two. 
	 int n1, n2						Integers such that n = n1 * n2. Each must be a power of two. 
	 int sign						Sign of exponent in FFT. -1 for forward FFT, 1 for inverse. 
	 struct complex *table			Table of precompted exponential factors. 
	 int tableSize					Size of table, must be at least n and a power of two. 
	 
	 Output:
	 struct complex * (returned)	FFT of x, interleaved as length n/2 complex. 
	 */



	int k;
	int m = n/2;
	int index; 

	struct complex w, d0, d1;
	struct complex i = newComplex(0.0, 1.0);

	struct complex *v = allocComplex(m);
	for(k=0; k<m; k++) {
		setComplex( &v[k], x[2*k], x[2*k + 1] ) ;
	}

	v = fourStepFFT(v, m, (n1)/2, n2, sign, table, tableSize) ;

	struct complex *y = allocComplex(m+1) ;
	setComplex(&y[0], v[0].real + v[0].imag, 0);
	setComplex(&y[n/4], v[n/4].real, sign * v[n/4].imag );
	setComplex(&y[n/2], v[0].real - v[0].imag, 0);

	for(k=1; k < n/4; k++){
		d0 = addComplex(v[k], conjugate( v[m-k] ) ) ;
	
		// get the index of exponential factor from the table
		if(sign == -1){  // must count backwards
			index = tableSize - ( tableSize /(m*2) * k) ;
		}
		else
			index = ( tableSize / (m*2) ) * k ; // division here is an exact integer division
		
		if(index == tableSize) index = 0; // if out of bounds, take the first
		w = table[index] ; 
		
		d1 = multComplexReal(multComplex(i, multComplex( w, subComplex( v[k], conjugate(v[m-k]) ))), -1.0) ;
		y[k] = multComplexReal(addComplex(d0, d1) , 0.5);
		y[m-k] = multComplexReal(subComplex( conjugate(d0), conjugate(d1) ), 0.5 );
	}

	return y;
}


struct complex * c_to_rFFT(struct complex *x, int n, int n1, int n2, int sign, struct complex *table, int tableSize){
	/*
	 Performs a complex to real FFT. 
	 Data must be correctly packed for transformation to invert properly.  
	 	 
	 Input:
	 struct complex *x				Input array of length n/2 +  1, overwritten in function. 
										First and last elements must be real for input data to reperesent output of R->C FFT. 
	 int n							Length of original real data, must be power of two. 
	 int n1, n2						Integers such that n = n1 * n2. Each must be a power of two. 
	 int sign						Sign of exponent in FFT. -1 for forward FFT, 1 for inverse. 
	 struct complex *table			Table of precompted exponential factors. 
	 int tableSize					Size of table, must be at least n and a power of two. 
	 
	 Output:
	 struct complex * (returned)	FFT of x, interleaved as length n/2 complex.
	 */


	int k;
	int m = n/2;
	int index; 

	struct complex w, d0, d1;
	struct complex i = newComplex(0.0, 1.0);

	struct complex *v = allocComplex(m);

	v[0] = addComplex( addComplex(x[0], x[m]), multComplex(i, subComplex(x[0], x[m])))   ;
	setComplex( &v[n/4], 2*x[n/4].real, -2 * sign * x[n/4].imag);

	for(k=1; k < n/4; k++){
		d0 = addComplex(x[k], conjugate( x[m-k] ) ) ;
		
		// get the index of exponential factor from the table
		if(sign == -1){  // must count backwards
			index = tableSize - ( tableSize /(m*2) * k) ;
		}
		else
			index = ( tableSize / (m*2) ) * k ; // division here is an exact integer division
		
		if(index == tableSize) index = 0; // if out of bounds, take the first
		w = table[index] ; 
		
		d1 = multComplex(i, multComplex( w, subComplex( x[k], conjugate( x[m-k])))) ;
		v[k] = addComplex(d0, d1) ;
		v[m-k] = subComplex( conjugate(d0), conjugate(d1)) ;
	}

	v = fourStepFFT(v, m, n1/2, n2, sign, table, tableSize) ;

	return v;
}


double * naiveConvolution(double a[], double b[], int n, int numToCompute){
	/*
	 Computes a portion of the convolution of real vectors a and b naively. 
	 Provided only for verification.
	 
	 Input:
	 double a[]				First vector for convolution. 
	 double b[]				Second vector for convolution. 
	 int n					Total length of both vectors. 
	 int numToCompute		Number of elements to compute. Only computes the first numToCompute 
								elements of the convolution for efficiency. 
	 
	 Output:
	 double * (returned)	The first numToCompute elements of the convolution. 
	 */ 

	int j, k, subscript ;

	if (2*n < numToCompute){
		fprintf(stderr, "Can only compute 2*n convolution elements. Dafaulting to 2*n.\n");
		numToCompute = 2*n;
	}

	double *f = (double *) malloc( numToCompute * sizeof(double)) ;
	double *aPad = (double *) malloc( 2 * n * sizeof(double)) ;
	double *bPad = (double *) malloc( 2 * n * sizeof(double)) ;

	// copy first half of input arrays to padded arrays
	for(k=0; k<n; k++) {
		aPad[k] = a[k];
		bPad[k] = b[k];
	}
	// and zero the second half
	for(k=n; k<2*n; k++){
		aPad[k] = 0.0;
		bPad[k] = 0.0;
	}

	for(k=0; k < numToCompute; k++){
		f[k] = 0.0;
		for(j=0; j<2*n; j++){
			if ((k - j) < 0)
				subscript = k - j + 2*n;
			else
				subscript = k - j;
			f[k] += aPad[j] * bPad[subscript] ;
		}
    }

	return f;
}

double * convolution(double a[], double b[], int n, int n1, int n2, struct complex *table, int tableSize){
	/*
	 Computes the convolution of real vectors a and b using an FFT. 
	 
	 Input:
	 double a[]					First vector for convolution. 
	 double b[]					Second vector for convolution. 
	 int n						Length of both vectors. Must be power of two. 
	 int n1, n2					Integers such that n = n1 * n2. Each must be a power of two.  
	 struct complex *table		Table of precompted exponential factors. 
	 int tableSize				Size of table, must be at least n and a power of two. 
	 
	 Output:
	 double * (returned)		Convolution of the input vectors. 
	 */ 

	int k;
	double oneOverTwoN = 1.0 /(double) (2*n);

	double *f = (double *) malloc( 2 * n * sizeof(double)) ;
	double *aPad = (double *) malloc( 2 * n * sizeof(double)) ;
	double *bPad = (double *) malloc( 2 * n * sizeof(double)) ;
	struct complex *aRes = allocComplex(n+1);
	struct complex *bRes = allocComplex(n+1);

	struct complex *c = allocComplex(n+1);

	// copy first half of input arrays to padded arrays
	for(k=0; k<n; k++) {
		aPad[k] = a[k];
		bPad[k] = b[k];
	}
	// and zero the second half
	for(k=n; k<2*n; k++){
		aPad[k] = 0.0;
		bPad[k] = 0.0;
	}

	aRes = r_to_cFFT(aPad, 2*n, 2*n1, n2, -1, table, tableSize); // r_to_cFFT returns (input length/2 + 1) complex numbers
	bRes = r_to_cFFT(bPad, 2*n, 2*n1, n2, -1, table, tableSize);

	for(k=0; k<n+1; k++) c[k] = multComplex(aRes[k], bRes[k]) ;

	// note that actual FFT is called within this routine with half input length as length.
	// pass length as 2*n.
	c = c_to_rFFT(c, 2*n, 2*n1, n2, 1, table, tableSize);
	for(k=0; k<n; k++) c[k] = multComplexReal( c[k], oneOverTwoN) ;

	for(k=0; k<n; k++){
		f[2*k] = c[k].real;
		f[2*k + 1] = c[k].imag;
	}

	return f;
}


struct complex *** fft3D(struct complex ***x, int m, int m1, int m2, int n, int n1, int n2, int p, int p1, int p2, int sign, struct complex *table, int tableSize){
	/*
	 Three dimensional FFT. 
	 To calculate the inverse FFT, use sign = 1 and devide output by m*n*p after function call.
	 
	 Input:
	 struct complex ***x			Input array, overwritten in function. 
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
	 struct complex *** (returned)	FFT of x. 

	 */

	int j, k, l;

	// allocated temp storage large enough for any vector in any dimension
	struct complex *temp = allocComplex(max(m, max(n,p))) ;

	// transformations down each column
	for(j=0; j<n; j++){
		for(k=0; k<p; k++){

			// copy into contiguous array
			for(l=0; l<m; l++) temp[l] = x[l][j][k] ;

			temp = fourStepFFT(temp, m, m1, m2, sign, table, tableSize);

			// copy back
			for(l=0; l<m; l++) x[l][j][k] = temp[l] ;
		}
	}

	// transformations across each row
	for(j=0; j<m; j++){
		for(k=0; k<p; k++){

			// copy into contiguous array
			for(l=0; l<n; l++) temp[l] = x[j][l][k] ;

			temp = fourStepFFT(temp, n, n1, n2, sign, table, tableSize);

			// copy back
			for(l=0; l<n; l++) x[j][l][k] = temp[l] ;
		}
	}

	// take down each depth
	// this is contiguous in memory, do not copy
	for(j=0; j<m; j++){
		for(k=0; k<n; k++){

			// no need to to copy here, as this dimension is contiguous in memory
			x[j][k] = fourStepFFT(x[j][k], p, p1, p2, sign, table, tableSize);

		}
	}

	free(temp);

	return x;
}



