#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#include "complexUtil.h"

/*
 Utility and arithmetic routines for complex numbers and data-structures. 
 
 Alex Kaiser, LBNL, 9/09
*/ 


struct complex newComplex(double real, double imag){
	// initializes a new complex number to the supplied double precision values
	// and returns it
	// Source - Harbison and Steele
	struct complex z ;
	z.real = real;
	z.imag = imag;
	return z;
}

struct complex * allocComplex(int n){
	// returns ptr to array of n double precision complex
	return (struct complex *) malloc( n * sizeof(struct complex) ) ;
}

struct complex ** allocComplex2d(int m, int n){
	// returns ptr to and (m by n) array of double precision complex
	struct complex **temp = (struct complex **) malloc( m * sizeof(struct complex *) ) ;
	int k;
	for(k=0; k<m; k++){
		temp[k] = (struct complex *) malloc( n * sizeof(struct complex));
	}
	return temp;
}

struct complex *** allocComplex3d(int m, int n, int p){
	// returns ptr to and (m by n by p) array of double precision complex
	struct complex *** temp = (struct complex ***) malloc( m * sizeof(struct complex **) ) ;
	int j,k;
	for(j=0; j<m; j++){
		temp[j] = (struct complex **) malloc( n * sizeof(struct complex *) ) ;
		for(k=0; k<n; k++){
			temp[j][k] = (struct complex *) malloc( p * sizeof(struct complex));
		}
	}
	return temp;
}

double *** allocDouble3d(int m, int n, int p){
	// returns ptr to and (m by n by p) array of double precision real
	double *** temp = (double ***) malloc( m * sizeof(double **) ) ;
	int j,k;
	for(j=0; j<m; j++){
		temp[j] = (double **) malloc( n * sizeof(double *) ) ;
		for(k=0; k<n; k++){
			temp[j][k] = (double *) malloc( p * sizeof(double));
		}
	}
	return temp;
}


void freeComplex2d(struct complex ** z, int m, int n){
	// frees ptr to an (m by n) array of double precision complex
	int k;
	for(k=0; k<m; k++){
		free( z[k] ); 
	}
	free(z) ; 
}


void freeComplex3d(struct complex ***z, int m, int n, int p){
	// frees (m by n by p) array of double precision complex
	int j,k;
	for(j=0; j<m; j++){
		for(k=0; k<n; k++){
			free( z[j][k] );  
		}
		free( z[j] );  
	}
	free(z) ; 
}

void freeDouble3d(double ***z, int m, int n, int p){
	// frees (m by n by p) array of double precision reals
	int j,k;
	for(j=0; j<m; j++){
		for(k=0; k<n; k++){
			free( z[j][k] );  
		}
		free( z[j] );  
	}
	free(z) ; 
}

void setComplex( struct complex *x, double real, double imag){
	// sets the value of the *x to the supplied double precision values
	// does not allocate a new complex number
	x->real = real;
	x->imag = imag;
}

void printComplex( struct complex z){
	// prints a complex number to stdout in C++ format
	// ( real, imag)
	printf("(%.20f, %.20f) ", z.real, z.imag);
}

struct complex addComplex(struct complex a, struct complex b){
	// adds two complex numbers
	struct complex sum;
	sum.real = a.real + b.real;
	sum.imag = a.imag + b.imag;
	return sum;
}

struct complex subComplex(struct complex a, struct complex b){
	// computes difference a-b
	struct complex diff;
	diff.real = a.real - b.real;
	diff.imag = a.imag - b.imag;
	return diff;
}


struct complex multComplex(struct complex a, struct complex b){
	// source - Harbison and Steele
	// multiplies two complex numbers
	// naive, 4 multiply, 2 add algorithm
	struct complex product;
	product.real = a.real*b.real - a.imag*b.imag ;
	product.imag = a.real*b.imag + a.imag*b.real ;
	return product;
}


struct complex expComplex( struct complex a){
	// computes complex exponential e^(i * a) = cos(a) + i*sin(a)
	// input must be a purely imaginary complex number
	if( a.real != 0){
		printf("errors. must call expComplex with 0 real part!\n");
	}
	return newComplex( cos(a.imag), sin(a.imag));
}

struct complex multComplexReal(struct complex a, double x){
	// multiplies complex a by double precision x
	struct complex product;
	product.real = a.real * x;
	product.imag = a.imag * x;
	return product;
}

int max(int m, int n){
	// simple integer comparison for max
	if (m > n) return m;
	return n;
}

struct complex ** transpose( struct complex **z, int m, int n ){
	// performs transpose of 2D complex array z
	// m, n - dimensions of input
	
	// dimensions must be sufficient before calling. No new space is allocated.
	// may want to optimize
	
	int j,k;
	struct complex temp;
	
	for(j=0; j<m; j++){
		for(k=j+1; k<n; k++){
			temp = z[j][k];
			z[j][k] = z[k][j];
			z[k][j] = temp;
		}
	}
	
	return z;
}

struct complex conjugate(struct complex z){
	// returns complex conjugate of z
	z.imag = -z.imag;
	return z;
}


void printComplexMatrix(struct complex ** zz, int m, int n){
	// outputs complex 2D array of dimensions (m by n)
	int j,k;
	for(j=0; j<m; j++){
		for(k=0; k<n; k++){
			printComplex(zz[j][k]);
		}
		printf("\n");
	}
	printf("\n");
}

void printComplexArray(struct complex * zz, int n){
	// outputs complex array of dimension n
	int k;
	for(k=0; k<n; k++){
		printComplex(zz[k]);
		printf("\n");
	}
	printf("\n");
}

struct complex rmsError(struct complex *x, struct complex *y, int n){
	// returns rms error between two complex vectors x and y of length n
	double errReal = 0.0;
	double errImag = 0.0;
	
	int k;
	for(k=0; k<n; k++){
		errReal += pow( x[k].real - y[k].real, 2);
		errImag += pow( x[k].imag - y[k].imag, 2);
	}
	
	errReal = sqrt( errReal / (double) n);
	errImag = sqrt( errImag / (double) n);
	
	return newComplex(errReal, errImag);
}

struct complex rmsError3D(struct complex ***x, struct complex ***y, int m, int n, int p){
	// returns rms error between two complex vectors x and y
	// of size (m by n by p)
	double errReal = 0.0;
	double errImag = 0.0;
	
	int k, j, l;
	for(j=0; j<m; j++){
		for(k=0; k<n; k++){
			for(l=0; l<p; l++){
				errReal += pow( x[j][k][l].real - y[j][k][l].real, 2);
				errImag += pow( x[j][k][l].imag - y[j][k][l].imag, 2);
			}
		}
	}
	
	errReal = sqrt( errReal / (double) n);
	errImag = sqrt( errImag / (double) n);
	
	return newComplex(errReal, errImag);
}


double norm(struct complex * x, int n){
	// l2 norm of complex double precision vector x
	double runningNorm = 0.0; 
	struct complex temp; 
	int j; 
	
	for (j=0; j<n; j++) {
		temp = multComplex( x[j], conjugate(x[j]) );
		runningNorm += temp.real; 
	}
	
	return sqrt(runningNorm); 
}


double l2RelativeErr(struct complex * guess, struct complex * trueValue, int n){
	// computes l2 relative error between two complex vectors. 
	
	int j; 
	
	struct complex * diffs = allocComplex(n); 
	
	for (j=0; j<n; j++) {
		diffs[j] = subComplex( guess[j], trueValue[j] ) ; 
	}
	
	double normTrue = norm(trueValue,n); 
	double normDiffs = norm(diffs, n); 
	
	// if true norm is equal, then difference must be exactly zero. 
	// else return NaN
	if(normTrue == 0.0 ){
		if(normDiffs == 0.0)
			return 0.0; 
		else
			return 0.0/0.0; 
	}
	
	return normDiffs / normTrue; 
}

double normDouble(double * x, int n){
	// l2 norm of double precision vector x
	double runningNorm = 0.0; 
	int j; 
	
	for (j=0; j<n; j++)
		runningNorm += x[j] * x[j] ; 
	
	return sqrt(runningNorm); 
}


double l2RelativeErrDouble(double * guess, double * trueValue, int n){
	// computes l2 relative error between two real vectors. 
	
	int j; 
	
	double * diffs = (double *) malloc(n * sizeof(double)); 
	
	for (j=0; j<n; j++)
		diffs[j] = guess[j] - trueValue[j] ; 
	
	double normTrue = normDouble(trueValue,n); 
	double normDiffs = normDouble(diffs, n); 
	
	// if true norm is equal, then difference must be exactly zero. 
	// else return NaN
	if(normTrue == 0.0 ){
		if(normDiffs == 0.0)
			return 0.0; 
		else
			return 0.0/0.0; 
	}
	
	return normDiffs / normTrue; 
}


int getRandomInt(int b){
	// returns random int in range [0, 2^b - 1]
	// uses bcnrand()
	int maxVal = (int) pow(2,b);
	return floor( maxVal * bcnrand()) ;
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
