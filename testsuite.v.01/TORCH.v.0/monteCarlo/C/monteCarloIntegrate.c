#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define DIMENSION 10

/* 
 Computes high dimensional integrals using a quasi-Monte Carlo method. 
 Method described in Crandall and Pomerance's "Prime Numbers" 

 Alex Kaiser, LBNL, 10/2009
*/ 


// globals
unsigned int primes[10] = {2,3,5,7,11,13,17,19,23,29} ;  // higher dimension than length of this array will BREAK this code.
int K[DIMENSION] ;



// headers

// random generation and integration
double * seed(int n, int Nmax, int dimension, double *x, int *** d, double ***q);
double * mcRandom(double x[], int ***d, double **q, int dimension);
double qMCintegrate( double integrand(double *x, double s), int seedValue, int points, int dimension, double s); 
double boxIntegrand(double *x, double s);
double boxIntegralValue(int n, double s); 

// allocation and freeing 
double ** allocDouble2d(int m, int n);
void freeDouble2d(double **z, int m, int n);
int ** allocInt2d(int m, int n);
void freeInt2d(int **z, int m, int n); 

// timer
double read_timer( );

//helpers
int maxValue(int K[], int length);




int main(){
	/*
	 Computes the value of a Box integral using quasi-Monte Carlo integration. 
	 Also computes the result using an infinite series with provable bounds. 
	 Details of the integrals and series are described in the tech report. 
	 Compares the two results, taking the series result as true. 
	 
	 Parameters:
		#define DIMENSION		Dimension of the function to integrate. 
		double s				Argument for box integral integrand function. 
		double tol				Tolerance for relative error in results. 
		int seedValue			Position in sequence at which to start random numbers. 
									Use in parallel compurations to obtain later portions 
									of the random number sequence. 
									Default = 0.
		int points				Number of points at which to evaluate integrand. 
	 */ 

	
	printf("Quasi-Monte Carlo integration test:\n");
	
	int n = DIMENSION;
	double s = 1.0; 
	double tol = 10e-6;
	
	// test and call integrate routine
	int seedValue = 0;
	double result, seriesResult ;
	int points = 10000000 ;

	double startTime, endTime; 
	startTime = read_timer(); 
	
	result = qMCintegrate( boxIntegrand, seedValue, points, DIMENSION, s) ;
	
	endTime = read_timer(); 
	printf("Elapsed time = %f seconds.\n", endTime - startTime) ;
	
	printf("result = %.15f\n", result);

	seriesResult = boxIntegralValue(n, s) ;
	printf("series result = %.15f\n", seriesResult);

	double relErr ;
	
	if(seriesResult == 0.0){ 
		if(result == 0.0)
			relErr = 0.0 ; 
		else 
			relErr = 1.0 / 0.0 ; 
	}
	else
		relErr = fabs(result - seriesResult) / fabs(seriesResult) ;
	
	
	printf("relative error = %.10f\n", relErr) ;

	if(relErr < tol)
		printf("Quasi-Monte Carlo integration test passed.\n\n\n");
	else
		fprintf(stderr, "Relative error must be less that specified tolerance.\nQuasi-Monte Carlo integration test failed.\n\n\n") ;

	return (relErr < tol);
}

double * seed(int n, int Nmax, int dimension, double *x, int *** d, double ***q){
	/*
	 Calculate the seed for random number generator.
	 Generates vectors of random points. 
	 
	 Input: 
		int n					Starting value of integers to generate. 
		int Nmax				Maximum number of points to generate. 
		int dimension			Dimension of random numbers to generate. 
		double *x				Starting random vector. 
		double ***d				Address of 2d array for "odometer constants". 
									Allocated here. 
		double ***q				Address of 2d array for inverses of integer powers. 
									Allocated here. 
	 
	 Output:
		double *x (returned)	Current random vector. 
		double ***d				Address of 2d array for "odometer constants".
									'fake returned' using pointers. 
		double ***q				Address of 2d array for inverses of integer powers.
									'fake returned' using pointers.  
	 */ 
	
	

	int i, j;

	unsigned int kk;

	// assign values to K
	for(i=0; i<dimension; i++){
		K[i] = ceil( log(Nmax + 1) / log(primes[i] )) ;
	}

	int **tempD = allocInt2d(dimension, maxValue(K, dimension)) ;
	double **tempQ = allocDouble2d(dimension, maxValue(K, dimension)) ;

	for(i=0; i< dimension; i++){

		// K values pre-computed
		kk = n;
		x[i] = 0;  

		for(j=0; j< K[i]; j++){
			if(j == 0){
				tempQ[i][j] = 1.0 / (double) primes[i] ;
			}
			else{
				tempQ[i][j] = tempQ[i][j-1] / (double) primes[i] ;
			}
			tempD[i][j] = kk % primes[i] ;
			kk = (kk - tempD[i][j]) / (double) primes[i] ;   
			x[i] += tempD[i][j] * tempQ[i][j] ;
		}

	}

	// fake returns by adding a passing these one reference level down 
	*d = tempD; 
	*q = tempQ;

	return x;
}


double * mcRandom(double x[], int ***d, double **q, int dimension){
	/*
	 Calculate the seed for random number generator.
	 Generates vectors of random points. 
	 
	 Input: 
		double x[]					Current random vector. 
		double ***d					Address of 2d array for "odometer constants".
										Modified and 'fake returned.'
		double **q					2d array of inverses of integer powers. 
		int dimension				Dimension of random numbers to generate.
	 
	 Output:
		double x[] (returned)		Current random vector. 
		double ***d					Address of 2d array for "odometer constants".
										'fake returned' using pointers. 
	 */ 

	
	int i, j ;

	for(i=0; i<dimension; i++){
		for(j=0; j<K[i]; j++){
			(*d)[i][j] ++ ;
			x[i] += q[i][j] ;

			if( (*d)[i][j] < primes[i]) break ;

			(*d)[i][j] = 0;
			if(j == 0){
				x[i] -- ;
			}
			else{
				x[i] -= q[i][j-1] ;
			}
		}
	}

	return x;
}


double qMCintegrate(double integrand(double *x, double s), int seedValue, int points, int dimension, double s){
	/* 
	 Integrates the supplied function on the unit cube of the specified dimension. 
	 Uses quasi-monte Carlo integration
	 
	 Input:
			double	integrand(double *x, double s)
								Integrand function. 
								Evaluates at x, a 'dimension' length double precision array.
								s - double precision parameter for integration function 
			int		seedValue	First index of random sequence to use. 	
			int		points		Number of pointes for which to evaluate function.
			int		dimension	Dimension of problem. 
			double	s			Parameter for integration function. 
	 
	 Output:
			double (returned)	Integral of integrand on 'dimension' dimensional unit cube. 
	 
	 */ 
	
	double *x = (double *) malloc(dimension * sizeof(double) ) ;
	int **d;
	double **q;

	x = seed(seedValue, seedValue + points + 1, dimension, x, &d, &q);

	double result = 0.0;

	int j;

	for(j=0; j<points; j++){
		x = mcRandom(x, &d, q, dimension) ;
		result += integrand(x, s);
	}
	
	// free resources
	free(x);
	freeInt2d(d, dimension, maxValue(K, dimension)); 
	freeDouble2d(q, dimension, maxValue(K, dimension)); 

	return result / (double) points ;
}


double boxIntegrand(double *x, double s){
	
	/* 
	 Computes the value of the integrand of B_DIMENSION(s).
	 See sources for additional information on this family of functions
	
	 Input:
			double *x				Point at which to evaluate function. 
									Length of vector = DIMENSION 
			double s				Parameter for function
	 
	 Output:
			double (returned)		Value of the function
	*/ 

	double val = 0.0;
	int i;
	for(i=0; i<DIMENSION; i++){
		val += x[i]*x[i];
	}

	return pow(val, s/2.0);
}


double boxIntegralValue(int n, double s){
	/* 
	 Crandall's Box Sum scheme. 
	 Computes B_n(s) via evaluation of an infinite series. 
	
	 See Richard Crandall's "Theory of Box Series"
	 
	 Input:
			int n					Dimension to evaluate
			double s				Parameter for function
	 
	 Output:
			double (returned)		Value of the function
	 */ 
	
	int k; 
	int mu; 
	int m = n-1; 
	double tol = 1e-16;
	
	double A = 1.0; 
	double p = 1.0; 
	
	double * gammas = (double *) malloc(m * sizeof(double)) ; 
	for (k=0; k<m; k++) 
		gammas[k] = 1.0; 

	double t = 2.0 / (double) n ; 

	double sigma ; 

	for (k=1; ; k++) {
		sigma = k - 1.0 - s/2.0 ; 
		gammas[0] *= sigma / (double) (1 + 2*k) ; 
		
		// algorithm statement is "one" indexed
		// add one to all numerical values of mu, but use indices as is
		for (mu=1; mu<m; mu++) 
			gammas[mu] = (sigma * gammas[mu] + gammas[mu-1]) / (1.0 + 2.0 * (double) k / (mu+1) ) ;
		
		p *= t; 
		A += gammas[m-1] * p ; 
		
		if(fabs(gammas[m-1]*p) < tol) {
			break; 
		}
	}

	return A * pow(n, 1.0 + s/2.0) / (double) (s + n) ; 
}



double ** allocDouble2d(int m, int n){
	/* 
	 Returns a pointer to an m by n array of double precision reals. 
	 
	 Input: 
			int m					Vertical dimension. 
			int n					Horizontal dimension. 
	 
	 Output:
			double ** (returned)	Pointer to m x n double precision array. 
	 */ 
	
	double **temp = (double **) malloc( m * sizeof(double *) ) ;
	int k;
	for(k=0; k<m; k++){
		temp[k] = (double *) malloc( n * sizeof(double));
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


int ** allocInt2d(int m, int n){
	/* 
	 Returns a pointer to an m by n array of ints. 
	 
	 Input: 
			int m					Vertical dimension. 
			int n					Horizontal dimension. 
	 
	 Output:
			double ** (returned)	Pointer to m x n int array. 
	 */ 
	int **temp = (int **) malloc( m * sizeof(int *) ) ;
	int k;
	for(k=0; k<m; k++){
		temp[k] = (int *) malloc( n * sizeof(int));
	}
	return temp;
}


void freeInt2d(int **z, int m, int n){
	/*
	 Frees (m by n) array of ints.
	 
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


int maxValue(int K[], int length){
	/* 
	 Returns max value in int array K of specified length
	 
	 Input: 
			int k[]				Array to evaluate max from 
			int length			Length of array
	 
	 Output:
			int	(returned)		Maximum value contained in array K
	 */ 
	
	int i;
	int maxSoFar = -1;

	for(i=0; i<length; i++){
		if(K[i] > maxSoFar) maxSoFar = K[i];
	}

	return maxSoFar;
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


