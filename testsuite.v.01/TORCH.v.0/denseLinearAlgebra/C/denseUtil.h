



/*
 Header for utilities for dense linear algebra operations.

 Alex Kaiser, LBNL, 8/2010.
 */

typedef struct{
	double **values;
	int m;
	int n;
}matrix;

typedef struct {
	double x[2];
}dd_real;

// allocators and freer routines.
double ** allocDouble2d(int m, int n) ;
void freeDouble2d(double **z, int m, int n) ;

matrix allocAndInitZeroMatrix(int m, int n) ;
matrix allocAndInitRandomizedMatrix(int m, int n) ;
void freeMatrix(matrix A);

// timer
double read_timer( ); 

// data movement
void copyMatrix(matrix in, matrix out) ;
void swapRow(matrix A, int first, int second) ;
int compareDoubles(const void *x, const void *y) ;


// I/O
void printMatrix(matrix A) ;
void printVector(double *z, int n) ;

// norms and measurement
double maxDiffMatrix(matrix guess, matrix true) ;
double maxDiffVector(double *guess, double *true, int n) ;

// util for LU factorization
int getIndexOfMax(matrix A, int startIndex, int columnNum) ;


// random number generator
dd_real ddSub(dd_real a, dd_real b) ;
dd_real ddMult(double a, double b) ;
double expm2(double p, double modulus) ;
double bcnrand( ) ;