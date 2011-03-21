
/*
 Header for arithmetic routines for sparse matrix operations.

 Alex Kaiser, LBNL, 2010
 */


// arithmetic routines
void spmv(csrMatrix A, double * x, double *y) ;
double * spts(csrMatrix A, double *b) ;
double * conjugateGradient(csrMatrix A, double *b, double *guess, double eps, int maxIt) ;
void matrixPowers(csrMatrix A, int k, double **x) ;


// dense support and debugging routines
double * denseMV(double ** dense, int m, int n, double * x) ;
double innerProduct(double x[], double y[], int n);
void vectorPlusConstantByVector(double toReturn[], double x[], double alpha, double y[], int n);

