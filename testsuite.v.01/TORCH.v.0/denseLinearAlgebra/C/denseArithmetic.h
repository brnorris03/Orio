


/*
 Header for arithmetic routines for dense linear algebra operations.

 Alex Kaiser, LBNL, 8/2010.
 */



void LUFactorize(matrix A, matrix *L, matrix *U) ;

double * symmetricQR(matrix A) ;
matrix symmetricQRStep(matrix A, int startIndex, int stopIndex) ;
matrix applyGivens(matrix A, double a, double b, int i, int k, int startIndex, int stopIndex) ;
matrix tridiagonalize(matrix A) ;
double householder(matrix A, int startIndex, int columnNum, double *v) ;



// blas type operations
void matrixVectorMultiply(matrix A, int startIndexRow, int startIndexColumn, double *v, double constant, double *out) ;
void vectorPlusConstantByVector(double toReturn[], double x[], double alpha, double y[], int n) ;
void matrixPlusConstantByMatrix(matrix *toReturn, matrix A, double alpha, matrix B) ;
void elementWiseVectorProduct(double x[], double y[], int n, double toReturn[]) ;
void scalarVectorProduct(double alpha, double x[], int n, double toReturn[]) ;
double innerProduct(double x[], double y[], int n) ;
void outerProduct(matrix *toReturn, double *column, int lengthCol, double *row, int lengthRow) ;

void matrixMatrixMultiply(matrix A, matrix B, matrix *res) ;
void transpose(matrix in, matrix *out) ;


int sign(double d) ;
