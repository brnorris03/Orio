


/*
 Generators for dense linear algebra operations.

 Alex Kaiser, LBNL, 8/2010.
 */


matrix get1_2_1Tridiagonal(int size) ;

void getCondNumberMatrix(matrix *A, double cond, int type);

matrix getRandomOrthogonalMatrix(int n) ;
void getEigenvalueTestMatrix(int n, int type, matrix *A, double *eigs) ;
