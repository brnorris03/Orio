
/*
 Sparse matrix utilities header file.


 Alex Kaiser, LBNL, 7/2010
 */


// CSR matrix strucure
typedef struct csrFiller{
	int m;
	int n;
	int nnz;
	int *rowPtr;
	int *columnIndices;
	double *values ;
} csrMatrix;


// utility routines

// csr matrix util routines
csrMatrix allocCSRMatrix(int m, int n, int nnz) ;
void printCSRMatrix(csrMatrix A);
void freeCSRMatrixArrays(csrMatrix A);

csrMatrix getCSRfromRowColumn(int m, int n, int row[], int column[], double valuesOrig[], int origLength) ;
void quickSortTriples(int row[], int column[], double valuesOrig[], int left, int right) ;
void swapTriples(int row[], int column[], double valuesOrig[], int i, int j) ;
int compareTriple(int row[], int column[], double valuesOrig[], int i, int j) ;
char isSortedAndNoDuplicates(int row[], int column[], double valuesOrig[], int origLength) ;

// dense, debugging and I/O util
double ** toDense(csrMatrix A);
double ** toDenseFromRowColumn(int m, int n, int row[], int column[], double valuesOrig[], int nnz);
double ** allocDouble2d(int m, int n);
void freeDouble2d(double ** z, int m, int n);
void printMatrix(double ** zz, int m, int n);
void printVector(double *z, int n);
void printIntegerVector(int *z, int n);
void spyPrimitive(double **x, int m, int n);
double normDoublel2(double * x, int n);
double l2RelativeErrDouble_local(double * guess, double * trueValue, int n);

// generators
int getRandomInt_local(int maxVal) ;

typedef struct {
	double x[2];
}dd_real_local;

dd_real_local ddSub_local(dd_real_local a, dd_real_local b) ;
dd_real_local ddMult_local(double a, double b);
double expm2_local(double p, double modulus);
double bcnrand_local( );

// timer
double read_timer_local( ); 


