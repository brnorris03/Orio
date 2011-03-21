
/*
 Header for complex utility functions. 
 
 Alex Kaiser, LBNL, 9/2010. 
 */ 


struct complex{
	double real;
	double imag;
}; 

// i/o
void printComplex( struct complex z);
void printComplexArray(struct complex * zz, int n);
void printComplexMatrix(struct complex ** zz, int m, int n);

// allocation and setters
struct complex newComplex(double real, double imag);
struct complex * allocComplex(int n);
struct complex ** allocComplex2d(int m, int n);
struct complex *** allocComplex3d(int m, int n, int p);
double *** allocDouble3d(int m, int n, int p);
void setComplex( struct complex *x, double real, double imag);

// free resources
void freeComplex2d(struct complex ** z, int m, int n); 
void freeComplex3d(struct complex ***z, int m, int n, int p); 
void freeDouble3d(double ***z, int m, int n, int p); 

// arithmetic
struct complex addComplex(struct complex a, struct complex b);
struct complex subComplex(struct complex a, struct complex b);
struct complex multComplex(struct complex a, struct complex b);
struct complex expComplex(struct complex a);
struct complex conjugate(struct complex z);
struct complex ** transpose( struct complex **z, int m, int n );
struct complex multComplexReal(struct complex a, double x);

// helpers
int max(int m, int n);
struct complex rmsError(struct complex *x, struct complex *y, int n);
struct complex rmsError3D(struct complex ***x, struct complex ***y, int m, int n, int p);
double l2RelativeErr(struct complex * guess, struct complex * trueValue, int n); 
double l2RelativeErrDouble(double * guess, double * trueValue, int n); 
double norm(struct complex * x, int n); 
double normDouble(double * x, int n); 

// random number generators and utils
int getRandomInt(int b);

typedef struct {
	double x[2];
}dd_real;

dd_real ddSub(dd_real a, dd_real b) ;
dd_real ddMult(double a, double b);
double expm2(double p, double modulus);
double bcnrand( );

// timer 
double read_timer( ); 

