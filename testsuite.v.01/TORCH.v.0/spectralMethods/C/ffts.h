

/*
 FFT algorithms and associated other routines. 
 
 Alex Kaiser, LBNL, 9/2009
 */ 


// helpers
struct complex *initTable(int tableSize);


// fft algorithms
struct complex * stockhamFFT( struct complex *x, int n, int sign, struct complex *table, int tableSize);
struct complex * cooleyTukeyFFT(struct complex *x, int n, int sign, struct complex *table, int tableSize);
struct complex * fourStepFFT( struct complex *x, int n, int n1, int n2, int sign, struct complex *table, int tableSize);

// real to complex and complex to real ffts
struct complex * r_to_cFFT(double x[], int n, int n1, int n2, int sign, struct complex *table, int tableSize);
struct complex * c_to_rFFT(struct complex *x, int n, int n1, int n2, int sign, struct complex *table, int tableSize);

// convolution of real data
double * convolution(double a[], double b[], int n, int n1, int n2, struct complex *table, int tableSize);
double * naiveConvolution(double a[], double b[], int n, int numToCompute);

// 3D fft
struct complex *** fft3D(struct complex ***x, int m, int m1, int m2, int n, int n1, int n2, int p, int p1, int p2, int sign, struct complex *table, int tableSize);



