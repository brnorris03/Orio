
void axpy4(int* _n, double *y, double* _a1, double *x1, double* _a2, double *x2, 
	   double* _a3, double *x3, double* _a4, double *x4) {

#pragma disjoint (*x1,*x2,*x3,*x4,*y) 

    int n = *_n;
    double a1 = *_a1;
    double a2 = *_a2;
    double a3 = *_a3;
    double a4 = *_a4;
    

}
