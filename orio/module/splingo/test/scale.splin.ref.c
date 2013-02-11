void SCALE(int n, double alpha, double* x, double* y) {
  int i;
  for (i=0; i<n; i++ ) 
    y[i]=alpha*x[i];
}
