void SCALE(int dim1, double alpha, double* x, double* y) {
  int i1;
  for (i1=0; i1<dim1; i1++ ) 
    y[i1]=alpha*x[i1];
}
