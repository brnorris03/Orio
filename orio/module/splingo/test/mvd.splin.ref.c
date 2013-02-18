void MV(int dim1, int dim2, double* A, double* x, double* y) {
  int i1,i2;
  for (i2=0; i2<dim2; i2++ ) 
    for (i1=0; i1<dim1; i1++ ) 
      y[i1]=A[dim1*i2+i1]*x[i1];
}
