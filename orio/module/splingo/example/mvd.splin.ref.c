void MV(int dim1, int dim2, double* A, double* x, double* y) {
  int i1,i2;
  register double tmp1;
  for (i2=0; i2<dim2; i2++ ) {
    tmp1=0.0;
    for (i1=0; i1<dim1; i1++ ) 
      tmp1+=A[dim1*i2+i1]*x[i1];
    y[i2]=tmp1;
  }
}
