
void init_input_vars() {

  int i, j;
  a = 1.5;
  b = 2.5;
  nx = Nx;
  ny = Ny;
  for (i=0; i<=Nx-1; i++) {
    u1[i]=(i+1)/Nx/2.0;
    u2[i]=(i+1)/Nx/4.0;
    y[i]=(i+1)/Nx/6.0;
    w[i]=(i+1)/Nx/8.0;
    for (j=0; j<=Ny-1; j++) {
      A[i][j]=(i*j)/Ny;
      B[i][j]=0;
    }
  }
  for (j=0; j<=Ny-1; j++) {
    v1[j]=(j+1)/Ny/2.0;
    v2[j]=(j+1)/Ny/4.0;
    z[j]=(j+1)/Ny/6.0;
    x[j]=(j+1)/Ny/8.0;
  }

}
