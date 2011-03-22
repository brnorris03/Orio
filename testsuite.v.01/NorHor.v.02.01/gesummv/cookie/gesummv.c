
int i,j;

for (i=0; i<=n-1; i++) {
  tmp[i] = 0;
  y[i] = 0;
  for (j=0; j<=n-1; j++) {
    tmp[i] = A[i][j]*x[j] + tmp[i];
    y[i] = B[i][j]*x[j] + y[i];
  }
  y[i] = a*tmp[i] + b*y[i];
 }
