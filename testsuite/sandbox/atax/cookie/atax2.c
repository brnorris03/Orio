
/* pluto start (nx,ny) */
for (i= 0; i<=ny-1; i++)
  y[i] = 0.0;
for (i = 0; i<=nx-1; i++) {
  tmp[i] = 0;
  for (j = 0; j<=ny-1; j++) 
    tmp[i] = tmp[i] + A[i][j]*x[j];
  for (j = 0; j<=ny-1; j++) 
    y[j] = y[j] + A[i][j]*tmp[i];
 }
/* pluto end */
