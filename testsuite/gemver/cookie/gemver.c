
/* pluto start (nx,ny,a,b) */

for (i=0; i<=nx-1; i++) 
  for (j=0; j<=ny-1; j++) 
    B[i][j] = A[i][j] + u1[i]*v1[j] + u2[i]*v2[j];

for (j=0; j<=ny-1; j++) 
  x[j] = 0;

for (i=0; i<=nx-1; i++) 
  for (j=0; j<=ny-1; j++) 
    x[j] = x[j] + y[i]*B[i][j];

for (j=0; j<=ny-1; j++) 
  x[j] = b*x[j] + z[j];
    
for (i=0; i<=nx-1; i++) {
  w[i] = 0;
  for (j=0; j<=ny-1; j++) 
    w[i] = w[i] + B[i][j]*x[j];
  w[i] = a*w[i];
 }

/* pluto end */
