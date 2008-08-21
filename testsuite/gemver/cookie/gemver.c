
/* pluto start (n,a,b) */
for (i = 0; i <= n-1; i++) {
  for (j = 0; j <= n-1; j++) {
    B[i][j] = A[i][j] + u1[i]*v1[j] + u2[i]*v2[j];
  }
 }

for (i = 0; i <= n-1; i++)
  x[i] = 0;

for (i = 0; i <= n-1; i++) {
  for (j = 0; j <= n-1; j++) {
    x[j] = x[j] + y[i]*B[i][j];
  }
 }

for (i = 0; i <= n-1; i++) {
  x[i] = z[i] + b*x[i];
 }

for (i = 0; i <= n-1; i++)
  w[i] = 0;

/* pluto end */

for (i = 0; i <= n-1; i++) {
  for (j = 0; j <= n-1; j++) {
    w[i] = w[i] + B[i][j]*x[j];
  }
  w[i] = a*w[i];
 }

