
/* pluto start (nx,ny) */
for (i = 0; i <= ny-1; i++)
  s[i] = 0;

for (i = 0; i <= nx-1; i++) {
  q[i] = 0;
  for (j = 0; j <= ny-1; j++) {
    s[j] = s[j] + r[i]*A[i][j];
    q[i] = q[i] + A[i][j]*p[j];
  }
 }
/* pluto end */

