
int i,j;

for (i=0; i<=m-1; i++) 
  {
    y[i] = 0.0;
    for (j=ii[i]; j<=ii[i+1]; j++)
      y[i] = y[i] + aa[j] * x[aj[j]];
  }

