for (t=0; t<=T-1; t++) 
  {
    for (i=1; i<=N-2; i++)
      for (j=1; j<=N-2; j++)
	for (k=1; k<=N-2; k++)
	  b[i][j][k] = f1*a[i][j][k] + f2*(a[i+1][j][k] + a[i-1][j][k] + a[i][j+1][k]
					   + a[i][j-1][k] + a[i][j][k+1] + a[i][j][k-1]);
    for (i=1; i<=N-2; i++)
      for (j=1; j<=N-2; j++)
	for (k=1; k<=N-2; k++)
	  a[i][j][k] = b[i][j][k];
  }
