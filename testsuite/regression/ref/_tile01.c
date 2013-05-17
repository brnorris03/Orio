
/*@ begin Loop(
  transform Tile(tsize=32, tindex='ii')
  for (i=0; i<=m-1; i++)
    transform Tile(tsize=32, tindex='jj')
    for (j=0; j<=n-1; j++)
      B[i][j]=A[i][j];
) @*/
for (ii=0; ii<=m-1; ii=ii+32) 
  for (i=ii; i<=min(m-1,ii+31); i=i+1) 
    for (jj=0; jj<=n-1; jj=jj+32) 
      for (j=jj; j<=min(n-1,jj+31); j=j+1) 
        B[i][j]=A[i][j];
/*@ end @*/
