void mxm(int M, int N, int K, double **A, double **B, double **C) {
  register int i, ii, j, jj, k, kk;

  /*@ begin Loop(
        transform ArrCopy(aref='C[i][j]', dimsizes=[32,32], suffix='_copy', dtype='double')
        transform ArrCopy(aref='A[i][k]', dimsizes=[32,32])
        transform ArrCopy(aref='B[k][j]', dimsizes=[32,32], dtype='double')

  for (ii=0; ii<=M-1; ii++)
    for (jj=0; jj<=N-1; jj++)
      for (kk=0; kk<=K-1; kk++)
        for (i=ii; i<=min(M-1,ii+31); i++)
          for (j=jj; j<=min(N-1,jj+31); j++)
            for (k=kk; k<=min(K-1,kk+31); k++)
              C[i][j]+=A[i][k]*B[k][j];

  ) @*/

  for (ii=0; ii<=M-1; ii++)
    for (jj=0; jj<=N-1; jj++)
      for (kk=0; kk<=K-1; kk++)
        for (i=ii; i<=min(M-1,ii+31); i++)
          for (j=jj; j<=min(N-1,jj+31); j++)
            for (k=kk; k<=min(K-1,kk+31); k++)
              C[i][j]+=A[i][k]*B[k][j];

  /*@ end @*/
}
