/*@ begin Loop(
  transform Composite(
    tile = [('i',16,'ii'),('j',32,'jj'),('k',64,'kk'),
            ('ii',32,'iii'),('jj',64,'jjj'),('kk',128,'kkk')]
  )
  for (i=0; i<=nx-1; i++) {
    tmp[i] = 0;
    for (j=0; j<=ny-1; j++)
      tmp[i] += A[i*ny+j]*x[j];
    for (k=0; k<=ny-1; k++)
      y[k] += A[i*ny+k]*tmp[i];
  }
) @*/

/*@ end @*/
