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
for (iii=0; iii<=nx-1; iii=iii+32) 
  for (ii=iii; ii<=min(nx-1,iii+16); ii=ii+16) 
    for (i=ii; i<=min(nx-1,ii+15); i=i+1) {
      tmp[i]=0;
      for (jjj=0; jjj<=ny-1; jjj=jjj+64) 
        for (jj=jjj; jj<=min(ny-1,jjj+32); jj=jj+32) 
          for (j=jj; j<=min(ny-1,jj+31); j=j+1) 
            tmp[i]=tmp[i]+A[i*ny+j]*x[j];
      for (kkk=0; kkk<=ny-1; kkk=kkk+128) 
        for (kk=kkk; kk<=min(ny-1,kkk+64); kk=kk+64) 
          for (k=kk; k<=min(ny-1,kk+63); k=k+1) 
            y[k]=y[k]+A[i*ny+k]*tmp[i];
    }
/*@ end @*/
