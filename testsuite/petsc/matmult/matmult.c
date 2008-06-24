// nsz = the number of rows of matrix v1
//  sz = the number of columns of matrix v1
//  yt = the output vector
//   x = the input vector
//  v1 = the input matrix (represented in 1-d array)
// idx = the index vector (to indicate the index position of non-zero elements stored in vector x)

int i1,i2;

for (i1=0; i1<=nsz-1; i1++) 
  {
    yt[i1] = 0;
    for (i2=0; i2<=sz-1; i2++)
      yt[i1] = yt[i1] + v1[i1*sz+i2] * x[idx[i2]];
  }

