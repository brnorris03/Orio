
int isValid() {
  
  double actual = 154057935.585664;

  double ax_sum = 0.0;
  double rand1=0.1, rand2=0.9;
  double expected=0.0;
  double diff=0.0;
  int i, j;

  for (i=0; i<T; i++) 
    for (j=0; j<N; j++) 
      ax_sum+=a[i][j]*rand1*rand2;

  expected = ax_sum;

  diff=abs(expected-actual);

  //printf("expected=%f\n",expected);
  //printf("actual=%f\n",actual);
  //printf("diff=%f\n",diff);
  //printf("diff=%d\n",(diff < 0.00000001));

  if (diff < 0.00000001)
    return 1;
  else
    return 0;

}




