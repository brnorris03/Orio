
int isValid() {
  
  double actual = -301105.392481; // 5821.103474; // or some other user-defined computation
  double x_sum = 0.0;
  double b_sum = 0.0;
  double rand1=0.1, rand2=0.9;
  double expected=0.0;
  int i1,i2;
  double diff=0.0;

  for (i1=0; i1<N; i1++)
    for (i2=0; i2<N +20; i2++){
	x_sum+=X[i1][i2]*rand1*rand2;
	b_sum+=B[i1][i2]*rand1*rand2;
    }

  expected = x_sum/b_sum;

  diff=abs(expected-actual);

  //printf("expected=%f\n",expected);
  //printf("actual=%f\n",actual);
  //printf("diff=%f\n",diff);
  //printf("diff=%d\n",(diff < 0.00000001));

  if (diff < 1.0/1000.0)
    return 1;
  else
    return 0;

}




