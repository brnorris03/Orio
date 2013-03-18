
int isValid() {
  
  double actual = 1.0; 
  double x_sum = 0.0;
  double b_sum = 0.0;
  double rand1=0.1, rand2=0.9;
  double expected=0.0;
  int i1,i2;
  double diff=0.0;

  for (i1=0; i1 < N; i1++){
	x_sum+=x1[i1]*rand1*rand2;
	b_sum+=x2[i1]*rand1*rand2;
    }

  expected = x_sum/b_sum;

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




