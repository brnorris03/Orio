
int isValid() {

  double actual = 66960.570634;

  double s_sum = 0.0;
  double q_sum = 0.0;
  double rand1=0.1, rand2=0.9;
  double expected=0.0;
  int i,j;
  double diff=0.0;



  for(i=1; i<=N-2; i++)
    for (j=1; j<=N-2; j++)
      s_sum+=A[i][j]*rand1*rand2;


  expected = s_sum;

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




