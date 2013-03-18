
int isValid() {

  double actual = 0.111111; 
  double s_sum = 0.0;
  double q_sum = 0.0;
  double rand1=0.1, rand2=0.9;
  double expected=0.0;
  int k,j;
  double diff=0.0;

    for (k = 1; k <= m-1; k++)
    {
      for (j = k+1; j <= m; j++)
	{
	  s_sum=s_sum+(symmat[k][j]*rand1);
	  q_sum=q_sum+(symmat[k][j]*rand2);
	}
    }

  expected = s_sum/q_sum;

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




