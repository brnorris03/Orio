
int isValid() {

  double actual = 675000000.0; 
  double t2_sum = 0.0;
  double t6_sum = 0.0;
  double t10_sum = 0.0;
  double rand1=0.1, rand2=0.5, rand3=0.9;
  double expected=0.0;
  int ii,jj;
  double diff=0.0;


  for (ii=0; ii<M; ii++) {
	t2_sum+=t2[ii]*rand1;
	t6_sum+=t6[ii]*rand2;
	t10_sum+=t10[ii]*rand3;
    }

  expected = t2_sum+t6_sum+t10_sum;

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




