
int isValid() {

  double actual = 3742936.111145; 
  double w_sum = 0.0;
  double rand1=0.1, rand2=0.5, rand3=0.9;
  double expected=0.0;
  int ii,jj;
  double diff=0.0;


  for (ii=0; ii<N-1; ii++) {
      for (jj=0; ii<N-1; ii++) {
	w_sum+=Y[ii][jj]*rand1*rand2;
    }
  }
  expected = w_sum;

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




