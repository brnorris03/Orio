
int isValid() {
  
  double actual = 2793016550086296.5; // or some other user-defined computation
  double ex_sum = 0.0;
  double ey_sum = 0.0;
  double hz_sum = 0.0;
  double rand1=0.1, rand2=0.9;
  double expected=0.0;
  int i1,i2;
  double diff=0.0;

  for (i1=0; i1<nx; i1++)
    for (i2=0; i2<ny +1; i2++){
    ex_sum+=ex[i1][i2]*rand1*rand2;
    ey_sum+=ey[i1][i2]*rand1*rand2;
    hz_sum+=hz[i1][i2]*rand1*rand2;
    }

  expected = ex_sum*ey_sum*hz_sum;

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




