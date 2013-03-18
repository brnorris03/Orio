
int isValid() {
  
  double actual = 9369846271501.798828; // or some other user-defined computation
  double y_sum = 0.0;
  double rand1=0.1, rand2=0.9;
  double expected=0.0;
  double diff=0.0;
  int i;

  for (i= 0; i<=ny-1; i++)
    y_sum+=y[i]*rand1*rand2;


  expected = y_sum;

  diff=abs(expected-actual);

  fprintf(stderr,"expected=%f\n",expected);
  fprintf(stderr,"actual=%f\n",actual);
  //printf("diff=%f\n",diff);
  //printf("diff=%d\n",(diff < 0.00000001));

  if (diff < 0.00000001)
    return 1;
  else
    return 0;

}




