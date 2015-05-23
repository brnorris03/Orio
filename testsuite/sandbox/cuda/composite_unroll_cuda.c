void VecAXPY(int n, double a, double *x, double *y) {

  register int i;
  int n=N;


  /*@ begin Loop(transform Composite(
    unrolljam = (['i'],[8]),
    cuda = (16,True,False,1)
    )
    
  for (i=0; i<=n-1; i++)
    y[i]+=a*x[i];
    
  ) @*/

  for (i=0; i<=n-1; i++)
    y[i]+=a*x[i];

  /*@ end @*/
}
