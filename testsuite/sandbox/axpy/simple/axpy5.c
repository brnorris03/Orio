
register int i;

/*@ begin Loop ( 
    transform Composite(
      unrolljam = (['i'],[UF]),
      vector = (VEC, ['ivdep','vector always'])
     )
  for (i=0; i<=N-1; i++)
    y[i]=y[i]+a1*x1[i]+a2*x2[i]+a3*x3[i]+a4*x4[i]+a5*x5[i];
) @*/

/*@ end @*/

