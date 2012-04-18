void axpy1(int n, double *y, double a1, double *x1)
{
register int i;


/*@ begin Loop(
  transform Composite(
    cuda = (16,False, False, 1)
    ,scalarreplace = (False, 'int')
, unrolljam = (['i'], [2])
  )
   {
    for (i=0; i<=n-1; i++) {
    	y[i]=y[i]+a1*x1[i];
    }
    
   }


   
  
) @*/
/*@ end @*/
}

