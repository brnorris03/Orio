#include <algorithm>

void axpy4(int n, double *y, double a1, double *x1, double a2, double *x2, double a3, double *x3,
           double a4, double *x4) {


 int i, ii;

    
/*@ begin Loop (
  transform Composite(
  	tile = [('i',4,'ii')]
  )
  transform Unroll(ufactor=2, parallelize = True)
    for (i=0; i<=N-1; i++)
      y[i]=y[i]+a1*x1[i]+a2*x2[i]+a3*x3[i]+a4*x4[i];
      ) @*/

    for (i=0; i<=N-1; i++)
        y[i]=y[i]+a1*x1[i]+a2*x2[i]+a3*x3[i]+a4*x4[i];

/*@ end @*/

}

