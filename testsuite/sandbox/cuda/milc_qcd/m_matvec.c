/****************  m_matvec.c  (in su3.a) *******************************
*                                                                       *
* matrix vector multiply                                                *
*  y[i]  <-  A[i]*x[i]                                                  *
*/
void mult_su3_mat_vec(double A[], double x[], double y[], int sites_on_node) {
  register int i,j,k;
  register double ar,ai,br,bi,cr,ci;

  /*@ PerfTuning (
        def performance_params {
          param TC[]  = range(32,1025,32);
          param BC[]  = range(14,113,14);
          param UIF[] = range(1,6);
          param PL[]  = [16,48];
          param CFLAGS[] = map(join, product(['-O0', '-O1', '-O2', '-O3']));
        }
        def input_params {
          param SITES[] = [2,4,6,8,10,12,14,16];
        }
        def input_vars {
          decl dynamic double A[18*SITES] = random;
          decl dynamic double x[6*SITES]  = random;
          decl dynamic double y[6*SITES]  = 0;
        }
  ) @*/
  int sites_on_node = SITES;

  #pragma Orio Loop(transform CUDA(threadCount=TC, blockCount=BC, preferL1Size=PL, unrollInner=UIF))
  for(i=0; i<=sites_on_node-1; i++) {
    for(j=0; j<=5; j+=2) {
      cr = ci = 0.0;
      for(k=0; k<=5; k+=2) {
        ar=A[18*i+3*j+k];
        ai=A[18*i+3*j+k+1];
        br=x[6*i+k];
        bi=x[6*i+k+1];
        cr += ar*br - ai*bi;
        ci += ar*bi + ai*br;
      }
      y[6*i+j]  =cr;
      y[6*i+j+1]=ci;
    }
  }
  #pragma Oiro
  /*@ @*/
}

