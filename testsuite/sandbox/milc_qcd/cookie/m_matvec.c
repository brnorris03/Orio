/****************  m_matvec.c  (in su3.a) *******************************
*                                                                       *
* matrix vector multiply                                                *
*  y[i]  <-  A[i]*x[i]                                                  *
*/
/* void mult_su3_mat_vec(double A[], double x[], double y[]) { */
  /*@ begin PerfTuning (
        def build {
          arg build_command = 'icc @CFLAGS';
        }
        def performance_params {
          param U1[] = [1]+range(2,10);
          param U2[] = [1]+range(2,10);
          param SREP[] = [False,True];
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
        def performance_counter {
          arg method = 'basic timer';
          arg repetitions = 6;
        }
  ) @*/

  register int i,j,k,jt,kt;
  register double ar,ai,br,bi,cr,ci;
  int sites_on_node=SITES;

  /*@ begin Loop(
  transform Composite(scalarreplace = (SREP, 'double'))
  transform RegTile(loops=['j','k'], ufactors=[U1,U2])

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

  ) @*/

  for(i=0; i<=sites_on_node-1; i++) {
    for(j=0; j<6; j+=2) {
      cr = ci = 0.0;
      for(k=0; k<6; k+=2) {
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
  /*@ end @*/
  /*@ end @*/
/* } */
