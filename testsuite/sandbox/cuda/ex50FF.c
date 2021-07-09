void FormFunction2DMDOF(double GRASHOF, double PRANDTL, double LID, int M, int N, double* x, double *f){
  register int i;
  double u,uxx,uyy,vx,vy,avx,avy,vxp,vxm,vyp,vym;

  /*@ begin PerfTuning(
        def performance_params {
          param TC[] = range(32,1025,32);
          param BC[] = range(14,113,14);
          param SC[] = range(1,6);
          param PL[] = [16,48];
          param CFLAGS[] = ['', '-O1', '-O2', '-O3'];
        }
        def build {
          arg build_command = 'nvcc -arch=sm_75 @CFLAGS';
        }
        def input_params {
          param M[] = [4,8,16,32];
          param N[] = [4,8,16,32];
          constraint c1 = (M==N);
        }
        def input_vars {
          decl dynamic double x[M*N*4] = random;
          decl dynamic double f[M*N*4] = 0;
          decl double GRASHOF = random;
          decl double PRANDTL = random;
          decl double LID     = random;
        }
        def performance_counter {
          arg method = 'basic timer';
          arg repetitions = 6;
        }
  ) @*/

  double grashof = GRASHOF;
  double prandtl = PRANDTL;
  double lid     = LID;
  int m     = M;
  int n     = N;
  int nrows = m*n*4;
  double dhx = m-1;
  double dhy = n-1;
  double hx = 1.0/dhx;
  double hy = 1.0/dhy;
  double hxdhy = hx*dhy;
  double hydhx = hy*dhx;

  /*@ begin Loop(transform CUDA(threadCount=TC, blockCount=BC, streamCount=SC, preferL1Size=PL)

  for(i=0; i<=nrows-1; i+=4){
    if(i<4*m){
      f[i]   = x[i];
      f[i+1] = x[i+1];
      f[i+2] = x[i+2] + (x[i+4] - x[i])*dhy;
      f[i+3] = x[i+3] - x[i+7];
    }else
    if(i>=nrows-4*m){
      f[i]   = x[i] - lid;
      f[i+1] = x[i+1];
      f[i+2] = x[i+2] + (x[i] - x[i-4])*dhy;
      f[i+3] = x[i+3] - x[i-1];
    }else
    if(i%(4*m)==0){
      f[i]   = x[i];
      f[i+1] = x[i+1];
      f[i+2] = x[i+2] - (x[i+5] - x[i+1])*dhx;
      f[i+3] = x[i+3];
    }else
    if(i%(4*m)==4*m-4){
      f[i]   = x[i];
      f[i+1] = x[i+1];
      f[i+2] = x[i+2] - (x[i+1] - x[i-3])*dhx;
      f[i+3] = x[i+3] - (grashof>0);
    }else{
      vx  = x[i];
      avx = fabs(vx);
      vxp = 0.5*(vx+avx);
      vxm = 0.5*(vx-avx);
      vy  = x[i+1];
      avy = fabs(vy);
      vyp = 0.5*(vy+avy);
      vym = 0.5*(vy-avy);

      u    = x[i];
      uxx  = (2.0*u - x[i-4] - x[i+4])*hydhx;
      uyy  = (2.0*u - x[i-4*m] - x[i+4*m])*hxdhy;
      f[i] = uxx + uyy - 0.5*(x[i+4*m+3]-x[i-4*m+3])*hx;

      u      = x[i+1];
      uxx    = (2.0*u - x[i-3] - x[i+5])*hydhx;
      uyy    = (2.0*u - x[i-4*m+1] - x[i+4*m+1])*hxdhy;
      f[i+1] = uxx + uyy + 0.5*(x[i+6]-x[i-2])*hy;

      u      = x[i+2];
      uxx    = (2.0*u - x[i-2] - x[i+6])*hydhx;
      uyy    = (2.0*u - x[i-4*m+2] - x[i+4*m+2])*hxdhy;
      f[i+2] = uxx + uyy + (vxp*(u - x[i-2]) + vxm*(x[i+6] - u)) * hy +
            (vyp*(u - x[i-4*m+2]) + vym*(x[i+4*m+2] - u)) * hx - 0.5 * grashof * (x[i+7] - x[i-1]) * hy;

      u      = x[i+3];
      uxx    = (2.0*u - x[i-1] - x[i+7])*hydhx;
      uyy    = (2.0*u - x[i-4*m+3] - x[i+4*m+3])*hxdhy;
      f[i+3] =  uxx + uyy  + prandtl * ((vxp*(u - x[i-1]) + vxm*(x[i+7] - u)) * hy +
            (vyp*(u - x[i-4*m+3]) + vym*(x[i+4*m+3] - u)) * hx);
    }
  }

  ) @*/

  // Field struct is inlined into the linearized array: e.g. x[i].u + x[i].v is now x[i] + x[i+1].
  //   - field order/sequence is [u,v,omega,temp]
  for(i=0; i<=nrows-1; i+=4){
    /* bottom edge */
    if(i<4*m){
      f[i]   = x[i];
      f[i+1] = x[i+1];
      f[i+2] = x[i+2] + (x[i+4] - x[i])*dhy;
      f[i+3] = x[i+3] - x[i+7];
    }else
    /* top edge */
    if(i>=nrows-4*m){
      f[i]   = x[i] - lid;
      f[i+1] = x[i+1];
      f[i+2] = x[i+2] + (x[i] - x[i-4])*dhy;
      f[i+3] = x[i+3] - x[i-1];
    }else
    /* left edge */
    if(i%(4*m)==0){
      f[i]   = x[i];
      f[i+1] = x[i+1];
      f[i+2] = x[i+2] - (x[i+5] - x[i+1])*dhx;
      f[i+3] = x[i+3];
    }else
    /* right edge */
    if(i%(4*m)==4*m-4){
      f[i]   = x[i];
      f[i+1] = x[i+1];
      f[i+2] = x[i+2] - (x[i+1] - x[i-3])*dhx;
      f[i+3] = x[i+3] - (grashof>0);
    }else{
      /* convective coefficients for upwinding */
      vx  = x[i];
      avx = fabs(vx);
      vxp = 0.5*(vx+avx);
      vxm = 0.5*(vx-avx);
      vy  = x[i+1];
      avy = fabs(vy);
      vyp = 0.5*(vy+avy);
      vym = 0.5*(vy-avy);

      /* U velocity */
      u    = x[i];
      uxx  = (2.0*u - x[i-4] - x[i+4])*hydhx;
      uyy  = (2.0*u - x[i-4*m] - x[i+4*m])*hxdhy;
      f[i] = uxx + uyy - 0.5*(x[i+4*m+3]-x[i-4*m+3])*hx;

      /* V velocity */
      u      = x[i+1];
      uxx    = (2.0*u - x[i-3] - x[i+5])*hydhx;
      uyy    = (2.0*u - x[i-4*m+1] - x[i+4*m+1])*hxdhy;
      f[i+1] = uxx + uyy + 0.5*(x[i+6]-x[i-2])*hy;

      /* Omega */
      u      = x[i+2];
      uxx    = (2.0*u - x[i-2] - x[i+6])*hydhx;
      uyy    = (2.0*u - x[i-4*m+2] - x[i+4*m+2])*hxdhy;
      f[i+2] = uxx + uyy + (vxp*(u - x[i-2]) + vxm*(x[i+6] - u)) * hy +
            (vyp*(u - x[i-4*m+2]) + vym*(x[i+4*m+2] - u)) * hx - 0.5 * grashof * (x[i+7] - x[i-1]) * hy;

      /* Temperature */
      u      = x[i+3];
      uxx    = (2.0*u - x[i-1] - x[i+7])*hydhx;
      uyy    = (2.0*u - x[i-4*m+3] - x[i+4*m+3])*hxdhy;
      f[i+3] =  uxx + uyy  + prandtl * ((vxp*(u - x[i-1]) + vxm*(x[i+7] - u)) * hy +
            (vyp*(u - x[i-4*m+3]) + vym*(x[i+4*m+3] - u)) * hx);
    }
  }
  /*@ end @*/
  /*@ end @*/
}
