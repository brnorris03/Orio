void FormFunction2DMDOF(double lambda, int M, int N, double* X, double *F){
  register int i;
  /*@ begin PerfTuning(
        def performance_params {
          param TC[] = range(32,1025,32);
          param BC[] = range(14,113,14);
          param SC[] = range(1,6);
          param PL[] = [16,48];
          param CFLAGS[] = map(join, product(['', '-O1', '-O2', '-O3']));
        }
        def build {
          arg build_command = 'nvcc -arch=sm_20 @CFLAGS';
        }
        def input_params {
          param lambda = 6;
          param M[] = [4,8,16,32];
          param N[] = [4,8,16,32];
          constraint c1 = (M==N);
        }
        def input_vars {
          decl dynamic double X[M*N] = random;
          decl dynamic double F[M*N] = 0;
        }
        def performance_counter {
          arg method = 'basic timer';
          arg repetitions = 6;
        }
  ) @*/
  double grashof,prandtl,lid;
  double u,uxx,uyy,vx,vy,avx,avy,vxp,vxm,vyp,vym;

  int m     = M;
  int n     = N;
  int nrows = m*n;

  double dhx = m-1;
  double dhy = n-1;
  double hx = 1.0/dhx;
  double hy = 1.0/dhy;
  double hxdhy = hx*dhy;
  double hydhx = hy*dhx;

  /*@ begin Loop(transform CUDA(threadCount=TC, blockCount=BC, streamCount=SC, preferL1Size=PL)

  for(i=0; i<=nrows-1; i++){
    if(i<m){
      f[i].u     = x[i].u;
      f[i].v     = x[i].v;
      f[i].omega = x[i].omega + (x[i+1].u - x[i].u)*dhy;
      f[i].temp  = x[i].temp  - x[i+1].temp;
    }else
    if(i>=nrows-m){
      f[i].u     = x[i].u - lid;
      f[i].v     = x[i].v;
      f[i].omega = x[i].omega + (x[i].u - x[i-1].u)*dhy;
      f[i].temp  = x[i].temp  - x[i-1].temp;
    }else
    if(i%m==0){
      f[i].u     = x[i].u;
      f[i].v     = x[i].v;
      f[i].omega = x[i].omega - (x[i+1].v - x[i].v)*dhx;
      f[i].temp  = x[i].temp;
    }else
    if(i%m==m-1){
      f[i].u     = x[i].u;
      f[i].v     = x[i].v;
      f[i].omega = x[i].omega - (x[i].v - x[i-1].v)*dhx;
      f[i].temp  = x[i].temp  - (grashof>0);
    }else{
      vx  = x[i].u;
      avx = fabs(vx);
      vxp = .5*(vx+avx);
      vxm = .5*(vx-avx);
      vy  = x[i].v;
      avy = fabs(vy);
      vyp = .5*(vy+avy);
      vym = .5*(vy-avy);

      u      = x[i].u;
      uxx    = (2.0*u - x[i-1].u - x[i+1].u)*hydhx;
      uyy    = (2.0*u - x[i-m].u - x[i+m].u)*hxdhy;
      f[i].u = uxx + uyy - .5*(x[i+m].omega-x[i-m].omega)*hx;

      u      = x[i].v;
      uxx    = (2.0*u - x[i-1].v - x[i+1].v)*hydhx;
      uyy    = (2.0*u - x[i-m].v - x[i+m].v)*hxdhy;
      f[i].v = uxx + uyy + .5*(x[i+1].omega-x[i-1].omega)*hy;

      u          = x[i].omega;
      uxx        = (2.0*u - x[i-1].omega - x[i+1].omega)*hydhx;
      uyy        = (2.0*u - x[i-m].omega - x[i+m].omega)*hxdhy;
      f[i].omega = uxx + uyy + (vxp*(u - x[i-1].omega) + vxm*(x[i+1].omega - u)) * hy +
            (vyp*(u - x[i-m].omega) + vym*(x[i+m].omega - u)) * hx - .5 * grashof * (x[i+1].temp - x[i-1].temp) * hy;

      u         = x[i].temp;
      uxx       = (2.0*u - x[i-1].temp - x[i+1].temp)*hydhx;
      uyy       = (2.0*u - x[i-m].temp - x[i+m].temp)*hxdhy;
      f[i].temp =  uxx + uyy  + prandtl * ((vxp*(u - x[i-1].temp) + vxm*(x[i+1].temp - u)) * hy +
            (vyp*(u - x[i-m].temp) + vym*(x[i+m].temp - u)) * hx);
    }
  }

  ) @*/

  for(i=0; i<=nrows-1; i++){
    /* bottom edge */
    if(i<m){
      f[i].u     = x[i].u;
      f[i].v     = x[i].v;
      f[i].omega = x[i].omega + (x[i+1].u - x[i].u)*dhy;
      f[i].temp  = x[i].temp  - x[i+1].temp;
    }else
    /* top edge */
    if(i>=nrows-m){
      f[i].u     = x[i].u - lid;
      f[i].v     = x[i].v;
      f[i].omega = x[i].omega + (x[i].u - x[i-1].u)*dhy;
      f[i].temp  = x[i].temp  - x[i-1].temp;
    }else
    /* left edge */
    if(i%m==0){
      f[i].u     = x[i].u;
      f[i].v     = x[i].v;
      f[i].omega = x[i].omega - (x[i+1].v - x[i].v)*dhx;
      f[i].temp  = x[i].temp;
    }else
    /* right edge */
    if(i%m==m-1){
      f[i].u     = x[i].u;
      f[i].v     = x[i].v;
      f[i].omega = x[i].omega - (x[i].v - x[i-1].v)*dhx;
      f[i].temp  = x[i].temp  - (grashof>0);
    }else{
      /* convective coefficients for upwinding */
      vx  = x[i].u;
      avx = fabs(vx);
      vxp = .5*(vx+avx);
      vxm = .5*(vx-avx);
      vy  = x[i].v;
      avy = fabs(vy);
      vyp = .5*(vy+avy);
      vym = .5*(vy-avy);

      /* U velocity */
      u      = x[i].u;
      uxx    = (2.0*u - x[i-1].u - x[i+1].u)*hydhx;
      uyy    = (2.0*u - x[i-m].u - x[i+m].u)*hxdhy;
      f[i].u = uxx + uyy - .5*(x[i+m].omega-x[i-m].omega)*hx;

      /* V velocity */
      u      = x[i].v;
      uxx    = (2.0*u - x[i-1].v - x[i+1].v)*hydhx;
      uyy    = (2.0*u - x[i-m].v - x[i+m].v)*hxdhy;
      f[i].v = uxx + uyy + .5*(x[i+1].omega-x[i-1].omega)*hy;

      /* Omega */
      u          = x[i].omega;
      uxx        = (2.0*u - x[i-1].omega - x[i+1].omega)*hydhx;
      uyy        = (2.0*u - x[i-m].omega - x[i+m].omega)*hxdhy;
      f[i].omega = uxx + uyy + (vxp*(u - x[i-1].omega) + vxm*(x[i+1].omega - u)) * hy +
            (vyp*(u - x[i-m].omega) + vym*(x[i+m].omega - u)) * hx - .5 * grashof * (x[i+1].temp - x[i-1].temp) * hy;

      /* Temperature */
      u         = x[i].temp;
      uxx       = (2.0*u - x[i-1].temp - x[i+1].temp)*hydhx;
      uyy       = (2.0*u - x[i-m].temp - x[i+m].temp)*hxdhy;
      f[i].temp =  uxx + uyy  + prandtl * ((vxp*(u - x[i-1].temp) + vxm*(x[i+1].temp - u)) * hy +
            (vyp*(u - x[i-m].temp) + vym*(x[i+m].temp - u)) * hx);
    }
  }
  /*@ end @*/
  /*@ end @*/
}
