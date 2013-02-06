/****************  m_matvec.c  (in su3.a) *******************************
*                                                                       *
* matrix vector multiply                                                *
*  y[i]  <-  A[i]*x[i]                                                  *
*/
void mult_su3_mat_vec(double A[], double x[], double y[]) {
  const int sites_on_node = 10; // or some other global constant value
  register int i,j,k;
  register double ar,ai,br,bi,cr,ci;

  for(i=0; i<sites_on_node; i++) {

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
}
