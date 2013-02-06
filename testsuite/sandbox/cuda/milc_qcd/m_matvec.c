/****************  m_matvec.c  (in su3.a) *******************************
*                                                                       *
* matrix vector multiply                                                *
*  y[i]  <-  A[i]*x[i]                                                  *
*/
void mult_su3_mat_vec(su3_matrix **A, su3_vector **x, su3_vector **y ){
  const int sites_on_node = 10; // or some other global constant value
  register int i,j,k;
  register double ar,ai,br,bi,cr,ci;

  for(i=0; i<sites_on_node; i++) {

    for(j=0; j<3; j++) {

      cr = ci = 0.0;

      for(k=0; k<3; k++) {

        ar=A[i]->e[j][k].real;
        ai=A[i]->e[j][k].imag;
        br=x[i]->c[k].real;
        bi=x[i]->c[k].imag;
        cr += ar*br - ai*bi;
        ci += ar*bi + ai*br;
      }

      y[i]->c[j].real=cr;
      y[i]->c[j].imag=ci;
    }
  }
}
