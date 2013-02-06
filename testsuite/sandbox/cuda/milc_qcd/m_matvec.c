/****************  m_matvec.c  (in su3.a) *******************************
*                                                                       *
* matrix vector multiply                                                *
*  y[i]  <-  A[i]*x[i]                                                  *
*/
void mult_su3_mat_vec(su3_matrix **A, su3_vector **x, su3_vector **y ){
  const int sites_on_node = 10; // or some other global constant value
  register int i,j;
  register double ar,ai,br,bi,cr,ci;

  for(i=0; i<sites_on_node; i++) {

    for(j=0; j<3; j++) {

      ar=A[i]->e[j][0].real;
      ai=A[i]->e[j][0].imag;
      br=x[i]->c[0].real;
      bi=x[i]->c[0].imag;
      cr  = ar*br - ai*bi;
      ci  = ar*bi + ai*br;

      ar=A[i]->e[j][1].real;
      ai=A[i]->e[j][1].imag;
      br=x[i]->c[1].real;
      bi=x[i]->c[1].imag;
      cr += ar*br - ai*bi;
      ci += ar*bi + ai*br;

      ar=A[i]->e[j][2].real;
      ai=A[i]->e[j][2].imag;
      br=x[i]->c[2].real;
      bi=x[i]->c[2].imag;
      cr += ar*br - ai*bi;
      ci += ar*bi + ai*br;

      y[i]->c[j].real=cr;
      y[i]->c[j].imag=ci;
    }
  }
}
