/****************  m_mv_s_4dir.c  (in su3.a) *****************************
*                                                                       *
* void mult_su3_mat_vec_sum_4dir( su3_matrix *A, su3_vector *x[0123],*y)*
* Multiply the elements of an array of four su3_matrices by the         *
* four su3_vectors, and add the results to                              *
* produce a single su3_vector.                                          *
* y[i]  <-  A[i][0]*x[0][i]+A[i][1]*x[1][i]+A[i][2]*x[2][i]+A[i][3]*x[3][i]
*/
void mult_su3_mat_vec_sum_4dir(su3_matrix ***A, su3_vector ***x, su3_vector **y) {
  const int sites_on_node = 10; // or some other global constant value
  register int i,j,k,d;
  register double ar,ai,br,bi,cr,ci;

  for(i=0; i<sites_on_node; i++) {
    for(d=0; d<4; d++) {
      for(j=0; j<3; j++) {

        cr = ci = 0.0;

        for(k=0; k<3; k++) {

          ar=A[i][d]->e[j][k].real;
          ai=A[i][d]->e[j][k].imag;
          br=x[d][i]->c[k].real;
          bi=x[d][i]->c[k].imag;
          cr += ar*br - ai*bi;
          ci += ar*bi + ai*br;
        }

        y[i]->c[j].real+=cr;
        y[i]->c[j].imag+=ci;
      }
    }
  }
}
