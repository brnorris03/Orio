/****************  m_matvec.c  (in su3.a) *******************************
*                                                                       *
* matrix vector multiply                                                *
*  C  <-  A*B                                                           *

auxiliary definitions:
typedef struct {
   double real;
   double imag;
} dcomplex;
typedef struct { dcomplex e[3][3]; } dsu3_matrix;
typedef struct { dcomplex c[3]; } dsu3_vector;
#define su3_matrix      dsu3_matrix
#define su3_vector      dsu3_vector
typedef struct {
  su3_matrix *fat;
  su3_matrix *lng;
  su3_matrix *fatback;  // NULL if unused
  su3_matrix *lngback;  // NULL if unused
} fn_links_t;
extern char ** gen_pt[16];
*/
void mult_su3_mat_vec(su3_matrix **a, su3_vector **b, su3_vector **c ){
  const int sites_on_node = 10; // or some other global constant value
  register int i;
  register double c0r,c0i,c1r,c1i,c2r,c2i;
  register double br,bi,a0,a1,a2;

  c0r = c0i = c1r = c1i = c2r = c2i = 0.0;

  for(i=0; i<sites_on_node; i++) {

    br=b[i]->c[0].real;    bi=b[i]->c[0].imag;
    a0=a[i]->e[0][0].real;
    a1=a[i]->e[1][0].real;
    a2=a[i]->e[2][0].real;

    c0r += a0*br;
    c1r += a1*br;
    c2r += a2*br;
    c0i += a0*bi;
    c1i += a1*bi;
    c2i += a2*bi;

    a0=mat->e[0][0].imag;
    a1=mat->e[1][0].imag;
    a2=mat->e[2][0].imag;

    c0r -= a0*bi;
    c1r -= a1*bi;
    c2r -= a2*bi;
    c0i += a0*br;
    c1i += a1*br;
    c2i += a2*br;

    br=b[i]->c[1].real;    bi=b[i]->c[1].imag;
    a0=mat->e[0][1].real;
    a1=mat->e[1][1].real;
    a2=mat->e[2][1].real;

    c0r += a0*br;
    c1r += a1*br;
    c2r += a2*br;
    c0i += a0*bi;
    c1i += a1*bi;
    c2i += a2*bi;

    a0=mat->e[0][1].imag;
    a1=mat->e[1][1].imag;
    a2=mat->e[2][1].imag;

    c0r -= a0*bi;
    c1r -= a1*bi;
    c2r -= a2*bi;
    c0i += a0*br;
    c1i += a1*br;
    c2i += a2*br;

    br=b[i]->c[2].real;    bi=b[i]->c[2].imag;
    a0=mat->e[0][2].real;
    a1=mat->e[1][2].real;
    a2=mat->e[2][2].real;

    c0r += a0*br;
    c1r += a1*br;
    c2r += a2*br;
    c0i += a0*bi;
    c1i += a1*bi;
    c2i += a2*bi;

    a0=mat->e[0][2].imag;
    a1=mat->e[1][2].imag;
    a2=mat->e[2][2].imag;

    c0r -= a0*bi;
    c1r -= a1*bi;
    c2r -= a2*bi;
    c0i += a0*br;
    c1i += a1*br;
    c2i += a2*br;

    c[i]->c[0].real = c0r;
    c[i]->c[0].imag = c0i;
    c[i]->c[1].real = c1r;
    c[i]->c[1].imag = c1i;
    c[i]->c[2].real = c2r;
    c[i]->c[2].imag = c2i;
  }
}
