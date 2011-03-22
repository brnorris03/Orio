

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#include "f2c.h"

#define GEMVER 

doublereal a,b,*A,*B,*x,*y;
int n;

/*
      double precision alpha, beta
      double precision w(n), x(n), y(n), z(n), xorig(n), q(n), s(n)
      double precision u1(n), u2(n), v1(n), v2(n), worig(n), yorig(n)
      double precision A(lda, n), B(lda, n)
      data alpha/22/, beta/7/
      data worig/1, 2, 3, 4, 5/, xorig/5, 10, 15, 20, 25/
      data z/2, 4, 6, 8, 10/, yorig/0, 3, 5, 7, 9/
      data u1/1, 2, 3, 4, 5/, u2/5, 10, 15, 20, 25/, v1/0, 3, 5, 7, 9/
      data v2/2, 4, 6, 8, 10/ 
      data A/1,2,3,4,5,2,3,4,5,6,0,1,0,1,1,1,1,1,1,1,2,1,2,1,2/
      data B/10,2,3,4,5,2,3,4,5,6,0,1,0,1,1,1,1,1,1,1,2,1,2,1,20/
*/


void init_arrays()
{
    int i, j;
    A=(doublereal*)malloc(N*N*sizeof(doublereal));
    B=(doublereal*)malloc(N*N*sizeof(doublereal));
    x = (doublereal *)malloc(N*sizeof(doublereal));
    y = (doublereal *)malloc(N*sizeof(doublereal));
    a = 1.5;
    b = 2.5;
    n = N;
    for (i=0; i<=N-1; i++) {
      x[i]=(i+1)/N/3.0;
      y[i]=0.0;
      for (j=0; j<=N-1; j++) {
        A[i*N+j]=(i*j)/N/2.0;
        B[i*N+j]=(i*j)/N/3.0;
      }
   }
}

int gemver_(doublereal *alpha, doublereal *beta, integer *
        lda, integer *n, doublereal *a, integer *ldb, doublereal *b, 
        doublereal *u1, doublereal *v1, doublereal *u2, doublereal *v2, 
        doublereal *w, doublereal *x, doublereal *y, doublereal *z__);
int vadd_(integer *n, doublereal *x, doublereal *w, doublereal *y, doublereal *z__, doublereal *yy) ;
int bicgkernel_(integer *lda, integer *n, doublereal *a, doublereal *p, doublereal *r__, doublereal *s, doublereal *q);
int gesummv_(doublereal *alpha, doublereal *beta, integer *n, integer *lda, doublereal *a, integer *ldb, doublereal *b, doublereal *x, doublereal *y);
int atax_(integer *lda, integer *n, doublereal *a, doublereal *x, doublereal *y, doublereal *z__) ;

double rtclock()
{
  struct timezone tzp;
  struct timeval tp;
  int stat;
  gettimeofday (&tp, &tzp);
  return (tp.tv_sec + tp.tv_usec*1.0e-6);
}

int main()
{
  int ret;

  double annot_t_start=0, annot_t_end=0, annot_t_total=0;
  doublereal alpha=22, beta=7, s=0;
  integer lda=N, ldb=N, n=N;
  int annot_i;

  init_arrays();
  for (annot_i=0; annot_i<REPS; annot_i++)
  {
    annot_t_start = rtclock();


    /* Func. call here */

    ret = gesummv_(&a, &b, &n, &n, A, &n, B, x, y);
    s+= y[13];

    annot_t_end = rtclock();
    annot_t_total += annot_t_end - annot_t_start;
  }
  annot_t_total = annot_t_total / REPS;

#ifndef TEST
  printf("%f\n", annot_t_total);
#else
  {
    int i, j;
    for (i=0; i<n; i++) {
      if (i%100==0)
        printf("\n");
      printf("%f ",y[i]);
    }
  }
#endif

  if (s > 0.1) return s;
  return ret;

}
                                    


/* gesummv.f -- translated by f2c (version 20050501).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

#include "f2c.h"

/* Subroutine */ int gesummv_(doublereal *alpha, doublereal *beta, integer *n,
	 integer *lda, doublereal *a, integer *ldb, doublereal *b, doublereal 
	*x, doublereal *y)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset;

    /* Local variables */
    static doublereal one;
    static integer incx, incy;
    static doublereal zero;
    extern /* Subroutine */ int dgemv_(char *, integer *, integer *, 
	    doublereal *, doublereal *, integer *, doublereal *, integer *, 
	    doublereal *, doublereal *, integer *, ftnlen);


/*   GESUMMV */
/*   in */
/*     A : column matrix, */
/*     B : column matrix, */
/*     x : vector, */
/*     a : scalar, */
/*     b : scalar */
/*   out */
/*     y : vector */
/*   { */
/*     y = alpha * (A * x) + beta * (B * x) */
/*   } */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    --x;
    --y;

    /* Function Body */
    incx = 1;
    incy = 1;

/*   Put beta*B*x into y. */

    zero = 0.;
    dgemv_("n", n, n, beta, &b[b_offset], lda, &x[1], &incx, &zero, &y[1], &
	    incy, (ftnlen)1);

/*   Put y = beta*B*x + alpha*A*x into y. */

    one = 1.;
    dgemv_("n", n, n, alpha, &a[a_offset], lda, &x[1], &incx, &one, &y[1], &
	    incy, (ftnlen)1);

    return (int)y[1];
} /* gesummv_ */

