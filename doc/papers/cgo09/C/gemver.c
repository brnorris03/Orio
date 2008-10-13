/* gemver.f -- translated by f2c (version 20050501).
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

/* Subroutine */ int gemver_(doublereal *alpha, doublereal *beta, integer *
	lda, integer *n, doublereal *a, integer *ldb, doublereal *b, 
	doublereal *u1, doublereal *v1, doublereal *u2, doublereal *v2, 
	doublereal *w, doublereal *x, doublereal *y, doublereal *z__)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1, i__2;

    /* Local variables */
    static integer i__, j;
    static doublereal one;
    extern /* Subroutine */ int dger_(integer *, integer *, doublereal *, 
	    doublereal *, integer *, doublereal *, integer *, doublereal *, 
	    integer *);
    static integer incx, incy;
    static doublereal zero;
    extern /* Subroutine */ int dgemv_(char *, integer *, integer *, 
	    doublereal *, doublereal *, integer *, doublereal *, integer *, 
	    doublereal *, doublereal *, integer *, ftnlen), dcopy_(integer *, 
	    doublereal *, integer *, doublereal *, integer *);


/*   GEMVER */
/*   in */
/*     A : column matrix, u1 : vector, u2 : vector, v1 : vector, v2 : vector, */
/*     alpha : scalar, beta : scalar, */
/*     y : vector, z : vector */
/*   out */
/*     B : column matrix, x : vector, w : vector */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --z__;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    --u1;
    --v1;
    --u2;
    --v2;
    --w;
    --x;
    --y;

    /* Function Body */
    incx = 1;
    incy = 1;
    one = 1.;
    zero = 0.;

/*     B = A + u1 * v1' + u2 * v2' */
/*     x = beta * (B' * y) + z */
/*     w = alpha * (B * x) */


/*  Copy A so that it doesn't get overwritten. */

    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *n;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    b[i__ + j * b_dim1] = a[i__ + j * a_dim1];
	}
/*      do j = 1,n */
/*         call dcopy(n, A(1,j), incx, B(1,j), incx) */
    }

/*  Rank 1 updates */

    dger_(n, n, &one, &u1[1], &incx, &v1[1], &incx, &b[b_offset], ldb);
    dger_(n, n, &one, &u2[1], &incx, &v2[1], &incx, &b[b_offset], ldb);

/*  Copy z so that it doesn't get overwritten. */

    dcopy_(n, &z__[1], &incx, &x[1], &incy);

/*     x = beta * (B' * y) + z  (z is x here) */

    dgemv_("t", n, n, beta, &b[b_offset], lda, &y[1], &incx, &one, &x[1], &
	    incx, (ftnlen)1);

/*     w = alpha * (B * x) */

    dgemv_("n", n, n, alpha, &b[b_offset], lda, &x[1], &incx, &zero, &w[1], &
	    incx, (ftnlen)1);

    return 0;
} /* gemver_ */

