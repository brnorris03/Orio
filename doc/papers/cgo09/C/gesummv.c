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

    return 0;
} /* gesummv_ */

