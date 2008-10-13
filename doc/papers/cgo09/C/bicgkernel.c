/* bicgkernel.f -- translated by f2c (version 20050501).
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

/* Subroutine */ int bicgkernel_(integer *lda, integer *n, doublereal *a, 
	doublereal *p, doublereal *r__, real *s, real *q)
{
    /* System generated locals */
    integer a_dim1, a_offset;

    /* Local variables */
    static doublereal one;
    static integer incx, incy;
    static doublereal zero;
    extern /* Subroutine */ int dgemv_(char *, integer *, integer *, 
	    doublereal *, doublereal *, integer *, doublereal *, integer *, 
	    doublereal *, real *, integer *, ftnlen);


/*   BICG */
/*   in */
/*     A : column matrix, p : vector, r : vector */
/*   out */
/*     q : vector, s : vector */
/*   { */
/*     q = A * p */
/*     s = A' * r */
/*   } */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --p;
    --r__;

    /* Function Body */
    incx = 1;
    incy = 1;
    one = 1.;
    zero = 0.;

/*    Put A*p in q */

    dgemv_("n", n, n, &one, &a[a_offset], lda, &p[1], &incx, &zero, q, &incy, 
	    (ftnlen)1);

/*    Put A'*r in s */

    dgemv_("t", n, n, &one, &a[a_offset], lda, &r__[1], &incx, &zero, s, &
	    incy, (ftnlen)1);

    return 0;
} /* bicgkernel_ */

