/* waxpby.f -- translated by f2c (version 20050501).
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

/* Subroutine */ int waxpby_(integer *n, doublereal *w, doublereal *alpha, 
	doublereal *x, real *beta, doublereal *y, doublereal *yy)
{
    static integer incx, incy;
    extern /* Subroutine */ int dscal_(integer *, real *, doublereal *, 
	    integer *), dcopy_(integer *, doublereal *, integer *, doublereal 
	    *, integer *), daxpy_(integer *, doublereal *, doublereal *, 
	    integer *, doublereal *, integer *);


/*   WAXPBY */
/*   in */
/*     alpha : scalar, */
/*     x : vector, */
/*     beta : scalar, */
/*     y : vector */
/*   out */
/*     w : vector */
/*   { */
/*     w = alpha * x + beta * y */
/*   } */

    /* Parameter adjustments */
    --yy;
    --y;
    --x;
    --w;

    /* Function Body */
    incx = 1;
    incy = 1;

/*  Copy y into yy so that input is not overwritten. */

    dcopy_(n, &y[1], &incx, &yy[1], &incy);

/*  Put beta*yy into yy */

    dscal_(n, beta, &yy[1], &incx);

/*  Put x + yy into yy */

    daxpy_(n, alpha, &x[1], &incx, &yy[1], &incy);

/*  Copy yy into w. */

    dcopy_(n, &yy[1], &incx, &w[1], &incy);


    return 0;
} /* waxpby_ */

