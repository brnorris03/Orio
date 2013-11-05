/* vadd.f -- translated by f2c (version 20050501).
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

/* Subroutine */ int vadd_(integer *n, doublereal *x, doublereal *w, 
	doublereal *y, doublereal *z__, doublereal *yy)
{
    static doublereal one;
    static integer incx, incy;
    extern /* Subroutine */ int dcopy_(integer *, doublereal *, integer *, 
	    doublereal *, integer *), daxpy_(integer *, doublereal *, 
	    doublereal *, integer *, doublereal *, integer *);


/*   vadd */
/*   in */
/*   w : vector, y : vector, z : vector */
/*   out */
/*   x : vector */
/*   { */
/*    x = w + y + z */
/*   } */

    /* Parameter adjustments */
    --yy;
    --z__;
    --y;
    --w;
    --x;

    /* Function Body */
    incx = 1;
    incy = 1;
    one = 1.;

/*  Copy y into yy so that input is not overwritten. */

    dcopy_(n, &y[1], &incx, &yy[1], &incy);

/*  y = w + y */

    daxpy_(n, &one, &w[1], &incx, &yy[1], &incy);

/*  y = z + y */

    daxpy_(n, &one, &z__[1], &incx, &yy[1], &incy);

/*  Copy result into x. */

    dcopy_(n, &yy[1], &incx, &x[1], &incy);

    return 0;
} /* vadd_ */

