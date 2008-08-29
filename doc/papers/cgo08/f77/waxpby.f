      subroutine waxpby(n, w, alpha, x, beta, y)
      integer n, incx, incy
      double precision alpha
      double precision yy(n)
      double precision w(*), x(*), y(*)
c
c   WAXPBY
c   in
c     alpha : scalar,
c     x : vector,
c     beta : scalar,
c     y : vector
c   out
c     w : vector
c   {
c     w = alpha * x + beta * y
c   }
c
      incx = 1
      incy = 1
c
c  Copy y into yy so that input is not overwritten.
c
      call dcopy(n, y, incx, yy, incy)
c
c  Put beta*yy into yy
c
      call dscal(n, beta, yy, incx )
c
c  Put x + yy into yy
c
      call daxpy(n, alpha, x, incx, yy, incy)
c
c  Copy yy into w.
c
      call dcopy(n, yy, incx, w, incy)
c
c
      return
      end

