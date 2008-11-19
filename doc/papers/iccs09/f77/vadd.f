      subroutine vadd(n, x, w, y, z) 
      integer n, incx, incy
      double precision one
      double precision yy(n) 
      double precision w(*), x(*), y(*), z(*)
c
c   vadd
c   in
c   w : vector, y : vector, z : vector
c   out
c   x : vector
c   {
c    x = w + y + z
c   }
c
      incx = 1
      incy = 1
      one = 1.0d0
c
c  Copy y into yy so that input is not overwritten.
c
      call dcopy(n, y, incx, yy, incy)
c
c  y = w + y
c
      call daxpy(n, one, w, incx, yy, incy)
c
c  y = z + y
c
      call daxpy(n, one, z, incx, yy, incy)
c
c  Copy result into x.
c
      call dcopy(n, yy, incx, x, incy)
c
      return
      end

