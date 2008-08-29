      subroutine gesummv(alpha, beta, n, lda, a, ldb, b, x, y)
      integer n, incx, incy
      double precision alpha, beta, zero, one
      double precision x(*), y(*)
      double precision A(lda,*), B(ldb,*)
c
c   GESUMMV
c   in
c     A : column matrix,
c     B : column matrix,
c     x : vector,
c     a : scalar,
c     b : scalar
c   out
c     y : vector
c   {
c     y = alpha * (A * x) + beta * (B * x)
c   }
c
      incx = 1
      incy = 1
c
c   Put beta*B*x into y.
c
      zero = 0.0d0
      call dgemv( 'n', n, n, beta, b, lda, x, incx, zero, y, incy )
      print*,'first y',(y(j),j=1,n)
c
c   Put y = beta*B*x + alpha*A*x into y.
c
      one = 1.0d0
      call dgemv( 'n', n, n, alpha, a, lda, x, incx, one, y, incy )
      print*,'second y',(y(j),j=1,n)
c
      return
      end
