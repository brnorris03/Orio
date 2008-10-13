      subroutine atax(lda, n, a, x, y, z)
c
c  y = A' * (A * x)
c
      integer lda, n, j, incx, incy
      double precision zero, one
      double precision x(*), y(*), z(*) 
      double precision A(lda,*)
c
      incx = 1
      incy = 1
      zero = 0.0d0
      zerox = 0
      one = 1.0d0
c
      call dgemv( 'n', n, n, one, a, lda, x, incx, zero, z, incy )
      call dgemv( 't', n, n, one, a, lda, z, incx, zero, y, incy )
c
      return 
      end
