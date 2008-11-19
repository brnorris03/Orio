      subroutine gemver(alpha,beta,lda,n,A,ldb,B,u1,v1,u2,v2,w,x,y,z)
      integer incx, incy, i, j, n
      double precision alpha, beta, zero, one
      double precision w(*), x(*), y(*), u1(*), u2(*), v1(*), v2(*)
      double precision z(n)
      double precision A(lda,*), B(ldb,*)
c
c   GEMVER
c   in
c     A : column matrix, u1 : vector, u2 : vector, v1 : vector, v2 : vector,
c     alpha : scalar, beta : scalar,
c     y : vector, z : vector
c   out
c     B : column matrix, x : vector, w : vector
c
      incx = 1
      incy = 1
      one = 1.0d0
      zero = 0.0d0
c
c     B = A + u1 * v1' + u2 * v2'
c     x = beta * (B' * y) + z
c     w = alpha * (B * x)
c
c
c  Copy A so that it doesn't get overwritten.
c
      do j = 1,n
         do  i = 1,n
            B(i,j) = A(i,j)
         enddo
c      do j = 1,n
c         call dcopy(n, A(1,j), incx, B(1,j), incx)
      enddo
c
c  Rank 1 updates
c
      call dger(n, n, one, u1, incx, v1, incx, B, ldb )	
      call dger(n, n, one, u2, incx, v2, incx, B, ldb )	
c
c  Copy z so that it doesn't get overwritten.
c
      call dcopy(n, z, incx, x, incy)
c
c     x = beta * (B' * y) + z  (z is x here)
c
      call dgemv( 't', n, n, beta, B, lda, y, incx, one, x, incx )
c
c     w = alpha * (B * x)
c
      call dgemv( 'n', n, n, alpha, B, lda, x, incx, zero, w, incx )
c
      return
      end
