      subroutine bicgkernel(lda, n, A, p, r, s, q)
      integer lda, n, incx, incy
      double precision zero, one
      double precision p(*), r(*)
      double precision s(*), q(*)
      double precision A(lda, *)
c
c   BICG
c   in
c     A : column matrix, p : vector, r : vector
c   out
c     q : vector, s : vector
c   {
c     q = A * p
c     s = A' * r
c   }
c
      incx = 1
      incy = 1
      one = 1.0d0
      zero = 0.0d0
c
c    Put A*p in q 
c
      call dgemv( 'n', n, n, one, a, lda, p, incx, zero, q, incy )	
c
c    Put A'*r in s 
c
      call dgemv( 't', n, n, one, a, lda, r, incx, zero, s, incy )	
c
      return
      end
