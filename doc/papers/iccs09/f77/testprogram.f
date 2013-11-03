      program testprogram
      integer n
      parameter (lda = 5, ldb = 5, n = 5)
c
      integer j
      double precision alpha, beta
      double precision w(n), x(n), y(n), z(n), xorig(n), q(n), s(n)
      double precision u1(n), u2(n), v1(n), v2(n), worig(n), yorig(n)
      double precision A(lda, n), B(lda, n)
      data alpha/22/, beta/7/
      data worig/1, 2, 3, 4, 5/, xorig/5, 10, 15, 20, 25/
      data z/2, 4, 6, 8, 10/, yorig/0, 3, 5, 7, 9/
      data u1/1, 2, 3, 4, 5/, u2/5, 10, 15, 20, 25/, v1/0, 3, 5, 7, 9/
      data v2/2, 4, 6, 8, 10/ 
      data A/1,2,3,4,5,2,3,4,5,6,0,1,0,1,1,1,1,1,1,1,2,1,2,1,2/
      data B/10,2,3,4,5,2,3,4,5,6,0,1,0,1,1,1,1,1,1,1,2,1,2,1,20/
c      
c    x = w + y + z
c      
      do j = 1,n
        x(j) = xorig(j)
        y(j) = yorig(j)
        w(j) = worig(j)
      enddo
      call vadd(n, x, w, y, z)
      print*,'vadd result x',(x(j),j=1,n)
      print*,'vadd result should be (3, 9, 14, 19, 24)' 
      print*,' '
      do j = 1,n
        x(j) = xorig(j)
      enddo
c
c     w = alpha * x + beta * y
c
      do j = 1,n
        x(j) = xorig(j)
        w(j) = worig(j)
        y(j) = yorig(j)
      enddo
      call waxpby(n, w, alpha, x, beta, y)
      print*,'waxpby result w',(w(j),j=1,n)
      print*,'vadd result should be (110   241   365   489   613)'
      print*,' '
      do j = 1,n
        w(j) = worig(j)
      enddo
c
c     q = A * x
c     s = A' * y
c
      call bicgkernel(lda, n, A, x, y, s, q)
      print*,'bicgkernel result q',(q(j),j=1,n)
      print*,'bicgkernel result s',(s(j),j=1,n)
      print*,'bicgkernel result should be (95 100 125 130 170)'
      print*,'bicgkernel result should be (94 118 19 24 38)'
      print*,' '
c
c     y = alpha * (A * x) + beta * (B * x)
c
      do j = 1,n
        y(j) = yorig(j)
      enddo
      call gesummv(alpha, beta, n, lda, a, ldb, b, x, y)
      print*,'gesummv 1st y should be (980 700 875 910 1190)'
      print*,'gesummv 2nd y should be (3070 2900 3625 3770 8080)'
      print*,' '
      do j = 1,n
        y(j) = yorig(j)
      enddo
c
c
c     B = A + u1 * v1' + u2 * v2'
c     x = beta * (B' * y) + z
c     w = alpha * (B * x)
c
      call gemver(alpha,beta,lda,n,A,ldb,B,u1,v1,u2,v2,w,x,v1,z)
      print*,'gemver result x',(x(j),j=1,n)
      print*,'gemver result w',(w(j),j=1,n)
      print*,'x should be 7240.  15964.  23169.  31102.  39098.'
      print*,'w should be 113685638. 224265074. 335545386. '
      print*, '446124822. 557914852.'
      print*,' '
c
c  y = A' * (A * x)
c
      do j = 1,n
        x(j) = xorig(j)
      enddo
      call waxpby(n, w, alpha, x, beta, y)
      call atax(lda, n, a, x, y, z)
      print*,'atax result y',(y(j),j=1,n)
      print*,'atax result y should be  2040.  2660.  400.  620.  1010.'
c
      end
