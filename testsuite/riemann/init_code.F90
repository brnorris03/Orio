 aux, game, gmin, gmax, gamfac, hy_gmelft, hy_gmergt, hy_gmclft, hy_gmcrgt &
				& pstar1, hy_prght, hy_plft, hy_crght, hy_clft, hy_ulft, hy_urght, &
				& gmstrl, gmstrr, &
				& scrch1, scrch2, wlft1, wrght1, hy_vright, hy_vlft


real(double) ge, gc, hy_smallp




  L(:,:) = 0.0
  U(:,:) = 0.0
  A(:,:) = 0.0

  do i=1, N
    do j=1, i
      L(j,i) = dble(i+j+1)
      U(j,i) = dble(i+j+1)
    end do
  end do

  do i=1, N
    do j=1, N
      do k=1, N
        A(i,j) = A(i,j) + L(i,k)*U(k,j)
      end do
    end do
  end do
