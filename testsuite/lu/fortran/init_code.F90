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
