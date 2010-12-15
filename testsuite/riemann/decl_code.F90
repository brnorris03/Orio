
	call random_seed
	aux(:) = 0.0
	game(:) = 0.0
	gmin(:) = 0.0
	gmax(:) = 0.0
	gamfac(:) = 0.0
	pstar1(:) = 0.0
	gmstrl(:) = 0.0
	gmstrr(:) = 0.0
	scrch1(:) = 0.0
	scrch2(:) = 0.0
	wlft1(:) = 0.0
	wrght1(:) = 0.0
	call random_number(hy_gmelft)
	call random_number(hy_gmergt)
	call random_number(hy_gmclft)
	call random_number(hy_gmcrgt)
	call random_number(hy_prght)
	call random_number(hy_plft)
	call random_number(hy_crght)
	call random_number(hy_clft)
	call random_number(hy_ulft)
	call random_number(hy_urght)
	call random_number(hy_vright)
	call random_number(hy_vlft)

	ge = 0.0
	gc = 0.0
	hy_small = 1.E-10

real(double) ge, gc, hy_smallp
