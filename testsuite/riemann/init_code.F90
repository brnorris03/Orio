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
	pstar1(:) = 0.0
	pstar2(:) = 0.0
	pstar(:) = 0.0
	hy_pstor(:) = 0.0
	scrch3(:) = 0.0
	scrch4(:) = 0.0
	ustar(:) = 0.0
	urell(:) = 0.0
	ugrdl(:) = 0.0
	ps(:) = 0.0
	us(:) = 0.0
	uts(:) = 0.0
	utts(:) = 0.0
	hy_utlft(:) = 0.0
	hy_utrght(:) = 0.0
	hy_uttlft(:) = 0.0
	hy_uttrgt(:) = 0.0
	vs(:) = 0.0
	ws(:) = 0.0
	games(:) = 0.0
	gamcs(:) = 0.0
	rhos(:) = 0.0
	ces(:) = 0.0
	vstar(:) = 0.0
	rhostr(:) = 0.0
	cestar(:) = 0.0
	wes(:) = 0.0

    pres_err = 0.0

	! Inputs
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
	call random_number(hy_vrght)
	call random_number(hy_vlft)
	call random_number(hy_numXn)

	ge = 0.0
	gc = 0.0
	hy_smallp = 1.E-10
	small_dp = 1.E-10
	hy_smallu = 1.E-10
	hy_smlrho = 1.E-10

	! check what the tolerance and max its should really be
	hy_nriem = 30
	hy_riemanTol = 1.E-4

	numintcells5 = 100
