real(double), allocatable, dimension (:) :: aux, game, gmin, gmax, gamfac, &
				& hy_gmelft, hy_gmergt, hy_gmclft, hy_gmcrgt, &
				& pstar1, hy_prght, hy_plft, hy_crght, hy_clft, hy_ulft, hy_urght, &
				& gmstrl, gmstrr, &
				& scrch1, scrch2, wlft1, wrght1, hy_vrght, hy_vlft, &
				& pstar2, wlft, wrght, pstar, &
				& scrch3, scrch4, ustar, urell, ugrdl, &
				& ps, us, uts, utts, hy_utlft, hy_utrght, hy_uttlft, hy_uttrgt, &
				& vs, ws, games, gamcs, rhos, ces, vstar, rhostr, cestar, wes, &
				& westar, gmstar, x, &
                & rhoav, uav, utav, uttav, pav, gameav


real(double), allocatable, dimension(:,:) :: xnav, hy_xnlft, hy_xnrght

real(double), allocatable, dimension(:) :: hy_pstor

real(double) ge, gc, hy_smallp, small_dp, hy_smallu, hy_smlrho,  &
				& ustrr1, ustrr2, ustrl1, ustrl2, delu1, delu2, hy_riemanTol, pres_err

real(double) one

integer i, n, numCells, numIntCells5, numIntCells, hy_nriem, hy_numXn, array_len
integer ii, nn, iii, nnn


#define goto10(i) 

#define abortcode()
