real(double), dimension (N) :: aux, game, gmin, gmax, gamfac, &
				& hy_gmelft, hy_gmergt, hy_gmclft, hy_gmcrgt, &
				& pstar1, hy_prght, hy_plft, hy_crght, hy_clft, hy_ulft, hy_urght, &
				& gmstrl, gmstrr, &
				& scrch1, scrch2, wlft1, wrght1, hy_vrght, hy_vlft, &
				& pstar2, wlft, wrght, pstar

real(double), dimension(2) :: hy_pstor

real(double) ge, gc, hy_smallp, small_dp, hy_smallu,  &
				& ustrr1, ustrr2, ustrl1, ustrl2, delu1, delu2, hy_riemanTol, pres_err

integer i, n, numintcells5, hy_nriem

#define rieman_err(i)\
     n = n - 1\
     \
     ! print error message and stop code if iteration fails to converge\
     \
     print *, ' '\
     print *, 'Nonconvergence in subroutine rieman'\
     print *, ' '\
     print *, 'Zone index       = ', i\
     print *, 'Zone center      = ', x(i)\
     print *, 'Iterations tried = ', n+2\
     print *, 'Pressure error   = ', pres_err\
     print *, 'rieman_tol       = ', hy_riemanTol\
     print *, ' '\
     print *, 'pL       = ', hy_plft(i),   ' pR       =', hy_prght(i)\
     print *, 'uL       = ', hy_ulft(i),   ' uR       =', hy_urght(i)\
     print *, 'cL       = ', hy_clft(i),   ' cR       =', hy_crght(i)\
     print *, 'gamma_eL = ', hy_gmelft(i), ' gamma_eR =', hy_gmergt(i)\
     print *, 'gamma_cL = ', hy_gmclft(i), ' gamma_cR =', hy_gmcrgt(i)\
     print *, ' '\
     print *, 'Iteration history:'\
     print *, ' '\
     print '(A4, 2X, A20)', 'n', 'p*'\
     do j = 1, n+2\
        print '(I4, 2X, E20.12)', j, hy_pstor(j)\
     enddo\
     print *, ' '\
     print *, 'Terminating execution.'\
     call Driver_abortFlash('Nonconvergence in subroutine rieman')
