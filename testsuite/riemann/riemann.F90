/*@ begin PerfTuning (  
  def build 
  { 
    #arg build_command = 'icc -O3 -openmp -I/usr/local/icc/include -lm'; 
    arg build_command = 'gfortran -fopenmp -O3';
    arg libs = '-lm -lrt';
  } 
    
  def performance_counter          
  { 
    arg repetitions = 11;
  }

  def performance_params 
  {
    param Rn[] = [1,4,8];
    param U1[] = range(1,8);
    param U2[] = range(1,8);
    param U3[] = range(1,8);
    param U4[] = range(1,8);
    param U5[] = range(1,8);
    param U7[] = range(1,8);
    param U8[] = range(1,8);
    param U9[] = range(1,8);

    param T1_I[] = [1,16,32,64,128,256];
    param T1_N[] = [1,16,32,64,128,256,521];
    param T2_I[] = [1,64,128,256,512,1024,2048];
    param T2_N[] = [1,64,128,256,512,1024,2048];
    param U_I[] = range(1,8);
    param U_N[] = range(1,12);
    


    #param IVEC1[] = [True,False];
    #param SCREP[] = [True,False];  # cannot do in fortran yet because vars must be declared
    param PAR[] = [True,False];

    constraint tileI = ((T2_I == 1) or (T2_I % T1_I == 0));
    constraint tileJ = ((T2_N == 1) or (T2_N % T1_N == 0));
    constraint unrollN = ((U_N == 1) or (U_N % 2 == 0));
    constraint unrollU1 = ((U1 == 1) or (U1 % 2 == 0));
    constraint unrollU2 = ((U2 == 1) or (U2 % 2 == 0));
    constraint unrollU3 = ((U3 == 1) or (U3 % 2 == 0));
    constraint unrollU4 = ((U4 == 1) or (U4 % 2 == 0));
    constraint unrollU5 = ((U5 == 1) or (U5 % 2 == 0));
    constraint unrollU7 = ((U7 == 1) or (U7 % 2 == 0));
    constraint unrollU8 = ((U8 == 1) or (U8 % 2 == 0));
    constraint unrollU9 = ((U9 == 1) or (U9 % 2 == 0));

  }

  def search 
  { 
#    arg algorithm = 'Randomsearch'; 
    arg algorithm = 'Simplex'; 
    arg total_runs = 1000;
    arg time_limit = 100;
  } 
   
  def input_params 
  {
    param N[] = [520];
    param NUMINTCELLS[] = [512];
    param HY_NUMXN[] = [0];
  }

  def input_vars
  {
    arg decl_file = 'decl_code.F90';
    arg init_file = 'init_code.F90';
  }
) @*/ 


/*@ begin Loop (

  transform UnrollJam(ufactor=U1)
  for (i = 5; i<=numIntCells5; i++) {
     aux[i]    = sqrt (0.5 * (game[i] - 1.0) / game[i]);
     ge        = 0.5 * (hy_gmelft[i] + hy_gmergt[i]);
     gc        = 0.5 * (hy_gmclft[i] + hy_gmcrgt[i]);
     gamfac(i) = (1.0 - ge / gc) * (ge - 1.0);
     gmin[i]   = min (game[i-1], game[i], game[i+1]);
     gmax[i]   = max (game[i-1], game[i], game[i+1]);
  }
  
  transform UnrollJam(ufactor=U2)
  for (i = 5; i<=numIntCells5; i++) {
     pstar1[i] = hy_prght[i] - hy_plft[i] - hy_crght[i] * (hy_urght[i] - hy_ulft[i]);
     pstar1[i] = hy_plft[i] + pstar1[i] * (hy_clft[i] / (hy_clft[i] + hy_crght[i]));
     pstar1[i] = max (hy_smallp, pstar1[i]);
  }
  
  transform UnrollJam(ufactor=U3)
  for (i = 5; i<=numIntCells5; i++) {
     gmstrl[i] = gamfac[i] * (pstar1[i] - hy_plft[i]);
     gmstrl[i] = hy_gmelft[i] + 2.0 * gmstrl[i] / (pstar1[i] + hy_plft[i]);
     
     gmstrr[i] = gamfac[i] * (pstar1[i] - hy_prght[i]);
     gmstrr[i] = hy_gmergt[i] + 2.0 * gmstrr[i] / (pstar1[i] + hy_prght[i]);
     
     gmstrl[i] = max (gmin[i], min( gmstrl[i], gmax[i]));
     gmstrr[i] = max (gmin[i], min( gmstrr[i], gmax[i]));
  }
  
  transform UnrollJam(ufactor=U4)
  for (i = 5; i<=numIntCells5; i++) {
     scrch1[i] = pstar1[i] - (gmstrl[i] - 1.0) * hy_plft[i] / (hy_gmelft[i] - 1.0);
     if (scrch1[i] == 0) scrch1[i] = hy_smallp;
     
     wlft1[i]  = pstar1[i] + 0.5 * (gmstrl[i] - 1.0) * (pstar1[i] + hy_plft[i]);
     wlft1[i]  = (pstar1[i] - hy_plft[i]) * wlft1[i] / (hy_vlft[i] * scrch1[i]);
     wlft1[i]  = sqrt(abs(wlft1[i]));
     

     scrch2[i] = pstar1[i] - (gmstrr[i] - 1.0) * hy_prght[i] /(hy_gmergt[i] - 1.0);
     
     if (scrch2[i] == 0.0) scrch2[i] = hy_smallp;
     
     wrght1[i] = pstar1[i] + 0.5 * (gmstrr[i] - 1.0) * (pstar1[i] + hy_prght[i]);
     wrght1[i] = (pstar1[i] - hy_prght[i]) * wrght1[i] / (hy_vrght[i] * scrch2[i]);
     wrght1[i] = sqrt(abs(wrght1[i]));
     
     if (abs (pstar1[i] - hy_plft[i]) < small_dp*(pstar1[i] + hy_plft[i])) wlft1[i] = hy_clft[i];
     wlft1[i]  = max (wlft1[i],  aux[i] * hy_clft[i]);
     
     if (abs (pstar1[i] - hy_prght[i]) < small_dp*((pstar1[i] + hy_prght[i]))) wrght1[i] = hy_crght[i];
     wrght1[i] = max (wrght1[i], aux[i] * hy_crght[i]);
  }

  transform UnrollJam(ufactor=U5)
  for (i = 5; i<=numIntCells5; i++) {
     pstar2[i] = hy_prght[i] - hy_plft[i] - wrght1[i] * (hy_urght[i] - hy_ulft[i]);
     pstar2[i] = hy_plft[i] + pstar2[i] * wlft1[i] / (wlft1[i] + wrght1[i]);
     pstar2[i] = max (hy_smallp, pstar2[i]);
  }
  
  transform Composite(
   tile = [('i',T1_I,'ii'),('n',T1_N,'nn'), 
           (('ii','i'),T2_I,'iii'),(('nn','n'),T2_N,'nnn')],
   unrolljam = (['n','i'], [U_N, U_I])
  )
  for (i = 5; i<=numIntCells5; i++) {

    hy_pstor[1] = pstar1[i];
    hy_pstor[2] = pstar2[i];

    for (n = 1; n <= hy_nriem; n++) {
        gmstrl[i] = gamfac[i] * (pstar2[i] - hy_plft[i]);
        gmstrl[i] = hy_gmelft[i] + 2.0 * gmstrl[i] / (pstar2[i] + hy_plft[i]);

        gmstrr[i] = gamfac[i] * (pstar2[i] - hy_prght[i]);
        gmstrr[i] = hy_gmergt[i] + 2.0 * gmstrr[i] / (pstar2[i] + hy_prght[i]);

        gmstrl[i] = max (gmin[i], min (gmax[i], gmstrl[i]));
        gmstrr[i] = max (gmin[i], min (gmax[i], gmstrr[i]));

        scrch1[i] = pstar2[i] - (gmstrl[i] - 1.0) * hy_plft[i] / (hy_gmelft[i] - 1.0);
        if (scrch1[i] == 0.0) scrch1[i] = hy_smallp;

        wlft[i]   = pstar2[i] + 0.5 * (gmstrl[i] - 1.0) * (pstar2[i] + hy_plft[i]);
        wlft[i]   = (pstar2[i] - hy_plft[i]) * wlft[i] / (hy_vlft[i] * scrch1[i]);
        wlft[i]   = sqrt(abs(wlft[i]));

        scrch2[i] = pstar2[i] - (gmstrr[i] - 1.0) * hy_prght[i] /(hy_gmergt[i] - 1.0);

        if (scrch2[i] == 0.0) scrch2[i] = hy_smallp;

        wrght[i]  = pstar2[i] + 0.5 * (gmstrr[i] - 1.0) * (pstar2[i] + hy_prght[i]);
        wrght[i]  = (pstar2[i] - hy_prght[i]) * wrght[i] / (hy_vrght[i] * scrch2[i]);
        wrght[i]  = sqrt(abs(wrght[i]));

        if (abs (pstar2[i] - hy_plft[i]) < small_dp*(pstar2[i] + hy_plft[i]))
        	wlft[i] = hy_clft[i];
        wlft[i]  = max (wlft[i], aux[i] * hy_clft[i]);

        if (abs (pstar2[i] - hy_prght[i]) < small_dp*(pstar2[i] + hy_prght[i])) wrght[i] = hy_crght[i];
        wrght[i] = max (wrght[i], aux[i] * hy_crght[i]);

        ustrl1    =  hy_ulft[i] - (pstar1[i] -  hy_plft[i]) /  wlft1[i];
        ustrr1    = hy_urght[i] + (pstar1[i] - hy_prght[i]) / wrght1[i];
        ustrl2    =  hy_ulft[i] - (pstar2[i] -  hy_plft[i]) /   wlft[i];
        ustrr2    = hy_urght[i] + (pstar2[i] - hy_prght[i]) /  wrght[i];

        delu1     = ustrl1 - ustrr1;
        delu2     = ustrl2 - ustrr2;
        scrch1[i] = delu2  - delu1;

        if (abs(pstar2[i]-pstar1[i]) <= hy_smallp) scrch1[i] = 0.0;

        if (abs(scrch1[i]) < hy_smallu)
        {
           delu2 = 0.0;
           scrch1[i] = 1.0;
        }

        pstar[i]  = pstar2[i] - delu2 * (pstar2[i] - pstar1[i]) / scrch1[i];
        pstar[i]  = max (hy_smallp, pstar[i]);

        pres_err = abs(pstar[i]-pstar2[i]) / pstar[i];
        if (pres_err < hy_riemanTol) 
            goto10(i);

        wlft1[i]  = wlft[i];
        wrght1[i] = wrght[i];

        pstar1[i] = pstar2[i];
        pstar2[i] = pstar[i];
        hy_pstor[n+2] = pstar[i];

     }
  n = n-1;
  abortcode();

  }

  transform UnrollJam(ufactor=U7)
  for (i = 5; i<=numIntCells5; i++) {

     scrch3[i] = hy_ulft [i] - (pstar[i] -  hy_plft[i]) /  wlft[i];
     scrch4[i] = hy_urght[i] + (pstar[i] - hy_prght[i]) / wrght[i];
     ustar[i]  = 0.5e0 * (scrch3[i] + scrch4[i]);

     urell[i]   = ustar[i] - ugrdl[i];
     scrch1[i]  = sign (one, urell[i]);

     scrch2[i] = 0.5e0 * ( 1.e0 + scrch1[i]);
     scrch3[i] = 0.5e0 * ( 1.e0 - scrch1[i]);

     ps[i]    = hy_plft[i]   * scrch2[i] + hy_prght[i]  * scrch3[i];
     us[i]    = hy_ulft[i]   * scrch2[i] + hy_urght[i]  * scrch3[i];
     uts[i]   = hy_utlft[i]  * scrch2[i] + hy_utrght[i] * scrch3[i];
     utts[i]  = hy_uttlft[i] * scrch2[i] + hy_uttrgt[i] * scrch3[i];
     vs[i]    = hy_vlft[i]   * scrch2[i] + hy_vrght[i]  * scrch3[i];
     games[i] = hy_gmelft[i] * scrch2[i] + hy_gmergt[i] * scrch3[i];
     gamcs[i] = hy_gmclft[i] * scrch2[i] + hy_gmcrgt[i] * scrch3[i];

     rhos[i]  = 1.e0 / vs[i];
     rhos[i]  = max (hy_smlrho, rhos[i]);

     vs[i]    = 1.e0 / rhos[i];
     ws[i]    = wlft[i] * scrch2[i] + wrght[i] * scrch3[i];
     ces[i]   = sqrt (gamcs[i] * ps[i] * vs[i]);

     vstar[i]  = vs[i] - (pstar[i] - ps[i]) / ws[i] / ws[i];
     rhostr[i] = 1.e0 / vstar[i];
     cestar[i] = sqrt (gamcs[i] * pstar[i] * vstar[i]);

     wes[i]    = ces[i]    - scrch1[i] * us[i];
     westar[i] = cestar[i] - scrch1[i] * ustar[i];

     scrch4[i] = ws[i] * vs[i] - scrch1[i] * us[i];


     if (pstar[i] - ps[i] > 0.0) {
        wes[i]    = scrch4[i];
        westar[i] = scrch4[i];
     }

     wes[i]    = wes[i]    + scrch1[i] * ugrdl[i];
     westar[i] = westar[i] + scrch1[i] * ugrdl[i];

     gamfac[i] = (1.e0 - games[i] / gamcs[i]) * (games[i] - 1.e0);
     gmstar[i] = gamfac[i] * (pstar[i] - ps[i]);
     gmstar[i] = games[i] + 2.e0 * gmstar[i] / (pstar[i] + ps[i]);
     gmstar[i] = max (gmin[i], min (gmax[i], gmstar[i]));
  }

  transform UnrollJam(ufactor=U8)
  for (n = 1; n<=hy_numXn; n++) {
     for (i = 5; i<=numIntCells5; i++) {
        xnav[i][n] = hy_xnlft[i][n] * scrch2[i] + hy_xnrght[i][n] * scrch3[i];
	 }
  }

  transform UnrollJam(ufactor=U9)
  for (i = 5; i<=numIntCells5; i++) {
     scrch1[i] = max (wes[i] - westar[i], wes[i] + westar[i], hy_smallu);
     scrch1[i] =     (wes[i] + westar[i]) / scrch1[i];

     scrch1[i] = 0.5e0 * (1.e0 + scrch1[i]);
     scrch2[i] =          1.e0 - scrch1[i];

     rhoav[i]  = scrch1[i] * rhostr[i] + scrch2[i] * rhos [i];
     uav  [i]  = scrch1[i] * ustar[i]  + scrch2[i] * us[i];
     utav [i]  = uts[i];
     uttav[i]  = utts[i];
     pav   [i] = scrch1[i] * pstar[i]  + scrch2[i] * ps[i];
     gameav[i] = scrch1[i] * gmstar[i] + scrch2[i] * games[i];

     if (westar[i] > 0.0) {
        rhoav[i]  = rhostr[i];
        uav[i]    = ustar[i];
        pav[i]    = pstar[i];
        gameav[i] = gmstar[i];
     }

     if (wes[i] < 0.0) {
        rhoav[i]  = rhos[i];
        uav[i]    = us[i];
        pav[i]    = ps[i];
        gameav[i] = games[i];
     }

     urell[i] = uav[i] - ugrdl[i];
  }


  ) @*/

  do i = 5, numIntCells5
     aux(i)    = sqrt (0.5e0 * (game(i) - 1.0e0) / game(i))
     ge        = 0.5e0 * (hy_gmelft(i) + hy_gmergt(i))
     gc        = 0.5e0 * (hy_gmclft(i) + hy_gmcrgt(i))
     gamfac(i) = (1.e0 - ge / gc) * (ge - 1.e0)
     gmin(i)   = min (game(i-1), game(i), game(i+1))
     gmax(i)   = max (game(i-1), game(i), game(i+1))
  enddo
  
    ! construct first guess for secant iteration by assuming that the nonlinear 
    ! wave speed is equal to the sound speed -- the resulting expression is the
    ! same as Toro, Eq. 9.28 in the Primitive Variable Riemann Solver (PVRS).
    ! See also Fry Eq. 72.
    
  do i = 5, numIntCells5
     pstar1(i) = hy_prght(i) - hy_plft(i) - hy_crght(i) * (hy_urght(i) - hy_ulft(i))
     pstar1(i) = hy_plft(i) + pstar1(i) * (hy_clft(i) / (hy_clft(i) + hy_crght(i)))
     pstar1(i) = max (hy_smallp, pstar1(i))
  enddo

    ! calculate approximation jump in gamma acrosss the interface based on the 
    ! first guess for the pressure jump.  There is a left and right 'star' region,
    ! so we need gamma add both places.  Use CG Eq. 31 and 32, with definitions
    ! as in CG Eq. 33.
    
  do i = 5, numIntCells5
     gmstrl(i) = gamfac(i) * (pstar1(i) - hy_plft(i))
     gmstrl(i) = hy_gmelft(i) + 2.e0 * gmstrl(i) / (pstar1(i) + hy_plft(i))
     
     gmstrr(i) = gamfac(i) * (pstar1(i) - hy_prght(i))
     gmstrr(i) = hy_gmergt(i) + 2.e0 * gmstrr(i) / (pstar1(i) + hy_prght(i))
     
     gmstrl(i) = max (gmin(i), min( gmstrl(i), gmax(i)))
     gmstrr(i) = max (gmin(i), min( gmstrr(i), gmax(i)))
  enddo

    ! calculate nonlinear wave speeds for the left and right moving waves based
    ! on the first guess for the pressure jump.  Again, there is a left and a 
    ! right wave speed.  Compute this using CG Eq. 34.
    
  do i = 5, numIntCells5
     scrch1(i) = pstar1(i) - (gmstrl(i) - 1.e0) * hy_plft(i) &
          & / (hy_gmelft(i) - 1.e0)
     if (scrch1(i) .EQ. 0.e0) scrch1(i) = hy_smallp
     
     wlft1(i)  = pstar1(i) + 0.5e0 * (gmstrl(i) - 1.e0) &
          & * (pstar1(i) + hy_plft(i))
     wlft1(i)  = (pstar1(i) - hy_plft(i)) * wlft1(i) / (hy_vlft(i) * scrch1(i))
     wlft1(i)  = sqrt(abs(wlft1(i)))
     

     scrch2(i) = pstar1(i) - (gmstrr(i) - 1.e0) * hy_prght(i) /(hy_gmergt(i) - 1.e0)
     
     if (scrch2(i) .EQ. 0.e0) scrch2(i) = hy_smallp
     
     wrght1(i) = pstar1(i) + 0.5e0 * (gmstrr(i) - 1.e0) &
          & * (pstar1(i) + hy_prght(i))
     wrght1(i) = (pstar1(i) - hy_prght(i)) * wrght1(i) / (hy_vrght(i) * scrch2(i))
     wrght1(i) = sqrt(abs(wrght1(i)))
     
       ! if the pressure jump is small, the wave speed is just the sound speed

     if (abs (pstar1(i) - hy_plft(i)) < small_dp*(pstar1(i) + hy_plft(i))) wlft1(i) = hy_clft(i)
     wlft1(i)  = max (wlft1(i),  aux(i) * hy_clft(i))
     
     if (abs (pstar1(i) - hy_prght(i)) < small_dp*((pstar1(i) + hy_prght(i)))) wrght1(i) = hy_crght(i)
     wrght1(i) = max (wrght1(i), aux(i) * hy_crght(i))
  enddo

    ! construct second guess for the pressure using the nonlinear wave speeds
    ! from the first guess.  This is basically the same thing we did to get
    ! pstar1, except now we are using the better wave speeds instead of the 
    ! sound speed.

  do i = 5, numIntCells5
     pstar2(i) = hy_prght(i) - hy_plft(i) - wrght1(i) * (hy_urght(i) - hy_ulft(i))
     pstar2(i) = hy_plft(i) + pstar2(i) * wlft1(i) / (wlft1(i) + wrght1(i))
     pstar2(i) = max (hy_smallp, pstar2(i))
  enddo

    ! begin the secant iteration -- see CG Eq. 17 for details.  We will continue to
    ! interate for convergence until the error falls below tol (in which case, 
    ! things are good), or we hit hy_nriem iterations (in which case we have a 
    ! problem, and we spit out an error).

  do i = 5, numIntCells5
     
     hy_pstor(1) = pstar1(i)
     hy_pstor(2) = pstar2(i)
     
     do n = 1, hy_nriem
        
        ! new values for the gamma at the "star" state -- again, using CG Eq. 31
          
        gmstrl(i) = gamfac(i) * (pstar2(i) - hy_plft(i))
        gmstrl(i) = hy_gmelft(i) + 2.e0 * gmstrl(i) / (pstar2(i) + hy_plft(i))
        
        gmstrr(i) = gamfac(i) * (pstar2(i) - hy_prght(i))
        gmstrr(i) = hy_gmergt(i) + 2.e0 * gmstrr(i) / (pstar2(i) + hy_prght(i))
        
        gmstrl(i) = max (gmin(i), min (gmax(i), gmstrl(i)))
        gmstrr(i) = max (gmin(i), min (gmax(i), gmstrr(i)))
        
        ! new nonlinear wave speeds, using CG Eq. 34 and the updated gammas
          
        scrch1(i) = pstar2(i) - (gmstrl(i) - 1.e0) * hy_plft(i) &
             & / (hy_gmelft(i) - 1.e0)
        if (scrch1(i) .EQ. 0.e0) scrch1(i) = hy_smallp
        
        wlft(i)   = pstar2(i) + 0.5e0 * (gmstrl(i) - 1.e0) &
             & * (pstar2(i) + hy_plft(i))
        wlft(i)   = (pstar2(i) - hy_plft(i)) * wlft(i) / (hy_vlft(i) * scrch1(i))
        wlft(i)   = sqrt(abs(wlft(i)))

        scrch2(i) = pstar2(i) - (gmstrr(i) - 1.e0) * hy_prght(i) /(hy_gmergt(i) - 1.e0)

        if (scrch2(i) .EQ. 0.e0) scrch2(i) = hy_smallp
        
        wrght(i)  = pstar2(i) + 0.5e0 * (gmstrr(i) - 1.e0) &
             & * (pstar2(i) + hy_prght(i))
        wrght(i)  = (pstar2(i) - hy_prght(i)) * wrght(i) / (hy_vrght(i) * scrch2(i))
        wrght(i)  = sqrt(abs(wrght(i)))
        
        ! if the pressure jump is small, the wave speed is just the sound speed

        if (abs (pstar2(i) - hy_plft(i)) < small_dp*(pstar2(i) + hy_plft(i))) wlft(i) = hy_clft(i)
        wlft(i)  = max (wlft(i), aux(i) * hy_clft(i))
        
        if (abs (pstar2(i) - hy_prght(i)) < small_dp*(pstar2(i) + hy_prght(i))) wrght(i) = hy_crght(i)
        wrght(i) = max (wrght(i), aux(i) * hy_crght(i))

        ! compute the velocities in the "star" state -- using CG Eq. 18 -- ustrl2 and
        ! ustrr2 are the velocities they define there.  ustrl1 and ustrl2 seem to be
        ! the velocities at the last time, since pstar1 is the old 'star' pressure, and
        ! wlft1 is the old wave speed.
        
        ustrl1    =  hy_ulft(i) - (pstar1(i) -  hy_plft(i)) /  wlft1(i)
        ustrr1    = hy_urght(i) + (pstar1(i) - hy_prght(i)) / wrght1(i)
        ustrl2    =  hy_ulft(i) - (pstar2(i) -  hy_plft(i)) /   wlft(i)
        ustrr2    = hy_urght(i) + (pstar2(i) - hy_prght(i)) /  wrght(i)
        
        delu1     = ustrl1 - ustrr1
        delu2     = ustrl2 - ustrr2
        scrch1(i) = delu2  - delu1
        
        if (abs(pstar2(i)-pstar1(i)) .le. hy_smallp) scrch1(i) = 0.e0
        
        if (abs(scrch1(i)) .lt. hy_smallu) then
           delu2 = 0.e0
           scrch1(i) = 1.e0
        endif

        ! pressure at the "star" state -- using CG Eq. 18

        pstar(i)  = pstar2(i) - delu2 * (pstar2(i) - pstar1(i)) / scrch1(i)
        pstar(i)  = max (hy_smallp, pstar(i))
        
        ! check for convergence of iteration, hy_riemanTol is a run-time parameter
        
        pres_err = abs(pstar(i)-pstar2(i)) / pstar(i)
        if (pres_err .lt. hy_riemanTol) goto 10
        
        ! reset variables for next iteration
          
        pstar1(i) = pstar2(i)
        pstar2(i) = pstar(i)
        hy_pstor(n+2) = pstar(i)
        
        wlft1(i)  = wlft(i)
        wrght1(i) = wrght(i)
        
     enddo
!$ end      
     n = n - 1
     
     ! print error message and stop code if iteration fails to converge
     
     print *, ' '
     print *, 'Nonconvergence in subroutine rieman'
     print *, ' '
     print *, 'Zone index       = ', i
     print *, 'Zone center      = ', x(i)
     print *, 'Iterations tried = ', n+2
     print *, 'Pressure error   = ', pres_err
     print *, 'rieman_tol       = ', hy_riemanTol
     print *, ' '
     print *, 'pL       = ', hy_plft(i),   ' pR       =', hy_prght(i)
     print *, 'uL       = ', hy_ulft(i),   ' uR       =', hy_urght(i)
     print *, 'cL       = ', hy_clft(i),   ' cR       =', hy_crght(i)
     print *, 'gamma_eL = ', hy_gmelft(i), ' gamma_eR =', hy_gmergt(i)
     print *, 'gamma_cL = ', hy_gmclft(i), ' gamma_cR =', hy_gmcrgt(i)
     print *, ' '
     print *, 'Iteration history:'
     print *, ' '
     print '(A4, 2X, A20)', 'n', 'p*'
     do j = 1, n+2
        print '(I4, 2X, E20.12)', j, hy_pstor(j)
     enddo
     print *, ' '
     print *, 'Terminating execution.'
     call Driver_abortFlash('Nonconvergence in subroutine rieman')
       
       ! land here if the iterations have converged
       
10     continue
     
  enddo

! end of secant iteration

! calculate fluid velocity for the "star" state -- this comes from the shock
! jump equations, Fry Eq. 68 and 69.  The ustar velocity can be computed
! using either the jump eq. for a left moving or right moving shock -- we use
! the average of the two.
! NOTE: Also look at Fry Eqn. 75 and 76.

  do i = 5, numIntCells5
     scrch3(i) = hy_ulft (i) - (pstar(i) -  hy_plft(i)) /  wlft(i)
     scrch4(i) = hy_urght(i) + (pstar(i) - hy_prght(i)) / wrght(i)
     ustar(i)  = 0.5e0 * (scrch3(i) + scrch4(i))
  enddo

! account for grid velocity

  do i = 5, numIntCells5
     urell(i)   = ustar(i) - ugrdl(i)
     scrch1(i)  = sign (one, urell(i))
  enddo

! decide which state is located at the zone iterface based on the values 
! of the wave speeds.  This is just saying that if ustar > 0, then the state
! is U_L.  if ustar < 0, then the state on the axis is U_R.

  do i = 5, numIntCells5
     
     scrch2(i) = 0.5e0 * ( 1.e0 + scrch1(i))
     scrch3(i) = 0.5e0 * ( 1.e0 - scrch1(i))
     
     ps(i)    = hy_plft(i)   * scrch2(i) + hy_prght(i)  * scrch3(i)
     us(i)    = hy_ulft(i)   * scrch2(i) + hy_urght(i)  * scrch3(i)
     uts(i)   = hy_utlft(i)  * scrch2(i) + hy_utrght(i) * scrch3(i)
     utts(i)  = hy_uttlft(i) * scrch2(i) + hy_uttrgt(i) * scrch3(i)
     vs(i)    = hy_vlft(i)   * scrch2(i) + hy_vrght(i)  * scrch3(i) !v for v=1/rho
     games(i) = hy_gmelft(i) * scrch2(i) + hy_gmergt(i) * scrch3(i)
     gamcs(i) = hy_gmclft(i) * scrch2(i) + hy_gmcrgt(i) * scrch3(i)
     
     rhos(i)  = 1.e0 / vs(i)
     rhos(i)  = max (hy_smlrho, rhos(i))
     
     vs(i)    = 1.e0 / rhos(i)
     ws(i)    = wlft(i) * scrch2(i) + wrght(i) * scrch3(i)
     ces(i)   = sqrt (gamcs(i) * ps(i) * vs(i))
     
     ! compute rhostar, using the shock jump condition (Fry Eq. 80)
     
     vstar(i)  = vs(i) - (pstar(i) - ps(i)) / ws(i) / ws(i)
     rhostr(i) = 1.e0 / vstar(i)
     cestar(i) = sqrt (gamcs(i) * pstar(i) * vstar(i))
     
! compute some factors, Fry Eq. 81 and 82       

     wes(i)    = ces(i)    - scrch1(i) * us(i)
     westar(i) = cestar(i) - scrch1(i) * ustar(i)
     
     scrch4(i) = ws(i) * vs(i) - scrch1(i) * us(i)
     
     
     if (pstar(i) - ps(i) .ge. 0.e0) then
        wes(i)    = scrch4(i)
        westar(i) = scrch4(i)
     endif
     
     wes(i)    = wes(i)    + scrch1(i) * ugrdl(i)
     westar(i) = westar(i) + scrch1(i) * ugrdl(i)
  enddo


  ! compute Fry Eq. 86
  do i = 5, numIntCells5
     gamfac(i) = (1.e0 - games(i) / gamcs(i)) * (games(i) - 1.e0)
     gmstar(i) = gamfac(i) * (pstar(i) - ps(i))
     gmstar(i) = games(i) + 2.e0 * gmstar(i) / (pstar(i) + ps(i))
     gmstar(i) = max (gmin(i), min (gmax(i), gmstar(i)))
  enddo
  
  do n = 1, hy_numXn 
     do i = 5, numIntCells5
        xnav(i,n) = hy_xnlft(i,n) * scrch2(i) + hy_xnrght(i,n) * scrch3(i)
     enddo
  enddo
  
! compute correct state for rarefaction fan by linear interpolation

  do i = 5, numIntCells5
     scrch1(i) = max (wes(i) - westar(i), wes(i) + westar(i), hy_smallu)
     scrch1(i) =     (wes(i) + westar(i)) / scrch1(i)
     
     scrch1(i) = 0.5e0 * (1.e0 + scrch1(i))
     scrch2(i) =          1.e0 - scrch1(i)
     
     rhoav(i)  = scrch1(i) * rhostr(i) + scrch2(i) * rhos (i)
     uav  (i)  = scrch1(i) * ustar(i)  + scrch2(i) * us(i)
     utav (i)  = uts(i)
     uttav(i)  = utts(i)
     pav   (i) = scrch1(i) * pstar(i)  + scrch2(i) * ps(i)
     gameav(i) = scrch1(i) * gmstar(i) + scrch2(i) * games(i)
  enddo
  
  do i = 5, numIntCells5
     if (westar(i) .ge. 0.e0) then
        rhoav(i)  = rhostr(i)
        uav(i)    = ustar(i)
        pav(i)    = pstar(i)
        gameav(i) = gmstar(i)
     endif
     
     if (wes(i) .lt. 0.e0) then
        rhoav(i)  = rhos(i)
        uav(i)    = us(i)
        pav(i)    = ps(i)
        gameav(i) = games(i)
     endif
     
     urell(i) = uav(i) - ugrdl(i)
  enddo 

/*@ end @*/
/*@ end @*/
